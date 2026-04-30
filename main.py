import cv2
cv2.ocl.setUseOpenCL(True)
import numpy as np
import sys
import filters
from filters.beauty   import apply as beauty_apply, get_last_landmarks, get_landmarks_and_bbox
from filters.overlays import apply_cat_ears
from filters.makeup   import apply as makeup_apply
import filters.liquify  as liquify
import filters.spotHeal as spotHeal
import filters.symmetry as symmetry


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CANVAS_W      = 780
MAIN_H        = 480
STRIP_H       = 150
CANVAS_H      = MAIN_H + STRIP_H
MAKEUP_H      = 90
EDIT_H        = 80
THUMB_W       = 96
THUMB_H       = 72
THUMB_PADDING = 10
THUMB_Y       = MAIN_H + 60

BTN_W, BTN_H  = 118, 30
BTN_GAP       = 7
BTN_Y         = MAIN_H + 12
_NUM_BTNS     = 5
BTN1_X        = (CANVAS_W - BTN_W * _NUM_BTNS - BTN_GAP * (_NUM_BTNS - 1)) // 2
BTN2_X        = BTN1_X + BTN_W + BTN_GAP
BTN3_X        = BTN2_X + BTN_W + BTN_GAP
BTN4_X        = BTN3_X + BTN_W + BTN_GAP
BTN5_X = BTN4_X + BTN_W + BTN_GAP

THUMB_UPDATE_EVERY = 15

# ---------------------------------------------------------------------------
# Makeup presets
# ---------------------------------------------------------------------------
BLUSH_PRESETS = [("Pink", 160), ("Red", 0), ("Peach", 15), ("Coral", 8), ("Rose", 170), ("Berry", 145), ("Nude", 15)]
LIP_PRESETS   = [("Pink", 160), ("Red", 0), ("Peach", 15), ("Coral", 8), ("Rose", 170), ("Berry", 145), ("Nude", 15)]

SWATCH_R       = 11
SWATCH_GAP     = 28
SWATCH_START_X = 90
BLUSH_ROW_Y    = CANVAS_H + 22
LIP_ROW_Y      = CANVAS_H + 62
OP_LABEL_X     = 480
OP_BTN_X       = 570
OP_BTN2_X      = 605
OP_BTN_W       = 28
OP_BTN_H       = 22

# Edit panel constants
EDIT_SUBMODE_Y    = 18    # y offset within panel for sub-mode buttons
EDIT_SUB_W        = 90
EDIT_SUB_H        = 26
EDIT_SUB1_X       = 10
EDIT_SUB2_X       = 108
EDIT_SIZE_LABEL_X = 215
EDIT_BTN_MINUS_X  = 310
EDIT_BTN_PLUS_X   = 345
EDIT_BTN_W        = 28
EDIT_BTN_H        = 26
EDIT_CLEAR_X      = 385
EDIT_CLEAR_W      = 70
EDIT_UNDO_X       = 465
EDIT_UNDO_W       = 50
EDIT_HINT_X       = 525

# ---------------------------------------------------------------------------
# Symmetry (If Dark)
# ---------------------------------------------------------------------------

DARK_BTN_W = 80
DARK_BTN_H = 22
DARK_BTN_X = BTN5_X + BTN_W - BTN_GAP
DARK_BTN_Y = BTN_Y + BTN_H - 6

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
selected      = 0
beauty_on     = False
overlay_on    = False
makeup_on     = False
edit_on       = False
symmetry_on = False
edit_mode     = "liquify"   # "liquify" or "spotheal"
edit_size     = 20

_current_landmarks = None

makeup_state = {
    "blush_idx": 0,
    "lip_idx":   0,
    "blush_op":  0,
    "lip_op":    0,
}

_drag_start = None
_frame_ox   = 0
_frame_oy   = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hue_to_bgr(hue):
    hsv = np.uint8([[[hue, 200, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0][0])

def get_filter_list():
    return [
        ("Original", filters.original),
        ("Warm",     filters.warm),
        ("Cool",     filters.cool),
        ("Vivid",    filters.vivid),
        ("Fade",     filters.fade),
    ]

def get_thumb_x(i, n):
    total_w = n * THUMB_W + (n - 1) * THUMB_PADDING
    start_x = (CANVAS_W - total_w) // 2
    return start_x + i * (THUMB_W + THUMB_PADDING)

def reload_filters():
    global filters
    for mod in list(sys.modules.keys()):
        if mod.startswith('filters'):
            del sys.modules[mod]
    import filters as filters
    print("Filters reloaded.")

def _edit_panel_y():
    return CANVAS_H + (MAKEUP_H if makeup_on else 0)

# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global selected, beauty_on, overlay_on, makeup_on, edit_on, symmetry_on
    global _drag_start, edit_size, edit_mode

    # ---- Edit interactions inside camera area ----
    if edit_on and y < MAIN_H:
        if edit_mode == "liquify":
            if event == cv2.EVENT_LBUTTONDOWN:
                _drag_start = (x, y)
                return
            if event == cv2.EVENT_LBUTTONUP and _drag_start is not None:
                fx0 = _drag_start[0] - _frame_ox
                fy0 = _drag_start[1] - _frame_oy
                dx  = x - _drag_start[0]
                dy  = y - _drag_start[1]
                if abs(dx) > 2 or abs(dy) > 2:
                    liquify.add_anchor(_current_landmarks, fx0, fy0, dx, dy, edit_size)
                _drag_start = None
                return
        elif edit_mode == "spotheal":
            if event == cv2.EVENT_LBUTTONDOWN:
                fx = x - _frame_ox
                fy = y - _frame_oy
                spotHeal.add_spot(_current_landmarks, fx, fy, edit_size)
                return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # ---- Toggle buttons ----
    if BTN1_X <= x <= BTN1_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        beauty_on = not beauty_on;  return
    if BTN2_X <= x <= BTN2_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        overlay_on = not overlay_on;  return
    if BTN3_X <= x <= BTN3_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        makeup_on = not makeup_on;  return
    if BTN4_X <= x <= BTN4_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        edit_on = not edit_on;  return
    if BTN5_X <= x <= BTN5_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        symmetry_on = not symmetry_on;  return
    if symmetry_on and DARK_BTN_X <= x <= DARK_BTN_X + DARK_BTN_W and DARK_BTN_Y <= y <= DARK_BTN_Y + DARK_BTN_H:
        symmetry.toggle_dark_mode();  return

    # ---- Filter thumbnails ----
    filter_list = get_filter_list()
    if THUMB_Y <= y <= THUMB_Y + THUMB_H:
        for i in range(len(filter_list)):
            tx = get_thumb_x(i, len(filter_list))
            if tx <= x <= tx + THUMB_W:
                selected = i;  return

    # ---- Makeup controls ----
    if makeup_on and CANVAS_H <= y <= CANVAS_H + MAKEUP_H:
        _handle_makeup_click(x, y);  return

    # ---- Edit panel controls ----
    if edit_on:
        panel_y = _edit_panel_y()
        if panel_y <= y <= panel_y + EDIT_H:
            _handle_edit_click(x, y, panel_y)

def _handle_makeup_click(x, y):
    for i in range(len(BLUSH_PRESETS)):
        sx = SWATCH_START_X + i * SWATCH_GAP
        if abs(x - sx) <= SWATCH_R and abs(y - BLUSH_ROW_Y) <= SWATCH_R:
            makeup_state["blush_idx"] = i;  return
    for i in range(len(LIP_PRESETS)):
        sx = SWATCH_START_X + i * SWATCH_GAP
        if abs(x - sx) <= SWATCH_R and abs(y - LIP_ROW_Y) <= SWATCH_R:
            makeup_state["lip_idx"] = i;  return
    by = BLUSH_ROW_Y - OP_BTN_H // 2
    if OP_BTN_X <= x <= OP_BTN_X + OP_BTN_W and by <= y <= by + OP_BTN_H:
        makeup_state["blush_op"] = max(0,   makeup_state["blush_op"] - 5);  return
    if OP_BTN2_X <= x <= OP_BTN2_X + OP_BTN_W and by <= y <= by + OP_BTN_H:
        makeup_state["blush_op"] = min(100, makeup_state["blush_op"] + 5);  return
    ly = LIP_ROW_Y - OP_BTN_H // 2
    if OP_BTN_X <= x <= OP_BTN_X + OP_BTN_W and ly <= y <= ly + OP_BTN_H:
        makeup_state["lip_op"] = max(0,   makeup_state["lip_op"] - 5);  return
    if OP_BTN2_X <= x <= OP_BTN2_X + OP_BTN_W and ly <= y <= ly + OP_BTN_H:
        makeup_state["lip_op"] = min(100, makeup_state["lip_op"] + 5);  return

def _handle_edit_click(x, y, panel_y):
    global edit_size, edit_mode
    sub_y  = panel_y + EDIT_SUBMODE_Y
    ctrl_y = panel_y + EDIT_H // 2

    # Sub-mode toggle buttons
    if EDIT_SUB1_X <= x <= EDIT_SUB1_X + EDIT_SUB_W and sub_y <= y <= sub_y + EDIT_SUB_H:
        edit_mode = "liquify";  return
    if EDIT_SUB2_X <= x <= EDIT_SUB2_X + EDIT_SUB_W and sub_y <= y <= sub_y + EDIT_SUB_H:
        edit_mode = "spotheal";  return

    by = ctrl_y - EDIT_BTN_H // 2
    if EDIT_BTN_MINUS_X <= x <= EDIT_BTN_MINUS_X + EDIT_BTN_W and by <= y <= by + EDIT_BTN_H:
        edit_size = max(0, edit_size - 5);  return
    if EDIT_BTN_PLUS_X <= x <= EDIT_BTN_PLUS_X + EDIT_BTN_W and by <= y <= by + EDIT_BTN_H:
        edit_size = min(120, edit_size + 5);  return
    if EDIT_CLEAR_X <= x <= EDIT_CLEAR_X + EDIT_CLEAR_W and by <= y <= by + EDIT_BTN_H:
        liquify.clear_anchors()
        spotHeal.clear_spots()
        return
    if EDIT_UNDO_X <= x <= EDIT_UNDO_X + EDIT_UNDO_W and by <= y <= by + EDIT_BTN_H:
        if edit_mode == "liquify":
            liquify.undo_anchor()
        else:
            spotHeal.undo_spot()
        return

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_buttons(canvas):
    edit_label = "Edit: ON" if edit_on else "Edit: OFF"
    for bx, label, active in [
        (BTN1_X, "Beauty: ON"   if beauty_on  else "Beauty: OFF",   beauty_on),
        (BTN2_X, "Cat Ears: ON" if overlay_on else "Cat Ears: OFF",  overlay_on),
        (BTN3_X, "Makeup: ON"   if makeup_on  else "Makeup: OFF",   makeup_on),
        (BTN4_X, edit_label,                                         edit_on),
        (BTN5_X, "Symmetry: ON" if symmetry_on else "Symmetry: OFF", symmetry_on),
    ]:
        color = (60, 160, 60) if active else (60, 60, 60)
        cv2.rectangle(canvas, (bx, BTN_Y), (bx + BTN_W, BTN_Y + BTN_H), color, -1)
        cv2.rectangle(canvas, (bx, BTN_Y), (bx + BTN_W, BTN_Y + BTN_H), (200, 200, 200), 1)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lx = bx + (BTN_W - tw) // 2
        cv2.putText(canvas, label, (lx, BTN_Y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        
    if symmetry_on:
        label = "Dark: ON" if symmetry.dark_mode else "Dark: OFF"
        color = (60, 160, 60) if symmetry.dark_mode else (60, 60, 60)
        cv2.rectangle(canvas, (DARK_BTN_X, DARK_BTN_Y),
                      (DARK_BTN_X + DARK_BTN_W, DARK_BTN_Y + DARK_BTN_H), color, -1)
        cv2.rectangle(canvas, (DARK_BTN_X, DARK_BTN_Y),
                      (DARK_BTN_X + DARK_BTN_W, DARK_BTN_Y + DARK_BTN_H), (200, 200, 200), 1)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx = DARK_BTN_X + (DARK_BTN_W - tw) // 2
        cv2.putText(canvas, label, (lx, DARK_BTN_Y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

def draw_makeup_controls(canvas):
    panel_y = CANVAS_H
    cv2.rectangle(canvas, (0, panel_y), (CANVAS_W, panel_y + MAKEUP_H), (20, 20, 20), -1)
    cv2.line(canvas, (0, panel_y), (CANVAS_W, panel_y), (60, 60, 60), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row_label, presets, row_y, idx_key, op_key in [
        ("Blush", BLUSH_PRESETS, BLUSH_ROW_Y, "blush_idx", "blush_op"),
        ("Lips",  LIP_PRESETS,   LIP_ROW_Y,   "lip_idx",   "lip_op"),
    ]:
        cv2.putText(canvas, row_label, (10, row_y + 5), font, 0.45, (180, 180, 180), 1)
        for i, (_, hue) in enumerate(presets):
            sx = SWATCH_START_X + i * SWATCH_GAP
            cv2.circle(canvas, (sx, row_y), SWATCH_R, _hue_to_bgr(hue), -1)
            if makeup_state[idx_key] == i:
                cv2.circle(canvas, (sx, row_y), SWATCH_R + 3, (255, 255, 255), 2)
        op_val = makeup_state[op_key]
        cv2.putText(canvas, f"Opacity: {op_val}%", (OP_LABEL_X, row_y + 5), font, 0.42, (180, 180, 180), 1)
        by = row_y - OP_BTN_H // 2
        cv2.rectangle(canvas, (OP_BTN_X, by), (OP_BTN_X + OP_BTN_W, by + OP_BTN_H), (70, 70, 70), -1)
        cv2.putText(canvas, "-", (OP_BTN_X + 8, by + 16), font, 0.55, (255, 255, 255), 1)
        cv2.rectangle(canvas, (OP_BTN2_X, by), (OP_BTN2_X + OP_BTN_W, by + OP_BTN_H), (70, 70, 70), -1)
        cv2.putText(canvas, "+", (OP_BTN2_X + 6, by + 16), font, 0.55, (255, 255, 255), 1)

def draw_edit_controls(canvas):
    panel_y = _edit_panel_y()
    cv2.rectangle(canvas, (0, panel_y), (CANVAS_W, panel_y + EDIT_H), (18, 18, 28), -1)
    cv2.line(canvas, (0, panel_y), (CANVAS_W, panel_y), (60, 60, 60), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Sub-mode buttons
    sub_y = panel_y + EDIT_SUBMODE_Y
    for sx, label, active in [
        (EDIT_SUB1_X, "Liquify",   edit_mode == "liquify"),
        (EDIT_SUB2_X, "Spot Heal", edit_mode == "spotheal"),
    ]:
        color = (80, 100, 160) if active else (50, 50, 50)
        cv2.rectangle(canvas, (sx, sub_y), (sx + EDIT_SUB_W, sub_y + EDIT_SUB_H), color, -1)
        cv2.rectangle(canvas, (sx, sub_y), (sx + EDIT_SUB_W, sub_y + EDIT_SUB_H), (150, 150, 150), 1)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx = sx + (EDIT_SUB_W - tw) // 2
        cv2.putText(canvas, label, (lx, sub_y + 17), font, 0.38, (255, 255, 255), 1)

    # Size control + action buttons on center row
    ctrl_y = panel_y + EDIT_H // 2 + 10
    by = ctrl_y - EDIT_BTN_H // 2

    cv2.putText(canvas, f"Size: {edit_size}", (EDIT_SIZE_LABEL_X, ctrl_y + 5),
                font, 0.45, (180, 180, 180), 1)

    cv2.rectangle(canvas, (EDIT_BTN_MINUS_X, by), (EDIT_BTN_MINUS_X + EDIT_BTN_W, by + EDIT_BTN_H), (70, 70, 70), -1)
    cv2.putText(canvas, "-", (EDIT_BTN_MINUS_X + 8, by + 18), font, 0.55, (255, 255, 255), 1)
    cv2.rectangle(canvas, (EDIT_BTN_PLUS_X, by), (EDIT_BTN_PLUS_X + EDIT_BTN_W, by + EDIT_BTN_H), (70, 70, 70), -1)
    cv2.putText(canvas, "+", (EDIT_BTN_PLUS_X + 6, by + 18), font, 0.55, (255, 255, 255), 1)

    cv2.rectangle(canvas, (EDIT_CLEAR_X, by), (EDIT_CLEAR_X + EDIT_CLEAR_W, by + EDIT_BTN_H), (80, 50, 50), -1)
    cv2.putText(canvas, "Clear", (EDIT_CLEAR_X + 14, by + 18), font, 0.42, (255, 255, 255), 1)

    cv2.rectangle(canvas, (EDIT_UNDO_X, by), (EDIT_UNDO_X + EDIT_UNDO_W, by + EDIT_BTN_H), (50, 70, 80), -1)
    cv2.putText(canvas, "Undo", (EDIT_UNDO_X + 6, by + 18), font, 0.42, (255, 255, 255), 1)

    # Hint
    if edit_mode == "liquify":
        count = liquify.get_anchor_count()
        hint  = f"{count} warp{'s' if count != 1 else ''} active  |  drag on face to warp"
    else:
        count = spotHeal.get_spot_count()
        hint  = f"{count} spot{'s' if count != 1 else ''} healed  |  click on spot to heal"
    cv2.putText(canvas, hint, (EDIT_HINT_X, ctrl_y + 5), font, 0.36, (120, 120, 120), 1)

    # Drag preview for liquify
    if edit_mode == "liquify" and _drag_start is not None:
        cv2.circle(canvas, _drag_start, 6, (100, 200, 100), 1)

def draw_canvas(main_frame, thumbs, filter_list):
    total_h = CANVAS_H + (MAKEUP_H if makeup_on else 0) + (EDIT_H if edit_on else 0)
    canvas = np.zeros((total_h, CANVAS_W, 3), dtype=np.uint8)
    canvas[MAIN_H:CANVAS_H, :] = (28, 28, 28)

    fh, fw = main_frame.shape[:2]
    ox = (CANVAS_W - fw) // 2
    oy = (MAIN_H - fh) // 2
    canvas[oy:oy + fh, ox:ox + fw] = main_frame

    draw_buttons(canvas)

    n = len(filter_list)
    for i, (name, _) in enumerate(filter_list):
        tx = get_thumb_x(i, n)
        if thumbs[i] is not None:
            canvas[THUMB_Y:THUMB_Y + THUMB_H, tx:tx + THUMB_W] = thumbs[i]
        border_color = (255, 255, 255) if i == selected else (90, 90, 90)
        border_t = 3 if i == selected else 1
        pad      = 3 if i == selected else 1
        cv2.rectangle(canvas, (tx - pad, THUMB_Y - pad),
                      (tx + THUMB_W + pad - 1, THUMB_Y + THUMB_H + pad - 1), border_color, border_t)
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx = tx + (THUMB_W - tw) // 2
        cv2.putText(canvas, name, (lx, THUMB_Y + THUMB_H + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    if makeup_on:
        draw_makeup_controls(canvas)
    if edit_on:
        draw_edit_controls(canvas)

    return canvas, ox, oy

# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------
def apply_filters(frame, filter_list):
    global _current_landmarks
    _, color_fn = filter_list[selected]
    output = color_fn(frame)

    if beauty_on:
        output = beauty_apply(output)

    landmarks = None
    if overlay_on or makeup_on or edit_on:
        if beauty_on:
            landmarks = get_last_landmarks()
        else:
            landmarks, _ = get_landmarks_and_bbox(output)

    if landmarks is not None:
        _current_landmarks = landmarks

    if overlay_on:
        output = apply_cat_ears(output, landmarks)
    if makeup_on:
        blush_hue = BLUSH_PRESETS[makeup_state["blush_idx"]][1]
        lip_hue   = LIP_PRESETS[makeup_state["lip_idx"]][1]
        output = makeup_apply(output, landmarks,
                              blush_hue, makeup_state["blush_op"],
                              lip_hue,   makeup_state["lip_op"])
    if edit_on:
        output = liquify.apply(output, landmarks)
        output = spotHeal.apply(output, landmarks)

    if symmetry_on:
        if landmarks is None:
            landmarks, _ = get_landmarks_and_bbox(output)
        if landmarks is not None:
            output = symmetry.draw(output, landmarks)

    return output

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    global _frame_ox, _frame_oy

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Photo Booth running.")
    print("  Toggle buttons | s = save | r = reload | q = quit")

    cv2.namedWindow("Photo Booth")
    cv2.setMouseCallback("Photo Booth", mouse_callback)

    thumbs      = [None] * 5
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filter_list = get_filter_list()

        fh, fw = frame.shape[:2]
        scale      = min(CANVAS_W / fw, MAIN_H / fh)
        main_size  = (int(fw * scale), int(fh * scale))
        main_frame = cv2.resize(frame, main_size)

        _frame_ox = (CANVAS_W - main_size[0]) // 2
        _frame_oy = (MAIN_H   - main_size[1]) // 2

        try:
            main_output = apply_filters(main_frame, filter_list)
        except Exception as e:
            print(f"Filter error: {e}")
            main_output = main_frame.copy()

        if frame_count % THUMB_UPDATE_EVERY == 0:
            small = cv2.resize(frame, (THUMB_W, THUMB_H))
            for i, (_, fn) in enumerate(filter_list):
                try:
                    thumbs[i] = fn(small)
                except Exception:
                    thumbs[i] = small.copy()

        canvas, ox, oy = draw_canvas(main_output, thumbs, filter_list)
        cv2.imshow("Photo Booth", canvas)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("snapshot.jpg", main_output)
            print("Snapshot saved.")
        elif key == ord('r'):
            reload_filters()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()