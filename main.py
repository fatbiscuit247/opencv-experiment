import cv2
cv2.ocl.setUseOpenCL(True)
import numpy as np
import sys
import filters
from filters.beauty import apply as beauty_apply, get_last_landmarks, get_landmarks_and_bbox
from filters.overlays import apply_cat_ears
from filters.makeup import apply as makeup_apply

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CANVAS_W      = 700
MAIN_H        = 480
STRIP_H       = 150
CANVAS_H      = MAIN_H + STRIP_H
MAKEUP_H      = 90          # extra height appended when makeup is ON
THUMB_W       = 96
THUMB_H       = 72
THUMB_PADDING = 10
THUMB_Y       = MAIN_H + 60

BTN_W, BTN_H  = 130, 30
BTN_GAP       = 8
BTN_Y         = MAIN_H + 12
BTN1_X        = (CANVAS_W - BTN_W * 3 - BTN_GAP * 2) // 2
BTN2_X        = BTN1_X + BTN_W + BTN_GAP
BTN3_X        = BTN2_X + BTN_W + BTN_GAP

THUMB_UPDATE_EVERY = 15

# ---------------------------------------------------------------------------
# Makeup presets
# ---------------------------------------------------------------------------

BLUSH_PRESETS   = [("Red", 0), ("Rose", 170), ("Pink", 160), ("Berry", 145), ("Nude", 15)]
LIP_PRESETS   = [("Red", 0), ("Rose", 170), ("Pink", 160), ("Berry", 145), ("Nude", 15)]

SWATCH_R      = 11
SWATCH_GAP    = 28
SWATCH_START_X = 90
BLUSH_ROW_Y   = CANVAS_H + 22
LIP_ROW_Y     = CANVAS_H + 62
OP_LABEL_X    = 430
OP_BTN_X      = 490   # "-" button
OP_BTN2_X     = 530   # "+" button
OP_BTN_W      = 28
OP_BTN_H      = 22

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
selected   = 0
beauty_on  = False
overlay_on = False
makeup_on  = False

makeup_state = {
    "blush_idx": 0,
    "lip_idx":   0,
    "blush_op":  40,
    "lip_op":    60,
}

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

# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global selected, beauty_on, overlay_on, makeup_on

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Toggle buttons
    if BTN1_X <= x <= BTN1_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        beauty_on = not beauty_on
        return
    if BTN2_X <= x <= BTN2_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        overlay_on = not overlay_on
        return
    if BTN3_X <= x <= BTN3_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        makeup_on = not makeup_on
        return

    # Color filter thumbnails
    filter_list = get_filter_list()
    if THUMB_Y <= y <= THUMB_Y + THUMB_H:
        for i in range(len(filter_list)):
            tx = get_thumb_x(i, len(filter_list))
            if tx <= x <= tx + THUMB_W:
                selected = i
                return

    # Makeup controls (only when makeup is ON)
    if makeup_on:
        _handle_makeup_click(x, y)

def _handle_makeup_click(x, y):
    # Blush swatches
    for i in range(len(BLUSH_PRESETS)):
        sx = SWATCH_START_X + i * SWATCH_GAP
        if abs(x - sx) <= SWATCH_R and abs(y - BLUSH_ROW_Y) <= SWATCH_R:
            makeup_state["blush_idx"] = i
            return

    # Lip swatches
    for i in range(len(LIP_PRESETS)):
        sx = SWATCH_START_X + i * SWATCH_GAP
        if abs(x - sx) <= SWATCH_R and abs(y - LIP_ROW_Y) <= SWATCH_R:
            makeup_state["lip_idx"] = i
            return

    # Blush opacity buttons
    by = BLUSH_ROW_Y - OP_BTN_H // 2
    if OP_BTN_X <= x <= OP_BTN_X + OP_BTN_W and by <= y <= by + OP_BTN_H:
        makeup_state["blush_op"] = max(0, makeup_state["blush_op"] - 10)
        return
    if OP_BTN2_X <= x <= OP_BTN2_X + OP_BTN_W and by <= y <= by + OP_BTN_H:
        makeup_state["blush_op"] = min(100, makeup_state["blush_op"] + 10)
        return

    # Lip opacity buttons
    ly = LIP_ROW_Y - OP_BTN_H // 2
    if OP_BTN_X <= x <= OP_BTN_X + OP_BTN_W and ly <= y <= ly + OP_BTN_H:
        makeup_state["lip_op"] = max(0, makeup_state["lip_op"] - 10)
        return
    if OP_BTN2_X <= x <= OP_BTN2_X + OP_BTN_W and ly <= y <= ly + OP_BTN_H:
        makeup_state["lip_op"] = min(100, makeup_state["lip_op"] + 10)
        return

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_buttons(canvas):
    for bx, label, active in [
        (BTN1_X, "Beauty: ON"   if beauty_on  else "Beauty: OFF",   beauty_on),
        (BTN2_X, "Cat Ears: ON" if overlay_on else "Cat Ears: OFF",  overlay_on),
        (BTN3_X, "Makeup: ON"   if makeup_on  else "Makeup: OFF",   makeup_on),
    ]:
        color = (60, 160, 60) if active else (60, 60, 60)
        cv2.rectangle(canvas, (bx, BTN_Y), (bx + BTN_W, BTN_Y + BTN_H), color, -1)
        cv2.rectangle(canvas, (bx, BTN_Y), (bx + BTN_W, BTN_Y + BTN_H), (200, 200, 200), 1)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lx = bx + (BTN_W - tw) // 2
        cv2.putText(canvas, label, (lx, BTN_Y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

def draw_makeup_controls(canvas):
    """Draw inline blush + lip controls below the main strip."""
    panel_y = CANVAS_H
    cv2.rectangle(canvas, (0, panel_y), (CANVAS_W, panel_y + MAKEUP_H), (20, 20, 20), -1)
    cv2.line(canvas, (0, panel_y), (CANVAS_W, panel_y), (60, 60, 60), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for row_label, presets, row_y, idx_key, op_key in [
        ("Blush", BLUSH_PRESETS, BLUSH_ROW_Y, "blush_idx", "blush_op"),
        ("Lips",  LIP_PRESETS,   LIP_ROW_Y,   "lip_idx",   "lip_op"),
    ]:
        # Row label
        cv2.putText(canvas, row_label, (10, row_y + 5), font, 0.45, (180, 180, 180), 1)

        # Color swatches
        for i, (name, hue) in enumerate(presets):
            sx = SWATCH_START_X + i * SWATCH_GAP
            bgr = _hue_to_bgr(hue)
            cv2.circle(canvas, (sx, row_y), SWATCH_R, bgr, -1)
            if makeup_state[idx_key] == i:
                cv2.circle(canvas, (sx, row_y), SWATCH_R + 3, (255, 255, 255), 2)

        # Opacity label
        op_val = makeup_state[op_key]
        cv2.putText(canvas, f"Opacity: {op_val}%", (OP_LABEL_X, row_y + 5),
                    font, 0.42, (180, 180, 180), 1)

        # "-" and "+" buttons
        by = row_y - OP_BTN_H // 2
        cv2.rectangle(canvas, (OP_BTN_X, by), (OP_BTN_X + OP_BTN_W, by + OP_BTN_H), (70, 70, 70), -1)
        cv2.putText(canvas, "-", (OP_BTN_X + 8, by + 16), font, 0.55, (255, 255, 255), 1)
        cv2.rectangle(canvas, (OP_BTN2_X, by), (OP_BTN2_X + OP_BTN_W, by + OP_BTN_H), (70, 70, 70), -1)
        cv2.putText(canvas, "+", (OP_BTN2_X + 6, by + 16), font, 0.55, (255, 255, 255), 1)

def draw_canvas(main_frame, thumbs, filter_list):
    total_h = CANVAS_H + (MAKEUP_H if makeup_on else 0)
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
        if i == selected:
            cv2.rectangle(canvas, (tx-3, THUMB_Y-3), (tx+THUMB_W+2, THUMB_Y+THUMB_H+2), (255, 255, 255), 3)
        else:
            cv2.rectangle(canvas, (tx-1, THUMB_Y-1), (tx+THUMB_W, THUMB_Y+THUMB_H), (90, 90, 90), 1)
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx = tx + (THUMB_W - tw) // 2
        cv2.putText(canvas, name, (lx, THUMB_Y + THUMB_H + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    if makeup_on:
        draw_makeup_controls(canvas)

    return canvas

# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------
def apply_filters(frame, filter_list):
    _, color_fn = filter_list[selected]
    output = color_fn(frame)

    if beauty_on:
        output = beauty_apply(output)

    landmarks = None
    if overlay_on or makeup_on:
        if beauty_on:
            landmarks = get_last_landmarks()
        else:
            landmarks, _ = get_landmarks_and_bbox(output)

    if overlay_on:
        output = apply_cat_ears(output, landmarks)

    if makeup_on:
        blush_hue = BLUSH_PRESETS[makeup_state["blush_idx"]][1]
        lip_hue   = LIP_PRESETS[makeup_state["lip_idx"]][1]
        output = makeup_apply(output, landmarks,
                              blush_hue, makeup_state["blush_op"],
                              lip_hue,   makeup_state["lip_op"])
    return output

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Photo Booth running.")
    print("  Click filter thumbnail | toggle buttons | s = save | r = reload | q = quit")

    cv2.namedWindow("Photo Booth")
    cv2.setMouseCallback("Photo Booth", mouse_callback)

    thumbs = [None] * 5
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filter_list = get_filter_list()

        fh, fw = frame.shape[:2]
        scale = min(CANVAS_W / fw, MAIN_H / fh)
        main_size = (int(fw * scale), int(fh * scale))
        main_frame = cv2.resize(frame, main_size)

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

        canvas = draw_canvas(main_output, thumbs, filter_list)
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