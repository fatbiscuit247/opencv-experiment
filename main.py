import cv2
cv2.ocl.setUseOpenCL(True)
import numpy as np
import importlib
import sys
import filters
from filters.beauty import apply as beauty_apply, get_last_landmarks, get_landmarks_and_bbox
from filters.overlays import apply_cat_ears





# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CANVAS_W      = 700
MAIN_H        = 480
STRIP_H       = 150
CANVAS_H      = MAIN_H + STRIP_H
THUMB_W       = 96
THUMB_H       = 72
THUMB_PADDING = 10
THUMB_Y       = MAIN_H + 60

BTN_W, BTN_H  = 150, 30
BTN_GAP       = 10
BTN_Y         = MAIN_H + 12
BTN1_X        = (CANVAS_W - BTN_W * 2 - BTN_GAP) // 2
BTN2_X        = BTN1_X + BTN_W + BTN_GAP

THUMB_UPDATE_EVERY = 15

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
selected  = 0
beauty_on = False
overlay_on = False

def get_filter_list():
    return [
        ("Original", filters.original),
        ("Warm",     filters.warm),
        ("Cool",     filters.cool),
        ("Vivid",    filters.vivid),
        ("Fade",     filters.fade),
    ]

def reload_filters():
    global filters
    for mod in list(sys.modules.keys()):
        if mod.startswith('filters'):
            del sys.modules[mod]
    import filters as filters
    print("Filters reloaded.")

def get_thumb_x(i, n):
    total_w = n * THUMB_W + (n - 1) * THUMB_PADDING
    start_x = (CANVAS_W - total_w) // 2
    return start_x + i * (THUMB_W + THUMB_PADDING)

def mouse_callback(event, x, y, flags, param):
    global selected, beauty_on, overlay_on
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if BTN1_X <= x <= BTN1_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        beauty_on = not beauty_on
        return
    if BTN2_X <= x <= BTN2_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
        overlay_on = not overlay_on
        return
    filter_list = get_filter_list()
    if y >= THUMB_Y:
        for i in range(len(filter_list)):
            tx = get_thumb_x(i, len(filter_list))
            if tx <= x <= tx + THUMB_W and THUMB_Y <= y <= THUMB_Y + THUMB_H:
                selected = i
                break

def draw_buttons(canvas):
    for bx, label, active in [
        (BTN1_X, "Beauty: ON" if beauty_on else "Beauty: OFF", beauty_on),
        (BTN2_X, "Cat Ears: ON" if overlay_on else "Cat Ears: OFF", overlay_on),
    ]:
        color = (60, 160, 60) if active else (60, 60, 60)
        cv2.rectangle(canvas, (bx, BTN_Y), (bx + BTN_W, BTN_Y + BTN_H), color, -1)
        cv2.rectangle(canvas, (bx, BTN_Y), (bx + BTN_W, BTN_Y + BTN_H), (200, 200, 200), 1)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx = bx + (BTN_W - tw) // 2
        cv2.putText(canvas, label, (lx, BTN_Y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_canvas(main_frame, thumbs, filter_list):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[MAIN_H:, :] = (28, 28, 28)

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

    return canvas

def apply_filters(frame, filter_list):
    _, color_fn = filter_list[selected]
    output = color_fn(frame)
    if beauty_on:
        output = beauty_apply(output)
    if overlay_on:
        if beauty_on:
            landmarks = get_last_landmarks()
        else:
            landmarks, _ = get_landmarks_and_bbox(output)
        output = apply_cat_ears(output, landmarks)
    return output

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Photo Booth running.")
    print("  Click filter thumbnail | click Beauty button | s = save | r = reload | q = quit")

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