import cv2
import numpy as np
import os

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

_cat_ears_img = None

def _get_cat_ears():
    global _cat_ears_img
    if _cat_ears_img is None:
        path = os.path.join(_ASSETS_DIR, "cat_ears.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: cat_ears.png not found at {path}")
        _cat_ears_img = img
    return _cat_ears_img

def _overlay_png(background, overlay, x, y, w, h):
    """Paste a 4-channel PNG onto background at (x, y) scaled to (w, h)."""
    if overlay is None:
        return background

    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    bg_h, bg_w = background.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return background

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    overlay_crop = overlay_resized[oy1:oy2, ox1:ox2]

    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
        bgr = overlay_crop[:, :, :3]
    else:
        alpha = np.ones((overlay_crop.shape[0], overlay_crop.shape[1], 1), dtype=np.float32)
        bgr = overlay_crop

    result = background.copy()
    bg_region = result[y1:y2, x1:x2].astype(np.float32)
    blended = bg_region * (1 - alpha) + bgr.astype(np.float32) * alpha
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result

def apply_cat_ears(frame, landmarks):
    """Position cat ears above the head using MediaPipe landmarks."""
    if landmarks is None:
        return frame

    cat_ears = _get_cat_ears()
    if cat_ears is None:
        return frame

    # Landmark 10 = top of forehead, 234 = left temple, 454 = right temple
    left_temple  = landmarks[234]
    right_temple = landmarks[454]
    top_forehead = landmarks[10]

    face_width = right_temple[0] - left_temple[0]
    if face_width <= 0:
        return frame

    # Scale ears to 1.4x face width, preserve aspect ratio
    ears_w = int(face_width * 1.62)
    orig_h, orig_w = cat_ears.shape[:2]
    ears_h = int(ears_w * orig_h / orig_w)

    # Center horizontally on face, sit above the forehead
    ears_x = left_temple[0] - (ears_w - face_width) // 2
    ears_y = top_forehead[1] - int(ears_h * 0.57)

    return _overlay_png(frame, cat_ears, ears_x, ears_y, ears_w, ears_h)