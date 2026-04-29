import cv2
import numpy as np

_anchors = []  # list of {lm_idx, offset_x, offset_y, warp_dx, warp_dy, radius}

def clear_anchors():
    global _anchors
    _anchors = []

def get_anchor_count():
    return len(_anchors)

def add_anchor(landmarks, frame_x, frame_y, warp_dx, warp_dy, radius):
    """
    Add a warp anchor at (frame_x, frame_y) with the given displacement.
    Stores position relative to nearest landmark so it tracks face movement.
    """
    if not landmarks:
        return

    pts = np.array(landmarks, dtype=np.float32)
    click = np.array([frame_x, frame_y], dtype=np.float32)
    dists = np.linalg.norm(pts - click, axis=1)
    nearest_idx = int(np.argmin(dists))

    lm = landmarks[nearest_idx]
    _anchors.append({
        "lm_idx":   nearest_idx,
        "offset_x": frame_x - lm[0],
        "offset_y": frame_y - lm[1],
        "warp_dx":  float(warp_dx) * 0.065,
        "warp_dy":  float(warp_dy) * 0.065,
        "radius":   float(radius),
    })

def apply(frame, landmarks):
    if not _anchors or landmarks is None:
        return frame

    h, w = frame.shape[:2]
    map_x = np.arange(w, dtype=np.float32)
    map_y = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(map_x, map_y)

    for anchor in _anchors:
        lm = landmarks[anchor["lm_idx"]]
        # Recompute absolute position using current landmark + stored offset
        ax = float(lm[0]) + anchor["offset_x"]
        ay = float(lm[1]) + anchor["offset_y"]

        radius = anchor["radius"]
        pixel_dist = np.sqrt((map_x - ax) ** 2 + (map_y - ay) ** 2)
        weight = np.clip(1.0 - pixel_dist / radius, 0, 1) ** 2

        # Backward remap: source pixel is pulled opposite to drag direction
        map_x -= anchor["warp_dx"] * weight
        map_y -= anchor["warp_dy"] * weight

    return cv2.remap(frame, map_x, map_y,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def undo_anchor():
    if _anchors:
        _anchors.pop()