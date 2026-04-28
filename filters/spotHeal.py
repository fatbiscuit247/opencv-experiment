import cv2
import numpy as np

_spots = []  # list of {lm_idx, offset_x, offset_y, radius}

def clear_spots():
    global _spots
    _spots = []

def get_spot_count():
    return len(_spots)

def undo_spot():
    if _spots:
        _spots.pop()

def add_spot(landmarks, frame_x, frame_y, radius):
    if not landmarks:
        return

    pts = np.array(landmarks, dtype=np.float32)
    click = np.array([frame_x, frame_y], dtype=np.float32)
    dists = np.linalg.norm(pts - click, axis=1)
    nearest_idx = int(np.argmin(dists))

    lm = landmarks[nearest_idx]
    _spots.append({
        "lm_idx":   nearest_idx,
        "offset_x": frame_x - lm[0],
        "offset_y": frame_y - lm[1],
        "radius":   radius,
    })

def _heal_spot(frame, cx, cy, radius):
    """Sample a ring of pixels around (cx, cy) and blend over the spot."""
    h, w = frame.shape[:2]
    cx, cy = int(cx), int(cy)
    r = int(radius)

    # Sample ring: pixels between radius and radius*1.8
    outer = int(radius * 1.2)
    y0 = max(0, cy - outer)
    y1 = min(h, cy + outer + 1)
    x0 = max(0, cx - outer)
    x1 = min(w, cx + outer + 1)

    region = frame[y0:y1, x0:x1].astype(np.float32)
    rh, rw = region.shape[:2]

    # Build ring mask (annulus between inner and outer radius)
    ys, xs = np.mgrid[0:rh, 0:rw]
    cy_r = cy - y0
    cx_r = cx - x0
    dist = np.sqrt((xs - cx_r) ** 2 + (ys - cy_r) ** 2)
    ring_mask = (dist >= r) & (dist <= outer)

    if ring_mask.sum() < 10:
        return frame

    # Average color of the ring (surrounding skin)
    ring_pixels = region[ring_mask]
    brightness = ring_pixels.mean(axis=1)
    skin_pixels = ring_pixels[brightness > 60]
    skin_color = np.median(skin_pixels, axis=0) if len(skin_pixels) > 5 else np.median(ring_pixels, axis=0)

    # Build soft feathered heal mask over the spot
    heal_mask = np.clip(1.0 - dist / r, 0, 1) ** 2
    heal_mask[dist > r] = 0
    heal_mask_3ch = np.stack([heal_mask] * 3, axis=-1)

    output = frame.copy().astype(np.float32)

    # Build a skin tone mask — only blend where pixels look like skin
    region_gray = region.mean(axis=2)
    skin_tone_mask = (region_gray > 60).astype(np.float32)  # exclude dark bg/hair
    skin_tone_mask = cv2.GaussianBlur(skin_tone_mask, (5, 5), 0)
    skin_tone_3ch = np.stack([skin_tone_mask] * 3, axis=-1)

    skin_layer = np.full_like(region, skin_color)
    combined_mask = heal_mask_3ch * skin_tone_3ch
    output[y0:y1, x0:x1] = (region * (1 - combined_mask) +
                            skin_layer * combined_mask)

    return output.astype(np.uint8)

def apply(frame, landmarks):
    if not _spots or landmarks is None:
        return frame

    output = frame.copy()
    for spot in _spots:
        lm = landmarks[spot["lm_idx"]]
        ax = float(lm[0]) + spot["offset_x"]
        ay = float(lm[1]) + spot["offset_y"]
        output = _heal_spot(output, ax, ay, spot["radius"])

    return output