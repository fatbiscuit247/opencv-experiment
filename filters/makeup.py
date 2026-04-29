import cv2
import numpy as np

# MediaPipe 478-point landmark indices
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             409, 270, 269, 267, 0, 37, 39, 40, 185]

RIGHT_CHEEK = 50
LEFT_CHEEK  = 280
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
             415, 310, 311, 312, 13, 82, 81, 80, 191]

def _hue_to_bgr(hue):
    """Convert HSV hue (0-179) to a BGR color."""
    hsv = np.uint8([[[hue, 200, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0][0])

def apply_blush(frame, landmarks, hue=160, opacity=40):
    if landmarks is None or opacity == 0:
        return frame

    rc = landmarks[RIGHT_CHEEK]
    lc = landmarks[LEFT_CHEEK]
    face_w = abs(lc[0] - rc[0])
    radius = max(10, int(face_w * 0.18))
    color = _hue_to_bgr(hue)

    # Build a soft feathered mask for both cheeks
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, rc, radius, 255, -1)
    cv2.circle(mask, lc, radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), radius // 2)
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

    alpha = (opacity / 100.0) * 0.55
    blush_layer = np.full_like(frame, color, dtype=np.float32)
    result = frame.astype(np.float32) * (1 - mask_3ch * alpha) + blush_layer * (mask_3ch * alpha)
    return result.astype(np.uint8)

def apply_lipstick(frame, landmarks, hue=0, opacity=60):
    if landmarks is None or opacity == 0:
        return frame

    pts = np.array([landmarks[i] for i in OUTER_LIP], dtype=np.int32)
    color = _hue_to_bgr(hue)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Cut out the inner mouth opening so teeth stay unaffected
    inner_pts = np.array([landmarks[i] for i in INNER_LIP], dtype=np.int32)
    cv2.fillPoly(mask, [inner_pts], 0)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

    alpha = (opacity / 100.0) * 0.65
    lip_layer = np.full_like(frame, color, dtype=np.float32)
    result = frame.astype(np.float32) * (1 - mask_3ch * alpha) + lip_layer * (mask_3ch * alpha)
    return result.astype(np.uint8)

def apply(frame, landmarks, blush_hue=160, blush_opacity=40, lip_hue=0, lip_opacity=60):
    output = apply_blush(frame, landmarks, blush_hue, blush_opacity)
    output = apply_lipstick(output, landmarks, lip_hue, lip_opacity)
    return output