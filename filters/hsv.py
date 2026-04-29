import cv2
import numpy as np

def _hsv_adjust(frame, h_shift=0, s_scale=1.0, v_scale=1.0):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def warm(frame):
    """Warm tone: shift hue toward red/orange, boost reds, reduce blues."""
    result = _hsv_adjust(frame, h_shift=-1, s_scale=1.1, v_scale=1.09)
    b, g, r = cv2.split(result)
    r = np.clip(r.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def cool(frame):
    """Cool tone: shift hue toward blue/cyan, reduce reds."""
    result = _hsv_adjust(frame, h_shift=2, s_scale=0.75, v_scale=1.08)
    b, g, r = cv2.split(result)
    b = np.clip(b.astype(np.float32) * 1.12, 0, 255).astype(np.uint8)
    r = np.clip(r.astype(np.float32) * 0.88, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def vivid(frame):
    """Vivid: strong saturation boost with slight brightness lift."""
    return _hsv_adjust(frame, s_scale=1.05, v_scale=1.2)

def fade(frame):
    """Fade: desaturated with lifted shadows, matte film look."""
    result = _hsv_adjust(frame, s_scale=0.45, v_scale=1)
    overlay = np.full_like(result, 35)
    return cv2.addWeighted(result, 0.82, overlay, 0.18, 0)