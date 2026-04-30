import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe 478-point landmark indices for symmetry analysis
# ---------------------------------------------------------------------------

# Feature contour indices (MediaPipe 478-point)
RIGHT_EYE_CONTOUR  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR   = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_BROW_CONTOUR = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LEFT_BROW_CONTOUR  = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
MOUTH_CONTOUR      = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSE_CONTOUR = [129, 98, 97, 2, 326, 327, 358, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64, 98, 129]
FACE_CORNERS        = [234, 454]                         # left/right cheek edges
JAW_OUTLINE = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 
               152, 
               377, 400, 378, 379, 365, 397, 288, 361, 323, 454]

# Iris centers (478-point model specific)
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER  = 473

# Feature pairs: (label, right_indices, left_indices)
# Averaged to a single center point per side before comparing
FEATURE_PAIRS = [
    ("Eyes",     [33,  133],  [263, 362]),   # outer/inner eye corners
    ("Brows",    [46,  107],  [276, 336]),   # outer/inner brow points
    ("Mouth",    [61],        [291]),         # mouth corners
    ("Nose",     [129],       [358]),         # nostril wings
]

MAX_DEVIATION = 20.0   # pixels — clamp for 0–100 scoring

COLOR_AXIS       = (210, 210, 210)   # soft white for dashed line
COLOR_SYMMETRIC  = (80,  200,  80)   # green
COLOR_ASYMMETRIC = (60,  120, 255)   # orange-red (BGR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

dark_mode = False

def toggle_dark_mode():
    global dark_mode
    dark_mode = not dark_mode

def _avg_point(landmarks, indices):
    pts = np.array([landmarks[i] for i in indices], dtype=np.float32)
    return pts.mean(axis=0)


def _draw_dashed_line(img, pt1, pt2, color, thickness=1, dash=10, gap=6):
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    total = np.hypot(x2 - x1, y2 - y1)
    if total == 0:
        return
    dx, dy = (x2 - x1) / total, (y2 - y1) / total
    pos, draw = 0.0, True
    while pos < total:
        end = min(pos + (dash if draw else gap), total)
        if draw:
            p1 = (int(x1 + dx * pos), int(y1 + dy * pos))
            p2 = (int(x1 + dx * end), int(y1 + dy * end))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        pos, draw = end, not draw


def _lerp_color(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze(landmarks, frame_shape):
    """
    Compute symmetry axis and per-feature deviations.

    Returns:
        results  -- list of (label, r_pt, l_pt, deviation_px)
        score    -- int 0–100 (100 = perfectly symmetric)
        axis_x   -- int pixel x of symmetry axis
    Or (None, None, None) if landmarks are unusable.
    """
    if landmarks is None or len(landmarks) < 478:
        return None, None, None

    r_iris = np.array(landmarks[RIGHT_IRIS_CENTER], dtype=np.float32)
    l_iris = np.array(landmarks[LEFT_IRIS_CENTER],  dtype=np.float32)
    axis_x = int((r_iris[0] + l_iris[0]) / 2)

    results    = []
    deviations = []

    for label, right_idx, left_idx in FEATURE_PAIRS:
        r_pt = _avg_point(landmarks, right_idx)
        l_pt = _avg_point(landmarks, left_idx)

        # How far each side sits from the axis
        r_dist   = axis_x - r_pt[0]   # person's right side
        l_dist   = l_pt[0] - axis_x   # person's left side
        deviation = abs(r_dist - l_dist)

        deviations.append(deviation)
        results.append((label, r_pt, l_pt, deviation))

    score = max(0, int(100 - (np.mean(deviations) / MAX_DEVIATION) * 100))
    return results, score, axis_x

def _draw_hud(output, results):
    panel_x     = output.shape[1] - 160
    panel_y     = 10
    row_h       = 22
    panel_h     = len(results) * row_h + 16
    bar_max_w   = 80
    bar_h       = 8

    # Semi-transparent dark background
    overlay = output.copy()
    cv2.rectangle(overlay, (panel_x - 8, panel_y),
                  (panel_x + 148, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)

    for i, (label, r_pt, l_pt, deviation) in enumerate(results):
        t         = min(deviation / MAX_DEVIATION, 1.0)
        score     = int((1 - t) * 100)
        color     = _lerp_color(COLOR_SYMMETRIC, COLOR_ASYMMETRIC, t)
        row_y     = panel_y + 12 + i * row_h

        # Label
        cv2.putText(output, label, (panel_x - 4, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

        # Bar background
        bar_x = panel_x + 40
        cv2.rectangle(output, (bar_x, row_y - 7),
                      (bar_x + bar_max_w, row_y - 7 + bar_h), (60, 60, 60), -1)

        # Bar fill
        fill_w = int(bar_max_w * (1 - t))
        if fill_w > 0:
            cv2.rectangle(output, (bar_x, row_y - 7),
                          (bar_x + fill_w, row_y - 7 + bar_h), color, -1)

        # Percentage
        cv2.putText(output, f"{score}%", (bar_x + bar_max_w + 5, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA)
        
def _draw_contours(output, landmarks, results):
    scores = {label: deviation for label, _, _, deviation in results}

    # --- Dark mode: dense contours with dashed lines ---
    if dark_mode:
        t     = min(np.mean([d for _, _, _, d in results]) / MAX_DEVIATION, 1.0)
        color = _lerp_color(COLOR_SYMMETRIC, COLOR_ASYMMETRIC, t)
        for i in range(468):
            cv2.circle(output, landmarks[i], 1, color, -1, cv2.LINE_AA)
        return

    # --- Light mode: sparse dots ---
    feature_groups = [
        ("Brows", [46, 53, 65, 70, 107],              [276, 283, 295, 300, 336]),
        ("Eyes",  [33, 159, 133, 145],                 [263, 386, 362, 374]),
        ("Nose",  [129, 98, 2, 327, 358, 122, 351],   None),
        ("Mouth", [61, 37, 0, 267, 291, 17, 61],       None),
    ]

    for label, right_pts, left_pts in feature_groups:
        dev   = scores.get(label, 0)
        t     = min(dev / MAX_DEVIATION, 1.0)
        color = _lerp_color(COLOR_SYMMETRIC, COLOR_ASYMMETRIC, t)

        for pt_group in ([right_pts, left_pts] if left_pts else [right_pts]):
            if pt_group is None:
                continue
            for i in pt_group:
                cv2.circle(output, landmarks[i], 1, color, -1, cv2.LINE_AA)

    # Jawline
    t     = min(np.mean([d for _, _, _, d in results]) / MAX_DEVIATION, 1.0)
    color = _lerp_color(COLOR_SYMMETRIC, COLOR_ASYMMETRIC, t)
    for i in JAW_OUTLINE:
        cv2.circle(output, landmarks[i], 2, color, -1, cv2.LINE_AA)

    # Hairline
    hairline = [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]
    for i in hairline:
        cv2.circle(output, landmarks[i], 2, color, -1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(frame, landmarks):
    if landmarks is None:
        return frame

    results, score, axis_x = analyze(landmarks, frame.shape)
    if results is None:
        return frame

    output = frame.copy()
    if dark_mode:
        output = np.zeros_like(frame)

    if not dark_mode:
        face_top    = landmarks[10][1]
        face_bottom = landmarks[152][1]
        _draw_dashed_line(output, (axis_x, face_top), (axis_x, face_bottom),
                          COLOR_AXIS, thickness=1, dash=12, gap=7)

    h, w = frame.shape[:2]

    # --- Feature contours ---
    _draw_contours(output, landmarks, results)

    # --- Symmetry score ---
    score_text = f"Symmetry  {score}%"
    cv2.putText(output, score_text, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, score_text, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    # --- HUD panel ---
    _draw_hud(output, results)

    return output