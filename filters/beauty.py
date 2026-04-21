import cv2
import numpy as np
import dlib

# ---------------------------------------------------------------------------
# Landmark index ranges (dlib 68-point model)
# ---------------------------------------------------------------------------
JAW         = list(range(0, 17))
RIGHT_BROW  = list(range(17, 22))
LEFT_BROW   = list(range(22, 27))
RIGHT_EYE   = list(range(36, 42))
LEFT_EYE    = list(range(42, 48))
OUTER_LIP   = list(range(48, 60))
INNER_LIP   = list(range(60, 68))

_face_net  = None
_predictor = None

def _get_face_net():
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000.caffemodel"
        )
    return _face_net

def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = dlib.shape_predictor(
            "models/shape_predictor_68_face_landmarks.dat"
        )
    return _predictor

def detect_faces(frame):
    net = _get_face_net()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

def get_landmarks(frame, face):
    predictor = _get_predictor()
    x, y, w, h = face
    rect = dlib.rectangle(x, y, x + w, y + h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, rect)
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]

def _points_to_mask(shape, indices, frame_shape, padding=0):
    pts = np.array([shape[i] for i in indices], dtype=np.int32)
    if padding > 0:
        center = pts.mean(axis=0)
        pts = (pts + (pts - center) * padding).astype(np.int32)
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    return mask

def build_skin_mask(frame, face, landmarks):
    x, y, w, h = face
    face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    center = (x + w // 2, y + h // 2)
    cv2.ellipse(face_mask, center, (w // 2, h // 2), 0, 0, 360, 255, -1)
    for indices, padding in [
        (RIGHT_EYE,  0.35), (LEFT_EYE,   0.35),
        (RIGHT_BROW, 0.2),  (LEFT_BROW,  0.2),
        (OUTER_LIP,  0.15), (INNER_LIP,  0.15),
    ]:
        feature_mask = _points_to_mask(landmarks, indices, frame.shape, padding)
        face_mask = cv2.subtract(face_mask, feature_mask)
    face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
    return face_mask

# ---------------------------------------------------------------------------
# Face slimming via landmark-guided warp via cv2.remap
# ---------------------------------------------------------------------------
SLIM_STRENGTH = 0.011  # 0.0 = no effect, 0.2 = strong, tweak here

def slim_face(frame, landmarks):
    """
    Warps jaw landmarks inward toward the face center using cv2.remap.
    Only the jaw region (points 0-16) is pulled in — cheekbones, chin etc.
    The warp uses a radial falloff so the effect fades naturally away from
    each jaw point, avoiding sharp distortion edges.
    """
    h, w = frame.shape[:2]

    # Build identity map (each pixel maps to itself)
    map_x = np.arange(w, dtype=np.float32)
    map_y = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(map_x, map_y)

    # Face center — used as the direction to pull jaw points toward
    all_pts = np.array(landmarks, dtype=np.float32)
    face_center = all_pts.mean(axis=0)

    # For each jaw point, apply a localized inward pull
    jaw_pts = [landmarks[i] for i in JAW]
    for pt in jaw_pts:
        px, py = float(pt[0]), float(pt[1])

        # Direction vector: from jaw point toward face center
        dx = face_center[0] - px
        dy = face_center[1] - py
        dist_to_center = np.sqrt(dx * dx + dy * dy) + 1e-6
        nx, ny = dx / dist_to_center, dy / dist_to_center

        # Influence radius — scales with face size
        radius = dist_to_center * 0.55

        # Distance of every pixel from this jaw point
        pixel_dist = np.sqrt((map_x - px) ** 2 + (map_y - py) ** 2)

        # Smooth falloff: pixels close to jaw point get pulled more
        weight = np.clip(1.0 - pixel_dist / radius, 0, 1) ** 2
        pull = SLIM_STRENGTH * dist_to_center * weight

        map_x -= pull * nx
        map_y -= pull * ny

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

# ---------------------------------------------------------------------------
# Main apply function
# ---------------------------------------------------------------------------

def apply(frame):
    """
    Full beauty enhancement pipeline:
    1. Detect face
    2. Get 68-point landmarks
    3. Bilateral smooth skin (landmark-masked, features preserved)
    4. Slim jaw via landmark warp
    """
    output = frame.copy()
    faces = detect_faces(frame)
    if not faces:
        return output

    # Step 1: Skin smoothing
    smoothed = cv2.bilateralFilter(frame, 15, 15, 15)
    result = smoothed.astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.04, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.02, 0, 255)
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.97, 0, 255)
    smoothed = result.astype(np.uint8)

    for face in faces:
        landmarks = get_landmarks(frame, face)

        # Composite smoothed skin onto original (feature-aware mask)
        mask = build_skin_mask(frame, face, landmarks)
        mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        output = (output.astype(np.float32) * (1 - mask_3ch) +
                  smoothed.astype(np.float32) * mask_3ch).astype(np.uint8)

        # Step 2: Slim jaw
        output = slim_face(output, landmarks)

    return output