import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

cv2.ocl.setUseOpenCL(True)

# ---------------------------------------------------------------------------
# Download MediaPipe face landmark model if needed
# ---------------------------------------------------------------------------
MODEL_PATH = "models/face_landmarker.task"

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            MODEL_PATH
        )
        print("Model ready.")

# ---------------------------------------------------------------------------
# MediaPipe landmark index groups (478-point model)
# ---------------------------------------------------------------------------
JAW       = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
             159, 160, 161, 246]
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
             386, 385, 384, 398]
RIGHT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LEFT_BROW  = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
OUTER_LIP  = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIP  = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              415, 310, 311, 312, 13, 82, 81, 80, 191]

SLIM_STRENGTH = 0.011
BILATERAL_D   = 9
BILATERAL_SIG = 15

PROCESS_EVERY_N_FRAMES = 3
_frame_count = 0
_last_output = None

_cached_map_x     = None
_cached_map_y     = None
_cached_landmarks = None
LANDMARK_MOVE_THRESHOLD = 4.0

_landmarker = None

def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _ensure_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _landmarks_moved(new_lm):
    global _cached_landmarks
    if _cached_landmarks is None:
        return True
    diff = np.array(new_lm, dtype=np.float32) - np.array(_cached_landmarks, dtype=np.float32)
    return np.max(np.abs(diff)) > LANDMARK_MOVE_THRESHOLD


def get_landmarks_and_bbox(frame):
    landmarker = _get_landmarker()
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None

    face_lm = result.face_landmarks[0]
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_lm]

    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    x1, y1 = max(0, min(xs)), max(0, min(ys))
    x2, y2 = min(w, max(xs)), min(h, max(ys))
    face = (x1, y1, x2 - x1, y2 - y1)

    return landmarks, face


def _points_to_mask(landmarks, indices, frame_shape, padding=0):
    pts = np.array([landmarks[i] for i in indices], dtype=np.int32)
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


def _build_warp_maps(frame, landmarks):
    global _cached_map_x, _cached_map_y, _cached_landmarks
    h, w = frame.shape[:2]
    map_x = np.arange(w, dtype=np.float32)
    map_y = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(map_x, map_y)

    all_pts = np.array(landmarks, dtype=np.float32)
    face_center = all_pts.mean(axis=0)

    for i in JAW:
        px, py = float(landmarks[i][0]), float(landmarks[i][1])
        dx = face_center[0] - px
        dy = face_center[1] - py
        dist_to_center = np.sqrt(dx * dx + dy * dy) + 1e-6
        nx, ny = dx / dist_to_center, dy / dist_to_center
        radius = dist_to_center * 0.55
        pixel_dist = np.sqrt((map_x - px) ** 2 + (map_y - py) ** 2)
        weight = np.clip(1.0 - pixel_dist / radius, 0, 1) ** 2
        pull = SLIM_STRENGTH * dist_to_center * weight
        map_x -= pull * nx
        map_y -= pull * ny

    _cached_map_x = map_x
    _cached_map_y = map_y
    _cached_landmarks = landmarks


def slim_face(frame, landmarks):
    if _landmarks_moved(landmarks):
        _build_warp_maps(frame, landmarks)
    return cv2.remap(frame, _cached_map_x, _cached_map_y,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply(frame):
    global _frame_count, _last_output
    _frame_count += 1

    if _last_output is not None and _frame_count % PROCESS_EVERY_N_FRAMES != 0:
        return _last_output

    landmarks, face = get_landmarks_and_bbox(frame)
    if landmarks is None:
        _last_output = frame.copy()
        return _last_output

    smoothed = cv2.bilateralFilter(frame, BILATERAL_D, BILATERAL_SIG, BILATERAL_SIG)
    result = smoothed.astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.04, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.02, 0, 255)
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.97, 0, 255)
    smoothed = result.astype(np.uint8)

    mask = build_skin_mask(frame, face, landmarks)
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    output = (frame.astype(np.float32) * (1 - mask_3ch) +
              smoothed.astype(np.float32) * mask_3ch).astype(np.uint8)
    output = slim_face(output, landmarks)

    _last_output = output
    return output