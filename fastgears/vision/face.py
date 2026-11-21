"""MediaPipe Face Landmarker wrapper for the driver's face.

Ported from afferent-junction2026 posture-analysis/posture/afferent/face.py
(MediaPipeLandmarker, model URL, mediapipe import fallback and the verified
`_euler_from_matrix`), switched to RunningMode.VIDEO on full frames and
extended with landmark-based eye/mouth geometry (EAR, MAR, iris offset).

mediapipe and cv2 are imported lazily so tests run without them.
"""
from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

MEDIAPIPE_FACE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task")
FACE_TASK_FILENAME = "face_landmarker.task"

# --- MediaPipe 478-point mesh indices ----------------------------------------
# "Left"/"right" are the subject's own sides. 6-point EAR order:
# p1 outer/inner corner, p2,p3 upper lid, p4 other corner, p5,p6 lower lid.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_LIDS = (159, 145)   # upper, lower (mid-lid)
LEFT_EYE_LIDS = (386, 374)
RIGHT_IRIS_CENTER = 468       # 468-472 right iris
LEFT_IRIS_CENTER = 473        # 473-477 left iris
MOUTH_INNER_VERT = (13, 14)   # upper / lower inner lip
MOUTH_CORNERS = (78, 308)     # inner mouth corners
N_LANDMARKS_IRIS = 478

# MediaPipe Face Mesh anatomical contours, ordered around each region.
MOUTH_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
               375, 321, 405, 314, 17, 84, 181, 91, 146)
MOUTH_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
               324, 318, 402, 317, 14, 87, 178, 88, 95)
NOSE_OUTLINE = (168, 193, 122, 49, 64, 98, 2, 327, 294, 279, 351, 417)
PULSE_REGIONS = ("forehead", "cheek_right", "cheek_left")
FACE_REGION_CONTOURS = {
    "forehead": (109, 10, 338, 336, 107),
    # Paired patches over the lateral cheekbones, below the outer eye corners.
    "cheek_right": (227, 111, 123, 137),
    "cheek_left": (447, 340, 352, 366),
    "nose": NOSE_OUTLINE, "mouth_outer": MOUTH_OUTER, "mouth_inner": MOUTH_INNER,
}


@dataclass
class FaceResult:
    landmarks_px: np.ndarray        # (478, 3) x,y in frame pixels, z relative
    bbox: tuple[int, int, int, int]  # x0,y0,x1,y1 px
    blendshapes: dict[str, float] = field(default_factory=dict)
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    roll_deg: float | None = None
    eyes: tuple[float, float, float] = (0.0, 0.0, 0.0)  # cx, cy, inter-eye px
    ear_left: float = 0.0
    ear_right: float = 0.0
    mar: float = 0.0
    iris_offset: tuple[float, float] | None = None


def _euler_from_matrix(mat: np.ndarray) -> tuple[float | None, ...]:
    """Extract pitch/yaw/roll (deg) from a 4x4 facial transform matrix.

    MediaPipe's python binding returns the rotation as a row-major 3x3 in
    the upper-left block; the head looks down the camera's -Z axis, so the
    usual ZYX Euler decomposition applies (verified against live output in
    the container smoke test).
    """
    import math
    m = np.asarray(mat, dtype=np.float64)
    if m.size == 16:
        m = m.reshape(4, 4)
    if m.shape != (4, 4):
        return None, None, None
    R = m[:3, :3]
    sy = max(-1.0, min(1.0, -float(R[2, 0])))
    yaw = math.degrees(math.asin(sy))
    pitch = math.degrees(math.atan2(float(R[2, 1]), float(R[2, 2])))
    roll = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
    return pitch, yaw, roll


# --- geometry helpers (pure numpy, unit-testable) ------------------------------
def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def eye_aspect_ratio(lm: np.ndarray, idx: tuple[int, ...]) -> float:
    """Soukupova & Cech EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
    p1, p2, p3, p4, p5, p6 = (lm[i] for i in idx)
    w = _d(p1, p4)
    if w < 1e-6:
        return 0.0
    return (_d(p2, p6) + _d(p3, p5)) / (2.0 * w)


def mouth_aspect_ratio(lm: np.ndarray) -> float:
    """Inner-lip opening / inner mouth width (~0 closed, >0.6 yawn)."""
    w = _d(lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]])
    if w < 1e-6:
        return 0.0
    return _d(lm[MOUTH_INNER_VERT[0]], lm[MOUTH_INNER_VERT[1]]) / w


def landmark_region(landmarks: np.ndarray, name: str) -> np.ndarray | None:
    indices = FACE_REGION_CONTOURS[name]
    if len(landmarks) <= max(indices):
        return None
    points = landmarks[list(indices), :2]
    if not np.isfinite(points).all():
        return None
    if name in PULSE_REGIONS:
        points = 0.82 * points + 0.18 * points.mean(axis=0)
    return points


def face_region_contours(face: FaceResult | None, frame_shape) -> dict[str, list | None]:
    """Return in-frame contours in the same pixel coordinates as the original frame."""
    height, width = frame_shape[:2]
    contours = dict.fromkeys(FACE_REGION_CONTOURS)
    if face is None:
        return contours
    for name in FACE_REGION_CONTOURS:
        points = landmark_region(face.landmarks_px, name)
        if points is not None and ((points >= 0) & (points < (width, height))).all():
            contours[name] = points.tolist()
    return contours


def _one_iris(lm, corners, lids, iris) -> tuple[float, float] | None:
    c0, c1 = lm[corners[0]], lm[corners[1]]
    half_w = _d(c0, c1) / 2.0
    if half_w < 1e-6:
        return None
    cx = (c0[0] + c1[0]) / 2.0
    top, bot = lm[lids[0]], lm[lids[1]]
    cy = (top[1] + bot[1]) / 2.0
    half_h = max(abs(float(bot[1] - top[1])) / 2.0, 0.25 * half_w)
    ix, iy = lm[iris][0], lm[iris][1]
    x = float(np.clip((ix - cx) / half_w, -1.0, 1.0))
    y = float(np.clip((iy - cy) / half_h, -1.0, 1.0))
    return x, y


def iris_offset(lm: np.ndarray) -> tuple[float, float] | None:
    """Mean normalised iris position in both eyes, image axes (+x right,
    +y down), -1..1. None without iris landmarks."""
    if lm.shape[0] < N_LANDMARKS_IRIS:
        return None
    r = _one_iris(lm, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_IRIS_CENTER)
    l = _one_iris(lm, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, LEFT_IRIS_CENTER)
    vals = [v for v in (r, l) if v is not None]
    if not vals:
        return None
    return (float(np.mean([v[0] for v in vals])),
            float(np.mean([v[1] for v in vals])))


def eye_centres(lm: np.ndarray) -> tuple[float, float, float]:
    """(cx, cy, d): midpoint between eye centres and inter-eye distance."""
    r = (lm[RIGHT_EYE_CORNERS[0], :2] + lm[RIGHT_EYE_CORNERS[1], :2]) / 2.0
    l = (lm[LEFT_EYE_CORNERS[0], :2] + lm[LEFT_EYE_CORNERS[1], :2]) / 2.0
    c = (r + l) / 2.0
    return float(c[0]), float(c[1]), _d(r, l)


def face_result_from_landmarks(lm_px: np.ndarray, frame_shape,
                               blendshapes: dict[str, float] | None = None,
                               matrix: np.ndarray | None = None) -> FaceResult:
    """Build a FaceResult from (N,3) pixel landmarks (also used by tests)."""
    lm = np.asarray(lm_px, dtype=np.float64)
    h, w = frame_shape[:2]
    x0 = int(max(0, math.floor(lm[:, 0].min())))
    y0 = int(max(0, math.floor(lm[:, 1].min())))
    x1 = int(min(w - 1, math.ceil(lm[:, 0].max())))
    y1 = int(min(h - 1, math.ceil(lm[:, 1].max())))
    pitch = yaw = roll = None
    if matrix is not None:
        pitch, yaw, roll = _euler_from_matrix(matrix)
    return FaceResult(
        landmarks_px=lm,
        bbox=(x0, y0, x1, y1),
        blendshapes=dict(blendshapes or {}),
        pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll,
        eyes=eye_centres(lm),
        ear_left=eye_aspect_ratio(lm, LEFT_EYE_EAR),
        ear_right=eye_aspect_ratio(lm, RIGHT_EYE_EAR),
        mar=mouth_aspect_ratio(lm),
        iris_offset=iris_offset(lm),
    )


def ensure_face_model(models_dir: str = "models") -> str:
    """Return the path to face_landmarker.task, downloading it if missing."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, FACE_TASK_FILENAME)
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(MEDIAPIPE_FACE_TASK_URL, tmp)
        os.replace(tmp, path)
    return path


class FaceTracker:
    """MediaPipe face landmarks with optional multi-face and driver crop passes."""

    def __init__(self, model_path: str | None = None,
                 models_dir: str = "models", max_input_side: int = 768, num_faces: int = 1):
        self._max_input_side = max_input_side
        if model_path is None or not os.path.exists(model_path):
            model_path = ensure_face_model(models_dir)
        try:
            from mediapipe.tasks.python import vision, BaseOptions
        except ImportError:  # older layout
            import mediapipe as mp
            vision = mp.tasks.vision
            BaseOptions = mp.tasks.BaseOptions
        import mediapipe as mp
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=num_faces,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1
        self._roi_landmarker = None
        self._scene_landmarker = None
        self._roi_options = vision.FaceLandmarkerOptions(
            base_options=opts.base_options, output_face_blendshapes=True,
            output_facial_transformation_matrixes=True, num_faces=1,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._scene_options = vision.FaceLandmarkerOptions(
            base_options=opts.base_options, output_face_blendshapes=True,
            output_facial_transformation_matrixes=True, num_faces=3,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._create_landmarker = vision.FaceLandmarker.create_from_options

    def process(self, frame_bgr: np.ndarray, t_ms: int) -> FaceResult | None:
        faces = self.process_all(frame_bgr, t_ms)
        return faces[0] if faces else None

    def process_all(self, frame_bgr: np.ndarray, t_ms: int) -> list[FaceResult]:
        import cv2
        # VIDEO mode requires strictly increasing integer timestamps.
        ts = int(t_ms)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        h, w = frame_bgr.shape[:2]
        image_bgr = frame_bgr
        if max(h, w) > self._max_input_side:
            ratio = self._max_input_side / max(h, w)
            image_bgr = cv2.resize(frame_bgr, (max(1, round(w * ratio)), max(1, round(h * ratio))),
                                   interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        return self._results(result, frame_bgr.shape, 0, 0, w, h)

    def process_roi(self, frame_bgr: np.ndarray, roi) -> FaceResult | None:
        import cv2

        x0, y0, x1, y1 = roi
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        if self._roi_landmarker is None:
            self._roi_landmarker = self._create_landmarker(self._roi_options)
        rgb = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._roi_landmarker.detect(image)
        faces = self._results(result, frame_bgr.shape, x0, y0, x1 - x0, y1 - y0)
        return faces[0] if faces else None

    def process_scene(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        """Detect passengers periodically without slowing the single-face video tracker."""
        import cv2

        if self._scene_landmarker is None:
            self._scene_landmarker = self._create_landmarker(self._scene_options)
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self._max_input_side / max(h, w))
        scene = cv2.resize(frame_bgr, (max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1 else frame_bgr
        rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._results(self._scene_landmarker.detect(image), frame_bgr.shape, 0, 0, w, h)

    @staticmethod
    def _results(result, frame_shape, x0, y0, width, height) -> list[FaceResult]:
        faces = []
        for index, points in enumerate(result.face_landmarks or []):
            landmarks = np.array([(x0 + p.x * width, y0 + p.y * height, p.z * width)
                                  for p in points], dtype=np.float64)
            shapes = result.face_blendshapes or []
            blendshapes = {}
            if index < len(shapes):
                for category in shapes[index]:
                    name = getattr(category, "category_name", None) or getattr(category, "name", None)
                    if name:
                        blendshapes[name] = float(category.score)
            matrices = result.facial_transformation_matrixes or []
            matrix = np.asarray(matrices[index], dtype=np.float64) if index < len(matrices) else None
            faces.append(face_result_from_landmarks(landmarks, frame_shape, blendshapes, matrix))
        return faces

    def close(self) -> None:
        for landmarker in (self._landmarker, self._roi_landmarker, self._scene_landmarker):
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass
