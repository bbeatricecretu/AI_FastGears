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
    """MediaPipe Face Landmarker (VIDEO mode, one face) on full BGR frames."""

    def __init__(self, model_path: str | None = None,
                 models_dir: str = "models", num_faces: int = 4):
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
            num_faces=num_faces,   # all faces: DriverLock picks the driver
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, t_ms: int) -> FaceResult | None:
        """First face only (compatibility); the app uses process_all + DriverLock."""
        faces = self.process_all(frame_bgr, t_ms)
        return faces[0] if faces else None

    def process_all(self, frame_bgr: np.ndarray, t_ms: int) -> list[FaceResult]:
        import cv2
        # VIDEO mode requires strictly increasing integer timestamps.
        ts = int(t_ms)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        h, w = frame_bgr.shape[:2]
        out: list[FaceResult] = []
        for i, pts in enumerate(result.face_landmarks or []):
            lm = np.array([(p.x * w, p.y * h, p.z * w) for p in pts],
                          dtype=np.float64)
            bs: dict[str, float] = {}
            if result.face_blendshapes and i < len(result.face_blendshapes):
                for c in result.face_blendshapes[i]:
                    name = getattr(c, "category_name", None) or \
                        getattr(c, "name", None)
                    if name:
                        bs[name] = float(c.score)
            mat = None
            mats = result.facial_transformation_matrixes
            if mats and i < len(mats):
                mat = np.asarray(mats[i], dtype=np.float64)
            out.append(face_result_from_landmarks(lm, frame_bgr.shape, bs, mat))
        return out

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
