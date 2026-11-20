"""Hands (MediaPipe Hand Landmarker): where the driver's hands are.

Used for two things:
  * the phone is almost always *in a hand*: the object detector searches a high-resolution
    crop around each raised hand, where an edge-on or half-covered phone is much easier to see;
  * a hand held up at face/chest level for a while = hands off the wheel (caution), even when
    the phone itself can't be recognised.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/1/hand_landmarker.task")
HAND_TASK_FILENAME = "hand_landmarker.task"
HAND_UP_BELOW_FACE = 1.2   # hand centre above face bottom + this x face height = "raised"
HAND_NEAR_FACE_W = 2.5     # and horizontally within this many face widths of the face


@dataclass
class Hand:
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 px
    score: float
    wrist: tuple[float, float] | None = None
    forearm: tuple[float, float] | None = None   # unit vector from the knuckles through the wrist,
                                                 # i.e. pointing down the forearm to the owner
    length: float = 0.0                          # wrist -> middle knuckle, px

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def ensure_hand_model(models_dir: str = "models") -> str:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, HAND_TASK_FILENAME)
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(HAND_TASK_URL, tmp)
        os.replace(tmp, path)
    return path


class HandTracker:
    def __init__(self, models_dir: str = "models", num_hands: int = 2):
        from mediapipe.tasks.python import vision, BaseOptions
        import mediapipe as mp
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_hand_model(models_dir)),
            num_hands=num_hands, running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4)
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, t_ms: int) -> list[Hand]:
        import cv2
        ts = max(int(t_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        h, w = frame_bgr.shape[:2]
        out = []
        for i, pts in enumerate(res.hand_landmarks or []):
            xs = [p.x * w for p in pts]
            ys = [p.y * h for p in pts]
            score = float(res.handedness[i][0].score) if res.handedness and res.handedness[i] else 1.0
            wx, wy = xs[0], ys[0]                    # landmark 0 = wrist, 9 = middle-finger knuckle
            dx, dy = wx - xs[9], wy - ys[9]
            n = (dx * dx + dy * dy) ** 0.5
            out.append(Hand((int(max(0, min(xs))), int(max(0, min(ys))),
                             int(min(w - 1, max(xs))), int(min(h - 1, max(ys)))), score,
                            wrist=(wx, wy), forearm=(dx / n, dy / n) if n > 1 else None, length=n))
        return out


def raised_hands(hands: list[Hand], face) -> list[Hand]:
    """Hands at face/chest level near the driver (not down on the wheel). Without a face: none."""
    if face is None:
        return []
    fx0, fy0, fx1, fy1 = face.bbox
    fw, fh = fx1 - fx0, fy1 - fy0
    fcx = (fx0 + fx1) / 2.0
    out = []
    for hd in hands:
        cx, cy = hd.centre
        if cy <= fy1 + HAND_UP_BELOW_FACE * fh and abs(cx - fcx) <= HAND_NEAR_FACE_W * fw:
            out.append(hd)
    return out


def hand_crop(hand: Hand, frame_shape, scale: float = 2.2):
    """Square crop around a hand big enough to contain the phone it holds."""
    h, w = frame_shape[:2]
    x0, y0, x1, y1 = hand.bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0, 40) * scale / 2.0 + 40
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    return (a, b, c, d) if c - a >= 48 and d - b >= 48 else None


FOREARM_REACH = 6.0   # follow the forearm this many knuckle-to-wrist lengths towards the body


def owner_distance(hand: Hand, face_bbox) -> float:
    """How well a hand's forearm points at a person's shoulders (smaller = more likely theirs),
    in that person's face widths. A hand next to someone's face can still be the driver's:
    the arm it hangs on decides, not the distance to the nearest face."""
    x0, y0, x1, y1 = face_bbox
    fw, fh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    mx, sy = (x0 + x1) / 2.0, y1 + 0.6 * fh          # shoulder height
    shoulders = ((mx - 0.9 * fw, sy), (mx + 0.9 * fw, sy))
    if hand.wrist is None or hand.forearm is None:
        cx, cy = hand.centre
        return min((((cx - sx) / fw) ** 2 + ((cy - sy_) / fh) ** 2) ** 0.5 for sx, sy_ in shoulders)
    wx, wy = hand.wrist
    ux, uy = hand.forearm
    reach = FOREARM_REACH * max(hand.length, 0.25 * fw)
    best = float("inf")
    for sx, sy_ in shoulders:   # closest point of the forearm ray to either shoulder
        k = max(0.0, min(reach, (sx - wx) * ux + (sy_ - wy) * uy))
        px, py = wx + k * ux, wy + k * uy
        best = min(best, ((px - sx) ** 2 + (py - sy_) ** 2) ** 0.5 / fw)
    return best
