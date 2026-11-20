"""The live loop: camera frame -> face/eyes -> rPPG -> objects -> attention -> state + overlay.

Runs on one worker thread. Each detector is optional: if a model fails to load the
rest keeps running and the dashboard shows the detector as unavailable.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .camera import Camera
from .config import Settings

log = logging.getLogger("pipeline")

COLORS = {"ok": (80, 200, 80), "caution": (0, 190, 255), "danger": (60, 60, 255)}


class Pipeline:
    def __init__(self, settings: Settings, on_state: Callable[[dict], None] | None = None):
        self.s = settings
        self.on_state = on_state
        self.camera = Camera(settings.camera, settings.width, settings.height, settings.mirror)
        self.face = self.rppg = self.objects = None
        self.errors: dict[str, str] = {}
        self._load_detectors()
        self._jpeg: bytes | None = None
        self._state: dict = {"attention": {}, "pulse": {}, "objects": [], "status": "starting"}
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.proc_fps = 0.0
        self._stop = threading.Event()
        self._recalibrate = threading.Event()

    def _load_detectors(self) -> None:
        from .vision.attention import AttentionMonitor
        thresholds = {"mirrored": self.s.mirror, **self.s.drowsiness_thresholds}
        self.attention = AttentionMonitor(calib_s=self.s.calib_s, **thresholds)
        try:
            from .vision.face import FaceTracker
            self.face = FaceTracker(models_dir=self.s.models_dir)
        except Exception as e:
            self.errors["face"] = str(e)
            log.exception("face tracker unavailable")
        if self.s.enable_rppg:
            try:
                from .vision.rppg import RPPG
                self.rppg = RPPG(device=self.s.camera if isinstance(self.s.camera, str)
                                 and self.s.camera.startswith("/dev/video") else None)
            except Exception as e:
                self.errors["rppg"] = str(e)
                log.exception("rPPG unavailable")
        if self.s.enable_objects:
            try:
                from .vision.objects import ObjectDetector
                self.objects = ObjectDetector(model=self.s.yolo_model, every_n=self.s.yolo_every,
                                              models_dir=self.s.models_dir)
            except Exception as e:
                self.errors["objects"] = str(e)
                log.exception("object detector unavailable")

    def start(self) -> "Pipeline":
        self.camera.start()
        threading.Thread(target=self._loop, name="pipeline", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.camera.stop()

    def recalibrate(self) -> None:
        self._recalibrate.set()

    # ── outputs ───────────────────────────────────────────────────────
    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def jpeg(self) -> bytes | None:
        return self._jpeg

    # ── loop ──────────────────────────────────────────────────────────
    def _loop(self) -> None:
        seq, n, t_fps = -1, 0, time.monotonic()
        last_ms = 0
        while not self._stop.is_set():
            seq2, frame = self.camera.read(seq, timeout=1.0)
            if frame is None or seq2 == seq:
                self._publish_no_camera()
                continue
            seq = seq2
            t = time.monotonic()
            t_ms = max(int(t * 1000), last_ms + 1)          # MediaPipe VIDEO mode needs strictly increasing
            last_ms = t_ms
            try:
                self._step(frame, t, t_ms)
            except Exception:
                log.exception("pipeline step failed")
            n += 1
            if t - t_fps >= 2:
                self.proc_fps, n, t_fps = n / (t - t_fps), 0, t

    def _step(self, frame: np.ndarray, t: float, t_ms: int) -> None:
        try:
            face = self.face.process(frame, t_ms) if self.face else None
        except Exception:
            log.exception("face tracking failed for this frame")
            face = None
        if self.rppg:
            self.rppg.update(frame, face.eyes if face else None, t)
        dets = self.objects.detect(frame) if self.objects else []
        phone = bool(self.objects and self.objects.phone_in_use(dets, face, frame.shape))
        self._apply_recalibration()
        att = self.attention.update(face, t, phone=phone)
        pulse = self.rppg.status() if self.rppg else {}
        state = {
            "t": time.time(),
            "uptime_s": round(t - self._t0, 1),
            "status": "running",
            "attention": att,
            "pulse": _jsonable(pulse),
            "objects": [d for d in dets if d.get("label") in ("cell phone", "person", "bottle", "cup", "book", "laptop")],
            "face_box": list(face.bbox) if face else None,
            "camera": {"connected": self.camera.connected, "fps": round(self.camera.fps, 1),
                       "proc_fps": round(self.proc_fps, 1), "size": [frame.shape[1], frame.shape[0]]},
            "detectors": {"face": self.face is not None, "rppg": self.rppg is not None,
                          "objects": self.objects is not None, "errors": self.errors},
        }
        self._jpeg = self._draw(frame, face, dets, att, pulse)
        with self._lock:
            self._state = state
        if self.on_state:
            self.on_state(state)

    def _apply_recalibration(self) -> None:
        if self._recalibrate.is_set():
            self._recalibrate.clear()
            self.attention.reset_calibration()

    def _publish_no_camera(self) -> None:
        self._apply_recalibration()
        att = self.attention.update(None, time.monotonic())
        self._jpeg = None
        with self._lock:
            self._state = {**self._state, "status": "no camera" if not self.camera.connected else "waiting",
                           "camera": {"connected": self.camera.connected}, "attention": att,
                           "face_box": None}
            state = self._state
        if self.on_state:
            self.on_state(state)

    # ── overlay (what the jury sees live) ─────────────────────────────
    def _draw(self, frame, face, dets, att, pulse) -> bytes | None:
        img = frame.copy()
        level = att.get("level", "ok")
        col = COLORS.get(level, (255, 255, 255))
        for d in dets:
            if d.get("label") in ("cell phone", "person"):
                x0, y0, x1, y1 = map(int, d["bbox"])
                c = (60, 60, 255) if d.get("label") == "cell phone" else (200, 200, 200)
                cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
                cv2.putText(img, f'{d["label"]} {d["conf"]:.2f}', (x0, max(15, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if face is not None:
            x0, y0, x1, y1 = map(int, face.bbox)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            lm = face.landmarks_px
            for i in (33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 468, 473):
                if i < len(lm):
                    cv2.circle(img, (int(lm[i][0]), int(lm[i][1])), 2, (255, 255, 0), 1)
        if self.rppg:
            for a, b, c2, d2 in self.rppg.rois():
                cv2.rectangle(img, (a, b), (c2, d2), (0, 255, 0), 1)
        def metric(key, fmt):
            value = att.get(key)
            return "N/A" if value is None else format(value, fmt)

        lines = [f"{level.upper()}  looking: {att.get('looking', '-')}",
                 f"EAR {metric('ear', '.2f')}  PERCLOS {metric('perclos', '.0%')}  "
                 f"blinks/min {metric('blinks_per_min', '.0f')}"]
        if pulse.get("bpm"):
            lines.append(f"pulse {pulse['bpm']} bpm ({(pulse.get('trust') or {}).get('label', '?')})")
        for k, s in enumerate(lines):
            cv2.putText(img, s, (12, 30 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(img, s, (12, 30 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        if img.shape[1] > 960:
            img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and x != x:   # NaN
        return None
    return x