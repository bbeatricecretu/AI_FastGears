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
from .vision.driver_assistance import DriverAssistance
from .vision.driver_signals import assess_driver
from .vision.face import FACE_REGION_CONTOURS, PULSE_REGIONS, face_region_contours

log = logging.getLogger("pipeline")

COLORS = {"ok": (80, 200, 80), "caution": (0, 190, 255), "danger": (60, 60, 255)}
PREVIEW_INTERVAL_S = 1 / 12


class Pipeline:
    def __init__(self, settings: Settings, on_state: Callable[[dict], None] | None = None):
        self.s = settings
        self.on_state = on_state
        self.camera = Camera(settings.camera, settings.width, settings.height, settings.mirror,
                             backend=settings.camera_backend)
        self.face = self.rppg = self.objects = self.rgb = None
        self.errors: dict[str, str] = {}
        self._load_detectors()
        self.assistance = DriverAssistance(settings, self.errors)
        self._last_frame_shape = None
        self._jpeg: bytes | None = None
        self._last_preview_t: float | None = None
        self._state: dict = {"attention": {}, "pulse": {}, "objects": [], "status": "starting"}
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.proc_fps = 0.0
        self._stop = threading.Event()
        self._recalibrate = threading.Event()

    def _load_detectors(self) -> None:
        from .vision.attention import AttentionMonitor
        from .vision.discomfort import DiscomfortMonitor
        self.attention = AttentionMonitor(calib_s=self.s.calib_s)
        self.discomfort = DiscomfortMonitor(calib_s=self.s.calib_s,
                                            overlap_s=self.s.discomfort_overlap_s)
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
        if self.s.enable_rgb:
            try:
                from RGB import ColorMonitor, Thresholds, parse_reference
                self.rgb = ColorMonitor(parse_reference(self.s.rgb_reference),
                                        Thresholds(calibration_s=float(self.s.rgb_calibration_s),
                                                   persistence_s=float(self.s.rgb_persistence_s)),
                                        overlay=self.s.rgb_overlay)
            except Exception as e:
                self.errors["rgb"] = str(e)
                log.exception("RGB analysis unavailable")

    def start(self) -> "Pipeline":
        self.camera.start()
        threading.Thread(target=self._loop, name="pipeline", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.camera.stop()
        if self.objects:
            self.objects.stop_live()

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
            if self._recalibrate.is_set():
                self._recalibrate.clear()
                self.assistance.reset()
                self.attention.reset_calibration()
                self.discomfort.reset()
                if self.rppg:
                    self.rppg.reset()
                if self.objects:
                    self.objects.reset_live()
                if self.rgb:
                    self.rgb.reset()
            seq2, frame, t = self.camera.read_timed(seq, timeout=1.0)
            if frame is None or t is None or seq2 == seq:
                self._publish_no_camera()
                continue
            seq = seq2
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
        self._last_frame_shape = frame.shape
        face = self.assistance.select_face(self.face, frame, t, t_ms)
        regions = face_region_contours(face, frame.shape)
        rgb_state = self._rgb_state(frame, face, regions, t)
        if self.rppg:
            self.rppg.update(frame, face.eyes if face else None, t,
                             landmarks=face.landmarks_px if face else None)
        dets, phone = self.assistance.update_objects(self.objects, frame, face, t, t_ms)
        att = self.attention.update(face, t, phone=phone)
        objects_available = self.objects is not None and not self.assistance.objects_failed
        gestures = self.discomfort.update(face, t, frame.shape,
                                          missing_reason="face_model_unavailable" if self.face is None else "face_not_visible")
        pulse = self.rppg.status() if self.rppg else {}
        state = {
            "t": time.time(),
            "uptime_s": round(t - self._t0, 1),
            "status": "running",
            "attention": att,
            **self.assistance.state(face, att, t, phone, objects_available),
            "discomfort": gestures,
            "rgb": rgb_state,
            "driver_signals": assess_driver(t, att, gestures, rgb_state, pulse),
            "pulse": _jsonable(pulse),
            "objects": [d for d in dets if d.get("label") in ("cell phone", "person", "bottle", "cup", "book", "laptop")],
            "face_box": list(face.bbox) if face else None,
            "face_regions": regions,
            "camera": {"connected": self.camera.connected, "fps": round(self.camera.fps, 1),
                       "proc_fps": round(self.proc_fps, 1), "size": [frame.shape[1], frame.shape[0]]},
            "detectors": {"face": self.face is not None, "rppg": self.rppg is not None,
                          "hands": self.assistance.hands is not None,
                          "phone_vision": self.assistance.phone_vision is not None,
                          "objects": objects_available, "errors": self.errors},
        }
        preview_due = self._last_preview_t is None or t - self._last_preview_t >= PREVIEW_INTERVAL_S
        masks_current = (not self.rgb or not self.s.rgb_overlay or not rgb_state.get("regions")
                         or rgb_state.get("sampled_monotonic_s") == t)
        if preview_due and masks_current:
            self._jpeg = self._draw(frame, face, dets, att, pulse, regions)
            self._last_preview_t = t
        with self._lock:
            self._state = state
        if self.on_state:
            self.on_state(state)

    def _publish_no_camera(self) -> None:
        t = time.monotonic()
        attention = self.attention.update(None, t)
        if self._last_frame_shape is not None:
            self.assistance.driver.update([], self._last_frame_shape, t)
        self.assistance.clear_hands()
        assistance = self.assistance.state(None, attention, t, False, False)
        gestures = self.discomfort.update(None, t, missing_reason="no_camera_frame")
        if self.rppg:
            self.rppg.update(None, None, t)
        pulse = _jsonable(self.rppg.status()) if self.rppg else {}
        rgb_state = self._rgb_state(None, None, {}, t)
        with self._lock:
            self._state = {**self._state, "status": "no camera" if not self.camera.connected else "waiting",
                           "attention": attention,
                           **assistance,
                           "discomfort": gestures,
                           "rgb": rgb_state,
                           "driver_signals": {"finding": "camera_unavailable"},
                           "pulse": pulse,
                           "face_box": None,
                           "face_regions": dict.fromkeys(FACE_REGION_CONTOURS),
                           "camera": {"connected": self.camera.connected}}

    def _rgb_state(self, frame, face, regions, t):
        if self.rgb:
            try:
                return self.rgb.update(frame, face, regions, t)
            except Exception as e:
                self.errors["rgb"] = str(e)
                self.rgb = None
                log.exception("RGB analysis stopped")
        return {"enabled": True, "lighting": "analysis_unavailable"} if self.s.enable_rgb else {}

    # ── overlay (what the jury sees live) ─────────────────────────────
    def _draw(self, frame, face, dets, att, pulse, regions) -> bytes | None:
        img = frame.copy()
        level = att.get("level", "ok")
        col = COLORS.get(level, (255, 255, 255))
        for d in dets:
            if d.get("label") in ("cell phone", "person"):
                x0, y0, x1, y1 = map(int, d["bbox"])
                c = (60, 60, 255) if d["label"] == "cell phone" else (200, 200, 200)
                cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
                cv2.putText(img, f'{d["label"]} {d["conf"]:.2f}', (x0, max(15, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        self.assistance.draw(img)
        if face is not None:
            x0, y0, x1, y1 = map(int, face.bbox)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            lm = face.landmarks_px
            for i in (33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 468, 473):
                if i < len(lm):
                    cv2.circle(img, (int(lm[i][0]), int(lm[i][1])), 2, (255, 255, 0), -1)
        region_colors = {name: (85, 225, 90) for name in PULSE_REGIONS}
        region_colors["nose"] = (255, 210, 80)
        for name, points in regions.items():
            if points is not None:
                contour = np.rint(points).astype(np.int32)
                region_color = region_colors.get(name, (170, 90, 255))
                cv2.polylines(img, [contour], True, region_color, 1, cv2.LINE_AA)
        if self.rgb and self.s.rgb_overlay:
            self.rgb.draw_masks(img)
        lines = [f"{level.upper()}  looking: {att.get('looking', '-')}",
                 f"EAR {att.get('ear') or 0:.2f}  PERCLOS {100 * (att.get('perclos') or 0):.0f}%  "
                 f"blinks/min {att.get('blinks_per_min') or 0:.0f}"]
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
