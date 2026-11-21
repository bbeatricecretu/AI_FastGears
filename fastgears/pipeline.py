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
from .vision.driver_signals import assess_driver
from .vision.face import FACE_REGION_CONTOURS, PULSE_REGIONS, face_region_contours

log = logging.getLogger("pipeline")

COLORS = {"ok": (80, 200, 80), "caution": (0, 190, 255), "danger": (60, 60, 255)}
PREVIEW_INTERVAL_S = 1 / 12
# The pulse estimator (main_final's, unchanged) needs >= 12 samples/s for a trusted reading and
# >= 8 for any reading; the detector loop runs at ~6-8 fps on a laptop CPU. So the pulse is fed
# every camera frame (~30 fps) from its own thread, using the driver's latest landmarks.
RPPG_FACE_MAX_AGE_S = 0.5        # landmarks older than this: no face for the pulse


class Pipeline:
    def __init__(self, settings: Settings, on_state: Callable[[dict], None] | None = None):
        self.s = settings
        self.on_state = on_state
        self.camera = Camera(settings.camera, settings.width, settings.height, settings.mirror,
                             backend=settings.camera_backend)
        self.face = self.rppg = self.objects = self.hands = self.vision = self.rgb = None
        self._hands: list = []
        self._phone_xy: tuple[float, float] | None = None   # where the driver's phone was last seen
        self._frame_i = 0
        self.errors: dict[str, str] = {}
        self._load_detectors()
        self._jpeg: bytes | None = None
        self._last_preview_t: float | None = None
        self._state: dict = {"attention": {}, "pulse": {}, "objects": [], "status": "starting"}
        self._lock = threading.Lock()
        self._rppg_face: tuple | None = None   # (driver FaceResult, its frame time) for the pulse thread
        self._rppg_lock = threading.Lock()
        self._rppg_threaded = False
        self._t0 = time.monotonic()
        self.proc_fps = 0.0
        self._stop = threading.Event()
        self._recalibrate = threading.Event()

    def _load_detectors(self) -> None:
        from .vision.attention import AttentionMonitor
        from .vision.discomfort import DiscomfortMonitor
        from .vision.driver import DriverLock
        self.attention = AttentionMonitor(calib_s=self.s.calib_s)
        self.driver = DriverLock(self.s.driver_seat)
        self.discomfort = DiscomfortMonitor(calib_s=self.s.calib_s,
                                            overlap_s=self.s.discomfort_overlap_s)
        try:
            from .vision.face import FaceTracker
            self.face = FaceTracker(models_dir=self.s.models_dir, num_faces=3)
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
                from .vision.objects import ObjectDetector, PhoneLatch
                self.objects = ObjectDetector(model=self.s.yolo_model, every_n=self.s.yolo_every,
                                              models_dir=self.s.models_dir)
                self.objects.warmup()
                self.phone_latch = PhoneLatch()
            except Exception as e:
                self.errors["objects"] = str(e)
                log.exception("object detector unavailable")
        if self.s.enable_objects and self.s.enable_hands:
            try:
                from .vision.hands import HandTracker
                # 4: the passenger's two hands must not use up the driver's slots
                self.hands = HandTracker(models_dir=self.s.models_dir, num_hands=4)
            except Exception as e:
                self.errors["hands"] = str(e)
                log.exception("hand tracker unavailable")
        if self.s.enable_objects:
            if self.s.phone_vision_check and self.s.llm_provider == "gemini" and self.s.gemini_api_key:
                try:
                    from .vision.phone_check import VisionPhoneCheck
                    self.vision = VisionPhoneCheck(self.s.gemini_api_key, self.s.gemini_vision_model)
                except Exception as e:
                    self.errors["phone_vision"] = str(e)
                    log.exception("vision phone check unavailable")
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
        if self.rppg:
            self._rppg_threaded = True
            threading.Thread(target=self._rppg_loop, name="rppg", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.camera.stop()

    def recalibrate(self) -> None:
        """Forget the driver lock and baselines (applied on the pipeline thread, next frame)."""
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
                self.driver.reset()
                self.attention.reset_calibration()
                self.discomfort.reset()
                if self.rppg:
                    with self._rppg_lock:
                        self._rppg_face = None
                    self.rppg.reset()
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
        faces = self.face.process_all(frame, t_ms) if self.face else []
        roi = self.driver.roi(frame.shape) if self.face else None
        if roi is not None:                                # driver crop: small faces are missed
            try:                                           # in the full frame
                faces = self.driver.merge(faces, self.face.process_roi(frame, roi))
            except Exception:
                log.exception("driver crop face pass failed")
        face = self.driver.update(faces, frame.shape, t)      # the driver only; others ignored
        regions = face_region_contours(face, frame.shape)
        rgb_state = self._rgb_state(frame, face, regions, t)
        if face is not None:                              # the pulse thread uses the newest driver face
            with self._rppg_lock:
                self._rppg_face = (face, t)
        if self.rppg and not self._rppg_threaded:         # no pulse thread (tests): feed this frame
            self._feed_rppg(frame, t)
        from .vision.hands import free_hands, hand_crop, on_open_palm
        self._frame_i += 1
        if self.hands and self._frame_i % 3 == 0:          # hands every 3rd frame (CPU)
            self._hands = self.hands.process(frame, t_ms)
        mine_hands = [h for h in self._hands if self.driver.owns_hand(h, face)]
        up = free_hands(mine_hands, face, frame.shape)    # off the wheel: open / at the face
        both_off = len(up) >= 2                           # one hand on the wheel is fine
        # phone search: around ANY of the driver's hands - a phone held low or at arm's length
        # to the side is not "raised" but is still in a hand
        crops = [(h, hand_crop(h, frame.shape)) for h in mine_hands]
        crops = [(h, b) for h, b in crops if b is not None]
        boxes = [b for _, b in crops]
        dets = self.objects.detect(frame, face, boxes) if self.objects else []
        phone, phone_src = False, None
        others = self.objects.other_people(dets, face) if self.objects else []
        if self.objects:
            own = [d["bbox"] for d in dets if d.get("label") == "person" and list(d["bbox"]) not in others]
            mine = [d for d in dets if d.get("label") != "cell phone" or (
                self.driver.belongs_to_driver((d["bbox"][0] + d["bbox"][2]) / 2,
                                              (d["bbox"][1] + d["bbox"][3]) / 2, face, frame.shape)
                and not self.objects.in_other_person(d["bbox"], others, own)  # a passenger's phone
                and not on_open_palm(d["bbox"], mine_hands)          # an open palm is not a phone
                and not self.objects.covers_face(d["bbox"], face))]  # nor is the driver's head
            best = self.objects.phone_best(mine, face, frame.shape)
            hit = None if best is None else float(best["conf"])
            fresh = self.objects.fresh
            if best is not None:
                phone_src = "yolo"
                if fresh:
                    bx = best["bbox"]
                    self._phone_xy = ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0)
            if self.vision:
                if crops and not self.phone_latch.on and hit is None:
                    self.vision.maybe_check(frame, crops[0][1], t)
                vc = self.vision.phone_conf(t)
                if vc is not None:
                    hit, fresh, phone_src = max(hit or 0.0, vc), True, phone_src or "gemini-vision"
            holding = self._phone_xy is not None and any(
                b[0] <= self._phone_xy[0] <= b[2] and b[1] <= self._phone_xy[1] <= b[3] for b in boxes)
            phone = self.phone_latch.update(hit, fresh, t, holding=holding)
            if not phone:
                self._phone_xy = None
        att = self.attention.update(face, t, phone=phone, hand_up=both_off)
        gestures = self.discomfort.update(face, t, frame.shape,
                                          missing_reason="face_model_unavailable" if self.face is None else "face_not_visible")
        pulse = self.rppg.status() if self.rppg else {}
        state = {
            "t": time.time(),
            "uptime_s": round(t - self._t0, 1),
            "status": "running",
            "attention": att,
            "discomfort": gestures,
            "rgb": rgb_state,
            "driver_signals": assess_driver(t, att, gestures, rgb_state, pulse),
            "pulse": _jsonable(pulse),
            "objects": [d for d in dets if d.get("label") in ("cell phone", "person", "bottle", "cup", "book", "laptop")],
            "face_box": list(face.bbox) if face else None,
            "driver": {**self.driver.state.as_dict(),     # a passenger looking down has no face: count bodies too
                       "passengers": max(self.driver.state.passengers, len(others))},
            "hands": {"available": self.hands is not None, "seen": len(self._hands), "driver": len(mine_hands), "free": len(up), "phone_source": phone_src if phone else None,
                      "vision_calls": self.vision.calls if self.vision else 0},
            "face_regions": regions,
            "camera": {"connected": self.camera.connected, "fps": round(self.camera.fps, 1),
                       "proc_fps": round(self.proc_fps, 1), "size": [frame.shape[1], frame.shape[0]]},
            "detectors": {"face": self.face is not None, "rppg": self.rppg is not None,
                          "hands": self.hands is not None, "phone_vision": self.vision is not None,
                          "objects": self.objects is not None, "errors": self.errors},
        }
        preview_due = self._last_preview_t is None or t - self._last_preview_t >= PREVIEW_INTERVAL_S
        masks_current = (not self.rgb or not self.s.rgb_overlay or not rgb_state.get("regions")
                         or rgb_state.get("sampled_monotonic_s") == t)
        if preview_due and masks_current:
            self._jpeg = self._draw(frame, face, dets, att, pulse, regions, mine_hands, up)
            self._last_preview_t = t
        with self._lock:
            self._state = state
        if self.on_state:
            self.on_state(state)

    def _rppg_loop(self) -> None:
        seq = -1
        while not self._stop.is_set():
            seq2, frame, t = self.camera.read_timed(seq, timeout=1.0)
            if frame is None or t is None or seq2 == seq:
                continue
            seq = seq2
            try:
                self._feed_rppg(frame, t)
            except Exception:
                log.exception("pulse sampling failed")

    def _feed_rppg(self, frame: np.ndarray, t: float) -> None:
        """One camera frame into the pulse estimator, with the newest driver landmarks."""
        with self._rppg_lock:
            face, face_t = self._rppg_face or (None, None)
        if face is not None and abs(t - face_t) > RPPG_FACE_MAX_AGE_S:
            face = None
        self.rppg.update(frame, face.eyes if face else None, t,
                         landmarks=face.landmarks_px if face else None)

    def _publish_no_camera(self) -> None:
        t = time.monotonic()
        gestures = self.discomfort.update(None, t, missing_reason="no_camera_frame")
        if self.rppg:
            self.rppg.update(None, None, t)
        pulse = _jsonable(self.rppg.status()) if self.rppg else {}
        rgb_state = self._rgb_state(None, None, {}, t)
        prev = self._state.get("attention") or {}
        # no frame: the AttentionMonitor is not fed (its timers stay as main calibrated them),
        # but the dashboard must not keep showing a stale face / microsleep
        attention = {"face_present": False, "calibrated": prev.get("calibrated", False), "level": "ok",
                     "eyes_closed": False, "microsleep": False, "yawning": False, "phone": False,
                     "looking": "-", "reasons": []}
        with self._lock:
            self._state = {**self._state, "status": "no camera" if not self.camera.connected else "waiting",
                           "attention": attention,
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
    def _draw(self, frame, face, dets, att, pulse, regions=None, hands=(), hands_free=()) -> bytes | None:
        img = frame.copy()
        for b in self.driver.state.passenger_boxes:
            x0, y0, x1, y1 = map(int, b)
            cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 1)
            cv2.putText(img, "passenger (ignored)", (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (150, 150, 150), 1)
        for hd in hands:                                   # the driver's hands only
            free = any(hd is f for f in hands_free)
            c = (255, 160, 0) if free else (80, 200, 80)
            x0, y0, x1, y1 = hd.bbox
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
            cv2.putText(img, "hand free" if free else "hand on wheel", (x0, max(15, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        level = att.get("level", "ok")
        col = COLORS.get(level, (255, 255, 255))
        for d in dets:
            # only phones that count: confident ones, or any while the phone latch is on
            if d.get("label") == "cell phone" and (d["conf"] >= 0.4 or att.get("phone")):
                x0, y0, x1, y1 = map(int, d["bbox"])
                c = (60, 60, 255)
                cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
                cv2.putText(img, f'{d["label"]} {d["conf"]:.2f}', (x0, max(15, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if face is not None:
            x0, y0, x1, y1 = map(int, face.bbox)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            cv2.putText(img, "DRIVER", (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            lm = face.landmarks_px
            for i in (33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 468, 473):
                if i < len(lm):
                    cv2.circle(img, (int(lm[i][0]), int(lm[i][1])), 2, (255, 255, 0), -1)
        region_colors = {name: (85, 225, 90) for name in PULSE_REGIONS}
        region_colors["nose"] = (255, 210, 80)
        for name, points in (regions or {}).items():
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
