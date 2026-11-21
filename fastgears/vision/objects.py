"""YOLOv8n object detection (COCO): phone in hand, passengers.

Conventions ported from afferent-junction2026 jetson/vision/app.py
(`_get_model`: lazy ultralytics YOLO load, prefer an exported .onnx next to
the .pt, CUDA only when torch reports it), adapted to a laptop CPU: the model
runs every `every_n` calls and the cached detections are returned between.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

PHONE_LABEL = "cell phone"
PERSON_LABEL = "person"
PHONE_NEAR_FACE = 1.0     # phone centre within this many face widths of the face box
PHONE_LOWER_FRAC = 1 / 3  # or in the lower 2/3 of the frame
PHONE_CONF = 0.20         # phones are small and often seen edge-on: lower bar, the latch filters noise
PHONE_COCO_ID = 67
CROP_IMGSZ = 480          # driver-crop pass: the phone gets ~2x more pixels than in the full frame
PHONE_CONFIRM_S = 1.5     # latch ON: 2 phone hits within this window (or one hit >= PHONE_SURE)
PHONE_SURE = 0.5
PHONE_FACE_COVER = 0.6    # a "phone" box holding this much of the driver's face is the face
PHONE_HOLD_S = 1.5        # latch OFF: no phone hit for this long...
PHONE_HAND_HOLD_S = 4.0   # ...or this long while the driver's hand is still where the phone was
                          # (YOLOv8n loses edge-on / half-covered phones for seconds at a time)
DRIVER_CROP_W = 3.0       # driver crop: face +- this many face widths (phones are often held
                          # at arm's length to the side, outside a narrow crop)
LIVE_MAX_AGE_S = 1.0
LIVE_INFERENCE_INTERVAL_S = 0.35


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class ObjectDetector:
    def __init__(self, model: str = "yolov8n.pt", every_n: int = 3,
                 conf: float = 0.35, models_dir: str = "models",
                 imgsz: int = 480):
        self.model_name = model
        self.every_n = max(1, int(every_n))
        self.conf = float(conf)
        self.models_dir = models_dir
        self.imgsz = int(imgsz)
        self._model = None
        self._device = None
        self._calls = 0
        self._cache: list[dict] = []
        self._phone_cache: list[dict] = []
        self.fresh = False        # True when the last detect() call ran inference
        self._live_lock = threading.Condition()
        self._live_worker = None
        self._live_stop = False
        self._live_pending = self._live_result = self._live_error = None
        self._live_generation = self._live_frame = 0
        self._live_seen = self._live_requested = None

    def _model_path(self) -> str:
        p = Path(self.model_name)
        if p.parent != Path(".") or p.exists():
            return str(p)
        os.makedirs(self.models_dir, exist_ok=True)
        local = Path(self.models_dir) / p.name
        onnx = local.with_suffix(".onnx")
        if onnx.exists():
            return str(onnx)
        # ultralytics downloads known assets (yolov8n.pt) to this path
        return str(local)

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._device = 0 if _has_cuda() else "cpu"
            self._model = YOLO(self._model_path())
        return self._model

    def _predict(self, img, imgsz, conf, classes=None, dx=0, dy=0) -> list[dict]:
        model = self._get_model()
        res = model.predict(img, conf=conf, imgsz=imgsz, classes=classes,
                            device=self._device, verbose=False)
        out: list[dict] = []
        if res:
            r = res[0]
            names = r.names
            boxes = r.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                for b, c, k in zip(xyxy, confs, clss):
                    x0, y0, x1, y1 = (int(v) for v in b)
                    out.append({"label": str(names[int(k)]),
                                "conf": round(float(c), 3),
                                "bbox": [x0 + dx, y0 + dy, x1 + dx, y1 + dy]})
        return out

    @staticmethod
    def driver_crop(face, frame_shape):
        """Region where the driver's hands and phone are: face +-DRIVER_CROP_W widths, down to the frame bottom."""
        if face is None:
            return None
        h, w = frame_shape[:2]
        fx0, fy0, fx1, fy1 = face.bbox
        fw, fh = fx1 - fx0, fy1 - fy0
        x0, x1 = max(0, int(fx0 - DRIVER_CROP_W * fw)), min(w, int(fx1 + DRIVER_CROP_W * fw))
        y0, y1 = max(0, int(fy0 - 0.5 * fh)), h
        if x1 - x0 < 64 or y1 - y0 < 64:
            return None
        return x0, y0, x1, y1

    def detect(self, frame_bgr, face=None, hand_boxes=None) -> list[dict]:
        """[{"label", "conf", "bbox": [x0,y0,x1,y1]}]; inference every_n calls.

        Inference ticks alternate between the full frame (everything, phones at a lower bar)
        and, when a face is known, a phone-only pass on the crop around the driver.
        """
        run = self._calls % self.every_n == 0
        tick = self._calls // self.every_n
        self._calls += 1
        self.fresh = run
        if not run:
            return self._cache + self._phone_cache
        return self._infer(frame_bgr, face, hand_boxes, tick)

    def detect_live(self, frame, face, hand_boxes, t) -> list[dict]:
        """Queue at most the newest frame so YOLO never blocks camera sampling."""
        with self._live_lock:
            if self._live_error is not None:
                raise RuntimeError("live YOLO inference failed") from self._live_error
            if self._live_stop:
                return []
            if self._live_worker is None:
                self._live_worker = threading.Thread(target=self._run_live, name="yolo", daemon=True)
                self._live_worker.start()
            if (self._live_frame % self.every_n == 0
                    and (self._live_requested is None or t - self._live_requested >= LIVE_INFERENCE_INTERVAL_S)):
                self._live_pending = (frame, face, hand_boxes, t, self._live_generation)
                self._live_requested = t
                self._live_lock.notify()
            self._live_frame += 1
            self.fresh = False
            if self._live_result is None:
                return []
            taken_at, detections = self._live_result
            if t - taken_at > LIVE_MAX_AGE_S:
                return []
            self.fresh = taken_at != self._live_seen
            self._live_seen = taken_at
            return detections

    def _run_live(self):
        try:
            import torch

            if not _has_cuda():
                torch.set_num_threads(1)
            tick = 0
            previous_generation = None
            while True:
                with self._live_lock:
                    self._live_lock.wait_for(lambda: self._live_stop or self._live_pending is not None)
                    if self._live_stop:
                        return
                    frame, face, boxes, taken_at, generation = self._live_pending
                    self._live_pending = None
                if generation != previous_generation:
                    self._cache, self._phone_cache = [], []
                    tick, previous_generation = 0, generation
                detections = self._infer(frame, face, boxes, tick)
                tick += 1
                with self._live_lock:
                    if generation == self._live_generation:
                        self._live_result = taken_at, detections
        except Exception as exc:
            with self._live_lock:
                self._live_error = exc

    def reset_live(self):
        with self._live_lock:
            self._live_generation += 1
            self._live_pending = self._live_result = self._live_seen = self._live_requested = None

    def stop_live(self):
        with self._live_lock:
            self._live_stop = True
            self._live_pending = None
            self._live_lock.notify_all()

    def _infer(self, frame_bgr, face, hand_boxes, tick):
        if face is None and hand_boxes is None:
            self._cache = self._predict(frame_bgr, self.imgsz, self.conf)
            self._phone_cache = []
            return list(self._cache)
        # phone-only pass: around each of the driver's hands (the phone is in one), else around the driver
        crops = [("hand-crop", b) for b in (hand_boxes or [])[:2]]
        if not crops:
            dc = self.driver_crop(face, frame_bgr.shape)
            crops = [("driver-crop", dc)] if dc is not None else []
        if crops and tick % 2 == 1:
            self._phone_cache = []
            for src, (x0, y0, x1, y1) in crops:
                for d in self._predict(frame_bgr[y0:y1, x0:x1], CROP_IMGSZ, PHONE_CONF,
                                       classes=[PHONE_COCO_ID], dx=x0, dy=y0):
                    d["source"] = src
                    self._phone_cache.append(d)
        else:
            dets = self._predict(frame_bgr, self.imgsz, min(self.conf, PHONE_CONF))
            self._cache = [d for d in dets if d["label"] != PHONE_LABEL and d["conf"] >= self.conf]
            self._phone_cache = [d for d in dets if d["label"] == PHONE_LABEL]
        return self._cache + self._phone_cache

    @staticmethod
    def count(dets: list[dict], label: str = PERSON_LABEL) -> int:
        return sum(1 for d in dets if d.get("label") == label)

    @staticmethod
    def covers_face(bbox, face, frac: float = PHONE_FACE_COVER) -> bool:
        """The box contains most of the driver's face: YOLO read the head as a phone. A phone at
        the ear sits beside the face; one held over the face would hide it from the tracker."""
        if face is None:
            return False
        fx0, fy0, fx1, fy1 = face.bbox
        ix = max(0.0, min(bbox[2], fx1) - max(bbox[0], fx0))
        iy = max(0.0, min(bbox[3], fy1) - max(bbox[1], fy0))
        return ix * iy >= frac * max(1.0, (fx1 - fx0) * (fy1 - fy0))

    @classmethod
    def phone_best(cls, dets: list[dict], face, frame_shape) -> dict | None:
        """The most confident phone detection that counts for the driver, or None."""
        best = None
        for d in dets:
            if d.get("label") == PHONE_LABEL and cls.phone_in_use([d], face, frame_shape):
                if best is None or float(d["conf"]) > float(best["conf"]):
                    best = d
        return best

    @staticmethod
    def phone_in_use(dets: list[dict], face, frame_shape) -> bool:
        """A "cell phone" is detected and, if a face exists, lies in the lower
        2/3 of the frame or near the face (held to the ear / in front)."""
        phones = [d for d in dets if d.get("label") == PHONE_LABEL]
        if not phones:
            return False
        if face is None:
            return True
        h = frame_shape[0]
        fx0, fy0, fx1, fy1 = face.bbox
        fw = max(1.0, float(fx1 - fx0))
        m = PHONE_NEAR_FACE * fw
        for d in phones:
            x0, y0, x1, y1 = d["bbox"]
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if cy >= h * PHONE_LOWER_FRAC:
                return True
            if fx0 - m <= cx <= fx1 + m and fy0 - m <= cy <= fy1 + m:
                return True
        return False


class PhoneLatch:
    """Turns noisy per-inference phone hits into a steady in-use signal.

    ON after 2 hits within PHONE_CONFIRM_S (or one confident hit), OFF after PHONE_HOLD_S
    without any hit - or after PHONE_HAND_HOLD_S while `holding` (the driver's hand is still
    where the phone was last seen). Only fresh inference results count; cached repeats are ignored.
    """

    def __init__(self, confirm_s: float = PHONE_CONFIRM_S, hold_s: float = PHONE_HOLD_S,
                 sure: float = PHONE_SURE, hand_hold_s: float = PHONE_HAND_HOLD_S):
        self.confirm_s, self.hold_s, self.sure = confirm_s, hold_s, sure
        self.hand_hold_s = hand_hold_s
        self.on = False
        self._hits: list[float] = []
        self._last_hit: float | None = None

    def update(self, hit_conf: float | None, fresh: bool, t: float, holding: bool = False) -> bool:
        if fresh and hit_conf is not None:
            self._hits = [h for h in self._hits if t - h <= self.confirm_s] + [t]
            self._last_hit = t
            if len(self._hits) >= 2 or hit_conf >= self.sure:
                self.on = True
        hold = self.hand_hold_s if holding else self.hold_s
        if self.on and (self._last_hit is None or t - self._last_hit > hold):
            self.on = False
            self._hits.clear()
        return self.on
