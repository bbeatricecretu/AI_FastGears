"""YOLOv8n object detection (COCO): phone in hand, passengers.

Conventions ported from afferent-junction2026 jetson/vision/app.py
(`_get_model`: lazy ultralytics YOLO load, prefer an exported .onnx next to
the .pt, CUDA only when torch reports it), adapted to a laptop CPU: the model
runs every `every_n` calls and the cached detections are returned between.
"""
from __future__ import annotations

import os
from pathlib import Path

PHONE_LABEL = "cell phone"
PERSON_LABEL = "person"
PHONE_NEAR_FACE = 1.0     # phone centre within this many face widths of the face box
PHONE_LOWER_FRAC = 1 / 3  # or in the lower 2/3 of the frame


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

    def detect(self, frame_bgr) -> list[dict]:
        """[{"label", "conf", "bbox": [x0,y0,x1,y1]}]; inference every_n calls."""
        run = self._calls % self.every_n == 0
        self._calls += 1
        if not run:
            return list(self._cache)
        model = self._get_model()
        res = model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
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
                    out.append({"label": str(names[int(k)]),
                                "conf": round(float(c), 3),
                                "bbox": [int(v) for v in b]})
        self._cache = out
        return list(out)

    @staticmethod
    def count(dets: list[dict], label: str = PERSON_LABEL) -> int:
        return sum(1 for d in dets if d.get("label") == label)

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
