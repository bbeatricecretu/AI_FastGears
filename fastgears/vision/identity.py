"""Optional driver recognition (InsightFace embeddings + cosine match).

Ported from afferent-junction2026 posture-analysis/posture/afferent/face.py
(InsightFaceDetector) and identity.py (`cosine`, enrollment de-duplication,
0.45 match threshold), reduced to a JSON gallery in data/drivers.json.
insightface is imported lazily; without it available() is False and every
call is a harmless no-op. Note: InsightFace buffalo models are licensed for
non-commercial use only.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

MATCH_THRESHOLD = 0.45     # cosine, as in afferent IdentityTracker
DEDUPE_COSINE = 0.985      # reject near-duplicate enrollment samples
MAX_PER_DRIVER = 20


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class DriverIdentity:
    def __init__(self, models_dir: str = "models",
                 db_path: str = os.path.join("data", "drivers.json"),
                 model_name: str = "buffalo_s",
                 threshold: float = MATCH_THRESHOLD):
        self.models_dir = models_dir
        self.db_path = db_path
        self.model_name = model_name
        self.threshold = threshold
        self._app = None
        self._failed = False
        self.gallery: dict[str, list[list[float]]] = self._load()

    @staticmethod
    def available() -> bool:
        return (importlib.util.find_spec("insightface") is not None
                and importlib.util.find_spec("onnxruntime") is not None)

    # -- storage ----------------------------------------------------------------
    def _load(self) -> dict:
        try:
            with open(self.db_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        tmp = self.db_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.gallery, f)
        os.replace(tmp, self.db_path)

    # -- model ------------------------------------------------------------------
    def _get_app(self):
        if self._app is None and not self._failed:
            if not self.available():
                self._failed = True
                return None
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name=self.model_name,
                    root=os.path.join(self.models_dir, "insightface"),
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(320, 320))
                self._app = app
            except Exception:
                self._failed = True
        return self._app

    def _embedding(self, frame_bgr) -> np.ndarray | None:
        """Embedding of the largest face, or None."""
        app = self._get_app()
        if app is None:
            return None
        try:
            faces = app.get(frame_bgr)
        except Exception:
            return None
        best, area = None, 0.0
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            x0, y0, x1, y1 = f.bbox
            a = float((x1 - x0) * (y1 - y0))
            if a > area:
                best, area = np.asarray(emb, dtype=np.float64), a
        return best

    # -- API --------------------------------------------------------------------
    def enroll(self, frame_bgr, name: str) -> bool:
        """Add one embedding of the largest face to `name`. True if stored."""
        emb = self._embedding(frame_bgr)
        if emb is None or not name:
            return False
        samples = self.gallery.setdefault(name, [])
        if any(cosine(emb, np.asarray(s)) > DEDUPE_COSINE for s in samples):
            return False
        samples.append([round(float(v), 6) for v in emb])
        del samples[:-MAX_PER_DRIVER]
        self._save()
        return True

    def identify(self, frame_bgr) -> dict:
        emb = self._embedding(frame_bgr)
        if emb is None:
            return {"name": None, "similarity": 0.0}
        best_name, best = None, 0.0
        for name, samples in self.gallery.items():
            for s in samples:
                c = cosine(emb, np.asarray(s, dtype=np.float64))
                if c > best:
                    best_name, best = name, c
        return {"name": best_name if best >= self.threshold else None,
                "similarity": round(best, 3)}
