"""Second opinion on "is the driver holding a phone?" from a vision LLM (Gemini).

YOLO (COCO) misses phones seen edge-on, half covered by fingers, or very close. When a hand
is raised but YOLO sees no phone, the driver crop is sent to Gemini at most every
VISION_EVERY_S on a background thread; a "yes" counts as a phone hit for VISION_VALID_S.
Never blocks the video loop; any error just means "no opinion".
"""
from __future__ import annotations

import json
import logging
import threading
import time

import cv2

log = logging.getLogger("phone_check")

VISION_EVERY_S = 5.0     # free-tier friendly (~12 requests/min while a hand is up)
VISION_VALID_S = 4.0
VISION_MIN_CONF = 0.6
PROMPT = ("You are checking a driver-monitoring camera frame. Is the person holding, looking at, "
          "or using a mobile phone (any orientation: screen, back, edge-on, at the ear)? "
          'Reply only JSON: {"phone": true|false, "confidence": 0.0-1.0}')


class VisionPhoneCheck:
    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types
        self._types = types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
        self.model = model
        self._busy = False
        self._last_call = -1e9
        self._result: tuple[float, bool, float] | None = None   # (t, phone, conf)
        self.calls = self.errors = 0
        self.last_latency_ms: int | None = None

    def maybe_check(self, frame_bgr, box, now: float) -> None:
        if self._busy or now - self._last_call < VISION_EVERY_S or box is None:
            return
        x0, y0, x1, y1 = box
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return
        if crop.shape[1] > 640:
            crop = cv2.resize(crop, (640, int(crop.shape[0] * 640 / crop.shape[1])))
        ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        self._busy, self._last_call = True, now
        threading.Thread(target=self._run, args=(jpg.tobytes(), now), daemon=True).start()

    def _run(self, jpg: bytes, asked_at: float) -> None:
        t = self._types
        t0 = time.monotonic()
        try:
            self.calls += 1
            r = self._client.models.generate_content(
                model=self.model,
                contents=[t.Part.from_bytes(data=jpg, mime_type="image/jpeg"), PROMPT],
                config=t.GenerateContentConfig(response_mime_type="application/json", temperature=0,
                                               max_output_tokens=100,
                                               thinking_config=t.ThinkingConfig(thinking_level="minimal")))
            txt = (r.text or "").strip()
            d = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            self._result = (asked_at, bool(d.get("phone")), float(d.get("confidence", 0.0)))
            self.last_latency_ms = int((time.monotonic() - t0) * 1000)
        except Exception as e:
            self.errors += 1
            log.warning("vision phone check failed: %s", e)
        finally:
            self._busy = False

    def phone_conf(self, now: float) -> float | None:
        """Confidence of a recent 'phone: yes', else None."""
        r = self._result
        if r and r[1] and r[2] >= VISION_MIN_CONF and now - r[0] <= VISION_VALID_S:
            return r[2]
        return None
