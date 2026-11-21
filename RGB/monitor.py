"""Conservative, time-based observations of visible facial colour changes."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite

import cv2
import numpy as np

from .color import CHANNELS, REGIONS, RegionReading, compensate_light, measure, region_mask


CHEEKS = ("cheek_right", "cheek_left")
OBSERVATIONS = {
    "cheek_redness": "Creștere persistentă a roșeții obrajilor",
    "cheek_color_reduction": "Reducere persistentă a colorației obrajilor",
    "lip_tone_shift": "Schimbare persistentă a nuanței buzelor",
}
VISIBLE_FINDINGS = {
    "cheek_redness": "Ambii obraji apar mai roșii decât la calibrare.",
    "cheek_color_reduction": "Ambii obraji au o colorație redusă față de reper.",
    "blue_axis": "Culoarea buzelor s-a deplasat spre albastru față de reper.",
    "lower_chroma": "Culoarea buzelor are o cromatică mai scăzută față de reper.",
}


@dataclass(frozen=True)
class Thresholds:
    calibration_s: float = 5.0
    persistence_s: float = 1.0
    recovery_s: float = 0.5
    interval_s: float = 0.1
    max_gap_s: float = 0.35
    min_focus: float = 9.0
    redness_a: float = 2.0
    reduced_a: float = 2.0
    reduced_c: float = 2.0
    lip_blue_b: float = 2.5
    lip_gray_c: float = 3.0
    variation_multiplier: float = 3.5
    dispersion_floor: float = 0.5
    reference_light_delta: float = 6.0
    reference_chroma_delta: float = 4.0
    forehead_light_delta: float = 9.0
    forehead_chroma_delta: float = 6.0
    cheek_light_delta: float = 8.0
    redness_light_delta: float = 16.0
    redness_light_imbalance: float = 7.0
    redness_a_with_light_change: float = 3.0
    lip_light_delta: float = 10.0
    hysteresis: float = 0.65


def parse_reference(value: str) -> tuple[float, float, float, float] | None:
    if not value.strip():
        return None
    parts = tuple(float(part) for part in value.split(","))
    if len(parts) != 4 or not all(isfinite(n) for n in parts):
        raise ValueError("RGB reference must be x0,y0,x1,y1 in 0–1 image coordinates")
    x0, y0, x1, y1 = parts
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("RGB reference must be x0,y0,x1,y1 in 0–1 image coordinates")
    return parts


@dataclass
class Baseline:
    center: np.ndarray | None = None
    compared_center: np.ndarray | None = None
    mad: np.ndarray | None = None
    illuminant: np.ndarray | None = None
    seconds: float = 0.0
    last: float | None = None
    samples: deque = field(default_factory=lambda: deque(maxlen=180))

    def add(self, color, t, settings, illuminant=None):
        if self.center is not None:
            return
        if color is None:
            self.last = None
            return
        if self.samples and np.linalg.norm(color[:3] - np.median(self.samples, axis=0)[:3]) > 7:
            self.samples.clear()
            self.seconds = 0.0
            self.last = None
        if self.last is not None and t - self.last <= settings.max_gap_s:
            self.seconds += t - self.last
        self.last = t
        self.samples.append(color if illuminant is None else np.concatenate((color, illuminant)))
        if self.seconds + 1e-9 >= settings.calibration_s and len(self.samples) >= 5:
            samples = np.asarray(self.samples)
            center = np.median(samples, axis=0)
            self.center = center[:4]
            compared = samples[:, :4]
            if len(center) > 4:
                self.illuminant = center[4:]
                compared = np.stack([compensate_light(sample[:4], sample[4:], self.illuminant)
                                     for sample in samples])
            self.compared_center = np.median(compared, axis=0)
            self.mad = np.median(np.abs(compared - self.compared_center), axis=0)
            self.samples.clear()


@dataclass
class Episode:
    since: float
    last: float
    event_id: int | None = None
    recovery_at: float | None = None
    details: dict = field(default_factory=dict)


class ColorMonitor:
    def __init__(self, reference=None, settings=None, *, overlay=False):
        self.reference = reference
        self.settings = settings or Thresholds()
        self.overlay = overlay
        if not all(isfinite(n) and n > 0 for n in (self.settings.calibration_s, self.settings.persistence_s)):
            raise ValueError("RGB durations must be positive seconds")
        self._next_id = 1
        self.reset()

    def reset(self):
        self.baselines = {name: Baseline() for name in (*REGIONS, "reference")}
        self._episodes = {}
        self._recent = deque(maxlen=8)
        self._last_frame = self._last_sample = self._last_face = None
        self._recalibrate = False
        self._mask_lines = []
        self._state = self._snapshot("waiting", {}, {}, None)

    def _snapshot(self, lighting, regions, observations, reference_rect):
        return {"enabled": True, "lighting": lighting, "regions": regions,
                "reference_rect": reference_rect, "reference_configured": self.reference is not None,
                "lighting_source": "external_patch" if self.reference is not None else "forehead",
                "sampled_monotonic_s": self._last_sample,
                "observations": observations, "recent_events": list(self._recent)}

    def _reference_reading(self, frame, face_box):
        if self.reference is None:
            return None, None
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = (round(n * side) for n, side in zip(self.reference, (w, h, w, h)))
        rect = (x0, y0, x1, y1)
        if face_box is not None:
            a, b, c, d = face_box
            if x0 < c and a < x1 and y0 < d and b < y1:
                return None, rect
        reading = measure(frame[y0:y1, x0:x1].reshape(-1, 3), reference=True)
        if reading.lab is None or reading.mad[0] > 5 or max(reading.mad[1:3]) > 4:
            return None, rect
        return reading.lab, rect

    def _light_state(self, reference, forehead):
        base = self.baselines["forehead"].center
        if self.reference is None:
            if forehead.lab is None:
                return "forehead_unavailable"
            if base is None:
                return "calibrating"
            delta = forehead.lab - base
            if (abs(delta[0]) > self.settings.forehead_light_delta
                    or np.linalg.norm(delta[1:3]) > self.settings.forehead_chroma_delta):
                return "lighting_changed"
            return "estimated"

        ref = self.baselines["reference"].center
        if reference is None:
            return "reference_unavailable"
        elif ref is None and (reference[3] > 14 or not 20 <= reference[0] <= 85):
            return "reference_not_neutral"
        elif ref is None:
            lighting = "calibrating"
        else:
            delta = reference - ref
            if (abs(delta[0]) > self.settings.reference_light_delta
                    or np.linalg.norm(delta[1:3]) > self.settings.reference_chroma_delta):
                return "lighting_changed"
            lighting = "stable"

        if (forehead.lab is not None and base is not None
                and abs(forehead.lab[0] - base[0]) > self.settings.forehead_light_delta):
            return "lighting_changed"
        return lighting

    @staticmethod
    def _neutral(face):
        keys = ("mouthSmileLeft", "mouthSmileRight", "mouthUpperUpLeft", "mouthUpperUpRight", "jawOpen")
        return face.mar < 0.22 and all(face.blendshapes.get(key, 0) < 0.35 for key in keys)

    def _measure_regions(self, frame, contours, face):
        observations = {}
        self._mask_lines = []
        for name in REGIONS:
            outer = contours.get("mouth_outer" if name == "lips" else name)
            inner = contours.get("mouth_inner") if name == "lips" else None
            if name == "lips" and face.mar > 0.23:
                observations[name] = RegionReading(None, None, reason="mouth_moving")
                continue
            pixels = None
            if outer is not None and (name != "lips" or inner is not None):
                scale = 0.006 if name == "lips" else 0.02
                raster = region_mask(frame.shape, outer, inner, inset=max(1, round(face.eyes[2] * scale)))
                if raster is not None:
                    mask, (x0, y0, x1, y1) = raster
                    pixels = frame[y0:y1, x0:x1][mask != 0]
                    if self.overlay:
                        lines, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        self._mask_lines.extend(line + (x0, y0) for line in lines)
            observations[name] = measure(pixels, lips=name == "lips")
        return observations

    def _assessment(self, kind, deltas, measured, lighting_source):
        areas = ("lips",) if kind == "lip_tone_shift" else CHEEKS
        if any(name not in deltas for name in areas):
            return None, {}, "region_unavailable"
        settings = self.settings

        def strength(name, axis, minimum, sign):
            mad = max(settings.dispersion_floor, self.baselines[name].mad[axis])
            return float(sign * deltas[name][axis] / max(minimum, settings.variation_multiplier * mad))

        limit = settings.lip_light_delta if kind == "lip_tone_shift" else settings.cheek_light_delta
        details = {"regions": list(areas), "delta": {name: _fields(measured[name]) for name in areas},
                   "compared_delta": {name: _fields(deltas[name]) for name in areas},
                   "lighting_source": lighting_source}
        luminance = [deltas[name][0] for name in areas]
        outside_light_limit = any(abs(value) > limit for value in luminance)
        if outside_light_limit and kind != "cheek_redness":
            return None, details, "luminance_changed"
        if kind == "cheek_redness":
            threshold = settings.redness_a
            if outside_light_limit:
                if abs(luminance[0] - luminance[1]) > settings.redness_light_imbalance:
                    return None, details, "uneven_cheek_light"
                if any(abs(value) > settings.redness_light_delta for value in luminance):
                    return None, details, "luminance_changed"
                threshold = settings.redness_a_with_light_change
            score = min(strength(name, 1, threshold, 1) for name in areas)
            if outside_light_limit and score < settings.hysteresis:
                return None, details, "luminance_changed"
        elif kind == "cheek_color_reduction":
            score = min(max(strength(name, 1, settings.reduced_a, -1),
                            strength(name, 3, settings.reduced_c, -1)) for name in areas)
        else:
            blue = strength("lips", 2, settings.lip_blue_b, -1)
            gray = strength("lips", 3, settings.lip_gray_c, -1)
            details["direction"] = "blue_axis" if blue >= gray else "lower_chroma"
            score = max(blue, gray)
        return score, details, None

    @staticmethod
    def _event(kind, episode, state, reason=None):
        finding_key = episode.details.get("direction") if kind == "lip_tone_shift" else kind
        return {"id": episode.event_id, "type": kind, "label": OBSERVATIONS[kind], "state": state,
                "started_monotonic_s": episode.since, "duration_ms": round((episode.last - episode.since) * 1000),
                "finding": VISIBLE_FINDINGS.get(finding_key), "reason": reason, **episode.details}

    def _end(self, kind, state, reason=None):
        episode = self._episodes.pop(kind, None)
        if episode and episode.event_id is not None:
            self._recent.append(self._event(kind, episode, state, reason))

    def _advance(self, kind, t, score, details, reason):
        episode = self._episodes.get(kind)
        if score is not None:
            details = {**details, "threshold_ratio": round(score, 3)}
        status = {"label": OBSERVATIONS[kind], "state": "idle", "reason": reason, **details}
        if score is None:
            self._end(kind, "interrupted", reason)
            return {**status, "state": "unavailable"}
        if episode is None and score >= 1:
            episode = self._episodes[kind] = Episode(t, t)
        if episode is None:
            return status
        threshold = self.settings.hysteresis if episode.event_id else 1
        if score >= threshold:
            episode.last, episode.details, episode.recovery_at = t, details, None
            if episode.event_id is None and t - episode.since + 1e-9 >= self.settings.persistence_s:
                episode.event_id = self._next_id
                self._next_id += 1
            if episode.event_id:
                return self._event(kind, episode, "active")
            return {**status, "state": "candidate", "duration_ms": round((t - episode.since) * 1000)}
        if episode.event_id is None:
            self._end(kind, "ended")
            return status
        if episode.recovery_at is None:
            episode.recovery_at = t
        if t - episode.recovery_at + 1e-9 >= self.settings.recovery_s:
            self._end(kind, "ended")
            return status
        return self._event(kind, episode, "recovering")

    def _unavailable(self, t, reason):
        observations = {kind: self._advance(kind, t, None, {}, reason) for kind in OBSERVATIONS}
        for baseline in self.baselines.values():
            baseline.add(None, t, self.settings)
        self._mask_lines = []
        self._state = self._snapshot(reason, {}, observations, None)
        return self._state

    def update(self, frame, face, contours, t):
        if self._last_frame is not None and t <= self._last_frame:
            return self._state
        if self._last_frame is not None and t - self._last_frame > self.settings.max_gap_s:
            self._unavailable(t, "frame_gap")
        self._last_frame = t
        if self._last_face is not None and t - self._last_face >= 2.0:
            self._recalibrate |= any(b.center is not None for name, b in self.baselines.items() if name in REGIONS)
        if frame is None or face is None:
            return self._unavailable(t, "no_camera_frame" if frame is None else "face_not_visible")
        self._last_face = t
        if self._recalibrate:
            return self._unavailable(t, "recalibration_needed")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            return self._unavailable(t, "color_unavailable")
        if self._last_sample is not None and t - self._last_sample + 1e-9 < self.settings.interval_s:
            return self._state
        self._last_sample = t
        if any(n is None or not isfinite(n) or abs(n) > 25 for n in (face.yaw_deg, face.pitch_deg)):
            return self._unavailable(t, "face_pose")
        x0, y0, x1, y1 = face.bbox
        crop = frame[y0:y1, x0:x1]
        if not crop.size or face.eyes[2] < 25:
            return self._unavailable(t, "face_pose")
        if np.percentile(np.ptp(crop[::8, ::8], axis=2), 95) <= 2:
            return self._unavailable(t, "color_unavailable")
        gray = cv2.cvtColor(cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_32F).var() < self.settings.min_focus:
            return self._unavailable(t, "blurred")

        readings = self._measure_regions(frame, contours, face)
        reference, rect = self._reference_reading(frame, face.bbox)
        light = self._light_state(reference, readings["forehead"])
        eligible = light in ("calibrating", "stable", "estimated") and self._neutral(face)
        if self.reference is not None and light in ("calibrating", "stable"):
            self.baselines["reference"].add(reference, t, self.settings)
            if self.baselines["reference"].center is not None:
                light = "stable"
        else:
            self.baselines["reference"].add(None, t, self.settings)
        regions, deltas, compared = {}, {}, {}
        anchor = reference if self.reference is not None else readings["forehead"].lab
        for name, reading in readings.items():
            baseline = self.baselines[name]
            illuminant = anchor[:3] if anchor is not None and name != "forehead" else None
            baseline.add(reading.lab if eligible else None, t, self.settings, illuminant)
            delta = None if reading.lab is None or baseline.center is None else reading.lab - baseline.center
            if delta is not None:
                deltas[name] = delta
                if illuminant is not None and baseline.illuminant is not None:
                    corrected = compensate_light(reading.lab, illuminant, baseline.illuminant)
                    compared[name] = corrected - baseline.compared_center
                    # A reference can sit under a different light. It must not create a
                    # chromatic event in a region whose observed chromaticity is unchanged.
                    if np.linalg.norm(delta[1:3]) < self.settings.dispersion_floor:
                        compared[name][1:] = delta[1:]
                else:
                    compared[name] = delta
            regions[name] = {"lab": _fields(reading.lab), "delta": _fields(delta), "dispersion": _fields(reading.mad),
                             "pixels": reading.pixels, "sampled": reading.sampled,
                             "valid_samples": round(reading.sampled * (1 - reading.rejected)),
                             "available": reading.lab is not None, "calibrated": baseline.center is not None,
                             "rejected": round(reading.rejected, 3),
                             "calibration_s": round(baseline.seconds, 2),
                             "reason": reading.reason or ("calibrating" if baseline.center is None else None)}
        if light == "calibrating" and self.reference is None and "forehead" in deltas:
            light = "estimated"
        if light in ("estimated", "stable"):
            for name in (*CHEEKS, "lips"):
                regions[name]["compared_delta"] = _fields(compared.get(name))
        lighting_source = "external_patch" if self.reference is not None else "forehead"
        observations = {}
        for kind in OBSERVATIONS:
            if light in ("estimated", "stable"):
                score, details, reason = self._assessment(kind, compared, deltas, lighting_source)
            else:
                score, details, reason = None, {}, light
            observations[kind] = self._advance(kind, t, score, details, reason)
        self._state = self._snapshot(light, regions, observations, rect)
        return self._state

    def draw_masks(self, image):
        if self._mask_lines:
            cv2.drawContours(image, [line.astype(np.int32) for line in self._mask_lines], -1, (255, 215, 55), 1)
        rect = self._state["reference_rect"]
        if rect:
            x0, y0, x1, y1 = rect
            cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (255, 215, 55), 1)


def _fields(values):
    return None if values is None else {key: round(float(value), 2) for key, value in zip(CHANNELS, values)}
