"""Temporal facial-expression proxies from the existing MediaPipe FaceResult."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite
from statistics import median

from .face import LEFT_EYE_EAR, MOUTH_CORNERS, MOUTH_INNER_VERT, RIGHT_EYE_EAR


@dataclass(frozen=True)
class Marker:
    region: str
    indices: tuple[int, ...]
    source: str
    threshold: float


# Landmark anchors check framing. Blendshapes estimate motion and can't confirm occlusion.
EYES = LEFT_EYE_EAR + RIGHT_EYE_EAR
NOSE_WINGS = (98, 327)
NOSE_BRIDGE = 168
NOSE_LIFT_THRESHOLD = 0.012  # fraction of inter-eye distance
MARKERS = {
    "au4": Marker("brows", (70, 107, 336, 300), "browDown", 0.18),
    "au6": Marker("cheeks", (50, 280), "cheekSquint", 0.15),
    "au7": Marker("eyelids", EYES, "eyeSquint", 0.16),
    "au9": Marker("nose", (1, 98, 327), "noseSneer", 0.14),
    "au10": Marker("upper_lip", (0, 13, *MOUTH_CORNERS), "mouthUpperUp", 0.18),
    "au43": Marker("eyelids", EYES, "bilateral_ear_closure", 0.80),
    "au25_27": Marker("mouth", MOUTH_INNER_VERT + MOUTH_CORNERS, "mouth_aspect_ratio", 0.15),
}
MAX_GAP_S = 0.25
REACQUIRE_S = 2.0
CALIBRATION_MAX_SAMPLES = 600
RECENT_EVENTS = 8


def _bilateral(blendshapes, stem):
    pair = [blendshapes.get(stem + side) for side in ("Left", "Right")]
    return pair if all(v is not None and isfinite(v) and 0 <= v <= 1 for v in pair) else None


def _in_frame(landmarks, indices, width, height):
    return len(landmarks) > max(indices) and all(
        1 <= landmarks[i][0] < width - 1 and 1 <= landmarks[i][1] < height - 1
        for i in indices
    )


def _span(landmarks, first, second):
    return hypot(*(landmarks[first][:2] - landmarks[second][:2]))


def _nose_wing_lift(face, width, height):
    points = face.landmarks_px
    if not _in_frame(points, (*NOSE_WINGS, NOSE_BRIDGE, 33, 263), width, height):
        return None
    axis_x, axis_y = points[263, :2] - points[33, :2]
    norm = hypot(axis_x, axis_y)
    if norm < 5:
        return None
    # Measure upwards in the eye-aligned plane, even if the frame is mirrored.
    sign = 1 if axis_x >= 0 else -1
    down_x, down_y = -sign * axis_y / norm, sign * axis_x / norm
    offsets = points[NOSE_BRIDGE, :2] - points[list(NOSE_WINGS), :2]
    return tuple(float((x * down_x + y * down_y) / face.eyes[2]) for x, y in offsets)


class DiscomfortMonitor:
    def __init__(self, calib_s=5.0, overlap_s=0.3):
        if not isfinite(overlap_s) or overlap_s <= 0:
            raise ValueError("Discomfort overlap must be a positive duration in seconds")
        self.calib_s = calib_s
        self.overlap_s = overlap_s
        self._last_t = None
        self._last_face_t = None
        self._next_id = 1
        self._recent = deque(maxlen=RECENT_EVENTS)
        self.reset()

    def reset(self):
        self._samples = deque(maxlen=CALIBRATION_MAX_SAMPLES)
        self._valid_s = 0.0
        self._last_neutral = False
        self._baseline = None
        self._last_face_t = None
        self._recent.clear()
        self._clear_episode()

    def _clear_episode(self):
        self._since = self._matched_t = None
        self._frames = 0
        self._evidence = set()
        self._event_id = None

    @staticmethod
    def _read(face, frame_shape, missing_reason):
        if face is None:
            return {}, dict.fromkeys(MARKERS, missing_reason)
        if any(v is None or not isfinite(v) or abs(v) > limit for v, limit in (
            (face.pitch_deg, 25), (face.yaw_deg, 25), (face.roll_deg, 30),
        )) or not isfinite(face.eyes[2]) or face.eyes[2] < 25:
            return {}, dict.fromkeys(MARKERS, "face_pose_or_scale")

        height, width = frame_shape[:2]
        readings, reasons = {}, {}
        for code, marker in MARKERS.items():
            if not _in_frame(face.landmarks_px, marker.indices, width, height):
                reasons[code] = "region_out_of_frame"
                continue
            if code == "au43":
                ears = (face.ear_left, face.ear_right)
                widths = [_span(face.landmarks_px, eye[0], eye[3])
                          for eye in (LEFT_EYE_EAR, RIGHT_EYE_EAR)]
                if min(widths) >= 5 and all(isfinite(ear) and ear >= 0 for ear in ears):
                    readings[code] = ears
            elif code == "au25_27":
                if _span(face.landmarks_px, *MOUTH_CORNERS) >= 5 and isfinite(face.mar) and face.mar >= 0:
                    readings[code] = face.mar
            else:
                pair = _bilateral(face.blendshapes, marker.source)
                if pair is not None:
                    readings[code] = max(pair) if code == "au9" else sum(pair) / 2
            if code == "au9":
                lift = _nose_wing_lift(face, width, height)
                if lift is not None:
                    readings["nose_lift"] = lift
                    readings["nose_pose"] = (face.pitch_deg, face.yaw_deg)
                    continue
            if code not in readings:
                reasons[code] = "missing_signal"
        return readings, reasons

    def _calibrate(self, face, readings, dt):
        blink = _bilateral(face.blendshapes, "eyeBlink") if face else None
        ears = readings.get("au43")
        # Region coefficients may be absent. Collect available signals while eyes are open.
        neutral = bool(ears and min(ears) >= 0.15 and (blink is None or max(blink) < 0.3)
                       and readings.get("au25_27", 0) < 0.25
                       and all(readings.get(code, 0) < 0.35 for code in ("au4", "au7", "au9", "au10")))
        if neutral:
            self._samples.append(readings)
            if self._last_neutral:
                self._valid_s += dt
        self._last_neutral = neutral
        if not neutral or self._valid_s + 1e-9 < self.calib_s or len(self._samples) < 5:
            return

        self._baseline = {}
        for code in MARKERS:
            values = [sample[code] for sample in self._samples if code in sample]
            if len(values) < 0.7 * len(self._samples):
                continue
            if code == "au43":
                self._baseline[code] = tuple(median(side) for side in zip(*values))
            else:
                centre = median(values)
                mad = median(abs(value - centre) for value in values)
                self._baseline[code] = centre, mad
        nose_samples = [sample["nose_lift"] for sample in self._samples if "nose_lift" in sample]
        if len(nose_samples) >= 0.7 * len(self._samples):
            centres = tuple(median(side) for side in zip(*nose_samples))
            noise = tuple(median(abs(x - centre) for x in side)
                          for side, centre in zip(zip(*nose_samples), centres))
            self._baseline["nose_lift"] = (centres, noise)
            poses = [sample["nose_pose"] for sample in self._samples if "nose_pose" in sample]
            self._baseline["nose_pose"] = tuple(median(angle) for angle in zip(*poses))
        self._samples.clear()

    def _nose_evidence(self, face, readings):
        if self._baseline is None or "nose_lift" not in self._baseline or "nose_lift" not in readings:
            return None
        neutral_pitch, neutral_yaw = self._baseline["nose_pose"]
        if max(abs(face.pitch_deg - neutral_pitch), abs(face.yaw_deg - neutral_yaw)) > 10:
            return None
        centres, noise = self._baseline["nose_lift"]
        evidence = [(value - centre, max(NOSE_LIFT_THRESHOLD, 4 * mad))
                    for value, centre, mad in zip(readings["nose_lift"], centres, noise)]
        return max(evidence, key=lambda candidate: candidate[0] / candidate[1])

    def _markers(self, face, readings, reasons):
        markers = {}
        for code, marker in MARKERS.items():
            reason = reasons.get(code)
            if reason is None and self._baseline is None:
                reason = "calibrating"
            if reason is None and (code not in self._baseline or code not in readings):
                reason = "no_neutral_reference"
            value = threshold = active = None
            source = marker.source
            unit = {"au43": "closure_fraction", "au25_27": "mar_delta"}.get(code, "score_delta")
            if reason is None:
                reference = self._baseline[code]
                threshold = marker.threshold
                if code == "au43":
                    left, right = (1 - ear / base for ear, base in zip(readings[code], reference))
                    value = max(0.0, min(1.0, left, right))
                else:
                    centre, mad = reference
                    value = readings[code] - centre
                    threshold = max(threshold, 4 * mad)
                active = value >= threshold
            if code == "au9":
                lift = self._nose_evidence(face, readings)
                if lift is not None and (value is None or lift[0] / lift[1] >= value / threshold):
                    value, threshold = lift
                    reason, active = None, value >= threshold
                    source, unit = "nose_wing_lift", "eye_distance_fraction"
            markers[code] = {"region": marker.region, "source": source, "unit": unit,
                             "available": reason is None, "reason": reason,
                             "value": None if value is None else round(value, 4),
                             "threshold": None if threshold is None else round(threshold, 4),
                             "active": active}
        return markers

    def _event(self, state):
        return {"id": self._event_id, "type": "discomfort_grimace", "state": state,
                "started_monotonic_s": self._since,
                "duration_ms": round((self._matched_t - self._since) * 1000),
                "evidence": sorted(self._evidence), "observed_frames": self._frames,
                "coverage": 1.0}

    def _end_episode(self, state):
        if self._event_id is not None:
            self._recent.append(self._event(state))
        self._clear_episode()

    def update(self, face, t, frame_shape=None, missing_reason="face_not_visible"):
        # The platform clock can give two updates the same timestamp. Only
        # elapsed time contributes to calibration and gesture duration.
        if self._last_t is None:
            dt = 0.0
        else:
            t = max(t, self._last_t)
            dt = t - self._last_t
        self._last_t = t
        if dt > MAX_GAP_S:
            self._end_episode("interrupted")
            dt = 0.0
        if self._last_face_t is not None and t - self._last_face_t >= REACQUIRE_S:
            self.reset()
        if face is not None:
            self._last_face_t = t
        readings, reasons = self._read(face, frame_shape, missing_reason)
        if self._baseline is None:
            self._calibrate(face, readings, dt)
        markers = self._markers(face, readings, reasons)
        active = {code for code, marker in markers.items() if marker["active"]}
        available = {code for code, marker in markers.items() if marker["available"]}
        core = {"au4", "au7"}
        lower = {"au9", "au10"}
        observable = core <= available and bool(lower & available)
        matching = core <= active and bool(lower & active)
        if matching:
            if self._since is None:
                self._since = t
            self._matched_t = t
            self._frames += 1
            self._evidence.update(active & (core | lower))
            if self._event_id is None and t - self._since + 1e-9 >= self.overlap_s:
                self._event_id = self._next_id
                self._next_id += 1
        else:
            ended = observable and (not self._evidence or bool(lower & self._evidence & available))
            self._end_episode("ended" if ended else "interrupted")

        return {"calibrated": self._baseline is not None, "calibration_s": round(self._valid_s, 2),
                "observable": observable, "pending_ms": round((t - self._since) * 1000) if matching else 0,
                "markers": markers, "event": self._event("active") if self._event_id else None,
                "recent_events": list(self._recent)}
