"""Driver attention monitor (R1): drowsiness, distraction, head pose.

New logic for FastGears. Inputs are FaceResult objects from
fastgears.vision.face (MediaPipe landmarks/blendshapes, head pose ported from
afferent-junction2026 posture-analysis/posture/afferent/face.py); the EAR
smoother is the One-Euro filter from posture-analysis/.../filters.py. Blink
definition mirrors the afferent iOS app (eyeBlinkLeft/Right blendshapes).

Deterministic: all timing comes from the `t` argument (seconds), never the
wall clock. Every threshold below can be overridden by passing its
lower-case name to AttentionMonitor(**thresholds), e.g.
AttentionMonitor(microsleep_s=2.0, mirrored=False).

Head-pose sign conventions (MediaPipe facial transformation matrix, via
_euler_from_matrix): +pitch = nose down; +yaw = face turned towards image
right. On a raw (unmirrored) webcam frame image-right is the driver's own
left; fastgears.camera mirrors frames by default, so MIRRORED defaults True
and +yaw then means the driver's right. Labels are from the driver's view.
"""
from __future__ import annotations

from statistics import median

from .drowsiness import DrowsinessSignals, THRESHOLDS
from .filters import OneEuroAxis
from .inattention import INATTENTION_THRESHOLDS, InattentionMonitor

# --- calibration ---------------------------------------------------------------
CALIB_MIN_SAMPLES = 5            # face frames needed to finish calibration
BASELINE_EAR_MIN = 0.15          # clamp learned open-eye EAR into a sane range
BASELINE_EAR_MAX = 0.45
CALIB_BLINK_BS_MAX = 0.3         # skip blinking frames when learning EAR

# --- eyes ------------------------------------------------------------------------
EAR_CLOSED_RATIO = 0.6           # closed when EAR < 0.6 x baseline open EAR
EAR_CLOSED_ABS = 0.20            # fallback absolute threshold before calibration
BLINK_BS_CLOSED = 0.5            # eyeBlinkLeft/Right mean >= 0.5 also = closed
EAR_FULLY_CLOSED_RATIO = 0.3     # EAR of a fully shut eye ~0.3 x open EAR
PERCLOS_CLOSURE = 0.8            # PERCLOS P80: eye >= 80 % closed
PERCLOS_WINDOW_S = 60.0
BLINK_MAX_S = 0.5                # closures shorter than this count as blinks
BLINK_WINDOW_S = 60.0
LONG_CLOSURE_S = 0.5             # closure reported as a reason beyond this
MICROSLEEP_S = 1.5
EAR_MIN_CUTOFF = 1.5             # One-Euro params for the reported EAR
EAR_BETA = 1.0

# --- mouth ----------------------------------------------------------------------
MAR_YAWN = 0.6                   # inner-lip MAR for a yawn
JAW_OPEN_YAWN = 0.6              # or jawOpen blendshape
YAWN_MIN_S = 1.0
YAWN_WINDOW_S = 300.0

# --- gaze / head pose (degrees, relative to calibrated neutral) -------------------
YAW_SIDE_DEG = 25.0              # |gaze yaw| beyond this -> left/right
YAW_AWAY_DEG = 60.0              # beyond this -> away (head fully turned)
PITCH_DOWN_DEG = 20.0            # gaze pitch beyond this -> down (lap, phone)
PITCH_UP_DEG = 20.0              # -> up
IRIS_YAW_GAIN = 25.0             # deg of gaze per unit iris offset (x)
IRIS_PITCH_GAIN = 20.0           # deg of gaze per unit iris offset (y)
FACE_MISSING_AWAY_S = 1.0        # accepted for compatibility with existing settings
DISTRACTED_S = 2.0               # continuous off-road seconds -> distracted
OFF_ROAD_REASON_S = 1.0
MIRRORED = True                  # frames are horizontally flipped (selfie view)

# --- scoring ----------------------------------------------------------------------
PERCLOS_FULL = 0.30              # PERCLOS mapping to drowsiness 100
YAWN_SCORE_EACH = 15.0           # drowsiness per yawn in the last 5 min (cap 60)
PHONE_DISTRACTION = 80.0
LEVEL_CAUTION = 40.0
LEVEL_DANGER = 70.0

_THRESHOLD_NAMES = (
    "CALIB_MIN_SAMPLES", "BASELINE_EAR_MIN", "BASELINE_EAR_MAX",
    "CALIB_BLINK_BS_MAX", "EAR_CLOSED_RATIO", "EAR_CLOSED_ABS",
    "BLINK_BS_CLOSED", "EAR_FULLY_CLOSED_RATIO", "PERCLOS_CLOSURE",
    "PERCLOS_WINDOW_S", "BLINK_MAX_S", "BLINK_WINDOW_S", "LONG_CLOSURE_S",
    "MICROSLEEP_S", "EAR_MIN_CUTOFF", "EAR_BETA", "MAR_YAWN", "JAW_OPEN_YAWN",
    "YAWN_MIN_S", "YAWN_WINDOW_S", "YAW_SIDE_DEG", "YAW_AWAY_DEG",
    "PITCH_DOWN_DEG", "PITCH_UP_DEG", "IRIS_YAW_GAIN", "IRIS_PITCH_GAIN",
    "FACE_MISSING_AWAY_S", "DISTRACTED_S", "OFF_ROAD_REASON_S", "MIRRORED",
    "PERCLOS_FULL", "YAWN_SCORE_EACH", "PHONE_DISTRACTION", "LEVEL_CAUTION",
    "LEVEL_DANGER",
)
DEFAULT_THRESHOLDS = {n.lower(): globals()[n] for n in _THRESHOLD_NAMES} | THRESHOLDS | INATTENTION_THRESHOLDS


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(float(x), nd)


class AttentionMonitor:
    def __init__(self, calib_s: float = 5.0, **thresholds):
        unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
        if unknown:
            raise TypeError(f"unknown thresholds: {sorted(unknown)}")
        self.calib_s = float(calib_s)
        self.th = {**DEFAULT_THRESHOLDS, **thresholds}
        self.signals = DrowsinessSignals(self.th)
        self.inattention = InattentionMonitor(self.th)
        self.reset_calibration()

    def reset_calibration(self) -> None:
        """Forget the driver baseline; the next calib_s seconds re-learn it."""
        self.signals.reset()
        self.inattention.reset()
        self._ear_filter = OneEuroAxis(self.th["ear_min_cutoff"], self.th["ear_beta"])
        self.calibrated = False
        self._calib_start: float | None = None
        self._calib = {"ear": [], "pitch": [], "yaw": [], "roll": [],
                       "ix": [], "iy": []}
        self.base_ear: float | None = None
        self.base_pose = (0.0, 0.0, 0.0)
        self.base_iris = (0.0, 0.0)

    # -- helpers -----------------------------------------------------------------
    @staticmethod
    def _blink_bs(face) -> float | None:
        bs = face.blendshapes or {}
        v = [bs[k] for k in ("eyeBlinkLeft", "eyeBlinkRight") if k in bs]
        return sum(v) / len(v) if v else None

    def _calibrate(self, face, t: float, ear: float, blink: float | None) -> None:
        if self._calib_start is None:
            self._calib_start = t
        c = self._calib
        if blink is None or blink < self.th["calib_blink_bs_max"]:
            c["ear"].append(ear)
        for k, v in (("pitch", face.pitch_deg), ("yaw", face.yaw_deg),
                     ("roll", face.roll_deg)):
            if v is not None:
                c[k].append(float(v))
        if face.iris_offset is not None:
            c["ix"].append(float(face.iris_offset[0]))
            c["iy"].append(float(face.iris_offset[1]))
        n = max(len(c["ear"]), len(c["yaw"]))
        if t - self._calib_start >= self.calib_s and n >= self.th["calib_min_samples"]:
            if c["ear"]:
                self.base_ear = min(self.th["baseline_ear_max"],
                                    max(self.th["baseline_ear_min"], median(c["ear"])))
            med = lambda xs: median(xs) if xs else 0.0  # noqa: E731
            self.base_pose = (med(c["pitch"]), med(c["yaw"]), med(c["roll"]))
            self.base_iris = (med(c["ix"]), med(c["iy"]))
            self.calibrated = True

    def _gaze_label(self, rp: float | None, ry: float | None,
                    iris: tuple[float, float] | None) -> str:
        th = self.th
        gy = ry or 0.0
        gp = rp or 0.0
        if iris is not None:
            gy += th["iris_yaw_gain"] * (iris[0] - self.base_iris[0])
            gp += th["iris_pitch_gain"] * (iris[1] - self.base_iris[1])
        if ry is not None and abs(ry) >= th["yaw_away_deg"]:
            return "away"
        if abs(gy) >= th["yaw_side_deg"] and abs(gy) >= abs(gp):
            # +yaw = towards image right; that is the driver's left unless mirrored
            image_right = gy > 0
            return "right" if image_right == bool(th["mirrored"]) else "left"
        if gp >= th["pitch_down_deg"]:
            return "down"
        if gp <= -th["pitch_up_deg"]:
            return "up"
        if abs(gy) >= th["yaw_side_deg"]:
            image_right = gy > 0
            return "right" if image_right == bool(th["mirrored"]) else "left"
        return "road"

    # -- main ---------------------------------------------------------------------
    def update(self, face, t: float, phone: bool = False) -> dict:
        th = self.th
        t = float(t)
        ear_s = None
        if face is not None:
            ear = (float(face.ear_left) + float(face.ear_right)) / 2.0
            if not self.calibrated:
                self._calibrate(face, t, ear, self._blink_bs(face))
            ear_s = self._ear_filter(ear, t)
        bp, by, br = self.base_pose
        rp = None if face is None or face.pitch_deg is None else face.pitch_deg - bp
        ry = None if face is None or face.yaw_deg is None else face.yaw_deg - by
        rr = None if face is None or face.roll_deg is None else face.roll_deg - br
        metrics = self.signals.update(face, t, rp if self.calibrated and self._calib["pitch"] else None, self.calib_s)
        eyes = metrics["eyes"]
        eyes_closed = all(eyes) if eyes is not None else None
        eye_closed_s = metrics["eye_closed_s"] or 0.0

        direction = None
        unavailable = None
        if face is None:
            unavailable = "face_not_visible"
        elif not self.calibrated:
            unavailable = "calibrating"
        elif eyes_closed:
            unavailable = "eyes_closed"
        elif face.iris_offset is None and (rp is None or ry is None):
            unavailable = "gaze_not_available"
        else:
            direction = self._gaze_label(rp, ry, face.iris_offset)
        gaze = self.inattention.update(direction, t, unavailable)
        looking = gaze["looking"]
        off_road_s = gaze["off_road_s"]
        distracted = gaze["distracted"]

        microsleep = eye_closed_s + 1e-9 >= th["microsleep_s"]
        perclos = metrics["perclos"]
        perclos_score = min(100.0, 100.0 * perclos / th["perclos_full"]) if (
            perclos is not None and metrics["perclos_ready"]) else 0.0
        closure = self.signals.active.get("prolonged_eye_closure")
        severe = bool(closure or microsleep or metrics["nod_detected"] or metrics["recent_nods"] >= th["repeated_nods"]
                      or (perclos_score >= th["level_danger"] and metrics["drooping"]))
        repeated_blinks = metrics["prolonged_blink_count"] >= th["repeated_blinks"]
        possible = perclos_score >= th["level_caution"] or metrics["drooping"] or repeated_blinks
        yawns = metrics["recent_yawns"]
        drowsiness = None if face is None else max(
            perclos_score if severe
            else min(perclos_score, th["level_danger"] - 0.1),
            th["level_danger"] if severe else th["level_caution"] if possible else 0.0,
            min(th["level_caution"] - 0.1, th["yawn_score_each"] * yawns),
        )
        risk = (None if face is None else "HIGH DROWSINESS" if severe
                else "POSSIBLE DROWSINESS" if possible else "NORMAL")
        warning = None
        warning_type = None
        if face is not None:
            if severe and not closure and not metrics["nod_detected"]:
                warning_type = "severe"
            elif metrics["nod_detected"]:
                warning_type = "nod"
            elif closure or (repeated_blinks and not severe):
                warning_type = "eyes"
            elif possible:
                warning_type = "fatigue"
            elif yawns >= th["repeated_yawns"]:
                warning_type = "yawning"
        if warning_type:
            message, voice = {
                "fatigue": ("You might be getting tired. Consider taking a break.", False),
                "yawning": ("Frequent yawning detected. Please stay attentive.", False),
                "eyes": ("Your eyes seem tired. Please find a safe place to rest.", True),
                "nod": ("Warning. You may be falling asleep. Pull over safely.", True),
                "severe": ("Warning. Severe drowsiness detected. Please stop in a safe location.", True),
            }[warning_type]
            severity = "HIGH" if severe or metrics["nod_detected"] else "MEDIUM" if possible else "LOW"
            warning = dict(message=message, reason=warning_type, voice=voice,
                           severity="alert" if severity == "HIGH" else "warning")
            if warning_type in ("eyes", "nod") and severity == "HIGH":
                event = closure if warning_type == "eyes" else next(
                    e for e in self.signals.transitions if e["type"] == "head_nodding")
                warning["event_id"] = event["id"]
            else:
                self.signals.event(warning_type, self.signals.active.get(warning_type, {}).get(
                    "start_s", t), t, "active", severity,
                    perclos=perclos, drooping=metrics["drooping"],
                    recent_yawns=yawns, recent_nods=metrics["recent_nods"],
                    prolonged_blinks=metrics["prolonged_blink_count"])
                warning["event_id"] = self.signals.active[warning_type]["id"]
        for kind, event in list(self.signals.active.items()):
            if kind not in (warning_type, "prolonged_eye_closure"):
                self.signals.event(kind, event["start_s"], t, "ended", event["severity"])

        gaze_warning = gaze["warning"]
        if not phone and gaze_warning and (
            not warning or (gaze_warning["severity"] == "alert" and warning["severity"] != "alert")
        ):
            warning = gaze_warning
        distraction = gaze["score"]
        if phone:
            distraction = max(distraction, th["phone_distraction"])

        reasons: list[str] = []
        if microsleep:
            reasons.append(f"microsleep: eyes closed {eye_closed_s:.1f} s")
        elif eye_closed_s >= th["long_closure_s"]:
            reasons.append(f"eyes closed {eye_closed_s:.1f} s")
        if metrics["perclos_ready"] and perclos is not None and perclos >= 0.15:
            reasons.append(f"PERCLOS {perclos * 100:.0f}%")
        if metrics["drooping"]:
            reasons.append("drooping eyelids")
        if metrics["nod_detected"]:
            reasons.append("head nod")
        if yawns >= th["repeated_yawns"]:
            reasons.append(f"{yawns} yawns in the recent window")
        if unavailable == "face_not_visible":
            reasons.append("driver not visible")
        if off_road_s is not None and off_road_s >= th["off_road_reason_s"]:
            reasons.append(f"looking {looking} {off_road_s:.1f} s")
        if gaze["repeated"]:
            reasons.append(f"{gaze['short_glances']} off-road glances in the recent window")
        if phone:
            reasons.append("phone in hand")

        worst = max(drowsiness or 0.0, distraction)
        if (warning and warning["severity"] == "alert") or worst >= th["level_danger"]:
            level = "danger"
        elif worst >= th["level_caution"]:
            level = "caution"
        else:
            level = "ok"

        return {
            **metrics,
            "face_present": face is not None,
            "eyelids_calibrated": self.signals.baseline is not None,
            "calibrated": self.calibrated,
            "ear": _r(ear_s),
            "ear_left": _r(face.ear_left) if face is not None else None,
            "ear_right": _r(face.ear_right) if face is not None else None,
            "eyes_closed": eyes_closed,
            "eye_closed_s": _r(metrics["eye_closed_s"], 4),
            "blinks_per_min": _r(metrics["blinks_per_min"], 1),
            "microsleep": microsleep,
            "drowsiness_level": risk,
            "active_warning": warning,
            "events": [dict(event) for event in self.signals.active.values()] + gaze["events"],
            "event_transitions": self.signals.transitions + gaze["event_transitions"],
            "inattention": {
                "state": "unavailable" if unavailable else "road" if looking == "road" else "off_road",
                "unavailable_reason": unavailable,
                "short_glances": gaze["short_glances"],
                "repeated": gaze["repeated"],
            },
            "notification_cooldown_s": th["notification_cooldown_s"],
            "pitch": _r(rp, 1),
            "yaw": _r(ry, 1),
            "roll": _r(rr, 1),
            "looking": looking,
            "off_road_s": _r(off_road_s, 2),
            "distracted": bool(distracted),
            "phone": bool(phone),
            "drowsiness": _r(drowsiness, 1),
            "distraction": round(distraction, 1),
            "level": level,
            "reasons": reasons,
        }