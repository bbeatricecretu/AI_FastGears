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

from collections import deque
from statistics import median

from .filters import OneEuroAxis

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
FACE_MISSING_AWAY_S = 1.0        # face lost this long -> "away"
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
DEFAULT_THRESHOLDS = {n.lower(): globals()[n] for n in _THRESHOLD_NAMES}


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(float(x), nd)


class AttentionMonitor:
    def __init__(self, calib_s: float = 5.0, **thresholds):
        unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
        if unknown:
            raise TypeError(f"unknown thresholds: {sorted(unknown)}")
        self.calib_s = float(calib_s)
        self.th = {**DEFAULT_THRESHOLDS, **thresholds}
        self._full_reset()

    # -- state -----------------------------------------------------------------
    def _full_reset(self) -> None:
        self.reset_calibration()
        self._ear_filter = OneEuroAxis(self.th["ear_min_cutoff"],
                                       self.th["ear_beta"])
        self._t0: float | None = None
        self._last_t: float | None = None
        self._closed_since: float | None = None
        self._blinks: deque[float] = deque()
        self._perclos: deque[tuple[float, float, bool]] = deque()  # t, dt, closed80
        self._mouth_open_since: float | None = None
        self._yawn_counted = False
        self._yawns: deque[float] = deque()
        self._face_missing_since: float | None = None
        self._off_road_since: float | None = None
        self._looking = "road"
        self._ear_s: float | None = None
        self._rel = (None, None, None)

    def reset_calibration(self) -> None:
        """Forget the driver baseline; the next calib_s seconds re-learn it."""
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

    def _closure(self, ear: float, blink: float | None) -> float:
        """0 = open (baseline), 1 = fully shut."""
        base = self.base_ear or (self.th["ear_closed_abs"] / self.th["ear_closed_ratio"])
        span = base * (1.0 - self.th["ear_fully_closed_ratio"])
        c = (base - ear) / span if span > 1e-6 else 0.0
        if blink is not None:
            c = max(c, blink)
        return min(1.0, max(0.0, c))

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
        if self._t0 is None:
            self._t0 = t
        dt = 0.0 if self._last_t is None else min(1.0, max(0.0, t - self._last_t))
        self._last_t = t

        eyes_closed = False
        eye_closed_s = 0.0
        yawning = False
        ear_s = None
        rel_p = rel_y = rel_r = None

        if face is not None:
            self._face_missing_since = None
            ear = (float(face.ear_left) + float(face.ear_right)) / 2.0
            blink = self._blink_bs(face)
            if not self.calibrated:
                self._calibrate(face, t, ear, blink)
            ear_s = self._ear_filter(ear, t)
            self._ear_s = ear_s

            # eyes: raw EAR for the decision (blinks are 100-300 ms)
            thr = (th["ear_closed_ratio"] * self.base_ear
                   if self.base_ear else th["ear_closed_abs"])
            eyes_closed = ear < thr or (blink is not None
                                        and blink >= th["blink_bs_closed"])
            if eyes_closed:
                if self._closed_since is None:
                    self._closed_since = t
                eye_closed_s = t - self._closed_since
            elif self._closed_since is not None:
                dur = t - self._closed_since
                if dur <= th["blink_max_s"]:
                    self._blinks.append(t)
                self._closed_since = None
            closure = self._closure(ear, blink)
            self._perclos.append((t, dt, closure >= th["perclos_closure"]))

            # mouth / yawns
            jaw = (face.blendshapes or {}).get("jawOpen", 0.0)
            if float(face.mar) >= th["mar_yawn"] or jaw >= th["jaw_open_yawn"]:
                if self._mouth_open_since is None:
                    self._mouth_open_since = t
                    self._yawn_counted = False
                if t - self._mouth_open_since >= th["yawn_min_s"]:
                    yawning = True
                    if not self._yawn_counted:
                        self._yawns.append(t)
                        self._yawn_counted = True
            else:
                self._mouth_open_since = None

            # head pose relative to the neutral baseline
            bp, by, br = self.base_pose
            rel_p = None if face.pitch_deg is None else face.pitch_deg - bp
            rel_y = None if face.yaw_deg is None else face.yaw_deg - by
            rel_r = None if face.roll_deg is None else face.roll_deg - br
            self._rel = (rel_p, rel_y, rel_r)
            # iris is unreliable with eyes shut; ignore it then
            iris = None if eyes_closed else face.iris_offset
            self._looking = self._gaze_label(rel_p, rel_y, iris)
        else:
            # face lost: eye/mouth episodes cannot continue
            self._closed_since = None
            self._mouth_open_since = None
            if self._face_missing_since is None:
                self._face_missing_since = t
            if t - self._face_missing_since >= th["face_missing_away_s"]:
                self._looking = "away"
                if self._off_road_since is None:
                    self._off_road_since = self._face_missing_since
            # else: hold the last label for brief dropouts

        looking = self._looking
        if looking == "road":
            self._off_road_since = None
        elif self._off_road_since is None:
            self._off_road_since = t
        off_road_s = 0.0 if self._off_road_since is None else t - self._off_road_since
        distracted = off_road_s >= th["distracted_s"]

        # windows
        while self._blinks and t - self._blinks[0] > th["blink_window_s"]:
            self._blinks.popleft()
        while self._perclos and t - self._perclos[0][0] > th["perclos_window_s"]:
            self._perclos.popleft()
        while self._yawns and t - self._yawns[0] > th["yawn_window_s"]:
            self._yawns.popleft()
        elapsed = min(th["blink_window_s"], max(10.0, t - self._t0))
        blinks_per_min = len(self._blinks) * 60.0 / elapsed
        tot = sum(d for _, d, _ in self._perclos)
        perclos = (sum(d for _, d, c in self._perclos if c) / tot) if tot > 0 else 0.0
        microsleep = eye_closed_s >= th["microsleep_s"]
        yawns_5min = len(self._yawns)

        # scores
        closure_score = 0.0
        if eye_closed_s >= th["long_closure_s"]:
            closure_score = min(100.0, 90.0 * eye_closed_s / th["microsleep_s"])
        drowsiness = max(min(100.0, 100.0 * perclos / th["perclos_full"]),
                         closure_score,
                         min(60.0, th["yawn_score_each"] * yawns_5min))
        if yawning:
            drowsiness = min(100.0, drowsiness + 10.0)
        distraction = min(100.0, 70.0 * off_road_s / th["distracted_s"])
        if phone:
            distraction = max(distraction, th["phone_distraction"])

        reasons: list[str] = []
        if microsleep:
            reasons.append(f"microsleep: eyes closed {eye_closed_s:.1f} s")
        elif eye_closed_s >= th["long_closure_s"]:
            reasons.append(f"eyes closed {eye_closed_s:.1f} s")
        if perclos >= 0.15:
            reasons.append(f"PERCLOS {perclos * 100:.0f}%")
        if yawning:
            reasons.append("yawning")
        if yawns_5min >= 2:
            reasons.append(f"{yawns_5min} yawns in 5 min")
        if off_road_s >= th["off_road_reason_s"]:
            if face is None:
                reasons.append(f"face not visible {off_road_s:.1f} s")
            else:
                reasons.append(f"looking {looking} {off_road_s:.1f} s")
        if phone:
            reasons.append("phone in hand")

        worst = max(drowsiness, distraction)
        if microsleep or worst >= th["level_danger"]:
            level = "danger"
        elif worst >= th["level_caution"] or yawning:
            level = "caution"
        else:
            level = "ok"

        rp, ry, rr = self._rel if face is not None else (None, None, None)
        return {
            "face_present": face is not None,
            "calibrated": bool(self.calibrated),
            "ear": _r(ear_s if face is not None else self._ear_s),
            "eyes_closed": bool(eyes_closed),
            "eye_closed_s": round(eye_closed_s, 2),
            "blinks_per_min": round(blinks_per_min, 1),
            "perclos": round(perclos, 3),
            "microsleep": bool(microsleep),
            "yawning": bool(yawning),
            "yawns_5min": int(yawns_5min),
            "pitch": _r(rp, 1),
            "yaw": _r(ry, 1),
            "roll": _r(rr, 1),
            "looking": looking,
            "off_road_s": round(off_road_s, 2),
            "distracted": bool(distracted),
            "phone": bool(phone),
            "drowsiness": round(drowsiness, 1),
            "distraction": round(distraction, 1),
            "level": level,
            "reasons": reasons,
        }
