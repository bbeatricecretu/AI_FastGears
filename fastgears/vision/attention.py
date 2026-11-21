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

import math
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
PERCLOS_WINDOW_S = 300.0         # PERCLOS = share of closed time over the last 5 min
PERCLOS_MIN_S = 60.0             # ...judged only once this much history exists (no "tired" at start-up)
BLINK_MAX_S = 0.5                # closures shorter than this count as blinks
BLINK_WINDOW_S = 60.0
LONG_CLOSURE_S = 0.5             # closure reported as a reason beyond this
EYES_WARN_S = 2.0                # eyes closed this long: warning (caution)
MICROSLEEP_S = 5.0               # eyes closed this long: microsleep (danger)
EAR_MIN_CUTOFF = 1.5             # One-Euro params for the reported EAR
EAR_BETA = 1.0
EYE_HYSTERESIS = 0.02            # per-eye open/closed label: an eye already shut must reopen
                                 # this far above the threshold (stops the label flickering)

# --- mouth ----------------------------------------------------------------------
MAR_YAWN = 0.6                   # inner-lip MAR for a yawn
JAW_OPEN_YAWN = 0.6              # or jawOpen blendshape
YAWN_MIN_S = 1.5                 # talking opens the mouth too, but rarely for 1.5 s
YAWN_WINDOW_S = 300.0
YAWNS_TIRED = 7                  # more than 6 yawns in the 5-min window = tired (caution)
YAWN_POSE_MAX_DEG = 20.0         # mouth shape is only trusted with yaw/roll near neutral
                                 # (a tilted/turned head distorts it: fake yawns). Pitch is
                                 # NOT gated: a real yawn tips the head back ~20-30 deg.

# --- gaze / head pose (degrees, relative to calibrated neutral) -------------------
HEAD_TILT_DEG = 35.0             # head rolled this far (lying sideways): "tilted" - pitch/yaw unreliable
HEAD_TILT_S = 1.0                # tilt held this long counts as nodding off...
HEAD_TILT_DANGER_S = 3.0         # ...and reaches danger (70) after this long
HEAD_DROP_CLOSURE = 0.3          # head pitched up/down past the gaze limits with eyes this
                                 # droopy (0 open .. 1 shut) = head dropping: "nodding",
                                 # scored as drowsiness (same timer/score as a tilt)
HEAD_BACK_DEG = 35.0             # head thrown back this far = nodding even with eyes open
                                 # (no driver watches the road from there)
NOD_MEMORY_S = 3.0               # face lost within this long after "nodding" = still nodding
DROP_PITCH_RISE_DEG = 8.0        # face lost while pitch was rising this much...
DROP_LOOKBACK_S = 0.3            # ...over this window = head falling forward fast (chin to chest)
YAW_SIDE_DEG = 25.0              # |gaze yaw| beyond this -> left/right
YAW_AWAY_DEG = 60.0              # beyond this -> away (head fully turned)
PITCH_DOWN_DEG = 20.0            # gaze pitch beyond this -> down (lap, phone)
PITCH_UP_DEG = 20.0              # -> up
IRIS_YAW_GAIN = 25.0             # deg of gaze per unit iris offset (x)
IRIS_PITCH_GAIN = 20.0           # deg of gaze per unit iris offset (y)
FACE_MISSING_AWAY_S = 1.0        # face lost this long -> "away"
OFF_ROAD_WINDOW_S = 60.0         # distraction = off-road seconds summed over this window...
DISTRACTED_TOTAL_S = 20.0        # ...reaching this total -> distracted (danger), not one short glance
OFF_ROAD_RECOVER_S = 2.0         # back on the road this long clears "distracted" (the total stays,
                                 # so the next glance re-triggers sooner)
OFF_ROAD_REASON_S = 1.0
OFF_ROAD_DECAY_S = 1.0           # back on the road: the off-road score fades with this time constant
                                 # (gone in ~3 s); the 60 s total still makes the next glance count more
MIRRORED = True                  # frames are horizontally flipped (selfie view)

# --- scoring ----------------------------------------------------------------------
PERCLOS_TIRED = 0.15             # PERCLOS at which the trend score reaches caution (40)
PERCLOS_SCORE_MAX = 60.0         # PERCLOS is a slow 60 s trend: it can raise caution, never hold danger
ACUTE_DECAY_S = 2.0              # eye-closure score fades with this time constant once eyes reopen
YAWN_TIRED_SCORE = 50.0          # drowsiness while yawns_5min >= YAWNS_TIRED (caution)
PHONE_WARN_DISTRACTION = 55.0    # phone seen: caution + spoken warning at once...
PHONE_DANGER_S = 10.0            # ...danger once it has been in use this long
PHONE_GAP_S = 2.0                # phone gone this long ends the phone episode
PHONE_DISTRACTION = 80.0
HAND_UP_S = 1.5                  # both hands off the wheel this long (pipeline decides "off";
HAND_UP_DISTRACTION = 45.0       # one hand on the wheel is fine): caution
LEVEL_CAUTION = 40.0
LEVEL_DANGER = 70.0

_THRESHOLD_NAMES = (
    "CALIB_MIN_SAMPLES", "BASELINE_EAR_MIN", "BASELINE_EAR_MAX",
    "CALIB_BLINK_BS_MAX", "EAR_CLOSED_RATIO", "EAR_CLOSED_ABS",
    "BLINK_BS_CLOSED", "EAR_FULLY_CLOSED_RATIO", "PERCLOS_CLOSURE",
    "PERCLOS_WINDOW_S", "PERCLOS_MIN_S", "BLINK_MAX_S", "BLINK_WINDOW_S", "LONG_CLOSURE_S",
    "EYES_WARN_S", "MICROSLEEP_S", "EAR_MIN_CUTOFF", "EAR_BETA", "MAR_YAWN", "JAW_OPEN_YAWN",
    "YAWN_MIN_S", "YAWN_WINDOW_S", "YAWNS_TIRED", "YAW_SIDE_DEG", "YAW_AWAY_DEG",
    "PITCH_DOWN_DEG", "PITCH_UP_DEG", "IRIS_YAW_GAIN", "IRIS_PITCH_GAIN",
    "FACE_MISSING_AWAY_S", "OFF_ROAD_WINDOW_S", "DISTRACTED_TOTAL_S", "OFF_ROAD_RECOVER_S",
    "OFF_ROAD_REASON_S", "OFF_ROAD_DECAY_S", "MIRRORED",
    "PERCLOS_TIRED", "EYE_HYSTERESIS", "PERCLOS_SCORE_MAX", "ACUTE_DECAY_S", "YAWN_TIRED_SCORE",
    "PHONE_WARN_DISTRACTION", "PHONE_DANGER_S", "PHONE_GAP_S", "PHONE_DISTRACTION", "HEAD_TILT_DEG", "HEAD_TILT_S", "HEAD_TILT_DANGER_S", "HEAD_DROP_CLOSURE", "HEAD_BACK_DEG", "NOD_MEMORY_S", "DROP_PITCH_RISE_DEG", "DROP_LOOKBACK_S", "YAWN_POSE_MAX_DEG", "HAND_UP_S", "HAND_UP_DISTRACTION", "LEVEL_CAUTION",
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
        self._eye_closed = [False, False]   # per-eye label state (driver's left, right)
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
        self._acute = 0.0
        self._hand_since: float | None = None
        self._perclos_closed = 0.0                                   # closed seconds in _perclos
        self._perclos_total = 0.0                                    # all seconds in _perclos
        self._off_samples: deque[tuple[float, float]] = deque()     # (t, dt) of off-road frames
        self._off_total = 0.0
        self._last_off_t: float | None = None
        self._phone_since: float | None = None
        self._phone_last: float | None = None
        self._tilt_since: float | None = None
        self._droop_since: float | None = None
        self._dropping = False       # last face frame: head going down with droopy eyes / nodding
        self._last_nod_t: float | None = None
        self._pitch_hist: deque[tuple[float, float]] = deque()   # (t, rel pitch), last ~1 s

    def reset_calibration(self) -> None:
        """Forget the driver baseline; the next calib_s seconds re-learn it."""
        self.calibrated = False
        self._calib_start: float | None = None
        self._calib = {"ear": [], "ear_l": [], "ear_r": [], "pitch": [], "yaw": [], "roll": [],
                       "ix": [], "iy": []}
        self.base_ear: float | None = None
        self.base_ear_lr: tuple[float, float] | None = None   # per eye, driver's left/right
        self.base_pose = (0.0, 0.0, 0.0)
        self.base_iris = (0.0, 0.0)

    # -- helpers -----------------------------------------------------------------
    @staticmethod
    def _blink_bs(face) -> float | None:
        bs = face.blendshapes or {}
        v = [bs[k] for k in ("eyeBlinkLeft", "eyeBlinkRight") if k in bs]
        return sum(v) / len(v) if v else None

    def _driver_ears(self, face) -> tuple[float, float]:
        """(left, right) EAR from the driver's own view. face.ear_right is landmarks 33..133,
        the eye on image-left: the driver's right on a raw frame, their left when mirrored."""
        if self.th["mirrored"]:
            return float(face.ear_right), float(face.ear_left)
        return float(face.ear_left), float(face.ear_right)

    def _calibrate(self, face, t: float, ear: float, blink: float | None) -> None:
        if self._calib_start is None:
            self._calib_start = t
        c = self._calib
        if blink is None or blink < self.th["calib_blink_bs_max"]:
            c["ear"].append(ear)
            el, er = self._driver_ears(face)
            c["ear_l"].append(el)
            c["ear_r"].append(er)
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
                clamp = lambda x: min(self.th["baseline_ear_max"],  # noqa: E731
                                      max(self.th["baseline_ear_min"], x))
                self.base_ear_lr = (clamp(median(c["ear_l"])), clamp(median(c["ear_r"])))
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

    def _fell_forward(self) -> bool:
        """Pitch was still rising fast (head falling forward) in the last frames before the loss."""
        h = self._pitch_hist
        if len(h) < 2:
            return False
        t_last, p_last = h[-1]
        earlier = [p for tt, p in h if t_last - tt >= self.th["drop_lookback_s"] - 1e-6]
        if not earlier:
            return False
        return p_last - earlier[-1] >= self.th["drop_pitch_rise_deg"] and p_last > 0

    # -- main ---------------------------------------------------------------------
    def update(self, face, t: float, phone: bool = False, hand_up: bool = False) -> dict:
        th = self.th
        t = float(t)
        if self._t0 is None:
            self._t0 = t
        dt = 0.0 if self._last_t is None else min(1.0, max(0.0, t - self._last_t))
        self._last_t = t

        eyes_closed = False
        eye_closed_s = 0.0
        eyes_lr: list[bool] | None = None
        ears_lr: tuple[float, float] | None = None
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
            # per-eye open/closed label (display only: the decisions above use both eyes)
            ears_lr = self._driver_ears(face)
            bases = self.base_ear_lr or (self.base_ear, self.base_ear)
            for i, e in enumerate(ears_lr):
                thr_i = (th["ear_closed_ratio"] * bases[i] if bases[i] else th["ear_closed_abs"])
                if self._eye_closed[i]:
                    thr_i += th["eye_hysteresis"]
                self._eye_closed[i] = e < thr_i
            eyes_lr = list(self._eye_closed)
            if eyes_closed and not any(eyes_lr):
                eyes_lr = [True, True]   # closed by the blink blendshapes: both lids down
            closure = self._closure(ear, blink)
            if self.calibrated and dt > 0:
                shut = closure >= th["perclos_closure"]
                self._perclos.append((t, dt, shut))
                self._perclos_total += dt
                self._perclos_closed += dt if shut else 0.0

            # mouth / yawns
            jaw = (face.blendshapes or {}).get("jawOpen", 0.0)
            _, by0, br0 = self.base_pose
            dev = [abs(v - b) for v, b in ((face.yaw_deg, by0), (face.roll_deg, br0))
                   if v is not None]
            pose_ok = not self.calibrated or not dev or max(dev) <= th["yawn_pose_max_deg"]
            mouth_open = pose_ok and (float(face.mar) >= th["mar_yawn"] or jaw >= th["jaw_open_yawn"])
            if mouth_open:
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
            if rel_r is not None and abs(rel_r) >= th["head_tilt_deg"]:
                self._looking = "tilted"
                self._droop_since = None
                if self._tilt_since is None:
                    self._tilt_since = t
            else:
                label = self._gaze_label(rel_p, rel_y, iris)
                if label == "up" and mouth_open:
                    # a yawn tips the head back and a wide-open jaw skews the pose fit
                    label = "road"
                    self._droop_since = None
                elif label in ("up", "down") and (closure >= th["head_drop_closure"]
                                                  or (rel_p is not None and rel_p <= -th["head_back_deg"])):
                    # head fallen back/forward with drooping eyes: nodding off, not looking.
                    # Held longer than a blink, so blinking while looking down stays "down".
                    if self._droop_since is None:
                        self._droop_since = t
                    if t - self._droop_since > th["blink_max_s"]:
                        label = "nodding"
                else:
                    self._droop_since = None
                if label == "nodding":
                    if self._tilt_since is None:
                        self._tilt_since = self._droop_since
                else:
                    self._tilt_since = None
                self._looking = label
            self._dropping = self._looking == "nodding" or (
                rel_p is not None and rel_p >= th["pitch_down_deg"] and closure >= th["head_drop_closure"])
            if self._looking == "nodding":
                self._last_nod_t = t
            if rel_p is not None:
                self._pitch_hist.append((t, rel_p))
            while self._pitch_hist and t - self._pitch_hist[0][0] > 1.0:
                self._pitch_hist.popleft()
        else:
            # face lost: eye/mouth episodes cannot continue
            self._closed_since = None
            self._eye_closed = [False, False]
            self._mouth_open_since = None
            if self._face_missing_since is None:
                self._face_missing_since = t
                # a fast drop takes the face out of view before the pitch/eyes say "nodding":
                # use what led up to the loss
                self._dropping = self._dropping or self._fell_forward() or (
                    self._last_nod_t is not None and t - self._last_nod_t <= th["nod_memory_s"])
            if self._dropping:
                # chin fell to the chest and took the face out of view: nodding off, not away
                self._looking = "nodding"
                if self._tilt_since is None:
                    self._tilt_since = self._droop_since or self._face_missing_since
            else:
                self._droop_since = None
                if t - self._face_missing_since >= th["face_missing_away_s"]:
                    self._looking = "away"
                    if self._off_road_since is None:
                        self._off_road_since = self._face_missing_since
            # else: hold the last label for brief dropouts

        looking = self._looking
        if face is None and looking != "nodding":
            self._tilt_since = None
        tilt_s = 0.0 if self._tilt_since is None else t - self._tilt_since
        if looking in ("road", "tilted", "nodding"):   # drowsiness, not distraction
            self._off_road_since = None
        elif self._off_road_since is None:
            self._off_road_since = t
        off_road_s = 0.0 if self._off_road_since is None else t - self._off_road_since

        # off-road time summed over the last OFF_ROAD_WINDOW_S: many glances add up, one doesn't
        if self._off_road_since is not None:
            # count from the episode start (a lost face is "away" from its first missing frame)
            since = self._off_road_since if self._last_off_t is None else max(self._off_road_since, self._last_off_t)
            add = max(0.0, t - since)
            self._last_off_t = t
            if add > 0:
                self._off_samples.append((t, add))
                self._off_total += add
        while self._off_samples and t - self._off_samples[0][0] > th["off_road_window_s"]:
            self._off_total -= self._off_samples.popleft()[1]
        off_total = max(0.0, self._off_total)
        still_off = self._last_off_t is not None and t - self._last_off_t < th["off_road_recover_s"]
        distracted = off_total >= th["distracted_total_s"] and still_off

        # windows
        while self._blinks and t - self._blinks[0] > th["blink_window_s"]:
            self._blinks.popleft()
        while self._yawns and t - self._yawns[0] > th["yawn_window_s"]:
            self._yawns.popleft()
        while self._perclos and t - self._perclos[0][0] > th["perclos_window_s"]:
            _, d, shut = self._perclos.popleft()
            self._perclos_total -= d
            self._perclos_closed -= d if shut else 0.0
        elapsed = min(th["blink_window_s"], max(10.0, t - self._t0))
        blinks_per_min = len(self._blinks) * 60.0 / elapsed
        perclos = self._perclos_closed / self._perclos_total if self._perclos_total > 1e-6 else 0.0
        perclos = min(1.0, max(0.0, perclos))
        perclos_ready = self._perclos_total >= th["perclos_min_s"]
        microsleep = eye_closed_s >= th["microsleep_s"]
        eyes_closing = eye_closed_s >= th["eyes_warn_s"] and not microsleep
        yawns_5min = len(self._yawns)
        tired_perclos = perclos_ready and perclos >= th["perclos_tired"]
        tired_yawns = yawns_5min >= th["yawns_tired"]
        tired = tired_perclos or tired_yawns

        # phone episode: warning at once, danger after PHONE_DANGER_S of use
        if phone:
            if self._phone_since is None or (self._phone_last is not None
                                             and t - self._phone_last > th["phone_gap_s"]):
                self._phone_since = t
            self._phone_last = t
        elif self._phone_last is not None and t - self._phone_last > th["phone_gap_s"]:
            self._phone_since = self._phone_last = None
        phone_s = 0.0 if not phone or self._phone_since is None else t - self._phone_since

        # scores
        closure_score = 0.0
        warn, micro = th["eyes_warn_s"], th["microsleep_s"]
        if microsleep:
            closure_score = 90.0
        elif eye_closed_s >= warn:     # caution (40) at EYES_WARN_S rising to just under danger
            closure_score = th["level_caution"] + (th["level_danger"] - 1.0 - th["level_caution"]) * (
                (eye_closed_s - warn) / max(1e-6, micro - warn))
        elif eye_closed_s >= th["long_closure_s"]:
            closure_score = (th["level_caution"] - 1.0) * eye_closed_s / warn
        # acute part: a closure keeps the score up only briefly after the eyes reopen
        self._acute = max(closure_score, self._acute * math.exp(-dt / th["acute_decay_s"]))
        perclos_score = 0.0
        if perclos_ready:
            perclos_score = min(th["perclos_score_max"], th["level_caution"] * perclos / th["perclos_tired"])
        yawn_score = th["yawn_tired_score"] if tired_yawns else 0.0
        tilt_score = 0.0
        if tilt_s >= th["head_tilt_s"]:
            tilt_score = min(100.0, th["level_danger"] * tilt_s / th["head_tilt_danger_s"])
        drowsiness = max(perclos_score, self._acute, yawn_score, tilt_score)
        distraction = min(100.0, th["level_danger"] * off_total / th["distracted_total_s"])
        if self._off_road_since is None and self._last_off_t is not None:
            # looking at the road again: the off-road history fades out within seconds
            distraction *= math.exp(-max(0.0, t - self._last_off_t) / th["off_road_decay_s"])
        off_score = distraction
        if hand_up:
            if self._hand_since is None:
                self._hand_since = t
        else:
            self._hand_since = None
        hand_off = self._hand_since is not None and t - self._hand_since >= th["hand_up_s"]
        if hand_off:
            distraction = max(distraction, th["hand_up_distraction"])
        if phone:
            distraction = max(distraction, th["phone_distraction"] if phone_s >= th["phone_danger_s"]
                              else th["phone_warn_distraction"])

        reasons: list[str] = []
        if microsleep:
            reasons.append(f"microsleep: eyes closed {eye_closed_s:.1f} s")
        elif eye_closed_s >= th["long_closure_s"]:
            reasons.append(f"eyes closed {eye_closed_s:.1f} s")
        if tired_perclos:
            reasons.append(f"tired: eyes closed {perclos * 100:.0f}% of the last 5 min")
        if tilt_s >= th["head_tilt_s"]:
            what = "dropping" if looking == "nodding" else "tilted"
            reasons.append(f"head {what} {tilt_s:.1f} s (nodding off?)")
        if yawning:
            reasons.append("yawning")
        if tired_yawns:
            reasons.append(f"tired: {yawns_5min} yawns in 5 min")
        elif yawns_5min >= 2:
            reasons.append(f"{yawns_5min} yawns in 5 min")
        if off_road_s >= th["off_road_reason_s"]:
            if face is None:
                reasons.append(f"face not visible {off_road_s:.1f} s")
            else:
                reasons.append(f"looking {looking} {off_road_s:.1f} s")
        if (off_total >= th["off_road_reason_s"] and off_score >= th["level_caution"]
                and (distracted or off_total >= 0.5 * th["distracted_total_s"])):
            reasons.append(f"eyes off road {off_total:.0f} s in the last {th['off_road_window_s']:.0f} s")
        if phone:
            reasons.append(f"phone in hand {phone_s:.0f} s" if phone_s >= 1.0 else "phone in hand")
        elif hand_off:
            reasons.append("both hands off the wheel")

        worst = max(drowsiness, distraction)
        if microsleep or worst >= th["level_danger"]:
            level = "danger"
        elif worst >= th["level_caution"]:
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
            "eye_left_closed": None if eyes_lr is None else bool(eyes_lr[0]),
            "eye_right_closed": None if eyes_lr is None else bool(eyes_lr[1]),
            "ear_left": _r(ears_lr[0]) if ears_lr else None,
            "ear_right": _r(ears_lr[1]) if ears_lr else None,
            "blinks_per_min": round(blinks_per_min, 1),
            "perclos": round(perclos, 3),
            "microsleep": bool(microsleep),
            "eyes_closing": bool(eyes_closing),
            "tired": bool(tired),
            "yawning": bool(yawning),
            "yawns_5min": int(yawns_5min),
            "pitch": _r(rp, 1),
            "yaw": _r(ry, 1),
            "roll": _r(rr, 1),
            "looking": looking,
            "off_road_s": round(off_road_s, 2),
            "off_road_total_s": round(off_total, 1),
            "distracted": bool(distracted),
            "phone": bool(phone),
            "phone_s": round(phone_s, 1),
            "hand_off_wheel": bool(hand_off),
            "drowsiness": round(drowsiness, 1),
            "distraction": round(distraction, 1),
            "level": level,
            "reasons": reasons,
        }
