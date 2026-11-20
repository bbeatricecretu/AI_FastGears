"""Temporal measurements for the existing attention score (seconds, monotonic)."""
from collections import deque
from statistics import mean, median
from uuid import uuid4


THRESHOLDS = dict(
    prolonged_s=0.6, tracking_gap_s=0.25, eye_hysteresis=0.03,
    perclos_short_s=0.2, perclos_coverage=0.7, prolonged_blink_s=0.3,
    droop_ratio=0.8, droop_s=1.0, droop_mad=3.0,
    neutral_pitch_deg=15.0, neutral_yaw_deg=20.0,
    yawn_window_short_s=60.0, yawn_max_s=6.0, mouth_release_ratio=0.6,
    neutral_mouth_ratio=0.5, yawn_min_peak_ratio=1.1, nod_deg=15.0, nod_recovery_deg=6.0,
    nod_min_s=0.15, nod_max_s=1.2, repeated_nods=2, repeated_blinks=2,
    repeated_yawns=2, nod_window_s=60.0, notification_cooldown_s=15.0,
)


class DrowsinessSignals:
    def __init__(self, th):
        self.th = th
        self.reset()

    def reset(self):
        self.start = self.last = None
        self.previous_p80 = None
        self.previous_ear = None
        self.coverage = 0.0
        self.pre_blink = None
        self.eye_samples = [[], []]
        self.calibration_time = 0.0
        self.calibration_valid = False
        self.baseline = None
        self.closed_since = [None, None]
        self.intervals, self.blinks, self.yawns, self.nods = (
            deque(), deque(), deque(), deque()
        )
        self.blink_samples = deque()
        self.normal_blink_durations = []
        self.p80_token = None
        self.short_episodes = set()
        self.mouth = self.nod = self.droop = None
        self.active = {}
        self.transitions = []
        self.blink_total = self.yawn_total = self.nod_total = 0

    def event(self, kind, start, t, status, severity, **evidence):
        if kind not in self.active:
            self.active[kind] = {"id": uuid4().hex, "type": kind, "start_s": start}
        record = self.active[kind]
        changed = "status" not in record or status != "active"
        record.update(duration_s=round(t - start, 4), status=status,
                      severity=severity, evidence=evidence, confidence=None,
                      coverage=getattr(self, "coverage", 0.0))
        if changed:
            self.transitions.append(dict(record))
        if status != "active":
            del self.active[kind]

    def interrupt(self, t):
        for kind, record in list(self.active.items()):
            self.event(kind, record["start_s"], t, "interrupted",
                       record["severity"], **record["evidence"])
        self.previous_p80 = self.previous_ear = self.p80_token = None
        self.closed_since = [None, None]
        self.mouth = self.nod = self.droop = None
        self.blink_samples.clear()
        self.calibration_valid = False

    def calibrate(self, ears, blink, mar, jaw, pitch, yaw, dt, calib_s):
        th = self.th
        valid = (min(ears) >= th["ear_closed_abs"]
                 and max(blink) < th["calib_blink_bs_max"]
                 and mar < th["mar_yawn"] * th["neutral_mouth_ratio"]
                 and jaw < th["jaw_open_yawn"] * th["neutral_mouth_ratio"]
                 and (pitch is None or abs(pitch) <= th["neutral_pitch_deg"])
                 and (yaw is None or abs(yaw) <= th["neutral_yaw_deg"]))
        if valid and self.baseline is None:
            if self.calibration_valid:
                self.calibration_time += dt
            for samples, ear in zip(self.eye_samples, ears):
                samples.append(ear)
            if self.calibration_time + 1e-9 >= calib_s:
                self.baseline = []
                for samples in self.eye_samples:
                    center = median(samples)
                    self.baseline.append((center, median(abs(x - center) for x in samples)))
                self.eye_samples = [[], []]
        self.calibration_valid = valid

    def update(self, face, t, pitch, calib_s):
        th = self.th
        self.transitions = []
        if self.start is None:
            self.start = t
        dt = t - self.last if self.last is not None else 0
        valid_interval = self.previous_p80 is not None and face is not None and 0 < dt <= th["tracking_gap_s"]
        if not valid_interval and self.last is not None:
            self.interrupt(self.last)
        if valid_interval:
            self.intervals.append((self.last, t, self.previous_p80, self.p80_token))
        self.last_observed_t, self.last = self.last, t
        eyes = None
        duration = None
        per_eye_duration = None
        yawning = False
        drooping = False
        nod_detected = False
        if face is not None:
            ears = (float(face.ear_left), float(face.ear_right))
            bs = face.blendshapes or {}
            blink = (bs.get("eyeBlinkLeft", 0), bs.get("eyeBlinkRight", 0))
            mar, jaw = float(face.mar), bs.get("jawOpen", 0)
            self.calibrate(ears, blink, mar, jaw, face.pitch_deg, face.yaw_deg, dt if valid_interval else 0, calib_s)
            bases = (self.baseline if self.baseline else
                     [(th["ear_closed_abs"] / th["ear_closed_ratio"], 0)] * 2)
            p80 = True
            for i, (ear, b, (base, _)) in enumerate(zip(ears, blink, bases)):
                started = self.closed_since[i]
                hysteresis = th["eye_hysteresis"] if started is not None else 0
                is_closed = (ear < base * th["ear_closed_ratio"] + hysteresis
                             or b >= th["blink_bs_closed"] - hysteresis)
                self.closed_since[i] = (t if started is None else started) if is_closed else None
                closure = (base - ear) / (base * (1 - th["ear_fully_closed_ratio"]))
                p80 &= max(closure, b) >= th["perclos_closure"]
            eyes = tuple(start is not None for start in self.closed_since)
            per_eye_duration = tuple(round(t - start, 4) if start is not None else 0.0
                                     for start in self.closed_since)
            duration = self.measure_blink(t, all(eyes), mean(ears))
            if p80 and self.p80_token is None:
                self.p80_token = t
            elif not p80 and self.p80_token is not None:
                if t - self.p80_token < th["perclos_short_s"]:
                    self.short_episodes.add(self.p80_token)
                self.p80_token = None
            yawning = self.measure_yawn(t, mar, jaw)
            nod_detected = self.measure_nod(t, pitch)
            partial = (self.baseline is not None and not any(eyes)
                       and all(ear < min(base * th["droop_ratio"], base - th["droop_mad"] * mad)
                               for ear, (base, mad) in zip(ears, bases)))
            self.droop = (t if self.droop is None else self.droop) if partial else None
            drooping = self.droop is not None and t - self.droop >= th["droop_s"]
            self.previous_p80 = p80
            self.previous_ear = mean(ears)
        cutoff = t - th["perclos_window_s"]
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        oldest = self.intervals[0][0] if self.intervals else t
        self.short_episodes = {token for token in self.short_episodes
                               if token + th["perclos_short_s"] > oldest}
        observed = closed = 0.0
        for start, end, p80, token in self.intervals:
            span = end - max(start, cutoff)
            observed += span
            if (p80 and token not in self.short_episodes
                    and (token != self.p80_token or t - token >= th["perclos_short_s"])):
                closed += span
        self.coverage = min(1.0, observed / th["perclos_window_s"])
        for history, window in ((self.blinks, th["blink_window_s"]),
                                (self.yawns, th["yawn_window_s"]),
                                (self.nods, th["nod_window_s"])):
            while history and t - history[0][0] > window:
                history.popleft()
        perclos = closed / observed if face is not None and observed else None
        recent_yawns = sum(t - end <= th["yawn_window_short_s"] for end, *_ in self.yawns)
        prolonged = sum(d >= th["prolonged_blink_s"] for _, d, *_ in self.blinks)
        recent_blink = self.blinks[-1] if self.blinks and face is not None else None
        blink_baseline = (median(self.normal_blink_durations)
                          if len(self.normal_blink_durations) >= th["calib_min_samples"] else None)
        return dict(
            blink_baseline_s=blink_baseline if face is not None else None,
            blink_duration_delta_s=recent_blink[1] - blink_baseline
            if recent_blink and blink_baseline is not None else None,
            eyes=eyes, eye_closed_s=duration, per_eye_closed_s=per_eye_duration, perclos=perclos,
            perclos_coverage=self.coverage,
            perclos_ready=t - self.start >= th["perclos_window_s"]
            and self.coverage >= th["perclos_coverage"],
            blink_count=self.blink_total, prolonged_blink_count=prolonged,
            blink_duration_s=recent_blink[1] if recent_blink else None,
            average_blink_s=mean(row[1] for row in self.blinks) if recent_blink else None,
            closing_speed=recent_blink[2] if recent_blink else None,
            opening_speed=recent_blink[3] if recent_blink else None,
            blinks_per_min=len(self.blinks) * 60 / min(th["blink_window_s"], t - self.start)
            if face is not None and t > self.start else None,
            yawns_5min=len(self.yawns), yawn_count=self.yawn_total,
            yawns_per_min=recent_yawns * 60 / th["yawn_window_short_s"] if face is not None else None,
            recent_yawns=recent_yawns, yawning=yawning if face is not None else None,
            drooping=drooping if face is not None and self.baseline is not None else None,
            nod_count=self.nod_total if pitch is not None else None,
            recent_nods=len(self.nods), nod_detected=nod_detected,
            tracking_confidence=None, frame_timestamp_s=t,
        )

    def measure_blink(self, t, bilateral, ear):
        th = self.th
        samples = self.blink_samples
        if bilateral:
            if not samples:
                self.pre_blink = (self.last_observed_t, self.previous_ear)
            samples.append((t, ear))
            start = samples[0][0]
            duration = t - start
            if duration + 1e-9 >= th["prolonged_s"]:
                self.event("prolonged_eye_closure", start, t, "active", "HIGH",
                           bilateral=True, ear=ear)
            # Keep the start and enough samples for short blink measurements.
            if len(samples) > 3 and duration > max(th["blink_max_s"], th["prolonged_s"]):
                samples.pop()
            return duration
        if samples:
            start = samples[0][0]
            duration = t - start
            if duration <= th["blink_max_s"]:
                closing = opening = None
                if len(samples) >= 2 and self.pre_blink[1] is not None:
                    low_t, low = min(samples, key=lambda sample: sample[1])
                    open_t, open_ear = self.pre_blink
                    if open_ear > low:
                        closing = (open_ear - low) / (low_t - open_t)
                    if ear > low:
                        opening = (ear - low) / (t - low_t)
                if (duration < th["prolonged_blink_s"]
                        and len(self.normal_blink_durations) < th["calib_min_samples"]):
                    self.normal_blink_durations.append(duration)
                self.blink_total += 1
                self.blinks.append((t, duration, closing, opening))
            if "prolonged_eye_closure" in self.active:
                self.event("prolonged_eye_closure", start, t, "ended", "HIGH", bilateral=True)
            samples.clear()
        return 0.0

    def measure_yawn(self, t, mar, jaw):
        th = self.th
        amplitude = max(mar / th["mar_yawn"], jaw / th["jaw_open_yawn"])
        if self.mouth is None:
            if amplitude >= 1:
                self.mouth = (t, amplitude)
            return False
        start, peak = self.mouth
        duration = t - start
        if amplitude < th["mouth_release_ratio"]:
            if (th["yawn_min_s"] <= duration <= th["yawn_max_s"]
                    and peak >= th["yawn_min_peak_ratio"]):
                self.yawn_total += 1
                self.yawns.append((t,))
                self.event("yawn", start, t, "ended", "LOW", peak_ratio=peak)
            self.mouth = None
            return False
        self.mouth = (start, max(peak, amplitude))
        return th["yawn_min_s"] <= duration <= th["yawn_max_s"]

    def measure_nod(self, t, pitch):
        th = self.th
        if pitch is None:
            self.nod = None
            return False
        if self.nod is None:
            if abs(pitch) <= th["nod_recovery_deg"]:
                self.nod = (None, 0.0)
            return False
        start, peak = self.nod
        if start is None:
            if pitch >= th["nod_deg"]:
                self.nod = (t, pitch)
            return False
        if abs(pitch) <= th["nod_recovery_deg"]:
            self.nod = (None, 0.0)
            if th["nod_min_s"] <= t - start <= th["nod_max_s"]:
                self.nod_total += 1
                self.nods.append((t,))
                self.event("head_nodding", start, t, "ended", "HIGH", peak_deg=peak)
                return True
        else:
            self.nod = (start, max(pitch, peak))
        return False