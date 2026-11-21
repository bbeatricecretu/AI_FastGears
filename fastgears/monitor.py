"""Session-scoped episode tracking, local alert arbitration and bounded pattern mining."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import threading
import time
from uuid import uuid4

from .llm.coach import local_message

FATIGUE = {"EYES_CLOSED", "MICROSLEEP", "YAWNING", "HEAD_DROPPING", "DRIVER_DROWSY"}
AWAY = {"LOOKING_LEFT", "LOOKING_RIGHT", "LOOKING_DOWN", "LOOKING_UP", "DRIVER_DISTRACTED"}


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def uid():
    return uuid4().hex


@dataclass(frozen=True)
class Thresholds:
    window: float = 30
    minimum: float = .45
    recovery: float = .25
    count: int = 3
    cooldown: float = 12
    support: int = 3
    growth: float = .25
    hold: float = 4


def signals(state):
    a = state.get("attention", {})
    result = {}
    if a.get("phone"):
        result["phone"] = ("PHONE_DETECTED", "attention")
    if not a.get("face_present") or not a.get("calibrated"):
        return result
    looking = a.get("looking")
    if looking in ("left", "right", "down", "up", "away") or a.get("distracted"):
        kind = "LOOKING_" + str(looking).upper()
        result["gaze"] = (kind if kind in AWAY else "DRIVER_DISTRACTED", "attention")
    if a.get("eyes_closed") or a.get("microsleep"):
        result["eyes"] = ("MICROSLEEP" if a.get("microsleep") else "EYES_CLOSED", "attention")
    if a.get("yawning"):
        result["mouth"] = ("YAWNING", "attention")
    if state.get("driver_posture", {}).get("active") and state["driver_posture"].get("state") == "nodding":
        result["posture"] = ("HEAD_DROPPING", "driver_posture")
    warning = state.get("driver_attention", {}).get("active_warning") or {}
    if warning.get("reason") == "severe":
        result["fatigue"] = ("DRIVER_DROWSY", "driver_attention")
    elif state.get("driver_signals", {}).get("drowsy") and "eyes" not in result:
        result["fatigue"] = ("DRIVER_DROWSY", "driver_signals")
    return result


class Monitor:
    def __init__(self, history, publish, thresholds=None):
        self.history, self.publish = history, publish
        self.cfg = thresholds or Thresholds()
        self.session = dict(id=uid(), timestamp=utc(), status="running", end_time=None)
        self.session["start_time"] = self.session["timestamp"]
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.open = {}
        self.events = deque(maxlen=128)
        self.alerts = deque(maxlen=64)
        self.occurrences = deque(maxlen=64)
        self.active = None
        self.expires = 0
        self.last_critical = None
        self.last_caution = None
        self.last_match = {}
        self.counts = Counter()
        self.event_order = 0
        self.detector_cooldown = {}
        self.patterns = {p["id"]: p for p in history.recent("patterns", 512)}
        self.candidates = defaultdict(lambda: deque(maxlen=max(2, self.cfg.support)))
        self.history.submit([("sessions", self.session.copy())])
        self.batch = []
        previous = deque(maxlen=128)
        self.seeding = True
        rows = history.recent("events", 512)
        for rec in sorted(rows, key=self._episode_order):
            if rec["status"] != "resolved" or rec["type"] == "ATTENTION_RECOVERED":
                continue
            if previous and previous[-1]["session_id"] != rec["session_id"]:
                previous.clear()
            previous.append(rec)
            self.mine([e for e in previous if e["start"] >= rec["end"] - self.cfg.window], rec["end"])
        self.seeding = False
        self.history.submit(self.batch)
        self.batch = []

    @staticmethod
    def _episode_order(record):
        return record["session_id"], record.get("ordinal", record["end"])

    def record(self, **fields):
        return dict(id=uid(), timestamp=utc(), session_id=self.session["id"], **fields)

    def snapshot(self):
        with self.lock:
            return dict(active=self.active, events=list(self.events)[-96:], alerts=list(self.alerts)[-20:],
                        occurrences=list(self.occurrences)[-20:], counts=dict(self.counts),
                        duration_s=int(time.monotonic()-self.started), session_id=self.session["id"],
                        storage_error=self.history.error, dropped_batches=self.history.dropped)

    def observe(self, state, now=None):
        now = state.get("sample_monotonic_s", time.monotonic()) if now is None else now
        with self.lock:
            self.batch = []
            self.pattern_alerts = []
            a = state.get("attention", {})
            visible = state.get("status") == "running" and a.get("face_present") and a.get("calibrated")
            current = signals(state) if state.get("status") == "running" else {}
            loc = local_message(state) if state.get("status") == "running" else None
            extra = state.get("driver_attention", {}).get("active_warning") or {}
            critical = extra.get("message") if extra.get("severity") == "alert" else None
            if state.get("status") != "running":
                critical = None
            elif a.get("level") == "danger" or loc and loc["severity"] == "alert":
                critical = loc["message"] if loc else "Focus on the road."
            if self.active and self.active["source"] == "detector" and self.last_critical and not critical:
                self.resolve()
            if self.active and now >= self.expires and not (critical and self.last_critical == critical):
                self.resolve()
            if critical and critical != self.last_critical:
                self.alert("WARNING", critical, None, "detector", now)
            caution = loc["message"] if loc and loc["severity"] == "warning" else None
            if (caution and not critical and caution != self.last_caution
                    and now - self.detector_cooldown.get(caution, -1e12) >= self.cfg.cooldown):
                self.detector_cooldown[caution] = now
                self.alert("CAUTION", caution, None, "detector", now)
            self.last_caution = caution
            self.last_critical = critical
            for channel, (kind, source) in current.items():
                if channel not in self.open:
                    rec = self.record(type=kind, source=source, start=now, end=now, duration=0,
                                      detector_severity=a.get("level"), severity="CAUTION",
                                      confidence=None, diagnostics={}, status="active")
                    self.open[channel] = rec
                rec = self.open[channel]
                if kind == "MICROSLEEP":
                    rec["type"] = kind
                if critical:
                    rec["detector_severity"] = a.get("level")
                    rec["severity"] = "WARNING"
                rec.update(end=now, duration=round(now-rec["start"], 3))
                rec["diagnostics"] = dict(looking=a.get("looking"), phone=bool(a.get("phone") or
                                           rec["diagnostics"].get("phone")))
                if channel == "phone":
                    rec["diagnostics"]["phone_source"] = state.get("hands", {}).get("phone_source")
                    scores = [d["conf"] for d in state.get("objects", []) if d.get("label") == "cell phone"
                              and isinstance(d.get("conf"), (int, float))]
                    if scores:
                        rec["diagnostics"]["highest_phone_detection_confidence"] = max(scores)
                if not rec.get("qualified") and (critical or rec["duration"] >= self.cfg.minimum):
                    rec["qualified"] = True
                    self.batch.append(("events", rec.copy()))
            for channel, rec in list(self.open.items()):
                if channel in current or visible and now-rec["end"] < self.cfg.recovery:
                    continue
                del self.open[channel]
                if rec["duration"] < self.cfg.minimum and rec["severity"] != "WARNING":
                    continue
                self.event_order += 1
                rec["ordinal"] = self.event_order
                rec["resolved_at"] = utc()
                rec["status"] = "resolved" if visible else "interrupted"
                self.events.append(rec)
                self.batch.append(("events", rec.copy()))
                if visible:
                    self.match(rec, now)
                    if channel == "gaze" and a.get("looking") == "road":
                        recovery = self.record(type="ATTENTION_RECOVERED", source="attention", start=now,
                                               end=now, duration=0, severity="NORMAL", confidence=None,
                                               detector_severity=a.get("level"), diagnostics={}, status="resolved")
                        self.events.append(recovery)
                        self.batch.append(("events", recovery))
            if self.pattern_alerts:
                severity, message, pattern, source = max(self.pattern_alerts, key=lambda match: match[0]=="WARNING")
                if not critical:
                    self.alert(severity, message, pattern, source, now)
            self.history.submit(self.batch)

    def resolve(self):
        if self.active:
            self.active.update(status="resolved", resolved_at=utc())
            self.batch.append(("alerts", self.active.copy()))
            self.active = None

    def alert(self, severity, message, pattern, source, now):
        if self.active and self.active["severity"] == "WARNING" and severity == "CAUTION":
            return
        self.resolve()
        item = self.record(severity=severity, message=message, pattern_id=pattern and pattern["id"],
                           name=pattern["name"] if pattern else "Driver attention", source=source,
                           explanation=pattern["description"] if pattern else message,
                           status="active", delivery_status="published")
        self.active, self.expires = item, now + self.cfg.hold
        self.alerts.append(item)
        self.counts[severity] += 1
        self.batch.append(("alerts", item.copy()))
        self.publish({"type":"alert", **item})

    def match(self, event, now):
        window = [e for e in self.events if e["start"] >= now-self.cfg.window and
                  e["status"] == "resolved" and e["type"] != "ATTENTION_RECOVERED"]
        gaze = [e for e in window if e["type"] in AWAY]
        fatigue = []
        for rec in window:
            if rec["type"] in FATIGUE and (not fatigue or rec["start"] > fatigue[-1]["end"]):
                fatigue.append(rec)
        n = self.cfg.count
        def rule(name, evidence, severity, message, **criteria):
            self.occurrence(name, evidence, severity, message, now, "configured",
                            dict(rule=name, window_s=self.cfg.window, minimum_s=self.cfg.minimum, **criteria))
        if event["type"] in AWAY and len(gaze) >= n:
            rule("Repeated distraction", gaze[-n:], "CAUTION", "Bring your attention back to the road.",
                 event_types=sorted(AWAY), count=n)
            if all(x["duration"] + self.cfg.growth <= y["duration"] for x,y in zip(gaze[-n:], gaze[-n+1:])):
                rule("Escalating inattention", gaze[-n:], "WARNING", "Keep your eyes on the road.",
                     event_types=sorted(AWAY), count=n, duration_growth_s=self.cfg.growth)
        if event["type"] in FATIGUE and len(fatigue) >= n:
            rule("Repeated fatigue indicators", fatigue[-n:], "WARNING", "Take a break when safe.",
                 event_types=sorted(FATIGUE), count=n)
        if len(gaze) >= 2 and fatigue and event["type"] in AWAY | FATIGUE:
            rule("Fatigue and distraction", [*gaze[-2:], fatigue[-1]], "WARNING", "Refocus and take a break when safe.",
                 gaze_count=2, fatigue_count=1)
        if event["type"] in AWAY and event["diagnostics"]["phone"]:
            rule("Phone-related distraction", [event], "WARNING", "Put the phone away when safe.",
                 event_types=sorted(AWAY), overlapping_phone=True)
        history = [e for e in self.events if e["type"] != "ATTENTION_RECOVERED" and e["status"] == "resolved"]
        self.match_saved(history, now)
        if window and window[-1]["id"] == event["id"]:
            self.mine(window, now)

    def validate_suggestion(self, types):
        """A model may propose only an ordered list of locally observed event types."""
        if not isinstance(types, list) or len(types) not in (2, 3) or not all(
            isinstance(kind, str) and kind in AWAY | FATIGUE | {"PHONE_DETECTED"} for kind in types
        ):
            return False
        rule = dict(mode="sequence", types=types, window_s=self.cfg.window)
        key = hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()[:20]
        # This method runs on the coach worker. Database reads never take the camera lock.
        rows = self.history.recent("events", 512)
        rows = sorted((r for r in rows if r["status"] == "resolved" and r["type"] != "ATTENTION_RECOVERED"),
                      key=self._episode_order)
        matches = []
        recent = deque(maxlen=len(types))
        for row in rows:
            if recent and recent[0]["session_id"] != row["session_id"]:
                recent.clear()
            recent.append(row)
            if [r["type"] for r in recent] == types and row["end"]-min(r["start"] for r in recent) <= self.cfg.window:
                matches.append(list(recent))
                recent.clear()
            if len(matches) >= self.cfg.support:
                break
        if len(matches) < self.cfg.support:
            return False
        with self.lock:
            if key in self.patterns:
                return True
            self.batch = []
            for group in matches:
                self.occurrence("Sequence · " + " / ".join(types), group, "NORMAL",
                                "Observed in separate episodes.", time.monotonic(),
                                "gemini-suggested", rule, key, historical=True)
            self.history.submit(self.batch)
        return True

    def match_saved(self, history, now):
        for pattern in list(self.patterns.values()):
            if pattern["source"] == "configured" or pattern["validation_status"] != "validated":
                continue
            rule = pattern["conditions"]
            expected = rule["types"]
            evidence = history[-len(expected):]
            if len(evidence) != len(expected) or min(e["start"] for e in evidence) < now-rule["window_s"]:
                continue
            if set(pattern["last_event_ids"]).intersection(e["id"] for e in evidence):
                continue
            actual = [e["type"] for e in evidence]
            mode = rule["mode"]
            if mode == "combination":
                actual.sort()
            if actual != expected:
                continue
            if mode == "increasing duration" and not all(a["duration"] < b["duration"] for a,b in zip(evidence,evidence[1:])):
                continue
            if mode == "increasing frequency":
                a,b,c = [e["start"] for e in evidence]
                if not b-a > c-b > 0:
                    continue
            self.occurrence(pattern["name"], evidence, pattern["severity"], pattern.get("guidance", pattern["description"]),
                            now, pattern["source"], rule, pattern["id"])

    def mine(self, window, now):
        if not window:
            return
        last = window[-1]
        def candidate(mode, evidence):
            kinds = [e["type"] for e in evidence]
            if mode == "combination":
                kinds.sort()
            conditions = dict(mode=mode, types=kinds, window_s=self.cfg.window)
            key = hashlib.sha256(json.dumps(conditions, sort_keys=True).encode()).hexdigest()[:20]
            if key in self.patterns:
                return
            bucket = self.candidates[key]
            if bucket and evidence[0]["session_id"] == bucket[-1][0]["session_id"] and any(e["start"] <= bucket[-1][-1]["end"] for e in evidence):
                return
            bucket.append(evidence)
            if len(bucket) < max(2, self.cfg.support):
                return
            name = mode.capitalize() + " · " + " / ".join(kinds)
            groups = list(bucket)
            for group in groups:
                severe = "MICROSLEEP" in kinds and "PHONE_DETECTED" in kinds
                severity = "WARNING" if severe else "NORMAL"
                guidance = "Put the phone away and stop when safe." if severe else "Observed in separate episodes."
                self.occurrence(name, group, severity, guidance, now,
                                "learned", conditions, key, historical=True)
            if not self.seeding:
                self.publish({"type":"discovery", "name":name, "pattern_id":key, "message":guidance,
                              "severity":severity, "id":uid(), "timestamp":utc()})
        candidate("recurrence", [last])
        if len(window) >= 2:
            pair = window[-2:]
            candidate("sequence", pair)
            if pair[0]["type"] != pair[1]["type"]:
                candidate("transition", pair)
                candidate("combination", pair)
        if len(window) >= 3:
            a,b,c = window[-3:]
            if a["type"] == b["type"] == c["type"]:
                if a["duration"] < b["duration"] < c["duration"]:
                    candidate("increasing duration", [a,b,c])
                if b["start"]-a["start"] > c["start"]-b["start"] > 0:
                    candidate("increasing frequency", [a,b,c])

    def occurrence(self, name, evidence, severity, message, now, source, conditions, key=None, historical=False):
        key = key or name
        ids = [e["id"] for e in evidence]
        if source == "configured" and now-self.last_match.get(key, -1e12) < self.cfg.cooldown:
            return
        if source == "configured":
            self.last_match[key] = now
        stamp = utc()
        old = self.patterns.get(key, {})
        description = (f"{len(ids)} observed episode{'s' if len(ids) != 1 else ''} in a "
                       f"{conditions['window_s']:g}-second window.")
        pattern = dict(id=key, name=name, description=description, guidance=message, conditions=conditions, severity=severity,
                       source=source, validation_status="validated", first_detection=old.get("first_detection",stamp),
                       timestamp=stamp, last_detection=stamp, last_event_ids=ids,
                       minimum_support=self.cfg.support if source!="configured" else 1,
                       support=old.get("support",0)+1, occurrence_count=old.get("occurrence_count",0)+1)
        self.patterns[key] = pattern
        occurrence = self.record(pattern_id=key, name=name, supporting_event_ids=ids, severity=severity)
        if evidence[-1]["session_id"] == self.session["id"]:
            self.occurrences.append(occurrence)
            self.counts["patterns"] += 1
        occurrence["session_id"] = evidence[-1]["session_id"]
        if source != "configured":
            occurrence["timestamp"] = evidence[-1].get("resolved_at", evidence[-1]["timestamp"])
        self.batch.extend([("patterns", pattern), ("pattern_occurrences", occurrence)])
        if severity != "NORMAL" and evidence[-1]["session_id"] == self.session["id"] and not historical:
            self.pattern_alerts.append((severity, message, pattern, source))

    def close(self):
        with self.lock:
            self.batch = []
            self.resolve()
            for rec in self.open.values():
                if not rec.get("qualified"):
                    continue
                self.batch.append(("events", {**rec, "status":"interrupted"}))
            self.session.update(status="closed", end_time=utc())
            self.history.pending.join()
            self.history.submit(self.batch + [("sessions", self.session)])
        self.history.close()
