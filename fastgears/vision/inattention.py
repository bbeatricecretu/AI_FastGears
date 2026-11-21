"""Track observed glances independently of drowsiness and phone use."""
from collections import deque
from uuid import uuid4


INATTENTION_THRESHOLDS = {
    "short_glance_min_s": 0.6,
    "repeated_glance_window_s": 30.0,
    "repeated_glance_count": 3,
}


class InattentionMonitor:
    def __init__(self, thresholds):
        self.th = thresholds
        self.reset()

    def reset(self):
        self.last_t = None
        self.started = None
        self.direction = None
        self.short_glances = deque()
        self.events = {}

    def update(self, direction, t, unavailable_reason=None):
        th = self.th
        transitions = []

        def finish(kind, when, status):
            event = self.events.pop(kind, None)
            if event is not None:
                transitions.append({
                    **event,
                    "status": status,
                    "duration_s": round(when - event["start_s"], 4),
                })

        def observe(kind, start, severity, **evidence):
            event = self.events.get(kind)
            if event is None:
                event = {
                    "id": uuid4().hex, "type": kind, "start_s": start,
                    "status": "active", "severity": severity,
                }
            first_frame = kind not in self.events
            event.update(duration_s=round(t - start, 4), evidence=evidence)
            self.events[kind] = event
            if first_frame:
                transitions.append(dict(event))
            return event

        gap = self.last_t is not None and t - self.last_t > th["tracking_gap_s"]
        tracking_lost = gap or (direction is None and unavailable_reason != "eyes_closed")
        if self.last_t is not None and (tracking_lost or direction is None):
            finish("sustained_off_road", self.last_t, "interrupted")
            self.started = self.direction = None
            if tracking_lost:
                finish("repeated_off_road", self.last_t, "interrupted")
                self.short_glances.clear()
        self.last_t = t

        cutoff = t - th["repeated_glance_window_s"]
        while self.short_glances and self.short_glances[0] <= cutoff:
            self.short_glances.popleft()

        duration = None if direction is None else 0.0
        sustained = False
        if direction == "road":
            if self.started is not None:
                elapsed = t - self.started
                if th["short_glance_min_s"] <= elapsed <= 3.0:
                    self.short_glances.append(t)
                finish("sustained_off_road", t, "ended")
                self.started = self.direction = None
        elif direction is not None:
            if self.started is None:
                self.started = t
            self.direction = direction
            duration = t - self.started
            sustained = duration + 1e-9 >= th["distracted_s"]

        count = len(self.short_glances)
        repeated = count >= th["repeated_glance_count"]
        repeated_event = self.events.get("repeated_off_road")
        if repeated and direction is not None:
            start = repeated_event["start_s"] if repeated_event else t
            repeated_event = observe("repeated_off_road", start, "MEDIUM", count=count)
        elif not repeated:
            finish("repeated_off_road", t, "ended")

        warning = None
        score = 0.0
        if sustained:
            event = observe("sustained_off_road", self.started, "HIGH", direction=self.direction)
            warning = dict(event_id=event["id"], reason="sustained_off_road",
                           severity="alert", voice=True, message="Eyes back on the road.")
            score = min(100.0, th["level_danger"] * duration / th["distracted_s"])
            score = max(score, th["level_danger"])
        elif repeated and direction is not None:
            warning = dict(event_id=repeated_event["id"], reason="repeated_off_road",
                           severity="warning", voice=False,
                           message="Repeated glances away from the road. Keep your eyes on the road.")
            score = th["level_caution"]

        return {
            "looking": direction or "unavailable",
            "off_road_s": duration,
            "distracted": sustained,
            "score": score,
            "short_glances": count,
            "repeated": repeated and direction is not None,
            "unavailable_reason": unavailable_reason,
            "warning": warning,
            "events": [dict(event) for event in self.events.values()],
            "event_transitions": transitions,
        }