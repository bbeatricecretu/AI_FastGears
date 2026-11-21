"""Regression tests for timestamp-driven drowsiness and the warning path."""
from dataclasses import replace

import numpy as np
import pytest

from fastgears.config import Settings
from fastgears.llm.coach import Coach, compact_state
from fastgears.pipeline import Pipeline
from fastgears.vision.attention import AttentionMonitor
from test_attention import fake_face


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_bilateral_closure_and_recovery(fps):
    mon = AttentionMonitor()
    mon.update(fake_face(), 0)
    wink = replace(fake_face(), ear_left=0.08)
    assert mon.update(wink, 1 / fps)["eyes"] == (True, False)

    started = []
    for i in range(2, int(0.7 * fps) + 3):
        out = mon.update(fake_face(ear=0.08), i / fps)
        started.extend(out["event_transitions"])
    assert len(started) == 1
    event = started[0]
    assert event["duration_s"] == pytest.approx(0.6, abs=1 / fps)
    assert out["drowsiness_level"] == "HIGH DROWSINESS"
    assert not out["calibrated"]
    assert not out["perclos_ready"]

    out = mon.update(fake_face(), (i + 1) / fps)
    assert out["drowsiness_level"] == "NORMAL"
    assert out["event_transitions"][0]["id"] == event["id"]
    assert out["event_transitions"][0]["status"] == "ended"


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_blinks_and_perclos(fps):
    mon = AttentionMonitor()
    for i in range(61 * fps):
        phase = (i % fps) / fps
        out = mon.update(fake_face(ear=0.08 if phase < 0.4 else 0.30), i / fps)
        if i < 60 * fps:
            assert not out["perclos_ready"]
    assert out["perclos_ready"]
    assert out["perclos_coverage"] == pytest.approx(1)
    assert out["perclos"] == pytest.approx(0.4, abs=1 / fps)
    assert out["average_blink_s"] == pytest.approx(0.4, abs=1 / fps)
    assert out["prolonged_blink_count"] >= 59
    assert out["drowsiness_level"] == "POSSIBLE DROWSINESS"


def test_short_blink_excluded_and_gap_interrupts():
    mon = AttentionMonitor()
    mon.update(fake_face(), 0)
    mon.update(fake_face(ear=0.08), 0.05)
    out = mon.update(fake_face(), 0.15)
    assert out["blink_duration_s"] == pytest.approx(0.10)
    assert out["perclos"] == 0
    for i in range(4, 22):
        out = mon.update(fake_face(ear=0.08), i / 20)
    active = out["events"][0]
    out = mon.update(None, 1.1)
    assert out["event_transitions"][0]["id"] == active["id"]
    assert out["event_transitions"][0]["status"] == "interrupted"
    assert out["drowsiness_level"] is None and out["perclos"] is None
    out = mon.update(fake_face(ear=0.08), 2)
    assert out["eye_closed_s"] == 0
    out = mon.update(fake_face(ear=0.08), 3)
    assert out["eye_closed_s"] == 0


def test_partial_eyelids_and_recalibration():
    mon = AttentionMonitor()
    for i in range(130):
        out = mon.update(fake_face(ear=0.30 if i < 105 else 0.21), i / 20)
    assert out["drooping"]
    assert out["drowsiness_level"] == "POSSIBLE DROWSINESS"
    mon.reset_calibration()
    for i in range(120):
        out = mon.update(fake_face(ear=0.22), 7 + i / 20)
    assert out["eyelids_calibrated"]
    assert not out["drooping"]
    assert out["drowsiness_level"] == "NORMAL"


def test_yawn_and_short_mouth_movements():
    mon = AttentionMonitor()
    for i in range(100):
        # Repeated brief movements, then one sustained opening with a peak and return.
        mar = (0.8 if i % 8 < 3 else 0.05) if i < 60 else 0.05
        if 65 <= i < 95:
            mar = 0.65 + 0.15 * (1 - abs(i - 80) / 15)
        out = mon.update(fake_face(mar=mar), i / 20)
        if i == 94:
            assert out["yawn_count"] == 0
    assert out["yawn_count"] == 1
    assert out["active_warning"] is None


def test_nod_vs_sustained_downward_glance():
    mon = AttentionMonitor(calib_s=0)
    for i in range(60):
        pitch = 25 if 10 <= i < 50 else 0
        out = mon.update(fake_face(pitch=pitch), i / 20)
    assert out["nod_count"] == 0
    for i in range(60, 69):
        out = mon.update(fake_face(pitch=25 if i < 68 else 0), i / 20)
    assert out["nod_count"] == 1
    assert out["drowsiness_level"] == "HIGH DROWSINESS"
    assert out["active_warning"]["reason"] == "nod"


def test_pipeline_warning_without_llm(monkeypatch):
    messages = []
    coach = Coach(None, messages.append)
    monkeypatch.setattr(Pipeline, "_load_detectors", lambda p: setattr(
        p, "attention", AttentionMonitor()))
    pipeline = Pipeline(Settings(), on_state=coach.observe)
    pipeline.face = type("Tracker", (), {
        "process": lambda self, frame, t_ms: fake_face(ear=0.08),
    })()
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    for i in range(20):
        pipeline._step(frame, i / 20, i * 50)
    assert len(messages) == 1
    assert messages[0]["severity"] == "alert" and messages[0]["voice"]
    assert pipeline.jpeg() is not None
    state = pipeline.state()
    assert state["attention"]["active_warning"]["event_id"] == messages[0]["event_id"]
    context = compact_state(state)
    assert context["events"][0]["type"] == "prolonged_eye_closure"
    assert "face_box" not in context


def test_warning_precedes_llm_and_llm_cannot_change_it():
    from threading import Event
    import json

    called = Event()

    class Client:
        name = "test"

        def complete(self, system, user, json_mode=False):
            self.context = json.loads(user.split("\n", 1)[1])
            called.set()
            return '{"message":"Keep driving","severity":"info","reason":"test"}'

    messages = []
    client = Client()
    coach = Coach(client, messages.append, min_interval_s=0)
    mon = AttentionMonitor()
    for i in range(15):
        out = mon.update(fake_face(ear=0.08), i / 20)
    coach.observe({"attention": out}, now=10)
    assert messages[0]["severity"] == "alert"
    assert called.wait(timeout=2)
    assert client.context["drowsiness_level"] == "HIGH DROWSINESS"
    assert client.context["events"][0]["confidence"] is None
    assert len(messages) == 1


def test_perclos_window_requires_enough_observed_time():
    mon = AttentionMonitor()
    for i in range(601):
        t = i / 10
        out = mon.update(fake_face() if t < 30 or t >= 59 else None, t)
    assert out["perclos_coverage"] == pytest.approx(30.9 / 60)
    assert not out["perclos_ready"]


def test_blink_speed_requires_valid_open_and_closed_samples():
    mon = AttentionMonitor()
    for i, ear in enumerate([0.30, 0.12, 0.07, 0.30]):
        out = mon.update(fake_face(ear=ear), i / 30)
    assert out["closing_speed"] == pytest.approx((0.30 - 0.07) / (2 / 30))
    assert out["opening_speed"] == pytest.approx((0.30 - 0.07) / (1 / 30))  