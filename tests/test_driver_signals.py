import pytest

from fastgears.vision.driver_signals import assess_driver


def readings(t=20):
    attention = {"face_present": True, "eye_closed_s": 0.8, "perclos": 0.03}
    gestures = {"observable": True, "event": {"state": "active"}}
    rgb = {"enabled": True, "lighting": "stable", "sampled_monotonic_s": t,
           "observations": {"cheek_color_reduction": {"state": "active"}}}
    pulse = {"bpm": 48, "trust": {"label": "reliable"}, "sampled_monotonic_s": t - 1}
    return attention, gestures, rgb, pulse


def test_report_presyncope_pattern_with_concurrent_signals():
    result = assess_driver(20, *readings())
    assert result["finding"] == "possible_presyncope"
    assert result["stop_safely"]
    assert (result["bpm"], result["color"], result["expression"]) == (48, "pallor", "grimace")


@pytest.mark.parametrize("change", ["old_pulse", "uncertain_pulse", "typical_pulse"])
def test_eye_closure_still_prompts_safe_stop_without_trusted_slow_pulse(change):
    attention, gestures, rgb, pulse = readings()
    updates = {"old_pulse": {"sampled_monotonic_s": 17},
               "uncertain_pulse": {"trust": {"label": "uncertain"}},
               "typical_pulse": {"bpm": 72}}
    pulse.update(updates[change])
    result = assess_driver(20, attention, gestures, rgb, pulse)
    assert result["finding"] == "pallor_and_eye_closure"
    assert result["stop_safely"]
    assert result["bpm"] == (72 if change == "typical_pulse" else None)


@pytest.mark.parametrize("rgb_update", [{"lighting": "lighting_changed"}, {"sampled_monotonic_s": 19.5}])
def test_unusable_color_does_not_corroborate_presyncope(rgb_update):
    attention, gestures, rgb, pulse = readings()
    rgb.update(rgb_update)
    result = assess_driver(20, attention, gestures, rgb, pulse)
    assert result["finding"] == "drowsiness"
    assert result["color"] == "unavailable"


def test_grimace_and_color_can_coincide_without_drowsiness():
    attention, gestures, rgb, pulse = readings()
    attention["eye_closed_s"] = 0
    result = assess_driver(20, attention, gestures, rgb, pulse)
    assert result["finding"] == "combined_discomfort_signals"
    assert not result["stop_safely"]


def test_face_loss_clears_previous_pattern():
    attention, gestures, rgb, pulse = readings()
    attention["face_present"] = False
    assert assess_driver(20, attention, gestures, rgb, pulse) == {"finding": "face_unavailable"}


def test_camera_loss_clears_stale_attention_and_combined_finding(monkeypatch):
    from fastgears.config import Settings
    from fastgears.pipeline import Pipeline

    monkeypatch.setattr("fastgears.vision.face.FaceTracker", lambda **kwargs: None)
    pipeline = Pipeline(Settings(enable_rppg=False, enable_objects=False, enable_rgb=False, enable_hands=False))
    pipeline._state["attention"] = {"face_present": True, "microsleep": True}
    pipeline._state["driver_signals"] = {"finding": "possible_presyncope", "stop_safely": True}
    pipeline._publish_no_camera()

    state = pipeline.state()
    assert not state["attention"]["face_present"]
    assert not state["attention"]["microsleep"]
    assert state["driver_signals"] == {"finding": "camera_unavailable"}
