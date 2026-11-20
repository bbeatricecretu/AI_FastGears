import numpy as np
import pytest

from fastgears.llm.coach import compact_state
from fastgears.vision.discomfort import MARKERS, DiscomfortMonitor
from fastgears.vision.face import FaceResult


SHAPE = (480, 640, 3)


def face(*active, missing=None, ears=(0.30, 0.30), mar=0.05, yaw=0, clipped=None):
    points = np.zeros((478, 3))
    points[:, 0] = 100 + np.arange(478) % 200
    points[:, 1] = 200
    if clipped:
        points[MARKERS[clipped].indices[0], 0] = -1
    blendshapes = {m.source + side: 0.55 if code in active else 0.04
                   for code, m in MARKERS.items() if code not in (missing, "au43", "au25_27")
                   for side in ("Left", "Right")}
    return FaceResult(points, (100, 100, 300, 300), blendshapes, 0, yaw, 0,
                      (200, 200, 75), *ears, mar)


def calibrated(fps=30):
    monitor = DiscomfortMonitor(calib_s=0.4)
    for i in range(1 + fps):
        state = monitor.update(face(), i / fps, SHAPE)
    assert state["calibrated"]
    return monitor


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_synchronized_grimace_has_one_id_and_ends(fps):
    monitor = calibrated(fps)
    events = []
    for i in range(fps + 1, 2 * fps):
        state = monitor.update(face("au4", "au7", "au10"), i / fps, SHAPE)
        if state["event"]:
            events.append(state["event"])
    assert events[0]["duration_ms"] == round(np.ceil(0.3 * fps) / fps * 1000)
    assert events[0]["evidence"] == ["au10", "au4", "au7"]
    assert {e["id"] for e in events} == {1}
    assert compact_state({"discomfort": state})["discomfort_grimace"]["evidence"] == events[0]["evidence"]
    final = monitor.update(face(), 2, SHAPE)
    assert final["event"] is None
    assert final["recent_events"] == [{**events[-1], "state": "ended"}]


def test_individual_markers_and_bilateral_closure_are_observations():
    monitor = calibrated()
    t = 1
    for code in ("au4", "au6", "au7", "au9", "au10"):
        t += 0.1
        state = monitor.update(face(code), t, SHAPE)
        assert state["markers"][code]["active"]
        assert not state["event"]
    t += 0.1
    state = monitor.update(face(ears=(0.02, 0.30), mar=0.5), t, SHAPE)
    assert state["markers"]["au25_27"]["active"]
    assert not state["markers"]["au43"]["active"]
    for i in range(1, 6):
        state = monitor.update(face("au4", "au9", ears=(0.02, 0.02)), t + i / 10, SHAPE)
    assert state["markers"]["au43"]["active"]
    assert not state["event"]


def test_sequential_signals_and_missing_required_region():
    monitor = calibrated()
    for i in range(11, 22):
        f = face("au4", "au7") if i < 16 else face("au9")
        assert monitor.update(f, i / 10, SHAPE)["event"] is None
    for i in range(22, 27):
        state = monitor.update(face("au4", "au7", "au9", clipped="au4"), i / 10, SHAPE)
        assert not state["observable"] and not state["event"]
        assert state["markers"]["au4"]["reason"] == "region_out_of_frame"


@pytest.mark.parametrize("outage", ["camera", "region", "gap", "pose"])
def test_outage_interrupts_confirmed_event(outage):
    monitor = calibrated()
    for i in range(11, 15):
        state = monitor.update(face("au4", "au7", "au9", missing="au10"), i / 10, SHAPE)
    assert state["event"]["id"] == 1
    assert state["markers"]["au10"]["reason"] == "missing_signal"
    if outage == "camera":
        state = monitor.update(None, 1.5, missing_reason="no_camera_frame")
    elif outage == "region":
        state = monitor.update(face("au4", "au7", clipped="au9"), 1.5, SHAPE)
    elif outage == "gap":
        state = monitor.update(face("au4", "au7", "au9"), 1.8, SHAPE)
    else:
        state = monitor.update(face("au4", "au7", "au9", yaw=40), 1.5, SHAPE)
    assert state["event"] is None
    assert state["recent_events"][-1]["state"] == "interrupted"
    assert state["recent_events"][-1]["duration_ms"] == 300


def test_calibration_skips_closed_eyes_and_resets_after_face_loss():
    monitor = DiscomfortMonitor(calib_s=0.3)
    for i in range(10):
        assert not monitor.update(face(ears=(0.05, 0.05)), i / 10, SHAPE)["calibrated"]
    for i in range(10, 15):
        state = monitor.update(face(), i / 10, SHAPE)
    assert state["calibrated"]
    assert not monitor.update(face(), 3.5, SHAPE)["calibrated"]


def test_repeated_clock_ticks_do_not_advance_or_crash_the_detector():
    monitor = calibrated()
    for _ in range(20):
        state = monitor.update(face("au4", "au7", "au9"), 1.1, SHAPE)
        assert state["pending_ms"] == 0 and state["event"] is None
    state = monitor.update(None, 1.1, missing_reason="no_camera_frame")
    assert state["markers"]["au4"]["reason"] == "no_camera_frame"
    assert not monitor.update(None, 1.1)["observable"]


@pytest.mark.parametrize("mirror", [False, True])
def test_nose_wing_lift_detected_with_flat_blendshape(mirror):
    def sample(lift=0, turn=0, tip=0):
        f = face(yaw=turn)
        f.blendshapes.update(noseSneerLeft=0, noseSneerRight=0)
        f.landmarks_px[98, 1] -= lift
        f.landmarks_px[1, 1] -= tip
        if mirror:
            f.landmarks_px[:, 0] = SHAPE[1] - f.landmarks_px[:, 0]
        return f

    monitor = DiscomfortMonitor(calib_s=0.4)
    for i in range(11):
        state = monitor.update(sample(), i / 10, SHAPE)
    assert not state["markers"]["au9"]["active"]
    assert not monitor.update(sample(tip=5), 1.1, SHAPE)["markers"]["au9"]["active"]
    active = monitor.update(sample(lift=2), 1.2, SHAPE)["markers"]["au9"]
    assert active["active"] and active["source"] == "nose_wing_lift"
    assert not monitor.update(sample(lift=2, turn=15), 1.3, SHAPE)["markers"]["au9"]["active"]


@pytest.mark.parametrize("mirror", [False, True])
def test_cheek_lift_works_with_flat_blendshapes_and_rejects_head_motion(mirror):
    def sample(lift=0, turn=0, shift=0):
        f = face(yaw=turn)
        f.blendshapes.update(cheekSquintLeft=0, cheekSquintRight=0)
        f.landmarks_px[:, 1] += shift
        f.landmarks_px[[50, 101], 1] -= lift
        if mirror:
            f.landmarks_px[:, 0] = SHAPE[1] - f.landmarks_px[:, 0]
        return f

    monitor = DiscomfortMonitor(calib_s=0.4)
    for i in range(11):
        monitor.update(sample(), i / 10, SHAPE)
    assert not monitor.update(sample(shift=12), 1.1, SHAPE)["markers"]["au6"]["active"]
    lifted = monitor.update(sample(lift=2), 1.2, SHAPE)["markers"]["au6"]
    assert lifted["active"] and lifted["source"] == "cheek_lift"
    assert lifted["unit"] == "eye_distance_fraction"
    assert not monitor.update(sample(lift=2, turn=15), 1.3, SHAPE)["markers"]["au6"]["active"]
