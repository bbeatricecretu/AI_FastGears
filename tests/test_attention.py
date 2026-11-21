"""AttentionMonitor / phone_in_use / head-pose tests. No camera, no models."""
import numpy as np
import pytest

from fastgears.vision.attention import AttentionMonitor
from fastgears.vision.face import FaceResult, _euler_from_matrix
from fastgears.vision.objects import ObjectDetector

FPS = 10
DT = 1.0 / FPS


def fake_face(ear=0.30, pitch=0.0, yaw=0.0, roll=0.0, mar=0.05,
              blendshapes=None, iris=(0.0, 0.0), bbox=(500, 200, 700, 450)):
    return FaceResult(
        landmarks_px=np.zeros((478, 3)),
        bbox=bbox,
        blendshapes=dict(blendshapes or {}),
        pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll,
        eyes=(600.0, 300.0, 65.0),
        ear_left=ear, ear_right=ear,
        mar=mar,
        iris_offset=iris,
    )


def run(mon, t0, seconds, face_fn, phone=False):
    """Feed frames at FPS for `seconds`; returns (last_out, t_end)."""
    n = int(round(seconds * FPS))
    out = None
    for i in range(n):
        t = t0 + i * DT
        out = mon.update(face_fn() if callable(face_fn) else face_fn, t, phone)
    return out, t0 + n * DT


def calibrated(**kw):
    mon = AttentionMonitor(calib_s=5.0, **kw)
    out, t = run(mon, 0.0, 5.5, lambda: fake_face())
    return mon, out, t


def test_calibration():
    mon = AttentionMonitor(calib_s=5.0)
    out, t = run(mon, 0.0, 2.0, lambda: fake_face(ear=0.28, yaw=5.0))
    assert out["calibrated"] is False
    out, t = run(mon, t, 3.5, lambda: fake_face(ear=0.28, yaw=5.0))
    assert out["calibrated"] is True
    assert mon.base_ear == pytest.approx(0.28)
    assert out["yaw"] == pytest.approx(0.0)  # relative to the neutral pose
    assert out["level"] == "ok" and out["looking"] == "road"
    mon.reset_calibration()
    assert mon.calibrated is False


def test_unknown_threshold_rejected():
    with pytest.raises(TypeError):
        AttentionMonitor(not_a_threshold=1)


def test_eyes_closed_microsleep_danger():
    mon, out, t = calibrated()
    assert out["eyes_closed"] is False and out["level"] == "ok"
    out, t2 = run(mon, t, 1.0, lambda: fake_face(ear=0.08))
    assert out["eyes_closed"] is True
    assert out["microsleep"] is False
    out, t3 = run(mon, t2, 1.0, lambda: fake_face(ear=0.08))
    assert out["eye_closed_s"] >= 1.5
    assert out["microsleep"] is True
    assert out["level"] == "danger"
    assert any(r.startswith("microsleep: eyes closed") for r in out["reasons"])
    assert out["perclos"] > 0
    assert out["drowsiness"] >= 70
    # eyes reopen: closure ends, not counted as a blink
    out, _ = run(mon, t3, 0.5, lambda: fake_face())
    assert out["eyes_closed"] is False and out["microsleep"] is False
    assert out["blinks_per_min"] == 0


def test_blendshape_blink_counts_as_closed():
    mon, out, t = calibrated()
    out = mon.update(fake_face(ear=0.28, blendshapes={"eyeBlinkLeft": 0.7,
                                                      "eyeBlinkRight": 0.6}), t)
    assert out["eyes_closed"] is True


def test_blink_counting():
    mon, out, t = calibrated()
    for _ in range(5):
        out, t = run(mon, t, 0.2, lambda: fake_face(ear=0.09))   # blink
        out, t = run(mon, t, 2.8, lambda: fake_face())
    elapsed = (t - DT) - 0.0
    assert out["blinks_per_min"] == pytest.approx(5 * 60 / elapsed, rel=0.02)
    assert out["level"] == "ok" and out["microsleep"] is False


@pytest.mark.parametrize("yaw,mirrored,label", [
    (35.0, False, "left"), (-35.0, False, "right"),
    (35.0, True, "right"), (-35.0, True, "left"),
])
def test_head_yaw_left_right_distracted(yaw, mirrored, label):
    mon, out, t = calibrated(mirrored=mirrored)
    out, t1 = run(mon, t, 1.5, lambda: fake_face(yaw=yaw))
    assert out["looking"] == label
    assert out["distracted"] is False
    out, _ = run(mon, t1, 1.5, lambda: fake_face(yaw=yaw))
    assert out["looking"] == label
    assert out["off_road_s"] >= 2.0
    assert out["distracted"] is True
    assert out["level"] == "danger"
    assert any(r.startswith(f"looking {label}") for r in out["reasons"])


def test_head_back_to_road_resets():
    mon, out, t = calibrated()
    out, t = run(mon, t, 3.0, lambda: fake_face(yaw=40.0))
    assert out["distracted"]
    out, _ = run(mon, t, 0.2, lambda: fake_face())
    assert out["looking"] == "road" and out["off_road_s"] == 0
    assert out["distracted"] is False


def test_iris_gaze_moves_label():
    mon, out, t = calibrated(mirrored=False)
    out = mon.update(fake_face(yaw=10.0, iris=(0.8, 0.0)), t)  # 10 + 25*0.8 = 30
    assert out["looking"] == "left"


def test_pitch_down():
    mon, out, t = calibrated()
    out, _ = run(mon, t, 2.5, lambda: fake_face(pitch=30.0))
    assert out["looking"] == "down"
    assert out["distracted"] is True
    assert any(r.startswith("looking down") for r in out["reasons"])
    out = mon.update(fake_face(pitch=-30.0), 100.0)
    assert out["looking"] == "up"


def test_phone_reason():
    mon, out, t = calibrated()
    out = mon.update(fake_face(), t, phone=True)
    assert out["phone"] is True
    assert "phone in hand" in out["reasons"]
    assert out["distraction"] >= 70 and out["level"] == "danger"


def test_yawn():
    mon, out, t = calibrated()
    out, t = run(mon, t, 0.5, lambda: fake_face(mar=0.8))
    assert out["yawning"] is False
    out, t = run(mon, t, 1.0, lambda: fake_face(mar=0.8))
    assert out["yawning"] is True and out["yawns_5min"] == 0
    assert out["level"] == "ok"
    out, _ = run(mon, t, 0.5, lambda: fake_face(blendshapes={"jawOpen": 0.1}))
    assert out["yawning"] is False and out["yawns_5min"] == 1


def test_face_missing_is_unavailable_without_distraction():
    mon, _, t = calibrated()
    out, t = run(mon, t, 2.3, lambda: fake_face(yaw=35.0))
    event_id = out["active_warning"]["event_id"]
    assert out["distracted"] is True

    out = mon.update(None, t)
    assert out["looking"] == "unavailable"
    assert out["off_road_s"] is None
    assert out["event_transitions"][0]["id"] == event_id
    assert out["event_transitions"][0]["status"] == "interrupted"
    out = mon.update(None, t + 3)
    assert out["distraction"] == 0
    assert out["level"] == "ok"
    assert out["inattention"]["unavailable_reason"] == "face_not_visible"
    out = mon.update(fake_face(), t + 3.1)
    assert out["looking"] == "road" and out["distracted"] is False


def test_repeated_short_glances_ignore_quick_mirror_checks():
    mon, _, t = calibrated()
    for _ in range(3):
        out, t = run(mon, t, 0.3, lambda: fake_face(yaw=35.0))
        out, t = run(mon, t, 0.4, fake_face)
    assert out["inattention"]["short_glances"] == 0

    for _ in range(3):
        out, t = run(mon, t, 0.8, lambda: fake_face(yaw=35.0))
        out, t = run(mon, t, 0.3, fake_face)
        out, t = run(mon, t, 0.1, lambda: fake_face(ear=0.08))
        out, t = run(mon, t, 0.2, fake_face)
    assert out["looking"] == "road"
    assert out["inattention"]["short_glances"] == 3
    assert out["level"] == "caution"
    assert out["active_warning"]["reason"] == "repeated_off_road"


def test_tracking_gap_restarts_the_off_road_timer():
    mon, _, t = calibrated()
    out, t = run(mon, t, 1.5, lambda: fake_face(yaw=35.0))
    out = mon.update(fake_face(yaw=35.0), t + 2)
    assert out["looking"] == "right"
    assert out["off_road_s"] == 0
    assert out["active_warning"] is None


def test_output_json_serialisable():
    import json
    mon, out, _ = calibrated()
    json.dumps(out)
    json.dumps(mon.update(None, 10.0))


# --- objects -------------------------------------------------------------------
FRAME = (720, 1280, 3)


def test_phone_in_use():
    phone_low = {"label": "cell phone", "conf": 0.6, "bbox": [600, 500, 680, 640]}
    phone_top_far = {"label": "cell phone", "conf": 0.6, "bbox": [10, 10, 60, 90]}
    phone_at_ear = {"label": "cell phone", "conf": 0.6, "bbox": [700, 150, 760, 230]}
    person = {"label": "person", "conf": 0.9, "bbox": [400, 100, 900, 720]}
    face = fake_face(bbox=(500, 120, 680, 340))
    pu = ObjectDetector.phone_in_use
    assert pu([], face, FRAME) is False
    assert pu([person], face, FRAME) is False
    assert pu([phone_low], None, FRAME) is True
    assert pu([phone_top_far], None, FRAME) is True
    assert pu([phone_low], face, FRAME) is True
    assert pu([phone_top_far], face, FRAME) is False
    assert pu([phone_at_ear], face, FRAME) is True
    assert ObjectDetector.count([person, phone_low, person]) == 2


def test_object_detector_construct_is_lazy():
    det = ObjectDetector()
    assert det._model is None


# --- landmark geometry ------------------------------------------------------------
def test_landmark_geometry():
    from fastgears.vision.face import face_result_from_landmarks
    lm = np.full((478, 3), 100.0)

    def eye(idx, x0, lids, iris):
        p1, p2, p3, p4, p5, p6 = idx
        lm[p1, :2] = (x0, 100)
        lm[p4, :2] = (x0 + 10, 100)
        lm[p2, :2] = (x0 + 3, 98.5)
        lm[p6, :2] = (x0 + 3, 101.5)
        lm[p3, :2] = (x0 + 7, 98.5)
        lm[p5, :2] = (x0 + 7, 101.5)
        lm[lids[0], :2] = (x0 + 5, 98.0)
        lm[lids[1], :2] = (x0 + 5, 102.0)
        lm[iris, :2] = (x0 + 5 + 2.5, 100)   # half-way to the outer corner

    eye((33, 160, 158, 133, 153, 144), 80, (159, 145), 468)
    eye((362, 385, 387, 263, 373, 380), 110, (386, 374), 473)
    lm[78, :2], lm[308, :2] = (90, 140), (110, 140)
    lm[13, :2], lm[14, :2] = (100, 135), (100, 145)
    f = face_result_from_landmarks(lm, (480, 640, 3))
    assert f.ear_left == pytest.approx(0.3) and f.ear_right == pytest.approx(0.3)
    assert f.mar == pytest.approx(0.5)
    assert f.iris_offset == pytest.approx((0.5, 0.0))
    assert f.eyes == pytest.approx((100.0, 100.0, 30.0))
    assert f.bbox[0] <= 80 and f.bbox[2] >= 120


# --- head pose -------------------------------------------------------------------
def test_euler_identity():
    p, y, r = _euler_from_matrix(np.eye(4))
    assert p == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert r == pytest.approx(0.0, abs=1e-9)
    assert _euler_from_matrix(np.eye(3)) == (None, None, None)


def test_euler_yaw_rotation():
    a = np.radians(30)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    p, y, r = _euler_from_matrix(m)
    assert y == pytest.approx(30.0) and abs(p) < 1e-6 and abs(r) < 1e-6