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
    out, t = run(mon, t, 1.5, lambda: fake_face(mar=0.8))
    assert out["yawning"] is True and out["yawns_5min"] == 1
    assert out["level"] in ("caution", "danger")
    out, _ = run(mon, t, 0.5, lambda: fake_face(blendshapes={"jawOpen": 0.1}))
    assert out["yawning"] is False and out["yawns_5min"] == 1


def test_face_missing_goes_away_after_1s():
    mon, out, t = calibrated()
    out = mon.update(None, t + 0.5)
    assert out["face_present"] is False
    assert out["looking"] == "road" and out["off_road_s"] == 0
    out = mon.update(None, t + 1.2)
    assert out["looking"] == "road"          # missing only 0.7 s
    out = mon.update(None, t + 1.6)
    assert out["looking"] == "away"
    assert out["off_road_s"] == pytest.approx(1.1)  # counted from first missing frame
    assert out["distracted"] is False
    out = mon.update(None, t + 2.6)
    assert out["distracted"] is True
    assert any(r.startswith("face not visible") for r in out["reasons"])
    out = mon.update(fake_face(), t + 2.7)
    assert out["looking"] == "road" and out["distracted"] is False


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


def test_drowsiness_recovers_after_eyes_reopen():
    mon, out, t = calibrated()
    out, t = run(mon, t, 6.0, lambda: fake_face(ear=0.08))      # head drop, eyes shut 6 s
    assert out["level"] == "danger" and out["microsleep"]
    out, t = run(mon, t, 2.0, lambda: fake_face())              # eyes open again
    assert out["level"] != "danger" and out["drowsiness"] < 70
    assert out["perclos"] > 0.05                                # trend still remembers it...
    out, t = run(mon, t, 62.0, lambda: fake_face())             # ...until it leaves the 60 s window
    assert out["level"] == "ok"


def test_talking_is_not_a_yawn():
    mon, out, t = calibrated()
    for _ in range(10):                                          # mouth open 0.6 s, closed 0.3 s
        out, t = run(mon, t, 0.6, lambda: fake_face(mar=0.8))
        out, t = run(mon, t, 0.3, lambda: fake_face())
    assert out["yawns_5min"] == 0


def test_phone_latch_debounces():
    from fastgears.vision.objects import PhoneLatch
    latch = PhoneLatch()
    assert latch.update(0.3, True, 0.0) is False                 # one weak hit: not yet
    assert latch.update(None, False, 0.1) is False
    assert latch.update(0.3, True, 0.5) is True                  # confirmed by a 2nd hit
    assert latch.update(None, True, 1.5) is True                 # missed detections: held
    assert latch.update(None, True, 2.1) is False                # > 1.5 s without a hit: off
    assert latch.update(0.7, True, 3.0) is True                  # one confident hit is enough


def test_tilted_head_mouth_is_not_a_yawn():
    mon, out, t = calibrated()
    out, t = run(mon, t, 3.0, lambda: fake_face(mar=0.8, roll=40.0))   # head on the side
    assert out["yawns_5min"] == 0


def test_yawn_score_fades():
    mon, out, t = calibrated()
    out, t = run(mon, t, 1.6, lambda: fake_face(mar=0.8))
    out, t = run(mon, t, 1.0, lambda: fake_face())
    fresh = out["drowsiness"]
    out, t = run(mon, t, 150.0, lambda: fake_face())
    assert out["yawns_5min"] == 1 and out["drowsiness"] < fresh


def test_hand_up_is_hand_off_wheel():
    mon, out, t = calibrated()
    out, t = run(mon, t, 2.0, lambda: fake_face(), )
    for _ in range(40):
        t += 0.05
        out = mon.update(fake_face(), t, hand_up=True)
    assert out["hand_off_wheel"] and "hand off the wheel" in out["reasons"] and out["level"] == "caution"


def test_head_lying_sideways_is_tilted_and_drowsy():
    mon, out, t = calibrated()
    out, t = run(mon, t, 0.5, lambda: fake_face(roll=60.0, pitch=-25.0))
    assert out["looking"] == "tilted"                       # not "up": pitch is unreliable when rolled
    out, t = run(mon, t, 3.0, lambda: fake_face(roll=60.0, pitch=-25.0))
    assert out["level"] == "danger" and any(r.startswith("head tilted") for r in out["reasons"])
    assert out["distraction"] < 40                           # it's drowsiness, not distraction
    out, t = run(mon, t, 1.0, lambda: fake_face())
    assert out["looking"] == "road" and out["level"] != "danger"


def test_drowsiness_recovers_within_20s_after_long_closure():
    mon, out, t = calibrated()
    out, t = run(mon, t, 9.0, lambda: fake_face(ear=0.08))      # 9 s eyes closed (screenshot)
    assert out["microsleep"]
    out, t = run(mon, t, 20.0, lambda: fake_face())
    assert out["level"] == "ok" and out["drowsiness"] < 40


def test_yawn_with_head_tipped_back_counts_and_is_not_looking_up():
    mon, out, t = calibrated()
    out, t = run(mon, t, 2.0, lambda: fake_face(mar=0.8, pitch=-29.0,
                                                 blendshapes={"jawOpen": 0.8}))
    assert out["yawning"] is True and out["yawns_5min"] == 1
    assert out["looking"] == "road" and out["distraction"] < 40


def test_turned_head_mouth_is_still_not_a_yawn():
    mon, out, t = calibrated()
    out, t = run(mon, t, 3.0, lambda: fake_face(mar=0.8, yaw=30.0))
    assert out["yawns_5min"] == 0


@pytest.mark.parametrize("pitch", [-40.0, 30.0])
def test_head_drop_with_droopy_eyes_is_drowsiness(pitch):
    mon, out, t = calibrated()
    out, t = run(mon, t, 0.3, lambda: fake_face(pitch=pitch, ear=0.20))
    assert out["looking"] in ("up", "down")              # not yet: could be a blink
    out, t = run(mon, t, 3.5, lambda: fake_face(pitch=pitch, ear=0.20))
    assert out["looking"] == "nodding"
    assert out["level"] == "danger" and out["drowsiness"] >= 70
    assert out["distraction"] < 40
    assert any(r.startswith("head dropping") for r in out["reasons"])
    out, t = run(mon, t, 1.0, lambda: fake_face())
    assert out["looking"] == "road" and out["level"] != "danger"


def test_looking_down_with_blinks_stays_distraction():
    mon, out, t = calibrated()
    for _ in range(3):
        out, t = run(mon, t, 0.9, lambda: fake_face(pitch=30.0))
        out, t = run(mon, t, 0.2, lambda: fake_face(pitch=30.0, ear=0.08))   # blink
    out, t = run(mon, t, 0.5, lambda: fake_face(pitch=30.0))
    assert out["looking"] == "down" and out["distracted"] is True
    assert out["off_road_s"] >= 3.0 and out["drowsiness"] < 40


def test_head_thrown_back_is_nodding_even_with_eyes_open():
    mon, out, t = calibrated()
    out, t = run(mon, t, 4.0, lambda: fake_face(pitch=-44.0, ear=0.27))
    assert out["looking"] == "nodding" and out["level"] == "danger"
    assert out["distraction"] < 40
    out, t = run(mon, t, 4.0, lambda: fake_face(pitch=-25.0, ear=0.30))   # glance up, eyes open
    assert out["looking"] == "up"


def test_chin_drop_losing_face_is_nodding_not_away():
    mon, out, t = calibrated()
    out, t = run(mon, t, 0.3, lambda: fake_face(pitch=30.0, ear=0.18))   # head going down, droopy
    out, t = run(mon, t, 4.0, lambda: None)                              # face out of view
    assert out["face_present"] is False
    assert out["looking"] == "nodding" and out["level"] == "danger"
    assert out["distraction"] < 40
    assert any(r.startswith("head dropping") for r in out["reasons"])


def test_looking_down_alert_then_face_lost_is_away():
    mon, out, t = calibrated()
    out, t = run(mon, t, 0.5, lambda: fake_face(pitch=30.0))             # eyes open
    out, t = run(mon, t, 3.0, lambda: None)
    assert out["looking"] == "away" and out["distracted"] is True


def test_fast_forward_drop_losing_face_is_nodding():
    # the head falls forward faster than the eyes/pitch can say "nodding" before the face is gone
    mon, out, t = calibrated()
    for p in (0.0, 6.0, 12.0, 18.0):                                     # pitch rising, eyes open
        out, t = run(mon, t, DT, lambda p=p: fake_face(pitch=p))
    out, t = run(mon, t, 4.0, lambda: None)
    assert out["looking"] == "nodding" and out["distraction"] < 40
    assert out["drowsiness"] >= 70


def test_face_lost_right_after_nodding_is_still_nodding():
    mon, out, t = calibrated()
    out, t = run(mon, t, 1.5, lambda: fake_face(pitch=-45.0))           # head thrown back: nodding
    assert out["looking"] == "nodding"
    out, t = run(mon, t, 0.5, lambda: fake_face(pitch=0.0))             # swings through neutral...
    out, t = run(mon, t, 3.0, lambda: None)                             # ...and drops out of view
    assert out["looking"] == "nodding"


def test_face_lost_long_after_nodding_is_away():
    mon, out, t = calibrated()
    out, t = run(mon, t, 1.5, lambda: fake_face(pitch=-45.0))
    out, t = run(mon, t, 5.0, lambda: fake_face(pitch=0.0))             # awake again for 5 s
    out, t = run(mon, t, 3.0, lambda: None)
    assert out["looking"] == "away"


def test_brief_face_dropout_is_not_away():
    mon, out, t = calibrated()
    out, t = run(mon, t, 0.7, lambda: None)                             # hand passes over the face
    assert out["looking"] == "road" and out["distraction"] == 0


def test_phone_latch_holds_longer_while_hand_still_holds_it():
    from fastgears.vision.objects import PhoneLatch
    latch = PhoneLatch()
    assert latch.update(0.7, True, 0.0) is True
    assert latch.update(None, True, 3.0, holding=True) is True          # YOLO lost it, hand still there
    assert latch.update(None, True, 3.5, holding=False) is False        # hand gone: normal 1.5 s hold


def test_driver_crop_reaches_arm_length_to_the_side():
    face = fake_face(bbox=(600, 200, 700, 320))                         # 100 px wide face
    x0, y0, x1, y1 = ObjectDetector.driver_crop(face, (720, 1280, 3))
    assert x0 <= 300 and x1 >= 1000 and y1 == 720
