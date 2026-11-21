import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from fastgears.config import Settings
from fastgears.pipeline import Pipeline
from fastgears.vision.driver_assistance import DriverAssistance
from fastgears.vision.hands import Hand
from fastgears.vision.objects import ObjectDetector, PhoneLatch
from test_attention import fake_face


def test_driver_only_uses_biometrics_but_reports_passengers_and_phone(monkeypatch):
    driver = fake_face(bbox=(350, 150, 550, 370))
    passenger = fake_face(bbox=(850, 160, 1010, 320))
    tracker = Mock()
    tracker.process_all.return_value = [passenger, driver]
    tracker.process_scene.return_value = [passenger, driver]
    monkeypatch.setattr("fastgears.vision.face.FaceTracker", lambda **kwargs: tracker)
    settings = Settings(driver_seat="left", enable_hands=False, enable_rgb=False, enable_rppg=False)
    pipeline = Pipeline(settings)
    pipeline._draw = Mock(return_value=b"preview")
    pipeline.objects.detect_live = lambda frame, face, boxes, t: pipeline.objects.detect(frame, face, boxes)
    pipeline.objects._predict = lambda *args, **kwargs: [
        {"label": "cell phone", "conf": 0.8, "bbox": [452, 385, 483, 466]},
        {"label": "cell phone", "conf": 0.9, "bbox": [938, 375, 974, 455]},
    ]
    frame = np.zeros((480, 1280, 3), dtype=np.uint8)
    for index in range(8):
        pipeline._step(frame, pipeline._t0 + index / 30, index + 1)
    state = pipeline.state()
    assert state["driver"]["status"] == "locked" and state["driver"]["passengers"] == 1
    assert state["face_box"] == list(driver.bbox) and state["attention"]["face_present"]
    assert len(state["objects"]) == 1 and state["driver_alerts"]["phone"]["active"]

    tracker.process_all.return_value = [passenger]
    tracker.process_roi.return_value = None
    pipeline._step(frame, pipeline._t0 + 0.3, 10)
    state = pipeline.state()
    assert state["driver"]["status"] == "lost" and state["face_box"] is None
    assert state["driver_alerts"]["phone"]["active"] is None


def test_hands_and_repeated_glances_have_independent_alerts():
    settings = Settings(enable_hands=False, enable_objects=False, calib_s=0.4)
    assistance = DriverAssistance(settings, {})
    assistance.hands = Mock()
    face = fake_face()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assistance.hands.process.return_value = [Hand((478, 370, 528, 427), 0.9), Hand((652, 360, 707, 430), 0.9)]
    for i in range(52):
        off_road = 8 <= i < 15 or 19 <= i < 26 or 30 <= i < 37
        observed = fake_face(pitch=35) if off_road else face
        assistance.update_objects(None, frame, observed, i / 10, i)
        state = assistance.state(observed, {"calibrated": True, "pitch": observed.pitch_deg, "roll": 0.0},
                                 i / 10, False, False)
    assert state["driver_alerts"]["attention"]["active"]
    assert state["driver_attention"]["inattention"]["short_glances"] == 3
    assert state["driver_alerts"]["hands"]["active"]
    assert state["driver_alerts"]["phone"]["active"] is None


def test_phone_confirmation_ignores_cached_detections_and_holds_a_tracked_phone():
    latch = PhoneLatch()
    assert not latch.update(0.3, True, 0.0)
    assert not latch.update(0.3, False, 0.1)
    assert latch.update(0.3, True, 0.2)
    assert latch.update(None, True, 2.0, holding=True)
    assert not latch.update(None, True, 2.1)
    assert ObjectDetector.covers_face([350, 150, 550, 370], fake_face(bbox=(350, 150, 550, 370)))


def test_object_worker_drops_intermediate_frames_without_blocking(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                            get_num_threads=lambda: 2, set_num_threads=lambda n: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    detector = ObjectDetector(every_n=1)
    entered, release, finished = Event(), Event(), Event()
    frames = []

    def infer(frame, face, boxes, tick):
        frames.append(frame)
        if tick == 0:
            entered.set()
            assert release.wait(timeout=3)
        else:
            finished.set()
        return []

    monkeypatch.setattr(detector, "_infer", infer)
    try:
        detector.detect_live("first", None, [], 0.0)
        assert entered.wait(timeout=3)
        detector.detect_live("discarded", None, [], 0.4)
        detector.detect_live("latest", None, [], 0.8)
        release.set()
        assert finished.wait(timeout=3)
        assert frames == ["first", "latest"]
    finally:
        release.set()
        detector.stop_live()
        detector._live_worker.join(timeout=3)
