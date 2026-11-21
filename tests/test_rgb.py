from unittest.mock import patch

import cv2
import numpy as np
import pytest

from RGB import ColorMonitor, Thresholds, parse_reference
from RGB.color import lab_values, measure, region_pixels
from fastgears.config import Settings
from fastgears.llm.coach import compact_state
from fastgears.pipeline import Pipeline
from fastgears.vision.face import FACE_REGION_CONTOURS, FaceResult


def rectangle(x0, y0, x1, y1):
    return np.array(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=float)


CONTOURS = {
    "cheek_right": rectangle(55, 68, 87, 97), "cheek_left": rectangle(148, 68, 177, 97),
    "forehead": rectangle(93, 35, 133, 57), "mouth_outer": rectangle(90, 104, 138, 132),
    "mouth_inner": rectangle(101, 114, 127, 122),
}
REFERENCE = (0.05, 0.05, 0.18, 0.19)
ORIGINAL = (58., 18., 22.)
LIPS = (47., 31., 14.)
FAST = Thresholds(calibration_s=0.4, persistence_s=0.4, recovery_s=0.3, min_focus=0)


def bgr(value):
    patch = cv2.cvtColor(np.asarray(value, dtype=np.float32).reshape(1, 1, 3), cv2.COLOR_LAB2BGR)
    return np.rint(255 * patch[0, 0]).astype(np.uint8)


def sample(right=ORIGINAL, left=ORIGINAL, lips=LIPS, witness=(53, 0, 0), forehead=ORIGINAL, mar=0.03):
    img = np.empty((170, 230, 3), dtype=np.uint8)
    img[:] = bgr((35, 7, 12))
    img[8:33, 11:42] = bgr(witness)
    for key, lab in zip(CONTOURS, (right, left, forehead, lips, (94, 0, 0))):
        cv2.fillPoly(img, [CONTOURS[key].astype(np.int32)], tuple(int(v) for v in bgr(lab)))
    face = FaceResult(np.zeros((478, 3)), (48, 26, 183, 145), pitch_deg=0,
                      yaw_deg=0, roll_deg=0, eyes=(112, 58, 65), mar=mar,
                      ear_left=0.30, ear_right=0.30)
    return img, face, CONTOURS


def feed(monitor, first, last, **kwargs):
    return [monitor.update(*sample(**kwargs), i / 10) for i in range(first, last)]


def calibrated(reference=REFERENCE, **kwargs):
    m = ColorMonitor(reference, FAST)
    base = feed(m, 0, 8, **kwargs)[-1]
    assert all(region["delta"] is not None for region in base["regions"].values())
    return m


def test_cielab_order_and_normalization():
    colors = np.array([[118, 118, 118], [37, 52, 165], [159, 63, 42]], np.uint8)
    neutral, red, blue = lab_values(colors)
    assert 45 < neutral[0] < 55 and neutral[3] < 1
    assert red[1] > 35 and blue[2] < -35
    np.testing.assert_allclose(lab_values(colors), lab_values(colors.astype(np.float32) / 255))
    with pytest.raises(ValueError):
        lab_values(colors.astype(float))


def test_lip_mask_excludes_teeth_and_is_stable_under_image_transforms():
    img, _, contours = sample()
    outer, inner = contours["mouth_outer"], contours["mouth_inner"]
    expected = measure(region_pixels(img, outer, inner), lips=True).lab
    original = np.array((outer, inner))
    rotated = np.stack((original[:, :, 1], img.shape[1] - 1 - original[:, :, 0]), axis=-1)
    mirrored = original.copy()
    mirrored[:, :, 0] = img.shape[1] - 1 - mirrored[:, :, 0]
    variants = ((np.fliplr(img), mirrored), (np.rot90(img), rotated),
                (img[10:, 20:], original - (20, 10)),
                (np.repeat(np.repeat(img, 2, axis=0), 2, axis=1), 2 * original))
    for frame, rings in variants:
        pixels = region_pixels(frame, *rings, inset=2)
        assert pixels is not None and pixels.max() < 225
        np.testing.assert_allclose(measure(pixels, lips=True).lab, expected, atol=1)
    assert region_pixels(img, outer + 1000, inner) is None
    assert measure(region_pixels(img, rectangle(1, 1, 3, 3))).reason == "region_too_small"


@pytest.mark.parametrize("lightness", [35, 58, 72])
@pytest.mark.parametrize("reference", [None, REFERENCE], ids=["forehead", "external_patch"])
@pytest.mark.parametrize("kind,change", [
    ("cheek_redness", (8, 0)), ("cheek_color_reduction", (-7, -6)), ("lip_tone_shift", (0, -9)),
])
def test_persistent_color_changes_across_base_lightness(kind, change, lightness, reference):
    base = (float(lightness), *ORIGINAL[1:])
    m = calibrated(reference=reference, right=base, left=base)
    if kind == "lip_tone_shift":
        settings = {"right": base, "left": base, "lips": (LIPS[0], LIPS[1], LIPS[2] + change[1])}
    else:
        shifted = (base[0], base[1] + change[0], base[2] + change[1])
        settings = {"right": shifted, "left": shifted}
    states = feed(m, 8, 16, **settings)
    events = [s["observations"][kind] for s in states]
    assert events[0]["state"] == "candidate" and events[4]["state"] == "active"
    assert "finding" not in events[0] and events[4]["finding"]
    assert events[4]["lighting_source"] == ("forehead" if reference is None else "external_patch")
    assert events[4]["id"] == events[-1]["id"]
    if kind == "lip_tone_shift":
        assert events[-1]["direction"] == "blue_axis"
        assert events[-1]["delta"]["lips"]["b"] < 0
    recovery = feed(m, 16, 21, right=base, left=base)
    assert recovery[0]["observations"][kind]["state"] == "recovering"
    assert recovery[-1]["recent_events"][-1]["state"] == "ended"


@pytest.mark.parametrize("witness", [(70, 0, 0), (53, 9, -10)])
def test_reference_vetoes_exposure_and_white_balance_change(witness):
    m = calibrated()
    red = (58, 29, 22)
    active = feed(m, 8, 14, right=red, left=red)[-1]
    assert active["observations"]["cheek_redness"]["state"] == "active"
    changed = feed(m, 14, 16, right=red, left=red, witness=witness)[-1]
    assert changed["lighting"] == "lighting_changed"
    assert changed["recent_events"][-1]["state"] == "interrupted"
    assert changed["recent_events"][-1]["reason"] == "lighting_changed"
    returned = feed(m, 16, 18)[-1]
    assert returned["lighting"] == "stable"
    assert returned["regions"]["cheek_left"]["delta"]["a"] == 0


def test_small_light_changes_are_tolerated_but_larger_drifts_gate_each_finding():
    m = calibrated()
    red = (65.5, 29, 22)
    normal = feed(m, 8, 14, right=red, left=red, forehead=(66.4, 18, 22), witness=(58.4, 3.6, 0))[-1]
    assert normal["lighting"] == "stable"
    assert normal["observations"]["cheek_redness"]["state"] == "active"
    assert normal["observations"]["cheek_color_reduction"]["state"] == "idle"
    assert normal["observations"]["lip_tone_shift"]["state"] == "idle"
    glare = feed(m, 14, 15, right=(68, 18, 22), left=red)[-1]
    assert glare["lighting"] == "stable"
    assert glare["observations"]["cheek_redness"]["reason"] == "luminance_changed"
    assert glare["observations"]["lip_tone_shift"]["state"] == "idle"
    exposed = feed(m, 15, 16, witness=(60.1, 0, 0))[-1]
    assert exposed["lighting"] == "lighting_changed"
    assert all(item["state"] == "unavailable" for item in exposed["observations"].values())


@pytest.mark.parametrize("lightness", [46, 70])
def test_bilateral_redness_is_detected_with_moderate_lightness_change(lightness):
    m = calibrated(reference=None)
    red = (lightness, 26, 22)
    states = feed(m, 8, 16, right=red, left=red)
    assert states[0]["observations"]["cheek_redness"]["state"] == "candidate"
    assert states[-1]["observations"]["cheek_redness"]["state"] == "active"
    assert states[-1]["observations"]["cheek_color_reduction"]["reason"] == "luminance_changed"


@pytest.mark.parametrize("right,left,reason", [
    ((58, 26, 22), (70, 26, 22), "uneven_cheek_light"),
    ((39, 27, 22), (39, 27, 22), "luminance_changed"),
])
def test_redness_remains_unevaluable_in_uneven_or_excessive_light(right, left, reason):
    m = calibrated(reference=None)
    red = feed(m, 8, 16, right=right, left=left)[-1]["observations"]["cheek_redness"]
    assert red["state"] == "unavailable"
    assert red["reason"] == reason


def test_unilateral_colour_change_stays_below_threshold_without_a_patch():
    red = (58, 30, 22)
    m = calibrated(reference=None)
    assert feed(m, 8, 18, right=red)[-1]["observations"]["cheek_redness"]["state"] == "idle"


def test_forehead_compensates_a_small_shared_shift_and_vetoes_a_large_one():
    m = calibrated(reference=None)
    face_shift = (60, 22.5, 22)
    lip_shift = (49, 35.5, 14)
    states = feed(m, 8, 17, right=face_shift, left=face_shift, forehead=face_shift, lips=lip_shift)
    result = states[-1]
    red = result["observations"]["cheek_redness"]
    assert result["lighting"] == "estimated"
    assert red["delta"]["cheek_right"]["a"] > 3
    assert abs(red["compared_delta"]["cheek_right"]["a"]) < 1
    assert all(item["state"] == "idle" for item in result["observations"].values())
    light_jump = feed(m, 17, 19, forehead=(71, 22.5, 22))[-1]
    assert light_jump["lighting"] == "lighting_changed"
    assert all(item["state"] == "unavailable" for item in light_jump["observations"].values())


def apply_camera_gains(frame, gains):
    encoded = frame / 255.0
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    adjusted = np.clip(linear * gains, 0, 1)
    encoded = np.where(adjusted <= 0.0031308, 12.92 * adjusted, 1.055 * adjusted ** (1 / 2.4) - 0.055)
    return np.rint(255 * encoded).astype(np.uint8)


@pytest.mark.parametrize("reference", [None, REFERENCE], ids=["forehead", "external_patch"])
@pytest.mark.parametrize("gains", [(1.20, 1.20, 1.20), (0.80, 0.80, 0.80), (1.08, 1.01, 0.95)])
def test_exposure_and_white_balance_alone_do_not_trigger_an_observation(reference, gains):
    monitor = calibrated(reference)
    frame, face, contours = sample()
    adjusted = apply_camera_gains(frame, gains)
    for i in range(8, 16):
        state = monitor.update(adjusted, face, contours, i / 10)
    assert all(item["state"] == "idle" for item in state["observations"].values())
    for name in ("cheek_right", "cheek_left", "lips"):
        assert max(abs(state["regions"][name]["compared_delta"][axis]) for axis in ("a", "b", "C")) < 1.0


@pytest.mark.parametrize("reference", [None, REFERENCE], ids=["forehead", "external_patch"])
@pytest.mark.parametrize("kind,changes", [
    ("cheek_redness", {"right": (58, 20.6, 22), "left": (58, 20.6, 22)}),
    ("cheek_color_reduction", {"right": (58, 15.4, 22), "left": (58, 15.4, 22)}),
    ("lip_tone_shift", {"lips": (47, 31, 10.8)}),
])
def test_subtle_local_color_change_survives_a_shared_exposure_change(reference, kind, changes):
    monitor = calibrated(reference)
    frame, face, contours = sample(**changes)
    adjusted = apply_camera_gains(frame, (1.12, 1.12, 1.12))
    for i in range(8, 16):
        state = monitor.update(adjusted, face, contours, i / 10)
    assert state["observations"][kind]["state"] == "active"


@pytest.mark.parametrize("reference", [None, REFERENCE], ids=["forehead", "external_patch"])
def test_calibration_removes_shared_light_variation_from_adaptive_threshold(reference):
    monitor = ColorMonitor(reference, Thresholds(calibration_s=0.5, persistence_s=0.4, min_focus=0))
    frame, face, contours = sample()
    dark = apply_camera_gains(frame, (0.94, 0.98, 1.04))
    bright = apply_camera_gains(frame, (1.06, 1.02, 0.96))
    for i in range(8):
        monitor.update((dark, bright)[i % 2], face, contours, i / 10)

    red = (58, 20.6, 22)
    event = feed(monitor, 8, 16, right=red, left=red)[-1]["observations"]["cheek_redness"]
    assert monitor.baselines["cheek_right"].mad[1] < 0.5
    assert event["state"] == "active" and event["threshold_ratio"] >= 1


def test_forehead_mode_rejects_an_occluded_forehead_without_replacing_baseline():
    m = calibrated(reference=None)
    red = (58, 30, 22)
    assert feed(m, 8, 14, right=red, left=red)[-1]["observations"]["cheek_redness"]["state"] == "active"
    frame, face, contours = sample(right=red, left=red)
    status = m.update(frame, face, {**contours, "forehead": None}, 1.4)
    assert status["lighting"] == "forehead_unavailable"
    assert status["recent_events"][-1]["reason"] == "forehead_unavailable"
    restored = feed(m, 15, 17)[-1]
    assert restored["lighting"] == "estimated"
    assert restored["observations"]["cheek_redness"]["state"] == "idle"


def test_lips_calibrate_against_forehead_colour_from_the_same_window():
    m = ColorMonitor(None, FAST)
    for i in range(8):
        frame, face, contours = sample()
        m.update(frame, face, {**contours, "mouth_outer": None}, i / 10)
    shift = (59, 22, 22)
    lip = (48, 35, 14)
    result = feed(m, 8, 19, right=shift, left=shift, forehead=shift, lips=lip)[-1]
    assert result["regions"]["lips"]["calibrated"]
    assert result["observations"]["lip_tone_shift"]["state"] == "idle"


def test_lipstick_at_calibration_mouth_open_and_occlusion():
    painted = (44, 44, 15)
    m = calibrated(lips=painted)
    assert feed(m, 8, 15, lips=painted)[-1]["observations"]["lip_tone_shift"]["state"] == "idle"
    grey = (44, 29, 15)
    active = feed(m, 15, 22, lips=grey)[-1]["observations"]["lip_tone_shift"]
    assert active["state"] == "active" and active["direction"] == "lower_chroma"
    occluded = feed(m, 22, 23, lips=(2, 0, 0))[-1]
    assert occluded["regions"]["lips"]["reason"] == "exposure_or_occlusion"
    assert occluded["recent_events"][-1]["state"] == "interrupted"
    mouth_open = feed(m, 23, 25, lips=grey, mar=0.4)[-1]
    assert mouth_open["regions"]["lips"]["reason"] == "mouth_moving"


def test_monochrome_and_lost_face_gate_and_reset():
    m = calibrated()
    frame, face, contours = sample()
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert m.update(gray, face, contours, 0.9)["lighting"] == "color_unavailable"
    assert m.update(frame, None, {}, 1.2)["lighting"] == "face_not_visible"
    assert m.update(frame, face, contours, 3.2)["lighting"] == "recalibration_needed"
    m.reset()
    assert feed(m, 40, 48)[-1]["lighting"] == "stable"


def test_frame_gap_interrupts_an_episode_and_requires_new_persistence():
    m = calibrated()
    red = (58, 29, 22)
    assert feed(m, 8, 14, right=red, left=red)[-1]["observations"]["cheek_redness"]["state"] == "active"
    after = feed(m, 19, 20, right=red, left=red)[-1]
    assert after["recent_events"][-1]["reason"] == "frame_gap"
    assert after["observations"]["cheek_redness"]["state"] == "candidate"


def _no_driver_lock(pipeline):
    """These tests are about colour, not driver selection: the one face is the driver."""
    pipeline.driver.roi = lambda shape: None
    pipeline.driver.update = lambda faces, shape, t: faces[0] if faces else None
    pipeline.driver.owns_hand = lambda hand, face: True
    return pipeline


def test_options_and_pipeline_preserve_gestures_when_rgb_cannot_measure():
    assert parse_reference("0.05,0.05,0.18,0.19") == REFERENCE
    with pytest.raises(ValueError):
        parse_reference("0.9,0,0.3,1")
    frame, face, _ = sample()
    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    with patch("fastgears.vision.face.FaceTracker") as tracker:
        tracker.return_value.process_all.return_value = [face]
        tracker.return_value.process_roi.return_value = None
        p = _no_driver_lock(Pipeline(Settings(enable_rppg=False, enable_objects=False, enable_rgb=True)))
        p._step(gray, p._t0 + 0.1, 100)
    state = p.state()
    assert state["rgb"]["lighting"] == "color_unavailable"
    assert state["attention"]["face_present"] and state["discomfort"]["markers"]
    assert all("rgb" not in name for name in compact_state(state))


def test_pipeline_detects_a_colour_change_without_a_reference_rectangle():
    config = Settings(enable_rppg=False, enable_objects=False, enable_rgb=True, rgb_reference="",
                      rgb_calibration_s="0.4", rgb_persistence_s="0.4")
    with patch("fastgears.vision.face.FaceTracker") as tracker, \
            patch("fastgears.pipeline.face_region_contours", return_value=CONTOURS):
        pipeline = _no_driver_lock(Pipeline(config))
        for index in range(15):
            cheek = ORIGINAL if index < 8 else (58, 29, 22)
            frame, face, _ = sample(right=cheek, left=cheek)
            tracker.return_value.process_all.return_value = [face]
            tracker.return_value.process_roi.return_value = None
            pipeline._step(frame, pipeline._t0 + index / 10, index * 100)
    state = pipeline.state()
    assert state["rgb"]["lighting_source"] == "forehead" and state["rgb"]["reference_rect"] is None
    assert state["rgb"]["observations"]["cheek_redness"]["state"] == "active"
    assert state["attention"]["face_present"] and state["discomfort"]["markers"]
