"""rPPG tests with synthetic frames (no camera)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastgears.vision.rppg import (  # noqa: E402
    RPPG, _display_provisional_bpm, _region_pixels, pos_spectrum, trust_score,
)
from fastgears.vision.face import (  # noqa: E402
    FACE_REGION_CONTOURS, PULSE_REGIONS, FaceResult, FaceTracker, face_region_contours,
)

FACE = (160.0, 110.0, 60.0)   # ROIs: forehead y 44-83, cheeks y 131-167, all inside 240x320


def _frames(bpm=72.0, seconds=12.0, fps=30.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([100.0, 140.0, 190.0])          # BGR skin tone
    amp = np.array([0.3, 1.5, 0.5])                  # pulse mostly in green
    f = bpm / 60.0
    t0 = 1000.0
    for i in range(int(seconds * fps)):
        t = t0 + i / fps
        px = base + amp * np.sin(2 * np.pi * f * t)
        img = px + rng.normal(0, 2.0, (240, 320, 3))
        yield np.clip(img, 0, 255).astype(np.uint8), t


def test_bpm_from_synthetic_pulse():
    r = RPPG()
    for frame, t in _frames():
        r.update(frame, FACE, t)
    s = r.status()
    assert s["face"] is True
    assert s["bpm"] is not None and abs(s["bpm"] - 72) <= 4, s
    assert s["snr"] > 0
    assert 15 <= s["fps"] <= 25                       # throttled to ~20 Hz from 30 fps input
    assert len(r.rois()) == 3
    assert s["trust"] is not None and set(s["trust"]) == {"score", "label", "reasons", "parts"}


def test_no_face_no_bpm():
    r = RPPG()
    for frame, t in _frames(seconds=8.0):
        r.update(frame, None, t)
    s = r.status()
    assert s["bpm"] is None
    assert s["face"] is False
    assert r.rois() == []


def test_face_dropped_after_two_seconds():
    r = RPPG()
    frames = list(_frames(seconds=4.0))
    for frame, t in frames[:30]:
        r.update(frame, FACE, t)
    for frame, t in frames[30:]:
        r.update(frame, None, t)                      # 3 s missing
    assert r.status()["face"] is False


def test_pos_spectrum_clean_sine():
    ts = np.arange(0, 10, 1 / 20.0)
    g = 140 + np.sin(2 * np.pi * 1.2 * ts)
    rgb = np.stack([190 + 0.3 * np.sin(2 * np.pi * 1.2 * ts), g, np.full_like(ts, 100.0)], axis=1)
    _, _, f0, snr = pos_spectrum(ts, rgb)
    assert abs(f0 * 60 - 72) <= 3 and snr > 0


def test_trust_score_sanity():
    good = trust_score(8.0, [72, 73, 71], 72, [72, 72, 73, 72], 0.01, 128, 0.0)
    assert good["label"] == "reliable" and good["score"] >= 90 and good["reasons"] == []
    bad = trust_score(-4.0, [60, 90, 110], 72, [72], 0.5, 30, 0.0)
    assert bad["label"] == "unreliable"
    assert "weak pulse signal" in bad["reasons"] and "head moving" in bad["reasons"]
    assert "face too dark" in bad["reasons"] and "still settling" in bad["reasons"]
    over = trust_score(8.0, [72, 72, 72], 72, [72, 72, 72], 0.0, 230, 0.0)
    assert over["parts"]["exposure"] == 0.5 and "face overexposed" in over["reasons"]


def _signals(rates, *, fps=20, seconds=10):
    for t in np.arange(0, seconds, 1 / fps):
        signals = [channel for bpm in rates for channel in
                   ((np.nan,)*3 if bpm is None else
                    np.array([190, 140, 100]) + np.array([0.5, 1.8, 0.2])*np.sin(2*np.pi*bpm*t/60))]
        yield (t, *FACE, *signals)


@pytest.mark.parametrize("rates,expected,reason", [
    ((110, 72, 72), 72, None),
    ((None, 72, 72), 72, None),
    ((58, 74, 96), None, "regions_disagree"),
    ((72, None, None), None, "insufficient_regions"),
])
def test_regional_consensus(rates, expected, reason):
    result = RPPG()._estimate(list(_signals(rates)), {})
    assert result["reason"] == reason
    if expected is None:
        assert result["bpm"] is not None
        assert result["method"] == "provisional"
        assert result["trust"]["label"] == "unreliable" and result["trust"]["score"] < 40
    else:
        assert result["method"] == "regional_consensus"
        assert abs(result["bpm"] - expected) <= 2
        assert result["region_quality"]["forehead"]["used"] is False
        assert all(result["region_quality"][name]["used"] for name in PULSE_REGIONS[1:])


def test_rejects_low_sample_rate_and_internal_gap():
    rppg = RPPG()
    sparse = rppg._estimate(list(_signals((72, 72, 72), fps=5)), {})
    assert sparse["bpm"] is None and sparse["reason"] == "insufficient_samples"
    gap = rppg._estimate([p for p in _signals((72, 72, 72)) if not 4 <= p[0] <= 4.5], {})
    assert gap["bpm"] is None and gap["max_gap_s"] > 0.35


def test_continuous_10_hz_signal_has_only_a_provisional_estimate():
    result = RPPG()._estimate(list(_signals((72, 72, 72), fps=10)), {})
    assert abs(result["bpm"] - 72) <= 2
    assert result["method"] == "provisional" and result["reason"] == "low_sample_rate"
    assert result["trust"]["label"] == "unreliable"


def test_fallback_display_rejects_an_isolated_spike_but_follows_a_sustained_change():
    assert _display_provisional_bpm([74, 128]) == 74
    assert _display_provisional_bpm([73, 74, 128]) == 74
    assert _display_provisional_bpm([74, 127, 128]) == 127


def test_motion_explaining_a_stronger_colour_change_is_removed():
    samples = np.array(list(_signals((72, 72, 72), seconds=10)))
    artifact = np.sin(2 * np.pi * 1.9 * samples[:, 0])
    samples[:, 1] += 5 * artifact
    samples[:, (5, 8, 11)] += 4 * artifact[:, None]
    result = RPPG()._estimate(samples.tolist(), {})
    assert abs(result["bpm"] - 72) <= 2
    assert result["method"] == "regional_consensus"


def test_regions_can_agree_on_a_secondary_peak_under_different_artifacts():
    samples = np.array(list(_signals((72, 72, 72), seconds=10)))
    seconds = samples[:, 0]
    for channel, bpm in zip((5, 8, 11), (100, 125, 151)):
        samples[:, channel] += 2.0 * np.sin(2 * np.pi * bpm * seconds / 60)
    result = RPPG()._estimate(samples.tolist(), {})
    assert abs(result["bpm"] - 72) <= 2
    assert result["method"] == "regional_consensus"
    assert result["trust"]["label"] != "reliable"


def test_no_usable_region_does_not_manufacture_a_pulse():
    result = RPPG()._estimate(list(_signals((None, None, None))), {})
    assert result["bpm"] is None and result["method"] is None


def test_loss_and_long_gap_invalidate_cached_estimate():
    rppg = RPPG()
    samples = list(_frames(seconds=8))
    for frame, t in samples:
        rppg.update(frame, FACE, t)
    assert rppg.status()["bpm"] is not None
    rppg.update(None, None, t + 0.1)
    assert rppg.status()["bpm"] is None
    rppg.update(frame, FACE, t + 0.6)
    assert rppg.status()["reason"] == "sampling_gap"


def _mesh():
    points = np.full((478, 3), np.nan)
    centres = ((160, 60), (121, 148), (199, 148), (160, 128), (160, 170), (160, 170))
    for (name, indices), centre in zip(FACE_REGION_CONTOURS.items(), centres):
        angles = np.arange(len(indices)) * 2 * np.pi / len(indices)
        radius = 8 if name == "mouth_inner" else 17
        points[list(indices), :2] = centre + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return points


def test_landmark_regions_mirror_and_reject_missing_points_independently():
    points = _mesh()
    shape = (240, 320, 3)
    original = face_region_contours(FaceResult(points, (0, 0, 0, 0)), shape)
    assert all(original.values())
    points[:, 0] = shape[1] - 1 - points[:, 0]
    mirrored = face_region_contours(FaceResult(points, (0, 0, 0, 0)), shape)
    for name, contour in original.items():
        coords = np.array(contour)
        coords[:, 0] = shape[1] - 1 - coords[:, 0]
        np.testing.assert_allclose(mirrored[name], coords)
    points[FACE_REGION_CONTOURS["mouth_inner"][0], 0] = -1
    affected = face_region_contours(FaceResult(points, (0, 0, 0, 0)), shape)
    assert affected["mouth_inner"] is None and affected["mouth_outer"] and affected["nose"]


def test_face_tracking_resize_keeps_original_pixel_coordinates():
    tracker = FaceTracker.__new__(FaceTracker)
    tracker._max_input_side = 768
    tracker._last_ts = -1
    tracker._mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda **kwargs: kwargs["data"])

    def detect(image, timestamp):
        assert image.shape == (432, 768, 3) and timestamp == 10
        point = SimpleNamespace(x=0.25, y=0.75, z=-0.1)
        return SimpleNamespace(face_landmarks=[[point] * 478], face_blendshapes=[],
                               facial_transformation_matrixes=[])

    tracker._landmarker = SimpleNamespace(detect_for_video=detect)
    face = tracker.process(np.zeros((720, 1280, 3), np.uint8), 10)
    np.testing.assert_allclose(face.landmarks_px[0], [320, 540, -128])


def test_roi_masks_sample_two_cheeks_and_reject_clipped_forehead():
    rppg = RPPG()
    landmarks = _mesh()
    for frame, t in _frames(seconds=8):
        frame[:90] = 255  # only forehead is overexposed
        rppg.update(frame, FACE, t, landmarks=landmarks)
    result = rppg.status()
    assert abs(result["bpm"] - 72) <= 2
    assert result["diag"]["region_skin_px"]["forehead"] == 0
    assert len(rppg.rois()) == 2


def test_subpixel_region_sampling_has_stable_support_across_integer_boundaries():
    gradient = np.broadcast_to(3 * np.arange(64, dtype=np.uint8)[None, :, None], (64, 64, 3)).copy()
    contour = np.array(((10.8, 10), (29.8, 10), (29.8, 28), (10.8, 28)))
    first, _ = _region_pixels(gradient, contour)
    shifted, _ = _region_pixels(gradient, contour + (0.4, 0))
    assert len(first) == len(shifted)
    assert shifted.mean() - first.mean() == pytest.approx(1.2, abs=0.1)
