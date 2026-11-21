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
    FACE_REGION_CONTOURS, PULSE_REGIONS, FaceResult, face_region_contours,
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


def test_filter_rejects_motion_holds_and_slew_limits():
    r = RPPG()
    good = lambda bpm: {"bpm": bpm, "snr": 5.0, "motion": 0.02, "method": "regional_consensus"}
    assert r._filter(good(72), 100.0) == {"bpm": 72, "bpm_raw": 72, "pulse_state": "tracking", "rejected": None}
    # head moving: a +30 BPM spike must not reach the user
    out = r._filter({"bpm": 105, "snr": 3.0, "motion": 0.3, "method": "regional_consensus"}, 101.0)
    assert out["bpm"] == 72 and out["pulse_state"] == "held" and out["rejected"] == "head moving"
    out = r._filter({"bpm": 110, "snr": -2.0, "motion": 0.01, "method": "regional_consensus"}, 102.0)
    assert out["bpm"] == 72 and out["rejected"] == "weak signal"
    # real change is followed, but at most SLEW_BPM_PER_S per second (median of last 3 accepted)
    r._filter(good(90), 103.0)
    out = r._filter(good(90), 104.0)
    assert 72 < out["bpm"] <= 72 + 3 * 4            # 4 s since the last good reading
    # rejected for longer than HOLD_MAX_S: back to measuring
    out = r._filter({"bpm": None}, 104.0 + 13)
    assert out["bpm"] is None and out["pulse_state"] == "measuring"


def test_filter_never_promotes_a_provisional_reading():
    r = RPPG()
    out = r._filter({"bpm": 80, "snr": 4.0, "motion": 0.01, "method": "provisional", "raw_bpm": 81}, 10.0)
    assert out == {"bpm": None, "bpm_raw": 81, "pulse_state": "provisional", "rejected": "no regional consensus"}
    r._filter({"bpm": 70, "snr": 4.0, "motion": 0.01, "method": "regional_consensus"}, 11.0)
    out = r._filter({"bpm": 95, "snr": 4.0, "motion": 0.01, "method": "provisional"}, 12.0)
    assert out["bpm"] == 70 and out["pulse_state"] == "held"


def test_head_jump_restarts_the_window():
    samples = np.array(list(_signals((72, 72, 72), seconds=10)))
    samples[samples[:, 0] >= 6, 1] += 0.4 * FACE[2]       # 0.4 eye distances jump at t=6 s
    result = RPPG()._estimate(samples.tolist(), {})
    assert result["bpm"] is None and result["reason"] == "settling" and result["window_s"] < 6


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


def test_pipeline_samples_the_pulse_on_every_camera_frame():
    """The detector loop runs at ~8 fps; the pulse thread feeds every 30 fps camera frame
    with the newest driver landmarks, so the estimator reaches a trusted reading."""
    from fastgears.pipeline import RPPG_FACE_MAX_AGE_S, Pipeline

    p = Pipeline.__new__(Pipeline)
    import threading
    p._rppg_lock = threading.Lock()
    p.rppg = RPPG()
    face = SimpleNamespace(eyes=FACE, landmarks_px=None)
    frames = list(_frames(seconds=10.0))
    for i, (frame, t) in enumerate(frames):
        if i % 4 == 0:                                   # detector loop: every 4th frame (~8 fps)
            p._rppg_face = (face, t)
        p._feed_rppg(frame, t)
    s = p.rppg.status()
    assert s["fps"] >= 15 and s["method"] == "regional_consensus"
    assert s["pulse_state"] == "tracking" and abs(s["bpm"] - 72) <= 3
    # landmarks too old: the pulse thread treats the face as gone
    frame, t = frames[-1]
    p._rppg_face = (face, t + 1 - RPPG_FACE_MAX_AGE_S - 0.1)
    p._feed_rppg(frame, t + 1)
    assert p.rppg.status()["face"] is False
