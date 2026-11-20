"""rPPG tests with synthetic frames (no camera)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastgears.vision.rppg import RPPG, pos_spectrum, trust_score  # noqa: E402

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
    good = lambda bpm: {"bpm": bpm, "snr": 5.0, "motion": 0.02}
    assert r._filter(good(72), 100.0) == {"bpm": 72, "bpm_raw": 72, "pulse_state": "tracking", "rejected": None}
    # head moving: a +30 BPM spike must not reach the user
    out = r._filter({"bpm": 105, "snr": 3.0, "motion": 0.3}, 101.0)
    assert out["bpm"] == 72 and out["pulse_state"] == "held" and out["rejected"] == "head moving"
    out = r._filter({"bpm": 110, "snr": -2.0, "motion": 0.01}, 102.0)
    assert out["bpm"] == 72 and out["rejected"] == "weak signal"
    # real change is followed, but at most SLEW_BPM_PER_S per second (median of last 3 accepted)
    r._filter(good(90), 103.0)
    out = r._filter(good(90), 104.0)
    assert 72 < out["bpm"] <= 72 + 3 * 4            # 4 s since the last good reading
    # rejected for longer than HOLD_MAX_S: back to measuring
    out = r._filter({"bpm": 150, "snr": 1.0, "motion": 0.5}, 104.0 + 13)
    assert out["bpm"] is None and out["pulse_state"] == "measuring"
