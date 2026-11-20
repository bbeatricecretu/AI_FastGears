"""Remote photoplethysmography (rPPG): heart rate from one webcam.

Ported from afferent-junction2026 jetson/vision/rppg.py (validated there against a
finger pulse sensor: mean absolute difference 1.8 BPM). The estimator uses
POS (Wang et al. 2017, "Algorithmic Principles of Remote PPG") over the last
10 s of forehead + cheek skin colour, dominant frequency in 45-180 BPM, plus the
per-region trust score. Dropped from the original: YOLO-pose face locator, second
camera / own capture, multi-camera fusion, GStreamer/JPEG feed path. This
version combines spectra from individually tracked and qualified regions.

Each heartbeat changes skin colour very slightly (mostly green). The live
caller supplies MediaPipe landmarks and the eye centre and scale for motion.

Honest requirements (from the source README):
  * the face must be still, 0.5-1 m from the camera, lit by steady white light;
  * snr < 0 dB means the reading is noise: don't trust it (see "trust" in status());
  * a light flashing at heart-rate frequency on the face fakes a result;
  * camera auto-exposure / white balance changes skin colour far more than a pulse.
    On Linux with v4l2-ctl, pass `device=` to lock them (best effort); elsewhere they
    stay automatic and readings are less reliable.
"""
from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from itertools import combinations
from statistics import median

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .face import PULSE_REGIONS, landmark_region

logger = logging.getLogger("rppg")

SAMPLE_HZ = 20.0
WINDOW_S = 10.0
MIN_WINDOW_S = 6.0
BAND = (0.75, 3.0)          # 45-180 BPM
ESTIMATE_EVERY_S = 1.0
MIN_SAMPLE_FRACTION = 0.6
PROVISIONAL_SAMPLE_FRACTION = 0.4
MAX_GAP_S = 0.35
MIN_REGION_COVERAGE = 0.85
MIN_REGION_SNR_DB = 0.0
REGION_AGREEMENT_BPM = 6.0
SECONDARY_MIN_SNR_DB = -3.5
SECONDARY_MIN_POWER = 0.42
MAX_TRACK_AGE_S = 5.0
REGION_RASTER_SIDE = 40

# forehead + two cheeks, relative to (cx, cy) in units of eye distance d
BOXES = [(-0.55, -1.10, 0.55, -0.45),   # forehead, above the eyes
         (-0.95, 0.35, -0.35, 0.95),     # cheek below one eye
         (0.35, 0.35, 0.95, 0.95)]       # cheek below the other

# Every auto-adjustment changes skin colour far more than a heartbeat, so lock them.
# 50 Hz mains; 30 ms exposure (300 x 100 us) = 3 full flicker cycles.
LOCKS = {
    "c920": [  # Logitech C920
        ("power_line_frequency", 1), ("white_balance_automatic", 0),
        ("focus_automatic_continuous", 0), ("exposure_dynamic_framerate", 0),
        ("auto_exposure", 1), ("exposure_time_absolute", 300), ("white_balance_temperature", 4100),
    ],
    "default": [      # generic UVC: same idea, controls that exist get set
        ("power_line_frequency", 1), ("white_balance_automatic", 0),
        ("focus_automatic_continuous", 0), ("auto_exposure", 1),
        ("exposure_time_absolute", 300), ("white_balance_temperature", 4500),
    ],
}


# ── v4l2 (Linux only, optional) ───────────────────────────────────────
def v4l2_available() -> bool:
    return platform.system() == "Linux" and shutil.which("v4l2-ctl") is not None


def v4l2_set(device, name, value):
    try:
        r = subprocess.run(["v4l2-ctl", "-d", device, "-c", f"{name}={value}"],
                           capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def v4l2_get(device, name):
    try:
        r = subprocess.run(["v4l2-ctl", "-d", device, "-C", name], capture_output=True, text=True, timeout=5)
        return int(r.stdout.split(":")[-1].strip())
    except Exception:
        return None


def _v4l2_card(device):
    try:
        r = subprocess.run(["v4l2-ctl", "-d", device, "--info"], capture_output=True, text=True, timeout=5)
        return r.stdout.lower()
    except Exception:
        return ""


def _v4l2_gain_max(device, default=64):
    try:
        r = subprocess.run(["v4l2-ctl", "-d", device, "-l"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            if line.strip().startswith("gain "):
                m = re.search(r"max=(\d+)", line)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return default


def lock_camera(device):
    if not v4l2_available():
        return
    locks = LOCKS["c920"] if "c920" in _v4l2_card(device) else LOCKS["default"]
    ok = [n for n, v in locks if v4l2_set(device, n, v)]
    logger.info("%s: locked %s", device, ", ".join(ok) or "nothing")


# ── signal processing ───────────────────────────────────────────────
def pos_spectrum(ts, rgb, sample_hz=SAMPLE_HZ):
    """POS on unevenly sampled RGB means -> (band freqs, band power, f0 Hz, SNR dB)."""
    grid = np.arange(ts[0], ts[-1], 1.0 / sample_hz)            # uniform resampling
    c = np.stack([np.interp(grid, ts, rgb[:, k]) for k in range(3)], axis=1)
    n, L = len(c), int(1.6 * sample_hz)
    h = np.zeros(n)
    windows = sliding_window_view(c, L, axis=0)                   # each 1.6 s window
    red, green, blue = (windows / (windows.mean(axis=2, keepdims=True) + 1e-9)).transpose(1, 0, 2)
    s0, s1 = green - blue, -2 * red + green + blue              # plane orthogonal to skin tone
    scale = s0.std(axis=1) / (s1.std(axis=1) + 1e-9)
    pulses = s0 + scale[:, None] * s1
    pulses -= pulses.mean(axis=1, keepdims=True)
    for offset in range(L):
        h[offset:offset + len(pulses)] += pulses[:, offset]
    k = int(sample_hz)                                           # detrend: 1 s moving average
    h = h - np.convolve(h, np.ones(k) / k, mode="same")
    h = (h - h.mean()) * np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    spec = np.abs(np.fft.rfft(h, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_hz)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    fb, sb = freqs[band], spec[band]
    f0 = fb[int(np.argmax(sb))]
    return fb, sb, float(f0), _spectral_snr(fb, sb, f0)


def _spectral_snr(freqs, power, peak):
    signal = (np.abs(freqs - peak) <= 0.1) | (np.abs(freqs - 2 * peak) <= 0.1)
    return float(10 * np.log10((power[signal].sum() + 1e-12) / (power[~signal].sum() + 1e-12)))


def _remove_motion(rgb, geometry):
    """Remove colour variation explained by tracked translation and face scale."""
    regressors = (geometry - geometry.mean(axis=0)) / max(geometry[:, 2].mean(), 1.0)
    if np.max(regressors.std(axis=0)) < 0.004:
        return rgb
    # Ridge limits subtraction when a geometry predictor is nearly constant.
    regularization = len(rgb) * 0.004**2
    matrix = regressors.T @ regressors + regularization * np.eye(regressors.shape[1])
    coefficients = np.linalg.solve(matrix, regressors.T @ (rgb - rgb.mean(axis=0)))
    return rgb - regressors @ coefficients


def _regional_peaks(freqs, power, peak, snr):
    if snr >= MIN_REGION_SNR_DB:
        yield peak, snr, False
    # A shared secondary local maximum can survive a stronger, unrelated
    # artifact in each region. Compare power with that region's own maximum.
    local = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
    local = local[np.argsort(power[local])[-2:]]
    for i in local:
        if abs(freqs[i] - peak) * 60 <= REGION_AGREEMENT_BPM or power[i] < SECONDARY_MIN_POWER * power.max():
            continue
        secondary_snr = _spectral_snr(freqs, power, freqs[i])
        if secondary_snr >= SECONDARY_MIN_SNR_DB:
            yield freqs[i], secondary_snr, True


def _tracked_peak(freqs, power, history):
    strongest = int(np.argmax(power))
    if not history:
        return freqs[strongest]
    local = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
    nearby = local[np.abs(freqs[local] * 60 - np.median(history[-3:])) <= 7]
    if len(nearby):
        peak = nearby[np.argmax(power[nearby])]
        if power[peak] >= 0.60 * power[strongest]:
            return freqs[peak]
    return freqs[strongest]


def _display_provisional_bpm(recent):
    if len(recent) == 2 and abs(recent[-1] - recent[0]) > REGION_AGREEMENT_BPM:
        return recent[0]
    return round(median(recent))


def _region_pixels(frame, polygon):
    """Sample an in-frame polygon on a fixed, subpixel-aligned grid."""
    height, width = frame.shape[:2]
    if not np.isfinite(polygon).all():
        return None
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    x0, y0 = np.floor(lo).astype(int)
    x1, y1 = np.ceil(hi).astype(int) + 1
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    if np.min(hi - lo) < 5 or cv2.contourArea(cv2.convexHull(polygon.astype(np.float32))) < 35:
        return None
    side = REGION_RASTER_SIDE
    step_x, step_y = (hi - lo) / (side - 1)
    transform = np.array([[step_x, 0, lo[0]], [0, step_y, lo[1]]], dtype=np.float32)
    crop = cv2.warpAffine(frame, transform, (side, side), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    mask = np.zeros((side, side), np.uint8)
    points = np.rint((polygon - lo) / (step_x, step_y)).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)
    pixels = crop[mask != 0]
    return (pixels, (x0, y0, x1, y1)) if len(pixels) else None


def _region_color(pixels):
    """Trim local shadows and highlights without a fixed skin colour range."""
    y = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb)[:, 0, 0]
    lo, hi = np.percentile(y, (15, 90))
    clipped = pixels.max(axis=1) >= 250
    # Bright, nearly neutral pixels are commonly specular reflections.
    glare = (y >= 215) & (np.ptp(pixels, axis=1) <= 20)
    valid = (y >= lo) & (y <= hi) & ~clipped & ~glare
    usable = valid.sum() >= max(35, 0.45 * len(pixels)) and 35 <= np.median(y) <= 225
    rgb = pixels[valid, ::-1].mean(axis=0) if usable else (np.nan,) * 3
    return rgb, int(valid.sum()), float(y.mean()), float(clipped.mean())


def _pulse_result(**values):
    return {"bpm": None, "snr": None, "window_s": 0.0, "fps": None,
            "max_gap_s": None, "regions_bpm": [], "region_quality": {},
            "motion": None, "trust": None, "reason": None, "method": None, **values}


def trust_score(snr, region_bpms, bpm, history, motion, luma, clipped):
    """0-100 reliability of the reading, plus the reasons it isn't higher.

    With a single camera the forehead and both cheeks are three independent
    measurements of the same pulse.
      signal     SNR -2 dB -> 0 ... +6 dB -> 1
      regions    share of face regions within 6 BPM of the reading
      stability  spread of the last readings (0 BPM -> 1, 8+ BPM -> 0)
      stillness  eye-midpoint movement over the window, relative to eye distance
      exposure   face not too dark / clipped (multiplies the rest)
    """
    reasons = []
    sig = min(max((snr + 2) / 8, 0), 1)
    if sig < 0.5:
        reasons.append("weak pulse signal")
    valid = [x for x in region_bpms if x is not None]
    agree = sum(1 for x in valid if abs(x - bpm) <= 6)
    reg = agree / len(valid) if valid else 0.0
    if agree * 3 < 2 * len(valid):          # fewer than 2 of 3 regions agree
        reasons.append(f"face regions disagree ({agree}/{len(valid)})")
    if len(history) >= 3:
        spread = float(np.std(history[-5:]))
        stab = min(max(1 - spread / 8, 0), 1)
        if stab < 0.5:
            reasons.append(f"reading jumping (±{spread:.0f} BPM)")
    else:
        stab = 0.3
        reasons.append("still settling")
    still = min(max(1 - motion / 0.15, 0), 1)
    if still < 0.5:
        reasons.append("head moving")
    expo = 1.0
    if clipped > 0.05 or luma > 215:
        expo, _ = 0.5, reasons.append("face overexposed")
    elif luma < 50:
        expo, _ = 0.5, reasons.append("face too dark")
    score = round(100 * expo * (0.30 * sig + 0.30 * reg + 0.20 * stab + 0.20 * still))
    label = "reliable" if score >= 70 else "uncertain" if score >= 40 else "unreliable"
    return {"score": score, "label": label, "reasons": reasons,
            "parts": {"signal": round(sig, 2), "regions": round(reg, 2), "stability": round(stab, 2),
                      "stillness": round(still, 2), "exposure": expo}}


# ── single-camera estimator ───────────────────────────────────────────
class RPPG:
    """Heart rate from one camera. Feed every frame with update(); read status()."""

    def __init__(self, sample_hz: float = 20.0, device: str | None = None):
        self.sample_hz = float(sample_hz)
        self.device = device
        self._lock = threading.Lock()          # guards samples / face / rois / diag / result
        self._est_lock = threading.Lock()      # one recompute at a time
        self._samples: deque = deque(maxlen=int(self.sample_hz * 20))
        self._face = None            # (cx, cy, d) for the current frame
        self._last_frame_t = None
        self._next_t = None          # next sample time (caller's clock)
        self._rois: list[tuple[int, int, int, int]] = []
        self._diag: dict = {}
        self._history: deque = deque(maxlen=8)
        self._provisional_bpms: deque = deque(maxlen=3)
        self._last_consensus_t = None
        self._result = _pulse_result(reason="face_not_visible")
        self._est_at = None          # wall-clock (monotonic) of last recompute
        self._est_sample_t = None
        self._gen = 0                # sample-buffer generation (bumped on gain change)
        # face-gain control (Linux + v4l2-ctl + device only)
        self._v4l2 = bool(device) and v4l2_available()
        self._gain = None
        self._gain_max = 64
        self._gain_t = None
        self._gain_busy = False
        if self._v4l2:
            threading.Thread(target=self._lock_later, daemon=True).start()

    # ── camera locks ──────────────────────────────────────────────────
    def _lock_later(self):
        time.sleep(8)  # after the pipeline has opened the camera
        try:
            lock_camera(self.device)
            self._gain_max = _v4l2_gain_max(self.device)
        except Exception as e:
            logger.warning("rPPG camera lock failed: %s", e)

    # ── per frame ─────────────────────────────────────────────────────
    def update(self, frame_bgr: np.ndarray | None, face: tuple[float, float, float] | None,
               t: float, *, landmarks: np.ndarray | None = None) -> None:
        with self._lock:
            if self._last_frame_t is not None and t <= self._last_frame_t:
                return
            self._last_frame_t = t
            if face is None or frame_bgr is None or not np.isfinite(face).all() or face[2] < 1:
                if self._face is not None:
                    self._gen += 1
                    self._face = None
                    self._rois = []
                    self._result = _pulse_result(reason="face_not_visible")
                    self._est_at = None
                if self._samples and t - self._samples[-1][0] > MAX_GAP_S:
                    self._reset_window("face_not_visible")
                return
            self._face = face
            if self._next_t is not None and t < self._next_t:
                return
            if self._samples and t - self._samples[-1][0] > MAX_GAP_S:
                self._reset_window("sampling_gap")
            if self._result["reason"] == "face_not_visible":
                self._result = _pulse_result(reason="collecting")
            period = 1.0 / self.sample_hz
            self._next_t = t + period if self._next_t is None else max(self._next_t + period, t)
            generation = self._gen
        exposure = self._sample(frame_bgr, face, t, landmarks, generation)
        if exposure is not None and self._v4l2:
            self._maybe_gain(*exposure, t)

    def _reset_window(self, reason):
        """Reset sampling state with the buffer lock held."""
        self._gen += 1
        self._samples.clear()
        self._history.clear()
        self._provisional_bpms.clear()
        self._last_consensus_t = None
        self._next_t = self._est_at = self._est_sample_t = None
        self._result = _pulse_result(reason=reason)

    def _sample(self, frame, face, t, landmarks, generation):
        cx, cy, d = face
        if landmarks is None:
            # Keep compatibility with callers that only have eye geometry.
            polygons = [np.asarray(((cx + a*d, cy + b*d), (cx + c*d, cy + b*d),
                                    (cx + c*d, cy + e*d), (cx + a*d, cy + e*d)))
                        for a, b, c, e in BOXES]
        else:
            polygons = [landmark_region(landmarks, name) for name in PULSE_REGIONS]
        regions, rects, counts, exposure = [], [], [], []
        for polygon in polygons:
            measurement = _region_pixels(frame, polygon) if polygon is not None else None
            if measurement is None:
                regions.extend((np.nan,) * 3)
                counts.append(0)
                continue
            pixels, rect = measurement
            rgb, selected, luma, clipped = _region_color(pixels)
            regions.extend(rgb)
            counts.append(selected)
            exposure.append((len(pixels), luma, clipped))
            if np.isfinite(rgb).all():
                rects.append(rect)
        n_pixels = sum(n for n, _, _ in exposure)
        luma = sum(n * y for n, y, _ in exposure) / n_pixels if n_pixels else 0.0
        clipped = sum(n * c for n, _, c in exposure) / n_pixels if n_pixels else 0.0
        height, width = frame.shape[:2]
        with self._lock:
            if generation != self._gen:
                return None
            self._rois = rects
            self._samples.append((t, cx, cy, d, *regions))
            self._diag = {"skin_px": sum(counts), "roi_px": n_pixels, "brightness": round(luma),
                          "clipped": round(clipped, 3), "eye_dist_px": round(d), "frame": f"{width}x{height}",
                          "gain": self._gain, "region_skin_px": dict(zip(PULSE_REGIONS, counts))}
        return (luma, clipped) if n_pixels else None

    # ── face exposure via v4l2 gain (Linux, optional) ──────────────────
    def _maybe_gain(self, luma, clipped, t):
        if self._gain_busy or (self._gain_t is not None and t - self._gain_t < 2.0):
            return
        self._gain_t = t
        self._gain_busy = True
        threading.Thread(target=self._face_gain, args=(luma, clipped), daemon=True).start()

    def _face_gain(self, luma, clipped):
        """Keep the *face* well exposed (luma ~65-190, nothing clipped). A clipped face has no
        pulse signal; a dark one is all sensor noise. Any change restarts the 10 s window, so
        the step itself can't show up as a heartbeat. Runs off the pipeline thread."""
        try:
            if self._gain is None:
                self._gain = v4l2_get(self.device, "gain")
                if self._gain is None:
                    self._v4l2 = False       # camera has no gain control
                    return
            step = max(1, self._gain_max // 16)
            new = self._gain
            # wide dead band: every change restarts the window, so only act when clearly off
            if luma > 190 or clipped > 0.03:
                new = max(0, self._gain - step)
            elif luma < 65:
                new = min(self._gain_max, self._gain + step)
            if new != self._gain and v4l2_set(self.device, "gain", new):
                logger.info("rPPG: face luma %.0f clipped %.1f%% -> gain %d", luma, clipped * 100, new)
                self._gain = new
                with self._lock:
                    self._reset_window("collecting")
        except Exception as e:
            logger.debug("rPPG gain control: %s", e)
        finally:
            self._gain_busy = False

    # ── estimate ──────────────────────────────────────────────────────
    def _estimate(self, pts, diag, history=()):
        if len(pts) < 2:
            return _pulse_result(reason="collecting")
        t_end = pts[-1][0]
        pts = [p for p in pts if p[0] >= t_end - WINDOW_S]
        span = pts[-1][0] - pts[0][0]
        base = _pulse_result(window_s=round(span, 1))
        if span < MIN_WINDOW_S:
            return {**base, "reason": "collecting"}
        arr = np.array(pts, dtype=float)
        ts = arr[:, 0]
        hz = (len(pts) - 1) / span
        max_gap = float(np.diff(ts).max())
        base.update(fps=round(hz, 1), max_gap_s=round(max_gap, 3))
        if hz < max(self.sample_hz * PROVISIONAL_SAMPLE_FRACTION, 2.5 * BAND[1]) or max_gap > MAX_GAP_S:
            return {**base, "reason": "insufficient_samples"}
        low_sample_rate = hz < self.sample_hz * MIN_SAMPLE_FRACTION

        # Keep the same FFT grid per region. Only bridge short gaps within a
        # region with good coverage of the current capture window.
        region_bpms, region_quality, candidates, signals = [], {}, [], []
        for k, name in enumerate(PULSE_REGIONS):
            rgb = arr[:, 4 + 3 * k:7 + 3 * k]
            valid = np.isfinite(rgb).all(axis=1)
            coverage = float(valid.mean())
            quality = {"coverage": round(coverage, 2), "snr": None, "used": False}
            region_quality[name] = quality
            region_bpms.append(None)
            valid_ts = ts[valid]
            if (coverage < MIN_REGION_COVERAGE or len(valid_ts) < 2
                    or np.diff(valid_ts).max() > MAX_GAP_S
                    or valid_ts[0] - ts[0] > MAX_GAP_S / 2
                    or ts[-1] - valid_ts[-1] > MAX_GAP_S / 2):
                continue
            filled = rgb if valid.all() else np.column_stack([np.interp(ts, valid_ts, c[valid]) for c in rgb.T])
            filled = _remove_motion(filled, arr[:, 1:4])
            freqs, power, peak, snr = pos_spectrum(ts, filled, self.sample_hz)
            energy = power.sum()
            if energy <= 1e-12 or not np.isfinite(snr):
                continue
            region_bpms[k] = round(peak * 60)
            quality["snr"] = round(snr, 1)
            signals.append(filled)
            for frequency, strength, secondary in _regional_peaks(freqs, power, peak, snr):
                weight = coverage * (1 + np.clip(strength, 0, 8))
                candidates.append((name, frequency, weight, secondary, power / energy))

        base.update(regions_bpm=region_bpms, region_quality=region_quality)
        groups = [group for n in (2, 3) for group in combinations(candidates, n)
                  if len({item[0] for item in group}) == n
                  and np.ptp([item[1] for item in group]) * 60 <= REGION_AGREEMENT_BPM]
        method = "regional_consensus"
        if groups and not low_sample_rate:
            group = max(groups, key=lambda items: (len(items), sum(not item[3] for item in items),
                                                   sum(item[2] for item in items)))
            names, peaks, weights, secondary, spectra = zip(*group)
            for name, selected_peak in zip(names, peaks):
                region_quality[name]["used"] = True
                region_quality[name]["selected_bpm"] = round(selected_peak * 60)
            power = np.average(spectra, axis=0, weights=weights)
            centre = np.average(peaks, weights=weights)
            nearby = np.flatnonzero(np.abs(freqs - centre) * 60 <= REGION_AGREEMENT_BPM)
            peak = freqs[nearby[np.argmax(power[nearby])]]
            snr = _spectral_snr(freqs, power, peak)
        else:
            if low_sample_rate:
                base["reason"] = "low_sample_rate"
            else:
                enough_regions = len({item[0] for item in candidates}) >= 2
                base["reason"] = "regions_disagree" if enough_regions else "insufficient_regions"
            if not signals:
                return base
            # Retain an indicative reading from the pooled RGB, as in the
            # previous estimator, while keeping it out of trusted consumers.
            freqs, power, _, _ = pos_spectrum(ts, np.mean(signals, axis=0), self.sample_hz)
            peak = _tracked_peak(freqs, power, history)
            snr = _spectral_snr(freqs, power, peak)
            if power.sum() <= 1e-12 or not np.isfinite(snr):
                return base
            method = "provisional"
        bpm = round(peak * 60)
        cxs, cys, ds = arr[:, 1], arr[:, 2], arr[:, 3]
        motion = float(np.sqrt(cxs.var() + cys.var()) / max(ds.mean(), 1.0))
        trust_bpms = [quality.get("selected_bpm", previous)
                      for quality, previous in zip(region_quality.values(), region_bpms)]
        trust = trust_score(snr, trust_bpms, bpm, [*history, bpm], motion,
                            diag.get("brightness", 128), diag.get("clipped", 0.0))
        if method == "provisional":
            trust.update(score=min(trust["score"], 39), label="unreliable")
            trust["reasons"].append("low sample rate" if low_sample_rate else "no qualified regional consensus")
        elif any(secondary):
            trust["reasons"].append("shared secondary spectral peak")
            if trust["label"] == "reliable":
                trust.update(score=min(trust["score"], 69), label="uncertain")
        return {**base, "bpm": bpm, "snr": round(snr, 1), "motion": round(motion, 3),
                "trust": trust, "method": method}

    def _refresh(self):
        now = time.monotonic()
        if self._est_at is not None and now - self._est_at < ESTIMATE_EVERY_S:
            return
        if not self._est_lock.acquire(blocking=False):
            return                         # another thread is recomputing; serve the cache
        try:
            with self._lock:
                if self._face is None or not self._samples:
                    return
                if self._samples[-1][0] == self._est_sample_t:
                    return
                self._est_at = now
                generation, pts = self._gen, list(self._samples)
                if self._last_consensus_t is None or pts[-1][0] - self._last_consensus_t > MAX_TRACK_AGE_S:
                    self._history.clear()
                history = list(self._history)
                diag = dict(self._diag)
            try:
                result = self._estimate(pts, diag, history)
            except Exception as e:
                logger.warning("rPPG estimate failed: %s", e)
                result = _pulse_result(reason="estimation_failed")
            with self._lock:
                if generation != self._gen:
                    return
                if self._result["reason"] == "sampling_gap" and result["reason"] == "collecting":
                    result["reason"] = "sampling_gap"
                if result["method"] == "regional_consensus":
                    self._history.append(result["bpm"])
                    self._last_consensus_t = pts[-1][0]
                    self._provisional_bpms.clear()
                    self._provisional_bpms.append(result["bpm"])
                elif result["method"] == "provisional":
                    result["raw_bpm"] = result["bpm"]
                    self._provisional_bpms.append(result["raw_bpm"])
                    result["bpm"] = _display_provisional_bpm(self._provisional_bpms)
                else:
                    self._provisional_bpms.clear()
                self._est_sample_t = pts[-1][0]
                self._result = result
        finally:
            self._est_lock.release()

    # ── outputs ───────────────────────────────────────────────────────
    def status(self) -> dict:
        """Thread-safe; recomputes at most once per second (lazily), otherwise cached."""
        self._refresh()
        with self._lock:
            return {**self._result, "face": self._face is not None, "diag": dict(self._diag)}

    def rois(self) -> list[tuple[int, int, int, int]]:
        with self._lock:
            return list(self._rois)
