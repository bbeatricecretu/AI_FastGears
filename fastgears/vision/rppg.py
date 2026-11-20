"""Remote photoplethysmography (rPPG): heart rate from one webcam.

Ported from afferent-junction2026 jetson/vision/rppg.py (validated there against a
finger pulse sensor: mean absolute difference 1.8 BPM). The algorithmic core is kept
verbatim: POS (Wang et al. 2017, "Algorithmic Principles of Remote PPG") over the last
10 s of forehead + cheek skin colour, dominant frequency in 45-180 BPM, plus the
per-region trust score. Dropped from the original: YOLO-pose face locator, second
camera / own capture, multi-camera fusion, GStreamer/JPEG feed path.

Each heartbeat changes skin colour very slightly (mostly green). The caller supplies the
face position from MediaPipe Face Landmarker as (cx, cy, d): midpoint between the eyes in
frame pixels and the inter-eye distance in pixels; this module averages the skin pixels
of three ROIs relative to that point at SAMPLE_HZ.

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

import cv2
import numpy as np

logger = logging.getLogger("rppg")

SAMPLE_HZ = 20.0
WINDOW_S = 10.0
MIN_WINDOW_S = 6.0
BAND = (0.75, 3.0)          # 45-180 BPM
FACE_LOST_S = 2.0           # drop the face after this long without a detection
FACE_EMA = 0.3              # source smoothed at each 0.5 s detection with this weight
FACE_EMA_PERIOD_S = 0.5
ESTIMATE_EVERY_S = 1.0

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


# ── algorithmic core (verbatim from the validated source) ─────────────
def pos_spectrum(ts, rgb, sample_hz=SAMPLE_HZ):
    """POS on unevenly sampled RGB means -> (band freqs, band power, f0 Hz, SNR dB)."""
    grid = np.arange(ts[0], ts[-1], 1.0 / sample_hz)            # uniform resampling
    c = np.stack([np.interp(grid, ts, rgb[:, k]) for k in range(3)], axis=1)
    n, L = len(c), int(1.6 * sample_hz)
    h = np.zeros(n)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    for i in range(n - L + 1):                                   # per 1.6 s window
        cn = c[i:i + L] / (c[i:i + L].mean(axis=0) + 1e-9)       # temporal normalisation
        s = P @ cn.T                                             # plane orthogonal to skin tone
        hh = s[0] + (s[0].std() / (s[1].std() + 1e-9)) * s[1]
        h[i:i + L] += hh - hh.mean()
    k = int(sample_hz)                                           # detrend: 1 s moving average
    h = h - np.convolve(h, np.ones(k) / k, mode="same")
    h = (h - h.mean()) * np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    spec = np.abs(np.fft.rfft(h, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_hz)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    fb, sb = freqs[band], spec[band]
    f0 = fb[int(np.argmax(sb))]
    sig = (np.abs(fb - f0) <= 0.1) | (np.abs(fb - 2 * f0) <= 0.1)
    snr = 10 * np.log10(sb[sig].sum() / (sb[~sig].sum() + 1e-12))
    return fb, sb, float(f0), float(snr)


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
        self._face = None            # (cx, cy, d) smoothed
        self._face_seen = None       # t of last detection
        self._face_t = None          # t of last smoothing step
        self._next_t = None          # next sample time (caller's clock)
        self._rois: list[tuple[int, int, int, int]] = []
        self._diag: dict = {}
        self._history: deque = deque(maxlen=8)
        self._result = {"bpm": None, "snr": None, "window_s": 0.0, "fps": None, "regions_bpm": [],
                        "motion": None, "trust": None}
        self._est_at = None          # wall-clock (monotonic) of last recompute
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
    def update(self, frame_bgr: np.ndarray, face: tuple[float, float, float] | None, t: float) -> None:
        # face tracking runs every frame (cheap); sampling is throttled to sample_hz
        with self._lock:
            if face is not None and face[2] >= 1:
                face = tuple(float(v) for v in face)
                if self._face is not None and self._face_t is not None:
                    # smooth so ROI jitter doesn't modulate the signal: the source applied
                    # 0.3 EMA per 0.5 s detection; same time constant at any frame rate
                    a = 1.0 - (1.0 - FACE_EMA) ** (max(t - self._face_t, 0.0) / FACE_EMA_PERIOD_S)
                    face = tuple(p + a * (f - p) for p, f in zip(self._face, face))
                self._face, self._face_seen, self._face_t = face, t, t
            elif self._face is not None and (self._face_seen is None or t - self._face_seen > FACE_LOST_S):
                self._face = None
                self._face_t = None
            if self._next_t is not None and t < self._next_t:
                return
            period = 1.0 / self.sample_hz
            self._next_t = t + period if self._next_t is None else max(self._next_t + period, t)
            if self._face is None or frame_bgr is None:
                self._rois = []
                return
            cx, cy, d = self._face
        sample = self._sample(frame_bgr, cx, cy, d, t)
        if sample is not None and self._v4l2:
            self._maybe_gain(sample[1], sample[2], t)

    def _sample(self, frame, cx, cy, d, t):
        H, W = frame.shape[:2]
        rois, rects = [], []
        for x0, y0, x1, y1 in BOXES:
            a, b = int(max(0, cx + x0 * d)), int(min(W, cx + x1 * d))
            c, e = int(max(0, cy + y0 * d)), int(min(H, cy + y1 * d))
            if b > a and e > c:
                rois.append(frame[c:e, a:b]); rects.append((a, c, b, e))
        if not rois:
            with self._lock:
                self._rois = []
            return None
        pix = np.concatenate([r.reshape(-1, 3) for r in rois], axis=0)
        ycc = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        # Skin filter that works under any light colour (a fixed YCrCb skin box rejected
        # ~100 % of the face under purple stage light): drop the darkest 15 % (hair, brows,
        # shadows) and brightest 5 % (specular highlights, clipped pixels) by luma.
        y = ycc[:, 0].astype(np.float32)
        lo, hi = np.percentile(y, [15, 95])
        skin = (y >= lo) & (y <= hi) & (pix.max(axis=1) < 250)
        n_skin = int(skin.sum())
        luma = float(ycc[:, 0].mean())
        clipped = float((pix.max(axis=1) >= 250).mean())
        allpix = pix[skin] if n_skin > 50 else pix
        b, g, r = allpix.mean(axis=0)
        # the same filter per region: forehead / cheek / cheek are three independent
        # measurements of one pulse, which is what the trust score checks
        regions = []
        for roi in rois:
            rp = roi.reshape(-1, 3)
            ry = cv2.cvtColor(rp.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)[:, 0]
            m = (ry >= lo) & (ry <= hi) & (rp.max(axis=1) < 250)
            rb, rg, rr = (rp[m] if m.sum() > 30 else rp).mean(axis=0)
            regions += [rr, rg, rb]
        regions += [np.nan] * (9 - len(regions))
        with self._lock:
            self._rois = rects
            self._samples.append((t, r, g, b, cx, cy, d, *regions))
            self._diag = {"skin_px": n_skin, "roi_px": int(len(skin)), "brightness": round(luma),
                          "clipped": round(clipped, 3), "eye_dist_px": round(d), "frame": f"{W}x{H}",
                          "gain": self._gain}
        return t, luma, clipped

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
                    self._samples.clear()
                    self._history.clear()
        except Exception as e:
            logger.debug("rPPG gain control: %s", e)
        finally:
            self._gain_busy = False

    # ── estimate ──────────────────────────────────────────────────────
    def _estimate(self, pts, diag):
        if len(pts) < 2:
            return {"bpm": None, "snr": None, "window_s": 0.0}
        t_end = pts[-1][0]
        pts = [p for p in pts if p[0] >= t_end - WINDOW_S]
        span = pts[-1][0] - pts[0][0]
        if span < MIN_WINDOW_S:
            return {"bpm": None, "snr": None, "window_s": round(span, 1)}
        arr = np.array(pts, dtype=float)
        ts = arr[:, 0]
        fb, sb, f0, snr = pos_spectrum(ts, arr[:, 1:4], self.sample_hz)
        bpm = round(f0 * 60)
        region_bpms = []
        for k in range(3):
            rgb = arr[:, 7 + 3 * k:10 + 3 * k]
            if np.isnan(rgb).any():
                region_bpms.append(None)
                continue
            region_bpms.append(round(pos_spectrum(ts, rgb, self.sample_hz)[2] * 60))
        cxs, cys, ds = arr[:, 4], arr[:, 5], arr[:, 6]
        motion = float(np.sqrt(cxs.var() + cys.var()) / max(ds.mean(), 1.0))
        with self._lock:
            self._history.append(bpm)
            history = list(self._history)
        trust = trust_score(snr, region_bpms, bpm, history, motion,
                            diag.get("brightness", 128), diag.get("clipped", 0.0))
        return {"bpm": bpm, "snr": round(snr, 1), "window_s": round(span, 1),
                "fps": round(len(pts) / span, 1), "regions_bpm": region_bpms,
                "motion": round(motion, 3), "trust": trust}

    def _refresh(self):
        now = time.monotonic()
        if self._est_at is not None and now - self._est_at < ESTIMATE_EVERY_S:
            return
        if not self._est_lock.acquire(blocking=False):
            return                         # another thread is recomputing; serve the cache
        try:
            self._est_at = now
            with self._lock:
                pts = list(self._samples)
                face = self._face is not None
                diag = dict(self._diag)
            res = None
            if face:
                try:
                    res = self._estimate(pts, diag)
                except Exception as e:
                    logger.warning("rPPG estimate failed: %s", e)
            base = {"bpm": None, "snr": None, "window_s": 0.0, "fps": None, "regions_bpm": [],
                    "motion": None, "trust": None}
            with self._lock:
                self._result = {**base, **(res or {})}
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
