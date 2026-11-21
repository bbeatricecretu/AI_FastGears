"""Measure colour in facial polygons on the original, unannotated BGR frame."""
from dataclasses import dataclass

import cv2
import numpy as np


REGIONS = ("cheek_right", "cheek_left", "forehead", "lips")
CHANNELS = ("L", "a", "b", "C")
MIN_PIXELS = 35
MAX_SAMPLES = 4096


@dataclass(frozen=True)
class RegionReading:
    lab: np.ndarray | None
    mad: np.ndarray | None
    pixels: int = 0
    sampled: int = 0
    rejected: float = 0.0
    reason: str | None = None


def lab_values(pixels: np.ndarray) -> np.ndarray:
    """Decode uint8 or normalized float BGR as sRGB and return L*, a*, b*, C*."""
    bgr = pixels.astype(np.float32).reshape(-1, 1, 3)
    if pixels.dtype == np.uint8:
        bgr /= 255.0
    elif not np.issubdtype(pixels.dtype, np.floating) or not ((0 <= bgr) & (bgr <= 1)).all():
        raise ValueError("Expected uint8 BGR or normalized floating-point BGR")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.column_stack((lab, np.hypot(lab[:, 1], lab[:, 2])))


def compensate_light(color: np.ndarray, reference: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Undo shared exposure and white-balance gains measured by the reference."""
    samples = np.asarray([color[:3], reference[:3], baseline[:3]], dtype=np.float32)
    region, current, neutral = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_Lab2LRGB)[:, 0]
    gains = current / np.maximum(neutral, 1e-4)
    corrected = np.clip(region / np.maximum(gains, 1e-4), 0.0, 1.0)
    lab = cv2.cvtColor(corrected.reshape(1, 1, 3), cv2.COLOR_LRGB2Lab)[0, 0]
    return np.append(lab, np.hypot(lab[1], lab[2]))


def region_pixels(frame, outline, inner=None, *, inset=1):
    """Crop and rasterize a polygon, optionally subtracting the mouth opening."""
    raster = region_mask(frame.shape, outline, inner, inset=inset)
    if raster is None:
        return None
    mask, (x0, y0, x1, y1) = raster
    return frame[y0:y1, x0:x1][mask != 0]


def region_mask(frame_shape, outline, inner=None, *, inset=1):
    outer = np.asarray(outline, dtype=np.float32)
    hole = None if inner is None else np.asarray(inner, dtype=np.float32)
    h, w = frame_shape[:2]
    for points in (outer,) if hole is None else (outer, hole):
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= (w, h))):
            return None
    lo = np.floor(outer.min(axis=0)).astype(int)
    hi = np.minimum(np.ceil(outer.max(axis=0)).astype(int) + 1, (w, h))
    x0, y0 = lo
    x1, y1 = hi
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.rint(outer - lo).astype(np.int32)], 1)
    if hole is not None:
        cv2.fillPoly(mask, [np.rint(hole - lo).astype(np.int32)], 0)
    if inset:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1))
        mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask, (x0, y0, x1, y1)


def measure(pixels, *, lips=False, reference=False):
    if pixels is None:
        return RegionReading(None, None, reason="region_unavailable")
    count = len(pixels)
    if count < MIN_PIXELS:
        return RegionReading(None, None, count, count, reason="region_too_small")
    sampled = pixels[::max(1, (count + MAX_SAMPLES - 1) // MAX_SAMPLES)]
    lab = lab_values(sampled)
    light = lab[:, 0]
    usable = (light >= 9) & (light <= 93) & (sampled.max(axis=1) < 250)
    if not reference:
        median_light = np.median(light)
        # Trim strong shadows and diffuse/white highlights relative to this region.
        usable &= (light >= median_light - 19) & (light <= median_light + 23)
        if lips:
            usable &= ~((light > 67) & (lab[:, 3] < 13) & (light > median_light + 7))
    valid = lab[usable]
    rejected = 1.0 - len(valid) / len(sampled)
    if len(valid) < MIN_PIXELS or rejected > 0.5:
        return RegionReading(None, None, count, len(sampled), rejected, "exposure_or_occlusion")
    center = np.median(valid, axis=0)
    spread = np.median(np.abs(valid - center), axis=0)
    if spread[0] > 13 or max(spread[1:3]) > 13:
        return RegionReading(None, None, count, len(sampled), rejected, "region_unstable")
    return RegionReading(center, spread, count, len(sampled), rejected)
