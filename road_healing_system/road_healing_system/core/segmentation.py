"""
Stage 3 - Road segmentation (classical computer vision; no trained model).

Input : preprocessed BGR image
Output: binary road mask (road = 255, background = 0) plus diagnostics.

Methods
-------
contextual (default) - combines several cues instead of one colour threshold:
    1. exclusion cues: vegetation-like (excess green), water-like (blue
       dominant) and shadow-like (very dark) pixels are never road;
    2. a loose colour gate (greyish, mid-bright) gives candidate pixels;
    3. regions wider than ``max_road_width`` are removed with a morphological
       opening (top-hat): open ground, yards, fields and car parks are wide,
       roads are not;
    4. a road colour model (Lab mean/covariance) is learned from strict seed
       pixels of this image, giving a per-pixel road likelihood;
    5. a Sato ridge filter on that likelihood measures line-likeness, so long
       thin structures score higher than patches;
    6. after thresholding the combined score, each connected component is
       checked: compact shapes (length/width below ``min_elongation``, e.g.
       roofs) and over-wide components are dropped.
hsv / adaptive / otsu - the earlier single-threshold methods, kept for comparison.

Limitations (stated honestly): no trained road-segmentation model is included.
The contextual method still relies on hand-set cues: long grey roofs, concrete
canals, grey riverbanks and dirt tracks can be misjudged, and roads under
dense canopy or deep shadow are missed (that is what gap reconstruction is
for). A road mask from a better source can be uploaded instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from config import SegmentationConfig


@dataclass
class SegmentationResult:
    mask: np.ndarray                      # uint8 {0,255}
    raw_mask: np.ndarray                  # before morphology / filtering
    method: str
    removed_small: int = 0
    removed_blobs: int = 0
    notes: list[str] = field(default_factory=list)
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        fg = float((self.mask > 0).mean())
        return {"method": self.method, "road_fraction": round(fg, 4),
                "removed_small_components": self.removed_small,
                "removed_compact_blobs": self.removed_blobs, "notes": self.notes, **self.stats}


def _threshold(image: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    if cfg.method == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        return (s <= cfg.sat_max) & (v >= cfg.val_min) & (v <= cfg.val_max)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cfg.method == "otsu":
        flag = cv2.THRESH_BINARY_INV if cfg.otsu_dark_roads else cv2.THRESH_BINARY
        _, th = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
        return th > 0
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                               int(cfg.adaptive_block), int(cfg.adaptive_offset))
    return th > 0


def clean_mask(binary: np.ndarray, cfg: SegmentationConfig) -> tuple[np.ndarray, int, int]:
    """Morphological clean-up and removal of tiny or roof-like components."""
    m = binary.astype(np.uint8) * 255
    if cfg.open_kernel >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if cfg.close_kernel >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    removed_small = removed_blobs = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < cfg.min_component_area:
            removed_small += 1
            continue
        if cfg.remove_compact_blobs:
            ys, xs = np.nonzero(labels == i)
            (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
            extent = area / max(rw * rh, 1.0)
            if long_side / short_side < cfg.blob_max_aspect and extent > cfg.blob_min_extent:
                removed_blobs += 1
                continue
        keep[i] = True
    out = np.where(keep[labels], 255, 0).astype(np.uint8)
    return out, removed_small, removed_blobs


# --------------------------------------------------------------------------- #
# Contextual method
# --------------------------------------------------------------------------- #
def exclusion_cues(image: np.ndarray, cfg: SegmentationConfig) -> dict[str, np.ndarray]:
    bgr = image.astype(np.float32)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    exg = (2 * g - r - b) / (b + g + r + 1e-6)
    v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
    return {"vegetation": exg > cfg.vegetation_exg,
            "water": ((b - np.maximum(r, g)) > cfg.water_blue_margin) & (b > 50),
            "shadow": v < cfg.shadow_val_max}


def _remove_wide(binary: np.ndarray, max_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a mask into the part narrower than max_width and the wide regions."""
    radius = max_width // 2 + 2
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    wide = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, disk) > 0
    wide = cv2.dilate(wide.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    return binary & ~wide, wide & binary


def _shape_filter(mask: np.ndarray, cfg: SegmentationConfig) -> tuple[np.ndarray, int]:
    """Drop components that are compact (roofs) or wider than a road."""
    from skimage.morphology import skeletonize
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask, 0
    skel = skeletonize(mask > 0)
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    length = np.bincount(labels[skel], minlength=n).astype(float)
    dsum = np.bincount(labels[skel], weights=dist[skel], minlength=n)
    keep = np.zeros(n, bool)
    removed = 0
    for i in range(1, n):
        if length[i] == 0:
            removed += 1
            continue
        width = max(2.0 * dsum[i] / length[i] - 1.0, 1.0)          # mean width along the centre line
        aspect = (length[i] + width) / width                         # ~ length / width of the shape
        if aspect < cfg.min_elongation or width > cfg.max_road_width:
            removed += 1
            continue
        keep[i] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8), removed


def segment_contextual(image: np.ndarray, cfg: SegmentationConfig) -> SegmentationResult:
    from skimage.filters import sato
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    cues = exclusion_cues(image, cfg)
    excluded = cues["vegetation"] | cues["water"] | cues["shadow"]
    candidate = (s <= cfg.candidate_sat_max) & (v >= cfg.candidate_val_min) & (v <= cfg.candidate_val_max) & ~excluded
    narrow, wide = _remove_wide(candidate, cfg.max_road_width)

    seeds = narrow & (s <= cfg.seed_sat_max)
    notes = []
    if seeds.sum() < 50:
        seeds = narrow if narrow.sum() >= 50 else candidate
        notes.append("Few strict road-colour seeds were found; the colour model uses looser seeds.")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    if seeds.sum() >= 10:
        samples = lab[seeds]
        if len(samples) > 20000:
            samples = samples[np.random.default_rng(0).choice(len(samples), 20000, replace=False)]
        mu = samples.mean(axis=0)
        cov = np.cov(samples.T) + np.eye(3) * 25.0
        diff = lab - mu
        d2 = np.einsum("...i,ij,...j->...", diff, np.linalg.inv(cov), diff)
        prob = np.exp(-d2 / (2.0 * cfg.likelihood_scale)).astype(np.float32)
    else:
        prob = candidate.astype(np.float32)
        notes.append("No road-coloured pixels were found to learn a colour model.")
    prob[excluded] = 0.0

    ridge = sato(prob, sigmas=[float(x) for x in cfg.ridge_sigmas], black_ridges=False)
    positive = ridge[ridge > 0]
    scale = float(np.percentile(positive, 99)) if positive.size else 1.0
    ridge = np.clip(ridge / max(scale, 1e-6), 0, 1).astype(np.float32)
    score = prob * ((1 - cfg.ridge_weight) + cfg.ridge_weight * ridge)

    allowed = candidate & ~wide                    # never road: excluded cues and wide regions
    binary = (score > cfg.score_threshold) & allowed
    binary, wide2 = _remove_wide(binary, cfg.max_road_width)
    cleaned, removed_small, _ = clean_mask(binary, cfg)
    mask, removed_shape = _shape_filter(cleaned, cfg)
    # line-likeness is weakest at road borders: regrow up to 3 px into allowed road-coloured pixels
    if mask.any():
        grown = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3) > 0
        mask = np.where((mask > 0) | (grown & allowed & ~wide2), 255, 0).astype(np.uint8)

    frac = float((mask > 0).mean())
    if frac == 0:
        notes.append("No road was found. Try raising the maximum road saturation or the widest road, "
                     "or upload a road mask.")
    to8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)  # noqa: E731
    removed = ((wide | wide2) | ((cleaned > 0) & (mask == 0))).astype(np.uint8) * 255
    return SegmentationResult(
        mask, binary.astype(np.uint8) * 255, "contextual", removed_small, removed_shape, notes,
        intermediates={"seg_candidates": candidate.astype(np.uint8) * 255, "seg_likelihood": to8(prob),
                       "seg_line_likeness": to8(ridge), "seg_removed": removed},
        stats={"excluded_vegetation_pct": round(100 * float(cues["vegetation"].mean()), 1),
               "excluded_water_pct": round(100 * float(cues["water"].mean()), 1),
               "excluded_shadow_pct": round(100 * float(cues["shadow"].mean()), 1),
               "removed_wide_pct": round(100 * float((wide | wide2).mean()), 2),
               "removed_compact_components": removed_shape})


def segment_roads(image: np.ndarray, cfg: SegmentationConfig) -> SegmentationResult:
    if cfg.method == "contextual":
        return segment_contextual(image, cfg)
    raw = _threshold(image, cfg)
    mask, removed_small, removed_blobs = clean_mask(raw, cfg)
    notes = []
    frac = float((mask > 0).mean())
    if frac == 0:
        notes.append("No road pixels were found. Adjust the thresholds or upload a road mask.")
    elif frac > 0.5:
        notes.append(f"{frac:.0%} of the image was labelled road, which is unusually high. "
                     "Check the thresholds; the background may be leaking in.")
    return SegmentationResult(mask, raw.astype(np.uint8) * 255, cfg.method,
                              removed_small, removed_blobs, notes)
