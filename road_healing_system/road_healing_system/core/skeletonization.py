"""
Stage 4a - Skeletonization.

Input : binary road mask (from segmentation)
Output: one-pixel-wide skeleton, distance map (for widths), and a traced list
        of branches between skeleton nodes (endpoints and intersections).

Why m-adjacency: on an 8-connected skeleton, a diagonal stair-step makes a
pixel look like it has three neighbours, which creates false intersections.
Mixed adjacency (diagonal neighbours count only when they share no
orthogonal neighbour) gives every chain pixel exactly two neighbours, so
endpoints (1 neighbour) and intersections (3+) are detected cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import SkeletonConfig

ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


# --------------------------------------------------------------------------- #
# Skeletonisation
# --------------------------------------------------------------------------- #
def _fill_small_holes(binary: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return binary
    inv = (~binary).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = binary.shape
    out = binary.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and area < max_area:
            out[labels == i] = True
    return out


def _zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Pure-NumPy Zhang-Suen thinning (fallback when scikit-image is unavailable)."""
    img = np.pad(binary.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b = sum(seq[:8])
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            core = img[1:-1, 1:-1]
            cond = (core == 1) & (b >= 2) & (b <= 6) & (a == 1)
            if step == 0:
                cond &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if cond.any():
                core[cond] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def skeletonize_mask(mask: np.ndarray, cfg: SkeletonConfig) -> tuple[np.ndarray, str]:
    """Return (skeleton bool array, method name)."""
    binary = mask > 0
    binary = _fill_small_holes(binary, cfg.fill_hole_area)
    if cfg.smooth_kernel and cfg.smooth_kernel >= 3:
        binary = cv2.medianBlur(binary.astype(np.uint8) * 255, int(cfg.smooth_kernel) | 1) > 0
    try:
        from skimage.morphology import skeletonize
        return skeletonize(binary).astype(bool), "scikit-image skeletonize"
    except ImportError:
        return _zhang_suen(binary), "NumPy Zhang-Suen thinning (scikit-image not installed)"


def distance_map(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from each road pixel to the nearest background pixel."""
    return cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)


# --------------------------------------------------------------------------- #
# m-adjacency
# --------------------------------------------------------------------------- #
def m_degree(skel: np.ndarray) -> np.ndarray:
    """Number of m-adjacent skeleton neighbours of every skeleton pixel (vectorised)."""
    s = skel.astype(bool)
    h, w = s.shape
    p = np.pad(s, 1)

    def shift(dr: int, dc: int) -> np.ndarray:
        return p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]

    deg = np.zeros((h, w), np.int32)
    for dr, dc in ORTH:
        deg += shift(dr, dc)
    for dr, dc in DIAG:
        deg += shift(dr, dc) & ~shift(dr, 0) & ~shift(0, dc)
    return np.where(s, deg, 0)


def m_neighbors(skel: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = skel.shape
    out = []
    for dr, dc in ORTH:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
            out.append((rr, cc))
    for dr, dc in DIAG:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and skel[rr, cc] and not skel[rr, c] and not skel[r, cc]:
            out.append((rr, cc))
    return out


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
@dataclass
class TraceResult:
    node_pixels: dict[int, list[tuple[int, int]]]   # node label -> pixels (r, c)
    branches: list[tuple[int, int, list[tuple[int, int]]]]  # (label_a, label_b, path r,c)
    degree: np.ndarray
    notes: list[str] = field(default_factory=list)


def trace_skeleton(skel: np.ndarray) -> TraceResult:
    """Split the skeleton into nodes (degree != 2) and the pixel chains between them."""
    skel = skel.astype(bool)
    deg = m_degree(skel)
    node_mask = skel & (deg != 2)
    n_lab, lab = cv2.connectedComponents(node_mask.astype(np.uint8), connectivity=8)
    lab = lab.astype(np.int32)

    node_pixels: dict[int, list[tuple[int, int]]] = {}
    rr, cc = np.nonzero(node_mask)
    for r, c in zip(rr.tolist(), cc.tolist()):
        node_pixels.setdefault(int(lab[r, c]), []).append((r, c))

    visited = np.zeros_like(skel)
    branches: list[tuple[int, int, list[tuple[int, int]]]] = []
    seen_direct: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def walk(start_label: int, start: tuple[int, int], first: tuple[int, int]) -> None:
        path = [start, first]
        visited[first] = True
        prev, cur = start, first
        while True:
            nbrs = [q for q in m_neighbors(skel, *cur) if q != prev]
            node_adj = [q for q in nbrs if lab[q]]
            if node_adj:
                # prefer a node other than the start node when both touch
                node_adj.sort(key=lambda q: lab[q] == start_label)
                q = node_adj[0]
                path.append(q)
                end_label = int(lab[q])
                if end_label == start_label and len(path) < 4:
                    return  # trivial bump on a node cluster
                branches.append((start_label, end_label, path))
                return
            nxt = [q for q in nbrs if not visited[q]]
            if not nxt:
                return
            q = nxt[0]
            visited[q] = True
            path.append(q)
            prev, cur = cur, q

    for label, pixels in node_pixels.items():
        for (r, c) in pixels:
            for q in m_neighbors(skel, r, c):
                lq = int(lab[q])
                if lq:
                    if lq != label:
                        key = (min((r, c), q), max((r, c), q))
                        if key not in seen_direct:
                            seen_direct.add(key)
                            branches.append((label, lq, [(r, c), q]))
                    continue
                if visited[q]:
                    continue
                walk(label, (r, c), q)

    # closed loops that contain no node at all
    next_label = n_lab
    leftover = skel & (deg == 2) & ~visited
    loops = 0
    for r, c in zip(*np.nonzero(leftover)):
        r, c = int(r), int(c)
        if visited[r, c]:
            continue
        lab[r, c] = next_label
        node_pixels[next_label] = [(r, c)]
        visited[r, c] = True
        nbrs = m_neighbors(skel, r, c)
        if nbrs:
            walk(next_label, (r, c), nbrs[0])
        next_label += 1
        loops += 1

    notes = [f"{loops} closed loop(s) without intersections were found."] if loops else []
    return TraceResult(node_pixels, branches, deg, notes)
