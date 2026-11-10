"""
Small geometry toolkit. Points and directions are (x, y) in image pixels
(x = column, y = row, y pointing down). Pixel arrays from NumPy are (row, col)
and are converted explicitly where needed.
"""

from __future__ import annotations

import math

import numpy as np


def unit(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=float)
    n = float(np.hypot(v[0], v[1]))
    return None if n < 1e-9 else v / n


def angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned angle in degrees, 0..180."""
    cos = float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def signed_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Signed angle from u to v in radians, in (-pi, pi]."""
    return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def pca_direction(points_xy: np.ndarray, outward_hint: np.ndarray) -> np.ndarray | None:
    """
    Principal direction of a short point sequence, oriented to agree with
    ``outward_hint``. Robust to the one-pixel stair-steps of a skeleton.
    """
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 2:
        return None
    if len(pts) < 4:
        return unit(outward_hint)
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    if np.dot(axis, outward_hint) < 0:
        axis = -axis
    return unit(axis)


def polyline_length(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


def hermite_curve(p1: np.ndarray, t1: np.ndarray, p2: np.ndarray, t2: np.ndarray,
                  n: int, tangent_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Cubic Hermite curve from p1 (leaving along t1) to p2 (arriving along t2).
    It is the smooth "expected road" between two fragment ends: it honours both
    end directions, so it bends for curved roads and is straight for straight ones.
    Returns (points (n,2), unit tangents (n,2)).
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    d = float(np.hypot(*(p2 - p1)))
    m1 = np.asarray(t1, float) * d * tangent_scale
    m2 = np.asarray(t2, float) * d * tangent_scale
    t = np.linspace(0.0, 1.0, max(n, 2))[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    pts = h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2
    d00 = 6 * t**2 - 6 * t
    d10 = 3 * t**2 - 4 * t + 1
    d01 = -6 * t**2 + 6 * t
    d11 = 3 * t**2 - 2 * t
    tan = d00 * p1 + d10 * m1 + d01 * p2 + d11 * m2
    norms = np.hypot(tan[:, 0], tan[:, 1])[:, None]
    fallback = (p2 - p1) / max(d, 1e-9)
    tan = np.where(norms > 1e-9, tan / np.maximum(norms, 1e-9), fallback)
    return pts, tan


def reference_curve(p1: np.ndarray, dir1: np.ndarray | None, p2: np.ndarray,
                    arrive_dir: np.ndarray | None, tangent_scale: float = 1.0,
                    samples_per_px: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Hermite curve with missing directions replaced by the straight chord direction."""
    chord = unit(np.asarray(p2, float) - np.asarray(p1, float))
    if chord is None:
        pts = np.array([p1, p2], dtype=float)
        return pts, np.zeros_like(pts)
    t1 = dir1 if dir1 is not None else chord
    t2 = arrive_dir if arrive_dir is not None else chord
    d = float(np.hypot(*(np.asarray(p2, float) - np.asarray(p1, float))))
    n = int(max(50, samples_per_px * d))
    return hermite_curve(p1, t1, p2, t2, n, tangent_scale)


def smooth_polyline(points_xy: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing that keeps both ends fixed."""
    pts = np.asarray(points_xy, dtype=float)
    if window < 3 or len(pts) <= window:
        return pts
    k = window // 2
    padded = np.vstack([np.repeat(pts[:1], k, 0), pts, np.repeat(pts[-1:], k, 0)])
    kernel = np.ones(window) / window
    out = np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)])
    out[0], out[-1] = pts[0], pts[-1]
    return out


def hermite_with_curvature(p1: np.ndarray, t1: np.ndarray, p2: np.ndarray, t2: np.ndarray,
                           n: int, tangent_scale: float = 1.0) -> dict[str, np.ndarray]:
    """Hermite curve plus analytic derivatives.
    Returns points, unit tangents, signed curvature (1/px, positive = turning with a
    positive cross product) and the arc-length element ds per sample step."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    d = float(np.hypot(*(p2 - p1)))
    m1 = np.asarray(t1, float) * d * tangent_scale
    m2 = np.asarray(t2, float) * d * tangent_scale
    t = np.linspace(0.0, 1.0, max(n, 3))[:, None]
    pts = ((2 * t**3 - 3 * t**2 + 1) * p1 + (t**3 - 2 * t**2 + t) * m1
           + (-2 * t**3 + 3 * t**2) * p2 + (t**3 - t**2) * m2)
    d1 = ((6 * t**2 - 6 * t) * p1 + (3 * t**2 - 4 * t + 1) * m1
          + (-6 * t**2 + 6 * t) * p2 + (3 * t**2 - 2 * t) * m2)
    d2 = (12 * t - 6) * p1 + (6 * t - 4) * m1 + (-12 * t + 6) * p2 + (6 * t - 2) * m2
    speed = np.hypot(d1[:, 0], d1[:, 1])
    safe = np.maximum(speed, 1e-9)
    curvature = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / safe**3
    curvature = np.where(speed > 1e-9, curvature, np.inf)   # a cusp is infinitely sharp
    tangents = d1 / safe[:, None]
    ds = speed / (len(t) - 1)
    return {"points": pts, "tangents": tangents, "curvature": curvature, "ds": ds}


def curve_shape_metrics(curve: dict[str, np.ndarray], chord: float) -> dict[str, float]:
    """Length, tightest turn radius and normalised bending energy of a sampled curve."""
    k = curve["curvature"]
    finite = np.isfinite(k)
    kmax = float(np.max(np.abs(k[finite]))) if finite.any() else np.inf
    if not finite.all():
        kmax = np.inf
    length = polyline_length(curve["points"])
    bending = float(np.sum((k[finite] ** 2) * curve["ds"][finite])) * max(chord, 1e-9)
    return {"length": length, "max_curvature": kmax,
            "min_radius": np.inf if kmax == 0 else (0.0 if not np.isfinite(kmax) else 1.0 / kmax),
            "bending_energy": bending if np.isfinite(kmax) else np.inf}


def offset_polygon(points: np.ndarray, tangents: np.ndarray, half_widths: np.ndarray) -> np.ndarray:
    """Closed outline of a band of varying width around a centre line."""
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    left = points + normals * half_widths[:, None]
    right = points - normals * half_widths[:, None]
    return np.vstack([left, right[::-1]])
