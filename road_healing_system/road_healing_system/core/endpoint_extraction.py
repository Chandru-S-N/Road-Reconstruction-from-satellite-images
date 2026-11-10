"""
Stage 4b - Endpoint and intersection extraction, and road attributes.

* ``detect_keypoints``: pixel-level endpoints (1 skeleton neighbour) and
  intersections (3+ neighbours) on the raw skeleton, using m-adjacency.
* ``extract_endpoints``: final endpoint table on the pruned road graph, with
  outward direction, width, curvature, component and border flag.

Within about half a road width of the mask boundary the medial axis curls
toward the corners of the cut (a tree canopy cuts a road in an arc), so the
direction is measured beyond that zone and the tip is projected back onto the
reliable centre line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import networkx as nx
import numpy as np

from config import SkeletonConfig
from core.geometry import pca_direction, signed_angle, unit
from core.skeletonization import m_degree


@dataclass
class Endpoint:
    id: int
    xy: tuple[float, float]                 # tip corrected onto the centre line (used for gaps)
    rc: tuple[int, int]                     # raw skeleton tip pixel (graph node)
    direction: tuple[float, float] | None   # outward unit vector (x, y)
    width: float | None
    curvature: float | None                 # radians per pixel, signed
    component: int
    at_border: bool
    branch_length: float

    def as_row(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.id, "x": round(self.xy[0], 1), "y": round(self.xy[1], 1),
            "dir_x": None if self.direction is None else round(self.direction[0], 3),
            "dir_y": None if self.direction is None else round(self.direction[1], 3),
            "width_px": None if self.width is None else round(self.width, 2),
            "curvature_rad_per_px": None if self.curvature is None else round(self.curvature, 4),
            "component": self.component, "at_border": self.at_border,
            "branch_length_px": round(self.branch_length, 1),
        }


def detect_keypoints(skel: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    """Raw endpoints and intersection centres on the unpruned skeleton, as (x, y)."""
    deg = m_degree(skel)
    ends = np.argwhere(skel & (deg == 1))
    junction = (skel & (deg >= 3)).astype(np.uint8)
    n, _, _, cents = cv2.connectedComponentsWithStats(junction, connectivity=8)
    return {"endpoints": [(int(c), int(r)) for r, c in ends],
            "intersections": [(int(round(x)), int(round(y))) for x, y in cents[1:n]]}


def path_xy(path_rc: np.ndarray) -> np.ndarray:
    return np.asarray(path_rc, dtype=float)[:, ::-1]


def tip_skip(path_rc: np.ndarray, width: float | None) -> int:
    """Pixels to skip at a skeleton tip. Within about half a road width of the mask
    boundary the medial axis curls toward the corners of the cut (a round or slanted
    cut bends it sideways), so direction and position are measured beyond that zone."""
    if width is None:
        return 0
    return int(np.clip(round(0.5 * width), 2, max(len(path_rc) // 3, 0)))


def endpoint_direction(path_rc: np.ndarray, window: int, skip: int = 0) -> np.ndarray | None:
    """Outward unit direction at the tip, from `window` skeleton pixels after `skip`."""
    seg = path_rc[skip:skip + window]
    if len(seg) < 3:
        seg = path_rc[:window]
    if len(seg) < 3:
        return None
    pts = path_xy(seg)
    return pca_direction(pts, pts[0] - pts[-1])


def corrected_tip(path_rc: np.ndarray, direction: np.ndarray | None, skip: int) -> np.ndarray:
    """Tip position projected onto the reliable centre line: removes the sideways drift
    of a curled skeleton tip while keeping how far the road actually extends."""
    tip = path_xy(path_rc[:1])[0]
    if direction is None or skip <= 0 or skip >= len(path_rc):
        return tip
    q = path_xy(path_rc[skip:skip + 1])[0]
    return q + direction * max(float(np.dot(tip - q, direction)), 0.0)


def endpoint_width(path_rc: np.ndarray, dist: np.ndarray, window: int) -> float | None:
    """Road width near an endpoint: 2 x distance-to-edge - 1 (pixel count across the road),
    75th percentile over the window (the skeleton tip tapers, so low values underestimate)."""
    k = min(window, len(path_rc))
    if k < 3:
        return None
    rc = np.asarray(path_rc[1:k])
    vals = dist[rc[:, 0], rc[:, 1]]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return None
    return float(max(2.0 * np.percentile(vals, 75) - 1.0, 1.0))


def endpoint_curvature(path_rc: np.ndarray, window: int, skip: int = 0) -> float | None:
    """Signed turning rate (radians / pixel) along `window` pixels after `skip`; None if too short."""
    seg = path_rc[skip:skip + window]
    k = len(seg)
    if k < 8:
        return None
    pts = path_xy(seg)
    mid = k // 2
    near = unit(pts[0] - pts[mid])
    far = unit(pts[mid] - pts[-1])
    if near is None or far is None:
        return None
    return float(signed_angle(far, near) / max(k / 2.0, 1.0))


def _edge_path_from(G: nx.Graph, a: int, b: int) -> np.ndarray:
    data = G.edges[a, b]
    return data["path"] if data["start"] == a else data["path"][::-1]


def extract_endpoints(G: nx.Graph, dist: np.ndarray, shape: tuple[int, int],
                      component_of: dict[int, int], cfg: SkeletonConfig) -> dict[int, Endpoint]:
    h, w = shape
    out: dict[int, Endpoint] = {}
    for n, d in G.degree():
        if d != 1:
            continue
        m = next(iter(G.neighbors(n)))
        path = _edge_path_from(G, n, m)
        r, c = G.nodes[n]["rc"]
        width = endpoint_width(path, dist, cfg.direction_window)
        skip = tip_skip(path, width)
        direction = endpoint_direction(path, cfg.direction_window, skip)
        tip = corrected_tip(path, direction, skip)
        tip = (float(np.clip(tip[0], 0, w - 1)), float(np.clip(tip[1], 0, h - 1)))
        # a road leaving the image ends its skeleton about half a width inside the border
        edge_gap = min(r, c, h - 1 - r, w - 1 - c)
        at_border = edge_gap <= cfg.border_margin + 0.75 * (width or 0.0)
        out[n] = Endpoint(
            id=int(n), xy=tip, rc=(int(r), int(c)),
            direction=None if direction is None else (float(direction[0]), float(direction[1])),
            width=width, curvature=endpoint_curvature(path, cfg.direction_window, skip),
            component=component_of.get(n, -1), at_border=bool(at_border),
            branch_length=float(G.edges[n, m]["length"]),
        )
    return out
