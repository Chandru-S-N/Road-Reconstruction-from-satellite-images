"""
Graph-based road reconstruction for one approved gap - a trellis graph solved by
dynamic programming. No Dijkstra, A* or any priority-queue shortest-path search.

1. Expected course: a cubic Hermite curve leaving fragment A along its road
   direction and arriving along fragment B's road direction.
2. Stations: the course is resampled every ``station_spacing`` px. At each station
   the road may lie at one of M lateral positions (``offset_step`` apart, up to
   ``max_offset_widths`` x road width either side). The first and last stations
   are pinned to the two road ends.
3. Trellis graph (a NetworkX DiGraph, acyclic by construction): node = (station,
   lateral position); edges join consecutive stations whose lateral positions differ
   by at most ``max_lateral_step`` px, so the route stays continuous.
4. Costs (all >= 0):
       node:  lambda_road * (1 - road_likelihood) + lambda_guide * min((offset/s)^2, cap)
              + other_road_penalty * [on an unrelated road]
       edge:  step * (1 + mean node cost) + step * lambda_direction * (1 - cos(step heading - course tangent)) / 2
5. Dynamic programming: nodes are processed in topological order (station by
   station) and each edge is relaxed once. On a directed acyclic graph this gives the
   exact minimum-cost route in O(nodes + edges) time.
6. Validation (done by the caller with the result): length ratio, path cost threshold,
   overlap with unrelated roads.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from config import PathConfig
from core.context_features import CATEGORY_KEYS
from core.geometry import polyline_length, smooth_polyline

TERMS = ("length", "non_road", "guide", "direction", "other_road")


@dataclass
class PathResult:
    found: bool
    valid: bool
    reason_code: str                      # ok | no_path | too_long | too_costly | other_road | degenerate
    reason: str
    path_xy: np.ndarray | None = None     # smoothed centre line
    raw_xy: np.ndarray | None = None
    guide_xy: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=dict)
    reference: dict[str, Any] | None = None   # the expected course itself, priced with the same costs
    cover: dict[str, float] = field(default_factory=dict)


def _stations(curve: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.hypot(*np.diff(curve, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    k = max(2, int(round(total / spacing)) + 1)
    s = np.linspace(0.0, total, k)
    pts = np.column_stack([np.interp(s, cum, curve[:, 0]), np.interp(s, cum, curve[:, 1])])
    tan = np.gradient(pts, axis=0)
    tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
    return pts, tan


def find_path(geo: dict[str, Any], labels: np.ndarray, own_labels: set[int], likelihood: np.ndarray | None,
              category: np.ndarray | None, cfg: PathConfig) -> PathResult:
    t0 = time.perf_counter()
    p1, p2, width = geo["p1"], geo["p2"], geo["width"]
    gap = float(np.hypot(*(p2 - p1)))
    h, w = labels.shape
    if gap < 2.0:
        pts = np.array([p1, p2])
        return PathResult(True, True, "ok", "the ends almost touch; a direct join was used", pts, pts, pts,
                          {"length": round(gap, 2), "gap": round(gap, 2), "total_cost": round(gap, 2), "mean_cost": 1.0,
                           "length_ratio": 1.0, "stations": 2, "positions": 1, "nodes": 2, "edges": 1})

    P, T = _stations(geo["curve"], cfg.station_spacing)
    P[0], P[-1] = p1, p2
    N = np.column_stack([-T[:, 1], T[:, 0]])
    reach = max(6.0, cfg.max_offset_widths * width)
    half = int(math.floor(reach / cfg.offset_step))
    offsets = np.arange(-half, half + 1) * cfg.offset_step
    K, M = len(P), len(offsets)
    zero = half
    X = P[:, None, :] + N[:, None, :] * offsets[None, :, None]                 # (K, M, 2)
    ix, iy = np.round(X[..., 0]).astype(int), np.round(X[..., 1]).astype(int)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ixc, iyc = np.clip(ix, 0, w - 1), np.clip(iy, 0, h - 1)
    lab = labels[iyc, ixc]
    own = np.isin(lab, list(own_labels)) if own_labels else np.zeros_like(lab, bool)
    other = (lab > 0) & ~own
    lik = likelihood[iyc, ixc].astype(np.float64) if likelihood is not None else np.full((K, M), 0.5)
    lik = np.where(own, 1.0, np.where(other, 0.0, lik))
    s = max(cfg.guide_min_scale, 0.75 * width)
    t_road = cfg.lambda_road * (1.0 - lik)
    t_guide = cfg.lambda_guide * np.minimum((offsets[None, :] / s) ** 2, cfg.guide_cap) * np.ones((K, 1))
    t_other = cfg.other_road_penalty * other
    valid = inside.copy()
    valid[0, :], valid[-1, :] = False, False
    valid[0, zero], valid[-1, zero] = True, True                             # pinned to the two road ends
    max_dm = int(math.floor(cfg.max_lateral_step / cfg.offset_step + 1e-9))

    def edge_terms(k: int, a: int, b: int) -> dict[str, float]:
        vec = X[k, b] - X[k - 1, a]
        ell = float(np.hypot(*vec))
        if ell < 1e-9:
            return dict.fromkeys(TERMS, 0.0)
        u = vec / ell
        return {"length": ell, "non_road": ell * 0.5 * (t_road[k - 1, a] + t_road[k, b]),
                "guide": ell * 0.5 * (t_guide[k - 1, a] + t_guide[k, b]),
                "other_road": ell * 0.5 * (t_other[k - 1, a] + t_other[k, b]),
                "direction": ell * cfg.lambda_direction * (1.0 - float(u @ T[k])) / 2.0}

    G = nx.DiGraph()
    for k in range(K):
        for m in np.nonzero(valid[k])[0]:
            G.add_node((k, int(m)))
    for k in range(1, K):
        for b in np.nonzero(valid[k])[0]:
            for a in range(max(0, b - max_dm), min(M, b + max_dm + 1)):
                if valid[k - 1, a]:
                    G.add_edge((k - 1, a), (k, int(b)), weight=sum(edge_terms(k, a, int(b)).values()))
    stats = {"stations": K, "positions": M, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
             "lateral_range_px": round(float(half * cfg.offset_step), 1), "station_spacing": cfg.station_spacing}

    src, dst = (0, zero), (K - 1, zero)
    best: dict[tuple[int, int], float] = {src: 0.0}
    pred: dict[tuple[int, int], tuple[int, int]] = {}
    for u in nx.topological_sort(G):                      # dynamic programming on the DAG
        if u not in best:
            continue
        du = best[u]
        for v, data in G[u].items():
            nd = du + data["weight"]
            if nd < best.get(v, math.inf):
                best[v], pred[v] = nd, u
    if dst not in best:
        stats["seconds"] = round(time.perf_counter() - t0, 3)
        return PathResult(False, False, "no_path", "no continuous route fits between the two road ends",
                          metrics=stats, guide_xy=P)

    nodes = [dst]
    while nodes[-1] != src:
        nodes.append(pred[nodes[-1]])
    nodes.reverse()
    pts = np.array([X[k, m] for k, m in nodes])
    breakdown = dict.fromkeys(TERMS, 0.0)
    for (k0, a), (k1, b) in zip(nodes[:-1], nodes[1:]):
        for key, val in edge_terms(k1, a, b).items():
            breakdown[key] += val
    breakdown = {k: round(v, 3) for k, v in breakdown.items()}
    reference = None
    if all(valid[k, zero] for k in range(K)):
        ref = dict.fromkeys(TERMS, 0.0)
        for k in range(1, K):
            for key, val in edge_terms(k, zero, zero).items():
                ref[key] += val
        reference = {"total_cost": round(sum(ref.values()), 3), "terms": {k: round(v, 3) for k, v in ref.items()}}

    cost = best[dst]
    length = polyline_length(pts)
    ks = np.array([k for k, _ in nodes])
    ms = np.array([m for _, m in nodes])
    on_other = other[ks, ms]
    devs = np.abs(offsets[ms])
    cover: dict[str, float] = {}
    if category is not None:
        cat = category[iyc[ks, ms], ixc[ks, ms]]
        new = lab[ks, ms] == 0
        if new.any():
            for code, key in CATEGORY_KEYS.items():
                cover[key] = round(float(np.mean(cat[new] == code)), 3)
    ratio = length / max(gap, 1e-6)
    mean_cost = cost / max(length, 1e-6)
    other_frac = float(on_other.mean()) if len(on_other) else 0.0
    stats.update({"length": round(length, 2), "gap": round(gap, 2), "length_ratio": round(ratio, 3),
                  "total_cost": round(cost, 3), "mean_cost": round(mean_cost, 3), "other_road_fraction": round(other_frac, 3),
                  "offset_mean": round(float(devs.mean()), 2), "offset_max": round(float(devs.max()), 2),
                  "seconds": round(time.perf_counter() - t0, 3)})
    code, reason = "ok", f"minimum-cost route: {length:.0f} px, mean cost {mean_cost:.2f} per px"
    if ratio > cfg.max_length_ratio:
        code, reason = "too_long", f"the cheapest route is {ratio:.2f} x the gap (limit {cfg.max_length_ratio:.2f})"
    elif mean_cost > cfg.max_mean_cost:
        code, reason = "too_costly", (f"the cheapest route costs {mean_cost:.2f} per px "
                                      f"(path cost threshold {cfg.max_mean_cost:.2f})")
    elif other_frac > cfg.max_other_road_fraction:
        code, reason = "other_road", f"the cheapest route runs over an unrelated road ({other_frac:.0%})"
    return PathResult(True, code == "ok", code, reason, smooth_polyline(pts, cfg.smoothing_window), pts, P,
                      stats, breakdown, reference, cover)
