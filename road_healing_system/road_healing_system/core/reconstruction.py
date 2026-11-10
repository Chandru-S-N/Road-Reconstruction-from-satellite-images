"""
One-by-one road reconstruction after user approval.

``ReconstructionState`` holds the evolving network: the current road mask, the
reconstructed-only layer, the current road graph and which road ends are already
used. The original road mask is never modified.

``reconstruct_gap`` handles exactly one approved gap:
1. pre-validation - the ends are not already joined and no road end is used twice;
2. graph-based trellis search on the *current* mask (core.trellis_reconstruction);
3. path validation - length ratio, path cost threshold, overlap with other roads;
4. the path is drawn with the measured road width (round caps at both ends);
5. post-validation - the two fragments must then be in one connected component;
6. only if every check passes: mask, reconstructed layer, graph and component
   counts are updated. Otherwise nothing changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import networkx as nx
import numpy as np

from config import PipelineConfig
from core.context_features import FeatureContext, candidate_geometry
from core.errors import ProcessingError
from core.evaluation import count_components
from core.gap_detection import Candidate
from core.graph_construction import _new_node_id, _step_length, update_node_types
from core.trellis_reconstruction import PathResult, find_path


@dataclass
class ReconstructionState:
    original_mask: np.ndarray
    current_mask: np.ndarray
    recon_only: np.ndarray
    graph: nx.Graph
    usage: dict[int, list[str]] = field(default_factory=dict)

    @classmethod
    def start(cls, mask: np.ndarray, graph: nx.Graph) -> "ReconstructionState":
        return cls(mask, mask.copy(), np.zeros_like(mask), graph.copy())

    def components(self) -> dict[str, int]:
        return {"mask": count_components(self.current_mask),
                "graph": nx.number_connected_components(self.graph) if self.graph.number_of_nodes() else 0}


def _label_near(labels: np.ndarray, xy: tuple[float, float]) -> int:
    h, w = labels.shape
    x, y = int(round(xy[0])), int(round(xy[1]))
    for r in range(4):
        win = labels[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
        vals = win[win > 0]
        if len(vals):
            return int(np.bincount(vals).argmax())
    return 0


def path_thickness(c: Candidate, cfg: PipelineConfig) -> int:
    r = cfg.reconstruction
    widths = [w for w in (c.features.get("width_source"), c.features.get("width_target")) if w]
    if c.kind == "endpoint_to_road" and c.features.get("width_source"):
        widths = [c.features["width_source"]]          # a side road keeps its own width
    base = float(np.mean(widths)) if widths else float(r.fallback_width)
    t = base * r.width_scale
    if widths:
        t = min(t, 1.5 * max(widths))
    return int(np.clip(round(t), r.min_width, r.max_width))


def reconstruct_gap(c: Candidate, ctx: FeatureContext, state: ReconstructionState,
                    cfg: PipelineConfig) -> dict[str, Any]:
    """Reconstruct one approved gap. Returns an outcome dict; state changes only on success."""
    t0 = time.perf_counter()
    before = state.components()
    limit = cfg.gap.max_connections_per_endpoint
    used = [e for e in c.endpoint_ids() if len(state.usage.get(e, [])) >= limit]
    if used:
        others = ", ".join(sorted({g for e in used for g in state.usage[e]}))
        return {"status": "blocked", "reason_code": "endpoint_used", "blocked_by": others, "end": used[0],
                "reason": f"road end {used[0]} is already connected by {others}",
                "components_before": before, "components_after": before}
    if c.target is not None and state.graph.has_edge(c.source, c.target):
        return {"status": "blocked", "reason_code": "already_joined", "reason": "these road ends are already joined",
                "components_before": before, "components_after": before}

    _, labels = cv2.connectedComponents((state.current_mask > 0).astype(np.uint8), connectivity=8)
    own = {_label_near(labels, c.source_xy), _label_near(labels, c.target_xy)} - {0}
    geo = candidate_geometry(c, ctx)
    res: PathResult = find_path(geo, labels, own, ctx.appearance.likelihood, ctx.appearance.category, cfg.path)
    out: dict[str, Any] = {"path": res, "components_before": before, "components_after": before,
                           "thickness": None}
    if not res.valid:
        out.update(status="failed", reason_code=res.reason_code, reason=res.reason)
        return out

    thickness = path_thickness(c, cfg)
    band = np.zeros_like(state.current_mask)
    pts = np.round(res.path_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(band, [pts], False, 255, thickness, lineType=cv2.LINE_8)
    for p in (pts[0, 0], pts[-1, 0]):
        cv2.circle(band, (int(p[0]), int(p[1])), max(thickness // 2, 1), 255, -1)
    trial = np.maximum(state.current_mask, band)
    _, lab2 = cv2.connectedComponents((trial > 0).astype(np.uint8), connectivity=8)
    a, b = _label_near(lab2, c.source_xy), _label_near(lab2, c.target_xy)
    if a == 0 or a != b:
        out.update(status="failed", reason_code="not_connected",
                   reason="after drawing the path the two road fragments were still not connected")
        return out

    added = (band > 0) & (state.current_mask == 0)
    state.current_mask = trial
    state.recon_only = np.maximum(state.recon_only,
                                  np.where(added & (state.original_mask == 0), 255, 0).astype(np.uint8))
    _add_edge(state.graph, c, res.path_xy, ctx.dist, res.metrics.get("total_cost"))
    for e in c.endpoint_ids():
        state.usage.setdefault(e, []).append(c.id)
    if not np.all(state.current_mask[state.original_mask > 0] == 255):
        raise ProcessingError("Reconstruction removed original road pixels; this must never happen.")
    out.update(status="reconstructed", reason_code="ok", reason=res.reason, thickness=thickness,
               added_pixels=int(added.sum()), components_after=state.components(),
               seconds=round(time.perf_counter() - t0, 3))
    return out


def _add_edge(H: nx.Graph, c: Candidate, path_xy: np.ndarray, dist: np.ndarray, cost: float | None) -> None:
    rc_path = np.round(path_xy[:, ::-1]).astype(np.int32)
    target = c.target if c.target is not None else _split_edge_at(
        H, (int(round(c.target_xy[1])), int(round(c.target_xy[0]))), dist)
    if target is None or H.has_edge(c.source, target):
        return
    H.add_edge(c.source, target, path=rc_path, start=c.source, length=_step_length(rc_path),
               width=float(c.features.get("width_source") or 0.0), kind="reconstructed",
               candidate_id=c.id, cars=round(float(c.cars), 4), path_cost=cost)
    update_node_types(H)


def _split_edge_at(H: nx.Graph, rc: tuple[int, int], dist: np.ndarray) -> int | None:
    best = None
    for u, v, d in H.edges(data=True):
        if d.get("kind") != "existing":
            continue
        dd = np.abs(d["path"] - np.array(rc)).sum(axis=1)
        i = int(np.argmin(dd))
        if best is None or dd[i] < best[0]:
            best = (int(dd[i]), u, v, i)
    if best is None or best[0] > 3:
        return None
    _, u, v, i = best
    data = H.edges[u, v]
    path, start = data["path"], data["start"]
    other = v if start == u else u
    if i == 0:
        return start
    if i == len(path) - 1:
        return other
    nid = _new_node_id(H)
    r, col = path[i]
    H.add_node(nid, rc=(int(r), int(col)), xy=(float(col), float(r)), type="junction",
               width=float(max(2 * dist[r, col] - 1, 0)))
    first, second = path[:i + 1], path[i:]
    H.remove_edge(u, v)
    H.add_edge(start, nid, path=first, start=start, length=_step_length(first), width=data["width"], kind="existing")
    H.add_edge(nid, other, path=second, start=nid, length=_step_length(second), width=data["width"], kind="existing")
    return nid


def preview_gap(c: Candidate, ctx: FeatureContext, state: ReconstructionState, cfg: PipelineConfig) -> PathResult:
    """The connection that approval would draw, computed on the current mask without changing anything."""
    _, labels = cv2.connectedComponents((state.current_mask > 0).astype(np.uint8), connectivity=8)
    own = {_label_near(labels, c.source_xy), _label_near(labels, c.target_xy)} - {0}
    return find_path(candidate_geometry(c, ctx), labels, own, ctx.appearance.likelihood,
                     ctx.appearance.category, cfg.path)
