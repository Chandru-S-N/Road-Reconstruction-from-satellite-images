"""
Stage 5 - Road graph construction.

Input : traced skeleton (nodes + branches), distance map
Output: NetworkX graph whose nodes are endpoints, intersections and a few
        waypoints, and whose edges carry the real skeleton geometry.

Node attributes : rc (row, col), xy (x, y), type, width
Edge attributes : path (N x 2 row/col array, oriented from ``start``), start,
                  length, width, kind ("existing" | "reconstructed")

A reduced graph is used (one edge per road branch, not one node per pixel),
which keeps the network small while preserving the exact road geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from config import SkeletonConfig
from core.endpoint_extraction import Endpoint, extract_endpoints
from core.skeletonization import TraceResult


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class GraphBuildResult:
    graph: nx.Graph
    endpoints: dict[int, Endpoint]
    component_of: dict[int, int]
    components: list[dict[str, Any]]
    pruned_spurs: int = 0
    removed_small_components: int = 0
    notes: list[str] = field(default_factory=list)

    def junction_ids(self) -> list[int]:
        return [n for n, d in self.graph.degree() if d >= 3]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _step_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    d = np.diff(path.astype(float), axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _width_along(path: np.ndarray, dist: np.ndarray) -> float:
    vals = dist[path[:, 0], path[:, 1]]
    vals = vals[vals > 0]
    return float(max(2.0 * np.median(vals) - 1.0, 1.0)) if len(vals) else 0.0


def edge_path_from(G: nx.Graph, a: int, b: int) -> np.ndarray:
    """Edge geometry oriented so that it starts at node a."""
    data = G.edges[a, b]
    path = data["path"]
    return path if data["start"] == a else path[::-1]


def _new_node_id(G: nx.Graph) -> int:
    return (max(G.nodes) + 1) if G.number_of_nodes() else 1


def _add_node(G: nx.Graph, nid: int, rc: tuple[int, int], dist: np.ndarray, ntype: str = "") -> None:
    r, c = int(rc[0]), int(rc[1])
    G.add_node(nid, rc=(r, c), xy=(float(c), float(r)), type=ntype,
               width=float(max(2.0 * dist[r, c] - 1.0, 0.0)))


def add_branch(G: nx.Graph, u: int, v: int, path: np.ndarray, dist: np.ndarray,
               kind: str = "existing", **extra: Any) -> None:
    """Add one branch; parallel edges and self-loops are split with waypoint nodes
    so a simple Graph can hold them without losing geometry."""
    path = np.asarray(path, dtype=np.int32)
    if u == v or G.has_edge(u, v):
        if len(path) < 4:
            return
        i1, i2 = len(path) // 3, 2 * len(path) // 3
        w1, w2 = _new_node_id(G), _new_node_id(G) + 1
        _add_node(G, w1, tuple(path[i1]), dist, "waypoint")
        _add_node(G, w2, tuple(path[i2]), dist, "waypoint")
        for a, b, seg in ((u, w1, path[:i1 + 1]), (w1, w2, path[i1:i2 + 1]), (w2, v, path[i2:])):
            G.add_edge(a, b, path=seg, start=a, length=_step_length(seg),
                       width=_width_along(seg, dist), kind=kind, **extra)
        return
    G.add_edge(u, v, path=path, start=u, length=_step_length(path),
               width=_width_along(path, dist), kind=kind, **extra)


def merge_degree_two(G: nx.Graph, dist: np.ndarray) -> int:
    """Collapse nodes with exactly two branches into one continuous edge."""
    merged = 0
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes):
            if n not in G or G.degree(n) != 2:
                continue
            a, b = list(G.neighbors(n))
            if G.has_edge(a, b):
                continue  # keep as waypoint to avoid a parallel edge
            if G.edges[a, n].get("kind") != G.edges[n, b].get("kind"):
                continue
            p1 = edge_path_from(G, a, n)
            p2 = edge_path_from(G, n, b)
            kind = G.edges[a, n].get("kind", "existing")
            combined = np.vstack([p1, p2[1:]])
            G.remove_node(n)
            G.add_edge(a, b, path=combined, start=a, length=_step_length(combined),
                       width=_width_along(combined, dist), kind=kind)
            merged += 1
            changed = True
    return merged


def update_node_types(G: nx.Graph) -> None:
    for n, d in G.degree():
        G.nodes[n]["type"] = {0: "isolated", 1: "endpoint", 2: "waypoint"}.get(d, "junction")


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #
def build_road_graph(trace: TraceResult, dist: np.ndarray, shape: tuple[int, int],
                     cfg: SkeletonConfig) -> GraphBuildResult:
    G = nx.Graph()
    for label, pixels in trace.node_pixels.items():
        arr = np.asarray(pixels)
        centroid = arr.mean(axis=0)
        rep = arr[np.argmin(np.sum((arr - centroid) ** 2, axis=1))]
        _add_node(G, int(label), (int(rep[0]), int(rep[1])), dist)
    for a, b, path in trace.branches:
        add_branch(G, int(a), int(b), np.asarray(path), dist)

    # spur pruning: short endpoint branches hanging off an intersection
    pruned = 0
    for _ in range(max(cfg.prune_iterations, 0)):
        degrees = dict(G.degree())
        to_remove = []
        for n, d in degrees.items():
            if d != 1:
                continue
            m = next(iter(G.neighbors(n)))
            if degrees[m] < 3:
                continue
            r, c = G.nodes[m]["rc"]
            limit = max(cfg.prune_min_length, cfg.prune_width_factor * 2.0 * float(dist[r, c]))
            if G.edges[n, m]["length"] < limit:
                to_remove.append(n)
        if not to_remove:
            break
        G.remove_nodes_from(to_remove)
        pruned += len(to_remove)
        merge_degree_two(G, dist)
    merge_degree_two(G, dist)

    # drop isolated pixels and tiny components
    removed_small = 0
    for comp in list(nx.connected_components(G)):
        total = sum(G.edges[e]["length"] for e in G.subgraph(comp).edges)
        if total < cfg.min_component_length:
            G.remove_nodes_from(comp)
            removed_small += 1
    update_node_types(G)

    component_of: dict[int, int] = {}
    components = []
    for cid, comp in enumerate(sorted(nx.connected_components(G), key=lambda c: -len(c))):
        sub = G.subgraph(comp)
        length = sum(d["length"] for _, _, d in sub.edges(data=True))
        rcs = np.array([G.nodes[n]["rc"] for n in comp])
        for n in comp:
            component_of[n] = cid
        components.append({
            "component": cid, "nodes": len(comp), "edges": sub.number_of_edges(),
            "length_px": round(length, 1),
            "endpoints": sum(1 for n in comp if G.degree(n) == 1),
            "junctions": sum(1 for n in comp if G.degree(n) >= 3),
            "bbox_xyxy": [int(rcs[:, 1].min()), int(rcs[:, 0].min()),
                          int(rcs[:, 1].max()), int(rcs[:, 0].max())],
        })

    endpoints = extract_endpoints(G, dist, shape, component_of, cfg)
    notes = list(trace.notes)
    if G.number_of_nodes() == 0:
        notes.append("The skeleton produced no usable road graph (mask empty or too fragmented).")
    return GraphBuildResult(G, endpoints, component_of, components, pruned, removed_small, notes)


def rasterize_graph(G: nx.Graph, shape: tuple[int, int], kinds: tuple[str, ...] = ("existing",)) -> np.ndarray:
    """Skeleton image rebuilt from graph edges (after pruning)."""
    out = np.zeros(shape, bool)
    for _, _, d in G.edges(data=True):
        if d.get("kind", "existing") in kinds:
            p = d["path"]
            out[p[:, 0], p[:, 1]] = True
    return out


def graph_summary(G: nx.Graph) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        kinds[d.get("kind", "existing")] = kinds.get(d.get("kind", "existing"), 0) + 1
    return {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "edges_by_kind": kinds,
        "endpoints": sum(1 for _, d in G.degree() if d == 1),
        "junctions": sum(1 for _, d in G.degree() if d >= 3),
        "components": nx.number_connected_components(G) if G.number_of_nodes() else 0,
    }


def graph_to_json(G: nx.Graph) -> dict[str, Any]:
    """Node-link export; long pixel paths are summarised by their end points and length."""
    nodes = [{"id": int(n), "x": d["xy"][0], "y": d["xy"][1], "type": d.get("type", ""),
              "width": round(d.get("width", 0.0), 2)} for n, d in G.nodes(data=True)]
    edges = []
    for u, v, d in G.edges(data=True):
        item = {"source": int(u), "target": int(v), "kind": d.get("kind", "existing"),
                "length": round(d["length"], 2), "width": round(d.get("width", 0.0), 2),
                "polyline_xy": d["path"][:, ::-1].tolist()}
        for key in ("candidate_id", "cars", "path_cost"):
            if key in d:
                item[key] = d[key]
        edges.append(item)
    return {"nodes": nodes, "edges": edges}
