"""
Stage 6 - Automatic gap detection.

Input : road graph, endpoint table, gap configuration
Output: list of ``Candidate`` objects (plausible gaps for contextual analysis)
        and a list of geometric pre-filter rejections (for diagnostics).

Distance and direction are only *preliminary* evidence here. Every retained
candidate still has to pass the CARS evaluation before anything is drawn.

Candidate kinds
---------------
endpoint_pair   : two fragment ends facing each other across a gap.
endpoint_to_road: a fragment end pointing at the side of another road
                  (a T-junction hidden by an occluder). Optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from config import GapConfig
from core.geometry import angle_between_deg, unit
from core.graph_construction import GraphBuildResult


@dataclass
class Candidate:
    id: str
    kind: str                                   # endpoint_pair | endpoint_to_road
    source: int                                 # endpoint node id
    target: int | None                          # endpoint node id (pairs only)
    source_xy: tuple[float, float]
    target_xy: tuple[float, float]
    distance: float
    origin: str = "auto"                        # auto | manual | auto+manual
    target_edge: tuple[int, int] | None = None  # for endpoint_to_road
    target_index: int | None = None             # pixel index along target edge path
    angle_source_deg: float | None = None
    angle_target_deg: float | None = None
    features: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float | None] = field(default_factory=dict)
    feature_notes: dict[str, str] = field(default_factory=dict)
    cars: float | None = None
    cars_details: dict[str, Any] = field(default_factory=dict)
    decision: str = "pending"                   # automatic: accepted | review | rejected
    reason: str = ""
    explanation: str = ""
    recommendation: str = "pending"             # recommended | uncertain | not_recommended
    rec_basis: str = ""                         # score | band | missing | conflict | veto
    rec_detail: str = ""
    xai: dict[str, Any] = field(default_factory=dict)
    user_decision: str = "pending"              # pending | approved | rejected | skipped
    decided_at: str = ""
    recon_status: str = "not_attempted"         # not_attempted | reconstructed | failed | blocked
    recon_reason: str = ""
    segment_xy: np.ndarray | None = None        # centre line of the reconstructed path
    guide_xy: np.ndarray | None = None
    recon_metrics: dict[str, Any] = field(default_factory=dict)
    recon_explanation: dict[str, Any] = field(default_factory=dict)
    preview_xy: np.ndarray | None = None        # the connection approval would draw (computed, not applied)
    preview_metrics: dict[str, Any] = field(default_factory=dict)
    priority: str = "other"                     # occlusion | visible road | other (review order)
    fragments_info: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple:
        if self.target is not None:
            return ("pair",) + tuple(sorted((self.source, self.target)))
        return ("road", self.source, round(self.target_xy[0]), round(self.target_xy[1]))

    def endpoint_ids(self) -> list[int]:
        return [self.source] + ([self.target] if self.target is not None else [])


@dataclass
class GapDetectionResult:
    candidates: list[Candidate]
    prefilter_rejected: list[dict[str, Any]]
    border_endpoints: list[int]
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for c in self.candidates:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        return {"candidates": len(self.candidates), "by_kind": kinds,
                "prefilter_rejected": len(self.prefilter_rejected),
                "border_endpoints_excluded": len(self.border_endpoints)}


def _graph_distances(G: nx.Graph, source: int, cache: dict[int, dict[int, float]]) -> dict[int, float]:
    if source not in cache:
        cache[source] = nx.single_source_bellman_ford_path_length(G, source, weight="length")
    return cache[source]


def _same_component_detour(G: nx.Graph, src: int, dst_nodes: list[tuple[int, float]], euclid: float,
                           cache: dict[int, dict[int, float]]) -> float:
    """Network distance / straight distance for a pair inside one component."""
    dists = _graph_distances(G, src, cache)
    best = min((dists.get(n, np.inf) + extra for n, extra in dst_nodes), default=np.inf)
    return float(best / max(euclid, 1e-6))


def _nearest_distance(points: np.ndarray, refs: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Distance from every point to its nearest reference point (chunked to bound memory)."""
    if len(refs) == 0:
        return np.full(len(points), np.inf)
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        d = np.hypot(p[:, None, 0] - refs[None, :, 0], p[:, None, 1] - refs[None, :, 1])
        out[i:i + chunk] = d.min(axis=1)
    return out


def fragment_evidence(build: GraphBuildResult, likelihood: np.ndarray | None, cfg: GapConfig) -> dict[str, Any]:
    """Image-based check that each fragment is an identifiable road structure.

    For every connected component of the road graph: centre-line length, width,
    elongation (length / width) and the mean road-colour likelihood of the image
    along its centre line; for every endpoint, the likelihood along its last 20 px.
    Without an image the likelihood is unknown and the fragment is 'unverified'
    (never assumed to be road)."""
    G = build.graph
    comps: dict[int, dict[str, Any]] = {}
    for comp in build.components:
        cid = comp["component"]
        nodes = [n for n, c in build.component_of.items() if c == cid]
        pix = [G.edges[e]["path"] for e in G.subgraph(nodes).edges]
        length = float(comp["length_px"])
        widths = [G.edges[e]["width"] for e in G.subgraph(nodes).edges if G.edges[e].get("width")]
        width = float(np.median(widths)) if widths else 0.0
        # the skeleton stops about half a width short of each end, so add one width (as road extraction does)
        full = length + width
        leaves_image = any(build.endpoints[n].at_border for n in nodes if n in build.endpoints)
        info: dict[str, Any] = {"component": cid, "length_px": round(full, 1), "width_px": round(width, 1),
                                "elongation": round(full / width, 2) if width else None, "road_likelihood": None,
                                "leaves_image": leaves_image}
        if likelihood is not None and pix:
            rc = np.vstack(pix)
            info["road_likelihood"] = round(float(likelihood[rc[:, 0], rc[:, 1]].mean()), 3)
        reasons = []
        if full < cfg.min_fragment_length:
            reasons.append(f"fragment F{cid} is only {full:.0f} px long")
        # a fragment cut by the image frame continues outside the view: its visible shape can be short
        if not leaves_image and info["elongation"] is not None and info["elongation"] < cfg.min_fragment_elongation:
            reasons.append(f"fragment F{cid} is not elongated like a road (length/width {info['elongation']})")
        if info["road_likelihood"] is not None and info["road_likelihood"] < cfg.min_fragment_road_likelihood:
            reasons.append(f"fragment F{cid} does not look like road in the image (likelihood {info['road_likelihood']:.2f})")
        info["verified"] = None if likelihood is None and not reasons else not reasons
        info["reasons"] = reasons
        comps[cid] = info
    ends: dict[int, float | None] = {}
    for e in build.endpoints.values():
        if likelihood is None:
            ends[e.id] = None
            continue
        m = next(iter(G.neighbors(e.id)))
        path = G.edges[e.id, m]["path"]
        path = path if G.edges[e.id, m]["start"] == e.id else path[::-1]
        seg = path[:20]
        ends[e.id] = round(float(likelihood[seg[:, 0], seg[:, 1]].mean()), 3)
    return {"fragments": comps, "end_likelihood": ends}


def _road_to_road_checks(ea, eb, v: np.ndarray, scene: dict[str, Any] | None, cfg: GapConfig) -> str | None:
    """Gate before CARS: both sides must be identifiable road structures whose trajectories can meet."""
    from core.geometry import signed_angle, wrap_angle
    if ea.direction is not None and eb.direction is not None:
        phi1 = signed_angle(v, np.array(ea.direction))
        phi2 = signed_angle(-v, np.array(eb.direction))
        bend = abs(np.degrees(wrap_angle(phi1 + phi2)))
        if bend > cfg.max_trajectory_bend_deg:
            return (f"the two trajectories would need an S-shaped bend of {bend:.0f} deg to meet "
                    f"(limit {cfg.max_trajectory_bend_deg:.0f}): not one continuous road")
    ev = (scene or {}).get("evidence")
    if not ev:
        return None
    for e in (ea, eb):
        f = ev["fragments"].get(e.component)
        if f and f["verified"] is False:
            return "not a road-to-road pair: " + "; ".join(f["reasons"])
        lk = ev["end_likelihood"].get(e.id)
        if lk is not None and lk < cfg.min_end_road_likelihood:
            return f"road end {e.id} does not look like road in the image (likelihood {lk:.2f})"
    return None


def _chord_checks(pa: np.ndarray, pb: np.ndarray, wa: float | None, wb: float | None,
                  scene: dict[str, Any] | None, cfg: GapConfig) -> str | None:
    """Context checks on the straight line between two ends. Returns a reason to exclude, or None."""
    if wa and wb and min(wa, wb) / max(wa, wb) < cfg.min_width_ratio:
        return (f"widths {wa:.0f} and {wb:.0f} px are too different (ratio below "
                f"{cfg.min_width_ratio:.2f}): different kinds of road")
    if not scene:
        return None
    labels, water = scene.get("mask_labels"), scene.get("water")
    h, w = labels.shape
    d = float(np.hypot(*(pb - pa)))
    n = max(int(d), 2)
    t = np.linspace(0, 1, n)[:, None]
    pts = pa + (pb - pa) * t
    ix = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    iy = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    ra, rb = (wa or 6) / 2 + 2, (wb or 6) / 2 + 2
    inner = (np.hypot(*(pts - pa).T) > ra) & (np.hypot(*(pts - pb).T) > rb)
    own = {int(labels[int(round(pa[1])), int(round(pa[0]))]), int(labels[int(round(pb[1])), int(round(pb[0]))]), 0}
    if cfg.exclude_crossing and inner.any():
        crossing = int(np.sum(~np.isin(labels[iy[inner], ix[inner]], list(own))))
        if crossing > 2:
            return f"another road lies across the gap ({crossing} px): the ends belong to different roads"
    if water is not None and inner.any():
        frac = float(np.mean(water[iy[inner], ix[inner]]))
        if frac > cfg.water_barrier_fraction:
            return f"the gap is {frac:.0%} water-like: a road does not continue under water"
    return None


def detect_candidates(build: GraphBuildResult, cfg: GapConfig,
                      scene: dict[str, Any] | None = None) -> GapDetectionResult:
    """``scene`` (optional): {"mask_labels": int array, "water": bool array} for context checks."""
    G = build.graph
    endpoints = build.endpoints
    tol = cfg.direction_tolerance_deg
    candidates: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    cache: dict[int, dict[int, float]] = {}
    border = sorted(e.id for e in endpoints.values() if e.at_border)
    usable = [e for e in endpoints.values() if not e.at_border]
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"C{counter:03d}"

    def reject(kind: str, a: int, b: Any, dist: float, reason: str) -> None:
        rejected.append({"kind": kind, "source": a, "target": b,
                         "distance": round(dist, 1), "reason": reason})

    pair_pool: list[dict[str, Any]] = []

    # ---------------- endpoint pairs ---------------------------------------
    if cfg.candidate_mode in ("auto", "both"):
        for i, ea in enumerate(usable):
            for eb in usable[i + 1:]:
                pa, pb = np.array(ea.xy), np.array(eb.xy)
                d = float(np.hypot(*(pb - pa)))
                if d > cfg.max_gap_distance:
                    continue  # too far: not recorded (would flood the diagnostics)
                if d < cfg.min_gap_distance:
                    reject("endpoint_pair", ea.id, eb.id, d, "closer than the minimum gap distance")
                    continue
                if ea.component == eb.component:
                    if not cfg.allow_same_component:
                        reject("endpoint_pair", ea.id, eb.id, d, "same component (same-component pairs disabled)")
                        continue
                    ratio = _same_component_detour(G, ea.id, [(eb.id, 0.0)], d, cache)
                    if ratio < cfg.min_detour_ratio:
                        reject("endpoint_pair", ea.id, eb.id, d,
                               f"already connected by a short route (detour x{ratio:.1f})")
                        continue
                v = unit(pb - pa)
                ang_a = angle_between_deg(np.array(ea.direction), v) if ea.direction else None
                ang_b = angle_between_deg(np.array(eb.direction), -v) if eb.direction else None
                bad = [x for x in (ang_a, ang_b) if x is not None and x > tol]
                if bad:
                    reject("endpoint_pair", ea.id, eb.id, d,
                           f"ends do not face each other ({max(bad):.0f} deg > {tol:.0f} deg tolerance)")
                    continue
                why = _chord_checks(pa, pb, ea.width, eb.width, scene, cfg) or _road_to_road_checks(ea, eb, v, scene, cfg)
                if why:
                    reject("endpoint_pair", ea.id, eb.id, d, why)
                    continue
                cos_a = np.cos(np.radians(ang_a)) if ang_a is not None else 0.5
                cos_b = np.cos(np.radians(ang_b)) if ang_b is not None else 0.5
                match = (1.0 - d / cfg.max_gap_distance) * 0.5 * (cos_a + cos_b)
                pair_pool.append({"a": ea, "b": eb, "d": d, "ang_a": ang_a, "ang_b": ang_b, "match": match})

        # each road end keeps only its best partner, and only if that partner chooses it back
        best: dict[int, dict[str, Any]] = {}
        for pr in pair_pool:
            for e in (pr["a"].id, pr["b"].id):
                if e not in best or pr["match"] > best[e]["match"]:
                    best[e] = pr
        for pr in pair_pool:
            a, b = pr["a"].id, pr["b"].id
            if cfg.mutual_best and (best[a] is not pr or best[b] is not pr):
                other = best[a] if best[a] is not pr else best[b]
                end = a if best[a] is not pr else b
                partner = other["b"].id if other["a"].id == end else other["a"].id
                reject("endpoint_pair", a, b, pr["d"],
                       f"road end {end} matches road end {partner} better (closer and better aligned)")
                continue
            candidates.append(Candidate(next_id(), "endpoint_pair", a, b, pr["a"].xy, pr["b"].xy, pr["d"],
                                        angle_source_deg=pr["ang_a"], angle_target_deg=pr["ang_b"]))

    # ---------------- endpoint -> road side (T-junction) -------------------
    if cfg.candidate_mode in ("auto", "both") and cfg.enable_t_junctions and usable:
        pix, owner_edge, owner_idx, owner_comp = [], [], [], []
        for u, v, d in G.edges(data=True):
            if d.get("kind", "existing") != "existing":
                continue
            path = d["path"]
            pix.append(path[:, ::-1].astype(float))
            owner_edge.extend([(u, v)] * len(path))
            owner_idx.extend(range(len(path)))
            owner_comp.extend([build.component_of.get(u, -1)] * len(path))
        if pix:
            pix_xy = np.vstack(pix)
            owner_comp_arr = np.asarray(owner_comp)
            end_xy = np.array([e.xy for e in endpoints.values()])
            near_end = _nearest_distance(pix_xy, end_xy)
            paired = {x for c in candidates for x in c.endpoint_ids()}
            for e in usable:
                if e.direction is None or e.id in paired:
                    continue
                p = np.array(e.xy)
                vec = pix_xy - p
                dist = np.hypot(vec[:, 0], vec[:, 1])
                ok = (dist >= cfg.min_gap_distance) & (dist <= cfg.max_gap_distance)
                if not cfg.allow_same_component:
                    ok &= owner_comp_arr != e.component
                else:
                    ok &= owner_comp_arr != -1
                if not ok.any():
                    continue
                cosang = (vec @ np.array(e.direction)) / np.maximum(dist, 1e-9)
                ok &= cosang >= np.cos(np.radians(tol))
                # leave targets near another endpoint to the endpoint-pair candidates
                ok &= near_end > cfg.t_junction_clearance
                idx = np.nonzero(ok)[0]
                if len(idx) == 0:
                    continue
                # prefer close targets straight ahead
                score = dist[idx] * (1.0 + 2.0 * (1.0 - cosang[idx]))
                order = idx[np.argsort(score)]
                chosen = None
                for k in order[:50]:
                    comp = owner_comp_arr[k]
                    if comp == e.component:
                        u, v = owner_edge[k]
                        path = G.edges[u, v]["path"]
                        start = G.edges[u, v]["start"]
                        along = float(owner_idx[k])
                        to_u = along if start == u else len(path) - 1 - along
                        to_v = len(path) - 1 - to_u
                        ratio = _same_component_detour(G, e.id, [(u, to_u), (v, to_v)], dist[k], cache)
                        if ratio < cfg.min_detour_ratio:
                            continue
                    chosen = k
                    break
                if chosen is None:
                    continue
                comp = owner_comp_arr[chosen]
                if any(c.source == e.id and c.target is not None and endpoints[c.target].component == comp
                       for c in candidates) or any(
                        c.target == e.id and endpoints[c.source].component == comp for c in candidates):
                    continue  # an endpoint pair already covers this connection
                tx, ty = pix_xy[chosen]
                d = float(dist[chosen])
                ang = angle_between_deg(np.array(e.direction), vec[chosen])
                u, v = owner_edge[chosen]
                why = _chord_checks(p, np.array([tx, ty]), e.width, G.edges[u, v].get("width"), scene, cfg)
                if why:
                    reject("endpoint_to_road", e.id, f"road edge {(u, v)}", d, why)
                    continue
                candidates.append(Candidate(next_id(), "endpoint_to_road", e.id, None, e.xy,
                                            (float(tx), float(ty)), d, target_edge=owner_edge[chosen],
                                            target_index=int(owner_idx[chosen]), angle_source_deg=ang))

    # ---------------- manual pairs ------------------------------------------
    notes: list[str] = []
    if cfg.candidate_mode in ("manual", "both"):
        existing = {c.key(): c for c in candidates}
        for pair in cfg.manual_pairs:
            a, b = int(pair[0]), int(pair[1])
            if a not in endpoints or b not in endpoints or a == b:
                notes.append(f"Manual pair {a}-{b} ignored: both ids must be different endpoint ids "
                             f"(see the endpoint table).")
                continue
            ea, eb = endpoints[a], endpoints[b]
            pa, pb = np.array(ea.xy), np.array(eb.xy)
            d = float(np.hypot(*(pb - pa)))
            key = ("pair",) + tuple(sorted((a, b)))
            if key in existing:
                existing[key].origin = "auto+manual"
                continue
            v = unit(pb - pa) if d > 0 else np.array([1.0, 0.0])
            ang_a = angle_between_deg(np.array(ea.direction), v) if ea.direction else None
            ang_b = angle_between_deg(np.array(eb.direction), -v) if eb.direction else None
            c = Candidate(next_id(), "endpoint_pair", a, b, ea.xy, eb.xy, d, origin="manual",
                          angle_source_deg=ang_a, angle_target_deg=ang_b)
            candidates.append(c)
            existing[key] = c

    if not usable:
        notes.append("No interior endpoints: every road fragment end touches the image border.")
    for i, c in enumerate(candidates, 1):                 # Gap 01, Gap 02, ...
        c.id = f"G{i:02d}"
    return GapDetectionResult(candidates, rejected, border, notes)


def describe_fragments(build: GraphBuildResult) -> list[dict[str, Any]]:
    """Connected road components with their open (non-border) ends: the fragments that can have gaps."""
    open_ends: dict[int, list[int]] = {}
    border_ends: dict[int, list[int]] = {}
    for e in build.endpoints.values():
        (border_ends if e.at_border else open_ends).setdefault(e.component, []).append(e.id)
    out = []
    for comp in build.components:
        cid = comp["component"]
        out.append({**comp, "open_ends": sorted(open_ends.get(cid, [])),
                    "border_ends": sorted(border_ends.get(cid, [])),
                    "is_fragment": bool(open_ends.get(cid))})
    return out
