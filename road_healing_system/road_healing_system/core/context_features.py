"""
Stage 7 - Context feature extraction.

Input : candidates, road graph, mask, image appearance maps
Output: each candidate gets raw feature values (``features``), normalised
        scores in [0, 1] or None (``scores``: A, O, W, G, N), and a short note
        per feature explaining the value (``feature_notes``).

    A  road alignment       - do the two ends face each other along a smooth line or arc?
    O  occlusion evidence   - does the gap look hidden (vegetation, shadow) or still road-like?
    W  width consistency    - are the two fragments about as wide as each other?
    G  local geometry       - can a smooth road of this width make the required turn,
                              and does it continue the fragments' own curvature?
    N  network connectivity - does the link make structural sense in the road graph?

Every score is computed by a documented rule (see README, "CARS features").
A score is None when its evidence is unavailable; it is never silently set
to a high value. How None is treated is decided later by the CARS policy.

Appearance heuristics (no semantic classifier is used):
    road-like   : colour close to the road colour model learned from the mask
    vegetation  : excess-green index ExG = (2G - R - B) / (R + G + B) above a threshold
    shadow      : darker than a fraction of the median road brightness
    textured    : high local grey-level variation (canopy, roofs, clutter - unverified)
    unexplained : none of the above (open field, water, bare soil, ...)
These are cues, not identifications of trees, buildings or shadows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import networkx as nx
import numpy as np

from config import PipelineConfig
from core.gap_detection import Candidate
from core.geometry import angle_between_deg, hermite_with_curvature, signed_angle, unit, wrap_angle
from core.graph_construction import GraphBuildResult

CATEGORY_NAMES = {0: "unexplained", 1: "road-like", 2: "vegetation-like", 3: "shadow-like",
                  4: "textured (unverified obstruction)", 5: "water-like"}
CATEGORY_KEYS = {0: "unexplained", 1: "road_like", 2: "vegetation", 3: "shadow", 4: "textured", 5: "water"}


# --------------------------------------------------------------------------- #
# Appearance maps (computed once per image)
# --------------------------------------------------------------------------- #
@dataclass
class AppearanceMaps:
    category: np.ndarray | None          # uint8 codes, see CATEGORY_NAMES
    likelihood: np.ndarray | None        # float32 road-colour likelihood in [0, 1]
    has_color: bool
    road_model_available: bool
    road_median_value: float | None
    notes: list[str] = field(default_factory=list)


def compute_appearance(image: np.ndarray | None, mask: np.ndarray, cfg: PipelineConfig,
                       has_color: bool = True) -> AppearanceMaps:
    o = cfg.occlusion
    if image is None:
        return AppearanceMaps(None, None, False, False, None,
                              ["No image supplied: occlusion evidence cannot be assessed."])
    notes: list[str] = []
    road = mask > 0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2].astype(np.float32)

    likelihood = None
    road_median_v = None
    if int(road.sum()) >= 50:
        samples = lab[road]
        if len(samples) > 20000:
            rng = np.random.default_rng(0)
            samples = samples[rng.choice(len(samples), 20000, replace=False)]
        mu = samples.mean(axis=0)
        cov = np.cov(samples.T) + np.eye(3) * 25.0       # regularised: avoids a singular model
        inv = np.linalg.inv(cov)
        diff = lab - mu
        d2 = np.einsum("...i,ij,...j->...", diff, inv, diff)
        likelihood = np.exp(-d2 / (2.0 * o.likelihood_scale)).astype(np.float32)
        road_median_v = float(np.median(value[road]))
    else:
        notes.append("Too few road pixels to learn a road colour model; road-likelihood is unavailable.")

    bgr = image.astype(np.float32)
    total = bgr.sum(axis=2) + 1e-6
    exg = (2 * bgr[..., 1] - bgr[..., 2] - bgr[..., 0]) / total
    vegetation = (exg > o.exg_threshold) if has_color else np.zeros(road.shape, bool)
    if not has_color:
        notes.append("Greyscale image: the vegetation cue is disabled.")
    if road_median_v is not None:
        shadow = value < o.shadow_relative_value * road_median_v
    else:
        shadow = value < o.shadow_absolute_value
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = int(o.texture_window) | 1
    mean = cv2.blur(gray, (k, k))
    sq = cv2.blur(gray * gray, (k, k))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    textured = std > o.texture_threshold

    water = ((bgr[..., 0] - np.maximum(bgr[..., 1], bgr[..., 2])) > o.water_blue_margin) & (bgr[..., 0] > 50) \
        if has_color else np.zeros(road.shape, bool)
    category = np.zeros(road.shape, np.uint8)
    category[textured] = 4
    category[shadow] = 3
    category[water] = 5
    category[vegetation] = 2
    if likelihood is not None:
        category[likelihood > o.road_likelihood_threshold] = 1
    return AppearanceMaps(category, likelihood, has_color, likelihood is not None, road_median_v, notes)


# --------------------------------------------------------------------------- #
# Feature context
# --------------------------------------------------------------------------- #
@dataclass
class FeatureContext:
    build: GraphBuildResult
    mask: np.ndarray
    mask_labels: np.ndarray
    dist: np.ndarray
    appearance: AppearanceMaps
    cfg: PipelineConfig
    graph_dist_cache: dict[int, dict[int, float]] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape[:2]


def make_context(build: GraphBuildResult, mask: np.ndarray, dist: np.ndarray,
                 appearance: AppearanceMaps, cfg: PipelineConfig) -> FeatureContext:
    _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return FeatureContext(build, mask, labels.astype(np.int32), dist, appearance, cfg)


def candidate_geometry(c: Candidate, ctx: FeatureContext) -> dict[str, Any]:
    """Shared geometry used by both the features and the path search."""
    ep = ctx.build.endpoints
    src = ep[c.source]
    p1, p2 = np.array(c.source_xy, float), np.array(c.target_xy, float)
    a = np.array(src.direction) if src.direction else None
    if c.target is not None:
        tgt = ep[c.target]
        arrive = -np.array(tgt.direction) if tgt.direction else None
        widths = [w for w in (src.width, tgt.width) if w]
    else:
        arrive = None  # approach the road along the chord
        u, v = c.target_edge
        r, col = int(round(p2[1])), int(round(p2[0]))
        tw = max(2.0 * float(ctx.dist[r, col]) - 1.0, 0.0)
        widths = [w for w in (src.width, tw) if w]
    width = float(np.mean(widths)) if widths else float(ctx.cfg.reconstruction.fallback_width)
    chord = unit(p2 - p1)
    t1 = a if a is not None else chord
    t2 = arrive if arrive is not None else chord
    if chord is None:   # both ends at the same point
        return {"p1": p1, "p2": p2, "a": a, "arrive": arrive, "width": width,
                "curve": np.array([p1, p2]), "tangents": np.zeros((2, 2)),
                "curvature": np.zeros(2), "ds": np.zeros(2)}
    n = int(max(50, 4 * float(np.hypot(*(p2 - p1)))))
    cur = hermite_with_curvature(p1, t1, p2, t2, n, ctx.cfg.reconstruction.tangent_scale)
    return {"p1": p1, "p2": p2, "a": a, "arrive": arrive, "width": width, "curve": cur["points"],
            "tangents": cur["tangents"], "curvature": cur["curvature"], "ds": cur["ds"]}


def _target_road_direction(c: Candidate, ctx: FeatureContext) -> np.ndarray | None:
    u, v = c.target_edge
    G = ctx.build.graph
    if not G.has_edge(u, v):
        return None
    path = G.edges[u, v]["path"]
    i = int(c.target_index)
    lo, hi = max(0, i - 6), min(len(path) - 1, i + 6)
    if hi - lo < 2:
        return None
    return unit(path[hi, ::-1].astype(float) - path[lo, ::-1].astype(float))


def _target_component(c: Candidate, ctx: FeatureContext) -> int:
    if c.target is not None:
        return ctx.build.endpoints[c.target].component
    return ctx.build.component_of.get(c.target_edge[0], -1)


# --------------------------------------------------------------------------- #
# A - alignment
# --------------------------------------------------------------------------- #
def _alignment(c: Candidate, ctx: FeatureContext, geo: dict) -> tuple[float | None, dict, str]:
    cfg = ctx.cfg
    tol = cfg.gap.direction_tolerance_deg
    p1, p2, a = geo["p1"], geo["p2"], geo["a"]
    v = unit(p2 - p1)
    if v is None or a is None:
        return None, {}, "source direction unavailable (fragment too short)"
    if c.target is not None:
        tgt = ctx.build.endpoints[c.target]
        if tgt.direction is None:
            return None, {}, "target direction unavailable (fragment too short)"
        b = np.array(tgt.direction)
        phi1 = signed_angle(v, a)
        phi2 = signed_angle(-v, b)
        dev = math.degrees(max(abs(phi1), abs(phi2)))
        bend = abs(math.degrees(wrap_angle(phi1 + phi2)))   # 0 for straight or circular arcs
        turn = angle_between_deg(a, -b)
        s_dev = float(np.clip(1 - dev / tol, 0, 1))
        s_co = float(np.clip(1 - bend / cfg.cars.cocircularity_tolerance_deg, 0, 1))
        score = 0.5 * s_dev + 0.5 * s_co
        feats = {"end_deviation_deg": round(dev, 1), "bend_inconsistency_deg": round(bend, 1),
                 "turn_angle_deg": round(turn, 1)}
        note = (f"ends deviate {dev:.0f} deg from the gap line; bend inconsistency {bend:.0f} deg "
                f"(0 = straight or smooth arc); total turn {turn:.0f} deg")
        return score, feats, note
    phi = angle_between_deg(a, v)
    t = _target_road_direction(c, ctx)
    s_dev = float(np.clip(1 - phi / tol, 0, 1))
    if t is None:
        s_meet, meet = 0.5, None
    else:
        meet_raw = angle_between_deg(v, t)
        meet = min(meet_raw, 180 - meet_raw)
        s_meet = float(np.clip(math.sin(math.radians(meet)) / math.sin(math.radians(30)), 0, 1))
    score = 0.5 * s_dev + 0.5 * s_meet
    feats = {"end_deviation_deg": round(phi, 1),
             "meeting_angle_deg": None if meet is None else round(meet, 1)}
    note = f"end deviates {phi:.0f} deg; meets the other road at " + \
        ("an unknown angle" if meet is None else f"{meet:.0f} deg (grazing joins are implausible)")
    return score, feats, note


# --------------------------------------------------------------------------- #
# O - occlusion / gap-explanation evidence
# --------------------------------------------------------------------------- #
def _occlusion(c: Candidate, ctx: FeatureContext, geo: dict) -> tuple[float | None, dict, str]:
    app = ctx.appearance
    o = ctx.cfg.occlusion
    if app.category is None:
        return None, {}, "uncertain: no image to inspect"
    h, w = ctx.shape
    thickness = int(max(3, round(geo["width"] + 2 * o.corridor_margin)))
    pts = geo["curve"]
    pad = thickness + int(max(6, round(geo["width"] * 1.5))) + 2
    x0 = int(max(0, np.floor(pts[:, 0].min()) - pad)); x1 = int(min(w - 1, np.ceil(pts[:, 0].max()) + pad))
    y0 = int(max(0, np.floor(pts[:, 1].min()) - pad)); y1 = int(min(h - 1, np.ceil(pts[:, 1].max()) + pad))
    canvas = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    local = np.round(pts - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [local], False, 1, thickness)
    corridor = (canvas > 0) & (ctx.mask[y0:y1 + 1, x0:x1 + 1] == 0)
    n = int(corridor.sum())
    if n < o.min_corridor_pixels:
        return None, {"corridor_pixels": n}, f"uncertain: gap corridor too small ({n} px) to inspect"
    cats = ctx.appearance.category[y0:y1 + 1, x0:x1 + 1][corridor]
    fr = {name: float(np.mean(cats == code)) for code, name in CATEGORY_NAMES.items()}
    weights = {"unexplained": o.weight_unexplained, "road-like": o.weight_road_like,
               "vegetation-like": o.weight_vegetation, "shadow-like": o.weight_shadow,
               "textured (unverified obstruction)": o.weight_textured, "water-like": o.weight_water}
    score = float(np.clip(sum(fr[k] * weights[k] for k in fr), 0, 1))
    dominant = max(fr, key=fr.get)
    feats = {"corridor_pixels": n, "frac_road_like": round(fr["road-like"], 3),
             "frac_vegetation": round(fr["vegetation-like"], 3), "frac_shadow": round(fr["shadow-like"], 3),
             "frac_textured": round(fr["textured (unverified obstruction)"], 3),
             "frac_water": round(fr["water-like"], 3),
             "frac_unexplained": round(fr["unexplained"], 3), "dominant_cover": dominant}
    # surrounding environment: a ring around the corridor, outside existing roads
    ring_px = int(max(6, round(geo["width"] * 1.5)))
    ring = cv2.dilate((canvas > 0).astype(np.uint8), np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)) > 0
    ring &= (canvas == 0) & (ctx.mask[y0:y1 + 1, x0:x1 + 1] == 0)
    if int(ring.sum()) >= o.min_corridor_pixels:
        rc = ctx.appearance.category[y0:y1 + 1, x0:x1 + 1][ring]
        for code, key in CATEGORY_KEYS.items():
            feats[f"env_{key}"] = round(float(np.mean(rc == code)), 3)
        feats["env_dominant"] = CATEGORY_NAMES[int(np.bincount(rc, minlength=6).argmax())]
        feats["env_pixels"] = int(ring.sum())
    parts = [f"{fr[k]:.0%} {k}" for k in sorted(fr, key=fr.get, reverse=True) if fr[k] >= 0.1]
    note = "gap corridor: " + (", ".join(parts) if parts else "mixed cover")
    if not app.has_color:
        note += " (no colour: vegetation not assessed)"
    return score, feats, note


# --------------------------------------------------------------------------- #
# W - width consistency
# --------------------------------------------------------------------------- #
def _width(c: Candidate, ctx: FeatureContext) -> tuple[float | None, dict, str]:
    ep = ctx.build.endpoints
    ws = ep[c.source].width
    if c.target is not None:
        wt = ep[c.target].width
        tol = ctx.cfg.cars.width_tolerance
        label = "fragment widths"
    else:
        r, col = int(round(c.target_xy[1])), int(round(c.target_xy[0]))
        wt = max(2.0 * float(ctx.dist[r, col]) - 1.0, 0.0) or None
        tol = 2.0 * ctx.cfg.cars.width_tolerance   # side roads are often narrower than main roads
        label = "end width vs. target road width"
    if not ws or not wt:
        return None, {"width_source": ws, "width_target": wt}, "uncertain: width estimate unavailable"
    rd = abs(ws - wt) / max(ws, wt)
    score = float(np.clip(1 - rd / tol, 0, 1))
    feats = {"width_source": round(ws, 2), "width_target": round(wt, 2), "width_rel_diff": round(rd, 3)}
    return score, feats, f"{label} {ws:.1f} vs {wt:.1f} px ({rd:.0%} difference)"


# --------------------------------------------------------------------------- #
# G - local geometry
# --------------------------------------------------------------------------- #
def _geometry(c: Candidate, ctx: FeatureContext, geo: dict) -> tuple[float | None, dict, str]:
    """Turn feasibility of the smooth connecting curve, and curvature continuity with
    the fragments. Curvatures are compared as kappa x road width (dimensionless)."""
    cc = ctx.cfg.cars
    k = geo["curvature"]
    width = geo["width"]
    if geo["a"] is None:
        return None, {}, "uncertain: source direction unavailable"
    if not np.all(np.isfinite(k)):
        return 0.0, {"min_turn_radius_px": 0.0}, "the connecting curve would need a cusp"
    kmax = float(np.max(np.abs(k)))
    rmin = float("inf") if kmax < 1e-9 else 1.0 / kmax
    s_rad = float(np.clip(rmin / (cc.min_radius_factor * width), 0, 1))
    feats = {"min_turn_radius_px": None if not np.isfinite(rmin) else round(rmin, 1),
             "min_turn_radius_widths": None if not np.isfinite(rmin) else round(rmin / width, 2)}
    ep = ctx.build.endpoints
    diffs = []
    ks = ep[c.source].curvature
    if ks is not None:
        diffs.append(abs(k[0] - ks) * width)
    if c.target is not None:
        kt = ep[c.target].curvature
        if kt is not None:
            diffs.append(abs(k[-1] + kt) * width)   # the target's curvature was measured travelling the other way
    radius_note = ("straight connection" if not np.isfinite(rmin)
                   else f"tightest turn radius {rmin:.0f} px = {rmin / width:.1f} road widths")
    if not diffs:
        return s_rad, feats, radius_note + "; fragment curvature unavailable"
    jump = float(np.mean(diffs))
    s_cont = float(np.clip(1 - jump / cc.curvature_tolerance, 0, 1))
    feats["curvature_jump"] = round(jump, 3)
    return 0.5 * s_rad + 0.5 * s_cont, feats, f"{radius_note}; curvature jump {jump:.2f} (x width)"


# --------------------------------------------------------------------------- #
# N - network connectivity
# --------------------------------------------------------------------------- #
def _network(c: Candidate, ctx: FeatureContext, geo: dict) -> tuple[float, dict, str]:
    G = ctx.build.graph
    cc = ctx.cfg.cars
    comp_s = ctx.build.endpoints[c.source].component
    comp_t = _target_component(c, ctx)
    feats: dict[str, Any] = {"component_source": comp_s, "component_target": comp_t,
                             "same_component": comp_s == comp_t}
    if c.target is not None and G.has_edge(c.source, c.target):
        feats["duplicate_edge"] = True
        return 0.0, feats, "duplicate: the two ends are already joined by an edge"
    feats["duplicate_edge"] = False

    if comp_s != comp_t:
        topo, topo_note = cc.merge_score, f"joins separate components {comp_s} and {comp_t}"
        detour = None
    else:
        if c.source not in ctx.graph_dist_cache:
            ctx.graph_dist_cache[c.source] = nx.single_source_bellman_ford_path_length(G, c.source, weight="length")
        dists = ctx.graph_dist_cache[c.source]
        targets = [c.target] if c.target is not None else list(c.target_edge)
        net = min((dists.get(t, np.inf) for t in targets), default=np.inf)
        detour = float(net / max(c.distance, 1e-6))
        if detour >= ctx.cfg.gap.min_detour_ratio:
            topo, topo_note = cc.loop_closure_score, f"closes a loop inside component {comp_s} (detour x{detour:.1f})"
        else:
            topo, topo_note = cc.shortcut_score, f"shortcut inside component {comp_s} (detour only x{detour:.1f})"
    feats["detour_ratio"] = None if detour is None or not np.isfinite(detour) else round(detour, 2)

    # does the expected connection run over an unrelated road?
    h, w = ctx.shape
    pts = np.round(geo["curve"]).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    labels = ctx.mask_labels[pts[:, 1], pts[:, 0]]
    own = {int(ctx.mask_labels[int(round(c.source_xy[1])), int(round(c.source_xy[0]))]),
           int(ctx.mask_labels[int(round(c.target_xy[1])), int(round(c.target_xy[0]))])}
    crossing = int(np.sum((labels > 0) & ~np.isin(labels, list(own))))
    feats["crossing_pixels"] = crossing
    cross_factor = 1.0 - cc.crossing_penalty if crossing > 2 else 1.0

    competitors = int(c.features.get("competing_candidates", 0))
    amb_factor = 1.0 / (1.0 + cc.ambiguity_penalty * competitors)
    score = float(np.clip(topo * cross_factor * amb_factor, 0, 1))
    notes = [topo_note]
    if crossing > 2:
        notes.append(f"crosses an unrelated road ({crossing} px)")
    if competitors:
        notes.append(f"{competitors} competing candidate(s) share an end")
    return score, feats, "; ".join(notes)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def extract_all_features(candidates: list[Candidate], ctx: FeatureContext) -> None:
    usage: dict[int, int] = {}
    for c in candidates:
        for e in c.endpoint_ids():
            usage[e] = usage.get(e, 0) + 1
    for c in candidates:
        c.features = {"distance_px": round(c.distance, 2), "kind": c.kind,
                      "competing_candidates": max(usage[e] - 1 for e in c.endpoint_ids()),
                      "source_curvature": ctx.build.endpoints[c.source].curvature,
                      "target_curvature": ctx.build.endpoints[c.target].curvature if c.target is not None else None}
        geo = candidate_geometry(c, ctx)
        a, fa, na = _alignment(c, ctx, geo)
        o, fo, no = _occlusion(c, ctx, geo)
        w, fw, nw = _width(c, ctx)
        g, fg, ng = _geometry(c, ctx, geo)
        n, fn, nn = _network(c, ctx, geo)
        for f in (fa, fo, fw, fg, fn):
            c.features.update(f)
        c.scores = {"A": a, "O": o, "W": w, "G": g, "N": n}
        c.feature_notes = {"A": na, "O": no, "W": nw, "G": ng, "N": nn}
