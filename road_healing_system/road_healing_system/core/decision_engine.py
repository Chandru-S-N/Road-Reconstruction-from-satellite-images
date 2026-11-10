"""
Decision engine: turns CARS into a recommendation. It never reconstructs anything;
reconstruction happens only after the user approves a gap.

For every candidate
1. Hard vetoes -> not_recommended: duplicate pair; ends already joined; gap length
   outside the configured range.
2. CARS >= threshold                          -> recommended
   threshold - review_margin <= CARS < thr    -> uncertain (basis "band")
   below the review band                      -> not_recommended
3. Recommended but with missing evidence and ``review_if_missing`` -> uncertain ("missing").
4. Two recommended candidates sharing a road end: the higher CARS stays recommended,
   the other becomes uncertain ("conflict") and names its competitor, so the user can
   approve at most one of them.

``decision`` keeps the automatic outcome in the original vocabulary
(accepted / review / rejected) for compatibility with exports.
"""

from __future__ import annotations

from typing import Any

from config import FEATURE_NAMES, FEATURES, PipelineConfig
from core.gap_detection import Candidate

_DECISION_OF = {"recommended": "accepted", "uncertain": "review", "not_recommended": "rejected"}


def _set(c: Candidate, rec: str, basis: str, reason: str, detail: str = "") -> None:
    c.recommendation, c.rec_basis, c.rec_detail, c.reason = rec, basis, detail, reason
    c.decision = _DECISION_OF[rec]


def weakest_factor(c: Candidate) -> str:
    contrib = c.cars_details.get("contributions", {})
    weights = c.cars_details.get("weights_used", {})
    shortfall = {k: weights[k] - contrib[k] for k in contrib}
    if not shortfall:
        return "no usable evidence"
    k = max(shortfall, key=shortfall.get)
    s = c.scores.get(k)
    return f"weakest factor: {FEATURE_NAMES[k]} ({'n/a' if s is None else f'{s:.2f}'})"


def decide(candidates: list[Candidate], cfg: PipelineConfig) -> dict[str, Any]:
    thr = cfg.cars.acceptance_threshold
    low = max(0.0, thr - cfg.cars.review_margin)
    seen: set[tuple] = set()
    for c in candidates:
        if c.key() in seen:
            _set(c, "not_recommended", "veto", "duplicate candidate (this pair was already listed)")
            continue
        seen.add(c.key())
        if c.features.get("duplicate_edge"):
            _set(c, "not_recommended", "veto", "an edge already joins these ends")
        elif not cfg.gap.min_gap_distance <= c.distance <= cfg.gap.max_gap_distance:
            _set(c, "not_recommended", "veto", "gap length outside the configured range")
        elif c.cars >= thr:
            missing = c.cars_details.get("missing_filled", [])
            if missing and cfg.cars.review_if_missing:
                names = ", ".join(FEATURE_NAMES[k] for k in missing)
                _set(c, "uncertain", "missing", f"CARS {c.cars:.3f} >= {thr:.3f} but {names} could not be measured")
            else:
                _set(c, "recommended", "score", f"CARS {c.cars:.3f} >= {thr:.3f}")
        elif c.cars >= low:
            _set(c, "uncertain", "band", f"CARS {c.cars:.3f} is just below {thr:.3f} (review band); {weakest_factor(c)}")
        else:
            _set(c, "not_recommended", "score", f"CARS {c.cars:.3f} < {low:.3f}; {weakest_factor(c)}")

    usage: dict[int, str] = {}
    for c in sorted((x for x in candidates if x.recommendation == "recommended"), key=lambda x: (-x.cars, x.id)):
        rival = next((usage[e] for e in c.endpoint_ids() if e in usage), None)
        if rival:
            _set(c, "uncertain", "conflict", f"competes with {rival} (higher CARS) for the same road end", rival)
            continue
        for e in c.endpoint_ids():
            usage[e] = c.id
    return recommendation_counts(candidates) | {"threshold": thr, "review_band": [round(low, 4), thr]}


def recommendation_counts(candidates: list[Candidate]) -> dict[str, int]:
    out = {"recommended": 0, "uncertain": 0, "not_recommended": 0}
    for c in candidates:
        out[c.recommendation] = out.get(c.recommendation, 0) + 1
    return out


def candidate_rows(candidates: list[Candidate]) -> list[dict[str, Any]]:
    """Flat records for CSV export."""
    rows = []
    for c in candidates:
        f = c.features
        w = c.cars_details.get("weights_used", {})
        row: dict[str, Any] = {
            "gap_id": c.id, "kind": c.kind, "source_endpoint": c.source,
            "target": c.target if c.target is not None else f"road edge {c.target_edge}",
            "source_x": round(c.source_xy[0], 1), "source_y": round(c.source_xy[1], 1),
            "target_x": round(c.target_xy[0], 1), "target_y": round(c.target_xy[1], 1),
            "distance_px": round(c.distance, 2),
        }
        for k in FEATURES:
            s = c.scores.get(k)
            row[k] = None if s is None else round(s, 4)
        for k in FEATURES:
            row[f"w{k}"] = None if k not in w else round(w[k], 4)
        row.update({
            "missing_evidence": ",".join(c.cars_details.get("missing_filled", [])),
            "CARS": None if c.cars is None else round(c.cars, 4),
            "recommendation": c.recommendation, "recommendation_reason": c.reason,
            "user_decision": c.user_decision, "decided_at": c.decided_at,
            "reconstruction": c.recon_status, "reconstruction_reason": c.recon_reason,
            "path_length_px": c.recon_metrics.get("length"), "path_total_cost": c.recon_metrics.get("total_cost"),
            "path_mean_cost": c.recon_metrics.get("mean_cost"),
        })
        for key in ("end_deviation_deg", "bend_inconsistency_deg", "turn_angle_deg", "meeting_angle_deg",
                    "width_source", "width_target", "width_rel_diff", "min_turn_radius_widths", "curvature_jump",
                    "corridor_pixels", "frac_road_like", "frac_vegetation", "frac_shadow", "frac_textured",
                    "frac_water", "frac_unexplained", "dominant_cover", "env_dominant", "component_source",
                    "component_target", "same_component", "crossing_pixels", "competing_candidates"):
            row[key] = f.get(key)
        if c.xai:
            row["explanation_en"] = " ".join(p["en"] for p in c.xai["points"]) + " " + c.xai["conclusion"]["en"]
        rows.append(row)
    return rows
