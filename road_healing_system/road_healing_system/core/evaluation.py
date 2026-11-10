"""
Stage 12 - Evaluation.

Connectivity (always available)
    mask and graph component counts before/after, candidate counts, path
    successes/failures. Fewer components is *reported*, not assumed better.

Pixel-level (only with a reference mask; foreground = road = 255)
    precision, recall, F1, IoU of the input mask and of the reconstructed mask.
    Relaxed precision/recall ("correctness/completeness") with a
    ``tolerance_px`` buffer, common in road-extraction evaluation.

Gap-focused (only with a reference mask)
    added     = reconstructed \\ input        (what the system drew)
    missing   = reference \\ input            (road the input mask lacks)
    gap precision = |added within tolerance of reference| / |added|
    gap recall    = |missing within tolerance of added|  / |missing|
    These isolate the effect of reconstruction from the (much larger)
    unchanged road area that dominates whole-mask scores.

Without a reference mask no precision/recall/F1/IoU is reported; only
connectivity statistics and clearly labelled proxy values.
"""

from __future__ import annotations

from typing import Any

import cv2
import networkx as nx
import numpy as np

from core.utils import f1_score, safe_div


def count_components(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return int(n - 1)


def pixel_counts(pred: np.ndarray, ref: np.ndarray) -> dict[str, int]:
    p, r = pred > 0, ref > 0
    return {"tp": int(np.sum(p & r)), "fp": int(np.sum(p & ~r)),
            "fn": int(np.sum(~p & r)), "tn": int(np.sum(~p & ~r))}


def metrics_from_counts(c: dict[str, int]) -> dict[str, float | None]:
    precision = safe_div(c["tp"], c["tp"] + c["fp"])
    recall = safe_div(c["tp"], c["tp"] + c["fn"])
    return {"precision": precision, "recall": recall, "f1": f1_score(precision, recall),
            "iou": safe_div(c["tp"], c["tp"] + c["fp"] + c["fn"])}


def _dilate(mask: np.ndarray, tol: int) -> np.ndarray:
    if tol <= 0:
        return mask > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    return cv2.dilate((mask > 0).astype(np.uint8), k) > 0


def relaxed_metrics(pred: np.ndarray, ref: np.ndarray, tol: int) -> dict[str, float | None]:
    p, r = pred > 0, ref > 0
    correctness = safe_div(int(np.sum(p & _dilate(ref, tol))), int(p.sum()))
    completeness = safe_div(int(np.sum(r & _dilate(pred, tol))), int(r.sum()))
    return {"relaxed_precision": correctness, "relaxed_recall": completeness,
            "relaxed_f1": f1_score(correctness, completeness)}


def gap_counts(input_mask: np.ndarray, recon: np.ndarray, ref: np.ndarray, tol: int) -> dict[str, int]:
    inp, rec, rf = input_mask > 0, recon > 0, ref > 0
    added = rec & ~inp
    missing = rf & ~inp
    return {"added": int(added.sum()), "added_correct": int(np.sum(added & _dilate(ref, tol))),
            "missing": int(missing.sum()),
            "missing_recovered": int(np.sum(missing & _dilate(added.astype(np.uint8), tol)))}


def gap_metrics_from_counts(c: dict[str, int]) -> dict[str, float | None]:
    precision = safe_div(c["added_correct"], c["added"])
    recall = safe_div(c["missing_recovered"], c["missing"])
    if precision is None and recall is not None:
        f1 = 0.0 if c["missing"] > 0 else None   # nothing drawn although road was missing
    else:
        f1 = f1_score(precision, recall)
    return {"gap_precision": precision, "gap_recall": recall, "gap_f1": f1}


def evaluate(input_mask: np.ndarray, final_mask: np.ndarray, graph_before: nx.Graph,
             graph_after: nx.Graph, candidates: list, prefilter_rejected: int,
             reference: np.ndarray | None, tol: int, timings: dict[str, float]) -> dict[str, Any]:
    def n_comp(G: nx.Graph) -> int:
        return nx.number_connected_components(G) if G.number_of_nodes() else 0

    status = lambda s: sum(c.recon_status == s for c in candidates)  # noqa: E731
    user = lambda s: sum(c.user_decision == s for c in candidates)  # noqa: E731
    rec = [c for c in candidates if c.recon_status == "reconstructed"]
    out: dict[str, Any] = {
        "connectivity": {
            "mask_components_before": count_components(input_mask),
            "mask_components_after": count_components(final_mask),
            "graph_components_before": n_comp(graph_before),
            "graph_components_after": n_comp(graph_after),
            "candidate_gaps_detected": len(candidates),
            "excluded_endpoint_pairs": prefilter_rejected,
            "recommended": sum(c.recommendation == "recommended" for c in candidates),
            "uncertain": sum(c.recommendation == "uncertain" for c in candidates),
            "not_recommended": sum(c.recommendation == "not_recommended" for c in candidates),
            "approved": user("approved"), "rejected": user("rejected"), "skipped": user("skipped"),
            "pending": user("pending") + user("skipped"),
            "reconstructed": status("reconstructed"),
            "reconstruction_failed": status("failed"),
            "reconstruction_blocked": status("blocked"),
        },
        "proxy_metrics_no_ground_truth": {
            "added_road_pixels": int(np.sum((final_mask > 0) & (input_mask == 0))),
            "mean_cars_reconstructed": round(float(np.mean([c.cars for c in rec])), 4) if rec else None,
            "mean_segment_length_px": round(float(np.mean([c.recon_metrics["length"] for c in rec])), 2) if rec else None,
            "note": "Proxy values only. They are not accuracy measurements.",
        },
        "timings_seconds": {k: round(v, 4) for k, v in timings.items()},
        "has_reference": reference is not None,
    }
    if reference is not None:
        before_c, after_c = pixel_counts(input_mask, reference), pixel_counts(final_mask, reference)
        gc = gap_counts(input_mask, final_mask, reference, tol)
        out["pixel_metrics"] = {
            "before": {**metrics_from_counts(before_c), **relaxed_metrics(input_mask, reference, tol), "counts": before_c},
            "after": {**metrics_from_counts(after_c), **relaxed_metrics(final_mask, reference, tol), "counts": after_c},
            "tolerance_px": tol,
        }
        out["gap_metrics"] = {**gap_metrics_from_counts(gc), "counts": gc, "tolerance_px": tol}
        out["connectivity"]["reference_components"] = count_components(reference)
    return out


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested metrics -> one flat row for CSV export."""
    flat: dict[str, Any] = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten_metrics(v, key + "."))
        elif isinstance(v, (list, tuple)):
            flat[key] = ";".join(map(str, v))
        else:
            flat[key] = round(v, 6) if isinstance(v, float) else v
    return flat
