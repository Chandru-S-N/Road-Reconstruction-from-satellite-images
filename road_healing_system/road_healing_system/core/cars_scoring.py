"""
Stage 8 - Context-Aware Reconnection Score (CARS).

    CARS = wA*A + wO*O + wW*W + wG*G + wN*N
    weights >= 0 and normalised to sum to 1; every score is in [0, 1]

Missing evidence (a score of None) is handled by ``cars.missing_policy``:
    neutral      use ``neutral_value`` (0.5): no evidence either way (default)
    pessimistic  use 0.0: missing evidence counts against the connection
    renormalize  drop the feature and rescale the other weights (can be optimistic)
Missing evidence is never filled with a strong positive value, and the filled
features are recorded for every candidate.
"""

from __future__ import annotations

from typing import Any

from config import FEATURES, PipelineConfig
from core.gap_detection import Candidate


def compute_cars(scores: dict[str, float | None], cfg: PipelineConfig) -> tuple[float, dict[str, Any]]:
    weights = cfg.cars_weights()
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    policy = cfg.cars.missing_policy
    used: dict[str, float] = {}
    values: dict[str, float] = {}
    filled: list[str] = []
    for k in FEATURES:
        s = scores.get(k)
        if s is None:
            filled.append(k)
            if policy == "renormalize":
                continue
            s = cfg.cars.neutral_value if policy == "neutral" else 0.0
        values[k] = float(min(max(s, 0.0), 1.0))
        used[k] = weights[k]
    if policy == "renormalize":
        tot = sum(used.values())
        used = {k: v / tot for k, v in used.items()} if tot > 0 else {}
    contributions = {k: used[k] * values[k] for k in used}
    cars = float(min(max(sum(contributions.values()), 0.0), 1.0))
    return cars, {"weights_used": used, "values_used": values, "contributions": contributions,
                  "missing_filled": filled, "missing_policy": policy}


def score_candidates(candidates: list[Candidate], cfg: PipelineConfig) -> None:
    for c in candidates:
        c.cars, c.cars_details = compute_cars(c.scores, cfg)
