"""CARS formula (unchanged) and recommendation rules."""

import pytest

from config import FEATURES, PipelineConfig
from core.cars_scoring import compute_cars
from core.decision_engine import decide
from core.gap_detection import Candidate


def cfg():
    c = PipelineConfig()
    c.validate()
    return c


def test_formula_and_bounds():
    c = cfg()
    assert c.cars_weights() == {"A": 0.25, "O": 0.30, "W": 0.10, "G": 0.10, "N": 0.25}
    scores = {"A": 0.8, "O": 0.5, "W": 1.0, "G": 0.2, "N": 0.75}
    val, d = compute_cars(scores, c)
    assert val == pytest.approx(sum(c.cars_weights()[k] * scores[k] for k in FEATURES))
    assert compute_cars({k: 1.0 for k in FEATURES}, c)[0] == pytest.approx(1.0)


def test_missing_evidence_is_never_strong_positive():
    c = cfg()
    val, d = compute_cars({"A": 1, "O": None, "W": 1, "G": 1, "N": 1}, c)
    assert d["missing_filled"] == ["O"] and val < 1.0


def _cand(cid, cars, s=1, t=2, missing=()):
    c = Candidate(cid, "endpoint_pair", s, t, (0, 0), (30, 0), 30.0)
    c.cars = cars
    c.scores = {k: (None if k in missing else 0.5) for k in FEATURES}
    c.cars_details = {"contributions": {}, "weights_used": {}, "missing_filled": list(missing), "missing_policy": "neutral"}
    c.features = {"duplicate_edge": False}
    return c


def test_recommendation_levels():
    a, b, r, m = _cand("G01", 0.8, 1, 2), _cand("G02", 0.62, 3, 4), _cand("G03", 0.3, 5, 6), _cand("G04", 0.9, 7, 8, ("O",))
    decide([a, b, r, m], cfg())
    assert [x.recommendation for x in (a, b, r, m)] == ["recommended", "uncertain", "not_recommended", "uncertain"]
    assert m.rec_basis == "missing" and b.rec_basis == "band"


def test_competing_gaps_leave_the_weaker_one_to_the_user():
    best, other = _cand("G01", 0.9, 1, 2), _cand("G02", 0.8, 1, 3)
    decide([other, best], cfg())
    assert best.recommendation == "recommended"
    assert other.recommendation == "uncertain" and other.rec_basis == "conflict" and other.rec_detail == "G01"
