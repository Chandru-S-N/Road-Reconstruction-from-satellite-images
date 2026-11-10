"""One-by-one Dijkstra reconstruction after approval."""

import networkx as nx
import numpy as np
import pytest

from config import PipelineConfig
from core.pipeline import DecisionRefused
from tests.conftest import run_scene


def test_no_dijkstra_anywhere_and_only_the_approved_gap_changes(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("a Dijkstra-based routine was called")
    for name in ("dijkstra_path", "dijkstra_path_length", "single_source_dijkstra", "single_source_dijkstra_path",
                 "single_source_dijkstra_path_length", "multi_source_dijkstra", "bidirectional_dijkstra", "astar_path",
                 "shortest_path", "shortest_path_length"):
        monkeypatch.setattr(nx, name, forbidden)
    _, pipe = run_scene("07_multiple_components")
    st = pipe.state
    g1, g2 = st.candidates[0], st.candidates[1]
    assert g1.preview_xy is not None                               # the preview exists before approval
    before = st.recon.current_mask.copy()
    pipe.decide_gap(g1.id, "approve")
    assert g1.recon_status == "reconstructed" and g2.recon_status == "not_attempted"
    assert g1.recon_metrics["stations"] > 2 and g1.recon_metrics["edges"] > g1.recon_metrics["stations"]
    changed = np.argwhere(st.recon.current_mask != before)
    x0, y0, x1, y1 = st.gap_regions[g1.id]
    assert changed[:, 1].min() >= x0 and changed[:, 1].max() <= x1 and changed[:, 0].min() >= y0 and changed[:, 0].max() <= y1
    assert pipe.summary()["components_after"] == pipe.summary()["components_before"] - 1
    assert sum(d.get("kind") == "reconstructed" for _, _, d in st.recon.graph.edges(data=True)) == 1
    assert {"result", "mask_after", "graph_after"} <= set(st.gap_images[g1.id])


def test_trellis_graph_is_a_dag_and_optimal_on_it():
    from core.reconstruction import preview_gap
    _, pipe = run_scene("02_curved_gap")
    st = pipe.state
    c = st.candidates[0]
    res = preview_gap(c, st.context, st.recon, pipe.config)
    assert res.valid and np.allclose(res.raw_xy[0], c.source_xy) and np.allclose(res.raw_xy[-1], c.target_xy)
    assert res.reference is None or res.reference["total_cost"] >= res.metrics["total_cost"] - 1e-6


def test_reject_and_skip_change_nothing():
    _, pipe = run_scene("01_straight_gap")
    before = pipe.state.recon.current_mask.copy()
    gid = pipe.state.candidates[0].id
    pipe.decide_gap(gid, "skip")
    pipe.decide_gap(gid, "reject")
    assert np.array_equal(before, pipe.state.recon.current_mask)
    assert pipe.summary()["rejected"] == 1 and pipe.summary()["reconstructed"] == 0


def test_not_recommended_cannot_be_approved_and_reconstructed_cannot_change():
    _, pipe = run_scene("04_width_mismatch")
    assert pipe.state.candidates[0].recommendation == "not_recommended"
    with pytest.raises(DecisionRefused):
        pipe.decide_gap(pipe.state.candidates[0].id, "approve")
    _, pipe = run_scene("01_straight_gap")
    gid = pipe.state.candidates[0].id
    pipe.decide_gap(gid, "approve")
    with pytest.raises(DecisionRefused):
        pipe.decide_gap(gid, "reject")


def test_failed_path_leaves_mask_unchanged_and_is_not_counted_as_success():
    cfg = PipelineConfig()
    _, pipe = run_scene("01_straight_gap", cfg)
    c = pipe.state.candidates[0]
    pipe.config.path.max_mean_cost = 0.5                   # tightened after the analysis: the approval must fail
    pipe.decide_gap(c.id, "approve")
    assert c.recon_status == "failed" and c.recon_explanation["succeeded"] is False
    assert int((pipe.state.recon.recon_only > 0).sum()) == 0
    assert pipe.summary()["reconstructed"] == 0 and pipe.summary()["failed"] == 1


def test_path_explanation_reports_actual_cost():
    _, pipe = run_scene("02_curved_gap")
    c = pipe.state.candidates[0]
    pipe.decide_gap(c.id, "approve")
    m = c.recon_metrics
    assert sum(m["breakdown"].values()) == pytest.approx(m["total_cost"], rel=0.02)
    text = " ".join(p["en"] for p in c.recon_explanation["points"])
    assert f"{m['total_cost']:.1f}" in text and str(m["stations"]) in text and "trellis" in text and "Dijkstra" not in text
    assert all(p["ta"] for p in c.recon_explanation["points"])


def test_blocked_when_road_end_already_connected():
    _, pipe = run_scene("01_straight_gap")
    st = pipe.state
    c = st.candidates[0]
    st.recon.usage[c.source] = ["G99"]
    pipe.decide_gap(c.id, "approve")
    assert c.recon_status == "blocked" and "G99" in c.recon_reason
