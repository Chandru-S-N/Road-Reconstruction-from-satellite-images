"""Genuine gap identification."""

import cv2
import numpy as np

from config import PipelineConfig
from core.gap_detection import describe_fragments, detect_candidates
from core.graph_construction import build_road_graph
from core.skeletonization import distance_map, skeletonize_mask, trace_skeleton


def build(mask):
    cfg = PipelineConfig()
    sk, _ = skeletonize_mask(mask, cfg.skeleton)
    b = build_road_graph(trace_skeleton(sk), distance_map(mask), mask.shape, cfg.skeleton)
    _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return b, cfg, {"mask_labels": labels, "water": None}


def test_one_gap_between_collinear_fragments_named_g01():
    m = np.zeros((200, 300), np.uint8)
    cv2.line(m, (0, 100), (120, 100), 255, 12)
    cv2.line(m, (170, 100), (299, 100), 255, 12)
    b, cfg, scene = build(m)
    res = detect_candidates(b, cfg.gap, scene)
    assert [c.id for c in res.candidates] == ["G01"]
    assert sum(f["is_fragment"] for f in describe_fragments(b)) == 2


def test_mutual_best_keeps_only_the_best_pair():
    m = np.zeros((240, 320), np.uint8)
    cv2.line(m, (0, 120), (120, 120), 255, 12)          # A, end at x=120
    cv2.line(m, (165, 120), (319, 120), 255, 12)        # B, collinear, 45 px
    cv2.line(m, (170, 150), (170, 239), 255, 10)        # C, end pointing up near the gap
    b, cfg, scene = build(m)
    cfg.gap.enable_t_junctions = False
    res = detect_candidates(b, cfg.gap, scene)
    assert len(res.candidates) == 1
    assert any("matches road end" in r["reason"] for r in res.prefilter_rejected)


def test_road_between_the_ends_excludes_the_pair():
    m = np.zeros((200, 300), np.uint8)
    cv2.line(m, (0, 100), (120, 100), 255, 12)
    cv2.line(m, (172, 100), (299, 100), 255, 12)        # ends about 54 px apart, within range
    cv2.line(m, (146, 0), (146, 199), 255, 8)           # an unrelated road running through the gap
    b, cfg, scene = build(m)
    cfg.gap.enable_t_junctions = False
    res = detect_candidates(b, cfg.gap, scene)
    assert res.candidates == []
    assert any("another road lies across" in r["reason"] for r in res.prefilter_rejected)


def test_very_different_widths_are_excluded():
    m = np.zeros((200, 300), np.uint8)
    cv2.line(m, (0, 100), (120, 100), 255, 30)
    cv2.line(m, (170, 100), (299, 100), 255, 5)
    b, cfg, scene = build(m)
    res = detect_candidates(b, cfg.gap, scene)
    assert res.candidates == []
    assert any("too different" in r["reason"] for r in res.prefilter_rejected)


def test_far_and_facing_away_ends_produce_nothing():
    m = np.zeros((200, 400), np.uint8)
    cv2.line(m, (0, 100), (80, 100), 255, 12)
    cv2.line(m, (250, 100), (399, 100), 255, 12)
    b, cfg, scene = build(m)
    assert detect_candidates(b, cfg.gap, scene).candidates == []


def test_s_bend_pairs_and_non_road_fragments_are_excluded_before_cars():
    from core.gap_detection import fragment_evidence
    m = np.zeros((240, 320), np.uint8)
    cv2.line(m, (0, 100), (130, 100), 255, 12)
    cv2.line(m, (170, 140), (319, 140), 255, 12)          # offset parallel: an S-bend would be needed
    b, cfg, scene = build(m)
    res = detect_candidates(b, cfg.gap, scene)
    assert res.candidates == [] and any("S-shaped" in r["reason"] for r in res.prefilter_rejected)
    m = np.zeros((200, 300), np.uint8)
    cv2.line(m, (0, 100), (120, 100), 255, 12)
    cv2.line(m, (170, 100), (299, 100), 255, 12)
    b, cfg, scene = build(m)
    lik = np.zeros(m.shape, np.float32)
    lik[:, :140] = 0.9                                     # only the left fragment looks like road in the image
    scene["evidence"] = fragment_evidence(b, lik, cfg.gap)
    res = detect_candidates(b, cfg.gap, scene)
    assert res.candidates == [] and any("does not look like road" in r["reason"] for r in res.prefilter_rejected)


def test_infeasible_connection_never_reaches_cars_and_occluded_gaps_come_first():
    from config import PipelineConfig
    from tests.conftest import run_scene
    cfg = PipelineConfig()
    cfg.path.max_mean_cost = 0.5                           # no continuous route can be this cheap
    _, pipe = run_scene("01_straight_gap", cfg)
    assert pipe.state.candidates == []
    assert any("no continuous road trajectory" in r["reason"] for r in pipe.state.gaps.prefilter_rejected)
    _, pipe = run_scene("07_multiple_components")
    pri = [c.priority for c in pipe.state.candidates]
    assert pri == sorted(pri, key=lambda p: {"occlusion": 0, "visible road": 1, "other": 2}[p])
