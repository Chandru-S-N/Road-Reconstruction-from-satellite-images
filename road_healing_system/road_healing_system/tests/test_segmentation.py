"""Road extraction: the contextual method must not label ground, roofs, car parks or water as road."""

import numpy as np

from config import PipelineConfig
from core.evaluation import metrics_from_counts, pixel_counts
from core.preprocessing import preprocess_image
from core.segmentation import segment_roads
from core.synthetic_data import SCENARIOS


def _seg(name, method):
    cfg = PipelineConfig()
    cfg.segmentation.method = method
    scene = SCENARIOS[name]()
    pre, _ = preprocess_image(scene.image, cfg.preprocess)
    return scene, segment_roads(pre, cfg.segmentation)


def test_contextual_rejects_non_road_regions_where_threshold_fails():
    scene, ctx = _seg("11_non_road_distractors", "contextual")
    _, hsv = _seg("11_non_road_distractors", "hsv")
    ref = scene.observed_mask
    p_ctx = metrics_from_counts(pixel_counts(ctx.mask, ref))
    p_hsv = metrics_from_counts(pixel_counts(hsv.mask, ref))
    assert p_hsv["precision"] < 0.5                     # the threshold takes ground, roof and car park
    assert p_ctx["precision"] > 0.95 and p_ctx["recall"] > 0.85


def test_mask_is_binary_and_reports_its_steps():
    _, r = _seg("02_curved_gap", "contextual")
    assert set(np.unique(r.mask)) <= {0, 255}
    for k in ("seg_candidates", "seg_likelihood", "seg_line_likeness", "seg_removed"):
        assert k in r.intermediates
    assert "excluded_vegetation_pct" in r.summary()


def test_water_is_excluded():
    scene, r = _seg("11_non_road_distractors", "contextual")
    assert int((r.mask[30:62, :] > 0).sum()) == 0      # the river band
