"""End-to-end behaviour: analysis waits for the user; approved gaps are rebuilt one at a time."""

import numpy as np
import pytest

from core.errors import StageOrderError
from core.pipeline import AUTO_STAGES, ReconstructionPipeline
from core.preprocessing import loaded_from_array
from core.synthetic_data import SCENARIOS
from tests.conftest import approve_supported, run_scene


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("use_mask", [True, False])
def test_scenarios(name, use_mask):
    scene, pipe = run_scene(name, use_mask=use_mask)
    st = pipe.state
    assert int((st.recon.recon_only > 0).sum()) == 0                 # nothing before approval
    assert all(c.user_decision == "pending" for c in st.candidates)
    approve_supported(pipe)
    expected = scene.expected["true_gaps" if use_mask else "true_gaps_when_segmented"]
    assert pipe.summary()["reconstructed"] == expected, [(c.id, c.recommendation, c.cars, c.reason) for c in st.candidates]
    assert np.all(st.recon.current_mask[st.mask > 0] == 255)          # original roads preserved
    if expected and use_mask:
        g = st.metrics["gap_metrics"]
        assert g["gap_precision"] >= 0.9 and g["gap_recall"] >= 0.9


def test_analysis_stops_before_review_and_completes_after_all_decisions():
    _, pipe = run_scene("07_multiple_components")
    assert pipe.state.completed == AUTO_STAGES
    ids = [c.id for c in pipe.state.candidates]
    pipe.decide_gap(ids[0], "approve")
    assert "final" not in pipe.state.completed
    for gid in ids[1:]:
        pipe.decide_gap(gid, "skip")
    assert pipe.state.completed[-2:] == ["review", "final"]
    assert pipe.summary()["pending"] == len(ids) - 1


def test_original_image_and_mask_are_never_modified():
    scene = SCENARIOS["01_straight_gap"]()
    image = loaded_from_array(scene.image, "a.png")
    before_img, before_mask = image.image.copy(), scene.observed_mask.copy()
    pipe = ReconstructionPipeline()
    pipe.load(image, scene.observed_mask, scene.reference_mask)
    pipe.run_all()
    approve_supported(pipe)
    assert np.array_equal(image.image, before_img) and np.array_equal(scene.observed_mask, before_mask)
    assert not pipe.state.mask.flags.writeable


def test_stage_order_and_review_before_run():
    scene = SCENARIOS["01_straight_gap"]()
    pipe = ReconstructionPipeline()
    pipe.load(loaded_from_array(scene.image, "a.png"), scene.observed_mask)
    with pytest.raises(StageOrderError):
        pipe.run_stage("cars")
    with pytest.raises(StageOrderError):
        pipe.decide_gap("G01", "approve")


def test_mask_only_input_is_uncertain_not_recommended_blindly():
    _, pipe = run_scene("01_straight_gap", use_image=False)
    c = pipe.state.candidates[0]
    assert c.scores["O"] is None and c.recommendation == "uncertain" and c.rec_basis == "missing"
