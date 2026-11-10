"""Explanations are derived from measured values, in both languages."""

from core.explanations import final_voice
from tests.conftest import approve_supported, run_scene


def test_every_statement_exists_in_english_and_tamil_and_matches_the_gap():
    _, pipe = run_scene("07_multiple_components")
    for c in pipe.state.candidates:
        x = c.xai
        assert x["points"] and all(p["en"] and p["ta"] for p in x["points"])
        for k in ("analysis", "cars", "decision", "reconstruct", "review"):
            assert c.id in x["voice"][k]["en"] and c.id in x["voice"][k]["ta"]
        assert f"{c.cars:.2f}" in x["voice"]["cars"]["en"]


def test_vegetation_is_only_mentioned_when_measured():
    _, pipe = run_scene("01_straight_gap")                          # gap under tree canopy
    c = pipe.state.candidates[0]
    assert c.features["frac_vegetation"] >= 0.25 and any("Vegetation-like" in p["en"] for p in c.xai["points"])
    _, pipe = run_scene("04_width_mismatch")                        # gap across open ground
    c = pipe.state.candidates[0]
    assert c.features.get("frac_vegetation", 0) < 0.25
    assert not any("Vegetation-like" in p["en"] for p in c.xai["points"])
    assert c.recommendation == "not_recommended" and "insufficient evidence" in c.xai["conclusion"]["en"]


def test_numeric_and_language_parts_are_separate():
    _, pipe = run_scene("02_curved_gap")
    num = pipe.state.candidates[0].xai["numeric"]
    assert num["formula"].startswith("CARS =") and len(num["rows"]) == 5


def test_final_summary_voice_uses_actual_counts():
    _, pipe = run_scene("07_multiple_components")
    approve_supported(pipe)
    s = pipe.summary()
    v = final_voice(s)
    assert f"{s['reconstructed']} were reconstructed" in v["en"] and str(s["reconstructed"]) in v["ta"]


def test_fragment_statement_uses_image_measurements():
    _, pipe = run_scene("01_straight_gap")
    c = pipe.state.candidates[0]
    f = [p for p in c.xai["points"] if p["feature"] == "F"]
    assert f and "road-colour likelihood" in f[0]["en"] and c.fragments_info["A"]["road_likelihood"] is not None
    assert c.priority == "occlusion"
