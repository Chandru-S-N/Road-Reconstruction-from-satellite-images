"""Shared fixtures. Puts the project root on sys.path."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import AppSettings, PipelineConfig  # noqa: E402
from core.pipeline import ReconstructionPipeline  # noqa: E402
from core.preprocessing import loaded_from_array  # noqa: E402
from core.synthetic_data import SCENARIOS  # noqa: E402


def run_scene(name, cfg=None, use_mask=True, use_image=True):
    scene = SCENARIOS[name]()
    pipe = ReconstructionPipeline(cfg or PipelineConfig())
    image = loaded_from_array(scene.image, f"{name}.png", synthetic=True) if use_image else None
    pipe.load(image, scene.observed_mask if use_mask else None, scene.reference_mask)
    pipe.run_all()
    return scene, pipe


def approve_supported(pipe):
    """Approve every gap the system does not advise against; reject the rest."""
    for c in pipe.state.candidates:
        pipe.decide_gap(c.id, "reject" if c.recommendation == "not_recommended" else "approve")


@pytest.fixture
def client(tmp_path):
    from app import create_app
    s = AppSettings()
    s.storage_dir = tmp_path / "storage"
    s.max_upload_mb = 2
    app = create_app(s)
    app.config["TESTING"] = True
    return app.test_client()
