"""
Write the synthetic test scenes to samples/.

    python scripts/generate_samples.py

Each folder holds image.png (the aerial-style picture with occluded roads),
observed_mask.png (the road mask with gaps), reference_mask.png (the complete
roads, ground truth) and meta.json (description, expected outcome, and
"synthetic": true). These are procedurally generated, not real imagery.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.synthetic_data import SCENARIOS, random_scene, save_scene, write_invalid_image  # noqa: E402


def generate(folder: Path, n_random: int = 3) -> list[Path]:
    folder = Path(folder)
    written = [save_scene(fn(), folder) for fn in SCENARIOS.values()]
    written += [save_scene(random_scene(100 + i), folder) for i in range(n_random)]
    written.append(write_invalid_image(folder))
    return written


if __name__ == "__main__":
    out = generate(ROOT / "samples")
    print(f"Wrote {len(out)} synthetic sample folders to {ROOT / 'samples'}")
