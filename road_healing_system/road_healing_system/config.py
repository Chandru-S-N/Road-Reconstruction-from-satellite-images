"""
Configuration.

* ``PipelineConfig`` holds every processing parameter, grouped by module, with
  validation. The web form is generated from ``PARAMETER_SCHEMA`` below.
* ``AppSettings`` holds server settings read from environment variables or a
  ``.env`` file (see ``.env.example``).

All processing defaults are design parameters chosen by reasoning and checked
on synthetic data only. They are not experimentally validated constants.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from core.errors import ConfigError

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Processing parameters
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessConfig:
    denoise: str = "bilateral"          # none | gaussian | median | bilateral
    denoise_strength: int = 5
    enhance_contrast: bool = True       # CLAHE on the L channel of Lab
    clahe_clip_limit: float = 2.0
    min_size: int = 32
    max_size: int = 6000


@dataclass
class SegmentationConfig:
    method: str = "contextual"          # contextual (default) | hsv | adaptive | otsu
    # --- contextual method (colour model + line-likeness + width and shape filters) ---
    candidate_sat_max: int = 70         # loose colour gate: roads are greyish
    seed_sat_max: int = 35              # strict colour gate used to learn the road colour model
    candidate_val_min: int = 70
    candidate_val_max: int = 225
    vegetation_exg: float = 0.08        # excess-green above this is never road
    water_blue_margin: int = 12         # blue exceeding red and green by this much is never road
    shadow_val_max: int = 60            # darker than this is never road
    likelihood_scale: float = 4.0
    ridge_sigmas: list = field(default_factory=lambda: [2.0, 4.0, 6.0])
    ridge_weight: float = 0.6           # how much line-likeness (Sato ridge filter) shapes the score
    score_threshold: float = 0.35
    max_road_width: int = 32            # wider regions (open ground, yards, fields) are removed
    min_elongation: float = 3.0         # length / width of a component (from its skeleton); roofs score low
    # --- threshold methods kept for comparison ---
    sat_max: int = 45
    val_min: int = 90
    val_max: int = 215
    otsu_dark_roads: bool = False
    adaptive_block: int = 51
    adaptive_offset: int = -5
    # --- shared clean-up ---
    open_kernel: int = 3
    close_kernel: int = 5
    min_component_area: int = 150
    remove_compact_blobs: bool = False
    blob_max_aspect: float = 2.5
    blob_min_extent: float = 0.75


@dataclass
class SkeletonConfig:
    fill_hole_area: int = 60
    smooth_kernel: int = 3
    prune_min_length: float = 6.0
    prune_width_factor: float = 1.0
    prune_iterations: int = 2
    min_component_length: float = 15.0
    direction_window: int = 20
    border_margin: int = 5


@dataclass
class GapConfig:
    max_gap_distance: float = 70.0
    min_gap_distance: float = 1.5
    direction_tolerance_deg: float = 60.0
    enable_t_junctions: bool = False    # endpoint -> side of a road creates a new junction: off (road-to-road only)
    t_junction_clearance: float = 10.0
    allow_same_component: bool = False  # optional: gaps inside an already-connected network (see README)
    min_detour_ratio: float = 2.0
    candidate_mode: str = "auto"        # auto (manual review happens in the decision stage)
    manual_pairs: list = field(default_factory=list)
    force_manual_accept: bool = False
    max_connections_per_endpoint: int = 1
    mutual_best: bool = True            # keep a pair only if each end is the other's best match
    min_width_ratio: float = 0.25       # narrower / wider below this: different kinds of road
    exclude_crossing: bool = True       # another road lying across the gap means a different road
    water_barrier_fraction: float = 0.5 # gap mostly water-like: not a hidden road
    # --- road-to-road gate (checked before CARS) ---
    min_fragment_length: float = 20.0   # a fragment shorter than this is not an identifiable road
    min_fragment_elongation: float = 2.5  # length / width of a fragment
    min_fragment_road_likelihood: float = 0.30  # image evidence: mean road-colour likelihood along the fragment
    min_end_road_likelihood: float = 0.25       # image evidence near the road end itself
    max_trajectory_bend_deg: float = 50.0       # the two trajectories must join without an S-bend
    require_feasible_path: bool = True          # a continuous connection must be constructible


@dataclass
class OcclusionConfig:
    corridor_margin: float = 3.0
    exg_threshold: float = 0.10
    shadow_relative_value: float = 0.6
    shadow_absolute_value: int = 70
    texture_window: int = 7
    texture_threshold: float = 18.0
    likelihood_scale: float = 4.0
    road_likelihood_threshold: float = 0.5
    weight_road_like: float = 0.9
    weight_vegetation: float = 1.0
    weight_shadow: float = 1.0
    weight_textured: float = 0.5
    weight_unexplained: float = 0.0
    weight_water: float = 0.0           # water explains nothing: a road does not hide under a river
    water_blue_margin: int = 12
    min_corridor_pixels: int = 15


@dataclass
class CARSConfig:
    weight_alignment: float = 0.25
    weight_occlusion: float = 0.30
    weight_width: float = 0.10
    weight_geometry: float = 0.10
    weight_network: float = 0.25
    acceptance_threshold: float = 0.65
    review_margin: float = 0.05          # CARS in [threshold - margin, threshold) -> manual review
    review_if_missing: bool = True       # accepted but with missing evidence -> manual review
    missing_policy: str = "neutral"      # neutral | pessimistic | renormalize
    neutral_value: float = 0.5
    cocircularity_tolerance_deg: float = 40.0
    width_tolerance: float = 0.8
    min_radius_factor: float = 1.5       # geometry: tightest plausible turn radius / road width
    curvature_tolerance: float = 0.5     # geometry: allowed curvature jump x width
    merge_score: float = 0.75
    loop_closure_score: float = 0.5
    shortcut_score: float = 0.1
    crossing_penalty: float = 0.6
    ambiguity_penalty: float = 0.15


@dataclass
class PathConfig:
    """Graph-based trellis reconstruction (dynamic programming on a DAG; no Dijkstra)."""
    station_spacing: float = 3.0        # px between stations along the expected road course
    offset_step: float = 1.0            # px between lateral positions at a station
    max_offset_widths: float = 1.5      # lateral search range = this x road width (at least 6 px)
    max_lateral_step: float = 2.0       # px the path may move sideways between two stations
    lambda_road: float = 1.0            # cost of crossing non-road-looking pixels
    lambda_guide: float = 2.0           # cost of leaving the smooth expected course
    guide_cap: float = 9.0
    guide_min_scale: float = 6.0
    lambda_direction: float = 1.0       # cost of turning away from the expected heading
    other_road_penalty: float = 6.0     # cost of running over an unrelated road
    max_length_ratio: float = 1.8       # path length / straight gap length
    max_mean_cost: float = 3.0          # PATH COST THRESHOLD: cost per pixel of path
    max_other_road_fraction: float = 0.15
    smoothing_window: int = 5


@dataclass
class ReconstructionConfig:
    width_scale: float = 1.0
    min_width: int = 3
    max_width: int = 40
    fallback_width: int = 8
    tangent_scale: float = 1.0           # reference curve shared by the features and the path guide


@dataclass
class EvaluationConfig:
    tolerance_px: int = 3


SECTIONS = {
    "preprocess": PreprocessConfig, "segmentation": SegmentationConfig, "skeleton": SkeletonConfig,
    "gap": GapConfig, "occlusion": OcclusionConfig, "cars": CARSConfig, "path": PathConfig,
    "reconstruction": ReconstructionConfig, "evaluation": EvaluationConfig,
}
FEATURES = ("A", "O", "W", "G", "N")
FEATURE_NAMES = {"A": "road alignment", "O": "occlusion evidence", "W": "width consistency",
                 "G": "local geometry", "N": "network connectivity"}
WEIGHT_FIELDS = {"A": "weight_alignment", "O": "weight_occlusion", "W": "weight_width",
                 "G": "weight_geometry", "N": "weight_network"}


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    gap: GapConfig = field(default_factory=GapConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    cars: CARSConfig = field(default_factory=CARSConfig)
    path: PathConfig = field(default_factory=PathConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def cars_weights(self) -> dict[str, float]:
        return {k: float(getattr(self.cars, f)) for k, f in WEIGHT_FIELDS.items()}

    def validate(self) -> list[str]:
        """Check every value and normalise the CARS weights. Returns notes; raises ConfigError."""
        e: list[str] = []
        p, s, k, g, o, c, r = (self.preprocess, self.segmentation, self.skeleton, self.gap,
                               self.occlusion, self.cars, self.reconstruction)
        pc = self.path
        if p.denoise not in {"none", "gaussian", "median", "bilateral"}:
            e.append("Denoise must be none, gaussian, median or bilateral.")
        if not 1 <= p.denoise_strength <= 31:
            e.append("Denoise strength must be between 1 and 31.")
        if s.method not in {"contextual", "hsv", "adaptive", "otsu"}:
            e.append("Segmentation method must be contextual, hsv, adaptive or otsu.")
        if not 0 <= s.candidate_sat_max <= 255 or not 0 <= s.seed_sat_max <= 255:
            e.append("Saturation limits must be between 0 and 255.")
        if s.candidate_val_min > s.candidate_val_max:
            e.append("Candidate brightness range is empty.")
        if not s.ridge_sigmas or any(not isinstance(v, (int, float)) or v <= 0 for v in s.ridge_sigmas):
            e.append("Ridge sigmas must be a list of positive numbers.")
        if not 0 <= s.ridge_weight <= 1 or not 0 < s.score_threshold < 1:
            e.append("Ridge weight must be in [0, 1] and the score threshold in (0, 1).")
        if s.max_road_width < 3:
            e.append("Maximum road width must be at least 3 pixels.")
        for name in ("sat_max", "val_min", "val_max"):
            if not 0 <= getattr(s, name) <= 255:
                e.append(f"Segmentation {name} must be between 0 and 255.")
        if s.val_min > s.val_max:
            e.append("Minimum brightness must not exceed maximum brightness.")
        if s.adaptive_block < 3 or s.adaptive_block % 2 == 0:
            e.append("Adaptive block size must be an odd number of at least 3.")
        if s.min_component_area < 0:
            e.append("Smallest road blob must be 0 or more.")
        if k.direction_window < 3:
            e.append("Direction window must be at least 3 pixels.")
        if g.max_gap_distance <= 0 or not 0 <= g.min_gap_distance < g.max_gap_distance:
            e.append("Gap distances must satisfy 0 <= minimum < maximum.")
        if not 0 < g.direction_tolerance_deg <= 180:
            e.append("Direction tolerance must be between 0 and 180 degrees.")
        if g.max_connections_per_endpoint < 1:
            e.append("Connections per endpoint must be at least 1.")
        for name in ("weight_road_like", "weight_vegetation", "weight_shadow", "weight_textured",
                     "weight_unexplained"):
            if not 0 <= getattr(o, name) <= 1:
                e.append(f"Occlusion {name} must be between 0 and 1.")
        weights = self.cars_weights()
        if any(w < 0 for w in weights.values()):
            e.append("CARS weights must not be negative.")
        total = sum(weights.values())
        if total <= 0:
            e.append("At least one CARS weight must be positive.")
        if not 0 <= c.acceptance_threshold <= 1:
            e.append("CARS threshold must be between 0 and 1.")
        if not 0 <= c.review_margin <= 0.5:
            e.append("Review margin must be between 0 and 0.5.")
        if c.missing_policy not in {"neutral", "pessimistic", "renormalize"}:
            e.append("Missing-evidence policy must be neutral, pessimistic or renormalize.")
        if not 0 <= c.neutral_value <= 1:
            e.append("Neutral value must be between 0 and 1.")
        if c.min_radius_factor <= 0 or c.curvature_tolerance <= 0:
            e.append("Geometry factors must be positive.")
        for name in ("lambda_road", "lambda_guide", "lambda_direction", "other_road_penalty"):
            if getattr(pc, name) < 0:
                e.append(f"Path cost {name} must not be negative (Dijkstra needs non-negative costs).")
        if pc.max_length_ratio < 1:
            e.append("Maximum path length ratio must be at least 1.")
        if pc.max_mean_cost <= 0:
            e.append("Path cost threshold must be positive.")
        if pc.station_spacing <= 0 or pc.offset_step <= 0 or pc.max_offset_widths <= 0 or pc.max_lateral_step <= 0:
            e.append("Trellis spacing, offset step, offset range and lateral step must be positive.")
        if not 0 <= g.min_fragment_road_likelihood <= 1 or not 0 <= g.min_end_road_likelihood <= 1:
            e.append("Fragment road-likelihood limits must be between 0 and 1.")
        if not 0 < g.min_width_ratio <= 1 or not 0 < g.water_barrier_fraction <= 1:
            e.append("Gap width ratio and water barrier fraction must be in (0, 1].")
        if r.min_width < 1 or r.max_width < r.min_width:
            e.append("Reconstruction widths must satisfy 1 <= minimum <= maximum.")
        if self.evaluation.tolerance_px < 0:
            e.append("Evaluation tolerance must be 0 or more.")
        if e:
            raise ConfigError(" ".join(e))
        notes = []
        if abs(total - 1.0) > 1e-9:
            for key, fname in WEIGHT_FIELDS.items():
                setattr(c, fname, weights[key] / total)
            notes.append(f"CARS weights summed to {total:.3f} and were normalised to 1.")
        return notes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        cfg = cls()
        if not isinstance(data, dict):
            raise ConfigError("Parameters must be a JSON object.")
        for section, values in data.items():
            if section not in SECTIONS:
                raise ConfigError(f"Unknown parameter section '{section}'.")
            if not isinstance(values, dict):
                raise ConfigError(f"Section '{section}' must be an object.")
            target = getattr(cfg, section)
            valid = {f.name: f for f in fields(target)}
            for key, value in values.items():
                if key not in valid:
                    raise ConfigError(f"Unknown parameter '{section}.{key}'.")
                default = getattr(target, key)
                setattr(target, key, _coerce(value, default, f"{section}.{key}"))
        return cfg

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def copy(self) -> "PipelineConfig":
        return PipelineConfig.from_dict(self.to_dict())


def _coerce(value: Any, default: Any, name: str) -> Any:
    """Convert form/JSON values to the type of the default; reject what does not fit."""
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if isinstance(default, int):
            f = float(value)
            if f != int(f):
                raise ValueError
            return int(f)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, list):
            if isinstance(value, str):
                value = [float(v) for v in value.split(",") if v.strip()]
            return list(value)
        return str(value)
    except (TypeError, ValueError):
        raise ConfigError(f"Parameter '{name}' has an invalid value: {value!r}.") from None


# --------------------------------------------------------------------------- #
# Parameters exposed in the web form: (section, field, label, kind, extra, help)
# --------------------------------------------------------------------------- #
PARAMETER_SCHEMA: list[dict[str, Any]] = [
    {"group": "Image processing", "items": [
        ("preprocess", "denoise", "Noise reduction", "select", ["none", "gaussian", "median", "bilateral"], ""),
        ("preprocess", "denoise_strength", "Noise filter size", "int", [1, 31, 2], ""),
        ("preprocess", "enhance_contrast", "Enhance contrast (CLAHE)", "bool", None, ""),
    ]},
    {"group": "Road segmentation", "items": [
        ("segmentation", "method", "Method", "select", ["contextual", "hsv", "adaptive", "otsu"],
         "contextual: colour model + line-likeness + width/shape filters (recommended). The others are simple thresholds."),
        ("segmentation", "candidate_sat_max", "Max road saturation (contextual)", "int", [0, 255, 1], "Raise if grey roads are missed."),
        ("segmentation", "max_road_width", "Widest road (px)", "int", [3, 200, 1],
         "Regions wider than this (open ground, fields, yards) are not roads."),
        ("segmentation", "min_elongation", "Min elongation (contextual)", "float", [0, 50, 0.5],
         "Compact shapes such as roofs score low; roads are long and thin."),
        ("segmentation", "score_threshold", "Road score threshold (contextual)", "float", [0.05, 0.95, 0.05], ""),
        ("segmentation", "ridge_weight", "Weight of line-likeness", "float", [0, 1, 0.05], ""),
        ("segmentation", "sat_max", "Max saturation (hsv)", "int", [0, 255, 1], ""),
        ("segmentation", "val_min", "Min brightness (hsv)", "int", [0, 255, 1], ""),
        ("segmentation", "val_max", "Max brightness (hsv)", "int", [0, 255, 1], ""),
        ("segmentation", "min_component_area", "Smallest road blob (px)", "int", [0, 5000, 10], ""),
    ]},
    {"group": "Skeleton and endpoints", "items": [
        ("skeleton", "prune_min_length", "Spur pruning length (px)", "float", [0, 50, 1], ""),
        ("skeleton", "min_component_length", "Smallest road fragment (px)", "float", [0, 200, 1], ""),
        ("skeleton", "direction_window", "Direction window (px)", "int", [3, 80, 1], ""),
        ("skeleton", "border_margin", "Border margin (px)", "int", [0, 50, 1],
         "Endpoints this close to the edge are roads leaving the image."),
    ]},
    {"group": "Gap detection", "items": [
        ("gap", "max_gap_distance", "Max gap distance (px)", "float", [5, 400, 5], ""),
        ("gap", "direction_tolerance_deg", "Direction tolerance (deg)", "float", [5, 180, 5], ""),
        ("gap", "allow_same_component", "Also look inside a connected network", "bool", None,
         "Optional loop closures. Measures the network detour with a graph distance (never used to draw roads)."),
        ("gap", "max_connections_per_endpoint", "Connections per endpoint", "int", [1, 4, 1], ""),
        ("gap", "mutual_best", "Pair each road end only with its best match", "bool", None,
         "Prevents candidates between every nearby endpoint."),
        ("gap", "min_width_ratio", "Min width ratio of a pair", "float", [0.05, 1, 0.05], ""),
        ("gap", "exclude_crossing", "Exclude pairs with another road between them", "bool", None, ""),
        ("gap", "min_fragment_length", "Shortest road fragment (px)", "float", [0, 200, 1], ""),
        ("gap", "min_fragment_road_likelihood", "Fragment must look like road (0-1)", "float", [0, 1, 0.05],
         "Image evidence along the fragment; the mask alone is not enough."),
        ("gap", "max_trajectory_bend_deg", "Max trajectory bend (deg)", "float", [5, 180, 5],
         "Larger bends would need an S-shaped join: not one continuous road."),
        ("gap", "enable_t_junctions", "Also allow end-to-road-side joins", "bool", None,
         "Off: only road-to-road disconnections between two road ends."),
    ]},
    {"group": "CARS weights and decision", "items": [
        ("cars", "weight_alignment", "Weight: alignment (A)", "float", [0, 1, 0.05], ""),
        ("cars", "weight_occlusion", "Weight: occlusion evidence (O)", "float", [0, 1, 0.05], ""),
        ("cars", "weight_width", "Weight: width consistency (W)", "float", [0, 1, 0.05], ""),
        ("cars", "weight_geometry", "Weight: local geometry (G)", "float", [0, 1, 0.05], ""),
        ("cars", "weight_network", "Weight: network connectivity (N)", "float", [0, 1, 0.05], ""),
        ("cars", "acceptance_threshold", "Acceptance threshold", "float", [0, 1, 0.01], ""),
        ("cars", "review_margin", "Review band below threshold", "float", [0, 0.5, 0.01],
         "Candidates this close below the threshold are flagged for manual review."),
        ("cars", "review_if_missing", "Review accepted gaps with missing evidence", "bool", None, ""),
        ("cars", "missing_policy", "Missing evidence counts as", "select", ["neutral", "pessimistic", "renormalize"],
         "neutral = 0.5, pessimistic = 0, renormalize = ignore the feature."),
    ]},
    {"group": "Occlusion cues", "items": [
        ("occlusion", "exg_threshold", "Vegetation (excess green) threshold", "float", [0, 0.5, 0.01], ""),
        ("occlusion", "shadow_relative_value", "Shadow: darker than x road brightness", "float", [0.1, 1, 0.05], ""),
        ("occlusion", "texture_threshold", "Texture threshold (grey std)", "float", [1, 80, 1], ""),
    ]},
    {"group": "Trellis path cost", "items": [
        ("path", "lambda_road", "Cost of non-road pixels", "float", [0, 10, 0.1], ""),
        ("path", "lambda_guide", "Cost of leaving the smooth curve", "float", [0, 10, 0.1], ""),
        ("path", "lambda_direction", "Cost of turning away", "float", [0, 10, 0.1], ""),
        ("path", "max_offset_widths", "Sideways search range (x road width)", "float", [0.2, 5, 0.1], ""),
        ("path", "other_road_penalty", "Cost of crossing another road", "float", [0, 50, 1], ""),
        ("path", "max_mean_cost", "Path cost threshold (per px)", "float", [0.5, 20, 0.1],
         "Separate from the CARS threshold: an approved gap whose best path costs more is not drawn."),
        ("path", "max_length_ratio", "Max path length / gap", "float", [1, 5, 0.1], ""),
        ("reconstruction", "width_scale", "Drawn width scale", "float", [0.3, 3, 0.1], ""),
    ]},
    {"group": "Evaluation", "items": [
        ("evaluation", "tolerance_px", "Metric tolerance (px)", "int", [0, 20, 1],
         "Buffer used by the relaxed and gap-focused metrics."),
    ]},
]


def schema_with_defaults() -> list[dict[str, Any]]:
    cfg = PipelineConfig()
    out = []
    for group in PARAMETER_SCHEMA:
        items = []
        for section, fname, label, kind, extra, help_ in group["items"]:
            items.append({"key": f"{section}.{fname}", "label": label, "kind": kind, "extra": extra,
                          "help": help_, "default": getattr(getattr(cfg, section), fname)})
        out.append({"group": group["group"], "items": items})
    return out


def config_from_flat(flat: dict[str, Any]) -> PipelineConfig:
    """Build a config from {"section.field": value} pairs sent by the web form."""
    nested: dict[str, dict[str, Any]] = {}
    for key, value in (flat or {}).items():
        if "." not in key:
            raise ConfigError(f"Unknown parameter '{key}'.")
        section, fname = key.split(".", 1)
        nested.setdefault(section, {})[fname] = value
    return PipelineConfig.from_dict(nested)


# --------------------------------------------------------------------------- #
# Server settings (.env / environment)
# --------------------------------------------------------------------------- #
def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines, # comments; real environment wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class AppSettings:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    max_upload_mb: int = 25
    storage_dir: Path = ROOT / "storage"
    samples_dir: Path = ROOT / "samples"
    keep_jobs: int = 50

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "AppSettings":
        _load_dotenv(env_file or ROOT / ".env")
        s = cls()
        s.host = os.environ.get("APP_HOST", s.host)
        s.port = int(os.environ.get("APP_PORT", s.port))
        s.debug = os.environ.get("APP_DEBUG", "false").lower() in {"1", "true", "yes"}
        s.max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", s.max_upload_mb))
        storage = os.environ.get("STORAGE_DIR")
        if storage:
            s.storage_dir = Path(storage) if Path(storage).is_absolute() else ROOT / storage
        s.keep_jobs = int(os.environ.get("KEEP_JOBS", s.keep_jobs))
        return s
