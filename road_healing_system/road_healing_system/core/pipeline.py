"""
Pipeline with a user-controlled review.

Automatic stages (each consumes the actual output of the previous one):
    1 upload -> 2 preprocess -> 3 segment -> 4 skeleton -> 5 graph -> 6 gaps
    -> 7 features -> 8 cars -> 9 recommend
Then the pipeline waits. Each gap is reconstructed only when the user approves it
(``decide_gap``), one at a time, with the graph-based trellis method on the current road mask.
    10 review (per gap: approve / reject / skip) -> 11 final (all gaps decided)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from config import PipelineConfig
from core import explanations as xai
from core import gap_visuals as gv
from core.cars_scoring import score_candidates
from core.context_features import (AppearanceMaps, FeatureContext, candidate_geometry, compute_appearance,
                                   extract_all_features, make_context)
from core.decision_engine import decide, recommendation_counts
from core.endpoint_extraction import detect_keypoints
from core.errors import MaskError, ProcessingError, RoadHealingError, StageOrderError
from core.evaluation import evaluate
from core.gap_detection import (Candidate, GapDetectionResult, describe_fragments, detect_candidates,
                                fragment_evidence)
from core.graph_construction import GraphBuildResult, build_road_graph, graph_summary, rasterize_graph
from core.preprocessing import LoadedImage, preprocess_image
from core.reconstruction import ReconstructionState, path_thickness, preview_gap, reconstruct_gap
from core.segmentation import SegmentationResult, segment_roads
from core.skeletonization import distance_map, skeletonize_mask, trace_skeleton
from core.utils import StageTimer, make_logger

STAGES = ["upload", "preprocess", "segment", "skeleton", "graph", "gaps", "features", "cars", "recommend",
          "review", "final"]
AUTO_STAGES = STAGES[:STAGES.index("recommend") + 1]
STAGE_LABELS = {
    "upload": "Satellite image upload", "preprocess": "Image processing", "segment": "Road extraction",
    "skeleton": "Skeleton & endpoints", "graph": "Road graph", "gaps": "Fragments & gap detection",
    "features": "Context features", "cars": "CARS scoring", "recommend": "Recommendations",
    "review": "User review & reconstruction", "final": "Final road network",
}
ProgressFn = Callable[[str, str], None]


class DecisionRefused(RoadHealingError):
    """A user action that the rules do not allow (e.g. approving a not-recommended gap)."""


@dataclass
class PipelineState:
    config: PipelineConfig
    image: LoadedImage | None = None
    uploaded_mask: np.ndarray | None = None
    reference: np.ndarray | None = None
    preprocessed: np.ndarray | None = None
    preprocess_info: dict[str, Any] = field(default_factory=dict)
    segmentation: SegmentationResult | None = None
    mask: np.ndarray | None = None
    mask_source: str = ""
    skeleton_raw: np.ndarray | None = None
    skeleton_method: str = ""
    keypoints: dict[str, list] = field(default_factory=dict)
    dist: np.ndarray | None = None
    build: GraphBuildResult | None = None
    skeleton: np.ndarray | None = None
    appearance: AppearanceMaps | None = None
    mask_labels: np.ndarray | None = None
    fragments: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    gaps: GapDetectionResult | None = None
    context: FeatureContext | None = None
    candidates: list[Candidate] = field(default_factory=list)
    decision_summary: dict[str, Any] = field(default_factory=dict)
    recon: ReconstructionState | None = None
    components_initial: dict[str, int] = field(default_factory=dict)
    gap_regions: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    gap_images: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    decision_log: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    completed: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        if self.image is not None:
            return (self.image.height, self.image.width)
        if self.uploaded_mask is not None:
            return self.uploaded_mask.shape[:2]
        raise ProcessingError("Nothing has been uploaded.")

    @property
    def name(self) -> str:
        return self.image.name if self.image is not None else "uploaded_mask.png"

    def gap(self, gap_id: str) -> Candidate:
        for c in self.candidates:
            if c.id == gap_id:
                return c
        raise ProcessingError(f"Unknown gap '{gap_id}'.")


class ReconstructionPipeline:
    def __init__(self, config: PipelineConfig | None = None, on_progress: ProgressFn | None = None) -> None:
        self.config = (config or PipelineConfig()).copy()
        self.config_notes = self.config.validate()
        self.logger, self.log_handler = make_logger()
        self.timer = StageTimer()
        self.state = PipelineState(self.config)
        self.on_progress = on_progress
        for note in self.config_notes:
            self.logger.info(note)

    @property
    def log_lines(self) -> list[str]:
        return self.log_handler.lines

    # ------------------------------------------------------------------ #
    def load(self, image: LoadedImage | None, mask: np.ndarray | None = None,
             reference: np.ndarray | None = None) -> None:
        t0 = time.perf_counter()
        if image is None and mask is None:
            raise ProcessingError("Upload a satellite image (or a road mask).")
        shape = (image.height, image.width) if image is not None else mask.shape
        if mask is not None and mask.shape != shape:
            raise MaskError("The road mask and the image have different sizes.")
        if reference is not None and reference.shape != shape:
            raise MaskError("The reference mask and the image have different sizes.")
        self.state = PipelineState(self.config, image=image, uploaded_mask=mask, reference=reference)
        self.timer = StageTimer()
        self.timer.timings["upload"] = time.perf_counter() - t0
        self.state.completed = ["upload"]
        parts = [f"image {image.name} ({image.width}x{image.height})"] if image is not None else []
        if image is not None:
            for note in image.notes:
                self.logger.info(note)
        if mask is not None:
            parts.append("road mask (used instead of road extraction)")
        if reference is not None:
            parts.append("reference mask (accuracy metrics enabled)")
        self.logger.info("Uploaded " + ", ".join(parts) + ".")
        self._progress("upload", "done")

    def _progress(self, stage: str, status: str) -> None:
        if self.on_progress:
            self.on_progress(stage, status)

    def _require(self, stage: str) -> None:
        missing = [s for s in STAGES[:STAGES.index(stage)] if s not in self.state.completed]
        if missing:
            raise StageOrderError(f"'{STAGE_LABELS[stage]}' needs '{STAGE_LABELS[missing[0]]}' first.")

    def run_stage(self, stage: str) -> None:
        if stage not in AUTO_STAGES or stage == "upload":
            raise ProcessingError(f"'{stage}' is not an automatic stage.")
        self._require(stage)
        self._progress(stage, "running")
        t0 = time.perf_counter()
        try:
            with self.timer.time(stage):
                getattr(self, f"_stage_{stage}")()
        except Exception as exc:
            self.logger.error(f"{STAGE_LABELS[stage]} failed: {exc}")
            self._progress(stage, "error")
            raise
        i = STAGES.index(stage)
        self.state.completed = [s for s in self.state.completed if STAGES.index(s) < i] + [stage]
        self.logger.info(f"{STAGE_LABELS[stage]}: done in {time.perf_counter() - t0:.2f} s.")
        self._progress(stage, "done")

    def run_all(self) -> PipelineState:
        for stage in AUTO_STAGES[1:]:
            if stage not in self.state.completed:
                self.run_stage(stage)
        self._update_review_progress()
        return self.state

    # ------------------------------------------------------------------ #
    # automatic stages
    # ------------------------------------------------------------------ #
    def _stage_preprocess(self) -> None:
        st = self.state
        if st.image is None:
            st.preprocessed, st.preprocess_info = None, {"steps": ["skipped (no image)"]}
            return
        st.preprocessed, st.preprocess_info = preprocess_image(st.image.image, self.config.preprocess)
        self.logger.info("Image processing: " + ", ".join(st.preprocess_info["steps"]) + ".")

    def _stage_segment(self) -> None:
        st = self.state
        if st.uploaded_mask is not None:
            st.mask, st.mask_source, st.segmentation = st.uploaded_mask.copy(), "uploaded road mask", None
            self.logger.info("Using the uploaded road mask instead of extracting roads from the image.")
        else:
            st.segmentation = segment_roads(st.preprocessed, self.config.segmentation)
            st.mask = st.segmentation.mask
            st.mask_source = f"road extraction ({st.segmentation.method})"
            for note in st.segmentation.notes:
                self.logger.warning(note)
        if not (st.mask > 0).any():
            raise ProcessingError("The road mask has no road pixels. Adjust the road extraction parameters "
                                  "or upload a road mask.")
        st.mask.setflags(write=False)                      # the original road mask is never modified
        self.logger.info(f"Road mask: {(st.mask > 0).mean():.1%} of pixels are road.")

    def _stage_skeleton(self) -> None:
        st = self.state
        st.skeleton_raw, st.skeleton_method = skeletonize_mask(st.mask, self.config.skeleton)
        st.dist = distance_map(st.mask)
        if not st.skeleton_raw.any():
            raise ProcessingError("Skeletonization produced no pixels; the road mask may be empty or too thin.")
        st.keypoints = detect_keypoints(st.skeleton_raw)
        self.logger.info(f"Skeleton: {int(st.skeleton_raw.sum())} px; {len(st.keypoints['endpoints'])} raw endpoints, "
                         f"{len(st.keypoints['intersections'])} intersections.")

    def _stage_graph(self) -> None:
        st = self.state
        st.build = build_road_graph(trace_skeleton(st.skeleton_raw), st.dist, st.shape, self.config.skeleton)
        st.skeleton = rasterize_graph(st.build.graph, st.shape)
        s = graph_summary(st.build.graph)
        self.logger.info(f"Graph: {s['nodes']} nodes, {s['edges']} edges, {s['endpoints']} endpoints, "
                         f"{s['junctions']} intersections, {s['components']} components.")

    def _stage_gaps(self) -> None:
        st = self.state
        has_color = st.image is not None and not st.image.was_grayscale
        st.appearance = compute_appearance(st.preprocessed, st.mask, self.config, has_color)
        for note in st.appearance.notes:
            self.logger.warning(note)
        _, labels = cv2.connectedComponents((st.mask > 0).astype(np.uint8), connectivity=8)
        st.mask_labels = labels.astype(np.int32)
        st.fragments = describe_fragments(st.build)
        st.evidence = fragment_evidence(st.build, st.appearance.likelihood, self.config.gap)
        for f in st.fragments:
            f.update({k: v for k, v in st.evidence["fragments"].get(f["component"], {}).items()
                      if k in ("road_likelihood", "elongation", "width_px", "verified", "reasons")})
        water = (st.appearance.category == 5) if st.appearance.category is not None else None
        st.gaps = detect_candidates(st.build, self.config.gap, {"mask_labels": st.mask_labels, "water": water,
                                                                "evidence": st.evidence})
        s = st.gaps.summary()
        n_frag = sum(f["is_fragment"] for f in st.fragments)
        self.logger.info(f"{n_frag} fragments with open ends; {s['candidates']} candidate gaps; "
                         f"{s['prefilter_rejected']} endpoint pairs excluded; {s['border_endpoints_excluded']} border "
                         "endpoints are roads leaving the image.")

    def _stage_features(self) -> None:
        st = self.state
        st.context = make_context(st.build, st.mask, st.dist, st.appearance, self.config)
        st.recon = ReconstructionState.start(st.mask, st.build.graph)
        cands = st.gaps.candidates
        extract_all_features(cands, st.context)
        kept = []
        for c in cands:                       # a continuous connection must be constructible
            res = preview_gap(c, st.context, st.recon, self.config)
            c.preview_xy = res.path_xy if res.valid else None
            c.preview_metrics = dict(res.metrics) | {"reason": res.reason, "valid": res.valid}
            if self.config.gap.require_feasible_path and not res.valid:
                st.gaps.prefilter_rejected.append({"kind": c.kind, "source": c.source, "target": c.target,
                                                   "distance": round(c.distance, 1),
                                                   "reason": f"no continuous road trajectory fits: {res.reason}"})
                continue
            kept.append(c)
        rank = {"occlusion": 0, "visible road": 1, "other": 2}
        for c in kept:
            f = c.features
            occl = (f.get("frac_vegetation") or 0) + (f.get("frac_shadow") or 0)
            c.priority = "occlusion" if occl >= 0.25 else "visible road" if (f.get("frac_road_like") or 0) >= 0.25 else "other"
        kept.sort(key=lambda c: (rank[c.priority], c.distance))      # occluded gaps are reviewed first
        for i, c in enumerate(kept, 1):
            c.id = f"G{i:02d}"
        extract_all_features(kept, st.context)                         # recount competitors among the kept gaps
        st.candidates = st.gaps.candidates = kept
        self.logger.info(f"{len(kept)} road-to-road gaps passed the pre-CARS checks "
                         f"({sum(c.priority == 'occlusion' for c in kept)} around possible occlusions, reviewed first).")

    def _stage_cars(self) -> None:
        score_candidates(self.state.candidates, self.config)

    def _stage_recommend(self) -> None:
        st = self.state
        st.decision_summary = decide(st.candidates, self.config)
        base = st.image.image if st.image is not None else None
        margin = self.config.occlusion.corridor_margin
        for c in st.candidates:
            frag = st.evidence.get("fragments", {})
            ends = st.evidence.get("end_likelihood", {})
            ea, eb = st.build.endpoints[c.source], st.build.endpoints.get(c.target)
            c.fragments_info = {k: {**frag.get(e.component, {"component": e.component}), "end": e.id,
                                    "end_xy": [round(e.xy[0], 1), round(e.xy[1], 1)], "end_width_px": e.width,
                                    "end_road_likelihood": ends.get(e.id),
                                    "direction_deg": None if e.direction is None else
                                    round(float(np.degrees(np.arctan2(e.direction[1], e.direction[0]))), 1)}
                                for k, e in (("A", ea), ("B", eb)) if e is not None}
            c.xai = xai.explain_gap(c, self.config)
            c.user_decision, c.recon_status = "pending", "not_attempted"
            geo = candidate_geometry(c, st.context)
            box = gv.gap_region(c, geo, st.shape)
            st.gap_regions[c.id] = box
            th = path_thickness(c, self.config)
            imgs = {"original_view": gv.original_view(base, st.mask, st.mask_labels, st.appearance.category, c, geo, margin),
                    "enlarged": gv.enlarged_view(base, st.mask, st.mask_labels, c, geo, box, st.build.endpoints),
                    "mask_view": gv.mask_view(st.mask, st.mask_labels, c, box),
                    "graph_view": gv.graph_view(st.build.graph, st.build, c, box, c.preview_xy, False),
                    "preview": gv.preview_view(base, st.mask, st.mask_labels, c, box, c.preview_xy, th, False)}
            ev = gv.evidence_view(base, st.mask, st.appearance.category, c, geo, box, margin)
            if ev is not None:
                imgs["evidence"] = ev
            st.gap_images[c.id] = imgs
        st.recon = ReconstructionState.start(st.mask, st.build.graph)
        st.components_initial = st.recon.components()
        d = st.decision_summary
        self.logger.info(f"Recommendations at threshold {d['threshold']:.2f}: {d['recommended']} recommended, "
                         f"{d['uncertain']} uncertain, {d['not_recommended']} not recommended. "
                         "Nothing is reconstructed until a gap is approved.")

    # ------------------------------------------------------------------ #
    # user review
    # ------------------------------------------------------------------ #
    def decide_gap(self, gap_id: str, action: str) -> dict[str, Any]:
        st = self.state
        if "recommend" not in st.completed or st.recon is None:
            raise StageOrderError("Run the analysis before reviewing gaps.")
        if action not in ("approve", "reject", "skip"):
            raise ProcessingError("The action must be approve, reject or skip.")
        c = st.gap(gap_id)
        if c.recon_status == "reconstructed":
            raise DecisionRefused(f"Gap {c.id} has already been reconstructed; its decision can no longer change.")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if action == "approve":
            if c.recommendation == "not_recommended":
                raise DecisionRefused(f"Gap {c.id} is not recommended: {c.xai['conclusion']['en']}")
            c.user_decision, c.decided_at = "approved", now
            self.logger.info(f"User approved {c.id}; reconstructing with the graph-based trellis method.")
            with self.timer.time("review"):
                outcome = reconstruct_gap(c, st.context, st.recon, self.config)
            c.recon_status = outcome["status"]
            c.recon_reason = outcome["reason"]
            res = outcome.get("path")
            c.recon_metrics = dict(res.metrics) if res is not None else {}
            c.recon_metrics["thickness"] = outcome.get("thickness")
            c.recon_metrics["components_before"] = outcome["components_before"]
            c.recon_metrics["components_after"] = outcome["components_after"]
            if res is not None:
                c.recon_metrics["breakdown"] = res.breakdown
                c.recon_metrics["reference"] = res.reference
                c.recon_metrics["cover"] = res.cover
            c.recon_explanation = xai.explain_reconstruction(c, outcome, self.config)
            if c.recon_status == "reconstructed":
                c.segment_xy, c.guide_xy = res.path_xy, res.guide_xy
                base = st.image.image if st.image is not None else None
                box = st.gap_regions[c.id]
                imgs = st.gap_images[c.id]
                imgs["result"] = gv.preview_view(base, st.mask, st.mask_labels, c, box, res.path_xy,
                                                 outcome["thickness"], True)
                imgs["mask_after"] = gv.mask_after_view(st.recon.current_mask, st.recon.recon_only, box)
                imgs["graph_after"] = gv.graph_view(st.recon.graph, st.build, c, box, None, True)
                self.logger.info(f"{c.id} reconstructed: {c.recon_reason}. Components "
                                 f"{outcome['components_before']['mask']} -> {outcome['components_after']['mask']}.")
            else:
                self.logger.warning(f"{c.id} not reconstructed ({c.recon_status}): {c.recon_reason}. Mask unchanged.")
        else:
            c.user_decision = "rejected" if action == "reject" else "skipped"
            c.decided_at = now
            c.recon_status, c.recon_reason = "not_attempted", f"{c.user_decision} by the user"
            c.recon_explanation = {"result": xai.decision_voice(c, c.user_decision), "points": [], "succeeded": False}
            self.logger.info(f"User {c.user_decision} {c.id}.")
        st.decision_log.append({"time": now, "gap": c.id, "action": action, "user_decision": c.user_decision,
                                "recommendation": c.recommendation, "reconstruction": c.recon_status})
        self._update_review_progress()
        return {"gap": c.id, "user_decision": c.user_decision, "recon_status": c.recon_status,
                "reason": c.recon_reason, "summary": self.summary()}

    def _update_review_progress(self) -> None:
        st = self.state
        if "recommend" not in st.completed:
            return
        base = [s for s in st.completed if s not in ("review", "final")]
        undecided = [c for c in st.candidates if c.user_decision == "pending"]
        st.completed = base + (["review", "final"] if not undecided else [])
        timings = dict(self.timer.timings)
        timings["total"] = sum(timings.values())
        st.metrics = evaluate(st.mask, st.recon.current_mask, st.build.graph, st.recon.graph, st.candidates,
                              len(st.gaps.prefilter_rejected), st.reference, self.config.evaluation.tolerance_px, timings)

    def summary(self) -> dict[str, Any]:
        st = self.state
        cs = st.candidates
        count = lambda attr, v: sum(getattr(c, attr) == v for c in cs)  # noqa: E731
        comps = st.recon.components() if st.recon else {"mask": None, "graph": None}
        initial = st.components_initial or comps
        out = {"total": len(cs), "approved": count("user_decision", "approved"),
               "rejected": count("user_decision", "rejected"), "skipped": count("user_decision", "skipped"),
               "undecided": count("user_decision", "pending"),
               "reconstructed": count("recon_status", "reconstructed"),
               "failed": count("recon_status", "failed") + count("recon_status", "blocked"),
               "components_before": initial.get("mask"), "components_after": comps.get("mask"),
               "graph_components_before": initial.get("graph"), "graph_components_after": comps.get("graph"),
               **recommendation_counts(cs)}
        out["pending"] = out["undecided"] + out["skipped"]
        out["complete"] = out["undecided"] == 0
        return out
