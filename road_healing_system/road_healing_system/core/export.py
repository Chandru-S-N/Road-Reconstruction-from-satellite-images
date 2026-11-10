"""
Results export: stage images, map layers, graph data, API payload and tables.

Files (relative to a job folder):
    stages/<name>.png         stage outputs (original, road_mask, skeleton_endpoints, road_graph, ...)
    gaps/<G01>/<name>.png     per-gap analysis: overview, detail, mask, evidence; after approval: path, mask_after
    layers/<name>.png         RGBA layers for the interactive map (mask, roads, reconstructed, environment)
    masks/<name>.png          original_road_mask, current_road_mask, reconstructed_only
    reports/                  CSV / JSON / log / pdf/
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np

from config import FEATURE_NAMES, FEATURES
from core import visualization as viz
from core.decision_engine import candidate_rows
from core.evaluation import flatten_metrics
from core.explanations import final_voice, stage_voices, view_voices
from core.gap_visuals import CATEGORY_COLORS
from core.graph_construction import graph_summary, graph_to_json
from core.pipeline import STAGE_LABELS, STAGES
from core.utils import to_jsonable

FILE_PATTERN = re.compile(r"^(stages|layers|masks)/[a-z_]+\.png$|^gaps/G\d{2,3}/[a-z_]+\.png$")
STAGE_TITLES = {
    "original": "Original satellite image", "preprocessed": "Preprocessed copy",
    "road_mask": "Binary road mask (roads white)", "seg_candidates": "Road-coloured candidates",
    "seg_likelihood": "Road colour likelihood", "seg_line_likeness": "Line-likeness (Sato ridge filter)",
    "seg_removed": "Removed as non-road (wide or compact regions)", "skeleton_endpoints": "Skeleton and endpoints",
    "road_graph": "Road graph by component", "fragments": "Road fragments and open ends",
    "candidate_gaps": "Genuine candidate gaps", "decisions": "Decisions so far",
    "before_overlay": "Original roads", "final_overlay": "Final road network",
    "current_mask": "Current road mask", "reconstructed_only": "Reconstructed segments only",
}


def _bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def fragments_view(image, mask, mask_labels, build, fragments) -> np.ndarray:
    base = (image.astype(np.float32) * 0.45).astype(np.uint8) if image is not None else np.zeros((*mask.shape, 3), np.uint8)
    out = base.copy()
    for lab in range(1, int(mask_labels.max()) + 1):
        out[mask_labels == lab] = viz.PALETTE[(lab - 1) % len(viz.PALETTE)]
    thick, fscale = viz._scale(mask.shape)
    for e in build.endpoints.values():
        col = (150, 150, 150) if e.at_border else viz.ENDPOINT
        cv2.circle(out, (int(e.xy[0]), int(e.xy[1])), 3 + thick, col, -1)
        cv2.circle(out, (int(e.xy[0]), int(e.xy[1])), 3 + thick, (0, 0, 0), 1)
    for f in fragments:
        if f["open_ends"]:
            e = build.endpoints[f["open_ends"][0]]
            viz.label(out, f"F{f['component']}", (e.xy[0] + 6, e.xy[1] + 14), (255, 255, 255), fscale)
    return out


def gaps_view(image, mask, candidates, endpoints, mode: str) -> np.ndarray:
    """mode: 'detected' (all yellow), 'recommendation', or 'decision'."""
    out = viz.tint(viz._base(image, mask, 0.55), mask, viz.EXISTING, 0.55)
    thick, fscale = viz._scale(mask.shape)
    colors = {"recommended": _bgr("#16a34a"), "uncertain": _bgr("#d97706"), "not_recommended": _bgr("#dc2626")}
    for c in candidates:
        if mode == "detected":
            col, solid = viz.DETECTED, False
        elif mode == "recommendation":
            col, solid = colors[c.recommendation], c.recommendation == "recommended"
        else:
            if c.recon_status == "reconstructed":
                col, solid = viz.RECON, True
            elif c.user_decision == "approved":
                col, solid = viz.FAILED, False
            elif c.user_decision == "rejected":
                col, solid = viz.REJECTED, False
            else:
                col, solid = (170, 170, 170), False
        p1, p2 = c.source_xy, c.target_xy
        if solid and c.segment_xy is not None:
            cv2.polylines(out, [np.round(c.segment_xy).astype(np.int32).reshape(-1, 1, 2)], False, col, 2 + thick, cv2.LINE_AA)
        elif solid:
            cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 2 + thick, cv2.LINE_AA)
        else:
            viz.dashed(out, p1, p2, col, 1 + thick)
        viz.label(out, c.id, ((p1[0] + p2[0]) / 2 + 5, (p1[1] + p2[1]) / 2 - 5), col, fscale)
    return out


def _rgba(mask: np.ndarray, color_bgr, alpha: int) -> np.ndarray:
    out = np.zeros((*mask.shape, 4), np.uint8)
    sel = mask > 0
    out[sel, :3] = color_bgr
    out[sel, 3] = alpha
    return out


# --------------------------------------------------------------------------- #
def stage_files(pipe, stage: str) -> dict[str, np.ndarray]:
    st = pipe.state
    base = st.image.image if st.image is not None else None
    f: dict[str, np.ndarray] = {}
    if stage == "upload" and base is not None:
        f["stages/original.png"] = base
    elif stage == "preprocess" and st.preprocessed is not None:
        f["stages/preprocessed.png"] = st.preprocessed
    elif stage == "segment":
        f["stages/road_mask.png"] = st.mask
        f["masks/original_road_mask.png"] = st.mask
        if st.segmentation is not None:
            for k, v in st.segmentation.intermediates.items():
                f[f"stages/{k}.png"] = v
        f["layers/mask.png"] = cv2.cvtColor(st.mask, cv2.COLOR_GRAY2BGRA)
        f["layers/roads.png"] = _rgba(st.mask, viz.EXISTING, 150)
    elif stage == "skeleton":
        f["stages/skeleton_endpoints.png"] = viz.skeleton_view(st.mask, st.skeleton_raw, st.keypoints)
    elif stage == "graph":
        f["stages/road_graph.png"] = viz.graph_view(st.mask, st.build.graph, st.build.endpoints, st.build.component_of)
    elif stage == "gaps":
        f["stages/fragments.png"] = fragments_view(base, st.mask, st.mask_labels, st.build, st.fragments)
        f["stages/candidate_gaps.png"] = gaps_view(base, st.mask, st.gaps.candidates, st.build.endpoints, "detected")
        if st.appearance is not None and st.appearance.category is not None:
            env = np.zeros((*st.mask.shape, 4), np.uint8)
            for code, col in CATEGORY_COLORS.items():
                if code == 0:
                    continue
                sel = (st.appearance.category == code) & (st.mask == 0)
                env[sel, :3] = col
                env[sel, 3] = 120
            f["layers/environment.png"] = env
    elif stage == "recommend":
        f["stages/recommendations.png"] = gaps_view(base, st.mask, st.candidates, st.build.endpoints, "recommendation")
        for c in st.candidates:
            for k, v in st.gap_images.get(c.id, {}).items():
                f[f"gaps/{c.id}/{k}.png"] = v
        f.update(review_files(pipe, None))
    return f


def review_files(pipe, gap_id: str | None) -> dict[str, np.ndarray]:
    st = pipe.state
    base = st.image.image if st.image is not None else None
    r = st.recon
    f = {"stages/decisions.png": gaps_view(base, st.mask, st.candidates, st.build.endpoints, "decision"),
         "stages/before_overlay.png": viz.before_view(base, st.mask),
         "stages/final_overlay.png": viz.final_overlay(base, st.mask, r.recon_only, []),
         "stages/current_mask.png": viz.final_mask_view(r.current_mask, r.recon_only),
         "stages/reconstructed_only.png": r.recon_only,
         "masks/current_road_mask.png": r.current_mask, "masks/reconstructed_only.png": r.recon_only,
         "layers/reconstructed.png": _rgba(r.recon_only, viz.RECON, 235),
         "layers/current_mask.png": cv2.cvtColor(r.current_mask, cv2.COLOR_GRAY2BGRA)}
    if gap_id is not None:
        for k, v in st.gap_images.get(gap_id, {}).items():
            f[f"gaps/{gap_id}/{k}.png"] = v
    return f


# --------------------------------------------------------------------------- #
def _num(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 4)
    return v


def gap_payload(c) -> dict[str, Any]:
    keep = ("end_deviation_deg", "bend_inconsistency_deg", "turn_angle_deg", "meeting_angle_deg", "width_source",
            "width_target", "width_rel_diff", "min_turn_radius_widths", "curvature_jump", "corridor_pixels",
            "frac_road_like", "frac_vegetation", "frac_shadow", "frac_textured", "frac_water", "frac_unexplained",
            "dominant_cover", "env_road_like", "env_vegetation", "env_shadow", "env_textured", "env_water",
            "env_unexplained", "env_dominant", "component_source", "component_target", "same_component",
            "crossing_pixels", "competing_candidates", "detour_ratio")
    rm = {k: v for k, v in c.recon_metrics.items() if k not in ("roi_xyxy",)}
    return to_jsonable({
        "id": c.id, "kind": c.kind, "source": c.source, "target": c.target, "source_xy": c.source_xy,
        "target_xy": c.target_xy, "distance": round(c.distance, 2),
        "features": {k: _num(c.features.get(k)) for k in keep if k in c.features},
        "scores": {k: _num(c.scores.get(k)) for k in FEATURES}, "cars": round(float(c.cars), 4),
        "recommendation": c.recommendation, "rec_basis": c.rec_basis, "reason": c.reason, "xai": c.xai,
        "user_decision": c.user_decision, "decided_at": c.decided_at, "recon_status": c.recon_status,
        "recon_reason": c.recon_reason, "recon_metrics": rm, "recon_explanation": c.recon_explanation,
        "path": None if c.segment_xy is None else np.round(c.segment_xy, 1)[:: max(1, len(c.segment_xy) // 120)].tolist(),
        "preview": None if c.preview_xy is None else np.round(c.preview_xy, 1)[:: max(1, len(c.preview_xy) // 120)].tolist(),
        "preview_metrics": {k: v for k, v in c.preview_metrics.items() if k != "roi_xyxy"},
        "priority": c.priority, "fragments_info": c.fragments_info,
    })


def results_payload(pipe) -> dict[str, Any]:
    st = pipe.state
    out: dict[str, Any] = {"stages": [{"key": s, "label": STAGE_LABELS[s], "done": s in st.completed} for s in STAGES],
                           "log": list(pipe.log_lines), "config": st.config.to_dict(),
                           "feature_names": FEATURE_NAMES}
    h, w = st.shape
    out["image_size"] = [w, h]
    info = st.image.info() if st.image is not None else {"name": "uploaded road mask", "width": w, "height": h,
                                                          "notes": ["No image supplied."]}
    info.update(has_mask=st.uploaded_mask is not None, has_reference=st.reference is not None)
    out["input"] = info
    vinfo: dict[str, Any] = {"width": w, "height": h}
    if st.preprocess_info:
        out["preprocess"] = st.preprocess_info
    if st.mask is not None:
        seg = st.segmentation.summary() if st.segmentation else {"method": "uploaded mask"}
        out["segmentation"] = {**seg, "mask_source": st.mask_source, "road_fraction": round(float((st.mask > 0).mean()), 4)}
        vinfo["road_pct"] = round(100 * float((st.mask > 0).mean()), 1)
    if st.keypoints:
        out["skeleton"] = {"pixels": int(st.skeleton_raw.sum()), "raw_endpoints": len(st.keypoints["endpoints"]),
                           "intersections": len(st.keypoints["intersections"])}
    if st.build is not None:
        g = graph_summary(st.build.graph)
        out["graph"] = {**g, "pruned_spurs": st.build.pruned_spurs}
        vinfo.update(nodes=g["nodes"], edges=g["edges"], components=g["components"])
    if st.gaps is not None:
        out["fragments"] = st.fragments
        out["gap_detection"] = {**st.gaps.summary(), "excluded": st.gaps.prefilter_rejected}
        vinfo.update(fragments=sum(f["is_fragment"] for f in st.fragments), gaps=len(st.gaps.candidates),
                     excluded=len(st.gaps.prefilter_rejected))
    if "recommend" in st.completed:
        out["gaps"] = [gap_payload(c) for c in st.candidates]
        out["regions"] = {k: list(v) for k, v in st.gap_regions.items()}
        out["decisions"] = st.decision_summary
        s = pipe.summary()
        out["summary"] = s
        out["decision_log"] = st.decision_log
        out["metrics"] = st.metrics
        voices = stage_voices(vinfo)
        voices["final"] = final_voice(s)
        voices.update(view_voices(s))
        out["voice"] = voices
    else:
        out["voice"] = stage_voices(vinfo)
    return to_jsonable(out)


def graph_payload(pipe) -> dict[str, Any]:
    """The actual NetworkX graph (current state) for the interactive graph view."""
    st = pipe.state
    if st.build is None:
        return {"nodes": [], "edges": [], "components": [], "stats": {}}
    G: nx.Graph = st.recon.graph if st.recon is not None else st.build.graph
    comp_of: dict[int, int] = {}
    comps = []
    for cid, comp in enumerate(sorted(nx.connected_components(G), key=lambda c: -len(c))):
        sub = G.subgraph(comp)
        for n in comp:
            comp_of[n] = cid
        comps.append({"id": cid, "nodes": len(comp), "edges": sub.number_of_edges(),
                      "length": round(sum(d["length"] for _, _, d in sub.edges(data=True)), 1),
                      "reconstructed_edges": sum(d.get("kind") == "reconstructed" for _, _, d in sub.edges(data=True))})
    border = {e.id for e in st.build.endpoints.values() if e.at_border}
    nodes = []
    for n, d in G.nodes(data=True):
        deg = G.degree(n)
        kind = "endpoint" if deg == 1 else "junction" if deg >= 3 else "waypoint" if deg == 2 else "isolated"
        nodes.append({"id": int(n), "x": round(d["xy"][0], 1), "y": round(d["xy"][1], 1), "type": kind, "degree": deg,
                      "component": comp_of.get(n), "at_border": n in border, "width": round(d.get("width", 0.0), 2),
                      "neighbors": [int(m) for m in G.neighbors(n)]})
    edges = []
    for u, v, d in G.edges(data=True):
        p = d["path"][:, ::-1]
        step = max(1, len(p) // 60)
        poly = np.vstack([p[::step], p[-1:]]).tolist()
        edges.append({"id": f"{int(u)}-{int(v)}", "source": int(u), "target": int(v), "kind": d.get("kind", "existing"),
                      "length": round(d["length"], 1), "width": round(d.get("width", 0.0), 2),
                      "component": comp_of.get(u), "gap_id": d.get("candidate_id"), "cars": d.get("cars"),
                      "path_cost": d.get("path_cost"), "polyline": poly})
    before = graph_summary(st.build.graph)
    after = graph_summary(G)
    frag = lambda H: sum(1 for comp in nx.connected_components(H)  # noqa: E731
                         if any(H.degree(n) == 1 and n not in border for n in comp))
    stats = {"before": {**before, "fragments": frag(st.build.graph)}, "after": {**after, "fragments": frag(G)}}
    return to_jsonable({"nodes": nodes, "edges": edges, "components": comps, "stats": stats})


# --------------------------------------------------------------------------- #
def rows_to_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return "no rows\n".encode("utf-8")
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    return buf.getvalue().encode("utf-8-sig")


def write_reports(pipe, folder: Path) -> None:
    st = pipe.state
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {
        "configuration.json": json.dumps(st.config.to_dict(), indent=2).encode("utf-8"),
        "processing_log.txt": ("\n".join(pipe.log_lines) + "\n").encode("utf-8"),
        "results.json": json.dumps(results_payload(pipe), indent=1, ensure_ascii=False).encode("utf-8"),
    }
    if st.build is not None:
        files["endpoints.csv"] = rows_to_csv([e.as_row() for e in st.build.endpoints.values()])
    if st.gaps is not None:
        files["fragments.csv"] = rows_to_csv([{k: (";".join(map(str, v)) if isinstance(v, list) else v)
                                               for k, v in f.items()} for f in st.fragments])
        files["excluded_pairs.csv"] = rows_to_csv(st.gaps.prefilter_rejected)
    if "recommend" in st.completed:
        files["gap_analysis.csv"] = rows_to_csv(candidate_rows(st.candidates))
        files["user_decisions.csv"] = rows_to_csv(st.decision_log)
        files["road_graph.json"] = json.dumps(to_jsonable(graph_to_json(st.recon.graph)), indent=1).encode("utf-8")
        files["evaluation_metrics.json"] = json.dumps(to_jsonable(st.metrics), indent=2).encode("utf-8")
        files["evaluation_metrics.csv"] = rows_to_csv([{"image": st.name, **flatten_metrics(st.metrics)}])
        for c in st.candidates:
            gdir = folder.parent / "gaps" / c.id
            gdir.mkdir(parents=True, exist_ok=True)
            (gdir / "analysis.json").write_text(json.dumps(gap_payload(c), indent=1, ensure_ascii=False), encoding="utf-8")
    for name, data in files.items():
        (folder / name).write_bytes(data)


def zip_folder(folder: Path, prefix: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "input" not in path.relative_to(folder).parts[:1]:
                zf.write(path, f"{prefix}/{path.relative_to(folder).as_posix()}")
    return buf.getvalue()
