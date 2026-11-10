"""
PDF report of the current analysis session (ReportLab).

Every value comes from the session: images are the actual stage outputs, gap
sections use the measured features and generated explanations, statistics are
the current counts. Missing items are reported as "not available", never filled
in. The report is in English (Tamil text needs an embedded Tamil font, which is
not bundled).
"""

from __future__ import annotations

import io
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from config import FEATURE_NAMES
from core.export import review_files, stage_files
from core.pipeline import AUTO_STAGES

TITLE = "Context-Aware Shortest-Path Healing System for Occlusion-Robust Road Network Reconstruction"
NAVY = colors.HexColor("#0c3772")
GRID = colors.HexColor("#d6dfeb")


class _NumberedCanvas(rl_canvas.Canvas):
    """Adds 'Page X of Y' and a running header once the total page count is known."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pages: list[dict] = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            w, h = A4
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#5a6b85"))
            self.drawString(15 * mm, h - 10 * mm, "Road network reconstruction report")
            self.drawRightString(w - 15 * mm, 10 * mm, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def _img(arr: np.ndarray | None, max_w: float, max_h: float) -> Any:
    if arr is None:
        return Paragraph("<i>not available</i>", getSampleStyleSheet()["BodyText"])
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    ok, buf = cv2.imencode(".png", arr)
    h, w = arr.shape[:2]
    scale = min(max_w / w, max_h / h)
    return Image(io.BytesIO(buf.tobytes()), width=w * scale, height=h * scale)


def _fmt(v: Any, d: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def _table(rows: list[list[Any]], widths: list[float] | None = None, header: bool = True) -> Table:
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.5), ("GRID", (0, 0), (-1, -1), 0.4, GRID),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fa")),
                  ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5)]
    t.setStyle(TableStyle(style))
    return t


def _versions() -> list[list[str]]:
    out = [["Python", platform.python_version()]]
    from importlib.metadata import PackageNotFoundError, version
    for dist, name in (("opencv-python", "OpenCV"), ("numpy", "NumPy"), ("networkx", "NetworkX"),
                       ("scikit-image", "scikit-image"), ("scipy", "SciPy"), ("flask", "Flask"), ("reportlab", "ReportLab")):
        try:
            out.append([name, version(dist)])
        except PackageNotFoundError:
            try:                                        # e.g. opencv-python-headless instead of opencv-python
                mod = {"OpenCV": "cv2"}.get(name)
                out.append([name, __import__(mod).__version__ if mod else "not installed"])
            except Exception:
                out.append([name, "not installed"])
    return out


def build_pdf(pipe, path: Path) -> Path:
    st = pipe.state
    if "recommend" not in st.completed:
        raise ValueError("Run the analysis before generating a report.")
    files: dict[str, np.ndarray] = {}
    for stage in AUTO_STAGES:
        files.update(stage_files(pipe, stage))
    files.update(review_files(pipe, None))
    for c in st.candidates:
        for k, v in st.gap_images.get(c.id, {}).items():
            files[f"gaps/{c.id}/{k}.png"] = v

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=NAVY, fontSize=15, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=12, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#5a6b85"))
    cap = ParagraphStyle("c", parent=small, alignment=TA_CENTER)
    W = A4[0] - 30 * mm
    s = pipe.summary()
    m = st.metrics.get("connectivity", {})
    story: list[Any] = []

    # A - project information -------------------------------------------------
    story += [Paragraph(TITLE, h1),
              Paragraph(f"Analysis report generated {time.strftime('%Y-%m-%d %H:%M:%S')}", small), Spacer(1, 6)]
    info = st.image.info() if st.image is not None else {"name": "uploaded road mask", "width": st.shape[1],
                                                          "height": st.shape[0], "format": "-", "file_size_kb": "-"}
    synthetic = st.image is not None and any("Synthetic" in n for n in st.image.notes)
    story.append(_table([["Input", "Value"], ["File", info.get("name")], ["Size", f"{info.get('width')} x {info.get('height')} px"],
                         ["Format", info.get("format")], ["Road mask", st.mask_source],
                         ["Reference mask", "supplied" if st.reference is not None else "not supplied"],
                         ["Synthetic test scene", "yes (not real imagery)" if synthetic else "no"]], [45 * mm, W - 45 * mm]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Processing summary", h2))
    failed_txt = f" and {s['failed']} could not be" if s["failed"] else ""
    story.append(Paragraph(
        f"{s['total']} candidate gaps were analysed: {s['recommended']} recommended, {s['uncertain']} uncertain, "
        f"{s['not_recommended']} not recommended. The user approved {s['approved']}, rejected {s['rejected']}; "
        f"{s['pending']} are pending. {s['reconstructed']} connections were reconstructed successfully{failed_txt}.", body))
    story.append(Paragraph("Software and algorithms", h2))
    story.append(_table([["Component", "Version"]] + _versions(), [45 * mm, 40 * mm]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Road extraction: classical contextual segmentation (colour model, Sato ridge filter, width and shape filters; "
        "no trained model). Graph: NetworkX on a pruned m-adjacency skeleton. Gap scoring: Context-Aware Reconnection "
        "Score (weighted sum of alignment, occlusion evidence, width consistency, local geometry and network "
        "connectivity). Reconstruction: graph-based trellis (dynamic programming on a directed acyclic graph of stations "
        "and lateral positions along the expected road course; no Dijkstra), one approved gap at a time.", body))

    # B - original image and mask ----------------------------------------------
    story += [PageBreak(), Paragraph("Original image and road mask", h1)]
    row = [_img(files.get("stages/original.png"), W / 2 - 4, 90 * mm), _img(files.get("stages/road_mask.png"), W / 2 - 4, 90 * mm)]
    story.append(Table([row, [Paragraph("Original satellite/aerial image", cap), Paragraph("Binary road mask (roads white)", cap)]],
                       colWidths=[W / 2, W / 2]))
    seg = st.segmentation.summary() if st.segmentation else {}
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Roads cover {100 * float((st.mask > 0).mean()):.1f}% of the image ({st.mask_source}). The road graph has "
        f"{st.metrics['connectivity']['graph_components_before']} connected components and "
        f"{sum(f['is_fragment'] for f in st.fragments)} fragments with open ends."
        + (f" Excluded as non-road: {seg.get('excluded_vegetation_pct', 'n/a')}% vegetation-like, "
           f"{seg.get('excluded_water_pct', 'n/a')}% water-like, {seg.get('excluded_shadow_pct', 'n/a')}% very dark; "
           f"{seg.get('removed_wide_pct', 'n/a')}% removed as wider than a road." if seg else ""), body))
    row = [_img(files.get("stages/road_graph.png"), W / 2 - 4, 80 * mm), _img(files.get("stages/fragments.png"), W / 2 - 4, 80 * mm)]
    story.append(Table([row, [Paragraph("Road graph by component", cap), Paragraph("Fragments and open ends", cap)]],
                       colWidths=[W / 2, W / 2]))

    # C - candidate gaps ------------------------------------------------------
    story += [PageBreak(), Paragraph("Candidate gap analysis", h1),
              _img(files.get("stages/candidate_gaps.png"), W, 95 * mm),
              Paragraph(f"{len(st.candidates)} genuine candidate gaps; {len(st.gaps.prefilter_rejected)} endpoint pairs "
                        "were excluded before analysis (distance, direction, width, crossing roads, water, or a better "
                        "match for the same road end).", body)]
    for c in st.candidates:
        x = c.xai
        g = f"gaps/{c.id}/"
        names = ("original_view", "enlarged", "mask_view", "graph_view", "preview", "evidence")
        titles = ("A. Original image", "B. Enlarged gap (A cyan, B magenta)", "C. Road mask", "D. Local road graph",
                  "E. Proposed path (preview)", "Measured cover")
        imgs = [_img(files.get(g + k + ".png"), W / 3 - 4, 45 * mm) for k in names]
        caps = [Paragraph(t, cap) for t in titles]
        rows = [["Feature", "Score", "Weight", "Contribution"]]
        for r in x["numeric"]["rows"]:
            rows.append([f"{r['feature']} {FEATURE_NAMES[r['feature']]}" + (" (missing)" if r["missing"] else ""),
                         _fmt(r["value"]), _fmt(r["weight"]), _fmt(r["contribution"])])
        rows.append(["CARS", _fmt(c.cars), "", ""])
        points = "".join(f"<br/>&bull; {p['en']}" for p in x["points"])
        block = [Paragraph(f"Gap {c.id}: {x['headline']['en']}", h2),
                 Table([imgs[:3], caps[:3], imgs[3:], caps[3:]], colWidths=[W / 3] * 3),
                 Spacer(1, 4), _table(rows, [60 * mm, 25 * mm, 25 * mm, 30 * mm]), Spacer(1, 4),
                 Paragraph(f"<b>Evidence</b>{points}", body),
                 Paragraph(f"<b>Recommendation:</b> {x['headline']['en']} ({c.reason}). {x['conclusion']['en']} "
                           f"{x['influence_text']['en']}", body)]
        story.append(KeepTogether(block[:4]))
        story += block[4:]

    # D - user decisions ------------------------------------------------------
    story += [PageBreak(), Paragraph("User decisions", h1)]
    rows = [["Gap", "Recommendation", "User decision", "Decided at", "Reconstruction"]]
    for c in st.candidates:
        rows.append([c.id, c.recommendation.replace("_", " "), c.user_decision, c.decided_at or "-",
                     c.recon_status.replace("_", " ")])
    story.append(_table(rows, [18 * mm, 35 * mm, 30 * mm, 40 * mm, 35 * mm]))

    # E - reconstruction results ------------------------------------------------
    story += [Spacer(1, 8), Paragraph("Reconstruction results", h1)]
    done = [c for c in st.candidates if c.user_decision == "approved"]
    if not done:
        story.append(Paragraph("No gap was approved, so nothing was reconstructed.", body))
    for c in done:
        g = f"gaps/{c.id}/"
        ex = c.recon_explanation
        block = [Paragraph(f"Gap {c.id}: {c.recon_status.replace('_', ' ')}", h2)]
        if c.recon_status == "reconstructed":
            block.append(Table([[_img(files.get(g + "result.png"), W / 3 - 4, 55 * mm), _img(files.get(g + "mask_after.png"), W / 3 - 4, 55 * mm),
                                 _img(files.get(g + "graph_after.png"), W / 3 - 4, 55 * mm)],
                                [Paragraph("Reconstructed connection (green)", cap), Paragraph("Updated road mask", cap),
                                 Paragraph("Updated local road graph", cap)]], colWidths=[W / 3] * 3))
            block.append(Paragraph("".join(f"&bull; {p['en']}<br/>" for p in ex.get("points", [])), body))
        block.append(Paragraph(ex.get("result", {}).get("en", c.recon_reason), body))
        story.append(KeepTogether(block))
    story += [PageBreak(), Paragraph("Before and after", h1),
              Table([[_img(files.get("stages/before_overlay.png"), W / 2 - 4, 85 * mm), _img(files.get("stages/final_overlay.png"), W / 2 - 4, 85 * mm)],
                     [Paragraph("Original roads", cap), Paragraph("Final network (reconstructed in green)", cap)],
                     [_img(files.get("masks/original_road_mask.png"), W / 2 - 4, 85 * mm), _img(files.get("stages/current_mask.png"), W / 2 - 4, 85 * mm)],
                     [Paragraph("Original road mask", cap), Paragraph("Final road mask (new segments green)", cap)]],
                    colWidths=[W / 2, W / 2])]
    from core import visualization as viz
    story += [Paragraph("Final road network graph", h2),
              _img(viz.graph_view(st.recon.current_mask, st.recon.graph, st.build.endpoints,
                                  {n: 0 for n in st.recon.graph.nodes}), W * 0.7, 80 * mm)]

    # F - statistics -------------------------------------------------------------
    story += [PageBreak(), Paragraph("Final statistics", h1)]
    before, after = m.get("mask_components_before"), m.get("mask_components_after")
    rows = [["Measure", "Value"], ["Total candidate gaps", s["total"]], ["Approved", s["approved"]],
            ["Rejected", s["rejected"]], ["Pending (undecided or skipped)", s["pending"]],
            ["Successfully reconstructed", s["reconstructed"]], ["Failed or blocked reconstruction attempts", s["failed"]],
            ["Connected road components before", before], ["Connected road components after", after],
            ["Graph components before / after", f"{m.get('graph_components_before')} / {m.get('graph_components_after')}"],
            ["Connectivity change", f"{before - after} fewer components" if isinstance(before, int) and isinstance(after, int) and before > after
             else "no change" if before == after else "n/a"]]
    if st.metrics.get("has_reference"):
        gm = st.metrics["gap_metrics"]
        rows += [["Gap precision / recall / F1 (vs reference)",
                  f"{_fmt(gm['gap_precision'])} / {_fmt(gm['gap_recall'])} / {_fmt(gm['gap_f1'])}"]]
    else:
        rows += [["Accuracy metrics", "not reported: no reference mask was supplied"]]
    story.append(_table([[str(a), str(b)] for a, b in rows], [90 * mm, 60 * mm]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("A lower component count is reported, not assumed to be better: an incorrect connection would "
                           "also reduce it. Only reconstructions that passed validation are counted as successful.", small))

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Road network reconstruction report", author="Road healing system")
    doc.build(story, canvasmaker=_NumberedCanvas)
    return path
