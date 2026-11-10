"""
Images for the dedicated review screen of one gap (BGR uint8, OpenCV only).
Everything is drawn from the actual image, mask, graph and computed path.

    A original_view   whole image: both fragments tinted, road ends marked, gap region
                      outlined, measured occlusion pixels outlined (only if measured)
    B enlarged_view   crop with fragments A/B, direction arrows, the endpoint distance
    C mask_view       mask crop: fragment A cyan, fragment B magenta, ends, candidate gap
    D graph_view      local road graph: fragment edges coloured, other edges grey, nodes;
                      proposed connection dashed orange, reconstructed edge solid green
    E preview_view    crop with the computed path as a dashed overlay labelled PREVIEW
                      (or, once reconstructed, the actual result labelled RECONSTRUCTED)
    evidence_view     measured cover inside / around the gap
    mask_after_view   updated mask crop after reconstruction

Colours (BGR): fragment A cyan, fragment B magenta, gap yellow, proposal orange, reconstructed green.
"""

from __future__ import annotations

import cv2
import networkx as nx
import numpy as np

from core.visualization import dashed, label, tint

FRAG_A = (238, 211, 34)
FRAG_B = (239, 70, 217)
GAP = (21, 204, 250)
PROPOSAL = (22, 115, 249)
RECON = (94, 197, 34)
OCCL = (0, 140, 255)
CATEGORY_COLORS = {0: (163, 163, 163), 1: (238, 211, 34), 2: (74, 163, 22), 3: (237, 58, 124),
                   4: (22, 115, 249), 5: (235, 99, 37)}
TARGET = 420


def gap_region(c, geo: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    pts = np.vstack([geo["curve"], [c.source_xy, c.target_xy]])
    pad = int(max(30, 1.1 * c.distance, 3 * geo["width"]))
    h, w = shape
    return (max(0, int(pts[:, 0].min()) - pad), max(0, int(pts[:, 1].min()) - pad),
            min(w - 1, int(pts[:, 0].max()) + pad), min(h - 1, int(pts[:, 1].max()) + pad))


def _crop(img: np.ndarray, box, interp=cv2.INTER_CUBIC) -> tuple[np.ndarray, float]:
    x0, y0, x1, y1 = box
    sub = img[y0:y1 + 1, x0:x1 + 1]
    scale = TARGET / max(sub.shape[0], sub.shape[1])
    out = cv2.resize(sub, (max(1, int(round(sub.shape[1] * scale))), max(1, int(round(sub.shape[0] * scale)))),
                     interpolation=interp)
    return (out if out.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)), scale


def _to(box, scale, xy) -> tuple[int, int]:
    return int(round((xy[0] - box[0] + 0.5) * scale)), int(round((xy[1] - box[1] + 0.5) * scale))


def _resize_mask(m: np.ndarray, box, shape) -> np.ndarray:
    return cv2.resize(m[box[1]:box[3] + 1, box[0]:box[2] + 1].astype(np.uint8), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST) > 0


def fragment_masks(c, mask_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from core.reconstruction import _label_near
    la, lb = _label_near(mask_labels, c.source_xy), _label_near(mask_labels, c.target_xy)
    return (mask_labels == la) & (la > 0), (mask_labels == lb) & (lb > 0) & (lb != la)


def _ends(out, box, scale, c, size=7) -> None:
    for xy, color, name in ((c.source_xy, FRAG_A, "A"), (c.target_xy, FRAG_B, "B")):
        p = _to(box, scale, xy)
        cv2.circle(out, p, size, color, -1, cv2.LINE_AA)
        cv2.circle(out, p, size, (0, 0, 0), 1, cv2.LINE_AA)
        label(out, name, (p[0] + size + 3, p[1] - size - 2), color, 0.6)


def _distance(out, box, scale, c) -> None:
    a, b = np.array(_to(box, scale, c.source_xy)), np.array(_to(box, scale, c.target_xy))
    dashed(out, a, b, GAP, 2, dash=8)
    mid = (a + b) / 2
    n = np.array([-(b - a)[1], (b - a)[0]], float)
    n = n / (np.linalg.norm(n) + 1e-9) * 16
    label(out, f"{c.distance:.0f} px", (mid[0] + n[0] - 20, mid[1] + n[1] + 4), GAP, 0.55)


def corridor_masks(mask: np.ndarray, geo: dict, corridor_margin: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    thickness = int(max(3, round(geo["width"] + 2 * corridor_margin)))
    cor = np.zeros((h, w), np.uint8)
    cv2.polylines(cor, [np.round(geo["curve"]).astype(np.int32).reshape(-1, 1, 2)], False, 1, thickness)
    ring_px = int(max(6, round(geo["width"] * 1.5)))
    ring = cv2.dilate(cor, np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)) > 0
    cor = (cor > 0) & (mask == 0)
    return cor, ring & ~cor & (mask == 0)


# --------------------------------------------------------------------------- #
def original_view(image, mask, mask_labels, category, c, geo, corridor_margin: float) -> np.ndarray:
    base = (image.astype(np.float32) * 0.8).astype(np.uint8) if image is not None else \
        (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * 0.4).astype(np.uint8)
    fa, fb = fragment_masks(c, mask_labels)
    out = tint(tint(base, fa, FRAG_A, 0.55), fb, FRAG_B, 0.55)
    th = max(2, max(mask.shape) // 350)
    cor, ring = corridor_masks(mask, geo, corridor_margin)
    occl_frac = c.features.get("frac_vegetation", 0) + c.features.get("frac_shadow", 0)
    if category is not None and occl_frac >= 0.25:
        occl = (cor | ring) & np.isin(category, [2, 3])
        contours, _ = cv2.findContours(occl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, OCCL, th, cv2.LINE_AA)
        ys, xs = np.nonzero(occl)
        if len(xs):
            label(out, "possible occlusion", (int(xs.min()), max(14, int(ys.min()) - 6)), OCCL, max(0.45, max(mask.shape) / 900))
    contours, _ = cv2.findContours(cor.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, GAP, th, cv2.LINE_AA)
    _ends(out, (0, 0, mask.shape[1] - 1, mask.shape[0] - 1), 1.0, c, size=max(5, th * 2))
    return out


def enlarged_view(image, mask, mask_labels, c, geo, box, endpoints) -> np.ndarray:
    src = image if image is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out, scale = _crop(src, box)
    fa, fb = fragment_masks(c, mask_labels)
    for frag, color in ((fa, FRAG_A), (fb, FRAG_B)):
        contours, _ = cv2.findContours(_resize_mask(frag, box, out.shape).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    for ep, color in ((endpoints[c.source], FRAG_A), (endpoints.get(c.target), FRAG_B)):
        if ep is None or ep.direction is None:
            continue
        p = np.array(ep.xy)
        q = p - np.array(ep.direction) * max(18.0, (ep.width or 8) * 2)        # arrow along the fragment, toward its end
        cv2.arrowedLine(out, _to(box, scale, q), _to(box, scale, p), color, 3, cv2.LINE_AA, tipLength=0.3)
    _distance(out, box, scale, c)
    _ends(out, box, scale, c)
    return out


def mask_view(mask, mask_labels, c, box) -> np.ndarray:
    out, scale = _crop(mask, box, cv2.INTER_NEAREST)
    fa, fb = fragment_masks(c, mask_labels)
    out[_resize_mask(fa, box, out.shape)] = FRAG_A
    out[_resize_mask(fb, box, out.shape)] = FRAG_B
    _distance(out, box, scale, c)
    _ends(out, box, scale, c)
    return out


def graph_view(graph: nx.Graph, build, c, box, proposal_xy=None, reconstructed: bool = False) -> np.ndarray:
    x0, y0, x1, y1 = box
    scale = TARGET / max(x1 - x0 + 1, y1 - y0 + 1)
    out = np.full((int(round((y1 - y0 + 1) * scale)), int(round((x1 - x0 + 1) * scale)), 3), 24, np.uint8)
    ca = build.component_of.get(c.source)
    cb = build.component_of.get(c.target) if c.target is not None else None
    for u, v, d in graph.edges(data=True):
        pts = np.array([_to(box, scale, (p[1], p[0])) for p in d["path"]], np.int32)
        if d.get("kind") == "reconstructed":
            color, th = RECON, 5
        elif build.component_of.get(u) == ca and build.component_of.get(v) == ca:
            color, th = FRAG_A, 4
        elif cb is not None and build.component_of.get(u) == cb and build.component_of.get(v) == cb:
            color, th = FRAG_B, 4
        else:
            color, th = (150, 150, 150), 2
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, color, th, cv2.LINE_AA)
    if proposal_xy is not None and not reconstructed:
        pts = [np.array(_to(box, scale, p)) for p in proposal_xy]
        for i in range(0, len(pts) - 1, 2):
            cv2.line(out, tuple(pts[i]), tuple(pts[min(i + 1, len(pts) - 1)]), PROPOSAL, 3, cv2.LINE_AA)
    for n, d in graph.nodes(data=True):
        x, y = _to(box, scale, d["xy"])
        if not (0 <= x < out.shape[1] and 0 <= y < out.shape[0]):
            continue
        deg = graph.degree(n)
        if n in (c.source, c.target):
            continue
        if deg == 1:
            cv2.circle(out, (x, y), 5, (21, 204, 250), -1, cv2.LINE_AA)
        elif deg >= 3:
            cv2.rectangle(out, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)
        else:
            cv2.circle(out, (x, y), 3, (200, 200, 200), -1, cv2.LINE_AA)
    _ends(out, box, scale, c)
    label(out, "proposed connection (dashed)" if not reconstructed else "reconstructed edge (green)", (8, 18),
          PROPOSAL if not reconstructed else RECON, 0.5)
    return out


def preview_view(image, mask, mask_labels, c, box, path_xy, thickness: int, reconstructed: bool) -> np.ndarray:
    src = image if image is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out, scale = _crop(src, box)
    fa, fb = fragment_masks(c, mask_labels)
    for frag, color in ((fa, FRAG_A), (fb, FRAG_B)):
        contours, _ = cv2.findContours(_resize_mask(frag, box, out.shape).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    if path_xy is not None:
        band = np.zeros(mask.shape, np.uint8)
        cv2.polylines(band, [np.round(path_xy).astype(np.int32).reshape(-1, 1, 2)], False, 1, max(int(thickness), 2))
        sel = _resize_mask(band, box, out.shape)
        color = RECON if reconstructed else PROPOSAL
        if reconstructed:
            out[sel] = (0.3 * out[sel] + 0.7 * np.array(color)).astype(np.uint8)
        else:
            contours, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                for i in range(0, len(cnt) - 1, 8):
                    cv2.line(out, tuple(cnt[i][0]), tuple(cnt[min(i + 4, len(cnt) - 1)][0]), color, 2, cv2.LINE_AA)
        pts = [np.array(_to(box, scale, p)) for p in path_xy]
        step = 1 if reconstructed else 2
        for i in range(0, len(pts) - 1, step):
            cv2.line(out, tuple(pts[i]), tuple(pts[min(i + 1, len(pts) - 1)]), (255, 255, 255) if reconstructed else color, 1, cv2.LINE_AA)
    _ends(out, box, scale, c)
    banner = "RECONSTRUCTED" if reconstructed else ("PREVIEW - not reconstructed" if path_xy is not None else "NO FEASIBLE PATH")
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (20, 20, 20), -1)
    label(out, banner, (8, 18), RECON if reconstructed else PROPOSAL, 0.55)
    return out


def evidence_view(image, mask, category, c, geo, box, corridor_margin: float) -> np.ndarray | None:
    if image is None or category is None:
        return None
    cor, ring = corridor_masks(mask, geo, corridor_margin)
    h, w = mask.shape
    color = np.zeros((h, w, 3), np.uint8)
    for code, col in CATEGORY_COLORS.items():
        color[category == code] = col
    base = (image.astype(np.float32) * 0.55).astype(np.uint8)
    over = base.copy()
    over[ring] = (0.55 * color[ring] + 0.45 * base[ring]).astype(np.uint8)
    over[cor] = color[cor]
    out, _ = _crop(over, box, cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(_resize_mask(cor, box, out.shape).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_after_view(current_mask, recon_only, box) -> np.ndarray:
    out, _ = _crop(current_mask, box, cv2.INTER_NEAREST)
    out[_resize_mask(recon_only > 0, box, out.shape)] = RECON
    return out
