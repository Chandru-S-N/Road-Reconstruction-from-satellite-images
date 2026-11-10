"""
Stage output images (BGR uint8, rendered with OpenCV only).

Colour legend (also shown in the dashboard)
    existing road              blue    #4287f5
    reconstructed segment      green   #22c55e
    rejected candidate         red     #ef4444, dashed
    flagged for review         amber   #f59e0b, dashed
    accepted, not reconstructed orange #f97316, dashed
    detected candidate         yellow  #facc15, dashed
    endpoint                   yellow  #facc15
    intersection               magenta #d946ef
"""

from __future__ import annotations

import cv2
import networkx as nx
import numpy as np


def _bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


EXISTING = _bgr("#4287f5")
RECON = _bgr("#22c55e")
REJECTED = _bgr("#ef4444")
REVIEW = _bgr("#f59e0b")
FAILED = _bgr("#f97316")
DETECTED = _bgr("#facc15")
ENDPOINT = _bgr("#facc15")
JUNCTION = _bgr("#d946ef")
PALETTE = [_bgr(c) for c in ("#4287f5", "#14b8a6", "#a855f7", "#ec4899", "#0ea5e9",
                             "#84cc16", "#f59e0b", "#6366f1", "#10b981", "#f43f5e")]

LEGEND = [("Existing road", "#4287f5"), ("Reconstructed segment", "#22c55e"),
          ("Rejected candidate", "#ef4444"), ("Flagged for review", "#f59e0b"),
          ("Accepted, not reconstructed", "#f97316"), ("Endpoint", "#facc15"), ("Intersection", "#d946ef")]


def _scale(shape) -> tuple[int, float]:
    side = max(shape[:2])
    return max(1, side // 400), max(0.35, side / 1100)


def mask_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def tint(base: np.ndarray, region: np.ndarray, color, alpha: float) -> np.ndarray:
    out = base.astype(np.float32).copy()
    sel = region > 0
    out[sel] = out[sel] * (1 - alpha) + np.array(color, np.float32) * alpha
    return out.astype(np.uint8)


def _base(image: np.ndarray | None, mask: np.ndarray, dim: float) -> np.ndarray:
    if image is None:
        return (mask_bgr(mask) * 0.3).astype(np.uint8)
    return (image.astype(np.float32) * dim).astype(np.uint8)


def dashed(img: np.ndarray, p1, p2, color, thickness: int = 2, dash: int = 6) -> None:
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    length = float(np.hypot(*(p2 - p1)))
    if length < 1:
        return
    n = max(int(length // dash), 1)
    for i in range(0, n, 2):
        a = p1 + (p2 - p1) * (i / n)
        b = p1 + (p2 - p1) * (min(i + 1, n) / n)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness, cv2.LINE_AA)


def label(img: np.ndarray, text: str, xy, color=(255, 255, 255), scale: float = 0.4) -> None:
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
def skeleton_view(mask: np.ndarray, skeleton: np.ndarray, keypoints: dict) -> np.ndarray:
    out = (mask_bgr(mask) * 0.25).astype(np.uint8)
    thick, _ = _scale(mask.shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1 + thick, 1 + thick))
    out[cv2.dilate(skeleton.astype(np.uint8), k) > 0] = (255, 255, 255)
    for x, y in keypoints.get("endpoints", []):
        cv2.circle(out, (x, y), 2 + thick, ENDPOINT, -1)
    for x, y in keypoints.get("intersections", []):
        cv2.circle(out, (x, y), 3 + thick, JUNCTION, -1)
    return out


def graph_view(mask: np.ndarray, G: nx.Graph, endpoints: dict, component_of: dict) -> np.ndarray:
    out = (mask_bgr(mask) * 0.2).astype(np.uint8)
    thick, fscale = _scale(mask.shape)
    for u, v, d in G.edges(data=True):
        color = PALETTE[component_of.get(u, 0) % len(PALETTE)] if d.get("kind") == "existing" else RECON
        pts = d["path"][:, ::-1].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, color, 1 + thick)
    for n, d in G.nodes(data=True):
        x, y = int(d["xy"][0]), int(d["xy"][1])
        deg = G.degree(n)
        if deg == 1:
            cv2.circle(out, (x, y), 3 + thick, ENDPOINT, -1)
            if n in endpoints:
                label(out, str(n), (x + 5, y - 5), ENDPOINT, fscale)
        elif deg >= 3:
            cv2.circle(out, (x, y), 3 + thick, JUNCTION, -1)
    return out


def candidates_view(image: np.ndarray | None, mask: np.ndarray, candidates: list, endpoints: dict,
                    decisions: bool) -> np.ndarray:
    out = tint(_base(image, mask, 0.55), mask, EXISTING, 0.55)
    thick, fscale = _scale(mask.shape)
    for e in endpoints.values():
        cv2.circle(out, (int(e.xy[0]), int(e.xy[1])), 2 + thick, (120, 120, 120) if e.at_border else ENDPOINT, -1)
    for c in candidates:
        p1, p2 = c.source_xy, c.target_xy
        if not decisions:
            color, solid = DETECTED, False
            text = c.id
        else:
            fd = c.final_decision
            if fd == "accepted" and c.recon_status == "reconstructed":
                color, solid = RECON, True
            elif fd == "accepted":
                color, solid = FAILED, False
            elif fd == "review":
                color, solid = REVIEW, False
            else:
                color, solid = REJECTED, False
            text = f"{c.id} {c.cars:.2f}" if c.cars is not None else c.id
        if solid:
            cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2 + thick, cv2.LINE_AA)
        else:
            dashed(out, p1, p2, color, 1 + thick)
        label(out, text, ((p1[0] + p2[0]) / 2 + 4, (p1[1] + p2[1]) / 2 - 4), color, fscale)
    return out


def before_view(image: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    return tint(_base(image, mask, 0.8), mask, EXISTING, 0.5)


def final_overlay(image: np.ndarray | None, mask: np.ndarray, recon_only: np.ndarray,
                  candidates: list) -> np.ndarray:
    out = tint(_base(image, mask, 0.8), mask, EXISTING, 0.5)
    out = tint(out, recon_only, RECON, 0.9)
    thick, _ = _scale(mask.shape)
    for c in candidates:
        if c.recon_status == "reconstructed" and c.segment_xy is not None:
            pts = np.round(c.segment_xy).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
        elif c.final_decision == "rejected":
            dashed(out, c.source_xy, c.target_xy, REJECTED, 1 + thick, dash=4)
        elif c.final_decision == "review":
            dashed(out, c.source_xy, c.target_xy, REVIEW, 1 + thick, dash=4)
        elif c.final_decision == "accepted":
            dashed(out, c.source_xy, c.target_xy, FAILED, 1 + thick, dash=4)
    return out


def final_mask_view(final_mask: np.ndarray, recon_only: np.ndarray) -> np.ndarray:
    out = mask_bgr(final_mask)
    out[recon_only > 0] = RECON
    return out
