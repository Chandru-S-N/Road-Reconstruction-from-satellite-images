"""
Synthetic test-data generator.

Every scene is SYNTHETIC: a procedurally drawn, aerial-style picture (grass,
fields, asphalt roads, roofs, tree canopy, cast shadows) with exactly known
road geometry. Results on these scenes are not results on real satellite
imagery.

For each scene the generator keeps
    reference_mask : the complete road network (ground truth),
    observed_mask  : the reference with the occluded / dropped-out parts removed,
    image          : the rendered picture, where gaps are hidden by trees or
                     shadows (occlusion gaps) or still visible (mask dropouts),
    meta           : description, expected outcome, and a "synthetic": true flag.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from core.utils import write_image

GRASS = (78, 122, 62)      # RGB
FIELD = (156, 136, 96)
ASPHALT = (122, 122, 118)
GRAVEL = (150, 144, 134)
ROOFS = [(168, 82, 58), (150, 70, 52), (120, 96, 150), (176, 110, 64)]
CANOPY = [(40, 78, 38), (52, 96, 44), (34, 64, 32), (62, 104, 50)]


@dataclass
class SyntheticScene:
    name: str
    description: str
    image: np.ndarray                 # BGR
    reference_mask: np.ndarray
    observed_mask: np.ndarray
    expected: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


class SceneBuilder:
    def __init__(self, size: tuple[int, int] = (400, 400), seed: int = 0) -> None:
        self.h, self.w = size
        self.rng = np.random.default_rng(seed)
        self.ref = np.zeros((self.h, self.w), np.uint8)
        self.surface = np.zeros((self.h, self.w, 3), np.float32)
        self.fields: list[tuple[np.ndarray, tuple]] = []
        self.rivers: list[tuple[np.ndarray, int, tuple]] = []
        self.occluded = np.zeros((self.h, self.w), bool)   # removed from observed mask
        self.dropout = np.zeros((self.h, self.w), bool)    # removed from mask only
        self.trees: list[tuple[tuple[int, int], int]] = []
        self.shadows: list[np.ndarray] = []
        self.buildings: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        self.gaps: list[dict[str, Any]] = []

    # roads -------------------------------------------------------------- #
    def road(self, pts: list[tuple[float, float]], width: int, surface=ASPHALT) -> None:
        arr = np.round(np.asarray(pts)).astype(np.int32).reshape(-1, 1, 2)
        layer = np.zeros_like(self.ref)
        cv2.polylines(layer, [arr], False, 255, width, cv2.LINE_8)
        self.ref = np.maximum(self.ref, layer)
        self.surface[layer > 0] = surface

    def arc(self, center, radius, a0_deg, a1_deg, n=200) -> list[tuple[float, float]]:
        t = np.radians(np.linspace(a0_deg, a1_deg, n))
        return list(zip(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)))

    # scenery ------------------------------------------------------------ #
    def field_patch(self, polygon: list[tuple[int, int]], color=FIELD) -> None:
        self.fields.append((np.asarray(polygon, np.int32), tuple(color)))

    def river(self, pts: list[tuple[float, float]], width: int, color=(72, 104, 142)) -> None:
        self.rivers.append((np.round(np.asarray(pts)).astype(np.int32).reshape(-1, 1, 2), width, tuple(color)))

    def tree(self, center: tuple[int, int], radius: int, occluder: bool = False) -> None:
        self.trees.append((center, radius))
        if occluder:
            m = np.zeros_like(self.ref)
            cv2.circle(m, center, radius, 255, -1)
            sh = np.zeros_like(self.ref)
            cv2.circle(sh, (center[0] + radius // 3, center[1] + radius // 3), radius, 255, -1)
            self.occluded |= (m > 0) | (sh > 0)
            self.gaps.append({"type": "tree canopy", "center": list(center)})

    def shadow(self, polygon: list[tuple[int, int]], occluder: bool = True) -> None:
        poly = np.asarray(polygon, np.int32)
        self.shadows.append(poly)
        if occluder:
            m = np.zeros_like(self.ref)
            cv2.fillPoly(m, [poly], 255)
            self.occluded |= m > 0
            c = poly.mean(axis=0).astype(int).tolist()
            self.gaps.append({"type": "shadow", "center": c})

    def building(self, x: int, y: int, bw: int, bh: int, shadow: bool = True, color=None) -> None:
        color = color or ROOFS[int(self.rng.integers(len(ROOFS)))]
        poly = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], np.int32)
        self.buildings.append((poly, color))
        if shadow:
            off = max(4, min(bw, bh) // 4)
            self.shadows.append(poly + off)

    def dropout_gap(self, center: tuple[int, int], radius: int) -> None:
        m = np.zeros_like(self.ref)
        cv2.circle(m, center, radius, 255, -1)
        self.dropout |= m > 0
        self.gaps.append({"type": "mask dropout (road visible in image)", "center": list(center)})

    def scatter_scenery(self, n_trees: int = 6, n_buildings: int = 3, clearance: int = 22) -> None:
        keepout = cv2.dilate(self.ref, np.ones((2 * clearance + 1, 2 * clearance + 1), np.uint8)) > 0
        for _ in range(n_buildings * 20):
            if n_buildings <= 0:
                break
            bw, bh = int(self.rng.integers(18, 34)), int(self.rng.integers(16, 30))
            x, y = int(self.rng.integers(5, self.w - bw - 15)), int(self.rng.integers(5, self.h - bh - 15))
            if keepout[y:y + bh + 12, x:x + bw + 12].any():
                continue
            self.building(x, y, bw, bh)
            keepout[max(0, y - 10):y + bh + 20, max(0, x - 10):x + bw + 20] = True
            n_buildings -= 1
        for _ in range(n_trees * 20):
            if n_trees <= 0:
                break
            r = int(self.rng.integers(8, 16))
            x, y = int(self.rng.integers(r, self.w - r)), int(self.rng.integers(r, self.h - r))
            if keepout[max(0, y - r - 6):y + r + 6, max(0, x - r - 6):x + r + 6].any():
                continue
            self.tree((x, y), r)
            keepout[max(0, y - r):y + r, max(0, x - r):x + r] = True
            n_trees -= 1

    # rendering ---------------------------------------------------------- #
    def _noise(self, sigma_low: float, sigma_high: float) -> np.ndarray:
        low = cv2.GaussianBlur(self.rng.normal(0, 1, (self.h, self.w)).astype(np.float32), (0, 0), 12)
        low = low / (low.std() + 1e-6) * sigma_low
        return low + self.rng.normal(0, sigma_high, (self.h, self.w)).astype(np.float32)

    def render(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = np.empty((self.h, self.w, 3), np.float32)
        img[:] = GRASS
        img += self._noise(10, 5)[..., None] * np.array([0.6, 1.0, 0.5], np.float32)
        for poly, color in self.fields:
            m = np.zeros((self.h, self.w), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            img[m > 0] = np.array(color, np.float32) + self._noise(4, 4)[m > 0, None]
        for pts, width, color in self.rivers:
            m = np.zeros((self.h, self.w), np.uint8)
            cv2.polylines(m, [pts], False, 1, width)
            img[m > 0] = np.array(color, np.float32) + self.rng.normal(0, 4, (int(m.sum()), 1)).astype(np.float32)
        road = self.ref > 0
        img[road] = self.surface[road] + self.rng.normal(0, 5, (int(road.sum()), 1)).astype(np.float32)
        for poly, color in self.buildings:
            m = np.zeros((self.h, self.w), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            img[m > 0] = np.array(color, np.float32) + self.rng.normal(0, 6, (int(m.sum()), 1)).astype(np.float32)
        for poly in self.shadows:
            m = np.zeros((self.h, self.w), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            sel = (m > 0)
            for bpoly, _ in self.buildings:   # roofs are not shaded by their own shadow
                bm = np.zeros_like(m)
                cv2.fillPoly(bm, [bpoly], 1)
                sel &= bm == 0
            img[sel] = img[sel] * 0.38 + np.array([6, 8, 18], np.float32)
        for (cx, cy), r in self.trees:   # tree shadow first, then canopy
            m = np.zeros((self.h, self.w), np.uint8)
            cv2.circle(m, (cx + r // 3, cy + r // 3), r, 1, -1)
            img[m > 0] = img[m > 0] * 0.4
        for (cx, cy), r in self.trees:
            base = np.zeros((self.h, self.w), np.uint8)
            cv2.circle(base, (cx, cy), r, 1, -1)
            img[base > 0] = np.array(CANOPY[2], np.float32) + self.rng.normal(0, 12, (int(base.sum()), 1)).astype(np.float32)
            for _ in range(max(4, r // 2)):
                ox, oy = self.rng.integers(-r // 2, r // 2 + 1, 2)
                rr = int(self.rng.integers(max(3, r // 3), max(4, (2 * r) // 3)))
                color = np.array(CANOPY[int(self.rng.integers(len(CANOPY)))], np.float32)
                m = np.zeros((self.h, self.w), np.uint8)
                cv2.circle(m, (int(cx + ox), int(cy + oy)), rr, 1, -1)
                img[m > 0] = color + self.rng.normal(0, 14, (int(m.sum()), 1)).astype(np.float32)
        img = cv2.GaussianBlur(img, (3, 3), 0) + self.rng.normal(0, 3, img.shape).astype(np.float32)
        bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        observed = self.ref.copy()
        observed[self.occluded | self.dropout] = 0
        return bgr, self.ref.copy(), observed

    def scene(self, name: str, description: str, expected: dict[str, Any]) -> SyntheticScene:
        image, ref, observed = self.render()
        expected = dict(expected)
        if "true_gaps" in expected:
            # a mask dropout is not a gap when the picture is segmented: the road is visible there
            expected["true_gaps_when_segmented"] = sum(not g["type"].startswith("mask dropout") for g in self.gaps)
        meta = {"synthetic": True, "name": name, "description": description,
                "expected": expected, "gaps": self.gaps,
                "note": "Procedurally generated test data, not real satellite imagery."}
        return SyntheticScene(name, description, image, ref, observed, expected, meta)


def _shadow_across(point, direction, along: float, across: float) -> list[tuple[int, int]]:
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    p = np.asarray(point, float)
    corners = [p + d * along / 2 + n * across / 2, p - d * along / 2 + n * across / 2,
               p - d * along / 2 - n * across / 2, p + d * along / 2 - n * across / 2]
    return [(int(c[0]), int(c[1])) for c in corners]


# --------------------------------------------------------------------------- #
# The ten named scenarios
# --------------------------------------------------------------------------- #
def straight_gap(seed: int = 1) -> SyntheticScene:
    b = SceneBuilder(seed=seed)
    b.road([(0, 200), (399, 200)], 14)
    b.tree((192, 196), 24, occluder=True)
    b.scatter_scenery(6, 3)
    return b.scene("01_straight_gap", "Straight road interrupted by tree canopy.",
                   {"true_gaps": 1, "should_connect": True})


def curved_gap(seed: int = 2) -> SyntheticScene:
    b = SceneBuilder(seed=seed)
    b.road(b.arc((0, 400), 260, -90, 0), 14)
    gp = (260 * math.cos(math.radians(-45)), 400 + 260 * math.sin(math.radians(-45)))
    tangent = (math.sin(math.radians(45)), math.cos(math.radians(45)))
    b.shadow(_shadow_across(gp, tangent, 40, 70))
    b.scatter_scenery(6, 3)
    return b.scene("02_curved_gap", "Curved road interrupted by a cast shadow.",
                   {"true_gaps": 1, "should_connect": True})


def nearby_should_not_connect(seed: int = 3) -> SyntheticScene:
    b = SceneBuilder(seed=seed)
    b.field_patch([(170, 150), (260, 150), (260, 245), (170, 245)])
    b.road([(0, 180), (190, 180)], 12)
    b.road([(230, 215), (399, 215)], 12)
    b.scatter_scenery(5, 3)
    return b.scene("03_nearby_no_connect", "Two offset dead ends 40 px apart across an open field.",
                   {"true_gaps": 0, "should_connect": False})


def width_mismatch(seed: int = 4) -> SyntheticScene:
    b = SceneBuilder(seed=seed)
    b.field_patch([(170, 160), (240, 160), (240, 240), (170, 240)])
    b.road([(0, 200), (180, 200)], 22)
    b.road([(225, 200), (399, 200)], 6, surface=GRAVEL)
    b.scatter_scenery(5, 3)
    return b.scene("04_width_mismatch", "A wide road ends; a narrow gravel track starts 45 px later "
                   "across an open field (different roads).", {"true_gaps": 0, "should_connect": False})


def vegetation_shadow_gap(seed: int = 5) -> SyntheticScene:
    b = SceneBuilder(seed=seed)
    b.road([(0, 399), (399, 0)], 12)
    b.tree((198, 198), 20, occluder=True)
    b.scatter_scenery(6, 3)
    return b.scene("05_vegetation_shadow_gap", "Diagonal road hidden under a tree and its shadow.",
                   {"true_gaps": 1, "should_connect": True})


def unrelated_roads(seed: int = 6) -> SyntheticScene:
    b = SceneBuilder(seed=seed)
    b.field_patch([(170, 120), (250, 120), (250, 210), (170, 210)])
    b.road([(0, 150), (180, 150)], 12)
    b.road([(225, 185), (399, 360)], 8, surface=GRAVEL)
    b.scatter_scenery(5, 3)
    return b.scene("06_unrelated_roads", "A road and an unrelated track end near each other at an angle.",
                   {"true_gaps": 0, "should_connect": False})


def multiple_components(seed: int = 7) -> SyntheticScene:
    b = SceneBuilder(seed=seed)
    b.road([(0, 110), (399, 110)], 14)            # main road
    b.road([(200, 110), (200, 399)], 12)          # side road
    b.road([(200, 290), (399, 290)], 12)          # branch
    b.tree((60, 108), 20, occluder=True)          # gap on main road (left)
    b.shadow(_shadow_across((200, 200), (0, 1), 38, 60))   # gap on side road
    b.dropout_gap((310, 290), 16)                 # mask dropout on branch
    b.road([(95, 300), (95, 360)], 8)             # isolated lane, no evidence of a link
    b.scatter_scenery(5, 3)
    return b.scene("07_multiple_components", "Tree-shaped network split into four components by "
                   "three gaps (tree, shadow, mask dropout) plus an isolated lane.",
                   {"true_gaps": 3, "should_connect": True})


def no_valid_connection(seed: int = 8) -> SyntheticScene:
    b = SceneBuilder(seed=seed)
    b.road([(0, 100), (150, 100)], 12)
    b.road([(300, 250), (300, 399)], 12)
    b.scatter_scenery(6, 3)
    return b.scene("08_no_valid_connection", "Two road ends far apart and pointing in unrelated directions.",
                   {"true_gaps": 0, "should_connect": False})


def already_connected(seed: int = 10) -> SyntheticScene:
    b = SceneBuilder(seed=seed)
    b.road([(0, 200), (399, 200)], 14)
    b.road([(200, 0), (200, 399)], 12)
    b.road([(0, 0), (399, 399)], 10)
    b.scatter_scenery(4, 2)
    return b.scene("10_already_connected", "A fully connected network with no gaps.",
                   {"true_gaps": 0, "should_connect": False})


def non_road_distractors(seed: int = 11) -> SyntheticScene:
    """Roads next to things a colour threshold mistakes for road."""
    b = SceneBuilder(seed=seed)
    b.river([(0, 40), (200, 52), (399, 44)], 18)                                   # water
    b.field_patch([(250, 190), (380, 200), (390, 330), (270, 345), (240, 280)], (134, 129, 121))  # bare grey ground
    b.road([(0, 130), (399, 130)], 14)
    b.road([(110, 130), (110, 399)], 12)
    b.field_patch([(122, 300), (190, 300), (190, 352), (122, 352)], (120, 120, 116))  # car park beside the road
    b.building(165, 185, 42, 30, color=(170, 170, 164))                            # grey concrete roof
    b.tree((262, 128), 22, occluder=True)
    b.scatter_scenery(4, 1, clearance=26)
    return b.scene("11_non_road_distractors", "Roads beside bare grey ground, a grey roof, a car park and a river; "
                   "one road gap under a tree.", {"true_gaps": 1, "should_connect": True})


SCENARIOS: dict[str, Callable[[], SyntheticScene]] = {
    "01_straight_gap": straight_gap,
    "02_curved_gap": curved_gap,
    "03_nearby_no_connect": nearby_should_not_connect,
    "04_width_mismatch": width_mismatch,
    "05_vegetation_shadow_gap": vegetation_shadow_gap,
    "06_unrelated_roads": unrelated_roads,
    "07_multiple_components": multiple_components,
    "08_no_valid_connection": no_valid_connection,
    "10_already_connected": already_connected,
    "11_non_road_distractors": non_road_distractors,
}
INVALID_SCENARIO = "09_invalid_image"


# --------------------------------------------------------------------------- #
# Randomised scenes for parameter sweeps
# --------------------------------------------------------------------------- #
def _random_curve(rng, w, h) -> list[tuple[float, float]]:
    side_pts = {
        0: lambda t: (t * w, 0), 1: lambda t: (w - 1, t * h), 2: lambda t: (t * w, h - 1), 3: lambda t: (0, t * h)}
    s1, s2 = rng.choice(4, 2, replace=False)
    a = np.array(side_pts[int(s1)](rng.uniform(0.15, 0.85)))
    c = np.array(side_pts[int(s2)](rng.uniform(0.15, 0.85)))
    mid = (a + c) / 2 + rng.normal(0, 45, 2)
    t = np.linspace(0, 1, 120)[:, None]
    pts = (1 - t) ** 2 * a + 2 * (1 - t) * t * mid + t ** 2 * c
    return [tuple(p) for p in pts]


def random_scene(seed: int, size: int = 400) -> SyntheticScene:
    """Random roads with true gaps (tree / shadow / dropout) and hard negatives
    (dead ends pointing at another road across open ground, or ending under trees)."""
    rng = np.random.default_rng(seed)
    b = SceneBuilder((size, size), seed=seed)
    curves = [_random_curve(rng, size, size) for _ in range(int(rng.integers(2, 5)))]
    widths = [int(rng.integers(9, 16)) for _ in curves]
    for pts, w in zip(curves, widths):
        b.road(pts, w)

    true_gaps = 0
    used: list[np.ndarray] = []
    for ci, (pts, w) in enumerate(zip(curves, widths)):
        other_pts = np.vstack([np.asarray(c) for j, c in enumerate(curves) if j != ci]) \
            if len(curves) > 1 else np.empty((0, 2))
        for _ in range(int(rng.integers(1, 3))):
            i = int(rng.integers(25, len(pts) - 25))
            p = np.array(pts[i])
            # keep gaps away from road crossings and from each other
            if len(other_pts) and np.min(np.hypot(*(other_pts - p).T)) < 45:
                continue
            if any(np.hypot(*(p - q)) < 70 for q in used):
                continue
            used.append(p)
            kind = rng.choice(["tree", "shadow", "dropout"], p=[0.45, 0.35, 0.2])
            d = np.array(pts[i + 1]) - np.array(pts[i - 1])
            if kind == "tree":
                b.tree((int(p[0]), int(p[1])), int(rng.integers(16, 24)), occluder=True)
            elif kind == "shadow":
                b.shadow(_shadow_across(p, d, float(rng.integers(26, 42)), 3.5 * w))
            else:
                b.dropout_gap((int(p[0]), int(p[1])), int(rng.integers(12, 18)))
            true_gaps += 1

    # hard negatives: a dead-end street aimed at another road that stops 30-60 px short.
    # Geometry alone would connect it; the gap is open ground (or, sometimes, woodland).
    negatives = 0
    wanted = int(rng.integers(1, 3))
    for _ in range(200):
        if negatives >= wanted or len(curves) < 2:
            break
        ia, ib = rng.choice(len(curves), 2, replace=False)
        A, B = np.asarray(curves[int(ia)]), np.asarray(curves[int(ib)])
        p = A[int(rng.integers(20, len(A) - 20))]
        dists = np.hypot(*(B - p).T)
        q = B[int(np.argmin(dists))]
        D = float(dists.min())
        if not 75 <= D <= 220 or any(np.hypot(*(q - u)) < 60 or np.hypot(*(p - u)) < 45 for u in used):
            continue
        direction = (q - p) / D
        gap = float(rng.uniform(32, 60))
        end = q - direction * gap
        probe = [p + direction * t for t in np.arange(18, D - gap, 3)]
        if any(b.ref[int(y), int(x)] for x, y in probe if 0 <= int(x) < size and 0 <= int(y) < size):
            continue   # the stub would cross another road
        b.road([tuple(p), tuple(end)], int(rng.integers(7, 12)))
        mid = (end + q) / 2
        if rng.random() < 0.25:
            b.tree((int(mid[0]), int(mid[1])), int(gap // 2.5))      # woodland: ambiguous on purpose
        else:
            half = gap * 0.8
            b.field_patch([(int(mid[0] - half), int(mid[1] - half)), (int(mid[0] + half), int(mid[1] - half)),
                           (int(mid[0] + half), int(mid[1] + half)), (int(mid[0] - half), int(mid[1] + half))])
        used.append(mid)
        negatives += 1

    b.scatter_scenery(int(rng.integers(3, 7)), int(rng.integers(1, 4)))
    return b.scene(f"random_{seed:03d}", "Randomised synthetic scene for parameter sweeps.",
                   {"true_gaps": true_gaps, "hard_negatives": negatives})


# --------------------------------------------------------------------------- #
# Writing to disk
# --------------------------------------------------------------------------- #
def save_scene(scene: SyntheticScene, folder: Path) -> Path:
    folder = Path(folder) / scene.name
    folder.mkdir(parents=True, exist_ok=True)
    write_image(folder / "image.png", scene.image)
    write_image(folder / "reference_mask.png", scene.reference_mask)
    write_image(folder / "observed_mask.png", scene.observed_mask)
    (folder / "meta.json").write_text(json.dumps(scene.meta, indent=2), encoding="utf-8")
    return folder


def write_invalid_image(folder: Path) -> Path:
    folder = Path(folder) / INVALID_SCENARIO
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n this is not a real PNG payload")
    (folder / "notes.txt").write_bytes(b"Scenario 9: a file with a .png name that cannot be decoded.")
    (folder / "meta.json").write_text(json.dumps({
        "synthetic": True, "name": INVALID_SCENARIO,
        "description": "Corrupted PNG; loading must fail with a clear error.",
        "expected": {"load_error": True}}, indent=2), encoding="utf-8")
    return folder


def generate_all(out_dir: str | Path, n_random: int = 24, dev_fraction: float = 0.6) -> dict[str, Any]:
    out = Path(out_dir)
    named = out / "scenarios"
    written = [str(save_scene(fn(), named)) for fn in SCENARIOS.values()]
    written.append(str(write_invalid_image(named)))
    sweep = out / "sweep"
    n_dev = int(round(n_random * dev_fraction))
    manifest = []
    for seed in range(n_random):
        split = "dev" if seed < n_dev else "test"
        scene = random_scene(seed + 100)
        folder = save_scene(scene, sweep / split)
        manifest.append({"name": scene.name, "split": split, "folder": str(folder),
                         "true_gaps": scene.expected["true_gaps"]})
    (sweep / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"scenarios": written, "sweep_scenes": len(manifest), "dev": n_dev, "test": n_random - n_dev}
