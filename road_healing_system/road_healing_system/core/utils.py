"""Shared helpers: stage timing, logging, Windows-safe image IO, JSON conversion."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class StageTimer:
    """Accumulates wall-clock seconds per named stage."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - start


class ListLogHandler(logging.Handler):
    """Keeps formatted log lines in memory so the UI and exports can show them."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def make_logger(name: str = "road_healing") -> tuple[logging.Logger, ListLogHandler]:
    """A logger with its own in-memory handler (one per pipeline run)."""
    logger = logging.getLogger(f"{name}.{id(object())}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = ListLogHandler()
    logger.addHandler(handler)
    return logger, handler


def read_image_bytes(path: str | Path) -> bytes:
    """Read raw bytes. np.fromfile + imdecode avoids cv2.imread's non-ASCII path bug on Windows."""
    return Path(path).read_bytes()


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write PNG/JPG through imencode so non-ASCII Windows paths work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise OSError(f"OpenCV could not encode an image as {ext}")
    buf.tofile(str(path))


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise OSError("OpenCV could not encode PNG")
    return buf.tobytes()


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / tuple / set values into JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
