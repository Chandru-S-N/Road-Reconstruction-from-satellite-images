"""
Stages 1-2 - Satellite image upload (validation) and image processing.

Input : a PNG/JPG/JPEG file (path, bytes or file-like object)
Output: ``LoadedImage`` (validated BGR uint8 original, never modified) and a
        separately allocated preprocessed copy for the later stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np

from config import PreprocessConfig
from core.errors import ImageLoadError, MaskError
from core.utils import SUPPORTED_IMAGE_EXTENSIONS

try:  # OpenCV prints its own decoder warnings for broken files; our errors explain them instead
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except AttributeError:
    pass


@dataclass
class LoadedImage:
    image: np.ndarray                 # BGR uint8, H x W x 3 - treat as read-only
    name: str
    file_format: str
    file_size_bytes: int
    original_channels: int
    was_grayscale: bool
    had_alpha: bool
    notes: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format": self.file_format,
            "width": self.width,
            "height": self.height,
            "channels_in_file": self.original_channels,
            "grayscale_input": self.was_grayscale,
            "alpha_channel_dropped": self.had_alpha,
            "file_size_kb": round(self.file_size_bytes / 1024, 1),
            "notes": list(self.notes),
        }


def _read_source(source: str | Path | bytes | BinaryIO, name: str | None) -> tuple[bytes, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ImageLoadError(f"File not found: {path}")
        if not path.is_file():
            raise ImageLoadError(f"Not a file: {path}")
        return path.read_bytes(), name or path.name
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), name or "uploaded_image"
    if hasattr(source, "read"):
        data = source.read()
        return data, name or getattr(source, "name", "uploaded_image")
    raise ImageLoadError(f"Unsupported image source type: {type(source).__name__}")


def _check_extension(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ImageLoadError(f"Unsupported file format '{ext or '(none)'}' for {name}. Use one of: {allowed}.")
    return ext.lstrip(".").upper().replace("JPEG", "JPG")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    return np.clip((img - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def load_image(source: str | Path | bytes | BinaryIO, name: str | None = None,
               cfg: PreprocessConfig | None = None) -> LoadedImage:
    """Load and validate an aerial image. Raises ImageLoadError with a clear message."""
    cfg = cfg or PreprocessConfig()
    data, name = _read_source(source, name)
    fmt = _check_extension(name)
    if not data:
        raise ImageLoadError(f"{name} is empty (0 bytes).")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise ImageLoadError(f"{name} could not be decoded. The file may be corrupted or not a real {fmt} image.")

    notes: list[str] = []
    if img.dtype != np.uint8:
        notes.append(f"Converted {img.dtype} pixel data to 8-bit.")
        img = _to_uint8(img)

    channels = 1 if img.ndim == 2 else img.shape[2]
    had_alpha = channels == 4
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif channels == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        notes.append("Alpha channel was dropped.")
    elif channels == 3:
        bgr = img.copy()
    else:
        raise ImageLoadError(f"{name} has {channels} channels; expected 1, 3 or 4.")

    was_gray = channels == 1 or bool(np.array_equal(bgr[..., 0], bgr[..., 1]) and np.array_equal(bgr[..., 1], bgr[..., 2]))
    if was_gray:
        notes.append("Image has no colour information: vegetation cues are unavailable.")

    h, w = bgr.shape[:2]
    if min(h, w) < cfg.min_size:
        raise ImageLoadError(f"{name} is {w}x{h}; the smallest side must be at least {cfg.min_size} px.")
    if max(h, w) > cfg.max_size:
        raise ImageLoadError(f"{name} is {w}x{h}; the largest side must be at most {cfg.max_size} px. "
                             "Crop or downscale it first, or raise preprocess.max_size.")
    if float(bgr.std()) < 1e-6:
        notes.append("Image is a single flat colour; segmentation will find nothing useful.")

    bgr.setflags(write=False)  # the original is never modified downstream
    return LoadedImage(bgr, name, fmt, len(data), channels, was_gray, had_alpha, notes)


def load_mask(source: str | Path | bytes | BinaryIO, name: str | None = None,
              expected_shape: tuple[int, int] | None = None, role: str = "road mask") -> np.ndarray:
    """Load a binary mask and normalise it to {0, 255} uint8."""
    try:
        data, name = _read_source(source, name)
        _check_extension(name)
    except ImageLoadError as exc:
        raise MaskError(f"{role}: {exc}") from exc
    if not data:
        raise MaskError(f"{role} {name} is empty (0 bytes).")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise MaskError(f"{role} {name} could not be decoded.")
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    img = _to_uint8(img) if img.dtype != np.uint8 else img
    hi = int(img.max())
    binary = img > (0 if hi <= 1 else hi // 2)
    mask = np.where(binary, 255, 0).astype(np.uint8)
    if expected_shape is not None and mask.shape != tuple(expected_shape):
        raise MaskError(f"{role} {name} is {mask.shape[1]}x{mask.shape[0]} but the image is "
                        f"{expected_shape[1]}x{expected_shape[0]}. Masks must match the image size.")
    return mask


def preprocess_image(image: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Denoise and contrast-enhance a *copy* of the image."""
    out = image.copy()
    steps: list[str] = []
    k = int(cfg.denoise_strength) | 1  # force odd
    if cfg.denoise == "gaussian":
        out = cv2.GaussianBlur(out, (k, k), 0)
        steps.append(f"gaussian blur {k}x{k}")
    elif cfg.denoise == "median":
        out = cv2.medianBlur(out, k)
        steps.append(f"median blur {k}")
    elif cfg.denoise == "bilateral":
        out = cv2.bilateralFilter(out, k, 40, 40)
        steps.append(f"bilateral filter d={k}")
    if cfg.enhance_contrast:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit), tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        steps.append(f"CLAHE clip={cfg.clahe_clip_limit}")
    return out, {"steps": steps or ["none"]}


def loaded_from_array(image_bgr: np.ndarray, name: str, synthetic: bool = False) -> LoadedImage:
    """Wrap an in-memory BGR image (e.g. a synthetic scene) as a validated LoadedImage."""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ImageLoadError("In-memory images must be H x W x 3 uint8 (BGR).")
    img = image_bgr.copy()
    img.setflags(write=False)
    gray = bool(np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]))
    return LoadedImage(img, name, "PNG", int(img.nbytes), 3, gray, False,
                       ["Synthetic test scene (not real satellite imagery)."] if synthetic else [])
