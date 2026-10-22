"""C1 terminal-image authority and counterfactual selection primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io

import numpy as np
from PIL import Image

from pipeline import record


BLOCK_L1_PROTOCOL = "block-l1.v1"


@dataclass(frozen=True)
class EncodedNativeRGB:
    """One deterministic lossless encoding of a native RGB raster."""

    png_bytes: bytes
    png_sha256: str
    pixel_sha256: str
    mode: str
    width_px: int
    height_px: int


def _resolution(value) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(isinstance(item, bool) or not isinstance(item, int) or
                item <= 0 for item in value)):
        raise ValueError("native RGB resolution is invalid")
    return int(value[0]), int(value[1])


def _pixel_sha256(rgb: np.ndarray) -> str:
    height, width, _channels = rgb.shape
    header = f"rgb8-pixels.v1\0{width}\0{height}\0".encode("ascii")
    return hashlib.sha256(header + np.ascontiguousarray(rgb).tobytes()).hexdigest()


def encode_native_rgb_png(
        rgb: np.ndarray, *, expected_resolution) -> EncodedNativeRGB:
    """Encode native uint8 HxWx3 RGB without conversion or resizing."""
    width, height = _resolution(expected_resolution)
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        raise ValueError("terminal RGB must be uint8")
    if array.shape != (height, width, 3):
        raise ValueError("terminal RGB shape disagrees with native resolution")
    canonical = np.ascontiguousarray(array)
    output = io.BytesIO()
    Image.fromarray(canonical, mode="RGB").save(
        output, format="PNG", optimize=False, compress_level=9)
    payload = output.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        decoded.load()
        if decoded.mode != "RGB" or decoded.size != (width, height):
            raise ValueError("terminal PNG mode or size changed during encoding")
        if not np.array_equal(np.asarray(decoded), canonical):
            raise ValueError("terminal PNG is not a lossless pixel encoding")
    return EncodedNativeRGB(
        png_bytes=payload,
        png_sha256=hashlib.sha256(payload).hexdigest(),
        pixel_sha256=_pixel_sha256(canonical),
        mode="RGB", width_px=width, height_px=height,
    )


def block_l1_features(rgb: np.ndarray) -> np.ndarray:
    """Return uint64 channel sums over native non-overlapping 8x8 blocks."""
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("block-l1.v1 requires native uint8 HxWx3 RGB")
    height, width, _channels = array.shape
    if height <= 0 or width <= 0 or height % 8 or width % 8:
        raise ValueError("block-l1.v1 requires H and W divisible by 8")
    blocks = np.ascontiguousarray(array).reshape(
        height // 8, 8, width // 8, 8, 3)
    return blocks.sum(axis=(1, 3), dtype=np.uint64)


def block_l1_certificate(left: np.ndarray, right: np.ndarray) -> dict:
    """Compute an order-independent exact rational native appearance distance."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError("block-l1.v1 images must have identical native shape")
    left_features = block_l1_features(left_array)
    right_features = block_l1_features(right_array)
    return _block_l1_certificate_from_features(
        left_array, right_array, left_features, right_features)


def _block_l1_certificate_from_features(
        left: np.ndarray, right: np.ndarray,
        left_features: np.ndarray, right_features: np.ndarray) -> dict:
    """Build the exact pair atom from already cached block sums."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (left_array.shape != right_array.shape or
            left_features.shape != right_features.shape):
        raise ValueError("block-l1.v1 cached feature shapes disagree")
    delta = left_features.astype(np.int64) - right_features.astype(np.int64)
    numerator = int(np.abs(delta).sum(dtype=np.uint64))
    height, width, _channels = left_array.shape
    denominator = int(height * width * 3 * 255)
    identities = sorted((_pixel_sha256(left_array), _pixel_sha256(right_array)))
    value = {
        "protocol": BLOCK_L1_PROTOCOL,
        "block_size_px": 8,
        "resolution": [int(width), int(height)],
        "pair_pixel_sha256": identities,
        "numerator": numerator,
        "denominator": denominator,
    }
    return {**value, "sha256": record.canonical_atom_sha256(value)}


