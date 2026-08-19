"""Pure-CPU readers and ranking for source-bound 3D Gaussian scenes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData

_SH_C0 = 0.28209479177387814


def read_gaussian_source(path) -> dict:
    """Read one InteriorGS PLY into canonical NumPy arrays.

    Returned arrays use the source Habitat world frame.  Scale and opacity
    activations are applied exactly once here so rendering, alignment and
    scene ranking cannot silently parse the same PLY differently.
    """
    resolved = Path(path)
    vertices = PlyData.read(str(resolved))["vertex"]

    def column(name: str, *, allow_infinite: bool = False) -> np.ndarray:
        try:
            value = np.asarray(vertices[name], dtype=np.float32)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"GS source is missing vertex property {name!r}") from error
        invalid_values = (
            np.isnan(value).any() if allow_infinite else
            not np.isfinite(value).all())
        if value.ndim != 1 or invalid_values:
            raise ValueError(f"GS vertex property {name!r} is invalid")
        return value

    means = np.stack([column("x"), column("y"), column("z")], axis=1)
    log_scales = np.stack(
        [column(f"scale_{index}") for index in range(3)], axis=1)
    quaternions = np.stack(
        [column(f"rot_{index}") for index in range(4)], axis=1)
    features = np.stack(
        [column(f"f_dc_{index}") for index in range(3)], axis=1)
    # InteriorGS serializes saturated opacity logits as +/- infinity.  These
    # are valid sigmoid endpoints, unlike NaN, and the original torch reader
    # consumed them as exactly zero or one.
    opacity_logits = column("opacity", allow_infinite=True)
    scales = np.exp(log_scales).astype(np.float32, copy=False)
    logits = opacity_logits.astype(np.float64)
    opacities = np.empty(logits.shape, dtype=np.float64)
    nonnegative = logits >= 0.0
    opacities[nonnegative] = 1.0 / (
        1.0 + np.exp(-logits[nonnegative]))
    exponent = np.exp(logits[~nonnegative])
    opacities[~nonnegative] = exponent / (1.0 + exponent)
    opacities = opacities.astype(np.float32)
    colors = np.clip(0.5 + _SH_C0 * features, 0.0, 1.0).astype(
        np.float32, copy=False)
    return {
        "means": np.ascontiguousarray(means),
        "quats": np.ascontiguousarray(quaternions),
        "scales": np.ascontiguousarray(scales),
        "opacities": np.ascontiguousarray(opacities),
        "colors": np.ascontiguousarray(colors),
    }


def rank_scene_diagnostics(rows) -> list[dict]:
    """Return every scene in deterministic capacity-priority order."""
    return sorted((dict(row) for row in rows), key=lambda row: (
        float(row["fraction_above_quality_band"]),
        float(row["support_radius_p90_m"]),
        -int(row["gaussian_count"]),
        str(row["scene_id"]),
    ))
