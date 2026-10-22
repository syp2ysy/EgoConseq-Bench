"""Floor plane ``n . p + d = 0`` in the agent pose-local frame.

Collection uses the fixed start floor ``(0, 1, 0), 0`` (``pipeline.meta.collect``).
The plane validates its coefficients, so every height above it is a signed normal
distance with an upward normal.

Frame: the agent pose-local frame produced by ``perception.to_agent_ground``,
where the camera centre is ``(0, nominal_camera_offset_m, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FloorPlaneEstimate:
    """A floor plane ``n . p + d = 0`` with a unit upward normal (``n_y > 0``)."""
    normal_local: Tuple[float, float, float]
    offset_m: float
    def __post_init__(self):
        # A zero or vertical normal is not a floor, and a downward one silently
        # flips the sign of every published height.
        normal = tuple(float(value) for value in self.normal_local)
        if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
            raise ValueError(
                f"floor plane normal must be three finite numbers: "
                f"{self.normal_local!r}")
        if not math.isfinite(float(self.offset_m)):
            raise ValueError(
                f"floor plane offset must be finite: {self.offset_m!r}")
        norm = math.sqrt(sum(value * value for value in normal))
        if abs(norm - 1.0) > 1e-9:
            raise ValueError(
                f"floor plane normal must be unit length, got {norm!r}")
        if normal[1] <= 0.0:
            raise ValueError(
                f"floor plane normal must point upwards (n_y > 0), got "
                f"{normal[1]!r}")
        object.__setattr__(self, "normal_local", normal)
        object.__setattr__(self, "offset_m", float(self.offset_m))
    def height_above(self, point: Sequence[float]) -> float:
        """Signed normal distance from ``point`` to the plane (Convention A)."""
        x, y, z = (float(value) for value in point)
        n_x, n_y, n_z = self.normal_local
        return n_x * x + n_y * y + n_z * z + self.offset_m
    def height_above_points(self, points) -> np.ndarray:
        """Vectorised :meth:`height_above` over an ``(N, 3)`` point array.
        Every band test in the pipeline -- obstacle band, ground support,
        contact attribution -- asks this one question, so they all share this
        one definition and cannot drift into a vertical-difference convention
        that would disagree with the published height.
        """
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(
                f"expected an (N, 3) point array, got {array.shape}")
        return array @ np.asarray(self.normal_local, dtype=np.float64) \
            + self.offset_m
    def y_at(self, x: float, z: float) -> float:
        """Height of the plane under ``(x, z)`` -- where a ground sample lies.
        Distinct from :meth:`height_above`, deliberately: this answers *where
        the surface is* (for reprojecting a ground sample), that one answers
        *how far above it* a point sits (for a band test). They coincide only
        on a level floor.
        """
        n_x, n_y, n_z = self.normal_local
        return -(n_x * float(x) + n_z * float(z) + self.offset_m) / n_y
