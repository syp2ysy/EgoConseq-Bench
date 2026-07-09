"""Body shapes. Only Cylinder for now; extension point for future forms."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline import config


@dataclass(frozen=True)
class Cylinder:
    radius_m: float = 0.25
    height_m: float = config.CYLINDER_HEIGHT_M
    shape: str = "cylinder"

    def to_dict(self) -> dict:
        return {"shape": self.shape,
                "radius_m": float(self.radius_m),
                "height_m": float(self.height_m)}
