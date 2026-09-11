"""Deterministic planar-body geometry queries."""

from __future__ import annotations

from dataclasses import dataclass, field


EXCLUDED_GEOMETRY_SOURCES = frozenset({
    "navmesh_boundary",
    "unsupported_floor",
})


@dataclass(frozen=True)
class Disc:
    """Circular ground-plane robot footprint."""
    radius_m: float = 0.25
    shape: str = field(default="disc", init=False)
    def __post_init__(self):
        if self.radius_m <= 0:
            raise ValueError("disc radius must be positive")
    def to_dict(self) -> dict:
        return {"shape": self.shape, "radius_m": float(self.radius_m)}


@dataclass(frozen=True)
class GeometryQuery:
    """One planar-footprint query against a physical geometry authority."""
    navigable: bool
    clearance_m: float
    obstacle_index: int | None = None
    geometry_source: str | None = None
    ground_covered_count: int | None = None
    ground_total_count: int | None = None
