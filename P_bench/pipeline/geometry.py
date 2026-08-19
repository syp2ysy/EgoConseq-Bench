"""Deterministic planar-body and ground-support geometry queries."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


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


def points_to_ground_support_distances_m(
        points_xz, support: dict) -> np.ndarray:
    """Return exact distances from finite ``Nx2`` XZ points to a support.

    Components are evaluated in float64. Triangle/segment operations match
    the scalar semantic authority while amortizing them across endpoint rows.
    """
    queries = np.asarray(points_xz, dtype=np.float64)
    if (queries.ndim != 2 or queries.shape[1:] != (2,) or
            not np.all(np.isfinite(queries))):
        raise ValueError("support query points must be finite Nx2 XZ")
    if not isinstance(support, dict):
        raise ValueError("ground support must be an object")
    try:
        triangles = np.asarray(
            support["triangles_xz_m"], dtype=np.float64).reshape(-1, 3, 2)
        segments = np.asarray(
            support["segments_xz_m"], dtype=np.float64).reshape(-1, 2, 2)
        points = np.asarray(
            support["points_xz_m"], dtype=np.float64).reshape(-1, 2)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ground support components are invalid") from error
    if not all(np.all(np.isfinite(value))
               for value in (triangles, segments, points)):
        raise ValueError("ground support components must be finite")

    triangle_edges = None
    triangle_valid = None
    if len(triangles):
        triangle_edges = np.stack([
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ], axis=1)
        triangle_valid = (
            triangle_edges[:, 0, 0] *
            (triangles[:, 2, 1] - triangles[:, 0, 1]) -
            triangle_edges[:, 0, 1] *
            (triangles[:, 2, 0] - triangles[:, 0, 0])) != 0.0
    segment_edges = segments[:, 1] - segments[:, 0]
    segment_lengths_sq = np.sum(segment_edges * segment_edges, axis=1)
    distances = []
    for query in queries:
        best = math.inf
        if len(triangles):
            relative = query - triangles
            signs = (
                triangle_edges[:, :, 0] * relative[:, :, 1] -
                triangle_edges[:, :, 1] * relative[:, :, 0])
            inside = triangle_valid & (
                np.all(signs >= 0.0, axis=1) |
                np.all(signs <= 0.0, axis=1))
            if np.any(inside):
                distances.append(0.0)
                continue
            lengths_sq = np.sum(
                triangle_edges * triangle_edges, axis=2)
            numerators = np.sum(relative * triangle_edges, axis=2)
            fractions = np.divide(
                numerators, lengths_sq, out=np.zeros_like(numerators),
                where=lengths_sq != 0.0)
            fractions = np.clip(fractions, 0.0, 1.0)
            nearest = triangles + fractions[:, :, None] * triangle_edges
            triangle_distances = np.linalg.norm(query - nearest, axis=2)
            triangle_distances[~triangle_valid, :] = math.inf
            best = float(np.min(triangle_distances))
        if len(segments):
            fractions = np.divide(
                np.sum((query - segments[:, 0]) * segment_edges, axis=1),
                segment_lengths_sq, out=np.zeros_like(segment_lengths_sq),
                where=segment_lengths_sq != 0.0)
            fractions = np.clip(fractions, 0.0, 1.0)
            nearest = segments[:, 0] + fractions[:, None] * segment_edges
            best = min(best, float(np.min(np.linalg.norm(
                query - nearest, axis=1))))
        if len(points):
            best = min(best, float(np.min(np.linalg.norm(
                query - points, axis=1))))
        if not math.isfinite(best):
            raise ValueError("ground support is empty")
        distances.append(best)
    return np.asarray(distances, dtype=np.float64)
