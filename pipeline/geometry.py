"""Deterministic planar-body geometry queries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryQuery:
    """One planar-footprint query against a physical geometry authority."""
    navigable: bool
    clearance_m: float
