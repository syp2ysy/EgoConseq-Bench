"""Accessors for normalized fields within the current outcome schema."""

from __future__ import annotations

import math

from pipeline.geometry import EXCLUDED_GEOMETRY_SOURCES

EXECUTION_DISTANCE_TOL_M = 1e-4


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def realized_pose(outcome: dict) -> dict:
    """Return the finite realized local SE(2) pose stored by execution."""
    pose = (outcome.get("execution") or {}).get("realized_pose")
    if not isinstance(pose, dict):
        raise ValueError("execution realized pose is missing")
    try:
        return {
            key: _finite_number(pose.get(key), f"realized pose {key}")
            for key in ("x", "z", "heading_deg")
        }
    except ValueError as error:
        raise ValueError("execution realized pose is invalid") from error


def derive_execution_regime(outcome: dict) -> str:
    """Classify how one physical action execution ended.
    The value is derived only from private physical-oracle fields. It is safe
    to persist for validation and review, but must never be exposed as public
    model input because it reveals the A1 collision label.
    """
    physical = outcome.get("physical") or {}
    execution = outcome.get("execution") or {}
    collision = physical.get("collision")
    authority = str(physical.get("authority") or "unavailable")
    source = (
        physical.get("collision_source") or
        physical.get("contact_source") or
        (physical.get("contact") or {}).get("source")
    )
    if authority == "unavailable" or source in EXCLUDED_GEOMETRY_SOURCES:
        return "invalid_geometry"
    if not isinstance(collision, bool):
        if collision is None:
            return "invalid_geometry"
        raise ValueError("physical collision must be Boolean or null")
    completed = execution.get("completed")
    stop_reason = execution.get("stop_reason")
    nominal = _finite_number(
        execution.get("nominal_forward_m"), "nominal_forward_m")
    executed = _finite_number(
        execution.get("executed_forward_m"), "executed_forward_m")
    if collision is False:
        if completed is not True or stop_reason != "completed":
            raise ValueError(
                "collision-free execution state is inconsistent")
        if abs(executed - nominal) > EXECUTION_DISTANCE_TOL_M:
            raise ValueError(
                "completed execution state does not cover the full action")
        return "completed_clear"
    if completed is not False or stop_reason != "collision":
        raise ValueError("collision execution state is inconsistent")
    contact_arc = _finite_number(
        physical.get("first_contact_arc_m"), "first_contact_arc_m")
    stop_arc = _finite_number(execution.get("stop_arc_m"), "stop_arc_m")
    if (abs(contact_arc - stop_arc) > EXECUTION_DISTANCE_TOL_M or
            executed > nominal + EXECUTION_DISTANCE_TOL_M):
        raise ValueError("collision execution state has an invalid stop")
    return "contact_truncated"


def is_completed_clear(outcome: dict) -> bool:
    """Return whether an outcome is a valid collision-free full execution."""
    try:
        return derive_execution_regime(outcome) == "completed_clear"
    except (TypeError, ValueError):
        return False
