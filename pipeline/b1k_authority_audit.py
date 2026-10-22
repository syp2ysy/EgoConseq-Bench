"""Read-only inventory of historical B1K scene-authority failures."""

from __future__ import annotations



def _failure_category(error: str) -> str:
    normalized = str(error).lower()
    if "empty clearance-conditioned free space" in normalized:
        return "empty_clearance"
    if "position drift" in normalized:
        return "reset_position_drift"
    if "orientation drift" in normalized:
        return "reset_orientation_drift"
    if "joint" in normalized and "drift" in normalized:
        return "reset_joint_drift"
    if "triangle authority drift" in normalized:
        return "triangle_authority_drift"
    if "timeoutexpired" in normalized or "timed out" in normalized:
        return "timeout"
    if "no such file or directory" in normalized:
        return "missing_fragment"
    return "other"


