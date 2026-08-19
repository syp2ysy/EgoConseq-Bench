"""Execution-regime derivation from untrusted physical rollout fields."""

import pytest

from pipeline import outcome


def _execution_outcome(*, collision, completed, stop_reason,
                       authority="navmesh", collision_source=None):
    value = {
        "physical": {
            "authority": authority,
            "collision": collision,
            "first_contact_arc_m": 0.4 if collision is True else None,
        },
        "execution": {
            "completed": completed,
            "stop_reason": stop_reason,
            "nominal_forward_m": 1.5,
            "executed_forward_m": 0.4 if collision is True else 1.5,
            "stop_arc_m": 0.4 if collision is True else None,
        },
    }
    if collision_source is not None:
        value["physical"]["collision_source"] = collision_source
    return value


def test_execution_regime_distinguishes_completed_contact_and_invalid():
    assert outcome.derive_execution_regime(_execution_outcome(
        collision=False, completed=True,
        stop_reason="completed")) == "completed_clear"
    assert outcome.derive_execution_regime(_execution_outcome(
        collision=True, completed=False,
        stop_reason="collision")) == "contact_truncated"
    assert outcome.derive_execution_regime(_execution_outcome(
        collision=None, completed=None, stop_reason="physical_gt_unavailable",
        authority="unavailable")) == "invalid_geometry"
    assert outcome.derive_execution_regime(_execution_outcome(
        collision=True, completed=False, stop_reason="collision",
        collision_source="unsupported_floor")) == "invalid_geometry"


def test_completed_clear_predicate_fails_closed_on_invalid_outcomes():
    clear = _execution_outcome(
        collision=False, completed=True, stop_reason="completed")
    contact = _execution_outcome(
        collision=True, completed=False, stop_reason="collision")

    assert outcome.is_completed_clear(clear) is True
    assert outcome.is_completed_clear(contact) is False
    assert outcome.is_completed_clear({}) is False


def test_realized_pose_comes_directly_from_execution():
    value = {
        "execution": {"realized_pose": {
            "x": 0.25, "z": 1.5, "heading_deg": -30.0,
        }},
    }

    assert outcome.realized_pose(value) == {
        "x": 0.25, "z": 1.5, "heading_deg": -30.0}


@pytest.mark.parametrize("pose", [
    None,
    {"x": 0.0, "z": 0.0},
    {"x": True, "z": 0.0, "heading_deg": 0.0},
    {"x": float("nan"), "z": 0.0, "heading_deg": 0.0},
])
def test_realized_pose_rejects_missing_or_nonfinite_execution(pose):
    with pytest.raises(ValueError, match="realized pose"):
        outcome.realized_pose({"execution": {"realized_pose": pose}})


@pytest.mark.parametrize(
    "collision,completed,stop_reason",
    [
        (False, False, "completed"),
        (False, True, "collision"),
        (True, True, "collision"),
        (True, False, "completed"),
    ],
)
def test_execution_regime_rejects_internally_inconsistent_rollouts(
        collision, completed, stop_reason):
    with pytest.raises(ValueError, match="execution state"):
        outcome.derive_execution_regime(_execution_outcome(
            collision=collision,
            completed=completed,
            stop_reason=stop_reason,
        ))
