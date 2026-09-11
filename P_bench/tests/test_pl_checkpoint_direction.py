"""A4 intermediate Forward checkpoint geometry."""

import pytest
from pipeline import actions, checkpoint_direction

def test_forward_checkpoint_is_inside_the_requested_forward_stage():
    program = [
        actions.Forward(1.0), actions.Turn(90.0), actions.Forward(2.0)]

    checkpoint = checkpoint_direction.build_forward_checkpoint(
        program, forward_stage=2, fraction=0.5)

    assert checkpoint["action_index"] == 3
    assert checkpoint["arc_m"] == 2.0
    assert checkpoint["pose"] == pytest.approx({
        "x": 1.0, "z": 1.0, "heading_deg": 90.0})


def test_checkpoint_selection_is_deterministic_and_uses_all_fractions():
    program = [actions.Forward(1.0)]
    first = checkpoint_direction.select_forward_checkpoint(
        program, seed="same")
    observed = {
        checkpoint_direction.select_forward_checkpoint(
            program, seed=f"seed-{index}")["fraction"]
        for index in range(60)
    }

    assert checkpoint_direction.select_forward_checkpoint(
        program, seed="same") == first
    assert observed == {0.25, 0.5, 0.75}
