from collections import Counter
from itertools import islice

import pytest

from pipeline import a2
from pipeline import actions as action_geometry
from pipeline.actions import Forward, Turn


def test_action_cell_uses_forward_ordinal_and_distance_rank():
    actions = [
        Forward(3.0), Turn(30.0), Forward(1.0), Turn(-15.0),
        Forward(2.0),
    ]

    assert a2.action_cell(actions, collision_action_index_1based=5) == \
        a2.A2Cell(3, "middle")
    assert a2.action_cell(actions, collision_action_index_1based=1) == \
        a2.A2Cell(1, "longest")


def test_action_cell_allows_non_target_ties_but_rejects_target_ties():
    actions = [
        Forward(0.5), Turn(30.0), Forward(0.5), Turn(-15.0),
        Forward(2.0),
    ]

    assert a2.action_cell(
        actions, collision_action_index_1based=5) == \
        a2.A2Cell(3, "longest")

    with pytest.raises(ValueError, match="distinct"):
        a2.action_cell(actions, collision_action_index_1based=1)


@pytest.mark.parametrize(
    ("length", "expected_cells", "minimum", "maximum"),
    [(3, 4, 25, 25), (4, 4, 25, 25),
     (5, 9, 11, 12), (6, 9, 11, 12)],
)
def test_balanced_cells_jointly_balance_ordinal_and_rank(
        length, expected_cells, minimum, maximum):
    counts = Counter(a2.balanced_cells(
        length, 100, starts_with="forward"))

    assert len(counts) == expected_cells
    assert min(counts.values()) == minimum
    assert max(counts.values()) == maximum


def test_candidate_programs_obey_requested_cell_and_action_grammar():
    cell = a2.A2Cell(2, "shortest")
    programs = list(islice(
        a2.candidate_programs(length=5, cell=cell, seed=7), 20))

    assert len(programs) == 20
    for actions in programs:
        assert [type(action) for action in actions] == [
            Forward, Turn, Forward, Turn, Forward]
        assert a2.action_cell(
            actions, collision_action_index_1based=3) == cell


def test_candidate_programs_support_turn_first_even_horizons():
    cell = a2.A2Cell(3, "longest")
    actions = next(a2.candidate_programs(
        length=6, cell=cell, starts_with="turn", seed=3))

    assert [type(action) for action in actions] == [
        Turn, Forward, Turn, Forward, Turn, Forward]
    assert a2.action_cell(
        actions, collision_action_index_1based=6) == cell


def test_a2_turn_first_programs_use_only_the_initial_turn_vocabulary():
    programs = list(islice(a2.candidate_programs(
        length=6, cell=a2.A2Cell(3, "longest"),
        starts_with="turn", seed=3), 100))

    assert programs
    assert {program[0].deg for program in programs} <= {-30, -15, 15, 30}
    assert any(abs(program[2].deg) == 45 for program in programs)


def test_a2_has_no_turn_first_l3_design_cell():
    assert a2.cells_for_length(3, starts_with="turn") == ()
    assert a2.balanced_cells(
        3, 100, starts_with="turn") == ()


def test_turn_first_l5_has_only_two_forward_ordinals():
    cells = a2.cells_for_length(5, starts_with="turn")

    assert {cell.forward_ordinal_1based for cell in cells} == {1, 2}
    assert {cell.distance_rank for cell in cells} == {"shortest", "longest"}


def test_collision_index_is_derived_from_the_requested_action_pattern():
    cell = a2.A2Cell(2, "longest")

    assert a2.collision_action_index(
        cell, length=5, starts_with="forward") == 3
    assert a2.collision_action_index(
        cell, length=5, starts_with="turn") == 4

    with pytest.raises(ValueError, match="Forward ordinal"):
        a2.collision_action_index(
            a2.A2Cell(3, "middle"), length=5, starts_with="turn")


class _TargetLegProxy:
    radius_m = 0.2

    def rollout(self, actions):
        target = float(actions[2].m)
        if target >= 1.5:
            return {
                "collision": True,
                "contact_action_index": 2,
                "contact_action_local_arc_m": 1.0,
                "first_contact_arc_m": float(actions[0].m) + 1.0,
            }
        return {
            "collision": False,
            "contact_action_index": None,
            "contact_action_local_arc_m": None,
            "first_contact_arc_m": None,
        }

    def coverage(self, actions, max_arc_m=None):
        return 1.0


def test_propose_collision_keeps_requested_rank():
    cell = a2.A2Cell(2, "shortest")

    proposal = a2.propose_collision(
        _TargetLegProxy(), length=5, cell=cell, seed=11,
        half_fov_deg=180.0)

    assert proposal is not None
    assert a2.action_cell(
        proposal.collision_actions,
        collision_action_index_1based=3) == cell


class _NoCollisionProxy:
    radius_m = 0.2

    def __init__(self):
        self.rollout_calls = 0

    def rollout(self, actions):
        self.rollout_calls += 1
        return {"collision": False}

    def coverage(self, actions, max_arc_m=None):
        return 1.0


def test_proposal_search_evaluates_each_causal_prefix_once():
    proxy = _NoCollisionProxy()

    assert list(a2.collision_proposals(
        proxy, length=6, cell=a2.A2Cell(1, "shortest"), seed=19,
        half_fov_deg=180.0, max_programs=1000)) == []

    # The first Forward fully determines whether Action 1 collides.  Suffix
    # variants must not repeat the same physical query hundreds of times.
    assert proxy.rollout_calls <= 12


def _outcome(actions, *, collision, collision_action_index=2):
    local = 0.6
    return {
        "actions": action_geometry.actions_to_dicts(actions),
        "body": {"shape": "disc", "radius_m": 0.2},
        "physical": {
            "collision": collision,
            "contact_action_index": collision_action_index,
            "contact_action_local_arc_m": local,
        },
        "depth_physical": {
            "collision": collision,
            "contact_action_index": collision_action_index,
            "contact_action_local_arc_m": local,
        },
        "oracle_consensus": {"accepted": True},
        "execution": {
            "completed": not collision,
            "stop_reason": "completed" if not collision else "collision",
        },
        "shared_oracle_stability": {"summary": {
            "collision": collision,
            "collision_label_stable": collision,
            "original_action_index_stable": collision,
            "original_action_index": collision_action_index + 1,
        }},
    }


def test_design_certificate_binds_the_balanced_cell():
    collision_actions = (
        Forward(3.0), Turn(30.0), Forward(1.0), Turn(-15.0), Forward(2.0))

    certificate = a2.build_design_certificate(
        _outcome(collision_actions, collision=True),
        collision_action_index_1based=3)

    assert certificate["protocol"] == a2.PROTOCOL
    assert certificate["cell"] == {
        "forward_ordinal_1based": 2,
        "distance_rank": "shortest",
    }
    assert a2.validate_design_certificate(
        _outcome(collision_actions, collision=True), certificate) == []


def test_design_certificate_rejects_an_unstable_collision_action():
    collision_actions = (Forward(1.0), Turn(30.0), Forward(3.0))
    outcome = _outcome(
        collision_actions, collision=True, collision_action_index=0)
    outcome["shared_oracle_stability"]["summary"][
        "original_action_index_stable"] = False

    with pytest.raises(ValueError, match="not stable"):
        a2.build_design_certificate(
            outcome, collision_action_index_1based=1)
