"""Depth-conditioned action proposal: per-pose materialisation of templates.

The proposal layer is pure geometry over one initial depth frame. It never sees
navmesh, consensus, or a published label, so every test here injects a scripted
depth proxy rather than a simulator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline import action_proposal as AP
from pipeline import action_control_catalog
from pipeline import actions as A
from pipeline import config
from pipeline import reach_probe
from pipeline.actions import Forward, Turn


# --------------------------------------------------------------------------
# A scripted stand-in for the initial depth frame.
#
# The model is deliberately the same one the real proxy measures: each Forward
# leg has its own longitudinal free distance, and a leg collides exactly when
# its materialised distance exceeds that leg's free distance. ``None`` means the
# probe saw nothing out to ``REACH_PROBE_MAX_M`` -- unbounded, not "free to
# exactly that distance".
#
# ``reach`` is scripted separately from the per-leg free distances because the
# two measure different things: free distance is where the swept disc would hit
# something, reach is how far the corridor's floor is actually evidenced by this
# one depth image. A wall 2.0 m ahead does not make the corridor before it
# unobserved. The default is a fully open probe so the per-leg cap only binds in
# the tests that deliberately script a short reach; pass a callable to make reach
# depend on the prefix (its travel, or its heading after a Turn).
# --------------------------------------------------------------------------

class ScriptedProxy:
    def __init__(self, free_per_leg, *, coverage=1.0,
                 reach=reach_probe.REACH_PROBE_MAX_M, state=None):
        self._free = list(free_per_leg)
        self._coverage = coverage
        self._reach = reach
        self._state = state
        self.free_calls = []
        self.reach_calls = []

    def prefix_supported_reach_m(self, prefix):
        prefix = list(prefix)
        self.reach_calls.append(A.pose_after(prefix))
        return float(self._reach(prefix) if callable(self._reach)
                     else self._reach)

    def free_distance_state(self, prefix):
        prefix = list(prefix)
        if self._state is not None:
            state = (self._state(prefix) if callable(self._state)
                     else self._state)
            return {"state": state}
        # Default: mirror the scripted geometry. A leg with a free distance is a
        # boundary this depth saw; ``None`` is the open-through-probe case.
        free = self._leg_free(
            sum(1 for action in prefix if isinstance(action, Forward)))
        return {"state": (reach_probe.STATE_COLLISION if free is not None
                          else reach_probe.STATE_OPEN),
                "first_contact_arc_m": free}

    def _leg_free(self, index):
        # Cycles, so repeated materialisations of the same template shape see
        # the same geometry instead of running off the end of the script.
        return self._free[index % len(self._free)] if self._free else None

    def free_distance_m(self, base_pose):
        self.free_calls.append(tuple(round(v, 6) for v in base_pose))
        return self._leg_free(len(self.free_calls) - 1)

    def rollout(self, actions):
        cumulative = 0.0
        leg = 0
        for index, action in enumerate(actions):
            if not isinstance(action, Forward):
                continue
            free = self._leg_free(leg)
            if free is not None and action.m > free + 1e-9:
                return {
                    "collision": True,
                    "first_contact_arc_m": cumulative + free,
                    "contact_action_index": index,
                    "contact_action_local_arc_m": free,
                }
            cumulative += action.m
            leg += 1
        return {
            "collision": False, "first_contact_arc_m": None,
            "contact_action_index": None, "contact_action_local_arc_m": None,
        }

    def coverage(self, actions, max_arc_m=None):
        return (self._coverage(actions, max_arc_m)
                if callable(self._coverage) else self._coverage)


def template(**overrides):
    values = {
        "template_id": "T-000",
        "length": 3,
        "starts_with": "forward",
        "turn_angles": (30.0,),
        "target_forward_leg_number": 2,
        "distance_ranks": (0,),
    }
    values.update(overrides)
    return AP.Template(**values)


# --------------------------------------------------------------------------
# Grid quantisation
# --------------------------------------------------------------------------

def test_grid_is_the_frozen_generator_vocabulary():
    assert AP.GRID_M == tuple(config.GEN_FORWARDS_M)
    assert AP.GLOBAL_MAX_FORWARD_M == max(config.GEN_FORWARDS_M)


def test_the_probe_bound_is_a_sensing_property_not_a_publication_one():
    """Reading it off the vocabulary is what conflated the two in v1.

    They are equal at v2 by coincidence of the chosen numbers, so the binding
    -- not the value -- is what this pins.
    """
    assert AP.PROBE_DISTANCE_M == reach_probe.REACH_PROBE_MAX_M


def test_production_bank_exposes_over_200_actions_and_uses_v4_pose_limits():
    """The bank is wide, the shortlist is narrow, and the gap is the saving.

    Progressive filling may inspect up to 48 ordinary actions, but stops once
    36 stable actions are found. The record can add at most 12 C1 neighbours,
    while its published ``candidate_budget`` remains capped at 48.
    """
    assert config.MAIN_ACTION_PROPOSAL_PER_LENGTH == 40
    assert AP.NATURAL_PER_LENGTH_DEFAULT == 40
    assert AP.PAIRS_PER_LENGTH_DEFAULT == 12
    assert config.ACTION_CANDIDATE_MAX_PER_POSE == 48
    assert config.ACTION_CANDIDATE_ORDINARY_PER_POSE == 36
    assert config.C1_QUERIES_PER_POSE == 2
    assert config.C1_NEIGHBORS_PER_QUERY == 6
    assert config.C1_NEIGHBOR_SLOTS_PER_POSE == 12
    assert (config.ACTION_CANDIDATE_ORDINARY_PER_POSE +
            config.C1_NEIGHBOR_SLOTS_PER_POSE ==
            config.ACTION_CANDIDATE_MAX_PER_POSE)
    assert len(config.GEN_FORWARDS_M) + 5 * AP.NATURAL_PER_LENGTH_DEFAULT >= 200


def test_dynamic_natural_draw_uses_one_pose_budget_without_a_boundary_query():
    """A natural action is drawn from depth support, then labelled downstream.

    The boundary-paired arm is allowed to query ``free_distance_m`` because it
    exists to straddle that boundary.  The natural arm must not: otherwise its
    action text would encode which side of the boundary the generator wanted.
    """
    proxy = ScriptedProxy([], reach=5.0)

    bank = AP.build_dynamic_natural_bank(
        np.random.default_rng(17), proxy, lengths=(6,), per_length=12,
        half_fov_deg=90.0)

    assert bank[6]
    assert proxy.free_calls == []
    assert {candidate.variant for candidate in bank[6]} == {
        AP.NATURAL_DYNAMIC_VARIANT}
    for candidate in bank[6]:
        forwards = [action.m for action in candidate.actions
                    if isinstance(action, Forward)]
        assert len(forwards) == 3
        assert max(forwards) <= 1.5
        assert all(value in AP.GRID_M for value in forwards)


def test_dynamic_natural_draw_is_seeded_per_pose_and_reproducible():
    def draw(seed):
        bank = AP.build_dynamic_natural_bank(
            np.random.default_rng(seed), ScriptedProxy([], reach=4.0),
            lengths=(3, 4), per_length=8, half_fov_deg=90.0)
        return {
            length: [A.actions_to_dicts(list(candidate.actions))
                     for candidate in candidates]
            for length, candidates in bank.items()
        }

    assert draw(29) == draw(29)
    assert draw(29) != draw(30)


def test_dynamic_natural_bank_stratifies_10_10_20_by_depth_reach():
    bank = AP.build_dynamic_natural_bank(
        np.random.default_rng(20260813), ScriptedProxy([], reach=6.0),
        lengths=(6,), per_length=40, half_fov_deg=180.0)

    strata = {
        name: [candidate for candidate in bank[6]
               if candidate.provenance["natural_distance_stratum"] == name]
        for name in ("short", "mid", "near")
    }
    assert {name: len(values) for name, values in strata.items()} == {
        "short": 10, "mid": 10, "near": 20}
    starts = [candidate.provenance["starts_with"]
              for values in strata.values() for candidate in values]
    assert starts.count("forward") == starts.count("turn") == 20
    maxima = {
        name: {
            max(action.m for action in candidate.actions
                if isinstance(action, Forward))
            for candidate in values
        }
        for name, values in strata.items()
    }
    assert max(maxima["short"]) <= min(maxima["mid"])
    assert max(maxima["mid"]) <= min(maxima["near"])
    assert all(candidate.variant == AP.NATURAL_DYNAMIC_VARIANT
               for values in strata.values() for candidate in values)


def test_small_natural_grids_assign_each_distance_to_one_stratum():
    assert AP._natural_stratum_grid((0.5,), "short") == ()
    assert AP._natural_stratum_grid((0.5,), "mid") == ()
    assert AP._natural_stratum_grid((0.5,), "near") == (0.5,)

    strata = {
        name: AP._natural_stratum_grid((0.5, 1.0), name)
        for name in ("short", "mid", "near")
    }
    assert strata == {
        "short": (0.5,),
        "mid": (),
        "near": (1.0,),
    }
    assert len(set().union(*map(set, strata.values()))) == sum(
        len(values) for values in strata.values())


def test_fov_mask_cache_preserves_draws_and_rejection_accounting():
    for seed in range(5):
        cached_rejections, direct_rejections = {}, {}
        cached_stats, direct_stats = {}, {}
        cached = AP.build_dynamic_natural_bank(
            np.random.default_rng(seed), ScriptedProxy([], reach=4.0),
            lengths=(3, 4, 5), per_length=12, half_fov_deg=39.5,
            rejections=cached_rejections, stats=cached_stats)
        direct = AP.build_dynamic_natural_bank(
            np.random.default_rng(seed), ScriptedProxy([], reach=4.0),
            lengths=(3, 4, 5), per_length=12, half_fov_deg=39.5,
            rejections=direct_rejections, stats=direct_stats,
            _use_fov_cache=False)

        assert cached == direct
        assert cached_rejections == direct_rejections
        assert cached_stats == direct_stats


def test_control_actions_are_screened_without_being_rewritten():
    anchors = action_control_catalog.load_catalog()[:5]
    proxy = ScriptedProxy([], reach=6.0)

    bank = AP.build_control_bank(
        anchors, proxy, half_fov_deg=90.0)

    candidates = [candidate for rows in bank.values()
                  for candidate in rows]
    assert len(candidates) == len(anchors)
    assert proxy.free_calls == []
    assert {candidate.variant for candidate in candidates} == {
        AP.A1_CONTROL_VARIANT}
    expected = {
        anchor.tag: A.actions_to_dicts(list(anchor.actions))
        for anchor in anchors
    }
    assert {
        candidate.provenance["template_id"]:
            A.actions_to_dicts(list(candidate.actions))
        for candidate in candidates
    } == expected
    assert all(
        candidate.provenance["control_catalog_sha256"] ==
        action_control_catalog.CATALOG_SHA256
        for candidate in candidates)


@pytest.mark.parametrize("value,expected", [
    (0.49, None), (0.5, 0.5), (0.99, 0.5), (1.0, 1.0), (3.4, 3.0), (99.0, 6.0),
])
def test_floor_grid(value, expected):
    assert AP.floor_grid(value) == expected


@pytest.mark.parametrize("value,expected", [
    (0.0, 0.5), (0.5, 0.5), (0.51, 1.0), (3.0, 3.0), (3.01, 3.5), (6.01, None),
])
def test_ceil_grid(value, expected):
    assert AP.ceil_grid(value) == expected


def test_unbounded_free_distance_keeps_the_whole_grid():
    """``None`` means the probe saw no boundary, so no grid point is excluded.

    Substituting the probe bound would apply ``g <= 6.0 - 0.30`` and silently
    cap the leg at 5.5 m -- a leg that saw nothing would become shorter than one
    that saw a wall at the probe horizon.
    """
    assert AP.feasible_grid(None) == AP.GRID_M
    assert AP.pick_distance(None, rank=0) == AP.GLOBAL_MAX_FORWARD_M
    assert AP.pick_distance(None, rank=0) > AP.pick_distance(6.0, rank=0)


def test_feasible_grid_reserves_the_safe_clearance():
    assert AP.feasible_grid(2.0) == (0.5, 1.0, 1.5)
    assert AP.feasible_grid(0.7) == ()
    assert AP.pick_distance(0.7, rank=0) is None


def test_pick_distance_ranks_downward_and_wraps():
    assert AP.pick_distance(2.0, rank=0) == 1.5
    assert AP.pick_distance(2.0, rank=1) == 1.0
    assert AP.pick_distance(2.0, rank=2) == 0.5
    assert AP.pick_distance(2.0, rank=3) == 1.5


# --------------------------------------------------------------------------
# Index semantics: primitive index vs Forward leg number
# --------------------------------------------------------------------------

def test_action_pattern_alternates_from_the_declared_start():
    assert AP.action_pattern(5, "forward") == (
        "forward", "turn", "forward", "turn", "forward")
    assert AP.action_pattern(4, "turn") == (
        "turn", "forward", "turn", "forward")


def test_target_action_index_is_not_the_forward_leg_number():
    """`contact_action_index` indexes primitives; the leg number counts Forwards."""
    spec = template(length=5, starts_with="forward", turn_angles=(15.0, -15.0),
                    target_forward_leg_number=2, distance_ranks=(0, 0))
    assert spec.forward_action_indices == (0, 2, 4)
    assert spec.target_forward_leg_number == 2
    assert spec.target_action_index == 2
    assert spec.forward_count == 3


def test_a2_capable_requires_two_forward_legs():
    assert template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1,
                    distance_ranks=()).a2_capable is False
    assert template(length=3, starts_with="turn", turn_angles=(30.0, 30.0),
                    target_forward_leg_number=1,
                    distance_ranks=()).a2_capable is False
    assert template(length=3, starts_with="forward", turn_angles=(30.0,),
                    target_forward_leg_number=1,
                    distance_ranks=(0,)).a2_capable is True


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def test_local_guard_publication_term_dominates_without_a_suffix():
    guard = AP.local_guard_m(suffix_forward_m=0.0, a2_capable=True)
    assert guard == pytest.approx(
        config.BENCH_COLLISION_REMAINING_M + config.ORACLE_CONTACT_TOL_M)


def test_local_guard_falls_to_the_a2_floor_once_the_suffix_pays():
    guard = AP.local_guard_m(suffix_forward_m=1.0, a2_capable=True)
    assert guard == pytest.approx(
        config.A2_ACTION_BOUNDARY_MARGIN_M + config.ORACLE_CONTACT_TOL_M)


def test_local_guard_drops_the_a2_term_for_single_forward_programs():
    # A non-A2 program has exactly one Forward and therefore no suffix, so this
    # first combination is unreachable from ``materialize_pair``; it is asserted
    # here to pin the pure function's contract, not a materialisable state.
    assert AP.local_guard_m(suffix_forward_m=2.0, a2_capable=False) == 0.0
    assert AP.local_guard_m(suffix_forward_m=0.0, a2_capable=False) == pytest.approx(
        config.BENCH_COLLISION_REMAINING_M + config.ORACLE_CONTACT_TOL_M)


def test_remaining_budget_cap_scales_only_with_the_suffix_leg_count():
    assert AP.remaining_budget_cap_m(0) == pytest.approx(1.30)
    assert AP.remaining_budget_cap_m(2) == pytest.approx(2.30)


# --------------------------------------------------------------------------
# Pair materialisation
# --------------------------------------------------------------------------

def test_pair_differs_only_at_the_target_leg():
    spec = template(length=5, starts_with="forward", turn_angles=(15.0, -15.0),
                    target_forward_leg_number=2, distance_ranks=(0, 3))
    proxy = ScriptedProxy([2.5, 2.0, 2.5])
    pair = AP.materialize_pair(spec, proxy, half_fov_deg=90.0)

    safe = list(pair.safe.actions)
    collision = list(pair.collision.actions)
    assert len(safe) == len(collision) == 5
    differing = [i for i, (a, b) in enumerate(zip(safe, collision)) if a != b]
    assert differing == [spec.target_action_index]
    assert all(value in AP.GRID_M
               for program in (safe, collision)
               for value in (a.m for a in program if isinstance(a, Forward)))


def test_collision_variant_contacts_inside_the_target_leg():
    spec = template(length=5, starts_with="forward", turn_angles=(15.0, -15.0),
                    target_forward_leg_number=2, distance_ranks=(0, 3))
    proxy = ScriptedProxy([2.5, 2.0, 2.5])
    pair = AP.materialize_pair(spec, proxy, half_fov_deg=90.0)

    verdict = proxy.rollout(list(pair.collision.actions))
    assert verdict["collision"] is True
    assert verdict["contact_action_index"] == spec.target_action_index
    assert proxy.rollout(list(pair.safe.actions))["collision"] is False


def test_collision_overshoot_respects_the_publication_and_a2_guards():
    spec = template(length=3, starts_with="forward", turn_angles=(15.0,),
                    target_forward_leg_number=2, distance_ranks=(0,))
    proxy = ScriptedProxy([2.0, 1.4])
    pair = AP.materialize_pair(spec, proxy, half_fov_deg=90.0)

    b_j = pair.collision.provenance["b_j_m"]
    target = list(pair.collision.actions)[spec.target_action_index]
    assert target.m - b_j >= config.BENCH_COLLISION_REMAINING_M + \
        config.ORACLE_CONTACT_TOL_M - 1e-9
    assert pair.collision.provenance["remaining_proxy_m"] <= \
        AP.remaining_budget_cap_m(0) + 1e-9


def test_a2_near_boundary_filter_is_dominated_by_the_safe_grid_floor():
    """Under the frozen constants this filter cannot fire, and that is fine.

    A pair also needs a safe variant, which needs
    ``b_j >= min(GRID_M) + BENCH_SAFE_CLEARANCE_M = 0.80``. That already exceeds
    the A2 near-end requirement of 0.60, so every ``b_j`` reaching the filter has
    passed it. The check stays because it is the constraint that is actually
    being asserted -- if the grid ever gains a finer first step, it starts
    mattering -- but its counter must stay zero, and a near boundary must be
    attributed to the bound that really killed it.
    """
    assert (config.A2_ACTION_BOUNDARY_MARGIN_M + config.ORACLE_CONTACT_TOL_M <
            min(AP.GRID_M) + config.BENCH_SAFE_CLEARANCE_M)
    rejections = {}
    two_leg = template(length=3, starts_with="forward", turn_angles=(15.0,),
                       target_forward_leg_number=1, distance_ranks=(0,))
    assert AP.materialize_pair(
        two_leg, ScriptedProxy([0.45, 2.0]), half_fov_deg=90.0,
        rejections=rejections) is None
    assert rejections == {"bracket_grid_exhausted_safe": 1}


def test_single_forward_programs_still_produce_pairs():
    single_leg = template(length=1, starts_with="forward", turn_angles=(),
                          target_forward_leg_number=1, distance_ranks=())
    pair = AP.materialize_pair(
        single_leg, ScriptedProxy([1.0]), half_fov_deg=90.0)
    assert pair is not None
    assert pair.collision.provenance["a2_capable"] is False
    # No suffix exists, so the publication term is the whole guard either way:
    # dropping the A2 term for these programs is a correctness statement, not a
    # yield change.
    assert pair.collision.provenance["local_guard_m"] == pytest.approx(
        config.BENCH_COLLISION_REMAINING_M + config.ORACLE_CONTACT_TOL_M)


def test_unbounded_target_leg_cannot_form_a_pair():
    rejections = {}
    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    assert AP.materialize_pair(
        spec, ScriptedProxy([None]), half_fov_deg=90.0,
        rejections=rejections) is None
    assert rejections == {"bracket_unbounded": 1}


def test_long_suffix_is_rejected_by_the_remaining_budget_cap():
    """A far suffix would revive the multi-metre nominal overshoot."""
    rejections = {}
    spec = template(length=5, starts_with="forward", turn_angles=(15.0, -15.0),
                    target_forward_leg_number=1, distance_ranks=(0, 0))
    proxy = ScriptedProxy([1.5, None, None])
    assert AP.materialize_pair(
        spec, proxy, half_fov_deg=90.0, rejections=rejections) is None
    assert rejections == {"template_remaining_budget_exhausted": 1}


def test_blocked_prefix_leg_abandons_the_template():
    rejections = {}
    spec = template(length=3, starts_with="forward", turn_angles=(15.0,),
                    target_forward_leg_number=2, distance_ranks=(0,))
    assert AP.materialize_pair(
        spec, ScriptedProxy([0.6, 2.0]), half_fov_deg=90.0,
        rejections=rejections) is None
    assert rejections == {"template_infeasible_prefix": 1}


def test_coverage_is_whole_path_for_safe_and_prefix_only_for_collision():
    seen = []

    def coverage(actions, max_arc_m):
        seen.append(max_arc_m)
        return 1.0

    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    proxy = ScriptedProxy([1.5], coverage=coverage)
    pair = AP.materialize_pair(spec, proxy, half_fov_deg=90.0)
    assert pair is not None
    # The safe variant is certified over its whole path; the collision variant
    # only up to first contact plus the oracle's arc tolerance, matching the
    # formal corridor gate rather than being stricter than it.
    assert seen[0] is None
    assert seen[1] == pytest.approx(1.5 + config.ORACLE_CONTACT_TOL_M)


def test_collision_fov_is_checked_only_up_to_contact(monkeypatch):
    """The nominal remainder is never executed, so it must not be required
    to stay in view -- that would be stricter than the formal corridor gate."""
    seen = []
    real = A.inside_initial_fov

    def spy(actions, half_fov_deg, *, max_arc_m=None):
        seen.append(max_arc_m)
        return real(actions, half_fov_deg, max_arc_m=max_arc_m)

    monkeypatch.setattr(A, "inside_initial_fov", spy)
    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    pair = AP.materialize_pair(
        spec, ScriptedProxy([1.5]), half_fov_deg=90.0)
    assert pair is not None
    assert seen == [None, pytest.approx(1.5)]


def test_materialisation_is_deterministic():
    spec = template(length=5, starts_with="forward", turn_angles=(15.0, -15.0),
                    target_forward_leg_number=2, distance_ranks=(1, 3))
    first = AP.materialize_pair(spec, ScriptedProxy([2.5, 2.0, 2.5]),
                                half_fov_deg=90.0)
    second = AP.materialize_pair(spec, ScriptedProxy([2.5, 2.0, 2.5]),
                                 half_fov_deg=90.0)
    assert A.actions_to_dicts(list(first.safe.actions)) == \
        A.actions_to_dicts(list(second.safe.actions))
    assert first.collision.provenance == second.collision.provenance


def test_blocked_suffix_leg_abandons_the_template():
    rejections = {}
    spec = template(length=3, starts_with="forward", turn_angles=(15.0,),
                    target_forward_leg_number=1, distance_ranks=(0,))
    assert AP.materialize_pair(
        spec, ScriptedProxy([2.0, 0.6]), half_fov_deg=90.0,
        rejections=rejections) is None
    assert rejections == {"template_infeasible_suffix": 1}


def test_provenance_records_every_derived_quantity():
    spec = template(length=5, starts_with="forward", turn_angles=(15.0, -15.0),
                    target_forward_leg_number=2, distance_ranks=(0, 3))
    pair = AP.materialize_pair(spec, ScriptedProxy([2.5, 2.0, 2.5]),
                               half_fov_deg=90.0)
    for side, variant in ((pair.safe, "safe"), (pair.collision, "collision")):
        assert side.provenance["variant"] == variant
        assert side.provenance["template_id"] == spec.template_id
        assert side.provenance["target_forward_leg_number"] == 2
        assert side.provenance["target_action_index"] == 2
        assert side.provenance["a2_capable"] is True
        assert side.provenance["b_j_m"] == pytest.approx(2.0)
        assert set(side.provenance) == set(AP.PROVENANCE_FIELDS)


# --------------------------------------------------------------------------
# Prefix-conditioned reach and the per-leg budget (proposal v2)
#
# One depth image cannot certify a 9 m programme. Each Forward gets a fair share
# of the evidence its own prefix still has, so an L6 spends its reach across
# three legs instead of letting each leg independently re-probe as if it were
# the only one.
# --------------------------------------------------------------------------

def test_leg_cap_walks_the_worked_example_down_a_five_metre_corridor():
    """reach 5.0 over three legs: 1.5 -> 1.5 -> 2.0, not 3.0 -> 3.0 -> 3.0."""
    assert AP.leg_cap_m(5.0, 3) == 1.5
    assert AP.leg_cap_m(5.0 - 1.5, 2) == 1.5
    assert AP.leg_cap_m(5.0 - 3.0, 1) == 2.0
    # One Forward divides its reach by one, so the fair share is the reach
    # itself on the grid; the publication ceiling is ``local_cap_m``'s job.
    assert AP.leg_cap_m(5.0, 1) == 5.0
    assert AP.leg_cap_m(0.3, 1) is None
    assert AP.leg_cap_m(1.0, 3) is None
    with pytest.raises(ValueError):
        AP.leg_cap_m(5.0, 0)


def test_local_cap_bounds_a_leg_by_its_own_prefix_then_the_vocabulary():
    """The hard bound, and the only one.

    ``leg_cap_m`` divides reach between legs and is a preference, so this is all
    that stands between a proposal and geometry the initial view never covered.
    Too little evidence for even the shortest publishable leg drops the
    template, not the pose.
    """
    assert AP.local_cap_m(2.6) == 2.5
    assert AP.local_cap_m(5.0) == 5.0
    assert AP.local_cap_m(99.0) == AP.GLOBAL_MAX_FORWARD_M
    assert AP.local_cap_m(0.3) is None


def test_a_long_leg_the_depth_supports_is_preferred_away_not_deleted():
    """The fair share may reorder a draw; it may not delete a template.

    ``reach / remaining_legs`` assumed every future Forward shares one heading's
    depth budget. After a Turn the body enters a new corridor whose reach is
    re-read in full, so a single leg above the average is not thereby
    unsupported -- dropping it measured the average rather than the scene.
    """
    # Nothing under the share -> the share steps aside instead of being fatal.
    assert AP.pick_distance(2.0, rank=0, cap_m=1.5, prefer_max_m=0.3) == 1.5
    # Something under it -> the draw narrows to it.
    assert AP.pick_distance(None, rank=0, cap_m=5.0, prefer_max_m=1.5) == 1.5
    # It can never widen past the hard cap.
    assert AP.pick_distance(None, rank=0, cap_m=1.5, prefer_max_m=5.0) == 1.5


@pytest.mark.parametrize("free", [None, 0.7, 2.0, 3.5])
@pytest.mark.parametrize("prefer", [0.3, 1.0, 3.0, 99.0])
@pytest.mark.parametrize("rank", [0, 1, 5])
def test_the_preference_never_admits_what_the_cap_excluded(
        free, prefer, rank):
    """Property form: preferring is a reordering inside an unchanged bound."""
    picked = AP.pick_distance(free, rank, cap_m=2.5, prefer_max_m=prefer)
    bounded = AP.pick_distance(free, rank, cap_m=2.5)
    assert (picked is None) == (bounded is None)
    if picked is not None:
        assert picked in AP.feasible_grid(free)
        assert picked <= 2.5 + 1e-9


def test_l6_materialises_its_three_legs_at_one_and_a_half_and_two():
    """The worked example end to end, through the production segment builder.

    Turns are 0 deg so the stand-in's ``reach = 5.0 - travelled`` model is
    exactly the straight 5 m corridor the example describes; what is under test
    is the cap arithmetic across legs, not the turn geometry.
    """
    proxy = ScriptedProxy([5.0, 5.0, 5.0],
                          reach=lambda prefix: 5.0 - A.total_forward_m(prefix))
    pattern = AP.action_pattern(6, "forward")
    assert sum(1 for kind in pattern if kind == "forward") == 3

    produced = AP._materialize_segment(
        pattern, [], proxy, iter([0.0, 0.0, 0.0]), iter([0, 0, 0]),
        remaining_forward_legs=3)

    legs = [action.m for action in produced if isinstance(action, Forward)]
    assert legs == [1.5, 1.5, 2.0]
    # The three legs together spend the corridor and nothing more. Without a
    # budget each leg re-probed independently and took the longest feasible
    # distance, so the same 5 m of evidence produced a far longer nominal
    # programme -- and the v2 vocabulary makes that worse, not better.
    assert sum(legs) == pytest.approx(5.0)
    assert AP.pick_distance(5.0, 0) * 3 == pytest.approx(13.5)


@pytest.mark.parametrize("free", [None, 0.7, 2.0, 3.5])
@pytest.mark.parametrize("cap", AP.GRID_M)
@pytest.mark.parametrize("rank", [0, 1, 2, 5])
def test_the_cap_can_only_tighten_the_feasible_grid(free, cap, rank):
    """``leg_cap`` is a filter over ``feasible_grid``, never a replacement.

    Written as a property because this is the invariant that makes the cap
    unable to lower a GT gate: the safe clearance stays the only source of the
    boundary bound, and the budget can subtract from that set but never add to
    it.
    """
    picked = AP.pick_distance(free, rank, cap_m=cap)
    if picked is not None:
        assert picked in AP.feasible_grid(free)
        assert picked <= cap + 1e-9
    # A cap above the published vocabulary is not a constraint at all.
    assert AP.pick_distance(free, rank, cap_m=99.0) == \
        AP.pick_distance(free, rank)


@pytest.mark.parametrize("free", [None, 0.7, 2.0, 3.5])
@pytest.mark.parametrize("cap", AP.GRID_M)
def test_the_cap_removes_outcomes_and_reranks_the_survivors(free, cap):
    """Every capped outcome was already reachable uncapped; the rank shifts.

    Ranks index the feasible set from the far end and wrap, so removing entries
    renumbers the ones that remain: at a fixed rank a capped leg can come out
    *longer* than the uncapped leg. That is a diversity dial being remapped, not
    a bound being loosened -- the set-level statement below is what actually
    matters, and filtering rather than clamping is what keeps the shortened legs
    spread over the grid instead of piling onto the cap value and deduplicating
    each other out of the bank.
    """
    ranks = range(12)
    capped = {AP.pick_distance(free, rank, cap_m=cap) for rank in ranks}
    uncapped = {AP.pick_distance(free, rank) for rank in ranks}
    assert capped - {None} <= uncapped - {None}
    assert all(value <= cap + 1e-9 for value in capped - {None})


def test_reach_is_reread_from_the_heading_after_a_turn():
    """A turn changes what the depth image supports, so the cap must change too.

    The stand-in gives 3.0 m ahead of the initial heading and 1.5 m off it; the
    leg after the turn is capped by the turned reach. A pose-level reach would
    have given that leg 1.5 m instead of 0.5 m.
    """
    def reach(prefix):
        return 3.0 if abs(A.pose_after(prefix)[2]) < 1e-9 else 1.5

    spec = template(length=5, starts_with="forward",
                    turn_angles=(30.0, -30.0), target_forward_leg_number=3,
                    distance_ranks=(0, 0))
    proxy = ScriptedProxy([3.0, 3.0, 2.0], reach=reach)
    pair = AP.materialize_pair(spec, proxy, half_fov_deg=90.0)
    assert pair is not None

    prefix_legs = [action.m for action in list(pair.safe.actions)[:4]
                   if isinstance(action, Forward)]
    # leg 1: floor_grid(3.0 / 3) = 1.0 off the initial heading.
    # leg 2: floor_grid(1.5 / 2) = 0.5 off the turned heading.
    assert prefix_legs == [1.0, 0.5]
    assert any(abs(pose[2] - 30.0) < 1e-9 for pose in proxy.reach_calls)

    # Same template, same boundaries, reach that ignores heading: the second leg
    # triples. That difference is the whole content of "re-read".
    flat = ScriptedProxy([3.0, 3.0, 2.0], reach=3.0)
    flat_pair = AP.materialize_pair(spec, flat, half_fov_deg=90.0)
    assert [action.m for action in list(flat_pair.safe.actions)[:4]
            if isinstance(action, Forward)] == [1.0, 1.5]


def test_real_frame_reach_is_measured_per_heading_and_memoised():
    frame = _wall_frame(2.5)
    proxy = AP.FrameDepthProxy(frame, radius=0.2)
    ahead = proxy.prefix_supported_reach_m([])
    assert ahead == 3.0
    # Rotating in place at the camera origin puts the body's own width outside
    # the horizontal field of view, so one depth image supports nothing there.
    # This is not new in v2: ``corridor_coverage`` already refused these paths,
    # and the cap only turns that refusal into a cheaper, earlier rejection.
    assert proxy.prefix_supported_reach_m([Turn(90.0)]) == 0.0
    assert AP.leg_cap_m(0.0, 1) is None
    # A turn taken after walking is a different matter -- the same depth still
    # supports part of that corridor, so this is per-heading, not anti-turn.
    assert proxy.prefix_supported_reach_m([Forward(1.0), Turn(90.0)]) == 0.5
    assert proxy.prefix_supported_reach_m([Forward(1.0)]) == 2.0

    # Templates share prefixes heavily and a scan costs a dozen corridor
    # queries; the cache is keyed by the pose the prefix ends at, which is the
    # only thing the answer depends on.
    assert sorted(proxy._reach_by_pose) == [
        (0.0, 0.0, 0.0), (0.0, 0.0, 90.0), (0.0, 1.0, 0.0), (0.0, 1.0, 90.0)]
    assert proxy.prefix_supported_reach_m([Forward(0.5), Forward(0.5)]) == 2.0
    assert len(proxy._reach_by_pose) == 4


def test_capping_a_leg_does_not_desynchronise_the_rank_stream():
    """The rank draw happens once per Forward whatever the cap decides.

    Skipping the draw on a capped leg would hand the next leg a rank that was
    meant for this one, so the bank would depend on which legs happened to be
    capped rather than on the template.
    """
    spec = template(length=5, starts_with="forward", turn_angles=(15.0, -15.0),
                    target_forward_leg_number=3, distance_ranks=(0, 2))
    free = [3.0, 3.0, 2.0]

    open_pair = AP.materialize_pair(
        spec, ScriptedProxy(free), half_fov_deg=90.0)
    capped_pair = AP.materialize_pair(
        spec, ScriptedProxy(free, reach=lambda prefix: (
            3.0 if not any(isinstance(a, Forward) for a in prefix) else 6.0)),
        half_fov_deg=90.0)

    open_legs = [a.m for a in list(open_pair.safe.actions)[:4]
                 if isinstance(a, Forward)]
    capped_legs = [a.m for a in list(capped_pair.safe.actions)[:4]
                   if isinstance(a, Forward)]
    assert open_legs == [2.0, 1.5]
    # Leg 1 tightened; leg 2 still consumed rank 2 and is unchanged. Had the cap
    # swallowed leg 1's draw, leg 2 would have taken rank 0 and come out at 2.5.
    assert capped_legs == [1.0, 1.5]


def test_the_colliding_leg_needs_an_observed_boundary_not_a_reach_allowance():
    """The one thing the colliding leg must satisfy, and the one it cannot.

    It exists to cross the boundary reach itself stops at, so holding it to that
    reach could never be satisfied -- the previous rule bought its way out with
    a one-grid-step allowance, which measured the allowance rather than the
    scene. ``STATE_COLLISION`` is the real requirement: the corridor up to
    contact plus tolerance is covered, and the executed part of the leg
    terminates there. An uncovered "boundary" is a hole in the depth image.
    """
    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    # Reach 2.0 is a full grid step short of the 3.0 m colliding leg, which the
    # old rule refused outright. The boundary at 2.2 m was observed, so it is
    # admitted -- and the safe branch is still bounded by that same reach.
    observed = ScriptedProxy([2.2], reach=2.0)
    pair = AP.materialize_pair(spec, observed, half_fov_deg=90.0)
    assert pair is not None
    assert list(pair.collision.actions)[0].m == 3.0
    assert list(pair.safe.actions)[0].m <= 2.0
    assert observed.free_distance_state([])["state"] == \
        reach_probe.STATE_COLLISION

    for state in (reach_probe.STATE_INSUFFICIENT, reach_probe.STATE_OPEN):
        rejections = {}
        assert AP.materialize_pair(
            spec, ScriptedProxy([2.2], reach=2.5, state=state),
            half_fov_deg=90.0, rejections=rejections) is None
        assert rejections == {"collision_boundary_unobserved": 1}


def test_a_template_its_prefix_cannot_support_costs_the_template_not_the_pose():
    over_reach = template(template_id="T-000", length=1,
                          starts_with="forward", turn_angles=(),
                          target_forward_leg_number=1, distance_ranks=())
    within_reach = template(template_id="T-001", length=1,
                            starts_with="forward", turn_angles=(),
                            target_forward_leg_number=1, distance_ranks=())

    class _PerTemplateReach(ScriptedProxy):
        """T-000 is materialised first, and only it sees the short reach."""

        def __init__(self):
            super().__init__([2.2])
            self._attempt = -1

        def prefix_supported_reach_m(self, prefix):
            super().prefix_supported_reach_m(prefix)
            self._attempt += 1
            return 0.3 if self._attempt == 0 else 2.5

    bank = AP.build_pose_bank(
        [over_reach, within_reach], _PerTemplateReach(),
        half_fov_deg=90.0, pairs_per_length=2)
    tags = {candidate.provenance["template_id"] for candidate in bank[1]}
    assert tags == {"T-001"}


def test_old_provenance_stays_frozen_while_the_collector_moves_to_v5():
    """Frozen contracts have no adapter: old records keep their field sets.

    The field tuple is pinned literally here because the danger is silent -- an
    edit that grew v2's fields by mutating the shared tuple would rewrite the
    contract that already-frozen Golden records were written under.
    """
    assert AP.PROPOSAL_PROTOCOL_V1 == "depth-conditioned-pair-v1"
    assert AP.PROPOSAL_PROTOCOL_V2 == "depth-conditioned-pair-v2"
    assert AP.PROPOSAL_PROTOCOL_V3 == "depth-conditioned-action-bank-v3"
    assert AP.PROPOSAL_PROTOCOL_V4 == "depth-conditioned-action-bank-v4"
    assert AP.PROPOSAL_PROTOCOL_V5 == "depth-conditioned-action-bank-v5"
    assert AP.PROPOSAL_PROTOCOL_VERSION == AP.PROPOSAL_PROTOCOL_V5
    assert set(AP.PROVENANCE_FIELDS_BY_PROTOCOL) == {
        AP.PROPOSAL_PROTOCOL_V1, AP.PROPOSAL_PROTOCOL_V2,
        AP.PROPOSAL_PROTOCOL_V3, AP.PROPOSAL_PROTOCOL_V4,
        AP.PROPOSAL_PROTOCOL_V5}
    assert AP.PROVENANCE_FIELDS_BY_PROTOCOL[AP.PROPOSAL_PROTOCOL_V1] == (
        "protocol", "template_id", "target_forward_leg_number",
        "target_action_index", "a2_capable", "variant", "b_j_m",
        "suffix_forward_m", "local_guard_m", "remaining_proxy_m",
        "safe_proxy_gap_m", "proxy_coverage",
    )
    assert AP.PROVENANCE_FIELDS_BY_PROTOCOL[AP.PROPOSAL_PROTOCOL_V2] == \
        AP.PROVENANCE_FIELDS_BY_PROTOCOL[AP.PROPOSAL_PROTOCOL_V1]
    assert AP.PROVENANCE_FIELDS_BY_PROTOCOL[AP.PROPOSAL_PROTOCOL_V3] == \
        AP.PROVENANCE_FIELDS_BY_PROTOCOL[AP.PROPOSAL_PROTOCOL_V1]
    assert AP.PROVENANCE_FIELDS == \
        AP.PROVENANCE_FIELDS_BY_PROTOCOL[AP.PROPOSAL_PROTOCOL_V5]

    # Reach and cap are sampling hints over a depth array no record stores, so
    # no validator could recompute them; v2 therefore publishes the same
    # authoritative fields and binds the new rule through the sampler digest.
    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    pair = AP.materialize_pair(spec, ScriptedProxy([1.5]), half_fov_deg=90.0)
    assert pair.safe.provenance["protocol"] == AP.PROPOSAL_PROTOCOL_V5
    assert set(pair.safe.provenance) == set(
        AP.PROVENANCE_FIELDS_BY_PROTOCOL[AP.PROPOSAL_PROTOCOL_V1])


# --------------------------------------------------------------------------
# The control arm's own prefix budget
# --------------------------------------------------------------------------

def test_a_control_is_bounded_per_prefix_not_by_its_path_average():
    """Whole-path coverage is a mean, and a mean hides its worst leg.

    A control whose early legs are fully observed and whose last one walks into
    unobserved space can still average above the coverage floor. That is where
    the two arms disagreed: the paired arm cannot *propose* such a leg at all,
    because every leg it builds is bounded by ``local_cap_m`` of that leg's own
    prefix. The control arm now screens on the same bound.
    """
    actions = [Forward(1.0), Turn(30.0), Forward(3.0)]
    rejections = {}
    assert AP.proxy_natural_coverage(
        actions, ScriptedProxy([], reach=2.0), half_fov_deg=90.0,
        rejections=rejections) is None
    assert rejections == {"proxy_natural_over_prefix_reach": 1}
    # Same program, same coverage: only the depth behind the last leg differs.
    assert AP.proxy_natural_coverage(
        actions, ScriptedProxy([], reach=3.0), half_fov_deg=90.0) == 1.0


def test_the_control_budget_is_reread_from_the_heading_after_a_turn():
    """A control that turns away from a short corridor is measured in the new
    one, exactly as a materialised leg is -- no pose-level number could be
    right for both headings."""
    actions = [Forward(1.0), Turn(-45.0), Forward(3.0)]

    def reach(prefix):
        return 1.5 if not any(isinstance(a, Turn) for a in prefix) else 4.0

    assert AP.proxy_natural_coverage(
        actions, ScriptedProxy([], reach=reach), half_fov_deg=90.0) == 1.0


def test_a_control_is_not_budgeted_past_the_contact_it_stops_at():
    """The label-blind half of the rule, and the reason it is not simply
    "every leg within reach".

    Reach stops where the depth stops seeing floor, which is close to where the
    obstacle is, so budgeting the leg that ends in contact -- or the legs after
    it, which never run at all -- would reject controls for colliding and hand
    the screen the label it exists not to know. The executed part is governed by
    the contact-arc coverage rule instead, the same evidence the colliding half
    of a pair is admitted on.
    """
    actions = [Forward(1.0), Turn(15.0), Forward(3.0)]
    # Second leg contacts 0.5 m in, so arc 1.5 of a nominal 4.0 is executed.
    assert AP.proxy_natural_coverage(
        actions, ScriptedProxy([None, 0.5], reach=1.5),
        half_fov_deg=90.0) == 1.0
    # Identical program and identical reach, nothing to hit: now the same leg is
    # fully executed, and the budget applies to it.
    rejections = {}
    assert AP.proxy_natural_coverage(
        actions, ScriptedProxy([], reach=1.5), half_fov_deg=90.0,
        rejections=rejections) is None
    assert rejections == {"proxy_natural_over_prefix_reach": 1}


def test_a_prefix_supporting_no_publishable_leg_screens_the_control_out():
    """``local_cap_m`` returning ``None`` means "this prefix's depth supports
    nothing publishable", not "unbounded" -- the same reading the paired arm
    takes when it drops the template."""
    rejections = {}
    assert AP.proxy_natural_coverage(
        [Forward(0.5)], ScriptedProxy([], reach=0.3), half_fov_deg=90.0,
        rejections=rejections) is None
    assert rejections == {"proxy_natural_over_prefix_reach": 1}


# --------------------------------------------------------------------------
# Per-pose bank: budget, dedup, ordering
# --------------------------------------------------------------------------

def _bank(templates, free=(2.5, 2.0, 2.5), **kwargs):
    options = {"half_fov_deg": 90.0, "pairs_per_length": 4}
    options.update(kwargs)
    return AP.build_pose_bank(
        templates, ScriptedProxy(list(free)), **options)


def test_bank_keys_candidates_by_length_and_tags_each_variant():
    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    bank = _bank([spec], free=(1.5,))
    assert set(bank) == {1}
    variants = [candidate.variant for candidate in bank[1]]
    assert sorted(variants) == ["collision", "safe"]
    assert len({candidate.tag for candidate in bank[1]}) == 2


def test_tags_are_opaque_and_content_addressed():
    """The tag becomes ``action_group_id``, then ``outcome_id``, then the QA's
    ``oracle_ref`` -- so anything readable in it is published. ``variant`` and
    ``template_id`` stay in the private provenance only."""
    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    bank = _bank([spec], free=(1.5,))
    for candidate in bank[1]:
        assert "safe" not in candidate.tag
        assert "col" not in candidate.tag
        assert "nat" not in candidate.tag
        assert spec.template_id not in candidate.tag
        assert candidate.tag == AP.candidate_tag(candidate.actions)
    # Content addressing: the same program always earns the same tag.
    assert AP.candidate_tag((Forward(1.0),)) == AP.candidate_tag((Forward(1.0),))
    assert AP.candidate_tag((Forward(1.0),)) != AP.candidate_tag((Forward(1.5),))


def test_duplicate_on_either_side_drops_the_whole_pair():
    """A pair is the atomic unit: an orphan half is not a controlled contrast."""
    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    twin = template(template_id="T-001", length=1, starts_with="forward",
                    turn_angles=(), target_forward_leg_number=1,
                    distance_ranks=())
    rejections = {}
    bank = _bank([spec, twin], free=(1.5,), rejections=rejections)
    assert len(bank[1]) == 2
    assert {c.provenance["template_id"] for c in bank[1]} == {"T-000"}
    assert rejections == {"pair_duplicate": 1}


def test_pair_budget_is_counted_in_pairs_not_actions():
    templates = [
        template(template_id=f"T-{index:03d}", length=1,
                 starts_with="forward", turn_angles=(),
                 target_forward_leg_number=1, distance_ranks=())
        for index in range(5)
    ]
    # Every template of length 1 materialises the same program, so dedup leaves
    # one pair regardless of budget; give each a distinct boundary instead.
    bank = AP.build_pose_bank(
        templates, _MultiBoundaryProxy([1.5, 2.0, 2.5, 3.0, 1.0]),
        half_fov_deg=90.0, pairs_per_length=2)
    assert len(bank[1]) == 4
    assert sum(1 for c in bank[1] if c.variant == "safe") == 2


class _MultiBoundaryProxy(ScriptedProxy):
    """One boundary per materialisation attempt, for budget tests."""

    def __init__(self, boundaries):
        super().__init__([])
        self._boundaries = list(boundaries)
        self._attempt = -1

    def free_distance_m(self, base_pose):
        self._attempt += 1
        self._free = [self._boundaries[
            self._attempt % len(self._boundaries)]]
        return self._free[0]


def test_bank_ordering_is_canonical_and_hash_covers_the_whole_bank():
    templates = [
        template(template_id="T-002", length=1, starts_with="forward",
                 turn_angles=(), target_forward_leg_number=1,
                 distance_ranks=()),
        template(template_id="T-001", length=1, starts_with="forward",
                 turn_angles=(), target_forward_leg_number=1,
                 distance_ranks=()),
    ]
    bank = AP.build_pose_bank(
        templates, _MultiBoundaryProxy([1.5, 2.0]),
        half_fov_deg=90.0, pairs_per_length=2)
    keys = [(c.length, c.provenance["template_id"], c.variant)
            for c in bank[1]]
    assert keys == sorted(keys)
    # The digest authenticates the full paired pre-oracle bank.
    digest = AP.materialized_action_bank_sha256(bank)
    assert digest == AP.materialized_action_bank_sha256(bank)
    trimmed = {1: bank[1][:-1]}
    assert AP.materialized_action_bank_sha256(trimmed) != digest


# --------------------------------------------------------------------------
# Run-level template bank and sampler contract
# --------------------------------------------------------------------------

def _templates(seed=7, lengths=(1, 2, 3, 4, 5, 6), per_length=12):
    import numpy as np
    return AP.build_template_bank(
        np.random.default_rng(seed), lengths=lengths, per_length=per_length)


def test_every_template_contains_a_forward_leg():
    for spec in _templates():
        assert spec.forward_count >= 1


def test_length_one_has_exactly_one_possible_structure():
    """[Forward] is the only alternating length-1 program with a Forward."""
    assert len([s for s in _templates() if s.length == 1]) == 1


def test_target_leg_is_spread_across_the_available_forward_legs():
    multi = [s for s in _templates() if s.forward_count >= 3]
    assert len({s.target_forward_leg_number for s in multi}) >= 3


def test_both_alternation_forms_appear_with_balanced_quota():
    """Filling one start form before the other biases primitive positions.

    Every length above 1 has a legal turn-start program; a bank that drew all
    its forward-start variants first and truncated would silently drop them.
    """
    bank = _templates()
    for length in (2, 3, 4, 5, 6):
        starts = [s.starts_with for s in bank if s.length == length]
        assert set(starts) == {"forward", "turn"}
        assert abs(starts.count("forward") - starts.count("turn")) <= 1


def test_target_leg_quota_is_balanced_within_each_structure():
    bank = _templates()
    grouped = {}
    for spec in bank:
        grouped.setdefault((spec.length, spec.starts_with), []).append(
            spec.target_forward_leg_number)
    for (length, start), targets in grouped.items():
        counts = [targets.count(leg)
                  for leg in range(1, AP._forward_count(length, start) + 1)]
        assert max(counts) - min(counts) <= 1, (length, start, counts)


def test_every_length_fills_its_budget_except_structurally_capped_ones():
    counts = {}
    for spec in _templates(per_length=12):
        counts[spec.length] = counts.get(spec.length, 0) + 1
    # Length 1 has one legal program shape, [Forward], and no turn or rank slot
    # to vary, so its structural capacity is a single template.
    assert counts[1] == 1
    # L2 stops at five Forward-first and four shared-FOV Turn-first templates;
    # taking the sixth Forward-first template would break the start balance.
    assert counts[2] == 9
    assert all(counts[length] == 12 for length in (3, 4, 5, 6))


def test_template_bank_is_deterministic_and_bounded():
    assert _templates() == _templates()
    counts = {}
    for spec in _templates(per_length=5):
        counts[spec.length] = counts.get(spec.length, 0) + 1
    assert all(count <= 5 for count in counts.values())


def test_templates_carry_no_distance():
    for spec in _templates():
        assert not hasattr(spec, "forward_distances")
        assert all(isinstance(angle, float) for angle in spec.turn_angles)


_NATURAL = [("L1-c000", [Forward(0.5)]), ("L2-c000", [Turn(15.0), Forward(1.0)])]


def _contract(templates=None, natural=None, **kwargs):
    options = {
        "pairs_per_length": 4,
        "natural_per_length": 4,
        "ordinary_actions_per_pose": 36,
    }
    options.update(kwargs)
    return AP.action_sampler_contract_sha256(
        _templates() if templates is None else templates,
        _NATURAL if natural is None else natural, **options)


def test_sampler_contract_binds_the_depth_oracle_parameters(monkeypatch):
    base = _contract()
    assert base == _contract()
    assert base != _contract(pairs_per_length=3)
    # An oracle whose march step or support threshold moved would sample a
    # different distribution from an unchanged rule, so the rule digest alone
    # cannot be the contract.
    monkeypatch.setattr(config, "MIN_SUPPORT_VOXELS",
                        config.MIN_SUPPORT_VOXELS + 1)
    assert base != _contract()


def test_sampler_contract_binds_the_natural_control_bank():
    """Controls are half the per-pose bank; a contract blind to them would let
    the control vocabulary change without any digest moving."""
    assert _contract() != _contract(
        natural=_NATURAL + [("L2-c001", [Turn(30.0), Forward(1.0)])])
    assert _contract() != _contract(natural=[])


def test_v3_sampler_contract_binds_the_committed_control_catalog(monkeypatch):
    base = _contract(natural=[])

    monkeypatch.setattr(
        AP.action_control_catalog, "CATALOG_SHA256", "0" * 64)

    assert base != _contract(natural=[])


@pytest.mark.parametrize("name", [
    "C1_QUERIES_PER_POSE",
    "C1_NEIGHBORS_PER_QUERY",
])
def test_v4_sampler_contract_binds_each_pose_allocation_limit(
        monkeypatch, name):
    base = _contract(natural=[])
    monkeypatch.setattr(config, name, getattr(config, name) + 1)
    assert base != _contract(natural=[])


def test_v4_sampler_contract_binds_selected_ordinary_action_budget():
    assert _contract(natural=[], ordinary_actions_per_pose=36) != \
        _contract(natural=[], ordinary_actions_per_pose=24)


def test_v4_sampler_contract_binds_natural_distance_strata(monkeypatch):
    base = _contract(natural=[])
    monkeypatch.setattr(
        config, "NATURAL_DYNAMIC_STRATUM_WEIGHTS",
        {"short": 1, "mid": 1, "near": 1})
    assert base != _contract(natural=[])


# --------------------------------------------------------------------------
# The real frame-backed proxy
# --------------------------------------------------------------------------

def _wall_frame(distance_m):
    import numpy as np
    from tests import _synthetic

    frame = _synthetic.make_frame()
    xs = np.linspace(-2.0, 2.0, 400)
    wall = np.stack(
        [xs, np.full_like(xs, 0.15), np.full_like(xs, distance_m)], axis=1)
    from pipeline import perception
    object.__setattr__(frame, "vf", perception.VoxelField(wall))
    return frame


def test_frame_proxy_measures_the_swept_disc_boundary():
    frame = _wall_frame(2.0)
    proxy = AP.FrameDepthProxy(frame, radius=0.2)
    free = proxy.free_distance_m((0.0, 0.0, 0.0))
    assert free is not None
    assert free == pytest.approx(2.0 - 0.2, abs=0.15)


def test_frame_proxy_reports_none_beyond_its_reach():
    frame = _wall_frame(8.0)
    proxy = AP.FrameDepthProxy(frame, radius=0.2)
    assert proxy.free_distance_m((0.0, 0.0, 0.0)) is None


def test_frame_proxy_advances_with_the_prefix_pose():
    """Probing after a prefix must measure from where the prefix ended."""
    frame = _wall_frame(2.5)
    proxy = AP.FrameDepthProxy(frame, radius=0.2)
    at_origin = proxy.free_distance_m((0.0, 0.0, 0.0))
    after_one_metre = proxy.free_distance_m((0.0, 1.0, 0.0))
    assert at_origin - after_one_metre == pytest.approx(1.0, abs=0.05)


def test_frame_proxy_memoizes_free_distance_by_prefix_pose(monkeypatch):
    frame = _wall_frame(2.5)
    real = AP.rollout.view_collision_rollout
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("base_pose"))
        return real(*args, **kwargs)

    monkeypatch.setattr(AP.rollout, "view_collision_rollout", counted)
    proxy = AP.FrameDepthProxy(frame, radius=0.2)

    first = proxy.free_distance_m((0.0, 0.0, 0.0))
    assert proxy.free_distance_m((0.0, 0.0, 0.0)) == first
    proxy.free_distance_m((0.0, 1.0, 0.0))

    assert calls == [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


def test_pair_checks_prefix_collision_state_before_materializing_suffix():
    class OrderingProxy(ScriptedProxy):
        state_checked = False

        def free_distance_state(self, prefix):
            self.state_checked = True
            return super().free_distance_state(prefix)

        def prefix_supported_reach_m(self, prefix):
            if any(isinstance(value, Turn) for value in prefix):
                assert self.state_checked
            return super().prefix_supported_reach_m(prefix)

    spec = template(
        length=3, starts_with="forward", turn_angles=(15.0,),
        target_forward_leg_number=1, distance_ranks=(0,))

    assert AP.materialize_pair(
        spec, OrderingProxy([1.5, 1.0]), half_fov_deg=90.0) is not None


def test_initial_fov_is_a_centerline_predicate_not_a_body_radius_gate():
    actions = [Turn(20.0), Forward(4.0)]

    assert A.inside_initial_fov(actions, 30.0)
    with pytest.raises(TypeError, match="radius_m"):
        A.inside_initial_fov(actions, 30.0, radius_m=1.5)


def test_turn_first_templates_never_start_with_45_degrees():
    turn_first = [
        spec for spec in _templates(seed=11, per_length=40)
        if spec.starts_with == "turn"
    ]

    assert turn_first
    assert {spec.turn_angles[0] for spec in turn_first} <= {
        -30.0, -15.0, 15.0, 30.0,
    }


def test_boundary_beyond_the_grid_reach_cannot_form_a_collision_variant():
    """The grid has a top, so a boundary near it has no room for the guard.

    Pairs only exist where an obstacle is within roughly
    ``max(GRID_M) - publication guard`` of the leg's start. v1 put that at 2.2 m
    and it bound most of the scene; at v2 it is 5.2 m and only a boundary that
    far out is refused for want of grid.
    """
    rejections = {}
    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    assert AP.materialize_pair(
        spec, ScriptedProxy([5.3]), half_fov_deg=90.0,
        rejections=rejections) is None
    assert rejections == {"bracket_grid_exhausted_collision": 1}
    assert AP.materialize_pair(
        spec, ScriptedProxy([5.2]), half_fov_deg=90.0) is not None
    # The v1 ceiling is no longer a ceiling: this pair did not exist before.
    assert AP.materialize_pair(
        spec, ScriptedProxy([2.3]), half_fov_deg=90.0) is not None


def test_a_suffix_buys_two_more_grid_decimetres_of_reach():
    """With a suffix the publication term is partly paid, so the guard drops to
    the A2 floor and the reachable boundary moves by 0.2 m -- 5.2 m to 5.4 m."""
    spec = template(length=3, starts_with="forward", turn_angles=(15.0,),
                    target_forward_leg_number=1, distance_ranks=(0,))
    pair = AP.materialize_pair(
        spec, ScriptedProxy([5.4, 0.9]), half_fov_deg=90.0)
    assert pair is not None
    assert pair.collision.provenance["local_guard_m"] == pytest.approx(
        config.A2_ACTION_BOUNDARY_MARGIN_M + config.ORACLE_CONTACT_TOL_M)
    assert pair.collision.provenance["suffix_forward_m"] == pytest.approx(0.5)

    rejections = {}
    assert AP.materialize_pair(
        spec, ScriptedProxy([5.5, 0.9]), half_fov_deg=90.0,
        rejections=rejections) is None
    assert rejections == {"bracket_grid_exhausted_collision": 1}


def test_real_frame_pair_collides_inside_its_target_leg():
    frame = _wall_frame(1.8)
    proxy = AP.FrameDepthProxy(frame, radius=0.2)
    spec = template(length=1, starts_with="forward", turn_angles=(),
                    target_forward_leg_number=1, distance_ranks=())
    pair = AP.materialize_pair(spec, proxy, half_fov_deg=39.5)
    assert pair is not None
    safe = proxy.rollout(list(pair.safe.actions))
    collision = proxy.rollout(list(pair.collision.actions))
    assert safe["collision"] is False
    assert collision["collision"] is True
    assert collision["contact_action_index"] == spec.target_action_index
    overshoot = (A.total_forward_m(list(pair.collision.actions)) -
                 collision["first_contact_arc_m"])
    assert overshoot >= config.BENCH_COLLISION_REMAINING_M - 1e-9
