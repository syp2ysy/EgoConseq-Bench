"""Pure-geometry action tests (no Habitat)."""

import collections
import importlib
import math

import numpy as np
import pytest

from pipeline import actions as action_geometry
from pipeline import config
from pipeline.actions import (
    Turn, Forward, parse_actions, actions_to_dicts,
    wrap_deg, net_turn_deg, total_forward_m,
    pose_after, pose_at_arc, pose_at_progress, arc_at_progress,
    program_progress_at_contact,
    balanced_action_pool, sample_path,
    contact_action_index,
    validate_alternating_actions, validate_physics_actions,
)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_contact_action_index_multi_forward():
    acts = [Turn(15), Forward(1.5), Forward(1.5)]     # arc spans: F0=[0,1.5], F1=[1.5,3]
    idx, local = contact_action_index(acts, 1.78)
    assert idx == 2 and _close(local, 0.28, 1e-6)     # forward at index 2, 0.28 m in


def test_contact_action_index_first_leg():
    idx, local = contact_action_index([Forward(1.0)], 0.6)
    assert idx == 0 and _close(local, 0.6)


def test_contact_action_index_after_turns():
    acts = [Turn(30), Forward(0.5), Turn(-30), Forward(1.0)]   # F0=[0,0.5], F1=[0.5,1.5]
    idx, local = contact_action_index(acts, 1.2)
    assert idx == 3 and _close(local, 0.7)             # turns consume no arc


def test_contact_action_index_beyond_total_is_none():
    idx, local = contact_action_index([Forward(1.0)], 5.0)
    assert idx is None and local is None


def test_forward_leg_location_reports_the_metric_stage():
    actions = [
        Forward(1.0), Turn(90), Forward(2.0),
        Turn(-45), Forward(1.5),
    ]

    location = action_geometry.forward_leg_location(actions, 1.7)

    assert location.action_index == 2
    assert location.forward_leg_number == 2
    assert location.cumulative_before_leg_m == pytest.approx(1.0)
    assert location.distance_into_leg_m == pytest.approx(0.7)
    assert location.leg_length_m == pytest.approx(2.0)


def test_forward_leg_location_assigns_an_exact_boundary_to_the_closing_leg():
    actions = [Forward(1.0), Turn(90), Forward(2.0)]

    location = action_geometry.forward_leg_location(actions, 1.0)

    assert location.action_index == 0
    assert location.forward_leg_number == 1
    assert location.distance_into_leg_m == pytest.approx(1.0)


@pytest.mark.parametrize("arc_m", [-0.01, 3.01, float("nan")])
def test_forward_leg_location_rejects_invalid_arcs(arc_m):
    with pytest.raises(ValueError, match="forward arc"):
        action_geometry.forward_leg_location(
            [Forward(1.0), Turn(90), Forward(2.0)], arc_m)


# --- pose_after known answers -------------------------------------------

def test_pose_after_forward():
    x, z, hd = pose_after([Forward(1.0)])
    assert _close(x, 0.0) and _close(z, 1.0) and _close(hd, 0.0)


def test_pose_after_right_then_forward():
    x, z, hd = pose_after([Turn(90), Forward(1.0)])
    assert _close(x, 1.0, 1e-9) and _close(z, 0.0, 1e-9) and _close(hd, 90.0)


def test_pose_after_square_back_to_axis():
    x, z, hd = pose_after([Turn(90), Forward(1.0), Turn(-90), Forward(1.0)])
    assert _close(x, 1.0, 1e-9) and _close(z, 1.0, 1e-9) and _close(hd, 0.0)


def test_pose_after_left_turn():
    x, z, hd = pose_after([Turn(-90), Forward(2.0)])
    assert _close(x, -2.0, 1e-9) and _close(z, 0.0, 1e-9) and _close(hd, -90.0)


# --- scalar summaries ---------------------------------------------------

def test_net_turn_and_total_forward():
    acts = [Turn(-30), Forward(1.2), Turn(15), Forward(0.5)]
    assert _close(net_turn_deg(acts), -15.0)
    assert _close(total_forward_m(acts), 1.7)


@pytest.mark.parametrize("d,expect", [
    (0, 0), (30, 30), (-30, -30), (180, 180), (-180, 180),
    (190, -170), (360, 0), (540, 180), (-540, 180),
])
def test_wrap_deg(d, expect):
    assert _close(wrap_deg(d), expect)


def test_planar_distance_is_bit_exact_across_interpreters():
    """The B1K route hashes this number in one interpreter and re-derives it in
    another, so "close enough" is not a passing grade.

    ``math.hypot`` is CPython's own algorithm and it changed in 3.10: the
    endpoint below returned 1.3989663259659064 under the supervisor's 3.9 and
    ...66 under the collector's 3.11, which is what failed a C1 shard whose
    data was entirely sound. Only correctly-rounded IEEE-754 primitives are
    portable, so the value is pinned rather than compared with a tolerance.
    """
    x_m, z_m = -0.35355339059327373, 1.3535533905932737
    assert action_geometry.planar_distance_m(x_m, z_m) == 1.3989663259659066
    assert action_geometry.planar_distance_m(x_m, z_m) == \
        math.sqrt(x_m * x_m + z_m * z_m)
    assert action_geometry.planar_distance_m(0.0, 0.0) == 0.0
    assert action_geometry.planar_distance_m(-3.0, 4.0) == 5.0


def test_hashed_geometry_summaries_do_not_use_hypot():
    """A grep the fix cannot silently regress.

    The two summaries below are content-hashed and re-derived by a different
    interpreter than the one that wrote them; anyone reaching for the obvious
    ``math.hypot`` in either module reintroduces the same shard failure.
    """
    import inspect
    from pipeline import future_view_selection
    assert "math.hypot" not in inspect.getsource(future_view_selection)


# --- pose_at_arc --------------------------------------------------------

def test_pose_at_arc_mid_leg():
    x, z, hd = pose_at_arc([Forward(1.0)], 0.5)
    assert _close(x, 0.0) and _close(z, 0.5) and _close(hd, 0.0)


def test_pose_at_arc_after_turn():
    x, z, hd = pose_at_arc([Turn(90), Forward(2.0)], 0.5)
    assert _close(x, 0.5, 1e-9) and _close(z, 0.0, 1e-9) and _close(hd, 90.0)


def test_pose_at_arc_boundary_does_not_apply_later_turn():
    # Stops exactly at end of first forward -> the R(90) is never applied.
    x, z, hd = pose_at_arc([Forward(1.0), Turn(90), Forward(1.0)], 1.0)
    assert _close(x, 0.0) and _close(z, 1.0) and _close(hd, 0.0)


def test_pose_at_arc_beyond_total_equals_pose_after():
    acts = [Turn(30), Forward(1.0), Turn(-15), Forward(0.5)]
    assert pose_at_arc(acts, 99.0) == pose_after(acts)


def test_pose_at_program_progress_uses_physical_elapsed_time():
    acts = [Turn(90), Forward(2.0)]
    assert pose_at_progress(acts, 0.0) == pytest.approx((0.0, 0.0, 0.0))
    assert pose_at_progress(acts, 0.25) == pytest.approx((0.0, 0.0, 67.5))
    assert pose_at_progress(acts, 0.50) == pytest.approx((0.5, 0.0, 90.0))
    assert pose_at_progress(acts, 0.75) == pytest.approx((1.25, 0.0, 90.0))
    assert pose_at_progress(acts, 1.0) == pytest.approx((2.0, 0.0, 90.0))


def test_arc_at_program_progress_uses_the_same_physical_clock():
    acts = [Turn(90), Forward(2.0), Turn(-90), Forward(1.0)]
    assert arc_at_progress(acts, 0.20) == pytest.approx(0.0)
    assert arc_at_progress(acts, 0.40) == pytest.approx(1.0)
    assert arc_at_progress(acts, 0.60) == pytest.approx(2.0)
    assert arc_at_progress(acts, 0.80) == pytest.approx(2.0)
    assert arc_at_progress(acts, 0.90) == pytest.approx(2.5)


def test_contact_program_progress_maps_into_forward_primitive():
    acts = [Turn(30), Forward(2.0), Turn(-15), Forward(1.0)]
    assert program_progress_at_contact(acts, action_index=1, local_arc_m=0.5) \
        == pytest.approx(
            ((30.0 / config.ANGULAR_SPEED_DEG_S) +
             (0.5 / config.LINEAR_SPEED_M_S)) /
            ((45.0 / config.ANGULAR_SPEED_DEG_S) +
             (3.0 / config.LINEAR_SPEED_M_S)))


def test_balanced_action_pool_is_deterministic_and_strictly_alternates():
    a = balanced_action_pool(
        np.random.default_rng(7), lengths=(4,), pool_per_length=200,
        half_fov_deg=39.5,
    )[4]
    b = balanced_action_pool(
        np.random.default_rng(7), lengths=(4,), pool_per_length=200,
        half_fov_deg=39.5,
    )[4]
    assert actions_to_dicts(a[0]) == actions_to_dicts(b[0])
    keys = {tuple((type(x).__name__, x.deg if isinstance(x, Turn) else x.m)
                  for x in seq) for seq in a}
    assert len(keys) == len(a)
    assert all(isinstance(x, Turn) != isinstance(y, Turn)
               for seq in a for x, y in zip(seq, seq[1:]))
    assert all(abs(math.degrees(math.atan2(x, z))) <= 39.5 + 1e-6
               for seq in a for x, z, _heading, arc in sample_path(seq, 0.02)
               if arc > 0)


def test_balanced_action_pool_contains_both_start_types_without_equal_neighbors():
    pool = balanced_action_pool(
        np.random.default_rng(11), lengths=(3,), pool_per_length=60,
        half_fov_deg=89.0,
    )[3]
    assert any(isinstance(seq[0], Turn) for seq in pool)
    assert any(isinstance(seq[0], Forward) for seq in pool)
    assert all(isinstance(x, Turn) != isinstance(y, Turn)
               for seq in pool for x, y in zip(seq, seq[1:]))
    assert all(any(isinstance(action, Forward) for action in seq) for seq in pool)


def test_main_length_one_pool_contains_forward_actions_only():
    pool = balanced_action_pool(
        np.random.default_rng(13), lengths=(1,), pool_per_length=10,
        half_fov_deg=39.5)[1]
    assert pool
    assert all(len(sequence) == 1 and isinstance(sequence[0], Forward)
               for sequence in pool)


def test_forward_legs_are_half_metre_multiples():
    # Every forward distance is a clean 0.5 m multiple: no arbitrary or
    # near-boundary metric values, so a question never hinges on sub-0.5 m
    # precision the model (or a human) could not resolve from one image.
    for value in config.GEN_FORWARDS_M:
        assert value >= 0.5 - 1e-9
        assert abs(value / 0.5 - round(value / 0.5)) < 1e-9


def test_main_action_library_and_formal_lengths():
    assert set(config.GEN_TURNS_DEG) == {-45, -30, -15, 15, 30, 45}
    assert set(config.GEN_FORWARDS_M) == {
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0}
    assert set(config.GEN_LENGTHS) == {1, 2, 3, 4, 5, 6}
    assert config.KEEP_PER_LENGTH == 1


@pytest.mark.parametrize("actions", [
    [Forward(0.5), Forward(1.0)],
    [Turn(-15), Turn(15)],
])
def test_action_validation_rejects_equal_adjacent_types(actions):
    with pytest.raises(ValueError, match="actions 1 and 2"):
        validate_alternating_actions(actions)


def test_action_validation_accepts_either_start_type_and_singletons():
    for actions in ([Turn(-15)], [Forward(0.5)],
                    [Turn(-15), Forward(0.5)],
                    [Forward(0.5), Turn(15)]):
        validate_alternating_actions(actions)


def test_physics_action_validation_accepts_main_vocabulary():
    validate_physics_actions([
        Turn(-45), Forward(0.5), Turn(30), Forward(3.0),
    ])
    validate_physics_actions([Forward(1.5)])


@pytest.mark.parametrize("actions, message", [
    ([Turn(15)], "forward"),
    ([Forward(-0.5)], "forward"),
    ([Forward(float("nan"))], "forward"),
    ([Forward(0.75)], "forward"),
    ([Turn(float("inf")), Forward(0.5)], "turn"),
    ([Turn(60), Forward(0.5)], "turn"),
])
def test_physics_action_validation_rejects_invalid_programs(actions, message):
    with pytest.raises(ValueError, match=message):
        validate_physics_actions(actions)


# --- sample_path --------------------------------------------------------

def test_sample_path_origin_and_exact_endpoint():
    s = sample_path([Forward(1.0)], step=0.02)
    assert s[0] == (0.0, 0.0, 0.0, 0.0)
    xe, ze, he, arce = s[-1]
    assert _close(xe, 0.0) and _close(ze, 1.0, 1e-9) and _close(arce, 1.0, 1e-9)
    # monotone arc
    arcs = [a for *_, a in s]
    assert all(arcs[i] < arcs[i + 1] + 1e-12 for i in range(len(arcs) - 1))


def test_sample_path_noninteger_leg_hits_exact_endpoint():
    s = sample_path([Forward(0.03)], step=0.02)
    # samples: origin, 0.02, 0.03(exact)
    assert _close(s[-1][3], 0.03)
    assert _close(s[-1][1], 0.03, 1e-9)


def test_sample_path_turn_adds_no_sample():
    only_turn = sample_path([Turn(45), Turn(-10)], step=0.02)
    assert only_turn == [(0.0, 0.0, 0.0, 0.0)]


def test_sample_path_multi_leg_arc_accumulates():
    s = sample_path([Turn(90), Forward(0.5), Turn(-90), Forward(0.5)], step=0.02)
    assert _close(s[-1][3], 1.0, 1e-9)   # total arc
    assert _close(s[-1][0], 0.5, 1e-9) and _close(s[-1][1], 0.5, 1e-9)


def test_parse_and_roundtrip():
    raw = [{"type": "turn", "deg": -30}, {"type": "forward", "m": 1.2}]
    acts = parse_actions(raw)
    assert acts == [Turn(-30.0), Forward(1.2)]
    assert actions_to_dicts(acts) == [{"type": "turn", "deg": -30.0},
                                      {"type": "forward", "m": 1.2}]


def _action_sampling():
    return importlib.import_module("pipeline.action_sampling")


def _matched_action_programs():
    return (
        [
            Turn(15.0), Forward(0.5),
            Turn(30.0), Forward(0.75),
            Turn(-15.0), Forward(0.5),
        ],
        [
            Turn(30.0), Forward(0.5),
            Turn(15.0), Forward(0.75),
            Turn(-15.0), Forward(0.5),
        ],
    )


def test_action_match_key_controls_every_preregistered_complexity_feature():
    sampling = _action_sampling()
    first, second = _matched_action_programs()

    first_key = sampling.action_match_key(first)
    second_key = sampling.action_match_key(second)

    assert first_key == second_key
    assert first_key == {
        "total_forward_bucket": 7,
        "primitive_count": 6,
        "total_turn_bucket": 4,
        "net_turn_bucket": 2,
        "turn_direction_sequence": [1, 1, -1],
        "forward_distance_profile": [2, 3, 2],
    }
    different_profile = [
        Turn(15.0), Forward(0.75),
        Turn(30.0), Forward(0.5),
        Turn(-15.0), Forward(0.5),
    ]
    different_directions = [
        Turn(15.0), Forward(0.5),
        Turn(-15.0), Forward(0.75),
        Turn(30.0), Forward(0.5),
    ]
    assert sampling.action_match_key(different_profile) != first_key
    assert sampling.action_match_key(different_directions) != first_key


def test_matched_action_units_are_variable_private_pairs_not_a_global_ratio():
    sampling = _action_sampling()
    first, second = _matched_action_programs()
    candidates = [
        {"group_id": "safe-a", "label": "safe", "actions": first},
        {"group_id": "collision-a", "label": "collision", "actions": second},
        {"group_id": "unmatched-safe", "label": "safe",
         "actions": [Forward(1.0)]},
    ]

    units = sampling.matched_action_units(candidates)

    assert len(units) == 1
    assert units[0]["safe_group_id"] == "safe-a"
    assert units[0]["collision_group_id"] == "collision-a"
    assert len(units[0]["pair_id"]) == 64
    assert units[0]["match_key"] == sampling.action_match_key(first)
    assert all("pair_id" not in candidate for candidate in candidates)


def test_matched_same_label_units_pair_safe_actions_without_label_transition():
    sampling = _action_sampling()
    first, second = _matched_action_programs()
    candidates = [
        {"group_id": "safe-a", "label": "safe", "actions": first},
        {"group_id": "safe-b", "label": "safe", "actions": second},
        {"group_id": "collision-a", "label": "collision", "actions": first},
    ]

    units = sampling.matched_same_label_units(candidates, label="safe")

    assert len(units) == 1
    assert units[0]["left_group_id"] == "safe-a"
    assert units[0]["right_group_id"] == "safe-b"
    assert units[0]["label"] == "safe"
    assert units[0]["match_key"] == sampling.action_match_key(first)
    assert units[0]["pair_id"] == sampling.expected_same_label_pair_id(
        units[0]["match_key"], "safe-a", "safe-b", label="safe")


def test_natural_candidate_selection_varies_count_without_reading_labels():
    sampling = _action_sampling()
    pools = {
        1: [(f"a-{index}", [Forward(0.5 + index * 0.25)])
            for index in range(6)],
    }
    labels = {
        group_id: ("safe" if index < 5 else "collision")
        for index, (group_id, _actions) in enumerate(pools[1])
    }

    two = sampling.select_natural_action_groups(
        pools, labels, pose_seed=0, min_actions=2, max_actions=5)
    five = sampling.select_natural_action_groups(
        pools, labels, pose_seed=3, min_actions=2, max_actions=5)
    flipped_labels = {
        group_id: ("collision" if label == "safe" else "safe")
        for group_id, label in labels.items()
    }
    two_after_label_flip = sampling.select_natural_action_groups(
        pools, flipped_labels, pose_seed=0, min_actions=2, max_actions=5)

    assert len(two) == 2
    assert len(five) == 5
    assert [group_id for group_id, _actions in two] == [
        group_id for group_id, _actions in two_after_label_flip]


def test_final_selection_keeps_every_certified_length_despite_c1_reservations():
    sampling = _action_sampling()
    pools = {
        length: [(f"L{length}-{index}", [Forward(0.5)] * length)
                 for index in range(3)]
        for length in config.GEN_LENGTHS
    }
    labels = {
        tag: ("safe" if index % 2 else "collision")
        for candidates in pools.values()
        for index, (tag, _actions) in enumerate(candidates)
    }
    reserved = ["L1-0", "L1-1", "L2-0", "L2-1"]

    selected = sampling.select_natural_action_groups(
        pools, labels, pose_seed=0, reserved_tags=reserved)

    assert set(reserved) <= {tag for tag, _actions in selected}
    assert {len(actions) for _tag, actions in selected} == \
        set(config.GEN_LENGTHS)


def test_v4_retention_keeps_every_certified_candidate_in_bank_order():
    """Dropping one certified ordinary tag would recreate the v3 final crop."""
    sampling = _action_sampling()
    pools = {
        1: [("L1-a", [Forward(0.5)]), ("L1-b", [Forward(1.0)])],
        2: [("L2-a", [Forward(0.5), Forward(0.5)]),
            ("L2-b", [Forward(1.0), Forward(1.0)])],
    }
    labels = {"L1-a": "safe", "L1-b": "collision", "L2-b": "safe"}

    selected = sampling.retain_certified_action_groups(
        pools, labels, maximum=3)

    assert [tag for tag, _actions in selected] == ["L1-a", "L1-b", "L2-b"]


def test_v4_retention_fails_closed_above_its_hard_limit():
    sampling = _action_sampling()
    pools = {1: [(f"tag-{index}", [Forward(0.5)]) for index in range(4)]}

    with pytest.raises(ValueError, match="certified actions exceed"):
        sampling.retain_certified_action_groups(
            pools, {tag: "safe" for tag, _actions in pools[1]}, maximum=3)


# --- the pre-certification shortlist ------------------------------------

def _shortlist_bank(per_cell=6, lengths=(1, 2, 3),
                    variants=("safe", "collision", "natural")):
    """A bank shaped like a real pose's: several lengths, all three variants."""
    pools, provenance = {}, {}
    for length in lengths:
        pools[length] = []
        for variant in variants:
            for index in range(per_cell):
                tag = f"{variant}-L{length}-{index}"
                pools[length].append((tag, [Forward(0.5)] * length))
                provenance[tag] = {"variant": variant}
    return pools, provenance


def _flat(shortlist):
    return [tag for length in sorted(shortlist)
            for tag, _actions in shortlist[length]]


def test_shortlist_gives_every_populated_cell_a_seat_before_seconds():
    """The failure this replaces: a flat hash order over an uneven bank left
    whole lengths with nothing certified, so L5/L6 published nothing.

    Round-robin makes the first pass over the cells *be* the floor, so the
    guarantee needs no separate pass to enforce it.
    """
    sampling = _action_sampling()
    pools, provenance = _shortlist_bank()
    shortlist = sampling.shortlist_action_bank(
        pools, provenance, pose_seed=7, budget=12)

    tags = _flat(shortlist)
    assert len(tags) == 12
    cells = {(tag.split("-")[0], tag.split("-")[1]) for tag in tags}
    assert len(cells) == 9          # 3 lengths x 3 variants, all present
    assert all(sum(1 for tag in tags
                   if tag.startswith(f"{variant}-L{length}-")) >= 1
               for length in (1, 2, 3)
               for variant in ("safe", "collision", "natural"))


def test_shortlist_treats_natural_distance_strata_as_distinct_cells():
    sampling = _action_sampling()
    pools = {3: []}
    provenance = {}
    for stratum in ("short", "mid", "near"):
        for index in range(2):
            tag = f"natural-{stratum}-{index}"
            pools[3].append((tag, [Forward(0.5), Turn(15), Forward(0.5)]))
            provenance[tag] = {
                "variant": "natural_dynamic",
                "natural_distance_stratum": stratum,
            }

    tags = _flat(sampling.shortlist_action_bank(
        pools, provenance, pose_seed=0, budget=3))

    assert {provenance[tag]["natural_distance_stratum"] for tag in tags} == {
        "short", "mid", "near"}


def test_k18_first_covers_each_length_and_natural_distance_stratum():
    sampling = _action_sampling()
    pools = {length: [] for length in range(1, 7)}
    provenance = {}
    for length in pools:
        for stratum in ("short", "mid", "near"):
            tag = f"natural-L{length}-{stratum}"
            pools[length].append((tag, [Forward(0.5)] * length))
            provenance[tag] = {
                "variant": "natural_dynamic",
                "natural_distance_stratum": stratum,
            }
        for variant in ("safe", "collision", "a1_control"):
            tag = f"{variant}-L{length}"
            pools[length].append((tag, [Forward(1.0)] * length))
            provenance[tag] = {"variant": variant}

    tags = _flat(sampling.shortlist_action_bank(
        pools, provenance, pose_seed=0, budget=18))
    lengths = {
        tag: length for length, candidates in pools.items()
        for tag, _actions in candidates}

    assert all(provenance[tag]["variant"] == "natural_dynamic"
               for tag in tags)
    assert {(lengths[tag], provenance[tag]["natural_distance_stratum"])
            for tag in tags} == {
        (length, stratum)
        for length in range(1, 7)
        for stratum in ("short", "mid", "near")
    }


def test_stratified_order_alternates_forward_and_turn_first_within_cell():
    sampling = _action_sampling()
    pools = {3: []}
    provenance = {}
    for index in range(3):
        for prefix, actions in (
                ("f", [Forward(0.5), Turn(15), Forward(0.5)]),
                ("t", [Turn(15), Forward(0.5), Turn(-15)])):
            tag = f"{prefix}-{index}"
            pools[3].append((tag, actions))
            provenance[tag] = {
                "variant": "natural_dynamic",
                "natural_distance_stratum": "mid",
            }

    ordered = sampling.stratified_action_order(
        pools, provenance, pose_seed=9)
    start_types = [tag.split("-")[0] for tag in ordered]

    assert start_types in (["f", "t", "f", "t", "f", "t"],
                           ["t", "f", "t", "f", "t", "f"])


def test_post_stability_retention_reuses_stratified_order_and_forces_query():
    sampling = _action_sampling()
    pools, provenance = _shortlist_bank(
        per_cell=2, lengths=(1, 2, 3),
        variants=("safe", "collision", "natural"))
    stable = {
        tag for candidates in pools.values() for tag, _actions in candidates}
    forced = "natural-L3-1"

    retained = sampling.retain_stratified_action_groups(
        pools, provenance, stable_tags=stable, pose_seed=13,
        maximum=10, forced_tags=(forced,))
    tags = [tag for tag, _actions in retained]

    assert tags[0] == forced
    assert {len(actions) for _tag, actions in retained} == {1, 2, 3}
    assert len(tags) == 10


def test_shortlist_cannot_read_a_label_because_none_exists_yet():
    """Label-blindness here is structural, not a rule to be obeyed.

    ``group_labels`` is the product of the certification this shortlist runs
    *before*, so the only way to prove independence is that the function has
    nowhere to be told a label -- passing one is a TypeError.
    """
    import inspect
    sampling = _action_sampling()
    parameters = inspect.signature(sampling.shortlist_action_bank).parameters
    assert "group_labels" not in parameters
    assert not any("label" in name for name in parameters)
    pools, provenance = _shortlist_bank()
    with pytest.raises(TypeError):
        sampling.shortlist_action_bank(
            pools, provenance, pose_seed=1, budget=8,
            group_labels={"safe-L1-0": "safe"})


def test_shortlist_is_deterministic_for_a_pose_and_moves_with_the_seed():
    sampling = _action_sampling()
    pools, provenance = _shortlist_bank()
    first = _flat(sampling.shortlist_action_bank(
        pools, provenance, pose_seed=11, budget=15))
    again = _flat(sampling.shortlist_action_bank(
        pools, provenance, pose_seed=11, budget=15))
    other = _flat(sampling.shortlist_action_bank(
        pools, provenance, pose_seed=12, budget=15))
    assert first == again
    assert first != other


def test_frozen_family_members_are_admitted_before_the_round_robin():
    """A family member that misses the shortlist never gets a certificate, and
    an incomplete family is discarded whole -- so it cannot compete for a seat.
    """
    sampling = _action_sampling()
    pools, provenance = _shortlist_bank()
    forced = ["natural-L3-5", "collision-L2-4"]
    tags = _flat(sampling.shortlist_action_bank(
        pools, provenance, pose_seed=3, budget=10, forced_tags=forced))
    assert set(forced) <= set(tags)
    assert len(tags) == 10
    with pytest.raises(ValueError):
        sampling.shortlist_action_bank(
            pools, provenance, pose_seed=3, budget=10,
            forced_tags=["not-in-the-bank"])


def test_shortlist_never_grows_a_bank_and_keeps_its_shape():
    sampling = _action_sampling()
    pools, provenance = _shortlist_bank(per_cell=1, lengths=(1, 2))
    shortlist = sampling.shortlist_action_bank(
        pools, provenance, pose_seed=2, budget=40)
    assert {length: [tag for tag, _actions in candidates]
            for length, candidates in shortlist.items()} == \
        {length: [tag for tag, _actions in candidates]
         for length, candidates in pools.items()}


def test_shortlist_reports_which_cells_it_starved():
    """A silent cut reads as "there was nothing there"; the funnel has to be
    able to tell that apart from "the budget ran out first"."""
    sampling = _action_sampling()
    pools, provenance = _shortlist_bank(per_cell=2, lengths=(1, 2))
    stats = collections.Counter()
    sampling.shortlist_action_bank(
        pools, provenance, pose_seed=4, budget=3, stats=stats)
    assert stats["shortlist_offered.safe.L1"] == 2
    assert sum(value for key, value in stats.items()
               if key.startswith("shortlist_admitted.")) == 3
    assert sum(value for key, value in stats.items()
               if key.startswith("shortlist_cell_starved.")) == 3
