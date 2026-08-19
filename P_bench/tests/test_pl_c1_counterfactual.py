"""Counterfactual C1 neighbour bank and candidate-only selection tests."""

import collections
import copy
import dataclasses
import hashlib
import inspect
import math
import numpy as np
import pytest
from PIL import Image

from pipeline import (
    action_proposal, action_sampling, actions as A, benchmark_tasks,
    c1_counterfactual,
    candidate_preview, collection_assets, collection_runtime,
    collection_support, config, consensus, dataset_contracts,
    future_view_selection, perception, record,
)
from tests._synthetic import LEVEL_FLOOR_FIT, make_frame, source_provenance
from tests.test_pl_v16_a_candidates import _case as _a_case


def _query(*primitives):
    return list(primitives)


TURN_15_FORWARD_1 = [
    {"type": "turn", "deg": 15.0}, {"type": "forward", "m": 1.0}]


def _clear_outcome(index, action_program):
    _rec, outcome = _a_case(collision=False)
    outcome = copy.deepcopy(outcome)
    outcome["outcome_id"] = f"clear-{index}"
    outcome["actions"] = copy.deepcopy(action_program)
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(
            outcome["actions"],
            outcome["shared_oracle_stability"]["rows"])
    parsed = A.parse_actions(outcome["actions"])
    x_m, z_m, heading_deg = A.pose_after(parsed)
    total_forward_m = A.total_forward_m(parsed)
    outcome["execution"] = {
        "completed": True,
        "stop_reason": "completed",
        "nominal_forward_m": total_forward_m,
        "executed_forward_m": total_forward_m,
        "realized_pose": {
            "x": x_m, "z": z_m, "heading_deg": heading_deg,
        },
    }
    outcome["checkpoints"] = [
        {
            "requested_progress": 0.0,
            "realized_progress": 0.0,
            "pose": {"x": 0.0, "z": 0.0, "heading_deg": 0.0},
        },
        {
            "requested_progress": 1.0,
            "realized_progress": 1.0,
            "pose": {"x": x_m, "z": z_m, "heading_deg": heading_deg},
        },
    ]
    return outcome


def _terminal_frame(base, outcome, value):
    endpoint = outcome["execution"]["realized_pose"]
    position = perception.world_from_local(
        np.asarray([[endpoint["x"], 0.0, endpoint["z"]]],
                   dtype=np.float64),
        base.position, base.yaw_rad)[0]
    return dataclasses.replace(
        base,
        frame_id=f"{base.frame_id}-{outcome['outcome_id']}",
        position=position,
        yaw_rad=-math.radians(endpoint["heading_deg"]),
        rgb=np.full_like(base.rgb, value, dtype=np.uint8),
    )


def test_the_neighbour_bank_is_a_pure_function_of_the_query():
    # Compilation has to rebuild this set from a published record, where no
    # pose seed survives.  A seed parameter here would mean the family could
    # not be re-derived, so the signature is part of the contract.
    signature = inspect.signature(c1_counterfactual.counterfactual_neighbors)
    assert list(signature.parameters) == ["query_actions"]

    first = c1_counterfactual.counterfactual_neighbors(TURN_15_FORWARD_1)
    second = c1_counterfactual.counterfactual_neighbors(
        copy.deepcopy(TURN_15_FORWARD_1))

    assert [value.tag for value in first] == [value.tag for value in second]
    assert [value.generator_id for value in first] == \
        [value.generator_id for value in second]
    assert c1_counterfactual.neighbor_generator_priority(
        TURN_15_FORWARD_1) != c1_counterfactual.NEIGHBOR_GENERATORS


def test_every_neighbour_stays_inside_the_published_action_vocabulary():
    queries = [
        [{"type": "forward", "m": 0.5}],
        TURN_15_FORWARD_1,
        [{"type": "turn", "deg": 15.0}, {"type": "forward", "m": 1.0},
         {"type": "turn", "deg": -30.0}, {"type": "forward", "m": 0.5},
         {"type": "turn", "deg": 45.0}, {"type": "forward", "m": 2.0}],
    ]
    for query in queries:
        neighbors = c1_counterfactual.counterfactual_neighbors(query)
        assert neighbors
        for neighbor in neighbors:
            A.validate_physics_actions(neighbor.actions)
            assert len(neighbor.actions) in config.GEN_LENGTHS
            assert neighbor.generator_id in \
                c1_counterfactual.NEIGHBOR_GENERATORS
        digests = [c1_counterfactual.action_sha256(value.actions)
                   for value in neighbors]
        assert len(set(digests)) == len(digests)
        assert c1_counterfactual.action_sha256(query) not in digests


def test_an_unpublishable_proposal_never_reaches_the_bank():
    # ``Q[:-1]`` of a two-primitive query is a bare Turn, which is not a
    # publishable program.  It has to vanish here rather than be offered and
    # rejected later, because an offered member would waste a bounded
    # second-pass slot.
    neighbors = c1_counterfactual.counterfactual_neighbors(TURN_15_FORWARD_1)
    assert not any(
        value.generator_id == "prefix_drop" for value in neighbors)

    six = [{"type": "turn", "deg": 15.0}, {"type": "forward", "m": 1.0},
           {"type": "turn", "deg": -30.0}, {"type": "forward", "m": 0.5},
           {"type": "turn", "deg": 45.0}, {"type": "forward", "m": 2.0}]
    families = {
        value.generator_id
        for value in c1_counterfactual.counterfactual_neighbors(six)}
    assert "extend_turn" not in families and "extend_turn_forward" not in \
        families
    assert families == {"prefix_drop", "last_forward_delta",
                        "reverse_last_turn"}

    # At the top of the grid ``d + 0.5`` leaves the vocabulary and only the
    # shorter sibling survives. The edge is read off the grid, not written as a
    # literal: v2 moved it from 3.0 to 6.0 and a 3.5 m neighbour is now
    # publishable rather than dropped.
    top = max(config.GEN_FORWARDS_M)
    edge = [{"type": "turn", "deg": 15.0}, {"type": "forward", "m": top}]
    deltas = [
        A.total_forward_m(value.actions)
        for value in c1_counterfactual.counterfactual_neighbors(edge)
        if value.generator_id == "last_forward_delta"]
    assert deltas == [pytest.approx(top - 0.5)]


def test_a_query_must_end_on_a_forward():
    assert c1_counterfactual.is_query_program(TURN_15_FORWARD_1)
    assert not c1_counterfactual.is_query_program(
        [{"type": "forward", "m": 1.0}, {"type": "turn", "deg": 15.0}])
    assert not c1_counterfactual.is_query_program([])
    with pytest.raises(ValueError):
        c1_counterfactual.counterfactual_neighbors(
            [{"type": "forward", "m": 1.0}, {"type": "turn", "deg": 15.0}])


# ---------------------------------------------------------------------------
# Collection side: the neighbours have to win shortlist slots.
# ---------------------------------------------------------------------------

def _bank(programs, *, variant=action_proposal.NATURAL_DYNAMIC_VARIANT):
    pools = collections.defaultdict(list)
    provenance = {}
    manifest = []
    for index, program in enumerate(programs):
        parsed = A.parse_actions(program)
        tag = c1_counterfactual.action_tag(parsed)
        pools[len(parsed)].append((tag, parsed))
        provenance[tag] = {
            "protocol": "test", "template_id": f"t{index:02d}",
            "variant": variant,
        }
        manifest.append({
            "length": len(parsed), "tag": tag, "variant": variant,
            "template_id": f"t{index:02d}",
            "actions": A.actions_to_dicts(parsed),
        })
    manifest.sort(key=lambda entry: (
        entry["length"], entry["template_id"], entry["variant"]))
    return dict(pools), provenance, manifest


def test_reserving_slots_banks_the_neighbours_the_compiler_will_need():
    pools, provenance, manifest = _bank([TURN_15_FORWARD_1])
    stats = collections.Counter()

    families = c1_counterfactual.reserve_pose_slots(
            pools, provenance, manifest, pose_seed=7, stats=stats,
            variant_of=collection_runtime._variant_of,
            group_labels={
                c1_counterfactual.action_tag(TURN_15_FORWARD_1): "safe"})
    family, = families
    query_tag, neighbor_tags = family.query_tag, family.neighbor_tags

    assert query_tag == c1_counterfactual.action_tag(TURN_15_FORWARD_1)
    assert len(neighbor_tags) == config.C1_NEIGHBORS_PER_QUERY
    banked = {tag for values in pools.values() for tag, _actions in values}
    assert set(neighbor_tags) <= banked
    assert all(provenance[tag]["variant"] == c1_counterfactual.VARIANT
               for tag in neighbor_tags)
    keys = [(entry["length"], entry["template_id"], entry["variant"])
            for entry in manifest]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert stats["c1_counterfactual_queries_banked"] == 1
    assert stats["c1_counterfactual_neighbors_banked"] == \
        config.C1_NEIGHBORS_PER_QUERY
    assert stats["c1_counterfactual_query_shortfall"] == 1


def test_a_pose_with_no_askable_program_costs_nothing():
    pools, provenance, manifest = _bank(
        [[{"type": "forward", "m": 1.0}, {"type": "turn", "deg": 15.0}]])
    stats = collections.Counter()
    before = copy.deepcopy(pools)

    families = c1_counterfactual.reserve_pose_slots(
            pools, provenance, manifest, pose_seed=7, stats=stats,
            variant_of=collection_runtime._variant_of,
            group_labels={})

    assert families == ()
    assert pools == before
    assert stats["c1_counterfactual_query_shortfall"] == 2


def test_a_collision_program_never_spends_a_query_slot():
    # A collision candidate is proposed in order to hit something, so it would
    # burn a query slot on a program that cannot complete clear.
    pools, provenance, manifest = _bank(
        [TURN_15_FORWARD_1], variant="collision")
    stats = collections.Counter()

    families = c1_counterfactual.reserve_pose_slots(
            pools, provenance, manifest, pose_seed=7, stats=stats,
            variant_of=collection_runtime._variant_of,
            group_labels={
                c1_counterfactual.action_tag(TURN_15_FORWARD_1):
                    "collision"})

    assert families == ()


def test_query_is_chosen_after_certification_from_a_completed_clear_program():
    rejected = TURN_15_FORWARD_1
    certified = [
        {"type": "forward", "m": 0.5},
        {"type": "turn", "deg": -15.0},
        {"type": "forward", "m": 1.0},
    ]
    pools, provenance, manifest = _bank([rejected, certified], variant="safe")
    rejected_tag = c1_counterfactual.action_tag(rejected)
    certified_tag = c1_counterfactual.action_tag(certified)

    families = c1_counterfactual.reserve_pose_slots(
        pools, provenance, manifest, pose_seed=7,
        stats=collections.Counter(),
        variant_of=collection_runtime._variant_of,
        group_labels={rejected_tag: "collision", certified_tag: "safe"})

    assert families[0].query_tag == certified_tag
    assert families[0].neighbor_tags


def test_post_label_recalled_neighbor_is_not_published_as_natural_a1():
    query = A.parse_actions(TURN_15_FORWARD_1)
    recalled = c1_counterfactual.counterfactual_neighbors(query)[0]
    pools, provenance, manifest = _bank(
        [TURN_15_FORWARD_1], variant="safe")
    _unused, recalled_provenance, recalled_manifest = _bank(
        [A.actions_to_dicts(recalled.actions)], variant="natural_dynamic")
    provenance.update(recalled_provenance)
    manifest.extend(recalled_manifest)
    query_tag = c1_counterfactual.action_tag(query)

    families = c1_counterfactual.reserve_pose_slots(
        pools, provenance, manifest, pose_seed=7,
        stats=collections.Counter(),
        variant_of=collection_runtime._variant_of,
        group_labels={query_tag: "safe"})

    assert recalled.tag in families[0].neighbor_tags
    assert provenance[recalled.tag]["variant"] == \
        c1_counterfactual.VARIANT
    stored = next(row for row in manifest if row["tag"] == recalled.tag)
    assert stored["variant"] == c1_counterfactual.VARIANT


def test_certified_query_and_neighbours_survive_the_final_selection():
    pools, provenance, manifest = _bank([TURN_15_FORWARD_1])
    stats = collections.Counter()
    families = c1_counterfactual.reserve_pose_slots(
            pools, provenance, manifest, pose_seed=7, stats=stats,
            variant_of=collection_runtime._variant_of,
            group_labels={
                c1_counterfactual.action_tag(TURN_15_FORWARD_1): "safe"})
    query_tag = families[0].query_tag
    neighbor_tags = families[0].neighbor_tags
    # Crowd the bank. C1 neighbours are generated only after the broad bank
    # has been shortlisted and the query passed the physical precheck.
    filler = [
        [{"type": "forward", "m": 0.5}, {"type": "turn", "deg": deg},
         {"type": "forward", "m": 1.0}]
        for deg in (-45, -30, -15, 15, 30, 45)
    ]
    for index, program in enumerate(filler):
        parsed = A.parse_actions(program)
        tag = c1_counterfactual.action_tag(parsed)
        pools.setdefault(len(parsed), []).append((tag, parsed))
        provenance[tag] = {
            "protocol": "test", "template_id": f"f{index:02d}",
            "variant": "safe",
        }

    # Certification is allowed to reject a neighbour, but the query and every
    # surviving reserved neighbour must also survive the *final* hash cut.
    lost = neighbor_tags[-1]
    certified = {
        tag: "safe" for tag in (query_tag, *neighbor_tags) if tag != lost
    }
    selected = action_sampling.select_natural_action_groups(
        pools, certified, pose_seed=1,
        reserved_tags=(query_tag, *neighbor_tags))

    assert {tag for tag, _actions in selected} == set(certified)


def test_two_safe_queries_each_reserve_six_globally_unique_neighbours():
    queries = [
        [{"type": "forward", "m": 1.0}],
        [{"type": "forward", "m": 1.5}],
    ]
    pools, provenance, manifest = _bank(queries)
    stats = collections.Counter()
    labels = {
        c1_counterfactual.action_tag(query): "safe" for query in queries}

    families = c1_counterfactual.reserve_pose_slots(
        pools, provenance, manifest, pose_seed=17, stats=stats,
        variant_of=collection_runtime._variant_of, group_labels=labels)

    assert len(families) == config.C1_QUERIES_PER_POSE
    assert all(len(family.neighbor_tags) == config.C1_NEIGHBORS_PER_QUERY
               for family in families)
    query_tags = {family.query_tag for family in families}
    neighbor_tags = [tag for family in families
                     for tag in family.neighbor_tags]
    assert not query_tags.intersection(neighbor_tags)
    assert len(neighbor_tags) == len(set(neighbor_tags))
    assert stats["c1_neighbor_failure.duplicate_program"] >= 1
    assert stats["c1_counterfactual_query_shortfall"] == 0


def test_two_query_reservation_keeps_total_pose_bank_at_or_below_48():
    queries = [
        [{"type": "forward", "m": 1.0}],
        [{"type": "forward", "m": 1.5}],
    ]
    filler = [
        [{"type": "turn", "deg": deg},
         {"type": "forward", "m": distance}]
        for distance in config.GEN_FORWARDS_M
        for deg in config.GEN_TURNS_DEG
    ][:34]
    pools, provenance, manifest = _bank(queries + filler)
    labels = {
        tag: "safe" for values in pools.values() for tag, _actions in values}
    assert sum(len(values) for values in pools.values()) == 36

    families = c1_counterfactual.reserve_pose_slots(
        pools, provenance, manifest, pose_seed=19,
        stats=collections.Counter(),
        variant_of=collection_runtime._variant_of, group_labels=labels)

    assert len(families) == 2
    assert sum(len(values) for values in pools.values()) <= 48


def test_c1_terminal_batch_preloads_the_query_and_surviving_neighbours():
    programs = _query_and_neighbours(5)
    tags = [c1_counterfactual.action_tag(program) for program in programs]
    accepted = {}
    expected_poses = []
    for index, (tag, program) in enumerate(zip(tags, programs), 1):
        value = _clear_outcome(index, program)
        accepted[tag] = {"frame": [value]}
        expected_poses.append(tuple(
            value["checkpoints"][-1]["pose"][key]
            for key in ("x", "z", "heading_deg")))
    # A failed neighbour remains absent and does not poison the batch.
    missing = c1_counterfactual.counterfactual_neighbors(programs[0])[-1].tag
    cache = {
        tuple(round(float(value), 6) for value in pose):
            "sequential-history-dependent-rgb"
        for pose in expected_poses
    }
    calls = []
    authorized = set()

    def render(poses):
        calls.append(list(poses))
        for pose in poses:
            cache[tuple(round(float(value), 6) for value in pose)] = object()

    family = c1_counterfactual.C1Family(
        query_tag=tags[0], neighbor_tags=(*tags[1:], missing))
    count = collection_assets.preload_counterfactual_terminal_rgb(
        accepted, frame_id="frame", families=(family,), render_cache=cache,
        terminal_batch_renderer=render,
        authorized_outcome_ids=authorized)

    assert count == len(expected_poses)
    assert calls == [expected_poses]
    assert authorized == {
        accepted[tag]["frame"][0]["outcome_id"] for tag in tags
    }


def test_c1_terminal_batch_does_not_render_without_a_clear_query():
    program = TURN_15_FORWARD_1
    query_tag = c1_counterfactual.action_tag(program)
    query = _clear_outcome(1, program)
    query["physical"]["authority"] = "unavailable"
    calls = []

    family = c1_counterfactual.C1Family(
        query_tag=query_tag, neighbor_tags=())
    count = collection_assets.preload_counterfactual_terminal_rgb(
        {query_tag: {"frame": [query]}}, frame_id="frame",
        families=(family,), render_cache={},
        terminal_batch_renderer=lambda poses: calls.append(poses),
        authorized_outcome_ids=set())

    assert count == 0
    assert calls == []


def test_counterfactual_terminal_ids_exclude_incidental_clear_outcomes():
    programs = _query_and_neighbours(3)
    tags = [c1_counterfactual.action_tag(program) for program in programs]
    accepted = {
        tag: {"frame": [_clear_outcome(index, program)]}
        for index, (tag, program) in enumerate(zip(tags, programs), 1)
    }
    incidental = _clear_outcome(99, [{"type": "forward", "m": 3.0}])
    accepted["incidental"] = {"frame": [incidental]}
    family = c1_counterfactual.C1Family(tags[0], tuple(tags[1:]))

    eligible = collection_assets.counterfactual_terminal_outcome_ids(
        accepted, frame_id="frame", families=(family,))

    assert eligible == {
        accepted[tag]["frame"][0]["outcome_id"] for tag in tags}
    assert incidental["outcome_id"] not in eligible


def test_c1_terminal_batch_renders_two_family_union_once():
    programs = _query_and_neighbours(4)
    tags = [c1_counterfactual.action_tag(program) for program in programs]
    accepted = {}
    expected_poses = []
    for index, (tag, program) in enumerate(zip(tags, programs), 1):
        value = _clear_outcome(index, program)
        accepted[tag] = {"frame": [value]}
        expected_poses.append(tuple(
            value["checkpoints"][-1]["pose"][key]
            for key in ("x", "z", "heading_deg")))
    families = (
        c1_counterfactual.C1Family(tags[0], (tags[1],)),
        c1_counterfactual.C1Family(tags[2], (tags[3],)),
    )
    cache = {}
    calls = []

    def render(poses):
        calls.append(list(poses))
        for pose in poses:
            cache[tuple(round(float(value), 6) for value in pose)] = object()

    count = collection_assets.preload_counterfactual_terminal_rgb(
        accepted, frame_id="frame", families=families, render_cache=cache,
        terminal_batch_renderer=render, authorized_outcome_ids=set())

    assert count == 4
    assert calls == [expected_poses[:4]]


def test_c1_collection_diagnostics_type_every_neighbor_failure():
    query = _clear_outcome(1, [{"type": "forward", "m": 1.0}])
    query["terminal_rgb_asset"] = {
        "pixel_sha256": "query-pixel", "png_sha256": "query-png"}
    missing_render = _clear_outcome(
        2, [{"type": "forward", "m": 1.5}])
    duplicate_image = _clear_outcome(
        3, [{"type": "forward", "m": 2.0}])
    duplicate_image["terminal_rgb_asset"] = {
        "pixel_sha256": "query-pixel", "png_sha256": "query-png"}
    unique_one = _clear_outcome(4, [{"type": "forward", "m": 2.5}])
    unique_one["terminal_rgb_asset"] = {
        "pixel_sha256": "unique-1", "png_sha256": "unique-png-1"}
    unique_two = _clear_outcome(5, [{"type": "forward", "m": 3.0}])
    unique_two["terminal_rgb_asset"] = {
        "pixel_sha256": "unique-2", "png_sha256": "unique-png-2"}
    family = c1_counterfactual.C1Family(
        query_tag="query",
        neighbor_tags=("physical", "render", "duplicate", "one", "two"),
        duplicate_program_count=2)
    accepted = {
        "query": {"frame": [query]},
        "render": {"frame": [missing_render]},
        "duplicate": {"frame": [duplicate_image]},
        "one": {"frame": [unique_one]},
        "two": {"frame": [unique_two]},
    }

    counts = c1_counterfactual.neighbor_failure_counts(
        (family,), accepted, frame_id="frame")

    assert counts == {"query": {
        "physical_rejection": 1,
        "render_rejection": 1,
        "duplicate_program": 2,
        "duplicate_image": 1,
        "insufficient_unique_distractors": 1,
    }}


def test_b1k_terminal_batch_renderer_has_no_sequential_fallback():
    render = collection_support.terminal_rgb_batch_renderer(
        object(), make_frame(), {})

    with pytest.raises(RuntimeError, match="simultaneous batch renderer"):
        render([(0.0, 0.0, 0.0)])


# ---------------------------------------------------------------------------
# Compile side: pick the query's own terminal frame plus three neighbours.
# ---------------------------------------------------------------------------

def _record(tmp_path, programs):
    rec, _unused = _a_case(collision=False)
    base = make_frame()
    base.frame_id = rec["frame_id"]
    base.scene_id = rec["source"]["scene_id"]
    rec.update({
        "scene_id": base.scene_id,
        "pose": {"position": base.position.tolist(), "yaw_rad": base.yaw_rad},
        "sensor": base.sensor.to_dict(),
        "collection_contract": record.r2r_v16_collection_contract(
            rec["source"], "main"),
        "camera_height_above_visible_floor_m":
            base.camera_height_above_visible_floor_m,
    })
    image = tmp_path / "initial.png"
    Image.fromarray(base.rgb, mode="RGB").save(
        image, format="PNG", optimize=False, compress_level=9)
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    outcomes = []
    for index, program in enumerate(programs, 1):
        outcome = _clear_outcome(index, list(program))
        terminal = _terminal_frame(base, outcome, 20 + index * 7)
        atom = future_view_selection.materialize_terminal_rgb_asset(
            tmp_path, base_frame=base, outcome=outcome,
            render_cache={
                future_view_selection.terminal_render_cache_key(outcome):
                    terminal,
            },
            source=rec["source"],
            collection_contract=rec["collection_contract"])
        outcome["terminal_rgb_asset"] = atom
        outcome["base_rollout_key"] = atom["binding"]["base_rollout_key"]
        outcomes.append(outcome)
    rec["outcomes"] = outcomes
    return rec, outcomes, image, image_sha256


def _query_and_neighbours(count):
    query = TURN_15_FORWARD_1
    neighbors = c1_counterfactual.counterfactual_neighbors(query)[:count]
    return [query] + [A.actions_to_dicts(value.actions)
                      for value in neighbors]


def test_the_query_answers_its_own_item_with_three_distinct_neighbours(
        tmp_path):
    programs = _query_and_neighbours(4)
    rec, outcomes, image, _sha = _record(tmp_path, programs)

    selection = c1_counterfactual.counterfactual_choices(
        rec, outcomes[0], asset_root=image.parent)

    assert selection["schema"] == "c1-counterfactual-selection.v2"
    assert selection["headline_eligible"] is False
    assert len(selection["choices"]) == 4
    assert selection["correct_outcome_id"] == outcomes[0]["outcome_id"]
    answer = next(value for value in selection["choices"]
                  if value["id"] == selection["canonical_answer"])
    assert answer["outcome_id"] == outcomes[0]["outcome_id"]
    assert answer["neighbor_generator_id"] == "query"
    assert len(selection["selected_neighbor_generator_ids"]) == 3
    assert selection["neighbor_generator_priority"] == list(
        c1_counterfactual.neighbor_generator_priority(
            TURN_15_FORWARD_1))
    assert len({value["action_sha256"]
                for value in selection["choices"]}) == 4
    assert len({value["terminal_rgb_sha256"]
                for value in selection["choices"]}) == 4
    assert len({value["terminal_pixel_sha256"]
                for value in selection["choices"]}) == 4
    assert selection["permutation_outcome_ids"] == [
        value["outcome_id"] for value in selection["choices"]]
    pairs = selection["pairwise_block_l1"]
    assert len(pairs) == 6
    assert {tuple(value["choice_ids"]) for value in pairs} == {
        ("image_1", "image_2"), ("image_1", "image_3"),
        ("image_1", "image_4"), ("image_2", "image_3"),
        ("image_2", "image_4"), ("image_3", "image_4"),
    }
    assert all(value["certificate"]["protocol"] == "block-l1.v1"
               for value in pairs)


def test_block_l1_is_diagnostic_and_never_changes_selection(
        tmp_path, monkeypatch):
    programs = _query_and_neighbours(4)
    rec, outcomes, image, _sha = _record(tmp_path, programs)
    baseline = c1_counterfactual.counterfactual_choices(
        rec, outcomes[0], asset_root=image.parent)

    counter = iter(range(6))

    def diagnostic_only(left, right, left_features, right_features):
        del left, right, left_features, right_features
        value = next(counter)
        return {
            "protocol": "block-l1.v1", "block_size_px": 8,
            "resolution": [640, 480], "pair_pixel_sha256": [
                f"{value:064x}", f"{value + 1:064x}"],
            "numerator": value, "denominator": 1,
            "sha256": f"{value + 2:064x}",
        }

    monkeypatch.setattr(
        future_view_selection, "_block_l1_certificate_from_features",
        diagnostic_only)
    changed = c1_counterfactual.counterfactual_choices(
        rec, outcomes[0], asset_root=image.parent)

    assert changed["choices"] == baseline["choices"]
    assert changed["canonical_answer"] == baseline["canonical_answer"]
    assert changed["permutation_outcome_ids"] == \
        baseline["permutation_outcome_ids"]
    assert changed["pairwise_block_l1"] != baseline["pairwise_block_l1"]


def test_pairwise_block_l1_is_recomputed_during_validation(tmp_path):
    programs = _query_and_neighbours(4)
    rec, outcomes, image, _sha = _record(tmp_path, programs)
    selection = c1_counterfactual.counterfactual_choices(
        rec, outcomes[0], asset_root=image.parent)
    selection["pairwise_block_l1"][0]["certificate"]["numerator"] += 1
    payload = {key: value for key, value in selection.items()
               if key != "sha256"}
    selection["sha256"] = record.canonical_atom_sha256(payload)

    with pytest.raises(
            ValueError, match="counterfactual selection certificate changed"):
        c1_counterfactual.validate_counterfactual_selection(
            rec, outcomes[0], selection, asset_root=image.parent)


def test_counterfactual_candidates_allow_new_terminal_observations(tmp_path):
    programs = _query_and_neighbours(4)
    rec, outcomes, image, image_sha = _record(tmp_path, programs)
    for index, outcome in enumerate(outcomes, 1):
        outcome.setdefault("future_view", {})["objects_entering_view"] = [
            100 + index]

    selection, reason = benchmark_tasks.c_candidate_selection(
        rec, outcomes[0], asset_root=image.parent)

    assert reason == "eligible"
    assert selection["correct_outcome_id"] == outcomes[0]["outcome_id"]
    assert len(selection["choices"]) == 4
    assert selection["headline_eligible"] is False


def test_gs_builds_counterfactual_c1_from_bound_terminal_observations(
        tmp_path):
    from tests.test_pl_v18_record import _v18_frame_and_source

    base, source, binding = _v18_frame_and_source()
    image = tmp_path / "initial-gs.png"
    Image.fromarray(base.rgb, mode="RGB").save(
        image, format="PNG", optimize=False, compress_level=9)
    programs = _query_and_neighbours(4)
    outcomes = []
    for index, program in enumerate(programs, 1):
        outcome = _clear_outcome(index, program)
        rows = copy.deepcopy(outcome["shared_oracle_stability"]["rows"])
        for row in rows:
            row["physical"]["authority"] = "gs_collision_mesh"
            row["physical"]["geometry_authority_sha256"] = \
                binding.collision_authority_sha256
        certificate = consensus.build_a_stability_certificate(
            program, rows, require_contact_instance_witness=True)
        nominal = certificate["rows"][0]
        outcome["physical"] = copy.deepcopy(nominal["physical"])
        outcome["depth_physical"] = copy.deepcopy(nominal["depth_physical"])
        outcome["oracle_consensus"] = copy.deepcopy(nominal["consensus"])
        outcome["evidence"]["physical"]["coverage"] = \
            nominal["corridor_coverage"]
        outcome["shared_oracle_stability"] = certificate
        outcome.setdefault("future_view", {})["objects_entering_view"] = [
            100 + index]
        terminal = _terminal_frame(base, outcome, 20 + index * 7)
        atom = future_view_selection.materialize_terminal_rgb_asset(
            tmp_path, base_frame=base, outcome=outcome,
            render_cache={
                future_view_selection.terminal_render_cache_key(outcome):
                    terminal,
            },
            source=source, collection_contract=None)
        outcome["terminal_rgb_asset"] = atom
        outcome["base_rollout_key"] = atom["binding"]["base_rollout_key"]
        outcomes.append(outcome)
    rec = record.build_record_v18(
        base, outcomes, image_path=image.name,
        floor_calibration=LEVEL_FLOOR_FIT, source_provenance=source)

    selection, reason = benchmark_tasks.c_candidate_selection(
        rec, rec["outcomes"][0], asset_root=tmp_path)

    assert reason == "eligible"
    assert selection["headline_eligible"] is False
    assert len(selection["choices"]) == 4
    assert {choice["neighbor_generator_id"]
            for choice in selection["choices"]} >= {"query"}


def test_the_option_order_is_a_deterministic_function_of_the_family(tmp_path):
    programs = _query_and_neighbours(4)
    rec, outcomes, image, _sha = _record(tmp_path, programs)

    first = c1_counterfactual.counterfactual_choices(
        rec, outcomes[0], asset_root=image.parent)
    second = c1_counterfactual.counterfactual_choices(
        copy.deepcopy(rec), copy.deepcopy(rec["outcomes"])[0],
        asset_root=image.parent)

    assert first == second
    assert first["permutation_outcome_ids"] == \
        second["permutation_outcome_ids"]
    c1_counterfactual.validate_counterfactual_selection(
        rec, outcomes[0], first, asset_root=image.parent)


def test_record_selection_index_is_exactly_equivalent_to_direct_selection(
        tmp_path):
    programs = _query_and_neighbours(4)
    rec, outcomes, image, _sha = _record(tmp_path, programs)
    expected = [
        c1_counterfactual.candidate_selection_or_reason(
            rec, outcome, asset_root=image.parent)
        for outcome in outcomes
    ]

    index = c1_counterfactual.build_record_selection_index(
        rec, asset_root=image.parent)
    actual = [
        c1_counterfactual.candidate_selection_or_reason(
            rec, outcome, asset_root=image.parent, selection_index=index)
        for outcome in outcomes
    ]

    assert actual == expected


def test_selection_index_decodes_only_the_four_selected_terminal_assets(
        tmp_path, monkeypatch):
    programs = _query_and_neighbours(4)
    rec, outcomes, image, _sha = _record(tmp_path, programs)
    calls = []
    feature_calls = []
    original = future_view_selection._decode_native_rgb_png
    original_features = future_view_selection.block_l1_features

    def counted(payload, **kwargs):
        calls.append(kwargs["expected_png_sha256"])
        return original(payload, **kwargs)

    def counted_features(pixels):
        feature_calls.append(pixels)
        return original_features(pixels)

    monkeypatch.setattr(
        future_view_selection, "_decode_native_rgb_png", counted)
    monkeypatch.setattr(
        future_view_selection, "block_l1_features", counted_features)

    index = c1_counterfactual.build_record_selection_index(
        rec, asset_root=tmp_path)
    assert calls == []
    selection = c1_counterfactual.counterfactual_choices(
        rec, outcomes[0], asset_root=image.parent, selection_index=index)

    assert sorted(calls) == sorted(
        choice["terminal_rgb_sha256"] for choice in selection["choices"])
    assert len(calls) == 4
    assert len(feature_calls) == 4


def test_compile_hashes_each_record_once_for_all_source_atoms(
        tmp_path, monkeypatch):
    programs = _query_and_neighbours(4)
    rec, _outcomes, image, _sha = _record(tmp_path, programs)
    rec["schema_version"] = record.SCHEMA_VERSION
    rec["oracle_contract_version"] = record.ORACLE_CONTRACT_VERSION
    rec["image_path"] = image.name
    calls = 0
    original = candidate_preview._canonical_sha256

    def counted(value):
        nonlocal calls
        if value is rec:
            calls += 1
        return original(value)

    monkeypatch.setattr(candidate_preview, "_canonical_sha256", counted)

    candidate_preview.compile_main_records(
        [rec], asset_root=tmp_path, build_root=tmp_path / "build")

    assert calls == 1


def test_one_lost_neighbour_costs_the_pose_a_distractor_not_the_item(
        tmp_path):
    # This is the whole reason the pose reserves more neighbours than an item
    # needs: the oracle is allowed to reject one and the item still exists.
    programs = _query_and_neighbours(4)
    rec, outcomes, image, _sha = _record(tmp_path, programs)
    full = c1_counterfactual.counterfactual_choices(
        rec, outcomes[0], asset_root=image.parent)
    dropped = copy.deepcopy(rec)
    dropped["outcomes"] = [
        value for value in dropped["outcomes"]
        if value["outcome_id"] != outcomes[1]["outcome_id"]]

    selection = c1_counterfactual.counterfactual_choices(
        dropped, dropped["outcomes"][0], asset_root=image.parent)

    def _distractors(value):
        return {row["action_sha256"] for row in value["choices"]
                if row["id"] != value["canonical_answer"]}

    assert len(selection["choices"]) == 4
    assert selection["canonical_answer"] in {
        value["id"] for value in selection["choices"]}
    assert _distractors(selection) != _distractors(full)
    assert len(_distractors(selection)) == 3


def test_too_few_neighbours_is_a_typed_shortfall_not_a_crash(tmp_path):
    programs = _query_and_neighbours(2)
    rec, outcomes, image, _sha = _record(tmp_path, programs)

    selection, reason = c1_counterfactual.candidate_selection_or_reason(
        rec, outcomes[0], asset_root=image.parent)

    assert selection is None
    assert reason == "counterfactual_distractor_shortfall"


def test_a_program_that_is_not_a_query_reports_its_own_reason(tmp_path):
    programs = _query_and_neighbours(4)
    rec, outcomes, image, _sha = _record(tmp_path, programs)
    turn_last = next(
        value for value in outcomes
        if not c1_counterfactual.is_query_program(value["actions"]))

    selection, reason = c1_counterfactual.candidate_selection_or_reason(
        rec, turn_last, asset_root=image.parent)

    assert selection is None
    assert reason == "counterfactual_query_not_eligible"


def test_counterfactual_is_the_only_c1_selector(tmp_path):
    programs = _query_and_neighbours(4)
    rec, outcomes, image, _image_sha256 = _record(tmp_path, programs)

    selection, reason = benchmark_tasks.c_candidate_selection(
        rec, outcomes[0], asset_root=image.parent)

    assert reason == "eligible"
    assert selection["schema"] == c1_counterfactual.SELECTION_SCHEMA
    assert selection["headline_eligible"] is False
    with pytest.raises(TypeError):
        benchmark_tasks.c_candidate_selection(
            rec, outcomes[0], asset_root=image.parent, gate={})


def _counterfactual_projection(tmp_path):
    """One compiled record whose only C1 item came from the neighbour bank."""
    import json

    programs = _query_and_neighbours(4)
    rec, _outcomes, image, _sha = _record(tmp_path, programs)
    rec["schema_version"] = record.SCHEMA_VERSION
    rec["oracle_contract_version"] = record.ORACLE_CONTRACT_VERSION
    rec["image_path"] = image.name
    (tmp_path / "records.jsonl").write_text(
        json.dumps(record.json_value(rec), sort_keys=True) + "\n")
    return candidate_preview.compile_main_records(
        [rec], asset_root=tmp_path, build_root=tmp_path / "build")


def test_a_counterfactual_c1_artifact_writes_without_an_appearance_gate(
        tmp_path):
    # The selection route already skips the uncalibrated appearance gate, but
    # the artifact writer used to demand a gate authority for any C1 item at
    # all -- which made every counterfactual run die at the last step.
    from tests.test_pl_v16_candidate_preview import _write_preview_artifact

    projection = _counterfactual_projection(tmp_path)
    assert any(item["task_id"] == "C1_future_view_selection"
               for item in projection["items"])

    result = _write_preview_artifact(
        projection, tmp_path / "candidate_qa",
        source_records_path=tmp_path / "records.jsonl")

    assert result["headline_eligible"] is False
    assert result["coverage"]["C1_future_view_selection"] == 1
