"""Frozen v16 shared visible-space oracle contract.

These tests collect the substrate behaviors reused by A, B, and C. They do not
introduce a second rollout, depth, corridor, or consensus implementation.
"""

import collections
import copy
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import (
    c1_counterfactual, collection_proposals, collection_runtime, config,
    consequence,
    consensus as consensus_module, record as record_fields, rollout,
    semantic as semantic_module, sim as sim_module, source_manifest, validate,
)
from pipeline.actions import Forward, Turn
from pipeline.consensus import oracle_consensus
from pipeline.geometry import Disc
from tests._synthetic import (
    _outcome, _record, clean_record, make_frame, source_provenance,
)


def _physical(collision, arc=None, *, instance_id=None, authority="navmesh"):
    contact = None
    if collision:
        contact = {
            "full_geometry_attribution": {
                "instance_id": instance_id,
                "category": "chair" if instance_id else None,
                "unattributed": instance_id is None,
            },
        }
    return {
        "authority": authority,
        "collision": collision,
        "first_contact_arc_m": arc,
        "contact": contact,
    }


def _depth(collision, arc=None, *, instance_id=None):
    contact = None
    if collision:
        contact = {
            "depth_mask_attribution": {
                "instance_id": instance_id,
                "category": "chair" if instance_id else None,
                "unattributed": instance_id is None,
            },
        }
    return {
        "authority": "depth",
        "collision": collision,
        "first_contact_arc_m": arc,
        "contact": contact,
    }


@pytest.mark.parametrize(
    ("full_id", "depth_id", "reason"),
    [
        (None, 7, "missing_contact_instance_witness"),
        (7, None, "missing_contact_instance_witness"),
        (7, 8, "contact_instance_mismatch"),
    ],
)
def test_r2r_collision_requires_non_null_same_instance_visible_witness(
        full_id, depth_id, reason):
    """Catches blind, unattributed, and cross-instance R2R collisions."""
    result = oracle_consensus(
        _physical(True, 1.0, instance_id=full_id),
        _depth(True, 1.0, instance_id=depth_id),
        1.0,
        require_contact_instance_witness=True,
    )

    assert result["accepted"] is False
    assert result["reason"] == reason


def test_r2r_collision_accepts_matching_visible_instance_witness():
    """Catches a witness gate that rejects valid same-instance attribution."""
    result = oracle_consensus(
        _physical(True, 1.0, instance_id=7),
        _depth(True, 1.0, instance_id=7),
        1.0,
        require_contact_instance_witness=True,
    )

    assert result["accepted"] is True
    assert result["full_contact_instance_id"] == 7
    assert result["depth_contact_instance_id"] == 7


def test_strict_safe_consensus_persists_the_route_marker():
    """Catches the safe early-return omitting the strict-route certificate."""
    result = oracle_consensus(
        _physical(False),
        _depth(False),
        1.0,
        require_contact_instance_witness=True,
    )

    assert result["accepted"] is True
    assert result["contact_instance_witness_required"] is True


@pytest.mark.parametrize("valid_id", [7, np.int64(7)])
def test_r2r_witness_accepts_only_positive_integer_instance_ids(valid_id):
    result = oracle_consensus(
        _physical(True, 1.0, instance_id=valid_id),
        _depth(True, 1.0, instance_id=valid_id),
        1.0,
        require_contact_instance_witness=True,
    )

    assert result["accepted"] is True
    assert result["full_contact_instance_id"] == 7
    assert result["depth_contact_instance_id"] == 7


@pytest.mark.parametrize(
    "malformed_id",
    [7.0, 7.1, 7.9, "7", True, False, 0, -3, None, [], {}],
)
def test_r2r_witness_rejects_malformed_instance_ids_without_raising(
        malformed_id):
    result = oracle_consensus(
        _physical(True, 1.0, instance_id=malformed_id),
        _depth(True, 1.0, instance_id=malformed_id),
        1.0,
        require_contact_instance_witness=True,
    )

    assert result["accepted"] is False
    assert result["reason"] == "missing_contact_instance_witness"
    assert result["full_contact_instance_id"] is None
    assert result["depth_contact_instance_id"] is None


class _HalfPlaneNav:
    authority = "navmesh"

    def __init__(self, limit):
        self.limit = float(limit)

    def is_navigable(self, pose):
        return float(pose[1]) <= self.limit + 1e-9

    def clearance(self, pose):
        return max(0.0, self.limit - float(pose[1]))

    def closest_obstacle(self, pose):
        return {
            "world_point": [float(pose[0]), 0.0, self.limit],
            "world_normal": [0.0, 0.0, -1.0],
            "distance_m": self.clearance(pose),
        }

    def rebase(self, pose):
        del pose
        return self


class _FullInstanceSeven:
    def assign(self, points):
        return np.full(len(points), 7, dtype=np.int64)

    def instance_points(self, instance_id):
        if int(instance_id) == 7:
            return np.array([[0.0, 0.15, -2.0]], dtype=np.float64)
        return np.empty((0, 3), dtype=np.float64)


def test_r2r_final_consensus_does_not_trust_cached_pre_attribution_acceptance():
    """Catches a cached precheck bypassing the final same-instance gate."""
    frame = make_frame()
    frame = dataclasses.replace(
        frame,
        pts_sem=np.full_like(frame.pts_sem, 8),
        id_to_cat={7: "chair", 8: "table"},
        semantic_index=_FullInstanceSeven(),
    )
    actions = [Forward(1.5)]
    nav = _HalfPlaneNav(1.0)
    physical = rollout.physical_rollout(nav, actions)
    depth = {
        "authority": "depth",
        "collision": True,
        "first_contact_arc_m": physical["physical"]["first_contact_arc_m"],
        "contact_action_index": 0,
        "contact_action_local_arc_m": 1.0,
        "center_local": [0.0, 1.0],
        "realized_pose": {"x": 0.0, "z": 1.0, "heading_deg": 0.0},
    }

    outcome = consequence.judge(
        frame,
        Disc(0.2),
        actions,
        nav=nav,
        target_ids=[],
        cached_physical=physical,
        cached_depth_physical=depth,
        cached_corridor_coverage=1.0,
        cached_oracle_consensus={
            "accepted": True,
            "verdict": "agree_collision",
            "reason": "accepted",
        },
        require_contact_instance_witness=True,
    )

    assert outcome["oracle_consensus"]["accepted"] is False
    assert outcome["oracle_consensus"]["reason"] == \
        "contact_instance_mismatch"


@pytest.mark.parametrize(
    ("collection_mode", "source_dataset", "expected"),
    [
        ("main", "r2r", True),
        ("not-main", "r2r", False),
        ("diagnostic", "r2r", False),
    ],
)
def test_only_v16_r2r_main_requires_the_shared_instance_gate(
        collection_mode, source_dataset, expected):
    """Catches implicit backend detection or leakage into legacy modes."""
    assert collection_runtime.v16_r2r_shared_oracle_required(
        collection_mode, source_dataset) is expected


def test_r2r_main_evaluator_wires_strict_gate_into_judge(
        monkeypatch):
    """Catches the strict helper existing without protecting collection."""
    class Sim:
        source_dataset = "r2r"

        def recompute_navmesh(self, _radius, height):
            del height
            pass

        def nav(self, _position, _yaw):
            return _AlwaysSafeNav()

    frame = make_frame()
    physical = {
        "physical": {"collision": False},
        "execution": {
            "realized_pose": {"x": 0.0, "z": 1.0, "heading_deg": 0.0},
            "nominal_pose": {"x": 0.0, "z": 1.0, "heading_deg": 0.0},
        },
    }
    seen = {}

    def fake_judge(*_args, **kwargs):
        seen["judge"] = kwargs["require_contact_instance_witness"]
        return {
            "physical": {
                "authority": "navmesh", "collision": False,
                "first_contact_arc_m": None,
            },
            "depth_physical": {
                "authority": "depth", "collision": False,
                "first_contact_arc_m": None,
            },
            "evidence": {"physical": {"coverage": 1.0}},
            "execution": {"completed": True},
        }

    def fake_disposition(_outcome):
        return "keep"

    assert not hasattr(collection_runtime.rollout, "terminal_options")
    monkeypatch.setattr(collection_runtime, "judge", fake_judge)
    monkeypatch.setattr(
        collection_runtime, "collect_a_stability_rows",
        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        collection_runtime,
        "structured_outcome_disposition",
        fake_disposition,
    )
    stats = collections.Counter()
    pending = []
    evaluator = collection_runtime._make_structured_spec_evaluator(
        args=SimpleNamespace(collection_mode="main"),
        variants=[(Sim(), frame)],
        target_ids_by_frame={frame.frame_id: []},
        group_labels={"action": "safe"},
        precheck_cache={},
        render_caches=collections.defaultdict(dict),
        stats=stats,
        skipped=collections.Counter(),
        pending_a_certificates=pending,
    )

    evaluator({
        "action_tag": "action",
        "actions": [Forward(1.0)],
        "radii": [0.2],
        "type": "main",
        "_physical_by_radius": {0.2: physical},
    })

    assert seen == {"judge": True}
    assert stats["a_stability_rerollouts"] == 7
    assert len(pending) == 1


def test_rejected_spec_runs_no_stability_for_an_earlier_passing_sibling(
        monkeypatch):
    class Sim:
        source_dataset = "r2r"

        def recompute_navmesh(self, _radius, height):
            del height

        def nav(self, _position, _yaw):
            return _AlwaysSafeNav()

    first = make_frame()
    second = dataclasses.replace(first, frame_id="second-frame")

    def fake_judge(_sim_frame, _body, _actions, **_kwargs):
        frame = _sim_frame
        return {
            "frame_id_for_test": frame.frame_id,
            "physical": {
                "authority": "navmesh", "collision": False,
                "first_contact_arc_m": None,
            },
            "depth_physical": {
                "authority": "depth", "collision": False,
                "first_contact_arc_m": None,
            },
            "evidence": {"physical": {"coverage": 1.0}},
            "execution": {"completed": True},
        }

    monkeypatch.setattr(collection_runtime, "judge", fake_judge)
    stability_calls = []

    def fake_stability(*args, **_kwargs):
        stability_calls.append(args)
        return []

    monkeypatch.setattr(
        collection_runtime, "collect_a_stability_rows", fake_stability)
    monkeypatch.setattr(
        collection_runtime, "structured_outcome_disposition",
        lambda outcome, **_kwargs: (
            "reject_spec" if outcome["frame_id_for_test"] == "second-frame"
            else "keep"))
    pending = []
    stats = collections.Counter()
    evaluator = collection_runtime._make_structured_spec_evaluator(
        args=SimpleNamespace(collection_mode="main"),
        variants=[(Sim(), first), (Sim(), second)],
        target_ids_by_frame={first.frame_id: [], second.frame_id: []},
        group_labels={"action": "safe"},
        precheck_cache={},
        render_caches=collections.defaultdict(dict),
        stats=stats,
        skipped=collections.Counter(),
        pending_a_certificates=pending,
    )

    result = evaluator({
        "action_tag": "action",
        "actions": [Forward(1.0)],
        "radii": [0.2],
        "type": "main",
    })

    assert result is None
    assert pending == []
    assert stability_calls == []
    assert stats["evaluated_outcomes"] == 2
    assert stats["a_stability_certificates"] == 0
    assert stats["a_stability_rerollouts"] == 0


def test_natural_evaluator_does_not_request_retired_semantic_rendering(
        monkeypatch):
    """A/B and counterfactual C1 use only the terminal RGB asset path."""
    class Sim:
        source_dataset = "r2r"

        def recompute_navmesh(self, _radius, height):
            del height

        def nav(self, _position, _yaw):
            return _AlwaysSafeNav()

    frame = make_frame()
    calls = []

    def fake_judge(*_args, **kwargs):
        calls.append(dict(kwargs))
        return {
            "physical": {
                "authority": "navmesh", "collision": False,
                "first_contact_arc_m": None,
            },
            "depth_physical": {
                "authority": "depth", "collision": False,
                "first_contact_arc_m": None,
            },
            "evidence": {"physical": {"coverage": 1.0}},
            "execution": {"completed": True},
        }

    monkeypatch.setattr(collection_runtime, "judge", fake_judge)
    monkeypatch.setattr(
        collection_runtime, "collect_a_stability_rows",
        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        collection_runtime, "structured_outcome_disposition",
        lambda _outcome: "keep")
    evaluator = collection_runtime._make_structured_spec_evaluator(
        args=SimpleNamespace(collection_mode="main"),
        variants=[(Sim(), frame)],
        target_ids_by_frame={frame.frame_id: []},
        group_labels={"action": "safe"},
        precheck_cache={},
        render_caches=collections.defaultdict(dict),
        stats=collections.Counter(),
        skipped=collections.Counter(),
        pending_a_certificates=[],
    )

    assert evaluator({
        "action_tag": "action",
        "actions": [Forward(1.0)],
        "radii": [0.2],
        "type": "main",
    }) is not None
    assert len(calls) == 1
    assert "future_renderer" not in calls[0]


def _minimal_pose_evaluation_state(frame):
    selected = [
        ("first", [Forward(1.0)]),
        ("second", [Forward(1.0)]),
    ]
    return {
        "variants": [(object(), frame)],
        "target_ids_by_frame": {frame.frame_id: []},
        "group_labels": {"first": "safe", "second": "safe"},
        "precheck_cache": {},
        "selected": selected,
        "selected_ids": ["first", "second"],
        "calibration": {},
        "position": np.zeros(3),
        "yaw": 0.0,
        "pools": {1: selected},
        "required_siblings": {},
        "selection_seed": 0,
        "base_frame": frame,
        "active_radii": [0.2],
        # The natural route plans no family, so nothing here is atomic and a
        # failed program drops itself instead of the pose.
        "family_candidates": (),
    }


def test_pose_rejection_drops_pending_certificates_without_finalizing(
        monkeypatch):
    captured = {}

    def fake_factory(**kwargs):
        captured["pending"] = kwargs["pending_a_certificates"]

        def evaluate(spec, future_cache_scope=None):
            del future_cache_scope
            captured["pending"].append(spec["action_tag"])
            if spec["action_tag"] == "second":
                return None
            return {"frame": [{"outcome_id": "first"}]}

        return evaluate

    monkeypatch.setattr(
        collection_runtime, "_make_structured_spec_evaluator", fake_factory)
    monkeypatch.setattr(
        collection_runtime, "finalize_a_stability_certificates",
        lambda _pending: pytest.fail("rejected pose must not finalize A3"))

    result = collection_runtime._evaluate_pose_candidates(
        _minimal_pose_evaluation_state(make_frame()),
        args=SimpleNamespace(), stats=collections.Counter(),
        skipped=collections.Counter())

    assert result is None
    assert captured["pending"] == ["first", "second"]


def test_successful_pose_finalizes_pending_certificates_once(monkeypatch):
    captured = {"finalized": [], "outcomes": []}

    def fake_factory(**kwargs):
        pending = kwargs["pending_a_certificates"]

        def evaluate(spec, future_cache_scope=None):
            del future_cache_scope
            pending.append(spec["action_tag"])
            outcome = {"outcome_id": spec["action_tag"]}
            captured["outcomes"].append(outcome)
            return {"frame": [outcome]}

        return evaluate

    monkeypatch.setattr(
        collection_runtime, "_make_structured_spec_evaluator", fake_factory)

    def finalize(pending):
        captured["finalized"].append(list(pending))
        for outcome in captured["outcomes"]:
            outcome["shared_oracle_stability"] = {
                "summary": {"collision_label_stable": True}}

    monkeypatch.setattr(
        collection_runtime, "finalize_a_stability_certificates", finalize)

    result = collection_runtime._evaluate_pose_candidates(
        _minimal_pose_evaluation_state(make_frame()),
        args=SimpleNamespace(), stats=collections.Counter(),
        skipped=collections.Counter())

    assert result is not None
    assert captured["finalized"] == [["first", "second"]]


def test_unstable_sibling_drops_its_action_group_before_persistence(
        monkeypatch):
    """A stable sibling must not let an unstable group reach the record."""
    frame = make_frame()
    state = _minimal_pose_evaluation_state(frame)
    state["group_labels"]["third"] = "safe"
    state["selected"].append(("third", [Forward(1.0)]))
    state["selected_ids"].append("third")
    pending_outcomes = []

    def fake_factory(**kwargs):
        pending = kwargs["pending_a_certificates"]

        def evaluate(spec, future_cache_scope=None):
            del future_cache_scope
            tag = spec["action_tag"]
            outcomes = [
                {"outcome_id": f"{tag}-a"},
                {"outcome_id": f"{tag}-b"},
            ]
            pending.extend(outcomes)
            pending_outcomes.extend(outcomes)
            return {"frame": outcomes}

        return evaluate

    def finalize(_pending):
        for outcome in pending_outcomes:
            stable = outcome["outcome_id"] != "second-b"
            outcome["shared_oracle_stability"] = {
                "summary": {"collision_label_stable": stable}}

    monkeypatch.setattr(
        collection_runtime, "_make_structured_spec_evaluator", fake_factory)
    monkeypatch.setattr(
        collection_runtime, "finalize_a_stability_certificates", finalize)
    skipped = collections.Counter()

    result = collection_runtime._evaluate_pose_candidates(
        state, args=SimpleNamespace(), stats=collections.Counter(),
        skipped=skipped)

    assert result is not None
    assert set(result["selected_ids"]) == {"first", "third"}
    assert set(result["accepted_outcomes"]) == {"first", "third"}
    assert skipped["stability_group_rejected"] == 1
    assert skipped["stability_sibling_collateral_outcomes"] == 1


def test_stability_reserve_refills_36_ordinary_and_12_c1_slots(monkeypatch):
    """Certification may try 48 ordinary actions without publishing >48."""
    frame = make_frame()
    ordinary = [f"ordinary-{index:02d}" for index in range(42)]
    neighbors = [f"neighbor-{index:02d}" for index in range(12)]
    query_tags = (ordinary[-2], ordinary[-1])
    selected = [
        (tag, [Forward(1.0)]) for tag in ordinary + neighbors]
    state = _minimal_pose_evaluation_state(frame)
    state.update({
        "selected": selected,
        "selected_ids": [tag for tag, _actions in selected],
        "pools": {1: selected},
        "group_labels": {tag: "safe" for tag, _actions in selected},
        "proposal_provenance": {
            **{tag: {"variant": "natural_dynamic"} for tag in ordinary},
            **{tag: {"variant": c1_counterfactual.VARIANT}
               for tag in neighbors},
        },
        "c1_families": (
            c1_counterfactual.C1Family(
                query_tag=query_tags[0],
                neighbor_tags=tuple(neighbors[:6])),
            c1_counterfactual.C1Family(
                query_tag=query_tags[1],
                neighbor_tags=tuple(neighbors[6:])),
        ),
    })
    pending_outcomes = []

    def fake_factory(**kwargs):
        del kwargs

        def evaluate(spec, future_cache_scope=None):
            del future_cache_scope
            outcome = {"outcome_id": spec["action_tag"]}
            pending_outcomes.append(outcome)
            return {frame.frame_id: [outcome]}

        return evaluate

    def finalize(_pending):
        for outcome in pending_outcomes:
            outcome["shared_oracle_stability"] = {
                "summary": {"collision_label_stable": True}}

    monkeypatch.setattr(
        collection_runtime, "_make_structured_spec_evaluator", fake_factory)
    monkeypatch.setattr(
        collection_runtime, "finalize_a_stability_certificates", finalize)

    result = collection_runtime._evaluate_pose_candidates(
        state, args=SimpleNamespace(), stats=collections.Counter(),
        skipped=collections.Counter())

    assert result is not None
    assert len(result["selected_ids"]) == 48
    assert set(query_tags).issubset(result["selected_ids"])
    assert set(neighbors).issubset(result["selected_ids"])
    retained_ordinary = [
        tag for tag in result["selected_ids"] if tag in ordinary]
    assert len(retained_ordinary) == 36
    assert len(set(ordinary) - set(retained_ordinary)) == 6


def test_validator_rebuilds_explicit_r2r_instance_witness_consensus():
    """Catches persisted accepted consensus hiding attribution tampering."""
    outcome = _outcome(collision=True, progress=0.4 / 1.5)
    outcome["depth_physical"]["contact"] = {
        "depth_mask_attribution": {
            "instance_id": 3,
            "category": "wall",
            "unattributed": False,
        },
    }
    outcome["oracle_consensus"] = oracle_consensus(
        outcome["physical"],
        outcome["depth_physical"],
        outcome["evidence"]["physical"]["coverage"],
        require_contact_instance_witness=True,
    )
    assert outcome["oracle_consensus"]["accepted"] is True
    rec = _record(outcome)
    rec["oracle_contract_version"] = record_fields.ORACLE_CONTRACT_VERSION
    rec["source"] = source_provenance("s", dataset="r2r")
    rec["collection_contract"] = record_fields.r2r_v16_collection_contract(
        rec["source"], "main")
    context = _r2r_v16_validation_context(rec["source"])
    rec["outcomes"][0]["depth_physical"]["contact"][
        "depth_mask_attribution"]["instance_id"] = 8

    errors = validate.validate_record_local(rec, context=context)

    assert any(
        "reconstructed shared-oracle consensus rejected: "
        "contact_instance_mismatch" in error
        for error in errors
    )


def _r2r_v16_validation_context(source, collection_mode="main"):
    return source_manifest.r2r_v16_registered_validation_context(
        [source],
        collection_mode=collection_mode,
        expected_schema_version=record_fields.SCHEMA_VERSION,
        expected_oracle_contract_version=(
            record_fields.ORACLE_CONTRACT_VERSION),
        authority_sha256="a" * 64,
    )


def test_external_v16_context_rejects_deleted_contract_and_route_marker():
    outcome = _outcome(collision=True, progress=0.4 / 1.5)
    outcome["depth_physical"]["contact"] = {
        "depth_mask_attribution": {
            "instance_id": 3,
            "category": "wall",
            "unattributed": False,
        },
    }
    outcome["oracle_consensus"] = oracle_consensus(
        outcome["physical"], outcome["depth_physical"], 1.0,
        require_contact_instance_witness=True)
    rec = _record(outcome)
    rec["oracle_contract_version"] = record_fields.ORACLE_CONTRACT_VERSION
    rec["source"] = source_provenance("s", dataset="r2r")
    context = _r2r_v16_validation_context(rec["source"])
    rec.pop("collection_contract", None)
    rec["outcomes"][0]["oracle_consensus"].pop(
        "contact_instance_witness_required", None)

    errors = validate.validate_record_local(rec, context=context)

    assert any("required collection contract is missing" in error
               for error in errors)
    assert any("strict shared-oracle consensus marker is missing" in error
               for error in errors)


def test_external_v16_context_rejects_record_contract_mismatch():
    outcome = _outcome(collision=False)
    outcome["oracle_consensus"] = oracle_consensus(
        outcome["physical"], outcome["depth_physical"], 1.0,
        require_contact_instance_witness=True)
    rec = _record(outcome)
    rec["oracle_contract_version"] = record_fields.ORACLE_CONTRACT_VERSION
    rec["source"] = source_provenance("s", dataset="r2r")
    context = _r2r_v16_validation_context(rec["source"])
    rec["collection_contract"] = record_fields.r2r_v16_collection_contract(
        rec["source"], "main")
    rec["collection_contract"]["collection_mode"] = "not-main"

    errors = validate.validate_record_local(rec, context=context)

    assert any("does not match external validation context" in error
               for error in errors)


def test_explicit_legacy_context_does_not_enable_strict_rebuild():
    rec = _record(_outcome(collision=False))
    rec["source"] = source_provenance("s", dataset="r2r")
    rec["outcomes"][0]["evidence"]["physical"]["coverage"] = 0.2

    implicit_errors = validate.validate_record_source_bound(rec)
    errors = validate.validate_record_local(
        rec, context=source_manifest.LEGACY_RECORD_VALIDATION_CONTEXT)

    assert implicit_errors == [
        f"[{rec['frame_id']}] source-bound validation context required"]
    assert not any("shared-oracle" in error for error in errors)
    assert not any("collection contract" in error for error in errors)


def test_missing_context_rejects_record_after_all_route_fields_are_deleted():
    rec = clean_record()
    rec.pop("source", None)
    rec.pop("collection_contract", None)
    rec["outcomes"][0]["oracle_consensus"].pop(
        "contact_instance_witness_required", None)

    errors = validate.validate_record_source_bound(rec)

    assert errors == [
        f"[{rec['frame_id']}] source-bound validation context required",
    ]


def test_registered_context_rejects_untrusted_run_contract_hash():
    source = source_provenance("s", dataset="r2r")
    run_contract = {
        "params": {"collection_mode": "main"},
        "resolved_scenes": [source],
    }

    with pytest.raises(ValueError, match="run contract hash"):
        source_manifest.r2r_v16_context_from_run_contract(
            run_contract,
            expected_run_contract_sha256="0" * 64,
            expected_schema_version=record_fields.SCHEMA_VERSION,
            expected_oracle_contract_version=(
                record_fields.ORACLE_CONTRACT_VERSION),
        )


@pytest.mark.parametrize("collision", [False, True])
def test_strict_r2r_record_rejects_missing_consensus_route_marker(collision):
    outcome = _outcome(
        collision=collision,
        progress=(0.4 / 1.5 if collision else 1.0),
    )
    if collision:
        outcome["depth_physical"]["contact"] = {
            "depth_mask_attribution": {
                "instance_id": 3,
                "category": "wall",
                "unattributed": False,
            },
        }
    outcome["oracle_consensus"] = oracle_consensus(
        outcome["physical"],
        outcome["depth_physical"],
        1.0,
        require_contact_instance_witness=collision,
    )
    outcome["oracle_consensus"].pop(
        "contact_instance_witness_required", None)
    rec = _record(outcome)
    rec["oracle_contract_version"] = record_fields.ORACLE_CONTRACT_VERSION
    rec["source"] = source_provenance("s", dataset="r2r")
    rec["collection_contract"] = record_fields.r2r_v16_collection_contract(
        rec["source"], "main")

    errors = validate.validate_record_local(
        rec, context=_r2r_v16_validation_context(rec["source"]))

    assert any(
        "strict shared-oracle consensus marker is missing" in error
        for error in errors
    )


def test_strict_r2r_safe_record_authoritatively_rebuilds_consensus():
    outcome = _outcome(collision=False)
    outcome["oracle_consensus"] = oracle_consensus(
        outcome["physical"],
        outcome["depth_physical"],
        1.0,
        require_contact_instance_witness=True,
    )
    rec = _record(outcome)
    rec["oracle_contract_version"] = record_fields.ORACLE_CONTRACT_VERSION
    rec["source"] = source_provenance("s", dataset="r2r")
    rec["collection_contract"] = record_fields.r2r_v16_collection_contract(
        rec["source"], "main")
    rec["outcomes"][0]["evidence"]["physical"]["coverage"] = 0.2

    errors = validate.validate_record_local(
        rec,
        context=_r2r_v16_validation_context(rec["source"], "main"),
    )

    assert any(
        "reconstructed shared-oracle consensus rejected: "
        "insufficient_depth_coverage" in error
        for error in errors
    )


def test_r2r_v16_collection_contract_is_hash_and_source_bound():
    outcome = _outcome(collision=False)
    outcome["oracle_consensus"] = oracle_consensus(
        outcome["physical"], outcome["depth_physical"], 1.0,
        require_contact_instance_witness=True)
    rec = _record(outcome)
    rec["oracle_contract_version"] = record_fields.ORACLE_CONTRACT_VERSION
    rec["source"] = source_provenance("s", dataset="r2r")
    rec["collection_contract"] = record_fields.r2r_v16_collection_contract(
        rec["source"], "main")
    context = _r2r_v16_validation_context(rec["source"])
    rec["collection_contract"]["sha256"] = "0" * 64

    errors = validate.validate_record_local(rec, context=context)

    assert any(
        "collection contract does not match external validation context" in error
        for error in errors
    )


def test_shared_oracle_safe_full_path_and_collision_prefix_are_distinct():
    """Locks full-corridor safety and executed-prefix collision evidence."""
    frame = make_frame()
    near = rollout.certified_near_field_m(frame, radius=0.2)
    depth = np.full_like(frame.depth, 0.35)
    occluding = dataclasses.replace(frame, depth=depth)
    actions = [Forward(near + 0.6)]

    safe_full_path = rollout.corridor_coverage(
        occluding, actions, radius=0.2)
    collision_prefix = rollout.corridor_coverage(
        occluding, actions, radius=0.2, max_arc_m=near * 0.9)

    assert safe_full_path < config.EVIDENCE_COVERAGE_MIN
    assert collision_prefix == 1.0


def test_shared_oracle_rejects_out_of_fov_and_occluded_corridors():
    """Catches blind swept space being treated as visible clear geometry."""
    frame = make_frame()
    near = rollout.certified_near_field_m(frame, radius=0.2)
    out_of_fov = rollout.corridor_coverage_details(
        frame,
        [Turn(45.0), Forward(min(2.0, near * 0.9))],
        radius=0.2,
    )
    depth = np.full_like(frame.depth, 5.0)
    horizon = int(round(frame.K[1, 2]))
    depth[horizon + 1:, :] = 1.0
    occluded = rollout.corridor_coverage_details(
        dataclasses.replace(frame, depth=depth),
        [Forward(rollout.certified_near_field_m(frame, 0.0) + 0.6)],
        radius=0.0,
    )

    assert out_of_fov["coverage"] == 0.0
    assert out_of_fov["out_of_frame_samples"] > 0
    assert occluded["coverage"] < config.EVIDENCE_COVERAGE_MIN
    assert occluded["occluded_samples"] > 0


def test_shared_oracle_does_not_let_an_evidence_gap_hide_near_field_contact():
    """Locks the certified blind-strip collision check at persistence."""
    rec = _record(_outcome(collision=True, progress=0.4 / 1.5))
    rec["oracle_contract_version"] = record_fields.ORACLE_CONTRACT_VERSION
    outcome = rec["outcomes"][0]
    outcome["depth_physical"] = {
        "authority": "depth",
        "collision": False,
        "first_contact_arc_m": None,
    }
    outcome["oracle_consensus"] = {
        "accepted": False,
        "reason": "insufficient_depth_coverage",
    }
    outcome["evidence"]["physical"]["status"] = "insufficient"

    errors = validate.validate_record_local(
        rec, context=source_manifest.LEGACY_RECORD_VALIDATION_CONTEXT)

    assert any(
        "hidden collision lies inside certified near field" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("full", "depth", "coverage", "reason"),
    [
        (_physical(True, 1.0), _depth(False), 1.0,
         "collision_state_mismatch"),
        (_physical(True, 1.0), _depth(True, 1.31), 1.0,
         "contact_arc_mismatch"),
        (_physical(False), _depth(False), 0.89,
         "insufficient_depth_coverage"),
    ],
)
def test_shared_oracle_fail_closed_consensus(full, depth, coverage, reason):
    """Locks label, arc, and complete-safe-corridor rejection branches."""
    result = oracle_consensus(full, depth, coverage)

    assert result["accepted"] is False
    assert result["reason"] == reason


def _safe_stability_row(perturbation_id, transform):
    return {
        "perturbation_id": perturbation_id,
        "transform": dict(transform),
        "physical": {
            "authority": "navmesh", "collision": False,
            "first_contact_arc_m": None,
        },
        "depth_physical": {
            "authority": "depth", "collision": False,
            "first_contact_arc_m": None,
        },
        "corridor_coverage": 1.0,
    }


def test_a_stability_certificate_uses_the_frozen_complete_se2_set():
    """Catches an optional, empty, reordered, or self-declared perturbation set."""
    expected = (
        ("nominal", 0.0, 0.0, 0.0),
        ("x_negative", -0.01, 0.0, 0.0),
        ("x_positive", 0.01, 0.0, 0.0),
        ("z_negative", 0.0, -0.01, 0.0),
        ("z_positive", 0.0, 0.01, 0.0),
        ("yaw_negative", 0.0, 0.0, -1.0),
        ("yaw_positive", 0.0, 0.0, 1.0),
    )
    assert tuple(
        (value["id"], value["x_m"], value["z_m"], value["yaw_deg"])
        for value in consensus_module.R2R_A_STABILITY_PERTURBATIONS
    ) == expected
    rows = [
        _safe_stability_row(value["id"], {
            "x_m": value["x_m"], "z_m": value["z_m"],
            "yaw_deg": value["yaw_deg"],
        })
        for value in consensus_module.R2R_A_STABILITY_PERTURBATIONS
    ]

    certificate = consensus_module.build_a_stability_certificate(
        [{"type": "forward", "m": 1.0}], rows)

    assert certificate["version"] == "r2r-a-stability.v1"
    assert len(certificate["sha256"]) == 64
    assert certificate["summary"] == {
        "collision": False,
        "collision_label_stable": True,
        "original_action_index": None,
        "original_action_index_stable": None,
        "contact_instance_id": None,
        "contact_instance_stable": None,
    }


def test_geometry_only_stability_is_explicit_and_reconstructible():
    """GS A1/A2 must not silently inherit the unavailable A3 witness."""
    actions = [{"type": "forward", "m": 1.0}]
    rows = []
    for value in consensus_module.R2R_A_STABILITY_PERTURBATIONS:
        rows.append({
            "perturbation_id": value["id"],
            "transform": {
                "x_m": value["x_m"], "z_m": value["z_m"],
                "yaw_deg": value["yaw_deg"],
            },
            "physical": _physical(True, 0.5, instance_id=None),
            "depth_physical": _depth(True, 0.5, instance_id=None),
            "corridor_coverage": 1.0,
        })

    certificate = consensus_module.build_a_stability_certificate(
        actions, rows, require_contact_instance_witness=False)
    nominal = {
        "physical": rows[0]["physical"],
        "depth_physical": rows[0]["depth_physical"],
        "evidence": {"physical": {"coverage": 1.0}},
    }

    assert certificate["contact_instance_witness_required"] is False
    assert certificate["summary"]["collision"] is True
    assert certificate["summary"]["collision_label_stable"] is True
    assert certificate["summary"]["original_action_index"] == 1
    assert certificate["summary"]["contact_instance_stable"] is False
    assert consensus_module.a_stability_certificate_mismatches(
        actions, certificate, nominal,
        require_contact_instance_witness=False) == []

    forged = copy.deepcopy(certificate)
    forged.pop("contact_instance_witness_required")
    assert "contact_instance_witness_required" in \
        consensus_module.a_stability_certificate_mismatches(
            actions, forged, nominal,
            require_contact_instance_witness=False)


def test_a_stability_certificate_rejects_an_incomplete_frozen_set():
    rows = [
        _safe_stability_row(value["id"], {
            "x_m": value["x_m"], "z_m": value["z_m"],
            "yaw_deg": value["yaw_deg"],
        })
        for value in consensus_module.R2R_A_STABILITY_PERTURBATIONS[:-1]
    ]

    with pytest.raises(ValueError, match="frozen perturbation set"):
        consensus_module.build_a_stability_certificate(
            [{"type": "forward", "m": 1.0}], rows)


def test_a3_stability_requires_the_same_exact_face_winner_per_perturbation():
    """Catches nominal-only exact identity plus sampled perturbation winners."""
    actions = [
        {"type": "forward", "m": 1.0},
        {"type": "turn", "deg": 30.0},
        {"type": "forward", "m": 1.0},
    ]
    rows = []
    for index, perturbation in enumerate(
            consensus_module.R2R_A_STABILITY_PERTURBATIONS):
        row = {
            "perturbation_id": perturbation["id"],
            "transform": {
                "x_m": perturbation["x_m"], "z_m": perturbation["z_m"],
                "yaw_deg": perturbation["yaw_deg"],
            },
            "physical": _physical(True, 1.5, instance_id=7),
            "depth_physical": _depth(True, 1.5, instance_id=7),
            "corridor_coverage": 1.0,
            "exact_contact_identity": {
                "confirmed": True,
                "reason": "confirmed",
                "instance_id": 9 if index == 3 else 7,
                "category": "chair",
                "streaming_instance_faces_sha256": "d" * 64,
                "contact_face_distance_m": 0.01,
                "runner_up_face_distance_m": 0.5,
                "face_distance_margin_m": 0.49,
            },
        }
        rows.append(row)

    certificate = consensus_module.build_a_stability_certificate(actions, rows)

    assert certificate["summary"]["contact_instance_stable"] is False
    assert certificate["summary"]["contact_instance_id"] is None


class _AlwaysSafeNav:
    authority = "navmesh"

    def is_navigable(self, _pose):
        return True

    def clearance(self, _pose):
        return 1.0

    def rebase(self, _pose):
        return self


class _RecordingSafeSim:
    def __init__(self):
        self.calls = []

    def nav(self, position, yaw):
        self.calls.append((tuple(float(value) for value in position), float(yaw)))
        return _AlwaysSafeNav()


def test_habitat_nav_nulls_nonfinite_contact_normal():
    """Catches Habitat NaN diagnostics escaping the simulator boundary."""
    class Pathfinder:
        def closest_obstacle_surface_point(self, point):
            return SimpleNamespace(
                hit_pos=np.asarray(point) + np.array([0.01, 0.0, 0.0]),
                hit_normal=np.array([np.nan, np.nan, np.nan]),
                hit_dist=0.01,
            )

    nav = sim_module.Nav(
        Pathfinder(), np.zeros(3), 0.0,
        radius_m=0.2, authority="navmesh")

    contact = nav.closest_obstacle((0.0, 0.0, 0.0))

    assert contact["world_normal"] is None
    json.dumps(contact, allow_nan=False)


def test_collection_normalizes_nonfinite_contact_diagnostic(monkeypatch):
    """Catches NaN equality rejecting two otherwise identical rerollouts."""
    physical = {
        "authority": "navmesh",
        "collision": True,
        "first_contact_arc_m": 0.5,
        "contact": {
            "world_normal": [np.nan, np.nan, np.nan],
            "full_geometry_attribution": {
                "instance_id": 7,
                "category": "chair",
                "unattributed": False,
            },
        },
    }
    depth = {
        "authority": "depth",
        "collision": True,
        "first_contact_arc_m": 0.5,
        "contact": {
            "depth_mask_attribution": {
                "instance_id": 7,
                "category": "chair",
                "unattributed": False,
            },
        },
    }
    oracle_inputs = {
        "physical": physical,
        "depth_physical": depth,
        "corridor_coverage": 1.0,
    }
    monkeypatch.setattr(
        consequence, "_stability_oracle_row",
        lambda *_args, **_kwargs: copy.deepcopy(oracle_inputs))
    monkeypatch.setattr(
        consequence, "_exact_face_identities_for_oracle_inputs",
        lambda _frame, rows: [None] * len(rows))
    nominal = {
        "physical": copy.deepcopy(physical),
        "depth_physical": copy.deepcopy(depth),
        "evidence": {"physical": {"coverage": 1.0}},
    }

    certificate = consequence.collect_a_stability_certificate(
        _RecordingSafeSim(), make_frame(), Disc(0.2),
        [Forward(1.0)], nominal)

    assert certificate["rows"][0]["physical"]["contact"][
        "world_normal"] == [None, None, None]
    json.dumps(certificate, allow_nan=False)


def test_collection_materializes_real_se2_stability_rerollouts():
    """Catches a collector attaching a self-reported stability summary."""
    frame = make_frame()
    sim = _RecordingSafeSim()
    nominal_inputs = consequence._stability_oracle_row(
        frame, Disc(0.2), [Forward(0.1)], _AlwaysSafeNav(),
        {"x_m": 0.0, "z_m": 0.0, "yaw_deg": 0.0})
    nominal = {
        "physical": nominal_inputs["physical"],
        "depth_physical": nominal_inputs["depth_physical"],
        "evidence": {"physical": {
            "coverage": nominal_inputs["corridor_coverage"]}},
    }

    certificate = consequence.collect_a_stability_certificate(
        sim, frame, Disc(0.2), [Forward(0.1)], nominal)

    assert len(sim.calls) == 7
    assert sim.calls[0] == ((0.0, 0.0, 0.0), 0.0)
    assert [row["perturbation_id"] for row in certificate["rows"]] == [
        "nominal", "x_negative", "x_positive", "z_negative",
        "z_positive", "yaw_negative", "yaw_positive",
    ]
    assert certificate["summary"]["collision_label_stable"] is True


def test_collection_rejects_fresh_nominal_disagreement():
    """Catches copied nominal inputs bypassing the real rerollout path."""
    frame = make_frame()
    nominal = {
        "physical": {
            "authority": "navmesh", "collision": True,
            "first_contact_arc_m": 0.5,
        },
        "depth_physical": {
            "authority": "depth", "collision": True,
            "first_contact_arc_m": 0.5,
        },
        "evidence": {"physical": {"coverage": 1.0}},
    }

    with pytest.raises(ValueError, match="fresh nominal rerollout"):
        consequence.collect_a_stability_certificate(
            _RecordingSafeSim(), frame, Disc(0.2), [Forward(0.1)], nominal)


def test_collection_batches_seven_collision_identity_queries(monkeypatch):
    class BatchIndex:
        def __init__(self):
            self.calls = []

        def confirm_contact_instances(self, requests):
            self.calls.append(list(requests))
            return [{
                "authority": "mp3d_full_face_universe",
                "schema": "mp3d-contact-face-identity.v1",
                "confirmed": True, "reason": "confirmed",
                "instance_id": 7, "category": "chair",
                "streaming_instance_faces_sha256": "d" * 64,
                "contact_face_distance_m": 0.01,
                "runner_up_face_distance_m": 0.5,
                "face_distance_margin_m": 0.49,
                "global_query_protocol":
                    "mp3d-complete-face-instance-universe.v1",
                "semantic_ply_sha256": "a" * 64,
                "global_universe_sha256":
                    semantic_module.complete_face_universe_sha256(
                        global_query_protocol=
                            "mp3d-complete-face-instance-universe.v1",
                        semantic_ply_sha256="a" * 64),
                "global_winner_instance_id": 7,
                "global_runner_up_instance_id": 9,
            } for _request in requests]

    index = BatchIndex()
    frame = dataclasses.replace(make_frame(), semantic_index=index)
    physical = _physical(True, 0.5, instance_id=7)
    physical["contact"]["world_point"] = [0.0, 0.15, 0.5]
    oracle_inputs = {
        "physical": physical,
        "depth_physical": _depth(True, 0.5, instance_id=7),
        "corridor_coverage": 1.0,
    }
    monkeypatch.setattr(
        consequence, "_stability_oracle_row",
        lambda *_args, **_kwargs: copy.deepcopy(oracle_inputs))
    nominal = {
        "physical": copy.deepcopy(oracle_inputs["physical"]),
        "depth_physical": copy.deepcopy(oracle_inputs["depth_physical"]),
        "evidence": {"physical": {"coverage": 1.0}},
    }

    consequence.collect_a_stability_certificate(
        _RecordingSafeSim(), frame, Disc(0.2), [Forward(1.0)], nominal)

    assert len(index.calls) == 1
    assert len(index.calls[0]) == 7


def test_batched_identity_preserves_mixed_safe_collision_row_alignment():
    class EchoBatchIndex:
        def confirm_contact_instances(self, requests):
            return [{"request_point": list(point)}
                    for _instance_id, point in requests]

    frame = dataclasses.replace(
        make_frame(), semantic_index=EchoBatchIndex())
    rows = []
    for index, collision in enumerate([False, True, False, True, True]):
        physical = _physical(
            collision, 0.5 if collision else None,
            instance_id=7 if collision else None)
        if collision:
            physical["contact"]["world_point"] = [float(index), 0.15, 0.5]
        rows.append({
            "physical": physical,
            "depth_physical": _depth(
                collision, 0.5 if collision else None,
                instance_id=7 if collision else None),
            "corridor_coverage": 1.0,
        })

    identities = consequence._exact_face_identities_for_oracle_inputs(
        frame, rows)

    assert identities == [
        None,
        {"request_point": [1.0, 0.15, 0.5]},
        None,
        {"request_point": [3.0, 0.15, 0.5]},
        {"request_point": [4.0, 0.15, 0.5]},
    ]


def _collision_stability_rows(point, *, instance_id=7):
    rows = []
    for perturbation in consensus_module.R2R_A_STABILITY_PERTURBATIONS:
        physical = _physical(True, 0.5, instance_id=instance_id)
        physical["contact"]["world_point"] = list(point)
        rows.append({
            "perturbation_id": perturbation["id"],
            "transform": {
                "x_m": perturbation["x_m"],
                "z_m": perturbation["z_m"],
                "yaw_deg": perturbation["yaw_deg"],
            },
            "physical": physical,
            "depth_physical": _depth(
                True, 0.5, instance_id=instance_id),
            "corridor_coverage": 1.0,
        })
    return rows


def _confirmed_identity(instance_id, point):
    return {
        "authority": "test",
        "schema": "test",
        "confirmed": True,
        "reason": "confirmed",
        "instance_id": int(instance_id),
        "category": "chair",
        "streaming_instance_faces_sha256": "d" * 64,
        "contact_face_distance_m": float(point[0]),
        "runner_up_face_distance_m": float(point[0]) + 0.5,
        "face_distance_margin_m": 0.5,
    }


def test_pose_finalizer_batches_exact_identity_across_outcomes():
    class BatchIndex:
        def __init__(self):
            self.calls = []

        def confirm_contact_instances(self, requests):
            self.calls.append(copy.deepcopy(list(requests)))
            return [
                _confirmed_identity(instance_id, point)
                for instance_id, point in requests
            ]

    index = BatchIndex()
    frame = dataclasses.replace(make_frame(), semantic_index=index)
    outcomes = [{}, {}]
    pending = [
        consequence.deferred_a_stability_certificate(
            frame, [Forward(1.0)],
            _collision_stability_rows([0.01, 0.15, 0.5]), outcomes[0]),
        consequence.deferred_a_stability_certificate(
            frame, [Forward(1.0)],
            _collision_stability_rows([0.02, 0.15, 0.5]), outcomes[1]),
    ]

    consequence.finalize_a_stability_certificates(pending)

    assert len(index.calls) == 1
    assert len(index.calls[0]) == 14
    assert all(
        row["exact_contact_identity"]["contact_face_distance_m"] == 0.01
        for row in outcomes[0]["shared_oracle_stability"]["rows"])
    assert all(
        row["exact_contact_identity"]["contact_face_distance_m"] == 0.02
        for row in outcomes[1]["shared_oracle_stability"]["rows"])


def test_pose_finalizer_replays_legacy_outcome_batches_after_combined_error():
    class PartiallyBadIndex:
        def __init__(self):
            self.calls = []

        def confirm_contact_instances(self, requests):
            requests = list(requests)
            self.calls.append(copy.deepcopy(requests))
            bad = next((instance_id for instance_id, _point in requests
                        if instance_id == 99), None)
            if bad is not None:
                raise ValueError(f"bad witness {bad}")
            return [
                _confirmed_identity(instance_id, point)
                for instance_id, point in requests
            ]

    consequence._reset_a_stability_batch_diagnostics()
    index = PartiallyBadIndex()
    frame = dataclasses.replace(make_frame(), semantic_index=index)
    outcomes = [{}, {}]
    pending = [
        consequence.deferred_a_stability_certificate(
            frame, [Forward(1.0)],
            _collision_stability_rows(
                [0.01, 0.15, 0.5], instance_id=99), outcomes[0]),
        consequence.deferred_a_stability_certificate(
            frame, [Forward(1.0)],
            _collision_stability_rows(
                [0.02, 0.15, 0.5], instance_id=7), outcomes[1]),
    ]

    consequence.finalize_a_stability_certificates(pending)

    assert [len(call) for call in index.calls] == [14, 7, 7]
    assert all(
        row["exact_contact_identity"] == {
            "confirmed": False,
            "reason": "contact_face_query_failed",
            "instance_id": None,
            "error": "bad witness 99",
        }
        for row in outcomes[0]["shared_oracle_stability"]["rows"])
    assert all(
        row["exact_contact_identity"]["instance_id"] == 7
        for row in outcomes[1]["shared_oracle_stability"]["rows"])
    assert consequence._a_stability_batch_diagnostics() == {
        "combined_query_batches": 1,
        "combined_query_requests": 14,
        "combined_query_fallbacks": 1,
        "legacy_outcome_replays": 2,
        "certificates_finalized": 2,
    }


def test_pose_finalizer_skips_exact_query_for_family_leg():
    class ForbiddenIndex:
        def confirm_contact_instances(self, _requests):
            raise AssertionError("family leg must not query exact A3 identity")

    outcome = {}
    frame = dataclasses.replace(
        make_frame(), semantic_index=ForbiddenIndex())
    pending = [consequence.deferred_a_stability_certificate(
        frame, [Forward(1.0)],
        _collision_stability_rows([0.01, 0.15, 0.5]), outcome,
        exact_contact_identity=False)]

    consequence.finalize_a_stability_certificates(pending)

    assert all(
        row["exact_contact_identity"] is None
        for row in outcome["shared_oracle_stability"]["rows"])


@pytest.mark.parametrize("c1_only, expected", [(False, True), (True, False)])
def test_exact_contact_identity_is_only_computed_for_publishable_a3_routes(
        c1_only, expected):
    assert collection_runtime.exact_contact_identity_required(
        c1_only=c1_only) is expected


def test_validator_rederives_and_rejects_tampered_a_stability_certificate():
    outcome = _outcome(collision=False)
    outcome["oracle_consensus"] = oracle_consensus(
        outcome["physical"], outcome["depth_physical"], 1.0,
        require_contact_instance_witness=True)
    rows = [
        _safe_stability_row(value["id"], {
            "x_m": value["x_m"], "z_m": value["z_m"],
            "yaw_deg": value["yaw_deg"],
        })
        for value in consensus_module.R2R_A_STABILITY_PERTURBATIONS
    ]
    outcome["shared_oracle_stability"] = \
        consensus_module.build_a_stability_certificate(
            outcome["actions"], rows)
    rec = _record(outcome)
    rec["oracle_contract_version"] = record_fields.ORACLE_CONTRACT_VERSION
    rec["source"] = source_provenance("s", dataset="r2r")
    rec["collection_contract"] = record_fields.r2r_v16_collection_contract(
        rec["source"], "main")
    rec["outcomes"][0]["shared_oracle_stability"]["rows"][1][
        "corridor_coverage"] = 0.0

    errors = validate.validate_record_local(
        rec, context=_r2r_v16_validation_context(rec["source"]))

    assert any("A stability certificate" in error for error in errors)


@pytest.mark.parametrize("tamper", ["delete_derived", "extra_field"])
def test_stability_rebuild_rejects_noncanonical_stored_rows(tamper):
    actions = [{"type": "forward", "m": 1.0}]
    rows = [
        _safe_stability_row(value["id"], {
            "x_m": value["x_m"], "z_m": value["z_m"],
            "yaw_deg": value["yaw_deg"],
        })
        for value in consensus_module.R2R_A_STABILITY_PERTURBATIONS
    ]
    stored = consensus_module.build_a_stability_certificate(actions, rows)
    if tamper == "delete_derived":
        stored["rows"][2].pop("full_original_action_index")
    else:
        stored["rows"][2]["unbound_extra"] = "not canonical"

    mismatches = consensus_module.a_stability_certificate_mismatches(
        actions, stored, {
            "physical": rows[0]["physical"],
            "depth_physical": rows[0]["depth_physical"],
            "evidence": {"physical": {"coverage": 1.0}},
        })

    assert "rows" in mismatches


def test_a3_stability_rejects_zero_margin_non_nominal_exact_identity():
    actions = [
        {"type": "forward", "m": 1.0},
        {"type": "turn", "deg": 30.0},
        {"type": "forward", "m": 1.0},
    ]
    rows = []
    for index, perturbation in enumerate(
            consensus_module.R2R_A_STABILITY_PERTURBATIONS):
        rows.append({
            "perturbation_id": perturbation["id"],
            "transform": {key: perturbation[key]
                          for key in ("x_m", "z_m", "yaw_deg")},
            "physical": _physical(True, 1.5, instance_id=7),
            "depth_physical": _depth(True, 1.5, instance_id=7),
            "corridor_coverage": 1.0,
            "exact_contact_identity": {
                "authority": "mp3d_full_face_universe",
                "schema": "mp3d-contact-face-identity.v1",
                "confirmed": True, "reason": "confirmed",
                "instance_id": 7, "category": "chair",
                "streaming_instance_faces_sha256": "d" * 64,
                "contact_face_distance_m": 0.01,
                "runner_up_face_distance_m": 0.01 if index == 4 else 0.5,
                "face_distance_margin_m": 0.0 if index == 4 else 0.49,
                "global_query_protocol":
                    "mp3d-complete-face-instance-universe.v1",
                "semantic_ply_sha256": "a" * 64,
                "global_universe_sha256":
                    semantic_module.complete_face_universe_sha256(
                        global_query_protocol=
                            "mp3d-complete-face-instance-universe.v1",
                        semantic_ply_sha256="a" * 64),
                "global_winner_instance_id": 7,
                "global_runner_up_instance_id": 9,
            },
        })

    certificate = consensus_module.build_a_stability_certificate(actions, rows)

    assert certificate["summary"]["contact_instance_stable"] is False
