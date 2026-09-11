"""Invariant validation for current consequence records."""

from __future__ import annotations

import math
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from pipeline import (
    action_proposal,
    action_sampling,
    actions as A,
    capability_contracts,
    config,
    consensus as consensus_module,
    dataset_contracts,
    future_view_selection,
    gs_semantic,
    objects as object_fields,
    outcome as outcome_fields,
    perception,
    record as R,
    rollout,
    scene_pool,
)
from pipeline import pose_setting
from pipeline.intervention_validation import validate_intervention_groups
from pipeline.source_manifest import RecordValidationContext

_POS = 1e-3
_ANG = 1e-3
def _floor_calibration_errors(fid, rec: dict, sensor: dict) -> List[str]:
    del sensor  # The shared authority reads the same persisted sensor itself.
    try:
        R.authenticated_camera_height_above_visible_floor_m(rec)
    except ValueError as error:
        return [f"[{fid}] {error}"]
    return []

def _pose_close(stored, expected) -> bool:
    if stored is None:
        return False
    return (abs(float(stored["x"]) - expected[0]) <= _POS and
            abs(float(stored["z"]) - expected[1]) <= _POS and
            abs(A.wrap_deg(float(stored["heading_deg"]) - expected[2])) <= _ANG)


_SUPPORTED_ACTION_POLICIES = frozenset({
    action_proposal.DEPTH_CONDITIONED_POLICY,
    action_proposal.DEPTH_CONDITIONED_POLICY_V3,
    action_proposal.DEPTH_CONDITIONED_POLICY_V4,
    action_proposal.DEPTH_CONDITIONED_POLICY_V5,
    action_proposal.EXPLICIT_ACTION_FILE_POLICY,
})
# Only a depth-conditioned run materialises programs from a pose's own depth, so
# only it may carry the proposal fields. A file-driven run that carried them
# would be claiming provenance it never produced.
_PROPOSAL_ONLY_FIELDS = ("proposal_provenance", "materialized_action_bank",
                         "materialized_action_bank_sha256")
# Forcing frozen members through selection reads the labels the plan was built
# from. Only the augmented policy may do it, and it must say so: a run that
# declares plain depth conditioning while carrying these fields would be
# presenting a label-aware selection as a label-blind one.
def _proposal_shape_errors(prefix, selection, policy) -> List[str]:
    present = [name for name in _PROPOSAL_ONLY_FIELDS if name in selection]
    if policy == action_proposal.EXPLICIT_ACTION_FILE_POLICY:
        return ([f"{prefix} file-driven runs must not carry depth proposal "
                 f"fields: {present}"] if present else [])
    missing = [name for name in _PROPOSAL_ONLY_FIELDS if name not in selection]
    if missing:
        return [f"{prefix} depth proposal fields are required: {missing}"]
    errors = []
    bank = selection.get("materialized_action_bank")
    try:
        recomputed = action_proposal.action_bank_manifest_sha256(bank)
    except (KeyError, TypeError, ValueError) as error:
        return [f"{prefix} materialized action bank is unreadable: {error}"]
    if recomputed != selection.get("materialized_action_bank_sha256"):
        errors.append(
            f"{prefix} materialized action bank digest does not match its bank")
    bank_by_tag = {entry.get("tag"): entry for entry in bank}
    banked = set(bank_by_tag)
    proposed = selection.get("proposal_provenance") or {}
    if not set(proposed).issubset(banked):
        errors.append(
            f"{prefix} proposal provenance names programs outside the bank")
    for tag, provenance in proposed.items():
        bank_row = bank_by_tag.get(tag)
        if not isinstance(provenance, dict) or not isinstance(bank_row, dict):
            errors.append(
                f"{prefix} proposal provenance for {tag!r} is invalid")
            continue
        if (provenance.get("variant") != bank_row.get("variant") or
                provenance.get("template_id") !=
                bank_row.get("template_id")):
            errors.append(
                f"{prefix} proposal provenance for {tag!r} does not match "
                "its action bank row")
    if not set(selection.get("action_group_ids") or []).issubset(banked):
        errors.append(
            f"{prefix} selected action groups are outside the proposal bank")
    return errors


def _validate_balanced_selection(
        rec: dict, expected_setting_policy: Optional[str] = None,
        expected_action_policy: Optional[str] = None) -> List[str]:
    selection = rec.get("selection") or {}
    policy = selection.get("policy")
    fid = rec.get("frame_id", "?")
    prefix = f"[{fid}:selection]"
    # Run first and unconditionally: a record that omits `selection` entirely
    # must not thereby escape the sampling policies its run committed to.
    setting_errors = pose_setting.selection_errors(
        prefix, selection, rec.get("outcomes", []),
        sorted(round(float(value), 6)
               for value in selection.get("required_radii_m", [])),
        expected_setting_policy)
    if expected_action_policy is not None and policy != expected_action_policy:
        return setting_errors + [
            f"{prefix} action sampling policy is {policy!r} but the run "
            f"declares {expected_action_policy!r}"]
    if policy is None:
        return setting_errors
    forbidden = {
        "formal_action_group_ids",
        "groups_per_length",
        "per_length_label_counts",
        "global_label_counts",
        "global_label_policy",
    }
    if (policy == "strict_full_depth_formal_six" or
            selection.get("global_label_policy") ==
            "exact_3_safe_3_collision" or
            forbidden.intersection(selection)):
        return [f"{prefix} fixed per-pose label contract is forbidden"]
    if policy not in _SUPPORTED_ACTION_POLICIES:
        return [f"{prefix} unsupported action sampling policy"]
    setting_errors.extend(_proposal_shape_errors(prefix, selection, policy))
    errors = []
    outcomes = rec.get("outcomes", [])
    groups = defaultdict(list)
    for outcome in outcomes:
        group_id = outcome.get("action_group_id")
        if group_id is not None:
            groups[group_id].append(outcome)
    expected_ids = selection.get("action_group_ids", [])
    if set(groups) != set(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        errors.append(f"{prefix} stored action groups do not match selection metadata")
    required_radii = sorted(round(float(value), 6) for value in
                            selection.get("required_radii_m", []))
    errors.extend(setting_errors)
    group_length_by_id = {}
    group_actions_by_id = {}
    group_label_by_id = {}
    for group_id, members in groups.items():
        lengths = {int(member.get("seq_len", -1)) for member in members}
        labels = {member.get("action_group_label") for member in members}
        action_keys = {repr(member.get("actions", [])) for member in members}
        if len(lengths) != 1 or len(labels) != 1 or len(action_keys) != 1:
            errors.append(f"{prefix} action group {group_id} has inconsistent siblings")
            continue
        length, label = next(iter(lengths)), next(iter(labels))
        group_length_by_id[group_id] = length
        try:
            parsed = A.parse_actions(members[0].get("actions", []))
            A.validate_physics_actions(parsed)
            if len(parsed) != length:
                raise ValueError("stored sequence length differs from its actions")
            group_actions_by_id[group_id] = parsed
        except ValueError:
            errors.append(
                f"{prefix} action group {group_id} violates the action grammar")
        metadata_label = selection.get("action_group_labels", {}).get(group_id)
        allowed_labels = {"safe", "collision"}
        if label not in allowed_labels or metadata_label != label:
            errors.append(f"{prefix} action group {group_id} label metadata is invalid")
        group_label_by_id[group_id] = label
        radii = sorted(round(float(member.get("body", {}).get("radius_m", -1)), 6)
                       for member in members)
        if required_radii and radii != required_radii:
            errors.append(f"{prefix} action group {group_id} does not cover all radii")
        collisions = {
            member.get("physical", {}).get("collision")
            for member in members}
        expected_collision = label == "collision"
        if collisions != {expected_collision}:
            errors.append(
                f"{prefix} action group {group_id} is not radius-consistent")
    try:
        candidate_budget = int(selection.get("candidate_budget"))
    except (TypeError, ValueError):
        candidate_budget = -1
    if (not config.ACTION_CANDIDATE_MIN_PER_POSE <= candidate_budget <=
            config.ACTION_CANDIDATE_MAX_PER_POSE or
            len(expected_ids) > candidate_budget):
        errors.append(f"{prefix} natural candidate budget is invalid")
    observed_lengths = sorted(set(group_length_by_id.values()))
    if selection.get("observed_lengths") != observed_lengths:
        errors.append(f"{prefix} observed action lengths do not match candidates")
    counted_labels = ("safe", "collision")
    actual_label_counts = {
        label: list(group_label_by_id.values()).count(label)
        for label in counted_labels}
    recorded_label_counts = selection.get("observed_label_counts") or {}
    if any(int(recorded_label_counts.get(label, 0)) != count
           for label, count in actual_label_counts.items()):
        errors.append(f"{prefix} observed label counts do not match candidates")
    used_pair_members = set()
    for unit in selection.get("matched_action_units") or []:
        safe_id = str(unit.get("safe_group_id") or "")
        collision_id = str(unit.get("collision_group_id") or "")
        pair_id = unit.get("pair_id")
        if (not safe_id or not collision_id or safe_id == collision_id or
                safe_id in used_pair_members or
                collision_id in used_pair_members):
            errors.append(f"{prefix} matched action membership is invalid")
            continue
        used_pair_members.update({safe_id, collision_id})
        if (group_label_by_id.get(safe_id) != "safe" or
                group_label_by_id.get(collision_id) != "collision"):
            errors.append(f"{prefix} matched action labels are invalid")
            continue
        if safe_id not in group_actions_by_id or collision_id not in group_actions_by_id:
            errors.append(f"{prefix} matched action group is missing")
            continue
        safe_key = action_sampling.action_match_key(group_actions_by_id[safe_id])
        collision_key = action_sampling.action_match_key(
            group_actions_by_id[collision_id])
        if safe_key != collision_key or unit.get("match_key") != safe_key:
            errors.append(f"{prefix} matched action key is invalid")
        expected_pair_id = action_sampling.expected_pair_id(
            safe_key, safe_id, collision_id)
        if (not isinstance(pair_id, str) or pair_id != expected_pair_id):
            errors.append(f"{prefix} matched action pair id is invalid")
    used_safe_pair_members = set()
    for unit in selection.get("matched_safe_action_units") or []:
        left_id = str(unit.get("left_group_id") or "")
        right_id = str(unit.get("right_group_id") or "")
        pair_id = unit.get("pair_id")
        if (
            unit.get("label") != "safe" or
            not left_id or not right_id or left_id == right_id or
            left_id in used_safe_pair_members or
            right_id in used_safe_pair_members
        ):
            errors.append(
                f"{prefix} matched safe action membership is invalid")
            continue
        used_safe_pair_members.update({left_id, right_id})
        if (
            group_label_by_id.get(left_id) != "safe" or
            group_label_by_id.get(right_id) != "safe"
        ):
            errors.append(f"{prefix} matched safe action labels are invalid")
            continue
        if (
            left_id not in group_actions_by_id or
            right_id not in group_actions_by_id
        ):
            errors.append(f"{prefix} matched safe action group is missing")
            continue
        left_key = action_sampling.action_match_key(
            group_actions_by_id[left_id])
        right_key = action_sampling.action_match_key(
            group_actions_by_id[right_id])
        if left_key != right_key or unit.get("match_key") != left_key:
            errors.append(f"{prefix} matched safe action key is invalid")
        expected_pair_id = action_sampling.expected_same_label_pair_id(
            left_key, left_id, right_id, label="safe")
        if not isinstance(pair_id, str) or pair_id != expected_pair_id:
            errors.append(f"{prefix} matched safe action pair id is invalid")
    if any("pair_id" in outcome for outcome in outcomes):
        errors.append(f"{prefix} pair ids must remain private selection metadata")
    return errors


def _validate_radius_monotonicity(rec: dict) -> List[str]:
    fid = rec.get("frame_id", "?")
    groups = defaultdict(list)
    for outcome in rec.get("outcomes", []):
        key = json.dumps(outcome.get("actions", []), sort_keys=True)
        groups[key].append(outcome)
    errors = []
    for members in groups.values():
        by_radius = {}
        for outcome in members:
            radius = round(float(outcome.get("body", {}).get("radius_m", -1)), 6)
            by_radius.setdefault(radius, outcome)
        ordered = sorted(by_radius.items())
        if len(ordered) < 2:
            continue
        collision_seen = False
        previous_clearance = None
        for radius, outcome in ordered:
            physical = outcome.get("physical", {})
            collision = physical.get("collision")
            if collision is True:
                collision_seen = True
            elif collision_seen and collision is False:
                errors.append(
                    f"[{fid}] body-radius collision monotonicity violated "
                    f"at radius {radius:g}")
                break
            clearance = physical.get("minimum_clearance_m")
            if clearance is not None:
                clearance = float(clearance)
                if (previous_clearance is not None and
                        clearance > previous_clearance +
                        config.BENCH_RADIUS_CLEARANCE_MONOTONIC_TOL_M):
                    errors.append(
                        f"[{fid}] body-radius clearance monotonicity violated "
                        f"at radius {radius:g}")
                    break
                previous_clearance = clearance
    return errors


def _validated_r2r_v16_collection_contract(
        rec: dict, context: RecordValidationContext,
        errors: List[str]) -> bool:
    """Validate record evidence against a trusted external route decision."""
    if context.route == "legacy":
        return False
    fid = rec.get("frame_id", "?")
    scene_id = str(rec.get("scene_id") or "")
    expected_entry = context.expected_collection_contracts.get(scene_id)
    if expected_entry is None:
        errors.append(
            f"[{fid}] external validation context has no collection contract "
            f"for scene {scene_id!r}")
        return True
    contract = rec.get("collection_contract")
    if contract is None:
        errors.append(f"[{fid}] required collection contract is missing")
        return True
    if not isinstance(contract, dict):
        errors.append(f"[{fid}] collection contract must be an object")
        return True
    expected = (
        expected_entry.get(contract.get("version"))
        if context.route == "b1k_v16_registered" else expected_entry)
    if not isinstance(expected, dict):
        errors.append(
            f"[{fid}] collection contract version is not registered")
        return True
    if contract != expected:
        errors.append(
            f"[{fid}] collection contract does not match external validation "
            "context")
    try:
        source_bound = R.collection_contract(
            rec.get("source") or {}, expected.get("collection_mode"),
            record_schema_version=context.expected_schema_version,
            contract_version=(
                expected.get("version")
                if context.route == "b1k_v16_registered" else None))
    except (TypeError, ValueError) as error:
        errors.append(f"[{fid}] collection contract is invalid: {error}")
        return True
    if source_bound != expected:
        errors.append(
            f"[{fid}] external collection contract source binding disagrees")
    return True


def _validate_declared_collection_contract(
        rec: dict, errors: List[str]) -> bool:
    """Validate a record's own contract without treating it as authority."""
    contract = rec.get("collection_contract")
    if contract is None:
        return False
    fid = rec.get("frame_id", "?")
    if not isinstance(contract, dict):
        errors.append(f"[{fid}] collection contract must be an object")
        return False
    try:
        rebuilt = R.collection_contract(
            rec.get("source") or {}, contract.get("collection_mode"),
            record_schema_version=rec.get("schema_version"),
            contract_version=(
                contract.get("version")
                if (rec.get("source") or {}).get("source_dataset") == "b1k"
                else None))
    except (TypeError, ValueError) as error:
        errors.append(f"[{fid}] collection contract is invalid: {error}")
        return False
    if rebuilt != contract:
        errors.append(
            f"[{fid}] collection contract source binding disagrees")
        return False
    return True


def _validate_record_local_context(
        rec: dict, context: Optional[RecordValidationContext],
        errors: List[str]) -> tuple:
    fid = rec.get("frame_id", "?")
    source = rec.get("source")
    if source is not None:
        dataset = str(source.get("source_dataset") or "")
        try:
            source_contract = dataset_contracts.dataset_source_contract(
                dataset)
        except ValueError:
            source_contract = None
            errors.append(f"[{fid}] unsupported source dataset {dataset!r}")
        if str(source.get("scene_id") or "") != str(rec.get("scene_id") or ""):
            errors.append(f"[{fid}] source scene_id does not match record")
        if source.get("official_split") not in \
                dataset_contracts.OFFICIAL_SOURCE_SPLITS:
            errors.append(f"[{fid}] source official split is unsupported")
        if (dataset == "b1k" and
                source.get("split_authority") != "project_defined"):
            errors.append(
                f"[{fid}] B1K source split authority is not project-defined")
        if dataset == "b1k" and source.get("official_split") != "train":
            errors.append(f"[{fid}] B1K source split is not train")
        semantic_format = str(source.get("semantic_format") or "")
        semantic_format_matches = (
            source_contract is not None and
            semantic_format == source_contract.semantic_format)
        if source_contract is not None and not semantic_format_matches:
            errors.append(
                f"[{fid}] source semantic format {semantic_format!r} does not "
                f"match {dataset!r}")
        manifest_hash = str(source.get("source_manifest_sha256") or "")
        if (len(manifest_hash) != 64 or
                any(char not in "0123456789abcdef" for char in manifest_hash)):
            errors.append(f"[{fid}] source manifest sha256 is invalid")
        if source_contract is None or semantic_format_matches:
            try:
                scene_pool.validate_source_asset_provenance(source)
            except scene_pool.SceneCatalogError as error:
                errors.append(f"[{fid}] {error}")
    if context is not None and context.route == "gs_v18_registered":
        strict_shared_oracle = True
    else:
        strict_shared_oracle = (
            _validated_r2r_v16_collection_contract(rec, context, errors)
            if context is not None else
            _validate_declared_collection_contract(rec, errors))
    sensor = rec.get("sensor", {})
    errors.extend(_floor_calibration_errors(fid, rec, sensor))
    if {"hfov_deg", "vfov_deg", "resolution"} <= sensor.keys():
        expected_resolution = list(config.resolution())
        if list(sensor["resolution"]) != expected_resolution:
            errors.append(f"[{fid}] sensor resolution does not match FOV/focal")
        expected_k = config.intrinsics(
            float(sensor["hfov_deg"]), float(sensor["vfov_deg"]))
        if (abs(float(sensor.get("focal_x_px", -1.0)) - float(expected_k[0, 0])) > _POS or
                abs(float(sensor.get("focal_y_px", -1.0)) -
                    float(expected_k[1, 1])) > _POS):
            errors.append(f"[{fid}] sensor intrinsics do not match dynamic FOV")
        try:
            config.render_resolution(
                float(sensor["hfov_deg"]), float(sensor["vfov_deg"]))
        except ValueError:
            errors.append(f"[{fid}] sensor profile violates formal FOV calibration")
    expected_substrate = {
        "base_rollout_key_version": R.BASE_ROLLOUT_KEY_VERSION,
    }
    if rec.get("substrate") != expected_substrate:
        errors.append(f"[{fid}] substrate contract is missing or stale")
    return sensor, strict_shared_oracle


def _validate_outcome_identity(
        rec: dict, outcome: dict, errors: List[str]) -> None:
    fid = rec.get("frame_id", "?")
    oid = outcome.get("outcome_id", "?")
    prefix = f"[{fid}:{oid}]"
    try:
        expected_regime = outcome_fields.derive_execution_regime(outcome)
    except (TypeError, ValueError) as error:
        errors.append(
            f"{prefix} execution regime cannot be derived: {error}")
    else:
        stored_regime = (
            (outcome.get("execution") or {}).get("execution_regime"))
        if stored_regime != expected_regime:
            errors.append(
                f"{prefix} execution regime disagrees with physical rollout")
    try:
        expected_base_key = R.stored_base_rollout_key(rec, outcome)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        errors.append(f"{prefix} base rollout key cannot be derived: {error}")
        expected_base_key = None
    if (expected_base_key is not None and
            outcome.get("base_rollout_key") != expected_base_key):
        errors.append(
            f"{prefix} base rollout key disagrees with record inputs")


_CONTACT_SOURCE_BY_AUTHORITY = {
    # Source-bound mesh contacts name their authority; a navmesh contact has
    # no positive source and is certified by its configuration boundary.
    "b1k_geometry": "b1k_geometry",
    "gs_collision_mesh": "gs_collision_mesh",
    "navmesh": None,
}


def _authority_label(expected_authority) -> str:
    if expected_authority == "b1k_geometry":
        return "B1K"
    return "GS" if expected_authority == "gs_collision_mesh" else "navmesh"


def _collision_source_matches(
        expected_authority, collision, collision_source) -> bool:
    """Hold both oracles to one taxonomy of collision provenance."""
    if expected_authority not in _CONTACT_SOURCE_BY_AUTHORITY:
        return True
    if collision is True:
        return collision_source == _CONTACT_SOURCE_BY_AUTHORITY[
            expected_authority]
    if collision is False:
        return collision_source is None
    return collision_source in outcome_fields.EXCLUDED_GEOMETRY_SOURCES


def _a_stability_validation_errors(
        outcome, actions, prefix, *,
        authority_binding: dataset_contracts.AuthorityBinding | None = None,
        require_contact_instance_witness: bool | None = None,
        ) -> list[str]:
    stability = outcome.get("shared_oracle_stability")
    if not isinstance(stability, dict):
        return [f"{prefix} A stability certificate is missing"]
    if stability.get("version") == "nominal-oracle.v1":
        rows = stability.get("rows") or []
        summary = stability.get("summary") or {}
        physical = outcome.get("physical") or {}
        depth = outcome.get("depth_physical") or {}
        consensus = outcome.get("oracle_consensus") or {}
        contact_index = physical.get("contact_action_index")
        expected_index = (
            None if contact_index is None else int(contact_index) + 1)
        full_id = consensus.get("full_contact_instance_id")
        depth_id = consensus.get("depth_contact_instance_id")
        expected_contact_id = full_id if full_id == depth_id else None
        body = {key: value for key, value in stability.items()
                if key != "sha256"}
        valid = (
            len(rows) == 1 and
            rows[0].get("perturbation_id") == "nominal" and
            rows[0].get("transform") == {
                "x_m": 0.0, "z_m": 0.0, "yaw_deg": 0.0} and
            rows[0].get("physical") == physical and
            rows[0].get("depth_physical") == depth and
            rows[0].get("consensus") == consensus and
            summary.get("evaluation") == "nominal" and
            summary.get("collision") is physical.get("collision") and
            summary.get("original_action_index") == expected_index and
            summary.get("contact_instance_id") == expected_contact_id and
            stability.get("sha256") == R.canonical_atom_sha256(body))
        return [] if valid else [
            f"{prefix} nominal A certificate differs from its outcome"]
    mismatches = consensus_module.a_stability_certificate_mismatches(
        actions, stability, outcome,
        authority_binding=authority_binding,
        require_contact_instance_witness=require_contact_instance_witness)
    return ([f"{prefix} A stability certificate differs from reconstruction: "
             f"{', '.join(mismatches)}"] if mismatches else [])


def _validate_navmesh_contact_surface(
        prefix: str, rec: dict, body: dict, physical: dict,
        expected_authority, errors: List[str]) -> None:
    """Cross-check a navmesh collision against its contact surface.

    Extracted verbatim from _validate_outcome_execution; the block is
    unchanged so the error inventory and its order are identical.
    """
    if (expected_authority == "navmesh" and
            physical.get("collision") is True):
        contact = physical.get("contact") or {}
        center = contact.get("center_local")
        world_point = contact.get("world_point")
        boundary_point = contact.get(
            "configuration_boundary_world_point")
        pose = rec.get("pose") or {}
        if (not isinstance(center, list) or len(center) != 2 or
                not isinstance(boundary_point, list) or
                len(boundary_point) != 3):
            errors.append(
                f"{prefix} navmesh configuration boundary point is missing")
        else:
            try:
                local_boundary = perception.local_from_world(
                    [boundary_point],
                    pose["position"], float(pose["yaw_rad"]))[0]
                boundary_delta = (
                    float(local_boundary[0]) - float(center[0]),
                    float(local_boundary[2]) - float(center[1]),
                )
                boundary_distance = math.hypot(*boundary_delta)
                radius = float(body["radius_m"])
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(
                    f"{prefix} navmesh configuration boundary point is invalid")
            else:
                boundary_is_near = (
                    math.isfinite(boundary_distance) and
                    1e-9 < boundary_distance <=
                    config.CONTACT_NAVMESH_BOUNDARY_MAX_M)
                if not boundary_is_near:
                    if (world_point is not None or
                            contact.get("surface_protocol") is not None):
                        errors.append(
                            f"{prefix} distant navmesh configuration boundary "
                            "must not expose a contact surface")
                else:
                    if (contact.get("surface_protocol") !=
                            "radius_extrapolated_navmesh_boundary_v2"):
                        errors.append(
                            f"{prefix} navmesh contact surface protocol is "
                            "missing")
                    if (not isinstance(world_point, list) or
                            len(world_point) != 3):
                        errors.append(
                            f"{prefix} navmesh contact surface point is missing")
                    else:
                        try:
                            local_point = perception.local_from_world(
                                [world_point],
                                pose["position"], float(pose["yaw_rad"]))[0]
                            surface_delta = (
                                float(local_point[0]) - float(center[0]),
                                float(local_point[2]) - float(center[1]),
                            )
                            distance = math.hypot(*surface_delta)
                        except (KeyError, TypeError, ValueError, OverflowError):
                            errors.append(
                                f"{prefix} navmesh contact surface point is "
                                "invalid")
                        else:
                            # The centre has already crossed the tolerance
                            # contour by ``boundary_distance``; the disc touches
                            # the obstacle at the contour, so the surface sits
                            # that much short of a full radius.
                            reach = radius - boundary_distance
                            if (not math.isfinite(distance) or
                                    abs(distance - reach) >
                                    config.CONTACT_SURFACE_RADIUS_TOL_M):
                                errors.append(
                                    f"{prefix} navmesh contact surface radius "
                                    "does not match the body radius")
                            expected_direction = (
                                -boundary_delta[0] / boundary_distance,
                                -boundary_delta[1] / boundary_distance,
                            )
                            direction_error = math.hypot(
                                surface_delta[0] -
                                expected_direction[0] * reach,
                                surface_delta[1] -
                                expected_direction[1] * reach,
                            )
                            if (not all(
                                    math.isfinite(value)
                                    for value in surface_delta) or
                                    direction_error >
                                    config.CONTACT_SURFACE_RADIUS_TOL_M):
                                errors.append(
                                    f"{prefix} navmesh contact surface "
                                    "direction does not oppose the "
                                    "configuration boundary")


def _validate_shared_oracle_certificate(
        prefix: str, outcome: dict, actions, consensus: dict,
        strict_shared_oracle: bool, errors: List[str], *,
        authority_binding: dataset_contracts.AuthorityBinding | None = None,
        source_dataset: str = "r2r",
        require_contact_instance_witness: bool = True) -> None:
    """Rebuild the shared full/depth certificate and compare it.

    Extracted verbatim from _validate_outcome_execution; the block is
    unchanged so the error inventory and its order are identical.
    """
    if strict_shared_oracle:
        if require_contact_instance_witness:
            if consensus.get("contact_instance_witness_required") is not True:
                errors.append(
                    f"{prefix} strict shared-oracle consensus marker is "
                    "missing")
        elif "contact_instance_witness_required" in consensus:
            errors.append(
                f"{prefix} geometry-only consensus claims an A3 witness")
        try:
            rebuilt_shared = consensus_module.rebuild_outcome_consensus(
                outcome, require_contact_instance_witness=
                    require_contact_instance_witness)
        except (TypeError, ValueError) as error:
            errors.append(
                f"{prefix} reconstructed shared-oracle consensus is "
                f"unavailable: {error}")
        else:
            if rebuilt_shared.get("accepted") is not True:
                errors.append(
                    f"{prefix} reconstructed shared-oracle consensus "
                    f"rejected: {rebuilt_shared.get('reason')}")
            shared_mismatches = consensus_module.stored_consensus_mismatches(
                consensus, rebuilt_shared, require_complete=True)
            if shared_mismatches:
                errors.append(
                    f"{prefix} stored shared-oracle consensus differs from "
                    f"reconstruction: {', '.join(shared_mismatches)}")
        errors.extend(_a_stability_validation_errors(
            outcome, actions, prefix,
            authority_binding=authority_binding,
            require_contact_instance_witness=
                require_contact_instance_witness))
    elif consensus.get("contact_instance_witness_required") is True:
        if source_dataset == "r2r":
            contract_label = "R2R v16"
        elif source_dataset == "b1k":
            contract_label = "B1K v11"
        else:
            errors.append(
                f"{prefix} contact instance witness has unsupported source "
                f"dataset {source_dataset!r}")
            return
        errors.append(
            f"{prefix} contact instance witness lacks a validated "
            f"{contract_label} collection contract")


def _validate_near_field_hidden_collision(
        prefix: str, rec: dict, outcome: dict, sensor: dict,
        body: dict, physical: dict, errors: List[str]) -> None:
    """Refuse a collision hidden inside the certified near field.

    Extracted verbatim from _validate_outcome_execution; the block is
    unchanged so the error inventory and its order are identical.
    """
    # The public prompt certifies that the floor-blind near field contains
    # no hidden collision. A selective-evidence probe may relax coverage
    # farther away, but it may not use that relaxation to hide an obstacle
    # inside the certified strip.
    contact_arc = physical.get("first_contact_arc_m")
    if physical.get("collision") is True and contact_arc is not None:
        near_field = rollout.certified_near_field_distance_m(
            camera_height_above_floor_m=float(
                rec["camera_height_above_visible_floor_m"]),
            hfov_deg=float(sensor["hfov_deg"]),
            vfov_deg=float(sensor["vfov_deg"]),
            radius=float(body.get("radius_m", 0.0)),
        )
        if float(contact_arc) <= near_field + _POS:
            depth = outcome.get("depth_physical", {}) or {}
            depth_arc = depth.get("first_contact_arc_m")
            confirmed = (
                depth.get("collision") is True and depth_arc is not None and
                abs(float(depth_arc) - float(contact_arc)) <=
                config.ORACLE_CONTACT_TOL_M + _POS
            )
            if not confirmed:
                errors.append(
                    f"{prefix} hidden collision lies inside certified near field")


def _validate_checkpoint_trace(
        prefix: str, execution: dict, actions, checkpoints,
        errors: List[str]) -> None:
    """Re-derive nominal pose and checkpoint schedule from the actions.

    Extracted verbatim from _validate_outcome_execution; the block is
    unchanged so the error inventory and its order are identical.
    """
    nominal_pose = execution.get("nominal_pose")
    if nominal_pose is not None and not _pose_close(nominal_pose, A.pose_at_progress(actions, 1.0)):
        errors.append(f"{prefix} nominal pose does not match actions")
    if checkpoints:
        requested = [float(c["requested_progress"]) for c in checkpoints]
        if requested != list(config.CHECKPOINT_PROGRESS):
            errors.append(f"{prefix} checkpoint schedule mismatch")
        previous = -1.0
        for checkpoint in checkpoints:
            req = float(checkpoint["requested_progress"])
            realized = float(checkpoint["realized_progress"])
            if realized < previous - _POS or realized > req + _POS:
                errors.append(f"{prefix} checkpoint progress is invalid")
            previous = realized
            if not _pose_close(checkpoint.get("pose"), A.pose_at_progress(actions, realized)):
                errors.append(f"{prefix} checkpoint pose does not match realized progress")
            arc = checkpoint.get("arc_m")
            expected_arc = A.arc_at_progress(actions, realized)
            if arc is not None and abs(float(arc) - expected_arc) > _POS:
                errors.append(f"{prefix} checkpoint arc does not match realized progress")


def _validate_collision_execution_consistency(
        prefix: str, physical: dict, execution: dict, actions,
        checkpoints, errors: List[str]) -> None:
    """Re-derive the execution summary and cross-check stored fields.

    Extracted verbatim from _validate_outcome_execution; the block is
    unchanged so the error inventory and its order are identical.
    """
    collision = physical.get("collision")
    stop_progress = execution.get("animation_stop_time_fraction")
    if collision is True:
        if execution.get("completed") is not False or execution.get("stop_reason") != "collision":
            errors.append(f"{prefix} collision execution state is inconsistent")
        if physical.get("first_contact_arc_m") is None or stop_progress is None:
            errors.append(f"{prefix} collision lacks stop location")
        contact_arc = physical.get("first_contact_arc_m")
        if contact_arc is not None:
            try:
                contact_location = A.forward_leg_location(
                    actions, float(contact_arc))
            except (TypeError, ValueError) as error:
                errors.append(
                    f"{prefix} contact arc cannot be located: {error}")
            else:
                if (physical.get("contact_action_index") !=
                        contact_location.action_index):
                    errors.append(
                        f"{prefix} contact action index does not match "
                        "contact arc")
                stored_local = physical.get("contact_action_local_arc_m")
                if (stored_local is None or abs(
                        float(stored_local) -
                        contact_location.distance_into_leg_m) > _POS):
                    errors.append(
                        f"{prefix} contact action local arc does not match "
                        "contact arc")
        if checkpoints and stop_progress is not None:
            stop_pose = A.pose_at_progress(actions, float(stop_progress))
            for checkpoint in checkpoints:
                if checkpoint["requested_progress"] > stop_progress + _POS and \
                        not _pose_close(checkpoint["pose"], stop_pose):
                    errors.append(f"{prefix} checkpoint pose moves after collision")
    elif collision is False and execution.get("completed") is not True:
        errors.append(f"{prefix} collision-free execution is not completed")
    # Independently re-derive the execution summary from the stored stop arc
    # and cross-check the persisted turn / forward / pose fields. This never
    # trusts a stored action index, so a record produced by the pre-fix
    # rollout (which dropped trailing Turns from executed_turn_deg) fails
    # here even though its raw checkpoint poses validate.
    if isinstance(collision, bool) and execution:
        stored_stop_arc = execution.get("stop_arc_m")
        if collision is True and stored_stop_arc is None:
            errors.append(f"{prefix} collision execution lacks stop_arc_m")
        else:
            summary = rollout.summarize_execution(
                actions, collision=collision, stop_arc_m=stored_stop_arc)
            expected_forward = (
                float(stored_stop_arc) if collision else
                A.total_forward_m(actions))
            stored_forward_total = execution.get("executed_forward_m")
            if (stored_forward_total is None or abs(
                    float(stored_forward_total) - expected_forward) > _POS):
                errors.append(
                    f"{prefix} executed_forward_m does not match "
                    f"{'stop arc' if collision else 'complete action'}")
            stored_turn = execution.get("executed_turn_deg")
            if stored_turn is not None and abs(
                    float(stored_turn) - summary["executed_turn_deg"]) > _ANG:
                errors.append(
                    f"{prefix} executed_turn_deg does not match re-rolled actions")
            stored_forward = execution.get("executed_forward_after_turn_m")
            if stored_forward is not None and abs(
                    float(stored_forward) -
                    summary["executed_forward_after_turn_m"]) > _POS:
                errors.append(
                    f"{prefix} executed_forward_after_turn_m does not match "
                    f"re-rolled actions")
            if (execution.get("realized_pose") is not None and
                    not _pose_close(execution.get("realized_pose"),
                                    summary["realized_pose"])):
                errors.append(
                    f"{prefix} realized pose does not match re-rolled actions")


def _validate_outcome_execution(
        rec: dict, outcome: dict, sensor: dict,
        strict_shared_oracle: bool, errors: List[str], *,
        require_contact_instance_witness: bool | None = None) -> None:
    prefix = f"[{rec.get('frame_id', '?')}:{outcome.get('outcome_id', '?')}]"
    body = outcome.get("body", {})
    if (body.get("shape") != "disc" or
            set(body) != {"shape", "radius_m"}):
        errors.append(f"{prefix} body must be a radius-only disc")
    actions = A.parse_actions(outcome.get("actions", []))
    execution = outcome.get("execution", {})
    physical = outcome.get("physical", {})
    source_dataset = str(
        (rec.get("source") or {}).get("source_dataset") or "")
    expected_authority = {
        "b1k": "b1k_geometry",
        "r2r": "navmesh",
        "gs": "gs_collision_mesh",
    }.get(source_dataset)
    if (expected_authority is not None and
            physical.get("authority") != expected_authority):
        errors.append(
            f"{prefix} physical authority {physical.get('authority')!r} "
            f"does not match source dataset {source_dataset!r}")
    collision_source = (
        physical.get("collision_source") or
        physical.get("contact_source") or
        (physical.get("contact") or {}).get("source"))
    if not _collision_source_matches(
            expected_authority, physical.get("collision"), collision_source):
        errors.append(
            f"{prefix} {_authority_label(expected_authority)} geometry source "
            "does not match its physical collision state")
    _validate_navmesh_contact_surface(
        prefix, rec, body, physical, expected_authority, errors)
    if ((outcome.get("depth_physical") or {}).get("authority") != "depth"):
        errors.append(f"{prefix} visible-depth authority must be 'depth'")
    checkpoints = outcome.get("checkpoints", [])
    consensus = outcome.get("oracle_consensus", {})
    authority_binding = None
    witness_required = (
        bool(strict_shared_oracle)
        if require_contact_instance_witness is None else
        bool(require_contact_instance_witness))
    if strict_shared_oracle and witness_required:
        try:
            authority_binding = R.authority_binding(rec.get("source"))
        except (KeyError, StopIteration, TypeError, ValueError):
            pass
    _validate_shared_oracle_certificate(
        prefix, outcome, actions, consensus, strict_shared_oracle, errors,
        authority_binding=authority_binding,
        source_dataset=source_dataset,
        require_contact_instance_witness=witness_required)
    evidence_protocol = outcome.get("evidence", {}).get(
        "physical", {}).get("coverage_protocol")
    if evidence_protocol not in rollout.SUPPORTED_EVIDENCE_PROTOCOLS:
        errors.append(
            f"{prefix} unsupported physical evidence protocol "
            f"{evidence_protocol!r}")
    if consensus.get("accepted") is not True:
        errors.append(
            f"{prefix} non-consensus outcome lacks accepted consensus")
    _validate_near_field_hidden_collision(
        prefix, rec, outcome, sensor, body, physical, errors)
    _validate_checkpoint_trace(
        prefix, execution, actions, checkpoints, errors)
    _validate_collision_execution_consistency(
        prefix, physical, execution, actions, checkpoints, errors)


_RETIRED_RECORD_FIELDS = frozenset({
    "review_evidence", "target_reference_sets", "targets",
})
_RETIRED_OUTCOME_FIELDS = frozenset({
    "future_state", "object_consequences", "probe",
    "start_terminal_options", "target_projection_keys",
    "target_projections", "terminal_options",
})
_RETIRED_EVIDENCE_FIELDS = frozenset({"goal", "terminal_options"})
# v10 named the per-run global bank digest `action_bank_sha256`. v11 replaces it
# with a per-pose digest over a materialised bank, which is a different quantity
# under the same name -- so the old name is rejected rather than reinterpreted.
_RETIRED_SELECTION_FIELDS = frozenset({"action_bank_sha256"})


def _validate_v11_shape(rec: dict) -> List[str]:
    """Reject retired branches instead of silently accepting an adapter."""
    fid = rec.get("frame_id", "?")
    errors = []
    record_fields = sorted(_RETIRED_RECORD_FIELDS.intersection(rec))
    if record_fields:
        errors.append(
            f"[{fid}] retired v9 record fields are forbidden: {record_fields}")
    selection_fields = sorted(
        _RETIRED_SELECTION_FIELDS.intersection(rec.get("selection") or {}))
    if selection_fields:
        errors.append(
            f"[{fid}] retired v10 selection fields are forbidden: "
            f"{selection_fields}")
    for outcome in rec.get("outcomes", []):
        prefix = f"[{fid}:{outcome.get('outcome_id', '?')}]"
        outcome_fields_present = sorted(
            _RETIRED_OUTCOME_FIELDS.intersection(outcome))
        if outcome_fields_present:
            errors.append(
                f"{prefix} retired v9 outcome fields are forbidden: "
                f"{outcome_fields_present}")
        evidence = outcome.get("evidence") or {}
        evidence_fields = sorted(
            _RETIRED_EVIDENCE_FIELDS.intersection(evidence))
        if evidence_fields:
            errors.append(
                f"{prefix} retired v9 evidence fields are forbidden: "
                f"{evidence_fields}")
    return errors


def _terminal_record_provenance_invalid(rec: dict, outcome: dict) -> bool:
    """Recompute record-visible provenance failures used by C1 collection."""
    source = rec.get("source")
    declared = rec.get("collection_contract") or {}
    try:
        expected_contract = R.collection_contract(
            source, "main", record_schema_version=rec.get(
                "schema_version", R.SCHEMA_VERSION),
            contract_version=(
                declared.get("version")
                if (source or {}).get("source_dataset") == "b1k" else None))
        future_view_selection._source_binding(
            source, scene_id=str(rec.get("scene_id") or ""))
    except (KeyError, StopIteration, TypeError, ValueError):
        return True
    return bool(
        rec.get("collection_contract") != expected_contract or
        not str(outcome.get("outcome_id") or ""))


def _terminal_asset_withhold_errors(
        prefix: str, outcome: dict, *, require_typed: bool,
        rec: Optional[dict] = None) -> List[str]:
    """Validate one policy-scoped terminal-asset failure declaration.

    Legacy depth-conditioned records predate typed attribution and remain
    valid for A/B compilation.  Only the externally authenticated directed-C1
    action policy opts into this frozen reason/authority contract.
    """
    if not require_typed:
        return []
    errors = []
    has_asset = outcome.get("terminal_rgb_asset") is not None
    has_reason = "terminal_rgb_asset_withhold" in outcome
    has_authority = "terminal_rgb_asset_withhold_authority" in outcome
    if has_asset and (has_reason or has_authority):
        errors.append(
            f"{prefix} terminal RGB asset and withhold are mutually exclusive")
    if not has_reason and not has_authority:
        return errors
    if not has_reason:
        errors.append(f"{prefix} terminal RGB withhold reason is required")
        return errors
    if not has_authority:
        errors.append(f"{prefix} terminal RGB withhold authority is required")
        return errors
    reason = outcome.get("terminal_rgb_asset_withhold")
    authority = outcome.get("terminal_rgb_asset_withhold_authority")
    legal = (
        future_view_selection.TERMINAL_ASSET_WITHHOLD_REASON_AUTHORITIES.get(
            reason))
    if legal is None or authority not in legal:
        errors.append(
            f"{prefix} terminal RGB reason/authority pair is invalid")
        return errors

    checkpoint_reasons = {
        "terminal_checkpoint_missing",
        "terminal_checkpoint_incomplete",
        "terminal_checkpoint_pose_invalid",
        "terminal_pose_disagreement",
    }
    if (authority ==
            future_view_selection.TERMINAL_ASSET_RECORD_AUTHORITY and
            reason in checkpoint_reasons):
        try:
            future_view_selection._terminal_checkpoint(outcome)
        except future_view_selection.TerminalRGBAssetError as error:
            if error.reason != reason or error.authority != authority:
                errors.append(
                    f"{prefix} terminal RGB record-recomputable reason "
                    "does not match the record")
        else:
            errors.append(
                f"{prefix} terminal RGB record-recomputable withhold is "
                "not reproduced by the record")
    if (authority ==
            future_view_selection.TERMINAL_ASSET_RECORD_AUTHORITY and
            reason == "terminal_provenance_invalid" and rec is not None and
            not _terminal_record_provenance_invalid(rec, outcome)):
        errors.append(
            f"{prefix} terminal RGB record-recomputable provenance withhold "
            "is not reproduced by the record")
    return errors


def validate_record_local(
        rec: dict, *,
        context: Optional[RecordValidationContext] = None,
        asset_root=None) -> List[str]:
    """Validate deterministic record integrity without loading source data.

    A context may pin run-level schema and policy declarations, but this level
    never opens R2R episodes, GLB, PLY, or house assets.  With no context it
    validates the supported schema declared by the record and treats any
    internally valid collection contract as a certificate-strength marker,
    never as external source authority.
    """
    errors: List[str] = []
    fid = rec.get("frame_id", "?")
    if context is not None and not isinstance(context, RecordValidationContext):
        raise TypeError("context must be a RecordValidationContext")
    expected_schema = (
        context.expected_schema_version if context is not None
        else rec.get("schema_version"))
    if expected_schema not in {
            R.SCHEMA_VERSION, R.V18_SCHEMA_VERSION} or \
            rec.get("schema_version") != expected_schema:
        errors.append(f"[{fid}] unsupported schema {rec.get('schema_version')}")
        return errors
    if expected_schema == R.V18_SCHEMA_VERSION:
        errors.extend(
            f"[{fid}] {error}" for error in R.validate_record_v18(rec))
        # The opt-in envelope remains dependency-light when no registered
        # route is supplied. Formal GS collection provides an explicit source
        # context and therefore also receives the complete outcome checks.
        if context is None or context.route != "gs_v18_registered":
            return errors
    else:
        expected_oracle = (
            context.expected_oracle_contract_version if context is not None
            else R.ORACLE_CONTRACT_VERSION)
        if rec.get("oracle_contract_version") != expected_oracle:
            errors.append(
                f"[{fid}] unsupported oracle contract "
                f"{rec.get('oracle_contract_version')!r}")
            return errors
        errors.extend(_validate_v11_shape(rec))
    sensor, strict_shared_oracle = (
        _validate_record_local_context(rec, context, errors))
    source_dataset = str(
        (rec.get("source") or {}).get("source_dataset") or "")
    try:
        if strict_shared_oracle and source_dataset == "gs":
            require_contact_instance_witness = \
                gs_semantic.scene_task_available(
                    rec.get("gs_scene_capability"), rec.get("source"), "A3")
        else:
            require_contact_instance_witness = (
                capability_contracts.task_available(source_dataset, "A3")
                if strict_shared_oracle else None)
    except ValueError:
        # Source validation above owns the diagnostic.  Keeping ``None`` here
        # preserves the strict shared-oracle default instead of weakening it.
        require_contact_instance_witness = None
    for outcome in rec.get("outcomes", []):
        _validate_outcome_identity(rec, outcome, errors)
        _validate_outcome_execution(
            rec, outcome, sensor, strict_shared_oracle, errors,
            require_contact_instance_witness=
                require_contact_instance_witness)
        prefix = (
            f"[{rec.get('frame_id', '?')}:"
            f"{outcome.get('outcome_id', '?')}]"
        )
        errors.extend(_terminal_asset_withhold_errors(
            prefix, outcome, require_typed=False,
            rec=rec))
        if outcome.get("terminal_rgb_asset") is not None:
            if asset_root is None:
                errors.append(f"{prefix} terminal RGB asset root is required")
            else:
                try:
                    future_view_selection.validate_terminal_rgb_asset(
                        rec, outcome, asset_root=asset_root)
                except ValueError as error:
                    errors.append(f"{prefix} {error}")
    errors.extend(_validate_radius_monotonicity(rec))
    errors.extend(_validate_balanced_selection(
        rec,
        (context.expected_setting_sampling_policy
         if context is not None else None),
        (context.expected_action_sampling_policy
         if context is not None else None)))
    return errors


def _trusted_b1k_source_errors(
        rec: dict, context: RecordValidationContext) -> List[str]:
    """Bind B1K record atoms to one externally rederived scene authority."""
    fid = rec.get("frame_id", "?")
    scene_id = str(rec.get("scene_id") or "")
    expected_by_version = context.expected_collection_contracts.get(scene_id)
    if expected_by_version is None:
        return [
            f"[{fid}] trusted B1K scene authority is unregistered for "
            f"scene {scene_id!r}"]
    expected_contract = expected_by_version.get(
        (rec.get("collection_contract") or {}).get("version"))
    if not isinstance(expected_contract, dict):
        return [f"[{fid}] trusted B1K collection contract is unregistered"]
    resolver = context.b1k_scene_authority_resolver
    try:
        resolved_sha256 = resolver(scene_id)
    except Exception as error:
        return [f"[{fid}] trusted B1K scene authority unavailable: {error}"]
    if (not isinstance(resolved_sha256, str) or
            len(resolved_sha256) != 64 or
            any(character not in "0123456789abcdef"
                for character in resolved_sha256)):
        return [f"[{fid}] trusted B1K scene authority digest is invalid"]
    try:
        source_sha256 = R.b1k_scene_authority_sha256(rec.get("source"))
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        return [f"[{fid}] trusted B1K source unavailable: {error}"]
    errors = []
    if (source_sha256 != resolved_sha256 or
            expected_contract.get("scene_authority_sha256") !=
            resolved_sha256):
        errors.append(
            f"[{fid}] B1K scene authority differs from trusted resolver")
    for outcome in rec.get("outcomes") or []:
        atom = outcome.get("terminal_rgb_asset")
        if atom is None:
            continue
        prefix = f"[{fid}:{outcome.get('outcome_id', '?')}]"
        if not isinstance(atom, dict):
            errors.append(
                f"{prefix} trusted B1K terminal RGB atom is invalid")
            continue
        source = atom.get("source")
        if not isinstance(source, dict):
            errors.append(
                f"{prefix} trusted B1K terminal source authority is invalid")
        elif source.get("scene_authority_sha256") != resolved_sha256:
            errors.append(
                f"{prefix} trusted B1K terminal source authority disagrees")
        renderer = atom.get("renderer")
        if not isinstance(renderer, dict):
            errors.append(
                f"{prefix} trusted B1K renderer authority is invalid")
        elif renderer.get("source_scene_sha256") != resolved_sha256:
            errors.append(
                f"{prefix} trusted B1K renderer authority disagrees")
    return errors


def validate_record_source_bound(
        rec: dict, *, context: Optional[RecordValidationContext] = None,
        asset_root=None) -> List[str]:
    """Run local validation, then authenticate the registered dataset."""
    fid = rec.get("frame_id", "?")
    if context is None:
        return [f"[{fid}] source-bound validation context required"]
    if not isinstance(context, RecordValidationContext):
        raise TypeError("context must be a RecordValidationContext")
    errors = validate_record_local(
        rec, context=context, asset_root=asset_root)
    if context.route == "gs_v18_registered":
        resolver = context.gs_scene_source_resolver
        try:
            trusted_source = resolver(str(rec.get("scene_id") or ""))
        except Exception as error:
            errors.append(f"[{fid}] trusted GS source unavailable: {error}")
        else:
            if rec.get("source") != trusted_source:
                errors.append(f"[{fid}] trusted GS source differs from record")
            try:
                trusted_capability = context.gs_scene_capability_resolver(
                    str(rec.get("scene_id") or ""))
            except Exception as error:
                errors.append(
                    f"[{fid}] trusted GS scene capability unavailable: "
                    f"{error}")
            else:
                if rec.get("gs_scene_capability") != trusted_capability:
                    errors.append(
                        f"[{fid}] trusted GS scene capability differs from "
                        "record")
        return errors
    if context.route == "b1k_v16_registered":
        errors.extend(_trusted_b1k_source_errors(rec, context))
        return errors
    binding_errors = scene_pool.trusted_r2r_source_binding_errors(
        rec, context)
    errors.extend(binding_errors)
    if not binding_errors:
        errors.extend(scene_pool.trusted_r2r_a3_source_errors(rec, context))
    return errors


def _decode_file_records(
        path: str, context: Optional[RecordValidationContext]) -> list[dict]:
    from pipeline import record as R
    payload = Path(path).read_bytes()
    if context is not None:
        decoder = {
            R.SCHEMA_VERSION: R.decode_records,
            R.V18_SCHEMA_VERSION: R.decode_v18_records,
        }.get(context.expected_schema_version, R.decode_records)
        return decoder(payload)
    decoded = [json.loads(line) for line in payload.decode("utf-8").splitlines()
               if line.strip()]
    schemas = {value.get("schema_version") for value in decoded
               if isinstance(value, dict)}
    decoder = {
        frozenset({R.SCHEMA_VERSION}): R.decode_records,
        frozenset({R.V18_SCHEMA_VERSION}): R.decode_v18_records,
    }.get(frozenset(schemas), R.decode_records)
    return decoder(payload)


def validate_file_local(
        path: str, *, context: Optional[RecordValidationContext] = None):
    """Validate a JSONL snapshot without opening dataset source assets."""
    records = _decode_file_records(path, context)
    violations = [
        error
        for rec in records
        for error in validate_record_local(
            rec, context=context, asset_root=Path(path).parent)
    ]
    violations.extend(validate_intervention_groups(records))
    return len(records), violations

def validate_file_source_bound(
        path: str, *, context: Optional[RecordValidationContext] = None):
    """Validate a JSONL snapshot against required external source authority."""
    if context is None:
        raise ValueError("source-bound validation context required")
    records = _decode_file_records(path, context)
    violations = [
        error
        for rec in records
        for error in validate_record_source_bound(
            rec, context=context, asset_root=Path(path).parent)
    ]
    violations.extend(validate_intervention_groups(records))
    return len(records), violations
