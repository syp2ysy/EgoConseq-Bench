"""Public structured consequence judge for the existing pipeline."""

from __future__ import annotations

import copy
from collections import Counter
import math

import numpy as np

from pipeline import (
    actions as A, config, dataset_contracts, gs_semantic, objects as OBJ,
    perception, record as record_fields,
)
from pipeline.consensus import (
    R2R_A_STABILITY_PERTURBATIONS, build_a_stability_certificate,
    oracle_consensus,
)
from pipeline import rollout


_A_STABILITY_BATCH_DIAGNOSTICS = Counter()


def _reset_a_stability_batch_diagnostics() -> None:
    _A_STABILITY_BATCH_DIAGNOSTICS.clear()


def _a_stability_batch_diagnostics() -> dict:
    """Return execution-only pose-batching counters, never record content."""
    return dict(_A_STABILITY_BATCH_DIAGNOSTICS)


def _full_contact_attribution(frame, contact) -> dict:
    world_point = contact.get("world_point")
    semantic_index = getattr(frame, "semantic_index", None)
    if (world_point is None or semantic_index is None or
            not hasattr(semantic_index, "assign")):
        return {"instance_id": None, "category": None, "unattributed": True}
    point = np.asarray(world_point, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        return {"instance_id": None, "category": None, "unattributed": True}
    point_local = perception.local_from_world(
        point[None, :], frame.position, frame.yaw_rad)[0]
    low, high = config.GROUND_OBSTACLE_BAND_M
    heights = np.linspace(
        low, high, config.CONTACT_ATTRIBUTION_PROBE_COUNT)
    probes_local = np.repeat(point_local[None, :], len(heights), axis=0)
    floor_y = frame.floor_plane.y_at(point_local[0], point_local[2])
    normal_y = frame.floor_plane.normal_local[1]
    probes_local[:, 1] = floor_y + heights / normal_y
    probes = perception.world_from_local(
        probes_local, frame.position, frame.yaw_rad)
    ids = np.asarray(semantic_index.assign(probes), dtype=np.int64)
    valid = []
    for value in ids:
        category = frame.predicate_category(int(value)).lower()
        if (int(value) != 0 and
                config.is_contact_obstacle_category(category)):
            valid.append(int(value))
    if not valid:
        return {"instance_id": None, "category": None, "unattributed": True}
    values, counts = np.unique(valid, return_counts=True)
    full_id = int(values[np.argmax(counts)])
    return {
        "instance_id": full_id,
        "category": frame.id_to_cat.get(full_id, "unknown"),
        "unattributed": False,
    }


def _attribute_full_contact(frame, physical):
    contact = physical.get("contact")
    if not contact:
        return
    # A missing or malformed world point degrades to an explicit
    # unattributed contact rather than silently omitting the attribution
    # fields (the GS oracle can return no obstacle geometry).
    attr = _full_contact_attribution(frame, contact)
    contact["full_geometry_attribution"] = copy.deepcopy(attr)
    contact.update(attr)
    # The contact anchor is the disc centre at the moment of contact, same
    # as the depth oracle; the obstacle surface point stays in point_3d.
    center = contact.get("center_local")
    contact["xy"] = (
        [float(center[0]), float(center[1])] if center is not None else None)
    contact["point_3d"] = contact.get("world_point")
    contact["pixel"] = None
    contact["pixel_in_frame"] = False


def _attribute_depth_contact(frame, body, estimate) -> dict:
    center = estimate.get("center_local")
    if center is not None:
        attr = OBJ.attribute_contact(
            frame.pts, frame.pts_sem, center, body.radius_m, frame.id_to_cat,
            predicate_categories=frame.id_to_predicate_cat,
            height_band=config.GROUND_OBSTACLE_BAND_M,
            floor_plane=frame.floor_plane)
        # Band midpoint above the floor *under the contact*, measured along
        # the plane normal — the same signed-normal convention as the
        # full-geometry probes — not above the plane's origin and not a
        # vertical offset: on a tilted floor those are different places.
        point = (center[0],
                 frame.floor_plane.y_at(center[0], center[1]) +
                 config.CONTACT_ATTRIBUTION_HEIGHT_M /
                 float(frame.floor_plane.normal_local[1]),
                 center[1])
        pixel = perception.project_ground(
            point, frame.K,
            camera_height=frame.sensor.nominal_camera_offset_m)
        estimate["contact"] = {
            "xy": list(center), "point_3d": list(point),
            "pixel": list(pixel) if pixel is not None else None,
            "pixel_in_frame": bool(pixel is not None and 0 <= pixel[0] < frame.depth.shape[1]
                                   and 0 <= pixel[1] < frame.depth.shape[0]),
            "depth_mask_attribution": copy.deepcopy(attr),
            **attr,
        }
    else:
        estimate["contact"] = None
    return estimate


def _view_estimate(frame, body, acts) -> dict:
    return _attribute_depth_contact(
        frame, body, rollout.view_collision_rollout(frame, acts, body.radius_m))


def _stability_oracle_row(
        frame, body, acts, nav, transform, *,
        require_contact_instance_witness: bool = True) -> dict:
    """Run the existing full/depth/corridor oracles at one SE(2) offset."""
    base_pose = (
        float(transform["x_m"]), float(transform["z_m"]),
        float(transform["yaw_deg"]),
    )
    phase = collect_visible_space_phase(
        frame, body, acts, nav, base_pose=base_pose,
        require_contact_instance_witness=require_contact_instance_witness)
    return {
        "physical": phase["physical"],
        "depth_physical": phase["depth_physical"],
        "corridor_coverage": phase["corridor_coverage"],
    }


def collect_visible_space_phase(
        frame, body, acts, nav, *, base_pose,
        require_contact_instance_witness: bool = True) -> dict:
    """Run one phase with nav physics and the original frame as evidence."""
    physical_bundle = rollout.physical_rollout(nav, acts)
    full = physical_bundle["physical"]
    _attribute_full_contact(frame, full)
    depth = _attribute_depth_contact(
        frame, body, rollout.view_collision_rollout(
            frame, acts, body.radius_m, base_pose=base_pose))
    if full.get("contact") is not None:
        full["contact"]["depth_mask_attribution"] = copy.deepcopy(
            (depth.get("contact") or {}).get("depth_mask_attribution") or {})
    coverage = rollout.corridor_coverage(
        frame, acts, body.radius_m, base_pose=base_pose,
        max_arc_m=rollout.realized_corridor_arc_m(full))
    return {
        "actions": A.actions_to_dicts(acts),
        "execution": physical_bundle["execution"],
        "checkpoints": physical_bundle["checkpoints"],
        "physical": full,
        "depth_physical": depth,
        "corridor_coverage": float(coverage),
        "consensus": oracle_consensus(
            full, depth, coverage,
            require_contact_instance_witness=
                bool(require_contact_instance_witness)),
        "base_pose": {
            "x_m": float(base_pose[0]), "z_m": float(base_pose[1]),
            "heading_deg": float(base_pose[2]),
        },
    }


def _prepare_exact_face_identity_requests(frame, oracle_rows):
    """Prepare exact A3 requests without crossing the semantic trust boundary."""
    semantic_index = getattr(frame, "semantic_index", None)
    confirm_batch = getattr(
        semantic_index, "confirm_contact_instances", None)
    identities = [None] * len(oracle_rows)
    requests = []
    request_indices = []
    for index, oracle_inputs in enumerate(oracle_rows):
        physical = oracle_inputs["physical"]
        if physical.get("collision") is not True:
            continue
        rebuilt = oracle_consensus(
            physical, oracle_inputs["depth_physical"],
            oracle_inputs["corridor_coverage"],
            require_contact_instance_witness=True)
        instance_id = rebuilt.get("full_contact_instance_id")
        point = (physical.get("contact") or {}).get("world_point")
        if (rebuilt.get("accepted") is not True or instance_id is None or
                point is None or not callable(confirm_batch)):
            identities[index] = {
                "confirmed": False,
                "reason": "contact_face_authority_unavailable",
                "instance_id": None,
            }
            continue
        request_indices.append(index)
        geometry_element_index = (
            physical.get("contact") or {}).get("geometry_element_index")
        requests.append(
            (int(instance_id), point, int(geometry_element_index))
            if geometry_element_index is not None else
            (int(instance_id), point))
    return identities, request_indices, requests, confirm_batch


def _exact_face_identities_for_oracle_inputs(frame, oracle_rows) -> list:
    """Legacy one-outcome exact attribution, retained as the fallback."""
    identities, request_indices, requests, confirm_batch = (
        _prepare_exact_face_identity_requests(frame, oracle_rows))
    if not requests:
        return identities
    try:
        confirmed = confirm_batch(requests)
    except (KeyError, TypeError, ValueError) as error:
        confirmed = [{
            "confirmed": False,
            "reason": "contact_face_query_failed",
            "instance_id": None,
            "error": str(error),
        } for _request in requests]
    if len(confirmed) != len(requests):
        raise ValueError("batched contact identity results are misaligned")
    for index, identity in zip(request_indices, confirmed):
        identities[index] = identity
    return identities


def collect_a_stability_rows(sim, frame, body, acts,
                             nominal_outcome: dict, *,
                             require_contact_instance_witness: bool = True
                             ) -> list[dict]:
    """Materialize the seven frozen SE(2) rerollout rows, without A3 lookup.

    The initial RGB-D frame remains the evidence frame for every perturbation.
    Habitat navmesh physics is rebound to the perturbed world origin; depth and
    corridor checks use the matching ``base_pose`` against that original frame.
    """
    rows = []
    for index, perturbation in enumerate(R2R_A_STABILITY_PERTURBATIONS):
        transform = {
            "x_m": float(perturbation["x_m"]),
            "z_m": float(perturbation["z_m"]),
            "yaw_deg": float(perturbation["yaw_deg"]),
        }
        local = np.array([[transform["x_m"], 0.0, transform["z_m"]]])
        world_position = perception.world_from_local(
            local, frame.position, frame.yaw_rad)[0]
        world_yaw = (
            float(frame.yaw_rad) - math.radians(transform["yaw_deg"]))
        nav = sim.nav(world_position, world_yaw)
        oracle_inputs = record_fields.json_value(_stability_oracle_row(
            frame, body, acts, nav, transform,
            require_contact_instance_witness=
                require_contact_instance_witness))
        if index == 0:
            nominal_coverage = float(
                (nominal_outcome.get("evidence") or {}).get(
                    "physical", {})["coverage"])
            nominal_physical = record_fields.json_value(
                nominal_outcome.get("physical") or {})
            nominal_depth = record_fields.json_value(
                nominal_outcome.get("depth_physical") or {})
            if (oracle_inputs["physical"] !=
                    nominal_physical or
                    oracle_inputs["depth_physical"] !=
                    nominal_depth or
                    abs(oracle_inputs["corridor_coverage"] -
                        nominal_coverage) > 1e-12):
                raise ValueError(
                    "fresh nominal rerollout disagrees with nominal outcome")
        rows.append({
            "perturbation_id": str(perturbation["id"]),
            "transform": transform,
            **oracle_inputs,
        })
    return rows


def deferred_a_stability_certificate(
        frame, acts, rows, outcome: dict, *,
        exact_contact_identity: bool = True,
        require_contact_instance_witness: bool = True) -> dict:
    """Describe one certificate whose exact A3 identity is pose-batched later."""
    return {
        "frame": frame,
        "actions": A.actions_to_dicts(acts),
        "rows": rows,
        "outcome": outcome,
        "exact_contact_identity": bool(exact_contact_identity),
        "require_contact_instance_witness": bool(
            require_contact_instance_witness),
    }


def _runtime_authority_binding(semantic_index):
    """Type the already-authenticated runtime identity source, if present."""
    semantic_sha256 = getattr(
        semantic_index, "_semantic_ply_sha256", None)
    scene_sha256 = getattr(
        semantic_index, "_scene_authority_sha256", None)
    gs_sha256 = getattr(
        semantic_index, "geometry_authority_sha256", None)
    gs_semantic_sha256 = getattr(
        semantic_index, "semantic_source_sha256", None)
    if semantic_sha256 is not None and scene_sha256 is None:
        return dataset_contracts.AuthorityBinding(
            source_dataset="r2r",
            identity_schema="mp3d-contact-face-identity.v1",
            authoritative_source_role="semantic",
            source_sha256=str(semantic_sha256),
        )
    if scene_sha256 is not None and semantic_sha256 is None:
        return dataset_contracts.AuthorityBinding(
            source_dataset="b1k",
            identity_schema="b1k-contact-triangle-identity.v1",
            authoritative_source_role="scene_authority",
            source_sha256=str(scene_sha256),
        )
    if (gs_sha256 is not None and gs_semantic_sha256 is not None and
            semantic_sha256 is None and scene_sha256 is None):
        return dataset_contracts.AuthorityBinding(
            source_dataset="gs",
            identity_schema=gs_semantic.VISIBLE_CONTACT_IDENTITY_SCHEMA,
            authoritative_source_role="source_bundle",
            source_sha256=str(gs_sha256),
            semantic_source_sha256=str(gs_semantic_sha256),
        )
    return None


def finalize_a_stability_certificates(pending) -> None:
    """Resolve exact A3 identities once per semantic index and attach certs.

    The successful path combines requests from every surviving outcome at a
    pose. If that combined query raises a legacy query exception or returns a
    misaligned result, each outcome is replayed through the retained legacy
    helper. This construction preserves the old outcome-wide failure payload
    and error text while keeping the common path batched.
    """
    entries = list(pending)
    prepared = {}
    groups = {}
    for entry_index, entry in enumerate(entries):
        rows = entry["rows"]
        if not entry["exact_contact_identity"]:
            prepared[entry_index] = [None] * len(rows)
            _A_STABILITY_BATCH_DIAGNOSTICS[
                "exact_identity_skipped_certificates"] += 1
            continue
        identities, request_indices, requests, confirm_batch = (
            _prepare_exact_face_identity_requests(entry["frame"], rows))
        prepared[entry_index] = identities
        if not requests:
            continue
        semantic_index = getattr(entry["frame"], "semantic_index", None)
        key = id(semantic_index)
        group = groups.setdefault(key, {
            "confirm": confirm_batch,
            "members": [],
            "requests": [],
            "scatter": [],
        })
        member_index = len(group["members"])
        group["members"].append(entry_index)
        for row_index, request in zip(request_indices, requests):
            group["requests"].append(request)
            group["scatter"].append((member_index, row_index))

    for group in groups.values():
        requests = group["requests"]
        _A_STABILITY_BATCH_DIAGNOSTICS["combined_query_batches"] += 1
        _A_STABILITY_BATCH_DIAGNOSTICS[
            "combined_query_requests"] += len(requests)
        try:
            confirmed = group["confirm"](requests)
            if len(confirmed) != len(requests):
                raise ValueError(
                    "batched contact identity results are misaligned")
        except (KeyError, TypeError, ValueError):
            # The old helper is both the differential reference and the exact
            # error-semantics fallback. One bad outcome cannot poison siblings.
            _A_STABILITY_BATCH_DIAGNOSTICS[
                "combined_query_fallbacks"] += 1
            _A_STABILITY_BATCH_DIAGNOSTICS[
                "legacy_outcome_replays"] += len(group["members"])
            for entry_index in group["members"]:
                entry = entries[entry_index]
                prepared[entry_index] = (
                    _exact_face_identities_for_oracle_inputs(
                        entry["frame"], entry["rows"]))
            continue
        for (member_index, row_index), identity in zip(
                group["scatter"], confirmed):
            entry_index = group["members"][member_index]
            prepared[entry_index][row_index] = identity

    for entry_index, entry in enumerate(entries):
        for row, identity in zip(entry["rows"], prepared[entry_index]):
            row["exact_contact_identity"] = identity
        entry["outcome"]["shared_oracle_stability"] = (
            build_a_stability_certificate(
                entry["actions"], entry["rows"],
                authority_binding=_runtime_authority_binding(
                    entry["frame"].semantic_index),
                require_contact_instance_witness=entry[
                    "require_contact_instance_witness"]))
        _A_STABILITY_BATCH_DIAGNOSTICS[
            "certificates_finalized"] += 1


def collect_a_stability_certificate(sim, frame, body, acts,
                                    nominal_outcome: dict, *,
                                    exact_contact_identity: bool = True,
                                    require_contact_instance_witness: bool = True
                                    ) -> dict:
    """Compatibility wrapper for callers that materialize one certificate.

    ``exact_contact_identity=False`` skips the complete-face A3 attribution.
    That pass feeds only ``contact_instance_*`` in the summary; A1/A2 stability
    still comes from all seven rerollouts. Such a certificate is diagnostic and
    must not be published as A3 ground truth.
    """
    rows = collect_a_stability_rows(
        sim, frame, body, acts, nominal_outcome,
        require_contact_instance_witness=require_contact_instance_witness)
    outcome = {}
    finalize_a_stability_certificates([
        deferred_a_stability_certificate(
            frame, acts, rows, outcome,
            exact_contact_identity=exact_contact_identity,
            require_contact_instance_witness=
                require_contact_instance_witness)
    ])
    return outcome["shared_oracle_stability"]


def judge(frame, body, acts, nav=None, *, target_ids=None,
          cached_physical=None, cached_depth_physical=None,
          cached_corridor_coverage=None,
          cached_oracle_consensus=None,
          require_contact_instance_witness: bool = False) -> dict:
    """Predict structured consequences for one body-action query."""
    targets = (OBJ.eligible_target_ids(frame) if target_ids is None
               else list(target_ids))
    physical = copy.deepcopy(cached_physical) if cached_physical is not None \
        else rollout.physical_rollout(nav, acts)
    _attribute_full_contact(frame, physical["physical"])
    checkpoints = physical["checkpoints"]
    future_view = rollout.future_view_rollout(frame, checkpoints, targets)
    execution = physical["execution"]
    view_estimate = (copy.deepcopy(cached_depth_physical)
                     if cached_depth_physical is not None else
                     _view_estimate(frame, body, acts))
    if cached_depth_physical is not None:
        _attribute_depth_contact(frame, body, view_estimate)
    view_collision = view_estimate["collision"]
    full_contact = physical["physical"].get("contact")
    depth_contact = view_estimate.get("contact")
    if full_contact is not None:
        full_contact["depth_mask_attribution"] = copy.deepcopy(
            (depth_contact or {}).get("depth_mask_attribution") or {})
    coverage = (float(cached_corridor_coverage)
                if cached_corridor_coverage is not None else
                rollout.corridor_coverage(
                    frame, acts, body.radius_m,
                    max_arc_m=rollout.realized_corridor_arc_m(
                        physical["physical"])))
    # A precheck runs before semantic attribution. Under the frozen R2R v16
    # contract it therefore cannot authorize a collision candidate: rebuild
    # the final consensus after both attribution paths have been materialized.
    if require_contact_instance_witness:
        consensus = oracle_consensus(
            physical["physical"], view_estimate, coverage,
            require_contact_instance_witness=True)
    elif cached_oracle_consensus is not None:
        consensus = copy.deepcopy(cached_oracle_consensus)
    else:
        consensus = oracle_consensus(
            physical["physical"], view_estimate, coverage)
    evidence = {
        "physical": {
            "status": "sufficient" if coverage >= config.EVIDENCE_COVERAGE_MIN else "insufficient",
            "coverage": coverage,
            "coverage_protocol": rollout.EVIDENCE_PROTOCOL_VERSION,
            "view_collision_estimate": view_collision,
        },
        "future_view": {
            "status": (
                "not_computed" if future_view["status"] != "computed" else
                "insufficient" if future_view.get("objects_entering_view") or
                not future_view.get("initial_depth_reprojection_agrees", False)
                else "sufficient"),
            "source_frame": "initial",
        },
    }
    result = {
        "body": body.to_dict(),
        "actions": A.actions_to_dicts(acts),
        "execution": execution,
        "checkpoints": checkpoints,
        "physical": physical["physical"],
        "depth_physical": view_estimate,
        "oracle_consensus": consensus,
        "future_view": future_view,
        "evidence": evidence,
        "provenance": {
            "physical_oracle": physical["physical"]["authority"],
            "sensor_profile": frame.sensor.to_dict(),
            "camera_height_above_visible_floor_m":
                frame.camera_height_above_visible_floor_m,
            "checkpoint_progress": list(config.CHECKPOINT_PROGRESS),
        },
    }
    return result
