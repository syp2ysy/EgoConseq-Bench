"""FrameRecord assembly + jsonl I/O (append-style; numpy -> python)."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from typing import Iterator, List

import numpy as np

from pipeline import (
    actions as action_geometry, config, consensus as consensus_fields,
    capability_contracts, dataset_contracts, gs_semantic, objects as OBJ,
    outcome as outcome_fields, perception, semantic,
)
from pipeline.floor_plane import FloorPlaneEstimate, FloorPlaneFitResult


def stored_floor_plane(rec: dict):
    """Rebuild a record's canonical plane, or say why it cannot be rebuilt.
    Returns ``(plane, reason)`` with exactly one set. The single place anything
    turns a persisted record back into a floor: the validator reads it to check
    the published height, and the viewer reads it to rebuild a frame. Neither
    may re-fit -- that is the per-image estimate returning through the back door
    -- and neither may fall back to a level floor at the origin, which would
    answer floor-relative questions against a surface nobody measured.
    """
    calibration = rec.get("floor_calibration")
    if not isinstance(calibration, dict):
        return None, "floor calibration is missing"
    reasons = calibration.get("rejection_reasons")
    if not isinstance(reasons, list):
        return None, "floor calibration has no rejection_reasons list"
    if not all(isinstance(reason, str) for reason in reasons):
        return None, ("floor calibration rejection_reasons must all be "
                      f"strings: {reasons!r}")
    if reasons:
        return None, f"floor calibration was rejected: {sorted(reasons)}"
    estimate = calibration.get("estimate")
    if estimate is None:
        return None, "floor calibration carries no plane"
    # A persisted record is untrusted input, so every shape is checked here
    # rather than left to raise from inside a JSON accessor: a list, a string
    # or a number would otherwise surface as AttributeError instead of a
    # validation reason a caller can report.
    if not isinstance(estimate, dict):
        return None, f"floor calibration plane must be an object: {estimate!r}"
    try:
        return FloorPlaneEstimate.from_json(estimate), None
    except (ValueError, TypeError) as error:
        return None, f"floor calibration plane is invalid: {error}"


def require_floor_plane(rec: dict) -> FloorPlaneEstimate:
    """:func:`stored_floor_plane`, raising instead of reporting."""
    plane, reason = stored_floor_plane(rec)
    if plane is None:
        raise ValueError(
            f"{rec.get('frame_id')}: {reason}; cannot rebuild this frame")
    return plane


def authenticated_camera_height_above_visible_floor_m(rec: dict) -> float:
    """Rederive the persisted public height from accepted floor authority.

    A record scalar is never authoritative on its own.  The only accepted
    value is ``n . (0, nominal_camera_offset_m, 0) + d`` from the record's
    accepted :class:`FloorPlaneEstimate`; the stored scalar is checked only as
    a second serialization of that identity.  JSON booleans, non-numbers and
    non-finite values are rejected before arithmetic.
    """
    if not isinstance(rec, dict):
        raise ValueError("camera-height record must be an object")
    sensor = rec.get("sensor")
    if not isinstance(sensor, dict):
        raise ValueError("record sensor is missing")
    nominal = sensor.get("nominal_camera_offset_m")
    if (isinstance(nominal, bool) or
            not isinstance(nominal, (int, float)) or
            not math.isfinite(float(nominal))):
        raise ValueError(
            "sensor nominal_camera_offset_m must be a finite JSON number")
    stored = rec.get("camera_height_above_visible_floor_m")
    if (isinstance(stored, bool) or
            not isinstance(stored, (int, float)) or
            not math.isfinite(float(stored))):
        raise ValueError(
            "camera_height_above_visible_floor_m must be a finite JSON number")
    plane = require_floor_plane(rec)
    derived = float(plane.height_above((0.0, float(nominal), 0.0)))
    if abs(float(stored) - derived) > \
            config.CAMERA_HEIGHT_CALIBRATION_TOLERANCE_M:
        raise ValueError(
            "camera_height_above_visible_floor_m does not match the "
            "record floor calibration")
    return derived

SCHEMA_VERSION = "conseq.v11"
V18_SCHEMA_VERSION = "conseq.v18"
ORACLE_CONTRACT_VERSION = "ground-disc-visible-v8"
BASE_ROLLOUT_KEY_VERSION = "base-rollout.v1"
R2R_V16_COLLECTION_CONTRACT_VERSION = "r2r-visible-space-abcd.v16"
B1K_V16_COLLECTION_CONTRACT_VERSION = "b1k-visible-space-abc1.v5"
B1K_RENDERER_INSTANCE_PROTOCOL = "b1k-observation.v5"
B1K_OBSERVATION_PROFILE_SHA256 = \
    "b02ba45708cf607eda82ee9d01671805c79670f46d7ca342661e16df5b881088"
B1K_C1_RENDER_MODE = "temporary-counterfactual-bank-batch.v1"
B1K_B_TARGET_GEOMETRY_SCHEMA = "b1k-b-target-geometry.v1"
B1K_INSTANCE_TRIANGLES_SCHEMA = "b1k-runtime-instance-triangles.v1"


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        json_value(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_atom_sha256(value: dict) -> str:
    """Public name for canonical hashes used by persisted v16 atoms."""
    return _canonical_sha256(value)


def _authority_surface_capability(source_dataset: str) -> dict:
    value = capability_contracts.snapshot_fields(source_dataset)
    return {**value, "sha256": _canonical_sha256(value)}


def _type_exact_json_equal(left, right) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if (len(left) != len(right) or set(left) != set(right) or
                not all(isinstance(key, str) for key in left)):
            return False
        return all(
            _type_exact_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _type_exact_json_equal(first, second)
            for first, second in zip(left, right))
    return left == right


def _require_v18_gs_source_provenance(
        source: dict, *, scene_id: str) -> str:
    """Authenticate the exact four-asset GS source used by v18."""
    dataset_contracts.validate_source_asset_provenance(source)
    source_dataset = source.get("source_dataset")
    source_scene_id = source.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("v18 record scene_id must be a nonempty string")
    if not isinstance(source_scene_id, str) or not source_scene_id:
        raise ValueError("v18 source scene_id must be a nonempty string")
    if source_scene_id != scene_id:
        raise ValueError("v18 source scene_id does not match record")
    if source.get("official_split") != "train":
        raise ValueError("v18 source is not from the official train split")
    if source_dataset != "gs":
        raise ValueError("conseq.v18 is reserved for GS collection")
    manifest_sha256 = source.get("source_manifest_sha256")
    if (not isinstance(manifest_sha256, str) or
            not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)):
        raise ValueError("v18 source manifest sha256 is invalid")
    return source_dataset


def action_program_sha256(action_dicts) -> str:
    """The single record-atom digest of one action program.

    Published as ``terminal_rgb_asset.binding.action_sha256`` and used as the
    join key between C1 families and the QA choice descriptors, so every caller
    must reach it through this function rather than re-spelling the wrapper.

    Takes already-canonical action dicts rather than parsing them: the value is
    frozen in published records, and re-parsing could renormalise a stored float
    and silently move an identifier that golden pins.

    The bare-list digest that becomes the candidate tag is a different value and
    lives in ``pipeline.actions.canonical_actions_sha256``.
    """
    return _canonical_sha256({"actions": list(action_dicts)})


def _valid_gs_visible_b_geometry(geometry: dict) -> bool:
    """Validate GS B geometry explicitly scoped to initial-visible depth."""
    geometry_keys = {
        "schema", "instance_id", "category",
        "geometry_authority_sha256", "semantic_source_sha256",
        "alignment_certificate_sha256", "visible_point_protocol",
        "visible_point_count", "visible_points_sha256", "ground_support",
        "reference_centroid", "sha256",
    }
    support_keys = {
        "protocol", "frame", "ground_band_m", "triangles_xz_m",
        "segments_xz_m", "points_xz_m", "sha256",
    }
    centroid_keys = {
        "protocol", "frame", "world_xyz_m", "world_xz_m", "sha256",
    }
    if not isinstance(geometry, dict) or set(geometry) != geometry_keys:
        return False
    support = geometry.get("ground_support")
    centroid = geometry.get("reference_centroid")
    if (not isinstance(support, dict) or set(support) != support_keys or
            not isinstance(centroid, dict) or
            set(centroid) != centroid_keys):
        return False
    try:
        triangles = np.asarray(support["triangles_xz_m"], dtype=np.float64)
        segments = np.asarray(support["segments_xz_m"], dtype=np.float64)
        points = np.asarray(support["points_xz_m"], dtype=np.float64)
        if not support["triangles_xz_m"]:
            triangles = np.empty((0, 3, 2), dtype=np.float64)
        if not support["segments_xz_m"]:
            segments = np.empty((0, 2, 2), dtype=np.float64)
        if not support["points_xz_m"]:
            points = np.empty((0, 2), dtype=np.float64)
        centroid_xyz = np.asarray(
            centroid["world_xyz_m"], dtype=np.float64)
        centroid_xz = np.asarray(
            centroid["world_xz_m"], dtype=np.float64)
        digests = (
            geometry["geometry_authority_sha256"],
            geometry["semantic_source_sha256"],
            geometry["alignment_certificate_sha256"],
            geometry["visible_points_sha256"], support["sha256"],
            centroid["sha256"], geometry["sha256"],
        )
        return bool(
            geometry["schema"] ==
                gs_semantic.VISIBLE_B_TARGET_GEOMETRY_SCHEMA and
            isinstance(geometry["instance_id"], int) and
            not isinstance(geometry["instance_id"], bool) and
            geometry["instance_id"] > 0 and
            isinstance(geometry["category"], str) and
            bool(geometry["category"].strip()) and
            geometry["visible_point_protocol"] ==
                gs_semantic.VISIBLE_INSTANCE_PROTOCOL and
            isinstance(geometry["visible_point_count"], int) and
            not isinstance(geometry["visible_point_count"], bool) and
            geometry["visible_point_count"] >=
                config.TARGET_GROUND_SUPPORT_MIN_POINTS and
            all(isinstance(value, str) and
                re.fullmatch(r"[0-9a-f]{64}", value)
                for value in digests) and
            support["protocol"] ==
                gs_semantic.VISIBLE_B_GROUND_SUPPORT_PROTOCOL and
            support["frame"] == "pbench_world_xz" and
            support["ground_band_m"] == [
                float(value) for value in config.GROUND_OBSTACLE_BAND_M] and
            triangles.ndim == 3 and triangles.shape[1:] == (3, 2) and
            segments.ndim == 3 and segments.shape[1:] == (2, 2) and
            points.ndim == 2 and points.shape[1:] == (2,) and
            len(triangles) > 0 and
            all(np.isfinite(value).all()
                for value in (triangles, segments, points)) and
            support["sha256"] == _canonical_sha256({
                key: value for key, value in support.items()
                if key != "sha256"}) and
            centroid["protocol"] ==
                gs_semantic.VISIBLE_B_REFERENCE_CENTROID_PROTOCOL and
            centroid["frame"] == "pbench_world_xyz" and
            centroid_xyz.shape == (3,) and centroid_xz.shape == (2,) and
            np.isfinite(centroid_xyz).all() and
            np.array_equal(centroid_xz, centroid_xyz[[0, 2]]) and
            centroid["sha256"] == _canonical_sha256({
                key: value for key, value in centroid.items()
                if key != "sha256"}) and
            geometry["sha256"] == _canonical_sha256({
                key: value for key, value in geometry.items()
                if key != "sha256"}))
    except (KeyError, TypeError, ValueError):
        return False


def _valid_b_geometry(geometry: dict) -> bool:
    if not isinstance(geometry, dict):
        return False
    schema = geometry.get("schema")
    if schema == gs_semantic.VISIBLE_B_TARGET_GEOMETRY_SCHEMA:
        return _valid_gs_visible_b_geometry(geometry)
    if schema == semantic.B_TARGET_GEOMETRY_SCHEMA:
        source_key = "semantic_ply_sha256"
        triangle_protocol = semantic.MP3D_INSTANCE_TRIANGLES_SCHEMA
        support_frame = "habitat_world_xz"
        centroid_frame = "habitat_world_xyz"
    elif schema == B1K_B_TARGET_GEOMETRY_SCHEMA:
        source_key = "scene_authority_sha256"
        triangle_protocol = B1K_INSTANCE_TRIANGLES_SCHEMA
        support_frame = "pbench_world_xz"
        centroid_frame = "pbench_world_xyz"
    else:
        return False
    geometry_keys = {
        "schema", "instance_id", "category", source_key,
        "full_triangle_protocol", "full_triangle_count",
        "full_triangles_sha256", "ground_support", "reference_centroid",
        "sha256",
    }
    support_keys = {
        "protocol", "frame", "ground_band_m", "triangles_xz_m",
        "segments_xz_m", "points_xz_m", "sha256",
    }
    centroid_keys = {
        "protocol", "frame", "world_xyz_m", "world_xz_m", "sha256",
    }
    if set(geometry) != geometry_keys:
        return False
    support = geometry.get("ground_support")
    centroid = geometry.get("reference_centroid")
    value = {key: geometry.get(key) for key in (
        "schema", "instance_id", "category", source_key,
        "full_triangle_protocol", "full_triangle_count",
        "full_triangles_sha256", "ground_support", "reference_centroid",
    )}
    try:
        if (not isinstance(support, dict) or set(support) != support_keys or
                not isinstance(centroid, dict) or
                set(centroid) != centroid_keys):
            return False
        raw_triangles = support["triangles_xz_m"]
        raw_segments = support["segments_xz_m"]
        raw_points = support["points_xz_m"]
        if not all(isinstance(component, list) for component in (
                raw_triangles, raw_segments, raw_points)):
            return False
        triangles = np.asarray(raw_triangles, dtype=np.float64)
        segments = np.asarray(raw_segments, dtype=np.float64)
        points = np.asarray(raw_points, dtype=np.float64)
        if not raw_triangles:
            triangles = np.empty((0, 3, 2), dtype=np.float64)
        if not raw_segments:
            segments = np.empty((0, 2, 2), dtype=np.float64)
        if not raw_points:
            points = np.empty((0, 2), dtype=np.float64)
        centroid_xyz = np.asarray(centroid["world_xyz_m"], dtype=np.float64)
        centroid_xz = np.asarray(centroid["world_xz_m"], dtype=np.float64)
        digests = (
            geometry.get(source_key),
            geometry.get("full_triangles_sha256"),
            support.get("sha256"), centroid.get("sha256"),
            geometry.get("sha256"),
        )
        return bool(
            isinstance(geometry.get("instance_id"), int) and
            not isinstance(geometry.get("instance_id"), bool) and
            geometry["instance_id"] > 0 and
            isinstance(geometry.get("category"), str) and
            bool(geometry["category"].strip()) and
            geometry.get("full_triangle_protocol") == triangle_protocol and
            isinstance(geometry.get("full_triangle_count"), int) and
            not isinstance(geometry.get("full_triangle_count"), bool) and
            geometry["full_triangle_count"] > 0 and
            all(isinstance(digest, str) and
                re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in digests) and
            support.get("protocol") == semantic.B_GROUND_SUPPORT_PROTOCOL and
            support.get("frame") == support_frame and
            support.get("ground_band_m") == [
                float(config.GROUND_OBSTACLE_BAND_M[0]),
                float(config.GROUND_OBSTACLE_BAND_M[1])] and
            triangles.ndim == 3 and triangles.shape[1:] == (3, 2) and
            segments.ndim == 3 and segments.shape[1:] == (2, 2) and
            points.ndim == 2 and points.shape[1:] == (2,) and
            len(triangles) > 0 and
            np.all(np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0]) != 0.0) and
            all(np.all(np.isfinite(component))
                for component in (triangles, segments, points)) and
            support.get("sha256") == _canonical_sha256({
                key: value for key, value in support.items()
                if key != "sha256"}) and
            centroid.get("protocol") == semantic.B_REFERENCE_CENTROID_PROTOCOL and
            centroid.get("frame") == centroid_frame and
            centroid_xyz.shape == (3,) and centroid_xz.shape == (2,) and
            np.all(np.isfinite(centroid_xyz)) and
            np.all(np.isfinite(centroid_xz)) and
            np.array_equal(centroid_xz, centroid_xyz[[0, 2]]) and
            centroid.get("sha256") == _canonical_sha256({
                key: value for key, value in centroid.items()
                if key != "sha256"}) and
            geometry.get("sha256") == _canonical_sha256(value))
    except (TypeError, ValueError):
        return False


def build_b_target_atom(*, selection: dict, geometry: dict,
                        pose: dict) -> dict:
    """Bind the one s0-selected target to complete source geometry."""
    if (not isinstance(selection, dict) or
            selection.get("eligible") is not True or
            selection.get("selection_protocol") !=
            config.B_TARGET_SELECTION_PROTOCOL):
        raise ValueError("B target selection is invalid")
    if not _valid_b_geometry(geometry):
        raise ValueError("B target geometry is invalid")
    if (selection.get("instance_id") != geometry.get("instance_id") or
            selection.get("category") != geometry.get("category")):
        raise ValueError("B target selection and exact geometry disagree")
    position = np.asarray(pose.get("position"), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("B target pose is invalid")
    initial_distance = semantic.point_to_ground_support_distance_m(
        position[[0, 2]], geometry["ground_support"])
    low, high = config.B_TARGET_VISIBLE_DISTANCE_RANGE_M
    if not low <= initial_distance <= high:
        raise ValueError("B target exact initial distance is out of range")
    value = {
        "schema": "b-target.v1",
        "selection": dict(selection),
        "geometry": dict(geometry),
        "initial_distance_m": float(initial_distance),
    }
    return {**value, "sha256": _canonical_sha256(value)}


def b_target_atom_valid(value: dict, *, pose: dict) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        rebuilt = build_b_target_atom(
            selection=value.get("selection"),
            geometry=value.get("geometry"), pose=pose)
    except (KeyError, TypeError, ValueError):
        return False
    return value == rebuilt


def r2r_semantic_source_sha256(source: dict) -> str:
    """Validate strict source provenance and return its semantic PLY hash."""
    binding = authority_binding(source)
    if binding.source_dataset != "r2r":
        raise ValueError("R2R source assets are invalid")
    return binding.source_sha256


def b1k_scene_authority_sha256(source: dict) -> str:
    """Validate B1K provenance and return its canonical runtime authority."""
    binding = authority_binding(source)
    if binding.source_dataset != "b1k":
        raise ValueError("B1K source assets are invalid")
    return binding.source_sha256


def authority_binding(source: dict) -> dataset_contracts.AuthorityBinding:
    """Resolve one typed A3 identity authority from strict provenance."""
    if not isinstance(source, dict):
        raise ValueError("source provenance must be an object")
    for key in ("source_manifest_sha256", "source_assets_sha256"):
        value = source.get(key)
        if (not isinstance(value, str) or
                not re.fullmatch(r"[0-9a-f]{64}", value)):
            raise ValueError(f"invalid {key}")
    return dataset_contracts.resolve_authority_binding(source)


def authenticated_b_target(rec: dict) -> tuple[dict | None, str]:
    """Rederive B's canonical s0 selector and bind it to source semantics."""
    target = rec.get("b_target")
    pose = rec.get("pose")
    objects = rec.get("objects")
    if (not isinstance(pose, dict) or
            not isinstance(objects, list) or
            any(not isinstance(value, dict) for value in objects) or
            not b_target_atom_valid(target, pose=pose)):
        return None, "b_target_invalid"
    source = rec.get("source") or {}
    dataset = (
        source.get("source_dataset") if isinstance(source, dict) else None)
    try:
        selector_record = rec
        if dataset == "r2r":
            expected_source = r2r_semantic_source_sha256(source)
            geometry_source = target["geometry"].get("semantic_ply_sha256")
        elif dataset == "b1k":
            binding = authority_binding(source)
            expected_source = binding.source_sha256
            geometry_source = target["geometry"].get(
                "scene_authority_sha256")
        elif dataset == "gs":
            binding = authority_binding(source)
            expected_source = binding.source_sha256
            geometry_source = target["geometry"].get(
                "geometry_authority_sha256")
            semantic_source = next(
                asset["sha256"] for asset in source["source_assets"]
                if asset["role"] == "semantic")
            if (target["geometry"].get("semantic_source_sha256") !=
                    semantic_source or
                    target["geometry"].get(
                        "alignment_certificate_sha256") !=
                    ((rec.get("gs_scene_capability") or {}).get(
                        "alignment_certificate") or {}).get("sha256")):
                return None, "b_target_invalid"
        else:
            return None, "strict_source_geometry_required"
    except (KeyError, StopIteration, TypeError, ValueError):
        return None, (
            "strict_b1k_main_contract_required"
            if dataset == "b1k" else
            "strict_gs_geometry_required"
            if dataset == "gs" else
            "strict_r2r_main_contract_required")
    selected = OBJ.select_b_target(selector_record)
    if (selected.get("eligible") is not True or
            selected != target.get("selection")):
        return None, "b_target_invalid"
    if geometry_source != expected_source:
        return None, "b_target_invalid"
    return target, "eligible"


def _b_direction(bearing_deg: float) -> tuple[str, float]:
    bearing = action_geometry.wrap_deg(float(bearing_deg))
    boundaries = (-135.0, -45.0, 45.0, 135.0)
    margin = min(abs(action_geometry.wrap_deg(bearing - boundary))
                 for boundary in boundaries)
    return action_geometry.bearing_sector(bearing), float(margin)


def b_direction_with_margin(bearing_deg: float) -> tuple[str, float]:
    """Quantize B2 only when strictly clear of every sector boundary."""
    label, margin = _b_direction(bearing_deg)
    if margin <= config.B_DIRECTION_BOUNDARY_MARGIN_DEG:
        raise ValueError("B2 bearing is too close to a sector boundary")
    return label, margin


def _explicit_endpoint(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("endpoint must be an object")
    keys = (("x_m", "z_m", "heading_deg")
            if "x_m" in value else ("x", "z", "heading_deg"))
    try:
        components = [value[key] for key in keys]
    except KeyError as error:
        raise ValueError("endpoint must contain a finite SE(2) pose") from error
    if any(isinstance(component, bool) or
           not isinstance(component, numbers.Real)
           for component in components):
        raise ValueError("endpoint must contain a finite SE(2) pose")
    result = {"x": float(components[0]), "z": float(components[1]),
              "heading_deg": float(components[2])}
    if not all(math.isfinite(component) for component in result.values()):
        raise ValueError("endpoint must contain a finite SE(2) pose")
    return result


def build_endpoint_relation_at_pose(
        *, pose: dict, endpoint: dict, b_target: dict) -> dict:
    """Project one explicit true endpoint onto the single B/D2 target."""
    if not b_target_atom_valid(b_target, pose=pose):
        raise ValueError("B target atom is invalid")
    return _build_endpoint_relation_from_validated_target(
        pose=pose, endpoint=endpoint, b_target=b_target)


def build_direction_relation_at_pose(
        *, pose: dict, checkpoint: dict, target_world_xz_m) -> dict:
    """Project one world-XZ target into a pose-local SE(2) checkpoint.

    ``checkpoint`` is expressed in the record's initial agent-ground frame;
    ``target_world_xz_m`` is expressed in the source world frame.  The result
    uses the same bearing and four-sector convention as B2.
    """
    checkpoint = _explicit_endpoint(checkpoint)
    checkpoint_world = local_ground_xz_to_world(
        pose, (checkpoint["x"], checkpoint["z"]))
    target = np.asarray(target_world_xz_m, dtype=np.float64)
    if target.shape != (2,) or not np.all(np.isfinite(target)):
        raise ValueError("direction target world XZ is invalid")
    terminal_yaw = (
        float(pose["yaw_rad"]) -
        math.radians(float(checkpoint["heading_deg"])))
    delta_x = float(target[0] - checkpoint_world[0])
    delta_z = float(target[1] - checkpoint_world[1])
    target_range = math.hypot(delta_x, delta_z)
    if target_range < config.B_TARGET_CENTROID_MIN_RANGE_M:
        bearing = direction = boundary_margin = None
        status = "undefined_centroid_range"
    else:
        cosine, sine = math.cos(terminal_yaw), math.sin(terminal_yaw)
        local_x = delta_x * cosine - delta_z * sine
        local_z = -delta_x * sine - delta_z * cosine
        bearing = action_geometry.wrap_deg(math.degrees(
            math.atan2(local_x, local_z)))
        direction, boundary_margin = _b_direction(bearing)
        status = "computed"
    return {
        "checkpoint_world_xz_m": [
            float(checkpoint_world[0]), float(checkpoint_world[1])],
        "checkpoint_heading_deg": float(checkpoint["heading_deg"]),
        "target_range_m": float(target_range),
        "direction_status": status,
        "bearing_deg": None if bearing is None else float(bearing),
        "direction": direction,
        "sector_boundary_margin_deg": boundary_margin,
    }


def _build_endpoint_relation_from_validated_target(
        *, pose: dict, endpoint: dict, b_target: dict,
        distance_after_m=None) -> dict:
    """Project an endpoint after the caller authenticated the fixed target."""
    endpoint = _explicit_endpoint(endpoint)
    after_xz = local_ground_xz_to_world(
        pose, (float(endpoint["x"]), float(endpoint["z"])))
    geometry = b_target["geometry"]
    if distance_after_m is None:
        distance_after = semantic.point_to_ground_support_distance_m(
            after_xz, geometry["ground_support"])
    else:
        if (isinstance(distance_after_m, bool) or
                not isinstance(distance_after_m, numbers.Real) or
                not math.isfinite(float(distance_after_m)) or
                float(distance_after_m) < 0.0):
            raise ValueError("B endpoint distance must be finite and nonnegative")
        distance_after = float(distance_after_m)
    direction_relation = build_direction_relation_at_pose(
        pose=pose, checkpoint=endpoint,
        target_world_xz_m=
            geometry["reference_centroid"]["world_xz_m"])
    centroid_range = direction_relation["target_range_m"]
    direction_status = direction_relation["direction_status"]
    bearing = direction_relation["bearing_deg"]
    direction = direction_relation["direction"]
    boundary_margin = direction_relation["sector_boundary_margin_deg"]
    value = {
        "schema": "b-endpoint-relation.v1",
        "target_geometry_sha256": geometry["sha256"],
        "target_instance_id": int(b_target["selection"]["instance_id"]),
        "endpoint_world_xz_m": [float(after_xz[0]), float(after_xz[1])],
        "endpoint_heading_deg": float(endpoint["heading_deg"]),
        "distance_before_m": float(b_target["initial_distance_m"]),
        "distance_after_m": float(distance_after),
        "distance_change_m": float(
            distance_after - b_target["initial_distance_m"]),
        "centroid_range_m": float(centroid_range),
        "direction_status": direction_status,
        "bearing_after_deg": (None if bearing is None else float(bearing)),
        "direction": direction,
        "sector_boundary_margin_deg": boundary_margin,
    }
    return {**value, "sha256": _canonical_sha256(value)}


def build_b_endpoint_relation_atom(
        *, pose: dict, outcome: dict, b_target: dict) -> dict:
    """Project one true completed endpoint onto the persisted B anchors."""
    if not outcome_fields.is_completed_clear(outcome):
        raise ValueError("B endpoint requires completed_clear execution")
    return build_endpoint_relation_at_pose(
        pose=pose, endpoint=_terminal_pose(outcome), b_target=b_target)


def build_b_endpoint_relation_from_validated_target(
        *, pose: dict, outcome: dict, b_target: dict,
        distance_after_m=None) -> dict:
    """Rebuild a relation after source authority authenticated ``b_target``."""
    if not outcome_fields.is_completed_clear(outcome):
        raise ValueError("B endpoint requires completed_clear execution")
    return _build_endpoint_relation_from_validated_target(
        pose=pose, endpoint=_terminal_pose(outcome), b_target=b_target,
        distance_after_m=distance_after_m)


def b_endpoint_relation_atom_valid(
        value: dict, *, pose: dict, outcome: dict, b_target: dict) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        rebuilt = build_b_endpoint_relation_atom(
            pose=pose, outcome=outcome, b_target=b_target)
    except (KeyError, TypeError, ValueError):
        return False
    return value == rebuilt




def r2r_v16_collection_contract(
        source_provenance: dict, collection_mode: str, *,
        record_schema_version: str = SCHEMA_VERSION) -> dict:
    """Build the hash-bound record route for strict R2R main oracles.

    v11 is the active R2R route.
    """
    source = source_provenance or {}
    mode = str(collection_mode)
    source_contract = dataset_contracts.dataset_source_contract("r2r")
    if mode != "main":
        raise ValueError("R2R v16 collection mode must be main")
    if (source.get("source_dataset") != "r2r" or
            source.get("official_split") != "train" or
            source.get("semantic_format") !=
            source_contract.semantic_format):
        raise ValueError(
            "R2R v16 collection requires train MP3D source provenance")
    schema = str(record_schema_version)
    if schema != SCHEMA_VERSION:
        raise ValueError("R2R collection record schema is unsupported")
    value = {
        "version": R2R_V16_COLLECTION_CONTRACT_VERSION,
        "record_schema_version": schema,
        "oracle_contract_version": ORACLE_CONTRACT_VERSION,
        "collection_mode": mode,
        "source_dataset": "r2r",
        "official_split": "train",
        "semantic_format": source_contract.semantic_format,
        "source_manifest_sha256": source.get("source_manifest_sha256"),
        "source_assets_sha256": source.get("source_assets_sha256"),
    }
    return {**value, "sha256": _canonical_sha256(value)}


def b1k_v16_collection_contract(
        source_provenance: dict, collection_mode: str, *,
        record_schema_version: str = SCHEMA_VERSION) -> dict:
    """Build the hash-bound v11 route for B1K main ABC1 collection."""
    if not isinstance(source_provenance, dict):
        raise ValueError("source provenance must be an object")
    source = source_provenance or {}
    mode = str(collection_mode)
    source_contract = dataset_contracts.dataset_source_contract("b1k")
    if mode != "main":
        raise ValueError("B1K collection mode must be main")
    if (source.get("source_dataset") != "b1k" or
            source.get("official_split") != "train" or
            source.get("split_authority") != "project_defined" or
            source.get("semantic_format") !=
            source_contract.semantic_format):
        raise ValueError(
            "B1K collection requires project-defined train OmniGibson "
            "source provenance")
    schema = str(record_schema_version)
    if schema != SCHEMA_VERSION:
        raise ValueError("B1K collection record schema must be conseq.v11")
    scene_authority_sha256 = b1k_scene_authority_sha256(source)
    value = {
        "version": B1K_V16_COLLECTION_CONTRACT_VERSION,
        "record_schema_version": schema,
        "oracle_contract_version": ORACLE_CONTRACT_VERSION,
        "collection_mode": mode,
        "source_dataset": "b1k",
        "official_split": "train",
        "split_authority": "project_defined",
        "semantic_format": source_contract.semantic_format,
        "source_manifest_sha256": source.get("source_manifest_sha256"),
        "source_assets_sha256": source.get("source_assets_sha256"),
        "scene_authority_sha256": scene_authority_sha256,
        "renderer_instance_protocol": B1K_RENDERER_INSTANCE_PROTOCOL,
        "observation_profile_sha256": B1K_OBSERVATION_PROFILE_SHA256,
        "c1_render_mode": B1K_C1_RENDER_MODE,
    }
    return {**value, "sha256": _canonical_sha256(value)}


def collection_contract(
        source_provenance: dict, collection_mode: str, *,
        record_schema_version: str = SCHEMA_VERSION) -> dict:
    """Dispatch a strict collection contract from trusted dataset identity."""
    if not isinstance(source_provenance, dict):
        raise ValueError("source provenance must be an object")
    source = source_provenance or {}
    dataset = str(source.get("source_dataset") or "")
    source_contract = dataset_contracts.dataset_source_contract(dataset)
    if not source_contract.main_collection_enabled:
        raise ValueError(
            f"unsupported strict collection source dataset {dataset!r}")
    if dataset == "r2r":
        return r2r_v16_collection_contract(
            source, collection_mode,
            record_schema_version=record_schema_version)
    if dataset == "b1k":
        return b1k_v16_collection_contract(
            source, collection_mode,
            record_schema_version=record_schema_version)
    raise ValueError(
        f"unsupported strict collection source dataset {dataset!r}")


def _base_rollout_payload(*, scene_id, position, yaw_rad, sensor,
                          outcome: dict) -> dict:
    return {
        "version": BASE_ROLLOUT_KEY_VERSION,
        "scene_id": str(scene_id),
        "pose": {
            "position": [float(value) for value in position],
            "yaw_rad": float(yaw_rad),
        },
        "sensor_profile": {
            "nominal_camera_offset_m":
                float(sensor["nominal_camera_offset_m"]),
            "hfov_deg": float(sensor["hfov_deg"]),
            "vfov_deg": float(sensor["vfov_deg"]),
        },
        "body": outcome.get("body") or {},
        "actions": outcome.get("actions") or [],
    }


def base_rollout_key(frame, outcome: dict) -> str:
    """Hash the target-independent physical rollout identity.
    The key names exactly the variables fixed by a base future state.  Target
    identity and any derived target measurement are intentionally absent so
    target-free A1/A2/A3/C1 cannot be multiplied by target count.
    """
    payload = _base_rollout_payload(
        scene_id=frame.scene_id,
        position=frame.position,
        yaw_rad=frame.yaw_rad,
        sensor={
            "nominal_camera_offset_m":
                frame.sensor.nominal_camera_offset_m,
            "hfov_deg": frame.sensor.hfov_deg,
            "vfov_deg": frame.sensor.vfov_deg,
        },
        outcome=outcome,
    )
    return _canonical_sha256(payload)


def stored_base_rollout_key(rec: dict, outcome: dict) -> str:
    """Recompute a base key from untrusted persisted record fields."""
    pose = rec.get("pose") or {}
    return _canonical_sha256(_base_rollout_payload(
        scene_id=rec.get("scene_id"),
        position=pose.get("position") or [],
        yaw_rad=pose.get("yaw_rad"),
        sensor=rec.get("sensor") or {},
        outcome=outcome,
    ))


def local_ground_xz_to_world(pose: dict, local_xz) -> tuple[float, float]:
    """Transform one pose-local ground point into Habitat world XZ."""
    position = np.asarray(pose.get("position"), dtype=np.float64)
    local = np.asarray(local_xz, dtype=np.float64)
    yaw = float(pose.get("yaw_rad"))
    if (position.shape != (3,) or not np.all(np.isfinite(position)) or
            local.shape != (2,) or not np.all(np.isfinite(local)) or
            not math.isfinite(yaw)):
        raise ValueError("record pose and local XZ must be finite")
    world = perception.world_from_local(
        np.array([[local[0], 0.0, local[1]]], dtype=np.float64),
        position, yaw,
    )[0]
    return float(world[0]), float(world[2])


def _exact_contact_instance_identity(
        frame, outcome: dict, collection_contract: dict | None, *,
        authority_binding: dataset_contracts.AuthorityBinding | None):
    """Return collection-time full-face identity, or ``None`` to withhold A3."""
    gs_identity = (
        isinstance(authority_binding, dataset_contracts.AuthorityBinding) and
        authority_binding.source_dataset == "gs")
    if ((not gs_identity and
         (collection_contract or {}).get("version") not in {
            R2R_V16_COLLECTION_CONTRACT_VERSION,
            B1K_V16_COLLECTION_CONTRACT_VERSION,
            }) or
            (outcome.get("physical") or {}).get("collision") is not True):
        return None
    consensus = outcome.get("oracle_consensus") or {}
    full_id = consensus.get("full_contact_instance_id")
    depth_id = consensus.get("depth_contact_instance_id")
    if (consensus.get("accepted") is not True or
            consensus.get("contact_instance_witness_required") is not True or
            not isinstance(full_id, int) or isinstance(full_id, bool) or
            full_id <= 0 or full_id != depth_id):
        return None
    visible = {
        int(value["instance_id"]): value
        for value in frame.objects
        if value.get("instance_id") is not None
    }
    witness = visible.get(int(full_id))
    certificate = outcome.get("shared_oracle_stability") or {}
    rows = certificate.get("rows") or []
    summary = certificate.get("summary") or {}
    exact = (rows[0].get("exact_contact_identity")
             if len(rows) == len(
                 consensus_fields.R2R_A_STABILITY_PERTURBATIONS) else None)
    if (witness is None or
            summary.get("contact_instance_stable") is not True or
            summary.get("contact_instance_id") != full_id or
            consensus_fields.a_stability_certificate_mismatches(
                outcome.get("actions") or [], certificate, outcome,
                authority_binding=authority_binding) or
            not consensus_fields.a3_exact_contact_identity_valid(
                exact, full_instance_id=full_id,
                depth_instance_id=depth_id,
                authority_binding=authority_binding)):
        return None
    if exact.get("category") != witness.get("category"):
        return None
    return dict(exact)


def _terminal_pose(outcome: dict) -> dict:
    return outcome_fields.realized_pose(outcome)


def _sampling_context(frame) -> dict:
    quality = frame.quality or {}
    clearance = float(quality.get("dist_to_obstacle_m", float("inf")))
    floor_ratio = float(quality.get("visible_floor_ratio", 0.0))
    object_count = sum(
        not obj.get("is_structural", False) for obj in frame.objects)
    return {
        "clearance": (
            "tight" if clearance < 0.5 else
            "near" if clearance < 1.5 else "open"),
        "visible_floor": (
            "low" if floor_ratio < 0.15 else
            "medium" if floor_ratio < 0.35 else "high"),
        "object_density": (
            "sparse" if object_count <= 1 else
            "moderate" if object_count <= 3 else "dense"),
    }


def json_value(o):
    """Convert arrays/scalars to strict JSON; unavailable numbers become null."""
    if isinstance(o, dict):
        return {k: json_value(v) for k, v in o.items()
                if not isinstance(k, str) or not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [json_value(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        value = float(o)
        return value if math.isfinite(value) else None
    if isinstance(o, np.ndarray):
        return json_value(o.tolist())
    return o


def _materialize_b_target(
        frame, objects: list, source_provenance, collection_contract_value):
    """Select once from s0, then verify only that selected exact instance."""
    contract = collection_contract_value or {}
    source = source_provenance or {}
    dataset = (
        source.get("source_dataset") if isinstance(source, dict) else None)
    if dataset == "r2r":
        if (contract.get("version") !=
                R2R_V16_COLLECTION_CONTRACT_VERSION or
                contract.get("collection_mode") != "main"):
            return None, None
        authority_role = "semantic"
        authority_keyword = "expected_semantic_ply_sha256"
    elif dataset == "b1k":
        try:
            expected_contract = collection_contract(source, "main")
        except (KeyError, StopIteration, TypeError, ValueError) as error:
            raise ValueError(
                "B1K B target requires the strict B1K main contract") \
                from error
        if contract != expected_contract:
            raise ValueError(
                "B1K B target requires the strict B1K main contract")
        authority_role = "scene_authority"
        authority_keyword = "expected_scene_authority_sha256"
    else:
        return None, None
    selected = OBJ.select_b_target({
        "objects": objects,
        "sensor": frame.sensor.to_dict(),
    })
    if selected.get("eligible") is not True:
        return None, {"reason": selected.get("reason", "no_visible_target")}
    instance_id = int(selected["instance_id"])
    authority_assets = [
        value for value in source.get("source_assets") or []
        if isinstance(value, dict) and value.get("role") == authority_role]
    authority = getattr(frame, "semantic_index", None)
    if (len(authority_assets) != 1 or authority is None or
            not hasattr(authority, "target_geometry_atom")):
        return None, {
            "reason": "exact_target_geometry_failed",
            "selected_instance_id": instance_id,
        }
    try:
        pose = {
            "position": list(frame.position),
            "yaw_rad": float(frame.yaw_rad),
        }
        if authority_keyword == "expected_semantic_ply_sha256":
            geometry = authority.target_geometry_atom(
                instance_id, frame.floor_plane,
                expected_semantic_ply_sha256=
                    authority_assets[0].get("sha256"),
                pose=pose)
        else:
            geometry = authority.target_geometry_atom(
                instance_id, frame.floor_plane,
                expected_scene_authority_sha256=
                    authority_assets[0].get("sha256"),
                pose=pose)
        target = build_b_target_atom(
            selection=selected, geometry=geometry, pose=pose)
    except (KeyError, MemoryError, OSError, TypeError, ValueError):
        return None, {
            "reason": "exact_target_geometry_failed",
            "selected_instance_id": instance_id,
        }
    return target, None


B1K_B_SELECTOR_PREDICATE_FACTS_SCHEMA = \
    "b1k-b-selector-predicate-facts.v1"


def _b1k_selector_predicate_facts(
        objects: list[dict], binding: dataset_contracts.AuthorityBinding
        ) -> dict:
    """Bind raw-taxonomy selector booleans without persisting raw labels."""
    if binding.source_dataset != "b1k":
        raise ValueError("B1K selector facts require a B1K authority binding")
    rows = []
    for obj in sorted(objects, key=lambda value: int(value["instance_id"])):
        instance_id = int(obj["instance_id"])
        category = str(obj.get("category") or "").strip()
        predicate_category = str(
            obj.get("_predicate_category") or category).strip()
        normalized = " ".join(predicate_category.casefold().split())
        rows.append({
            "instance_id": instance_id,
            "category": category,
            "is_structural": bool(config.is_structural(predicate_category)),
            "is_specific": bool(config.is_specific_semantic_category(
                predicate_category)),
            "excluded_material": bool(any(
                token in normalized
                for token in config.B_TARGET_EXCLUDED_MATERIAL_TOKENS)),
        })
    value = {
        "schema": B1K_B_SELECTOR_PREDICATE_FACTS_SCHEMA,
        "source_dataset": binding.source_dataset,
        "identity_schema": binding.identity_schema,
        "authoritative_source_role": binding.authoritative_source_role,
        "source_sha256": binding.source_sha256,
        "rows": rows,
    }
    return {**value, "sha256": canonical_atom_sha256(value)}


def _b1k_selector_objects_from_facts(
        objects: list[dict], facts: dict,
        binding: dataset_contracts.AuthorityBinding) -> list[dict]:
    """Validate source-bound selector facts and restore private booleans.

    The canonical digest catches corruption and implementation drift; it is
    not an adversarial signature.  Arbitrary in-process record re-signing is
    outside the repository threat model.  Record validation separately
    replays selection so even a validly resealed but behaviorally inconsistent
    atom cannot disagree with the persisted target/withhold outcome.
    """
    if not isinstance(facts, dict):
        raise ValueError("B1K selector predicate facts are missing")
    if set(facts) != {
            "schema", "source_dataset", "identity_schema",
            "authoritative_source_role", "source_sha256", "rows",
            "sha256"}:
        raise ValueError("B1K selector predicate fact fields are invalid")
    body = {key: value for key, value in facts.items() if key != "sha256"}
    if (facts.get("sha256") != canonical_atom_sha256(body) or
            facts.get("schema") != B1K_B_SELECTOR_PREDICATE_FACTS_SCHEMA or
            facts.get("source_dataset") != binding.source_dataset or
            facts.get("identity_schema") != binding.identity_schema or
            facts.get("authoritative_source_role") !=
            binding.authoritative_source_role or
            facts.get("source_sha256") != binding.source_sha256):
        raise ValueError("B1K selector predicate facts are unauthenticated")
    rows = facts.get("rows")
    if not isinstance(rows, list) or len(rows) != len(objects):
        raise ValueError("B1K selector predicate fact rows are invalid")
    by_id = {}
    observed_ids = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {
                "instance_id", "category", "is_structural", "is_specific",
                "excluded_material"}):
            raise ValueError("B1K selector predicate fact row is invalid")
        instance_id = row.get("instance_id")
        if (not isinstance(instance_id, int) or isinstance(instance_id, bool) or
                instance_id <= 0 or instance_id in by_id or
                not isinstance(row.get("category"), str) or
                not row["category"] or
                any(not isinstance(row.get(key), bool) for key in (
                    "is_structural", "is_specific", "excluded_material"))):
            raise ValueError("B1K selector predicate fact row is invalid")
        by_id[instance_id] = row
        observed_ids.append(instance_id)
    if observed_ids != sorted(observed_ids):
        raise ValueError("B1K selector predicate fact rows are not canonical")
    restored = []
    for obj in objects:
        instance_id = obj.get("instance_id")
        row = by_id.get(instance_id)
        if (row is None or row.get("category") != obj.get("category") or
                row.get("is_structural") != obj.get("is_structural")):
            raise ValueError(
                "B1K selector predicate facts disagree with objects")
        restored.append({
            **obj,
            "_predicate_is_specific": row["is_specific"],
            "_predicate_excluded_material": row["excluded_material"],
        })
    return restored


def _materialize_b_target_v18(
        frame, objects: list, source_provenance
        ) -> tuple[dict | None, dict | None]:
    """Materialize the GS visible-anchor B track from initial RGB-D."""
    source = source_provenance or {}
    try:
        binding = dataset_contracts.resolve_authority_binding(source)
    except (KeyError, StopIteration, TypeError, ValueError):
        return None, {"reason": "exact_target_geometry_failed"}
    selected = OBJ.select_b_target({
        "objects": objects,
        "sensor": frame.sensor.to_dict(),
    })
    if selected.get("eligible") is not True:
        return None, {"reason": selected.get("reason", "no_visible_target")}
    instance_id = int(selected["instance_id"])
    authority = getattr(frame, "semantic_index", None)
    target_geometry = getattr(authority, "target_geometry_atom", None)
    if not callable(target_geometry):
        return None, {
            "reason": "exact_target_geometry_failed",
            "selected_instance_id": instance_id,
        }
    pose = {
        "position": list(frame.position),
        "yaw_rad": float(frame.yaw_rad),
    }
    try:
        if binding.source_dataset == "gs":
            geometry = target_geometry(
                instance_id, frame.floor_plane,
                expected_geometry_authority_sha256=binding.source_sha256,
                pose=pose)
        else:
            raise ValueError("v18 B target authority is not certified")
        target = build_b_target_atom(
            selection=selected, geometry=geometry, pose=pose)
    except (KeyError, MemoryError, OSError, TypeError, ValueError):
        return None, {
            "reason": "exact_target_geometry_failed",
            "selected_instance_id": instance_id,
        }
    return target, None


def _build_record(frame, outcomes: List[dict], *,
                  record_schema_version: str,
                  image_path: str, floor_calibration, depth_path=None,
                  intervention=None, selection=None,
                  source_provenance=None, collection_contract=None) -> dict:
    """Assemble one private frame record.
    ``floor_calibration`` is the pose's ``FloorPlaneFitResult``. It is required
    and stored in full: the plane decides target eligibility, the obstacle band
    and the published height, so a record that cannot be re-derived from its own
    stored calibration cannot be audited.
    """
    if not isinstance(floor_calibration, FloorPlaneFitResult):
        raise TypeError("build_record needs the pose's FloorPlaneFitResult")
    if floor_calibration.estimate != frame.floor_plane:
        raise ValueError(
            "the stored calibration is not the plane this frame was built with")
    objs = [dict(value) for value in frame.objects]
    if record_schema_version == V18_SCHEMA_VERSION:
        points = np.asarray(frame.pts, dtype=np.float64)
        labels = np.asarray(frame.pts_sem)
        for obj in objs:
            instance_id = int(obj["instance_id"])
            obj["initial_ground_support"] = (
                OBJ.initial_ground_support_grid(
                    points[labels == instance_id], frame.floor_plane))
    v18_capability = None
    gs_collision_binding = None
    gs_scene_capability = None
    source_dataset = None
    if record_schema_version == V18_SCHEMA_VERSION:
        source_dataset = _require_v18_gs_source_provenance(
            source_provenance, scene_id=frame.scene_id)
        if collection_contract is not None:
            raise ValueError(
                "v18 collection contracts are not defined; do not reuse a "
                "legacy contract")
        v18_capability = _authority_surface_capability(source_dataset)
        gs_collision_binding = \
            dataset_contracts.resolve_gs_collision_binding(
                source_provenance)
        gs_scene_capability = gs_semantic.scene_capability_atom(
            source_provenance,
            getattr(
                getattr(frame, "semantic_index", None),
                "alignment_certificate", None),
        )
    n_nonstruct = sum(0 if o["is_structural"] else 1 for o in objs)
    b_available = (
        v18_capability is None or
        all(
            v18_capability["task_statuses"][task]["status"] == "available"
            for task in ("B1", "B2"))
    )
    if gs_scene_capability is not None:
        b_available = all(
            gs_semantic.scene_task_available(
                gs_scene_capability, source_provenance, task)
            for task in ("B1", "B2"))
    b_selector_predicate_facts = None
    if b_available:
        if record_schema_version == V18_SCHEMA_VERSION:
            b_target, b_target_withhold = _materialize_b_target_v18(
                frame, objs, source_provenance)
        else:
            b_target, b_target_withhold = _materialize_b_target(
                frame, objs, source_provenance, collection_contract)
    else:
        b_target, b_target_withhold = None, None
    identity_authority_binding = None
    if ((collection_contract or {}).get("version") ==
            R2R_V16_COLLECTION_CONTRACT_VERSION):
        identity_authority_binding = authority_binding(source_provenance)
    elif ((collection_contract or {}).get("version") ==
          B1K_V16_COLLECTION_CONTRACT_VERSION):
        identity_authority_binding = authority_binding(source_provenance)
    elif (record_schema_version == V18_SCHEMA_VERSION and
          source_dataset == "gs"):
        identity_authority_binding = authority_binding(source_provenance)
    stored_outcomes = []
    for outcome in outcomes:
        if gs_collision_binding is not None:
            physical = outcome.get("physical") or {}
            if (physical.get("authority") != "gs_collision_mesh" or
                    physical.get("geometry_authority_sha256") !=
                    gs_collision_binding.collision_authority_sha256):
                raise ValueError(
                    "GS outcome collision authority is not bound")
        stored = dict(outcome)
        stored["execution"] = dict(outcome.get("execution") or {})
        stored["execution"]["execution_regime"] = (
            outcome_fields.derive_execution_regime(outcome))
        base_key = base_rollout_key(frame, stored)
        stored["base_rollout_key"] = base_key
        contact_identity = _exact_contact_instance_identity(
            frame, stored, collection_contract,
            authority_binding=identity_authority_binding)
        if contact_identity is not None:
            stored["contact_instance_identity"] = contact_identity
        if b_target is not None and outcome_fields.is_completed_clear(stored):
            try:
                stored["b_endpoint_relation"] = \
                    build_b_endpoint_relation_atom(
                        pose={
                            "position": list(frame.position),
                            "yaw_rad": float(frame.yaw_rad),
                        },
                        outcome=stored, b_target=b_target)
            except (KeyError, TypeError, ValueError):
                stored["b_endpoint_relation_withhold"] = \
                    "endpoint_relation_invalid"
        stored_outcomes.append(stored)
    rec = {
        "schema_version": record_schema_version,
        "oracle_contract_version": ORACLE_CONTRACT_VERSION,
        "substrate": {
            "base_rollout_key_version": BASE_ROLLOUT_KEY_VERSION,
        },
        "frame_id": frame.frame_id,
        "scene_id": frame.scene_id,
        "scene_glb": frame.scene_glb,
        "pose": {"position": list(frame.position), "yaw_rad": frame.yaw_rad},
        "sensor": frame.sensor.to_dict(),
        # Private provenance. The public height is derived from this plane and
        # the nominal offset; neither the plane nor the offset may be published.
        "floor_calibration": floor_calibration.to_json(),
        "camera_height_above_visible_floor_m":
            frame.camera_height_above_visible_floor_m,
        "image_path": image_path,
        "depth_path": depth_path,
        "quality": frame.quality,
        "n_objects": len(objs),
        "n_nonstructural": n_nonstruct,
        "category_inventory": frame.category_inventory,
        "objects": objs,
        "outcomes": stored_outcomes,
        "sampling_context": _sampling_context(frame),
    }
    if intervention is not None:
        rec["intervention"] = intervention
    if selection is not None:
        rec["selection"] = selection
    if source_provenance is not None:
        rec["source"] = dict(source_provenance)
    if collection_contract is not None:
        rec["collection_contract"] = dict(collection_contract)
    if b_target is not None:
        rec["b_target"] = b_target
    if b_target_withhold is not None:
        rec["b_target_withhold"] = b_target_withhold
    if v18_capability is not None:
        rec["authority_surface_capability"] = v18_capability
    if gs_collision_binding is not None:
        rec["gs_collision_authority"] = \
            dataset_contracts.gs_collision_binding_atom(gs_collision_binding)
        expected_plane = {
            "normal_local": [0.0, 1.0, 0.0],
            "offset_m": 0.0,
        }
        if not _type_exact_json_equal(
                rec["floor_calibration"].get("estimate"), expected_plane):
            raise ValueError(
                "GS floor calibration must use the source-bound level "
                "floor reference")
        rec["gs_floor_reference_authority"] = \
            dataset_contracts.gs_collision_floor_reference_atom(
                gs_collision_binding,
                ground_y_m=float(frame.position[1]))
        rec["gs_scene_capability"] = gs_scene_capability
    if b_selector_predicate_facts is not None:
        rec["b_selector_predicate_facts"] = b_selector_predicate_facts
    return json_value(rec)


def build_record(frame, outcomes: List[dict], *,
                 image_path: str, floor_calibration, depth_path=None,
                 intervention=None, selection=None,
                 source_provenance=None, collection_contract=None) -> dict:
    """Build the active v11 record."""
    return _build_record(
        frame, outcomes, record_schema_version=SCHEMA_VERSION,
        image_path=image_path, floor_calibration=floor_calibration,
        depth_path=depth_path, intervention=intervention, selection=selection,
        source_provenance=source_provenance,
        collection_contract=collection_contract)


def build_record_v18(frame, outcomes: List[dict], *,
                     image_path: str, floor_calibration, depth_path=None,
                     intervention=None, selection=None,
                     source_provenance=None, collection_contract=None) -> dict:
    """Build the GS-only v18 record backed by official collision geometry."""
    return _build_record(
        frame, outcomes, record_schema_version=V18_SCHEMA_VERSION,
        image_path=image_path, floor_calibration=floor_calibration,
        depth_path=depth_path, intervention=intervention, selection=selection,
        source_provenance=source_provenance,
        collection_contract=collection_contract)


def decode_records(payload: bytes | str) -> list[dict]:
    """Decode one immutable JSONL snapshot under the current contract."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        source = value.get("schema_version", "conseq.v1")
        if source != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported record schema {source}; recollect as "
                f"{SCHEMA_VERSION}")
        contract = value.get("oracle_contract_version")
        if contract != ORACLE_CONTRACT_VERSION:
            raise ValueError(
                "unsupported oracle contract "
                f"{contract!r}; recollect as {ORACLE_CONTRACT_VERSION}")
        records.append(value)
    return records


def validate_record_v18(value: dict) -> list[str]:
    """Validate the GS-only official-collision record envelope."""
    if not isinstance(value, dict):
        return ["v18 record must be an object"]
    errors = []
    if value.get("schema_version") != V18_SCHEMA_VERSION:
        errors.append(
            f"unsupported record schema {value.get('schema_version')}; "
            f"recollect as {V18_SCHEMA_VERSION}")
    if value.get("oracle_contract_version") != ORACLE_CONTRACT_VERSION:
        errors.append(
            "unsupported oracle contract "
            f"{value.get('oracle_contract_version')!r}; recollect as "
            f"{ORACLE_CONTRACT_VERSION}")
    binding = None
    try:
        _require_v18_gs_source_provenance(
            value.get("source"), scene_id=value.get("scene_id"))
        binding = dataset_contracts.resolve_gs_collision_binding(
            value["source"])
    except (AttributeError, KeyError, StopIteration, TypeError,
            ValueError) as error:
        errors.append(f"v18 GS source provenance is invalid: {error}")
    expected_capability = _authority_surface_capability("gs")
    if not _type_exact_json_equal(
            value.get("authority_surface_capability"), expected_capability):
        errors.append(
            "authority surface capability differs from exact recomputation")
    if binding is not None:
        expected_atom = dataset_contracts.gs_collision_binding_atom(binding)
        if not _type_exact_json_equal(
                value.get("gs_collision_authority"), expected_atom):
            errors.append(
                "GS collision authority differs from exact recomputation")
        try:
            gs_semantic.validate_scene_capability_atom(
                value.get("gs_scene_capability"), value.get("source"))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"GS scene capability is invalid: {error}")
        try:
            ground_y = value["pose"]["position"][1]
            expected_floor = \
                dataset_contracts.gs_collision_floor_reference_atom(
                    binding, ground_y_m=ground_y)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            errors.append(f"GS floor reference authority is invalid: {error}")
        else:
            if not _type_exact_json_equal(
                    value.get("gs_floor_reference_authority"),
                    expected_floor):
                errors.append(
                    "GS floor reference authority differs from exact "
                    "recomputation")
            if not _type_exact_json_equal(
                    (value.get("floor_calibration") or {}).get("estimate"),
                    expected_floor["plane_local"]):
                errors.append(
                    "GS floor calibration differs from official collision "
                    "floor reference")
    for index, outcome in enumerate(value.get("outcomes") or []):
        physical = (
            outcome.get("physical") if isinstance(outcome, dict) else None)
        if not isinstance(physical, dict) or binding is None:
            errors.append(
                f"GS collision physical authority is missing at outcome "
                f"{index}")
        elif (physical.get("authority") != "gs_collision_mesh" or
              physical.get("geometry_authority_sha256") !=
              binding.collision_authority_sha256):
            errors.append(
                f"GS collision physical authority differs at outcome {index}")
    if "collection_contract" in value:
        errors.append("v18 records must not carry a legacy collection contract")
    if "gs_geometry_authority" in value:
        errors.append("v18 records must not carry Gaussian physical authority")
    if "b_selector_predicate_facts" in value:
        errors.append("v18 GS records must not carry B1K selector facts")
    objects = value.get("objects")
    if not isinstance(objects, list):
        errors.append("v18 objects must be a list")
    else:
        seen = set()
        for index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                errors.append(f"v18 object {index} must be an object")
                continue
            instance_id = obj.get("instance_id")
            if (not isinstance(instance_id, int) or
                    isinstance(instance_id, bool) or instance_id <= 0 or
                    instance_id in seen):
                errors.append(
                    f"v18 object {index} instance id is invalid or duplicated")
            else:
                seen.add(instance_id)
            errors.extend(
                f"v18 object {index} {error}" for error in
                OBJ.initial_ground_support_errors(
                    obj.get("initial_ground_support")))
    if "b_target" in value:
        if "b_target_withhold" in value:
            errors.append(
                "v18 record carries both B target and B target withhold")
        target, reason = authenticated_b_target(value)
        if target is None:
            errors.append("v18 B target is not authenticated: " + str(reason))
    return errors


def decode_v18_records(payload: bytes | str) -> list[dict]:
    """Decode and locally validate GS official-collision JSONL snapshots."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        errors = validate_record_v18(value)
        if errors:
            raise ValueError("invalid conseq.v18 record: " + "; ".join(errors))
        records.append(value)
    return records


def decode_records_for_schema(
        payload: bytes | str, schema_version: str) -> list[dict]:
    """Decode under one exact authenticated schema; never guess from rows."""
    decoders = {
        SCHEMA_VERSION: decode_records,
        V18_SCHEMA_VERSION: decode_v18_records,
    }
    try:
        decoder = decoders[schema_version]
    except KeyError as error:
        raise ValueError(
            f"candidate source schema is unsupported: {schema_version!r}") \
            from error
    return decoder(payload)


def read_records(path: str) -> Iterator[dict]:
    with open(path, "rb") as handle:
        yield from decode_records(handle.read())
