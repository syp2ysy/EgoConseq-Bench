"""Pure structured rollout helpers used by :func:`pipeline.consequence.judge`."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np

from pipeline import actions as A, config, perception
from pipeline.geometry import EXCLUDED_GEOMETRY_SOURCES


EVIDENCE_PROTOCOL_VERSION = "swept_floor_v6"
# v5 differs only in the near-field exemption: it used the straight-ahead
# wedge at every bearing, so a turned body's own flank counted as missing
# evidence. Frozen v5 artefacts stay verifiable as history; only v6 is emitted.
SUPPORTED_EVIDENCE_PROTOCOLS = ("swept_floor_v5", EVIDENCE_PROTOCOL_VERSION)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

@dataclass(frozen=True)
class _PoseGeometryQuery:
    navigable: bool
    clearance_m: float
    obstacle_index: int | None = None
    geometry_source: str | None = None


@dataclass(frozen=True)
class PhysicalPathTrace:
    """Coarse path and contact refinement shared by precheck and rollout."""

    authority: str
    collision: bool | None
    collision_source: str | None
    first_contact_arc_m: float | None
    stop_arc_m: float | None
    minimum_clearance_m: float | None
    geometry_authority_sha256: str | None

    def precheck(self) -> dict:
        value = {
            "authority": self.authority,
            "collision": self.collision,
            "first_contact_arc_m": self.first_contact_arc_m,
            "minimum_clearance_m": self.minimum_clearance_m,
        }
        if self.geometry_authority_sha256 is not None:
            value["geometry_authority_sha256"] = \
                self.geometry_authority_sha256
        if self.collision_source is not None:
            value["collision_source"] = self.collision_source
        return value

    def __getitem__(self, key: str):
        if key == "authority":
            return self.authority
        if key == "collision":
            return self.collision
        if key == "first_contact_arc_m":
            return self.first_contact_arc_m
        if key == "minimum_clearance_m":
            return self.minimum_clearance_m
        if key == "geometry_authority_sha256" and \
                self.geometry_authority_sha256 is not None:
            return self.geometry_authority_sha256
        if key == "collision_source" and self.collision_source is not None:
            return self.collision_source
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _query_pose(nav, pose) -> _PoseGeometryQuery:
    query = getattr(nav, "query_pose", None)
    if query is not None:
        return query(pose)
    navigable = bool(nav.is_navigable(pose))
    return _PoseGeometryQuery(
        navigable=navigable,
        clearance_m=(
            float(nav.clearance(pose)) if navigable else 0.0),
    )


def _query_many(nav, poses) -> list[_PoseGeometryQuery]:
    values = list(poses)
    query = getattr(nav, "query_many", None)
    if query is not None:
        return list(query(values))
    return [_query_pose(nav, pose) for pose in values]


def _excluded_geometry_source(query) -> str | None:
    source = getattr(query, "geometry_source", None)
    return source if source in EXCLUDED_GEOMETRY_SOURCES else None


def _geometry_authority_fields(nav, authority: str) -> dict:
    """Require the official GS collision binding in physical evidence."""
    if authority != "gs_collision_mesh":
        return {}
    digest = getattr(nav, "geometry_authority_sha256", None)
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("GS geometry authority binding is missing or invalid")
    return {"geometry_authority_sha256": digest}


def _unavailable_physical_rollout(
        authority: str, actions: Sequence[A.Action], checkpoints, *,
        geometry_source: str | None = None,
        geometry_authority_fields: dict | None = None) -> dict:
    nominal_arc = A.total_forward_m(actions)
    nominal_pose = A.pose_at_progress(actions, 1.0)
    cps = [{
        "requested_progress": float(progress),
        "realized_progress": float(progress),
        "arc_m": None,
        "pose": pose_dict(A.pose_at_progress(actions, progress)),
        "clearance_m": None,
    } for progress in checkpoints]
    physical = {
        "authority": authority,
        "collision": None,
        "minimum_clearance_m": None,
        "first_contact_arc_m": None,
        "contact_action_index": None,
        "contact_action_local_arc_m": None,
        "contact": None,
        **(geometry_authority_fields or {}),
    }
    if geometry_source is not None:
        physical["collision_source"] = geometry_source
    return {
        "execution": {
            "completed": None,
            "stop_reason": "physical_gt_unavailable",
            "executed_forward_fraction": None,
            "animation_stop_time_fraction": None,
            "nominal_forward_m": nominal_arc,
            "executed_forward_m": None,
            "executed_turn_deg": None,
            "executed_forward_after_turn_m": None,
            "stop_arc_m": None,
            "nominal_pose": pose_dict(nominal_pose),
            "realized_pose": None,
        },
        "checkpoints": cps,
        "physical": physical,
    }


def pose_dict(pose) -> dict:
    return {"x": float(pose[0]), "z": float(pose[1]), "heading_deg": float(pose[2])}


def summarize_execution(actions: Sequence[A.Action], *, collision: bool,
                        stop_arc_m: float | None) -> dict:
    """Reconstruct the executed action prefix and its terminal pose.
    ``stop_arc_m`` is the *last traversable* forward arc-length (the ``lo``
    endpoint of the contact binary search); it governs where execution
    terminates and is deliberately distinct from ``first_contact_arc_m`` (the
    first *non*-traversable position, used only for contact attribution). The
    summary is derived purely from ``actions`` + ``stop_arc_m`` and never trusts
    a stored action index, so :mod:`pipeline.validate` can independently
    re-derive and cross-check the persisted execution fields.
    Semantics (kept consistent with :func:`pipeline.actions._fold`):
    * safe (``collision=False``): every action executes, *including trailing
      Turns*; ``executed_turn_deg`` is the raw net turn (``net_turn_deg``) and
      the realized pose is :func:`pipeline.actions.pose_after`.
    * forward-leg collision: execution stops inside (or at the closing boundary
      of) the forward leg that first becomes non-traversable; any Turn *after*
      that leg is not executed. ``executed_turn_deg`` sums only the Turns up to
      that leg and the realized pose is :func:`pipeline.actions.pose_at_arc`
      at ``stop_arc_m``.
    """
    budget = math.inf
    if collision:
        if stop_arc_m is None:
            raise ValueError("collision execution summary requires stop_arc_m")
        budget = float(stop_arc_m)
    executed_turn_deg = 0.0
    executed_forward_after_turn_m = 0.0
    arc = 0.0
    turn_seen = False
    for action in actions:
        if isinstance(action, A.Turn):
            executed_turn_deg += float(action.deg)
            turn_seen = True
            continue
        remaining = budget - arc
        if float(action.m) >= remaining:  # reaches / passes the stop point here
            if turn_seen:
                executed_forward_after_turn_m += max(0.0, remaining)
            break  # later actions (incl. trailing Turns) never execute
        if turn_seen:
            executed_forward_after_turn_m += float(action.m)
        arc += float(action.m)
    realized_pose = (A.pose_at_arc(actions, budget) if collision
                     else A.pose_after(actions))
    return {
        "executed_turn_deg": float(executed_turn_deg),
        "executed_forward_after_turn_m": float(executed_forward_after_turn_m),
        "realized_pose": realized_pose,
    }


def _physical_path_trace_from_queries(
        nav, actions: Sequence[A.Action], samples, queries,
        ) -> PhysicalPathTrace:
    """Preserve scalar contact refinement after a coarse batched query."""
    authority = getattr(nav, "authority", "unavailable")
    authority_fields = _geometry_authority_fields(nav, authority)
    authority_digest = authority_fields.get("geometry_authority_sha256")
    previous_arc = 0.0
    clearances = []
    for sample, query in zip(samples, queries):
        _x, _z, _heading_rad, arc = sample
        excluded_source = _excluded_geometry_source(query)
        if excluded_source is not None:
            return PhysicalPathTrace(
                authority=authority, collision=None,
                collision_source=excluded_source,
                first_contact_arc_m=None, stop_arc_m=None,
                minimum_clearance_m=None,
                geometry_authority_sha256=authority_digest)
        if query.navigable:
            clearances.append(max(0.0, float(query.clearance_m)))
            previous_arc = arc
            continue
        lo, hi = previous_arc, arc
        contact_query = query
        for _ in range(config.CONTACT_REFINE_ITERS):
            mid = (lo + hi) / 2.0
            mid_query = _query_pose(nav, A.pose_at_arc(actions, mid))
            excluded_source = _excluded_geometry_source(mid_query)
            if excluded_source is not None:
                return PhysicalPathTrace(
                    authority=authority, collision=None,
                    collision_source=excluded_source,
                    first_contact_arc_m=None, stop_arc_m=None,
                    minimum_clearance_m=None,
                    geometry_authority_sha256=authority_digest)
            if mid_query.navigable:
                lo = mid
            else:
                hi = mid
                contact_query = mid_query
        source = getattr(contact_query, "geometry_source", None)
        return PhysicalPathTrace(
            authority=authority, collision=True, collision_source=source,
            first_contact_arc_m=float(hi), stop_arc_m=float(lo),
            minimum_clearance_m=0.0,
            geometry_authority_sha256=authority_digest)
    minimum_clearance = (
        float(min(clearances)) if clearances else
        max(0.0, float(
            _query_pose(nav, (0.0, 0.0, 0.0)).clearance_m)))
    return PhysicalPathTrace(
        authority=authority, collision=False, collision_source=None,
        first_contact_arc_m=None, stop_arc_m=None,
        minimum_clearance_m=minimum_clearance,
        geometry_authority_sha256=authority_digest)


def physical_path_trace(
        nav, actions: Sequence[A.Action]) -> PhysicalPathTrace:
    """Walk and refine a path without materializing rollout checkpoints."""
    authority = (
        getattr(nav, "authority", "unavailable")
        if nav is not None else "unavailable")
    if nav is None or authority == "unavailable":
        return PhysicalPathTrace(
            authority="unavailable", collision=None, collision_source=None,
            first_contact_arc_m=None, stop_arc_m=None,
            minimum_clearance_m=None, geometry_authority_sha256=None)
    samples = A.sample_path(actions, config.MARCH_STEP_M)
    queries = _query_many(nav, [
        (x, z, math.degrees(heading_rad))
        for x, z, heading_rad, _arc in samples])
    return _physical_path_trace_from_queries(nav, actions, samples, queries)


def physical_path_traces(
        nav, action_programs: Sequence[Sequence[A.Action]],
        ) -> list[PhysicalPathTrace]:
    """Batch coarse geometry for independent paths on one bound B1K nav."""
    programs = list(action_programs)
    authority = (
        getattr(nav, "authority", "unavailable")
        if nav is not None else "unavailable")
    if nav is None or authority == "unavailable":
        return [physical_path_trace(nav, actions) for actions in programs]
    sample_groups = [
        A.sample_path(actions, config.MARCH_STEP_M) for actions in programs]
    poses = [
        (x, z, math.degrees(heading_rad))
        for samples in sample_groups
        for x, z, heading_rad, _arc in samples
    ]
    queries = _query_many(nav, poses)
    traces = []
    offset = 0
    for actions, samples in zip(programs, sample_groups):
        end = offset + len(samples)
        traces.append(_physical_path_trace_from_queries(
            nav, actions, samples, queries[offset:end]))
        offset = end
    return traces


def physical_collision_precheck(
        nav, actions: Sequence[A.Action], *,
        path_trace: PhysicalPathTrace | None = None) -> dict:
    """Return collision and clearance fields needed by candidate gates."""
    trace = path_trace or physical_path_trace(nav, actions)
    return trace.precheck()


def physical_rollout(
        nav, actions: Sequence[A.Action],
        checkpoints=config.CHECKPOINT_PROGRESS, *,
        path_trace: PhysicalPathTrace | None = None) -> dict:
    """Roll a circular footprint through a radius-conditioned navigation oracle."""
    trace = path_trace or physical_path_trace(nav, actions)
    authority = trace.authority
    authority_fields = (
        {"geometry_authority_sha256": trace.geometry_authority_sha256}
        if trace.geometry_authority_sha256 is not None else {})
    if trace.collision is None:
        return _unavailable_physical_rollout(
            authority, actions, checkpoints,
            geometry_source=trace.collision_source,
            geometry_authority_fields=authority_fields)
    nominal_arc = A.total_forward_m(actions)
    nominal_pose = A.pose_at_progress(actions, 1.0)
    collision = bool(trace.collision)
    contact_arc = trace.first_contact_arc_m
    stop_arc = trace.stop_arc_m
    collision_source = trace.collision_source
    contact_index = contact_local = None
    stop_progress = 1.0
    if collision:
        contact_index, contact_local = A.contact_action_index(actions, contact_arc)
        stop_index, stop_local = A.contact_action_index(actions, stop_arc)
        stop_progress = A.program_progress_at_contact(
            actions, stop_index, stop_local)
    executed_arc = stop_arc if collision else nominal_arc
    executed_forward_fraction = (
        float(executed_arc / nominal_arc) if nominal_arc > 0.0 else 1.0)
    summary = summarize_execution(
        actions, collision=collision, stop_arc_m=stop_arc)
    executed_turn_deg = summary["executed_turn_deg"]
    executed_forward_after_turn_m = summary["executed_forward_after_turn_m"]
    realized_pose = summary["realized_pose"]
    checkpoint_specs = []
    for requested in checkpoints:
        realized = min(float(requested), stop_progress)
        pose = A.pose_at_progress(actions, realized)
        checkpoint_specs.append((requested, realized, pose))
    checkpoint_queries = _query_many(
        nav, [value[2] for value in checkpoint_specs])
    cps = []
    for (requested, realized, pose), query in zip(
            checkpoint_specs, checkpoint_queries):
        cps.append({
            "requested_progress": float(requested),
            "realized_progress": float(realized),
            "arc_m": float(min(executed_arc, A.arc_at_progress(actions, realized))),
            "pose": pose_dict(pose),
            "clearance_m": max(0.0, float(query.clearance_m)),
        })
    contact = None
    if collision:
        contact_pose = A.pose_at_arc(actions, contact_arc)
        hit = nav.closest_obstacle(contact_pose) or {}
        contact = {
            "center_local": [float(contact_pose[0]), float(contact_pose[1])],
            "world_point": hit.get("world_point"),
            "world_normal": hit.get("world_normal"),
            "distance_m": hit.get("distance_m"),
            "configuration_boundary_world_point": hit.get(
                "configuration_boundary_world_point"),
            "configuration_boundary_distance_m": hit.get(
                "configuration_boundary_distance_m"),
            "surface_protocol": hit.get("surface_protocol"),
        }
        if "obstacle_identity" in hit:
            contact["obstacle_identity"] = hit["obstacle_identity"]
        if "gaussian_index" in hit:
            contact["geometry_element_index"] = int(
                hit["gaussian_index"])
    physical_result = {
        "authority": authority,
        **authority_fields,
        "collision": bool(collision),
        "minimum_clearance_m": trace.minimum_clearance_m,
        "first_contact_arc_m": (
            float(contact_arc) if contact_arc is not None else None),
        "contact_action_index": contact_index,
        "contact_action_local_arc_m": contact_local,
        "contact": contact,
    }
    if collision_source is not None:
        physical_result["collision_source"] = collision_source
    return {
        "execution": {
            "completed": not collision,
            "stop_reason": "collision" if collision else "completed",
            "executed_forward_fraction": executed_forward_fraction,
            "animation_stop_time_fraction": float(stop_progress),
            "nominal_forward_m": nominal_arc,
            "executed_forward_m": float(executed_arc),
            "executed_turn_deg": executed_turn_deg,
            "executed_forward_after_turn_m": float(executed_forward_after_turn_m),
            "stop_arc_m": float(stop_arc) if stop_arc is not None else None,
            "nominal_pose": pose_dict(nominal_pose),
            "realized_pose": pose_dict(realized_pose),
        },
        "checkpoints": cps,
        "physical": physical_result,
    }


def compose_pose(base_pose, local_pose):
    """Compose two project-convention local SE(2) poses exactly once."""
    bx, bz, base_heading_deg = (float(value) for value in base_pose)
    x, z, heading_deg = (float(value) for value in local_pose)
    heading = math.radians(base_heading_deg)
    return (
        bx + x * math.cos(heading) + z * math.sin(heading),
        bz - x * math.sin(heading) + z * math.cos(heading),
        A.wrap_deg(base_heading_deg + heading_deg),
    )


_compose_pose = compose_pose


def realized_corridor_arc_m(full_physical: dict) -> float | None:
    """Return the physical prefix whose public visibility must be certified."""
    if full_physical.get("collision") is not True:
        return None
    contact_arc = full_physical.get("first_contact_arc_m")
    return None if contact_arc is None else float(contact_arc)


def view_collision_rollout(frame, actions, radius: float,
                           base_pose=(0.0, 0.0, 0.0)) -> dict:
    for i, (x, z, _heading, arc) in enumerate(A.sample_path(actions, config.MARCH_STEP_M)):
        pose = _compose_pose(base_pose, A.pose_at_arc(actions, arc))
        if i and frame.vf.support_count((pose[0], pose[1]), radius) >= config.MIN_SUPPORT_VOXELS:
            action_index, local = A.contact_action_index(actions, arc)
            return {"authority": "depth", "collision": True,
                    "first_contact_arc_m": float(arc),
                    "contact_action_index": action_index,
                    "contact_action_local_arc_m": local,
                    "center_local": [float(pose[0]), float(pose[1])],
                    "realized_pose": pose_dict(pose)}
    final_pose = _compose_pose(base_pose, A.pose_after(actions))
    return {"authority": "depth", "collision": False, "first_contact_arc_m": None,
            "contact_action_index": None, "contact_action_local_arc_m": None,
            "center_local": None, "realized_pose": pose_dict(final_pose)}


def certified_near_field_distance_m(*, camera_height_above_floor_m: float,
                                    hfov_deg: float, vfov_deg: float,
                                    radius: float) -> float:
    """Return the calibrated floor blind strip certified by the task contract.
    The height is the *calibrated* one -- where the floor actually is under this
    camera. The nominal mount offset would place the ground entry ray on a floor
    at the agent root, which is the assumption P0-4 removes.
    """
    horizontal = (float(radius) /
                  max(math.tan(math.radians(float(hfov_deg)) / 2.0), 1e-9))
    ground_entry = (float(camera_height_above_floor_m) /
                    max(math.tan(math.radians(float(vfov_deg)) / 2.0), 1e-9))
    return float(max(horizontal, ground_entry))


def certified_near_field_m(frame, radius: float) -> float:
    """Frame-aware wrapper for the public calibrated near-field distance."""
    return certified_near_field_distance_m(
        camera_height_above_floor_m=frame.camera_height_above_visible_floor_m,
        hfov_deg=frame.sensor.hfov_deg,
        vfov_deg=frame.sensor.vfov_deg,
        radius=radius)


def near_field_blind_distance_m(*, radius: float, half_hfov_rad: float,
                                bearing_rad: float,
                                certified_near_field_m: float) -> float:
    """Range within which a corridor cross-section cannot be depth-verified.

    A cross-section of half-width ``radius`` centred at ``bearing_rad`` fits
    inside the horizontal cone only where ``|bearing| + atan(radius / d) <=
    half_hfov``, that is beyond ``radius / tan(half_hfov - |bearing|)``.
    Straight ahead this is exactly the v5 ``radius / tan(half_hfov)``; off-axis
    the body leaves the frame farther out, and charging that difference as
    missing evidence measures the formula rather than the scene.

    The result is capped at the certified near field, so the exemption can
    never reach past the radius the hidden-collision gate guards.
    """
    if radius <= 0:
        return 0.0
    margin = float(half_hfov_rad) - abs(float(bearing_rad))
    if margin <= 1e-9:
        return float(certified_near_field_m)
    return min(float(radius) / max(math.tan(margin), 1e-9),
               float(certified_near_field_m))


def corridor_coverage_details(frame, actions, radius: float,
                              base_pose=(0.0, 0.0, 0.0),
                              max_arc_m=None) -> dict:
    """Measure visible-floor support across the circular body's swept corridor.
    The task contract certifies the calibrated near-floor blind strip while
    retaining visible obstacles there as collision evidence. Beyond that strip,
    every lateral sample is projected onto the local ground plane. ``max_arc_m``
    limits collision evidence to the actually executed prefix.
    """
    seen = total = 0
    out_of_frame_samples = invalid_depth_samples = occluded_samples = 0
    # Two different heights, deliberately: reprojection is camera-relative and
    # takes the nominal mount offset, while where the floor enters the frame
    # depends on how far the floor actually is below the camera.
    nominal_offset = frame.sensor.nominal_camera_offset_m
    floor_plane = frame.floor_plane
    bx, bz, base_heading_deg = (float(value) for value in base_pose)
    base_heading = math.radians(base_heading_deg)
    sample_count = max(1, int(config.EVIDENCE_CORRIDOR_LATERAL_SAMPLES))
    lateral_offsets = np.linspace(-float(radius), float(radius), sample_count)
    half_hfov = math.radians(float(frame.sensor.hfov_deg)) / 2.0
    horizontal_near_field = (
        float(radius) /
        max(math.tan(half_hfov), 1e-9)
        if radius > 0 else 0.0)
    ground_visibility_start = (
        frame.camera_height_above_visible_floor_m /
        max(math.tan(math.radians(float(frame.sensor.vfov_deg)) / 2.0), 1e-9))
    near_field = max(horizontal_near_field, ground_visibility_start)
    rows = []
    last_arc = 0.0
    for i, (x, z, heading, arc) in enumerate(A.sample_path(
            actions, config.MARCH_STEP_M)):
        if i == 0:
            continue
        if max_arc_m is not None and float(arc) > float(max_arc_m) + 1e-9:
            break
        global_x = bx + x * math.cos(base_heading) + z * math.sin(base_heading)
        global_z = bz - x * math.sin(base_heading) + z * math.cos(base_heading)
        global_heading = base_heading + heading
        distance = math.hypot(global_x, global_z)
        # Vertical floor blindness never certifies a path outside the camera's
        # horizontal view. Only the close-body width strip uses the center ray.
        center_in_hfov = (
            global_z > 0.0 and
            abs(math.atan2(global_x, global_z)) <= half_hfov + 1e-9)
        if not center_in_hfov:
            total += sample_count
            out_of_frame_samples += sample_count
            rows.append({
                "arc_m": float(arc),
                "seen": 0,
                "total": sample_count,
                "out_of_frame": sample_count,
                "invalid_depth": 0,
                "occluded": 0,
            })
            last_arc = float(arc)
            continue
        row_blind = near_field_blind_distance_m(
            radius=radius, half_hfov_rad=half_hfov,
            bearing_rad=math.atan2(global_x, global_z),
            certified_near_field_m=near_field)
        if distance <= row_blind + 1e-9:
            continue
        row_seen = row_total = 0
        row_out_of_frame = row_invalid_depth = row_occluded = 0
        for lateral in lateral_offsets:
            px = global_x + lateral * math.cos(global_heading)
            pz = global_z - lateral * math.sin(global_heading)
            total += 1
            row_total += 1
            if (pz <= 0.0 or
                    abs(math.atan2(px, pz)) > half_hfov + 1e-9):
                out_of_frame_samples += 1
                row_out_of_frame += 1
                continue
            if distance <= ground_visibility_start + 1e-9:
                seen += 1
                row_seen += 1
                continue
            # The sample lies on the canonical plane, not at a hard-coded y=0:
            # a tilted or offset floor puts the ground sample somewhere else,
            # and reprojecting it at 0 would compare depth against a surface
            # that is not the one the body travels over.
            uv = perception.project_ground(
                (px, floor_plane.y_at(px, pz), pz), frame.K,
                camera_height=nominal_offset)
            if uv is None:
                out_of_frame_samples += 1
                row_out_of_frame += 1
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            if not (0 <= u < frame.depth.shape[1] and 0 <= v < frame.depth.shape[0]):
                out_of_frame_samples += 1
                row_out_of_frame += 1
                continue
            depth = float(frame.depth[v, u])
            if not np.isfinite(depth) or depth <= 0.0:
                invalid_depth_samples += 1
                row_invalid_depth += 1
                continue
            if depth + config.DEPTH_SUPPORT_TOL_M < pz:
                occluded_samples += 1
                row_occluded += 1
                continue
            seen += 1
            row_seen += 1
        rows.append({
            "arc_m": float(arc),
            "seen": int(row_seen),
            "total": int(row_total),
            "out_of_frame": int(row_out_of_frame),
            "invalid_depth": int(row_invalid_depth),
            "occluded": int(row_occluded),
        })
        last_arc = float(arc)
    raw_coverage = seen / total if total else 1.0
    terminal_start = max(0.0, last_arc - config.EVIDENCE_TERMINAL_LENGTH_M)
    terminal_rows = [
        value for value in rows
        if value["arc_m"] >= terminal_start - 1e-9
    ]
    terminal_seen = sum(value["seen"] for value in terminal_rows)
    terminal_total = sum(value["total"] for value in terminal_rows)
    terminal_out_of_frame = sum(
        value["out_of_frame"] for value in terminal_rows)
    terminal_invalid_depth = sum(
        value["invalid_depth"] for value in terminal_rows)
    terminal_occluded = sum(value["occluded"] for value in terminal_rows)
    terminal_coverage = terminal_seen / terminal_total if terminal_total else 1.0
    row_supported = [
        value["out_of_frame"] == 0 and
        value["occluded"] == 0 and
        value["invalid_depth"] <= config.EVIDENCE_MAX_INVALID_DEPTH_SAMPLES
        for value in rows]
    max_unsupported_run = current_run = 0.0
    previous_arc = None
    for value, supported in zip(rows, row_supported):
        arc = value["arc_m"]
        step = (float(arc) - float(previous_arc)
                if previous_arc is not None else config.MARCH_STEP_M)
        current_run = 0.0 if supported else current_run + max(0.0, step)
        max_unsupported_run = max(max_unsupported_run, current_run)
        previous_arc = float(arc)
    meets_terminal = (
        terminal_coverage >= config.EVIDENCE_MIN_LATERAL_FRACTION - 1e-9)
    meets_gap = (
        max_unsupported_run <= config.EVIDENCE_MAX_UNSUPPORTED_RUN_M + 1e-9)
    meets_visibility_contract = bool(
        raw_coverage >= config.EVIDENCE_COVERAGE_MIN - 1e-9 and
        out_of_frame_samples == 0 and
        occluded_samples == 0 and
        invalid_depth_samples <= config.EVIDENCE_MAX_INVALID_DEPTH_SAMPLES and
        meets_terminal and meets_gap)
    qualified_coverage = raw_coverage if meets_visibility_contract else 0.0
    return {
        "protocol": EVIDENCE_PROTOCOL_VERSION,
        "coverage": float(qualified_coverage),
        "raw_coverage": float(raw_coverage),
        "supported_samples": int(seen),
        "total_samples": int(total),
        "out_of_frame_samples": int(out_of_frame_samples),
        "invalid_depth_samples": int(invalid_depth_samples),
        "occluded_samples": int(occluded_samples),
        "lateral_sample_count": sample_count,
        "horizontal_near_field_m": float(horizontal_near_field),
        "ground_visibility_start_m": float(ground_visibility_start),
        "near_field_exempt_m": float(near_field),
        "evaluated_arc_m": float(last_arc),
        "terminal_length_m": float(config.EVIDENCE_TERMINAL_LENGTH_M),
        "terminal_coverage": float(terminal_coverage),
        "terminal_out_of_frame_samples": int(terminal_out_of_frame),
        "terminal_invalid_depth_samples": int(terminal_invalid_depth),
        "terminal_occluded_samples": int(terminal_occluded),
        "max_unsupported_run_m": float(max_unsupported_run),
        "meets_visibility_contract": meets_visibility_contract,
    }


def corridor_coverage(frame, actions, radius: float,
                      base_pose=(0.0, 0.0, 0.0), max_arc_m=None) -> float:
    return float(corridor_coverage_details(
        frame, actions, radius, base_pose=base_pose,
        max_arc_m=max_arc_m)["coverage"])
