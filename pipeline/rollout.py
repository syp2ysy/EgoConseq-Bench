"""Pure structured rollout helpers used by :func:`pipeline.consequence.judge`."""

from __future__ import annotations

import math

import numpy as np

from pipeline import actions as A, config, perception


EVIDENCE_PROTOCOL_VERSION = "swept_floor_v6"
# v5 differs only in the near-field exemption: it used the straight-ahead
# wedge at every bearing, so a turned body's own flank counted as missing
# evidence. Frozen v5 artefacts stay verifiable as history; only v6 is emitted.


def pose_dict(pose) -> dict:
    return {"x": float(pose[0]), "z": float(pose[1]), "heading_deg": float(pose[2])}


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
