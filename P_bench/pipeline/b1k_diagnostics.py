"""Private, non-gating diagnostics for real B1K oracle probes."""

from __future__ import annotations

import collections

import numpy as np

from pipeline import actions as action_module, b1k_geometry, config, perception


_ORACLE_PRECHECK_ROWS = []


def reset_oracle_precheck_diagnostics() -> None:
    _ORACLE_PRECHECK_ROWS.clear()


def record_oracle_precheck_diagnostic(
        *, scene_id: str, pose_index: int, frame_id: str, radius: float,
        length: int, action_tag: str, label: str, precheck: dict,
        depth_alignment: dict | None = None) -> None:
    """Keep private real-pilot A2 measurements rejected before records."""
    _ORACLE_PRECHECK_ROWS.append({
        "scene_id": str(scene_id),
        "pose_index": int(pose_index),
        "frame_id": str(frame_id),
        "body_radius_m": float(radius),
        "action_length": int(length),
        "action_tag": str(action_tag),
        "full_geometry_label": str(label),
        **{
            key: precheck.get(key) for key in (
                "accepted", "reason", "full_collision", "depth_collision",
                "full_contact_arc_m", "depth_contact_arc_m",
                "contact_arc_difference_m", "corridor_coverage")
        },
        **({"depth_alignment": depth_alignment}
           if depth_alignment is not None else {}),
    })


def _depth_patch(depth: np.ndarray, pixel) -> list | None:
    if pixel is None:
        return None
    u, v = map(int, pixel)
    height, width = depth.shape
    if not (0 <= u < width and 0 <= v < height):
        return None
    return np.asarray(depth[
        max(0, v - 1):min(height, v + 2),
        max(0, u - 1):min(width, u + 2),
    ], dtype=np.float64).tolist()


def depth_alignment_diagnostic(
        *, sim, frame, actions, radius: float,
        full_physical: dict, depth_physical: dict) -> dict:
    """Expose the exact raw axial-depth ray behind one A2 comparison."""
    position = np.asarray(frame.position, dtype=np.float64)
    camera_position = position + np.array([
        0.0, float(frame.sensor.nominal_camera_offset_m), 0.0])
    forward_point = perception.world_from_local(
        np.array([[0.0, 0.0, 1.0]]), position, frame.yaw_rad)[0]
    forward = forward_point - position
    center_pixel = [frame.depth.shape[1] // 2, frame.depth.shape[0] // 2]
    result = {
        "schema": "b1k-depth-alignment-diagnostic.v1",
        "depth_input": {
            "modality": "depth_linear",
            "renderer_semantics": "distance_to_image_plane",
            "unprojection_semantics": "axial_camera_z",
            "units": "m",
            "no_hit_value_after_adapter": 0.0,
        },
        "camera": {
            "position_pbench_world_xyz_m": camera_position.tolist(),
            "forward_pbench_world_xyz": forward.tolist(),
            "position_og_world_xyz_m":
                b1k_geometry.pbench_to_og_xyz(camera_position).tolist(),
            "forward_og_world_xyz":
                b1k_geometry.pbench_to_og_xyz(forward).tolist(),
            "pbench_from_og": "[og_x, og_z, -og_y]",
            "yaw_rad": float(frame.yaw_rad),
        },
        "raw_image_center": {
            "pixel_uv": center_pixel,
            "depth_patch_m": _depth_patch(frame.depth, center_pixel),
        },
        "actions": action_module.actions_to_dicts(actions),
        "full_contact_arc_m": full_physical.get("first_contact_arc_m"),
        "depth_contact_arc_m": depth_physical.get("first_contact_arc_m"),
    }
    full_arc = full_physical.get("first_contact_arc_m")
    full_contact = None
    if full_arc is not None:
        contact_pose = action_module.pose_at_arc(actions, float(full_arc))
        nav = sim.proposal_nav(
            frame.position, frame.yaw_rad, radius_m=float(radius))
        hit = dict(nav.closest_obstacle(contact_pose) or {})
        world_point = hit.get("world_point")
        local_point = None
        pixel = None
        if world_point is not None:
            local_point = perception.local_from_world(
                np.asarray([world_point], dtype=np.float64),
                frame.position, frame.yaw_rad)[0]
            projected = perception.project_ground(
                local_point, frame.K,
                camera_height=frame.sensor.nominal_camera_offset_m)
            if projected is not None:
                pixel = [int(round(projected[0])), int(round(projected[1]))]
        full_contact = {
            "center_local_xz_heading": [float(value)
                                        for value in contact_pose],
            "obstacle_identity": hit.get("obstacle_identity"),
            "world_point_pbench_xyz_m": world_point,
            "local_xyz_m": (
                local_point.tolist() if local_point is not None else None),
            "projected_pixel_uv": pixel,
            "expected_axial_depth_m": (
                float(local_point[2]) if local_point is not None else None),
            "raw_depth_patch_m": _depth_patch(frame.depth, pixel),
        }
    result["full_contact"] = full_contact
    result["depth_contact"] = {
        "center_local_xz_m": depth_physical.get("center_local"),
    }
    return result


def oracle_precheck_report() -> dict:
    grouped = collections.defaultdict(list)
    for row in _ORACLE_PRECHECK_ROWS:
        grouped[str(float(row["body_radius_m"]))].append(row)
    by_radius = {}
    for radius, rows in sorted(grouped.items(), key=lambda item: float(item[0])):
        differences = sorted(
            float(row["contact_arc_difference_m"])
            for row in rows
            if row.get("contact_arc_difference_m") is not None)
        reasons = collections.Counter(str(row.get("reason")) for row in rows)
        by_radius[radius] = {
            "total": len(rows),
            "accepted": sum(row.get("accepted") is True for row in rows),
            "reasons": dict(sorted(reasons.items())),
            "contact_arc_difference_m": {
                "count": len(differences),
                "minimum": min(differences) if differences else None,
                "maximum": max(differences) if differences else None,
                "values": differences,
            },
        }
    return {
        "schema": "b1k-oracle-precheck-diagnostics.v1",
        "contact_tolerance_m": float(config.ORACLE_CONTACT_TOL_M),
        "consensus_by_radius_m": by_radius,
        "rows": list(_ORACLE_PRECHECK_ROWS),
    }
