"""Visible instance points and camera relations; no extra renders."""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from pipeline import actions, checkpoint_direction, config, perception


def normalize_target(record: dict) -> dict | None:
    """Convert the historical single local point at the input boundary."""
    target = record.get("surface_point_target")
    if not target:
        return None
    if "points" in target:
        return copy.deepcopy(target)
    anchor = target.get("surface_anchor", target)
    local = anchor.get("initial_robot_xyz_m", anchor.get("point_initial_robot_xyz_m"))
    if local is None or anchor.get("pixel_xy_px") is None:
        return None
    pose = record["pose"]
    world = perception.world_from_local(
        np.asarray([local]), pose["position"], float(pose["yaw_rad"]))[0]
    return {
        "instance_id": int(target["instance_id"]),
        "category": str(target.get("source_category") or target["category"]),
        "points": [{"point_id": "p1", "pixel_xy_px": list(anchor["pixel_xy_px"]),
                    "world_xyz_m": world.tolist()}],
    }


def select_target(frame, *, record_uid: str, seed: int = 20260906,
                  existing: dict | None = None) -> dict | None:
    """Sample one visible object uniformly, then up to three separated pixels."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        f"{seed}:{record_uid}".encode()).digest()[:8], "big"))
    width, height = frame.sensor.width_px, frame.sensor.height_px
    marker = int(config.TARGET_POINT_MARKER_RADIUS_PX)
    candidates = {}
    for instance in np.unique(frame.pts_sem):
        instance = int(instance)
        if instance <= 0:
            continue
        category = frame.predicate_category(instance)
        if (config.is_structural(category) or
                not config.is_specific_semantic_category(category) or
                any(token in category.casefold()
                    for token in config.SURFACE_TARGET_EXCLUDED_MATERIAL_TOKENS)):
            continue
        indices = np.flatnonzero(frame.pts_sem == instance)
        uv = frame.pts_uv[indices]
        x0, y0 = uv.min(axis=0)
        x1, y1 = uv.max(axis=0)
        mask = np.zeros((y1 - y0 + 3, x1 - x0 + 3), dtype=bool)
        mask[uv[:, 1] - y0 + 1, uv[:, 0] - x0 + 1] = True
        interior = distance_transform_edt(mask)[uv[:, 1] - y0 + 1, uv[:, 0] - x0 + 1]
        keep = ((interior > 2) & (uv[:, 0] >= marker) & (uv[:, 0] < width - marker)
                & (uv[:, 1] >= marker) & (uv[:, 1] < height - marker))
        if np.any(keep):
            candidates[instance] = indices[keep]
    if existing is not None:
        target = copy.deepcopy(existing)
        instance = int(target["instance_id"])
        if instance not in candidates:
            return target
    elif candidates:
        instance = int(rng.choice(sorted(candidates)))
        target = {"instance_id": instance, "category": frame.predicate_category(instance),
                  "points": []}
    else:
        return None
    for index in rng.permutation(candidates[instance]):
        if len(target["points"]) >= 3:
            break
        pixel = frame.pts_uv[index]
        if any(np.sum((pixel - np.asarray(point["pixel_xy_px"])) ** 2)
               < (2 * marker) ** 2 for point in target["points"]):
            continue
        world = perception.world_from_local(
            frame.pts[index:index + 1], frame.position, frame.yaw_rad)[0]
        target["points"].append({
            "point_id": f"p{len(target['points']) + 1}",
            "pixel_xy_px": pixel.tolist(), "world_xyz_m": world.tolist(),
        })
    return target


def relation(point_local, endpoint: dict, camera_height_m: float) -> dict:
    """Point relative to a camera in the initial robot frame (metres)."""
    delta = np.asarray(point_local) - [endpoint["x"], camera_height_m, endpoint["z"]]
    heading = math.radians(endpoint["heading_deg"])
    local_x = float(delta[0] * math.cos(heading) - delta[2] * math.sin(heading))
    local_z = float(delta[0] * math.sin(heading) + delta[2] * math.cos(heading))
    planar = math.hypot(local_x, local_z)
    result = {"endpoint_distance_m": float(np.linalg.norm(delta))}
    if planar >= config.TARGET_DIRECTION_MIN_RANGE_M:
        horizontal = actions.horizontal_direction(actions.wrap_deg(
            math.degrees(math.atan2(local_x, local_z))))
        vertical = actions.vertical_direction(math.degrees(math.atan2(float(delta[1]), planar)))
        result.update(horizontal_direction=horizontal, vertical_direction=vertical,
                      answer=actions.spatial_direction_key(horizontal, vertical))
    return result


def checkpoint_distances(record: dict, case: dict):
    """Derive B3 from saved geometry; no new oracle or record fields needed."""
    target = record.get("surface_point_target")
    if record["dataset"] == "gs" or case["collision"] or not case["completed"] or not target:
        return
    points = target["points"]
    pose = record["pose"]
    height = float(record["sensor"]["nominal_camera_offset_m"])
    local = perception.local_from_world(
        np.asarray([p["world_xyz_m"] for p in points]),
        pose["position"], float(pose["yaw_rad"]))
    before = np.linalg.norm(local - [0, height, 0], axis=1)
    program = actions.parse_actions(case["actions"])
    for ordinal in range(1, sum(isinstance(a, actions.Forward) for a in program) + 1):
        for fraction in checkpoint_direction.FRACTIONS:
            checkpoint = checkpoint_direction.build_forward_checkpoint(
                program, forward_stage=ordinal, fraction=fraction)
            for point, xyz, initial in zip(points, local, before):
                distance = relation(xyz, checkpoint["pose"], height)["endpoint_distance_m"]
                if (distance >= config.B_ENDPOINT_DISTANCE_MIN_M and
                        abs(distance - initial) >= max(config.B_DISTANCE_CHANGE_MIN_M,
                                                       config.B_DISTANCE_CHANGE_MIN_RATIO * initial)):
                    yield point["point_id"], checkpoint, distance


def refresh_outputs(record: dict) -> None:
    """Refresh only surface tasks; collision tasks and C1 remain untouched."""
    target = record.get("surface_point_target")
    pose = record["pose"]
    height = float(record["sensor"]["nominal_camera_offset_m"])
    points = [] if not target else target["points"]
    local = (perception.local_from_world(np.asarray([p["world_xyz_m"] for p in points]),
                                       pose["position"], float(pose["yaw_rad"]))
             if points else [])
    initial_distances = [float(np.linalg.norm(point - [0, height, 0])) for point in local]
    for case in record["cases"]:
        outputs = case["task_outputs"]
        prior = outputs.get("A4", {})
        for task in ("A4", "B1", "B2"):
            outputs.pop(task, None)
        if case["collision"] or not case["completed"] or not points:
            continue
        program = actions.parse_actions(case["actions"])
        seed = prior.get("checkpoint_seed", f"{record['record_uid']}:{case['case_id']}")
        checkpoint = prior.get("checkpoint")
        if checkpoint is None and any(isinstance(a, actions.Forward) for a in program):
            checkpoint = checkpoint_direction.select_forward_checkpoint(program, seed=seed)
        a4, b1, b2 = [], [], []
        for point, xyz, before in zip(points, local, initial_distances):
            point_id = point["point_id"]
            if checkpoint is not None:
                value = relation(xyz, checkpoint["pose"], height)
                if "answer" in value:
                    a4.append({"point_id": point_id, **{k: v for k, v in value.items()
                                                       if k != "endpoint_distance_m"}})
            value = relation(xyz, case["endpoint_pose"], height)
            after = value["endpoint_distance_m"]
            if (record["dataset"] != "gs" and after >= config.B_ENDPOINT_DISTANCE_MIN_M
                    and abs(after - before) >= max(config.B_DISTANCE_CHANGE_MIN_M,
                                                  config.B_DISTANCE_CHANGE_MIN_RATIO * before)):
                b1.append({"point_id": point_id, "endpoint_distance_m": after})
            if "answer" in value:
                b2.append({"point_id": point_id, **{k: v for k, v in value.items()
                                                   if k != "endpoint_distance_m"}})
        if a4:
            outputs["A4"] = {"checkpoint_seed": seed, "checkpoint": checkpoint, "points": a4}
        if b1:
            outputs["B1"] = {"points": b1}
        if b2:
            outputs["B2"] = {"points": b2}
