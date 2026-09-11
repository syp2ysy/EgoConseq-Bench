"""Compact records that contain exactly what the ABC1 compiler consumes."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Iterable, Mapping

from pipeline import abc1_task_outputs


SCHEMA_VERSION = "abc1.record.v3"
CASE_TASKS_BY_DATASET = {
    "b1k": ("A1", "A2", "A3", "A4", "B1", "B2"),
    "gs": ("A1", "A2", "A4", "B2"),
    "r2r": ("A1", "A2", "A3", "A4", "B1", "B2"),
}
TASKS_BY_DATASET = {
    dataset: (*tasks, "C1")
    for dataset, tasks in CASE_TASKS_BY_DATASET.items()
}
INITIAL_TURNS_DEG = frozenset((-30.0, -15.0, 15.0, 30.0))


def _digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def historical_record_uid(
        dataset: str, source_records_sha256: str, byte_offset: int) -> str:
    identity = [str(dataset), str(source_records_sha256), int(byte_offset)]
    return f"{dataset}-{_digest(identity)[:24]}"


def new_record_uid(
        *, dataset: str, shard_id: str, scene_id: str, pose_index: int,
        sensor_tag: str, pose: Mapping[str, object]) -> str:
    identity = [
        str(dataset), str(shard_id), str(scene_id), int(pose_index),
        str(sensor_tag), pose,
    ]
    return f"{dataset}-{_digest(identity)[:24]}"


def supported_tasks(dataset: str) -> tuple[str, ...]:
    return TASKS_BY_DATASET[str(dataset)]


def is_compact(record: Mapping[str, object]) -> bool:
    return record.get("schema_version") in {"abc1.record.v2", SCHEMA_VERSION}


def normalize(record: Mapping[str, object]) -> dict:
    """Normalize a compact input once; do not rejudge frozen task outputs."""
    from pipeline import surface_points

    result = copy.deepcopy(dict(record))
    result["schema_version"] = SCHEMA_VERSION
    target = surface_points.normalize_target(result)
    if target is not None:
        result["surface_point_target"] = target
    result.pop("surface_relation_schema", None)
    for case in result.get("cases", []):
        for task in ("A4", "B1", "B2"):
            output = case.get("task_outputs", {}).get(task)
            if output is None or "points" in output:
                continue
            checkpoint = {k: output[k] for k in ("checkpoint", "checkpoint_seed") if k in output}
            point = {k: v for k, v in output.items()
                     if k not in {"checkpoint", "checkpoint_seed", "initial_distance_m", "distance_change_m"}}
            case["task_outputs"][task] = {
                **checkpoint, "points": [{"point_id": "p1", **point}] if point else []}
    return result


def install_surface_delta(record: Mapping[str, object], result: dict) -> dict:
    """Append at most one certified safe group without evicting old cases."""
    from pipeline import surface_points

    updated = normalize(record)
    if result.get("surface_point_target") is not None:
        updated["surface_point_target"] = copy.deepcopy(result["surface_point_target"])
    if result.get("observation_profile") is not None:
        updated["observation_profile"] = copy.deepcopy(result["observation_profile"])
    existing = {case["case_id"] for case in updated["cases"]}
    updated["cases"].extend(copy.deepcopy(case) for case in result.get("cases", [])
                            if case["case_id"] not in existing)
    updated["action_group_capacity"] = max(int(updated["action_group_capacity"]),
                                          len({c["group_id"] for c in updated["cases"]}))
    surface_points.refresh_outputs(updated)
    return updated


def starts_with(actions: Iterable[Mapping[str, object]]) -> str:
    values = list(actions)
    return str(values[0].get("type")) if values else ""


def compliant_initial_action(actions: Iterable[Mapping[str, object]]) -> bool:
    values = list(actions)
    if not values:
        return False
    if values[0].get("type") != "turn":
        return values[0].get("type") == "forward"
    return float(values[0].get("deg")) in INITIAL_TURNS_DEG


def physical_fingerprint(record: Mapping[str, object]) -> str:
    """Hash only immutable pose/body/camera geometry."""
    sensor = record["sensor"]
    radii = record.get("body_radii_m")
    if radii is None:
        selection = record.get("selection") or {}
        radii = selection.get("required_radii_m") or sorted({
            float(outcome["body"]["radius_m"])
            for outcome in record.get("outcomes") or []
            if outcome.get("body")
        })
    return _digest({
        "pose": record["pose"],
        "body_radii_m": sorted(map(float, radii)),
        "sensor": {
            "nominal_camera_offset_m": float(
                sensor["nominal_camera_offset_m"]),
            "hfov_deg": float(sensor["hfov_deg"]),
            "vfov_deg": float(sensor["vfov_deg"]),
            "resolution": list(sensor.get("resolution") or []),
        },
    })


def _certified(outcome: Mapping[str, object]) -> bool:
    physical = outcome.get("physical") or {}
    depth = outcome.get("depth_physical") or {}
    collision = physical.get("collision")
    if (not isinstance(collision, bool) or
            depth.get("collision") is not collision or
            (outcome.get("oracle_consensus") or {}).get("accepted") is not True):
        return False
    summary = (outcome.get("shared_oracle_stability") or {}).get("summary") or {}
    stability = outcome.get("shared_oracle_stability") or {}
    nominal = (stability.get("version") == "nominal-oracle.v1" and
               summary.get("evaluation") == "nominal")
    if (summary.get("collision") is not collision or
            (not nominal and summary.get("collision_label_stable") is not True)):
        return False
    return (outcome.get("execution") or {}).get("completed") is (not collision)


def _stable_collision_index(outcome: Mapping[str, object]) -> int | None:
    physical = outcome.get("physical") or {}
    depth = outcome.get("depth_physical") or {}
    index = physical.get("contact_action_index")
    if not isinstance(index, int) or isinstance(index, bool) or \
            depth.get("contact_action_index") != index:
        return None
    stability = outcome.get("shared_oracle_stability") or {}
    summary = stability.get("summary") or {}
    nominal = (stability.get("version") == "nominal-oracle.v1" and
               summary.get("evaluation") == "nominal")
    if (summary.get("original_action_index") != index + 1 or
            (not nominal and
             summary.get("original_action_index_stable") is not True)):
        return None
    return index + 1


def _stable_contact(outcome: Mapping[str, object]) -> dict | None:
    contact = (outcome.get("physical") or {}).get("contact")
    if not isinstance(contact, Mapping) or contact.get("unattributed") is True:
        return None
    instance_id = contact.get("instance_id")
    category = contact.get("category")
    if instance_id is None or not category:
        return None
    stability = outcome.get("shared_oracle_stability") or {}
    summary = stability.get("summary") or {}
    nominal = (stability.get("version") == "nominal-oracle.v1" and
               summary.get("evaluation") == "nominal")
    if (summary.get("contact_instance_id") != instance_id or
            (not nominal and summary.get("contact_instance_stable") is not True)):
        return None
    return {"instance_id": int(instance_id), "category": str(category)}


def compact_case(
        record: Mapping[str, object], outcome: Mapping[str, object], *,
        dataset: str, record_uid: str,
        provenance: Mapping[str, object]) -> dict | None:
    if not _certified(outcome):
        return None
    actions = copy.deepcopy(list(outcome.get("actions") or []))
    if not compliant_initial_action(actions):
        return None
    physical = outcome["physical"]
    execution = outcome["execution"]
    collision = bool(physical["collision"])
    source = provenance.get(str(outcome["action_group_id"])) or {}
    protocol = source.get("protocol") if isinstance(source, Mapping) else None
    variant = source.get("variant") if isinstance(source, Mapping) else None
    value = {
        "case_id": str(outcome["outcome_id"]),
        "group_id": str(outcome["action_group_id"]),
        "actions": actions,
        "starts_with": starts_with(actions),
        "body_radius_m": float(outcome["body"]["radius_m"]),
        "collision": collision,
        "completed": bool(execution["completed"]),
        "first_collision_action_index_1based": (
            _stable_collision_index(outcome) if collision else None),
        "endpoint_pose": copy.deepcopy(execution["realized_pose"]),
        "minimum_clearance_m": (
            None if physical.get("minimum_clearance_m") is None else
            float(physical["minimum_clearance_m"])),
        "proposal_protocol": protocol,
        "proposal_variant": variant,
    }
    contact = _stable_contact(outcome) if collision else None
    if contact is not None:
        value["contact"] = contact
    terminal = outcome.get("terminal_rgb_asset") or {}
    if terminal.get("path"):
        value["terminal_rgb_path"] = str(terminal["path"])
        value["terminal_rgb_sha256"] = terminal.get("png_sha256")
        value["terminal_pixel_sha256"] = terminal.get("pixel_sha256")
    value["task_outputs"] = abc1_task_outputs.build_task_outputs(
        record, outcome, record_uid=record_uid,
        proposal_protocol=protocol, proposal_variant=variant,
        allowed_tasks=CASE_TASKS_BY_DATASET[str(dataset)])
    return value


def _visible_entities(record: Mapping[str, object]) -> list[dict]:
    entities = {}
    for value in record.get("objects") or []:
        instance_id = value.get("instance_id")
        category = value.get("source_category") or value.get("category")
        if instance_id is not None and category:
            entities[int(instance_id)] = {
                "instance_id": int(instance_id), "category": str(category)}
    return [entities[key] for key in sorted(entities)]


def _action_sha256(actions: Iterable[Mapping[str, object]]) -> str:
    return _digest({"actions": list(actions)})


def action_program_sha256(actions: Iterable[Mapping[str, object]]) -> str:
    return _action_sha256(actions)


def _c1_families(cases: list[dict], provenance: Mapping[str, object]) -> list[dict]:
    """Recover explicit query/distractor links without opening any image."""
    by_group = {}
    queries = {}
    for case in cases:
        if (case.get("terminal_rgb_path") and
                case.get("terminal_rgb_sha256") and
                case.get("terminal_pixel_sha256") and
                not case["collision"]):
            by_group.setdefault(case["group_id"], []).append(case)
            queries.setdefault(_action_sha256(case["actions"]), []).append(case)
    neighbors = {}
    for group_id, metadata in provenance.items():
        if not isinstance(metadata, Mapping):
            continue
        base = metadata.get("base_action_sha256")
        if base and group_id in by_group:
            neighbors.setdefault(str(base), []).extend(by_group[group_id])
    result = []
    for digest, query_cases in sorted(queries.items()):
        for query in query_cases:
            members = [query]
            seen_paths = {query["terminal_rgb_path"]}
            seen_pngs = {query["terminal_rgb_sha256"]}
            seen_pixels = {query["terminal_pixel_sha256"]}
            for candidate in sorted(
                    neighbors.get(digest, ()),
                    key=lambda value: value["case_id"]):
                if (candidate["body_radius_m"] != query["body_radius_m"] or
                        candidate["terminal_rgb_path"] in seen_paths or
                        candidate["terminal_rgb_sha256"] in seen_pngs or
                        candidate["terminal_pixel_sha256"] in seen_pixels):
                    continue
                members.append(candidate)
                seen_paths.add(candidate["terminal_rgb_path"])
                seen_pngs.add(candidate["terminal_rgb_sha256"])
                seen_pixels.add(candidate["terminal_pixel_sha256"])
                if len(members) == 4:
                    result.append({
                        "query_case_id": query["case_id"],
                        "member_case_ids": [
                            case["case_id"] for case in members],
                    })
                    break
    return result


def from_legacy(
        record: Mapping[str, object], *, dataset: str,
        source_records_sha256: str, byte_offset: int,
        record_uid: str | None = None) -> dict:
    """Convert one authenticated legacy record into its compact form."""
    if is_compact(record):
        return normalize(record)
    selection = record.get("selection") or {}
    final_uid = record_uid or historical_record_uid(
        dataset, source_records_sha256, byte_offset)
    provenance = selection.get("proposal_provenance") or {}
    selected = set(map(str, selection.get("action_group_ids") or []))
    cases = []
    for outcome in record.get("outcomes") or []:
        if selected and str(outcome.get("action_group_id")) not in selected:
            continue
        case = compact_case(
            record, outcome, dataset=dataset, record_uid=final_uid,
            provenance=provenance)
        if case is not None:
            cases.append(case)
    cases.sort(key=lambda value: (value["group_id"], value["body_radius_m"]))
    sensor = record["sensor"]
    plane = (record.get("floor_calibration") or {}).get("estimate") or {
        "normal_local": [0.0, 1.0, 0.0], "offset_m": 0.0}
    radii = selection.get("required_radii_m") or sorted({
        case["body_radius_m"] for case in cases})
    original_groups = list(dict.fromkeys(map(
        str, selection.get("action_group_ids") or
        [case["group_id"] for case in cases])))
    compact = {
        "schema_version": SCHEMA_VERSION,
        "record_uid": final_uid,
        "dataset": str(dataset),
        "scene_id": str(record["scene_id"]),
        "frame_id": str(record.get("observation_id") or record["frame_id"]),
        "pose": copy.deepcopy(record["pose"]),
        "sensor": {
            "nominal_camera_offset_m": float(
                sensor["nominal_camera_offset_m"]),
            "hfov_deg": float(sensor["hfov_deg"]),
            "vfov_deg": float(sensor["vfov_deg"]),
            "resolution": list(sensor.get("resolution") or []),
        },
        "floor_plane": {
            "normal_local": list(plane["normal_local"]),
            "offset_m": float(plane["offset_m"]),
        },
        "body_radii_m": sorted(map(float, radii)),
        "image_path": str(record["image_path"]),
        "action_group_capacity": int(
            record.get("action_group_capacity") or len(original_groups) + 1),
        "visible_entities": _visible_entities(record),
        "cases": cases,
        "c1_families": _c1_families(
            cases, provenance),
    }
    if record.get("surface_point_target"):
        compact["surface_point_target"] = copy.deepcopy(record["surface_point_target"])
    compact = normalize(compact)
    from pipeline import surface_points
    surface_points.refresh_outputs(compact)
    return compact


def _start_collision_cells(
        cases: Iterable[Mapping[str, object]]) -> set[tuple[str, bool]]:
    return {
        (str(case["starts_with"]), bool(case["collision"]))
        for case in cases
    }


def _has_scarce_task(cases: Iterable[Mapping[str, object]]) -> bool:
    return any(
        {"A2", "A3"} & set((case.get("task_outputs") or {}).keys())
        for case in cases)


def install_delta(
        record: Mapping[str, object], *, outcomes: Iterable[Mapping[str, object]],
        provenance: Mapping[str, object], mode: str = "added") -> dict | None:
    """Install certified groups without changing geometry or growing capacity."""
    updated = copy.deepcopy(dict(record))
    new_cases = [compact_case(
        updated, outcome, dataset=str(updated["dataset"]),
        record_uid=str(updated["record_uid"]), provenance=provenance)
        for outcome in outcomes]
    if any(case is None for case in new_cases):
        return None
    new_cases = [case for case in new_cases if case is not None]
    new_groups = list(dict.fromkeys(case["group_id"] for case in new_cases))
    if mode == "c1" and len(new_groups) != 4:
        return None
    capacity = int(updated["action_group_capacity"])
    if len(new_groups) > capacity:
        return None

    existing_by_group = {}
    for case in updated.get("cases") or []:
        existing_by_group.setdefault(case["group_id"], []).append(case)
    new_by_group = {}
    for case in new_cases:
        new_by_group.setdefault(case["group_id"], []).append(case)

    selected = list(new_groups)
    selected_set = set(selected)
    family_groups = set()
    for family in sorted(
            updated.get("c1_families") or [],
            key=lambda value: str(value["query_case_id"])):
        member_ids = set(map(str, family["member_case_ids"]))
        bundle = sorted({
            group for group, group_cases in existing_by_group.items()
            if member_ids & {str(case["case_id"]) for case in group_cases}
        })
        family_groups.update(bundle)
        if (member_ids <= {
                str(case["case_id"])
                for group in bundle for case in existing_by_group[group]
        } and not selected_set.intersection(bundle) and
                len(selected) + len(bundle) <= capacity):
            selected.extend(bundle)
            selected_set.update(bundle)

    covered = _start_collision_cells(
        case for group in selected
        for case in new_by_group.get(group, existing_by_group.get(group, ())))
    available = [
        group for group in existing_by_group
        if group not in selected_set and group not in family_groups]
    while available and len(selected) < capacity:
        group = min(available, key=lambda value: (
            -int(_has_scarce_task(existing_by_group[value])),
            -len(_start_collision_cells(existing_by_group[value]) - covered),
            value))
        available.remove(group)
        selected.append(group)
        covered.update(_start_collision_cells(existing_by_group[group]))

    cases = []
    for group in selected:
        cases.extend(new_by_group.get(group, existing_by_group.get(group, ())))
    kept_ids = {case["case_id"] for case in cases}
    old_families = [
        family for family in updated.get("c1_families") or []
        if set(family["member_case_ids"]) <= kept_ids
    ]
    derived = _c1_families(cases, provenance)
    family_keys = {
        (family["query_case_id"], tuple(family["member_case_ids"]))
        for family in old_families
    }
    old_families.extend(
        family for family in derived
        if (family["query_case_id"], tuple(family["member_case_ids"]))
        not in family_keys)
    updated["cases"] = cases
    updated["c1_families"] = old_families
    from pipeline import surface_points

    updated = normalize(updated)
    surface_points.refresh_outputs(updated)
    return updated


def from_collected(
        record: Mapping[str, object], *, dataset: str, shard_id: str,
        pose_index: int, sensor_tag: str | None = None,
        frame=None) -> dict:
    """Compact a freshly certified pose before it enters a records shard."""
    sensor = record["sensor"]
    sensor_identity = sensor_tag or ":".join(
        f"{float(sensor[key]):.6f}"
        for key in ("nominal_camera_offset_m", "hfov_deg", "vfov_deg"))
    record_uid = new_record_uid(
        dataset=dataset, shard_id=shard_id,
        scene_id=str(record["scene_id"]), pose_index=pose_index,
        sensor_tag=sensor_identity, pose=record["pose"])
    compact = from_legacy(
        record, dataset=dataset,
        source_records_sha256=_digest([dataset, shard_id]),
        byte_offset=pose_index, record_uid=record_uid)
    compact["collection_group_id"] = str(
        (record.get("intervention") or {})["group_id"])
    compact["action_group_capacity"] = len({
        case["group_id"] for case in compact["cases"]})
    if dataset == "b1k":
        contract = record["collection_contract"]
        compact["observation_profile"] = {
            "collection_contract_version": contract["version"],
            "sha256": contract["observation_profile_sha256"],
        }
    if frame is not None:
        from pipeline import surface_points

        target = surface_points.select_target(frame, record_uid=record_uid)
        compact = install_surface_delta(compact, {"surface_point_target": target})
    return compact
