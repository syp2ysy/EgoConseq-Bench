"""Project benchmark candidates from one authenticated consequence record."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Iterable

from pipeline import a2
from pipeline import abc1_record
from pipeline import candidate_sources
from pipeline import config
from pipeline import surface_points


@dataclass(frozen=True)
class Candidate:
    source_path: str
    byte_offset: int
    record_sha256: str
    record_id: str
    item_id: str
    task_id: str
    dataset: str
    scene_id: str
    outcome_id: str
    action_length: int
    starts_with: str
    camera_height_m: float
    body_radius_m: float
    hfov_deg: float
    answer_bucket: str
    action_signature: str
    action_family: str
    image_path: str
    position: tuple[float, float, float]
    numeric_value_m: float | None = None
    a2_ordinal: int | None = None
    a2_rank: str | None = None
    template_id: str | None = None
    c1_label: str | None = None
    point_id: str | None = None
    terminal_paths: tuple[str, ...] = ()
    checkpoint_ordinal: int | None = None
    checkpoint_fraction: float | None = None



def _category(value: object) -> str:
    value = re.sub(
        r"\.[anvr]\.\d+$", "", str(value or "").strip(),
        flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip().casefold()


def describe(
        *, dataset: str, source_path: str, byte_offset: int,
        record_sha256: str, record: dict, outcome: dict,
        task_id: str, item_id: str, answer_bucket: str,
        numeric_value_m: float | None = None,
        point_id: str | None = None) -> Candidate | None:
    actions = list(outcome.get("actions") or [])
    sensor = record["sensor"]
    length = len(actions)
    height = round(float(sensor["nominal_camera_offset_m"]), 2)
    radius_value = outcome.get("body_radius_m")
    if radius_value is None:
        radius_value = (outcome.get("body") or {})["radius_m"]
    radius = round(float(radius_value), 2)
    hfov = round(float(sensor["hfov_deg"]), 1)
    if (length not in range(1, 7) or
            height not in {0.5, 1.0, 1.5} or
            radius not in {0.15, 0.2, 0.25} or
            hfov not in {79.0, 110.0}):
        return None
    starts = str(actions[0].get("type") or "")
    if (starts == "turn" and
            float(actions[0].get("deg")) not in
            set(map(float, config.INITIAL_TURNS_DEG))):
        return None
    family = "-".join(
        "F" if action.get("type") == "forward" else
        ("R" if float(action.get("deg", 0.0)) >= 0.0 else "L")
        for action in actions)
    cell = outcome.get("a2_design", {}).get("cell", {})
    frame_id = str(record.get("observation_id") or record["frame_id"])
    record_uid = str(record.get("record_uid") or
                     f"{dataset}:{frame_id}:{int(byte_offset)}")
    return Candidate(
        source_path=str(source_path), byte_offset=int(byte_offset),
        record_sha256=str(record_sha256),
        record_id=record_uid, item_id=str(item_id),
        task_id=str(task_id), dataset=str(dataset),
        scene_id=str(record["scene_id"]),
        outcome_id=str(outcome.get("case_id") or outcome["outcome_id"]),
        action_length=length,
        starts_with=starts, camera_height_m=height,
        body_radius_m=radius, hfov_deg=hfov,
        answer_bucket=str(answer_bucket),
        action_signature=(outcome.get("action_signature") or
                          candidate_sources.canonical_sha256(actions)),
        action_family=family, image_path=str(record.get("image_path") or ""),
        position=tuple(float(value) for value in record["pose"]["position"]),
        numeric_value_m=numeric_value_m, point_id=point_id,
        a2_ordinal=(int(cell["forward_ordinal_1based"])
                    if cell else None),
        a2_rank=(str(cell["distance_rank"]) if cell else None),
    )


def _item_id(task_id: str, record_uid: str, case_id: str,
             point_id: str | None = None) -> str:
    key = f"{task_id}:{record_uid}:{case_id}"
    value = (key if point_id is None else f"{key}:{point_id}").encode()
    return f"{task_id.lower()}-{hashlib.sha256(value).hexdigest()[:16]}"


def _compact_describe(
        *, dataset: str, source_path: str, byte_offset: int,
        record_sha256: str, record: dict, case: dict, task_id: str,
        answer_bucket: str, numeric_value_m: float | None = None,
        a2_cell: a2.A2Cell | None = None,
        point_id: str | None = None) -> Candidate | None:
    outcome = dict(case)
    if a2_cell is not None:
        outcome["a2_design"] = {"cell": {
            "forward_ordinal_1based": a2_cell.forward_ordinal_1based,
            "distance_rank": a2_cell.distance_rank,
        }}
    return describe(
        dataset=dataset, source_path=source_path, byte_offset=byte_offset,
        record_sha256=record_sha256, record=record, outcome=outcome,
        task_id=task_id,
        item_id=_item_id(task_id, record["record_uid"], case["case_id"], point_id),
        answer_bucket=answer_bucket, numeric_value_m=numeric_value_m,
        point_id=point_id)


def _project_compact(
        *, dataset: str, source_path: str, byte_offset: int,
        record_sha256: str, record: dict,
        allowed_tasks: Iterable[str]) -> list[Candidate]:
    allowed = set(map(str, allowed_tasks))
    result = []
    for case in record.get("cases") or []:
        case = dict(case, action_signature=candidate_sources.canonical_sha256(case["actions"]))
        outputs = case.get("task_outputs") or {}
        if "A1" in allowed and "A1" in outputs:
            row = _compact_describe(
                dataset=dataset, source_path=source_path,
                byte_offset=byte_offset, record_sha256=record_sha256,
                record=record, case=case, task_id="A1",
                answer_bucket=str(outputs["A1"]["answer"]))
            if row is not None:
                result.append(row)
        if "A2" in allowed and "A2" in outputs:
            output = outputs["A2"]
            cell = a2.A2Cell(
                int(output["forward_ordinal_1based"]),
                str(output["distance_rank"]))
            row = _compact_describe(
                dataset=dataset, source_path=source_path,
                byte_offset=byte_offset, record_sha256=record_sha256,
                record=record, case=case, task_id="A2",
                answer_bucket=(
                    f"o{cell.forward_ordinal_1based}-{cell.distance_rank}"),
                a2_cell=cell)
            if row is not None:
                result.append(row)
        if "A3" in allowed and "A3" in outputs:
            row = _compact_describe(
                dataset=dataset, source_path=source_path,
                byte_offset=byte_offset, record_sha256=record_sha256,
                record=record, case=case, task_id="A3",
                answer_bucket=_category(outputs["A3"]["category"]))
            if row is not None:
                result.append(row)
        for task in ("A4", "B1", "B2"):
            if task not in allowed or task not in outputs:
                continue
            for point in outputs[task]["points"]:
                row = _compact_describe(
                    dataset=dataset, source_path=source_path,
                    byte_offset=byte_offset, record_sha256=record_sha256,
                    record=record, case=case, task_id=task,
                    point_id=point["point_id"],
                    answer_bucket="distance" if task == "B1" else point["answer"],
                    numeric_value_m=(float(point["endpoint_distance_m"])
                                     if task == "B1" else None))
                if row is not None:
                    result.append(row)
        if "B3" in allowed:
            for point_id, checkpoint, distance in surface_points.checkpoint_distances(record, case):
                ordinal, fraction = checkpoint["forward_stage"], checkpoint["fraction"]
                row = _compact_describe(
                    dataset=dataset, source_path=source_path, byte_offset=byte_offset,
                    record_sha256=record_sha256, record=record, case=case,
                    task_id="B3", point_id=point_id, answer_bucket="distance",
                    numeric_value_m=distance)
                if row is not None:
                    result.append(replace(row, checkpoint_ordinal=ordinal,
                        checkpoint_fraction=fraction,
                        item_id=_item_id("B3", record["record_uid"], case["case_id"],
                                         f"{point_id}:{ordinal}:{fraction}")))
    if "C1" in allowed:
        cases = {str(case["case_id"]): case
                 for case in record.get("cases") or []}
        for family in record.get("c1_families") or []:
            query = cases.get(str(family.get("query_case_id")))
            members = [cases.get(str(case_id))
                       for case_id in family.get("member_case_ids") or []]
            if (query is None or len(members) != 4 or any(
                    member is None or not member.get("terminal_rgb_path")
                    for member in members)):
                continue
            row = _compact_describe(
                dataset=dataset, source_path=source_path,
                byte_offset=byte_offset, record_sha256=record_sha256,
                record=record, case=query, task_id="C1", answer_bucket="")
            if row is not None:
                result.append(replace(row, terminal_paths=tuple(
                    member["terminal_rgb_path"] for member in members)))
    return result


def project_record(
        *, dataset: str, source_path: str, byte_offset: int,
        record_sha256: str, record: dict,
        allowed_tasks: Iterable[str]) -> list[Candidate]:
    """Project certified compact outputs; old records adapt at this boundary."""
    if not abc1_record.is_compact(record):
        record = abc1_record.from_legacy(
            record, dataset=dataset, source_records_sha256=record_sha256,
            byte_offset=byte_offset)
    elif record["schema_version"] != abc1_record.SCHEMA_VERSION:
        record = abc1_record.normalize(record)
    return _project_compact(
        dataset=dataset, source_path=source_path,
        byte_offset=byte_offset, record_sha256=record_sha256,
        record=record, allowed_tasks=allowed_tasks)
