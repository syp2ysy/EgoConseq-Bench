"""Compile selected record outcomes into public QA and a private index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

from pipeline import abc1_record
from pipeline import candidate_sources
from pipeline import io_utils
from pipeline import record as record_schema
from pipeline import viz
from pipeline import actions, checkpoint_direction
from pipeline.benchmark_candidates import Candidate
from post_QA import answers, categories, templates
from post_QA.seen_build import catalog, spec


Source = catalog.Source


def sources(records_root: Path) -> dict[str, Source]:
    return {source.dataset: source for source in catalog.load(records_root)}


def humanize_category(value: object) -> str:
    value = re.sub(
        r"\.[anvr]\.\d+$", "", str(value or "").strip(),
        flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip().casefold()


def release_records_path(dataset: str) -> str:
    """Path stored relative to an immutable release version directory."""
    return f"records/{str(dataset)}/records.jsonl"



def _case(record: dict, case_id: str) -> dict:
    return next(
        value for value in record["cases"]
        if str(value["case_id"]) == str(case_id))



def _compact_model(record: dict, case: dict, image: Path,
                   image_sha256: str) -> dict:
    sensor = record["sensor"]
    return {
        "initial_rgb": str(image),
        "initial_rgb_sha256": image_sha256,
        "camera_height_m": float(sensor["nominal_camera_offset_m"]),
        "hfov_deg": float(sensor["hfov_deg"]),
        "vfov_deg": float(sensor["vfov_deg"]),
        "body_radius_m": float(case["body_radius_m"]),
        "actions": list(case["actions"]),
    }


def _compact_c1(record: dict, case: dict, asset_root: Path,
                candidate: Candidate, cache: dict) -> tuple[dict, list[dict]]:
    family = next(
        value for value in record.get("c1_families") or []
        if str(value["query_case_id"]) == str(case["case_id"]))
    members = [_case(record, case_id)
               for case_id in family["member_case_ids"]]
    members.sort(key=lambda value: hashlib.sha256(
        f"{candidate.item_id}:{value['case_id']}".encode()).hexdigest())
    choices = []
    canonical = None
    for index, member in enumerate(members, 1):
        choice_id = f"image_{index}"
        image = candidate_sources.resolve_record_asset(
            asset_root, member["terminal_rgb_path"])
        key = ("sha256", str(image))
        if key not in cache:
            cache[key] = io_utils.sha256_file(image)
        choices.append({
            "id": choice_id,
            "outcome_id": member["case_id"],
            "action_sha256": record_schema.action_program_sha256(
                member["actions"]),
            "image": str(image),
            "image_sha256": cache[key],
        })
        if member["case_id"] == case["case_id"]:
            canonical = choice_id
    return {"canonical_answer": canonical}, choices


def _project_compact(candidate: Candidate, source: Source, record: dict,
                     build_root: Path, cache: dict) -> tuple[dict, dict, list, str]:
    case = _case(record, candidate.outcome_id)
    asset_root = source.records_path.parent
    image = candidate_sources.resolve_record_asset(
        asset_root, record["image_path"])
    surface_task = candidate.task_id in {"A4", "B1", "B2", "B3"}
    raw_key = ("rgb", str(image))
    if surface_task and raw_key not in cache:
        cache[raw_key] = viz.authenticate_raw_rgb_image(
            image, expected_resolution=record["sensor"]["resolution"])
    authenticated = cache.get(raw_key)
    sha_key = ("sha256", str(image))
    if sha_key not in cache:
        cache[sha_key] = (authenticated.sha256 if authenticated is not None
                          else io_utils.sha256_file(image))
    image_sha256 = cache[sha_key]
    model = _compact_model(record, case, image, image_sha256)
    choices = []
    output = (case.get("task_outputs") or {}).get(candidate.task_id) or {}
    if candidate.task_id == "B3":
        output = {
            "checkpoint": checkpoint_direction.build_forward_checkpoint(
                actions.parse_actions(case["actions"]), forward_stage=candidate.checkpoint_ordinal,
                fraction=candidate.checkpoint_fraction),
            "points": [{"point_id": candidate.point_id,
                        "endpoint_distance_m": candidate.numeric_value_m}],
        }
    if candidate.task_id == "A1":
        private = {"canonical_answer": str(output["answer"])}
    elif candidate.task_id == "A2":
        private = {"canonical_answer": (
            f"action_{int(output['collision_action_index_1based'])}")}
    elif candidate.task_id == "A3":
        private = {"canonical_answer": categories.a3_answer(record, output)}
    elif candidate.task_id == "C1":
        private, choices = _compact_c1(
            record, case, asset_root, candidate, cache)
    else:
        target = record["surface_point_target"]
        point_id = candidate.point_id or "p1"
        point = next(p for p in target["points"] if p["point_id"] == point_id)
        point_output = next(p for p in output["points"]
                            if p["point_id"] == point_id)
        model["target"] = "the marked point"
        marked_key = ("marked", record["record_uid"], point_id)
        if marked_key not in cache:
            cache[marked_key] = viz.materialize_target_point_image(
                authenticated, point, build_root / "marked_inputs" /
                f"{record['record_uid']}-{point_id}.png")
        marked = cache[marked_key]
        model["initial_rgb"] = marked["path"]
        model["initial_rgb_sha256"] = marked["sha256"]
        if candidate.task_id in {"A4", "B3"}:
            model["checkpoint"] = output["checkpoint"]
        relation = {
            "schema": "surface-point-relation.v4",
            "horizontal_direction": point_output.get("horizontal_direction"),
            "vertical_direction": point_output.get("vertical_direction"),
        }
        if candidate.task_id in {"B1", "B3"}:
            relation["camera_to_target_distance_m"] = float(
                point_output["endpoint_distance_m"])
        private = {
            "canonical_answer": str(point_output.get("answer") or "distance"),
            "horizontal_direction": point_output.get("horizontal_direction"),
            "vertical_direction": point_output.get("vertical_direction"),
            "relation": relation,
            "target_instance_id": int(target["instance_id"]),
            "input_asset": marked,
        }
        if candidate.task_id == "A4":
            private.update({
                "instance_id": int(target["instance_id"]),
                "outcome_id": case["case_id"],
            })
        elif candidate.task_id in {"B1", "B3"}:
            private.update({
                "canonical_answer": "distance",
                "camera_to_target_distance_m": relation[
                    "camera_to_target_distance_m"],
            })
    return model, private, choices, image_sha256


def _compact(record: dict, candidate: Candidate) -> dict:
    if not abc1_record.is_compact(record):
        return abc1_record.from_legacy(
            record, dataset=candidate.dataset,
            source_records_sha256=candidate.record_sha256,
            byte_offset=candidate.byte_offset)
    if record["schema_version"] != abc1_record.SCHEMA_VERSION:
        return abc1_record.normalize(record)
    return record


def project(candidate: Candidate, source: Source, record: dict,
            build_root: Path, *, image_cache: dict | None = None
            ) -> tuple[dict, dict, list, str]:
    """Render certified task outputs, without repeating their eligibility checks."""
    return _project_compact(candidate, source, _compact(record, candidate), build_root,
                            {} if image_cache is None else image_cache)


def _number(value: object) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _inputs(model: dict) -> dict:
    actions = []
    for index, raw in enumerate(model.get("actions") or [], 1):
        if raw["type"] == "forward":
            meters = float(raw.get("m", raw.get("meters")))
            action = {"index": index, "type": "forward", "meters": meters,
                      "text": f"forward {_number(meters)} m"}
        else:
            degrees = float(raw.get("deg", raw.get("degrees")))
            side = "right" if degrees >= 0 else "left"
            action = {"index": index, "type": "turn", "degrees": degrees,
                      "text": f"turn {side} {_number(abs(degrees))} degree"}
        actions.append(action)
    value = {
        "camera": {
            "optical_center_height_m": float(
                model["camera_height_m"]),
            "hfov_deg": float(model["hfov_deg"]),
            "vfov_deg": float(model["vfov_deg"]),
        },
        "robot": {"radius_m": float(model["body_radius_m"])},
        "actions": actions,
    }
    if model.get("target") is not None:
        value["target"] = str(model["target"])
    if model.get("checkpoint") is not None:
        checkpoint = model["checkpoint"]
        fraction = float(checkpoint["fraction"])
        value["checkpoint"] = {
            "action_index": int(checkpoint["action_index"]),
            "fraction": fraction,
        }
    return value


def _stage_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    io_utils.link_or_copy_file(source, destination)


def _template(task_id: str, template_id: str) -> dict:
    return next(row for row in templates.QUESTION_TEMPLATES[task_id]
                if row["id"] == template_id)



def materialize_item(
        candidate: Candidate, source: Source, record: dict,
        artifact_root: Path, build_root: Path) -> tuple[dict, dict]:
    record = _compact(record, candidate)
    model, private, choices, raw_image_sha256 = project(
        candidate, source, record, build_root)
    outcome = _case(record, candidate.outcome_id)
    sensor = record["sensor"]
    usage = {
        "template_id": candidate.template_id,
        "outcome_id": candidate.outcome_id,
        "action_program_sha256": record_schema.action_program_sha256(
            outcome["actions"]),
        "actions": outcome["actions"],
        "action_length": candidate.action_length,
        "starts_with": candidate.starts_with,
        "body_radius_m": candidate.body_radius_m,
        "camera_height_m": float(sensor["nominal_camera_offset_m"]),
        "nominal_camera_offset_m": candidate.camera_height_m,
        "hfov_deg": candidate.hfov_deg,
        "vfov_deg": float(sensor["vfov_deg"]),
        "resolution_px": [int(value) for value in sensor["resolution"]],
    }
    initial_relative = Path("images/egocentric") / f"{candidate.item_id}.png"
    _stage_image(Path(model["initial_rgb"]), artifact_root / initial_relative)
    image_paths = [initial_relative.as_posix()]
    if candidate.task_id == "C1":
        canonical_id = str(private["canonical_answer"])
        correct = next(choice for choice in choices
                       if str(choice["id"]) == canonical_id)
        distractors = [choice for choice in choices if choice is not correct]
        correct_index = "ABCD".index(str(candidate.c1_label))
        choices = [correct if index == correct_index else distractors.pop(0)
                   for index in range(4)]
        for label, choice in zip("ABCD", choices):
            relative = Path("images/c1") / f"{candidate.item_id}_{label}.png"
            _stage_image(Path(choice["image"]), artifact_root / relative)
            image_paths.append(relative.as_posix())
        usage["candidate_labels"] = (
            {
                label: {
                    "outcome_id": str(choice["outcome_id"]),
                    "action_program_sha256": str(choice["action_sha256"]),
                }
                for label, choice in zip("ABCD", choices)
            })
        private["canonical_answer"] = candidate.c1_label
    elif candidate.task_id == "A3":
        usage["contact_category"] = private["canonical_answer"]
    elif candidate.task_id == "A2":
        usage["collision_action_index_1based"] = int(
            str(private["canonical_answer"]).removeprefix("action_"))
        usage["a2_design"] = {
            "cell": {
                "forward_ordinal_1based": candidate.a2_ordinal,
                "distance_rank": candidate.a2_rank,
            },
        }
    elif candidate.task_id in {"A4", "B3"}:
        checkpoint = model["checkpoint"]
        usage.update({
            "target_instance_id": int(private["target_instance_id"]),
            "checkpoint": {
                "action_index_1based": int(checkpoint["action_index"]),
                "forward_ordinal_1based": int(checkpoint["forward_stage"]),
                "fraction": float(checkpoint["fraction"]),
            },
        })
    elif candidate.task_id in {"B1", "B2"}:
        usage["target_instance_id"] = int(private["target_instance_id"])
    if candidate.task_id in {"A4", "B1", "B2", "B3"}:
        target = record["surface_point_target"]
        input_asset = private["input_asset"]
        source_category = str(target.get("source_category") or target["category"])
        surface_anchor = next(
            point for point in target["points"]
            if point["point_id"] == (candidate.point_id or "p1"))
        usage["target_point"] = {
            "instance_id": int(target["instance_id"]),
            "category": {
                "source_label": source_category,
                "display_name": humanize_category(source_category),
            },
            "surface_anchor": surface_anchor,
            "relation": private["relation"],
            "marker": input_asset["marker"],
            "marked_image_sha256": input_asset["sha256"],
        }
    question = templates.render_question(
        candidate.task_id, _inputs(model),
        _template(candidate.task_id, str(candidate.template_id)))
    item = {
        "id": candidate.item_id,
        "task_id": candidate.task_id,
        "dataset": candidate.dataset,
        "images": image_paths,
        "messages": [
            {"role": "system", "content": templates.SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answers.canonical_answer(
                candidate.task_id, private)},
        ],
    }
    entry = {
        "item_id": candidate.item_id,
        "task_id": candidate.task_id,
        "dataset": candidate.dataset,
        "scene_id": candidate.scene_id,
        "record_uid": str(record.get("record_uid") or candidate.record_id),
        "frame_id": str(record.get("observation_id") or record["frame_id"]),
        "record_sha256": candidate.record_sha256,
        "initial_image_sha256": raw_image_sha256,
        "pose": record["pose"],
        "source": {
            "records_path": (source.index_path or
                             release_records_path(candidate.dataset)),
            "records_sha256": source.records_sha256,
            "run_meta_sha256": source.run_meta_sha256,
            "byte_offset": candidate.byte_offset,
        },
        "usage": usage,
    }
    return item, entry


def write_index(entries: Iterable[dict], qa_path: Path,
                output_path: Path) -> dict:
    rows = list(entries)
    qa = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    if {row["item_id"] for row in rows} != {item["id"] for item in qa}:
        raise ValueError("record index differs from qa.json")
    document = {
        "schema": "egoconseq.seen-record-index.v3",
        "benchmark": spec._SPEC["benchmark_id"],
        "qa_sha256": io_utils.sha256_file(qa_path),
        "record_count": len(rows),
        "items": rows,
    }
    io_utils.atomic_write_json(output_path, document, allow_nan=False)
    return document
