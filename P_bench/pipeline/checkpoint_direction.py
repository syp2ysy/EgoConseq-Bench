"""Compile the non-headline A4 Forward-in-progress direction diagnostic."""

from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from pipeline import actions as action_geometry
from pipeline import config, io_utils, objects, outcome as outcome_fields
from pipeline import record as record_fields
from pipeline import viz


DIAGNOSTIC_ID = "checkpoint_direction"
SCHEMA_VERSION = "checkpoint_direction.v1"
FRACTIONS = (0.25, 0.5, 0.75)
FRACTION_TEXT = {
    0.25: "one quarter",
    0.5: "half",
    0.75: "three quarters",
}
CHOICES = [
    {"id": value, "text": value}
    for value in ("front", "left", "right", "rear")
]
MAX_ITEMS_PER_FRAME = 3


def _digest(*parts) -> str:
    return hashlib.sha256(
        ":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def build_forward_checkpoint(actions, *, forward_stage: int,
                             fraction: float) -> dict:
    """Return one strictly interior point of a 1-based Forward stage."""
    if fraction not in FRACTIONS:
        raise ValueError("checkpoint fraction must be 0.25, 0.5, or 0.75")
    if (not isinstance(forward_stage, int) or
            isinstance(forward_stage, bool) or forward_stage <= 0):
        raise ValueError("forward_stage must be a positive integer")
    cumulative = 0.0
    stage = 0
    for primitive_index, action in enumerate(actions):
        if not isinstance(action, action_geometry.Forward):
            continue
        stage += 1
        if stage == forward_stage:
            arc = cumulative + float(fraction) * float(action.m)
            x, z, heading = action_geometry.pose_at_arc(actions, arc)
            return {
                "action_index": int(primitive_index + 1),
                "forward_stage": int(stage),
                "fraction": float(fraction),
                "arc_m": float(arc),
                "pose": {
                    "x": float(x), "z": float(z),
                    "heading_deg": float(heading),
                },
            }
        cumulative += float(action.m)
    raise ValueError("forward_stage is outside the action program")


def select_forward_checkpoint(actions, *, seed: str) -> dict:
    """Choose one Forward and one interior fraction without reading GT."""
    stages = sum(
        isinstance(action, action_geometry.Forward) for action in actions)
    if stages <= 0:
        raise ValueError("checkpoint direction requires a Forward action")
    digest = bytes.fromhex(_digest(seed))
    stage = int.from_bytes(digest[:8], "big") % stages + 1
    fraction = FRACTIONS[int.from_bytes(digest[8:16], "big") % len(FRACTIONS)]
    return build_forward_checkpoint(
        actions, forward_stage=stage, fraction=fraction)


def _finite_xz(value) -> list[float] | None:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return [float(point[0]), float(point[1])]


def _b_target_referent(record: dict) -> dict | None:
    target, _reason = record_fields.authenticated_b_target(record)
    if target is None:
        return None
    selection = target["selection"]
    centroid = target["geometry"]["reference_centroid"]
    return {
        "track": "b_target",
        "instance_id": int(selection["instance_id"]),
        "category": str(selection["category"]),
        "name": str(selection["name"]),
        "marker": selection.get("marker"),
        "anchor_protocol": str(centroid["protocol"]),
        "centroid_world_xz_m": [
            float(value) for value in centroid["world_xz_m"]],
        "nearest_world_xz_m": None,
    }


def _visible_object_referents(record: dict) -> list[dict]:
    inventory = {
        row["instance_id"]: row
        for row in objects.initial_visible_entity_inventory(record)
    }
    category_counts = collections.Counter(
        str(row.get("category") or "").casefold()
        for row in record.get("objects") or [])
    result = []
    pose = record.get("pose") or {}
    for raw in record.get("objects") or []:
        category = str(raw.get("category") or "").strip()
        try:
            instance_id = int(raw.get("instance_id"))
        except (TypeError, ValueError):
            continue
        entity = inventory.get(instance_id)
        centroid = _finite_xz(raw.get("ground_xy_centroid"))
        nearest = _finite_xz(raw.get("ground_xy_nearest"))
        if (raw.get("is_structural") is not False or not category or
                category.casefold() == "unknown" or
                not config.is_specific_semantic_category(category) or
                category_counts[category.casefold()] != 1 or
                entity is None or entity.get("marker") is not None or
                centroid is None or nearest is None):
            continue
        try:
            centroid_world = record_fields.local_ground_xz_to_world(
                pose, centroid)
            nearest_world = record_fields.local_ground_xz_to_world(
                pose, nearest)
        except (TypeError, ValueError):
            continue
        result.append({
            "track": "visible_object",
            "instance_id": instance_id,
            "category": category,
            "name": str(entity["name"]),
            "marker": None,
            "anchor_protocol": "initial-visible-depth-centroid.v1",
            "centroid_world_xz_m": [
                float(centroid_world[0]), float(centroid_world[1])],
            "nearest_world_xz_m": [
                float(nearest_world[0]), float(nearest_world[1])],
        })
    return sorted(result, key=lambda row: row["instance_id"])


def referents_for_record(record: dict) -> list[dict]:
    """Return one authenticated B target plus unambiguous visible objects."""
    target = _b_target_referent(record)
    visible = _visible_object_referents(record)
    if target is not None:
        visible = [
            row for row in visible
            if row["instance_id"] != target["instance_id"]]
        return [target, *visible]
    return visible


def _direction_at_checkpoint(record: dict, checkpoint: dict,
                             referent: dict) -> tuple[dict | None, str | None]:
    relation = record_fields.build_direction_relation_at_pose(
        pose=record["pose"], checkpoint=checkpoint["pose"],
        target_world_xz_m=referent["centroid_world_xz_m"])
    if relation["direction_status"] != "computed":
        return None, "target_range_undefined"
    try:
        direction, margin = record_fields.b_direction_with_margin(
            relation["bearing_deg"])
    except (TypeError, ValueError):
        return None, "bearing_boundary_margin_failed"
    if direction != relation["direction"]:
        raise ValueError("checkpoint direction quantization disagrees")
    if referent["nearest_world_xz_m"] is not None:
        nearest = record_fields.build_direction_relation_at_pose(
            pose=record["pose"], checkpoint=checkpoint["pose"],
            target_world_xz_m=referent["nearest_world_xz_m"])
        if (nearest["direction_status"] != "computed" or
                nearest["direction"] != direction):
            return None, "anchor_sector_disagreement"
    return {**relation, "sector_boundary_margin_deg": float(margin)}, None


def _marker_set(record: dict, referent: dict) -> list[dict]:
    category = referent["category"].casefold()
    return [
        row["marker"]
        for row in objects.initial_visible_entity_inventory(record)
        if row["category"].casefold() == category and
        row["marker"] is not None
    ]


def _input_asset(record: dict, image: Path, referent: dict, *,
                 item_digest: str, asset_dir: Path, raw_sha256: str,
                 encoded_cache: dict, authenticated_cache: dict) -> dict:
    artifact_root = asset_dir.parent.resolve()
    raw_path = os.path.relpath(image.resolve(), artifact_root)
    marker = referent.get("marker")
    if marker is None:
        return {
            "marked": False,
            "path": raw_path,
            "sha256": raw_sha256,
            "raw_path": raw_path,
            "raw_sha256": raw_sha256,
        }
    sensor = record.get("sensor") or {}
    authenticated = authenticated_cache.get("image")
    if authenticated is None:
        authenticated = viz.authenticate_raw_rgb_image(
            image, expected_sha256=raw_sha256,
            expected_resolution=sensor.get("resolution"))
        authenticated_cache["image"] = authenticated
    materialized = viz.materialize_numbered_dot_image(
        authenticated, _marker_set(record, referent), asset_dir,
        "d-" + item_digest[:16], encoded_cache=encoded_cache)
    materialized["path"] = os.path.relpath(
        Path(materialized["path"]).resolve(), artifact_root)
    materialized["raw_path"] = raw_path
    return materialized


def _safe_outcome(outcome: dict) -> bool:
    summary = (outcome.get("shared_oracle_stability") or {}).get("summary")
    return (
        isinstance(summary, dict) and
        summary.get("collision") is False and
        summary.get("collision_label_stable") is True and
        outcome_fields.is_completed_clear(outcome)
    )


def compile_record(record: dict, *, asset_root, asset_dir,
                   encoded_cache=None) -> dict:
    """Compile at most two A4 items per safe outcome from one record."""
    asset_root = Path(asset_root)
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    encoded_cache = {} if encoded_cache is None else encoded_cache
    image_value = Path(str(record.get("image_path") or ""))
    image = image_value if image_value.is_absolute() else asset_root / image_value
    if not image.is_file():
        raise ValueError(f"record RGB asset is missing: {image}")
    raw_sha256 = io_utils.sha256_file(image)
    authenticated_cache = {}
    referents = referents_for_record(record)
    items = []
    answers = []
    funnel = collections.Counter()
    funnel["records"] = 1
    dataset = str((record.get("source") or {}).get("source_dataset") or "unknown")
    frame_id = str(record.get("observation_id") or record.get("frame_id"))
    for outcome in record.get("outcomes") or []:
        funnel["outcomes"] += 1
        if not _safe_outcome(outcome):
            funnel["outcome_not_completed_clear"] += 1
            continue
        funnel["safe_outcomes"] += 1
        parsed = action_geometry.parse_actions(outcome.get("actions") or [])
        try:
            checkpoint = select_forward_checkpoint(
                parsed, seed=f"{frame_id}:{outcome.get('outcome_id')}")
        except ValueError:
            funnel["outcome_without_forward"] += 1
            continue
        fraction_key = str(checkpoint["fraction"])
        funnel[f"chosen_fraction.{fraction_key}"] += 1
        eligible = []
        for referent in referents:
            funnel["referents_considered"] += 1
            relation, reason = _direction_at_checkpoint(
                record, checkpoint, referent)
            if reason is not None:
                funnel[reason] += 1
                continue
            eligible.append((referent, relation))
        if not eligible:
            funnel["chosen_checkpoint_all_rejected"] += 1
            continue
        target = [row for row in eligible if row[0]["track"] == "b_target"]
        visible = [
            row for row in eligible if row[0]["track"] == "visible_object"]
        visible.sort(key=lambda row: _digest(
            frame_id, outcome.get("outcome_id"),
            checkpoint["arc_m"], row[0]["instance_id"]))
        selected = (target[:1] + visible[:1]) if target else visible[:2]
        for referent, relation in selected:
            item_digest = _digest(
                frame_id, outcome.get("outcome_id"),
                checkpoint["action_index"], checkpoint["fraction"],
                referent["instance_id"])
            item_id = "a4-" + item_digest[:16]
            try:
                input_asset = _input_asset(
                    record, image, referent, item_digest=item_digest,
                    asset_dir=asset_dir, raw_sha256=raw_sha256,
                    encoded_cache=encoded_cache,
                    authenticated_cache=authenticated_cache)
            except ValueError:
                funnel["marker_unrenderable"] += 1
                continue
            initial = record_fields.build_direction_relation_at_pose(
                pose=record["pose"],
                checkpoint={"x": 0.0, "z": 0.0, "heading_deg": 0.0},
                target_world_xz_m=referent["centroid_world_xz_m"])
            fraction_text = FRACTION_TEXT[checkpoint["fraction"]]
            question = (
                f"During action {checkpoint['action_index']} "
                f"(Forward move {checkpoint['forward_stage']}), when the "
                f"robot has traveled {fraction_text} of that Forward, where "
                f"is {referent['name']} relative to the robot's facing "
                "direction at that moment?")
            model_input = {
                "initial_rgb": input_asset["path"],
                "initial_rgb_sha256": input_asset["sha256"],
                "camera_height_above_visible_floor_m": round(float(
                    record["camera_height_above_visible_floor_m"]), 2),
                "hfov_deg": float(record["sensor"]["hfov_deg"]),
                "vfov_deg": float(record["sensor"]["vfov_deg"]),
                "body_radius_m": float(outcome["body"]["radius_m"]),
                "actions": list(outcome["actions"]),
                "target": referent["name"],
                "checkpoint": {
                    key: checkpoint[key]
                    for key in ("action_index", "forward_stage", "fraction",
                                "arc_m")
                },
            }
            items.append({
                "id": item_id,
                "diagnostic_id": DIAGNOSTIC_ID,
                "protocol": SCHEMA_VERSION,
                "headline_eligible": False,
                "result_head": "A",
                "question": question,
                "answer_format": "closed_exact",
                "model_input": model_input,
                "choices": list(CHOICES),
                "metadata": {
                    "dataset": dataset,
                    "scene_id": str(record.get("scene_id") or
                                    (record.get("source") or {}).get(
                                        "scene_id") or "unknown"),
                    "referent_track": referent["track"],
                    "anchor_protocol": referent["anchor_protocol"],
                },
            })
            answers.append({
                "id": item_id,
                "diagnostic_id": DIAGNOSTIC_ID,
                "canonical_answer": relation["direction"],
                "precise_bearing_deg": relation["bearing_deg"],
                "sector_boundary_margin_deg":
                    relation["sector_boundary_margin_deg"],
                "initial_direction": initial["direction"],
                "sector_change": (
                    "same_sector" if initial["direction"] ==
                    relation["direction"] else "changed_sector"),
                "frame_id": record.get("frame_id"),
                "outcome_id": outcome.get("outcome_id"),
                "instance_id": referent["instance_id"],
                "checkpoint": dict(model_input["checkpoint"]),
                "referent_track": referent["track"],
                "anchor_protocol": referent["anchor_protocol"],
                "input_asset": input_asset,
            })
            funnel["final_items"] += 1
            funnel[f"items.dataset.{dataset}"] += 1
            funnel[f"items.fraction.{fraction_key}"] += 1
            funnel[f"items.track.{referent['track']}"] += 1
    return {
        "items": items,
        "answers": answers,
        "funnel": dict(sorted(funnel.items())),
    }


def _write_jsonl(path: Path, rows) -> None:
    io_utils.atomic_write_text(
        path, "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows))


def select_diverse_run_items(items: list[dict], answers: list[dict]):
    """Keep at most three distinct directions from each raw source frame."""
    answer_by_id = {str(row.get("id")): row for row in answers}
    if (len(answer_by_id) != len(answers) or
            set(answer_by_id) != {str(item.get("id")) for item in items}):
        raise ValueError("A4 public/private ids are not bijective")
    direction_counts = collections.Counter(
        str(row["canonical_answer"]) for row in answers)
    by_frame = collections.defaultdict(list)
    for item in items:
        answer = answer_by_id[str(item["id"])]
        dataset = str((item.get("metadata") or {}).get("dataset") or "")
        asset = answer.get("input_asset") or {}
        frame_sha256 = str(
            asset.get("raw_sha256") or asset.get("sha256") or "")
        if not dataset or not frame_sha256:
            raise ValueError("A4 source-frame identity is missing")
        by_frame[(dataset, frame_sha256)].append((item, answer))

    keep_ids = set()
    for (_dataset, frame_sha256), rows in sorted(by_frame.items()):
        by_direction = collections.defaultdict(list)
        for item, answer in rows:
            by_direction[str(answer["canonical_answer"])].append(
                (item, answer))
        directions = sorted(
            by_direction,
            key=lambda direction: (
                direction_counts[direction],
                _digest(frame_sha256, direction)))
        used_fractions = set()
        for direction in directions[:MAX_ITEMS_PER_FRAME]:
            candidates = sorted(
                by_direction[direction],
                key=lambda row: (
                    (row[0].get("model_input") or {}).get(
                        "checkpoint", {}).get("fraction") in used_fractions,
                    _digest(frame_sha256, row[0]["id"])))
            item, _answer = candidates[0]
            keep_ids.add(str(item["id"]))
            used_fractions.add(
                (item.get("model_input") or {}).get(
                    "checkpoint", {}).get("fraction"))

    kept_items = [item for item in items if str(item["id"]) in keep_ids]
    kept_answers = [
        answer for answer in answers if str(answer["id"]) in keep_ids]
    retained_directions = collections.Counter(
        str(row["canonical_answer"]) for row in kept_answers)
    report = {
        "policy": "checkpoint-direction-diversity.v1",
        "source_frame_identity": "source_dataset+raw_rgb_sha256",
        "unique_source_frames": len(by_frame),
        "input_items": len(items),
        "retained_items": len(kept_items),
        "dropped_items": len(items) - len(kept_items),
        "by_sector": {
            direction: int(retained_directions.get(direction, 0))
            for direction in ("front", "left", "right", "rear")
        },
    }
    return kept_items, kept_answers, report


def cap_run_items(
        items: list[dict], answers: list[dict], *,
        max_items: int = config.BACKGROUND_MAX_A4_ITEMS
        ) -> tuple[list[dict], list[dict]]:
    """Apply the deterministic corpus-level A4 publication cap."""
    if len(items) <= int(max_items):
        return items, answers
    keep_ids = {
        str(item["id"])
        for item in sorted(
            items, key=lambda row: _digest("global-cap", row["id"]))[
                :int(max_items)]
    }
    return (
        [item for item in items if str(item["id"]) in keep_ids],
        [answer for answer in answers if str(answer["id"]) in keep_ids],
    )


def build_run_artifact(run_root, output) -> dict:
    """Compile every records.jsonl below one run into an A4 sidecar."""
    run_root = Path(run_root)
    output = Path(output)
    records_paths = sorted((run_root / "records").rglob("records.jsonl"))
    if not records_paths:
        raise ValueError(f"run has no records.jsonl files: {run_root}")
    items = []
    answers = []
    funnel = collections.Counter()
    source_files = []
    encoded_cache = {}
    seen_ids = set()
    record_count = 0
    for records_path in records_paths:
        rows = io_utils.read_jsonl(records_path, require_dict=True)
        source_files.append({
            "path": str(records_path.relative_to(run_root)),
            "sha256": io_utils.sha256_file(records_path),
            "record_count": len(rows),
        })
        for record in rows:
            record_count += 1
            projection = compile_record(
                record, asset_root=records_path.parent,
                asset_dir=output / "assets",
                encoded_cache=encoded_cache)
            for key, value in projection["funnel"].items():
                funnel[key] += int(value)
            for item, answer in zip(
                    projection["items"], projection["answers"]):
                if item["id"] != answer["id"] or item["id"] in seen_ids:
                    raise ValueError("duplicate or mismatched A4 item id")
                seen_ids.add(item["id"])
                items.append(item)
                answers.append(answer)
    items, answers, diversity_report = select_diverse_run_items(items, answers)
    pre_cap_items = len(items)
    items, answers = cap_run_items(items, answers)
    diversity_report["global_max_items"] = config.BACKGROUND_MAX_A4_ITEMS
    diversity_report["global_cap_dropped_items"] = pre_cap_items - len(items)
    diversity_report["retained_items"] = len(items)
    funnel["diversity_retained_items"] = len(items)
    funnel["diversity_dropped_items"] = (
        diversity_report["dropped_items"] +
        diversity_report["global_cap_dropped_items"])
    referenced_assets = {
        Path(str(item["model_input"]["initial_rgb"])).name
        for item in items}
    for path in (output / "assets").glob("*.png"):
        if path.name not in referenced_assets:
            path.unlink()
    by_dataset = collections.Counter(
        item["metadata"]["dataset"] for item in items)
    by_fraction = collections.Counter(
        str(item["model_input"]["checkpoint"]["fraction"])
        for item in items)
    by_forward_stage = collections.Counter(
        str(item["model_input"]["checkpoint"]["forward_stage"])
        for item in items)
    by_sector = collections.Counter(
        answer["canonical_answer"] for answer in answers)
    by_track = collections.Counter(
        answer["referent_track"] for answer in answers)
    by_anchor_protocol = collections.Counter(
        answer["anchor_protocol"] for answer in answers)
    by_sector_change = collections.Counter(
        answer["sector_change"] for answer in answers)
    item_count = len(items)
    majority = (
        max(by_sector.values()) / item_count if item_count else None)
    persistence = (
        sum(answer["initial_direction"] == answer["canonical_answer"]
            for answer in answers) / item_count if item_count else None)
    report = {
        "schema": "checkpoint_direction.report.v1",
        "diagnostic_id": DIAGNOSTIC_ID,
        "headline_eligible": False,
        "record_count": record_count,
        "item_count": item_count,
        "funnel": dict(sorted(funnel.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_fraction": {
            str(value): int(by_fraction.get(str(value), 0))
            for value in FRACTIONS
        },
        "by_forward_stage": dict(sorted(by_forward_stage.items())),
        "by_sector": {
            value: int(by_sector.get(value, 0))
            for value in ("front", "left", "right", "rear")
        },
        "by_referent_track": dict(sorted(by_track.items())),
        "by_anchor_protocol": dict(sorted(by_anchor_protocol.items())),
        "by_sector_change": dict(sorted(by_sector_change.items())),
        "majority_answer_accuracy": majority,
        "s0_persistence_oracle_accuracy": persistence,
        "diversity_selection": diversity_report,
    }
    manifest = {
        "schema": SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "paper_alias": "A4",
        "headline_eligible": False,
        "source_files": source_files,
        "source_record_count": record_count,
        "item_count": item_count,
        "answer_count": len(answers),
    }
    _write_jsonl(output / "items.jsonl", items)
    _write_jsonl(output / "private" / "answers.jsonl", answers)
    io_utils.atomic_write_json(
        output / "manifest.json", manifest, allow_nan=False)
    io_utils.atomic_write_json(
        output / "report.json", report, allow_nan=False)
    return {
        "artifact": str(output),
        "headline_eligible": False,
        "record_count": record_count,
        "item_count": item_count,
        "report": report,
    }
