"""A4 Forward-in-progress checkpoint-direction diagnostics."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from PIL import Image
import pytest

from pipeline import actions, checkpoint_direction, record
from tests.test_pl_v16_b_candidates import _b_case


def _object(instance_id: int, category: str, centroid, nearest=None,
            *, structural: bool = False) -> dict:
    nearest = centroid if nearest is None else nearest
    return {
        "instance_id": instance_id,
        "category": category,
        "is_structural": structural,
        "mask_area_px": 900,
        "depth_backed_px": 900,
        "bbox_xyxy_px": [10, 10, 49, 49],
        "centroid_px": [30.0, 30.0],
        "ground_xy_centroid": list(centroid),
        "ground_xy_nearest": list(nearest),
    }


def _safe_record(objects: list[dict]) -> dict:
    return {
        "frame_id": "frame-a4",
        "scene_id": "scene-a4",
        "image_path": "img/frame-a4.png",
        "source": {"source_dataset": "r2r"},
        "pose": {
            "position": [0.0, 0.0, 0.0],
            "yaw_rad": 0.0,
        },
        "sensor": {
            "resolution": [64, 48],
            "hfov_deg": 110.0,
            "vfov_deg": 90.0,
        },
        "camera_height_above_visible_floor_m": 1.0,
        "objects": objects,
        "outcomes": [{
            "outcome_id": "outcome-a4",
            "body": {"shape": "disc", "radius_m": 0.2},
            "actions": [{"type": "forward", "m": 1.0}],
            "physical": {
                "authority": "test_geometry",
                "collision": False,
            },
            "execution": {
                "completed": True,
                "stop_reason": "completed",
                "nominal_forward_m": 1.0,
                "executed_forward_m": 1.0,
            },
            "shared_oracle_stability": {
                "summary": {
                    "collision": False,
                    "collision_label_stable": True,
                },
            },
        }],
    }


def test_direction_relation_uses_world_target_and_local_checkpoint():
    """Catches mixing world-space B anchors with pose-local checkpoints."""
    relation = record.build_direction_relation_at_pose(
        pose={
            "position": [10.0, 0.0, 20.0],
            "yaw_rad": math.pi / 2.0,
        },
        checkpoint={"x": 0.0, "z": 1.0, "heading_deg": 90.0},
        target_world_xz_m=[7.0, 20.0],
    )

    assert relation["checkpoint_world_xz_m"] == [9.0, 20.0]
    assert relation["bearing_deg"] == -90.0
    assert relation["direction"] == "left"
    assert relation["sector_boundary_margin_deg"] == 45.0


def test_forward_checkpoint_is_inside_second_forward_after_turn():
    """Catches using action boundaries or Forward-stage as primitive index."""
    program = [
        actions.Forward(1.0),
        actions.Turn(90.0),
        actions.Forward(2.0),
    ]

    checkpoint = checkpoint_direction.build_forward_checkpoint(
        program, forward_stage=2, fraction=0.5)

    assert checkpoint["action_index"] == 3
    assert checkpoint["forward_stage"] == 2
    assert checkpoint["fraction"] == 0.5
    assert checkpoint["arc_m"] == 2.0
    assert checkpoint["pose"] == pytest.approx({
        "x": 1.0, "z": 1.0, "heading_deg": 90.0})


def test_checkpoint_selection_is_deterministic_and_uses_all_fractions():
    """Catches a fixed fraction or unstable selection replacing hash spread."""
    program = [actions.Forward(1.0)]
    first = checkpoint_direction.select_forward_checkpoint(
        program, seed="same")
    second = checkpoint_direction.select_forward_checkpoint(
        program, seed="same")
    observed = {
        checkpoint_direction.select_forward_checkpoint(
            program, seed=f"seed-{index}")["fraction"]
        for index in range(60)
    }

    assert first == second
    assert observed == {0.25, 0.5, 0.75}


def test_referents_include_authenticated_b_target_with_actual_protocol():
    """Catches hard-coding every B target as complete source geometry."""
    rec, _outcome = _b_case()

    referents = checkpoint_direction.referents_for_record(rec)

    b_target = next(row for row in referents if row["track"] == "b_target")
    assert b_target["instance_id"] == 7
    assert b_target["anchor_protocol"] == \
        "full-triangle-area-weighted-centroid.v1"
    assert b_target["centroid_world_xz_m"] == [0.0, -2.0]


def test_visible_referents_reject_duplicates_structures_and_bad_coordinates():
    """Catches ambiguous or non-geometric objects entering public questions."""
    rec = _safe_record([
        _object(1, "chair", [0.0, 4.0]),
        _object(2, "table", [1.0, 4.0]),
        _object(3, "table", [-1.0, 4.0]),
        _object(4, "wall", [0.0, 3.0], structural=True),
        _object(5, "lamp", [float("nan"), 2.0]),
    ])

    referents = checkpoint_direction.referents_for_record(rec)

    assert [(row["instance_id"], row["name"]) for row in referents] == [
        (1, "chair")]


def test_compile_record_is_safe_only_and_caps_two_referents(tmp_path):
    """Catches correlated object-rich frames emitting unbounded A4 items."""
    rec = _safe_record([
        _object(1, "chair", [0.0, 5.0]),
        _object(2, "table", [1.0, 5.0]),
        _object(3, "lamp", [-1.0, 5.0]),
        _object(4, "sofa", [4.0, 2.0], nearest=[-4.0, 2.0]),
    ])
    image = tmp_path / "img" / "frame-a4.png"
    image.parent.mkdir()
    Image.new("RGB", (64, 48), (8, 9, 10)).save(image)

    projection = checkpoint_direction.compile_record(
        rec, asset_root=tmp_path, asset_dir=tmp_path / "marked")

    assert len(projection["items"]) == 2
    assert len(projection["answers"]) == 2
    assert projection["funnel"]["final_items"] == 2
    assert projection["funnel"]["anchor_sector_disagreement"] == 1
    assert all(
        item["protocol"] == "checkpoint_direction.v1" and
        item["headline_eligible"] is False
        for item in projection["items"])
    assert all(
        "relative to the robot's facing direction" in item["question"]
        for item in projection["items"])
    assert {
        item["model_input"]["checkpoint"]["fraction"]
        for item in projection["items"]
    } <= {0.25, 0.5, 0.75}
    assert all(item["choices"] == [
        {"id": "front", "text": "front"},
        {"id": "left", "text": "left"},
        {"id": "right", "text": "right"},
        {"id": "rear", "text": "rear"},
    ] for item in projection["items"])

    collision = copy.deepcopy(rec)
    outcome = collision["outcomes"][0]
    outcome["physical"]["collision"] = True
    outcome["execution"].update({
        "completed": False,
        "stop_reason": "collision",
        "stop_arc_m": 0.4,
        "executed_forward_m": 0.4,
    })
    outcome["shared_oracle_stability"]["summary"]["collision"] = True

    rejected = checkpoint_direction.compile_record(
        collision, asset_root=tmp_path, asset_dir=tmp_path / "marked")
    assert rejected["items"] == []
    assert rejected["funnel"]["outcome_not_completed_clear"] == 1


def test_run_selector_caps_a4_per_raw_frame_and_prioritizes_rare_directions():
    items = []
    answers = []
    directions = [
        "front", "front", "front", "left", "left", "right", "rear"]
    for index, direction in enumerate(directions):
        item_id = f"a4-{index}"
        items.append({
            "id": item_id,
            "metadata": {"dataset": "gs"},
            "model_input": {"checkpoint": {
                "fraction": (0.25, 0.5, 0.75)[index % 3]}},
        })
        answers.append({
            "id": item_id,
            "canonical_answer": direction,
            "input_asset": {
                "sha256": f"marked-{index}",
                "raw_sha256": "same-raw-frame",
            },
        })

    kept_items, kept_answers, report = \
        checkpoint_direction.select_diverse_run_items(items, answers)

    assert len(kept_items) == checkpoint_direction.MAX_ITEMS_PER_FRAME == 3
    kept_directions = {
        row["canonical_answer"] for row in kept_answers}
    assert "rear" in kept_directions
    assert len(kept_directions) == 3
    assert report["unique_source_frames"] == 1
    assert report["dropped_items"] == 4


def test_global_a4_cap_is_deterministic_and_order_independent():
    items = [{"id": f"a4-{index}"} for index in range(7)]
    answers = [{"id": row["id"]} for row in items]

    first_items, first_answers = checkpoint_direction.cap_run_items(
        items, answers, max_items=3)
    second_items, second_answers = checkpoint_direction.cap_run_items(
        list(reversed(items)), list(reversed(answers)), max_items=3)

    assert checkpoint_direction.config.BACKGROUND_MAX_A4_ITEMS == 60000
    assert {row["id"] for row in first_items} == {
        row["id"] for row in second_items}
    assert {row["id"] for row in first_answers} == {
        row["id"] for row in second_answers}
    assert len(first_items) == len(first_answers) == 3


def test_build_run_artifact_writes_minimal_sidecar_and_funnel(tmp_path):
    """Catches a diagnostic build depending on the six-task compiler."""
    run_root = tmp_path / "run"
    shard = run_root / "records" / "r2r" / "g0" / "scene"
    image = shard / "img" / "frame-a4.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48), (8, 9, 10)).save(image)
    rec = _safe_record([_object(1, "chair", [0.0, 5.0])])
    (shard / "records.jsonl").write_text(
        json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
    output = run_root / "candidate_qa" / "diagnostics" / \
        "checkpoint_direction.v1"

    result = checkpoint_direction.build_run_artifact(run_root, output)

    assert result["headline_eligible"] is False
    assert result["record_count"] == 1
    assert result["item_count"] == 1
    assert (output / "items.jsonl").is_file()
    assert (output / "private" / "answers.jsonl").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "report.json").read_text())
    item = json.loads((output / "items.jsonl").read_text())
    assert manifest["schema"] == "checkpoint_direction.v1"
    assert manifest["headline_eligible"] is False
    assert manifest["source_record_count"] == 1
    assert item["metadata"]["scene_id"] == "scene-a4"
    assert not Path(item["model_input"]["initial_rgb"]).is_absolute()
    assert (output / item["model_input"]["initial_rgb"]).is_file()
    assert report["funnel"]["final_items"] == 1
    assert sum(report["by_fraction"].values()) == 1
