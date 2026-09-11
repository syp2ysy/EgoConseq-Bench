"""Physical point binding and task-preserving surface refinement."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import abc1_record


def _record():
    checkpoint = {
        "action_index": 1, "forward_stage": 1, "fraction": 0.5,
        "arc_m": 0.5, "pose": {"x": 0.0, "z": 0.5, "heading_deg": 0.0},
    }
    safe = {
        "case_id": "safe", "group_id": "safe", "body_radius_m": 0.2,
        "actions": [{"type": "forward", "m": 1.0},
                    {"type": "turn", "deg": 90.0}],
        "starts_with": "forward", "collision": False, "completed": True,
        "endpoint_pose": {"x": 0.0, "z": 1.0, "heading_deg": 90.0},
        "task_outputs": {
            "A1": {"answer": "no_collision"},
            "A4": {"checkpoint_seed": "fixed", "checkpoint": checkpoint},
        },
    }
    collision = {
        **copy.deepcopy(safe), "case_id": "collision", "group_id": "collision",
        "collision": True, "completed": False,
        "task_outputs": {"A1": {"answer": "collision"},
                         "A3": {"instance_id": 7, "category": "chair"}},
    }
    return {
        "schema_version": "abc1.record.v2", "record_uid": "r2r-original",
        "dataset": "r2r", "scene_id": "scene", "frame_id": "frame",
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {"nominal_camera_offset_m": 1.0,
                   "camera_height_above_floor_m": 1.1,
                   "hfov_deg": 79.0, "vfov_deg": 63.45,
                   "resolution": [100, 80]},
        "body_radii_m": [0.2], "action_group_capacity": 2,
        "cases": [safe, collision], "c1_families": [],
        "surface_point_target": {
            "instance_id": 7, "category": "chair",
            "pixel_xy_px": [50, 40],
            "point_initial_robot_xyz_m": [1.0, 1.0, 3.0],
        },
    }


def test_single_point_migration_keeps_world_binding_and_existing_labels():
    original = _record()
    before = copy.deepcopy(original)
    assert hasattr(abc1_record, "normalize"), "one compact input adapter is needed"
    result = abc1_record.normalize(original)

    assert result["surface_point_target"]["points"] == [{
        "point_id": "p1", "pixel_xy_px": [50, 40],
        "world_xyz_m": [1.0, 1.0, -3.0],
    }]
    assert result["cases"][1]["task_outputs"]["A3"] == \
        original["cases"][1]["task_outputs"]["A3"]
    assert original == before


def test_surface_relations_use_safe_endpoints_and_preserve_checkpoint():
    from pipeline import surface_points

    record = abc1_record.normalize(_record())
    before = copy.deepcopy(record)
    surface_points.refresh_outputs(record)
    outputs = record["cases"][0]["task_outputs"]

    assert outputs["A4"]["checkpoint"] == \
        before["cases"][0]["task_outputs"]["A4"]["checkpoint"]
    assert outputs["B1"]["points"][0]["endpoint_distance_m"] == pytest.approx(5 ** 0.5)
    assert outputs["B2"]["points"][0]["horizontal_direction"] == "front-left"
    assert outputs["B2"]["points"][0]["vertical_direction"] == "level"
    assert record["cases"][1] == before["cases"][1]
    assert record["sensor"] == before["sensor"]


def test_surface_points_use_instance_pixels_without_old_object_area_gate():
    from pipeline import surface_points

    uv = np.asarray([[u, v] for v in range(20, 60) for u in range(20, 60)])
    points = np.column_stack((uv[:, 0] / 100, np.ones(len(uv)), 3 + uv[:, 1] / 100))
    frame = SimpleNamespace(
        pts=points, pts_uv=uv, pts_sem=np.full(len(uv), 7),
        id_to_cat={7: "chair"},
        predicate_category=lambda _iid: "chair",
        sensor=SimpleNamespace(width_px=1000, height_px=800),
        position=np.asarray([2.0, 1.0, 5.0]), yaw_rad=0.0,
    )
    target = surface_points.select_target(frame, record_uid="record", seed=12)
    assert target["instance_id"] == 7
    assert len(target["points"]) == 3
    for point in target["points"]:
        u, v = point["pixel_xy_px"]
        assert 22 <= u <= 57 and 22 <= v <= 57
        assert point["world_xyz_m"] == pytest.approx([2 + u / 100, 2, 2 - v / 100])
    assert target == surface_points.select_target(frame, record_uid="record", seed=12)


def test_surface_delta_never_evicts_existing_cases_or_changes_physics():
    assert hasattr(abc1_record, "install_surface_delta")
    original = _record()
    new = copy.deepcopy(original["cases"][0])
    new.update(case_id="added", group_id="added")
    new["task_outputs"] = {}
    result = abc1_record.install_surface_delta(original, {
        "surface_point_target": {
            "instance_id": 7, "category": "chair", "points": [{
                "point_id": "p1", "pixel_xy_px": [50, 40],
                "world_xyz_m": [1.0, 1.0, -3.0]}]},
        "cases": [new],
    })
    assert [case["case_id"] for case in result["cases"]] == ["safe", "collision", "added"]
    assert result["cases"][1] == original["cases"][1]
    assert result["action_group_capacity"] == 3
    assert abc1_record.physical_fingerprint(result) == abc1_record.physical_fingerprint(original)
    assert result["c1_families"] == original["c1_families"]


def test_each_point_compiles_its_own_answer_and_exact_pixel(tmp_path):
    from dataclasses import replace
    from PIL import Image
    from pipeline import benchmark_candidates, surface_points
    from post_QA import templates
    from post_QA.seen_build import artifact

    record = abc1_record.normalize(_record())
    record["image_path"] = "initial.png"
    Image.new("RGB", (100, 80), (40, 50, 60)).save(tmp_path / "initial.png")
    record["surface_point_target"]["points"].append({
        "point_id": "p2", "pixel_xy_px": [25, 40],
        "world_xyz_m": [-1.0, 1.0, -3.0],
    })
    surface_points.refresh_outputs(record)
    source = artifact.Source("r2r", tmp_path / "records.jsonl", "a" * 64, "b" * 64)
    rows = benchmark_candidates.project_record(
        dataset="r2r", source_path=str(source.records_path), byte_offset=0,
        record_sha256="c" * 64, record=record, allowed_tasks=("A4", "B1", "B2"))
    assert len(rows) == len({row.item_id for row in rows}) == 6
    assert {row.point_id for row in rows} == {"p1", "p2"}
    row = next(row for row in rows if row.point_id == "p2" and row.task_id == "B2")
    row = replace(row, template_id=templates.QUESTION_TEMPLATES["B2"][0]["id"])
    item, index = artifact.materialize_item(
        row, source, record, tmp_path / "qa", tmp_path / "build")
    target = index["usage"]["target_point"]
    assert target["surface_anchor"]["world_xyz_m"] == [-1.0, 1.0, -3.0]
    assert target["marker"]["center_xy"] == [25, 40]
    assert target["relation"]["horizontal_direction"] == "rear-left"
    assert (tmp_path / "qa" / item["images"][0]).is_file()


def test_b3_compiles_partial_forward_distance_without_changing_record(tmp_path):
    from dataclasses import replace
    from PIL import Image
    from pipeline import benchmark_candidates
    from post_QA.seen_build import artifact

    record = abc1_record.normalize(_record())
    record["image_path"] = "initial.png"
    Image.new("RGB", (100, 80), (40, 50, 60)).save(tmp_path / "initial.png")
    before = copy.deepcopy(record)
    source = artifact.Source("r2r", tmp_path / "records.jsonl", "a" * 64, "b" * 64)
    rows = benchmark_candidates.project_record(
        dataset="r2r", source_path=str(source.records_path), byte_offset=0,
        record_sha256="c" * 64, record=record, allowed_tasks=("B3",))
    assert rows
    assert all(row.outcome_id == "safe" for row in rows)
    assert len({row.item_id for row in rows}) == len(rows)
    row = next(row for row in rows if row.checkpoint_fraction == .5)
    # Target (1, 1, 3), camera at (0, 1, .5): sqrt(1 + 2.5**2).
    assert row.numeric_value_m == pytest.approx(7.25 ** .5)
    row = replace(row, template_id="B3_01")
    item, entry = artifact.materialize_item(
        row, source, record, tmp_path / "qa", tmp_path / "build")
    assert item["messages"][-1]["content"] == "2.69 m"
    assert "50%" in item["messages"][1]["content"]
    assert "action 1" in item["messages"][1]["content"]
    assert entry["usage"]["checkpoint"]["fraction"] == .5
    assert entry["usage"]["target_point"]["relation"]["camera_to_target_distance_m"] == pytest.approx(7.25 ** .5)
    assert record == before
    record["dataset"] = "gs"
    assert not benchmark_candidates.project_record(
        dataset="gs", source_path=str(source.records_path), byte_offset=0,
        record_sha256="c" * 64, record=record, allowed_tasks=("B3",))
