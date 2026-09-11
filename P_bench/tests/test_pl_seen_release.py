import copy
import json
import pytest

from pipeline import abc1_record, action_proposal, actions
from post_QA.seen_build import inventory, release
from scripts import build_seen_benchmark


def _old_record(scene, frame):
    group = f"old-{frame}"
    return {
        "schema_version": "conseq.v11",
        "scene_id": scene,
        "frame_id": frame,
        "pose": {"position": [1.0, 0.0, 2.0], "yaw_rad": 0.25},
        "sensor": {
            "nominal_camera_offset_m": 1.0,
            "hfov_deg": 79.0, "vfov_deg": 63.45,
        },
        "camera_height_above_visible_floor_m": 1.0,
        "floor_calibration": {"estimate": {
            "normal_local": [0.0, 1.0, 0.0], "offset_m": 0.0}},
        "image_path": f"img/{frame}.png",
        "objects": [],
        "selection": {
            "required_radii_m": [0.2], "action_group_ids": [group],
            "action_group_labels": {group: "safe"},
            "materialized_action_bank": [{
                "tag": group, "length": 1,
                "actions": [{"type": "forward", "m": 1.0}],
                "variant": "old", "template_id": group,
            }],
            "proposal_provenance": {group: {"variant": "old"}},
        },
        "outcomes": [{
            "outcome_id": f"b020-{group}", "action_group_id": group,
            "action_group_label": "safe", "seq_len": 1,
            "actions": [{"type": "forward", "m": 1.0}],
            "body": {"shape": "disc", "radius_m": 0.2},
            "physical": {
                "collision": False, "contact_action_index": None,
                "minimum_clearance_m": 0.4,
            },
            "depth_physical": {
                "collision": False, "contact_action_index": None,
            },
            "oracle_consensus": {"accepted": True},
            "shared_oracle_stability": {
                "version": "nominal-oracle.v1",
                "summary": {
                    "evaluation": "nominal", "collision": False,
                    "original_action_index": None,
                },
            },
            "execution": {"completed": True, "realized_pose": {
                "x": 0.0, "z": 1.0, "heading_deg": 0.0}},
        }],
    }


def _write_source(root, rows):
    directory = root / "gs"
    directory.mkdir(parents=True)
    records = directory / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "run_meta.json").write_text(json.dumps({
        "schema": "egoconseq.seen-records-dataset.v2",
        "dataset": "gs", "record_count": len(rows),
    }))
    (directory / "img").mkdir()
    for row in rows:
        (directory / row["image_path"]).write_bytes(b"png")
    return records


def _result(row, **values):
    return {
        "schema": "egoconseq.seen-build-result.v2",
        "plan_id": row["plan_id"],
        "plan_row_id": row["plan_row_id"],
        "record_uid": row["record_uid"],
        "source_record_sha256": row["source_record_sha256"],
        "dataset": row["dataset"], "scene_id": row["scene_id"],
        "frame_id": row["frame_id"], "status": "complete",
        **values,
    }


def test_explicit_stop_preserves_uncollected_records_without_faking_results(tmp_path):
    record = _old_record("s", "f")
    source = _write_source(tmp_path / "source", [record])
    plan = tmp_path / "plan.jsonl"
    inventory.write_plan({"gs": source}, plan)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        release.materialize_records(plan, work, tmp_path / "strict")
    result = release.materialize_records(
        plan, work, tmp_path / "output", reuse_uncollected=True)
    saved = json.loads((tmp_path / "output/gs/records.jsonl").read_text())
    assert result["uncollected_reused"] == 1
    assert saved["pose"] == record["pose"]
    assert saved["cases"][0]["actions"] == record["outcomes"][0]["actions"]
    assert list(work.iterdir()) == []


def test_materialize_preserves_unseen_split(tmp_path):
    source = _write_source(tmp_path / "source", [_old_record("s", "f")])
    plan = tmp_path / "plan.jsonl"
    inventory.write_plan({"gs": source}, plan, split="test/unseen")
    release.materialize_records(plan, tmp_path / "work", tmp_path / "output",
                                reuse_uncollected=True)
    for relative in ("manifest.json", "gs/run_meta.json"):
        assert json.loads((tmp_path / "output" / relative).read_text())["split"] == "test/unseen"
    from post_QA.seen_build import catalog
    from pathlib import Path
    assert Path(catalog.load(tmp_path / "output")[0].index_path).is_file()


def test_materialize_records_applies_deltas_and_writes_scene_ranges(tmp_path):
    source_root = tmp_path / "source"
    records = _write_source(source_root, [
        _old_record("z", "z0"), _old_record("a", "a0")])
    plan = tmp_path / "plan.jsonl"
    inventory.write_plan({"gs": records}, plan)
    _header, plan_rows = inventory.read_plan(plan)
    target = next(row for row in plan_rows if row["frame_id"] == "a0")
    action_rows = [{"type": "turn", "deg": 15.0},
                   {"type": "forward", "m": 1.0}]
    group = action_proposal.candidate_tag(
        actions.parse_actions(action_rows))
    outcome = {
        "outcome_id": f"b020-{group}", "action_group_id": group,
        "action_group_label": "safe", "seq_len": 2,
        "actions": action_rows,
        "body": {"shape": "disc", "radius_m": 0.2},
        "physical": {
            "collision": False, "contact_action_index": None,
            "minimum_clearance_m": 0.3,
        },
        "depth_physical": {
            "collision": False, "contact_action_index": None,
        },
        "oracle_consensus": {"accepted": True},
        "shared_oracle_stability": {
            "version": "nominal-oracle.v1",
            "summary": {
                "evaluation": "nominal", "collision": False,
                "original_action_index": None,
            },
        },
        "execution": {"completed": True, "realized_pose": {
            "x": 0.2, "z": 1.0, "heading_deg": 15.0}},
    }
    work = tmp_path / "work"
    work.mkdir()
    with (work / "gs.jsonl").open("w") as stream:
        for row in plan_rows:
            result = _result(row, mode="reused")
            if row == target:
                result.update({
                    "mode": "added", "outcomes": [outcome],
                    "provenance": {group: {"variant": "fixed_pose_turn"}},
                })
            stream.write(json.dumps(result) + "\n")

    summary = release.materialize_records(plan, work, tmp_path / "output")

    built = [json.loads(line) for line in (
        tmp_path / "output" / "gs" / "records.jsonl").read_text().splitlines()]
    assert [row["scene_id"] for row in built] == ["a", "z"]
    assert built[0]["pose"] == _old_record("a", "a0")["pose"]
    assert built[0]["schema_version"] == abc1_record.SCHEMA_VERSION
    assert len({case["group_id"] for case in built[0]["cases"]}) == 2
    assert {case["group_id"] for case in built[1]["cases"]} == {"old-z0"}
    assert built[0]["record_uid"] == target["record_uid"]
    meta = json.loads((tmp_path / "output" / "gs" / "run_meta.json").read_text())
    assert meta["schema"] == "egoconseq.abc1-records-dataset.v1"
    assert [row["scene_id"] for row in meta["scene_byte_ranges"]] == ["a", "z"]
    assert summary["record_count"] == 2


def test_materialize_records_compacts_unchanged_rows(tmp_path):
    source_root = tmp_path / "source"
    record_value = _old_record("s", "f")
    records = _write_source(source_root, [record_value])
    plan = tmp_path / "plan.jsonl"
    inventory.write_plan({"gs": records}, plan)
    _header, plan_rows = inventory.read_plan(plan)
    work = tmp_path / "work"
    work.mkdir()
    (work / "gs.jsonl").write_text(json.dumps(
        _result(plan_rows[0], mode="reused")) + "\n")

    release.materialize_records(plan, work, tmp_path / "output")

    built = json.loads((tmp_path / "output" / "gs" / "records.jsonl")
                       .read_text())
    assert built["schema_version"] == abc1_record.SCHEMA_VERSION
    assert built["pose"] == record_value["pose"]
    assert "selection" not in built
    assert "outcomes" not in built


def test_materialize_records_prunes_legacy_initial_turn(tmp_path):
    source_root = tmp_path / "source"
    record_value = _old_record("s", "f")
    record_value["selection"]["action_group_ids"] = ["legacy"]
    record_value["selection"]["action_group_labels"] = {"legacy": "safe"}
    record_value["selection"]["materialized_action_bank"] = [{
        "tag": "legacy", "length": 2,
        "actions": [{"type": "turn", "deg": 45.0},
                    {"type": "forward", "m": 1.0}],
        "variant": "old", "template_id": "legacy",
    }]
    record_value["selection"]["proposal_provenance"] = {
        "legacy": {"variant": "old"}}
    record_value["outcomes"] = [{
        "outcome_id": "b020-legacy", "action_group_id": "legacy",
        "action_group_label": "safe", "seq_len": 2,
        "actions": [{"type": "turn", "deg": 45.0},
                    {"type": "forward", "m": 1.0}],
        "body": {"shape": "disc", "radius_m": 0.2},
    }]
    records = _write_source(source_root, [record_value])
    plan = tmp_path / "plan.jsonl"
    inventory.write_plan({"gs": records}, plan)
    _header, plan_rows = inventory.read_plan(plan)
    work = tmp_path / "work"
    work.mkdir()
    (work / "gs.jsonl").write_text(json.dumps(
        _result(plan_rows[0], status="unavailable")) + "\n")

    release.materialize_records(plan, work, tmp_path / "output")

    built = json.loads((tmp_path / "output" / "gs" / "records.jsonl")
                       .read_text())
    assert built["cases"] == []


def test_materialize_records_command_dispatches_without_building_benchmark(
        monkeypatch, tmp_path):
    called = {}

    def fake_materialize(plan, work_dir, output):
        called.update(plan=plan, work_dir=work_dir, output=output)
        return {"record_count": 3}

    monkeypatch.setattr(release, "materialize_records", fake_materialize)
    plan = tmp_path / "plan.jsonl"
    work = tmp_path / "work"
    output = tmp_path / "records"

    assert build_seen_benchmark.main([
        "materialize-records", "--plan", str(plan),
        "--work-dir", str(work), "--output", str(output),
    ]) == 0
    assert called == {"plan": plan, "work_dir": work, "output": output}


def test_surface_merge_audit_rejects_changes_to_existing_case_physics():
    original = {
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {"nominal_camera_offset_m": 1.0,
                   "camera_height_above_floor_m": 1.0,
                   "hfov_deg": 79.0, "vfov_deg": 63.45,
                   "resolution": [640, 480]},
        "body_radii_m": [0.2], "floor_plane": {"offset_m": 0.0},
        "cases": [{
            "case_id": "old", "group_id": "old", "actions": [],
            "collision": False, "completed": True,
            "endpoint_pose": {"x": 0.0, "z": 1.0, "heading_deg": 0.0},
            "task_outputs": {"A1": {"answer": "no_collision"}},
        }],
        "c1_families": [{"query_case_id": "old", "member_case_ids": ["old"]}],
    }
    changed = copy.deepcopy(original)
    changed["cases"][0]["endpoint_pose"]["z"] = 2.0

    with pytest.raises(ValueError, match="protected"):
        release.audit_surface_merge(original, changed)


def test_source_catalog_is_flattened_into_useful_references(tmp_path):
    records = _write_source(tmp_path / "source", [_old_record("s", "f")])
    meta_path = records.with_name("run_meta.json")
    meta = json.loads(meta_path.read_text())
    meta["source_catalog"] = [[
        {"dataset": "gs", "records_path": "old/records.jsonl"},
        {"dataset": "gs", "records_path": "old/records.jsonl"},
    ]]
    meta_path.write_text(json.dumps(meta))
    plan = tmp_path / "plan.jsonl"
    inventory.write_plan({"gs": records}, plan)
    _header, rows = inventory.read_plan(plan)
    work = tmp_path / "work"
    work.mkdir()
    (work / "gs.jsonl").write_text(json.dumps(
        _result(rows[0], mode="reused")) + "\n")

    release.materialize_records(plan, work, tmp_path / "output")

    output_meta = json.loads(
        (tmp_path / "output/gs/run_meta.json").read_text())
    assert output_meta["source_catalog"] == [
        {"dataset": "gs", "records_path": "old/records.jsonl"}]
