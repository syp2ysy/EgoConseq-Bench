from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import copy
import json

from PIL import Image
import pytest

from pipeline import abc1_record, benchmark_candidates, candidate_sources, io_utils
from post_QA import categories, templates
from post_QA.seen_build import artifact, release


@pytest.mark.parametrize("height", [0.5, 1.0, 1.5])
def test_qa_uses_configured_relative_height_without_mesh_measurement(tmp_path, height):
    record = _record(tmp_path)
    record["sensor"]["camera_height_above_floor_m"] = None
    record["sensor"]["nominal_camera_offset_m"] = height
    candidates = benchmark_candidates.project_record(
        dataset="r2r", source_path="records.jsonl", byte_offset=0,
        record_sha256="", record=record, allowed_tasks=("A1", "C1"))
    assert candidates
    model = artifact._compact_model(record, record["cases"][0], tmp_path/"img/initial.png", "")
    assert artifact._inputs(model)["camera"]["optical_center_height_m"] == height


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def _record(root: Path) -> dict:
    _write_rgb(root / "img/initial.png", (31, 32, 33))
    cases = []
    for index in range(4):
        _write_rgb(root / f"terminal/{index}.png", (30 + index * 30, 40, 50))
        cases.append({
            "case_id": f"safe-{index}", "group_id": f"safe-{index}",
            "actions": [{"type": "forward", "m": 1.0 + index * 0.5}],
            "starts_with": "forward", "body_radius_m": 0.2,
            "collision": False, "completed": True,
            "first_collision_action_index_1based": None,
            "endpoint_pose": {
                "x": float(index), "z": 1.0, "heading_deg": 0.0},
            "minimum_clearance_m": 0.4,
            "terminal_rgb_path": f"terminal/{index}.png",
            "task_outputs": {
                "A1": {"answer": "no_collision"},
                "A4": {
                    "answer": "front", "checkpoint_seed": "seed",
                    "checkpoint": {
                        "action_index": 1, "forward_stage": 1,
                        "fraction": 0.5, "arc_m": 0.5,
                        "pose": {"x": 0.0, "z": 0.5,
                                 "heading_deg": 0.0},
                    },
                    "horizontal_direction": "front",
                    "vertical_direction": "level",
                },
                "B1": {"initial_distance_m": 4.0,
                       "endpoint_distance_m": 3.0,
                       "distance_change_m": 1.0},
                "B2": {"answer": "front",
                       "horizontal_direction": "front",
                       "vertical_direction": "level"},
            },
        })
    cases.append({
        "case_id": "collision", "group_id": "collision",
        "actions": [
            {"type": "forward", "m": 3.0},
            {"type": "turn", "deg": 15.0},
            {"type": "forward", "m": 1.0},
        ],
        "starts_with": "forward", "body_radius_m": 0.2,
        "collision": True, "completed": False,
        "first_collision_action_index_1based": 3,
        "endpoint_pose": {"x": 0.0, "z": 3.5, "heading_deg": 15.0},
        "minimum_clearance_m": 0.0,
        "contact": {"instance_id": 7, "category": "chair"},
        "task_outputs": {
            "A1": {"answer": "collision"},
            "A2": {"collision_action_index_1based": 3,
                   "forward_ordinal_1based": 2,
                   "distance_rank": "shortest",
                   "minimum_action_boundary_margin_m": 0.5},
            "A3": {"instance_id": 7, "category": "chair",
                   "choices": [{"id": "chair", "text": "chair"},
                               {"id": "table", "text": "table"}]},
        },
    })
    return {
        "schema_version": "abc1.record.v2",
        "record_uid": "r2r-record",
        "dataset": "r2r", "scene_id": "scene", "frame_id": "frame",
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {
            "nominal_camera_offset_m": 1.0,
            "camera_height_above_floor_m": 1.0,
            "hfov_deg": 79.0, "vfov_deg": 63.45,
            "resolution": [32, 24],
        },
        "body_radii_m": [0.2], "image_path": "img/initial.png",
        "action_group_capacity": 8,
        "visible_entities": [{"instance_id": 7, "category": "chair"}],
        "surface_point_target": {
            "instance_id": 7, "category": "chair",
            "point_initial_robot_xyz_m": [2.0, 1.0, 4.0],
            "pixel_xy_px": [16, 12],
        },
        "cases": cases,
        "c1_families": [{
            "query_case_id": "safe-0",
            "member_case_ids": [f"safe-{index}" for index in range(4)],
        }],
    }


def _rows(record: dict, source: Path):
    return benchmark_candidates.project_record(
        dataset="r2r", source_path=str(source), byte_offset=0,
        record_sha256="a" * 64, record=record,
        allowed_tasks=("A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1"))


def test_direction_publication_reuses_existing_items_when_their_positions_change(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from post_QA.seen_build import catalog, direction_balance, images, selection, spec

    assets = tmp_path / "sources"
    records = [abc1_record.normalize(_record(assets)) for _ in range(2)]
    for i, record in enumerate(records):
        record["record_uid"] = f"record-{i}"
    source_path = assets / "records.jsonl"
    source = catalog.Source("r2r", source_path, "b" * 64, "c" * 64,
                            index_path="records/r2r/records.jsonl")
    payload = b""
    selected = []
    for record in records:
        rows = benchmark_candidates.project_record(
            dataset="r2r", source_path=str(source_path), byte_offset=len(payload),
            record_sha256=candidate_sources.canonical_sha256(record), record=record,
            allowed_tasks=("A4",))
        selected.append(replace(rows[0], template_id="A4_01"))
        payload += (json.dumps(record) + "\n").encode()
    source_path.write_bytes(payload)
    monkeypatch.setattr(catalog, "load", lambda _root: (source,))
    monkeypatch.setattr(spec, "DATASET_TASK_TOTALS", {"r2r": {"A4": 2}})
    root = tmp_path / "benchmark"
    root.mkdir()
    release.materialize_benchmark(selected, assets, root / "benchmark", root / "private")
    original = json.loads((root / "benchmark/qa.json").read_text())
    (root / "report.json").write_text(json.dumps({"seed": 7, "visual_selection": {"cosine_threshold": .9}}))
    (root / "index.html").write_text("old browser")
    bank = SimpleNamespace(metrics={str(assets / "img/initial.png"): {"accepted": True}},
                           features={}, timings={}, nearest_pairs=lambda *_args: {})
    monkeypatch.setattr(images, "ImageBank", lambda _device: bank)
    monkeypatch.setattr(selection, "enumerate_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(direction_balance, "refine_images", lambda rows, *_args, **_kwargs: list(reversed(rows)))

    result = release.rebalance_directions(assets, root)

    assert result["replaced_items"] == 0
    assert sorted(json.loads((root / "benchmark/qa.json").read_text()), key=lambda row: row["id"]) == sorted(original, key=lambda row: row["id"])
    assert "directions covered" in (root / "index.html").read_text()
    assert not list(tmp_path.glob(".benchmark-directions-*"))


def test_compact_artifact_compiles_answers_without_legacy_evidence(
        tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    record = _record(tmp_path)
    source = artifact.Source("r2r", records_path, "b" * 64, "c" * 64)

    projected = {}
    for row in _rows(record, records_path):
        projected.setdefault(row.task_id, row)
    for task_id in ("A1", "A2", "A3", "A4", "B1", "B2"):
        model, private, _choices, _digest = artifact.project(
            projected[task_id], source, record, tmp_path / "build")
        assert model["actions"]
        assert private["canonical_answer"]
        if task_id in ("A4", "B1", "B2"):
            assert model["target"] == "the marked point"
    assert projected["A2"].action_length == 3


@pytest.mark.parametrize(("record", "output", "expected"), (
    ({"dataset": "b1k", "visible_entities": []},
     {"instance_id": 1, "source_category": "elliptical_machine",
      "category": "exercise_bike.n.01"}, "elliptical machine"),
    ({"dataset": "b1k", "visible_entities": [
        {"instance_id": 2, "source_category": "locker",
         "category": "cabinet.n.03"}]},
     {"instance_id": 2, "category": "cabinet.n.03"}, "locker"),
    ({"dataset": "b1k", "visible_entities": [
        {"instance_id": 3, "category": "stairs"}]},
     {"instance_id": 3, "category": "step.n.01"}, "stairs"),
    ({"dataset": "b1k", "visible_entities": []},
     {"source_category": "gym_mat", "category": "mat.n.03"}, "gym mat"),
    ({"dataset": "b1k", "visible_entities": []},
     {"source_category": "commercial_kitchen_table",
      "category": "table.n.02"}, "commercial kitchen table"),
    ({"dataset": "b1k", "visible_entities": []},
     {"source_category": "bar", "category": "counter.n.01"}, "bar counter"),
    ({"dataset": "b1k", "visible_entities": []},
     {"source_category": "new_fixture_type", "category": "fixture.n.01"},
     "new fixture type"),
    ({"dataset": "b1k", "visible_entities": []},
     {"category": "exercise_bike.n.01"}, "exercise bike"),
    ({"dataset": "r2r", "visible_entities": []},
     {"category": "chair"}, "chair"),
))
def test_a3_answer_preserves_official_source_category_when_available(
        record, output, expected):
    assert categories.a3_answer(record, output) == expected


def test_a3_materialization_reuses_the_projected_canonical_answer(tmp_path):
    records_path = tmp_path / "records.jsonl"
    record = _record(tmp_path)
    record["dataset"] = "b1k"
    record["visible_entities"] = [{
        "instance_id": 7,
        "source_category": "eames_chair",
        "category": "chair.n.01",
    }]
    output = record["cases"][-1]["task_outputs"]["A3"]
    output.update({
        "source_category": "eames_chair",
        "category": "chair.n.01",
    })
    source = artifact.Source("b1k", records_path, "b" * 64, "c" * 64)
    candidate = benchmark_candidates.project_record(
        dataset="b1k", source_path=str(records_path), byte_offset=0,
        record_sha256="a" * 64, record=record, allowed_tasks=("A3",))[0]
    candidate = replace(
        candidate, template_id=templates.QUESTION_TEMPLATES["A3"][0]["id"])

    _model, private, _choices, _digest = artifact.project(
        candidate, source, record, tmp_path / "build")
    item, entry = artifact.materialize_item(
        candidate, source, record, tmp_path / "qa", tmp_path / "build")

    assert private["canonical_answer"] == "Eames chair"
    assert item["messages"][-1]["content"] == "Eames chair"
    assert entry["usage"]["contact_category"] == "Eames chair"


def test_compact_c1_uses_the_four_saved_terminal_paths_directly(
        tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    record = _record(tmp_path)
    source = artifact.Source("r2r", records_path, "b" * 64, "c" * 64)

    candidate = next(row for row in _rows(record, records_path)
                     if row.task_id == "C1")

    _model, private, choices, _digest = artifact.project(
        candidate, source, record, tmp_path / "build")

    assert len(choices) == 4
    assert {Path(choice["image"]).name for choice in choices} == {
        "0.png", "1.png", "2.png", "3.png"}
    assert private["canonical_answer"] in {choice["id"] for choice in choices}

    from dataclasses import replace
    from post_QA import templates

    candidate = replace(candidate, c1_label="D",
                        template_id=templates.QUESTION_TEMPLATES["C1"][0]["id"])
    _item, entry = artifact.materialize_item(
        candidate, source, record, tmp_path / "qa", tmp_path / "build")
    assert entry["usage"]["candidate_labels"]["D"]["outcome_id"] == candidate.outcome_id


@pytest.mark.parametrize("fail_publish", [False, True])
def test_presentation_refresh_preserves_records_answers_and_untouched_images(
        tmp_path, monkeypatch, fail_publish):
    records_root = tmp_path / "records"
    record = abc1_record.normalize(_record(records_root))
    source_path = records_root / "records.jsonl"
    source_path.write_text(json.dumps(record) + "\n")
    original_records = source_path.read_bytes()
    source = artifact.Source("r2r", source_path, "b" * 64, "c" * 64,
                             "records.jsonl")
    (records_root / "manifest.json").write_text(json.dumps({"datasets": [{
        "dataset": "r2r", "records_path": "records.jsonl",
        "index_path": "records.jsonl", "records_sha256": "b" * 64,
        "run_meta_sha256": "c" * 64}]}))
    root = tmp_path / "benchmark"
    by_task = {}
    for candidate in _rows(record, source_path):
        by_task.setdefault(candidate.task_id, candidate)
    items, entries = [], []
    for task in ("A1", "A4", "B1", "B2", "B3", "C1"):
        candidate = replace(by_task[task],
            record_sha256=candidate_sources.canonical_sha256(record),
            template_id=templates.QUESTION_TEMPLATES[task][0]["id"], c1_label="A")
        item, entry = artifact.materialize_item(
            candidate, source, record, root / "benchmark", tmp_path / "draw")
        if task in ("A4", "B1", "B2", "B3"):
            item["messages"][1]["content"] = "<image>\nQuestion: old point marked 1 on the chair"
            entry["usage"]["target_point"]["marker"] = {
                "kind": "surface_point", "number": 1, "center_xy": [16, 12]}
        items.append(item)
        entries.append(entry)
    qa_path = root / "benchmark/qa.json"
    io_utils.atomic_write_json(qa_path, items)
    (root / "private").mkdir()
    artifact.write_index(entries, qa_path, root / "private/record_index.json")
    (root / "index.html").write_text("old page")
    io_utils.atomic_write_json(root / "report.json", {"total": 6, "seed": 42})
    before = {str(p.relative_to(root)): p.read_bytes()
              for p in root.rglob("*") if p.is_file()}
    before_entries = copy.deepcopy(entries)
    if fail_publish:
        rename = Path.rename

        def fail_new_directory(path, destination):
            if Path(destination) == root and path.name == "new":
                raise OSError("simulated publish failure")
            return rename(path, destination)

        monkeypatch.setattr(Path, "rename", fail_new_directory)
        with pytest.raises(OSError, match="simulated publish failure"):
            release.refresh_presentation(records_root, root)
        assert {str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob("*") if p.is_file()} == before
    else:
        result = release.refresh_presentation(records_root, root)
        assert result["refreshed_items"] == 4
        updated = json.loads(qa_path.read_text())
        index = json.loads((root / "private/record_index.json").read_text())
        assert index["qa_sha256"] == io_utils.sha256_file(qa_path)
        for old, new, entry, original_entry in zip(items, updated, index["items"], before_entries):
            assert old["id"] == new["id"]
            assert old["messages"][-1] == new["messages"][-1]
            if old["task_id"] not in ("A4", "B1", "B2", "B3"):
                assert old == new
                assert entry == original_entry
                for image in old["images"]:
                    assert (root / "benchmark" / image).read_bytes() == before["benchmark/" + image]
            else:
                prompt = new["messages"][1]["content"]
                assert "the marked point" in prompt
                assert "marked 1" not in prompt and "chair" not in prompt
                assert "(1) forward" in prompt
                target = entry["usage"]["target_point"]
                assert "number" not in target["marker"]
                assert target["marked_image_sha256"] == io_utils.sha256_file(
                    root / "benchmark" / new["images"][0])
                for key in ("marker", "marked_image_sha256"):
                    target[key] = original_entry["usage"]["target_point"][key]
                assert entry == original_entry
        assert "benchmark-data" in (root / "index.html").read_text()
    assert source_path.read_bytes() == original_records
    assert not list(tmp_path.glob(".benchmark-refresh-*"))
