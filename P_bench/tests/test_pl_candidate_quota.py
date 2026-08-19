"""Compiled-QA quotas stop shards without feeding labels back into poses."""

from __future__ import annotations

import json

import pytest

from pipeline import candidate_quota


def _write_artifact(root, rows):
    (root / "public").mkdir(parents=True)
    (root / "private").mkdir(parents=True)
    items = []
    answers = []
    atoms = []
    contexts = {}
    for index, row in enumerate(rows):
        task_id, length, scene_id, *dataset_values = row
        dataset = dataset_values[0] if dataset_values else "r2r"
        frame_sha256 = (
            dataset_values[1] if len(dataset_values) > 1 else
            f"frame-{index}")
        item_id = f"item-{index}"
        atom_id = f"atom-{index}"
        record_sha256 = f"record-{index}"
        items.append({
            "id": item_id,
            "task_id": task_id,
            "model_input": {
                "actions": [
                    {"type": "forward", "m": 0.5}
                    for _ in range(length)
                ],
            },
        })
        answers.append({
            "id": item_id,
            "task_id": task_id,
            "canonical_answer": "answer",
            "atom_ref": atom_id,
            "input_asset": {
                "sha256": f"marked-{index}",
                "raw_sha256": frame_sha256,
            },
        })
        atoms.append({
            "id": atom_id,
            "record_sha256": record_sha256,
        })
        contexts[record_sha256] = {
            "record_sha256": record_sha256,
            "context": {
                "scene_id": scene_id,
                "source": {"source_dataset": dataset},
            },
        }
    for path, values in (
            (root / "public" / "items.jsonl", items),
            (root / "private" / "answers.jsonl", answers),
            (root / "private" / "atoms.jsonl", atoms),
            (root / "private" / "record_contexts.jsonl",
             list(contexts.values()))):
        path.write_text("".join(
            json.dumps(value, sort_keys=True) + "\n" for value in values))


def test_quota_stops_only_when_every_task_length_and_scene_gate_passes(
        tmp_path):
    _write_artifact(tmp_path, [
        ("A1_collision", 1, "scene-a"),
        ("A1_collision", 2, "scene-b"),
        ("A2_collision_step_grounding", 1, "scene-a"),
        ("A2_collision_step_grounding", 2, "scene-b"),
    ])

    report = candidate_quota.summarize_artifact(
        tmp_path, dataset="r2r",
        supported_tasks=("A1_collision", "A2_collision_step_grounding"),
        min_total_items=4, min_per_task=2,
        required_lengths=(1, 2), min_per_length=1,
        min_unique_frames=4, min_scene_families=2,
        max_scene_family_fraction=0.5,
        scene_families={"scene-a": "family-a", "scene-b": "family-b"})

    assert report["decision"] == "stop"
    assert report["complete"] is True
    assert report["counts_by_length"] == {"L1": 2, "L2": 2}
    assert report["tasks"]["A1_collision"] == {
        "count": 2,
        "count_shortfall": 0,
        "counts_by_length": {"L1": 1, "L2": 1},
        "scene_count": 2,
        "scene_counts": {"scene-a": 1, "scene-b": 1},
        "complete": True,
    }
    assert report["unique_source_frames"] == 4
    assert report["scene_family_count"] == 2
    assert report["dominant_scene_family_fraction"] == 0.5


def test_quota_reports_capacity_shortfall_instead_of_lowering_a_gate(tmp_path):
    _write_artifact(tmp_path, [
        ("A1_collision", 1, "scene-a", "b1k"),
        ("A1_collision", 2, "scene-b", "b1k"),
        ("A2_collision_step_grounding", 1, "scene-a", "b1k"),
    ])

    report = candidate_quota.summarize_artifact(
        tmp_path, dataset="b1k",
        supported_tasks=("A1_collision", "A2_collision_step_grounding"),
        min_total_items=4, min_per_task=2,
        required_lengths=(1, 2), min_per_length=2,
        min_unique_frames=4, min_scene_families=2,
        max_scene_family_fraction=0.6)

    assert report["decision"] == "append_shards"
    assert report["complete"] is False
    a2 = report["tasks"]["A2_collision_step_grounding"]
    assert a2["count_shortfall"] == 1
    assert report["length_shortfall"] == {"L2": 1}


def test_quota_counts_only_the_requested_dataset(tmp_path):
    _write_artifact(tmp_path, [
        ("A1_collision", 1, "r2r-scene", "r2r"),
        ("A1_collision", 2, "b1k-scene", "b1k"),
    ])

    report = candidate_quota.summarize_artifact(
        tmp_path, dataset="b1k", supported_tasks=("A1_collision",),
        min_total_items=1, min_per_task=1,
        required_lengths=(1, 2), min_per_length=1,
        min_unique_frames=1, min_scene_families=1,
        max_scene_family_fraction=1.0)

    assert report["total_supported_items"] == 1
    assert report["tasks"]["A1_collision"]["scene_counts"] == {
        "b1k-scene": 1}
    assert report["counts_by_length"] == {"L1": 0, "L2": 1}
    assert report["length_shortfall"] == {"L1": 1}


def test_one_loaded_artifact_index_serves_all_dataset_summaries(
        tmp_path, monkeypatch):
    _write_artifact(tmp_path, [
        ("A1_collision", 1, "r2r-scene", "r2r"),
        ("A1_collision", 1, "b1k-scene", "b1k"),
    ])
    calls = 0
    original = candidate_quota.io_utils.read_jsonl

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate_quota.io_utils, "read_jsonl", counted)
    artifact_index = candidate_quota.load_artifact_index(tmp_path)
    for dataset in ("r2r", "b1k"):
        report = candidate_quota.summarize_artifact(
            tmp_path, dataset=dataset, supported_tasks=("A1_collision",),
            min_total_items=1, min_per_task=1,
            required_lengths=(1,), min_per_length=1,
            min_unique_frames=1, min_scene_families=1,
            max_scene_family_fraction=1.0,
            artifact_index=artifact_index)
        assert report["total_supported_items"] == 1

    assert calls == 4


def test_quota_rejects_an_answer_with_a_missing_source_atom(tmp_path):
    _write_artifact(tmp_path, [("A1_collision", 1, "scene-a")])
    (tmp_path / "private" / "atoms.jsonl").write_text("")

    with pytest.raises(ValueError, match="source atom"):
        candidate_quota.summarize_artifact(
            tmp_path, dataset="r2r", supported_tasks=("A1_collision",),
            min_total_items=1, min_per_task=1,
            required_lengths=(1,), min_per_length=1,
            min_unique_frames=1, min_scene_families=1,
            max_scene_family_fraction=1.0)


def test_total_item_quota_is_part_of_the_pure_stop_decision(tmp_path):
    """Catches a controller mutating an otherwise complete quota report."""
    _write_artifact(tmp_path, [
        ("A1_collision", 1, "scene-a"),
        ("A1_collision", 2, "scene-b"),
    ])

    report = candidate_quota.summarize_artifact(
        tmp_path, dataset="r2r", supported_tasks=("A1_collision",),
        min_total_items=3, min_per_task=2,
        required_lengths=(1, 2), min_per_length=1,
        min_unique_frames=2, min_scene_families=2,
        max_scene_family_fraction=0.5)

    assert report["total_supported_items"] == 2
    assert report["min_total_items"] == 3
    assert report["total_item_shortfall"] == 1
    assert report["complete"] is False
    assert report["decision"] == "append_shards"


def test_quota_cli_writes_an_append_decision_with_exit_two(tmp_path):
    from scripts import check_candidate_quota

    benchmark = tmp_path / "candidate_qa"
    _write_artifact(benchmark, [("A1_collision", 1, "scene-a")])
    output = tmp_path / "quota.json"

    exit_code = check_candidate_quota.main([
        "--benchmark", str(benchmark),
        "--dataset", "r2r",
        "--supported-task", "A1_collision",
        "--min-total-items", "2",
        "--min-per-task", "2",
        "--length", "1",
        "--min-per-length", "1",
        "--min-unique-frames", "1",
        "--min-scene-families", "1",
        "--max-scene-family-fraction", "1.0",
        "--out", str(output),
    ])

    assert exit_code == 2
    assert json.loads(output.read_text())["decision"] == "append_shards"


def test_quota_cli_defaults_reuse_the_frozen_r2r_gates(tmp_path):
    from pipeline import config
    from scripts import check_candidate_quota

    benchmark = tmp_path / "candidate_qa"
    _write_artifact(benchmark, [("A1_collision", 1, "scene-a")])
    output = tmp_path / "quota.json"

    assert check_candidate_quota.main([
        "--benchmark", str(benchmark), "--dataset", "r2r",
        "--supported-task", "A1_collision", "--out", str(output),
    ]) == 2
    report = json.loads(output.read_text())
    assert report["min_total_items"] == config.BACKGROUND_MIN_TOTAL_ITEMS
    assert report["min_per_task"] == config.BACKGROUND_MIN_ITEMS_PER_TASK
    assert report["min_per_length"] == \
        config.BACKGROUND_MIN_ITEMS_PER_LENGTH
    assert report["min_unique_frames"] == \
        config.BACKGROUND_MIN_UNIQUE_FRAMES_BY_DATASET["r2r"]
    assert report["min_scene_families"] == \
        config.BACKGROUND_MIN_SCENE_FAMILIES_BY_DATASET["r2r"]
    assert report["max_scene_family_fraction"] == \
        config.BACKGROUND_MAX_SCENE_FAMILY_FRACTION["r2r"]


def test_quota_counts_unique_raw_frames_instead_of_marked_variants(tmp_path):
    _write_artifact(tmp_path, [
        ("A1_collision", 1, "scene-a", "r2r", "same-frame"),
        ("A2_collision_step_grounding", 2, "scene-a", "r2r",
         "same-frame"),
        ("B1_endpoint_distance", 1, "scene-b", "r2r", "second-frame"),
    ])

    report = candidate_quota.summarize_artifact(
        tmp_path, dataset="r2r",
        supported_tasks=(
            "A1_collision", "A2_collision_step_grounding",
            "B1_endpoint_distance"),
        min_total_items=3, min_per_task=1,
        required_lengths=(1, 2), min_per_length=1,
        min_unique_frames=3, min_scene_families=2,
        max_scene_family_fraction=0.5,
        scene_families={"scene-a": "family-a", "scene-b": "family-b"})

    assert report["total_supported_items"] == 3
    assert report["unique_source_frames"] == 2
    assert report["unique_source_frame_shortfall"] == 1
    assert report["decision"] == "append_shards"
