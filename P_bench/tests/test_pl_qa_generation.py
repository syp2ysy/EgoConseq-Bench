import hashlib
import json
import random

import pytest

from post_QA import answers, templates
from post_QA.seen_build import artifact, spec
from scripts import build_seen_benchmark


def _inputs():
    return {
        "camera": {
            "optical_center_height_m": 1.0,
            "hfov_deg": 110.0,
            "vfov_deg": 93.93,
        },
        "robot": {"radius_m": 0.2},
        "actions": [
            {"index": 1, "type": "forward", "meters": 0.5,
             "text": "forward 0.5 m"},
            {"index": 2, "type": "turn", "degrees": -30.0,
             "text": "turn left 30 degree"},
            {"index": 3, "type": "forward", "meters": 1.0,
             "text": "forward 1 m"},
        ],
        "target": "the marked point",
        "checkpoint": {
            "action_index": 3,
            "fraction": 0.5,
        },
    }


def _item(task_id, answer):
    return {
        "id": f"{task_id.lower()}-example",
        "task_id": task_id,
        "dataset": "gs",
        "images": ["images/egocentric/example.png"],
        "messages": [
            {"role": "system", "content": templates.SYSTEM_PROMPT},
            {"role": "user", "content": (
                "<image>\nAction sequence (execute in order):\n"
                "(1) forward 50 cm, (2) turn left 30 degree, "
                "(3) forward 100 cm")},
            {"role": "assistant", "content": answer},
        ],
    }


@pytest.mark.parametrize(("meters", "expected"), (
    (0.5, "forward 0.5 m"),
    (1.0, "forward 1 m"),
    (3.0, "forward 3 m"),
))
def test_action_inputs_format_forward_distances_in_meters(meters, expected):
    model = {
        "camera_height_m": 1.0,
        "hfov_deg": 79.0, "vfov_deg": 63.45,
        "body_radius_m": 0.2,
        "actions": [{"type": "forward", "m": meters}],
    }
    assert artifact._inputs(model)["actions"][0]["text"] == expected


def test_sources_follow_manifest_paths_without_path_shape_rules(tmp_path):
    path = tmp_path / "snapshots" / "custom.jsonl"
    path.parent.mkdir()
    path.write_text("{}\n")
    run_meta = tmp_path / "run_meta.json"
    run_meta.write_text("{}\n")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "datasets": [{
            "dataset": "r2r", "records_path": "snapshots/custom.jsonl",
            "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "run_meta_sha256": hashlib.sha256(
                run_meta.read_bytes()).hexdigest(),
        }],
    }))

    source = artifact.sources(tmp_path)["r2r"]

    assert source.records_path == path.resolve()


def test_private_index_uses_a_release_relative_records_path():
    assert artifact.release_records_path("r2r") == \
        "records/r2r/records.jsonl"


def test_seen_builder_exposes_the_read_only_preflight_stage(tmp_path):
    parser = build_seen_benchmark.build_arg_parser()
    args = parser.parse_args(["plan", "--records-root", str(tmp_path)])

    assert args.command == "plan"
    assert args.records_root == tmp_path
    assert {parser.parse_args([command]).command for command in (
        "collect", "dry-run")} == {"collect", "dry-run"}
    assert parser.parse_args([
        "compile", "--output", str(tmp_path / "qa")]).command == "compile"


def test_templates_render_every_public_input():
    assert tuple(templates.QUESTION_TEMPLATES) == spec.TASKS
    for task, bank in templates.QUESTION_TEMPLATES.items():
        prompts = [templates.render_question(task, _inputs(), row) for row in bank]
        assert len(set(prompts)) >= 10
        assert all(prompt.count("<image>") == (5 if task == "C1" else 1)
                   for prompt in prompts)
        if task in {"A4", "B3"}:
            assert all(prompt.count("50%") == 1 and "Query moment:" in prompt
                       for prompt in prompts)
    rng_a = random.Random(19)
    rng_b = random.Random(19)
    first = [templates.choose_template("A4", rng_a)["id"] for _ in range(20)]
    second = [templates.choose_template("A4", rng_b)["id"] for _ in range(20)]
    assert first == second
    assert len(set(first)) > 1

    question = templates.render_question(
        "A4", _inputs(), templates.QUESTION_TEMPLATES["A4"][0])
    for value in (
            "Collision footprint radius: 0.2 m",
            "Camera optical-center height relative to the local ground reference: 1.0 m",
            "Horizontal field of view: 110°",
            "Vertical field of view: 93.93°",
            "(1) forward 0.5 m", "(2) turn left 30 degree",
            "the marked point", "50%", "forward distance"):
        assert value in question
    assert "fixed surface point" in question
    assert "second forward action" not in question
    c1 = templates.render_question(
        "C1", _inputs(), templates.QUESTION_TEMPLATES["C1"][0])
    assert c1.count("<image>") == 5


def test_template_assignment_is_balanced_without_cycling_with_c1_answers():
    tasks = ["C1"] * 400 + ["A1"] * 23
    ids = templates.balanced_template_ids(tasks, seed=19)
    assert ids == templates.balanced_template_ids(tasks, seed=19)
    from collections import Counter
    counts = Counter(ids)
    assert len({name for name in counts if name.startswith("C1_")}) == 10
    assert {count for name, count in counts.items() if name.startswith("C1_")} == {40}
    assert {count for name, count in counts.items() if name.startswith("A1_")} == {2, 3}
    for name in set(ids[:400]):
        assert {"ABCD"[i % 4] for i, value in enumerate(ids[:400]) if value == name} == set("ABCD")


def test_answer_instructions_stay_with_their_task():
    prompts = {
        task: templates.render_question(task, _inputs(), rows[0])
        for task, rows in templates.QUESTION_TEMPLATES.items()
    }
    assert "\n(2) turn left 30 degree\n" in prompts["A2"]
    assert "Answer only: Action <number>" in prompts["A2"]
    assert "object category only" in prompts["A3"]
    assert "A, B, C, or D" in prompts["C1"]
    assert all("A, B, C, or D" not in prompt
               for task, prompt in prompts.items() if task != "C1")


def test_record_index_binds_private_rows_to_public_qa(tmp_path):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps([_item("A2", "Action 3.")]))
    entry = {
        "item_id": "a2-example", "record_sha256": "1" * 64,
        "initial_image_sha256": "2" * 64,
        "usage": {"starts_with": "turn"},
    }

    document = artifact.write_index(
        [entry], qa_path, tmp_path / "record_index.json")

    assert document["schema"] == "egoconseq.seen-record-index.v3"
    assert document["record_count"] == 1
    assert document["items"] == [entry]


def test_judge_summary_keeps_refusals_and_separates_judge_errors():
    rows = [
        {"task_id": "A1", "status": "scored", "correct": True, "score": 1, "gt_collision": True},
        {"task_id": "A1", "status": "scored", "correct": False, "score": 0, "gt_collision": True},
        {"task_id": "A1", "status": "scored", "correct": True, "score": 1, "gt_collision": False},
        {"task_id": "A2", "status": "judge_error"},
        {"task_id": "B2", "status": "scored", "correct": False, "score": 2 / 3,
         "axis_correct": {"front_back": False, "left_right": True, "up_down": True}},
    ]
    report = answers.summarize_evaluation(rows)
    assert report["tasks"]["A1"]["accuracy"] == pytest.approx(2 / 3)
    assert report["tasks"]["A1"]["collision_recall"] == 0.5
    assert report["tasks"]["A2"]["accuracy"] is None
    assert report["tasks"]["A2"]["pending"] == 1
    assert report["tasks"]["A3"]["accuracy"] is None
    assert report["tasks"]["B2"]["joint_accuracy"] == 0
    assert report["tasks"]["B2"]["axis_accuracy"] == pytest.approx(2 / 3)
