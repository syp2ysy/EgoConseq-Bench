import importlib
from pathlib import Path
import re

import pytest


TASKS = (
    "A1_collision",
    "A2_collision_step_grounding",
    "A3_contact_object",
    "B1_endpoint_distance",
    "B2_endpoint_direction",
    "C1_future_view_selection",
)


def _case(task_id: str, *, answer: str, choices: list[dict], index: int):
    model_input = {
        "actions": [
            {"type": "forward", "m": 1.0},
            {"type": "turn", "deg": 30.0},
        ],
        "body_radius_m": 0.2,
        "camera_height_above_visible_floor_m": 1.0,
        "hfov_deg": 79.0,
        "vfov_deg": 63.45,
        "initial_rgb": f"public/images/initial-{index}.png",
        "initial_rgb_sha256": f"{index + 1:064x}",
    }
    if task_id.startswith("B"):
        model_input["target"] = {"label": "chair 2"}
    return {
        "atom_ref": f"atom-{index}",
        "canonical_answer": answer,
        "oracle_ref": {
            "frame_id": f"F-scene-{index}-review-p000",
            "outcome_id": f"outcome-{index}",
        },
        "public": {
            "choices": choices,
            "id": f"case-{index}",
            "model_input": model_input,
            "question": f"Question for {task_id}?",
            "result_head": task_id[0],
            "schema_version": "egoconseq.qa.v16-candidate-preview",
            "task_id": task_id,
            "task_metadata": {"answer_format": "closed_exact"},
        },
    }


def _six_task_benchmark() -> dict:
    cases = [
        _case("A1_collision", answer="no_collision", choices=[
            {"id": "collision", "text": "collision"},
            {"id": "no_collision", "text": "no_collision"},
        ], index=0),
        _case("A2_collision_step_grounding", answer="action_2", choices=[
            {"id": "action_1", "text": "action_1"},
            {"id": "action_2", "text": "action_2"},
        ], index=1),
        _case("A3_contact_object", answer="wall", choices=[
            {"id": "door", "text": "door"},
            {"id": "wall", "text": "wall"},
        ], index=2),
        _case("B1_endpoint_distance", answer="choice_2", choices=[
            {"id": "choice_1", "text": "0.95 m", "value_m": 0.95},
            {"id": "choice_2", "text": "0.70 m", "value_m": 0.70},
        ], index=3),
        _case("B2_endpoint_direction", answer="right", choices=[
            {"id": "front", "text": "front"},
            {"id": "right", "text": "right"},
        ], index=4),
        _case("C1_future_view_selection", answer="image_3", choices=[
            {
                "id": f"image_{number}",
                "image": f"public/images/future-{number}.png",
                "image_sha256": f"{number + 10:064x}",
            }
            for number in range(1, 5)
        ], index=5),
    ]
    return {
        "schema": "egoconseq.candidate-preview.v2",
        "candidate_only": True,
        "headline_eligible": False,
        "coverage": {task_id: 1 for task_id in TASKS},
        "cases": cases,
        "task_catalog": list(TASKS),
        "task_scope": "ABC",
        "task_status": {task_id: "available_real" for task_id in TASKS},
    }


def test_browser_payload_renders_ab_as_open_text_and_c1_as_image_mcq(
        tmp_path: Path):
    case_browser = importlib.import_module("pipeline.case_browser")
    source = case_browser.BrowserSource(
        label="combined",
        artifact_root=tmp_path / "candidate_qa",
        benchmark=_six_task_benchmark(),
        scene_by_case_id={f"case-{index}": f"scene-{index}"
                          for index in range(6)},
        technical_by_case_id={},
    )

    payload = case_browser.build_browser_payload(
        [source], report_root=tmp_path / "report")

    assert payload["schema"] == "egoconseq.case-browser.v1"
    assert payload["coverage"] == {task_id: 1 for task_id in TASKS}
    by_task = {case["task_id"]: case for case in payload["cases"]}
    for task_id in TASKS[:-1]:
        assert by_task[task_id]["response_mode"] == "open_text"
        assert "choices" not in by_task[task_id]
    assert by_task["B1_endpoint_distance"]["answer"] == {
        "raw": "choice_2",
        "display": "0.70 m",
    }
    c1 = by_task["C1_future_view_selection"]
    assert c1["response_mode"] == "image_mcq"
    assert c1["answer"] == {"raw": "image_3", "display": "image_3"}
    assert [choice["id"] for choice in c1["choices"]] == [
        "image_1", "image_2", "image_3", "image_4"]
    assert [choice["is_correct"] for choice in c1["choices"]] == [
        False, False, True, False]


def test_browser_payload_rejects_duplicate_case_ids_across_sources(tmp_path):
    case_browser = importlib.import_module("pipeline.case_browser")
    benchmark = _six_task_benchmark()
    first = case_browser.BrowserSource(
        label="first", artifact_root=tmp_path / "first",
        benchmark=benchmark)
    second = case_browser.BrowserSource(
        label="second", artifact_root=tmp_path / "second",
        benchmark=benchmark)

    with pytest.raises(ValueError, match="duplicate case id: case-0"):
        case_browser.build_browser_payload(
            [first, second], report_root=tmp_path / "report")


@pytest.mark.parametrize("mutation", ["three_choices", "answer_outside"])
def test_browser_payload_rejects_malformed_c1_family(tmp_path, mutation):
    case_browser = importlib.import_module("pipeline.case_browser")
    benchmark = _six_task_benchmark()
    c1 = benchmark["cases"][-1]
    if mutation == "three_choices":
        c1["public"]["choices"].pop()
        expected = "exactly four image choices"
    else:
        c1["canonical_answer"] = "image_9"
        expected = "answer is outside image choices"
    source = case_browser.BrowserSource(
        label="source", artifact_root=tmp_path / "source",
        benchmark=benchmark)

    with pytest.raises(ValueError, match=expected):
        case_browser.build_browser_payload(
            [source], report_root=tmp_path / "report")


def test_static_browser_embeds_data_but_does_not_eager_render_case_images(
        tmp_path):
    case_browser = importlib.import_module("pipeline.case_browser")
    benchmark = _six_task_benchmark()
    benchmark["cases"][0]["public"]["question"] = \
        "Can </script><img src=x> break the report?"
    source = case_browser.BrowserSource(
        label="A/B + C1", artifact_root=tmp_path / "candidate_qa",
        benchmark=benchmark)
    report_root = tmp_path / "candidate_qa_report"

    output = case_browser.render_case_browser([source], report_root)

    assert output == report_root / "index.html"
    document = output.read_text()
    assert 'data-browser-schema="egoconseq.case-browser.v1"' in document
    assert 'id="case-browser-data" type="application/json"' in document
    assert "全部任务" in document
    assert all(f'data-task-filter="{task_id}"' in document
               for task_id in TASKS)
    assert 'id="case-grid"' in document
    assert 'id="case-drawer"' in document
    assert 'id="image-lightbox"' in document
    assert 'class="action-timeline"' in document
    assert "fetch(" not in document
    assert "A4_checkpoint_direction" not in document
    assert "A4 diagnostic" not in document
    assert re.search(r"<img[^>]+\ssrc=", document) is None
    assert "</script><img src=x>" not in document
    assert "\\u003c/script\\u003e\\u003cimg src=x\\u003e" in document


def test_validated_browser_source_joins_scene_and_outcome_metadata(
        tmp_path):
    case_browser = importlib.import_module("pipeline.case_browser")
    from tests.test_pl_v16_candidate_preview import (
        _one_a1_projection,
        _validate_preview_artifact,
        _write_preview_artifact,
    )

    artifact = tmp_path / "candidate_qa"
    projection = _one_a1_projection(tmp_path)
    _write_preview_artifact(
        projection, artifact,
        source_records_path=tmp_path / "records.jsonl")
    expected_case_id = projection["items"][0]["id"]
    expected_scene = projection["record_contexts"][0]["context"][
        "source"]["scene_id"]
    expected_outcome = projection["atoms"][0]["outcome_id"]

    validated = _validate_preview_artifact(artifact)
    source = case_browser.browser_source_from_validated_artifact(
        "A/B", artifact, validated)

    assert source.label == "A/B"
    assert source.scene_by_case_id == {expected_case_id: expected_scene}
    assert source.technical_by_case_id[expected_case_id]["outcome_id"] == \
        expected_outcome
    assert source.technical_by_case_id[expected_case_id]["record_sha256"] == \
        projection["atoms"][0]["record_sha256"]


def test_case_browser_cli_accepts_repeatable_labeled_artifacts(tmp_path):
    cli = importlib.import_module("scripts.build_case_browser")

    args = cli.build_arg_parser().parse_args([
        "--source", "AB=/runs/ab/candidate_qa",
        "--source", "C1=/runs/c1/candidate_qa",
        "--source-authority-manifest", "/runs/ab/authority.json",
        "--source-authority-manifest", "/runs/c1/authority.json",
        "--source-authority-root", "/trusted/repository",
        "--output", str(tmp_path / "candidate_qa_report"),
    ])

    assert args.source == [
        ("AB", Path("/runs/ab/candidate_qa")),
        ("C1", Path("/runs/c1/candidate_qa")),
    ]
    assert args.source_authority_manifest == [
        Path("/runs/ab/authority.json"),
        Path("/runs/c1/authority.json"),
    ]
    assert args.source_authority_root == Path("/trusted/repository")
    assert args.output == tmp_path / "candidate_qa_report"


def test_case_browser_cli_requires_one_authority_per_source(
        tmp_path, capsys):
    cli = importlib.import_module("scripts.build_case_browser")

    with pytest.raises(SystemExit):
        cli.main([
            "--source", "AB=/runs/ab/candidate_qa",
            "--source-authority-manifest", "/runs/ab/authority.json",
            "--source-authority-manifest", "/runs/other/authority.json",
            "--output", str(tmp_path / "candidate_qa_report"),
        ])

    assert "one source authority manifest per source" in capsys.readouterr().err


def test_standard_renderer_requires_and_uses_external_source_authority(
        tmp_path):
    from tests.test_pl_v16_candidate_preview import (
        _one_a1_projection,
        _render_preview_html,
        _write_preview_artifact,
    )

    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")

    output = _render_preview_html(
        artifact / "benchmark.json", tmp_path / "candidate_qa_report")

    assert output.is_file()
    assert 'data-browser-schema="egoconseq.case-browser.v1"' in \
        output.read_text()
