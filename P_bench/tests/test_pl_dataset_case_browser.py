import copy
import hashlib
import importlib
import json
from pathlib import Path
import re

import pytest

from tests.test_pl_case_browser import _six_task_benchmark, TASKS


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _materialize_images(benchmark: dict, artifact_root: Path, prefix: str):
    images = artifact_root / "public" / "images"
    images.mkdir(parents=True)
    for case_index, case in enumerate(benchmark["cases"]):
        model_input = case["public"]["model_input"]
        initial = f"{prefix}-initial-{case_index}.png"
        initial_bytes = b"\x89PNG\r\n\x1a\n" + initial.encode()
        (images / initial).write_bytes(initial_bytes)
        model_input["initial_rgb"] = f"public/images/{initial}"
        model_input["initial_rgb_sha256"] = _sha256(initial_bytes)
        for choice_index, choice in enumerate(
                case["public"].get("choices") or []):
            if "image" not in choice:
                continue
            name = f"{prefix}-future-{case_index}-{choice_index}.png"
            payload = b"\x89PNG\r\n\x1a\n" + name.encode()
            (images / name).write_bytes(payload)
            choice["image"] = f"public/images/{name}"
            choice["image_sha256"] = _sha256(payload)


def _write_input_report(tmp_path: Path, name: str, benchmark: dict,
                        source_label: str) -> Path:
    case_browser = importlib.import_module("pipeline.case_browser")
    artifact = tmp_path / name / "candidate_qa"
    _materialize_images(benchmark, artifact, name)
    scene_by_case = {
        case["public"]["id"]: f"{name}-scene"
        for case in benchmark["cases"]
    }
    source = case_browser.BrowserSource(
        label=source_label,
        artifact_root=artifact,
        benchmark=benchmark,
        scene_by_case_id=scene_by_case,
    )
    return case_browser.render_case_browser(
        [source], tmp_path / name / "candidate_qa_report")


def _payload_from_html(path: Path) -> dict:
    document = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="case-browser-data" type="application/json">(.*?)</script>',
        document, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _b1k_benchmark() -> dict:
    benchmark = copy.deepcopy(_six_task_benchmark())
    benchmark["cases"] = benchmark["cases"][:3]
    for case in benchmark["cases"]:
        case["public"]["id"] = f"b1k-{case['public']['id']}"
    benchmark["coverage"] = {
        task_id: int(index < 3)
        for index, task_id in enumerate(TASKS)
    }
    return benchmark


def test_merge_dataset_reports_bundles_images_and_preserves_dataset_source(
        tmp_path: Path):
    module = importlib.import_module("pipeline.dataset_case_browser")
    r2r = _write_input_report(
        tmp_path, "r2r", _six_task_benchmark(), "AB + C1")
    b1k = _write_input_report(
        tmp_path, "b1k", _b1k_benchmark(), "Ihlen + grocery")
    output = tmp_path / "all" / "candidate_qa_report"

    index = module.merge_dataset_reports([
        module.DatasetReport("r2r", "R2R", r2r),
        module.DatasetReport(
            "behavior1k", "BEHAVIOR-1K Pilot", b1k,
            status="pilot"),
    ], output)

    payload = _payload_from_html(index)
    assert payload["schema"] == "egoconseq.case-browser.v1"
    assert payload["headline_eligible"] is False
    assert payload["datasets"] == [
        {
            "id": "r2r", "label": "R2R", "status": "candidate",
            "case_count": 6,
            "coverage": {task_id: 1 for task_id in TASKS},
        },
        {
            "id": "behavior1k", "label": "BEHAVIOR-1K Pilot",
            "status": "pilot", "case_count": 3,
            "coverage": {
                "A1_collision": 1,
                "A2_collision_step_grounding": 1,
                "A3_contact_object": 1,
                "B1_endpoint_distance": 0,
                "B2_endpoint_direction": 0,
                "C1_future_view_selection": 0,
            },
        },
    ]
    b1k_cases = [row for row in payload["cases"]
                 if row["dataset_id"] == "behavior1k"]
    assert {row["source_label"] for row in b1k_cases} == {
        "BEHAVIOR-1K Pilot"}
    assert {row["technical"]["dataset_source"] for row in b1k_cases} == {
        "Ihlen + grocery"}
    all_images = [row["initial_image"] for row in payload["cases"]]
    all_images += [choice for row in payload["cases"]
                   for choice in row.get("choices", [])]
    assert all(image["path"].startswith("_assets/")
               for image in all_images)
    assert all((output / image["path"]).is_file() for image in all_images)
    assert len(list((output / "_assets").glob("*.png"))) == \
        len({image["sha256"] for image in all_images})


def test_merge_dataset_reports_rejects_duplicate_case_id(tmp_path: Path):
    module = importlib.import_module("pipeline.dataset_case_browser")
    first = _write_input_report(
        tmp_path, "first", _b1k_benchmark(), "first")
    second = _write_input_report(
        tmp_path, "second", _b1k_benchmark(), "second")

    with pytest.raises(ValueError, match="duplicate case id: b1k-case-0"):
        module.merge_dataset_reports([
            module.DatasetReport("first", "First", first),
            module.DatasetReport("second", "Second", second),
        ], tmp_path / "merged")


def test_merge_dataset_reports_rejects_image_digest_mismatch(tmp_path: Path):
    module = importlib.import_module("pipeline.dataset_case_browser")
    source = _write_input_report(
        tmp_path, "source", _b1k_benchmark(), "source")
    payload = _payload_from_html(source)
    image = source.parent / payload["cases"][0]["initial_image"]["path"]
    image.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="browser image digest changed"):
        module.merge_dataset_reports([
            module.DatasetReport("source", "Source", source),
        ], tmp_path / "merged")


def test_dataset_browser_html_has_dataset_switches_and_zero_c1_count(
        tmp_path: Path):
    module = importlib.import_module("pipeline.dataset_case_browser")
    r2r = _write_input_report(
        tmp_path, "r2r", _six_task_benchmark(), "AB + C1")
    b1k = _write_input_report(
        tmp_path, "b1k", _b1k_benchmark(), "Ihlen + grocery")

    index = module.merge_dataset_reports([
        module.DatasetReport("r2r", "R2R", r2r),
        module.DatasetReport(
            "behavior1k", "BEHAVIOR-1K Pilot", b1k,
            status="pilot"),
    ], tmp_path / "merged")
    document = index.read_text(encoding="utf-8")

    assert 'id="dataset-switcher"' in document
    assert 'data-dataset-source="R2R"' in document
    assert 'data-dataset-source="BEHAVIOR-1K Pilot"' in document
    assert '<label for="filter-source">数据集</label>' in document
    assert 'data-task-filter="C1_future_view_selection"' in document
    assert "C1 Pending" in document
    assert "dispatchEvent(new Event('change'" in document


def test_dataset_browser_adds_checkpoint_direction_diagnostic_in_place(
        tmp_path: Path):
    module = importlib.import_module("pipeline.dataset_case_browser")
    r2r = _write_input_report(
        tmp_path, "r2r", _six_task_benchmark(), "R2R source")
    report_root = tmp_path / "atlas"
    index = module.merge_dataset_reports([
        module.DatasetReport("r2r", "R2R", r2r),
    ], report_root)
    diagnostic = tmp_path / "checkpoint_direction.v1"
    image = diagnostic / "assets" / "a4.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG\r\n\x1a\na4")
    image_sha256 = _sha256(image.read_bytes())
    item = {
        "id": "a4-case-1",
        "diagnostic_id": "checkpoint_direction",
        "protocol": "checkpoint_direction.v1",
        "headline_eligible": False,
        "result_head": "A",
        "question": "Where is the chair during this Forward?",
        "model_input": {
            "initial_rgb": "assets/a4.png",
            "initial_rgb_sha256": image_sha256,
            "camera_height_above_visible_floor_m": 1.0,
            "hfov_deg": 79.0,
            "vfov_deg": 63.45,
            "body_radius_m": 0.2,
            "actions": [{"type": "forward", "m": 1.0}],
            "target": "chair",
            "checkpoint": {
                "action_index": 1,
                "forward_stage": 1,
                "fraction": 0.5,
                "arc_m": 0.5,
            },
        },
        "choices": [
            {"id": value, "text": value}
            for value in ("front", "left", "right", "rear")
        ],
        "metadata": {
            "dataset": "r2r",
            "scene_id": "scene-a4",
            "referent_track": "visible_object",
            "anchor_protocol": "initial-visible-depth-centroid.v1",
        },
    }
    answer = {
        "id": "a4-case-1",
        "canonical_answer": "right",
        "precise_bearing_deg": 80.0,
        "sector_boundary_margin_deg": 35.0,
        "initial_direction": "front",
        "sector_change": "changed_sector",
        "frame_id": "frame-a4",
        "outcome_id": "outcome-a4",
        "instance_id": 7,
        "checkpoint": dict(item["model_input"]["checkpoint"]),
        "referent_track": "visible_object",
        "anchor_protocol": "initial-visible-depth-centroid.v1",
    }
    (diagnostic / "private").mkdir()
    (diagnostic / "items.jsonl").write_text(
        json.dumps(item) + "\n", encoding="utf-8")
    (diagnostic / "private" / "answers.jsonl").write_text(
        json.dumps(answer) + "\n", encoding="utf-8")
    (diagnostic / "manifest.json").write_text(json.dumps({
        "schema": "checkpoint_direction.v1",
        "diagnostic_id": "checkpoint_direction",
        "headline_eligible": False,
        "item_count": 1,
        "answer_count": 1,
    }), encoding="utf-8")

    updated = module.add_checkpoint_direction_diagnostic(index, diagnostic)

    assert updated == index
    payload = _payload_from_html(index)
    assert payload["coverage"]["A4_checkpoint_direction"] == 1
    assert payload["datasets"][0]["coverage"][
        "A4_checkpoint_direction"] == 1
    assert payload["datasets"][0]["case_count"] == 7
    assert payload["sources"] == [{"label": "R2R", "case_count": 7}]
    row = next(row for row in payload["cases"]
               if row["task_id"] == "A4_checkpoint_direction")
    assert row["scene_id"] == "scene-a4"
    assert row["model_input"]["checkpoint"]["fraction"] == 0.5
    assert row["technical"]["anchor_protocol"] == \
        "initial-visible-depth-centroid.v1"
    assert row["initial_image"]["path"].startswith("_assets/")
    assert (report_root / row["initial_image"]["path"]).is_file()
    document = index.read_text(encoding="utf-8")
    assert 'data-task-filter="A4_checkpoint_direction"' in document
    assert "A4 过程方位" in document
    assert "checkpoint.action_index" in document
    assert "A4 diagnostic" in document


def test_extend_dataset_atlas_adds_truthful_zero_case_dataset(
        tmp_path: Path):
    module = importlib.import_module("pipeline.dataset_case_browser")
    r2r = _write_input_report(
        tmp_path, "r2r", _six_task_benchmark(), "AB + C1")
    b1k = _write_input_report(
        tmp_path, "b1k", _b1k_benchmark(), "Ihlen + grocery")
    base = module.merge_dataset_reports([
        module.DatasetReport("r2r", "R2R", r2r),
        module.DatasetReport(
            "behavior1k", "BEHAVIOR-1K Pilot", b1k,
            status="pilot"),
    ], tmp_path / "base")

    note = (
        "已有 GS source 与 authority smoke；当前没有通过 conseq.v17 "
        "验证并编译成 A1–C1 的 QA。")
    index = module.extend_dataset_atlas(
        base,
        [module.DatasetPlaceholder(
            "gs", "GS", status="collecting", note=note)],
        tmp_path / "three-dataset")

    payload = _payload_from_html(index)
    assert len(payload["cases"]) == 9
    assert payload["datasets"][-1] == {
        "id": "gs",
        "label": "GS",
        "status": "collecting",
        "case_count": 0,
        "coverage": {task_id: 0 for task_id in TASKS},
        "availability_note": note,
    }
    assert payload["dataset_inputs"][-1] == {
        "dataset_id": "gs",
        "kind": "placeholder",
        "status": "collecting",
        "note": note,
    }
    assert {row["dataset_id"] for row in payload["cases"]} == {
        "r2r", "behavior1k"}
    document = index.read_text(encoding="utf-8")
    assert 'data-dataset-source="GS"' in document
    assert "收集中 · 暂无认证 QA" in document
    assert "availability_note" in document
    assert all((index.parent / image["path"]).is_file()
               for row in payload["cases"]
               for image in [row["initial_image"], *row.get("choices", [])])


@pytest.mark.parametrize("kwargs", [
    {"dataset_id": "Bad ID", "label": "GS", "note": "pending"},
    {"dataset_id": "gs", "label": "", "note": "pending"},
    {"dataset_id": "gs", "label": "GS", "status": "candidate",
     "note": "pending"},
    {"dataset_id": "gs", "label": "GS", "note": ""},
])
def test_dataset_placeholder_rejects_misleading_or_invalid_metadata(kwargs):
    module = importlib.import_module("pipeline.dataset_case_browser")
    with pytest.raises(ValueError):
        module.DatasetPlaceholder(**kwargs)


def test_dataset_case_browser_cli_parses_dataset_specs(tmp_path: Path):
    cli = importlib.import_module("scripts.build_dataset_case_browser")

    args = cli.build_arg_parser().parse_args([
        "--dataset", f"r2r:candidate:R2R={tmp_path / 'r2r.html'}",
        "--dataset",
        f"behavior1k:pilot:BEHAVIOR-1K Pilot={tmp_path / 'b1k.html'}",
        "--output", str(tmp_path / "report"),
    ])

    assert [(row.dataset_id, row.label, row.report_path, row.status)
            for row in args.dataset] == [
        ("r2r", "R2R", tmp_path / "r2r.html", "candidate"),
        ("behavior1k", "BEHAVIOR-1K Pilot",
         tmp_path / "b1k.html", "pilot"),
    ]


@pytest.mark.parametrize("value", [
    "missing-separators",
    "Bad ID:candidate:R2R=/tmp/r2r.html",
    "r2r:candidate:=/tmp/r2r.html",
    "r2r:candidate:R2R=",
    "r2r:formal:R2R=/tmp/r2r.html",
])
def test_dataset_case_browser_cli_rejects_invalid_dataset_specs(value):
    cli = importlib.import_module("scripts.build_dataset_case_browser")

    with pytest.raises(SystemExit):
        cli.build_arg_parser().parse_args([
            "--dataset", value, "--output", "/tmp/report"])
