"""Merge static candidate browsers into one self-contained dataset atlas.

This module operates only on the read-only browser projection.  It does not
compile QA or derive GT.  Each input browser must already have been produced
from its dataset-specific validated candidate artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from pipeline import benchmark, case_browser, io_utils


_DATASET_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_PAYLOAD = re.compile(
    r'<script id="case-browser-data" type="application/json">(.*?)</script>',
    re.DOTALL)
_STATUSES = frozenset({"candidate", "pilot", "calibration_only"})
_PLACEHOLDER_STATUSES = frozenset({"collecting", "unavailable"})
_HEX = frozenset("0123456789abcdef")


def _browser_task_ids() -> tuple[str, ...]:
    return (
        *benchmark.ABC_CANDIDATE_TASK_IDS,
        *case_browser.BROWSER_DIAGNOSTIC_TASK_IDS,
    )


@dataclass(frozen=True)
class DatasetReport:
    """One dataset-level browser input and its review label."""

    dataset_id: str
    label: str
    report_path: Path
    status: str = "candidate"

    def __post_init__(self) -> None:
        if not _DATASET_ID.fullmatch(str(self.dataset_id)):
            raise ValueError("dataset id must be a lowercase slug")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("dataset label must be non-empty")
        if self.status not in _STATUSES:
            raise ValueError("dataset status is invalid")
        object.__setattr__(self, "report_path", Path(self.report_path))


@dataclass(frozen=True)
class DatasetPlaceholder:
    """Truthful atlas entry for a dataset with no validated QA artifact."""

    dataset_id: str
    label: str
    note: str
    status: str = "collecting"

    def __post_init__(self) -> None:
        if not _DATASET_ID.fullmatch(str(self.dataset_id)):
            raise ValueError("dataset id must be a lowercase slug")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("dataset label must be non-empty")
        if self.status not in _PLACEHOLDER_STATUSES:
            raise ValueError("dataset placeholder status is invalid")
        if not isinstance(self.note, str) or not self.note.strip():
            raise ValueError("dataset placeholder note must be non-empty")


def load_browser_payload(report_path: Path) -> dict:
    """Read and structurally validate one embedded case-browser payload."""
    path = Path(report_path)
    if not path.is_file():
        raise ValueError(f"dataset browser report is missing: {path}")
    match = _PAYLOAD.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("dataset browser report has no embedded payload")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ValueError("dataset browser payload is invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != \
            case_browser.BROWSER_SCHEMA:
        raise ValueError("dataset browser payload schema is invalid")
    if payload.get("candidate_only") is not True or \
            payload.get("headline_eligible") is not False:
        raise ValueError("dataset browser input is not candidate-only")
    cases = payload.get("cases")
    coverage = payload.get("coverage")
    if not isinstance(cases, list) or not isinstance(coverage, dict):
        raise ValueError("dataset browser cases or coverage are invalid")
    required = set(benchmark.ABC_CANDIDATE_TASK_IDS)
    allowed = set(_browser_task_ids())
    coverage_keys = set(coverage)
    if not required.issubset(coverage_keys) or not coverage_keys.issubset(
            allowed):
        raise ValueError("dataset browser coverage tasks are invalid")
    observed = {task_id: 0 for task_id in coverage}
    seen = set()
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("dataset browser case is invalid")
        case_id = row.get("id")
        task_id = row.get("task_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("dataset browser case ids are invalid")
        if task_id not in observed:
            raise ValueError("dataset browser task id is invalid")
        seen.add(case_id)
        observed[task_id] += 1
    normalized = {task_id: int(count)
                  for task_id, count in coverage.items()}
    if normalized != observed:
        raise ValueError("dataset browser coverage differs from its cases")
    return payload


def _digest(value: object) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in _HEX
                                for character in digest):
        raise ValueError("browser image digest is invalid")
    return digest


def _stage_image(image: dict, *, input_root: Path,
                 output_root: Path) -> None:
    if not isinstance(image, dict):
        raise ValueError("browser image entry is invalid")
    expected = _digest(image.get("sha256"))
    relative = image.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("browser image path is invalid")
    source = (input_root / relative).resolve()
    if not source.is_file() or io_utils.sha256_file(source) != expected:
        raise ValueError(f"browser image digest changed: {source}")
    assets = output_root / "_assets"
    assets.mkdir(parents=True, exist_ok=True)
    target = assets / f"{expected}.png"
    if target.exists():
        if not target.is_file() or io_utils.sha256_file(target) != expected:
            raise ValueError("staged browser image digest changed")
    else:
        linked = io_utils.link_or_copy_file(source, target)
        if not linked:
            if io_utils.sha256_file(target) != expected:
                target.unlink(missing_ok=True)
                raise ValueError("copied browser image digest changed")
    image["path"] = f"_assets/{expected}.png"


def _stage_case_images(row: dict, *, input_root: Path,
                       output_root: Path) -> None:
    _stage_image(
        row.get("initial_image"), input_root=input_root,
        output_root=output_root)
    choices = row.get("choices") or []
    if not isinstance(choices, list):
        raise ValueError("browser image choices are invalid")
    for choice in choices:
        _stage_image(
            choice, input_root=input_root, output_root=output_root)


def merge_dataset_reports(
        reports: list[DatasetReport], output_root: Path) -> Path:
    """Bundle ordered dataset reports into one static review browser."""
    if not reports:
        raise ValueError("at least one dataset report is required")
    dataset_ids = [report.dataset_id for report in reports]
    labels = [report.label for report in reports]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("dataset ids must be unique")
    if len(set(labels)) != len(labels):
        raise ValueError("dataset labels must be unique")
    output_root = Path(output_root)
    coverage = {task_id: 0
                for task_id in benchmark.ABC_CANDIDATE_TASK_IDS}
    cases = []
    datasets = []
    inputs = []
    seen_case_ids = set()
    for report in reports:
        source_payload = load_browser_payload(report.report_path)
        dataset_coverage = {
            task_id: int(count)
            for task_id, count in source_payload["coverage"].items()
        }
        dataset_cases = []
        for original in source_payload["cases"]:
            row = json.loads(json.dumps(original))
            case_id = row["id"]
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case id: {case_id}")
            seen_case_ids.add(case_id)
            technical = row.get("technical")
            if not isinstance(technical, dict):
                raise ValueError("browser technical metadata is invalid")
            if "dataset_source" in technical:
                raise ValueError("browser case already has dataset provenance")
            technical["dataset_source"] = str(row.get("source_label") or "")
            row["dataset_id"] = report.dataset_id
            row["dataset_label"] = report.label
            row["source_label"] = report.label
            _stage_case_images(
                row, input_root=report.report_path.parent,
                output_root=output_root)
            dataset_cases.append(row)
        cases.extend(dataset_cases)
        for task_id, count in dataset_coverage.items():
            coverage.setdefault(task_id, 0)
            coverage[task_id] += count
        datasets.append({
            "id": report.dataset_id,
            "label": report.label,
            "status": report.status,
            "case_count": len(dataset_cases),
            "coverage": dataset_coverage,
        })
        inputs.append({
            "dataset_id": report.dataset_id,
            "report_path": str(report.report_path.resolve()),
            "report_sha256": io_utils.sha256_file(report.report_path),
        })
    for dataset in datasets:
        for task_id in coverage:
            dataset["coverage"].setdefault(task_id, 0)
    payload = {
        "schema": case_browser.BROWSER_SCHEMA,
        "candidate_only": True,
        "headline_eligible": False,
        "coverage": coverage,
        "datasets": datasets,
        "dataset_inputs": inputs,
        "sources": [{
            "label": dataset["label"],
            "case_count": dataset["case_count"],
        } for dataset in datasets],
        "cases": cases,
    }
    return case_browser.render_case_browser_payload(
        payload, output_root, dataset_mode=True)


def _checkpoint_direction_cases(
        diagnostic_root: Path, *, report_root: Path,
        labels: dict[str, str]) -> list[dict]:
    """Project the A4 sidecar into the existing browser row shape."""
    diagnostic_root = Path(diagnostic_root)
    manifest = json.loads(
        (diagnostic_root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "checkpoint_direction.v1" or
            manifest.get("headline_eligible") is not False):
        raise ValueError("checkpoint-direction manifest is invalid")
    items = io_utils.read_jsonl(
        diagnostic_root / "items.jsonl", require_dict=True)
    answers = {
        str(row["id"]): row for row in io_utils.read_jsonl(
            diagnostic_root / "private" / "answers.jsonl",
            require_dict=True)
    }
    if (len(items) != int(manifest.get("item_count", -1)) or
            len(answers) != len(items)):
        raise ValueError("checkpoint-direction artifact counts differ")
    rows = []
    staged = set()
    for item in items:
        case_id = str(item.get("id") or "")
        answer = answers.get(case_id)
        metadata = item.get("metadata") or {}
        model_input = dict(item.get("model_input") or {})
        dataset_id = str(metadata.get("dataset") or "")
        if (not case_id or answer is None or dataset_id not in labels or
                item.get("protocol") != "checkpoint_direction.v1" or
                item.get("headline_eligible") is not False):
            raise ValueError("checkpoint-direction item is invalid")
        image = {
            "path": str(model_input.pop("initial_rgb")),
            "sha256": str(model_input.pop("initial_rgb_sha256")),
        }
        if image["sha256"] not in staged:
            _stage_image(
                image, input_root=diagnostic_root,
                output_root=report_root)
            staged.add(image["sha256"])
        else:
            image["path"] = f'_assets/{image["sha256"]}.png'
        choices = {
            str(choice.get("id")): str(
                choice.get("text", choice.get("id")))
            for choice in item.get("choices") or []
            if isinstance(choice, dict)
        }
        canonical = str(answer.get("canonical_answer"))
        if canonical not in choices:
            raise ValueError("checkpoint-direction answer is not a choice")
        rows.append({
            "id": case_id,
            "task_id": case_browser.CHECKPOINT_DIRECTION_TASK_ID,
            "head": "A",
            "source_label": labels[dataset_id],
            "dataset_id": dataset_id,
            "dataset_label": labels[dataset_id],
            "scene_id": str(metadata.get("scene_id") or "unknown"),
            "question": str(item["question"]),
            "response_mode": "open_text",
            "answer": {"raw": canonical, "display": choices[canonical]},
            "initial_image": image,
            "model_input": model_input,
            "atom_ref": f"diagnostic:{case_id}",
            "oracle_ref": {
                "frame_id": answer.get("frame_id"),
                "outcome_id": answer.get("outcome_id"),
                "checkpoint": answer.get("checkpoint"),
            },
            "technical": {
                "diagnostic_only": True,
                "referent_track": answer.get("referent_track"),
                "anchor_protocol": answer.get("anchor_protocol"),
                "instance_id": answer.get("instance_id"),
                "precise_bearing_deg": answer.get("precise_bearing_deg"),
                "sector_boundary_margin_deg": answer.get(
                    "sector_boundary_margin_deg"),
                "initial_direction": answer.get("initial_direction"),
                "sector_change": answer.get("sector_change"),
            },
        })
    return rows


def add_checkpoint_direction_diagnostic(
        report_path: Path, diagnostic_root: Path) -> Path:
    """Add or replace A4 rows in one existing dataset atlas in place."""
    report_path = Path(report_path)
    payload = json.loads(json.dumps(load_browser_payload(report_path)))
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("checkpoint direction requires a dataset atlas")
    labels = {str(row["id"]): str(row["label"]) for row in datasets}
    cases = [
        row for row in payload["cases"]
        if row.get("task_id") != case_browser.CHECKPOINT_DIRECTION_TASK_ID
    ]
    cases.extend(_checkpoint_direction_cases(
        diagnostic_root, report_root=report_path.parent, labels=labels))
    ids = [str(row["id"]) for row in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("dataset browser case ids are not unique")
    task_ids = _browser_task_ids()
    payload["cases"] = cases
    payload["coverage"] = {
        task_id: sum(row["task_id"] == task_id for row in cases)
        for task_id in task_ids
    }
    for dataset in datasets:
        dataset_cases = [
            row for row in cases
            if row.get("dataset_id") == dataset["id"]]
        dataset["case_count"] = len(dataset_cases)
        dataset["coverage"] = {
            task_id: sum(
                row["task_id"] == task_id for row in dataset_cases)
            for task_id in task_ids
        }
    payload["sources"] = [{
        "label": dataset["label"],
        "case_count": dataset["case_count"],
    } for dataset in datasets]
    inputs = payload.get("dataset_inputs")
    if not isinstance(inputs, list):
        raise ValueError("dataset atlas inputs are invalid")
    payload["dataset_inputs"] = [
        row for row in inputs
        if row.get("kind") != "checkpoint-direction-diagnostic"
    ]
    payload["dataset_inputs"].append({
        "kind": "checkpoint-direction-diagnostic",
        "schema": "checkpoint_direction.v1",
        "artifact_path": str(Path(diagnostic_root).resolve()),
        "manifest_sha256": io_utils.sha256_file(
            Path(diagnostic_root) / "manifest.json"),
        "item_count": payload["coverage"][
            case_browser.CHECKPOINT_DIRECTION_TASK_ID],
    })
    return case_browser.render_case_browser_payload(
        payload, report_path.parent, dataset_mode=True)


def extend_dataset_atlas(
        input_report: Path, placeholders: list[DatasetPlaceholder],
        output_root: Path) -> Path:
    """Copy one validated atlas and add explicit zero-case datasets.

    This is a review-only projection.  A placeholder never creates a case or
    changes coverage; it only makes missing current-schema QA visible instead
    of silently omitting the dataset from the browser.
    """
    if not placeholders:
        raise ValueError("at least one dataset placeholder is required")
    input_report = Path(input_report)
    output_root = Path(output_root)
    if input_report.resolve().parent == output_root.resolve():
        raise ValueError("extended dataset atlas requires a new output root")
    payload = json.loads(json.dumps(load_browser_payload(input_report)))
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("base browser is not a dataset atlas")
    dataset_ids = [row.get("id") for row in datasets
                   if isinstance(row, dict)]
    dataset_labels = [row.get("label") for row in datasets
                      if isinstance(row, dict)]
    if len(dataset_ids) != len(datasets) or \
            len(set(dataset_ids)) != len(dataset_ids) or \
            len(set(dataset_labels)) != len(dataset_labels):
        raise ValueError("base dataset atlas metadata is invalid")
    for row in payload["cases"]:
        if row.get("dataset_id") not in dataset_ids:
            raise ValueError("base atlas case has unknown dataset provenance")
        _stage_case_images(
            row, input_root=input_report.parent, output_root=output_root)
    inputs = payload.get("dataset_inputs")
    sources = payload.get("sources")
    if not isinstance(inputs, list) or not isinstance(sources, list):
        raise ValueError("base dataset atlas provenance is invalid")
    zero_coverage = {task_id: 0 for task_id in payload["coverage"]}
    for placeholder in placeholders:
        if placeholder.dataset_id in dataset_ids:
            raise ValueError("dataset ids must be unique")
        if placeholder.label in dataset_labels:
            raise ValueError("dataset labels must be unique")
        dataset_ids.append(placeholder.dataset_id)
        dataset_labels.append(placeholder.label)
        datasets.append({
            "id": placeholder.dataset_id,
            "label": placeholder.label,
            "status": placeholder.status,
            "case_count": 0,
            "coverage": dict(zero_coverage),
            "availability_note": placeholder.note,
        })
        inputs.append({
            "dataset_id": placeholder.dataset_id,
            "kind": "placeholder",
            "status": placeholder.status,
            "note": placeholder.note,
        })
        sources.append({"label": placeholder.label, "case_count": 0})
    return case_browser.render_case_browser_payload(
        payload, output_root, dataset_mode=True)
