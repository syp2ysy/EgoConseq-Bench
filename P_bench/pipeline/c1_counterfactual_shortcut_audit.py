"""Out-of-band shortcut audit for counterfactual C1 artifacts."""

from __future__ import annotations

import collections
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image

from pipeline import future_view_selection, io_utils


AUDIT_SCHEMA = "egoconseq.c1-counterfactual-shortcut-audit.v1"
TASK_ID = "C1_future_view_selection"
_BASELINE_FIELDS = {
    "initial_to_option_similarity": "initial_similarity_prediction",
    "options_only_visual_medoid": "visual_medoid_prediction",
    "motion_only": "motion_prediction",
}


def _scene_key(row: dict) -> tuple[str, str]:
    return str(row["source_dataset"]), str(row["scene_id"])


def _accuracy(rows: list[dict], field: str) -> float | None:
    if not rows:
        return None
    return sum(
        row[field] == row["answer_position"] for row in rows) / len(rows)


def _cluster_interval(
        rows: list[dict], field: str, *, resamples: int,
        seed: int) -> list[float] | None:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[_scene_key(row)].append(row)
    if len(groups) < 2 or int(resamples) <= 0:
        return None
    scenes = sorted(groups)
    rng = random.Random(int(seed))
    values = []
    for _index in range(int(resamples)):
        sampled = [scenes[rng.randrange(len(scenes))] for _scene in scenes]
        sample_rows = [
            row for scene_id in sampled for row in groups[scene_id]
        ]
        accuracy = _accuracy(sample_rows, field)
        if accuracy is None:
            raise ValueError("C1 shortcut bootstrap sample is empty")
        values.append(accuracy)
    return [
        io_utils.linear_quantile(values, 0.025),
        io_utils.linear_quantile(values, 0.975),
    ]


def _baseline_summary(rows: list[dict], *, resamples: int, seed: int) -> dict:
    return {
        name: {
            "accuracy": _accuracy(rows, field),
            "scene_clustered_95pct_ci": _cluster_interval(
                rows, field, resamples=resamples, seed=seed + index),
        }
        for index, (name, field) in enumerate(_BASELINE_FIELDS.items())
    }


def audit_rows(
        rows: Iterable[dict], *, resamples: int = 2000,
        seed: int = 1729) -> dict:
    """Summarize three fixed C1 predictions with scene uncertainty."""
    values = [dict(row) for row in rows]
    required = {
        "source_dataset", "scene_id", "answer_position",
        *_BASELINE_FIELDS.values(),
    }
    for row in values:
        if (not required.issubset(row) or not isinstance(
                row.get("scene_id"), str) or not row["scene_id"] or
                row.get("source_dataset") not in {"r2r", "b1k", "gs"}):
            raise ValueError("C1 shortcut audit row is incomplete")
        answer = row["answer_position"]
        visual_predictions = [
            row["initial_similarity_prediction"],
            row["visual_medoid_prediction"],
        ]
        motion_prediction = row["motion_prediction"]
        if (not isinstance(answer, int) or isinstance(answer, bool) or
                answer not in range(1, 5) or
                any(not isinstance(value, int) or isinstance(value, bool) or
                    value not in range(1, 5)
                    for value in visual_predictions) or
                not isinstance(motion_prediction, int) or
                isinstance(motion_prediction, bool) or
                motion_prediction not in range(0, 5)):
            raise ValueError("C1 shortcut prediction is invalid")
    baselines = _baseline_summary(
        values, resamples=resamples, seed=seed)
    datasets = sorted({row["source_dataset"] for row in values})
    strata = {}
    for index, dataset in enumerate(datasets):
        selected = [
            row for row in values if row["source_dataset"] == dataset]
        strata[dataset] = {
            "item_count": len(selected),
            "scene_count": len({_scene_key(row) for row in selected}),
            "baselines": _baseline_summary(
                selected, resamples=resamples,
                seed=seed + 100 * (index + 1)),
        }
    return {
        "schema": AUDIT_SCHEMA,
        "formal_artifact": False,
        "headline_eligible": False,
        "threshold_decision": None,
        "item_count": len(values),
        "scene_count": len({_scene_key(row) for row in values}),
        "baselines": baselines,
        "dataset_strata": strata,
    }


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        opened.load()
        if opened.mode != "RGB":
            raise ValueError("C1 shortcut image is not RGB")
        value = np.asarray(opened)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("C1 shortcut image pixels are invalid")
    return np.ascontiguousarray(value)


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    certificate = future_view_selection.block_l1_certificate(left, right)
    return certificate["numerator"] / certificate["denominator"]


def _index_task(rows: Iterable[dict], *, label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if row.get("task_id") != TASK_ID:
            continue
        item_id = str(row.get("id") or "")
        if not item_id or item_id in result:
            raise ValueError(f"{label} ids are missing or duplicated")
        result[item_id] = row
    return result


def _scene_bindings(
        answers: dict[str, dict], atoms: list[dict],
        contexts: list[dict]) -> dict[str, tuple[str, str]]:
    atom_by_id = {str(row.get("id") or ""): row for row in atoms}
    if "" in atom_by_id or len(atom_by_id) != len(atoms):
        raise ValueError("C1 shortcut atom ids are missing or duplicated")
    context_by_digest = {}
    for row in contexts:
        digest = str(row.get("record_sha256") or "")
        context = row.get("context")
        if (not digest or digest in context_by_digest or
                not isinstance(context, dict)):
            raise ValueError("C1 shortcut record contexts are malformed")
        context_by_digest[digest] = context
    result = {}
    for item_id, answer in answers.items():
        atom = atom_by_id.get(str(answer.get("atom_ref") or ""))
        context = None if atom is None else context_by_digest.get(
            str(atom.get("record_sha256") or ""))
        if context is None:
            raise ValueError(f"C1 scene binding is missing for {item_id}")
        source = context.get("source") or {}
        scene_id = context.get("scene_id") or source.get("scene_id")
        dataset = source.get("source_dataset")
        if (not isinstance(scene_id, str) or not scene_id or
                dataset not in {"r2r", "b1k", "gs"}):
            raise ValueError(f"C1 scene binding is missing for {item_id}")
        result[item_id] = (str(dataset), scene_id)
    return result


def _motion_predictions(rows: list[dict]) -> list[int]:
    """Predict action-only answer positions without fitting on the test row."""
    groups = collections.defaultdict(collections.Counter)
    overall = collections.Counter()
    for row in rows:
        position = int(row["answer_position"])
        groups[str(row["action_key"])][position] += 1
        overall[position] += 1
    predictions = []
    for row in rows:
        truth = int(row["answer_position"])
        counts = collections.Counter(groups[str(row["action_key"])])
        counts[truth] -= 1
        counts += collections.Counter()
        if not counts:
            counts = collections.Counter(overall)
            counts[truth] -= 1
            counts += collections.Counter()
        predictions.append(
            0 if not counts else min(
                counts, key=lambda value: (-counts[value], value)))
    return predictions


def _artifact_rows(
        root: Path, items: dict[str, dict], answers: dict[str, dict],
        scene_by_id: dict[str, tuple[str, str]]) -> tuple[list[dict], list[float]]:
    prepared = []
    for item_id in sorted(items):
        item = items[item_id]
        answer = answers[item_id]
        choices = item.get("choices")
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError(f"C1 choices are invalid for {item_id}")
        choice_ids = [str(row.get("id") or "") for row in choices]
        canonical = str(answer.get("canonical_answer") or "")
        if (len(set(choice_ids)) != 4 or canonical not in choice_ids):
            raise ValueError(f"C1 answer is invalid for {item_id}")
        initial = _rgb(root / str(item["model_input"]["initial_rgb"]))
        option_pixels = [
            _rgb(root / str(choice["image"])) for choice in choices
        ]
        initial_prediction = min(
            range(4), key=lambda index: (
                _distance(initial, option_pixels[index]), choice_ids[index]))
        medoid_prediction = min(
            range(4), key=lambda index: (
                sum(_distance(option_pixels[index], other)
                    for other in option_pixels), choice_ids[index]))
        action_key = json.dumps(
            item["model_input"].get("actions") or [], sort_keys=True,
            separators=(",", ":"))
        certificate = answer.get("selection_certificate") or {}
        pair_values = []
        for pair in certificate.get("pairwise_block_l1") or []:
            atom = pair.get("certificate") or {}
            pair_values.append(
                int(atom["numerator"]) / int(atom["denominator"]))
        if len(pair_values) != 6:
            raise ValueError("C1 pairwise block-L1 diagnostics are incomplete")
        dataset, scene_id = scene_by_id[item_id]
        prepared.append({
            "source_dataset": dataset,
            "scene_id": scene_id,
            "answer_position": choice_ids.index(canonical) + 1,
            "initial_similarity_prediction": initial_prediction + 1,
            "visual_medoid_prediction": medoid_prediction + 1,
            "action_key": action_key,
            "pair_values": pair_values,
        })
    motion_predictions = _motion_predictions(prepared)
    rows = [{
        **{key: value for key, value in row.items()
           if key not in {"action_key", "pair_values"}},
        "motion_prediction": motion_prediction,
    } for row, motion_prediction in zip(prepared, motion_predictions)]
    pair_values = [value for row in prepared for value in row["pair_values"]]
    return rows, pair_values


def audit_artifact(
        artifact_root: Path, *, resamples: int = 2000,
        seed: int = 1729, expected_source_authority=None) -> dict:
    """Authenticate a candidate artifact and recompute all C1 baselines."""
    root = Path(artifact_root)
    from pipeline import candidate_preview
    candidate_preview.validate_preview_artifact(
        root, expected_source_authority=expected_source_authority)
    paths = {
        "public/items.jsonl": root / "public" / "items.jsonl",
        "private/answers.jsonl": root / "private" / "answers.jsonl",
        "private/atoms.jsonl": root / "private" / "atoms.jsonl",
        "private/record_contexts.jsonl":
            root / "private" / "record_contexts.jsonl",
    }
    items = _index_task(io_utils.read_jsonl(
        paths["public/items.jsonl"], require_dict=True), label="public C1")
    answers = _index_task(io_utils.read_jsonl(
        paths["private/answers.jsonl"], require_dict=True), label="private C1")
    if set(items) != set(answers):
        raise ValueError("C1 item/answer ids do not match")
    atoms = io_utils.read_jsonl(
        paths["private/atoms.jsonl"], require_dict=True)
    contexts = io_utils.read_jsonl(
        paths["private/record_contexts.jsonl"], require_dict=True)
    scene_by_id = _scene_bindings(answers, atoms, contexts)
    rows, pair_values = _artifact_rows(root, items, answers, scene_by_id)
    report = audit_rows(rows, resamples=resamples, seed=seed)
    return {
        **report,
        "artifact_root": str(root),
        "pairwise_block_l1": {
            "pair_count": len(pair_values),
            "minimum": min(pair_values, default=None),
            "median": (
                float(np.median(pair_values)) if pair_values else None),
            "maximum": max(pair_values, default=None),
        },
        "source_sha256": {
            name: io_utils.sha256_file(path)
            for name, path in paths.items()
        },
    }


def write_artifact_audit(
        artifact_root: Path, output_path: Path, *,
        resamples: int = 2000, seed: int = 1729,
        expected_source_authority=None) -> dict:
    """Write a deterministic audit outside the benchmark tree."""
    root = Path(artifact_root).resolve()
    output = Path(output_path).resolve()
    if output == root or root in output.parents:
        raise ValueError("C1 shortcut audit output must be outside benchmark")
    report = audit_artifact(
        Path(artifact_root), resamples=resamples, seed=seed,
        expected_source_authority=expected_source_authority)
    io_utils.atomic_write_json(
        Path(output_path), report, allow_nan=False, sort_keys=True)
    return report
