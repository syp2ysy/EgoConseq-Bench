"""Stop/append decisions from compiled candidate QA, never from pose labels."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

from pipeline import benchmark, config, dataset_contracts, io_utils


QUOTA_REPORT_SCHEMA = "egoconseq.candidate-quota.v3"
SIX_TASK_MACRO_SCHEMA = "egoconseq.six-task-macro.v1"
_DATASETS = tuple(config.BACKGROUND_MIN_UNIQUE_FRAMES_BY_DATASET)
_TASKS = tuple(benchmark.ABC_CANDIDATE_TASK_IDS)


@dataclass(frozen=True)
class ArtifactIndex:
    """The four joined QA tables loaded once for all dataset summaries."""

    items: dict
    answers: dict
    atoms: dict
    contexts: dict


def collection_quota_rows() -> dict:
    """Return the single configured quota contract for every dataset."""
    return {
        dataset: {
            "supported_tasks": list(_TASKS),
            "min_total_items": config.BACKGROUND_MIN_TOTAL_ITEMS,
            "min_per_task": config.BACKGROUND_MIN_ITEMS_PER_TASK,
            "required_lengths": list(config.GEN_LENGTHS),
            "min_per_length": config.BACKGROUND_MIN_ITEMS_PER_LENGTH,
            "min_unique_frames":
                config.BACKGROUND_MIN_UNIQUE_FRAMES_BY_DATASET[dataset],
            "min_scene_families":
                config.BACKGROUND_MIN_SCENE_FAMILIES_BY_DATASET[dataset],
            "max_scene_family_fraction":
                config.BACKGROUND_MAX_SCENE_FAMILY_FRACTION[dataset],
        }
        for dataset in _DATASETS
    }


def allows_dataset_stop(
        dataset: str, *, own_complete: bool, state: Mapping) -> bool:
    """Keep the configured overflow dataset running to the global target."""
    if not own_complete:
        return False
    if str(dataset) != config.BACKGROUND_OVERFLOW_DATASET:
        return True
    headline_items = sum(
        int((((state.get("datasets") or {}).get(name) or {}).get("quota") or
             {}).get("total_supported_items") or 0)
        for name in _DATASETS)
    return headline_items >= config.BACKGROUND_MIN_HEADLINE_ITEMS_GLOBAL


def six_task_macro_report(task_scores: Mapping[str, float]) -> dict:
    """Report an equal-weight six-task macro without rewriting QA items."""
    if not isinstance(task_scores, Mapping) or set(task_scores) != set(_TASKS):
        raise ValueError("six task macro requires all six task scores")
    scores = {}
    for task_id in _TASKS:
        value = task_scores[task_id]
        if value is None:
            scores[task_id] = None
            continue
        if not math.isfinite(float(value)):
            raise ValueError("six task macro scores are invalid")
        scores[task_id] = float(value)
    return {
        "schema": SIX_TASK_MACRO_SCHEMA,
        "tasks": scores,
        "macro": (
            None if any(value is None for value in scores.values()) else
            sum(scores.values()) / len(scores)),
    }


def dataset_macro_reports(
        artifact_root: Path, item_scores: Mapping[str, float], *,
        artifact_index: ArtifactIndex = None) -> dict:
    """Compute six-task macros within each authenticated dataset stratum."""
    index = artifact_index or load_artifact_index(artifact_root)
    items = index.items
    answers = index.answers
    atoms = index.atoms
    contexts = {
        record_sha256: row["context"]
        for record_sha256, row in index.contexts.items()
    }
    if set(items) != set(answers) or set(item_scores) != set(items):
        raise ValueError("dataset macro public/private scores differ")
    values = {dataset: {task_id: [] for task_id in _TASKS}
              for dataset in _DATASETS}
    for item_id, item in items.items():
        answer = answers[item_id]
        try:
            atom = atoms[str(answer["atom_ref"])]
            context = contexts[str(atom["record_sha256"])]
        except (KeyError, TypeError) as error:
            raise ValueError("dataset macro source binding is missing") \
                from error
        dataset = _source_dataset(context)
        task_id = str(item.get("task_id") or "")
        if dataset not in values:
            raise ValueError("dataset macro source dataset is invalid")
        if task_id not in values[dataset]:
            raise ValueError("dataset macro task is invalid")
        values[dataset][task_id].append(float(item_scores[item_id]))
    return {dataset: six_task_macro_report({
        task_id: sum(scores) / len(scores) if scores else None
        for task_id, scores in tasks.items()})
        for dataset, tasks in values.items()}


def _unique_index(rows: Iterable[dict], field: str, *, label: str) -> dict:
    result = {}
    for row in rows:
        value = str(row.get(field) or "")
        if not value or value in result:
            raise ValueError(f"candidate quota {label} ids are invalid")
        result[value] = row
    return result


def load_artifact_index(artifact_root: Path) -> ArtifactIndex:
    """Read the joined QA tables once for quota and macro summaries."""
    root = Path(artifact_root)
    return ArtifactIndex(
        items=_unique_index(io_utils.read_jsonl(
            root / "public" / "items.jsonl", require_dict=True), "id",
            label="public item"),
        answers=_unique_index(io_utils.read_jsonl(
            root / "private" / "answers.jsonl", require_dict=True), "id",
            label="private answer"),
        atoms=_unique_index(io_utils.read_jsonl(
            root / "private" / "atoms.jsonl", require_dict=True), "id",
            label="source atom"),
        contexts=_unique_index(io_utils.read_jsonl(
            root / "private" / "record_contexts.jsonl", require_dict=True),
            "record_sha256", label="record context"),
    )


def _positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _source_dataset(context: dict) -> str:
    values = {
        str(container.get("source_dataset"))
        for container in (
            context.get("source") or {},
            context.get("collection_contract") or {},
        )
        if container.get("source_dataset")
    }
    if len(values) != 1:
        raise ValueError("candidate quota dataset binding is invalid")
    return values.pop()


def summarize_artifact(
        artifact_root: Path, *, dataset: str,
        supported_tasks: Iterable[str], min_total_items: int,
        min_per_task: int,
        required_lengths: Iterable[int], min_per_length: int,
        min_unique_frames: int, min_scene_families: int,
        max_scene_family_fraction: float,
        scene_families: Mapping[str, str] = None,
        artifact_index: ArtifactIndex = None) -> dict:
    """Summarize one compiled artifact and decide whether to append shards.

    The controller intentionally runs after QA compilation.  It never changes
    an action bank or a pose-local distribution; a shortfall asks the external
    supervisor for another independent shard.
    """
    root = Path(artifact_root)
    dataset = str(dataset).strip().lower()
    dataset_contracts.dataset_source_contract(dataset)
    tasks = tuple(str(value) for value in supported_tasks)
    if not tasks or any(not value for value in tasks) or \
            len(set(tasks)) != len(tasks) or any(
                value not in benchmark.ABC_CANDIDATE_TASK_IDS
                for value in tasks):
        raise ValueError("candidate quota supported tasks are invalid")
    lengths = tuple(int(value) for value in required_lengths)
    if (not lengths or any(value < 1 for value in lengths) or
            len(set(lengths)) != len(lengths)):
        raise ValueError("candidate quota required lengths are invalid")
    total_minimum = _positive_int(
        min_total_items, label="min_total_items")
    task_minimum = _positive_int(min_per_task, label="min_per_task")
    length_minimum = _positive_int(
        min_per_length, label="min_per_length")
    frame_minimum = _positive_int(
        min_unique_frames, label="min_unique_frames")
    family_minimum = _positive_int(
        min_scene_families, label="min_scene_families")
    family_fraction_limit = float(max_scene_family_fraction)
    if not 0.0 < family_fraction_limit <= 1.0:
        raise ValueError("max_scene_family_fraction must be in (0, 1]")
    family_by_scene = {
        str(scene_id): str(family)
        for scene_id, family in (scene_families or {}).items()}
    if any(not scene_id or not family
           for scene_id, family in family_by_scene.items()):
        raise ValueError("candidate quota scene families are invalid")

    index = artifact_index or load_artifact_index(root)
    items = index.items
    answers = index.answers
    atoms = index.atoms
    contexts = index.contexts
    if set(items) != set(answers):
        raise ValueError("candidate quota public/private ids do not match")

    task_lengths = {
        task_id: collections.Counter() for task_id in tasks}
    task_scenes = {
        task_id: collections.Counter() for task_id in tasks}
    frame_routes = {}
    for item_id, item in items.items():
        answer = answers[item_id]
        if answer.get("task_id") != item.get("task_id"):
            raise ValueError("candidate quota public/private task changed")
        atom = atoms.get(str(answer.get("atom_ref") or ""))
        if atom is None:
            raise ValueError("candidate quota source atom is missing")
        context_row = contexts.get(str(atom.get("record_sha256") or ""))
        if context_row is None:
            raise ValueError("candidate quota record context is missing")
        context = context_row.get("context") or {}
        if _source_dataset(context) != dataset:
            continue
        task_id = str(item.get("task_id") or "")
        if task_id not in task_lengths:
            continue
        actions = (item.get("model_input") or {}).get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("candidate quota action program is invalid")
        length = len(actions)
        if length not in lengths:
            raise ValueError("candidate quota action length is out of scope")
        scene_id = context.get("scene_id") or \
            (context.get("source") or {}).get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("candidate quota scene binding is missing")
        input_asset = answer.get("input_asset") or {}
        frame_sha256 = str(
            input_asset.get("raw_sha256") or
            input_asset.get("sha256") or "")
        if not frame_sha256:
            raise ValueError("candidate quota source-frame identity is missing")
        route = (scene_id, family_by_scene.get(scene_id, scene_id))
        previous = frame_routes.setdefault(frame_sha256, route)
        if previous != route:
            raise ValueError("candidate quota source frame crosses scenes")
        task_lengths[task_id][length] += 1
        task_scenes[task_id][scene_id] += 1

    task_reports = {}
    all_lengths = collections.Counter()
    for task_id in tasks:
        length_counts = task_lengths[task_id]
        scene_counts = task_scenes[task_id]
        count = sum(length_counts.values())
        all_lengths.update(length_counts)
        scene_count = len(scene_counts)
        count_shortfall = max(0, task_minimum - count)
        task_reports[task_id] = {
            "count": count,
            "count_shortfall": count_shortfall,
            "counts_by_length": {
                f"L{length}": int(length_counts.get(length, 0))
                for length in lengths},
            "scene_count": scene_count,
            "scene_counts": dict(sorted(scene_counts.items())),
            "complete": count_shortfall == 0,
        }

    counts_by_length = {
        f"L{length}": int(all_lengths.get(length, 0))
        for length in lengths}
    length_shortfall = {
        key: length_minimum - count
        for key, count in counts_by_length.items()
        if count < length_minimum}
    total_supported_items = sum(all_lengths.values())
    total_item_shortfall = max(0, total_minimum - total_supported_items)
    unique_frames = len(frame_routes)
    frame_shortfall = max(0, frame_minimum - unique_frames)
    family_frame_counts = collections.Counter(
        family for _scene, family in frame_routes.values())
    family_count = len(family_frame_counts)
    family_shortfall = max(0, family_minimum - family_count)
    dominant_family_fraction = (
        max(family_frame_counts.values()) / unique_frames
        if unique_frames else None)
    family_fraction_exceeded = (
        dominant_family_fraction is not None and
        dominant_family_fraction > family_fraction_limit + 1e-12)
    complete = (
        total_item_shortfall == 0 and not length_shortfall and
        frame_shortfall == 0 and family_shortfall == 0 and
        not family_fraction_exceeded and
        all(value["complete"] for value in task_reports.values()))
    return {
        "schema": QUOTA_REPORT_SCHEMA,
        "dataset": dataset,
        "supported_tasks": list(tasks),
        "min_total_items": total_minimum,
        "min_per_task": task_minimum,
        "required_lengths": list(lengths),
        "min_per_length": length_minimum,
        "min_unique_frames": frame_minimum,
        "unique_source_frames": unique_frames,
        "unique_source_frame_shortfall": frame_shortfall,
        "min_scene_families": family_minimum,
        "scene_family_count": family_count,
        "scene_family_shortfall": family_shortfall,
        "scene_family_frame_counts": dict(sorted(family_frame_counts.items())),
        "dominant_scene_family_fraction": dominant_family_fraction,
        "max_scene_family_fraction": family_fraction_limit,
        "scene_family_fraction_exceeded": family_fraction_exceeded,
        "total_supported_items": total_supported_items,
        "total_item_shortfall": total_item_shortfall,
        "counts_by_length": counts_by_length,
        "length_shortfall": length_shortfall,
        "tasks": task_reports,
        "complete": complete,
        "decision": "stop" if complete else "append_shards",
    }
