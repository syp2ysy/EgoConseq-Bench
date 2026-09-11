"""Benchmark totals; length allocations may adapt to image-qualified supply."""

from __future__ import annotations

import json
from pathlib import Path


SPEC_PATH = Path(__file__).resolve().parents[1] / "specs" / "seen_5000_v1.json"
_SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

DATASETS = tuple(_SPEC["datasets"])
TASKS = tuple(_SPEC["tasks"])
DATASET_TASK_TOTALS = {
    dataset: {task: int(count) for task, count in totals.items()}
    for dataset, totals in _SPEC["dataset_task_totals"].items()
}
C1_LENGTH_TOTALS = {
    dataset: tuple(int(value) for value in counts)
    for dataset, counts in _SPEC["c1_length_totals"].items()
}
A2_START_TOTALS = {
    dataset: {
        int(length): {start: int(count) for start, count in totals.items()}
        for length, totals in lengths.items()
    }
    for dataset, lengths in _SPEC["a2_start_totals"].items()
}
MINIMUM_TURN_FIRST_FRACTION = float(_SPEC["minimum_turn_first_fraction"])


def configure(path: Path) -> None:
    """Select one benchmark spec per CLI run; keep imported quota references live."""
    global SPEC_PATH, _SPEC
    SPEC_PATH = Path(path).resolve()
    _SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    for target, values in (
        (DATASET_TASK_TOTALS, _SPEC["dataset_task_totals"]),
        (C1_LENGTH_TOTALS, {ds: tuple(counts) for ds, counts in _SPEC["c1_length_totals"].items()}),
        (A2_START_TOTALS, {ds: {int(length): starts for length, starts in lengths.items()}
                           for ds, lengths in _SPEC["a2_start_totals"].items()}),
    ):
        target.clear()
        target.update(values)
    from post_QA.seen_build import selection
    selection._a2_slot_counts.cache_clear()


def supported_tasks(dataset: str) -> tuple[str, ...]:
    """Record capabilities do not depend on benchmark quotas."""
    from pipeline.abc1_record import supported_tasks as record_tasks
    tasks = record_tasks(dataset)
    return tasks if dataset == "gs" else (*tasks[:-1], "B3", "C1")
