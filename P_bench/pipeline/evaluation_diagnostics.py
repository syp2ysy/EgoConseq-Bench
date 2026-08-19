"""Non-headline diagnostics for the ABC candidate evaluator."""

from __future__ import annotations

import collections
import random

from pipeline import io_utils


def _optional_mean(values) -> float | None:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else None


def clustered_bootstrap_summary(
        rows: list[dict], *, cluster_field: str,
        resamples: int = 2000, seed: int = 1729) -> dict:
    """Bootstrap the pooled item mean by resampling whole clusters."""
    groups = collections.defaultdict(list)
    for row in rows:
        cluster = row.get(cluster_field)
        if cluster is not None:
            groups[str(cluster)].append(float(row["score"]))
    item_scores = [value for values in groups.values() for value in values]
    point = _optional_mean(item_scores)
    summary = {
        "item_count": len(item_scores),
        "cluster_count": len(groups),
        "point_estimate": point,
        "resamples": int(resamples),
        "interval_95": None,
    }
    if len(groups) < 2 or not item_scores or int(resamples) <= 0:
        return summary
    keys = sorted(groups)
    rng = random.Random(int(seed))
    estimates = []
    for _index in range(int(resamples)):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        values = [score for key in sampled for score in groups[key]]
        estimates.append(sum(values) / len(values))
    summary["interval_95"] = [
        io_utils.linear_quantile(estimates, 0.025),
        io_utils.linear_quantile(estimates, 0.975),
    ]
    return summary


def exact_group_summary(
        rows: list[dict], *, group_field: str,
        required_size: int | None = None) -> dict:
    """Score a group only when every one of its published members is right."""
    groups = collections.defaultdict(list)
    for row in rows:
        group = row.get(group_field)
        if group is not None:
            groups[str(group)].append(float(row["score"]))
    eligible = [
        values for values in groups.values()
        if required_size is None or len(values) == int(required_size)
    ]
    return {
        "group_count": len(eligible),
        "exact_accuracy": (
            _optional_mean(
                float(all(score == 1.0 for score in values))
                for values in eligible)
            if eligible else None),
    }


def chance_diagnostics(rows_by_task: dict, by_task: dict) -> dict:
    diagnostics = {}
    for task_id, rows in rows_by_task.items():
        chance = _optional_mean(row["random_chance"] for row in rows)
        accuracy = by_task.get(task_id)
        normalized = None
        if (accuracy is not None and chance is not None and chance < 1.0):
            normalized = (float(accuracy) - chance) / (1.0 - chance)
        diagnostics[task_id] = {
            "mean_random_chance": chance,
            "chance_normalized_accuracy": normalized,
        }
    return diagnostics


def a1_safety_diagnostics(rows: list[dict]) -> dict:
    collision = [row for row in rows if row["canonical_answer"] == "collision"]
    clear = [row for row in rows if row["canonical_answer"] == "no_collision"]
    collision_recall = _optional_mean(row["score"] for row in collision)
    clear_recall = _optional_mean(row["score"] for row in clear)
    return {
        "balanced_accuracy": (
            (collision_recall + clear_recall) / 2.0
            if collision_recall is not None and clear_recall is not None
            else None),
        "collision_recall": collision_recall,
        "false_safe_rate": (
            _optional_mean(
                float(row["submitted_answer"] == "no_collision")
                for row in collision)
            if collision else None),
        "collision_count": len(collision),
        "clear_count": len(clear),
    }


def b1_numeric_diagnostics(rows: list[dict]) -> dict:
    errors = []
    for row in rows:
        predicted = row.get("submitted_choice_value_m")
        precise = row.get("precise_distance_m")
        if predicted is not None and precise is not None:
            errors.append(abs(float(predicted) - float(precise)))
    return {
        "valid_numeric_choice_count": len(errors),
        "mae_m": _optional_mean(errors),
        "within_0_25m_accuracy": (
            _optional_mean(float(value <= 0.25) for value in errors)
            if errors else None),
    }


def joint_task_summary(
        rows: list[dict], required_tasks: tuple[str, ...]) -> dict:
    by_rollout = collections.defaultdict(list)
    for row in rows:
        by_rollout[row["atom_ref"]].append(row)
    selected = []
    required = set(required_tasks)
    for atom_ref, values in by_rollout.items():
        by_task = {value["task_id"]: value for value in values}
        if set(by_task) >= required:
            selected.extend({
                "joint_group": atom_ref,
                "score": by_task[task_id]["score"],
            } for task_id in required_tasks)
    return exact_group_summary(
        selected, group_field="joint_group",
        required_size=len(required_tasks))
