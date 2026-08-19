"""Out-of-band audits for learnable QA-policy shortcuts.

This module reads an already compiled candidate artifact.  It does not alter
Record/Oracle facts, Task-GT, QA eligibility, choices, or scoring.  In
particular, short B1 distances remain valid geometric facts even when the
four-choice policy cannot realise all four numeric truth ranks.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Iterable

from pipeline import io_utils


B1_SHORTCUT_AUDIT_SCHEMA = "egoconseq.b1-shortcut-audit.v1"
_B1_TASK_ID = "B1_endpoint_distance"


def _index_b1(rows: Iterable[dict], *, label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if row.get("task_id") != _B1_TASK_ID:
            continue
        item_id = str(row.get("id") or "")
        if not item_id or item_id in result:
            raise ValueError(f"{label} B1 ids are missing or duplicated")
        result[item_id] = row
    return result


def _feasible_numeric_ranks(displayed_m: float,
                            separation_m: float) -> list[int]:
    if (not math.isfinite(displayed_m) or displayed_m < 0.0 or
            not math.isfinite(separation_m) or separation_m <= 0.0):
        raise ValueError("B1 distance or choice separation is invalid")
    negative_steps = [
        step for step in range(1, 4)
        if displayed_m - step * separation_m >= 0.0
    ]
    return list(range(1, min(3, len(negative_steps)) + 2))


def _distance_stratum(displayed_m: float, separation_m: float) -> str:
    ratio = displayed_m / separation_m
    if ratio < 1.0:
        return "below_1x_separation"
    if ratio < 2.0:
        return "from_1x_to_2x_separation"
    if ratio < 3.0:
        return "from_2x_to_3x_separation"
    return "at_least_3x_separation"


def _counter(counter: collections.Counter) -> dict[str, int]:
    return dict(sorted(
        ((str(key), int(value)) for key, value in counter.items()),
        key=lambda item: item[0]))


def audit_b1_rows(items: Iterable[dict], answers: Iterable[dict]) -> dict:
    """Recompute empirical B1 rank baselines from the compiled artifact."""
    item_by_id = _index_b1(items, label="public item")
    answer_by_id = _index_b1(answers, label="private answer")
    if set(item_by_id) != set(answer_by_id):
        raise ValueError("B1 item/answer ids do not match")

    truth_ranks = collections.Counter()
    answer_positions = collections.Counter()
    support_counts = collections.Counter()
    support_rank_counts = collections.defaultdict(collections.Counter)
    strata = collections.defaultdict(lambda: {
        "item_count": 0,
        "truth_numeric_rank_counts": collections.Counter(),
        "feasible_rank_support_counts": collections.Counter(),
    })
    full_support_count = 0
    random_feasible_sum = 0.0
    separations = set()

    for item_id in sorted(item_by_id):
        item = item_by_id[item_id]
        answer_row = answer_by_id[item_id]
        answer = str(answer_row.get("canonical_answer") or "")
        choices = item.get("choices") or []
        if len(choices) != 4:
            raise ValueError(f"B1 must have four choices for {item_id}")
        try:
            values = [float(choice["value_m"]) for choice in choices]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"B1 choice values are invalid for {item_id}") from error
        if any(not math.isfinite(value) or value < 0.0 for value in values) or \
                len(set(values)) != 4:
            raise ValueError(f"B1 choice values are invalid for {item_id}")
        ordered = sorted(
            choices, key=lambda choice: (
                float(choice["value_m"]), str(choice.get("id") or "")))
        rank = next((
            index for index, choice in enumerate(ordered, 1)
            if str(choice.get("id") or "") == answer), None)
        position = next((
            index for index, choice in enumerate(choices, 1)
            if str(choice.get("id") or "") == answer), None)
        if rank is None or position is None:
            raise ValueError(f"B1 answer is absent from choices for {item_id}")

        certificate = answer_row.get("choice_certificate") or {}
        try:
            displayed = float(certificate["displayed_distance_m"])
            precise = float(certificate["precise_distance_m"])
            separation = float(certificate["minimum_separation_m"])
            declared_rank = int(certificate["truth_numeric_rank_1based"])
            declared_support = [
                int(value) for value in certificate[
                    "feasible_truth_numeric_ranks_1based"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"B1 choice certificate is incomplete for {item_id}") \
                from error
        if not math.isfinite(precise) or precise < 0.0:
            raise ValueError(f"B1 precise distance is invalid for {item_id}")
        feasible = _feasible_numeric_ranks(displayed, separation)
        if declared_support != feasible:
            raise ValueError(
                f"B1 feasible numeric ranks disagree for {item_id}")
        if declared_rank != rank or rank not in feasible:
            raise ValueError(f"B1 truth numeric rank disagrees for {item_id}")
        answer_choice = next(
            choice for choice in choices
            if str(choice.get("id") or "") == answer)
        if not math.isclose(
                float(answer_choice["value_m"]), displayed,
                abs_tol=1e-12, rel_tol=0.0):
            raise ValueError(f"B1 displayed truth disagrees for {item_id}")

        support_key = ",".join(str(value) for value in feasible)
        stratum = _distance_stratum(displayed, separation)
        truth_ranks[rank] += 1
        answer_positions[position] += 1
        support_counts[support_key] += 1
        support_rank_counts[support_key][rank] += 1
        strata[stratum]["item_count"] += 1
        strata[stratum]["truth_numeric_rank_counts"][rank] += 1
        strata[stratum]["feasible_rank_support_counts"][support_key] += 1
        full_support_count += int(feasible == [1, 2, 3, 4])
        random_feasible_sum += 1.0 / len(feasible)
        separations.add(separation)

    item_count = len(item_by_id)
    support_conditioned_correct = sum(
        max(counts.values()) for counts in support_rank_counts.values())
    ordered_strata = {}
    for name in (
            "below_1x_separation", "from_1x_to_2x_separation",
            "from_2x_to_3x_separation", "at_least_3x_separation"):
        row = strata[name]
        ordered_strata[name] = {
            "item_count": int(row["item_count"]),
            "truth_numeric_rank_counts": _counter(
                row["truth_numeric_rank_counts"]),
            "feasible_rank_support_counts": _counter(
                row["feasible_rank_support_counts"]),
        }
    return {
        "schema": B1_SHORTCUT_AUDIT_SCHEMA,
        "item_count": item_count,
        "minimum_separation_m_values": sorted(separations),
        "truth_numeric_rank_counts": _counter(truth_ranks),
        "answer_position_counts": _counter(answer_positions),
        "feasible_rank_support_counts": _counter(support_counts),
        "full_numeric_rank_support": {
            "item_count": full_support_count,
            "fraction": (
                full_support_count / item_count if item_count else None),
        },
        "distance_strata": ordered_strata,
        "blind_baselines": {
            "nominal_four_way_random_accuracy": (
                0.25 if item_count else None),
            "feasible_rank_uniform_random_accuracy": (
                random_feasible_sum / item_count if item_count else None),
            "best_constant_numeric_rank_accuracy": (
                max(truth_ranks.values()) / item_count
                if item_count else None),
            "best_middle_numeric_rank_accuracy": (
                max(truth_ranks.get(2, 0), truth_ranks.get(3, 0)) /
                item_count if item_count else None),
            "best_constant_answer_position_accuracy": (
                max(answer_positions.values()) / item_count
                if item_count else None),
            "best_feasible_support_conditioned_rank_accuracy": (
                support_conditioned_correct / item_count
                if item_count else None),
        },
    }


def audit_artifact(artifact_root: Path) -> dict:
    """Read one candidate artifact and bind the audit to its exact files."""
    root = Path(artifact_root)
    paths = {
        "public/items.jsonl": root / "public" / "items.jsonl",
        "private/answers.jsonl": root / "private" / "answers.jsonl",
        "private/atoms.jsonl": root / "private" / "atoms.jsonl",
    }
    report = audit_b1_rows(
        io_utils.read_jsonl(paths["public/items.jsonl"], require_dict=True),
        io_utils.read_jsonl(paths["private/answers.jsonl"], require_dict=True),
    )
    return {
        **report,
        "artifact_root": str(root),
        "source_sha256": {
            name: io_utils.sha256_file(path)
            for name, path in paths.items()
        },
    }


def write_artifact_audit(artifact_root: Path, output_path: Path) -> dict:
    """Write outside the benchmark tree so frozen artifact hashes cannot move."""
    root = Path(artifact_root).resolve()
    output = Path(output_path).resolve()
    if output == root or root in output.parents:
        raise ValueError("B1 shortcut audit output must be outside benchmark")
    report = audit_artifact(Path(artifact_root))
    io_utils.atomic_write_json(
        Path(output_path), report, allow_nan=False, sort_keys=True)
    return report
