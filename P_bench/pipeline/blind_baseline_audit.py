"""Out-of-band blind baselines for A1 (collision) and B2 (endpoint direction).

Both audits read an already compiled candidate artifact and predict its answers
from the *public* item alone -- never the image, never a private field.  They
change no Record/Oracle fact, no Task-GT, no eligibility, no choice and no
score.  A high blind baseline here is a number to report, not a licence to move
any gate.

Every grouped predictor is reported twice.  ``in_sample_accuracy`` fits the
majority label on the same rows it scores, so it inflates with ``group_count``
and is an upper bound, not an estimate.  ``leave_one_out_accuracy`` refits
without the scored row (backing off to the global majority for singleton
groups) and is the honest number; both are deterministic, so this module needs
no resampling and no RNG.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Callable, Iterable

from pipeline import (
    a1_common_support, actions as action_geometry,
)
from pipeline import io_utils


A1_BLIND_AUDIT_SCHEMA = "egoconseq.a1-blind-audit.v1"
B2_ROTATION_AUDIT_SCHEMA = "egoconseq.b2-rotation-audit.v1"
BLIND_BASELINE_AUDIT_SCHEMA = "egoconseq.blind-baseline-audit.v1"

_A1_TASK_ID = "A1_collision"
_B2_TASK_ID = "B2_endpoint_direction"
_A1_CHOICE_IDS = frozenset({"collision", "no_collision"})
_B2_CHOICE_IDS = ("front", "left", "right", "rear")

# Grouped predictors shared by both tasks.  Keys are blind feature names
# produced by ``_program_features``; the label is the canonical answer.
_SHARED_PREDICTORS = (
    "constant",
    "action_count",
    "sequence_pattern",
    "forward_leg_count",
    "forward_total_m",
    "turn_count",
    "net_turn_deg",
    "body_radius_m",
)


def _index(rows: Iterable[dict], task_id: str, *, label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        if row.get("task_id") != task_id:
            continue
        item_id = str(row.get("id") or "")
        if not item_id or item_id in result:
            raise ValueError(f"{label} ids are missing or duplicated")
        result[item_id] = row
    return result


def _counter(counter: collections.Counter) -> dict[str, int]:
    return dict(sorted(
        ((str(key), int(value)) for key, value in counter.items()),
        key=lambda item: item[0]))


def _majority_label(counter: collections.Counter) -> str:
    """Most frequent label, ties broken by the smallest label string."""
    return min(counter.items(), key=lambda item: (-item[1], str(item[0])))[0]


def _program_features(model_input: dict, *, item_id: str) -> dict[str, str]:
    """Blind features derived from the public item -- no image, no GT."""
    try:
        actions = action_geometry.parse_actions(model_input["actions"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"item actions are invalid for {item_id}") from error
    if not actions:
        raise ValueError(f"item action program is empty for {item_id}")
    forwards = [
        action for action in actions
        if isinstance(action, action_geometry.Forward)]
    try:
        radius = float(model_input["body_radius_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"item body radius is invalid for {item_id}") \
            from error
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"item body radius is invalid for {item_id}")
    net_turn = action_geometry.net_turn_deg(actions)
    return {
        "constant": "all",
        "action_count": str(len(actions)),
        "sequence_pattern": ",".join(
            "forward" if isinstance(action, action_geometry.Forward)
            else "turn" for action in actions),
        "forward_leg_count": str(len(forwards)),
        "forward_total_m": f"{action_geometry.total_forward_m(actions):.2f}",
        "turn_count": str(len(actions) - len(forwards)),
        "net_turn_deg": f"{net_turn:.2f}",
        "body_radius_m": f"{radius:.3f}",
        "net_turn_sector": action_geometry.bearing_sector(
            action_geometry.wrap_deg(-net_turn)),
    }


def _answer_position(item: dict, answer: str, *, item_id: str) -> int:
    choices = item.get("choices") or []
    position = next((
        index for index, choice in enumerate(choices, 1)
        if str(choice.get("id") or "") == answer), None)
    if position is None:
        raise ValueError(f"answer is absent from choices for {item_id}")
    return position


def _in_sample_accuracy(rows: list[dict], key_of: Callable[[dict], str],
                        label_of: Callable[[dict], str]) -> float:
    groups: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for row in rows:
        groups[key_of(row)][label_of(row)] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(rows)


def _leave_one_out_accuracy(rows: list[dict], key_of: Callable[[dict], str],
                            label_of: Callable[[dict], str]) -> float:
    groups: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    overall: collections.Counter = collections.Counter()
    for row in rows:
        groups[key_of(row)][label_of(row)] += 1
        overall[label_of(row)] += 1
    correct = 0
    for row in rows:
        truth = label_of(row)
        held_out = collections.Counter(groups[key_of(row)])
        held_out[truth] -= 1
        held_out += collections.Counter()  # drop non-positive entries
        if not held_out:  # singleton group: back off to the rest of the corpus
            held_out = collections.Counter(overall)
            held_out[truth] -= 1
            held_out += collections.Counter()
        if not held_out:  # a corpus of one row predicts nothing
            continue
        correct += int(_majority_label(held_out) == truth)
    return correct / len(rows)


def _predictor_report(rows: list[dict], names: Iterable[str],
                      label_key: str = "label") -> dict[str, dict]:
    report: dict[str, dict] = {}
    for name in names:
        def key_of(row, name=name):
            return row["features"][name]

        def label_of(row):
            return str(row[label_key])

        report[name] = {
            "group_count": len({key_of(row) for row in rows}),
            "in_sample_accuracy": _in_sample_accuracy(rows, key_of, label_of),
            "leave_one_out_accuracy": _leave_one_out_accuracy(
                rows, key_of, label_of),
        }
    return report


def _familywise(report: dict[str, dict], field: str) -> float | None:
    values = [entry[field] for entry in report.values()]
    return max(values) if values else None


def _empty_report(schema: str) -> dict:
    return {
        "schema": schema,
        "item_count": 0,
        "canonical_answer_counts": {},
        "answer_position_counts": {},
        "blind_baselines": {
            "nominal_uniform_random_accuracy": None,
            "predictors": {},
            "familywise_max_in_sample_accuracy": None,
            "familywise_max_leave_one_out_accuracy": None,
        },
    }


def _collect_rows(items: Iterable[dict], answers: Iterable[dict], task_id: str,
                  *, choice_check: Callable[[list, str], None]) -> list[dict]:
    item_by_id = _index(items, task_id, label=f"public {task_id} item")
    answer_by_id = _index(answers, task_id, label=f"private {task_id} answer")
    if set(item_by_id) != set(answer_by_id):
        raise ValueError(f"{task_id} item/answer ids do not match")
    rows = []
    for item_id in sorted(item_by_id):
        item = item_by_id[item_id]
        answer_row = answer_by_id[item_id]
        answer = str(answer_row.get("canonical_answer") or "")
        choices = item.get("choices") or []
        choice_check(choices, item_id)
        model_input = item.get("model_input")
        if not isinstance(model_input, dict):
            raise ValueError(f"item model input is invalid for {item_id}")
        rows.append({
            "id": item_id,
            "label": answer,
            "position": _answer_position(item, answer, item_id=item_id),
            "features": _program_features(model_input, item_id=item_id),
            "item": item,
            "answer": answer_row,
        })
    return rows


def _baselines(rows: list[dict], predictors: Iterable[str],
               choice_count: int) -> dict:
    report = _predictor_report(rows, predictors)
    report["answer_position"] = {
        "group_count": 1,
        "in_sample_accuracy": _in_sample_accuracy(
            rows, lambda row: "all", lambda row: str(row["position"])),
        "leave_one_out_accuracy": _leave_one_out_accuracy(
            rows, lambda row: "all", lambda row: str(row["position"])),
    }
    return {
        "nominal_uniform_random_accuracy": 1.0 / choice_count,
        "predictors": dict(sorted(report.items())),
        "familywise_max_in_sample_accuracy": _familywise(
            report, "in_sample_accuracy"),
        "familywise_max_leave_one_out_accuracy": _familywise(
            report, "leave_one_out_accuracy"),
    }


def _check_a1_choices(choices: list, item_id: str) -> None:
    ids = [str(choice.get("id") or "") for choice in choices]
    if len(ids) != len(set(ids)) or set(ids) != _A1_CHOICE_IDS:
        raise ValueError(f"A1 choices are not the frozen pair for {item_id}")


def _check_b2_choices(choices: list, item_id: str) -> None:
    ids = [str(choice.get("id") or "") for choice in choices]
    if len(ids) != len(set(ids)) or set(ids) != set(_B2_CHOICE_IDS):
        raise ValueError(f"B2 choices are not the four sectors for {item_id}")


def audit_a1_rows(items: Iterable[dict], answers: Iterable[dict]) -> dict:
    """Score public-input-only predictors of the A1 collision answer."""
    rows = _collect_rows(
        items, answers, _A1_TASK_ID, choice_check=_check_a1_choices)
    if not rows:
        return _empty_report(A1_BLIND_AUDIT_SCHEMA)
    return {
        "schema": A1_BLIND_AUDIT_SCHEMA,
        "item_count": len(rows),
        "canonical_answer_counts": _counter(
            collections.Counter(row["label"] for row in rows)),
        "answer_position_counts": _counter(
            collections.Counter(row["position"] for row in rows)),
        "blind_baselines": _baselines(
            rows, _SHARED_PREDICTORS, len(_A1_CHOICE_IDS)),
    }


def audit_a1_by_source(
        items: Iterable[dict], answers: Iterable[dict],
        atoms: Iterable[dict], record_contexts: Iterable[dict]) -> dict:
    """Run the same blind audit separately for each authenticated A1 arm."""
    items = list(items)
    answers = list(answers)
    answer_by_id = _index(
        answers, _A1_TASK_ID, label="private A1 source answer")
    atoms_by_id = {
        str(row.get("id") or ""): row for row in atoms}
    contexts = {
        str(row.get("record_sha256") or ""): row.get("context")
        for row in record_contexts
    }
    if (not atoms_by_id or "" in atoms_by_id or not contexts or
            "" in contexts):
        raise ValueError("A1 source audit tables are invalid")
    grouped_items = collections.defaultdict(list)
    grouped_answers = collections.defaultdict(list)
    for item in items:
        if item.get("task_id") != _A1_TASK_ID:
            continue
        item_id = str(item.get("id") or "")
        answer = answer_by_id.get(item_id)
        if answer is None:
            raise ValueError("A1 source audit item/answer ids do not match")
        protocol, variant = a1_common_support.proposal_source_for_item(
            item, answer_by_id, atoms_by_id, contexts)
        source = (variant if protocol in
                  a1_common_support.V3_PUBLICATION_PROTOCOLS
                  else f"legacy:{protocol}:{variant}")
        grouped_items[source].append(item)
        grouped_answers[source].append(answer)
    return {
        source: audit_a1_rows(
            grouped_items[source], grouped_answers[source])
        for source in sorted(grouped_items)
    }


def _rotation_only_rule(rows: list[dict]) -> dict:
    """Parameter-free rule: rotate a dead-ahead target by the net turn.

    ``_fold`` integrates a positive ``Turn`` as an increase in heading, so a
    target that starts on the optical axis ends at bearing ``-net_turn`` in the
    final egocentric frame.  Nothing here is fitted: if this rule already beats
    chance, the net turn alone carries the answer.
    """
    confusion: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    correct = 0
    for row in rows:
        predicted = row["features"]["net_turn_sector"]
        confusion[predicted][row["label"]] += 1
        correct += int(predicted == row["label"])
    return {
        "parameter_free": True,
        "accuracy": correct / len(rows),
        "predicted_sector_counts": _counter(collections.Counter(
            {sector: sum(counts.values())
             for sector, counts in confusion.items()})),
        "confusion": {
            sector: _counter(counts)
            for sector, counts in sorted(confusion.items())},
    }


def _precise_bearing_crosscheck(rows: list[dict]) -> dict:
    """Re-derive each sector label from the private bearing it claims."""
    checked = 0
    missing = 0
    for row in rows:
        bearing = row["answer"].get("precise_bearing_deg")
        if bearing is None:
            missing += 1
            continue
        try:
            value = float(bearing)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"B2 precise bearing is invalid for {row['id']}") from error
        if not math.isfinite(value):
            raise ValueError(f"B2 precise bearing is invalid for {row['id']}")
        if action_geometry.bearing_sector(value) != row["label"]:
            raise ValueError(
                f"B2 precise bearing disagrees with the answer for {row['id']}")
        checked += 1
    return {
        "checked_item_count": checked,
        "missing_bearing_item_count": missing,
        "disagreement_count": 0,
    }


def audit_b2_rows(items: Iterable[dict], answers: Iterable[dict]) -> dict:
    """Score rotation-only and other public-input-only predictors of B2."""
    rows = _collect_rows(
        items, answers, _B2_TASK_ID, choice_check=_check_b2_choices)
    if not rows:
        report = _empty_report(B2_ROTATION_AUDIT_SCHEMA)
        report["rotation_only_frontal_prior"] = None
        report["precise_bearing_crosscheck"] = {
            "checked_item_count": 0,
            "missing_bearing_item_count": 0,
            "disagreement_count": 0,
        }
        report["target_label_counts"] = {}
        return report
    for row in rows:
        target = row["item"]["model_input"].get("target")
        if not isinstance(target, str) or not target:
            raise ValueError(f"B2 target is invalid for {row['id']}")
        row["features"]["target"] = target
    predictors = _SHARED_PREDICTORS + ("net_turn_sector", "target")
    return {
        "schema": B2_ROTATION_AUDIT_SCHEMA,
        "item_count": len(rows),
        "canonical_answer_counts": _counter(
            collections.Counter(row["label"] for row in rows)),
        "answer_position_counts": _counter(
            collections.Counter(row["position"] for row in rows)),
        "target_label_counts": _counter(
            collections.Counter(row["features"]["target"] for row in rows)),
        "rotation_only_frontal_prior": _rotation_only_rule(rows),
        "precise_bearing_crosscheck": _precise_bearing_crosscheck(rows),
        "blind_baselines": _baselines(
            rows, predictors, len(_B2_CHOICE_IDS)),
    }


def audit_artifact(
        artifact_root: Path, *, expected_source_authority=None) -> dict:
    """Read one candidate artifact and bind both audits to its exact files."""
    root = Path(artifact_root)
    from pipeline import candidate_preview
    candidate_preview.validate_preview_artifact(
        root, expected_source_authority=expected_source_authority)
    paths = {
        "public/items.jsonl": root / "public" / "items.jsonl",
        "private/answers.jsonl": root / "private" / "answers.jsonl",
    }
    items = io_utils.read_jsonl(paths["public/items.jsonl"], require_dict=True)
    answers = io_utils.read_jsonl(
        paths["private/answers.jsonl"], require_dict=True)
    source_paths = {
        "private/atoms.jsonl": root / "private" / "atoms.jsonl",
        "private/record_contexts.jsonl":
            root / "private" / "record_contexts.jsonl",
    }
    atoms = io_utils.read_jsonl(
        source_paths["private/atoms.jsonl"], require_dict=True)
    contexts = io_utils.read_jsonl(
        source_paths["private/record_contexts.jsonl"], require_dict=True)
    source_map = json.loads(
        (root / "private" / "source_map.json").read_text(encoding="utf-8"))
    by_source = (
        audit_a1_by_source(items, answers, atoms, contexts)
        if source_map.get("publication_selection") == {
            "A1_collision": a1_common_support.V3_POLICY}
        else {})
    paths.update(source_paths)
    return {
        "schema": BLIND_BASELINE_AUDIT_SCHEMA,
        "artifact_root": str(root),
        "a1_blind": audit_a1_rows(items, answers),
        "a1_blind_by_source": by_source,
        "b2_rotation": audit_b2_rows(items, answers),
        "source_sha256": {
            name: io_utils.sha256_file(path)
            for name, path in paths.items()
        },
    }


def write_artifact_audit(
        artifact_root: Path, output_path: Path, *,
        expected_source_authority=None) -> dict:
    """Write outside the benchmark tree so frozen artifact hashes cannot move."""
    root = Path(artifact_root).resolve()
    output = Path(output_path).resolve()
    if output == root or root in output.parents:
        raise ValueError("blind baseline audit output must be outside benchmark")
    report = audit_artifact(
        Path(artifact_root),
        expected_source_authority=expected_source_authority)
    io_utils.atomic_write_json(
        Path(output_path), report, allow_nan=False, sort_keys=True)
    return report
