"""Out-of-band scene-clustered audit for A2 policy shortcuts."""

from __future__ import annotations

import collections
from pathlib import Path
import random
from typing import Iterable

from pipeline import actions as action_geometry
from pipeline import io_utils


A2_SHORTCUT_AUDIT_SCHEMA = "egoconseq.a2-shortcut-audit.v1"
_A2_TASK_ID = "A2_collision_step_grounding"


def _index_a2(rows: Iterable[dict], *, label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if row.get("task_id") != _A2_TASK_ID:
            continue
        item_id = str(row.get("id") or "")
        if not item_id or item_id in result:
            raise ValueError(f"{label} ids are missing or duplicated")
        result[item_id] = row
    return result


def _best_group_accuracy(rows: list[dict], *, key: str, label: str) -> float:
    counts = collections.defaultdict(collections.Counter)
    for row in rows:
        counts[str(row[key])][int(row[label])] += 1
    return sum(max(values.values()) for values in counts.values()) / len(rows)


def _predictor_accuracies(rows: list[dict]) -> dict[str, float | None]:
    """Fit in-sample majority predictors on one scene-cluster resample.

    Canonical sequence pattern refines action count and the global constant
    partitions, so its in-sample accuracy cannot be lower than the coarser
    predictors. Familywise max reduction remains explicit to freeze the named
    predictor family without silently substituting one coordinate.
    """
    if not rows:
        return {
            "constant_forward_ordinal": None,
            "answer_position": None,
            "action_count": None,
            "canonical_sequence_pattern": None,
        }
    constant_rows = [{**row, "constant": "all"} for row in rows]
    return {
        "constant_forward_ordinal": _best_group_accuracy(
            constant_rows, key="constant", label="forward_ordinal"),
        "answer_position": _best_group_accuracy(
            constant_rows, key="constant", label="answer_position"),
        "action_count": _best_group_accuracy(
            rows, key="action_count", label="forward_ordinal"),
        "canonical_sequence_pattern": _best_group_accuracy(
            rows, key="canonical_sequence_pattern", label="forward_ordinal"),
    }


def _familywise_max(accuracies: dict[str, float | None]) -> float | None:
    """Return the maximum named-predictor accuracy for one resample."""
    values = [
        value for value in accuracies.values() if value is not None
    ]
    return max(values, default=None)


def audit_rows(
        items: Iterable[dict], answers: Iterable[dict], *,
        scene_by_id: dict[str, str], resamples: int = 2000,
        seed: int = 1729) -> dict:
    """Audit four public-input-only A2 predictors with scene uncertainty."""
    item_by_id = _index_a2(items, label="public A2 item")
    answer_by_id = _index_a2(answers, label="private A2 answer")
    if set(item_by_id) != set(answer_by_id):
        raise ValueError("A2 item/answer ids do not match")
    if set(scene_by_id) != set(item_by_id):
        raise ValueError("A2 scene bindings do not match task ids")
    rows = []
    for item_id in sorted(item_by_id):
        item = item_by_id[item_id]
        answer = str(answer_by_id[item_id].get("canonical_answer") or "")
        scene_id = str(scene_by_id[item_id] or "")
        if not scene_id:
            raise ValueError(f"A2 scene binding is missing for {item_id}")
        model_input = item.get("model_input")
        if not isinstance(model_input, dict):
            raise ValueError(f"A2 model input is invalid for {item_id}")
        try:
            actions = action_geometry.parse_actions(model_input["actions"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"A2 actions are invalid for {item_id}") from error
        forward_indexes = [
            index for index, action in enumerate(actions, 1)
            if isinstance(action, action_geometry.Forward)
        ]
        choices = item.get("choices")
        if not isinstance(choices, list):
            raise ValueError(f"A2 choices are invalid for {item_id}")
        choice_ids = [
            str(choice.get("id") or "")
            for choice in choices if isinstance(choice, dict)
        ]
        expected_ids = [f"action_{index}" for index in forward_indexes]
        if (len(forward_indexes) < 2 or len(choice_ids) != len(choices) or
                choice_ids != expected_ids):
            raise ValueError(
                f"A2 choices must be original Forward indexes for {item_id}")
        if answer not in choice_ids:
            raise ValueError(f"A2 answer is absent from choices for {item_id}")
        answer_index = int(answer.removeprefix("action_"))
        rows.append({
            "scene_id": scene_id,
            "forward_ordinal": forward_indexes.index(answer_index) + 1,
            "answer_position": choice_ids.index(answer) + 1,
            "action_count": len(actions),
            "canonical_sequence_pattern": ",".join(
                "forward" if isinstance(action, action_geometry.Forward)
                else "turn" for action in actions),
        })

    accuracies = _predictor_accuracies(rows)
    predictors = {
        name: {"item_count": len(rows), "accuracy": accuracy}
        for name, accuracy in accuracies.items()
    }
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["scene_id"]].append(row)
    point = _familywise_max(accuracies)
    upper = None
    if len(groups) >= 2 and rows and int(resamples) > 0:
        keys = sorted(groups)
        rng = random.Random(int(seed))
        maxima = []
        for _index in range(int(resamples)):
            sampled = [keys[rng.randrange(len(keys))] for _key in keys]
            bootstrap_rows = [
                row for scene_id in sampled for row in groups[scene_id]
            ]
            maximum = _familywise_max(
                _predictor_accuracies(bootstrap_rows))
            if maximum is None:
                raise ValueError("A2 bootstrap resample is empty")
            maxima.append(maximum)
        upper = io_utils.linear_quantile(maxima, 0.95)
    return {
        "schema": A2_SHORTCUT_AUDIT_SCHEMA,
        "item_count": len(rows),
        "scene_count": len(groups),
        "blind_predictors": predictors,
        "max_statistic_cluster_bootstrap": {
            "point_estimate": point,
            "resamples": int(resamples),
            "confidence_level": 0.95,
            "upper_confidence_bound": upper,
        },
    }


def audit_artifact(
        artifact_root: Path, *, resamples: int = 2000,
        seed: int = 1729, expected_source_authority=None) -> dict:
    """Read A2 rows and private scene bindings from one candidate artifact."""
    root = Path(artifact_root)
    # Authenticate manifests, source-map/source-atom joins, and every public
    # and private row before using private record contexts as cluster labels.
    # Keep this import at the artifact boundary so the in-memory audit remains
    # independent and candidate compilation never needs to import this module.
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
    items = io_utils.read_jsonl(
        paths["public/items.jsonl"], require_dict=True)
    answers = io_utils.read_jsonl(
        paths["private/answers.jsonl"], require_dict=True)
    atoms = io_utils.read_jsonl(
        paths["private/atoms.jsonl"], require_dict=True)
    contexts = io_utils.read_jsonl(
        paths["private/record_contexts.jsonl"], require_dict=True)
    atom_by_id = {}
    for atom in atoms:
        atom_id = str(atom.get("id") or "")
        if not atom_id or atom_id in atom_by_id:
            raise ValueError("A2 atom ids are missing or duplicated")
        atom_by_id[atom_id] = atom
    context_by_digest = {}
    for row in contexts:
        digest = str(row.get("record_sha256") or "")
        context = row.get("context")
        if (not digest or digest in context_by_digest or
                not isinstance(context, dict)):
            raise ValueError("A2 record contexts are malformed")
        context_by_digest[digest] = context
    a2_answers = _index_a2(answers, label="private A2 answer")
    scene_by_id = {}
    for item_id, answer in a2_answers.items():
        atom = atom_by_id.get(str(answer.get("atom_ref") or ""))
        if atom is None:
            raise ValueError(f"A2 atom binding is missing for {item_id}")
        context = context_by_digest.get(str(atom.get("record_sha256") or ""))
        if context is None:
            raise ValueError(f"A2 record context is missing for {item_id}")
        source = context.get("source") or {}
        scene_id = context.get("scene_id") or source.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"A2 scene binding is missing for {item_id}")
        scene_by_id[item_id] = scene_id
    report = audit_rows(
        items, answers, scene_by_id=scene_by_id,
        resamples=resamples, seed=seed)
    return {
        **report,
        "artifact_root": str(root),
        "source_sha256": {
            name: io_utils.sha256_file(path)
            for name, path in paths.items()
        },
    }


def write_artifact_audit(
        artifact_root: Path, output_path: Path, *,
        resamples: int = 2000, seed: int = 1729,
        expected_source_authority=None) -> dict:
    """Write a deterministic A2 audit outside the benchmark tree."""
    root = Path(artifact_root).resolve()
    output = Path(output_path).resolve()
    if output == root or root in output.parents:
        raise ValueError("A2 shortcut audit output must be outside benchmark")
    report = audit_artifact(
        Path(artifact_root), resamples=resamples, seed=seed,
        expected_source_authority=expected_source_authority)
    io_utils.atomic_write_json(
        Path(output_path), report, allow_nan=False, sort_keys=True)
    return report
