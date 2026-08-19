"""Deterministic frame-first publication selection for candidate QA."""

from __future__ import annotations

import collections
import copy
import hashlib
from typing import Mapping

from pipeline import a1_common_support
from pipeline import action_proposal
from pipeline import config


POLICY = "diversity-selection.v1"
MAX_ITEMS_PER_FRAME = 8
TASK_CAPS = {
    "A1_collision": 2,
    "A2_collision_step_grounding": 3,
    "A3_contact_object": 3,
    "B1_endpoint_distance": 3,
    "B2_endpoint_direction": 4,
    "C1_future_view_selection": 2,
}
HEAD_TASKS = {
    "A": ("A1_collision", "A2_collision_step_grounding",
          "A3_contact_object"),
    "B": ("B1_endpoint_distance", "B2_endpoint_direction"),
    "C": ("C1_future_view_selection",),
}


def _rotated(values: tuple[str, ...], seed: str) -> tuple[str, ...]:
    if not values:
        return values
    start = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % len(values)
    return values[start:] + values[:start]


def _stable_key(frame_sha256: str, item: Mapping) -> str:
    return hashlib.sha256(
        f"{frame_sha256}\0{item.get('task_id')}\0{item.get('id')}".encode()
    ).hexdigest()


def _b1_distance_bin(answer: Mapping) -> str:
    distance = float(answer["precise_distance_m"])
    near_max, mid_max = config.DIVERSITY_B1_DISTANCE_BINS_M
    if distance < near_max:
        return "near"
    if distance < mid_max:
        return "mid"
    return "far"


def semantic_answer(item: Mapping, answer: Mapping) -> str:
    """Return the answer identity used only for within-frame deduplication."""
    if item.get("task_id") == "B1_endpoint_distance":
        return _b1_distance_bin(answer)
    return str(answer.get("canonical_answer"))


def _head_queue(
        head: str, task_rows: Mapping[str, list[dict]], *,
        frame_sha256: str) -> list[dict]:
    tasks = _rotated(
        tuple(HEAD_TASKS[head]), f"{frame_sha256}:tasks:{head}")
    queues = {task: list(task_rows.get(task) or []) for task in tasks}
    result = []
    if head == "A" and len(queues.get("A1_collision", ())) >= 2:
        result.extend(queues["A1_collision"][:2])
        queues["A1_collision"] = queues["A1_collision"][2:]
    depth = 0
    while any(len(queues[task]) > depth for task in tasks):
        for task in tasks:
            if len(queues[task]) > depth:
                result.append(queues[task][depth])
        depth += 1
    return result


def _select_frame(
        rows: list[dict], *, frame_sha256: str,
        allowed_a1_ids: set[str] | None = None) -> list[dict]:
    by_task_answer = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for row in rows:
        task_id = str(row["item"]["task_id"])
        if (task_id == "A1_collision" and allowed_a1_ids is not None and
                str(row["item"]["id"]) not in allowed_a1_ids):
            continue
        key = semantic_answer(row["item"], row["answer"])
        by_task_answer[task_id][key].append(row)

    task_rows = {}
    for task_id, answer_groups in by_task_answer.items():
        unique = [
            min(group, key=lambda row: _stable_key(
                frame_sha256, row["item"]))
            for group in answer_groups.values()
        ]
        unique.sort(key=lambda row: _stable_key(frame_sha256, row["item"]))
        task_rows[task_id] = unique[:TASK_CAPS[task_id]]

    head_rows = {
        head: _head_queue(head, task_rows, frame_sha256=frame_sha256)
        for head in ("A", "B", "C")
    }
    heads = _rotated(("A", "B", "C"), f"{frame_sha256}:heads")
    selected = []
    depth = 0
    while len(selected) < MAX_ITEMS_PER_FRAME and any(
            len(head_rows[head]) > depth for head in heads):
        for head in heads:
            if (len(selected) < MAX_ITEMS_PER_FRAME and
                    len(head_rows[head]) > depth):
                selected.append(head_rows[head][depth])
        depth += 1
    return selected


def _v3_a1_cell(
        row: dict, *, answer_by_id: Mapping, atom_by_id: Mapping,
        contexts: Mapping) -> tuple | None:
    item = row["item"]
    protocol, variant = a1_common_support.proposal_source_for_item(
        item, answer_by_id, atom_by_id, contexts)
    if protocol not in a1_common_support.V3_PUBLICATION_PROTOCOLS:
        return None
    dataset = a1_common_support.source_dataset_for_item(
        item, answer_by_id, atom_by_id, contexts)
    if variant == action_proposal.NATURAL_DYNAMIC_VARIANT:
        return ("natural", dataset, *a1_common_support.natural_v3_cell_key(
            item))
    if variant == action_proposal.A1_CONTROL_VARIANT:
        return ("control", dataset, *a1_common_support.control_v3_cell_key(
            item))
    return None


def _v3_balanced_a1_ids(
        projection: Mapping, by_frame: Mapping) -> set[str]:
    answers = list(projection.get("answers") or [])
    atoms = list(projection.get("atoms") or [])
    contexts_rows = list(projection.get("record_contexts") or [])
    answer_by_id = {str(row["id"]): row for row in answers}
    atom_by_id = {str(row["id"]): row for row in atoms}
    contexts = {
        str(row["record_sha256"]): row["context"]
        for row in contexts_rows}
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    frame_for_id = {}
    for frame_key, rows in by_frame.items():
        for row in rows:
            item = row["item"]
            if item.get("task_id") != "A1_collision":
                continue
            cell = _v3_a1_cell(
                row, answer_by_id=answer_by_id,
                atom_by_id=atom_by_id, contexts=contexts)
            if cell is None:
                continue
            item_id = str(item["id"])
            label = str(row["answer"]["canonical_answer"])
            cells[cell][label].append(row)
            frame_for_id[item_id] = frame_key

    used_labels = collections.defaultdict(set)
    keep_ids = set()

    def fits(row: dict, label: str) -> bool:
        frame = frame_for_id[str(row["item"]["id"])]
        return label not in used_labels[frame] and len(used_labels[frame]) < 2

    ordered_cells = sorted(
        cells,
        key=lambda cell: (
            -min(len(cells[cell].get("collision", ())),
                 len(cells[cell].get("no_collision", ()))),
            repr(cell)))
    for cell in ordered_cells:
        collision = sorted(
            cells[cell].get("collision", ()),
            key=lambda row: _stable_key(
                frame_for_id[str(row["item"]["id"])][1], row["item"]))
        clear = sorted(
            cells[cell].get("no_collision", ()),
            key=lambda row: _stable_key(
                frame_for_id[str(row["item"]["id"])][1], row["item"]))
        while True:
            pair = None
            for collision_row in collision:
                if not fits(collision_row, "collision"):
                    continue
                collision_frame = frame_for_id[
                    str(collision_row["item"]["id"])]
                used_labels[collision_frame].add("collision")
                clear_row = next(
                    (row for row in clear if fits(row, "no_collision")),
                    None)
                used_labels[collision_frame].remove("collision")
                if clear_row is not None:
                    pair = collision_row, clear_row
                    break
            if pair is None:
                break
            for row, label in zip(pair, ("collision", "no_collision")):
                item_id = str(row["item"]["id"])
                keep_ids.add(item_id)
                used_labels[frame_for_id[item_id]].add(label)
                if label == "collision":
                    collision.remove(row)
                else:
                    clear.remove(row)
    return keep_ids


def _rows_by_frame(projection: Mapping) -> dict[tuple[str, str], list[dict]]:
    items = list(projection.get("items") or [])
    answers = list(projection.get("answers") or [])
    answer_by_id = {str(row.get("id")): row for row in answers}
    atom_by_id = {
        str(row.get("id")): row for row in projection.get("atoms") or []}
    context_by_digest = {
        str(row.get("record_sha256")): row.get("context")
        for row in projection.get("record_contexts") or []}
    if (len(answer_by_id) != len(answers) or
            set(answer_by_id) != {str(item.get("id")) for item in items}):
        raise ValueError("diversity selection requires bijective QA rows")
    by_frame = collections.defaultdict(list)
    for item in items:
        item_id = str(item["id"])
        answer = answer_by_id[item_id]
        atom = atom_by_id.get(str(answer.get("atom_ref")))
        if atom is None:
            raise ValueError("diversity selection source atom is missing")
        context = context_by_digest.get(str(atom.get("record_sha256")))
        if not isinstance(context, dict):
            raise ValueError("diversity selection record context is missing")
        source = context.get("source") or {}
        dataset = str(source.get("source_dataset") or "")
        input_asset = answer.get("input_asset") or {}
        frame_sha256 = str(
            input_asset.get("raw_sha256") or input_asset.get("sha256") or "")
        if not dataset or not frame_sha256:
            raise ValueError("diversity selection frame identity is missing")
        by_frame[(dataset, frame_sha256)].append({
            "item": item, "answer": answer})
    return dict(by_frame)


def summarize_selection(projection: Mapping) -> dict:
    """Summarize the final selected rows without reopening image assets."""
    by_frame = _rows_by_frame(projection)
    task_counts = collections.Counter()
    semantic_counts = collections.defaultdict(collections.Counter)
    dataset_frames = collections.Counter()
    a1_labels = collections.defaultdict(set)
    item_counts = []
    for (dataset, frame_sha256), rows in by_frame.items():
        dataset_frames[dataset] += 1
        item_counts.append(len(rows))
        for row in rows:
            item = row["item"]
            answer = row["answer"]
            task_id = str(item["task_id"])
            value = semantic_answer(item, answer)
            task_counts[task_id] += 1
            semantic_counts[task_id][value] += 1
            if task_id == "A1_collision":
                a1_labels[(dataset, frame_sha256)].add(value)
    ordered_counts = sorted(item_counts)

    def percentile(fraction: float) -> int | None:
        if not ordered_counts:
            return None
        index = round((len(ordered_counts) - 1) * fraction)
        return int(ordered_counts[index])

    a1_eligible = len(a1_labels)
    a1_contrast = sum(len(values) >= 2 for values in a1_labels.values())
    return {
        "unique_source_frames": len(by_frame),
        "unique_source_frames_by_dataset": dict(sorted(dataset_frames.items())),
        "retained_items": sum(item_counts),
        "items_per_frame": {
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "max": max(item_counts) if item_counts else None,
        },
        "task_counts": dict(sorted(task_counts.items())),
        "semantic_answer_counts": {
            task_id: dict(sorted(counts.items()))
            for task_id, counts in sorted(semantic_counts.items())
        },
        "a1_contrast_frames": a1_contrast,
        "a1_eligible_frames": a1_eligible,
        "a1_contrast_frame_rate": (
            a1_contrast / a1_eligible if a1_eligible else None),
    }


def apply_selection(
        projection: dict, *, balance_a1_v3: bool = False) -> dict:
    """Keep diverse semantic answers under per-frame and per-task caps."""
    selected = copy.deepcopy(projection)
    items = list(selected.get("items") or [])
    answers = list(selected.get("answers") or [])
    by_frame = _rows_by_frame(selected)
    allowed_a1_ids = (
        _v3_balanced_a1_ids(selected, by_frame)
        if balance_a1_v3 else None)

    retained_rows = []
    for (_dataset, frame_sha256), rows in sorted(by_frame.items()):
        retained_rows.extend(_select_frame(
            rows, frame_sha256=frame_sha256,
            allowed_a1_ids=allowed_a1_ids))
    retained_ids = {str(row["item"]["id"]) for row in retained_rows}
    selected["items"] = [
        item for item in items if str(item["id"]) in retained_ids]
    selected["answers"] = [
        answer for answer in answers if str(answer["id"]) in retained_ids]

    used_atoms = {str(answer["atom_ref"]) for answer in selected["answers"]}
    selected["atoms"] = [
        atom for atom in selected.get("atoms") or []
        if str(atom.get("id")) in used_atoms]
    used_records = {
        str(atom["record_sha256"]) for atom in selected["atoms"]}
    selected["record_contexts"] = [
        row for row in selected.get("record_contexts") or []
        if str(row.get("record_sha256")) in used_records]

    summary = summarize_selection(selected)
    input_task_counts = collections.Counter(
        str(item["task_id"]) for item in items)
    dropped_by_task = {
        task_id: int(count - summary["task_counts"].get(task_id, 0))
        for task_id, count in sorted(input_task_counts.items())
    }
    selected["diversity_selection"] = {
        "policy": POLICY,
        "source_frame_identity": "source_dataset+raw_rgb_sha256",
        "input_items": len(items),
        "dropped_items": len(items) - len(selected["items"]),
        "input_task_counts": dict(sorted(input_task_counts.items())),
        "dropped_by_task": dropped_by_task,
        "before_a1_common_support": copy.deepcopy(summary),
        **summary,
    }
    return selected


def refresh_report(projection: dict) -> dict:
    """Refresh final counts after downstream A1 common-support selection."""
    report = dict(projection.get("diversity_selection") or {})
    if report.get("policy") != POLICY:
        raise ValueError("diversity selection policy is missing")
    report.update(summarize_selection(projection))
    projection["diversity_selection"] = report
    return projection
