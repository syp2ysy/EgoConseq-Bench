"""Record-exclusive matching shared by supply reports and benchmark selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from dataclasses import replace
import heapq
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping

from pipeline import a2
from pipeline import benchmark_candidates, candidate_sources
from pipeline.benchmark_candidates import Candidate
from post_QA import templates
from post_QA.seen_build import catalog, spec


class SelectionShortfall(RuntimeError):
    pass


@lru_cache(maxsize=None)
def _a2_slot_counts(dataset: str) -> Counter:
    counts = Counter()
    for length, starts in spec.A2_START_TOTALS[dataset].items():
        for start, total in starts.items():
            for cell in a2.balanced_cells(
                    length, total, starts_with=start,
                    rotation=spec.DATASETS.index(dataset) + length):
                counts[(dataset, "A2", length, start,
                        cell.forward_ordinal_1based,
                        cell.distance_rank)] += 1
    return counts


def _a1_slot_counts(total: int, turn_fraction: float) -> dict[tuple, int]:
    turns = int(math.ceil(int(total) * float(turn_fraction) - 1e-12))
    result = {}
    for start, count in (("turn", turns), ("forward", int(total) - turns)):
        no_collision = (count + (start == "turn")) // 2
        result[start, "no_collision"] = no_collision
        result[start, "collision"] = count - no_collision
    return result


def candidate_slot(row: Candidate, *, include_length=False) -> tuple:
    if row.task_id == "A2":
        return (row.dataset, row.task_id, row.action_length,
                row.starts_with, row.a2_ordinal, row.a2_rank)
    if row.task_id == "C1":
        return (row.dataset, row.task_id, row.action_length, row.starts_with)
    if row.task_id == "A1":
        slot = (row.dataset, row.task_id, row.starts_with, row.answer_bucket)
    else:
        slot = row.dataset, row.task_id, row.starts_with
    return slot + (row.action_length,) if include_length else slot


def _slot_quota(slot: tuple) -> int:
    dataset, task = slot[:2]
    if task == "A2":
        return _a2_slot_counts(dataset)[slot]
    if task == "C1":
        return spec.C1_LENGTH_TOTALS[dataset][int(slot[2]) - 1]
    if task == "A1":
        return _a1_slot_counts(
            spec.DATASET_TASK_TOTALS[dataset][task],
            spec.MINIMUM_TURN_FIRST_FRACTION)[slot[2:]]
    total = spec.DATASET_TASK_TOTALS[dataset][task]
    turns = int(math.ceil(
        total * spec.MINIMUM_TURN_FIRST_FRACTION - 1e-12))
    return turns if slot[2] == "turn" else total - turns


def enumerate_candidates(records_root: Path, *, seed: int,
                         progress=None, pool_factor=10, preserve_lengths=False,
                         task_ids=None) -> list[Candidate]:
    """Scan records once and retain a bounded deterministic pool per slot."""
    root = Path(records_root).resolve()
    heaps = defaultdict(list)
    serial = 0
    scanned = 0
    for source in catalog.load(root):
        dataset = source.dataset
        path = source.records_path
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                payload = line.rstrip(b"\r\n")
                if not payload:
                    continue
                record = json.loads(payload)
                record_sha256 = candidate_sources.canonical_sha256(record)
                record = _selection_facets(record, seed, preserve_lengths)
                projected = benchmark_candidates.project_record(
                    dataset=dataset, source_path=str(path),
                    byte_offset=offset, record_sha256=record_sha256,
                    record=record, allowed_tasks=[task for task in spec.supported_tasks(dataset)
                                                  if task_ids is None or task in task_ids])
                best = {}
                for row in projected:
                    slot = candidate_slot(row, include_length=preserve_lengths)
                    if row.task_id in ("A4", "B2"):
                        slot += (row.answer_bucket,)
                    if preserve_lengths and row.task_id == "C1":
                        slot += (row.outcome_id,)
                    key = _stable(seed, row)
                    if slot not in best or key < best[slot][0]:
                        best[slot] = key, row
                for slot, (key, row) in best.items():
                    quota = _slot_quota(candidate_slot(row))
                    if quota <= 0:
                        continue
                    divisor = 6 if preserve_lengths and row.task_id not in ("A2", "C1") else 1
                    capacity = max(100, quota * pool_factor // divisor)
                    entry = (-int(key, 16), serial, row)
                    serial += 1
                    pool_slot = candidate_slot(row, include_length=preserve_lengths)
                    if row.task_id in ("A4", "B2"):
                        pool_slot += (row.answer_bucket,)
                    heap = heaps[pool_slot]
                    if len(heap) < capacity:
                        heapq.heappush(heap, entry)
                    elif entry[0] > heap[0][0]:
                        heapq.heapreplace(heap, entry)
                scanned += 1
                if progress is not None and scanned % 1000 == 0:
                    progress(dataset, scanned)
    return [entry[2] for heap in heaps.values() for entry in heap]


def _selection_facets(record, seed, preserve_lengths=False):
    """Choose lightweight case/point references before constructing QA candidates.

    The SFT projector still emits every facet. Benchmark matching needs only one
    alternative per record/slot, so projecting all surface points first is waste.
    """
    if record.get("schema_version") != "abc1.record.v3":
        return record
    chosen = {}
    for case in record["cases"]:
        key = hashlib.sha256(f"{seed}:{record['record_uid']}:{case['case_id']}".encode()).digest()
        start, length = case["actions"][0]["type"], len(case["actions"])
        for task, output in case.get("task_outputs", {}).items():
            slot = (task, start)
            if preserve_lengths:
                slot += (length,)
            if task == "A1":
                slot += (output["answer"],)
            elif task == "A2":
                slot += (length, output["forward_ordinal_1based"], output["distance_rank"])
            if task in ("A4", "B2"):
                for point in output["points"]:
                    direction_slot = slot + (point["answer"],)
                    if direction_slot not in chosen or key < chosen[direction_slot][0]:
                        chosen[direction_slot] = key, case["case_id"], task, dict(output, points=[point])
                continue
            if slot not in chosen or key < chosen[slot][0]:
                if task in ("A4", "B1", "B2"):
                    points = output["points"]
                    output = dict(output, points=[points[int.from_bytes(key[:4], "big") % len(points)]])
                chosen[slot] = key, case["case_id"], task, output
    outputs = defaultdict(dict)
    for _, case_id, task, output in chosen.values():
        if task in ("A4", "B2"):
            outputs[case_id].setdefault(task, dict(output, points=[]))["points"].extend(output["points"])
        else:
            outputs[case_id][task] = output
    return dict(record, cases=[dict(case, task_outputs=outputs[case["case_id"]])
                               for case in record["cases"]])


def _stable(seed: int, row: Candidate) -> str:
    return hashlib.sha256(
        f"{int(seed)}:{row.item_id}:{row.record_sha256}:"
        f"{row.outcome_id}".encode()).hexdigest()


def _exact_slot_quotas(
        dataset_task_totals, c1_length_totals,
        a2_start_totals, minimum_turn_first_fraction) -> Counter:
    quotas = Counter()
    for dataset, tasks in dataset_task_totals.items():
        if tasks.get("A1"):
            for (start, answer), count in _a1_slot_counts(
                    tasks["A1"], minimum_turn_first_fraction).items():
                quotas[(dataset, "A1", start, answer)] = count
        if tasks.get("A2"):
            for length, starts in a2_start_totals[dataset].items():
                for start, count in starts.items():
                    for cell in a2.balanced_cells(
                            int(length), int(count), starts_with=start,
                            rotation=spec.DATASETS.index(dataset) +
                            int(length)):
                        quotas[(dataset, "A2", int(length), start,
                                cell.forward_ordinal_1based,
                                cell.distance_rank)] += 1
        if tasks.get("C1"):
            for length, count in enumerate(c1_length_totals[dataset], 1):
                quotas[(dataset, "C1", length,
                        "turn" if length % 2 == 0 else "forward")] = count
    return +quotas


def all_slot_quotas(
        dataset_task_totals=spec.DATASET_TASK_TOTALS,
        c1_length_totals=spec.C1_LENGTH_TOTALS,
        a2_start_totals=spec.A2_START_TOTALS,
        minimum_turn_first_fraction=spec.MINIMUM_TURN_FIRST_FRACTION,
        ) -> Counter:
    quotas = _exact_slot_quotas(
        dataset_task_totals, c1_length_totals,
        a2_start_totals, minimum_turn_first_fraction)
    for dataset, tasks in dataset_task_totals.items():
        for task_id in ("A3", "A4", "B1", "B2", "B3"):
            total = int(tasks.get(task_id, 0))
            if not total:
                continue
            turns = int(math.ceil(
                total * float(minimum_turn_first_fraction) - 1e-12))
            quotas[(dataset, task_id, "turn")] = turns
            quotas[(dataset, task_id, "forward")] = total - turns
    return +quotas


def _match_exact_slots(
        rows: list[Candidate], quotas: Counter, *, seed: int,
        allow_record=None, accept_record=None, order_options=None,
        slot_for=candidate_slot,
        ) -> tuple[list[Candidate], Counter]:
    """Match all requested slots while enforcing one QA per record."""
    options = defaultdict(dict)
    keys = {row.item_id: _stable(seed, row) for row in rows}
    for row in rows:
        slot = slot_for(row)
        if slot not in quotas:
            continue
        prior = options[slot].get(row.record_id)
        if prior is None or keys[row.item_id] < keys[prior.item_id]:
            options[slot][row.record_id] = row
    ordered_options = {
        slot: (order_options(list(values.values())) if order_options else
               sorted(values.values(), key=lambda row: keys[row.item_id]))
        for slot, values in options.items()
    }
    requests = [
        (slot, index)
        for slot, count in quotas.items()
        for index in range(int(count))
    ]
    requests.sort(key=lambda request: (
        len(ordered_options.get(request[0], ())) /
        max(1, quotas[request[0]]), str(request[0]), request[1]))
    assignment = {}
    matched_record = {}
    missing = Counter()
    for request in requests:
        queue = [request]
        visited_requests = {request}
        visited_records = set()
        parent = {}
        free_record = None
        for active in queue:
            slot = active[0]
            for candidate in ordered_options.get(slot, ()):
                record_uid = candidate.record_id
                if record_uid in visited_records:
                    continue
                visited_records.add(record_uid)
                parent[record_uid] = active, candidate
                occupied = matched_record.get(record_uid)
                if occupied is None:
                    if allow_record is not None and not allow_record(record_uid):
                        continue
                    free_record = record_uid
                    break
                if occupied not in visited_requests:
                    visited_requests.add(occupied)
                    queue.append(occupied)
            if free_record is not None:
                break
        if free_record is None:
            missing[request[0]] += 1
            continue
        if accept_record is not None:
            accept_record(free_record)
        record_uid = free_record
        while True:
            active, candidate = parent[record_uid]
            previous = assignment.get(active)
            assignment[active] = candidate
            matched_record[record_uid] = active
            if previous is None:
                break
            record_uid = previous.record_id
    return list(assignment.values()), missing


def _present(rows: list[Candidate], seed: int) -> list[Candidate]:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row.task_id].append(row)
    result = []
    for task_id, values in by_task.items():
        ordered = sorted(values, key=lambda row: _stable(seed, row))
        ids = templates.balanced_template_ids([task_id] * len(ordered), seed=seed)
        for row, template_id in zip(ordered, ids):
            result.append(replace(
                row,
                template_id=template_id))
    labelled = {}
    for dataset_index, dataset in enumerate(spec.DATASETS):
        c1 = sorted(
            (row for row in result
             if row.task_id == "C1" and row.dataset == dataset),
            key=lambda row: _stable(seed + 1, row))
        labels = list(("ABCD" * math.ceil(len(c1) / 4))[:len(c1)])
        random.Random(seed + 1 + dataset_index).shuffle(labels)
        labelled.update(
            (row.item_id, label) for row, label in zip(c1, labels))
    return [replace(row, c1_label=labelled[row.item_id])
            if row.task_id == "C1" else row for row in result]


def select(
        candidates: Iterable[Candidate], *, seed: int,
        dataset_task_totals: Mapping[str, Mapping[str, int]] =
        spec.DATASET_TASK_TOTALS,
        c1_length_totals: Mapping[str, tuple[int, ...]] =
        spec.C1_LENGTH_TOTALS,
        a2_start_totals: Mapping[str, Mapping[int, Mapping[str, int]]] =
        spec.A2_START_TOTALS,
        minimum_turn_first_fraction: float =
        spec.MINIMUM_TURN_FIRST_FRACTION,
        ) -> tuple[tuple[Candidate, ...], dict]:
    rows = list(candidates)
    quotas = all_slot_quotas(
        dataset_task_totals, c1_length_totals,
        a2_start_totals, minimum_turn_first_fraction)
    selected, missing = _match_exact_slots(
        rows, quotas, seed=seed)
    if missing:
        details = ", ".join(
            f"{slot[0]}/{slot[1]}/{slot[2:]}:-{count}"
            for slot, count in sorted(missing.items()))
        raise SelectionShortfall(f"slot shortfalls: {details}")
    selected = _present(selected, seed)
    validate(
        selected, dataset_task_totals=dataset_task_totals,
        c1_length_totals=c1_length_totals,
        a2_start_totals=a2_start_totals,
        minimum_turn_first_fraction=minimum_turn_first_fraction)
    report = summarize(selected)
    return tuple(selected), report


def validate(
        rows: Iterable[Candidate], *,
        dataset_task_totals: Mapping[str, Mapping[str, int]] =
        spec.DATASET_TASK_TOTALS,
        c1_length_totals: Mapping[str, tuple[int, ...]] =
        spec.C1_LENGTH_TOTALS,
        a2_start_totals: Mapping[str, Mapping[int, Mapping[str, int]]] =
        spec.A2_START_TOTALS,
        minimum_turn_first_fraction: float =
        spec.MINIMUM_TURN_FIRST_FRACTION) -> None:
    """Check the few distribution guarantees that define the benchmark."""
    rows = list(rows)
    wanted = Counter({
        (dataset, task): int(count)
        for dataset, tasks in dataset_task_totals.items()
        for task, count in tasks.items() if count
    })
    actual = Counter((row.dataset, row.task_id) for row in rows)
    if actual != wanted or len({row.record_id for row in rows}) != len(rows):
        raise SelectionShortfall("selection totals or record uniqueness differ")
    for (dataset, task), total in wanted.items():
        cell = [row for row in rows
                if row.dataset == dataset and row.task_id == task]
        if task == "A1":
            counts = Counter(
                (row.starts_with, row.answer_bucket) for row in cell)
            if counts != Counter(_a1_slot_counts(
                    total, minimum_turn_first_fraction)):
                raise SelectionShortfall(f"{dataset}/A1 answer balance differs")
        elif task == "A2":
            counts = Counter((
                row.action_length, row.starts_with,
                row.a2_ordinal, row.a2_rank) for row in cell)
            expected = Counter()
            for length, starts in a2_start_totals[dataset].items():
                for start, count in starts.items():
                    expected.update(
                        (int(length), start,
                         slot.forward_ordinal_1based, slot.distance_rank)
                        for slot in a2.balanced_cells(
                            int(length), int(count), starts_with=start,
                            rotation=spec.DATASETS.index(dataset) +
                            int(length)))
            if counts != expected:
                raise SelectionShortfall(f"{dataset}/A2 design differs")
        elif task == "C1":
            counts = Counter((row.action_length, row.starts_with)
                             for row in cell)
            expected = +Counter({
                (length, "turn" if length % 2 == 0 else "forward"): count
                for length, count in enumerate(c1_length_totals[dataset], 1)
            })
            if counts != expected:
                raise SelectionShortfall(f"{dataset}/C1 lengths differ")
        else:
            minimum = int(math.ceil(
                total * float(minimum_turn_first_fraction) - 1e-12))
            if sum(row.starts_with == "turn" for row in cell) < minimum:
                raise SelectionShortfall(
                    f"{dataset}/{task} Turn-first coverage differs")


def summarize(rows: Iterable[Candidate]) -> dict:
    rows = list(rows)
    totals = Counter((row.dataset, row.task_id) for row in rows)
    turns = Counter((row.dataset, row.task_id) for row in rows
                    if row.starts_with == "turn")
    lengths = Counter((row.dataset, row.task_id, row.action_length)
                      for row in rows)
    a1_answers = Counter((row.dataset, row.answer_bucket) for row in rows
                         if row.task_id == "A1")
    c1_labels = Counter((row.dataset, row.c1_label) for row in rows
                        if row.task_id == "C1")
    return {
        "total": len(rows),
        "dataset_task_totals": {
            f"{dataset}/{task}": count
            for (dataset, task), count in sorted(totals.items())
        },
        "turn_first": {
            f"{dataset}/{task}": {
                "count": turns[dataset, task],
                "fraction": round(turns[dataset, task] / count, 6),
            }
            for (dataset, task), count in sorted(totals.items())
        },
        "action_lengths": {
            f"{dataset}/{task}/L{length}": count
            for (dataset, task, length), count in sorted(lengths.items())
        },
        "a1_answers": {
            f"{dataset}/{answer}": count
            for (dataset, answer), count in sorted(a1_answers.items())
        },
        "c1_labels": {
            f"{dataset}/{label}": count
            for (dataset, label), count in sorted(c1_labels.items())
        },
    }
