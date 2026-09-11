"""Coverage-first SFT sampling, with soft task targets and no duplicate QA."""

from collections import Counter, defaultdict, deque
from dataclasses import replace
import heapq
import random

from pipeline import a2
from post_QA import templates


def bucket(row):
    """Only balance meaningful marginals; do not cross all physical settings."""
    if row.task_id == "A1":
        return row.dataset, row.starts_with, row.answer_bucket
    if row.task_id == "A2":
        return row.dataset, row.action_length, row.starts_with, row.a2_rank
    if row.task_id == "C1":
        return row.dataset, row.action_length
    if row.task_id in {"B1", "B3"}:
        return row.dataset, min(int(row.numeric_value_m), 5)
    return row.dataset, row.answer_bucket


def secondary(row):
    return (row.action_length, row.starts_with, row.a2_ordinal or 0,
            row.checkpoint_fraction or 0, row.checkpoint_ordinal or 0)


def question_key(row):
    return (row.action_signature, row.body_radius_m, row.point_id,
            row.checkpoint_ordinal, row.checkpoint_fraction)


def _labels(task, key):
    if task == "A1":
        return ("collision", "no_collision")
    return {cell.distance_rank for cell in a2.cells_for_length(
        key[1], starts_with=key[2])}


def representatives(rows, rng):
    """Keep at most two alternatives per record/bucket, without storing all facets."""
    rng.shuffle(rows)
    kept = defaultdict(set)
    result = []
    for row in rows:
        if row.task_id == "A2" and row.action_length == 3 and row.starts_with == "turn":
            continue
        key = row.task_id, bucket(row), secondary(row)
        question = question_key(row)
        if len(kept[key]) < 2 and question not in kept[key]:
            kept[key].add(question)
            result.append(row)
    return result


def _interleave(rows, rng):
    rng.shuffle(rows)
    groups = defaultdict(deque)
    for row in rows:
        groups[secondary(row)].append(row)
    active = deque(groups.values())
    result = deque()
    while active:
        group = active.popleft()
        result.append(group.popleft())
        if group:
            active.append(group)
    return result


def select(records, *, seed):
    """One QA per eligible record first, then fill tasks up to the smallest supply.

    Two QA per record/task at most. A2 caps the distance-rank marginals within
    dataset/length/start; missing ranks reduce its total, not the whole dataset.
    Coverage takes precedence when a record offers only an overfull bucket.
    """
    rng = random.Random(seed)
    records = [rows for rows in records if rows]
    rng.shuffle(records)
    pools = defaultdict(lambda: defaultdict(list))
    capacity = Counter()
    for rows in records:
        by_task = defaultdict(set)
        for row in rows:
            by_task[row.task_id].add(question_key(row))
            pools[row.task_id][bucket(row)].append(row)
        capacity.update({task: min(2, len(keys)) for task, keys in by_task.items()})
    target = min(capacity.values(), default=0)
    limits = {}
    for task in ("A1", "A2"):
        supply = {}
        for key, rows in pools[task].items():
            by_record = defaultdict(set)
            for row in rows:
                by_record[row.record_id].add(question_key(row))
            supply[key] = sum(min(2, len(keys)) for keys in by_record.values())
        for key in supply:
            limits[task, key] = min((supply.get((*key[:-1], label), 0)
                                     for label in _labels(task, key)), default=0)

    totals, counts, detail_counts = Counter(), Counter(), Counter()
    used = defaultdict(set)
    selected = []
    overrides = 0

    def available(row, balanced=True):
        prior = used[row.record_id, row.task_id]
        return (len(prior) < 2 and question_key(row) not in prior and
                (not balanced or counts[row.task_id, bucket(row)] <
                 limits.get((row.task_id, bucket(row)), float("inf"))))

    def take(row):
        selected.append(row)
        used[row.record_id, row.task_id].add(question_key(row))
        totals[row.task_id] += 1
        counts[row.task_id, bucket(row)] += 1
        detail_counts[row.task_id, secondary(row)] += 1

    records.sort(key=lambda rows: len({row.task_id for row in rows}))
    for rows in records:
        choices = [row for row in rows if available(row)]
        if not choices:
            choices = rows
            overrides += 1
        take(min(choices, key=lambda row: (
            totals[row.task_id], capacity[row.task_id],
            counts[row.task_id, bucket(row)], detail_counts[row.task_id, secondary(row)])))

    for task, groups in pools.items():
        queues = {key: _interleave(rows, rng) for key, rows in groups.items()}
        def priority(key):
            weight = (0.3 if key[1] == "turn" else 0.7) if task == "A1" else 1.0
            return counts[task, key] / weight
        heap = [(priority(key), key) for key in queues]
        heapq.heapify(heap)
        while heap and totals[task] < target:
            _, key = heapq.heappop(heap)
            queue = queues[key]
            while queue and not available(queue[0]):
                queue.popleft()
            if queue:
                take(queue.popleft())
                heapq.heappush(heap, (priority(key), key))

    # Bucket supply overlaps at the two-QA record/task cap. Balance the actual
    # survivors, not the hypothetical supply, without erasing a record's last QA.
    record_use = Counter(row.record_id for row in selected)
    conditional = defaultdict(list)
    for row in selected:
        if row.task_id in {"A1", "A2"}:
            conditional[row.task_id, bucket(row)[:-1]].append(row)
    dropped, residual = set(), {}
    for (task, key), rows in conditional.items():
        achieved = Counter(bucket(row)[-1] for row in rows)
        ceiling = min(achieved[label] for label in _labels(task, bucket(rows[0])))
        for row in reversed(rows):
            label = bucket(row)[-1]
            if achieved[label] > ceiling and record_use[row.record_id] > 1:
                dropped.add(row.item_id)
                achieved[label] -= 1
                record_use[row.record_id] -= 1
                totals[task] -= 1
        excess = sum(count - ceiling for count in achieved.values())
        if excess:
            residual["/".join(map(str, (task, *key)))] = excess
    selected = [row for row in selected if row.item_id not in dropped]
    rng.shuffle(selected)
    c1_counts = Counter()
    presented = []
    template_ids = templates.balanced_template_ids(
        (row.task_id for row in selected), seed=seed)
    labels = list("ABCD")
    rng.shuffle(labels)
    for row, template_id in zip(selected, template_ids):
        task = row.task_id
        label = labels[c1_counts[row.dataset] % 4] if task == "C1" else None
        c1_counts[row.dataset] += task == "C1"
        presented.append(replace(row, template_id=template_id, c1_label=label))
    return presented, {
        "target_per_task": target,
        "capacity_per_task": dict(sorted(capacity.items())),
        "task_shortfalls": {task: max(0, target - totals[task]) for task in capacity},
        "coverage_balance_overrides": overrides,
        "balance_trimmed_qa": len(dropped),
        "coverage_required_balance_excess": residual,
    }
