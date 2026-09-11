"""Balance 24 direction labels by swapping QA, never modifying source records."""

from collections import Counter, defaultdict
from pathlib import Path

from pipeline import actions
from post_QA.seen_build import selection


TASKS = ("A4", "B2")
LABELS = tuple(actions.spatial_direction_key(h, v)
               for h in actions.HORIZONTAL_DIRECTION_LABELS
               for v in actions.VERTICAL_DIRECTION_LABELS)


def summary(rows):
    counts = Counter((row.task_id, row.answer_bucket) for row in rows)
    return {task: {
        "counts": {label: counts[task, label] for label in LABELS},
        "covered": sum(counts[task, label] > 0 for label in LABELS),
        "total_classes": len(LABELS),
    } for task in TASKS}


def rebalance(selected, candidates, *, seed, allow=None, accept=None):
    """Reduce label imbalance using only same dataset/start/length replacements.

    Counts strictly improve on every swap, so this terminates even when a label
    is unavailable. Existing records get first choice; all other tasks stay fixed.
    """
    result = list(selected)
    by_record = {row.record_id: i for i, row in enumerate(result)}
    slots, options = defaultdict(list), defaultdict(list)
    counts = Counter((r.task_id, r.answer_bucket) for r in result)
    for i, row in enumerate(result):
        if row.task_id in TASKS:
            slots[selection.candidate_slot(row, include_length=True)].append(i)
    for row in sorted(candidates, key=lambda r: (
            r.record_id not in by_record, selection._stable(seed, r))):
        if row.task_id in TASKS:
            options[row.task_id, row.answer_bucket].append(row)
    while True:
        changed = False
        for key in sorted(options, key=lambda key: (counts[key], len(options[key]), key)):
            if max(counts[key[0], label] for label in LABELS) <= counts[key] + 1:
                continue
            for candidate in options[key]:
                slot = selection.candidate_slot(candidate, include_length=True)
                occupied = by_record.get(candidate.record_id)
                indices = ([occupied] if occupied is not None else slots[slot])
                indices = sorted(indices, key=lambda i: -counts[
                    result[i].task_id, result[i].answer_bucket])
                for i in indices:
                    old = result[i]
                    if (selection.candidate_slot(old, include_length=True) != slot or
                            counts[old.task_id, old.answer_bucket] <= counts[key] + 1 or
                            (occupied is not None and old.body_radius_m != candidate.body_radius_m)):
                        continue
                    if allow is not None and not allow(candidate, old):
                        continue
                    if accept is not None:
                        accept(candidate, old, i)
                    counts[old.task_id, old.answer_bucket] -= 1
                    counts[key] += 1
                    del by_record[old.record_id]
                    by_record[candidate.record_id] = i
                    result[i] = candidate
                    changed = True
                    break
                else:
                    continue
                break
        if not changed:
            return result


def image_path(row):
    return str(Path(row.source_path).parent / row.image_path)


def refine_images(selected, candidates, bank, *, seed, threshold):
    """Use the same raw-image DINO bank; accept a new view only if it fits."""
    torch = bank.torch
    paths = [image_path(row) for row in selected]
    bank.add(paths)
    matrix = torch.stack([bank.features[path] for path in paths]).to(bank.device)
    positions = {row.record_id: i for i, row in enumerate(selected)}
    # Decode small batches on demand, only for a direction we are trying to add.
    pending = defaultdict(list)
    for row in sorted(candidates, key=lambda r: selection._stable(seed, r), reverse=True):
        pending[row.task_id, row.answer_bucket].append(image_path(row))

    def allow(new, old):
        if new.record_id == old.record_id:
            return True
        path = image_path(new)
        if path not in bank.metrics:
            queue = pending[new.task_id, new.answer_bucket]
            batch = {path}
            while queue and len(batch) < 64:
                value = queue.pop()
                if value not in bank.metrics:
                    batch.add(value)
            bank.add(batch)
        if path not in bank.features:
            return False
        with torch.inference_mode():
            similarities = matrix @ bank.features[path].to(bank.device)
            similarities[positions[old.record_id]] = -1
            return not bool((similarities >= threshold).any())

    def accept(new, old, index):
        del positions[old.record_id]
        positions[new.record_id] = index
        matrix[index] = bank.features[image_path(new)].to(bank.device)

    before = summary(selected)
    result = rebalance(selected, candidates, seed=seed, allow=allow, accept=accept)
    print(f"[directions] before={before} after={summary(result)}", flush=True)
    return result
