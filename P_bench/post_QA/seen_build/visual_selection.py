"""Stratified candidate batches, visual exclusion, then record-exclusive matching."""

from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import math
import time

from pipeline import a2
from post_QA.seen_build import direction_balance, images, selection, spec


def balanced_order(rows, seed):
    """Round-robin scenes and action/answer buckets; keep deterministic ties."""
    scenes = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda row: selection._stable(seed, row)):
        answer = (int(row.numeric_value_m // 2) if row.task_id in {"B1", "B3"}
                  else row.answer_bucket)
        scenes[row.scene_id][row.action_length, answer, row.action_family].append(row)
    per_scene = [
        [row for batch in zip_longest(*buckets.values()) for row in batch if row is not None]
        for buckets in scenes.values()]
    return [row for batch in zip_longest(*per_scene) for row in batch if row is not None]


def _spread(total, capacities, preferred=None, initial=None):
    counts = dict(initial) if initial is not None else dict.fromkeys(capacities, 0)
    if sum(counts.values()) > total:
        raise selection.SelectionShortfall(f"length lower bounds exceed {total}: {counts}")
    for _ in range(total - sum(counts.values())):
        available = [key for key, cap in capacities.items() if counts[key] < cap]
        if not available:
            raise selection.SelectionShortfall(f"length supply cannot fill {total}: {capacities}")
        key = min(available, key=lambda key: (
            counts[key] / max(1, preferred[key]) if preferred else counts[key], key))
        counts[key] += 1
    return counts


def visual_slot(row):
    return selection.candidate_slot(row, include_length=True)


def ordinary_quotas(rows, totals=spec.DATASET_TASK_TOTALS):
    supply = defaultdict(set)
    answer_supply = defaultdict(set)
    for row in rows:
        supply[row.dataset, row.task_id, row.starts_with, row.action_length].add(row.record_id)
        if row.task_id == "A1":
            answer_supply[row.dataset, row.starts_with, row.action_length, row.answer_bucket].add(row.record_id)
    quotas = Counter()
    for dataset, tasks in totals.items():
        for task, total in tasks.items():
            if not total or task in ("A2", "C1"):
                continue
            lengths = _spread(total, {length: len(
                supply[dataset, task, "forward", length] |
                supply[dataset, task, "turn", length]) for length in range(1, 7)})
            turn_total = math.ceil(total * spec.MINIMUM_TURN_FIRST_FRACTION - 1e-12)
            turns = _spread(turn_total, {length: min(lengths[length], len(
                supply[dataset, task, "turn", length])) for length in lengths},
                initial={length: max(0, lengths[length] - len(
                    supply[dataset, task, "forward", length])) for length in lengths})
            for start in ("forward", "turn"):
                counts = {length: turns[length] if start == "turn" else lengths[length] - turns[length]
                          for length in lengths}
                if task == "A1":
                    target = selection._a1_slot_counts(total, spec.MINIMUM_TURN_FIRST_FRACTION)[start, "collision"]
                    collision = _spread(target, {
                        length: min(count, len(answer_supply[dataset, start, length, "collision"]))
                        for length, count in counts.items()}, preferred=counts,
                        initial={length: max(0, count - len(answer_supply[
                            dataset, start, length, "no_collision"])) for length, count in counts.items()})
                    for length, count in counts.items():
                        quotas[dataset, task, start, "collision", length] = collision[length]
                        quotas[dataset, task, start, "no_collision", length] = count - collision[length]
                else:
                    for length, count in counts.items():
                        quotas[dataset, task, start, length] = count
    return +quotas


def balanced_capacity(counts, *, rotation):
    """Exact capacity of balanced_cells, including its partial final cycle."""
    offset = rotation % len(counts)
    rotated = counts[offset:] + counts[:offset]
    return min(count * len(counts) + index for index, count in enumerate(rotated))


def length_design(rows):
    """Adapt lengths, never task totals or A2 within-length ordinal/rank balance."""
    supply = defaultdict(set)
    for row in rows:
        supply[selection.candidate_slot(row)].add(row.record_id)
    a2_totals, c1_totals = {}, {}
    for dataset in spec.DATASETS:
        a2_totals[dataset] = {length: {"forward": 0, "turn": 0} for length in range(3, 7)}
        for start in ("turn", "forward"):
            target = sum(starts[start] for starts in spec.A2_START_TOTALS[dataset].values())
            capacities = {}
            for length in range(3, 7):
                cells = a2.cells_for_length(length, starts_with=start)
                if not cells:
                    continue
                available = [len(supply[dataset, "A2", length, start,
                                       cell.forward_ordinal_1based, cell.distance_rank])
                             for cell in cells]
                capacities[length] = balanced_capacity(
                    available, rotation=spec.DATASETS.index(dataset) + length)
            preferred = {length: spec.DATASET_TASK_TOTALS[dataset]["A2"] / 4 -
                         a2_totals[dataset][length]["turn"] for length in capacities}
            if sum(capacities.values()) < target:
                raise selection.SelectionShortfall(
                    f"{dataset}/A2/{start} needs {target}; balanced capacities {capacities}")
            allocated = _spread(target, capacities, preferred if start == "forward" else None)
            for length, count in allocated.items():
                a2_totals[dataset][length][start] = count
        total = spec.DATASET_TASK_TOTALS[dataset]["C1"]
        lengths = [0] * 6
        for start, target in (("turn", total // 2), ("forward", total - total // 2)):
            capacities = {length: len(supply[dataset, "C1", length, start])
                          for length in range(1, 7)
                          if (length % 2 == 0) == (start == "turn")}
            for length, count in _spread(target, capacities).items():
                lengths[length - 1] = count
        c1_totals[dataset] = tuple(lengths)
    return a2_totals, c1_totals


def select(candidates, *, seed, device="cuda:0", threshold=0.90):
    started = time.monotonic()
    length_design(candidates)  # Cheap structural supply before opening any image.
    bank = images.ImageBank(device)
    slots = defaultdict(list)
    for row in candidates:
        slots[visual_slot(row)].append(row)
    slots = {slot: balanced_order(rows, seed) for slot, rows in slots.items()}
    initial_quotas = ordinary_quotas(candidates)
    initial_quotas.update({slot: count for slot, count in selection.all_slot_quotas().items()
                          if slot[1] in ("A2", "C1")})
    factor = 3
    while True:
        pool = [row for slot, rows in slots.items()
                for row in rows[:max(40, initial_quotas.get(slot, 1) * factor)]]
        path_for = {row.record_id: str(Path(row.source_path).parent / row.image_path) for row in pool}
        bank.add(path_for.values())
        eligible = [row for row in pool if path_for[row.record_id] in bank.features and
                    (row.task_id != "C1" or bank.good_endpoints(
                        Path(row.source_path).parent / path for path in row.terminal_paths))]
        print(f"[select] pool={len(pool)} records={len(path_for)} qualified={len(eligible)}", flush=True)
        try:
            a2_totals, c1_totals = length_design(eligible)
            quotas = ordinary_quotas(eligible)
        except selection.SelectionShortfall as error:
            print(f"[select] expanding candidate supply: {error}", flush=True)
            if len(pool) == len(candidates):
                raise
            factor *= 2
            continue
        quotas.update({slot: count for slot, count in selection.all_slot_quotas(
            a2_start_totals=a2_totals, c1_length_totals=c1_totals).items()
                       if slot[1] in ("A2", "C1")})
        uids = list(dict.fromkeys(row.record_id for row in eligible))
        positions = {uid: index for index, uid in enumerate(uids)}
        paths = [path_for[uid] for uid in uids]
        conflicts = bank.conflicts(paths, threshold)
        blocked = set()

        def allow(uid):
            return positions[uid] not in blocked

        def accept(uid):
            blocked.update(conflicts[positions[uid]])

        for order_seed in ((seed, seed + 1) if len(pool) == len(candidates) else (seed,)):
            blocked.clear()

            def order(rows):
                ordered = balanced_order(rows, order_seed)
                if order_seed != seed:
                    ordered.sort(key=lambda row: len(conflicts[positions[row.record_id]]))
                return ordered

            selected, missing = selection._match_exact_slots(
                eligible, quotas, seed=order_seed, allow_record=allow, accept_record=accept,
                slot_for=visual_slot, order_options=order)
            if not missing:
                break
        print(f"[select] selected={len(selected)} shortfall={sum(missing.values())} "
              f"dino_encoded={len(bank.features)}", flush=True)
        if not missing:
            break
        if len(pool) == len(candidates):
            raise selection.SelectionShortfall(f"no selection found in two orderings: {dict(missing)}")
        factor *= 2
    selected = direction_balance.refine_images(
        selected, candidates, bank, seed=seed, threshold=threshold)
    selected = selection._present(selected, seed)
    selection.validate(selected, a2_start_totals=a2_totals, c1_length_totals=c1_totals)
    report = selection.summarize(selected)
    report["seed"] = seed
    selected_paths = [direction_balance.image_path(row) for row in selected]
    report["visual_selection"] = {
        "model": images.MODEL, "input_size_wh": images.INPUT_SIZE,
        "checkpoint": bank.checkpoint, "cosine_threshold": threshold,
        "quality_limits": images.QUALITY_LIMITS,
        "encoded_initial_images": len(bank.features),
        "quality_rejected": sum(not row["accepted"] for row in bank.metrics.values()),
        "seconds": time.monotonic() - started, **dict(bank.timings),
        **bank.nearest_pairs(selected_paths, [row.item_id for row in selected]),
    }
    report["effective_length_design"] = {"a2": a2_totals, "c1": c1_totals}
    report["scenes"] = dict(sorted(Counter(f"{row.dataset}/{row.scene_id}" for row in selected).items()))
    report["action_signatures"] = {task: len({row.action_signature for row in selected if row.task_id == task})
                                   for task in spec.TASKS}
    report["physical_parameters"] = {name: dict(sorted(Counter(
        str(getattr(row, name)) for row in selected).items()))
        for name in ("body_radius_m", "camera_height_m", "hfov_deg")}
    report["answer_buckets"] = dict(sorted(Counter(
        f"{row.dataset}/{row.task_id}/{row.answer_bucket}" for row in selected).items()))
    report["a2_baselines"] = {}
    for start in ("forward", "turn"):
        rows = [row for row in selected if row.task_id == "A2" and row.starts_with == start]
        report["a2_baselines"][start] = {
            "longest_forward_accuracy": sum(row.a2_rank == "longest" for row in rows) / len(rows),
            "random_forward_accuracy": sum(1 / (row.action_length // 2 +
                int(start == "forward" and row.action_length % 2 == 1)) for row in rows) / len(rows)}
    report["image_quality"] = {row.item_id: bank.metrics[direction_balance.image_path(row)] for row in selected}
    report["direction_balance"] = direction_balance.summary(selected)
    return selected, report
