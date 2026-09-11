"""Read-only supply analysis before records or benchmark materialization."""

from __future__ import annotations

from collections import defaultdict
import heapq
import json
from pathlib import Path
from typing import Iterable, Mapping

from pipeline import benchmark_candidates, candidate_sources, io_utils
from pipeline.benchmark_candidates import Candidate
from post_QA.seen_build import inventory, release, selection, spec


def analyze_candidates(
        candidates: Iterable[Candidate], *, quotas: Mapping[tuple, int] | None = None,
        seed: int) -> dict:
    rows = list(candidates)
    required = selection.all_slot_quotas() if quotas is None else dict(quotas)
    by_slot = defaultdict(set)
    for row in rows:
        by_slot[selection.candidate_slot(row)].add(row.record_id)
    _matched, missing = selection._match_exact_slots(
        rows, required, seed=seed)
    slots = []
    for slot, count in sorted(required.items()):
        available = len(by_slot.get(slot, ()))
        slots.append({
            "slot": list(slot),
            "required": int(count),
            "unique_records": available,
            "raw_shortfall": max(0, int(count) - available),
            "shortfall": int(missing.get(slot, 0)),
        })
    return {
        "feasible": not missing,
        "candidate_pool": len(rows),
        "slots": slots,
        "shortfall_total": sum(missing.values()),
    }


def _bounded_pool(
        projected_records: Iterable[list[Candidate]], *, seed: int
        ) -> list[Candidate]:
    quotas = selection.all_slot_quotas()
    heaps = defaultdict(list)
    serial = 0
    for projected in projected_records:
        best = {}
        for row in projected:
            slot = selection.candidate_slot(row)
            if slot not in quotas:
                continue
            key = selection._stable(seed, row)
            if slot not in best or key < best[slot][0]:
                best[slot] = key, row
        for slot, (key, row) in best.items():
            capacity = max(100, int(quotas[slot]) * 10)
            entry = (-int(key, 16), serial, row)
            serial += 1
            heap = heaps[slot]
            if len(heap) < capacity:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)
    return [entry[2] for heap in heaps.values() for entry in heap]


def virtual_candidates(
        plan_path: Path, work_dir: Path, *, seed: int, progress=None
        ) -> tuple[list[Candidate], dict]:
    """Project source records plus bound deltas without writing large files."""
    header, rows = inventory.read_plan(plan_path)
    results = release.load_results(work_dir, header, rows)
    streams = {}
    capacity_rejected = 0
    scanned = 0

    def records():
        nonlocal capacity_rejected, scanned
        try:
            for row in rows:
                source = release._read_source_record(row, streams)
                result = results.get(str(row["plan_row_id"]), {
                    "status": "unavailable", "mode": "reused"})
                compact = release.compact_record(row, source, result)
                if compact is None:
                    capacity_rejected += 1
                    compact = release.compact_record(
                        row, source, {"mode": "reused"})
                record_sha256 = candidate_sources.canonical_sha256(compact)
                yield benchmark_candidates.project_record(
                    dataset=str(row["dataset"]),
                    source_path=str(row["source_path"]),
                    byte_offset=int(row["byte_offset"]),
                    record_sha256=record_sha256, record=compact,
                    allowed_tasks=spec.supported_tasks(str(row["dataset"])))
                scanned += 1
                if progress is not None and scanned % 1000 == 0:
                    progress(str(row["dataset"]), scanned)
        finally:
            for stream in streams.values():
                stream.close()

    pool = _bounded_pool(records(), seed=seed)
    return pool, {
        "plan_id": header["plan_id"],
        "records": len(rows),
        "bound_results": len(results),
        "unbound_plan_rows": len(rows) - len(results),
        "capacity_rejected": capacity_rejected,
    }


def dry_run(
        plan_path: Path, work_dir: Path, *, seed: int,
        output_path: Path | None = None, progress=None) -> dict:
    candidates, collection = virtual_candidates(
        plan_path, work_dir, seed=seed, progress=progress)
    report = {
        "schema": "egoconseq.abc1-supply.v1",
        "collection": collection,
        **analyze_candidates(candidates, seed=seed),
    }
    if output_path is not None:
        io_utils.atomic_write_json(Path(output_path), report, allow_nan=False)
    return report
