"""Lightweight catalog for one or many compact records shards."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from pipeline import io_utils


@dataclass(frozen=True)
class Source:
    dataset: str
    records_path: Path
    records_sha256: str
    run_meta_sha256: str
    index_path: str | None = None


def load(records_root: Path) -> tuple[Source, ...]:
    root = Path(records_root).resolve()
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = []
    for dataset_row in document.get("datasets") or []:
        dataset = str(dataset_row["dataset"])
        for row in dataset_row.get("shards") or (dataset_row,):
            path = (root / str(row["records_path"])).resolve()
            meta = path.with_name("run_meta.json")
            result.append(Source(
                dataset=dataset,
                records_path=path,
                records_sha256=str(
                    row.get("records_sha256") or io_utils.sha256_file(path)),
                run_meta_sha256=str(
                    row.get("run_meta_sha256") or
                    (io_utils.sha256_file(meta) if meta.is_file() else "")),
                index_path=str(row.get("index_path") or path)))
    return tuple(result)


def by_path(records_root: Path) -> dict[str, Source]:
    return {str(source.records_path): source for source in load(records_root)}


def write(
        source_rows: Iterable[Mapping[str, object]], output_path: Path,
        *, split: str = "train") -> dict:
    """Write shard references; never concatenate or copy record payloads."""
    output_path = Path(output_path).resolve()
    root = output_path.parent
    grouped = {}
    total = 0
    for row in source_rows:
        path = Path(str(row["path"])).resolve()
        meta = path.with_name("run_meta.json")
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        dataset = str(row.get("dataset") or metadata.get("dataset") or
                      metadata.get("source_dataset") or "")
        if not dataset:
            with path.open(encoding="utf-8") as stream:
                first = json.loads(next(line for line in stream if line.strip()))
            dataset = str(first.get("dataset") or
                          (first.get("source") or {}).get("source_dataset"))
        count = int(metadata.get("record_count") or sum(
            1 for line in path.open("rb") if line.strip()))
        grouped.setdefault(dataset, []).append({
            "records_path": os.path.relpath(path, root),
            "records_sha256": str(
                row.get("records_sha256") or io_utils.sha256_file(path)),
            "run_meta_sha256": str(
                row.get("run_meta_sha256") or io_utils.sha256_file(meta)),
            "record_count": count,
        })
        total += count
    document = {
        "schema": "egoconseq.abc1-record-catalog.v1",
        "split": str(split),
        "record_count": total,
        "datasets": [
            {"dataset": dataset, "shards": shards}
            for dataset, shards in sorted(grouped.items())
        ],
    }
    io_utils.atomic_write_json(output_path, document, allow_nan=False)
    return document
