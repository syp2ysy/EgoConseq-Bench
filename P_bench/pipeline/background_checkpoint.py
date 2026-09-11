"""Small atomic checkpoint for background ABC1 supply reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

from pipeline import io_utils


SCHEMA = "egoconseq.background-compile-checkpoint.v2"
SUMMARY_NAME = "checkpoint.json"


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def paths(manifest: Mapping, checkpoint: str) -> tuple[Path, Path, Path]:
    root = Path(str(manifest["output_root"])).resolve() / "artifacts" / "global"
    final = root / str(checkpoint)
    return root, final, root / f".{checkpoint}.staging"


def prepare(manifest: Mapping, checkpoint: str) -> tuple[Path, Path]:
    root, final, staging = paths(manifest, checkpoint)
    root.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    return final, staging


def publish(
        manifest: Mapping, checkpoint: str, staging: Path, *,
        result: Mapping, result_files: Sequence[str],
        sources: Sequence[Mapping]) -> dict:
    _root, final, expected = paths(manifest, checkpoint)
    files = {
        relative: io_utils.sha256_file(expected / relative)
        for relative in result_files
    }
    body = {
        "schema": SCHEMA,
        "manifest_sha256": manifest["sha256"],
        "checkpoint": str(checkpoint),
        "files": files,
        "sources": [dict(row) for row in sources],
        "result": dict(result),
    }
    summary = {**body, "sha256": _canonical_sha256(body)}
    io_utils.atomic_write_json(
        expected / SUMMARY_NAME, summary, allow_nan=False, durable=True)
    expected.rename(final)
    path = final / SUMMARY_NAME
    return {
        "path": str(path),
        "sha256": io_utils.sha256_file(path),
        "content_sha256": summary["sha256"],
    }


def load(
        manifest: Mapping, checkpoint: str,
        identity: Mapping | None = None) -> dict:
    _root, final, _staging = paths(manifest, checkpoint)
    path = final / SUMMARY_NAME
    summary = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in summary.items() if key != "sha256"}
    if (summary.get("schema") != SCHEMA or
            summary.get("manifest_sha256") != manifest["sha256"] or
            summary.get("checkpoint") != str(checkpoint) or
            summary.get("sha256") != _canonical_sha256(body)):
        raise ValueError("checkpoint identity differs")
    for relative, digest in summary["files"].items():
        if io_utils.sha256_file(final / relative) != digest:
            raise ValueError(f"checkpoint file changed: {relative}")
    if identity is not None:
        expected = {
            "path": str(path),
            "sha256": io_utils.sha256_file(path),
            "content_sha256": summary["sha256"],
        }
        if dict(identity) != expected:
            raise ValueError("checkpoint state identity differs")
    return summary
