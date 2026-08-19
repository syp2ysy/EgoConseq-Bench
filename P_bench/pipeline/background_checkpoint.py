"""Recoverable publication and verification for controller checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

from pipeline import (
    background_authorities, candidate_preview, gate_authority, io_utils,
)


SCHEMA = "egoconseq.background-compile-checkpoint.v1"
SUMMARY_NAME = "checkpoint.json"
_ARTIFACT_FILES = (
    "candidate_qa/benchmark.json",
    "candidate_qa/report.json",
    "candidate_qa/public/manifest.json",
    "candidate_qa/private/manifest.json",
    "candidate_qa/private/source_map.json",
)


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def paths(manifest: Mapping, checkpoint: str) -> tuple[Path, Path, Path]:
    root = Path(str(manifest["output_root"])).resolve() / "artifacts" / "global"
    final = root / checkpoint
    staging = root / f".{checkpoint}.staging"
    if final.parent != root or staging.parent != root:
        raise ValueError("background checkpoint escapes artifact root")
    return root, final, staging


def prepare(manifest: Mapping, checkpoint: str) -> tuple[Path, Path]:
    """Create clean controller-owned staging, preserving all final output."""
    root, final, staging = paths(manifest, checkpoint)
    root.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(
            f"finalized background checkpoint already exists: {final}")
    if staging.exists():
        if staging.resolve().parent != root.resolve() or \
                staging.name != f".{checkpoint}.staging":
            raise ValueError("background checkpoint staging path is invalid")
        shutil.rmtree(staging)
    staging.mkdir()
    return final, staging


def _file_identities(root: Path, relatives: Sequence[str]) -> dict:
    identities = {}
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            raise ValueError(
                f"background checkpoint output is missing: {relative}")
        identities[str(relative)] = io_utils.sha256_file(path)
    return identities


def _validate_candidate(final: Path, sources: Sequence[Mapping]) -> dict:
    authority = background_authorities.resolve_artifact_source_authority(
        final / "candidate_qa", sources)
    return candidate_preview.validate_preview_artifact(
        final / "candidate_qa", expected_source_authority=authority)


def publish(
        manifest: Mapping, checkpoint: str, staging: Path, *,
        result: Mapping, result_files: Sequence[str],
        sources: Sequence[Mapping]) -> dict:
    """Authenticate complete staging and atomically rename it to final."""
    _root, final, expected_staging = paths(manifest, checkpoint)
    if Path(staging).resolve() != expected_staging.resolve() or final.exists():
        raise ValueError("background checkpoint publication target is invalid")
    relatives = [*_ARTIFACT_FILES, *result_files]
    source_rows = [dict(row) for row in sources]
    if not source_rows:
        raise ValueError("background checkpoint sources are absent")
    body = {
        "schema": SCHEMA,
        "manifest_sha256": manifest["sha256"],
        "checkpoint": checkpoint,
        "artifact": str(final / "candidate_qa"),
        "files": _file_identities(expected_staging, relatives),
        "sources": source_rows,
        "result": dict(result),
    }
    summary = {**body, "sha256": _canonical_sha256(body)}
    io_utils.atomic_write_json(
        expected_staging / SUMMARY_NAME, summary,
        allow_nan=False, durable=True)
    expected_staging.rename(final)
    summary_path = final / SUMMARY_NAME
    return {
        "path": str(summary_path),
        "sha256": io_utils.sha256_file(summary_path),
        "content_sha256": summary["sha256"],
    }


def load(
        manifest: Mapping, checkpoint: str, identity: Mapping = None) -> dict:
    """Reopen a final checkpoint and all files named by its summary."""
    _root, final, _staging = paths(manifest, checkpoint)
    summary_path = final / SUMMARY_NAME
    summary = gate_authority.load_authority_manifest(summary_path)
    body = {key: value for key, value in summary.items() if key != "sha256"}
    if (summary.get("sha256") != _canonical_sha256(body) or
            summary.get("schema") != SCHEMA or
            summary.get("manifest_sha256") != manifest["sha256"] or
            summary.get("checkpoint") != checkpoint or
            summary.get("artifact") != str(final / "candidate_qa")):
        raise ValueError("background checkpoint summary binding differs")
    if identity is not None:
        expected_identity = {
            "path": str(summary_path),
            "sha256": io_utils.sha256_file(summary_path),
            "content_sha256": summary["sha256"],
        }
        if dict(identity) != expected_identity:
            raise ValueError("background checkpoint state identity differs")
    files = summary.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("background checkpoint file identities are absent")
    for relative, digest in files.items():
        path = final / str(relative)
        try:
            path.resolve().relative_to(final.resolve())
        except ValueError as error:
            raise ValueError("background checkpoint file escapes root") \
                from error
        if not path.is_file() or io_utils.sha256_file(path) != digest:
            raise ValueError("background checkpoint file identity differs")
    sources = summary.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("background checkpoint source authority is absent")
    validated = _validate_candidate(final, sources)
    result = summary.get("result") or {}
    if validated.get("coverage") != result.get("coverage"):
        raise ValueError("background checkpoint candidate coverage differs")
    datasets = result.get("datasets") or {}
    for dataset, row in datasets.items():
        quota = gate_authority.load_authority_manifest(
            final / f"quota-{dataset}.json")
        macro = gate_authority.load_authority_manifest(
            final / f"six_task_macro-{dataset}.json")
        if (not isinstance(row, Mapping) or row.get("quota") != quota or
                row.get("six_task_macro") != macro):
            raise ValueError(
                "background checkpoint dataset result differs from files")
    macro = gate_authority.load_authority_manifest(
        final / "six_task_macro.json")
    replay = gate_authority.load_authority_manifest(final / "gt_replay.json")
    if (result.get("six_task_macro") != macro or
            result.get("gt_as_pred") != replay.get("overall")):
        raise ValueError("background checkpoint result differs from files")
    return summary
