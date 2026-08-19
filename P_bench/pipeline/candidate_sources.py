"""Canonical source atoms and record contexts for candidate QA artifacts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from pipeline import io_utils
from pipeline import record as record_fields
from pipeline import source_manifest
from pipeline import validate as record_validation


def canonical_record_image_path(value: object) -> str:
    """Return the stable record-relative identity of an initial RGB asset."""
    raw = str(value or "")
    relative = Path(raw)
    if (not raw or "\\" in raw or relative.is_absolute() or
            ".." in relative.parts or relative == Path(".")):
        raise ValueError("private input record image path is invalid")
    return relative.as_posix()


def canonicalize_private_input_assets(answers: Iterable[dict]) -> None:
    """Remove checkout/build-root identity from private input audit rows."""
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("private answer must be an object")
        asset = answer.get("input_asset")
        if not isinstance(asset, dict):
            raise ValueError("private answer input asset is missing")
        record_path = canonical_record_image_path(
            asset.get("record_image_path"))
        asset["record_image_path"] = record_path
        if "raw_path" in asset:
            asset["raw_path"] = record_path
        if asset.get("marked") is True:
            answer_id = str(answer.get("id") or "")
            expected_name = f"{answer_id}.png"
            derivative = Path(str(asset.get("path") or ""))
            if (not answer_id or derivative.parent.name != "marked_inputs" or
                    derivative.name != expected_name):
                raise ValueError("private marked input path is invalid")
            asset["path"] = f"marked_inputs/{expected_name}"
        else:
            asset["path"] = record_path


def resolve_record_asset(asset_root: Path, value: object) -> Path:
    """Resolve one shard-relative asset without allowing root escape."""
    path = Path(str(value))
    if path.is_absolute():
        return path
    root = Path(asset_root).resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("record asset escapes its shard root") from error
    return resolved


def canonical_sha256(value: object) -> str:
    """Hash one canonical JSON value."""
    encoded = json.dumps(
        record_fields.json_value(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CandidateSourceSnapshot:
    """One authenticated, decoded, locally validated candidate source."""

    source_index: int
    records_path: Path
    records_sha256: str
    run_meta_sha256: str
    metadata: dict
    context: object
    records: tuple[dict, ...]
    record_sha256_by_identity: dict[int, str]


def load_candidate_source(
        records_path, *, expected_run_meta_sha256: str,
        source_authority, source_index: int) -> CandidateSourceSnapshot:
    """Read, decode, and locally validate one finalized shard exactly once."""
    path = Path(records_path).resolve()
    source = source_authority.sources[int(source_index)]
    if (Path(source["path"]) != path or
            source["run_meta_sha256"] != expected_run_meta_sha256):
        raise ValueError("candidate source authority order differs")
    records_sha256 = io_utils.sha256_file(path)
    if records_sha256 != source.get("records_sha256"):
        raise ValueError("candidate source records digest changed")
    metadata, actual_meta_sha256 = source_manifest.load_object_identity(
        path.with_name("run_meta.json"), label="candidate run metadata")
    if actual_meta_sha256 != expected_run_meta_sha256:
        raise ValueError("run metadata digest does not match authority")
    params = metadata.get("params") or {}
    backend = params.get("backend")
    if backend not in {"r2r", "b1k", "gs"} or \
            params.get("collection_mode") != "main":
        raise ValueError(
            "candidate source must be an R2R, B1K, or GS main shard")
    context_builder = {
        "r2r": source_manifest.r2r_v16_context_from_run_meta,
        "b1k": source_manifest.b1k_v16_context_from_run_meta,
        "gs": source_manifest.gs_v18_context_from_run_meta,
    }[backend]
    context = context_builder(
        metadata, authority_sha256=expected_run_meta_sha256)
    records = tuple(record_fields.decode_records_for_schema(
        path.read_bytes(), context.expected_schema_version))
    violations = [
        error
        for record in records
        for error in record_validation.validate_record_local(
            record, context=context, asset_root=path.parent)
    ]
    violations.extend(record_validation.validate_intervention_groups(records))
    if violations:
        details = "; ".join(violations[:3])
        raise ValueError(
            f"candidate source record validation failed: {details}")
    return CandidateSourceSnapshot(
        source_index=int(source_index),
        records_path=path,
        records_sha256=records_sha256,
        run_meta_sha256=actual_meta_sha256,
        metadata=metadata,
        context=context,
        records=records,
        record_sha256_by_identity={
            id(record): canonical_sha256(record) for record in records},
    )


def record_context(record: dict) -> dict:
    """Return the immutable record fields shared by all its outcomes."""
    return {
        key: copy.deepcopy(value)
        for key, value in record.items() if key != "outcomes"
    }


def source_atom(
        record: dict, outcome: dict, *, record_sha256: str | None = None
        ) -> dict:
    """Build one outcome atom that references its deduplicated record."""
    payload = {
        "schema": "egoconseq.qa-source-atom.v17-preview",
        "frame_id": record.get("frame_id"),
        "outcome_id": outcome.get("outcome_id"),
        "record_sha256": (
            record_sha256 if record_sha256 is not None
            else canonical_sha256(record)),
        "outcome": copy.deepcopy(outcome),
    }
    payload["outcome_sha256"] = canonical_sha256(payload["outcome"])
    payload["id"] = "atom-" + canonical_sha256(payload)[:24]
    return record_fields.json_value(payload)


def collect_record_context(
        contexts: dict, record: dict, *, record_sha256: str | None = None,
        context: dict | None = None) -> None:
    """Store one unambiguous context under its complete-record digest."""
    digest = (
        record_sha256 if record_sha256 is not None
        else canonical_sha256(record))
    value = (
        context if context is not None
        else record_fields.json_value(record_context(record)))
    previous = contexts.get(digest)
    if previous is not None and previous != value:
        raise ValueError(
            f"two distinct record contexts share sha256 {digest}")
    contexts[digest] = value


def validate_record_contexts(
        rows: list[dict], atoms: dict, source_rows: dict) -> dict:
    """Rebuild the deduplicated context table from authenticated sources."""
    digests = [str(row.get("record_sha256")) for row in rows]
    if digests != sorted(digests) or len(set(digests)) != len(digests):
        raise ValueError("preview record contexts are not canonically ordered")
    expected = {}
    seen_records = set()
    for atom in atoms.values():
        key = (str(atom["frame_id"]), str(atom["outcome_id"]))
        record, _outcome, _asset_root, digest = source_rows[key]
        if digest in seen_records:
            continue
        seen_records.add(digest)
        collect_record_context(expected, record, record_sha256=digest)
    stored = {str(row["record_sha256"]): row.get("context") for row in rows}
    if stored != expected:
        raise ValueError("preview record context content changed")
    return stored
