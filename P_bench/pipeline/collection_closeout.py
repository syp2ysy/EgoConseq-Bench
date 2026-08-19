"""Label-blind capacity stopping and durable collection finalization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from pipeline import io_utils


FINALIZATION_SCHEMA = "egoconseq.collection-finalization.v2"
FINALIZATION_FILE = "collection_finalization.json"
TERMINAL_STATUSES = frozenset({"completed", "partial", "failed"})
SOURCE_VALIDATION_STATUSES = frozenset({
    "pending", "passed", "skipped", "failed",
})


def emit_backend_ready(
        *, backend: str, scene_id: str, time_unix: float = None) -> None:
    """Emit the point after simulator initialization, before pose work."""
    print(json.dumps({
        "event": "backend_ready",
        "backend": str(backend),
        "scene_id": str(scene_id),
        "time_unix": float(time.time() if time_unix is None else time_unix),
    }, sort_keys=True), flush=True)


def capacity_stop_descriptor(
        args, *, scene_started_mono: float, last_record_mono: float | None,
        accepted_records: int, pose_attempts: int,
        now_mono: float) -> dict | None:
    """Return an ordinary stop based only on time and accepted-record count."""
    active_seconds = max(0.0, float(now_mono) - float(scene_started_mono))
    wallclock = getattr(args, "scene_wallclock_stop_s", None)
    idle = getattr(args, "record_idle_stop_s", None)
    reason = None
    if wallclock is not None and active_seconds >= float(wallclock):
        reason = "scene_wallclock"
    elif idle is not None:
        reference = (
            float(last_record_mono)
            if last_record_mono is not None else float(scene_started_mono))
        if float(now_mono) - reference >= float(idle):
            reason = (
                "inter_record_idle" if last_record_mono is not None else
                "first_record_idle")
    if reason is None:
        return None
    return {
        "reason": reason,
        "accepted_records": int(accepted_records),
        "pose_attempts": int(pose_attempts),
        "active_seconds": round(active_seconds, 6),
        "record_idle_stop_s": None if idle is None else float(idle),
        "scene_wallclock_stop_s": (
            None if wallclock is None else float(wallclock)),
    }


def begin_finalization(
        output_dir, *, run_contract_sha256: str,
        capacity_stop=None) -> dict:
    """Publish the marker before validation or metadata I/O can block."""
    marker = {
        "schema": FINALIZATION_SCHEMA,
        "status": "finalizing",
        "pid": os.getpid(),
        "started_time_unix": float(time.time()),
        "run_contract_sha256": str(run_contract_sha256),
        "capacity_stop_reason": (
            None if capacity_stop is None else capacity_stop.get("reason")),
    }
    io_utils.atomic_write_json(
        Path(output_dir) / FINALIZATION_FILE, marker,
        sort_keys=False, durable=True)
    return marker


def seal_finalization(
        output_dir, marker: dict, *, records_path, run_meta_path, funnel_path,
        status: str, source_validation: str, record_count: int) -> None:
    """Bind one terminal verdict to the exact durable shard bytes."""
    terminal_status = str(status)
    validation_status = str(source_validation)
    if terminal_status not in TERMINAL_STATUSES:
        raise ValueError("collection finalization status is invalid")
    if validation_status not in SOURCE_VALIDATION_STATUSES - {"pending"}:
        raise ValueError("collection source-validation status is invalid")
    count = int(record_count)
    if count < 0:
        raise ValueError("collection finalization record count is invalid")
    records = Path(records_path)
    run_meta = Path(run_meta_path)
    funnel = Path(funnel_path)
    for path in (records, run_meta, funnel):
        if not path.is_file():
            raise ValueError(f"collection finalization input is absent: {path}")
    sealed = {
        **marker,
        "status": terminal_status,
        "source_validation": validation_status,
        "record_count": count,
        "finished_time_unix": float(time.time()),
        "records_sha256": io_utils.sha256_file(records),
        "run_meta_sha256": io_utils.sha256_file(run_meta),
        "funnel_sha256": io_utils.sha256_file(funnel),
    }
    io_utils.atomic_write_json(
        Path(output_dir) / FINALIZATION_FILE, sealed,
        sort_keys=False, durable=True)


def load_finalization(
        output_dir, *, expected_run_contract_sha256: str | None = None,
        allowed_statuses=None) -> dict:
    """Verify and return the collector-authenticated terminal shard atom.

    The collector already performed the unique source-bound record validation.
    Downstream consumers therefore verify these three file digests once instead
    of decoding and re-validating the same records again.
    """
    root = Path(output_dir)
    path = root / FINALIZATION_FILE
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("collection finalization is unreadable") from error
    if not isinstance(value, dict) or value.get("schema") != FINALIZATION_SCHEMA:
        raise ValueError("collection finalization schema is invalid")
    status = str(value.get("status") or "")
    accepted = TERMINAL_STATUSES if allowed_statuses is None else {
        str(item) for item in allowed_statuses}
    if status not in accepted:
        raise ValueError("collection finalization status is invalid")
    source_validation = str(value.get("source_validation") or "")
    if source_validation not in SOURCE_VALIDATION_STATUSES - {"pending"}:
        raise ValueError("collection source-validation status is invalid")
    count = value.get("record_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("collection finalization record count is invalid")
    contract_sha256 = str(value.get("run_contract_sha256") or "")
    if (expected_run_contract_sha256 is not None and
            contract_sha256 != str(expected_run_contract_sha256)):
        raise ValueError("collection run contract digest differs")
    inputs = (
        ("records", root / "records.jsonl", "records_sha256"),
        ("run metadata", root / "run_meta.json", "run_meta_sha256"),
        ("funnel", root / "collection_funnel.json", "funnel_sha256"),
    )
    for label, input_path, field in inputs:
        if not input_path.is_file() or \
                io_utils.sha256_file(input_path) != value.get(field):
            raise ValueError(f"collection {label} digest differs")
    return value
