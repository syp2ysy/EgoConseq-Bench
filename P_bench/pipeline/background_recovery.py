"""Process recovery primitives for the background collection controller."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from pipeline import (
    background_collection, background_scheduler, collection_closeout,
)


class RecoveredProcess:
    """Minimal poll handle for a child surviving a controller restart."""

    def __init__(self, pid: int):
        self.pid = int(pid)

    def poll(self):
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return background_collection.RECOVERED_UNKNOWN_RETURNCODE
        return None

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.time() + float(timeout)
        while self.poll() is None:
            if deadline is not None and time.time() >= deadline:
                raise subprocess.TimeoutExpired("recovered-process", timeout)
            time.sleep(0.1)
        return background_collection.RECOVERED_UNKNOWN_RETURNCODE


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    return True


def finalization_status(job: dict) -> str | None:
    root = Path(job["output_dir"])
    if job["dataset"] == "b1k":
        root /= job["scenes"][0]
    path = root / "collection_finalization.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    status = value.get("status")
    if status not in {"finalizing", *collection_closeout.TERMINAL_STATUSES}:
        raise ValueError("collection finalization status is invalid")
    return str(status)


def settled_runtime_status(
        *, returncode: int, durable_records: int, catalog_status: str,
        stopped_for_capacity: bool) -> str:
    """Name a terminal runtime only after transaction validation."""
    if catalog_status == "zero_yield":
        return (
            "completed_zero_yield"
            if int(returncode) == 0 or stopped_for_capacity else
            "failed_zero_yield")
    if catalog_status in {"completed", "partial_valid"}:
        if int(returncode) == 0:
            return "completed"
        if stopped_for_capacity:
            return "completed_capacity_shortfall"
    return "failed_partial" if int(durable_records) else "failed_zero_yield"


def settle_recovered_runtime(
        definition: dict, runtime: dict, *, now: float) -> None:
    returncode = background_collection.RECOVERED_UNKNOWN_RETURNCODE
    transaction = background_collection.validate_scene_transaction(
        definition, returncode=returncode)
    catalog_status = transaction.pop("catalog_status")
    terminal_status = transaction.pop("terminal_status", None)
    if catalog_status not in {"completed", "partial_valid"}:
        raise ValueError(f"recovered scene transaction is {catalog_status}")
    count = int(transaction["source_validated_records"])
    runtime.update({
        "status": terminal_status or settled_runtime_status(
            returncode=returncode, durable_records=count,
            catalog_status=catalog_status, stopped_for_capacity=False),
        "catalog_status": catalog_status,
        "source_validation": transaction,
        "durable_records": count,
        "returncode": returncode,
        "finished_time_unix": float(now),
    })
    runtime.pop("source_validation_error", None)


def restore_running_processes(
        state: dict, definitions: dict[str, dict], processes: dict, *,
        now: float,
        launcher=background_scheduler.default_launcher,
        ) -> tuple[list[dict], bool]:
    """Attach, adopt, or resume each manifest-bound running transaction."""
    terminal = []
    changed = False
    for job_id, runtime in state.get("jobs", {}).items():
        if (job_id not in definitions or runtime.get("status") not in
                background_scheduler.RUNNING_STATUSES):
            continue
        definition = definitions[job_id]
        pid = int(runtime["pid"])
        if pid_alive(pid):
            processes[job_id] = RecoveredProcess(pid)
            continue
        if finalization_status(definition) in \
                collection_closeout.TERMINAL_STATUSES:
            settle_recovered_runtime(definition, runtime, now=now)
            terminal.append(definition)
            changed = True
            continue
        process = launcher(
            definition, definition["environment"], definition["log_path"])
        processes[job_id] = process
        runtime.update({
            "pid": int(process.pid),
            "status": "running",
            "started_time_unix": float(now),
            "finished_time_unix": None,
        })
        for name in (
                "capacity_backend_ready_recorded", "performance_violation",
                "termination_requested_time_unix", "termination_signal",
                "finalization_status", "finalization_started_time_unix",
                "sealed_seen_time_unix", "scene_wallclock_seen_time_unix"):
            runtime.pop(name, None)
        changed = True
    return terminal, changed


def wait_for_process_event(
        processes: dict, *, poll_s: float,
        heartbeat_deadline: float) -> tuple[float, bool]:
    """Poll child exit cheaply; defer filesystem health work to heartbeats."""
    time.sleep(float(poll_s))
    now = time.time()
    return now, (
        any(process.poll() is not None for process in processes.values()) or
        now >= float(heartbeat_deadline))
