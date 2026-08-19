"""Signal policy for background collector initialization and closeout."""

from __future__ import annotations

import json
from pathlib import Path
import signal
from typing import Mapping

from pipeline import config
from pipeline import collection_closeout


def b1k_supervisor_timeout_s(scene_wallclock_s: float) -> int:
    """Bound B1K init, active collection, cooperative close, and sealing."""
    return int(
        config.BACKGROUND_INITIALIZATION_DEADLINE_S +
        float(scene_wallclock_s) +
        config.BACKGROUND_CAPACITY_HANDOFF_GRACE_S +
        config.BACKGROUND_FINALIZATION_GRACE_S)


def health_requires_termination(status: str) -> bool:
    """Return true only for a genuine hang, not normal capacity exhaustion."""
    return str(status) == "initialization_slow"


def _finalization_marker(job: Mapping, *, started_time: float) -> dict | None:
    output = Path(str(job["output_dir"]))
    if str(job.get("dataset") or "") == "b1k":
        scenes = job.get("scenes") or []
        if len(scenes) != 1:
            return None
        output /= str(scenes[0])
    path = output / "collection_finalization.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return None
    if (value.get("schema") != collection_closeout.FINALIZATION_SCHEMA or
            value.get("status") not in (
                {"finalizing"} | collection_closeout.TERMINAL_STATUSES) or
            float(value.get("started_time_unix") or 0.0) <
            float(started_time)):
        return None
    return value


def capacity_watchdog_action(
        job: Mapping, runtime: dict, *, health_status: str,
        now: float):
    """Cooperate with an ordinary collector stop and protect finalization."""
    timestamp = float(now)
    marker = _finalization_marker(
        job, started_time=float(runtime["started_time_unix"]))
    if marker is not None:
        runtime["finalization_status"] = marker["status"]
        runtime["finalization_started_time_unix"] = float(
            marker["started_time_unix"])
        if marker["status"] in collection_closeout.TERMINAL_STATUSES:
            sealed_seen = runtime.setdefault(
                "sealed_seen_time_unix", timestamp)
            if timestamp - float(sealed_seen) < \
                    config.BACKGROUND_FINALIZATION_GRACE_S:
                return None
            return _force_kill(runtime, timestamp, "sealed_exit_timeout")
        if timestamp - float(marker["started_time_unix"]) < \
                config.BACKGROUND_FINALIZATION_GRACE_S:
            return None
        return _force_kill(runtime, timestamp, "finalization_timeout")
    if str(health_status) in {"first_record_slow", "inter_record_slow"}:
        return None
    if str(health_status) != "scene_wallclock_reached" or \
            str(job.get("dataset") or "") == "b1k":
        return None
    handoff_started = runtime.setdefault(
        "scene_wallclock_seen_time_unix", timestamp)
    if timestamp - handoff_started < \
            config.BACKGROUND_FINALIZATION_GRACE_S:
        return None
    runtime["status"] = "stopping_sigint"
    runtime["performance_violation"] = "scene_wallclock_reached"
    runtime["termination_requested_time_unix"] = timestamp
    runtime["termination_signal"] = "SIGINT"
    return signal.SIGINT


def _force_kill(runtime: dict, timestamp: float, reason: str):
    runtime["status"] = "stopping_sigkill"
    runtime["performance_violation"] = str(reason)
    runtime["termination_requested_time_unix"] = float(timestamp)
    runtime["termination_signal"] = "SIGKILL"
    return signal.SIGKILL


def wallclock_escalation(
        runtime: dict, *, now: float, scene_wallclock_s: float):
    """Advance the durable SIGINT/SIGTERM/SIGKILL deadline state."""
    timestamp = float(now)
    status = str(runtime.get("status") or "")
    if status == "stopping_sigkill":
        return None
    if status == "stopping_sigterm":
        requested = float(runtime["termination_requested_time_unix"])
        if timestamp - requested < config.BACKGROUND_SIGTERM_GRACE_S:
            return None
        return _force_kill(runtime, timestamp, "sigterm_timeout")
    if status in {"stopping_sigint", "stopping_slow"}:
        requested = float(runtime["termination_requested_time_unix"])
        if timestamp - requested < config.BACKGROUND_SIGINT_GRACE_S:
            return None
        runtime["status"] = "stopping_sigterm"
        runtime["termination_requested_time_unix"] = timestamp
        runtime["termination_signal"] = "SIGTERM"
        return signal.SIGTERM
    if timestamp - float(runtime["started_time_unix"]) < \
            float(scene_wallclock_s):
        return None
    runtime["status"] = "stopping_sigint"
    runtime["performance_violation"] = "scene_wallclock_reached"
    runtime["termination_requested_time_unix"] = timestamp
    runtime["termination_signal"] = "SIGINT"
    return signal.SIGINT


def controller_watchdog_action(
        job: Mapping, runtime: dict, *, health_status: str, now: float,
        scene_wallclock_s: float):
    """Return the sole controller signal decision for one live process."""
    if runtime.get("status") in {
            "stopping_slow", "stopping_sigint",
            "stopping_sigterm", "stopping_sigkill"}:
        return wallclock_escalation(
            runtime, now=now, scene_wallclock_s=scene_wallclock_s)
    requested = capacity_watchdog_action(
        job, runtime, health_status=health_status, now=now)
    if requested is not None:
        return requested
    if not health_requires_termination(health_status):
        return None
    runtime["status"] = "stopping_sigint"
    runtime["performance_violation"] = str(health_status)
    runtime["termination_requested_time_unix"] = float(now)
    runtime["termination_signal"] = "SIGINT"
    return signal.SIGINT
