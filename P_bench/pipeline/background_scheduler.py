"""Small asynchronous scheduler for immutable background scene jobs.

Jobs remain bound to the GPU, source, seed, and output directory frozen in the
controller manifest.  The scheduler only removes the artificial wave barrier:
when one GPU finishes, that GPU may start its next scene in the same catalog
pass while other GPUs continue their current scenes.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from typing import Callable, Iterable, Mapping, Sequence

from pipeline import background_capacity, background_pose_exclusions


RUNNING_STATUSES = frozenset({
    "running", "running_slow", "stopping_slow", "stopping_sigint",
    "stopping_sigterm", "stopping_sigkill",
})
TERMINAL_CATALOG_STATUSES = frozenset({
    "completed", "partial_valid", "zero_yield", "failed",
})


def bind_source_catalog(
        scheduled: Mapping[str, Sequence[str]],
        source: Mapping[str, Sequence[str]] | None,
        datasets: Sequence[str]) -> dict[str, list[str]]:
    """Bind complete source catalogs around scheduled train subsets."""
    values = source or scheduled
    result = {}
    for dataset in datasets:
        scenes = [str(scene).strip() for scene in values.get(dataset) or []]
        if not scenes or any(not scene for scene in scenes) or \
                len(scenes) != len(set(scenes)):
            raise ValueError(f"{dataset} source catalog is invalid")
        if not set(scheduled[dataset]).issubset(set(scenes)):
            raise ValueError(
                f"scheduled {dataset} scenes are outside source catalog")
        result[dataset] = scenes
    return result


def validate_catalog_binding(
        scheduled: Mapping, source: Mapping, datasets: Sequence[str],
        profile: Mapping) -> None:
    """Validate source counts and the train-only scheduled subset."""
    if set(scheduled) != set(datasets) or set(source) != set(datasets):
        raise ValueError("background collection scene catalog is invalid")
    for dataset in datasets:
        scenes = scheduled.get(dataset)
        source_scenes = source.get(dataset)
        if not isinstance(scenes, list) or not scenes or \
                len(scenes) != len(set(scenes)):
            raise ValueError(
                f"background collection {dataset} scene catalog is invalid")
        if not isinstance(source_scenes, list) or not source_scenes or \
                len(source_scenes) != len(set(source_scenes)) or not \
                set(scenes).issubset(set(source_scenes)):
            raise ValueError(
                f"background collection {dataset} source catalog is invalid")
        if profile["datasets"][dataset]["catalog_scene_count"] != len(
                source_scenes):
            raise ValueError(
                f"background collection {dataset} capacity count differs")
        canary_ids = {
            row["scene_id"]
            for row in profile["datasets"][dataset]["canary_scenes"]}
        if not canary_ids.issubset(set(scenes)):
            raise ValueError(
                f"background collection {dataset} canary differs from catalog")


def default_launcher(job: dict, environment: Mapping[str, str], log_path):
    """Launch one manifest-bound scene transaction in its own process group."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            job["command"], cwd=job.get("cwd"),
            env={**os.environ, **dict(environment)}, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        stream.close()
    return process


def jobs_for_pass(manifest: Mapping, catalog_pass: int) -> list[dict]:
    """Return one catalog pass in deterministic manifest order."""
    return [
        job
        for round_value in manifest.get("rounds") or []
        if int(round_value["catalog_pass"]) == int(catalog_pass)
        for job in round_value.get("jobs") or []
    ]


def catalog_pass_complete(
        manifest: Mapping, state: Mapping, *, catalog_pass: int,
        completed_datasets: Iterable[str] = ()) -> bool:
    """Whether every required transaction in one catalog pass is terminal."""
    completed = set(completed_datasets)
    jobs = jobs_for_pass(manifest, catalog_pass)
    if not jobs:
        return False
    runtime = state.get("jobs") or {}
    return all(
        job["dataset"] in completed or
        (runtime.get(job["job_id"]) or {}).get("catalog_status") in
        TERMINAL_CATALOG_STATUSES
        for job in jobs
    )


def next_jobs(
        manifest: Mapping, state: Mapping, *, catalog_pass: int,
        completed_datasets: Iterable[str] = ()) -> list[dict]:
    """Choose at most one next manifest job per idle GPU.

    A later catalog pass never starts before every required transaction in all
    earlier passes is terminal.  Within a pass, each GPU advances through its
    own deterministic queue independently of other GPUs.
    """
    completed = set(completed_datasets)
    for earlier in range(int(catalog_pass)):
        if not catalog_pass_complete(
                manifest, state, catalog_pass=earlier,
                completed_datasets=completed):
            return []
    runtime = state.get("jobs") or {}
    busy_gpus = {
        int(row["gpu_id"])
        for row in runtime.values()
        if row.get("status") in RUNNING_STATUSES and "gpu_id" in row
    }
    queues: dict[int, list[dict]] = {}
    for job in jobs_for_pass(manifest, catalog_pass):
        queues.setdefault(int(job["gpu_id"]), []).append(job)
    ready = []
    for gpu_id in sorted(queues):
        if gpu_id in busy_gpus:
            continue
        for job in queues[gpu_id]:
            if job["dataset"] in completed:
                continue
            row = runtime.get(job["job_id"])
            if row is None:
                ready.append(job)
                break
            if row.get("catalog_status") in TERMINAL_CATALOG_STATUSES:
                continue
            # A known non-terminal transaction must be recovered or retried;
            # never run a later scene over its durable state.
            break
    return ready


def launch_jobs(
        manifest: Mapping, state: dict, jobs: Sequence[dict], *,
        launcher: Callable = default_launcher,
        initialize: Callable = background_capacity.initialize_controller_events,
        now: float = None) -> dict:
    """Launch selected jobs while preserving unrelated running GPUs."""
    timestamp = float(time.time() if now is None else now)
    gpu_ids = [int(job["gpu_id"]) for job in jobs]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("background scheduler launch reuses a GPU")
    busy = {
        int(row["gpu_id"])
        for row in (state.get("jobs") or {}).values()
        if row.get("status") in RUNNING_STATUSES and "gpu_id" in row
    }
    if busy & set(gpu_ids):
        raise ValueError("background scheduler launch targets a busy GPU")
    processes = {}
    for job in jobs:
        background_pose_exclusions.materialize_for_job(manifest, state, job)
        initialize(manifest, job, time_unix=timestamp)
        process = launcher(job, job["environment"], job["log_path"])
        processes[job["job_id"]] = process
        state.setdefault("jobs", {})[job["job_id"]] = {
            "job_id": job["job_id"],
            "dataset": job["dataset"],
            "scene_id": job["scenes"][0],
            "gpu_id": int(job["gpu_id"]),
            "round_index": int(job["round_index"]),
            "pid": int(process.pid),
            "status": "running",
            "attempt": 1,
            "started_time_unix": timestamp,
            "finished_time_unix": None,
            "durable_records": 0,
        }
    if processes:
        state["status"] = "running"
        active_rounds = [
            int(row["round_index"])
            for row in state["jobs"].values()
            if row.get("status") in RUNNING_STATUSES
        ]
        state["current_round"] = min(active_rounds)
    state["updated_time_unix"] = timestamp
    return processes
