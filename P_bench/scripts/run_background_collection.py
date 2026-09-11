#!/usr/bin/env python3
"""Build, run, inspect, or stop the local three-dataset collector."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import (  # noqa: E402
    background_canary, background_collection, background_pose_exclusions,
    background_recovery, background_scheduler, collection_closeout,
    b1k_source_builder, config, gate_authority, gs_semantic,
    io_utils, scene_partitions, scene_pool,
)
from post_QA.seen_build import catalog as record_catalog  # noqa: E402
from post_QA.seen_build import selection as seen_selection  # noqa: E402
from post_QA.seen_build import supply as seen_supply  # noqa: E402


def _manifest_path(args) -> Path:
    return Path(args.manifest).resolve()


def _load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    background_collection.validate_manifest(value)
    return value


def _state_path(manifest: dict) -> Path:
    return Path(manifest["output_root"]) / "controller" / "state.json"


def _load_state(manifest: dict) -> dict:
    path = _state_path(manifest)
    if not path.is_file():
        return background_collection.initial_state(manifest)
    value = json.loads(path.read_text())
    background_collection.validate_state(manifest, value)
    return value


def _load_control_manifest(path: Path) -> dict:
    """Load only the manifest identity needed by status and stop."""
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("background controller manifest is unreadable") \
            from error
    body = {key: item for key, item in value.items() if key != "sha256"}
    if (value.get("schema") != background_collection.CONTROLLER_SCHEMA or
            value.get("sha256") !=
            background_collection._canonical_sha256(body) or
            not isinstance(value.get("rounds"), list) or
            not str(value.get("output_root") or "")):
        raise ValueError("background controller manifest identity is invalid")
    return value


def _load_control_state(manifest: dict) -> dict:
    """Load local controller shape without reopening compiled artifacts."""
    path = _state_path(manifest)
    if not path.is_file():
        return {
            "schema": background_collection.STATE_SCHEMA,
            "manifest_sha256": manifest["sha256"],
            "status": "ready", "current_round": 0, "jobs": {},
            "datasets": {}, "compile_checkpoints": [],
            "updated_time_unix": None,
        }
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("background controller state is unreadable") from error
    if (not isinstance(value, dict) or
            value.get("schema") != background_collection.STATE_SCHEMA or
            value.get("manifest_sha256") != manifest["sha256"] or
            not isinstance(value.get("current_round"), int) or
            not isinstance(value.get("jobs"), dict) or
            not isinstance(value.get("datasets"), dict) or
            not isinstance(value.get("compile_checkpoints"), list)):
        raise ValueError("background controller state shape is invalid")
    return value


def _controller_pid_path(manifest: dict) -> Path:
    return Path(manifest["output_root"]) / "controller" / "controller.pid"


def _controller_pid(manifest: dict) -> int | None:
    try:
        return int(_controller_pid_path(manifest).read_text().strip())
    except (OSError, TypeError, ValueError):
        return None


def _interrupt_controller(
        manifest: dict, state: dict, processes: dict, *, now: float) -> None:
    """Reap live collectors before publishing one interrupted state."""
    for job_id, process in list(processes.items()):
        _terminate_canary_process(process)
        runtime = (state.get("jobs") or {}).get(job_id)
        if isinstance(runtime, dict) and runtime.get("status") in \
                background_scheduler.RUNNING_STATUSES:
            runtime["status"] = "interrupted"
            runtime["finished_time_unix"] = float(now)
    processes.clear()
    state["status"] = "interrupted"
    _write_state(manifest, state)


def _write_state(manifest: dict, state: dict) -> None:
    state["updated_time_unix"] = time.time()
    io_utils.atomic_write_json(
        _state_path(manifest), state, allow_nan=False, durable=True)


def reconcile_recovery_state(manifest: dict, state: dict) -> tuple[dict, dict]:
    """Reauthenticate recoverable shards in a sidecar state copy."""
    sidecar = copy.deepcopy(state)
    recovered_shards = {
        dataset: 0 for dataset in background_collection.DATASETS}
    recovered_records = {
        dataset: 0 for dataset in background_collection.DATASETS}
    report = {
        "attempted_shards": 0,
        "recovered_shards": recovered_shards,
        "recovered_records": recovered_records,
        "rejected": [],
    }
    definitions = {
        job["job_id"]: job for round_value in manifest.get("rounds") or []
        for job in round_value.get("jobs") or []}
    if background_collection.is_continuous(manifest):
        for job_id, runtime in sidecar.get("jobs", {}).items():
            if job_id in definitions or not isinstance(runtime, dict):
                continue
            definitions[job_id] = background_collection.continuous_job(
                manifest, runtime["dataset"],
                catalog_pass=int(runtime["catalog_pass"]),
                scene_index=int(runtime["scene_index"]))
    for job_id, job in definitions.items():
        runtime = sidecar.get("jobs", {}).get(job_id)
        if not isinstance(runtime, dict):
            continue
        binding_failure = (
            runtime.get("catalog_status") == "failed" and
            int(runtime.get("durable_records") or 0) >= 1 and
            "transaction binding differs" in str(
                runtime.get("source_validation_error") or ""))
        dead_running = (
            runtime.get("status") in background_scheduler.RUNNING_STATUSES and
            not background_recovery.pid_alive(int(runtime["pid"])))
        if not binding_failure and not dead_running:
            continue
        if dead_running and background_recovery.finalization_status(job) not in \
                collection_closeout.TERMINAL_STATUSES:
            continue
        report["attempted_shards"] += 1
        try:
            if dead_running:
                background_recovery.settle_recovered_runtime(
                    job, runtime, now=time.time())
            else:
                transaction = \
                    background_collection.validate_scene_transaction(
                        job, returncode=int(runtime.get("returncode") or 0))
                catalog_status = transaction.pop("catalog_status")
                transaction.pop("terminal_status", None)
                if catalog_status not in {"completed", "partial_valid"}:
                    raise ValueError(f"revalidated as {catalog_status}")
                runtime["catalog_status"] = catalog_status
                runtime["source_validation"] = transaction
                runtime["durable_records"] = int(
                    transaction["source_validated_records"])
                runtime.pop("source_validation_error", None)
        except (OSError, TypeError, ValueError) as error:
            report["rejected"].append({
                "job_id": job_id, "error": str(error)})
            continue
        dataset = str(job["dataset"])
        count = int(runtime["durable_records"])
        recovered_shards[dataset] += 1
        recovered_records[dataset] += count
    return sidecar, report


def _clean_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip()
    if changed:
        raise ValueError("background collection requires a clean tracked tree")
    return revision


def _gpu_memory_used_mib() -> list[int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def _rank_gs_schedule_rows(rows) -> list[dict]:
    """Prefer lower-cost, higher-quality GS render sources.

    Gaussian count is a static source-cost proxy for gsplat rendering.
    Gaussian quality diagnostics only order independent shards; they never
    alter pose-local proposals, collision authority, or labels.
    """
    return sorted((dict(row) for row in rows), key=lambda row: (
        float(row["fraction_above_quality_band"]) >
        config.GS_SUPPORT_RADIUS_OUTLIER_FRACTION,
        float(row["support_radius_p90_m"]) >
        config.GS_SUPPORT_RADIUS_QUALITY_M,
        int(row["gaussian_count"]),
        float(row["fraction_above_quality_band"]),
        float(row["support_radius_p90_m"]),
        str(row["scene_id"]),
    ))


def _ranked_gs_scenes(specs) -> list[str]:
    rows = []
    for index, spec in enumerate(specs, 1):
        print(json.dumps({
            "event": "gs_scene_diagnostic",
            "scene_id": spec.scene_id,
            "index": index,
            "total": len(specs),
        }, sort_keys=True), flush=True)
        rows.append(gs_semantic.scene_diagnostics(
            spec.provenance(), spec.scene_path, spec.semantic_path))
    return [row["scene_id"] for row in _rank_gs_schedule_rows(rows)]


def _controller_gs_specs(root: Path, manifest_path: Path):
    """Discover the approved 54-scene catalog with one bound exclusion."""
    manifest = gate_authority.load_authority_manifest(manifest_path)
    rows = manifest.get("scenes") or []
    train_rows = [row for row in rows
                  if isinstance(row, dict) and row.get("split") == "train"]
    by_id = {str(row.get("scene_id") or ""): row for row in train_rows}
    exclusions = config.BACKGROUND_GS_CATALOG_EXCLUSIONS
    if (len(by_id) != len(train_rows) or set(exclusions) - set(by_id) or
            len(train_rows) - len(exclusions) !=
            config.BACKGROUND_GS_CATALOG_SCENE_COUNT):
        raise ValueError("controller GS manifest partition is invalid")
    for scene_id, reason in exclusions.items():
        row = by_id[scene_id]
        relative = Path(str(row.get("path") or ""))
        directory = (Path(root).resolve() / relative).resolve()
        try:
            directory.relative_to(Path(root).resolve())
        except ValueError as error:
            raise ValueError("controller GS exclusion escapes root") from error
        if (reason != "missing_authenticated_collision_artifact" or
                (directory / "scene.collision.npz").exists()):
            raise ValueError("controller GS exclusion evidence differs")
    requested = sorted(set(by_id) - set(exclusions))
    specs = scene_pool.discover_gs_train_scenes(
        root, manifest_path, requested=requested)
    if ({spec.scene_id for spec in specs} != set(requested) or
            len(specs) != config.BACKGROUND_GS_CATALOG_SCENE_COUNT):
        raise ValueError("controller GS accepted catalog differs")
    return specs


def _ranked_b1k_scenes(specs) -> list[str]:
    """Order trusted B1K scenes by exact semantic-query work.

    Runtime-instance triangle counts are frozen source metadata, not an
    oracle outcome.  Sorting on them only decides which independent shard to
    run first; it cannot alter a pose-local proposal or label distribution.
    """
    rows = []
    for spec in specs:
        instances = (spec.b1k_scene_authority or {}).get(
            "runtime_instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError(
                f"B1K scene {spec.scene_id!r} has no runtime instances")
        counts = [value.get("triangle_count")
                  for value in instances if isinstance(value, dict)]
        if (len(counts) != len(instances) or
                any(not isinstance(value, int) or value <= 0
                    for value in counts)):
            raise ValueError(
                f"B1K scene {spec.scene_id!r} has invalid triangle counts")
        rows.append((sum(counts), len(counts), str(spec.scene_id)))
    return [scene_id for _triangles, _instances, scene_id in sorted(rows)]


def _profile(args) -> int:
    measurements = json.loads(Path(args.measurements).read_text())
    profile = background_collection.derive_capacity_profile(measurements)
    io_utils.atomic_write_json(
        args.out, profile, allow_nan=False, durable=True)
    print(json.dumps({
        "profile": str(Path(args.out).resolve()),
        "sha256": profile["sha256"],
    }, indent=2, sort_keys=True))
    return 0


def _canary_paths(args) -> dict:
    return {
        "repository": str(ROOT),
        "python": str(args.python),
        "r2r_train_episodes": config.R2R_TRAIN_EPISODES,
        "mp3d_root": config.MP3D_ROOT,
        "gs_data_root": config.GS_ROOT,
        "gs_source_manifest": config.GS_TRAIN_MANIFEST,
        "b1k_python": str(b1k_source_builder.BEHAVIOR_PYTHON),
        "b1k_data_root": str(args.b1k_data_root),
        "b1k_source_manifest": str(args.b1k_source_manifest),
        # Normal job builders do not read these bootstrap-irrelevant paths.
        "b1k_catalog_audit": "capacity-canary-not-required",
        "capacity_profile": "capacity-canary-not-required",
    }


def _canary_build(args) -> int:
    revision = _clean_revision()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("capacity canary output root is not empty")
    r2r_specs = scene_pool.discover_r2r_train_scenes(
        config.R2R_TRAIN_EPISODES, config.MP3D_ROOT)
    gs_specs = _controller_gs_specs(
        Path(config.GS_ROOT), Path(config.GS_TRAIN_MANIFEST))
    b1k_specs = scene_pool.discover_b1k_train_scenes(
        args.b1k_data_root, args.b1k_source_manifest)
    catalogs = {
        "r2r": [row.scene_id for row in r2r_specs],
        "gs": [row.scene_id for row in gs_specs],
        "b1k": [row.scene_id for row in b1k_specs],
    }
    selected = {
        "r2r": list(args.r2r_scenes), "gs": list(args.gs_scenes),
        "b1k": list(args.b1k_scenes),
    }
    manifest = background_collection.build_canary_manifest(
        revision=revision, output_root=output_root,
        scene_catalog=catalogs, selected_scenes=selected,
        paths=_canary_paths(args),
        source_manifest_sha256={
            "r2r": io_utils.sha256_file(config.R2R_TRAIN_EPISODES),
            "gs": io_utils.sha256_file(config.GS_TRAIN_MANIFEST),
            "b1k": io_utils.sha256_file(args.b1k_source_manifest),
        },
        ordinary_actions_per_pose=args.ordinary_actions_per_pose)
    path = output_root / "controller" / "canary-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.atomic_write_json(path, manifest, allow_nan=False, durable=True)
    print(json.dumps({"manifest": str(path), "jobs": len(manifest["jobs"])},
                     indent=2, sort_keys=True))
    return 0


def _compile_canary_source(parent: dict, job: dict, source: dict) -> Path:
    final, staging = _canary_publication_paths(job)
    if final.is_dir():
        return _load_canary_publication(parent, job, source)[0]
    if final.exists():
        raise ValueError("capacity canary publication target is invalid")
    if staging.exists():
        if staging.resolve().parent != final.parent.resolve() or \
                staging.name != ".capacity_candidate.staging":
            raise ValueError("capacity canary staging path is invalid")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    catalog_root = staging / "records"
    catalog_root.mkdir()
    try:
        record_catalog.write(
            [{**source, "dataset": job["dataset"]}],
            catalog_root / "manifest.json")
        candidates = seen_selection.enumerate_candidates(
            catalog_root, seed=int(parent.get("seed") or 0))
        supply = {
            "schema": "egoconseq.abc1-canary-supply.v1",
            **seen_supply.analyze_candidates(
                candidates, seed=int(parent.get("seed") or 0)),
            "measurements": {
                "candidate_records": len({row.record_id for row in candidates}),
                "task_counts": {
                    task: sum(row.task_id == task for row in candidates)
                    for task in background_collection.SUPPORTED_TASKS
                },
            },
        }
        io_utils.atomic_write_json(
            staging / "supply.json", supply, allow_nan=False, durable=True)
        body = {
            "schema": _CANARY_PUBLICATION_SCHEMA,
            "parent_manifest_sha256": parent["sha256"],
            "job_id": job["job_id"],
            "source": dict(source),
            "artifact": "supply.json",
            "files": _canary_publication_files(staging),
        }
        publication = {**body, "sha256": _canonical_sha256(body)}
        io_utils.atomic_write_json(
            staging / _CANARY_PUBLICATION_NAME, publication,
            allow_nan=False, durable=True)
        staging.rename(final)
        io_utils.fsync_directory(final.parent)
    except BaseException:
        # A partial staging tree is controller-owned and safely rebuilt on the
        # next invocation.  A renamed final is deliberately left for adoption.
        raise
    return final / "supply.json"


_CANARY_PUBLICATION_SCHEMA = \
    background_collection.background_capacity.CANARY_PUBLICATION_SCHEMA
_CANARY_PUBLICATION_NAME = \
    background_collection.background_capacity.CANARY_PUBLICATION_NAME


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canary_publication_paths(job: dict) -> tuple[Path, Path]:
    root = Path(job["output_dir"]).resolve()
    final = root / "capacity_candidate"
    staging = root / ".capacity_candidate.staging"
    if final.parent != root or staging.parent != root:
        raise ValueError("capacity canary publication escapes job root")
    return final, staging


def _canary_publication_files(root: Path) -> dict[str, str]:
    return background_collection.background_capacity.\
        candidate_publication_files(root)


def _load_canary_publication(
        parent: dict, job: dict, source: dict, *, identity=None,
        ) -> tuple[Path, Path, dict]:
    final, _staging = _canary_publication_paths(job)
    return background_collection.background_capacity.\
        validate_canary_publication(
            final / "supply.json",
            parent_manifest_sha256=parent["sha256"],
            job_id=job["job_id"], source=source, identity=identity)


_CANARY_STATE_SCHEMA = "egoconseq.capacity-canary-state.v3"


def _canary_state_path(parent: dict) -> Path:
    return Path(parent["output_root"]) / "controller" / "canary-state.json"


def _write_canary_state(parent: dict, state: dict) -> None:
    path = _canary_state_path(parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.atomic_write_json(path, state, allow_nan=False, durable=True)


def _load_canary_state(parent: dict) -> dict:
    path = _canary_state_path(parent)
    if not path.is_file():
        return {
            "schema": _CANARY_STATE_SCHEMA,
            "parent_manifest_sha256": parent["sha256"],
            "status": "ready", "jobs": {}, "failures": {},
            "attempts": {},
        }
    state = gate_authority.load_authority_manifest(path)
    if (state.get("schema") != _CANARY_STATE_SCHEMA or
            state.get("parent_manifest_sha256") != parent["sha256"] or
            state.get("status") not in {
                "ready", "running", "interrupted", "capacity_shortfall",
                "complete"} or not isinstance(state.get("jobs"), dict) or
            not isinstance(state.get("failures"), dict) or
            not isinstance(state.get("attempts"), dict)):
        raise ValueError("capacity canary state is invalid")
    jobs = {job["job_id"]: job for job in parent["jobs"]}
    max_attempts = 1 + int(
        background_collection.config.
        BACKGROUND_RETRYABLE_INITIALIZATION_ATTEMPTS)
    for job_id, attempts in state["attempts"].items():
        if (job_id not in jobs or type(attempts) is not int or
                attempts < 1 or attempts > max_attempts):
            raise ValueError("capacity canary state attempt is invalid")
    for job_id, row in state["jobs"].items():
        if job_id not in jobs or row.get("status") != "completed":
            raise ValueError("capacity canary state job is invalid")
        source = row.get("source") or {}
        records = Path(str(source.get("path") or ""))
        metadata = records.with_name("run_meta.json")
        if (not records.is_file() or not metadata.is_file() or
                io_utils.sha256_file(records) !=
                source.get("records_sha256") or
                io_utils.sha256_file(metadata) !=
                source.get("run_meta_sha256")):
            raise ValueError("capacity canary state source identity differs")
        artifact, report, _identity = _load_canary_publication(
            parent, jobs[job_id], source, identity=row.get("publication"))
        if (Path(str(row.get("artifact") or "")).resolve() != artifact or
                Path(str(row.get("report") or "")).resolve() != report):
            raise ValueError("capacity canary state publication path differs")
    allowed_phases = {
        "supervisor", "transaction", "source_validation",
        "candidate_build", "publication",
    }
    for job_id, row in state["failures"].items():
        if (job_id not in jobs or job_id in state["jobs"] or
                row.get("status") != "failed" or
                row.get("dataset") != jobs[job_id]["dataset"] or
                row.get("scene_id") != jobs[job_id]["scenes"][0] or
                row.get("phase") not in allowed_phases or
                not isinstance(row.get("error_type"), str) or
                not isinstance(row.get("error"), str) or
                type(row.get("attempt")) is not int or
                type(row.get("time_unix")) not in {int, float}):
            raise ValueError("capacity canary state failure is invalid")
    return state


def _start_canary_state_attempt(
        parent: dict, state: dict, job: dict, attempt: int) -> None:
    job_id = job["job_id"]
    expected = int(state["attempts"].get(job_id, 0)) + 1
    if int(attempt) != expected:
        raise ValueError("capacity canary attempt sequence differs")
    state["attempts"][job_id] = int(attempt)
    state["status"] = "running"
    _write_canary_state(parent, state)


def _complete_canary_state_job(
        parent: dict, state: dict, job: dict, artifact: Path,
        source: dict) -> None:
    reopened, report, publication = _load_canary_publication(
        parent, job, source)
    if Path(artifact).resolve() != reopened:
        raise ValueError("capacity canary completed artifact path differs")
    state["jobs"][job["job_id"]] = {
        "status": "completed", "dataset": job["dataset"],
        "scene_id": job["scenes"][0], "artifact": str(artifact),
        "report": str(report), "publication": publication,
        "source": dict(source),
    }
    state["status"] = "running"
    _write_canary_state(parent, state)


def _fail_canary_state_job(
        parent: dict, state: dict, job: dict, *, phase: str,
        error: Exception, attempt: int, time_unix: float) -> None:
    state["failures"][job["job_id"]] = {
        "status": "failed", "dataset": job["dataset"],
        "scene_id": job["scenes"][0], "phase": str(phase),
        "error_type": type(error).__name__, "error": str(error),
        "attempt": int(attempt), "time_unix": float(time_unix),
    }
    state["status"] = "capacity_shortfall"
    _write_canary_state(parent, state)


def _require_positive_canary_records(record_count: int) -> None:
    """Admit measurable canary shortfalls while rejecting zero-yield runs."""
    if int(record_count) <= 0:
        raise ValueError("capacity canary produced no accepted records")


def _terminate_canary_process(process) -> None:
    """Give a supervisor time to reap its isolated worker before escalation."""
    stages = (
        (signal.SIGINT, config.BACKGROUND_SIGINT_GRACE_S),
        (signal.SIGTERM, config.BACKGROUND_SIGTERM_GRACE_S),
    )
    for requested, grace_s in stages:
        try:
            background_collection.terminate_process_group(
                process.pid, requested)
        except ProcessLookupError:
            return
        if process.poll() is not None:
            return
        try:
            process.wait(timeout=float(grace_s))
            return
        except subprocess.TimeoutExpired:
            continue
    try:
        background_collection.terminate_process_group(
            process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _run_canary_wave(parent_path: Path, parent: dict,
                     jobs: list[dict], *, completed=None, attempts=None,
                     attempt_started=None, failed=None,
                     ) -> dict[str, tuple[Path, dict]]:
    """Run one deterministic four-GPU canary wave through normal health APIs."""
    processes = {}
    runtimes = {}

    def launch(job, *, attempt: int):
        if attempt_started is not None:
            attempt_started(job, int(attempt))
        started = time.time()
        background_collection.initialize_capacity_events(
            parent, job, parent_manifest_path=parent_path,
            time_unix=started)
        process = background_collection._default_launcher(
            job, job["environment"], job["log_path"])
        processes[job["job_id"]] = process
        runtimes[job["job_id"]] = {
            "status": "running", "started_time_unix": started,
            "attempt": int(attempt),
        }

    attempts = {} if attempts is None else attempts
    max_attempts = 1 + int(
        background_collection.config.
        BACKGROUND_RETRYABLE_INITIALIZATION_ATTEMPTS)
    for job in jobs:
        if int(attempts.get(job["job_id"], 0)) >= max_attempts:
            raise ValueError("capacity canary retry budget exhausted")
    for job in jobs:
        launch(job, attempt=int(attempts.get(job["job_id"], 0)) + 1)
    results = {}
    failures = []
    by_id = {job["job_id"]: job for job in jobs}

    def fail_job(job, runtime, phase, error, now):
        failures.append((job["job_id"], phase, str(error)))
        if failed is not None:
            failed(
                job, phase, error, int(runtime["attempt"]), float(now))
        del processes[job["job_id"]]

    try:
        while processes:
            time.sleep(1.0)
            now = time.time()
            for job_id, process in list(processes.items()):
                job = by_id[job_id]
                runtime = runtimes[job_id]
                health = background_collection.job_health(
                    job, now=now, started_time=runtime["started_time_unix"],
                    scene_wallclock_s=job["scene_wallclock_s"])
                ready = health.get("backend_ready_time")
                if ready is not None and not runtime.get("ready_recorded"):
                    background_collection.record_capacity_event(
                        parent, job, "backend_ready", time_unix=ready)
                    runtime["ready_recorded"] = True
                requested = None
                if process.poll() is None:
                    requested = \
                        background_collection.controller_watchdog_action(
                            job, runtime, health_status=health["status"],
                            now=now,
                            scene_wallclock_s=job["scene_wallclock_s"])
                if requested is not None:
                    try:
                        background_collection.terminate_process_group(
                            process.pid, requested)
                    except ProcessLookupError:
                        pass
                returncode = process.poll()
                if returncode is None:
                    continue
                retryable = background_collection.retryable_failure(
                    job["log_path"], failure_status=runtime.get(
                        "performance_violation"))
                if retryable and int(runtime["attempt"]) <= int(
                        background_collection.config.
                        BACKGROUND_RETRYABLE_INITIALIZATION_ATTEMPTS):
                    launch(job, attempt=int(runtime["attempt"]) + 1)
                    continue
                if not runtime.get("ready_recorded"):
                    fail_job(
                        job, runtime, "supervisor",
                        ValueError(
                            "capacity canary lacks backend_ready event"),
                        now)
                    continue
                background_collection.record_capacity_event(
                    parent, job, "controller_scene_finished", time_unix=now)
                try:
                    transaction = \
                        background_collection.validate_scene_transaction(
                        job, returncode=returncode,
                        stopped_for_capacity=runtime.get(
                            "performance_violation") in {
                                "first_record_slow", "inter_record_slow",
                                "scene_wallclock_reached"})
                except Exception as error:
                    fail_job(job, runtime, "transaction", error, now)
                    continue
                if transaction["catalog_status"] not in {
                        "completed", "partial_valid"}:
                    error = ValueError(
                        "capacity canary produced no accepted records")
                    fail_job(job, runtime, "source_validation", error, now)
                    continue
                sources = transaction.get("sources") or []
                try:
                    _require_positive_canary_records(
                        int(transaction.get("source_validated_records") or 0))
                    if len(sources) != 1:
                        raise ValueError(
                            "capacity canary must produce exactly one source")
                except Exception as error:
                    fail_job(job, runtime, "source_validation", error, now)
                    continue
                try:
                    artifact = _compile_canary_source(parent, job, sources[0])
                except Exception as error:
                    fail_job(job, runtime, "candidate_build", error, now)
                    continue
                results[job_id] = (artifact, sources[0])
                if completed is not None:
                    try:
                        completed(job, artifact, sources[0])
                    except Exception as error:
                        results.pop(job_id, None)
                        fail_job(job, runtime, "publication", error, now)
                        continue
                del processes[job_id]
        if failures and failed is None:
            raise ValueError("; ".join(
                f"{job_id} {phase}: {error}"
                for job_id, phase, error in failures))
    finally:
        for process in processes.values():
            if process.poll() is not None:
                continue
            _terminate_canary_process(process)
    return results


def run_canary_manifest(manifest_path: Path, *, execute_job=None) -> dict:
    """Run six canaries in two fixed-dataset GPU waves."""
    path = Path(manifest_path).resolve()
    parent = background_canary.load(path)
    state = _load_canary_state(parent)
    state["status"] = "running"
    _write_canary_state(parent, state)
    datasets = {
        dataset: {
            "catalog_scene_count": len(parent["scene_catalog"][dataset]),
            "canary_scenes": [],
        } for dataset in background_collection.DATASETS
    }
    source_rows = []
    pending = []
    real_results = {
        job_id: (Path(row["artifact"]), dict(row["source"]))
        for job_id, row in state["jobs"].items()
    }
    try:
        if execute_job is None:
            waves = sorted({
                int(job["canary_wave"]) for job in parent["jobs"]
                if job["job_id"] not in real_results})
            for wave in waves:
                jobs = [job for job in parent["jobs"]
                        if int(job["canary_wave"]) == wave and
                        job["job_id"] not in real_results]
                real_results.update(_run_canary_wave(
                    path, parent, jobs, attempts=state["attempts"],
                    attempt_started=lambda job, attempt:
                    _start_canary_state_attempt(
                        parent, state, job, attempt),
                    completed=lambda job, artifact, source:
                    _complete_canary_state_job(
                        parent, state, job, artifact, source),
                    failed=lambda job, phase, error, attempt, failed_time:
                    _fail_canary_state_job(
                        parent, state, job, phase=phase, error=error,
                        attempt=attempt, time_unix=failed_time)))
                if state["failures"]:
                    raise ValueError(
                        "capacity shortfall: " + "; ".join(
                            f"{job_id} {row['phase']}: {row['error']}"
                            for job_id, row in sorted(
                                state["failures"].items())))
        for job in parent["jobs"]:
            if job["job_id"] in real_results:
                artifact, source = real_results[job["job_id"]]
            else:
                artifact = Path(execute_job(parent, job)).resolve()
                records = background_collection.job_progress(job)["record_paths"]
                if len(records) != 1:
                    raise ValueError(
                        "capacity canary must produce one record file")
                records_path = Path(records[0]).resolve()
                accepted = sum(
                    1 for line in records_path.read_text().splitlines()
                    if line.strip())
                _require_positive_canary_records(accepted)
                metadata = records_path.with_name("run_meta.json")
                source = {
                    "path": str(records_path),
                    "records_sha256": io_utils.sha256_file(records_path),
                    "run_meta_sha256": io_utils.sha256_file(metadata),
                }
                # Dependency-injected jobs are test probes, not resumable
                # production publications.  Production state is written only
                # by the authenticated bundle callback above.
            source_rows.append(source)
            records_path = Path(source["path"])
            controller_root = Path(job["output_dir"])
            pending.append((job, artifact, records_path, controller_root))
    except BaseException as error:
        state["status"] = (
            "capacity_shortfall"
            if "capacity shortfall" in str(error) else "interrupted")
        _write_canary_state(parent, state)
        raise
    authority_path = Path(parent["output_root"]) / \
        "capacity-source-authority.json"
    background_canary.write_source_authority(
        authority_path, parent, source_rows)
    for job, artifact, records_path, controller_root in pending:
        datasets[job["dataset"]]["canary_scenes"].append({
            "scene_id": job["scenes"][0],
            "records_path": str(records_path),
            "run_meta_path": str(records_path.with_name("run_meta.json")),
            "funnel_path": str(
                records_path.with_name("collection_funnel.json")),
            "controller_events_path": str(
                controller_root / "controller_events.jsonl"),
            "controller_manifest_path": str(
                controller_root / "capacity_controller_manifest.json"),
            "compiled_qa_path": str(artifact),
            "parent_manifest_path": str(path),
            "source_authority_path": str(authority_path),
        })
    evidence = {
        "schema": background_collection.background_capacity.EVIDENCE_SCHEMA,
        "parent_manifest_path": str(path),
        "source_authority_path": str(authority_path),
        "datasets": datasets,
    }
    evidence_path = Path(parent["output_root"]) / "capacity-evidence.json"
    io_utils.atomic_write_json(
        evidence_path, evidence, allow_nan=False, durable=True)
    state["status"] = "complete"
    _write_canary_state(parent, state)
    return evidence


def _canary_run(args) -> int:
    path = _manifest_path(args)
    parent = background_canary.load(path)
    if _clean_revision() != parent["revision"]:
        raise ValueError("capacity canary revision differs from clean HEAD")
    evidence = run_canary_manifest(path)
    print(json.dumps({
        "evidence": str(Path(parent["output_root"]) /
                        "capacity-evidence.json"),
        "source_authority": evidence["source_authority_path"],
    }, indent=2, sort_keys=True))
    return 0


def _load_catalog_audit(path: Path, expected_sha256: str) -> dict:
    value = gate_authority.load_authority_manifest(path)
    if io_utils.sha256_file(path) != str(expected_sha256):
        raise ValueError("B1K catalog audit digest differs")
    return value


def _load_capacity_profile(path: Path, expected_sha256: str) -> dict:
    value = gate_authority.load_authority_manifest(path)
    background_collection.validate_capacity_profile(value)
    if value["sha256"] != str(expected_sha256):
        raise ValueError("capacity profile digest differs")
    return value


def _build(args) -> int:
    revision = _clean_revision()
    output_root = Path(args.output_root).resolve()
    controller_root = output_root / "controller"
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("background collection output root is not empty")
    r2r_source_specs = scene_pool.deterministic_scene_order(
        scene_pool.discover_r2r_train_scenes(
            config.R2R_TRAIN_EPISODES, config.MP3D_ROOT),
        seed=args.collection_seed)
    gs_source_specs = _controller_gs_specs(
        Path(config.GS_ROOT), Path(config.GS_TRAIN_MANIFEST))
    b1k_source_specs = scene_pool.discover_b1k_train_scenes(
        args.b1k_data_root, args.b1k_source_manifest)
    partitions = scene_partitions.load()
    r2r_specs = partitions.select_catalog(
        "r2r", r2r_source_specs, benchmark_partition="train_seen")
    gs_specs = partitions.select_catalog(
        "gs", gs_source_specs, benchmark_partition="train_seen")
    b1k_specs = partitions.select_catalog(
        "b1k", b1k_source_specs, benchmark_partition="train_seen")
    catalog_audit = _load_catalog_audit(
        args.b1k_catalog_audit, args.b1k_catalog_audit_sha256)
    capacity_profile = _load_capacity_profile(
        args.capacity_profile, args.capacity_profile_sha256)
    if bool(args.baseline_checkpoint) != bool(
            args.baseline_checkpoint_sha256):
        raise ValueError(
            "baseline checkpoint path and SHA256 must be supplied together")
    if bool(args.seed_checkpoint) != bool(args.seed_checkpoint_sha256):
        raise ValueError(
            "seed checkpoint path and SHA256 must be supplied together")
    baseline_checkpoint = (
        Path(args.baseline_checkpoint).resolve()
        if args.baseline_checkpoint else None)
    seed_checkpoint = (
        Path(args.seed_checkpoint).resolve()
        if args.seed_checkpoint else baseline_checkpoint)
    seed_checkpoint_sha256 = (
        args.seed_checkpoint_sha256
        if args.seed_checkpoint else args.baseline_checkpoint_sha256)
    seed_identities = (
        background_pose_exclusions.build_seed_from_checkpoint(
            seed_checkpoint,
            expected_sha256=seed_checkpoint_sha256,
            output_dir=controller_root / "seed_pose_exclusions")
        if seed_checkpoint is not None else None)
    manifest = background_collection.build_manifest(
        revision=revision, output_root=output_root,
        r2r_scenes=[value.scene_id for value in r2r_specs],
        gs_scenes=_ranked_gs_scenes(gs_specs),
        b1k_scenes=_ranked_b1k_scenes(b1k_specs),
        source_scene_catalog={
            "r2r": [value.scene_id for value in r2r_source_specs],
            "gs": [value.scene_id for value in gs_source_specs],
            "b1k": [value.scene_id for value in b1k_source_specs],
        },
        paths={
            "repository": str(ROOT),
            "python": str(args.python),
            "r2r_train_episodes": config.R2R_TRAIN_EPISODES,
            "mp3d_root": config.MP3D_ROOT,
            "gs_data_root": config.GS_ROOT,
            "gs_source_manifest": config.GS_TRAIN_MANIFEST,
            "b1k_python": str(b1k_source_builder.BEHAVIOR_PYTHON),
            "b1k_data_root": str(args.b1k_data_root),
            "b1k_source_manifest": str(args.b1k_source_manifest),
            "b1k_catalog_audit": str(args.b1k_catalog_audit),
            "capacity_profile": str(args.capacity_profile),
        },
        b1k_catalog_audit=catalog_audit,
        b1k_catalog_audit_sha256=args.b1k_catalog_audit_sha256,
        r2r_source_manifest_sha256=io_utils.sha256_file(
            config.R2R_TRAIN_EPISODES),
        gs_source_manifest_sha256=io_utils.sha256_file(
            config.GS_TRAIN_MANIFEST),
        b1k_source_manifest_sha256=io_utils.sha256_file(
            args.b1k_source_manifest),
        b1k_source_scene_ids=[value.scene_id for value in b1k_source_specs],
        capacity_profile=capacity_profile,
        capacity_profile_sha256=args.capacity_profile_sha256,
        seed_pose_exclusions=seed_identities,
        baseline_checkpoint=(
            {"path": str(baseline_checkpoint),
             "sha256": args.baseline_checkpoint_sha256}
            if baseline_checkpoint is not None else None),
        target_pose_diverse_frames=args.target_pose_diverse_frames,
        collection_seed=args.collection_seed,
        saturated_datasets=args.saturated_dataset,
        rounds=args.rounds)
    controller_root.mkdir(parents=True, exist_ok=True)
    io_utils.atomic_write_json(
        controller_root / "manifest.json", manifest,
        allow_nan=False, durable=True)
    _write_state(manifest, background_collection.initial_state(manifest))
    print(json.dumps({
        "manifest": str(controller_root / "manifest.json"),
        "revision": revision,
        "rounds": len(manifest["rounds"]),
        "jobs_per_round": 4,
        "quota": manifest["quota"],
        "baseline_pose_diverse_frames": {
            dataset: (
                seed_identities[dataset]["representative_count"]
                if seed_identities is not None else 0)
            for dataset in background_collection.DATASETS
        },
        "target_pose_diverse_frames": args.target_pose_diverse_frames,
        "collection_seed": args.collection_seed,
        "saturated_datasets": sorted(args.saturated_dataset),
        "scheduled_scenes": {
            dataset: len({scene for round_value in manifest["rounds"]
                          for job in round_value["jobs"]
                          if job["dataset"] == dataset
                          for scene in job["scenes"]})
            for dataset in ("r2r", "b1k", "gs")
        },
    }, indent=2, sort_keys=True))
    return 0


def _status_value(manifest: dict, state: dict) -> dict:
    job_by_id = {
        job["job_id"]: job for round_value in manifest["rounds"]
        for job in round_value["jobs"]}
    jobs = {}
    for job_id, value in state.get("jobs", {}).items():
        definition = job_by_id.get(job_id)
        if definition is None and background_collection.is_continuous(
                manifest):
            try:
                definition = background_collection.continuous_job(
                    manifest, value["dataset"],
                    catalog_pass=int(value["catalog_pass"]),
                    scene_index=int(value["scene_index"]))
            except (KeyError, TypeError, ValueError):
                definition = None
        progress = (
            background_collection.job_progress(definition)
            if definition is not None and value.get("status") in
            background_scheduler.RUNNING_STATUSES else {})
        jobs[job_id] = {**value, **progress}
    seeds = manifest.get("seed_pose_exclusions") or {}
    saturated = set(
        (manifest.get("collection") or {}).get("saturated_datasets") or [])
    exhausted = set(state.get("exhausted_datasets") or [])
    collection_progress = {}
    for dataset in background_collection.DATASETS:
        baseline = seeds.get(dataset) or {}
        incremental = sum(
            int(row.get("durable_records") or 0)
            for job_id, row in jobs.items()
            if str(row.get("dataset") or
                   (job_by_id.get(job_id) or {}).get("dataset") or "") ==
            dataset)
        baseline_records = int(baseline.get("record_count") or 0)
        collection_progress[dataset] = {
            "baseline_records": baseline_records,
            "baseline_pose_diverse_frames": int(
                baseline.get("representative_count") or 0),
            "incremental_records": incremental,
            "total_records": baseline_records + incremental,
            "saturated": dataset in saturated,
            "exhausted": dataset in exhausted,
        }
    return {
        "schema": state["schema"],
        "status": state["status"],
        "current_round": state["current_round"],
        "dataset_cursors": state.get("dataset_cursors"),
        "exhausted_datasets": state.get("exhausted_datasets") or [],
        "progress": collection_progress,
        "jobs": jobs,
        "datasets": state.get("datasets") or {},
        "catalogs": {
            dataset: background_collection.dataset_catalog_coverage(
                manifest, state, dataset)
            for dataset in background_collection.DATASETS
        },
        "compile_checkpoints": state.get("compile_checkpoints") or [],
        "updated_time_unix": state.get("updated_time_unix"),
    }


def _status(args) -> int:
    manifest = _load_control_manifest(_manifest_path(args))
    state = _load_control_state(manifest)
    pid = _controller_pid(manifest)
    alive = pid is not None and background_recovery.pid_alive(pid)
    value = _status_value(manifest, state)
    value.update({
        "status": (
            "stale" if value["status"] == "running" and not alive
            else value["status"]),
        "state_status": state["status"],
        "controller_pid": pid,
        "controller_alive": alive,
    })
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _stop(args) -> int:
    manifest = _load_control_manifest(_manifest_path(args))
    pid = _controller_pid(manifest)
    if pid is None:
        print(json.dumps({"status": "not_running"}, indent=2))
        return 0
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        print(json.dumps({"status": "not_running", "pid": pid}, indent=2))
        return 0
    print(json.dumps({
        "status": "interrupt_requested", "controller_pid": pid}, indent=2))
    return 0


def _recover(args) -> int:
    """Reconcile recoverable shards into a sidecar, then compile it."""
    checkpoint = str(args.checkpoint)
    if not checkpoint or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in checkpoint):
        raise ValueError("recovery checkpoint name is invalid")
    manifest = _load_control_manifest(_manifest_path(args))
    original_state = _load_control_state(manifest)
    sidecar, report = reconcile_recovery_state(manifest, original_state)
    controller = Path(manifest["output_root"]) / "controller"
    controller.mkdir(parents=True, exist_ok=True)
    state_path = controller / f"{checkpoint}-state.json"
    report_path = controller / f"{checkpoint}-report.json"
    io_utils.atomic_write_json(
        state_path, sidecar, allow_nan=False, durable=True)
    io_utils.atomic_write_json(
        report_path, report, allow_nan=False, durable=True)
    recovered = sum(report["recovered_shards"].values())
    if report["rejected"] or recovered != report["attempted_shards"]:
        raise ValueError(
            "recovery sidecar contains rejected scene transactions")
    compiled = background_collection.compile_global(
        manifest, checkpoint, state=sidecar)
    print(json.dumps({
        "checkpoint": checkpoint,
        "checkpoint_summary": compiled["checkpoint_summary"],
        "recovery_report": str(report_path),
        "recovery_state": str(state_path),
        **report,
    }, indent=2, sort_keys=True))
    return 0


def _record_compile_result(
        manifest: dict, state: dict, *, round_index: int, checkpoint: str,
        compile_fn=background_collection.compile_global) -> bool:
    """Persist one retryable compile checkpoint transition."""
    try:
        compiled = compile_fn(manifest, checkpoint, state=state)
    except Exception as error:
        state["status"] = "compile_failed"
        state["current_round"] = int(round_index)
        for dataset in background_collection.DATASETS:
            state.setdefault("datasets", {})[dataset] = {
                "status": "compile_failed", "checkpoint": checkpoint,
                "error": str(error)}
        return False
    checkpoint_identity = compiled.get("checkpoint_summary")
    if not isinstance(checkpoint_identity, dict):
        state["status"] = "compile_failed"
        state["current_round"] = int(round_index)
        for dataset in background_collection.DATASETS:
            state.setdefault("datasets", {})[dataset] = {
                "status": "compile_failed", "checkpoint": checkpoint,
                "error": "compile result lacks checkpoint provenance"}
        return False
    for dataset in background_collection.DATASETS:
        result = {
            **compiled["datasets"][dataset],
            "artifact": compiled["artifact"],
            "coverage": compiled["coverage"],
            "gt_as_pred": compiled["gt_as_pred"],
            "combined_six_task_macro": compiled["six_task_macro"],
            "checkpoint": checkpoint,
            "checkpoint_summary": dict(checkpoint_identity),
        }
        state.setdefault("datasets", {})[dataset] = result
        result["catalog_coverage"] = \
            background_collection.dataset_catalog_coverage(
                manifest, state, dataset)
    state.setdefault("compile_checkpoints", []).append(checkpoint)
    state["status"] = "running"
    return True


def _poll_controller_processes(
        manifest: dict, state: dict, processes: dict,
        definitions: dict[str, dict], collection: dict, *,
        now: float) -> list[dict]:
    """Settle finished workers and return their validated job definitions."""
    terminal = []
    for job_id, process in list(processes.items()):
        definition = definitions[job_id]
        runtime = state["jobs"][job_id]
        health = background_collection.job_health(
            definition, now=now,
            first_record_deadline_s=collection[
                "first_record_deadline_s"],
            initialization_deadline_s=collection[
                "initialization_deadline_s"],
            scene_wallclock_s=definition["scene_wallclock_s"],
            started_time=runtime["started_time_unix"])
        runtime["durable_records"] = health["durable_records"]
        if (health.get("backend_ready_time") is not None and
                not runtime.get("capacity_backend_ready_recorded")):
            background_collection.record_capacity_event(
                manifest, definition, "backend_ready",
                time_unix=health["backend_ready_time"])
            runtime["capacity_backend_ready_recorded"] = True
        requested_signal = None
        if process.poll() is None:
            requested_signal = \
                background_collection.controller_watchdog_action(
                    definition, runtime,
                    health_status=health["status"], now=now,
                    scene_wallclock_s=definition["scene_wallclock_s"])
        if requested_signal is not None:
            try:
                background_collection.terminate_process_group(
                    process.pid, requested_signal)
            except ProcessLookupError:
                pass
        returncode = process.poll()
        print(json.dumps({
            "event": "controller_heartbeat", "job_id": job_id,
            "pid": process.pid,
            "elapsed_s": round(
                now - runtime["started_time_unix"], 3),
            **health, "returncode": returncode,
        }, sort_keys=True), flush=True)
        if returncode is None:
            continue
        background_collection.record_capacity_event(
            manifest, definition, "controller_scene_finished",
            time_unix=now)
        runtime["finished_time_unix"] = now
        stopped_for_capacity = runtime.get(
            "performance_violation") == "scene_wallclock_reached"
        runtime["returncode"] = int(returncode)
        retryable = (
            not health["durable_records"] and returncode != 0 and
            background_collection.retryable_failure(
                definition["log_path"], failure_status=runtime.get(
                    "performance_violation")))
        if retryable and int(runtime.get("attempt") or 0) <= int(collection[
                "retryable_initialization_attempts"]):
            runtime["status"] = "failed_retryable"
            processes[job_id] = background_collection.retry_job(
                manifest, state, job_id, now=now)
            continue
        catalog_status = "failed"
        recovered_status = None
        try:
            transaction = background_collection.validate_scene_transaction(
                definition, returncode=int(returncode),
                stopped_for_capacity=stopped_for_capacity)
        except (OSError, TypeError, ValueError) as error:
            runtime["source_validation_error"] = str(error)
        else:
            catalog_status = transaction.pop("catalog_status")
            recovered_status = transaction.pop("terminal_status", None)
            runtime["source_validation"] = transaction
        runtime["catalog_status"] = catalog_status
        runtime["status"] = background_recovery.settled_runtime_status(
            returncode=int(returncode),
            durable_records=int(health["durable_records"]),
            catalog_status=catalog_status,
            stopped_for_capacity=stopped_for_capacity)
        if recovered_status is not None:
            runtime["status"] = recovered_status
        if runtime["status"] == "completed_capacity_shortfall":
            runtime["capacity_shortfall"] = runtime["performance_violation"]
        del processes[job_id]
        terminal.append(definition)
    return terminal


def _run_continuous_controller(
        manifest: dict, state: dict, args, processes: dict) -> int:
    """Run three independent dataset cursors until operator interruption."""
    collection = manifest["collection"]
    poll_s = float(
        args.poll_interval_s or config.BACKGROUND_PROCESS_POLL_INTERVAL_S)
    heartbeat_s = float(collection["heartbeat_interval_s"])
    heartbeat_deadline = time.time() + heartbeat_s
    definitions = {}
    for job_id, runtime in state.get("jobs", {}).items():
        if runtime.get("status") not in background_scheduler.RUNNING_STATUSES:
            continue
        job = background_collection.continuous_job(
            manifest, runtime["dataset"],
            catalog_pass=int(runtime["catalog_pass"]),
            scene_index=int(runtime["scene_index"]))
        if job["job_id"] != job_id:
            raise ValueError("continuous running job identity differs")
        definitions[job_id] = job
    terminal, changed = background_recovery.restore_running_processes(
        state, definitions, processes, now=time.time())
    for job in terminal:
        background_scheduler.advance_dataset_cursor(manifest, state, job)
    if changed:
        _write_state(manifest, state)
    while True:
        ready = background_scheduler.next_continuous_jobs(
            manifest, state, job_factory=background_collection.continuous_job)
        if ready:
            definitions.update({job["job_id"]: job for job in ready})
            processes.update(background_scheduler.launch_jobs(
                manifest, state, ready))
            _write_state(manifest, state)
        if not processes:
            if set(state.get("exhausted_datasets") or []) == set(
                    background_collection.DATASETS):
                state["status"] = "capacity_shortfall"
                _write_state(manifest, state)
                return 2
            raise RuntimeError(
                "continuous background collection has no runnable job")
        now, due = background_recovery.wait_for_process_event(
            processes, poll_s=poll_s,
            heartbeat_deadline=heartbeat_deadline)
        if not due:
            continue
        terminal = _poll_controller_processes(
            manifest, state, processes, definitions, collection,
            now=now)
        for job in terminal:
            background_scheduler.advance_dataset_cursor(
                manifest, state, job)
        _write_state(manifest, state)
        heartbeat_deadline = now + heartbeat_s


def _run_controller(args, control: dict) -> int:
    manifest = _load_manifest(_manifest_path(args))
    background_collection.validate_external_authorities(manifest)
    if _clean_revision() != manifest["revision"]:
        raise ValueError("controller revision differs from clean HEAD")
    state = _load_state(manifest)
    processes = control.setdefault("processes", {})
    control.update({"manifest": manifest, "state": state})
    if not state.get("jobs"):
        background_collection.validate_launch_resources(
            _gpu_memory_used_mib(),
            free_bytes=shutil.disk_usage(
                Path(manifest["output_root"]).parent).free,
            capacity_profile=manifest["capacity_profile"])
    if background_collection.is_continuous(manifest):
        return _run_continuous_controller(
            manifest, state, args, processes)
    collection = manifest["collection"]
    poll_s = float(
        args.poll_interval_s or config.BACKGROUND_PROCESS_POLL_INTERVAL_S)
    heartbeat_s = float(collection["heartbeat_interval_s"])
    completed_datasets = {
        dataset for dataset in background_collection.DATASETS
        if background_collection.dataset_can_stop(manifest, state, dataset)}
    job_by_id = {
        job["job_id"]: job for round_value in manifest["rounds"]
        for job in round_value["jobs"]}
    pass_values = sorted({
        int(row["catalog_pass"]) for row in manifest["rounds"]})
    for catalog_pass in pass_values:
        if completed_datasets == set(background_collection.DATASETS):
            break
        pass_jobs = [
            job for job in job_by_id.values()
            if int(job["catalog_pass"]) == catalog_pass]
        last_round = max(int(job["round_index"]) for job in pass_jobs)
        processes.clear()
        running_definitions = {
            job_id: job_by_id[job_id]
            for job_id, value in state.get("jobs", {}).items()
            if job_id in job_by_id and
            int(job_by_id[job_id]["catalog_pass"]) == catalog_pass and
            value.get("status") in background_scheduler.RUNNING_STATUSES}
        _, changed = background_recovery.restore_running_processes(
            state, running_definitions, processes, now=time.time())
        if changed:
            _write_state(manifest, state)
        heartbeat_deadline = time.time() + heartbeat_s
        while True:
            completed_datasets = {
                dataset for dataset in background_collection.DATASETS
                if background_collection.dataset_can_stop(
                    manifest, state, dataset)}
            ready = background_scheduler.next_jobs(
                manifest, state, catalog_pass=catalog_pass,
                completed_datasets=completed_datasets)
            if ready:
                processes.update(background_scheduler.launch_jobs(
                    manifest, state, ready))
                _write_state(manifest, state)
            if not processes:
                if background_scheduler.catalog_pass_complete(
                        manifest, state, catalog_pass=catalog_pass,
                        completed_datasets=completed_datasets):
                    break
                raise RuntimeError(
                    "background collection catalog pass cannot advance")
            now, due = background_recovery.wait_for_process_event(
                processes, poll_s=poll_s,
                heartbeat_deadline=heartbeat_deadline)
            if not due:
                continue
            _poll_controller_processes(
                manifest, state, processes, job_by_id, collection,
                now=now)
            _write_state(manifest, state)
            heartbeat_deadline = now + heartbeat_s
        state["current_round"] = last_round + 1
        completed_datasets = {
            dataset for dataset in background_collection.DATASETS
            if background_collection.dataset_can_stop(
                manifest, state, dataset)}
        state["frame_progress"] = background_collection.frame_progress(
            manifest, state)
        _write_state(manifest, state)
    complete = completed_datasets == set(background_collection.DATASETS)
    final_checkpoint = "final-frame-target"
    if final_checkpoint not in state.get("compile_checkpoints", []):
        if not _record_compile_result(
                manifest, state, round_index=len(manifest["rounds"]),
                checkpoint=final_checkpoint):
            _write_state(manifest, state)
            return 2
    state["current_round"] = len(manifest["rounds"])
    state["frame_progress"] = background_collection.frame_progress(
        manifest, state)
    state["status"] = "complete" if complete else "capacity_shortfall"
    _write_state(manifest, state)
    print(json.dumps(_status_value(manifest, state), indent=2, sort_keys=True))
    return 0 if complete else 2


def _run(args) -> int:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    light_manifest = _load_control_manifest(_manifest_path(args))
    pid_path = _controller_pid_path(light_manifest)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.atomic_write_text(
        pid_path, f"{os.getpid()}\n", durable=True)
    control = {"processes": {}}
    try:
        return _run_controller(args, control)
    except KeyboardInterrupt:
        manifest = control.get("manifest") or light_manifest
        state = control.get("state") or _load_control_state(manifest)
        _interrupt_controller(
            manifest, state, control["processes"], now=time.time())
        return 130
    finally:
        pid_path.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    habitat_python = Path(os.environ.get(
        "EGOCONSEQ_HABITAT_PYTHON", sys.executable))
    b1k_data_root = Path(os.environ.get(
        "B1K_DATA_ROOT",
        ROOT / "data" / "sources" / "behavior-1k-v3.9.1"))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--measurements", required=True, type=Path)
    profile.add_argument("--out", required=True, type=Path)
    profile.set_defaults(run=_profile)
    canary_build = commands.add_parser("canary-build")
    canary_build.add_argument("--output-root", required=True, type=Path)
    canary_build.add_argument("--r2r-scenes", nargs=2, required=True)
    canary_build.add_argument("--gs-scenes", nargs=2, required=True)
    canary_build.add_argument("--b1k-scenes", nargs=2, required=True)
    canary_build.add_argument(
        "--python", type=Path, default=habitat_python)
    canary_build.add_argument(
        "--b1k-data-root", type=Path, default=b1k_data_root)
    canary_build.add_argument(
        "--b1k-source-manifest", required=True, type=Path)
    canary_build.add_argument(
        "--ordinary-actions-per-pose", type=int,
        choices=config.ACTION_CANDIDATE_ORDINARY_LADDER,
        default=config.ACTION_CANDIDATE_ORDINARY_PER_POSE)
    canary_build.set_defaults(run=_canary_build)
    canary_run = commands.add_parser("canary-run")
    canary_run.add_argument("--manifest", required=True, type=Path)
    canary_run.set_defaults(run=_canary_run)
    build = commands.add_parser("build")
    build.add_argument("--output-root", required=True, type=Path)
    build.add_argument(
        "--python", type=Path, default=habitat_python)
    build.add_argument(
        "--b1k-data-root", type=Path, default=b1k_data_root)
    build.add_argument(
        "--b1k-source-manifest", required=True, type=Path)
    build.add_argument(
        "--b1k-catalog-audit", required=True, type=Path)
    build.add_argument(
        "--b1k-catalog-audit-sha256", required=True)
    build.add_argument("--capacity-profile", required=True, type=Path)
    build.add_argument("--capacity-profile-sha256", required=True)
    build.add_argument("--baseline-checkpoint", type=Path)
    build.add_argument("--baseline-checkpoint-sha256")
    build.add_argument("--seed-checkpoint", type=Path)
    build.add_argument("--seed-checkpoint-sha256")
    build.add_argument("--collection-seed", required=True, type=int)
    build.add_argument(
        "--saturated-dataset", action="append", default=[],
        choices=background_collection.DATASETS)
    build.add_argument(
        "--target-pose-diverse-frames", type=int,
        default=config.BACKGROUND_TARGET_POSE_DIVERSE_FRAMES)
    build.add_argument(
        "--rounds", type=int, default=0,
        help="finite catalog passes; 0 runs continuously until stopped")
    build.set_defaults(run=_build)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--poll-interval-s", type=float)
    run.set_defaults(run=_run)
    status = commands.add_parser("status")
    status.add_argument("--manifest", required=True, type=Path)
    status.set_defaults(run=_status)
    stop = commands.add_parser("stop")
    stop.add_argument("--manifest", required=True, type=Path)
    stop.set_defaults(run=_stop)
    recover = commands.add_parser("recover")
    recover.add_argument("--manifest", required=True, type=Path)
    recover.add_argument("--checkpoint", default="recovery-562")
    recover.set_defaults(run=_recover)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
