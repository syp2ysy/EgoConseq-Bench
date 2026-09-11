"""The local controller schedules shards; it never changes pose-local GT."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from pipeline import (
    background_capacity, background_collection, background_scheduler,
    collection_runtime,
)
from scripts import run_background_collection
from tests._synthetic import capacity_evidence


TASKS = (
    "A1", "A2", "A3", "A4", "B1", "B2", "C1",
)


@pytest.fixture(autouse=True)
def _synthetic_capacity_record_validation(monkeypatch):
    monkeypatch.setattr(
        background_capacity, "_authenticate_canary",
        lambda *_args, **_kwargs: {
            "parent_manifest": {"path": "synthetic", "sha256": "0" * 64},
            "source_authority": {"path": "synthetic", "sha256": "1" * 64},
        })
    monkeypatch.setattr(
        background_capacity, "_authenticate_evidence_parent",
        lambda value: ({
            "revision": "a" * 40,
            "paths": {
                "r2r_train_episodes": "/data/r2r/train.json.gz",
                "gs_source_manifest": "/data/gs/train.json",
                "b1k_source_manifest": "/data/b1k/source.json",
            },
            "source_manifest_sha256": {
                "r2r": "4" * 64, "gs": "5" * 64, "b1k": "3" * 64},
            "scene_catalog": {
                dataset: [f"{dataset}-{index}"
                          for index in range(row["catalog_scene_count"])]
                for dataset, row in value["datasets"].items()},
            "selected_scenes": {
                dataset: [canary["scene_id"]
                          for canary in row["canary_scenes"]]
                for dataset, row in value["datasets"].items()},
        }, object()))
def _manifest(tmp_path: Path, *, rounds=1, seed_pose_exclusions=None,
              baseline_checkpoint=None, target_pose_diverse_frames=None,
              collection_seed=None, saturated_datasets=()) -> dict:
    r2r_scenes = [f"r2r-{index}" for index in range(8)]
    gs_scenes = [f"gs-{index}" for index in range(8)]
    b1k_scenes = [f"b1k-{index}" for index in range(12)]
    capacity_profile = background_collection.derive_capacity_profile(
        capacity_evidence(tmp_path / "measurements", counts={
            "r2r": len(r2r_scenes), "gs": len(gs_scenes),
            "b1k": len(b1k_scenes),
        }))
    audit = {
        "schema": "b1k-catalog-authority-audit.v1",
        "installed_scene_ids": [*b1k_scenes, "excluded"],
        "accepted": [{"scene_id": scene_id} for scene_id in b1k_scenes],
        "excluded": [{"scene_id": "excluded"}],
    }
    optional = {}
    if collection_seed is not None:
        optional["collection_seed"] = collection_seed
    if saturated_datasets:
        optional["saturated_datasets"] = saturated_datasets
    return background_collection.build_manifest(
        revision="a" * 40,
        output_root=tmp_path / "run",
        r2r_scenes=r2r_scenes,
        gs_scenes=gs_scenes,
        b1k_scenes=b1k_scenes,
        paths={
            "repository": str(tmp_path / "repo"),
            "python": "/env/habitat/bin/python",
            "r2r_train_episodes": "/data/r2r/train.json.gz",
            "mp3d_root": "/data/mp3d",
            "gs_data_root": "/data/gs",
            "gs_source_manifest": "/data/gs/train.json",
            "b1k_python": "/env/behavior/bin/python",
            "b1k_data_root": "/data/b1k",
            "b1k_source_manifest": "/data/b1k/source.json",
            "b1k_catalog_audit": "/data/b1k/audit.json",
            "capacity_profile": "/data/capacity.json",
        },
        b1k_catalog_audit=audit,
        b1k_catalog_audit_sha256="2" * 64,
        r2r_source_manifest_sha256="4" * 64,
        gs_source_manifest_sha256="5" * 64,
        b1k_source_manifest_sha256="3" * 64,
        b1k_source_scene_ids=b1k_scenes,
        capacity_profile=capacity_profile,
        capacity_profile_sha256=capacity_profile["sha256"],
        seed_pose_exclusions=seed_pose_exclusions,
        baseline_checkpoint=baseline_checkpoint,
        target_pose_diverse_frames=target_pose_diverse_frames,
        rounds=rounds,
        **optional,
    )


def test_manifest_schedules_every_scene_as_one_transaction(tmp_path):
    manifest = _manifest(tmp_path)

    assert manifest["schema"] == background_collection.CONTROLLER_SCHEMA
    assert manifest["revision"] == "a" * 40
    assert manifest["quota"]["r2r"]["task_totals"] == \
        dict(background_collection.seen_spec.DATASET_TASK_TOTALS["r2r"])
    assert manifest["quota"]["gs"]["supported_tasks"] == [
        "A1", "A2", "A4", "B2", "C1"]
    assert len(manifest["rounds"]) == 12
    gpu_by_dataset = {"gs": 0, "b1k": 1, "r2r": 2}
    scheduled = {dataset: [] for dataset in ("r2r", "gs", "b1k")}
    for round_value in manifest["rounds"]:
        jobs = round_value["jobs"]
        assert 1 <= len(jobs) <= 4
        assert len({job["gpu_id"] for job in jobs}) == len(jobs)
        assert len({job["output_dir"] for job in jobs}) == len(jobs)
        for job in jobs:
            assert len(job["scenes"]) == 1
            assert job["gpu_id"] == gpu_by_dataset[job["dataset"]]
            scheduled[job["dataset"]].extend(job["scenes"])
            command = job["command"]
            assert "--poses-per-scene" in command
            capacity = manifest["capacity_profile"]["datasets"][job["dataset"]]
            assert command[command.index("--poses-per-scene") + 1] == \
                str(capacity["records_per_scene"])
            assert "--pose-candidates-per-scene" in command
            assert command[
                command.index("--pose-candidates-per-scene") + 1] == \
                str(capacity["pose_attempt_cap"])
            assert command[command.index("--lengths") + 1:][0:6] == [
                "1", "2", "3", "4", "5", "6"]
            assert "--code-revision" in command
            assert command[command.index("--benchmark-partition") + 1] == \
                "train_seen"
            assert command[
                command.index("--ordinary-actions-per-pose") + 1] == "36"
            assert job["environment"]["OMP_NUM_THREADS"] == "1"
            if job["dataset"] == "b1k":
                assert "CUDA_VISIBLE_DEVICES" not in job["environment"]
                assert job["environment"]["OMNIGIBSON_GPU_ID"] == \
                    str(job["gpu_id"])
            else:
                assert job["environment"]["CUDA_VISIBLE_DEVICES"] == \
                    str(job["gpu_id"])
    assert {key: sorted(value) for key, value in scheduled.items()} == {
        "r2r": sorted(f"r2r-{index}" for index in range(8)),
        "gs": sorted(f"gs-{index}" for index in range(8)),
        "b1k": sorted(f"b1k-{index}" for index in range(12)),
    }
    assert manifest["collection"]["catalog_passes"] == 1
    assert "scene_wallclock_s" not in manifest["collection"]


def test_continuous_manifest_freezes_one_catalog_pass(tmp_path):
    manifest = _manifest(tmp_path, rounds=0)

    assert manifest["collection"]["catalog_passes"] == 0
    assert {row["catalog_pass"] for row in manifest["rounds"]} == {0}
    scheduled = {dataset: [] for dataset in ("r2r", "gs", "b1k")}
    for round_value in manifest["rounds"]:
        for job in round_value["jobs"]:
            scheduled[job["dataset"]].extend(job["scenes"])
    assert scheduled == manifest["scene_catalog"]
    background_collection.validate_manifest(manifest)


def test_continuous_job_derivation_is_deterministic(tmp_path):
    manifest = _manifest(tmp_path, rounds=0)

    first = background_collection.continuous_job(
        manifest, "gs", catalog_pass=1, scene_index=0)
    second = background_collection.continuous_job(
        manifest, "gs", catalog_pass=1, scene_index=0)

    assert first == second
    assert first["job_id"] == "gs-r08-g0"
    assert first["round_index"] == 8
    assert first["catalog_pass"] == 1
    assert first["scenes"] == ["gs-0"]
    assert first["pose_exclusions_path"].endswith(
        "/gs/gs-0-pass01.json")
    assert first["transaction_binding"]["seed"] == 20260811 + 8 * 101


def test_continuous_scheduler_retries_scenes_after_zero_yield_pass(tmp_path):
    manifest = _manifest(tmp_path, rounds=0)
    state = background_collection.initial_state(manifest)
    state["dataset_cursors"]["gs"] = {
        "catalog_pass": 1, "scene_index": 0}
    state["exhausted_datasets"] = ["b1k", "r2r"]
    for scene_index in range(8):
        job = background_collection.continuous_job(
            manifest, "gs", catalog_pass=0, scene_index=scene_index)
        state["jobs"][job["job_id"]] = {
            "dataset": "gs",
            "scene_id": f"gs-{scene_index}",
            "catalog_pass": 0,
            "status": "completed_zero_yield",
            "catalog_status": "zero_yield",
        }

    ready = background_scheduler.next_continuous_jobs(
        manifest, state,
        job_factory=background_collection.continuous_job)

    assert [job["job_id"] for job in ready] == ["gs-r08-g0"]
    assert state["exhausted_datasets"] == ["b1k", "r2r"]


def test_continuous_manifest_binds_seed_and_starts_saturated_dataset_exhausted(
        tmp_path):
    seeds = {}
    for dataset, representatives, records in (
            ("r2r", 4000, 4200), ("gs", 3000, 3200),
            ("b1k", 2500, 2700)):
        path = tmp_path / f"{dataset}-seed.json"
        path.write_text(json.dumps({
            "schema_version": "egoconseq.pose_exclusions.v1",
            "scenes": {},
        }))
        seeds[dataset] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "representative_count": representatives,
            "record_count": records,
        }
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    manifest = _manifest(
        tmp_path, rounds=0, collection_seed=700,
        saturated_datasets=("gs",), seed_pose_exclusions=seeds,
        baseline_checkpoint={
            "path": str(checkpoint.resolve()),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })

    assert manifest["schema"] == background_collection.CONTROLLER_SCHEMA
    assert manifest["collection"]["seed_base"] == 700
    assert manifest["collection"]["saturated_datasets"] == ["gs"]
    assert background_collection.initial_state(manifest)[
        "exhausted_datasets"] == ["gs"]
    first = background_collection.continuous_job(
        manifest, "b1k", catalog_pass=1, scene_index=0)
    assert first["transaction_binding"]["seed"] == 700 + 12 * 101 + 1


def test_continuous_state_validates_a_derived_runtime_job(tmp_path):
    manifest = _manifest(tmp_path, rounds=0)
    state = background_collection.initial_state(manifest)
    assert state["dataset_cursors"] == {
        dataset: {"catalog_pass": 0, "scene_index": 0}
        for dataset in ("r2r", "gs", "b1k")
    }
    job = background_collection.continuous_job(
        manifest, "gs", catalog_pass=1, scene_index=0)
    state["status"] = "running"
    state["current_round"] = job["round_index"]
    state["jobs"][job["job_id"]] = {
        "job_id": job["job_id"],
        "dataset": "gs",
        "scene_id": "gs-0",
        "gpu_id": 0,
        "round_index": job["round_index"],
        "catalog_pass": 1,
        "scene_index": 0,
        "pid": 99,
        "status": "running",
        "attempt": 1,
        "started_time_unix": 1.0,
        "finished_time_unix": None,
        "durable_records": 0,
    }

    background_collection.validate_state(manifest, state)


def test_continuous_state_allows_terminal_scene_exhaustion(tmp_path):
    manifest = _manifest(tmp_path, rounds=0)
    state = background_collection.initial_state(manifest)
    state["status"] = "capacity_shortfall"
    state["exhausted_datasets"] = list(background_collection.DATASETS)

    background_collection.validate_state(manifest, state)


def test_revisit_jobs_bind_the_existing_pose_exclusion_protocol(tmp_path):
    manifest = _manifest(tmp_path, rounds=2)

    for round_value in manifest["rounds"]:
        for job in round_value["jobs"]:
            command = job["command"]
            if job["catalog_pass"] == 0:
                assert job["pose_exclusions_path"] is None
                assert "--pose-exclusions" not in command
                continue
            path = job["pose_exclusions_path"]
            assert path.endswith(
                f"/{job['dataset']}/{job['scenes'][0]}-pass01.json")
            assert command[command.index("--pose-exclusions") + 1] == path
            assert job["transaction_binding"]["pose_exclusions_path"] == path


def test_manifest_binds_seed_exclusions_from_pass_zero(tmp_path):
    seeds = {}
    for dataset in ("r2r", "gs", "b1k"):
        path = tmp_path / f"{dataset}-seed.json"
        path.write_text(json.dumps({
            "schema_version": "egoconseq.pose_exclusions.v1",
            "scenes": {},
        }))
        seeds[dataset] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "representative_count": 0,
            "record_count": 0,
        }

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    baseline = {
        "path": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    manifest = _manifest(
        tmp_path, rounds=2, seed_pose_exclusions=seeds,
        baseline_checkpoint=baseline, target_pose_diverse_frames=40000)

    assert manifest["seed_pose_exclusions"] == seeds
    assert manifest["baseline_checkpoint"] == baseline
    assert manifest["target_pose_diverse_frames"] == 40000
    for round_value in manifest["rounds"]:
        for job in round_value["jobs"]:
            path = job["pose_exclusions_path"]
            assert path.endswith(
                f"/{job['dataset']}/{job['scenes'][0]}-"
                f"pass{job['catalog_pass']:02d}.json")
            assert job["command"][
                job["command"].index("--pose-exclusions") + 1] == path


def test_manifest_rejects_scene_reuse_and_nonclean_revision(tmp_path):
    valid = _manifest(tmp_path)
    common = {
        "paths": valid["paths"],
        "b1k_catalog_audit": {
            "schema": "b1k-catalog-authority-audit.v1",
            "installed_scene_ids": valid["scene_catalog"]["b1k"],
            "accepted": [{"scene_id": scene_id}
                         for scene_id in valid["scene_catalog"]["b1k"]],
            "excluded": [],
        },
        "b1k_catalog_audit_sha256": "2" * 64,
        "r2r_source_manifest_sha256": "4" * 64,
        "gs_source_manifest_sha256": "5" * 64,
        "b1k_source_manifest_sha256": "3" * 64,
        "b1k_source_scene_ids": valid["scene_catalog"]["b1k"],
        "capacity_profile": valid["capacity_profile"],
        "capacity_profile_sha256": valid["capacity_profile"]["sha256"],
    }
    with pytest.raises(ValueError, match="revision"):
        background_collection.build_manifest(
            revision="dirty", output_root=tmp_path / "run",
            r2r_scenes=["r"] * 8, gs_scenes=["g"] * 8,
            b1k_scenes=["b"] * 12, rounds=1, **common)
    with pytest.raises(ValueError, match="unique"):
        background_collection.build_manifest(
            revision="a" * 40, output_root=tmp_path / "run",
            r2r_scenes=["r"] * 8,
            gs_scenes=[f"gs-{x}" for x in range(8)],
            b1k_scenes=[f"b1k-{x}" for x in range(12)], rounds=1,
            **common)


class _FakeProcess:
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None


def test_start_round_launches_one_process_per_gpu_and_persists_state(tmp_path):
    manifest = _manifest(tmp_path, rounds=1)
    state = background_collection.initial_state(manifest)
    calls = []

    def launch(job, environment, log_path):
        calls.append((job["job_id"], dict(environment), Path(log_path)))
        return _FakeProcess(1000 + len(calls))

    processes = background_collection.start_round(
        manifest, state, round_index=0, launcher=launch, now=123.0)

    assert set(processes) == {
        job["job_id"] for job in manifest["rounds"][0]["jobs"]}
    assert {value["pid"] for value in state["jobs"].values()} == {
        1001, 1002, 1003}
    assert {value["status"] for value in state["jobs"].values()} == {
        "running"}
    assert len({value["gpu_id"] for value in state["jobs"].values()}) == 3
    assert all(call[1]["PYTHONUNBUFFERED"] == "1" for call in calls)


def test_scheduler_retires_only_normally_exhausted_scenes(tmp_path):
    manifest = _manifest(tmp_path, rounds=2)
    state = background_collection.initial_state(manifest)
    pass_zero = background_scheduler.jobs_for_pass(manifest, 0)
    pass_one = background_scheduler.jobs_for_pass(manifest, 1)
    target = pass_one[0]
    previous = next(
        job for job in pass_zero
        if (job["dataset"], job["scenes"][0]) ==
        (target["dataset"], target["scenes"][0]))
    for job in pass_zero:
        state["jobs"][job["job_id"]] = {
            "status": "completed", "catalog_status": "completed"}
    state["jobs"][previous["job_id"]] = {
        "status": "completed_zero_yield", "catalog_status": "zero_yield"}

    ready = background_scheduler.next_jobs(
        manifest, state, catalog_pass=1)

    assert target["job_id"] not in {job["job_id"] for job in ready}
    for job in pass_one[1:]:
        state["jobs"][job["job_id"]] = {
            "status": "completed", "catalog_status": "completed"}
    assert background_scheduler.catalog_pass_complete(
        manifest, state, catalog_pass=1) is True

    state["jobs"][previous["job_id"]]["status"] = "failed_zero_yield"
    assert background_scheduler.catalog_pass_complete(
        manifest, state, catalog_pass=1) is False
    ready = background_scheduler.next_jobs(
        manifest, state, catalog_pass=1)
    assert target["job_id"] in {job["job_id"] for job in ready}


def test_gs_catalog_coverage_requires_every_clean_scene_transaction(tmp_path):
    manifest = _manifest(tmp_path)
    state = background_collection.initial_state(manifest)
    gs_jobs = [
        job for round_value in manifest["rounds"]
        for job in round_value["jobs"] if job["dataset"] == "gs"]
    for job in gs_jobs:
        state["jobs"][job["job_id"]] = {
            "catalog_status": "zero_yield"}

    assert background_collection.dataset_catalog_coverage(
        manifest, state, "gs")["complete"] is True
    state["jobs"][gs_jobs[-1]["job_id"]] = {"status": "interrupted"}
    coverage = background_collection.dataset_catalog_coverage(
        manifest, state, "gs")
    assert coverage["complete"] is False
    assert coverage["incomplete"] == [{
        "job_id": gs_jobs[-1]["job_id"],
        "scene_id": gs_jobs[-1]["scenes"][0],
        "status": "interrupted",
    }]


def test_retry_is_limited_to_one_initialization_or_oom_attempt(tmp_path):
    manifest = _manifest(tmp_path, rounds=1)
    job = manifest["rounds"][0]["jobs"][0]
    state = background_collection.initial_state(manifest)
    state["jobs"][job["job_id"]] = {
        "job_id": job["job_id"], "dataset": job["dataset"],
        "gpu_id": job["gpu_id"], "round_index": 0,
        "pid": 9, "status": "failed_retryable", "attempt": 1,
        "started_time_unix": 1.0, "finished_time_unix": 2.0,
        "durable_records": 0,
    }
    calls = []

    def launch(job_value, environment, log_path):
        calls.append(job_value["job_id"])
        return _FakeProcess(20)

    process = background_collection.retry_job(
        manifest, state, job["job_id"], launcher=launch, now=3.0)

    assert process.pid == 20
    assert state["jobs"][job["job_id"]]["attempt"] == 2
    assert state["jobs"][job["job_id"]]["status"] == "running"
    state["jobs"][job["job_id"]]["status"] = "failed_retryable"
    with pytest.raises(ValueError, match="retry budget"):
        background_collection.retry_job(
            manifest, state, job["job_id"], launcher=launch, now=4.0)


def test_retry_clears_attempt_health_and_ignores_old_backend_ready(tmp_path):
    manifest = _manifest(tmp_path, rounds=1)
    job = manifest["rounds"][0]["jobs"][0]
    Path(job["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["log_path"]).write_text(
        json.dumps({"event": "backend_ready", "time_unix": 100.0}) + "\n")
    state = background_collection.initial_state(manifest)
    state["jobs"][job["job_id"]] = {
        "job_id": job["job_id"], "dataset": job["dataset"],
        "gpu_id": job["gpu_id"], "round_index": 0,
        "pid": 9, "status": "failed_retryable", "attempt": 1,
        "started_time_unix": 50.0, "finished_time_unix": 150.0,
        "durable_records": 0, "capacity_backend_ready_recorded": True,
        "performance_violation": "initialization_slow",
        "termination_requested_time_unix": 140.0,
        "termination_signal": "SIGINT",
    }

    background_collection.retry_job(
        manifest, state, job["job_id"],
        launcher=lambda *_args: _FakeProcess(20), now=200.0)
    runtime = state["jobs"][job["job_id"]]
    health = background_collection.job_health(
        job, now=231.0, initialization_deadline_s=30.0,
        started_time=runtime["started_time_unix"])

    assert health["backend_ready_time"] is None
    assert health["status"] == "initialization_slow"
    assert "capacity_backend_ready_recorded" not in runtime
    assert "performance_violation" not in runtime
    assert "termination_requested_time_unix" not in runtime
    assert "termination_signal" not in runtime


def test_empty_log_initialization_timeout_is_retryable(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("")

    assert background_collection.retryable_failure(
        log, failure_status="initialization_slow") is True
    assert background_collection.retryable_failure(
        log, failure_status="scene_wallclock_reached") is False


def test_job_progress_uses_backend_ready_and_durable_jsonl(tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    log = tmp_path / "job.log"
    log.write_text(
        json.dumps({"event": "backend_ready", "time_unix": 100.0}) + "\n")
    records = output / "records.jsonl"
    records.write_text('{"frame_id":"one"}\n{"frame_id":"two"}\n')

    progress = background_collection.job_progress({
        "dataset": "r2r", "output_dir": str(output), "log_path": str(log),
    })

    assert progress["backend_ready_time"] == 100.0
    assert progress["durable_records"] == 2
    assert progress["record_paths"] == [str(records)]


def test_two_minute_post_ready_delay_is_reported_not_hidden(tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    log = tmp_path / "job.log"
    log.write_text(
        json.dumps({"event": "backend_ready", "time_unix": 100.0}) + "\n")

    health = background_collection.job_health({
        "dataset": "gs", "output_dir": str(output), "log_path": str(log),
    }, now=221.0, first_record_deadline_s=120)

    assert health["status"] == "first_record_slow"
    assert health["seconds_since_ready"] == pytest.approx(121.0)


def test_two_minute_inter_record_delay_is_reported(tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    log = tmp_path / "job.log"
    log.write_text(
        json.dumps({"event": "backend_ready", "time_unix": 100.0}) + "\n")
    records = output / "records.jsonl"
    records.write_text('{"frame_id":"one"}\n')
    os.utime(records, (150.0, 150.0))

    health = background_collection.job_health({
        "dataset": "b1k", "output_dir": str(output),
        "log_path": str(log),
    }, now=271.0, first_record_deadline_s=120)

    assert health["status"] == "inter_record_slow"
    assert health["seconds_since_record"] == pytest.approx(121.0)


def test_slow_record_and_scene_wallclock_both_request_normal_stop(
        tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    log = tmp_path / "job.log"
    log.write_text(
        json.dumps({"event": "backend_ready", "time_unix": 100.0}) + "\n")
    job = {"dataset": "gs", "output_dir": str(output),
           "log_path": str(log), "started_time_unix": 1.0}

    slow = background_collection.job_health(
        job, now=221.0, first_record_deadline_s=120,
        scene_wallclock_s=5400)
    timed_out = background_collection.job_health(
        job, now=5501.0, first_record_deadline_s=120,
        scene_wallclock_s=5400)

    assert slow["status"] == "first_record_slow"
    assert background_collection.health_requires_termination(
        slow["status"]) is False
    assert timed_out["status"] == "scene_wallclock_reached"
    assert background_collection.health_requires_termination(
        timed_out["status"]) is False


def test_scene_wallclock_starts_at_backend_ready_not_process_launch(tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    log = tmp_path / "job.log"
    log.write_text(
        json.dumps({"event": "backend_ready", "time_unix": 700.0}) + "\n")
    job = {"dataset": "b1k", "output_dir": str(output),
           "log_path": str(log)}

    health = background_collection.job_health(
        job, now=1501.0, started_time=1.0,
        initialization_deadline_s=600, first_record_deadline_s=900,
        scene_wallclock_s=900)

    assert health["status"] == "healthy"
    assert health["backend_ready_time"] == 700.0


def test_finalizing_marker_allows_authenticated_validation_to_finish(
        tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    marker = output / "collection_finalization.json"
    marker.write_text(json.dumps({
        "schema": "egoconseq.collection-finalization.v2",
        "status": "finalizing", "started_time_unix": 100.0,
    }))
    runtime = {"status": "running", "started_time_unix": 1.0}
    job = {"dataset": "r2r", "output_dir": str(output)}

    assert background_collection.capacity_watchdog_action(
        job, runtime, health_status="inter_record_slow",
        now=399.0) is None
    assert runtime["status"] == "running"
    assert background_collection.capacity_watchdog_action(
        job, runtime, health_status="inter_record_slow",
        now=401.0) == signal.SIGKILL
    assert runtime["status"] == "stopping_sigkill"


def test_sealed_marker_keeps_short_process_exit_grace(tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    marker = output / "collection_finalization.json"
    marker.write_text(json.dumps({
        "schema": "egoconseq.collection-finalization.v2",
        "status": "completed", "started_time_unix": 100.0,
    }))
    runtime = {"status": "running", "started_time_unix": 1.0}
    job = {"dataset": "r2r", "output_dir": str(output)}

    assert background_collection.capacity_watchdog_action(
        job, runtime, health_status="healthy", now=200.0) is None
    assert background_collection.capacity_watchdog_action(
        job, runtime, health_status="healthy", now=231.0) == signal.SIGKILL
    assert runtime["performance_violation"] == "sealed_exit_timeout"


def test_slow_records_are_metrics_and_never_parent_signals(tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    runtime = {"status": "running", "started_time_unix": 1.0}
    job = {"dataset": "gs", "output_dir": str(output)}

    assert background_collection.capacity_watchdog_action(
        job, runtime, health_status="first_record_slow",
        now=200.0) is None
    assert runtime["status"] == "running"
    assert background_collection.capacity_watchdog_action(
        job, runtime, health_status="inter_record_slow",
        now=321.0) is None
    assert runtime["status"] == "running"


def test_scene_wallclock_signals_r2r_but_not_b1k(tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    runtime = {"status": "running", "started_time_unix": 1.0}
    job = {"dataset": "r2r", "output_dir": str(output)}

    assert background_collection.capacity_watchdog_action(
        job, runtime, health_status="scene_wallclock_reached",
        now=200.0) is None
    assert runtime["status"] == "running"
    assert background_collection.capacity_watchdog_action(
        job, runtime, health_status="scene_wallclock_reached",
        now=231.0) == signal.SIGINT
    b1k_runtime = {"status": "running", "started_time_unix": 1.0}
    b1k = {
        "dataset": "b1k", "scenes": ["scene-a"],
        "output_dir": str(output),
    }
    assert background_collection.capacity_watchdog_action(
        b1k, b1k_runtime, health_status="scene_wallclock_reached",
        now=2000.0) is None
    assert b1k_runtime["status"] == "running"


def test_signal_escalation_precedes_new_health_classification(tmp_path):
    runtime = {
        "status": "stopping_sigint", "started_time_unix": 1.0,
        "termination_requested_time_unix": 100.0,
    }
    job = {"dataset": "gs", "output_dir": str(tmp_path)}

    requested = background_collection.controller_watchdog_action(
        job, runtime, health_status="first_record_slow",
        now=100.0 + background_collection.config.BACKGROUND_SIGINT_GRACE_S,
        scene_wallclock_s=600.0)

    assert requested == signal.SIGTERM
    assert runtime["status"] == "stopping_sigterm"


def test_collector_emits_machine_readable_backend_ready_event(capsys):
    collection_runtime.emit_backend_ready(
        backend="b1k", scene_id="scene-a", time_unix=123.5)

    assert json.loads(capsys.readouterr().out) == {
        "backend": "b1k",
        "event": "backend_ready",
        "scene_id": "scene-a",
        "time_unix": 123.5,
    }


def test_cli_exposes_build_run_status_and_stop(monkeypatch):
    monkeypatch.setenv(
        "EGOCONSEQ_HABITAT_PYTHON", "/opt/egoconseq/bin/python")
    monkeypatch.setenv("B1K_DATA_ROOT", "/datasets/behavior-1k")
    parser = run_background_collection.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "build", "--output-root", "/tmp/run",
            "--capacity-profile", "/tmp/capacity.json",
            "--capacity-profile-sha256", "1" * 64,
            "--b1k-catalog-audit", "/tmp/audit.json",
            "--b1k-catalog-audit-sha256", "2" * 64])
    build = parser.parse_args([
        "build", "--output-root", "/tmp/run",
        "--b1k-source-manifest", "/tmp/source.json",
        "--capacity-profile", "/tmp/capacity.json",
        "--capacity-profile-sha256", "1" * 64,
        "--b1k-catalog-audit", "/tmp/audit.json",
        "--b1k-catalog-audit-sha256", "2" * 64,
        "--collection-seed", "700",
        "--baseline-checkpoint", "/tmp/checkpoint.json",
        "--baseline-checkpoint-sha256", "3" * 64,
        "--seed-checkpoint", "/tmp/seed-checkpoint.json",
        "--seed-checkpoint-sha256", "4" * 64])
    assert build.command == "build"
    assert build.python == Path("/opt/egoconseq/bin/python")
    assert build.b1k_data_root == Path("/datasets/behavior-1k")
    assert build.b1k_source_manifest == Path("/tmp/source.json")
    assert build.baseline_checkpoint == Path("/tmp/checkpoint.json")
    assert build.seed_checkpoint == Path("/tmp/seed-checkpoint.json")
    assert build.seed_checkpoint_sha256 == "4" * 64
    assert build.collection_seed == 700
    assert build.target_pose_diverse_frames == 40000
    canary = parser.parse_args([
        "canary-build", "--output-root", "/tmp/canary",
        "--r2r-scenes", "r2r-a", "r2r-b",
        "--gs-scenes", "gs-a", "gs-b",
        "--b1k-scenes", "b1k-a", "b1k-b",
        "--b1k-source-manifest", "/tmp/source.json"])
    assert canary.python == Path("/opt/egoconseq/bin/python")
    assert canary.b1k_data_root == Path("/datasets/behavior-1k")
    assert parser.parse_args([
        "run", "--manifest", "/tmp/manifest.json"]
    ).command == "run"
    assert parser.parse_args([
        "status", "--manifest", "/tmp/manifest.json"]
    ).command == "status"
    assert parser.parse_args([
        "stop", "--manifest", "/tmp/manifest.json"]
    ).command == "stop"
    assert parser.parse_args([
        "profile", "--measurements", "/tmp/measurements.json",
        "--out", "/tmp/profile.json"]
    ).command == "profile"


def test_b1k_scene_schedule_prefers_lower_static_triangle_cost():
    def scene(scene_id, triangle_counts):
        return SimpleNamespace(
            scene_id=scene_id,
            b1k_scene_authority={
                "runtime_instances": [
                    {"triangle_count": value}
                    for value in triangle_counts
                ],
            },
        )

    ordered = run_background_collection._ranked_b1k_scenes([
        scene("large", [100, 200]),
        scene("small-b", [30, 20]),
        scene("small-a", [25, 25]),
    ])

    assert ordered == ["small-a", "small-b", "large"]


def test_gs_scene_schedule_prefers_render_quality_then_cost():
    rows = [
        {
            "scene_id": "large-capable",
            "gaussian_count": 300,
            "fraction_above_quality_band": 0.01,
            "support_radius_p90_m": 0.10,
        },
        {
            "scene_id": "small-compatible",
            "gaussian_count": 100,
            "fraction_above_quality_band": 0.03,
            "support_radius_p90_m": 0.20,
        },
        {
            "scene_id": "tiny-but-expanded",
            "gaussian_count": 50,
            "fraction_above_quality_band": 0.20,
            "support_radius_p90_m": 0.40,
        },
        {
            "scene_id": "tiny-high-quality",
            "gaussian_count": 10,
            "fraction_above_quality_band": 0.0,
            "support_radius_p90_m": 0.01,
        },
    ]

    ordered = run_background_collection._rank_gs_schedule_rows(rows)

    assert [row["scene_id"] for row in ordered] == [
        "tiny-high-quality", "small-compatible", "large-capable",
        "tiny-but-expanded"]


def test_launch_preflight_rejects_busy_gpus_and_low_storage():
    background_collection.validate_launch_resources(
        [10, 20, 30, 40], free_bytes=121 * 1024 ** 3)
    with pytest.raises(ValueError, match="GPU"):
        background_collection.validate_launch_resources(
            [10, 501, 30, 40], free_bytes=121 * 1024 ** 3)
    with pytest.raises(ValueError, match="storage"):
        background_collection.validate_launch_resources(
            [10, 20, 30, 40], free_bytes=119 * 1024 ** 3)
