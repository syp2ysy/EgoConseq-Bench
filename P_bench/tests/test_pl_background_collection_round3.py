"""Round-three controller trust-boundary regressions."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import signal
from types import SimpleNamespace

import pytest

from pipeline import (
    a1_common_support, action_proposal, background_capacity,
    background_collection, candidate_preview,
)
from scripts import run_background_collection
from tests.test_pl_background_collection_closeout import (
    TASKS, _manifest, _paths, _raw_profile,
)
from tests.test_pl_v16_candidate_preview import (
    _one_a1_projection, _png, _validate_preview_artifact,
    _write_preview_artifact,
)
from tests.test_pl_check_records import _strict_r2r_record


def _synthetic_manifest(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        background_capacity, "_validate_candidate_artifact",
        lambda *_args, **_kwargs: None)
    return _manifest(tmp_path)


def _write_r2r_run_meta_for_records(path: Path, records: Path) -> str:
    record_value = json.loads(records.read_text().splitlines()[0])
    path.write_text(json.dumps({
        "record_schema_version": "conseq.v11",
        "oracle_contract_version": "ground-disc-visible-v8",
        "run_contract_sha256": "c" * 64,
        "params": {
            "backend": "r2r", "collection_mode": "main",
            "r2r_train_episodes": "/datasets/r2r/train.json",
            "mp3d_root": "/datasets/mp3d",
        },
        "resolved_scenes": [record_value["source"]],
    }) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_a1_projection(root: Path) -> dict:
    record_value = _strict_r2r_record()
    image = root / record_value["image_path"]
    image.parent.mkdir(parents=True)
    _png(image, size=(640, 480))
    (root / "records.jsonl").write_text(
        json.dumps(record_value, allow_nan=False) + "\n")
    return candidate_preview.compile_main_records(
        [record_value], asset_root=root, build_root=root / "build")


def test_public_canary_manifest_needs_no_capacity_profile(tmp_path):
    catalogs = {
        "r2r": ["r2r-0", "r2r-1", "r2r-2"],
        "gs": ["gs-0", "gs-1", "gs-2"],
        "b1k": ["b1k-0", "b1k-1", "b1k-2"],
    }
    selected = {dataset: scenes[:2]
                for dataset, scenes in catalogs.items()}

    manifest = background_collection.build_canary_manifest(
        revision="a" * 40,
        output_root=tmp_path / "canary",
        scene_catalog=catalogs,
        selected_scenes=selected,
        paths=_paths(tmp_path),
        source_manifest_sha256={
            "r2r": "4" * 64, "gs": "5" * 64, "b1k": "3" * 64},
        ordinary_actions_per_pose=18,
    )

    background_collection.validate_canary_manifest(manifest)
    assert "capacity_profile" not in manifest
    assert len(manifest["jobs"]) == 6
    assert {job["transaction_binding"]["poses_per_scene"]
            for job in manifest["jobs"]} == {20}
    attempt_caps = {
        job["dataset"]:
        job["transaction_binding"]["pose_candidates_per_scene"]
        for job in manifest["jobs"]
    }
    assert attempt_caps == {"r2r": 800, "gs": 800, "b1k": 6000}
    assert manifest["pose_attempt_cap"] == attempt_caps
    assert manifest["ordinary_actions_per_pose"] == 18
    assert all(
        job["transaction_binding"]["ordinary_actions_per_pose"] == 18 and
        job["command"][
            job["command"].index("--ordinary-actions-per-pose") + 1] == "18"
        for job in manifest["jobs"])
    assert all(len(manifest["selected_scenes"][name]) == 2
               for name in background_collection.DATASETS)


def test_canary_workflow_persists_positive_shortfall_evidence(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "canary-manifest.json"
    output_root = tmp_path / "canary"
    manifest = background_collection.build_canary_manifest(
        revision="a" * 40, output_root=output_root,
        scene_catalog={name: [f"{name}-0", f"{name}-1"]
                       for name in background_collection.DATASETS},
        selected_scenes={name: [f"{name}-0", f"{name}-1"]
                         for name in background_collection.DATASETS},
        paths=_paths(tmp_path),
        source_manifest_sha256={
            "r2r": "4" * 64, "gs": "5" * 64, "b1k": "3" * 64},
        )
    manifest_path.write_text(json.dumps(manifest))
    calls = []

    def execute(parent, job):
        calls.append(job["job_id"])
        output = Path(job["output_dir"])
        if job["dataset"] == "b1k":
            output /= job["scenes"][0]
        output.mkdir(parents=True)
        for name in ("records.jsonl", "run_meta.json",
                     "collection_funnel.json"):
            (output / name).write_text("{}\n")
        # The frozen target is 20, but a positive shortfall is itself the
        # capacity measurement the canary exists to preserve.
        (output / "records.jsonl").write_text("{}\n" * 19)
        background_collection.initialize_capacity_events(
            parent, job, parent_manifest_path=manifest_path,
            time_unix=1.0)
        background_collection.record_capacity_event(
            parent, job, "backend_ready", time_unix=2.0)
        background_collection.record_capacity_event(
            parent, job, "controller_scene_finished", time_unix=3.0)
        artifact = output / "candidate_qa"
        artifact.mkdir()
        return artifact

    evidence = run_background_collection.run_canary_manifest(
        manifest_path, execute_job=execute)

    assert len(calls) == 6
    assert evidence["schema"] == background_capacity.EVIDENCE_SCHEMA
    assert Path(evidence["parent_manifest_path"]) == manifest_path.resolve()
    assert all(len(row["canary_scenes"]) == 2
               for row in evidence["datasets"].values())
    assert (output_root / "capacity-evidence.json").is_file()


def test_canary_record_admission_measures_positive_shortfall():
    run_background_collection._require_positive_canary_records(19)


def test_canary_record_admission_rejects_zero_yield():
    with pytest.raises(ValueError, match="no accepted records"):
        run_background_collection._require_positive_canary_records(0)


def test_failed_canary_job_does_not_kill_productive_wave_sibling(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "canary-manifest.json"
    parent = background_collection.build_canary_manifest(
        revision="a" * 40, output_root=tmp_path / "canary",
        scene_catalog={name: [f"{name}-0", f"{name}-1"]
                       for name in background_collection.DATASETS},
        selected_scenes={name: [f"{name}-0", f"{name}-1"]
                         for name in background_collection.DATASETS},
        paths=_paths(tmp_path),
        source_manifest_sha256={
            "r2r": "4" * 64, "gs": "5" * 64, "b1k": "3" * 64})
    manifest_path.write_text(json.dumps(parent))
    jobs = parent["jobs"][:2]

    class Process:
        def __init__(self, pid, terminal=False):
            self.pid = pid
            self.alive = not terminal
            self.terminal = terminal

        def poll(self):
            return 1 if self.terminal else (None if self.alive else -2)

    processes = {
        jobs[0]["job_id"]: Process(101, terminal=True),
        jobs[1]["job_id"]: Process(102),
    }
    monkeypatch.setattr(
        background_collection, "_default_launcher",
        lambda job, *_args: processes[job["job_id"]])
    ticks = []

    def sleep(_seconds):
        ticks.append(True)
        if len(ticks) >= 2:
            processes[jobs[1]["job_id"]].alive = False

    monkeypatch.setattr(run_background_collection.time, "sleep", sleep)
    monkeypatch.setattr(
        background_collection, "job_health",
        lambda job, **kwargs: {
            "status": "healthy", "backend_ready_time": 2.0})
    source = {
        "path": str(tmp_path / "records.jsonl"),
        "records_sha256": "1" * 64, "run_meta_sha256": "2" * 64,
    }

    def validate(job, **_kwargs):
        if job["job_id"] == jobs[0]["job_id"]:
            raise ValueError("failed canary")
        return {
            "catalog_status": "completed", "sources": [source],
            "source_validated_records": 1,
        }

    monkeypatch.setattr(
        background_collection, "validate_scene_transaction", validate)
    compiled = []
    monkeypatch.setattr(
        run_background_collection, "_compile_canary_source",
        lambda _parent, job, _source: compiled.append(job["job_id"]) or
        (tmp_path / "candidate_qa"))
    signals = []

    def terminate(pid, requested_signal=signal.SIGINT):
        signals.append((pid, requested_signal))
        if pid == 102:
            processes[jobs[1]["job_id"]].alive = False

    monkeypatch.setattr(
        background_collection, "terminate_process_group", terminate)

    with pytest.raises(ValueError, match="failed canary"):
        run_background_collection._run_canary_wave(
            manifest_path, parent, jobs)

    assert processes[jobs[1]["job_id"]].poll() is not None
    assert signals == []
    assert compiled == [jobs[1]["job_id"]]


def test_canary_state_persists_typed_job_failure(tmp_path):
    parent = background_collection.build_canary_manifest(
        revision="a" * 40, output_root=tmp_path / "canary",
        scene_catalog={name: [f"{name}-0", f"{name}-1"]
                       for name in background_collection.DATASETS},
        selected_scenes={name: [f"{name}-0", f"{name}-1"]
                         for name in background_collection.DATASETS},
        paths=_paths(tmp_path),
        source_manifest_sha256={
            "r2r": "4" * 64, "gs": "5" * 64, "b1k": "3" * 64})
    state = run_background_collection._load_canary_state(parent)
    job = parent["jobs"][0]

    run_background_collection._fail_canary_state_job(
        parent, state, job, phase="candidate_build",
        error=ValueError("bad candidate"), attempt=1, time_unix=123.0)

    reopened = run_background_collection._load_canary_state(parent)
    assert reopened["failures"][job["job_id"]] == {
        "status": "failed", "dataset": job["dataset"],
        "scene_id": job["scenes"][0], "phase": "candidate_build",
        "error_type": "ValueError", "error": "bad candidate",
        "attempt": 1, "time_unix": 123.0,
    }


def test_zero_yield_canary_does_not_kill_productive_wave_sibling(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "canary-manifest.json"
    parent = background_collection.build_canary_manifest(
        revision="a" * 40, output_root=tmp_path / "canary",
        scene_catalog={name: [f"{name}-0", f"{name}-1"]
                       for name in background_collection.DATASETS},
        selected_scenes={name: [f"{name}-0", f"{name}-1"]
                         for name in background_collection.DATASETS},
        paths=_paths(tmp_path),
        source_manifest_sha256={
            "r2r": "4" * 64, "gs": "5" * 64, "b1k": "3" * 64})
    manifest_path.write_text(json.dumps(parent))
    jobs = parent["jobs"][:2]

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 130 if self.pid == 101 else 0

    monkeypatch.setattr(
        background_collection, "_default_launcher",
        lambda job, *_args: Process(101 if job is jobs[0] else 102))
    monkeypatch.setattr(run_background_collection.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        background_collection, "job_health",
        lambda job, **kwargs: {
            "status": "healthy", "backend_ready_time": 2.0})
    source = {
        "path": str(tmp_path / "records.jsonl"),
        "records_sha256": "6" * 64,
        "run_meta_sha256": "7" * 64,
    }
    transactions = {
        jobs[0]["job_id"]: {
            "catalog_status": "zero_yield",
            "source_validated_records": 0,
        },
        jobs[1]["job_id"]: {
            "catalog_status": "completed",
            "source_validated_records": 1,
            "sources": [source],
        },
    }
    monkeypatch.setattr(
        background_collection, "validate_scene_transaction",
        lambda job, **kwargs: transactions[job["job_id"]])
    compiled = []
    monkeypatch.setattr(
        run_background_collection, "_compile_canary_source",
        lambda parent, job, value: compiled.append(job["job_id"]) or
        (tmp_path / "candidate_qa"))

    with pytest.raises(ValueError, match="no accepted records"):
        run_background_collection._run_canary_wave(
            manifest_path, parent, jobs)

    assert compiled == [jobs[1]["job_id"]]


def test_canary_cleanup_gives_b1k_supervisor_time_to_reap_worker(
        monkeypatch):
    signals = []

    class Supervisor:
        pid = 4321

        def __init__(self):
            self.alive = True
            self.waits = []

        def poll(self):
            return None if self.alive else -signal.SIGINT

        def wait(self, timeout=None):
            self.waits.append(timeout)
            self.alive = False
            return -signal.SIGINT

    supervisor = Supervisor()
    monkeypatch.setattr(
        background_collection, "terminate_process_group",
        lambda pid, requested: signals.append((pid, requested)))

    run_background_collection._terminate_canary_process(supervisor)

    assert signals == [(4321, signal.SIGINT)]
    assert supervisor.waits == [
        background_collection.config.BACKGROUND_SIGINT_GRACE_S]


def test_canary_initialization_timeout_gets_exactly_one_retry(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "canary-manifest.json"
    parent = background_collection.build_canary_manifest(
        revision="a" * 40, output_root=tmp_path / "canary",
        scene_catalog={name: [f"{name}-0", f"{name}-1"]
                       for name in background_collection.DATASETS},
        selected_scenes={name: [f"{name}-0", f"{name}-1"]
                         for name in background_collection.DATASETS},
        paths=_paths(tmp_path),
        source_manifest_sha256={
            "r2r": "4" * 64, "gs": "5" * 64, "b1k": "3" * 64})
    manifest_path.write_text(json.dumps(parent))
    job = parent["jobs"][0]

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.alive = True

        def poll(self):
            return None if self.alive else -2

    launched = []

    def launch(*_args):
        process = Process(100 + len(launched))
        launched.append(process)
        return process

    monkeypatch.setattr(background_collection, "_default_launcher", launch)
    monkeypatch.setattr(run_background_collection.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        background_collection, "job_health",
        lambda *_args, **_kwargs: {
            "status": "initialization_slow", "backend_ready_time": None})

    def terminate(pid, requested_signal=signal.SIGINT):
        assert requested_signal == signal.SIGINT
        next(process for process in launched if process.pid == pid).alive = False

    monkeypatch.setattr(
        background_collection, "terminate_process_group", terminate)

    with pytest.raises(ValueError, match="lacks backend_ready"):
        run_background_collection._run_canary_wave(
            manifest_path, parent, [job])

    assert len(launched) == 2


def test_canary_retryable_initialization_log_gets_exactly_one_retry(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "canary-manifest.json"
    parent = background_collection.build_canary_manifest(
        revision="a" * 40, output_root=tmp_path / "canary",
        scene_catalog={name: [f"{name}-0", f"{name}-1"]
                       for name in background_collection.DATASETS},
        selected_scenes={name: [f"{name}-0", f"{name}-1"]
                         for name in background_collection.DATASETS},
        paths=_paths(tmp_path),
        source_manifest_sha256={
            "r2r": "4" * 64, "gs": "5" * 64, "b1k": "3" * 64})
    manifest_path.write_text(json.dumps(parent))
    job = parent["jobs"][0]
    Path(job["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["log_path"]).write_text("CUDA out of memory\n")

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 1

    launched = []

    def launch(*_args):
        process = Process(100 + len(launched))
        launched.append(process)
        return process

    monkeypatch.setattr(background_collection, "_default_launcher", launch)
    monkeypatch.setattr(run_background_collection.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        background_collection, "job_health",
        lambda *_args, **_kwargs: {
            "status": "healthy", "backend_ready_time": None})

    with pytest.raises(ValueError, match="lacks backend_ready"):
        run_background_collection._run_canary_wave(
            manifest_path, parent, [job])

    assert len(launched) == 2


def test_canary_retry_budget_survives_supervisor_restart(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "canary-manifest.json"
    parent = background_collection.build_canary_manifest(
        revision="a" * 40, output_root=tmp_path / "canary",
        scene_catalog={name: [f"{name}-0", f"{name}-1"]
                       for name in background_collection.DATASETS},
        selected_scenes={name: [f"{name}-0", f"{name}-1"]
                         for name in background_collection.DATASETS},
        paths=_paths(tmp_path),
        source_manifest_sha256={
            "r2r": "4" * 64, "gs": "5" * 64, "b1k": "3" * 64})
    manifest_path.write_text(json.dumps(parent))
    for job in parent["jobs"]:
        path = Path(job["log_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("CUDA out of memory\n")

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 1

    launched = []

    def launch(*_args):
        process = Process(100 + len(launched))
        launched.append(process)
        return process

    monkeypatch.setattr(background_collection, "_default_launcher", launch)
    monkeypatch.setattr(run_background_collection.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        background_collection, "job_health",
        lambda *_args, **_kwargs: {
            "status": "healthy", "backend_ready_time": None})

    with pytest.raises(ValueError, match="lacks backend_ready"):
        run_background_collection.run_canary_manifest(manifest_path)
    first_launch_count = len(launched)
    # All six jobs get their own initial attempt plus one retry; one broken
    # scene no longer aborts the remaining wave or later wave.
    assert first_launch_count == 12

    with pytest.raises(ValueError, match="retry budget exhausted"):
        run_background_collection.run_canary_manifest(manifest_path)
    assert len(launched) == first_launch_count


def test_capacity_profile_reopens_parent_and_real_candidate_validator(
        tmp_path, monkeypatch):
    evidence = _raw_profile(measurement_root=tmp_path / "measurements")
    parent = tmp_path / "canary-parent.json"
    parent.write_text(json.dumps({"schema": "tampered"}))
    evidence["parent_manifest_path"] = str(parent)
    for row in evidence["datasets"].values():
        for canary in row["canary_scenes"]:
            canary["parent_manifest_path"] = str(parent)
    calls = []
    monkeypatch.setattr(
        background_capacity.candidate_preview,
        "validate_preview_artifact",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {})

    with pytest.raises(ValueError, match="canary parent"):
        background_collection.derive_capacity_profile(evidence)
    assert calls == []


def test_capacity_qa_counts_require_real_candidate_validation(tmp_path):
    source_root = tmp_path / "real-source"
    source_root.mkdir()
    projection = _validated_a1_projection(source_root)
    records = source_root / "records.jsonl"
    metadata = source_root / "run_meta.json"
    metadata_sha = _write_r2r_run_meta_for_records(metadata, records)
    artifact = source_root / "candidate_qa"
    _write_preview_artifact(
        projection, artifact, source_records_path=records,
        source_run_meta_sha256=metadata_sha)

    background_capacity._validate_candidate_artifact(
        artifact, records, metadata)
    items = [json.loads(line) for line in
             (artifact / "public/items.jsonl").read_text().splitlines()]
    (artifact / "public/items.jsonl").write_text("".join(
        json.dumps(row) + "\n" for row in items * 60))

    with pytest.raises(ValueError):
        background_capacity._validate_candidate_artifact(
            artifact, records, metadata)


def test_canary_candidate_publication_adopts_crash_window(
        tmp_path, monkeypatch):
    """Publishing before the state callback must not brick the rerun."""
    output = tmp_path / "canary-job"
    output.mkdir()
    records = output / "records.jsonl"
    metadata = output / "run_meta.json"
    records.write_text("{}\n")
    metadata.write_text("{}\n")
    source = {
        "path": str(records),
        "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "run_meta_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    job = {
        "job_id": "canary-r2r-00", "output_dir": str(output),
        "dataset": "r2r", "scenes": ["r2r-0"],
    }
    parent = {
        "sha256": "a" * 64, "output_root": str(tmp_path),
        "jobs": [job],
    }
    builds = []

    def build(_records, artifact, report, **_kwargs):
        builds.append(Path(artifact))
        (Path(artifact) / "public").mkdir(parents=True)
        (Path(artifact) / "private").mkdir()
        for relative in (
                "benchmark.json", "report.json", "public/manifest.json",
                "private/manifest.json", "private/source_map.json"):
            path = Path(artifact) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        Path(report).mkdir(parents=True)
        (Path(report) / "index.html").write_text("canary")

    monkeypatch.setattr(
        run_background_collection.candidate_preview,
        "build_main_preview", build)
    monkeypatch.setattr(
        run_background_collection.candidate_preview,
        "validate_preview_artifact", lambda *_args, **_kwargs: {})
    validations = []
    monkeypatch.setattr(
        background_capacity, "_validate_candidate_artifact",
        lambda *_args, **_kwargs: validations.append("validated"))

    first = run_background_collection._compile_canary_source(
        parent, job, source)
    # build_main_preview already runs production source validation, compiler,
    # artifact validation, and report rendering.  Publishing its result must
    # not repeat the deterministic compiler projection.
    assert validations == []
    # Simulate termination after publication and before the state callback.
    second = run_background_collection._compile_canary_source(
        parent, job, source)
    assert validations == []

    assert second == first
    assert len(builds) == 1
    assert first.is_dir()
    bundle = output / "capacity_candidate"
    assert (bundle / "candidate_qa_report/index.html").is_file()
    assert (bundle / "publication.json").is_file()
    state = {
        "schema": run_background_collection._CANARY_STATE_SCHEMA,
        "parent_manifest_sha256": parent["sha256"],
        "status": "running", "jobs": {}, "failures": {}, "attempts": {},
    }
    run_background_collection._complete_canary_state_job(
        parent, state, job, first, source)
    assert validations == []
    reopened = run_background_collection._load_canary_state(parent)
    assert validations == []
    assert reopened["jobs"][job["job_id"]]["artifact"] == str(first)
    (bundle / "candidate_qa_report/index.html").write_text("tampered")
    with pytest.raises(ValueError, match="publication binding"):
        run_background_collection._compile_canary_source(
            parent, job, source)
    with pytest.raises(ValueError, match="publication binding"):
        run_background_collection._load_canary_state(parent)


def test_reopening_authenticated_canary_publication_does_not_recompile(
        tmp_path, monkeypatch):
    """A bound publication is the durable compiler trust boundary."""
    output = tmp_path / "canary-job"
    bundle = output / "capacity_candidate"
    artifact = bundle / "candidate_qa"
    report = bundle / "candidate_qa_report"
    artifact.mkdir(parents=True)
    report.mkdir()
    (artifact / "payload.json").write_text("{}\n")
    (report / "index.html").write_text("canary")
    records = output / "records.jsonl"
    metadata = output / "run_meta.json"
    records.write_text("{}\n")
    metadata.write_text("{}\n")
    source = {
        "path": str(records),
        "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "run_meta_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    parent = {"sha256": "a" * 64}
    job = {"job_id": "canary-r2r-00", "output_dir": str(output)}
    files = {
        "candidate_qa/payload.json": hashlib.sha256(
            (artifact / "payload.json").read_bytes()).hexdigest(),
        "candidate_qa_report/index.html": hashlib.sha256(
            (report / "index.html").read_bytes()).hexdigest(),
    }
    body = {
        "schema": "egoconseq.capacity-canary-publication.v1",
        "parent_manifest_sha256": parent["sha256"],
        "job_id": job["job_id"],
        "source": source,
        "artifact": "candidate_qa",
        "report": "candidate_qa_report",
        "files": files,
    }
    (bundle / "publication.json").write_text(json.dumps({
        **body,
        "sha256": hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("ascii")).hexdigest(),
    }))

    def reject_recompile(*_args, **_kwargs):
        raise AssertionError("published candidate was recompiled")

    monkeypatch.setattr(
        background_capacity, "_validate_candidate_artifact",
        reject_recompile)

    reopened, reopened_report, _identity = \
        run_background_collection._load_canary_publication(
            parent, job, source)

    assert reopened == artifact.resolve()
    assert reopened_report == report.resolve()


@pytest.mark.parametrize("relative", [
    "candidate_qa/public/items.jsonl", "candidate_qa/benchmark.json",
])
def test_checkpoint_real_payload_mutation_fails_every_trust_boundary(
        tmp_path, monkeypatch, relative):
    manifest = _synthetic_manifest(tmp_path, monkeypatch)
    source_root = tmp_path / "real-checkpoint-source"
    source_root.mkdir()
    projection = _validated_a1_projection(source_root)
    records = source_root / "records.jsonl"
    metadata = source_root / "run_meta.json"
    metadata_sha = _write_r2r_run_meta_for_records(metadata, records)
    artifact = source_root / "candidate_qa"
    _write_preview_artifact(
        projection, artifact, source_records_path=records,
        source_run_meta_sha256=metadata_sha)
    coverage = json.loads(
        (artifact / "benchmark.json").read_text())["coverage"]
    checkpoint = "real-payload"
    final, staging = background_collection.background_checkpoint.prepare(
        manifest, checkpoint)
    shutil.copytree(artifact, staging / "candidate_qa")
    macro = background_collection.six_task_macro_report({
        task: 1.0 for task in TASKS})
    result = {
        "artifact": str(final / "candidate_qa"), "coverage": coverage,
        "gt_as_pred": 1.0, "six_task_macro": macro,
        "datasets": {dataset: {
            "quota": {"complete": True}, "six_task_macro": macro,
        } for dataset in background_collection.DATASETS},
    }
    result_files = []
    for dataset in background_collection.DATASETS:
        for name, value in (
                (f"quota-{dataset}.json", {"complete": True}),
                (f"six_task_macro-{dataset}.json", macro)):
            (staging / name).write_text(json.dumps(value))
            result_files.append(name)
    (staging / "gt_replay.json").write_text(json.dumps({"overall": 1.0}))
    (staging / "six_task_macro.json").write_text(json.dumps(macro))
    result_files.extend(("gt_replay.json", "six_task_macro.json"))
    source = {
        "path": str(records),
        "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "run_meta_sha256": metadata_sha,
    }
    identity = background_collection.background_checkpoint.publish(
        manifest, checkpoint, staging, result=result,
        result_files=result_files, sources=[source])
    state = background_collection.initial_state(manifest)
    state.update({"status": "running", "compile_checkpoints": [checkpoint]})
    for dataset in background_collection.DATASETS:
        state["datasets"][dataset] = {
            **result["datasets"][dataset], "artifact": result["artifact"],
            "coverage": coverage, "gt_as_pred": 1.0,
            "combined_six_task_macro": macro, "checkpoint": checkpoint,
            "checkpoint_summary": identity,
        }
    background_collection.validate_state(manifest, state)
    payload = final / relative
    payload.write_bytes(payload.read_bytes() + b"\n")

    with pytest.raises(ValueError):
        background_collection.background_checkpoint.load(
            manifest, checkpoint, identity=identity)
    with pytest.raises(ValueError):
        background_collection.validate_state(manifest, state)
    assert background_collection.dataset_can_stop(
        manifest, state, "r2r") is False


def test_checkpoint_load_revalidates_candidate_payloads(
        tmp_path, monkeypatch):
    manifest, state, checkpoint = _published_checkpoint(
        tmp_path, monkeypatch)
    summary = state["datasets"]["r2r"]["checkpoint_summary"]
    calls = []

    def reject(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("candidate payload changed")

    monkeypatch.setattr(
        background_collection.candidate_preview,
        "validate_preview_artifact", reject)

    with pytest.raises(ValueError, match="candidate payload"):
        background_collection.background_checkpoint.load(
            manifest, checkpoint, identity=summary)
    with pytest.raises(ValueError, match="candidate payload"):
        background_collection.validate_state(manifest, state)
    assert background_collection.dataset_can_stop(
        manifest, state, "r2r") is False
    assert len(calls) == 2


def _published_checkpoint(tmp_path, monkeypatch):
    """Build the smallest internally consistent persisted checkpoint."""
    manifest = _synthetic_manifest(tmp_path, monkeypatch)
    checkpoint = "catalog-pass-00"
    final, staging = background_collection.background_checkpoint.prepare(
        manifest, checkpoint)
    artifact_files = (
        "candidate_qa/report.json",
        "candidate_qa/public/manifest.json",
        "candidate_qa/private/manifest.json",
        "candidate_qa/private/source_map.json",
        "candidate_qa/benchmark.json",
        "candidate_qa/public/items.jsonl",
        "candidate_qa/private/answers.jsonl",
        "candidate_qa/private/atoms.jsonl",
        "candidate_qa/private/record_contexts.jsonl",
    )
    for relative in artifact_files:
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    macro = background_collection.six_task_macro_report({
        task: 1.0 for task in TASKS})
    compiled = {
        "artifact": str(final / "candidate_qa"),
        "coverage": {}, "gt_as_pred": 1.0,
        "six_task_macro": macro,
        "datasets": {dataset: {
            "quota": {"complete": True}, "six_task_macro": macro,
        } for dataset in background_collection.DATASETS},
    }
    result_files = []
    for dataset in background_collection.DATASETS:
        for name, value in (
                (f"quota-{dataset}.json",
                 compiled["datasets"][dataset]["quota"]),
                (f"six_task_macro-{dataset}.json", macro)):
            (staging / name).write_text(json.dumps(value))
            result_files.append(name)
    (staging / "gt_replay.json").write_text(json.dumps({"overall": 1.0}))
    (staging / "six_task_macro.json").write_text(json.dumps(macro))
    result_files.extend(("gt_replay.json", "six_task_macro.json"))
    source = tmp_path / "checkpoint-source" / "records.jsonl"
    source.parent.mkdir()
    source.write_text("{}\n")
    metadata = source.with_name("run_meta.json")
    metadata.write_text("{}\n")
    (staging / "candidate_qa/private/source_map.json").write_text(json.dumps({
        "source_authority": {
            "schema": "egoconseq.preview-source-authority.v3",
            "authority_kind": "cli_external",
            "authority_id": "round3-checkpoint",
            "sources": [{
                "locator": {"kind": "authority_id", "id": "source-0"},
                "records_sha256": hashlib.sha256(
                    source.read_bytes()).hexdigest(),
                "run_meta_sha256": hashlib.sha256(
                    metadata.read_bytes()).hexdigest(),
            }],
        },
    }))
    identity = background_collection.background_checkpoint.publish(
        manifest, checkpoint, staging, result=compiled,
        result_files=result_files,
        sources=[{
            "path": str(source),
            "records_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "run_meta_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }])
    state = background_collection.initial_state(manifest)
    state.update({"status": "running", "compile_checkpoints": [checkpoint]})
    for dataset in background_collection.DATASETS:
        state["datasets"][dataset] = {
            **compiled["datasets"][dataset],
            "artifact": compiled["artifact"], "coverage": {},
            "gt_as_pred": 1.0, "combined_six_task_macro": macro,
            "checkpoint": checkpoint, "checkpoint_summary": identity,
        }
    return manifest, state, checkpoint


def test_recovered_returncode_requires_recovered_valid_status(
        tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path, monkeypatch)
    state = background_collection.initial_state(manifest)
    state["status"] = "running"
    job = manifest["rounds"][0]["jobs"][0]
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    metadata = output / "run_meta.json"
    records.write_text("{}\n")
    metadata.write_text("{}\n")
    state["jobs"][job["job_id"]] = {
        "job_id": job["job_id"], "dataset": job["dataset"],
        "scene_id": job["scenes"][0],
        "round_index": job["round_index"],
        "returncode": background_collection.RECOVERED_UNKNOWN_RETURNCODE,
        "status": "failed_partial", "catalog_status": "completed",
        "source_validation": {"sources": [{
            "path": str(records),
            "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
            "run_meta_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }]},
    }

    with pytest.raises(ValueError, match="recovered.*transition"):
        background_collection.validate_state(manifest, state)


def test_gs_exclusion_is_counted_in_catalog_coverage(tmp_path, monkeypatch):
    manifest = _synthetic_manifest(tmp_path, monkeypatch)
    template = next(job for row in manifest["rounds"] for job in row["jobs"]
                    if job["dataset"] == "gs")
    jobs = []
    state = background_collection.initial_state(manifest)
    state["status"] = "running"
    for index in range(54):
        job = copy.deepcopy(template)
        job["job_id"] = f"gs-catalog-{index:02d}"
        job["scenes"] = [f"gs-{index:02d}"]
        job["catalog_pass"] = 0
        jobs.append(job)
        state["jobs"][job["job_id"]] = {
            "catalog_status": "completed"}
    manifest["rounds"] = [{
        "round_index": 0, "catalog_pass": 0, "jobs": jobs}]

    coverage = background_collection.dataset_catalog_coverage(
        manifest, state, "gs")

    assert coverage["counts"] == {
        "catalog": 55, "scheduled": 54, "touched": 54,
        "terminal": 55, "produced": 54,
    }
    assert coverage["states"]["excluded"] == 1
    assert coverage["complete"] is True
