"""Controller closeout keeps quota, authority, and scheduling fail-closed."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from pipeline import (
    action_proposal, background_capacity, background_collection,
    background_recovery, collection_closeout, collection_cli,
    validate as record_validation,
)
from scripts import run_background_collection
from post_QA.seen_build import spec as seen_spec
from tests._synthetic import capacity_evidence


TASKS = seen_spec.TASKS


def _write_finalization(
        output: Path, *, status: str, run_contract_sha256: str,
        source_validation: str = "passed", record_count: int = 1) -> None:
    paths = {
        "records_sha256": output / "records.jsonl",
        "run_meta_sha256": output / "run_meta.json",
        "funnel_sha256": output / "collection_funnel.json",
    }
    (output / "collection_finalization.json").write_text(json.dumps({
        "schema": collection_closeout.FINALIZATION_SCHEMA,
        "status": status,
        "source_validation": source_validation,
        "record_count": record_count,
        "run_contract_sha256": run_contract_sha256,
        **{
            field: hashlib.sha256(path.read_bytes()).hexdigest()
            for field, path in paths.items()
        },
    }, sort_keys=True))


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
def _capacity_binding_fields() -> dict:
    return {
        "poses_per_scene": 10,
        "pose_candidates_per_scene": 20,
        "record_idle_stop_s":
            background_collection.config.BACKGROUND_RECORD_IDLE_STOP_S,
        "scene_wallclock_stop_s": 600,
        "supervisor_timeout_s": None,
        "seed": 7,
        "benchmark_partition": "train_seen",
        "ordinary_actions_per_pose": 36,
    }


def _capacity_run_params() -> dict:
    return {
        "poses_per_scene": 10,
        "pose_candidates_per_scene": 20,
        "record_idle_stop_s":
            background_collection.config.BACKGROUND_RECORD_IDLE_STOP_S,
        "scene_wallclock_stop_s": 600.0,
        "seed": 7,
        "radii": list(background_collection.config.RADII_M),
        "camera_heights": list(
            background_collection.config.BENCH_CAMERA_HEIGHTS_M),
        "lengths": list(background_collection.config.GEN_LENGTHS),
        "proposal_pairs_per_length": action_proposal.PAIRS_PER_LENGTH_DEFAULT,
        "proposal_natural_per_length":
            action_proposal.NATURAL_PER_LENGTH_DEFAULT,
        "ordinary_actions_per_pose": 36,
        "benchmark_partition": "train_seen",
        "keep_per_length": background_collection.config.KEEP_PER_LENGTH,
        "action_mode": "balanced",
    }


def _write_spool_run_contract(
        records: Path, *, params: dict, scene_id: str,
        revision: str = "a" * 40,
        source_manifest_sha256: str = "4" * 64) -> str:
    run_contract = {
        "params": copy.deepcopy(params),
        "resolved_scenes": [{
            "scene_id": scene_id,
            "source_dataset": params["backend"],
            "source_manifest_sha256": source_manifest_sha256,
        }],
        "code_revision": revision,
        "code_dirty": False,
    }
    spool = records.parent / f".{records.name}.groups"
    spool.mkdir()
    (spool / "contract.json").write_text(json.dumps({
        "oracle_contract_version": collection_cli.ORACLE_CONTRACT_VERSION,
        "expected_siblings": 1,
        "run": run_contract,
    }, sort_keys=True))
    return background_collection._canonical_sha256(run_contract)


def _raw_profile(counts=None, *, measurement_root: Path = None) -> dict:
    if measurement_root is None:
        raise ValueError("capacity evidence fixture requires a path")
    return capacity_evidence(measurement_root, counts=counts)


def _audit(b1k_scenes) -> dict:
    installed = [*b1k_scenes, "b1k-excluded"]
    return {
        "schema": "b1k-catalog-authority-audit.v1",
        "selection_scope": "b1k-catalog-authority-audit.v1",
        "installed_scene_ids": installed,
        "accepted": [{"scene_id": scene_id} for scene_id in b1k_scenes],
        "excluded": [{
            "scene_id": "b1k-excluded",
            "terminal": "excluded_deterministic_failure",
        }],
        "source_identities": {
            "install_manifest": {"path": "/data/install.json", "sha256": "1" * 64},
            "shards": [],
        },
    }


def _paths(tmp_path: Path) -> dict:
    return {
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
    }


def _manifest(tmp_path: Path, *, rounds=1) -> dict:
    r2r = [f"r2r-{index}" for index in range(8)]
    gs = [f"gs-{index}" for index in range(8)]
    b1k = [f"b1k-{index}" for index in range(12)]
    profile = background_collection.derive_capacity_profile(_raw_profile(
        measurement_root=tmp_path / "measurements"))
    return background_collection.build_manifest(
        revision="a" * 40,
        output_root=tmp_path / "run",
        r2r_scenes=r2r,
        gs_scenes=gs,
        b1k_scenes=b1k,
        paths=_paths(tmp_path),
        b1k_catalog_audit=_audit(b1k),
        b1k_catalog_audit_sha256="2" * 64,
        r2r_source_manifest_sha256="4" * 64,
        gs_source_manifest_sha256="5" * 64,
        b1k_source_manifest_sha256="3" * 64,
        b1k_source_scene_ids=b1k,
        capacity_profile=profile,
        capacity_profile_sha256=profile["sha256"],
        rounds=rounds,
    )


def test_manifest_separates_source_catalog_from_train_schedule(tmp_path):
    source = {
        "r2r": [f"r2r-{index}" for index in range(8)],
        "gs": [f"gs-{index}" for index in range(8)],
        "b1k": [f"b1k-{index}" for index in range(12)],
    }
    scheduled = {
        "r2r": source["r2r"][:6],
        "gs": source["gs"][:6],
        "b1k": source["b1k"][:10],
    }
    profile = background_collection.derive_capacity_profile(_raw_profile(
        measurement_root=tmp_path / "source-schedule"))

    manifest = background_collection.build_manifest(
        revision="b" * 40,
        output_root=tmp_path / "run",
        r2r_scenes=scheduled["r2r"],
        gs_scenes=scheduled["gs"],
        b1k_scenes=scheduled["b1k"],
        source_scene_catalog=source,
        paths=_paths(tmp_path),
        b1k_catalog_audit=_audit(source["b1k"]),
        b1k_catalog_audit_sha256="2" * 64,
        r2r_source_manifest_sha256="4" * 64,
        gs_source_manifest_sha256="5" * 64,
        b1k_source_manifest_sha256="3" * 64,
        b1k_source_scene_ids=source["b1k"],
        capacity_profile=profile,
        capacity_profile_sha256=profile["sha256"],
        rounds=2,
    )

    assert manifest["source_scene_catalog"] == source
    assert manifest["scene_catalog"] == scheduled
    for catalog_pass in range(2):
        for dataset in background_collection.DATASETS:
            actual = [
                job["scenes"][0]
                for row in manifest["rounds"]
                if row["catalog_pass"] == catalog_pass
                for job in row["jobs"]
                if job["dataset"] == dataset
            ]
            assert sorted(actual) == sorted(scheduled[dataset])
    background_collection.validate_manifest(manifest)


def test_capacity_profile_derives_the_approved_two_scene_formulas(tmp_path):
    raw = _raw_profile(
        {"r2r": 10, "gs": 10, "b1k": 10},
        measurement_root=tmp_path / "measurements")
    for row in raw["datasets"].values():
        first, second = row["canary_scenes"]
        for canary, ready, finished in (
                (first, 120.0, 160.0), (second, 115.0, 135.0)):
            path = Path(canary["controller_events_path"])
            events = [json.loads(line) for line in path.read_text().splitlines()]
            events[0]["time_unix"] = 100.0
            events[1]["time_unix"] = ready
            events[2]["time_unix"] = finished
            path.write_text("".join(json.dumps(event) + "\n"
                                    for event in events))

    profile = background_collection.derive_capacity_profile(raw)
    row = profile["datasets"]["r2r"]

    assert row["pooled_total_yield_per_record"] == 7.0
    assert row["pooled_unique_frame_yield_per_record"] == 1.0
    assert row["ordinary_actions_per_pose"] == 36
    assert row["pooled_task_yield_per_record"] == {
        # The historical canary fixture predates B3; its B3 yield is zero.
        task_id: 0.0 if task_id == "B3" else 1.0 for task_id in TASKS}
    assert row["pooled_pose_acceptance_rate"] == 0.5
    assert row["max_initialization_s"] == 20.0
    assert row["max_pose_attempt_s"] == 2.0
    assert row["unique_frames_needed"] == 3000
    assert row["records_needed"] == 3000
    assert row["records_per_scene"] == 20
    assert row["pose_attempt_cap"] == 800
    assert row["scene_wallclock_s"] == 600
    assert len(profile["sha256"]) == 64
    assert profile["ordinary_actions_per_pose"] == 36
    background_collection.validate_capacity_profile(profile)


def test_capacity_wallclock_preserves_per_scene_cost_correlation():
    tasks = {task_id: 10 for task_id in TASKS}
    tasks["C1"] = 1
    canaries = [
        {
            "scene_id": "low-yield-fast",
            "initialization_s": 36.0,
            "pose_attempts": 8000,
            "accepted_records": 2,
            "accepted_unique_frames": 2,
            "ordinary_actions_per_pose": 36,
            "pose_attempt_s": 0.14,
            "task_qa_counts": tasks,
        },
        {
            "scene_id": "high-yield-slow",
            "initialization_s": 157.0,
            "pose_attempts": 40,
            "accepted_records": 20,
            "accepted_unique_frames": 20,
            "ordinary_actions_per_pose": 36,
            "pose_attempt_s": 20.0,
            "task_qa_counts": tasks,
        },
    ]

    row = background_capacity._derive_dataset("b1k", 50, canaries)

    assert row["records_per_scene"] == 40
    assert row["pose_attempt_cap"] == 6000
    assert row["scene_wallclock_s"] == 1800


@pytest.mark.parametrize("mutation,match", [
    ("one_canary", "exactly two"),
    ("zero_records", "accepted records"),
    ("zero_attempts", "pose attempts"),
])
def test_capacity_profile_fails_closed_on_unlaunchable_canaries(
        tmp_path, mutation, match):
    raw = _raw_profile(measurement_root=tmp_path / mutation)
    row = raw["datasets"]["r2r"]
    if mutation == "one_canary":
        row["canary_scenes"] = row["canary_scenes"][:1]
    elif mutation == "zero_records":
        meta = Path(row["canary_scenes"][0]["run_meta_path"])
        value = json.loads(meta.read_text())
        value["record_count"] = 0
        meta.write_text(json.dumps(value))
    elif mutation == "zero_attempts":
        canary = row["canary_scenes"][0]
        run_meta = Path(canary["run_meta_path"])
        value = json.loads(run_meta.read_text())
        value["stats"]["pose_attempts"] = 0
        run_meta.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        background_collection.derive_capacity_profile(raw)


def test_manifest_binds_dataset_quota_audit_capacity_and_gpu_affinity(
        tmp_path):
    manifest = _manifest(tmp_path)

    assert manifest["schema"] == background_collection.CONTROLLER_SCHEMA
    assert manifest["quota"] == {
        dataset: {
            "supported_tasks": list(seen_spec.supported_tasks(dataset)),
            "task_totals": dict(seen_spec.DATASET_TASK_TOTALS[dataset]),
            "minimum_turn_first_fraction":
                seen_spec.MINIMUM_TURN_FIRST_FRACTION,
        }
        for dataset in background_collection.DATASETS
    }
    assert manifest["authorities"]["b1k_catalog_audit"]["sha256"] == \
        "2" * 64
    assert manifest["authorities"]["capacity_profile"]["sha256"] == \
        manifest["capacity_profile"]["sha256"]
    assert manifest["authorities"]["b1k_source_manifest"] == {
        "sha256": "3" * 64, "accepted_scene_ids": manifest["scene_catalog"]["b1k"]}
    assert manifest["rounds"][0]["jobs"]
    assert {job["dataset"] for job in manifest["rounds"][0]["jobs"]} == {
        "r2r", "gs", "b1k"}
    gpu_by_dataset = {"gs": 0, "b1k": 1, "r2r": 2}
    for round_value in manifest["rounds"]:
        for job in round_value["jobs"]:
            assert job["gpu_id"] == gpu_by_dataset[job["dataset"]]
            capacity = manifest["capacity_profile"]["datasets"][job["dataset"]]
            assert job["weight"] == capacity["scene_wallclock_s"]
            assert job["scene_wallclock_s"] == capacity["scene_wallclock_s"]
            command = job["command"]
            assert command[command.index("--poses-per-scene") + 1] == \
                str(capacity["records_per_scene"])
            assert command[
                command.index("--pose-candidates-per-scene") + 1] == \
                str(capacity["pose_attempt_cap"])
            assert command[command.index("--record-idle-stop-s") + 1] == \
                str(background_collection.config.BACKGROUND_RECORD_IDLE_STOP_S)
            assert command[
                command.index("--scene-wallclock-stop-s") + 1] == \
                str(capacity["scene_wallclock_s"])
            if job["dataset"] == "b1k":
                assert command[command.index("--scene-timeout-s") + 1] == str(int(
                    background_collection.config.
                    BACKGROUND_INITIALIZATION_DEADLINE_S +
                    capacity["scene_wallclock_s"] +
                    background_collection.config.
                    BACKGROUND_CAPACITY_HANDOFF_GRACE_S +
                    300.0 +
                    background_collection.config.
                    BACKGROUND_FINALIZATION_GRACE_S))
            assert job["transaction_binding"]["scene_id"] == job["scenes"][0]
            assert job["transaction_binding"]["collection_shard_id"] == (
                f"{job['job_id']}-{job['scenes'][0]}"
                if job["dataset"] == "b1k" else job["job_id"])
            assert job["transaction_binding"]["revision"] == "a" * 40


def test_controller_gs_catalog_has_one_manifest_bound_exclusion(
        tmp_path, monkeypatch):
    excluded = "interior_0505_839970"
    accepted = [f"interior_{index:04d}_scene" for index in range(54)]
    rows = [
        {"scene_id": scene_id, "path": f"train/{scene_id}",
         "split": "train"}
        for scene_id in [*accepted, excluded]
    ]
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({
        "schema_version": "egoconseq.gs-scene-manifest.v1",
        "dataset": "gs", "scenes": rows,
    }))
    root = tmp_path / "gs"
    (root / "train" / excluded).mkdir(parents=True)

    def discover(root_value, manifest_value, *, requested):
        assert Path(root_value) == root
        assert Path(manifest_value) == manifest
        assert set(requested) == set(accepted)
        return [SimpleNamespace(scene_id=scene_id) for scene_id in requested]

    monkeypatch.setattr(
        run_background_collection.scene_pool, "discover_gs_train_scenes",
        discover)
    specs = run_background_collection._controller_gs_specs(root, manifest)
    assert len(specs) == 54
    (root / "train" / excluded / "scene.collision.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="exclusion evidence"):
        run_background_collection._controller_gs_specs(root, manifest)


def test_manifest_rejects_audit_or_profile_identity_mismatch(tmp_path):
    r2r = [f"r2r-{index}" for index in range(8)]
    gs = [f"gs-{index}" for index in range(8)]
    b1k = [f"b1k-{index}" for index in range(12)]
    profile = background_collection.derive_capacity_profile(_raw_profile(
        measurement_root=tmp_path / "identity"))
    audit = _audit(b1k)

    with pytest.raises(ValueError, match="accepted set"):
        background_collection.build_manifest(
            revision="a" * 40, output_root=tmp_path / "run",
            r2r_scenes=r2r, gs_scenes=gs, b1k_scenes=[*b1k[:-1], "other"],
            paths=_paths(tmp_path), b1k_catalog_audit=audit,
            b1k_catalog_audit_sha256="2" * 64,
            r2r_source_manifest_sha256="4" * 64,
            gs_source_manifest_sha256="5" * 64,
            b1k_source_manifest_sha256="3" * 64,
            b1k_source_scene_ids=b1k,
            capacity_profile=profile,
            capacity_profile_sha256=profile["sha256"])
    with pytest.raises(ValueError, match="capacity profile digest"):
        background_collection.build_manifest(
            revision="a" * 40, output_root=tmp_path / "run",
            r2r_scenes=r2r, gs_scenes=gs, b1k_scenes=b1k,
            paths=_paths(tmp_path), b1k_catalog_audit=_audit(b1k),
            b1k_catalog_audit_sha256="2" * 64,
            r2r_source_manifest_sha256="4" * 64,
            gs_source_manifest_sha256="5" * 64,
            b1k_source_manifest_sha256="3" * 64,
            b1k_source_scene_ids=b1k,
            capacity_profile=profile,
            capacity_profile_sha256="3" * 64)

    revised = background_collection.build_manifest(
        revision="b" * 40, output_root=tmp_path / "revision-run",
        r2r_scenes=r2r, gs_scenes=gs, b1k_scenes=b1k,
        paths=_paths(tmp_path), b1k_catalog_audit=_audit(b1k),
        b1k_catalog_audit_sha256="2" * 64,
        r2r_source_manifest_sha256="4" * 64,
        gs_source_manifest_sha256="5" * 64,
        b1k_source_manifest_sha256="3" * 64,
        b1k_source_scene_ids=b1k,
        capacity_profile=profile,
        capacity_profile_sha256=profile["sha256"])
    assert revised["revision"] == "b" * 40
    assert revised["capacity_profile"]["canary_revision"] == "a" * 40

    with pytest.raises(ValueError, match="source authority"):
        background_collection.build_manifest(
            revision="a" * 40, output_root=tmp_path / "source-run",
            r2r_scenes=r2r, gs_scenes=gs, b1k_scenes=b1k,
            paths=_paths(tmp_path), b1k_catalog_audit=_audit(b1k),
            b1k_catalog_audit_sha256="2" * 64,
            r2r_source_manifest_sha256="6" * 64,
            gs_source_manifest_sha256="5" * 64,
            b1k_source_manifest_sha256="3" * 64,
            b1k_source_scene_ids=b1k,
            capacity_profile=profile,
            capacity_profile_sha256=profile["sha256"])


def test_manifest_validation_rejects_a_rehashed_weakened_quota(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["quota"]["b1k"]["min_per_task"] = 1
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = background_collection._canonical_sha256(body)

    with pytest.raises(ValueError, match="dataset quotas"):
        background_collection.validate_manifest(manifest)


def test_catalog_state_counts_partial_valid_as_reusable_terminal(tmp_path):
    manifest = _manifest(tmp_path)
    state = {"jobs": {}}
    jobs = [
        job for round_value in manifest["rounds"]
        for job in round_value["jobs"]
        if job["dataset"] == "b1k" and job["catalog_pass"] == 0
    ]
    statuses = [
        *(["completed"] * 9), "partial_valid", "zero_yield", "failed"]
    for job, status in zip(jobs, statuses):
        state["jobs"][job["job_id"]] = {
            "catalog_status": status,
            "source_validation": {
                "source_validated_records": 4 if status in {
                    "completed", "partial_valid"} else 0,
                "length_shortfall": ["L6"] if status == "partial_valid" else [],
            },
        }

    coverage = background_collection.dataset_catalog_coverage(
        manifest, state, "b1k")

    assert coverage["counts"] == {
        "catalog": 13,
        "scheduled": 12,
        "touched": 12,
        "terminal": 13,
        "produced": 10,
    }
    assert coverage["states"] == {
        "completed": 9,
        "partial_valid": 1,
        "zero_yield": 1,
        "excluded": 1,
        "failed": 1,
        "unresolved": 0,
    }
    assert coverage["completed_scene_transactions"] == 9
    assert coverage["reusable_scene_transactions"] == 10
    assert coverage["complete"] is True
    assert coverage["capacity_exhausted"] is False
    del state["jobs"][jobs[-1]["job_id"]]
    incomplete = background_collection.dataset_catalog_coverage(
        manifest, state, "b1k")
    assert incomplete["complete"] is False
    assert incomplete["capacity_exhausted"] is None


def test_frame_first_stop_uses_baseline_plus_authenticated_records(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["target_pose_diverse_frames"] = 10000
    manifest["seed_pose_exclusions"] = {
        "r2r": {"representative_count": 4000},
        "gs": {"representative_count": 3000},
        "b1k": {"representative_count": 2500},
    }
    state = {"jobs": {}}
    for round_value in manifest["rounds"]:
        for job in round_value["jobs"]:
            if job["catalog_pass"] == 0:
                state["jobs"][job["job_id"]] = {
                    "catalog_status": "completed",
                    "source_validation": {
                        "source_validated_records": (
                            500 if job["dataset"] == "b1k" else 10),
                    },
                }

    progress = background_collection.frame_progress(manifest, state)

    assert progress["datasets"]["b1k"]["pose_diverse_frames"] == 8500
    assert progress["global_pose_diverse_frames"] > 10000
    assert background_collection.dataset_can_stop(
        manifest, state, "b1k") is True
    assert background_collection.dataset_can_stop(
        manifest, state, "r2r") is True


def test_continuous_frame_progress_counts_derived_jobs(tmp_path):
    manifest = _manifest(tmp_path, rounds=0)
    job = background_collection.continuous_job(
        manifest, "gs", catalog_pass=1, scene_index=0)
    state = {"jobs": {job["job_id"]: {
        "dataset": "gs", "scene_id": job["scenes"][0],
        "catalog_pass": 1, "scene_index": 0,
        "catalog_status": "completed",
        "source_validation": {"source_validated_records": 7},
    }}}

    progress = background_collection.frame_progress(manifest, state)

    assert progress["datasets"]["gs"]["pose_diverse_frames"] == 7


def test_dataset_stop_is_a_pure_in_memory_decision(monkeypatch):
    manifest = {
        "target_pose_diverse_frames": 3,
        "seed_pose_exclusions": {
            dataset: {"representative_count": 1}
            for dataset in background_collection.DATASETS},
        "rounds": [],
    }
    state = {"jobs": {}}
    monkeypatch.setattr(
        background_collection, "validate_state",
        lambda *_args: pytest.fail("dataset stop reopened controller state"))
    monkeypatch.setattr(
        background_collection.background_checkpoint, "load",
        lambda *_args, **_kwargs: pytest.fail(
            "dataset stop reopened a compiled artifact"))
    monkeypatch.setattr(background_collection, "dataset_catalog_coverage",
                        lambda *_args, **_kwargs: {"complete": True})

    assert background_collection.dataset_can_stop(
        manifest, state, "r2r") is False


def test_status_scans_only_live_jobs(monkeypatch):
    manifest = {
        "collection": {"saturated_datasets": ["gs"]},
        "seed_pose_exclusions": {
            "r2r": {"representative_count": 10, "record_count": 12},
            "gs": {"representative_count": 20, "record_count": 23},
            "b1k": {"representative_count": 30, "record_count": 34},
        },
        "rounds": [{"jobs": [
        {"job_id": "done", "dataset": "r2r", "scenes": ["a"]},
        {"job_id": "live", "dataset": "gs", "scenes": ["b"]},
    ]}]}
    state = {
        "schema": "state", "status": "running", "current_round": 1,
        "jobs": {
            "done": {"status": "completed", "durable_records": 7},
            "live": {"status": "running", "durable_records": 1},
        },
        "datasets": {}, "compile_checkpoints": [],
        "exhausted_datasets": ["gs"],
        "updated_time_unix": 1.0,
    }
    calls = []
    monkeypatch.setattr(
        background_collection, "job_progress",
        lambda job: calls.append(job["job_id"]) or {"durable_records": 2})
    monkeypatch.setattr(
        background_collection, "dataset_catalog_coverage",
        lambda _manifest, _state, dataset: {"dataset": dataset})

    value = run_background_collection._status_value(manifest, state)

    assert calls == ["live"]
    assert value["jobs"]["done"]["durable_records"] == 7
    assert value["jobs"]["live"]["durable_records"] == 2
    assert value["progress"]["r2r"] == {
        "baseline_records": 12,
        "baseline_pose_diverse_frames": 10,
        "incremental_records": 7,
        "total_records": 19,
        "saturated": False,
        "exhausted": False,
    }
    assert value["progress"]["gs"] == {
        "baseline_records": 23,
        "baseline_pose_diverse_frames": 20,
        "incremental_records": 2,
        "total_records": 25,
        "saturated": True,
        "exhausted": True,
    }


def test_status_derives_a_live_continuous_job(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, rounds=0)
    job = background_collection.continuous_job(
        manifest, "gs", catalog_pass=1, scene_index=0)
    state = background_collection.initial_state(manifest)
    state["status"] = "running"
    state["jobs"][job["job_id"]] = {
        "job_id": job["job_id"], "dataset": "gs",
        "scene_id": job["scenes"][0], "catalog_pass": 1,
        "scene_index": 0, "round_index": job["round_index"],
        "status": "running", "durable_records": 1,
    }
    calls = []
    monkeypatch.setattr(
        background_collection, "job_progress",
        lambda value: calls.append(value["job_id"]) or {
            "durable_records": 3})
    monkeypatch.setattr(
        background_collection, "dataset_catalog_coverage",
        lambda _manifest, _state, dataset: {"dataset": dataset})

    value = run_background_collection._status_value(manifest, state)

    assert calls == [job["job_id"]]
    assert value["jobs"][job["job_id"]]["durable_records"] == 3
    assert value["dataset_cursors"] == state["dataset_cursors"]


def test_status_uses_lightweight_control_loaders(
        tmp_path, monkeypatch, capsys):
    manifest = {"output_root": str(tmp_path), "rounds": []}
    state = {
        "schema": "state", "status": "running", "current_round": 0,
        "jobs": {}, "datasets": {}, "compile_checkpoints": [],
        "updated_time_unix": 1.0,
    }
    monkeypatch.setattr(
        run_background_collection, "_load_control_manifest",
        lambda _path: manifest, raising=False)
    monkeypatch.setattr(
        run_background_collection, "_load_control_state",
        lambda _manifest: state, raising=False)
    monkeypatch.setattr(
        run_background_collection, "_load_manifest",
        lambda _path: pytest.fail("status performed full manifest validation"))
    monkeypatch.setattr(
        run_background_collection, "_load_state",
        lambda _manifest: pytest.fail("status performed full state validation"))
    monkeypatch.setattr(
        background_collection, "dataset_catalog_coverage",
        lambda _manifest, _state, dataset: {"dataset": dataset})

    result = run_background_collection._status(SimpleNamespace(
        manifest=tmp_path / "manifest.json"))

    assert result == 0
    assert json.loads(capsys.readouterr().out)["status"] == "stale"


def test_status_marks_a_dead_running_controller_stale(
        tmp_path, monkeypatch, capsys):
    controller = tmp_path / "controller"
    controller.mkdir()
    (controller / "controller.pid").write_text("123\n")
    manifest = {"output_root": str(tmp_path), "rounds": []}
    state = {
        "schema": "state", "status": "running", "current_round": 0,
        "jobs": {}, "datasets": {}, "compile_checkpoints": [],
        "updated_time_unix": 1.0,
    }
    monkeypatch.setattr(
        run_background_collection, "_load_control_manifest",
        lambda _path: manifest)
    monkeypatch.setattr(
        run_background_collection, "_load_control_state",
        lambda _manifest: state)
    monkeypatch.setattr(
        background_recovery, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        background_collection, "dataset_catalog_coverage",
        lambda _manifest, _state, dataset: {"dataset": dataset})

    assert run_background_collection._status(SimpleNamespace(
        manifest=tmp_path / "manifest.json")) == 0

    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "stale"
    assert value["state_status"] == "running"
    assert value["controller_pid"] == 123
    assert value["controller_alive"] is False


def test_stop_signals_only_the_controller(tmp_path, monkeypatch, capsys):
    controller = tmp_path / "controller"
    controller.mkdir()
    (controller / "controller.pid").write_text("123\n")
    manifest = {"output_root": str(tmp_path), "rounds": []}
    state = {
        "jobs": {"live": {
            "job_id": "live", "status": "running", "pid": 999,
        }},
    }
    monkeypatch.setattr(
        run_background_collection, "_load_control_manifest",
        lambda _path: manifest, raising=False)
    monkeypatch.setattr(
        run_background_collection, "_load_control_state",
        lambda _manifest: state, raising=False)
    monkeypatch.setattr(
        run_background_collection, "_load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        run_background_collection, "_load_state", lambda _manifest: state)
    monkeypatch.setattr(
        run_background_collection, "_write_state",
        lambda *_args: pytest.fail("stop wrote controller state"))
    monkeypatch.setattr(
        background_collection, "terminate_process_group",
        lambda *_args: pytest.fail("stop signaled a collector directly"))
    signals = []
    monkeypatch.setattr(
        run_background_collection.os, "kill",
        lambda pid, requested: signals.append((pid, requested)))

    result = run_background_collection._stop(SimpleNamespace(
        manifest=tmp_path / "manifest.json"))

    assert result == 0
    assert signals == [(123, signal.SIGINT)]
    assert state["jobs"]["live"]["status"] == "running"
    assert json.loads(capsys.readouterr().out)["status"] == \
        "interrupt_requested"


def test_controller_interrupt_reaps_children_then_writes_state(
        monkeypatch):
    state = {
        "status": "running",
        "jobs": {
            "live": {"status": "running", "pid": 10},
            "done": {"status": "completed", "pid": 11},
        },
    }
    live = SimpleNamespace(pid=10)
    events = []
    monkeypatch.setattr(
        run_background_collection, "_terminate_canary_process",
        lambda process: events.append(("reaped", process.pid)))
    monkeypatch.setattr(
        run_background_collection, "_write_state",
        lambda _manifest, value: events.append(
            ("written", value["status"], value["jobs"]["live"]["status"])))

    run_background_collection._interrupt_controller(
        {"output_root": "/tmp/run"}, state, {"live": live}, now=123.0)

    assert events == [
        ("reaped", 10), ("written", "interrupted", "interrupted")]
    assert state["jobs"]["live"]["finished_time_unix"] == 123.0
    assert state["jobs"]["done"]["status"] == "completed"


def test_capacity_terminal_name_requires_a_valid_scene_transaction():
    assert background_recovery.settled_runtime_status(
        returncode=-2, durable_records=8, catalog_status="failed",
        stopped_for_capacity=True) == "failed_partial"
    assert background_recovery.settled_runtime_status(
        returncode=1, durable_records=8, catalog_status="partial_valid",
        stopped_for_capacity=True) == "completed_capacity_shortfall"
    assert background_recovery.settled_runtime_status(
        returncode=0, durable_records=8, catalog_status="completed",
        stopped_for_capacity=False) == "completed"


def test_run_catches_keyboard_interrupt_and_removes_controller_pid(
        tmp_path, monkeypatch):
    controller = tmp_path / "controller"
    controller.mkdir()
    manifest = {"output_root": str(tmp_path), "rounds": []}
    state = {"status": "running", "jobs": {}}
    monkeypatch.setattr(
        run_background_collection, "_load_control_manifest",
        lambda _path: manifest)
    installed_handlers = []
    monkeypatch.setattr(
        run_background_collection.signal, "signal",
        lambda requested, handler:
            installed_handlers.append((requested, handler)))

    def interrupt(_args, control):
        control.update({
            "manifest": manifest, "state": state, "processes": {}})
        raise KeyboardInterrupt

    monkeypatch.setattr(
        run_background_collection, "_run_controller", interrupt,
        raising=False)
    handled = []
    monkeypatch.setattr(
        run_background_collection, "_interrupt_controller",
        lambda manifest_value, state_value, processes, now:
            handled.append((manifest_value, state_value, processes)))

    result = run_background_collection._run(SimpleNamespace(
        manifest=tmp_path / "manifest.json"))

    assert result == 130
    assert installed_handlers == [
        (signal.SIGINT, signal.default_int_handler),
        (signal.SIGTERM, signal.default_int_handler),
    ]
    assert handled == [(manifest, state, {})]
    assert not (controller / "controller.pid").exists()


def test_dead_running_job_resumes_without_advancing_its_cursor(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, rounds=0)
    job = background_collection.continuous_job(
        manifest, "gs", catalog_pass=0, scene_index=0)
    state = background_collection.initial_state(manifest)
    state["status"] = "running"
    state["jobs"][job["job_id"]] = {
        "job_id": job["job_id"], "dataset": "gs",
        "scene_id": job["scenes"][0], "gpu_id": job["gpu_id"],
        "round_index": job["round_index"], "catalog_pass": 0,
        "scene_index": 0, "pid": 999, "status": "running",
        "attempt": 1, "started_time_unix": 10.0,
        "finished_time_unix": None, "durable_records": 4,
    }
    original_cursor = dict(state["dataset_cursors"]["gs"])
    launched = []
    process = SimpleNamespace(pid=321)
    monkeypatch.setattr(
        background_recovery, "pid_alive", lambda _pid: False)

    terminal, changed = background_recovery.restore_running_processes(
        state, {job["job_id"]: job}, {}, now=20.0,
        launcher=lambda value, environment, log_path: (
            launched.append((value, environment, log_path)) or process))

    runtime = state["jobs"][job["job_id"]]
    assert terminal == []
    assert changed is True
    assert launched == [(job, job["environment"], job["log_path"])]
    assert state["dataset_cursors"]["gs"] == original_cursor
    assert runtime["pid"] == 321
    assert runtime["started_time_unix"] == 20.0
    assert runtime["durable_records"] == 4


def test_dead_running_job_adopts_valid_terminal_finalization(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, rounds=0)
    job = background_collection.continuous_job(
        manifest, "r2r", catalog_pass=0, scene_index=0)
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    (output / "collection_finalization.json").write_text(json.dumps({
        "status": "completed",
    }))
    state = background_collection.initial_state(manifest)
    state["status"] = "running"
    state["jobs"][job["job_id"]] = {
        "job_id": job["job_id"], "dataset": "r2r",
        "scene_id": job["scenes"][0], "gpu_id": job["gpu_id"],
        "round_index": job["round_index"], "catalog_pass": 0,
        "scene_index": 0, "pid": 999, "status": "running",
        "attempt": 1, "started_time_unix": 10.0,
        "finished_time_unix": None, "durable_records": 4,
    }
    monkeypatch.setattr(
        background_recovery, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        background_collection, "validate_scene_transaction",
        lambda *_args, **_kwargs: {
            "catalog_status": "completed",
            "terminal_status": "recovered_valid",
            "source_validated_records": 7,
            "length_shortfall": [],
            "record_paths": ["records.jsonl"],
            "sources": [{"path": "records.jsonl"}],
        })

    terminal, changed = background_recovery.restore_running_processes(
        state, {job["job_id"]: job}, {}, now=20.0,
        launcher=lambda *_args: pytest.fail(
            "a valid terminal transaction must not relaunch"))

    runtime = state["jobs"][job["job_id"]]
    assert terminal == [job]
    assert changed is True
    assert runtime["status"] == "recovered_valid"
    assert runtime["catalog_status"] == "completed"
    assert runtime["returncode"] == \
        background_collection.RECOVERED_UNKNOWN_RETURNCODE
    assert runtime["durable_records"] == 7


def test_process_poll_waits_lightly_until_exit_or_heartbeat(monkeypatch):
    sleeps = []
    monkeypatch.setattr(
        background_recovery.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        background_recovery.time, "time", lambda: 10.0)

    now, due = background_recovery.wait_for_process_event(
        {"live": SimpleNamespace(poll=lambda: None)},
        poll_s=1.0, heartbeat_deadline=30.0)
    assert (now, due, sleeps) == (10.0, False, [1.0])

    now, due = background_recovery.wait_for_process_event(
        {"done": SimpleNamespace(poll=lambda: 0)},
        poll_s=1.0, heartbeat_deadline=30.0)
    assert (now, due, sleeps) == (10.0, True, [1.0, 1.0])


def test_continuous_controller_launches_independent_dataset_cursors(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, rounds=0)
    state = background_collection.initial_state(manifest)
    state["status"] = "running"
    state["dataset_cursors"]["gs"] = {
        "catalog_pass": 1, "scene_index": 0}
    launched = []
    monkeypatch.setattr(
        run_background_collection, "_load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        run_background_collection, "_load_state", lambda _manifest: state)
    monkeypatch.setattr(
        run_background_collection, "_clean_revision",
        lambda: manifest["revision"])
    monkeypatch.setattr(
        background_collection, "validate_external_authorities",
        lambda _manifest: None)
    monkeypatch.setattr(
        background_collection, "validate_launch_resources",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        background_collection, "dataset_can_stop",
        lambda *_args, **_kwargs: True)

    def stop_after_launch(_manifest, _state, jobs, **_kwargs):
        launched.extend(jobs)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        run_background_collection.background_scheduler,
        "launch_jobs", stop_after_launch)
    monkeypatch.setattr(
        run_background_collection, "_record_compile_result",
        lambda *_args, **_kwargs: pytest.fail(
            "continuous collection must not stop on capacity"))

    with pytest.raises(KeyboardInterrupt):
        run_background_collection._run_controller(
            SimpleNamespace(
                manifest=tmp_path / "manifest.json", poll_interval_s=0.01),
            {"processes": {}})

    assert {(job["dataset"], job["catalog_pass"])
            for job in launched} == {
        ("gs", 1), ("b1k", 0), ("r2r", 0)}


def test_record_compile_result_trusts_the_just_published_identity(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    state = background_collection.initial_state(manifest)
    compiled = {
        "artifact": "supply.json",
        "coverage": {},
        "gt_as_pred": None,
        "six_task_macro": None,
        "datasets": {
            dataset: {"quota": {"complete": False}}
            for dataset in background_collection.DATASETS
        },
        "checkpoint_summary": {
            "path": "checkpoint.json",
            "sha256": "1" * 64,
            "content_sha256": "2" * 64,
        },
    }
    monkeypatch.setattr(
        background_collection.background_checkpoint, "load",
        lambda *_args, **_kwargs: pytest.fail(
            "a successful compile must not reopen its artifact"))

    assert run_background_collection._record_compile_result(
        manifest, state, round_index=0, checkpoint="catalog-pass-00",
        compile_fn=lambda *_args, **_kwargs: compiled) is True
    assert state["compile_checkpoints"] == ["catalog-pass-00"]


def test_scene_transaction_reuses_the_collectors_validated_partial_seal(
        tmp_path):
    output = tmp_path / "scene"
    output.mkdir()
    records = output / "records.jsonl"
    records.write_text('{}\n')
    metadata = output / "run_meta.json"
    contract_params = {
        "backend": "r2r", "collection_mode": "main",
        "scenes": ["scene-a"], "collection_shard_id": "job-a",
        "r2r_train_episodes": "/data/r2r/train.json.gz",
        "pose_exclusions": None,
        **_capacity_run_params(),
    }
    run_contract_sha256 = _write_spool_run_contract(
        records, params=contract_params, scene_id="scene-a")
    run_meta = {
        "params": {
            **contract_params,
            "out": str(output), "code_revision": "a" * 40,
        },
        "code_revision": "a" * 40, "code_dirty": False,
        "source_catalog": {
            "datasets": ["r2r"], "scene_ids": ["scene-a"],
            "manifest_sha256": ["4" * 64]},
        "resolved_scenes": [{"scene_id": "scene-a"}],
        "run_contract_sha256": run_contract_sha256,
    }
    metadata.write_text(json.dumps(run_meta))
    (output / "collection_funnel.json").write_text('{}\n')
    _write_finalization(
        output, status="partial",
        run_contract_sha256=run_contract_sha256)

    result = background_collection.validate_scene_transaction({
        "job_id": "job-a", "dataset": "r2r",
        "scenes": ["scene-a"],
        "output_dir": str(output),
        "transaction_binding": {
            "dataset": "r2r", "scene_id": "scene-a",
            "collection_shard_id": "job-a", "output_dir": str(output),
            "revision": "a" * 40, "backend": "r2r",
            **_capacity_binding_fields(),
            "source_authority": {
                "path": "/data/r2r/train.json.gz", "sha256": "4" * 64},
        },
    }, returncode=2)

    assert result["catalog_status"] == "partial_valid"
    assert result["source_validated_records"] == 1
    assert result["length_shortfall"] == []


def test_scene_transaction_uses_the_authenticated_spool_run_contract(
        tmp_path):
    output = tmp_path / "scene"
    output.mkdir()
    records = output / "records.jsonl"
    records.write_text('{}\n')
    exclusions = tmp_path / "pose-exclusions.json"
    exclusions.write_text('{"poses": []}\n')
    exclusion_sha256 = hashlib.sha256(exclusions.read_bytes()).hexdigest()
    contract_params = {
        "backend": "r2r",
        "collection_mode": "main",
        "scenes": ["scene-a"],
        "collection_shard_id": "job-a",
        "r2r_train_episodes": "/data/r2r/train.json.gz",
        "pose_exclusions": str(exclusions),
        "pose_exclusions_sha256": exclusion_sha256,
        **_capacity_run_params(),
    }
    run_contract_sha256 = _write_spool_run_contract(
        records, params=contract_params, scene_id="scene-a")
    run_meta = {
        # This forensic argv view intentionally lacks the derived SHA.  The
        # authenticated spool contract is the scientific run authority.
        "params": {
            **contract_params,
            "out": str(output),
            "code_revision": "a" * 40,
        },
        "code_revision": "a" * 40,
        "code_dirty": False,
        "source_catalog": {
            "datasets": ["r2r"],
            "scene_ids": ["scene-a"],
            "manifest_sha256": ["4" * 64],
        },
        "resolved_scenes": [{"scene_id": "scene-a"}],
        "run_contract_sha256": run_contract_sha256,
        "formal_action_length_coverage": {
            "complete": True,
            "shortfall": {},
        },
    }
    run_meta["params"].pop("pose_exclusions_sha256")
    (output / "run_meta.json").write_text(json.dumps(run_meta))
    (output / "collection_funnel.json").write_text('{}\n')
    _write_finalization(
        output, status="completed",
        run_contract_sha256=run_contract_sha256)
    job = {
        "job_id": "job-a",
        "dataset": "r2r",
        "scenes": ["scene-a"],
        "output_dir": str(output),
        "transaction_binding": {
            "dataset": "r2r",
            "scene_id": "scene-a",
            "collection_shard_id": "job-a",
            "output_dir": str(output),
            "revision": "a" * 40,
            "backend": "r2r",
            "pose_exclusions_path": str(exclusions),
            **_capacity_binding_fields(),
            "source_authority": {
                "path": "/data/r2r/train.json.gz",
                "sha256": "4" * 64,
            },
        },
    }

    result = background_collection.validate_scene_transaction(
        job, returncode=0)

    assert result["catalog_status"] == "completed"
    assert result["source_validated_records"] == 1


def test_scene_transaction_distinguishes_zero_yield_failure_and_bad_source(
        tmp_path):
    output = tmp_path / "scene"
    output.mkdir()
    job = {
        "job_id": "job-a", "dataset": "r2r", "scenes": ["scene-a"],
        "output_dir": str(output),
        "transaction_binding": {
            "dataset": "r2r", "scene_id": "scene-a",
            "collection_shard_id": "job-a", "output_dir": str(output),
            "revision": "a" * 40, "backend": "r2r",
            **_capacity_binding_fields(),
            "source_authority": {
                "path": "/data/r2r/train.json.gz", "sha256": "4" * 64},
        },
    }
    assert background_collection.validate_scene_transaction(
        job, returncode=0)["catalog_status"] == "zero_yield"
    assert background_collection.validate_scene_transaction(
        job, returncode=3)["catalog_status"] == "failed"

    records = output / "records.jsonl"
    records.write_text("")
    assert background_collection.validate_scene_transaction(
        job, returncode=130,
        stopped_for_capacity=True)["catalog_status"] == "zero_yield"

    records.write_text('{}\n')
    (output / "run_meta.json").write_text('{}\n')
    contract_params = {
        "backend": "r2r", "collection_mode": "main",
        "scenes": ["scene-a"], "collection_shard_id": "job-a",
        "r2r_train_episodes": "/data/r2r/train.json.gz",
        "pose_exclusions": None,
        **_capacity_run_params(),
    }
    run_contract_sha256 = _write_spool_run_contract(
        records, params=contract_params, scene_id="scene-a")
    run_meta = {
        "params": {
            **contract_params,
            "out": str(output), "code_revision": "a" * 40,
        },
        "code_revision": "a" * 40, "code_dirty": False,
        "source_catalog": {
            "datasets": ["r2r"], "scene_ids": ["scene-a"],
            "manifest_sha256": ["4" * 64]},
        "resolved_scenes": [{"scene_id": "scene-a"}],
        "run_contract_sha256": run_contract_sha256,
        "formal_action_length_coverage": {"complete": True, "shortfall": {}},
    }
    (output / "run_meta.json").write_text(json.dumps(run_meta))
    (output / "collection_funnel.json").write_text('{}\n')
    _write_finalization(
        output, status="completed",
        run_contract_sha256=run_contract_sha256)
    recovered = background_collection.validate_scene_transaction(
        job, returncode=background_collection.RECOVERED_UNKNOWN_RETURNCODE)
    assert recovered["catalog_status"] == "completed"
    assert recovered["terminal_status"] == "recovered_valid"
    _write_finalization(
        output, status="completed",
        run_contract_sha256=run_contract_sha256,
        source_validation="failed")
    with pytest.raises(ValueError, match="passed source validation"):
        background_collection.validate_scene_transaction(job, returncode=0)


def test_scene_transaction_rejects_substituted_scene_and_rehashed_command(
        tmp_path):
    manifest = _manifest(tmp_path)
    job = next(job for row in manifest["rounds"] for job in row["jobs"]
               if job["dataset"] == "r2r")
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    records.write_text('{}\n')
    binding = job["transaction_binding"]
    contract_params = {
        "backend": "r2r", "collection_mode": "main",
        "scenes": ["scene-b"],
        "collection_shard_id": binding["collection_shard_id"],
        "poses_per_scene": binding["poses_per_scene"],
        "pose_candidates_per_scene": binding["pose_candidates_per_scene"],
        "record_idle_stop_s": binding["record_idle_stop_s"],
        "scene_wallclock_stop_s": float(binding["scene_wallclock_stop_s"]),
        "seed": binding["seed"],
        "benchmark_partition": binding["benchmark_partition"],
        "ordinary_actions_per_pose": binding["ordinary_actions_per_pose"],
        "pose_exclusions": binding["pose_exclusions_path"],
        "r2r_train_episodes": binding["source_authority"]["path"],
    }
    run_contract_sha256 = _write_spool_run_contract(
        records, params=contract_params, scene_id="scene-b",
        revision=manifest["revision"])
    substituted = {
        "params": {
            "backend": "r2r", "collection_mode": "main",
            "scenes": ["scene-b"], "collection_shard_id": job["job_id"],
            "out": str(output), "code_revision": manifest["revision"],
        },
        "code_revision": manifest["revision"], "code_dirty": False,
        "source_catalog": {"datasets": ["r2r"], "scene_ids": ["scene-b"]},
        "resolved_scenes": [{"scene_id": "scene-b"}],
        "run_contract_sha256": run_contract_sha256,
        "formal_action_length_coverage": {"complete": True, "shortfall": {}},
    }
    (output / "run_meta.json").write_text(json.dumps(substituted))
    (output / "collection_funnel.json").write_text('{}\n')
    _write_finalization(
        output, status="completed",
        run_contract_sha256=run_contract_sha256)
    with pytest.raises(ValueError, match="transaction binding") as error:
        background_collection.validate_scene_transaction(job, returncode=0)
    assert "resolved_scene_id" in str(error.value)
    assert "scene-b" in str(error.value)
    assert binding["scene_id"] in str(error.value)

    tampered = copy.deepcopy(manifest)
    target = next(job for row in tampered["rounds"] for job in row["jobs"]
                  if job["dataset"] == "r2r")
    target["command"][target["command"].index("--scenes") + 1] = "scene-b"
    body = {key: value for key, value in tampered.items() if key != "sha256"}
    tampered["sha256"] = background_collection._canonical_sha256(body)
    with pytest.raises(ValueError, match="job command"):
        background_collection.validate_manifest(tampered)


def test_recovery_sidecar_changes_only_reauthenticated_jobs(monkeypatch):
    manifest = {
        "rounds": [{"jobs": [
            {"job_id": "recover", "dataset": "r2r"},
            {"job_id": "orphan", "dataset": "b1k"},
        ]}],
    }
    state = {
        "schema": "state", "status": "running", "current_round": 4,
        "compile_checkpoints": ["catalog-pass03"],
        "jobs": {
            "recover": {
                "status": "completed", "catalog_status": "failed",
                "returncode": 0, "durable_records": 2,
                "source_validation_error":
                    "scene transaction binding differs from run metadata",
            },
            "orphan": {
                "status": "completed_capacity_shortfall",
                "catalog_status": "failed", "returncode": -2,
                "durable_records": 1,
                "source_validation_error":
                    "collection finalization is unreadable",
            },
        },
    }
    monkeypatch.setattr(
        background_collection, "validate_scene_transaction",
        lambda *_args, **_kwargs: {
            "catalog_status": "completed",
            "terminal_status": None,
            "source_validated_records": 2,
            "length_shortfall": [],
            "record_paths": ["records.jsonl"],
            "sources": [{"path": "records.jsonl"}],
        })

    sidecar, report = run_background_collection.reconcile_recovery_state(
        manifest, state)

    assert state["jobs"]["recover"]["catalog_status"] == "failed"
    assert sidecar["status"] == state["status"]
    assert sidecar["current_round"] == state["current_round"]
    assert sidecar["compile_checkpoints"] == state["compile_checkpoints"]
    assert sidecar["jobs"]["recover"]["status"] == "completed"
    assert sidecar["jobs"]["recover"]["catalog_status"] == "completed"
    assert "source_validation_error" not in sidecar["jobs"]["recover"]
    assert sidecar["jobs"]["orphan"] == state["jobs"]["orphan"]
    assert report == {
        "attempted_shards": 1,
        "recovered_shards": {"r2r": 1, "gs": 0, "b1k": 0},
        "recovered_records": {"r2r": 2, "gs": 0, "b1k": 0},
        "rejected": [],
    }


def test_recover_command_writes_a_sidecar_and_compiles_it(
        tmp_path, monkeypatch):
    output_root = tmp_path / "run"
    manifest = {"output_root": str(output_root), "sha256": "1" * 64}
    original_state = {"jobs": {"old": {"catalog_status": "failed"}}}
    sidecar = {"jobs": {"old": {"catalog_status": "completed"}}}
    report = {
        "attempted_shards": 1,
        "recovered_shards": {"r2r": 1, "gs": 0, "b1k": 0},
        "recovered_records": {"r2r": 2, "gs": 0, "b1k": 0},
        "rejected": [],
    }
    compiled_states = []
    monkeypatch.setattr(
        run_background_collection, "_load_manifest",
        lambda _path: pytest.fail(
            "recovery must not rederive a historical run manifest"))
    monkeypatch.setattr(
        run_background_collection, "_load_control_manifest",
        lambda _path: manifest)
    monkeypatch.setattr(
        run_background_collection, "_load_control_state",
        lambda _manifest: original_state)
    monkeypatch.setattr(
        run_background_collection, "reconcile_recovery_state",
        lambda _manifest, _state: (sidecar, report))
    monkeypatch.setattr(
        background_collection, "compile_global",
        lambda _manifest, checkpoint, *, state: (
            compiled_states.append((checkpoint, state)) or {
                "checkpoint_summary": {"sha256": "2" * 64}}))

    assert run_background_collection._recover(SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        checkpoint="recovery-562")) == 0

    controller = output_root / "controller"
    assert json.loads((controller / "recovery-562-state.json").read_text()) \
        == sidecar
    assert json.loads((controller / "recovery-562-report.json").read_text()) \
        == report
    assert compiled_states == [("recovery-562", sidecar)]
    assert original_state["jobs"]["old"]["catalog_status"] == "failed"


@pytest.mark.parametrize("option,value", [
    ("--scenes", "substituted-scene"),
    ("--poses-per-scene", "999999"),
    ("--pose-candidates-per-scene", "999999"),
    ("--seed", "0"),
    ("--radii", "0.99"),
    ("--camera-heights", "9.9"),
    ("--lengths", "9"),
    ("--proposal-pairs-per-length", "999"),
    ("--proposal-natural-per-length", "999"),
    ("--keep-per-length", "999"),
    ("--action-mode", "natural"),
])
def test_manifest_rejects_appended_parser_duplicate_options(
        tmp_path, option, value):
    manifest = _manifest(tmp_path)
    target = next(job for row in manifest["rounds"] for job in row["jobs"]
                  if job["dataset"] == "r2r")
    target["command"].extend([option, value])
    body = {key: item for key, item in manifest.items() if key != "sha256"}
    manifest["sha256"] = background_collection._canonical_sha256(body)

    with pytest.raises(ValueError, match="duplicate.*option"):
        background_collection.validate_manifest(manifest)


def test_manifest_rejects_appended_unknown_or_stray_command_tokens(tmp_path):
    for suffix in (("--overwrite",), ("stray-nargs-value",)):
        manifest = _manifest(tmp_path / suffix[0].replace("-", "x"))
        target = next(job for row in manifest["rounds"] for job in row["jobs"]
                      if job["dataset"] == "r2r")
        target["command"].extend(suffix)
        body = {key: item for key, item in manifest.items() if key != "sha256"}
        manifest["sha256"] = background_collection._canonical_sha256(body)
        with pytest.raises(ValueError, match="exact contract"):
            background_collection.validate_manifest(manifest)


def test_capacity_profile_identity_and_catalog_membership_are_required(tmp_path):
    raw = _raw_profile(measurement_root=tmp_path / "measurements")
    profile = background_collection.derive_capacity_profile(raw)
    Path(raw["datasets"]["r2r"]["canary_scenes"][0][
        "records_path"]).write_text("{}\n")
    with pytest.raises(ValueError, match="capacity profile"):
        background_collection.build_manifest(
            revision="a" * 40, output_root=tmp_path / "digest-run",
            r2r_scenes=[f"r2r-{i}" for i in range(8)],
            gs_scenes=[f"gs-{i}" for i in range(8)],
            b1k_scenes=[f"b1k-{i}" for i in range(12)], paths=_paths(tmp_path),
            b1k_catalog_audit=_audit([f"b1k-{i}" for i in range(12)]),
            b1k_catalog_audit_sha256="2" * 64,
            r2r_source_manifest_sha256="4" * 64,
            gs_source_manifest_sha256="5" * 64,
            b1k_source_manifest_sha256="3" * 64,
            b1k_source_scene_ids=[f"b1k-{i}" for i in range(12)],
            capacity_profile=profile,
            capacity_profile_sha256=profile["sha256"])

    raw = _raw_profile(measurement_root=tmp_path / "membership")
    del raw["datasets"]["r2r"]["canary_scenes"][0]["records_path"]
    with pytest.raises(ValueError, match="source paths"):
        background_collection.derive_capacity_profile(raw)

    raw = _raw_profile(measurement_root=tmp_path / "outside")
    profile = background_collection.derive_capacity_profile(raw)
    with pytest.raises(ValueError, match="canary.*catalog"):
        background_collection.build_manifest(
            revision="a" * 40, output_root=tmp_path / "run",
            r2r_scenes=[f"r2r-{i}" for i in range(2, 10)],
            gs_scenes=[f"gs-{i}" for i in range(8)],
            b1k_scenes=[f"b1k-{i}" for i in range(12)], paths=_paths(tmp_path),
            b1k_catalog_audit=_audit([f"b1k-{i}" for i in range(12)]),
            b1k_catalog_audit_sha256="2" * 64,
            r2r_source_manifest_sha256="4" * 64,
            gs_source_manifest_sha256="5" * 64,
            b1k_source_manifest_sha256="3" * 64,
            b1k_source_scene_ids=[f"b1k-{i}" for i in range(12)],
            capacity_profile=profile,
            capacity_profile_sha256=profile["sha256"])


def test_external_authorities_are_reopened_and_exactly_revalidated(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    audit_path = tmp_path / "audit.json"
    source_path = tmp_path / "source.json"
    audit = _audit(manifest["scene_catalog"]["b1k"])
    source = {"scenes": [{"scene_id": value}
                         for value in manifest["scene_catalog"]["b1k"]]}
    audit_path.write_text(json.dumps(audit))
    source_path.write_text(json.dumps(source))
    manifest["paths"]["b1k_catalog_audit"] = str(audit_path)
    manifest["paths"]["b1k_source_manifest"] = str(source_path)
    manifest["authorities"]["b1k_catalog_audit"]["sha256"] = \
        hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest["authorities"]["b1k_source_manifest"]["sha256"] = \
        hashlib.sha256(source_path.read_bytes()).hexdigest()
    for round_value in manifest["rounds"]:
        for job in round_value["jobs"]:
            if job["dataset"] != "b1k":
                continue
            binding = job["transaction_binding"]["source_authority"]
            binding["path"] = str(source_path)
            binding["sha256"] = manifest["authorities"][
                "b1k_source_manifest"]["sha256"]
            command = job["command"]
            command[command.index("--source-manifest") + 1] = str(source_path)
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = background_collection._canonical_sha256(body)

    background_collection.validate_external_authorities(manifest)
    audit["selection_scope"] = "substituted"
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="audit file digest"):
        background_collection.validate_external_authorities(manifest)
    audit["selection_scope"] = "b1k-catalog-authority-audit.v1"
    audit_path.write_text(json.dumps(audit))
    source["scenes"] = source["scenes"][:-1]
    source_path.write_text(json.dumps(source))
    manifest["authorities"]["b1k_source_manifest"]["sha256"] = \
        hashlib.sha256(source_path.read_bytes()).hexdigest()
    for round_value in manifest["rounds"]:
        for job in round_value["jobs"]:
            if job["dataset"] == "b1k":
                job["transaction_binding"]["source_authority"]["sha256"] = \
                    manifest["authorities"]["b1k_source_manifest"]["sha256"]
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = background_collection._canonical_sha256(body)
    with pytest.raises(ValueError, match="accepted.*source"):
        background_collection.validate_external_authorities(manifest)


def test_state_validation_rejects_mismatched_jobs_and_fake_recovery_success(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    state = background_collection.initial_state(manifest)
    state["status"] = "running"
    job = manifest["rounds"][0]["jobs"][0]
    state["jobs"][job["job_id"]] = {
        "job_id": job["job_id"], "dataset": "r2r",
        "scene_id": job["scenes"][0], "round_index": 0,
        "catalog_status": "failed", "source_validation": {"sources": []},
        "status": "failed_zero_yield", "pid": 1,
    }
    with pytest.raises(ValueError, match="state job"):
        background_collection.validate_state(manifest, state)
    runtime = state["jobs"][job["job_id"]]
    runtime["dataset"] = job["dataset"]
    runtime["catalog_status"] = "completed"
    with pytest.raises(ValueError, match="reusable state lacks sources"):
        background_collection.validate_state(manifest, state)
    runtime["catalog_status"] = "failed"
    runtime["source_validation"] = {"sources": [{
        "path": str(Path(job["output_dir"]) / "records.jsonl"),
        "records_sha256": "1" * 64, "run_meta_sha256": "2" * 64,
    }]}
    with pytest.raises(ValueError, match="nonreusable state claims sources"):
        background_collection.validate_state(manifest, state)

    monkeypatch.setattr(background_recovery.os, "kill",
                        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    assert background_recovery.RecoveredProcess(123).poll() != 0


def test_wallclock_escalation_is_durable_and_reaches_sigkill():
    runtime = {"status": "running", "started_time_unix": 0.0}
    assert background_collection.wallclock_escalation(
        runtime, now=100.0, scene_wallclock_s=100.0) == signal.SIGINT
    assert runtime["status"] == "stopping_sigint"
    assert background_collection.wallclock_escalation(
        runtime, now=100.0 + background_collection.config.
        BACKGROUND_SIGINT_GRACE_S, scene_wallclock_s=100.0) == signal.SIGTERM
    assert runtime["status"] == "stopping_sigterm"
    assert background_collection.wallclock_escalation(
        runtime, now=100.0 + background_collection.config.
        BACKGROUND_SIGINT_GRACE_S + background_collection.config.
        BACKGROUND_SIGTERM_GRACE_S, scene_wallclock_s=100.0) == signal.SIGKILL
    assert runtime["status"] == "stopping_sigkill"


def test_controller_emits_digest_bound_capacity_timing_events(tmp_path):
    manifest = _manifest(tmp_path)
    state = background_collection.initial_state(manifest)
    job = manifest["rounds"][0]["jobs"][0]
    background_collection.start_round(
        manifest, state, round_index=0,
        launcher=lambda *_args: SimpleNamespace(pid=123), now=10.0)
    background_collection.record_capacity_event(
        manifest, job, "backend_ready", time_unix=20.0)
    background_collection.record_capacity_event(
        manifest, job, "controller_scene_finished", time_unix=30.0)

    output = Path(job["output_dir"])
    controller = json.loads(
        (output / "capacity_controller_manifest.json").read_text())
    events = [json.loads(line) for line in
              (output / "controller_events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events] == [
        "controller_scene_started", "backend_ready",
        "controller_scene_finished"]
    assert {row["controller_manifest_sha256"] for row in events} == {
        controller["sha256"]}
    assert controller["parent_manifest_sha256"] == manifest["sha256"]


def test_compile_checkpoint_failure_is_retryable_without_relaunch(tmp_path):
    manifest = _manifest(tmp_path)
    state = background_collection.initial_state(manifest)
    round_index = 0
    for job in manifest["rounds"][0]["jobs"]:
        output = Path(job["output_dir"])
        output.mkdir(parents=True)
        records = output / "records.jsonl"
        metadata = output / "run_meta.json"
        records.write_text("{}\n")
        metadata.write_text("{}\n")
        state["jobs"][job["job_id"]] = {
            "job_id": job["job_id"], "dataset": job["dataset"],
            "scene_id": job["scenes"][0], "round_index": 0,
            "catalog_status": "completed", "status": "completed",
            "source_validation": {"sources": [{
                "path": str(records),
                "records_sha256": hashlib.sha256(
                    records.read_bytes()).hexdigest(),
                "run_meta_sha256": hashlib.sha256(
                    metadata.read_bytes()).hexdigest(),
            }]},
        }
    checkpoint = "catalog-pass-00"
    assert run_background_collection._record_compile_result(
        manifest, state, round_index=round_index, checkpoint=checkpoint,
        compile_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("compile boom"))) is False
    assert state["current_round"] == round_index
    assert state["status"] == "compile_failed"
    launches = []
    assert background_collection.start_round(
        manifest, state, round_index=round_index,
        launcher=lambda *args: launches.append(args), now=3.0) == {}
    assert launches == []
    retried = []

    def compile_success(manifest_value, checkpoint_value, *, state):
        retried.append(checkpoint_value)
        final, staging = background_collection.background_checkpoint.prepare(
            manifest_value, checkpoint_value)
        (staging / "records").mkdir()
        (staging / "records" / "manifest.json").write_text("{}\n")
        (staging / "supply.json").write_text("{}\n")
        compiled = {
            "artifact": str(final / "supply.json"),
            "coverage": {}, "gt_as_pred": None,
            "six_task_macro": None,
            "datasets": {
                dataset: {"quota": {"complete": False}}
                for dataset in background_collection.DATASETS},
        }
        identity = background_collection.background_checkpoint.publish(
            manifest_value, checkpoint_value, staging, result=compiled,
            result_files=["supply.json", "records/manifest.json"],
            sources=next(iter(state["jobs"].values()))[
                "source_validation"]["sources"])
        return {**compiled, "checkpoint_summary": identity}

    assert run_background_collection._record_compile_result(
        manifest, state, round_index=round_index, checkpoint=checkpoint,
        compile_fn=compile_success) is True
    assert retried == [checkpoint]
    assert state["compile_checkpoints"] == [checkpoint]
    background_collection.validate_state(manifest, state)
    supply_path = Path(manifest["output_root"]) / \
        f"artifacts/global/{checkpoint}/supply.json"
    supply_path.write_text('{"changed":true}\n')
    assert background_collection.dataset_can_stop(
        manifest, state, "r2r") is False
    with pytest.raises(ValueError, match="checkpoint file changed"):
        background_collection.validate_state(manifest, state)


def test_compile_global_recreates_only_staging_after_partial_failure(
        tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    state = background_collection.initial_state(manifest)
    job = next(job for row in manifest["rounds"] for job in row["jobs"]
               if job["dataset"] == "r2r")
    output = Path(job["output_dir"])
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    records.write_text('{}\n')
    run_meta = output / "run_meta.json"
    run_meta.write_text(json.dumps({
        "dataset": "r2r", "record_count": 1}) + "\n")
    state["jobs"][job["job_id"]] = {
        "catalog_status": "completed",
        "source_validation": {"sources": [{
            "path": str(records),
            "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
            "run_meta_sha256": hashlib.sha256(run_meta.read_bytes()).hexdigest(),
        }]},
    }
    calls = []

    def enumerate_candidates(records_root, **_kwargs):
        calls.append(records_root)
        if len(calls) == 1:
            (records_root.parent / "partial").write_text("incomplete")
            raise ValueError("interrupted compile")
        return []

    monkeypatch.setattr(
        background_collection.seen_selection, "enumerate_candidates",
        enumerate_candidates)

    with pytest.raises(ValueError, match="interrupted compile"):
        background_collection.compile_global(
            manifest, "catalog-pass-00", state=state)
    root = Path(manifest["output_root"]) / "artifacts/global"
    assert not (root / "catalog-pass-00").exists()
    assert (root / ".catalog-pass-00.staging/partial").is_file()

    result = background_collection.compile_global(
        manifest, "catalog-pass-00", state=state)

    assert len(calls) == 2
    assert not (root / ".catalog-pass-00.staging").exists()
    assert result["artifact"] == str(
        root / "catalog-pass-00/supply.json")
    assert Path(result["checkpoint_summary"]["path"]).is_file()


def test_bare_quota_complete_has_no_authenticated_stop_provenance(tmp_path):
    manifest = _manifest(tmp_path)
    state = background_collection.initial_state(manifest)
    state["datasets"]["r2r"] = {"quota": {"complete": True}}

    assert background_collection.dataset_can_stop(
        manifest, state, "r2r") is False
    with pytest.raises(ValueError, match="dataset.*provenance"):
        background_collection.validate_state(manifest, state)


def test_capacity_profile_cli_derives_and_writes_digest(tmp_path):
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps(_raw_profile(
        measurement_root=tmp_path / "cli-evidence"), sort_keys=True))
    output = tmp_path / "capacity.json"

    assert run_background_collection.main([
        "profile", "--measurements", str(measurements),
        "--out", str(output),
    ]) == 0

    profile = json.loads(output.read_text())
    background_collection.validate_capacity_profile(profile)


def test_capacity_evidence_rejects_declared_numeric_measurements(tmp_path):
    evidence = _raw_profile(measurement_root=tmp_path / "declared")
    canary = evidence["datasets"]["r2r"]["canary_scenes"][0]
    Path(canary["records_path"]).write_text("{}\n")
    canary["accepted_records"] = 10
    canary["task_qa_counts"] = {task: 60 for task in TASKS}

    with pytest.raises(ValueError, match="only source paths"):
        background_collection.derive_capacity_profile(evidence)


def test_capacity_profile_does_not_repeat_published_source_validation(
        tmp_path, monkeypatch):
    evidence = _raw_profile(measurement_root=tmp_path / "source-published")
    monkeypatch.setattr(
        record_validation, "validate_file_source_bound",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("published records were source-validated again")))

    profile = background_collection.derive_capacity_profile(evidence)

    assert profile["schema"] == background_capacity.PROFILE_SCHEMA


def test_global_record_discovery_reuses_only_source_valid_terminals(tmp_path):
    manifest = _manifest(tmp_path)
    state = background_collection.initial_state(manifest)
    jobs = [
        job for round_value in manifest["rounds"]
        for job in round_value["jobs"] if job["dataset"] == "r2r"
    ][:3]
    paths = []
    for job, status in zip(jobs, ("completed", "partial_valid", "failed")):
        output = Path(job["output_dir"])
        output.mkdir(parents=True)
        records = output / "records.jsonl"
        records.write_text('{}\n')
        paths.append(records)
        state["jobs"][job["job_id"]] = {
            "catalog_status": status,
            "source_validation": {
                "sources": [{
                    "path": str(records),
                    "records_sha256": "1" * 64,
                    "run_meta_sha256": "2" * 64,
                }],
            },
        }

    assert background_collection.discover_dataset_records(
        manifest, "r2r", state=state) == paths[:2]


def test_continuous_record_discovery_includes_derived_passes(tmp_path):
    manifest = _manifest(tmp_path, rounds=0)
    state = background_collection.initial_state(manifest)
    jobs = [
        background_collection.continuous_job(
            manifest, "gs", catalog_pass=catalog_pass, scene_index=0)
        for catalog_pass in (0, 1)]
    expected = []
    for job in jobs:
        output = Path(job["output_dir"])
        output.mkdir(parents=True)
        records = output / "records.jsonl"
        records.write_text('{}\n')
        expected.append(records)
        state["jobs"][job["job_id"]] = {
            "dataset": "gs", "scene_id": job["scenes"][0],
            "catalog_pass": job["catalog_pass"],
            "scene_index": job["scene_index"],
            "catalog_status": "completed",
            "source_validation": {"sources": [{
                "path": str(records),
                "records_sha256": "1" * 64,
                "run_meta_sha256": "2" * 64,
            }]},
        }

    assert background_collection.discover_dataset_records(
        manifest, "gs", state=state) == expected


def test_global_record_discovery_spans_all_datasets_in_stable_order(tmp_path):
    manifest = _manifest(tmp_path)
    state = background_collection.initial_state(manifest)
    expected = []
    for dataset in ("r2r", "gs", "b1k"):
        job = next(
            job for round_value in manifest["rounds"]
            for job in round_value["jobs"] if job["dataset"] == dataset)
        output = Path(job["output_dir"])
        output.mkdir(parents=True)
        records = output / "records.jsonl"
        records.write_text('{}\n')
        expected.append(records)
        state["jobs"][job["job_id"]] = {
            "catalog_status": "completed",
            "source_validation": {
                "sources": [{
                    "path": str(records),
                    "records_sha256": "1" * 64,
                    "run_meta_sha256": "2" * 64,
                }],
            },
        }

    assert background_collection.discover_global_records(
        manifest, state=state) == expected
