"""Simulator-free orchestration for quota-driven three-dataset collection.

The controller operates only between shards.  It never reads an oracle label
to alter a pose-local action bank: compiled QA shortfall can request another
independent shard, and nothing more.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
from typing import Callable, Mapping, Sequence

from pipeline import (
    a1_common_support, action_proposal, background_authorities,
    background_canary, background_capacity, background_checkpoint,
    background_job_contract, background_scheduler,
    benchmark, candidate_preview, candidate_quota, collection_closeout,
    collection_cli, config, dataset_contracts, gate_authority, io_utils,
    scene_partitions,
)
from pipeline.background_lifecycle import (
    b1k_supervisor_timeout_s, capacity_watchdog_action,
    controller_watchdog_action, health_requires_termination,
    wallclock_escalation,
)


CONTROLLER_SCHEMA = "egoconseq.three-dataset-background-controller.v3"
STATE_SCHEMA = "egoconseq.three-dataset-background-state.v2"
CAPACITY_PROFILE_SCHEMA = background_capacity.PROFILE_SCHEMA
SUPPORTED_TASKS = tuple(benchmark.ABC_CANDIDATE_TASK_IDS)
DATASETS = dataset_contracts.main_collection_datasets()
RECOVERED_UNKNOWN_RETURNCODE = 255
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def derive_capacity_profile(value: Mapping) -> dict:
    """Derive launch budgets from exactly two authenticated canaries."""
    return background_capacity.build_profile(value)


def validate_capacity_profile(value: Mapping) -> None:
    """Reopen evidence inputs and require the exact derived profile."""
    background_capacity.validate_profile(value)


six_task_macro_report = candidate_quota.six_task_macro_report
dataset_macro_reports = candidate_quota.dataset_macro_reports


def _dataset_quota_rows() -> dict:
    return candidate_quota.collection_quota_rows()


def _unique(values: Sequence[str], *, label: str) -> list[str]:
    result = [str(value).strip() for value in values]
    if not result or any(not value for value in result) or \
            len(result) != len(set(result)):
        raise ValueError(f"{label} scenes must be nonempty and unique")
    return result


def _required_paths(paths: Mapping[str, str]) -> dict[str, str]:
    names = (
        "repository", "python", "r2r_train_episodes", "mp3d_root",
        "gs_data_root", "gs_source_manifest", "b1k_python",
        "b1k_data_root", "b1k_source_manifest", "b1k_catalog_audit",
        "capacity_profile",
    )
    result = {name: str(paths.get(name) or "").strip() for name in names}
    missing = [name for name, value in result.items() if not value]
    if missing:
        raise ValueError(
            "background collection paths are missing: " + ", ".join(missing))
    return result


def _common_collect_args(
        *, scenes: Sequence[str], output_dir: Path, revision: str,
        shard_id: str, poses_per_scene: int,
        pose_candidates_per_scene: int, seed: int,
        scene_wallclock_s: int,
        ordinary_actions_per_pose: int,
        pose_exclusions: Path = None) -> list[str]:
    values = [
        "--scenes", *scenes,
        "--poses-per-scene", str(int(poses_per_scene)),
        "--pose-candidates-per-scene", str(int(pose_candidates_per_scene)),
        "--record-idle-stop-s", str(config.BACKGROUND_RECORD_IDLE_STOP_S),
        "--scene-wallclock-stop-s", str(int(scene_wallclock_s)),
        "--collection-shard-id", shard_id,
        "--benchmark-partition", "train_seen",
        "--seed", str(int(seed)),
        "--radii", *[str(value) for value in config.RADII_M],
        "--camera-heights", *[
            str(value) for value in config.BENCH_CAMERA_HEIGHTS_M],
        "--lengths", *[str(value) for value in config.GEN_LENGTHS],
        "--proposal-pairs-per-length",
        str(action_proposal.PAIRS_PER_LENGTH_DEFAULT),
        "--proposal-natural-per-length",
        str(action_proposal.NATURAL_PER_LENGTH_DEFAULT),
        "--ordinary-actions-per-pose",
        str(int(ordinary_actions_per_pose)),
        "--keep-per-length", str(config.KEEP_PER_LENGTH),
        "--action-mode", "balanced",
    ]
    if pose_exclusions is not None:
        values.extend(["--pose-exclusions", str(pose_exclusions)])
    return [
        *values, "--out", str(output_dir),
        "--code-revision", revision,
        "--resume",
    ]


def _direct_job(
        *, dataset: str, gpu_id: int, round_index: int,
        scenes: Sequence[str], output_root: Path, revision: str, paths: dict,
        poses_per_scene: int, pose_candidates_per_scene: int,
        scene_wallclock_s: int,
        ordinary_actions_per_pose: int,
        pose_exclusions: Path = None) -> dict:
    job_id = f"{dataset}-r{round_index:02d}-g{gpu_id}"
    output_dir = output_root / "records" / dataset / job_id
    command = [
        paths["python"], str(Path(paths["repository"]) / "scripts" / "collect.py"),
        "--backend", dataset,
        *_common_collect_args(
            scenes=scenes, output_dir=output_dir, revision=revision,
            shard_id=job_id, poses_per_scene=poses_per_scene,
            pose_candidates_per_scene=pose_candidates_per_scene,
            scene_wallclock_s=scene_wallclock_s,
            ordinary_actions_per_pose=ordinary_actions_per_pose,
            seed=20260811 + int(round_index) * 101 + int(gpu_id),
            pose_exclusions=pose_exclusions),
    ]
    if dataset == "r2r":
        command.extend([
            "--r2r-train-episodes", paths["r2r_train_episodes"],
            "--mp3d-root", paths["mp3d_root"],
            "--semantic-query-workers", "8",
        ])
    elif dataset == "gs":
        command.extend([
            "--gs-data-root", paths["gs_data_root"],
            "--gs-source-manifest", paths["gs_source_manifest"],
            "--min-objects", "0",
        ])
    else:
        raise ValueError(f"unsupported direct dataset {dataset!r}")
    return _job_value(
        job_id=job_id, dataset=dataset, gpu_id=gpu_id,
        round_index=round_index, scenes=scenes, output_dir=output_dir,
        output_root=output_root, command=command,
        pose_exclusions=pose_exclusions)


def _b1k_job(
        *, gpu_id: int, round_index: int, scenes: Sequence[str],
        output_root: Path, revision: str, paths: dict,
        poses_per_scene: int, pose_candidates_per_scene: int,
        scene_wallclock_s: int, ordinary_actions_per_pose: int,
        pose_exclusions: Path = None) -> dict:
    dataset = "b1k"
    job_id = f"b1k-r{round_index:02d}-g{gpu_id}"
    output_dir = output_root / "records" / dataset / job_id
    child_args = _common_collect_args(
        scenes=scenes, output_dir=output_dir, revision=revision,
        shard_id=job_id, poses_per_scene=poses_per_scene,
        pose_candidates_per_scene=pose_candidates_per_scene,
        scene_wallclock_s=scene_wallclock_s,
        ordinary_actions_per_pose=ordinary_actions_per_pose,
        seed=20260811 + int(round_index) * 101 + int(gpu_id),
        pose_exclusions=pose_exclusions)
    # The supervisor owns these values and rejects attempts to forward them.
    protected = {
        "--scenes", "--collection-shard-id", "--out", "--code-revision",
        "--resume",
    }
    forwarded = []
    index = 0
    while index < len(child_args):
        option = child_args[index]
        if option in protected:
            if option in {"--resume"}:
                index += 1
            elif option == "--scenes":
                index += 1 + len(scenes)
            else:
                index += 2
            continue
        forwarded.append(option)
        index += 1
    command = [
        paths["b1k_python"],
        str(Path(paths["repository"]) / "scripts" /
            "run_b1k_collection_shard.py"),
        "run",
        "--data-root", paths["b1k_data_root"],
        "--source-manifest", paths["b1k_source_manifest"],
        "--output-dir", str(output_dir),
        "--gpu-id", str(gpu_id),
        "--shard-id", job_id,
        "--code-revision", revision,
        "--resume",
        "--scene-timeout-s", str(
            b1k_supervisor_timeout_s(scene_wallclock_s)),
        "--scenes", *scenes,
        "--collect-args", *forwarded,
    ]
    return _job_value(
        job_id=job_id, dataset=dataset, gpu_id=gpu_id,
        round_index=round_index, scenes=scenes, output_dir=output_dir,
        output_root=output_root, command=command,
        isolate_cuda=False,
        pose_exclusions=pose_exclusions,
        extra_environment={
            "OMNIGIBSON_DATA_PATH": paths["b1k_data_root"],
            "OMNIGIBSON_APPDATA_PATH": str(
                Path(paths["b1k_data_root"]) / "appdata"),
            "OMNIGIBSON_HEADLESS": "True",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "OMNIGIBSON_GPU_ID": str(gpu_id),
        })


def _job_value(
        *, job_id: str, dataset: str, gpu_id: int, round_index: int,
        scenes: Sequence[str], output_dir: Path, output_root: Path,
        command: Sequence[str], extra_environment: Mapping[str, str] = None,
        isolate_cuda: bool = True, pose_exclusions: Path = None,
        ) -> dict:
    environment = {
        **_THREAD_ENVIRONMENT,
        **dict(extra_environment or {}),
    }
    if isolate_cuda:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return {
        "job_id": job_id,
        "dataset": dataset,
        "gpu_id": int(gpu_id),
        "round_index": int(round_index),
        "scenes": list(scenes),
        "output_dir": str(output_dir),
        "log_path": str(output_root / "controller" / "logs" /
                        f"{job_id}.log"),
        "command": [str(value) for value in command],
        "environment": environment,
        "pose_exclusions_path": (
            str(Path(pose_exclusions).resolve())
            if pose_exclusions is not None else None),
    }


def _bind_job_transaction(
        job: dict, *, revision: str, paths: Mapping[str, str],
        source_manifest_sha256: Mapping[str, str],
        poses_per_scene: int, pose_candidates_per_scene: int,
        ordinary_actions_per_pose: int) -> None:
    dataset = job["dataset"]
    scene_id = job["scenes"][0]
    source_path_key = dataset_contracts.source_path_key(dataset)
    child_shard = (
        f"{job['job_id']}-{scene_id}"
        if dataset == "b1k" else job["job_id"])
    job["transaction_binding"] = {
        "dataset": dataset,
        "backend": dataset,
        "scene_id": scene_id,
        "collection_shard_id": child_shard,
        "controller_job_id": job["job_id"],
        "output_dir": job["output_dir"],
        "revision": revision,
        "poses_per_scene": int(poses_per_scene),
        "pose_candidates_per_scene": int(pose_candidates_per_scene),
        "record_idle_stop_s": config.BACKGROUND_RECORD_IDLE_STOP_S,
        "benchmark_partition": "train_seen",
        "ordinary_actions_per_pose":
            int(ordinary_actions_per_pose),
        "scene_wallclock_stop_s": int(job["scene_wallclock_s"]),
        "supervisor_timeout_s": (
            b1k_supervisor_timeout_s(job["scene_wallclock_s"])
            if dataset == "b1k" else None),
        "seed": 20260811 + int(job["round_index"]) * 101 +
            int(job["gpu_id"]),
        "pose_exclusions_path": job.get("pose_exclusions_path"),
        "source_authority": {
            "path": str(paths[source_path_key]),
            "sha256": source_manifest_sha256[dataset],
        },
    }


def build_canary_manifest(
        *, revision: str, output_root: Path,
        scene_catalog: Mapping[str, Sequence[str]],
        selected_scenes: Mapping[str, Sequence[str]], paths: Mapping[str, str],
        source_manifest_sha256: Mapping[str, str],
        ordinary_actions_per_pose: int =
        config.ACTION_CANDIDATE_ORDINARY_PER_POSE) -> dict:
    """Build the profile bootstrap from the normal collection job builders."""
    return background_canary.build_manifest(
        revision=revision, output_root=output_root,
        scene_catalog=scene_catalog, selected_scenes=selected_scenes,
        paths=paths, source_manifest_sha256=source_manifest_sha256,
        ordinary_actions_per_pose=ordinary_actions_per_pose,
        direct_job_builder=_direct_job, b1k_job_builder=_b1k_job,
        bind_job=_bind_job_transaction)


validate_canary_manifest = background_canary.validate_manifest
def build_manifest(
        *, revision: str, output_root: Path, r2r_scenes: Sequence[str],
        gs_scenes: Sequence[str], b1k_scenes: Sequence[str],
        paths: Mapping[str, str], b1k_catalog_audit: Mapping,
        b1k_catalog_audit_sha256: str,
        r2r_source_manifest_sha256: str,
        gs_source_manifest_sha256: str,
        b1k_source_manifest_sha256: str,
        b1k_source_scene_ids: Sequence[str], capacity_profile: Mapping,
        capacity_profile_sha256: str, rounds: int = 1,
        source_scene_catalog: Mapping[str, Sequence[str]] = None) -> dict:
    """Build full-catalog, one-scene transaction jobs for four GPUs."""
    revision = str(revision).strip()
    if _HEX40.fullmatch(revision) is None:
        raise ValueError("background collection revision must be a clean SHA-1")
    r2r = _unique(r2r_scenes, label="R2R")
    gs = _unique(gs_scenes, label="GS")
    b1k = _unique(b1k_scenes, label="B1K")
    scheduled_catalog = {"r2r": r2r, "gs": gs, "b1k": b1k}
    source_catalog = background_scheduler.bind_source_catalog(
        scheduled_catalog, source_scene_catalog, DATASETS)
    catalog_passes = int(rounds)
    if catalog_passes < 1:
        raise ValueError("background collection catalog passes must be positive")
    resolved_paths = _required_paths(paths)
    audit_digest = str(b1k_catalog_audit_sha256 or "")
    if _HEX64.fullmatch(audit_digest) is None:
        raise ValueError("B1K catalog audit digest is invalid")
    if not isinstance(b1k_catalog_audit, Mapping) or \
            b1k_catalog_audit.get("schema") != \
            "b1k-catalog-authority-audit.v1":
        raise ValueError("B1K catalog audit schema is invalid")
    installed = _unique(
        b1k_catalog_audit.get("installed_scene_ids") or [],
        label="B1K installed catalog")
    accepted_rows = b1k_catalog_audit.get("accepted")
    excluded_rows = b1k_catalog_audit.get("excluded")
    if not isinstance(accepted_rows, list) or \
            not isinstance(excluded_rows, list):
        raise ValueError("B1K catalog audit terminal rows are invalid")
    accepted = [str((row or {}).get("scene_id") or "")
                for row in accepted_rows if isinstance(row, Mapping)]
    excluded = [str((row or {}).get("scene_id") or "")
                for row in excluded_rows if isinstance(row, Mapping)]
    if (len(accepted) != len(accepted_rows) or
            len(excluded) != len(excluded_rows) or
            not accepted or len(accepted) != len(set(accepted)) or
            len(excluded) != len(set(excluded)) or
            set(accepted) & set(excluded) or
            set(accepted) | set(excluded) != set(installed)):
        raise ValueError("B1K catalog audit terminal partition is invalid")
    if set(accepted) != set(source_catalog["b1k"]):
        raise ValueError(
            "B1K catalog audit accepted set differs from source manifest scenes")
    source_manifest_digest = str(b1k_source_manifest_sha256 or "")
    if _HEX64.fullmatch(source_manifest_digest) is None:
        raise ValueError("B1K source manifest digest is invalid")
    source_manifest_digests = {
        "r2r": str(r2r_source_manifest_sha256 or ""),
        "gs": str(gs_source_manifest_sha256 or ""),
        "b1k": source_manifest_digest,
    }
    if any(_HEX64.fullmatch(value) is None
           for value in source_manifest_digests.values()):
        raise ValueError("dataset source manifest digest is invalid")
    source_scene_ids = _unique(
        b1k_source_scene_ids, label="B1K source manifest")
    if set(source_scene_ids) != set(accepted):
        raise ValueError(
            "B1K accepted and source manifest scene sets differ")
    validate_capacity_profile(capacity_profile)
    profile = dict(capacity_profile)
    ordinary_actions_per_pose = int(profile["ordinary_actions_per_pose"])
    profile_digest = str(capacity_profile_sha256 or "")
    if profile_digest != profile.get("sha256"):
        raise ValueError("capacity profile digest differs")
    profile_sources = profile.get("source_authorities")
    expected_profile_sources = {
        dataset: {
            "path": resolved_paths[
                dataset_contracts.source_path_key(dataset)],
            "sha256": source_manifest_digests[dataset],
        }
        for dataset in DATASETS
    }
    if profile_sources != expected_profile_sources:
        raise ValueError("capacity profile source authority differs")
    for dataset, scenes in scheduled_catalog.items():
        if profile["datasets"][dataset]["catalog_scene_count"] != len(
                source_catalog[dataset]):
            raise ValueError(
                f"capacity profile {dataset} catalog scene count differs")
        canary_ids = {
            row["scene_id"]
            for row in profile["datasets"][dataset]["canary_scenes"]}
        if not canary_ids.issubset(set(scenes)):
            raise ValueError(
                f"capacity profile {dataset} canary is outside catalog")
    root = Path(output_root).resolve()
    round_values = []
    round_index = 0
    dataset_rank = {dataset: index for index, dataset in enumerate(DATASETS)}
    for catalog_pass in range(catalog_passes):
        pending = []
        for dataset, scenes in (("r2r", r2r), ("gs", gs), ("b1k", b1k)):
            weight = int(profile["datasets"][dataset]["scene_wallclock_s"])
            pending.extend({
                "dataset": dataset,
                "scene_id": scene_id,
                "scene_index": scene_index,
                "weight": weight,
            } for scene_index, scene_id in enumerate(scenes))
        pending.sort(key=lambda row: (
            -row["weight"], dataset_rank[row["dataset"]],
            row["scene_index"], row["scene_id"]))
        gpu_loads = [0, 0, 0, 0]
        gpu_queues = [[], [], [], []]
        for row in pending:
            gpu_id = min(range(4), key=lambda value: (gpu_loads[value], value))
            gpu_queues[gpu_id].append(row)
            gpu_loads[gpu_id] += row["weight"]
        for queue_index in range(max(map(len, gpu_queues))):
            jobs = []
            for gpu_id, queue in enumerate(gpu_queues):
                if queue_index >= len(queue):
                    continue
                scheduled = queue[queue_index]
                dataset = scheduled["dataset"]
                scene_id = scheduled["scene_id"]
                capacity = profile["datasets"][dataset]
                common = {
                    "gpu_id": gpu_id,
                    "round_index": round_index,
                    "scenes": [scene_id],
                    "output_root": root,
                    "revision": revision,
                    "paths": resolved_paths,
                    "poses_per_scene": capacity["records_per_scene"],
                    "pose_candidates_per_scene":
                        capacity["pose_attempt_cap"],
                    "ordinary_actions_per_pose": ordinary_actions_per_pose,
                    "pose_exclusions": (
                        root / "controller" / "pose_exclusions" / dataset /
                        f"{scene_id}-pass{catalog_pass:02d}.json"
                        if catalog_pass > 0 else None),
                }
                job = (
                    _b1k_job(
                        **common,
                        scene_wallclock_s=capacity["scene_wallclock_s"])
                    if dataset == "b1k" else
                    _direct_job(
                        dataset=dataset, **common,
                        scene_wallclock_s=capacity["scene_wallclock_s"]))
                job["catalog_pass"] = catalog_pass
                job["weight"] = scheduled["weight"]
                job["scene_wallclock_s"] = capacity["scene_wallclock_s"]
                _bind_job_transaction(
                    job, revision=revision, paths=resolved_paths,
                    source_manifest_sha256=source_manifest_digests,
                    poses_per_scene=capacity["records_per_scene"],
                    pose_candidates_per_scene=capacity["pose_attempt_cap"],
                    ordinary_actions_per_pose=ordinary_actions_per_pose)
                jobs.append(job)
            round_values.append({
                "round_index": round_index,
                "catalog_pass": catalog_pass,
                "jobs": jobs,
            })
            round_index += 1
    value = {
        "schema": CONTROLLER_SCHEMA,
        "revision": revision,
        "output_root": str(root),
        "paths": resolved_paths,
        "collection": {
            "catalog_passes": catalog_passes,
            "ordinary_actions_per_pose": ordinary_actions_per_pose,
            "heartbeat_interval_s": config.BACKGROUND_HEARTBEAT_INTERVAL_S,
            "initialization_deadline_s":
                config.BACKGROUND_INITIALIZATION_DEADLINE_S,
            "first_record_deadline_s":
                config.BACKGROUND_FIRST_RECORD_DEADLINE_S,
            "retryable_initialization_attempts":
                config.BACKGROUND_RETRYABLE_INITIALIZATION_ATTEMPTS,
        },
        "quota": _dataset_quota_rows(),
        "authorities": {
            "b1k_catalog_audit": {
                "sha256": audit_digest,
                "installed_scene_ids": installed,
                "accepted_scene_ids": accepted,
                "excluded_scene_ids": excluded,
            },
            "b1k_source_manifest": {
                "sha256": source_manifest_digest,
                "accepted_scene_ids": source_scene_ids,
            },
            "capacity_profile": {"sha256": profile_digest},
        },
        "capacity_profile": profile,
        "catalog_exclusions": {
            "gs": [
                {"scene_id": scene_id, "reason": reason}
                for scene_id, reason in sorted(
                    config.BACKGROUND_GS_CATALOG_EXCLUSIONS.items())
            ],
        },
        "source_scene_catalog": source_catalog,
        "scene_catalog": scheduled_catalog,
        "rounds": round_values,
    }
    return {**value, "sha256": _canonical_sha256(value)}


def validate_manifest(value: dict) -> None:
    if not isinstance(value, dict) or value.get("schema") != CONTROLLER_SCHEMA:
        raise ValueError("background collection manifest schema is invalid")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != _canonical_sha256(body):
        raise ValueError("background collection manifest digest is invalid")
    if _HEX40.fullmatch(str(value.get("revision") or "")) is None:
        raise ValueError("background collection manifest revision is invalid")
    passes = int((value.get("collection") or {}).get("catalog_passes") or 0)
    if passes < 1:
        raise ValueError("background collection catalog passes are invalid")
    profile = value.get("capacity_profile")
    validate_capacity_profile(profile)
    ordinary_actions_per_pose = int(profile["ordinary_actions_per_pose"])
    if (value.get("collection") or {}).get(
            "ordinary_actions_per_pose") != ordinary_actions_per_pose:
        raise ValueError("background collection action budget differs")
    authorities = value.get("authorities") or {}
    if (authorities.get("capacity_profile") or {}).get("sha256") != \
            profile["sha256"]:
        raise ValueError("background collection capacity authority differs")
    expected_exclusions = {
        "gs": [
            {"scene_id": scene_id, "reason": reason}
            for scene_id, reason in sorted(
                config.BACKGROUND_GS_CATALOG_EXCLUSIONS.items())
        ],
    }
    if value.get("catalog_exclusions") != expected_exclusions:
        raise ValueError("background collection catalog exclusions differ")
    catalog = value.get("scene_catalog") or {}
    source_catalog = value.get("source_scene_catalog") or {}
    background_scheduler.validate_catalog_binding(
        catalog, source_catalog, DATASETS, profile)
    audit = authorities.get("b1k_catalog_audit") or {}
    if _HEX64.fullmatch(str(audit.get("sha256") or "")) is None or \
            set(audit.get("accepted_scene_ids") or []) != \
            set(source_catalog["b1k"]):
        raise ValueError("background collection B1K audit binding is invalid")
    installed = audit.get("installed_scene_ids") or []
    excluded = audit.get("excluded_scene_ids") or []
    if set(installed) != set(source_catalog["b1k"]) | set(excluded) or \
            set(source_catalog["b1k"]) & set(excluded):
        raise ValueError("background collection B1K audit partition is invalid")
    source_authority = authorities.get("b1k_source_manifest") or {}
    if (_HEX64.fullmatch(str(source_authority.get("sha256") or "")) is None or
            set(source_authority.get("accepted_scene_ids") or []) !=
            set(source_catalog["b1k"])):
        raise ValueError(
            "background collection B1K source manifest binding is invalid")
    quotas = value.get("quota")
    if quotas != _dataset_quota_rows():
        raise ValueError("background collection dataset quotas are invalid")
    scheduled = {
        catalog_pass: {name: [] for name in DATASETS}
        for catalog_pass in range(passes)
    }
    rounds = value.get("rounds") or []
    if not rounds:
        raise ValueError("background collection rounds are absent")
    for expected_index, round_value in enumerate(rounds):
        if round_value.get("round_index") != expected_index:
            raise ValueError("background collection round indices are invalid")
        catalog_pass = round_value.get("catalog_pass")
        if catalog_pass not in scheduled:
            raise ValueError("background collection catalog pass is invalid")
        gpu_ids = []
        for job in round_value.get("jobs") or []:
            dataset = job.get("dataset")
            scenes = job.get("scenes")
            if dataset not in DATASETS or not isinstance(scenes, list) or \
                    len(scenes) != 1:
                raise ValueError(
                    "background collection jobs must bind one catalog scene")
            if job.get("catalog_pass") != catalog_pass:
                raise ValueError(
                    "background collection job catalog pass differs")
            capacity = profile["datasets"][dataset]
            binding = job.get("transaction_binding")
            expected_pose_exclusions = (
                str((Path(value["output_root"]) / "controller" /
                     "pose_exclusions" / dataset /
                     f"{scenes[0]}-pass{int(catalog_pass):02d}.json").
                    resolve())
                if int(catalog_pass) > 0 else None)
            expected_shard = (
                f"{job.get('job_id')}-{scenes[0]}"
                if dataset == "b1k" else job.get("job_id"))
            if not isinstance(binding, Mapping) or any((
                    binding.get("dataset") != dataset,
                    binding.get("backend") != dataset,
                    binding.get("scene_id") != scenes[0],
                    binding.get("collection_shard_id") != expected_shard,
                    binding.get("controller_job_id") != job.get("job_id"),
                    binding.get("output_dir") != job.get("output_dir"),
                    binding.get("revision") != value["revision"],
                    binding.get("poses_per_scene") !=
                    capacity["records_per_scene"],
                    binding.get("pose_candidates_per_scene") !=
                    capacity["pose_attempt_cap"],
                    binding.get("record_idle_stop_s") !=
                    config.BACKGROUND_RECORD_IDLE_STOP_S,
                    binding.get("ordinary_actions_per_pose") !=
                    ordinary_actions_per_pose,
                    binding.get("scene_wallclock_stop_s") !=
                    capacity["scene_wallclock_s"],
                    binding.get("supervisor_timeout_s") != (
                        b1k_supervisor_timeout_s(
                            capacity["scene_wallclock_s"])
                        if dataset == "b1k" else None),
                    binding.get("seed") != 20260811 +
                    int(job.get("round_index")) * 101 +
                    int(job.get("gpu_id")),
                    job.get("pose_exclusions_path") !=
                    expected_pose_exclusions,
                    binding.get("pose_exclusions_path") !=
                    expected_pose_exclusions,
            )):
                raise ValueError(
                    "background collection job transaction binding differs")
            source_binding = binding.get("source_authority") or {}
            expected_source_path = value["paths"][
                dataset_contracts.source_path_key(dataset)]
            expected_source_sha = source_binding.get("sha256")
            if _HEX64.fullmatch(str(expected_source_sha or "")) is None:
                raise ValueError(
                    "background collection job source digest is invalid")
            if dataset == "b1k" and \
                    expected_source_sha != source_authority["sha256"]:
                raise ValueError(
                    "background collection B1K job source digest differs")
            if source_binding != {
                    "path": expected_source_path,
                    "sha256": expected_source_sha}:
                raise ValueError(
                    "background collection job source binding differs")
            if (job.get("weight") != capacity["scene_wallclock_s"] or
                    job.get("scene_wallclock_s") !=
                    capacity["scene_wallclock_s"]):
                raise ValueError(
                    "background collection job capacity binding differs")
            background_job_contract.validate_job_command(job, value["paths"])
            scheduled[catalog_pass][dataset].extend(scenes)
            gpu_ids.append(job.get("gpu_id"))
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError("background collection round reuses a GPU")
    for catalog_pass, datasets in scheduled.items():
        for dataset, scenes in datasets.items():
            if sorted(scenes) != sorted(catalog[dataset]):
                raise ValueError(
                    f"background collection pass {catalog_pass} does not "
                    f"cover the full {dataset} catalog")


def validate_external_authorities(manifest: dict) -> None:
    """Reopen B1K authorities and bind their bytes and terminal scene sets."""
    validate_manifest(manifest)
    background_authorities.validate_external_authorities(manifest)


def validate_launch_resources(
        gpu_memory_used_mib: Sequence[int], *, free_bytes: int,
        max_gpu_memory_mib: int = config.BACKGROUND_MAX_IDLE_GPU_MEMORY_MIB,
        minimum_free_bytes: int = config.BACKGROUND_MIN_FREE_STORAGE_BYTES,
        capacity_profile: Mapping = None) -> None:
    """Fail before the first wave when GPUs or durable storage are occupied."""
    if capacity_profile is not None:
        validate_capacity_profile(capacity_profile)
    values = [int(value) for value in gpu_memory_used_mib]
    if len(values) != 4 or any(value < 0 for value in values):
        raise ValueError("background collection GPU inventory is invalid")
    busy = [index for index, value in enumerate(values)
            if value > int(max_gpu_memory_mib)]
    if busy:
        raise ValueError(
            "background collection GPU(s) are already busy: " +
            ", ".join(map(str, busy)))
    if int(free_bytes) < int(minimum_free_bytes):
        raise ValueError("background collection storage has under 120 GiB free")


def initial_state(manifest: dict) -> dict:
    validate_manifest(manifest)
    return {
        "schema": STATE_SCHEMA,
        "manifest_sha256": manifest["sha256"],
        "status": "ready",
        "current_round": 0,
        "jobs": {},
        "datasets": {},
        "compile_checkpoints": [],
        "updated_time_unix": None,
    }


def validate_state(manifest: dict, state: Mapping) -> None:
    """Validate every persisted job against its immutable manifest row."""
    validate_manifest(manifest)
    if (not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA or
            state.get("manifest_sha256") != manifest["sha256"]):
        raise ValueError("background collection state does not match manifest")
    current_round = state.get("current_round")
    if (isinstance(current_round, bool) or not isinstance(current_round, int) or
            not 0 <= current_round <= len(manifest["rounds"])):
        raise ValueError("background collection state round is invalid")
    allowed_statuses = {
        "ready", "running", "compile_failed", "complete",
        "capacity_shortfall", "interrupted",
    }
    if state.get("status") not in allowed_statuses:
        raise ValueError("background collection state status is invalid")
    checkpoints = state.get("compile_checkpoints")
    if (not isinstance(checkpoints, list) or
            any(not isinstance(value, str) or not value for value in checkpoints)
            or len(checkpoints) != len(set(checkpoints))):
        raise ValueError("background collection state checkpoints are invalid")
    summaries = {
        checkpoint: background_checkpoint.load(manifest, checkpoint)
        for checkpoint in checkpoints
    }
    datasets = state.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("background collection state datasets are invalid")
    if datasets and set(datasets) != set(DATASETS):
        raise ValueError(
            "background collection state dataset provenance rows are invalid")
    jobs = state.get("jobs") or {}
    if not isinstance(jobs, Mapping):
        raise ValueError("background collection state jobs are invalid")
    status = state.get("status")
    if status == "ready" and any((current_round, jobs, datasets, checkpoints)):
        raise ValueError("background collection ready state is not empty")
    if status == "compile_failed" and (
            set(datasets) != set(DATASETS) or any(
                (datasets[name] or {}).get("status") != "compile_failed"
                for name in DATASETS)):
        raise ValueError(
            "background collection compile failure transition is invalid")
    if status in {"complete", "capacity_shortfall"} and (
            current_round != len(manifest["rounds"]) or not checkpoints or
            set(datasets) != set(DATASETS)):
        raise ValueError(
            "background collection terminal transition is invalid")
    for dataset, row in datasets.items():
        if not isinstance(row, Mapping):
            raise ValueError("background collection state dataset row is invalid")
        if row.get("status") == "compile_failed":
            if (state.get("status") != "compile_failed" or
                    row.get("checkpoint") in checkpoints):
                raise ValueError(
                    "background collection compile failure transition is invalid")
            continue
        checkpoint = row.get("checkpoint")
        identity = row.get("checkpoint_summary")
        if checkpoint not in summaries or not isinstance(identity, Mapping):
            raise ValueError(
                "background collection dataset provenance is invalid")
        summary = summaries[checkpoint]
        _root, final, _staging = background_checkpoint.paths(
            manifest, checkpoint)
        summary_path = final / background_checkpoint.SUMMARY_NAME
        expected_identity = {
            "path": str(summary_path),
            "sha256": io_utils.sha256_file(summary_path),
            "content_sha256": summary["sha256"],
        }
        if dict(identity) != expected_identity:
            raise ValueError(
                "background checkpoint state identity differs")
        compiled = summary.get("result") or {}
        expected_dataset = (compiled.get("datasets") or {}).get(dataset)
        expected = {
            **dict(expected_dataset or {}),
            "artifact": compiled.get("artifact"),
            "coverage": compiled.get("coverage"),
            "gt_as_pred": compiled.get("gt_as_pred"),
            "combined_six_task_macro": compiled.get("six_task_macro"),
            "checkpoint": checkpoint,
            "checkpoint_summary": dict(identity),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "background collection dataset provenance differs")
    definitions = {
        job["job_id"]: job for round_value in manifest["rounds"]
        for job in round_value["jobs"]}
    reusable = {"completed", "partial_valid"}
    nonreusable = {"zero_yield", "failed", "unresolved", None}
    runtime_statuses = {
        "running", "running_slow", "stopping_slow", "stopping_sigint",
        "stopping_sigterm", "stopping_sigkill", "completed",
        "completed_zero_yield", "completed_capacity_shortfall",
        "failed_partial", "failed_retryable", "failed_zero_yield",
        "recovered_valid", "interrupted",
    }
    for job_id, runtime in jobs.items():
        definition = definitions.get(job_id)
        if definition is None or not isinstance(runtime, Mapping):
            raise ValueError("background collection state job is unknown")
        if any((
                runtime.get("job_id") != job_id,
                runtime.get("dataset") != definition["dataset"],
                runtime.get("scene_id") != definition["scenes"][0],
                runtime.get("round_index") != definition["round_index"],
        )):
            raise ValueError("background collection state job binding differs")
        catalog_status = runtime.get("catalog_status")
        runtime_status = runtime.get("status")
        if runtime_status not in runtime_statuses:
            raise ValueError(
                "background collection state job status is invalid")
        if (runtime_status == "recovered_valid" and
                (catalog_status != "completed" or runtime.get("returncode") !=
                 RECOVERED_UNKNOWN_RETURNCODE)):
            raise ValueError(
                "background collection recovered job transition is invalid")
        if (runtime.get("returncode") == RECOVERED_UNKNOWN_RETURNCODE and
                catalog_status == "completed" and
                runtime_status != "recovered_valid"):
            raise ValueError(
                "background collection recovered job transition is invalid")
        validation = runtime.get("source_validation")
        sources = (
            validation.get("sources")
            if isinstance(validation, Mapping) else None)
        if catalog_status in reusable:
            if not isinstance(sources, list) or not sources:
                raise ValueError(
                    "background collection reusable state lacks sources")
            output = Path(definition["output_dir"]).resolve()
            for source in sources:
                if not isinstance(source, Mapping):
                    raise ValueError(
                        "background collection reusable source is invalid")
                path = Path(str(source.get("path") or "")).resolve()
                try:
                    path.relative_to(output)
                except ValueError as error:
                    raise ValueError(
                        "background collection reusable source escapes job") \
                        from error
                if any(_HEX64.fullmatch(str(source.get(key) or "")) is None
                       for key in ("records_sha256", "run_meta_sha256")):
                    raise ValueError(
                        "background collection reusable source digest is invalid")
                metadata = path.with_name("run_meta.json")
                if (not path.is_file() or not metadata.is_file() or
                        io_utils.sha256_file(path) !=
                        source["records_sha256"] or
                        io_utils.sha256_file(metadata) !=
                        source["run_meta_sha256"]):
                    raise ValueError(
                        "background collection reusable source identity differs")
        elif catalog_status in nonreusable:
            if sources:
                raise ValueError(
                    "background collection nonreusable state claims sources")
        else:
            raise ValueError(
                "background collection state catalog status is invalid")


initialize_capacity_events = background_capacity.initialize_controller_events
record_capacity_event = background_capacity.record_controller_event
# Kept as a narrow compatibility hook for the canary supervisor and tests.
_default_launcher = background_scheduler.default_launcher


def start_round(
        manifest: dict, state: dict, *, round_index: int,
        launcher: Callable = background_scheduler.default_launcher,
        now: float = None,
        completed_datasets: Sequence[str] = ()) -> dict:
    """Launch one wave; returned process handles stay outside durable state."""
    validate_manifest(manifest)
    timestamp = float(time.time() if now is None else now)
    round_value = manifest["rounds"][int(round_index)]
    terminal = {"completed", "partial_valid", "zero_yield", "failed"}
    jobs = [
        job for job in round_value["jobs"]
        if job["dataset"] not in set(completed_datasets) and
        (state.get("jobs", {}).get(job["job_id"], {}).get("catalog_status")
         not in terminal)]
    gpu_ids = [int(job["gpu_id"]) for job in jobs]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("background collection round reuses a GPU")
    if any(value.get("status") in {"running", "running_slow"}
           for value in state.get("jobs", {}).values()):
        raise ValueError("background collection already has running jobs")
    return background_scheduler.launch_jobs(
        manifest, state, jobs, launcher=launcher, now=timestamp)


def _manifest_job(manifest: dict, job_id: str) -> dict:
    matches = [
        job for round_value in manifest["rounds"]
        for job in round_value["jobs"] if job["job_id"] == str(job_id)]
    if len(matches) != 1:
        raise ValueError(f"background collection job is unknown: {job_id!r}")
    return matches[0]


def retry_job(
        manifest: dict, state: dict, job_id: str, *,
        launcher: Callable = background_scheduler.default_launcher,
        now: float = None):
    """Retry exactly one initialization/OOM failure, never zero-yield GT."""
    validate_manifest(manifest)
    runtime = state.get("jobs", {}).get(str(job_id))
    if runtime is None or runtime.get("status") != "failed_retryable":
        raise ValueError("background collection job is not retryable")
    allowed_retries = int(
        manifest["collection"]["retryable_initialization_attempts"])
    attempt = int(runtime.get("attempt") or 0)
    if attempt > allowed_retries:
        raise ValueError("background collection retry budget is exhausted")
    job = _manifest_job(manifest, str(job_id))
    timestamp = float(time.time() if now is None else now)
    initialize_capacity_events(manifest, job, time_unix=timestamp)
    process = launcher(job, job["environment"], job["log_path"])
    runtime.update({
        "pid": int(process.pid),
        "status": "running",
        "attempt": attempt + 1,
        "started_time_unix": timestamp,
        "finished_time_unix": None,
        "durable_records": 0,
    })
    for name in (
            "capacity_backend_ready_recorded", "performance_violation",
            "termination_requested_time_unix", "termination_signal",
            "returncode"):
        runtime.pop(name, None)
    state["updated_time_unix"] = timestamp
    return process


def _record_paths(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return sorted(path for path in output_dir.rglob("records.jsonl")
                  if path.is_file())


def _json_event_paths(job: Mapping) -> list[Path]:
    values = [Path(str(job["log_path"]))]
    output = Path(str(job["output_dir"]))
    if output.is_dir():
        values.extend(sorted(output.rglob("*.log")))
    return values


def job_progress(job: Mapping, *, started_time: float = None) -> dict:
    output_dir = Path(str(job["output_dir"]))
    paths = _record_paths(output_dir)
    funnel_paths = (
        sorted(output_dir.rglob("collection_funnel.json"))
        if output_dir.is_dir() else [])
    count = 0
    first_record_time = None
    last_record_time = None
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            count += sum(bool(line.strip()) for line in stream)
        if path.stat().st_size:
            first_record_time = (
                path.stat().st_mtime if first_record_time is None else
                min(first_record_time, path.stat().st_mtime))
            last_record_time = (
                path.stat().st_mtime if last_record_time is None else
                max(last_record_time, path.stat().st_mtime))
    activity_paths = [path for path in _json_event_paths(job)
                      if path.is_file()] + paths + funnel_paths
    last_activity_time = max(
        (path.stat().st_mtime for path in activity_paths), default=None)
    return {
        "backend_ready_time": background_capacity.backend_ready_time(
            _json_event_paths(job), started_time=started_time),
        "durable_records": int(count),
        "first_record_time": first_record_time,
        "last_record_time": last_record_time,
        "last_activity_time": last_activity_time,
        "record_paths": [str(path) for path in paths],
        "funnel_paths": [str(path) for path in funnel_paths],
    }


def job_health(
        job: Mapping, *, now: float = None,
        first_record_deadline_s: float =
        config.BACKGROUND_FIRST_RECORD_DEADLINE_S,
        initialization_deadline_s: float =
        config.BACKGROUND_INITIALIZATION_DEADLINE_S,
        scene_wallclock_s: float = config.BACKGROUND_MAX_SCENE_WALLCLOCK_S,
        started_time: float = None) -> dict:
    timestamp = float(time.time() if now is None else now)
    attempt_started = (
        started_time if started_time is not None else
        job.get("started_time_unix"))
    started = (
        float(attempt_started) if attempt_started is not None else timestamp)
    progress = job_progress(
        job,
        started_time=(
            float(attempt_started) if attempt_started is not None else None))
    ready = progress["backend_ready_time"]
    if ready is None and timestamp - started > float(initialization_deadline_s):
        return {
            **progress, "status": "initialization_slow",
            "seconds_since_start": timestamp - started,
        }
    if ready is not None and timestamp - ready > float(scene_wallclock_s):
        return {
            **progress, "status": "scene_wallclock_reached",
            "seconds_since_ready": timestamp - ready,
        }
    record_after_ready = (
        ready is not None and progress["last_record_time"] is not None and
        progress["last_record_time"] >= ready)
    if ready is not None and not record_after_ready and \
            timestamp - ready > float(first_record_deadline_s):
        return {
            **progress, "status": "first_record_slow",
            "seconds_since_ready": timestamp - ready,
        }
    if record_after_ready and \
            timestamp - progress["last_record_time"] > \
            float(first_record_deadline_s):
        return {
            **progress, "status": "inter_record_slow",
            "seconds_since_record": timestamp - progress["last_record_time"],
        }
    return {**progress, "status": "healthy"}


def dataset_catalog_coverage(
        manifest: dict, state: dict, dataset: str,
        catalog_pass: int = 0) -> dict:
    """Report the first-pass catalog transaction state for one dataset."""
    dataset = str(dataset)
    if dataset not in DATASETS:
        raise ValueError("background collection dataset is invalid")
    planned = [
        job for round_value in manifest.get("rounds") or []
        for job in round_value.get("jobs") or []
        if job.get("dataset") == dataset and
        job.get("catalog_pass") == int(catalog_pass)
    ]
    if dataset == "b1k":
        excluded_ids = list((manifest.get("authorities") or {}).get(
            "b1k_catalog_audit", {}).get("excluded_scene_ids") or [])
    else:
        excluded_ids = [
            str(row.get("scene_id") or "")
            for row in (manifest.get("catalog_exclusions") or {}).get(
                dataset, []) if isinstance(row, Mapping)]
    state_names = (
        "completed", "partial_valid", "zero_yield", "excluded", "failed",
        "unresolved",
    )
    scenes_by_state = {name: [] for name in state_names}
    scenes_by_state["excluded"].extend(excluded_ids)
    unresolved_rows = []
    touched = 0
    produced = 0
    shortfall_scenes = {
        f"L{length}": []
        for length in manifest["quota"][dataset]["required_lengths"]
    }
    for job in planned:
        runtime = (state.get("jobs") or {}).get(job["job_id"])
        scene_id = job["scenes"][0]
        if runtime is not None:
            touched += 1
        status = (
            runtime.get("catalog_status")
            if isinstance(runtime, Mapping) else None)
        if status not in state_names or status == "excluded":
            status = "unresolved"
        scenes_by_state[status].append(scene_id)
        if status in {"completed", "partial_valid"}:
            produced += 1
            source_validation = runtime.get("source_validation") or {}
            for length in source_validation.get("length_shortfall") or []:
                if length in shortfall_scenes:
                    shortfall_scenes[length].append(scene_id)
        if status == "unresolved":
            unresolved_rows.append({
                "job_id": job["job_id"],
                "scene_id": scene_id,
                "status": (
                    runtime.get("status")
                    if isinstance(runtime, Mapping) else None),
            })
    states = {
        name: len(scenes_by_state[name]) for name in state_names}
    catalog_count = len(planned) + len(excluded_ids)
    terminal_count = catalog_count - states["unresolved"]
    per_length = {
        length: {
            "scene_count": len(scene_ids),
            "rate": len(scene_ids) / produced if produced else 0.0,
            "scene_ids": scene_ids,
        }
        for length, scene_ids in shortfall_scenes.items()
    }
    return {
        "dataset": dataset,
        "catalog_pass": int(catalog_pass),
        "counts": {
            "catalog": catalog_count,
            "scheduled": len(planned),
            "touched": touched,
            "terminal": terminal_count,
            "produced": produced,
        },
        "states": states,
        "scene_ids_by_state": scenes_by_state,
        "per_length_scene_shortfall": per_length,
        "planned_scene_transactions": len(planned),
        "completed_scene_transactions": states["completed"],
        "reusable_scene_transactions": (
            states["completed"] + states["partial_valid"]),
        "complete": bool(catalog_count and not unresolved_rows),
        "incomplete": unresolved_rows,
    }


def dataset_can_stop(manifest: dict, state: dict, dataset: str) -> bool:
    """Require both compiled-QA quota and a terminal first catalog pass."""
    row = (state.get("datasets") or {}).get(str(dataset)) or {}
    quota = row.get("quota") or {}
    checkpoint = row.get("checkpoint")
    identity = row.get("checkpoint_summary")
    if (checkpoint not in (state.get("compile_checkpoints") or []) or
            not isinstance(identity, Mapping)):
        return False
    own_complete = bool(
        quota.get("complete") is True and
        dataset_catalog_coverage(manifest, state, dataset)["complete"])
    return candidate_quota.allows_dataset_stop(
        dataset, own_complete=own_complete, state=state)


def compilation_checkpoints_due(
        manifest: dict, state: dict, *, round_index: int,
        completed_datasets: Sequence[str] = ()) -> list[str]:
    """Return the only global QA checkpoints permitted after one wave."""
    completed = set(state.get("compile_checkpoints") or [])
    produced = {dataset: set() for dataset in DATASETS}
    for round_value in manifest.get("rounds") or []:
        for job in round_value.get("jobs") or []:
            runtime = (state.get("jobs") or {}).get(job["job_id"]) or {}
            if runtime.get("catalog_status") in {"completed", "partial_valid"}:
                produced[job["dataset"]].add(job["scenes"][0])
    due = []
    early = "early-two-scenes"
    if early not in completed and all(
            len(produced[dataset]) >= 2 for dataset in DATASETS):
        due.append(early)
    current = manifest["rounds"][int(round_index)]
    catalog_pass = int(current["catalog_pass"])
    final_catalog_pass = max(
        int(row["catalog_pass"]) for row in manifest["rounds"])
    if catalog_pass not in {0, final_catalog_pass}:
        return due
    last_round = max(
        row["round_index"] for row in manifest["rounds"]
        if int(row["catalog_pass"]) == catalog_pass)
    boundary = f"catalog-pass-{catalog_pass:02d}"
    if (int(round_index) == last_round and boundary not in completed and
            background_scheduler.catalog_pass_complete(
                manifest, state, catalog_pass=catalog_pass,
                completed_datasets=completed_datasets)):
        due.append(boundary)
    return due


def _reusable_dataset_sources(
        manifest: dict, dataset: str, state: Mapping) -> list[dict]:
    sources = []
    seen = set()
    for round_value in manifest.get("rounds") or []:
        for job in round_value.get("jobs") or []:
            if job.get("dataset") != str(dataset):
                continue
            runtime = (state.get("jobs") or {}).get(job["job_id"]) or {}
            if runtime.get("catalog_status") not in {
                    "completed", "partial_valid"}:
                continue
            validation = runtime.get("source_validation") or {}
            rows = validation.get("sources")
            if not isinstance(rows, list) or not rows:
                raise ValueError(
                    "reusable scene transaction has no source identities")
            output = Path(job["output_dir"]).resolve()
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError(
                        "reusable scene source identity is invalid")
                path = Path(str(row.get("path") or "")).resolve()
                try:
                    path.relative_to(output)
                except ValueError as error:
                    raise ValueError(
                        "reusable scene source escapes its transaction") \
                        from error
                if path in seen:
                    raise ValueError("reusable scene source is duplicated")
                if _HEX64.fullmatch(str(row.get("records_sha256") or "")) \
                        is None or _HEX64.fullmatch(str(
                            row.get("run_meta_sha256") or "")) is None:
                    raise ValueError(
                        "reusable scene source digest is invalid")
                seen.add(path)
                sources.append(dict(row))
    return sources


def discover_dataset_records(
        manifest: dict, dataset: str, *, state: Mapping) -> list[Path]:
    """Return only sources from immediately authenticated reusable scenes."""
    return [Path(row["path"])
            for row in _reusable_dataset_sources(manifest, dataset, state)]


def discover_global_records(
        manifest: dict, *, state: Mapping) -> list[Path]:
    """Return the stable three-dataset source order for global QA selection."""
    return [
        Path(row["path"])
        for dataset in DATASETS
        for row in _reusable_dataset_sources(manifest, dataset, state)
    ]


def _validate_transaction_run_binding(
        job: Mapping, metadata: Mapping, records_path: Path) -> dict:
    binding = job.get("transaction_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("scene transaction binding is absent")
    dataset = binding["dataset"]
    scene_id = binding["scene_id"]
    run_contract = collection_cli.load_spool_run_contract(records_path)
    params = run_contract.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("records spool run params are invalid")
    expected_output = Path(binding["output_dir"])
    if dataset == "b1k":
        expected_output /= scene_id
    expected_params = {
        "backend": binding["backend"],
        "collection_mode": "main",
        "scenes": [scene_id],
        "collection_shard_id": binding["collection_shard_id"],
        "poses_per_scene": binding["poses_per_scene"],
        "pose_candidates_per_scene": binding["pose_candidates_per_scene"],
        "record_idle_stop_s": binding["record_idle_stop_s"],
        "scene_wallclock_stop_s": float(
            binding["scene_wallclock_stop_s"]),
        "seed": binding["seed"],
        "benchmark_partition": binding["benchmark_partition"],
        "ordinary_actions_per_pose": binding["ordinary_actions_per_pose"],
        "pose_exclusions": binding.get("pose_exclusions_path"),
    }
    source_key = dataset_contracts.source_path_key(dataset)
    expected_params[source_key] = binding["source_authority"]["path"]
    resolved = run_contract.get("resolved_scenes")
    resolved_row = (
        resolved[0] if isinstance(resolved, list) and len(resolved) == 1 and
        isinstance(resolved[0], Mapping) else {})
    pose_exclusions_path = binding.get("pose_exclusions_path")
    pose_exclusions_digest = (
        io_utils.sha256_file(Path(pose_exclusions_path))
        if pose_exclusions_path is not None else None)
    expected = {
        **{f"params.{key}": value for key, value in expected_params.items()},
        "params.pose_exclusions_sha256": pose_exclusions_digest,
        "code_revision": binding["revision"],
        "code_dirty": False,
        "resolved_scene_id": scene_id,
        "resolved_source_dataset": dataset,
        "resolved_source_manifest_sha256":
            binding["source_authority"]["sha256"],
        "records_parent": str(expected_output.resolve()),
        "run_contract_sha256": _canonical_sha256(run_contract),
    }
    actual = {
        **{f"params.{key}": params.get(key) for key in expected_params},
        "params.pose_exclusions_sha256":
            params.get("pose_exclusions_sha256"),
        "code_revision": run_contract.get("code_revision"),
        "code_dirty": run_contract.get("code_dirty"),
        "resolved_scene_id": resolved_row.get("scene_id"),
        "resolved_source_dataset": resolved_row.get("source_dataset"),
        "resolved_source_manifest_sha256":
            resolved_row.get("source_manifest_sha256"),
        "records_parent": str(records_path.parent.resolve()),
        "run_contract_sha256": metadata.get("run_contract_sha256"),
    }
    mismatches = [
        key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        details = "; ".join(
            f"{key}: expected={expected[key]!r}, actual={actual.get(key)!r}"
            for key in mismatches)
        raise ValueError(f"scene transaction binding differs: {details}")
    return run_contract


def validate_scene_transaction(job: Mapping, *, returncode: int,
        stopped_for_capacity: bool = False) -> dict:
    """Source-authenticate one finished scene before publishing its state."""
    dataset = str(job.get("dataset") or "")
    if dataset not in DATASETS:
        raise ValueError("scene transaction dataset is invalid")
    paths = _record_paths(Path(str(job["output_dir"])))
    if not paths or not any(path.stat().st_size for path in paths):
        return {
            "catalog_status": (
                "zero_yield" if int(returncode) == 0 or stopped_for_capacity
                else "failed"),
            "source_validated_records": 0,
            "length_shortfall": [],
            "record_paths": [],
        }
    total = 0
    complete = True
    length_shortfall = set()
    source_identities = []
    for path in paths:
        metadata_path = path.with_name("run_meta.json")
        finalization = collection_closeout.load_finalization(
            path.parent, allowed_statuses={"completed", "partial"})
        if finalization["source_validation"] != "passed":
            raise ValueError(
                "scene transaction lacks passed source validation")
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "background scene run metadata is unreadable") from error
        if not isinstance(metadata, Mapping):
            raise ValueError("background scene run metadata is invalid")
        if metadata.get("run_contract_sha256") != \
                finalization["run_contract_sha256"]:
            raise ValueError(
                "scene transaction finalization run contract differs")
        _validate_transaction_run_binding(job, metadata, path)
        count = int(finalization["record_count"])
        if int(count) < 1:
            raise ValueError(
                "scene transaction source validation produced no records")
        total += int(count)
        source_identities.append({
            "path": str(path.resolve()),
            "records_sha256": finalization["records_sha256"],
            "run_meta_sha256": finalization["run_meta_sha256"],
        })
        coverage = metadata.get("formal_action_length_coverage")
        if not isinstance(coverage, Mapping) or \
                not isinstance(coverage.get("complete"), bool) or \
                not isinstance(coverage.get("shortfall"), Mapping):
            raise ValueError(
                "scene transaction formal action coverage is invalid")
        complete = complete and coverage["complete"]
        length_shortfall.update(str(value) for value in coverage["shortfall"])
        expected_terminal = "completed" if coverage["complete"] else "partial"
        if finalization["status"] != expected_terminal:
            raise ValueError(
                "scene transaction finalization disagrees with coverage")
    if complete and int(returncode) in {0, RECOVERED_UNKNOWN_RETURNCODE}:
        status = "completed"
    elif not complete:
        status = "partial_valid"
    else:
        status = "failed"
    return {
        "catalog_status": status,
        "terminal_status": (
            "recovered_valid"
            if status == "completed" and
            int(returncode) == RECOVERED_UNKNOWN_RETURNCODE else None),
        "source_validated_records": total,
        "length_shortfall": sorted(length_shortfall),
        "record_paths": [str(path) for path in paths],
        "sources": source_identities,
    }


def compile_global(
        manifest: dict, checkpoint, *, state: Mapping) -> dict:
    """Compile one corpus-global artifact, then compute dataset quota views."""
    checkpoint_id = (
        f"round-{int(checkpoint):02d}"
        if isinstance(checkpoint, int) else str(checkpoint))
    if not checkpoint_id or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in checkpoint_id):
        raise ValueError("background compile checkpoint is invalid")
    sources = [
        row for dataset in DATASETS
        for row in _reusable_dataset_sources(manifest, dataset, state)
    ]
    records = [Path(row["path"]) for row in sources]
    if not records:
        raise ValueError("global background corpus has no durable records")
    for path in records:
        rows = io_utils.read_jsonl(path, require_dict=True)
        if not rows or any(
                not a1_common_support.has_uniform_v3_ordinary_provenance(
                    record)
                for record in rows):
            raise ValueError(
                "global background corpus requires one uniform proposal-v3 "
                "policy")
    metadata_digests = []
    for path, source in zip(records, sources):
        metadata = path.with_name("run_meta.json")
        if not metadata.is_file():
            raise ValueError(f"run metadata is missing beside {path}")
        metadata_digests.append(source["run_meta_sha256"])
    authority = gate_authority.resolve_preview_source_authority(
        records_paths=records,
        expected_records_sha256=[
            source["records_sha256"] for source in sources],
        expected_run_meta_sha256=metadata_digests,
        authority_kind=gate_authority.CLI_EXTERNAL_AUTHORITY_KIND,
        authority_id=(
            f"background:{manifest['sha256']}:global:{checkpoint_id}"))
    artifact_parent = (
        Path(manifest["output_root"]) / "artifacts" / "global" /
        checkpoint_id)
    if artifact_parent.exists():
        summary = background_checkpoint.load(
            manifest, checkpoint_id)
        return {
            **dict(summary["result"]),
            "checkpoint_summary": {
                "path": str(artifact_parent /
                            background_checkpoint.SUMMARY_NAME),
                "sha256": io_utils.sha256_file(
                    artifact_parent / background_checkpoint.SUMMARY_NAME),
                "content_sha256": summary["sha256"],
            },
        }
    final_parent, staging = background_checkpoint.prepare(
        manifest, checkpoint_id)
    artifact = staging / "candidate_qa"
    report_root = staging / "candidate_qa_report"
    result = candidate_preview.build_main_preview(
        records, artifact, report_root,
        expected_source_authority=authority,
        run_meta_sha256=metadata_digests)
    expected_policy = {"A1_collision": a1_common_support.V3_POLICY}
    report = gate_authority.load_authority_manifest(artifact / "report.json")
    source_map = gate_authority.load_authority_manifest(
        artifact / "private" / "source_map.json")
    if (report.get("publication_selection") != expected_policy or
            source_map.get("publication_selection") != expected_policy):
        raise ValueError(
            "global background artifact lacks proposal-v3 publication policy")
    replay = candidate_preview.evaluate_preview_artifact(
        artifact, gt_as_pred=True, expected_source_authority=authority,
        _validated_benchmark=result)
    if replay["item_scores"] and \
            any(value != 1.0 for value in replay["item_scores"].values()):
        raise ValueError("GT replay is not exact")
    quota_reports = {}
    partitions = scene_partitions.load()
    artifact_index = candidate_quota.load_artifact_index(artifact)
    for dataset in DATASETS:
        quota = manifest["quota"][dataset]
        families = {
            scene_id: row["family"]
            for scene_id, row in partitions.rows[dataset].items()}
        quota_reports[dataset] = candidate_quota.summarize_artifact(
            artifact, dataset=dataset,
            supported_tasks=quota["supported_tasks"],
            min_total_items=quota["min_total_items"],
            min_per_task=quota["min_per_task"],
            required_lengths=quota["required_lengths"],
            min_per_length=quota["min_per_length"],
            min_unique_frames=quota["min_unique_frames"],
            min_scene_families=quota["min_scene_families"],
            max_scene_family_fraction=quota[
                "max_scene_family_fraction"],
            scene_families=families, artifact_index=artifact_index)
    macro = six_task_macro_report({
        task_id: replay["by_task"].get(task_id)
        for task_id in SUPPORTED_TASKS
    })
    if macro["macro"] != replay.get("six_task_macro"):
        raise ValueError("six-task macro implementations disagree")
    dataset_macros = dataset_macro_reports(
        artifact, replay["item_scores"], artifact_index=artifact_index)
    for dataset, quota_report in quota_reports.items():
        io_utils.atomic_write_json(
            staging / f"quota-{dataset}.json", quota_report,
            allow_nan=False, durable=True)
        io_utils.atomic_write_json(
            staging / f"six_task_macro-{dataset}.json",
            dataset_macros[dataset], allow_nan=False, durable=True)
    io_utils.atomic_write_json(
        staging / "gt_replay.json", replay,
        allow_nan=False, durable=True)
    io_utils.atomic_write_json(
        staging / "six_task_macro.json", macro,
        allow_nan=False, durable=True)
    compiled = {
        "artifact": str(final_parent / "candidate_qa"),
        "coverage": result["coverage"],
        "gt_as_pred": replay["overall"],
        "six_task_macro": macro,
        "datasets": {
            dataset: {
                "quota": quota_report,
                "six_task_macro": dataset_macros[dataset],
            }
            for dataset, quota_report in quota_reports.items()
        },
    }
    identity = background_checkpoint.publish(
        manifest, checkpoint_id, staging, result=compiled,
        sources=sources,
        result_files=[
            *[f"quota-{dataset}.json" for dataset in DATASETS],
            *[f"six_task_macro-{dataset}.json" for dataset in DATASETS],
            "gt_replay.json", "six_task_macro.json",
        ])
    return {**compiled, "checkpoint_summary": identity}


def terminate_process_group(pid: int, requested_signal=signal.SIGINT) -> None:
    """Signal the collector process group at one durable escalation stage."""
    os.killpg(int(pid), int(requested_signal))


def retryable_failure(log_path, *, failure_status: str = None) -> bool:
    if str(failure_status or "") == "initialization_slow":
        return True
    path = Path(log_path)
    if not path.is_file():
        return False
    tail = path.read_text(encoding="utf-8", errors="replace")[-200_000:].lower()
    return any(token in tail for token in (
        "cuda out of memory", "outofmemoryerror", "failed to initialize",
        "failed to create cuda", "carb.gym.plugin"))
