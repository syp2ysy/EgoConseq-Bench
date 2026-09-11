"""Deterministic construction of background collection jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from pipeline import action_proposal, config, dataset_contracts
from pipeline.background_lifecycle import b1k_supervisor_timeout_s


_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


def collection_seed(seed_base: int, round_index: int, gpu_id: int) -> int:
    """Return the deterministic seed bound to one collection transaction."""
    return int(seed_base) + int(round_index) * 101 + int(gpu_id)


def _common_collect_args(
        *, scenes: Sequence[str], output_dir: Path, revision: str,
        shard_id: str, poses_per_scene: int,
        pose_candidates_per_scene: int, seed: int,
        scene_wallclock_s: int, ordinary_actions_per_pose: int,
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


def _job_value(
        *, job_id: str, dataset: str, gpu_id: int, round_index: int,
        scenes: Sequence[str], output_dir: Path, output_root: Path,
        command: Sequence[str], extra_environment: Mapping[str, str] = None,
        isolate_cuda: bool = True, pose_exclusions: Path = None) -> dict:
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


def direct_job(
        *, dataset: str, gpu_id: int, round_index: int,
        scenes: Sequence[str], output_root: Path, revision: str, paths: dict,
        poses_per_scene: int, pose_candidates_per_scene: int,
        scene_wallclock_s: int, ordinary_actions_per_pose: int,
        pose_exclusions: Path = None,
        seed_base: int = config.BACKGROUND_COLLECTION_SEED_BASE) -> dict:
    """Build one R2R or GS collection job."""
    job_id = f"{dataset}-r{round_index:02d}-g{gpu_id}"
    output_dir = output_root / "records" / dataset / job_id
    command = [
        paths["python"], str(Path(paths["repository"]) / "scripts" /
                             "collect.py"),
        "--backend", dataset,
        *_common_collect_args(
            scenes=scenes, output_dir=output_dir, revision=revision,
            shard_id=job_id, poses_per_scene=poses_per_scene,
            pose_candidates_per_scene=pose_candidates_per_scene,
            scene_wallclock_s=scene_wallclock_s,
            ordinary_actions_per_pose=ordinary_actions_per_pose,
            seed=collection_seed(seed_base, round_index, gpu_id),
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
        ])
    else:
        raise ValueError(f"unsupported direct dataset {dataset!r}")
    return _job_value(
        job_id=job_id, dataset=dataset, gpu_id=gpu_id,
        round_index=round_index, scenes=scenes, output_dir=output_dir,
        output_root=output_root, command=command,
        pose_exclusions=pose_exclusions)


def b1k_job(
        *, gpu_id: int, round_index: int, scenes: Sequence[str],
        output_root: Path, revision: str, paths: dict,
        poses_per_scene: int, pose_candidates_per_scene: int,
        scene_wallclock_s: int, ordinary_actions_per_pose: int,
        pose_exclusions: Path = None,
        seed_base: int = config.BACKGROUND_COLLECTION_SEED_BASE) -> dict:
    """Build one B1K supervisor job."""
    dataset = "b1k"
    job_id = f"b1k-r{round_index:02d}-g{gpu_id}"
    output_dir = output_root / "records" / dataset / job_id
    child_args = _common_collect_args(
        scenes=scenes, output_dir=output_dir, revision=revision,
        shard_id=job_id, poses_per_scene=poses_per_scene,
        pose_candidates_per_scene=pose_candidates_per_scene,
        scene_wallclock_s=scene_wallclock_s,
        ordinary_actions_per_pose=ordinary_actions_per_pose,
        seed=collection_seed(seed_base, round_index, gpu_id),
        pose_exclusions=pose_exclusions)
    protected = {
        "--scenes", "--collection-shard-id", "--out", "--code-revision",
        "--resume",
    }
    forwarded = []
    index = 0
    while index < len(child_args):
        option = child_args[index]
        if option in protected:
            if option == "--resume":
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
        output_root=output_root, command=command, isolate_cuda=False,
        pose_exclusions=pose_exclusions,
        extra_environment={
            "OMNIGIBSON_DATA_PATH": paths["b1k_data_root"],
            "OMNIGIBSON_APPDATA_PATH": str(
                Path(paths["b1k_data_root"]) / "appdata"),
            "OMNIGIBSON_HEADLESS": "True",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "OMNIGIBSON_GPU_ID": str(gpu_id),
        })


def bind_job_transaction(
        job: dict, *, revision: str, paths: Mapping[str, str],
        source_manifest_sha256: Mapping[str, str],
        poses_per_scene: int, pose_candidates_per_scene: int,
        ordinary_actions_per_pose: int,
        seed_base: int = config.BACKGROUND_COLLECTION_SEED_BASE) -> None:
    """Bind one job to its source and immutable collection parameters."""
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
        "ordinary_actions_per_pose": int(ordinary_actions_per_pose),
        "scene_wallclock_stop_s": int(job["scene_wallclock_s"]),
        "supervisor_timeout_s": (
            b1k_supervisor_timeout_s(job["scene_wallclock_s"])
            if dataset == "b1k" else None),
        "seed": collection_seed(
            seed_base, job["round_index"], job["gpu_id"]),
        "pose_exclusions_path": job.get("pose_exclusions_path"),
        "source_authority": {
            "path": str(paths[source_path_key]),
            "sha256": source_manifest_sha256[dataset],
        },
    }


def collection_job(
        *, dataset: str, scene_id: str, scene_index: int,
        catalog_pass: int, round_index: int, output_root: Path,
        revision: str, paths: Mapping[str, str], profile: Mapping,
        source_manifest_sha256: Mapping[str, str],
        seed_pose_exclusions: Mapping | None,
        seed_base: int = config.BACKGROUND_COLLECTION_SEED_BASE) -> dict:
    """Build one manifest-bound scene transaction."""
    capacity = profile["datasets"][dataset]
    gpu_id = dataset_contracts.collection_gpu_id(dataset)
    pose_exclusions = (
        Path(output_root) / "controller" / "pose_exclusions" / dataset /
        f"{scene_id}-pass{int(catalog_pass):02d}.json"
        if int(catalog_pass) > 0 or seed_pose_exclusions is not None
        else None)
    common = {
        "gpu_id": gpu_id,
        "round_index": int(round_index),
        "scenes": [scene_id],
        "output_root": Path(output_root),
        "revision": revision,
        "paths": dict(paths),
        "poses_per_scene": capacity["records_per_scene"],
        "pose_candidates_per_scene": capacity["pose_attempt_cap"],
        "ordinary_actions_per_pose": int(profile["ordinary_actions_per_pose"]),
        "pose_exclusions": pose_exclusions,
        "seed_base": int(seed_base),
    }
    job = (
        b1k_job(**common, scene_wallclock_s=capacity["scene_wallclock_s"])
        if dataset == "b1k" else
        direct_job(
            dataset=dataset, **common,
            scene_wallclock_s=capacity["scene_wallclock_s"]))
    job.update({
        "catalog_pass": int(catalog_pass),
        "scene_index": int(scene_index),
        "weight": int(capacity["scene_wallclock_s"]),
        "scene_wallclock_s": int(capacity["scene_wallclock_s"]),
    })
    bind_job_transaction(
        job, revision=revision, paths=paths,
        source_manifest_sha256=source_manifest_sha256,
        poses_per_scene=capacity["records_per_scene"],
        pose_candidates_per_scene=capacity["pose_attempt_cap"],
        ordinary_actions_per_pose=int(profile["ordinary_actions_per_pose"]),
        seed_base=seed_base)
    return job
