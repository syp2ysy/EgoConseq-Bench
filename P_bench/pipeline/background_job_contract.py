"""Immutable, parser-equivalent command bindings for controller jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from pipeline import action_proposal, config


def _common(binding: Mapping) -> list[str]:
    values = [
        "--poses-per-scene", str(binding.get("poses_per_scene")),
        "--pose-candidates-per-scene",
        str(binding.get("pose_candidates_per_scene")),
        "--record-idle-stop-s", str(binding.get("record_idle_stop_s")),
        "--scene-wallclock-stop-s",
        str(binding.get("scene_wallclock_stop_s")),
        "--benchmark-partition",
        str(binding.get("benchmark_partition")),
        "--seed", str(binding.get("seed")),
        "--radii", *[str(value) for value in config.RADII_M],
        "--camera-heights", *[
            str(value) for value in config.BENCH_CAMERA_HEIGHTS_M],
        "--lengths", *[str(value) for value in config.GEN_LENGTHS],
        "--proposal-pairs-per-length",
        str(action_proposal.PAIRS_PER_LENGTH_DEFAULT),
        "--proposal-natural-per-length",
        str(action_proposal.NATURAL_PER_LENGTH_DEFAULT),
        "--ordinary-actions-per-pose",
        str(binding.get("ordinary_actions_per_pose")),
        "--keep-per-length", str(config.KEEP_PER_LENGTH),
        "--action-mode", "balanced",
    ]
    if binding.get("pose_exclusions_path") is not None:
        values.extend([
            "--pose-exclusions", str(binding["pose_exclusions_path"])])
    return values


def _reject_duplicate_options(command: Sequence[str]) -> None:
    seen = set()
    for token in command:
        value = str(token)
        if not value.startswith("--"):
            continue
        if value in seen:
            raise ValueError(
                f"background collection duplicate protected option: {value}")
        seen.add(value)


def validate_job_command(job: Mapping, paths: Mapping[str, str]) -> None:
    """Require the full argv to equal the immutable canonical job argv."""
    binding = job.get("transaction_binding") or {}
    command = [str(value) for value in (job.get("command") or [])]
    dataset = str(job.get("dataset") or "")
    scene_id = str(binding.get("scene_id") or "")
    repository = Path(paths["repository"])
    common = _common(binding)
    if dataset == "b1k":
        expected = [
            paths["b1k_python"], str(
                repository / "scripts" / "run_b1k_collection_shard.py"),
            "run",
            "--data-root", paths["b1k_data_root"],
            "--source-manifest",
            str((binding.get("source_authority") or {}).get("path")),
            "--output-dir", str(binding.get("output_dir")),
            "--gpu-id", str(job.get("gpu_id")),
            "--shard-id", str(binding.get("controller_job_id")),
            "--code-revision", str(binding.get("revision")),
            "--resume",
            "--scene-timeout-s", str(binding.get("supervisor_timeout_s")),
            "--scenes", scene_id,
            "--collect-args", *common,
        ]
    else:
        expected = [
            paths["python"], str(repository / "scripts" / "collect.py"),
            "--backend", dataset,
            "--scenes", scene_id,
            "--poses-per-scene", str(binding.get("poses_per_scene")),
            "--pose-candidates-per-scene",
            str(binding.get("pose_candidates_per_scene")),
            "--record-idle-stop-s", str(binding.get("record_idle_stop_s")),
            "--scene-wallclock-stop-s",
            str(binding.get("scene_wallclock_stop_s")),
            "--collection-shard-id", str(binding.get("collection_shard_id")),
            *common[8:],
            "--out", str(binding.get("output_dir")),
            "--code-revision", str(binding.get("revision")),
            "--resume",
        ]
        if dataset == "r2r":
            expected.extend([
                "--r2r-train-episodes", paths["r2r_train_episodes"],
                "--mp3d-root", paths["mp3d_root"],
                "--semantic-query-workers", "8",
            ])
        elif dataset == "gs":
            expected.extend([
                "--gs-data-root", paths["gs_data_root"],
                "--gs-source-manifest", paths["gs_source_manifest"],
            ])
    _reject_duplicate_options(command)
    if command != expected:
        raise ValueError(
            "background collection job command differs from exact contract")
