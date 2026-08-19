"""Bootstrap manifest for authenticated background-capacity canaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from pipeline import (
    background_job_contract, config, dataset_contracts, gate_authority,
    io_utils,
)


SCHEMA = "egoconseq.capacity-canary-parent.v2"
SOURCE_AUTHORITY_SCHEMA = "egoconseq.capacity-canary-source-authority.v1"
DATASETS = dataset_contracts.main_collection_datasets()
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
        *, revision: str, output_root: Path,
        scene_catalog: Mapping[str, Sequence[str]],
        selected_scenes: Mapping[str, Sequence[str]], paths: Mapping[str, str],
        source_manifest_sha256: Mapping[str, str],
        ordinary_actions_per_pose: int,
        direct_job_builder: Callable, b1k_job_builder: Callable,
        bind_job: Callable) -> dict:
    """Build six normal one-scene jobs without requiring a capacity profile."""
    if _HEX40.fullmatch(str(revision or "")) is None:
        raise ValueError("capacity canary revision must be a clean SHA-1")
    if set(scene_catalog) != set(DATASETS) or \
            set(selected_scenes) != set(DATASETS):
        raise ValueError("capacity canary dataset catalogs are invalid")
    catalogs = {}
    selected = {}
    for dataset in DATASETS:
        catalogs[dataset] = [str(value) for value in scene_catalog[dataset]]
        selected[dataset] = [str(value) for value in selected_scenes[dataset]]
        if (not catalogs[dataset] or
                len(catalogs[dataset]) != len(set(catalogs[dataset])) or
                len(selected[dataset]) != 2 or
                len(set(selected[dataset])) != 2 or
                not set(selected[dataset]).issubset(catalogs[dataset])):
            raise ValueError(
                "capacity canary requires two distinct catalog scenes")
        if _HEX64.fullmatch(str(source_manifest_sha256.get(dataset) or "")) \
                is None:
            raise ValueError("capacity canary source digest is invalid")
    root = Path(output_root).resolve()
    ordinary_actions = int(ordinary_actions_per_pose)
    if ordinary_actions not in config.ACTION_CANDIDATE_ORDINARY_LADDER:
        raise ValueError("capacity canary action budget is invalid")
    attempt_caps = {
        dataset: int(config.BACKGROUND_CANARY_POSE_ATTEMPT_CAP_BY_DATASET[
            dataset])
        for dataset in DATASETS
    }
    jobs = []
    index = 0
    for dataset in DATASETS:
        for scene_id in selected[dataset]:
            common = {
                "gpu_id": index % 4, "round_index": index // 4,
                "scenes": [scene_id],
                "output_root": root, "revision": revision,
                "paths": dict(paths),
                "poses_per_scene": config.BACKGROUND_CANARY_ACCEPTED_RECORDS,
                "pose_candidates_per_scene": attempt_caps[dataset],
                "ordinary_actions_per_pose": ordinary_actions,
            }
            job = (
                b1k_job_builder(
                    **common,
                    scene_wallclock_s=
                    config.BACKGROUND_SCENE_WALLCLOCK_S_BY_DATASET[dataset])
                if dataset == "b1k" else
                direct_job_builder(
                    dataset=dataset, **common,
                    scene_wallclock_s=
                    config.BACKGROUND_SCENE_WALLCLOCK_S_BY_DATASET[dataset]))
            job["catalog_pass"] = 0
            job["canary_wave"] = index // 4
            job["scene_wallclock_s"] = \
                config.BACKGROUND_SCENE_WALLCLOCK_S_BY_DATASET[dataset]
            job["weight"] = \
                config.BACKGROUND_SCENE_WALLCLOCK_S_BY_DATASET[dataset]
            bind_job(
                job, revision=revision, paths=paths,
                source_manifest_sha256=source_manifest_sha256,
                poses_per_scene=config.BACKGROUND_CANARY_ACCEPTED_RECORDS,
                pose_candidates_per_scene=attempt_caps[dataset],
                ordinary_actions_per_pose=ordinary_actions)
            jobs.append(job)
            index += 1
    body = {
        "schema": SCHEMA,
        "revision": revision,
        "output_root": str(root),
        "paths": {str(key): str(value) for key, value in paths.items()},
        "scene_catalog": catalogs,
        "selected_scenes": selected,
        "source_manifest_sha256": dict(source_manifest_sha256),
        "accepted_records_target":
            config.BACKGROUND_CANARY_ACCEPTED_RECORDS,
        "pose_attempt_cap": attempt_caps,
        "ordinary_actions_per_pose": ordinary_actions,
        "jobs": jobs,
    }
    return {**body, "sha256": canonical_sha256(body)}


def validate_manifest(value: Mapping) -> None:
    """Validate the parent and every exact normal collection command."""
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError("capacity canary parent schema is invalid")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != canonical_sha256(body):
        raise ValueError("capacity canary parent digest is invalid")
    if (_HEX40.fullmatch(str(value.get("revision") or "")) is None or
            value.get("accepted_records_target") !=
            config.BACKGROUND_CANARY_ACCEPTED_RECORDS or
            value.get("ordinary_actions_per_pose") not in
            config.ACTION_CANDIDATE_ORDINARY_LADDER or
            value.get("pose_attempt_cap") != {
                dataset: int(
                    config.BACKGROUND_CANARY_POSE_ATTEMPT_CAP_BY_DATASET[
                        dataset])
                for dataset in DATASETS
            }):
        raise ValueError("capacity canary collection budget is invalid")
    catalogs = value.get("scene_catalog") or {}
    selected = value.get("selected_scenes") or {}
    sources = value.get("source_manifest_sha256") or {}
    if set(catalogs) != set(DATASETS) or set(selected) != set(DATASETS) or \
            set(sources) != set(DATASETS):
        raise ValueError("capacity canary catalogs are invalid")
    expected_pairs = set()
    for dataset in DATASETS:
        catalog = catalogs.get(dataset)
        scenes = selected.get(dataset)
        if (not isinstance(catalog, list) or not catalog or
                len(catalog) != len(set(catalog)) or
                not isinstance(scenes, list) or len(scenes) != 2 or
                len(set(scenes)) != 2 or not set(scenes).issubset(catalog) or
                _HEX64.fullmatch(str(sources.get(dataset) or "")) is None):
            raise ValueError("capacity canary catalog selection is invalid")
        expected_pairs.update((dataset, scene) for scene in scenes)
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 6:
        raise ValueError("capacity canary jobs are invalid")
    actual_pairs = set()
    job_ids = []
    for index, job in enumerate(jobs):
        binding = job.get("transaction_binding") or {}
        dataset = job.get("dataset")
        scene = (job.get("scenes") or [None])[0]
        actual_pairs.add((dataset, scene))
        job_ids.append(job.get("job_id"))
        if (len(job.get("scenes") or []) != 1 or
                job.get("gpu_id") != index % 4 or
                job.get("canary_wave") != index // 4 or
                job.get("round_index") != index // 4 or
                binding.get("dataset") != dataset or
                binding.get("scene_id") != scene or
                binding.get("revision") != value["revision"] or
                binding.get("poses_per_scene") !=
                config.BACKGROUND_CANARY_ACCEPTED_RECORDS or
                binding.get("pose_candidates_per_scene") !=
                config.BACKGROUND_CANARY_POSE_ATTEMPT_CAP_BY_DATASET[
                    dataset] or
                binding.get("ordinary_actions_per_pose") !=
                value["ordinary_actions_per_pose"] or
                    (binding.get("source_authority") or {}).get("sha256") !=
                    sources.get(dataset) or
                    job.get("scene_wallclock_s") !=
                    config.BACKGROUND_SCENE_WALLCLOCK_S_BY_DATASET[dataset]):
            raise ValueError("capacity canary job binding differs")
        background_job_contract.validate_job_command(job, value["paths"])
    if actual_pairs != expected_pairs or len(job_ids) != len(set(job_ids)):
        raise ValueError("capacity canary jobs differ from selection")


def load(path: Path) -> dict:
    value = gate_authority.load_authority_manifest(Path(path).resolve())
    validate_manifest(value)
    return value


def write_source_authority(path: Path, parent: Mapping,
                           source_rows: Sequence[Mapping]) -> dict:
    """Finalize a path-only authority consumable by shortcut audits."""
    root = Path(parent["output_root"]).resolve()
    inputs = []
    for source in source_rows:
        records = Path(str(source["path"])).resolve()
        try:
            shard = records.parent.relative_to(root)
        except ValueError as error:
            raise ValueError("capacity canary source escapes output root") \
                from error
        inputs.append({
            "shard": str(shard),
            "records.jsonl": str(source["records_sha256"]),
            "run_meta.json": str(source["run_meta_sha256"]),
        })
    body = {
        "schema": SOURCE_AUTHORITY_SCHEMA,
        "name": f"capacity-canary:{parent['sha256']}",
        "inputs": sorted(inputs, key=lambda row: row["shard"]),
    }
    value = {**body, "sha256": canonical_sha256(body)}
    io_utils.atomic_write_json(path, value, allow_nan=False, durable=True)
    return value


def validate_source_authority(path: Path, parent: Mapping) -> dict:
    """Reopen the finalized six-source authority and its canonical digest."""
    value = gate_authority.load_authority_manifest(path)
    body = {key: item for key, item in value.items() if key != "sha256"}
    inputs = value.get("inputs")
    if (value.get("schema") != SOURCE_AUTHORITY_SCHEMA or
            value.get("name") != f"capacity-canary:{parent['sha256']}" or
            value.get("sha256") != canonical_sha256(body) or
            not isinstance(inputs, list) or len(inputs) != 6 or
            len({str((row or {}).get("shard") or "")
                 for row in inputs}) != 6):
        raise ValueError("capacity canary source authority is invalid")
    return value
