"""Derive revisit pose exclusions from authenticated prior scene records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from pipeline import io_utils


_REUSABLE = frozenset({"completed", "partial_valid"})
_DATASETS = ("r2r", "gs", "b1k")


def _pose(record: Mapping, scene_id: str) -> dict | None:
    if str(record.get("scene_id") or "") != scene_id:
        return None
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        raise ValueError("background revisit record pose is absent")
    position = pose.get("position")
    yaw = pose.get("yaw_rad")
    if (not isinstance(position, list) or len(position) != 3 or
            isinstance(yaw, bool)):
        raise ValueError("background revisit record pose is invalid")
    values = [float(value) for value in position]
    yaw_value = float(yaw)
    if not all(math.isfinite(value) for value in [*values, yaw_value]):
        raise ValueError("background revisit record pose is non-finite")
    return {"position": values, "yaw_rad": yaw_value}


def _seed_poses(
        manifest: Mapping, dataset: str, scene_id: str) -> list[dict]:
    identities = manifest.get("seed_pose_exclusions")
    if identities is None:
        return []
    if not isinstance(identities, Mapping):
        raise ValueError("background seed pose exclusions are invalid")
    identity = identities.get(dataset)
    if not isinstance(identity, Mapping):
        raise ValueError(
            f"background {dataset} seed pose exclusion identity is invalid")
    path = Path(str(identity.get("path") or ""))
    if (not path.is_file() or
            io_utils.sha256_file(path) != identity.get("sha256")):
        raise ValueError("background seed pose exclusion identity differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, Mapping) or
            value.get("schema_version") !=
            "egoconseq.pose_exclusions.v1" or
            not isinstance(value.get("scenes"), Mapping)):
        raise ValueError("background seed pose exclusion schema is invalid")
    rows = value["scenes"].get(scene_id) or []
    if not isinstance(rows, list):
        raise ValueError("background seed scene poses are invalid")
    poses = []
    for row in rows:
        pose = _pose({"scene_id": scene_id, "pose": row}, scene_id)
        if pose is None:  # pragma: no cover - scene is fixed above
            raise AssertionError("seed pose scene binding was lost")
        poses.append(pose)
    return poses


def build_seed_from_checkpoint(
        checkpoint_path: Path, *, expected_sha256: str,
        output_dir: Path) -> dict[str, dict]:
    """Build one deterministic pose-only seed file per dataset."""
    checkpoint = Path(checkpoint_path).resolve()
    if (not checkpoint.is_file() or
            io_utils.sha256_file(checkpoint) != str(expected_sha256)):
        raise ValueError("background baseline checkpoint identity differs")
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("background baseline checkpoint sources are invalid")
    sources = value.get("sources")
    if (value.get("schema") !=
            "egoconseq.background-compile-checkpoint.v1" or
            not isinstance(sources, list) or not sources):
        raise ValueError("background baseline checkpoint sources are invalid")

    grouped = {dataset: {} for dataset in _DATASETS}
    record_counts = {dataset: 0 for dataset in _DATASETS}
    for source in sources:
        path = Path(str((source or {}).get("path") or ""))
        if (not path.is_file() or io_utils.sha256_file(path) !=
                (source or {}).get("records_sha256")):
            raise ValueError("background baseline records identity differs")
        for record in io_utils.read_jsonl(path, require_dict=True):
            dataset = str((record.get("source") or {}).get(
                "source_dataset") or "")
            if dataset not in grouped:
                raise ValueError("background baseline record dataset is invalid")
            record_counts[dataset] += 1
            scene_id = str(record.get("scene_id") or "")
            pose = _pose(record, scene_id)
            if not scene_id or pose is None:
                raise ValueError("background baseline record pose is invalid")
            grouped[dataset].setdefault(scene_id, set()).add((
                *pose["position"], pose["yaw_rad"]))

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identities = {}
    for dataset in _DATASETS:
        scenes = {}
        for scene_id, poses in sorted(grouped[dataset].items()):
            scenes[scene_id] = [
                {"position": list(pose[:3]), "yaw_rad": pose[3]}
                for pose in sorted(poses)]
        path = root / f"{dataset}.json"
        io_utils.atomic_write_json(path, {
            "schema_version": "egoconseq.pose_exclusions.v1",
            "scenes": scenes,
        }, allow_nan=False, durable=True)
        identities[dataset] = {
            "path": str(path),
            "sha256": io_utils.sha256_file(path),
            "representative_count": sum(map(len, scenes.values())),
            "record_count": record_counts[dataset],
        }
    return identities


def materialize_for_job(
        manifest: Mapping, state: Mapping, job: Mapping) -> Path | None:
    """Write seed plus authenticated prior poses for one transaction."""
    destination = job.get("pose_exclusions_path")
    catalog_pass = int(job.get("catalog_pass") or 0)
    if destination is None:
        if catalog_pass > 0:
            raise ValueError("revisit transaction lacks pose exclusions")
        return None
    dataset = str(job["dataset"])
    scene_id = str(job["scenes"][0])
    runtime = state.get("jobs") or {}
    seed_poses = _seed_poses(manifest, dataset, scene_id)
    poses = []
    seen = set()
    for pose in seed_poses:
        key = (*pose["position"], pose["yaw_rad"])
        if key not in seen:
            seen.add(key)
            poses.append(pose)
    definitions = {
        prior["job_id"]: prior
        for round_value in manifest.get("rounds") or []
        for prior in round_value.get("jobs") or []}
    prior_rows = []
    for job_id, row in runtime.items():
        definition = definitions.get(job_id) or {}
        prior_dataset = str(
            row.get("dataset") or definition.get("dataset") or "")
        prior_scene = str(
            row.get("scene_id") or
            ((definition.get("scenes") or [""])[0]))
        prior_pass = int(
            row.get("catalog_pass")
            if row.get("catalog_pass") is not None else
            definition.get("catalog_pass") or 0)
        if (prior_dataset == dataset and prior_scene == scene_id and
                prior_pass < catalog_pass):
            prior_rows.append((prior_pass, str(job_id), row))
    for _prior_pass, _job_id, row in sorted(prior_rows):
        if row.get("catalog_status") not in _REUSABLE:
            if int(row.get("durable_records") or 0) > 0:
                raise ValueError(
                    "background prior durable records are not reusable")
            continue
        sources = (row.get("source_validation") or {}).get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(
                "background revisit source identity is absent")
        for source in sources:
            path = Path(str(source.get("path") or ""))
            if (not path.is_file() or
                    io_utils.sha256_file(path) !=
                    source.get("records_sha256")):
                raise ValueError(
                    "background revisit source identity differs")
            source_rows = 0
            source_poses = 0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    source_rows += 1
                    value = _pose(json.loads(line), scene_id)
                    if value is None:
                        continue
                    source_poses += 1
                    key = (*value["position"], value["yaw_rad"])
                    if key in seen:
                        continue
                    seen.add(key)
                    poses.append(value)
            if source_rows and not source_poses:
                raise ValueError(
                    f"background authenticated source contains no pose "
                    f"for {scene_id}")
    path = Path(str(destination)).resolve()
    io_utils.atomic_write_json(path, {
        "schema_version": "egoconseq.pose_exclusions.v1",
        "scenes": {scene_id: poses},
    }, allow_nan=False, durable=True)
    return path
