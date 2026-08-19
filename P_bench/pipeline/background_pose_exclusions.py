"""Derive revisit pose exclusions from authenticated prior scene records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from pipeline import io_utils


_REUSABLE = frozenset({"completed", "partial_valid"})


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


def materialize_for_job(
        manifest: Mapping, state: Mapping, job: Mapping) -> Path | None:
    """Write the v1 exclusion file required by one revisit transaction."""
    destination = job.get("pose_exclusions_path")
    catalog_pass = int(job.get("catalog_pass") or 0)
    if catalog_pass == 0:
        if destination is not None:
            raise ValueError("first catalog pass must not bind pose exclusions")
        return None
    if not destination:
        raise ValueError("revisit transaction lacks pose exclusions")
    dataset = str(job["dataset"])
    scene_id = str(job["scenes"][0])
    runtime = state.get("jobs") or {}
    poses = []
    seen = set()
    for round_value in manifest.get("rounds") or []:
        for prior in round_value.get("jobs") or []:
            if (prior.get("dataset") != dataset or
                    prior.get("scenes") != [scene_id] or
                    int(prior.get("catalog_pass") or 0) >= catalog_pass):
                continue
            row = runtime.get(prior["job_id"]) or {}
            if row.get("catalog_status") not in _REUSABLE:
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
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        value = _pose(json.loads(line), scene_id)
                        if value is None:
                            continue
                        key = (*value["position"], value["yaw_rad"])
                        if key in seen:
                            continue
                        seen.add(key)
                        poses.append(value)
    path = Path(str(destination)).resolve()
    io_utils.atomic_write_json(path, {
        "schema_version": "egoconseq.pose_exclusions.v1",
        "scenes": {scene_id: poses},
    }, allow_nan=False, durable=True)
    return path
