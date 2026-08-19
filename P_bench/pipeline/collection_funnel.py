"""Crash-persistent collection-funnel reporting.

The collector's records are transactional, but a zero-yield or interrupted
shard still needs an independently auditable output. This module owns that
small deterministic report. It has no simulator dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from pipeline.io_utils import atomic_write_json


FUNNEL_SCHEMA_VERSION = "egoconseq.collection-funnel.v3"
_POSE_CHANNELS = {
    "accepted": (
        "_scene_positions",
        "position_heading_distribution",
        "accepted_poses",
    ),
    "searched": (
        "_scene_searched_positions",
        "searched_position_heading_distribution",
        "searched_poses",
    ),
}


def canonical_sha256(value) -> str:
    """Hash JSON-compatible data with stable key and whitespace rules."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_CONFIG_SHA256_EXECUTION_ONLY_FIELDS = frozenset({
    "A3_FACE_AABB_PRUNE_SLACK_M",
    "MP3D_AABB_MADVISE_INTERVAL_CHUNKS",
    "SEMANTIC_ASSIGN_PARALLEL_MIN_POINTS",
})


def _hashable_config_value(name: str, value):
    """Return a JSON form of one constant, or raise if it has none.

    Silently dropping a constant the encoder cannot take is the dangerous
    failure: the category frozensets decide A3 contact attribution and B
    target exclusion, so a dropped one lets their semantics change while the
    digest -- and therefore every collection reuse guard bound to it -- stays
    put. Sets are ordered here; anything still unencodable is an error.
    """
    if isinstance(value, (set, frozenset)):
        value = sorted(value)
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"config constant {name} is not hashable by config_sha256; add it "
            f"to _CONFIG_SHA256_EXECUTION_ONLY_FIELDS only if it cannot "
            f"change collection output") from error
    return value


def config_sha256(config_module) -> str:
    """Hash public constant values that can affect collection behaviour."""
    values = {}
    for name in sorted(dir(config_module)):
        if (not name.isupper() or
                name in _CONFIG_SHA256_EXECUTION_ONLY_FIELDS):
            continue
        values[name] = _hashable_config_value(
            name, getattr(config_module, name))
    return canonical_sha256(values)


def _empty_pose_state(count_key: str) -> dict:
    return {"cells": set(), "headings": {}, count_key: 0}


def _pose_distribution(state: dict, count_key: str) -> dict:
    return {
        count_key: int(state[count_key]),
        "position_cells": sorted(state["cells"]),
        "heading_bins_deg": dict(sorted(state["headings"].items())),
    }


def _counter_delta(current: Mapping, baseline: Mapping) -> dict:
    keys = set(current) | set(baseline)
    return {
        str(key): int(current.get(key, 0)) - int(baseline.get(key, 0))
        for key in sorted(keys)
        if int(current.get(key, 0)) - int(baseline.get(key, 0))
    }


def _counter_sum(left: Mapping, right: Mapping) -> dict:
    keys = set(left) | set(right)
    return {
        str(key): int(left.get(key, 0)) + int(right.get(key, 0))
        for key in sorted(keys)
        if int(left.get(key, 0)) + int(right.get(key, 0))
    }


def _merge_pose_distribution(previous: Mapping, current: Mapping, count_key):
    return {
        count_key: int(previous.get(count_key, 0)) + int(current[count_key]),
        "position_cells": sorted(
            set(previous.get("position_cells", [])) |
            set(current["position_cells"])),
        "heading_bins_deg": _counter_sum(
            previous.get("heading_bins_deg") or {},
            current["heading_bins_deg"]),
    }


def _aggregate_pose_distribution(rows, field, count_key):
    cells, headings, count = set(), {}, 0
    for row in rows:
        distribution = row.get(field) or {}
        count += int(distribution.get(count_key, 0))
        cells.update(map(str, distribution.get("position_cells", [])))
        headings = _counter_sum(
            headings, distribution.get("heading_bins_deg") or {})
    return {
        count_key: count,
        "distinct_position_cells": len(cells),
        "position_cells": sorted(cells),
        "heading_bins_deg": headings,
    }


class CollectionFunnel:
    """Maintain one self-describing funnel report with per-scene checkpoints."""
    def __init__(
            self, path, *, record_schema_version: str,
            oracle_contract_version: str, code_revision: str,
            code_dirty: bool, config_sha256: str, run_contract: dict,
            backend: str, mode: str):
        self.path = Path(path)
        self._run_contract_sha256 = canonical_sha256(run_contract)
        self._scene_baselines = {}
        for state_attr, _field, _count_key in _POSE_CHANNELS.values():
            setattr(self, state_attr, {})
        if self.path.is_file():
            existing = json.loads(self.path.read_text())
            expected = {
                "schema_version": FUNNEL_SCHEMA_VERSION,
                "record_schema_version": str(record_schema_version),
                "oracle_contract_version": str(oracle_contract_version),
                "code_revision": code_revision,
                "code_dirty": bool(code_dirty),
                "config_sha256": str(config_sha256),
                "backend": str(backend),
                "collection_mode": str(mode),
                "run_contract_sha256": self._run_contract_sha256,
            }
            for field, expected_value in expected.items():
                if existing.get(field) != expected_value:
                    raise ValueError(
                        f"collection funnel {field} differs from the "
                        "requested resume")
            self.value = existing
            self.value["status"] = "running"
            self.value.pop("interruption_reason", None)
        else:
            self.value = {
                "schema_version": FUNNEL_SCHEMA_VERSION,
                "record_schema_version": str(record_schema_version),
                "oracle_contract_version": str(oracle_contract_version),
                "code_revision": code_revision,
                "code_dirty": bool(code_dirty),
                "config_sha256": str(config_sha256),
                "run_contract_sha256": self._run_contract_sha256,
                "backend": str(backend),
                "collection_mode": str(mode),
                "status": "running",
                "scene_counts": {
                    "total": 0,
                    "completed": 0,
                    "interrupted": 0,
                    "failed": 0,
                },
                "stage_counts": {},
                "skip_reasons": {},
                **{
                    field: {
                        count_key: 0,
                        "distinct_position_cells": 0,
                        "position_cells": [],
                        "heading_bins_deg": {},
                    }
                    for _state_attr, field, count_key
                    in _POSE_CHANNELS.values()
                },
                "per_scene": [],
            }
        self._write()
    def restore_counters(self, stats, skipped) -> None:
        """Restore persisted aggregate counters before a resumed scene loop.
        Assignment, never Counter addition: the collector rebuilds its
        deterministic action pools (``pool_L*``) before
        this call, so persisted keys must overwrite the live value while
        process-only keys survive. Restoring twice is therefore a no-op.
        """
        for key, value in (self.value.get("stage_counts") or {}).items():
            stats[str(key)] = int(value)
        for key, value in (self.value.get("skip_reasons") or {}).items():
            skipped[str(key)] = int(value)
    def _write(self) -> None:
        # Durable: the funnel is the crash audit trail for fsync-backed
        # records.jsonl, so it must never lag those records after power loss.
        atomic_write_json(self.path, self.value, allow_nan=False, durable=True)
    def begin_scene(self, scene_id: str, stats: Mapping, skipped: Mapping) -> None:
        self._scene_baselines[str(scene_id)] = (
            dict(stats), dict(skipped))
        for state_attr, _field, count_key in _POSE_CHANNELS.values():
            getattr(self, state_attr)[str(scene_id)] = \
                _empty_pose_state(count_key)
    @staticmethod
    def _record_pose_distribution(
            scene: dict, position, yaw_rad: float, count_key: str) -> None:
        x, _y, z = (float(value) for value in position)
        cell = f"{math.floor(x):d}:{math.floor(z):d}"
        yaw_deg = math.degrees(float(yaw_rad)) % 360.0
        heading = int(round(yaw_deg / 45.0) * 45) % 360
        heading_key = str(heading)
        scene["cells"].add(cell)
        scene["headings"][heading_key] = \
            int(scene["headings"].get(heading_key, 0)) + 1
        scene[count_key] += 1
    def _record_pose(
            self, channel: str, scene_id: str, position,
            yaw_rad: float) -> None:
        state_attr, _field, count_key = _POSE_CHANNELS[channel]
        states = getattr(self, state_attr)
        scene = states.setdefault(str(scene_id), _empty_pose_state(count_key))
        self._record_pose_distribution(
            scene, position, yaw_rad, count_key)
    def record_pose(self, scene_id: str, position, yaw_rad: float) -> None:
        """Record persisted coverage only; raw physical poses stay private."""
        self._record_pose("accepted", scene_id, position, yaw_rad)
    def record_searched_pose(
            self, scene_id: str, position, yaw_rad: float) -> None:
        """Record coarse coverage of poses that reached witness search."""
        self._record_pose("searched", scene_id, position, yaw_rad)
    def finish_scene(
            self, scene_id: str, stats: Mapping, skipped: Mapping, *,
            status: str) -> None:
        if status not in {"completed", "interrupted", "failed"}:
            raise ValueError(f"unsupported scene funnel status: {status}")
        scene_id = str(scene_id)
        baseline_stats, baseline_skipped = self._scene_baselines.pop(
            scene_id, ({}, {}))
        previous = next((
            value for value in self.value.get("per_scene", [])
            if str(value.get("scene_id")) == scene_id
        ), None)
        coverage = {}
        for state_attr, field, count_key in _POSE_CHANNELS.values():
            state = getattr(self, state_attr).pop(
                scene_id, _empty_pose_state(count_key))
            coverage[field] = _merge_pose_distribution(
                (previous or {}).get(field) or {},
                _pose_distribution(state, count_key),
                count_key,
            )
        row_stage_counts = _counter_sum(
            (previous or {}).get("stage_counts") or {},
            _counter_delta(stats, baseline_stats))
        row = {
            "scene_id": scene_id,
            "status": status,
            "stage_counts": row_stage_counts,
            "skip_reasons": _counter_sum(
                (previous or {}).get("skip_reasons") or {},
                _counter_delta(skipped, baseline_skipped)),
            **coverage,
        }
        rows = [
            value for value in self.value.get("per_scene", [])
            if str(value.get("scene_id")) != scene_id
        ]
        rows.append(row)
        self.value["per_scene"] = rows
        self.value["stage_counts"] = {
            str(key): int(value) for key, value in sorted(stats.items())
            if int(value)
        }
        self.value["skip_reasons"] = {
            str(key): int(value) for key, value in sorted(skipped.items())
            if int(value)
        }
        self._rebuild_coverage()
        self._write()
    def _rebuild_coverage(self) -> None:
        rows = self.value.get("per_scene", [])
        self.value["scene_counts"] = {
            "total": len(rows),
            "completed": sum(row.get("status") == "completed" for row in rows),
            "interrupted": sum(
                row.get("status") == "interrupted" for row in rows),
            "failed": sum(row.get("status") == "failed" for row in rows),
        }
        for _state_attr, field, count_key in _POSE_CHANNELS.values():
            self.value[field] = _aggregate_pose_distribution(
                rows, field, count_key)
    def complete(self) -> None:
        self.value["status"] = "completed"
        self._write()
    @staticmethod
    def _mark_terminal_status(
            path, *, status: str, reason_field: str, reason: str) -> None:
        path = Path(path)
        if not path.is_file():
            return
        value = json.loads(path.read_text())
        value["status"] = status
        value[reason_field] = str(reason)
        atomic_write_json(path, value, allow_nan=False, durable=True)
    @staticmethod
    def mark_interrupted(path, *, reason: str) -> None:
        CollectionFunnel._mark_terminal_status(
            path, status="interrupted",
            reason_field="interruption_reason", reason=reason)
    @staticmethod
    def mark_failed(path, *, reason: str) -> None:
        CollectionFunnel._mark_terminal_status(
            path, status="failed",
            reason_field="failure_reason", reason=reason)
