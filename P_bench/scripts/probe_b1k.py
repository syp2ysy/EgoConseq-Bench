#!/usr/bin/env python3
"""Run B1K probes with one isolated OmniGibson child per scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import (
    b1k_probe, b1k_process, b1k_source_builder, config, io_utils,
)
from pipeline.scene_pool import discover_b1k_train_scenes


_PROBE_SCHEMA = "b1k-adapter-probe.v1"
_SUMMARY_SCHEMA = "b1k-adapter-probe-summary.v2"


def _write(path: Path, value: dict) -> None:
    io_utils.atomic_write_json(
        path, value, allow_nan=False, durable=True)


def _shutdown() -> None:
    from pipeline.b1k_sim import shutdown_b1k_runtime

    try:
        shutdown_b1k_runtime()
    except SystemExit as error:
        if error.code not in (None, 0):
            raise


def _run_isolated_probe_child(command, *, timeout_s: int) -> int:
    """Run one probe in its own terminable process group."""
    process = subprocess.Popen(list(command), start_new_session=True)
    return b1k_process.wait_isolated_process(
        process, timeout_s=timeout_s)


def _catalog(args):
    catalog = discover_b1k_train_scenes(
        args.data_root, args.source_manifest)
    requested = list(
        args.scenes or [scene.scene_id for scene in catalog])
    by_id = {scene.scene_id: scene for scene in catalog}
    if (len(requested) != len(set(requested)) or
            not set(requested).issubset(by_id)):
        raise ValueError("requested probe scenes are absent or duplicated")
    return requested, by_id


def _load_probe(path: Path, scene_id: str) -> dict:
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or
            value.get("schema") != _PROBE_SCHEMA or
            value.get("scene_id") != scene_id):
        raise ValueError(f"invalid B1K probe output: {path}")
    return value


def _fail_scene_child(
        path: Path, scene_id: str, *, stage: str, error: BaseException,
        hard_exit=None) -> int:
    """Persist a required probe failure before native runtime teardown."""
    status = 130 if isinstance(error, KeyboardInterrupt) else 1
    blocked = {
        "schema": _PROBE_SCHEMA,
        "scene_id": scene_id,
        "smoke_pass": False,
        "blocked": {
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    }
    _write(path, blocked)
    print(json.dumps(blocked["blocked"], sort_keys=True),
          file=sys.stderr, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    (os._exit if hard_exit is None else hard_exit)(status)
    return status


def _run_scene_child(args, *, hard_exit=None) -> int:
    """Load, probe, and clean up exactly one OG scene."""
    requested, by_id = _catalog(args)
    scene_id = str(args._child_scene)
    if requested != [scene_id] and scene_id not in requested:
        raise ValueError("probe child scene is absent from requested scope")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{scene_id}.json"

    from pipeline.b1k_sim import B1KSimSession

    try:
        session = B1KSimSession(
            by_id[scene_id], fovs=config.BENCH_FOVS_DEG)
    except BaseException as error:
        return _fail_scene_child(
            path, scene_id, stage="runtime_geometry_or_cspace",
            error=error, hard_exit=hard_exit)

    try:
        result = b1k_probe.run_adapter_probe(
            session, fovs=config.BENCH_FOVS_DEG,
            seed=int(args.seed),
            semantic_sample_count=int(args.semantic_sample_count))
    except b1k_probe.B1KProbeBlockingError as error:
        return _fail_scene_child(
            path, scene_id, stage="runtime_geometry_or_cspace",
            error=error, hard_exit=hard_exit)
    except KeyboardInterrupt as error:
        return _fail_scene_child(
            path, scene_id, stage="probe_interrupted",
            error=error, hard_exit=hard_exit)
    except Exception as error:
        result = {
            "schema": _PROBE_SCHEMA,
            "scene_id": scene_id,
            "scene_authority_sha256": session.scene_authority_sha256,
            "smoke_pass": False,
            "diagnostic_error": {
                "error_type": type(error).__name__,
                "error": str(error),
            },
            "semantic_resolution": {"resolution_rate": None},
            "collision_visual": {"status": "not_reached"},
        }
    try:
        # Sensor removal and scene clear precede process-wide app shutdown.
        session.close()
        _write(path, result)
        print(json.dumps({
            "scene_id": scene_id,
            "smoke_pass": result["smoke_pass"],
            "semantic_resolution_rate":
                result["semantic_resolution"]["resolution_rate"],
            "collision_visual_status": result["collision_visual"]["status"],
        }, sort_keys=True), flush=True)
        _shutdown()
    except BaseException as error:
        return _fail_scene_child(
            path, scene_id, stage="probe_cleanup",
            error=error, hard_exit=hard_exit)
    return 0


def _run_parent(args) -> int:
    """Supervise scene children without importing OmniGibson."""
    requested, _by_id = _catalog(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    timeout_s = int(args.scene_timeout_s)
    if timeout_s <= 0:
        raise ValueError("B1K probe scene timeout must be positive")
    attempts = []
    completed = []
    completed_results = {}

    def write_summary() -> None:
        failed = [
            {
                "scene_id": row["scene_id"],
                "returncode": row.get("returncode"),
                "error": row["error"],
            }
            for row in attempts if row["status"] == "failed"
        ]
        _write(output / "probe-summary.json", {
            "schema": _SUMMARY_SCHEMA,
            "requested_scene_ids": requested,
            "completed_scene_ids": completed,
            "smoke_pass_count": sum(
                result.get("smoke_pass") is True
                for result in completed_results.values()),
            "failed_scenes": failed,
            "attempts": attempts,
            "complete": completed == requested and not failed,
        })

    write_summary()
    for index, scene_id in enumerate(requested):
        command = [
            str(b1k_source_builder.BEHAVIOR_PYTHON),
            str(Path(__file__).resolve()),
            "--data-root", str(Path(args.data_root).resolve()),
            "--source-manifest", str(Path(args.source_manifest).resolve()),
            "--output-dir", str(output),
            "--scenes", scene_id,
            "--seed", str(int(args.seed) + index),
            "--semantic-sample-count", str(int(args.semantic_sample_count)),
            "--scene-timeout-s", str(timeout_s),
            "--_child-scene", scene_id,
        ]
        try:
            returncode = _run_isolated_probe_child(
                command, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            attempts.append({
                "scene_id": scene_id, "status": "failed",
                "returncode": None,
                "error": f"TimeoutExpired: scene exceeded {timeout_s} seconds",
            })
            write_summary()
            continue
        path = output / f"{scene_id}.json"
        try:
            result = _load_probe(path, scene_id)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            attempts.append({
                "scene_id": scene_id, "status": "failed",
                "returncode": int(returncode),
                "error": ("probe child returned nonzero" if returncode else
                          f"{type(error).__name__}: {error}"),
            })
            write_summary()
            continue
        if returncode != 0 or result.get("blocked"):
            attempts.append({
                "scene_id": scene_id, "status": "failed",
                "returncode": int(returncode),
                "error": "runtime geometry or C-space extraction blocked",
            })
            write_summary()
            continue
        completed.append(scene_id)
        completed_results[scene_id] = result
        attempts.append({
            "scene_id": scene_id, "status": "completed",
            "returncode": int(returncode),
        })
        write_summary()
    write_summary()
    return int(completed != requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--semantic-sample-count", type=int, default=256)
    parser.add_argument(
        "--scene-timeout-s", type=int,
        default=b1k_source_builder.DEFAULT_SCENE_TIMEOUT_S)
    parser.add_argument("--_child-scene", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args._child_scene:
        return _run_scene_child(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
