#!/usr/bin/env python3
"""Run one B1K collection shard with process isolation per scene.

This controller never imports OmniGibson.  Each scene is delegated to one
``collect.py --backend b1k`` child and remains an independent source shard.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import (
    b1k_process, b1k_source_builder, collection_cli, collection_funnel,
    collection_closeout, collection_runtime, collection_support,
    io_utils,
)
from pipeline.scene_pool import B1K_SOURCE_MANIFEST_VERSION


ROOT = Path(__file__).resolve().parents[1]
_FUNNEL_SCHEMA = "egoconseq.collection-funnel.v3"
_PROGRESS_SCHEMA = "b1k-collection-shard-progress.v2"
_SUPERVISOR_CONTRACT_SCHEMA = "b1k-supervised-collection-contract.v1"
_WORKER_HEARTBEAT_INTERVAL_S = 30.0
_PROTECTED_COLLECT_OPTIONS = frozenset({
    "--allow-dirty-code",
    "--auto-scenes",
    "--backend",
    "--b1k-data-root",
    "--b1k-source-manifest",
    "--b1k-supervisor-contract-sha256",
    "--code-revision",
    "--collection-shard-id",
    "--out",
    "--overwrite",
    "--resume",
    "--scenes",
})
_SUPERVISOR_POLICY_EXCLUDED = frozenset({
    "allow_dirty_code", "b1k_supervisor_contract_sha256", "code_revision",
    "debug_images", "debug_outcomes_per_frame", "out",
    "overwrite", "resume", "semantic_query_workers",
})


def _write(path: Path, value: Mapping) -> None:
    io_utils.atomic_write_json(
        path, dict(value), allow_nan=False, durable=True)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sealed_artifact_identity(
        path: Path, root: Path, *, sha256: str) -> dict:
    """Describe bytes already authenticated by finalization.v2."""
    if not path.is_file():
        raise ValueError(f"completed B1K shard artifact is absent: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": str(sha256),
    }


def _effective_child_policy(
        *, scene_id: str, output: Path, data_root: Path,
        source_manifest: Path, collection_shard_id: str, revision: str,
        code_dirty: bool, collect_args: Sequence[str]) -> dict:
    """Parse the child's full scientific policy using collector defaults.

    Pruned through the collector's own rule rather than the parser's raw view:
    the child writes backend-scoped params, so an expected policy taken
    straight off the parser would name GS roots a b1k child never records and
    fail the readback for every scene.
    """
    arguments = [
        "--backend", "b1k", "--scenes", str(scene_id),
        "--out", str(Path(output).resolve()),
        "--b1k-data-root", str(Path(data_root).resolve(strict=True)),
        "--b1k-source-manifest",
        str(Path(source_manifest).resolve(strict=True)),
        "--collection-shard-id", str(collection_shard_id),
        "--code-revision", str(revision),
    ]
    if code_dirty:
        arguments.append("--allow-dirty-code")
    arguments.extend(str(value) for value in collect_args)
    parser = collection_cli.build_parser()
    parsed = parser.parse_args(arguments)
    collection_runtime._resolve_collection_mode_defaults(parsed, parser)
    params = collection_support.prune_backend_scoped_params(
        {key: value for key, value in vars(parsed).items()
         if key not in _SUPERVISOR_POLICY_EXCLUDED},
        getattr(parsed, "backend", "r2r"))
    return json.loads(json.dumps(params, sort_keys=True))


def _expected_child_contract(
        *, scene_id: str, output: Path, data_root: Path,
        source_manifest: Path, shard_id: str, revision: str,
        code_dirty: bool, collect_args: Sequence[str] | None = None) -> dict:
    """Freeze the supervisor-owned child arguments and source identity."""
    manifest = Path(source_manifest).resolve(strict=True)
    contract = {
        "scene_id": str(scene_id),
        "out": str(Path(output).resolve()),
        "data_root": str(Path(data_root).resolve(strict=True)),
        "source_manifest_path": str(manifest),
        "source_manifest_sha256": io_utils.sha256_file(manifest),
        "collection_shard_id": f"{shard_id}-{scene_id}",
        "code_revision": str(revision),
        "code_dirty": bool(code_dirty),
    }
    if collect_args is not None:
        policy = _effective_child_policy(
            scene_id=scene_id, output=output, data_root=data_root,
            source_manifest=manifest,
            collection_shard_id=contract["collection_shard_id"],
            revision=revision, code_dirty=code_dirty,
            collect_args=collect_args)
        contract["collect_policy"] = policy
        contract["collect_policy_sha256"] = (
            collection_funnel.canonical_sha256({
                "schema": _SUPERVISOR_CONTRACT_SCHEMA,
                "effective_policy": policy,
            }))
    return contract


def _validate_child_contract(
        run_meta: Mapping, funnel: Mapping, *, expected: Mapping) -> str:
    params = run_meta.get("params") or {}
    source_catalog = run_meta.get("source_catalog") or {}
    expected_params = {
        "backend": "b1k",
        "scenes": [expected["scene_id"]],
        "out": expected["out"],
        "b1k_data_root": expected["data_root"],
        "b1k_source_manifest": expected["source_manifest_path"],
        "collection_shard_id": expected["collection_shard_id"],
        "code_revision": expected["code_revision"],
        "allow_dirty_code": expected["code_dirty"],
    }
    if any(params.get(key) != value
           for key, value in expected_params.items()):
        raise ValueError("B1K output differs from the expected child contract")
    expected_policy = expected.get("collect_policy")
    if expected_policy is not None:
        expected_digest = expected.get("collect_policy_sha256")
        observed_digest = params.get("b1k_supervisor_contract_sha256")
        policy_digest = collection_funnel.canonical_sha256({
            "schema": _SUPERVISOR_CONTRACT_SCHEMA,
            "effective_policy": expected_policy,
        })
        if (policy_digest != expected_digest or
                observed_digest != expected_digest):
            raise ValueError(
                "B1K output differs from the expected child contract")
        try:
            observed_policy = {
                key: params[key] for key in expected_policy}
        except KeyError as error:
            raise ValueError(
                "B1K output differs from the expected child contract") \
                from error
        if observed_policy != expected_policy:
            raise ValueError(
                "B1K output differs from the expected child contract")
    if (run_meta.get("code_revision") != expected["code_revision"] or
            run_meta.get("code_dirty") is not expected["code_dirty"] or
            funnel.get("code_revision") != expected["code_revision"] or
            funnel.get("code_dirty") is not expected["code_dirty"] or
            source_catalog.get("datasets") != ["b1k"] or
            source_catalog.get("scene_ids") != [expected["scene_id"]] or
            source_catalog.get("manifest_sha256") != [
                expected["source_manifest_sha256"]]):
        raise ValueError("B1K output differs from the expected child contract")
    digest = run_meta.get("run_contract_sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(character not in "0123456789abcdef" for character in digest) or
            funnel.get("run_contract_sha256") != digest):
        raise ValueError("B1K child run contract digest is invalid")
    return digest


def _validate_terminal_output(
        output: Path, scene_id: str, *, expected: Mapping,
        partial: bool) -> dict:
    """Validate one completed or source-valid partial child output."""
    label = "partial" if partial else "completed"
    finalization_status = "partial" if partial else "completed"
    funnel_status = "failed" if partial else "completed"
    terminal_status = "partial_valid" if partial else "completed"
    funnel_path = output / "collection_funnel.json"
    run_meta_path = output / "run_meta.json"
    records_path = output / "records.jsonl"
    finalization = collection_closeout.load_finalization(
        output, allowed_statuses={finalization_status})
    if finalization["source_validation"] != "passed":
        raise ValueError(
            f"B1K {label} scene lacks passed source validation")
    funnel = _json(funnel_path)
    rows = funnel.get("per_scene")
    if (funnel.get("schema_version") != _FUNNEL_SCHEMA or
            funnel.get("backend") != "b1k" or
            funnel.get("status") != funnel_status or
            not isinstance(rows, list) or len(rows) != 1 or
            rows[0].get("scene_id") != scene_id or
            rows[0].get("status") != "completed"):
        description = (
            "an eligible partial shard" if partial
            else "a completed one-scene shard")
        raise ValueError(f"B1K scene output is not {description}: {output}")
    counts = funnel.get("scene_counts") or {}
    if (counts.get("total") != 1 or counts.get("completed") != 1 or
            counts.get("failed") != 0 or counts.get("interrupted") != 0):
        raise ValueError(
            f"B1K {label} scene funnel counts are invalid: {output}")
    run_meta = _json(run_meta_path)
    run_contract_sha256 = _validate_child_contract(
        run_meta, funnel, expected=expected)
    if finalization["run_contract_sha256"] != run_contract_sha256:
        raise ValueError("B1K finalization run contract differs")
    source_validated_records = int(finalization["record_count"])
    if partial and source_validated_records < 1:
        raise ValueError(
            f"B1K {label} scene has no source-valid records")
    if not partial and source_validated_records == 0:
        terminal_status = "zero_yield"
    stored_coverage = run_meta.get("formal_action_length_coverage")
    if partial and (not isinstance(stored_coverage, Mapping) or
                    stored_coverage.get("complete") is not False):
        raise ValueError("B1K historical partial coverage is invalid")
    result = {
        "scene_id": scene_id,
        "terminal_status": terminal_status,
        "output_dir": str(output),
        "source_validated_records": source_validated_records,
        "expected_child_contract": dict(expected),
        "run_contract_sha256": run_contract_sha256,
        "artifacts": [
            _sealed_artifact_identity(path, output, sha256=digest)
            for path, digest in (
                (funnel_path, finalization["funnel_sha256"]),
                (records_path, finalization["records_sha256"]),
                (run_meta_path, finalization["run_meta_sha256"]),
            )
        ],
    }
    if isinstance(stored_coverage, Mapping):
        result["formal_action_length_coverage"] = dict(stored_coverage)
    return result


def _validate_completed_output(
        output: Path, scene_id: str, *, expected: Mapping) -> dict:
    """Validate one completed child terminal boundary."""
    return _validate_terminal_output(
        output, scene_id, expected=expected, partial=False)


def _validate_partial_output(
        output: Path, scene_id: str, *, expected: Mapping) -> dict:
    """Accept a source-valid general shard with action-length shortfall."""
    return _validate_terminal_output(
        output, scene_id, expected=expected, partial=True)


def _run_isolated_collection(
        command: Sequence[str], *, env: Mapping[str, str], log_path: Path,
        timeout_s: int) -> int:
    """Run one scene in its own process group and durably retain its log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            list(command), env=dict(env), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            _emit_worker_heartbeat(
                pid=process.pid, log_path=log_path, elapsed_s=0.0)
            return b1k_process.wait_isolated_process(
                process, timeout_s=timeout_s,
                heartbeat_interval_s=_WORKER_HEARTBEAT_INTERVAL_S,
                on_heartbeat=lambda elapsed: _emit_worker_heartbeat(
                    pid=process.pid, log_path=log_path,
                    elapsed_s=elapsed))
        finally:
            os.fsync(log.fileno())


def _emit_worker_heartbeat(
        *, pid: int, log_path: Path, elapsed_s: float) -> None:
    """Emit bounded-latency evidence that an isolated worker is still live."""
    try:
        log_bytes = int(Path(log_path).stat().st_size)
    except FileNotFoundError:
        log_bytes = 0
    print(json.dumps({
        "event": "b1k_worker_heartbeat",
        "pid": int(pid),
        "elapsed_s": round(float(elapsed_s), 3),
        "log_bytes": log_bytes,
    }, sort_keys=True), file=sys.stderr, flush=True)


def _safe_identifier(value: str, *, field: str) -> str:
    normalized = str(value).strip()
    if (not normalized or not all(
            character.isalnum() or character in "-_"
            for character in normalized)):
        raise ValueError(f"B1K {field} contains unsafe characters")
    return normalized


def _validate_forwarded_args(values: Sequence[str]) -> list[str]:
    forwarded = [str(value) for value in values]
    forbidden = sorted(
        option for option in _PROTECTED_COLLECT_OPTIONS
        if any(value.split("=", 1)[0] == option for value in forwarded))
    if forbidden:
        raise ValueError(
            "B1K shard runner owns collection option(s): " +
            ", ".join(forbidden))
    return forwarded


def _load_source_scenes(path: Path) -> set[str]:
    manifest = _json(path)
    if manifest.get("schema_version") != B1K_SOURCE_MANIFEST_VERSION:
        raise ValueError("B1K source manifest schema is invalid")
    rows = manifest.get("scenes")
    if not isinstance(rows, list):
        raise ValueError("B1K source manifest scenes are invalid")
    scenes = [str((row or {}).get("scene_id") or "")
              for row in rows if isinstance(row, dict)]
    if len(scenes) != len(rows) or not scenes or \
            len(scenes) != len(set(scenes)):
        raise ValueError("B1K source manifest scenes are invalid")
    return set(scenes)


def _run_shard(args) -> int:
    data_root = Path(args.data_root).resolve(strict=True)
    source_manifest = Path(args.source_manifest).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_ids = [
        _safe_identifier(value, field="scene ID") for value in args.scenes]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("B1K collection shard scenes must be unique")
    source_scenes = _load_source_scenes(source_manifest)
    if not set(scene_ids).issubset(source_scenes):
        raise ValueError(
            "B1K collection shard contains unauthenticated scenes")
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        raise ValueError("B1K collection GPU ID must be non-negative")
    shard_id = _safe_identifier(args.shard_id, field="shard ID")
    revision = str(args.code_revision).strip()
    if not revision:
        raise ValueError("B1K collection code revision must be nonempty")
    timeout_s = int(args.scene_timeout_s)
    if timeout_s <= 0:
        raise ValueError("B1K collection scene timeout must be positive")
    forwarded = _validate_forwarded_args(args.collect_args)
    progress_path = output_dir / "shard-progress.json"
    manifest_identity = {
        "path": str(source_manifest),
        "sha256": io_utils.sha256_file(source_manifest),
    }
    contract = {
        "source_manifest": manifest_identity,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "gpu_id": gpu_id,
        "shard_id": shard_id,
        "code_revision": revision,
        "code_dirty": bool(args.allow_dirty_code),
        "scene_timeout_s": timeout_s,
        "requested_scene_ids": scene_ids,
        "collect_args": forwarded,
    }

    def expected_for(scene_id: str) -> dict:
        return _expected_child_contract(
            scene_id=scene_id, output=output_dir / scene_id,
            data_root=data_root, source_manifest=source_manifest,
            shard_id=shard_id, revision=revision,
            code_dirty=bool(args.allow_dirty_code), collect_args=forwarded)
    attempts = []
    completed = {}
    partial_valid = {}
    if progress_path.is_file():
        if not args.resume:
            raise ValueError(
                f"B1K collection progress exists; use --resume: {progress_path}")
        existing = _json(progress_path)
        expected = {"schema": _PROGRESS_SCHEMA, **contract}
        for field, expected_value in expected.items():
            if existing.get(field) != expected_value:
                raise ValueError(
                    f"B1K collection progress {field} differs on resume")
        raw_attempts = existing.get("attempts")
        if not isinstance(raw_attempts, list):
            raise ValueError("B1K collection progress attempts are invalid")
        attempts = list(raw_attempts)
        rows = existing.get("completed_scenes")
        if not isinstance(rows, list):
            raise ValueError(
                "B1K collection completed-scene progress is invalid")
        completed = {str(row.get("scene_id")): dict(row) for row in rows}
        if len(completed) != len(rows):
            raise ValueError(
                "B1K collection completed-scene progress is invalid")
        for scene_id, declared in completed.items():
            if scene_id not in scene_ids:
                raise ValueError(
                    "B1K collection progress contains an unexpected scene")
            try:
                observed = _validate_completed_output(
                    output_dir / scene_id, scene_id,
                    expected=expected_for(scene_id))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"B1K completed artifact changed: {scene_id}") from error
            if observed != declared:
                raise ValueError(
                    f"B1K completed artifact changed: {scene_id}")
        rows = existing.get("partial_valid_scenes")
        if not isinstance(rows, list):
            raise ValueError(
                "B1K collection partial-valid progress is invalid")
        partial_valid = {
            str(row.get("scene_id")): dict(row) for row in rows}
        if len(partial_valid) != len(rows):
            raise ValueError(
                "B1K collection partial-valid progress is invalid")
        if set(completed) & set(partial_valid):
            raise ValueError(
                "B1K scene cannot be completed and partial-valid")
        for scene_id, declared in partial_valid.items():
            if scene_id not in scene_ids:
                raise ValueError(
                    "B1K collection progress contains an unexpected scene")
            try:
                observed = _validate_partial_output(
                    output_dir / scene_id, scene_id,
                    expected=expected_for(scene_id))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"B1K partial-valid artifact changed: {scene_id}") \
                    from error
            if observed != declared:
                raise ValueError(
                    f"B1K partial-valid artifact changed: {scene_id}")

    def write_progress() -> None:
        reusable = {**completed, **partial_valid}
        latest = {}
        for attempt in attempts:
            latest[str(attempt["scene_id"])] = attempt
        failed = [
            {
                "scene_id": scene_id,
                "returncode": latest[scene_id].get("returncode"),
                "error": str(latest[scene_id]["error"]),
            }
            for scene_id in scene_ids
            if scene_id not in reusable and
            latest.get(scene_id, {}).get("status") == "failed"
        ]
        complete = len(reusable) == len(scene_ids)
        _write(progress_path, {
            "schema": _PROGRESS_SCHEMA,
            **contract,
            "completed_scene_ids": [
                scene_id for scene_id in scene_ids if scene_id in completed],
            "completed_scenes": [
                completed[scene_id]
                for scene_id in scene_ids if scene_id in completed],
            "partial_valid_scene_ids": [
                scene_id for scene_id in scene_ids
                if scene_id in partial_valid],
            "partial_valid_scenes": [
                partial_valid[scene_id]
                for scene_id in scene_ids if scene_id in partial_valid],
            "reusable_scene_ids": [
                scene_id for scene_id in scene_ids if scene_id in reusable],
            "failed_scenes": failed,
            "attempts": attempts,
            "complete": complete,
        })

    write_progress()
    environment = dict(os.environ)
    environment.update({
        "OMNIGIBSON_DATA_PATH": str(data_root),
        "OMNIGIBSON_APPDATA_PATH": str((data_root / "appdata").resolve()),
        "OMNIGIBSON_HEADLESS": "True",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "OMNIGIBSON_GPU_ID": str(gpu_id),
    })
    for scene_id in scene_ids:
        if scene_id in completed or scene_id in partial_valid:
            print(f"resume {scene_id}", flush=True)
            continue
        scene_output = (output_dir / scene_id).resolve()
        command = [
            str(b1k_source_builder.BEHAVIOR_PYTHON),
            str(ROOT / "scripts" / "collect.py"),
            "--backend", "b1k", "--scenes", scene_id,
            "--out", str(scene_output),
            "--b1k-data-root", str(data_root),
            "--b1k-source-manifest", str(source_manifest),
            "--collection-shard-id", f"{shard_id}-{scene_id}",
            "--code-revision", revision,
        ]
        if args.resume:
            command.append("--resume")
        if args.allow_dirty_code:
            command.append("--allow-dirty-code")
        command.extend([
            "--b1k-supervisor-contract-sha256",
            expected_for(scene_id)["collect_policy_sha256"],
        ])
        command.extend(forwarded)
        try:
            returncode = _run_isolated_collection(
                command, env=environment,
                log_path=(output_dir / "logs" / f"{scene_id}.log").resolve(),
                timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            attempts.append({
                "scene_id": scene_id,
                "status": "failed",
                "returncode": None,
                "error": f"TimeoutExpired: scene exceeded {timeout_s} seconds",
            })
            write_progress()
            print(json.dumps(attempts[-1], sort_keys=True),
                  file=sys.stderr, flush=True)
            continue
        if returncode != 0:
            attempts.append({
                "scene_id": scene_id,
                "status": "failed",
                "returncode": int(returncode),
                "error": "collect.py returned nonzero",
            })
            write_progress()
            print(json.dumps(attempts[-1], sort_keys=True),
                  file=sys.stderr, flush=True)
            continue
        try:
            completed[scene_id] = _validate_completed_output(
                scene_output, scene_id, expected=expected_for(scene_id))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            attempts.append({
                "scene_id": scene_id,
                "status": "failed",
                "returncode": int(returncode),
                "error": f"{type(error).__name__}: {error}",
            })
            write_progress()
            print(json.dumps(attempts[-1], sort_keys=True),
                  file=sys.stderr, flush=True)
            continue
        attempts.append({
            "scene_id": scene_id,
            "status": "completed",
            "returncode": int(returncode),
        })
        write_progress()
        print(json.dumps({
            "scene_id": scene_id,
            "output_dir": str(scene_output),
            "records_sha256": completed[scene_id]["artifacts"][1]["sha256"],
        }, sort_keys=True), flush=True)
    write_progress()
    progress = _json(progress_path)
    return int(progress.get("complete") is not True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--data-root", required=True)
    run.add_argument("--source-manifest", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--gpu-id", type=int, required=True)
    run.add_argument("--shard-id", required=True)
    run.add_argument("--code-revision", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-dirty-code", action="store_true")
    run.add_argument(
        "--scene-timeout-s", type=int,
        default=b1k_source_builder.DEFAULT_COLLECTION_SCENE_TIMEOUT_S)
    run.add_argument("--scenes", nargs="+", required=True)
    run.add_argument("--collect-args", nargs=argparse.REMAINDER, default=[])
    run.set_defaults(run=_run_shard)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
