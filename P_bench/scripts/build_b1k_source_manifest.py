#!/usr/bin/env python3
"""Verify a B1K install and build pilot or full authenticated manifests."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import (
    b1k_authority_audit, b1k_process, b1k_source_builder, config, io_utils,
    record,
)


_FRAGMENT_SCHEMAS = frozenset({
    "b1k-scene-authority-fragment.v1",
    "b1k-scene-authority-fragment.v2",
})
_FRAGMENT_FIELDS = frozenset({
    "schema", "scene_id", "scene_json_sha256", "scene_authority",
    "bootstrap",
})
_SHARD_PROGRESS_SCHEMA = "b1k-source-manifest-shard-progress.v2"
_SHARD_PROGRESS_FIELDS = frozenset({
    "schema", "data_root", "scene_timeout_s", "requested_scene_ids",
    "completed_scene_ids", "completed_scenes", "failed_scenes", "attempts",
    "complete",
})
_FAILURE_SCHEMA = "b1k-scene-authority-failure.v1"
_CURRENT_FRAGMENT_SCHEMA = "b1k-scene-authority-fragment.v2"
_CURRENT_BOOTSTRAP_SCHEMA = "b1k-scene-authority-bootstrap.v2"
_TRANSIENT_FAILURE_TOKENS = (
    "timeout", "timed out", "oom", "out of memory", "outofmemory",
    "memoryerror", "keyboardinterrupt", "interrupted", "segmentation fault",
    "killed",
)
_EXCLUDABLE_FAILURE_CATEGORIES = frozenset({
    "empty_clearance",
    "reset_position_drift",
    "reset_orientation_drift",
    "reset_joint_drift",
    "triangle_authority_drift",
})
_EXCLUDABLE_FAILURE_TYPES = frozenset({"RuntimeError", "ValueError"})
_REPLAY_ERROR_FIELDS = frozenset({
    "max_object_position_error_m",
    "max_object_orientation_error_rad",
    "max_joint_position_error_rad",
})


def _validate_replay_errors(value, *, context: str) -> None:
    if not isinstance(value, dict) or frozenset(value) != \
            _REPLAY_ERROR_FIELDS:
        raise ValueError(
            f"B1K catalog replay {context} diagnostics are invalid")
    errors = {}
    for field in sorted(_REPLAY_ERROR_FIELDS):
        raw = value[field]
        if isinstance(raw, bool):
            raise ValueError(
                f"B1K catalog replay {context} diagnostics are invalid")
        try:
            error = float(raw)
        except (TypeError, ValueError) as cause:
            raise ValueError(
                f"B1K catalog replay {context} diagnostics are invalid") \
                from cause
        if not math.isfinite(error) or error < 0.0:
            raise ValueError(
                f"B1K catalog replay {context} diagnostics are invalid")
        errors[field] = error
    if errors["max_object_position_error_m"] > \
            config.B1K_RESET_POSITION_TOL_M:
        raise ValueError(
            f"B1K catalog replay {context} position drift exceeds "
            "the frozen tolerance")
    if errors["max_object_orientation_error_rad"] > \
            config.B1K_RESET_ORIENTATION_TOL_RAD:
        raise ValueError(
            f"B1K catalog replay {context} orientation drift exceeds "
            "the frozen tolerance")
    if errors["max_joint_position_error_rad"] > \
            config.B1K_RESET_ORIENTATION_TOL_RAD:
        raise ValueError(
            f"B1K catalog replay {context} joint drift exceeds "
            "the frozen tolerance")


def _write(path, value) -> None:
    io_utils.atomic_write_json(
        Path(path), value, allow_nan=False, durable=True)


def _observed_install(source_root: Path, data_root: Path) -> dict:
    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=source_root, check=True,
            capture_output=True, text=True).stdout.strip()

    distribution = importlib.metadata.distribution("omnigibson")
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json"))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("OmniGibson direct_url.json is absent or invalid") \
            from error
    parsed = urlparse(str(direct_url.get("url") or ""))
    if (parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or
            (direct_url.get("dir_info") or {}).get("editable") is not True):
        raise ValueError("OmniGibson is not an editable local checkout")
    editable_root = Path(unquote(parsed.path)).resolve(strict=True)
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "source_tag": git("describe", "--tags", "--exact-match", "HEAD"),
        "python": platform.python_version(),
        "omnigibson": importlib.metadata.version("omnigibson"),
        "omnigibson_editable_root": str(editable_root),
        "bddl": importlib.metadata.version("bddl"),
        "isaac_sim": importlib.metadata.version("isaacsim"),
        "torch": importlib.metadata.version("torch"),
        "behavior-1k-assets": (
            data_root / "behavior-1k-assets" / "VERSION").read_text().strip(),
        "omnigibson-robot-assets": (
            data_root / "omnigibson-robot-assets" / "VERSION").read_text(
                ).strip(),
    }


def _verify_install(args) -> int:
    data_root = Path(args.data_root).resolve(strict=True)
    source_root = Path(args.source_root).resolve(strict=True)
    manifest = b1k_source_builder.build_install_manifest(
        data_root=data_root, source_root=source_root,
        observed=_observed_install(source_root, data_root),
        scene_ids=b1k_source_builder.installed_scene_ids(data_root))
    _write(args.output, manifest)
    print(json.dumps({
        "verified": True,
        "installed_catalog_count": manifest["installed_catalog_count"],
        "output": str(Path(args.output).resolve()),
    }, sort_keys=True))
    return 0


def _shutdown_runtime() -> None:
    from pipeline.b1k_sim import shutdown_b1k_runtime

    try:
        shutdown_b1k_runtime()
    except SystemExit as error:
        if error.code not in (None, 0):
            raise


def _derive_one(data_root: Path, scene_id: str, output: Path, runtime) -> dict:
    from pipeline.b1k_sim import bootstrap_scene_authority

    scene_json = b1k_source_builder.canonical_scene_json(
        data_root, scene_id)
    result = bootstrap_scene_authority(
        scene_id, scene_json, runtime=runtime)
    fragment = {
        "schema": "b1k-scene-authority-fragment.v2",
        "scene_id": scene_id,
        "scene_json_sha256": io_utils.sha256_file(scene_json),
        "scene_authority": result["scene_authority"],
        "bootstrap": result,
    }
    _write(output, fragment)
    return fragment


def _derive_scene(args, *, hard_exit=None) -> int:
    from pipeline.b1k_sim import load_b1k_runtime

    data_root = Path(args.data_root).resolve(strict=True)
    output = Path(args.output).resolve()
    failure_output = Path(f"{output}.failure.json")
    try:
        runtime = load_b1k_runtime()
        fragment = _derive_one(data_root, args.scene, output, runtime)
    except BaseException as error:
        failure = {
            "schema": "b1k-scene-authority-failure.v1",
            "scene_id": args.scene,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        attempt_id = getattr(args, "attempt_id", None)
        if attempt_id is not None:
            failure["attempt_id"] = str(attempt_id)
        _write(failure_output, failure)
        print(json.dumps(failure, sort_keys=True),
              file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        status = 130 if isinstance(error, KeyboardInterrupt) else 1
        (os._exit if hard_exit is None else hard_exit)(status)
        return status
    failure_output.unlink(missing_ok=True)
    print(json.dumps({
        "scene_id": args.scene,
        "scene_authority_sha256": fragment["scene_authority"]["sha256"],
        "output": str(output),
    }, sort_keys=True), flush=True)
    _shutdown_runtime()
    return 0


def _load_valid_fragment(
        path: Path, scene_id: str, data_root: Path) -> dict:
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or
            value.get("schema") not in _FRAGMENT_SCHEMAS or
            frozenset(value) != _FRAGMENT_FIELDS or
            value.get("scene_id") != scene_id):
        raise ValueError(
            f"B1K authority fragment does not match the exact schema: {path}")
    authority = b1k_source_builder._validate_authority(
        value.get("scene_authority") or {})
    scene_json = b1k_source_builder.canonical_scene_json(
        data_root, scene_id)
    if value.get("scene_json_sha256") != io_utils.sha256_file(scene_json):
        raise ValueError(f"B1K scene JSON changed: {scene_id}")
    bootstrap = value.get("bootstrap")
    if (not isinstance(bootstrap, dict) or
            bootstrap.get("schema") not in {
                "b1k-scene-authority-bootstrap.v1",
                "b1k-scene-authority-bootstrap.v2",
            } or
            bootstrap.get("scene_id") != scene_id):
        raise ValueError(f"B1K bootstrap scene differs: {scene_id}")
    if bootstrap.get("scene_authority") != authority:
        raise ValueError(f"B1K bootstrap authority differs: {scene_id}")
    return value


def _load_current_audit_fragment(
        path: Path, scene_id: str, data_root: Path) -> dict:
    """Require the v2 fragment/replay certificate used by catalog audit."""
    value = _load_valid_fragment(path, scene_id, data_root)
    if value.get("schema") != _CURRENT_FRAGMENT_SCHEMA:
        raise ValueError("B1K catalog fragment schema is not current")
    bootstrap = value["bootstrap"]
    if bootstrap.get("schema") != _CURRENT_BOOTSTRAP_SCHEMA:
        raise ValueError("B1K catalog bootstrap schema is not current")
    from pipeline import b1k_semantic

    replay_binding = b1k_semantic.canonical_replay_binding(
        value["scene_authority"])
    if replay_binding is None:
        raise ValueError("B1K catalog fragment lacks canonical replay binding")
    replay = bootstrap.get("canonical_replay")
    if not isinstance(replay, dict):
        raise ValueError("B1K catalog bootstrap lacks canonical replay report")
    if (replay.get("protocol") != replay_binding["canonical_replay_protocol"] or
            replay.get("canonical_state") != "state_2" or
            replay.get("canonical_state_sha256") !=
            replay_binding["canonical_state_sha256"]):
        raise ValueError("B1K catalog canonical replay binding differs")
    triangle_digests = replay.get("triangle_authority_sha256_by_state")
    if (not isinstance(triangle_digests, dict) or
            set(triangle_digests) != {"state_2", "state_3", "state_4"} or
            any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
                for value in triangle_digests.values()) or
            len(set(triangle_digests.values())) != 1):
        raise ValueError("B1K catalog replay state-2--4 certificate is invalid")
    authority = value["scene_authority"]
    triangle_authority = {
        key: value for key, value in authority.items()
        if key not in {
            "sha256", "canonical_replay_protocol",
            "canonical_state_sha256",
        }
    }
    if next(iter(triangle_digests.values())) != \
            record.canonical_atom_sha256(triangle_authority):
        raise ValueError(
            "B1K catalog replay triangle authority differs from its scene")
    for field in (
            "state_2_to_state_3_errors", "state_2_to_state_4_errors",
            "state_3_to_state_4_errors"):
        _validate_replay_errors(replay.get(field), context=field)
    return value


def _fragment_identity(path: Path, output_dir: Path) -> dict:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": io_utils.sha256_file(path),
    }


def _run_isolated_scene_child(command, *, timeout_s: int) -> int:
    """Run one OG scene in its own terminable process group."""
    process = subprocess.Popen(command, start_new_session=True)
    return b1k_process.wait_isolated_process(
        process, timeout_s=timeout_s)


def _derive_shard(args) -> int:
    data_root = Path(args.data_root).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = set(b1k_source_builder.installed_scene_ids(data_root))
    scenes = [str(value) for value in args.scenes]
    if len(scenes) != len(set(scenes)) or not set(scenes).issubset(catalog):
        raise ValueError("derive-shard scene selection is invalid")
    timeout_s = int(getattr(
        args, "scene_timeout_s",
        b1k_source_builder.DEFAULT_SCENE_TIMEOUT_S))
    if timeout_s <= 0:
        raise ValueError("derive-shard scene timeout must be positive")
    progress_path = output_dir / "shard-progress.json"
    completed = []
    completed_rows = {}
    attempts = []

    def validated_row(scene_id: str) -> dict:
        path = output_dir / f"{scene_id}.json"
        _load_valid_fragment(path, scene_id, data_root)
        return {
            "scene_id": scene_id,
            "fragment": _fragment_identity(path, output_dir),
        }

    if progress_path.is_file():
        if not args.resume:
            raise ValueError(
                f"B1K authority progress exists; use --resume: {progress_path}")
        existing = json.loads(progress_path.read_text())
        schema = existing.get("schema")
        if schema == _SHARD_PROGRESS_SCHEMA:
            if (frozenset(existing) != _SHARD_PROGRESS_FIELDS or
                    existing.get("data_root") != str(data_root) or
                    existing.get("requested_scene_ids") != scenes or
                    existing.get("scene_timeout_s") != timeout_s):
                raise ValueError("B1K authority shard progress contract changed")
            rows = existing.get("completed_scenes")
            if not isinstance(rows, list):
                raise ValueError("B1K authority completed progress is invalid")
            for row in rows:
                scene_id = str((row or {}).get("scene_id") or "")
                if scene_id not in scenes or scene_id in completed_rows:
                    raise ValueError(
                        "B1K authority completed progress is invalid")
                observed = validated_row(scene_id)
                if observed != row:
                    raise ValueError(
                        f"B1K authority fragment changed: {scene_id}")
                completed_rows[scene_id] = observed
            raw_attempts = existing.get("attempts")
            if not isinstance(raw_attempts, list):
                raise ValueError("B1K authority attempts are invalid")
            attempts = []
            attempt_counts = {}
            for raw_attempt in raw_attempts:
                attempt = dict(raw_attempt)
                scene_id = str(attempt.get("scene_id") or "")
                count = attempt_counts.get(scene_id, 0)
                attempt_counts[scene_id] = count + 1
                attempt.setdefault(
                    "attempt_id", f"{scene_id}:attempt-{count:04d}")
                attempts.append(attempt)
            observed_completed = [
                scene_id for scene_id in scenes
                if scene_id in completed_rows]
            if (existing.get("completed_scene_ids") != observed_completed or
                    existing.get("complete") is not
                    (observed_completed == scenes)):
                raise ValueError(
                    "B1K authority completed progress is inconsistent")
        elif schema == "b1k-source-manifest-shard-progress.v1":
            # One-time migration: never trust the old completed list.  Adopt
            # only fragments that independently pass current byte/source and
            # bootstrap validation, then immediately publish v2 identities.
            for scene_id in scenes:
                path = output_dir / f"{scene_id}.json"
                if path.is_file():
                    try:
                        completed_rows[scene_id] = validated_row(scene_id)
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
        else:
            raise ValueError("B1K authority shard progress schema is invalid")
    elif args.resume:
        # Recover a child fragment written before the first parent progress
        # publication, using the same independent validation as assembly.
        for scene_id in scenes:
            path = output_dir / f"{scene_id}.json"
            if path.is_file():
                try:
                    completed_rows[scene_id] = validated_row(scene_id)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
    completed = [scene_id for scene_id in scenes
                 if scene_id in completed_rows]

    def write_progress() -> None:
        latest = {}
        for attempt in attempts:
            latest[str(attempt["scene_id"])] = attempt
        failed = [
            {
                "scene_id": scene_id,
                "attempt_id": latest[scene_id]["attempt_id"],
                "returncode": latest[scene_id].get("returncode"),
                "error": latest[scene_id]["error"],
                **({"child_failure": latest[scene_id]["child_failure"]}
                   if "child_failure" in latest[scene_id] else {}),
            }
            for scene_id in scenes
            if scene_id not in completed_rows and
            latest.get(scene_id, {}).get("status") == "failed"
        ]
        _write(progress_path, {
            "schema": _SHARD_PROGRESS_SCHEMA,
            "data_root": str(data_root),
            "scene_timeout_s": timeout_s,
            "requested_scene_ids": scenes,
            "completed_scene_ids": completed,
            "completed_scenes": [
                completed_rows[scene_id] for scene_id in completed],
            "failed_scenes": failed,
            "attempts": attempts,
            "complete": completed == scenes,
        })

    write_progress()
    for scene_id in scenes:
        output = output_dir / f"{scene_id}.json"
        if scene_id in completed_rows:
            print(f"resume {scene_id}", flush=True)
            continue
        attempt_number = sum(
            attempt.get("scene_id") == scene_id for attempt in attempts)
        attempt_id = f"{scene_id}:attempt-{attempt_number:04d}"
        command = [
            sys.executable, str(Path(__file__).resolve()), "derive-scene",
            "--data-root", str(data_root), "--scene", scene_id,
            "--output", str(output), "--attempt-id", attempt_id,
        ]
        try:
            returncode = _run_isolated_scene_child(
                command, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            attempts.append({
                "attempt_id": attempt_id,
                "scene_id": scene_id,
                "status": "failed",
                "returncode": None,
                "error": (
                    f"TimeoutExpired: scene exceeded {timeout_s} seconds"),
            })
            write_progress()
            print(json.dumps(attempts[-1], sort_keys=True),
                  file=sys.stderr, flush=True)
            continue
        try:
            fragment = _load_valid_fragment(output, scene_id, data_root)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            child_failure_path = Path(f"{output}.failure.json")
            child_failure = None
            if child_failure_path.is_file():
                try:
                    child_failure = json.loads(child_failure_path.read_text())
                except (OSError, json.JSONDecodeError):
                    child_failure = None
            attempts.append({
                "attempt_id": attempt_id,
                "scene_id": scene_id,
                "status": "failed",
                "returncode": int(returncode),
                "error": f"{type(error).__name__}: {error}",
                **({"child_failure": child_failure}
                   if child_failure is not None else {}),
            })
            write_progress()
            print(json.dumps(attempts[-1], sort_keys=True),
                  file=sys.stderr, flush=True)
            continue
        if returncode != 0:
            attempts.append({
                "attempt_id": attempt_id,
                "scene_id": scene_id,
                "status": "failed",
                "returncode": int(returncode),
                "error": "derive-scene returned nonzero after fragment write",
            })
            write_progress()
            print(json.dumps(attempts[-1], sort_keys=True),
                  file=sys.stderr, flush=True)
            continue
        completed_rows[scene_id] = validated_row(scene_id)
        completed = [value for value in scenes if value in completed_rows]
        attempts.append({
            "attempt_id": attempt_id,
            "scene_id": scene_id,
            "status": "completed",
            "returncode": int(returncode),
        })
        write_progress()
        print(json.dumps({
            "scene_id": scene_id,
            "scene_authority_sha256": fragment["scene_authority"]["sha256"],
        }, sort_keys=True), flush=True)
    write_progress()
    return int(completed != scenes)


def _assemble(args) -> int:
    data_root = Path(args.data_root).resolve(strict=True)
    install_path = Path(args.install_manifest).resolve(strict=True)
    expected_install_sha256 = str(
        args.expected_install_manifest_sha256).strip()
    if (len(expected_install_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in expected_install_sha256)):
        raise ValueError("expected B1K install manifest digest is invalid")
    if io_utils.sha256_file(install_path) != expected_install_sha256:
        raise ValueError("B1K install manifest digest changed")
    install = json.loads(install_path.read_text())
    if install.get("schema") != "b1k-verified-install.v1" or \
            install.get("verified") is not True:
        raise ValueError("B1K install manifest is not verified")
    expected_editable_root = str(
        (Path(install.get("source_root") or "") / "OmniGibson").resolve())
    if (install.get("data_root") != str(data_root) or
            install.get("omnigibson_editable_root") !=
            expected_editable_root or
            install.get("versions") != b1k_source_builder.PINNED_INSTALL):
        raise ValueError("B1K install manifest pinning changed")
    catalog = b1k_source_builder.installed_scene_ids(data_root)
    if (install.get("installed_scene_ids") != catalog or
            install.get("installed_catalog_count") != len(catalog)):
        raise ValueError("B1K install manifest catalog changed")
    fragment_dirs = [Path(value).resolve(strict=True)
                     for value in args.fragment_dir]
    entries = []
    for scene_id in args.scenes:
        matches = [directory / f"{scene_id}.json"
                   for directory in fragment_dirs
                   if (directory / f"{scene_id}.json").is_file()]
        if len(matches) != 1:
            raise ValueError(
                f"B1K scene {scene_id} has {len(matches)} authority fragments")
        fragment = _load_valid_fragment(matches[0], scene_id, data_root)
        entries.append(b1k_source_builder.build_scene_entry(
            data_root, scene_id, fragment["scene_authority"]))
    versions = install["versions"]
    manifest = b1k_source_builder.build_source_manifest(
        entries, installed_scene_ids=catalog,
        selection_scope=args.selection_scope,
        simulator={
            "omnigibson": versions["omnigibson"],
            "bddl": versions["bddl"],
            "isaac_sim": versions["isaac_sim"],
            "torch": versions["torch"],
            "source_tag": versions["source_tag"],
            "source_commit": versions["source_commit"],
        },
        asset_versions={
            "behavior-1k-assets": versions["behavior-1k-assets"],
            "omnigibson-robot-assets":
                versions["omnigibson-robot-assets"],
        })
    _write(args.output, manifest)
    print(json.dumps({
        "installed_catalog_count": manifest["installed_catalog_count"],
        "selected_scene_count": manifest["selected_scene_count"],
        "selection_scope": manifest["selection_scope"],
        "output": str(Path(args.output).resolve()),
    }, sort_keys=True))
    return 0


def _audit_file_identity(path: Path) -> dict:
    """Return a durable identity for an authority-audit input file."""
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"B1K audit input is not a file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": io_utils.sha256_file(resolved),
    }


def _load_audited_install(args, data_root: Path) -> tuple[dict, list[str], Path]:
    """Load a digest-pinned 51-scene install authority for catalog assembly."""
    install_path = Path(args.install_manifest).resolve(strict=True)
    expected_install_sha256 = str(
        args.expected_install_manifest_sha256).strip()
    if (len(expected_install_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in expected_install_sha256)):
        raise ValueError("expected B1K install manifest digest is invalid")
    if io_utils.sha256_file(install_path) != expected_install_sha256:
        raise ValueError("B1K install manifest digest changed")
    install = json.loads(install_path.read_text())
    if install.get("schema") != "b1k-verified-install.v1" or \
            install.get("verified") is not True:
        raise ValueError("B1K install manifest is not verified")
    expected_editable_root = str(
        (Path(install.get("source_root") or "") / "OmniGibson").resolve())
    if (install.get("data_root") != str(data_root) or
            install.get("omnigibson_editable_root") !=
            expected_editable_root or
            install.get("versions") != b1k_source_builder.PINNED_INSTALL):
        raise ValueError("B1K install manifest pinning changed")
    catalog = b1k_source_builder.installed_scene_ids(data_root)
    if (install.get("installed_scene_ids") != catalog or
            install.get("installed_catalog_count") != len(catalog)):
        raise ValueError("B1K install manifest catalog changed")
    if len(catalog) != b1k_source_builder.CATALOG_AUTHORITY_AUDIT_CATALOG_COUNT:
        raise ValueError("B1K catalog authority audit requires 51 installed scenes")
    return install, catalog, install_path


def _typed_failure(scene_id: str, attempt: dict) -> tuple[str, str]:
    """Validate and normalize one child-persisted deterministic failure."""
    child = attempt.get("child_failure")
    if (not isinstance(child, dict) or
            frozenset(child) != {
                "schema", "scene_id", "attempt_id", "error_type", "error"} or
            child.get("schema") != _FAILURE_SCHEMA or
            child.get("scene_id") != scene_id or
            child.get("attempt_id") != attempt.get("attempt_id")):
        raise ValueError("missing typed child failure")
    error_type = str(child.get("error_type") or "").strip()
    error = str(child.get("error") or "").strip()
    if not error_type or not error:
        raise ValueError("incomplete typed child failure")
    return error_type, error


def _is_transient_failure(error_type: str, error: str) -> bool:
    value = f"{error_type}: {error}".lower()
    return any(token in value for token in _TRANSIENT_FAILURE_TOKENS)


def _attempt_id(scene_id: str, value, sequence: int) -> str:
    """Validate the child-issued monotonically sequenced shard attempt ID."""
    expected = f"{scene_id}:attempt-{sequence:04d}"
    if value != expected:
        raise ValueError("B1K audit attempt identity or sequence is invalid")
    return expected


def _audit_progress_error(detail: str) -> ValueError:
    return ValueError(f"B1K catalog authority audit unresolved: {detail}")


def _load_audit_shards(data_root: Path, catalog: list[str], fragment_dirs) -> tuple[
        dict[str, Path], dict[str, list[tuple[str, int, dict]]], set[str],
        list[dict]]:
    """Authenticate four disjoint resumable histories against ``catalog``."""
    directories = [Path(value).resolve(strict=True) for value in fragment_dirs]
    if len(directories) != 4 or len(set(directories)) != 4:
        raise _audit_progress_error("four shard directories are required")
    assigned = {}
    failures = {}
    completed_attempts = set()
    identities = []
    for directory in directories:
        progress_path = directory / "shard-progress.json"
        try:
            progress = json.loads(progress_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise _audit_progress_error(
                f"shard progress is unreadable: {directory}") \
                from error
        if (not isinstance(progress, dict) or
                progress.get("schema") != _SHARD_PROGRESS_SCHEMA or
                frozenset(progress) != _SHARD_PROGRESS_FIELDS or
                progress.get("data_root") != str(data_root)):
            raise _audit_progress_error(f"shard progress is invalid: {directory}")
        scenes = progress.get("requested_scene_ids")
        if not isinstance(scenes, list) or not scenes:
            raise _audit_progress_error(
                f"shard scene list is invalid: {directory}")
        for scene_id in scenes:
            if (not isinstance(scene_id, str) or scene_id not in catalog or
                    scene_id in assigned):
                raise _audit_progress_error("shard assignments are invalid")
            assigned[scene_id] = directory
        completed_ids = progress.get("completed_scene_ids")
        completed_rows = progress.get("completed_scenes")
        if (not isinstance(completed_ids, list) or
                completed_ids != [scene_id for scene_id in scenes
                                  if scene_id in completed_ids] or
                not isinstance(completed_rows, list) or
                [row.get("scene_id") if isinstance(row, dict) else None
                 for row in completed_rows] != completed_ids):
            raise _audit_progress_error(
                f"completed summary is invalid: {directory}")
        for row in completed_rows:
            if frozenset(row) != {"scene_id", "fragment"}:
                raise _audit_progress_error(
                    f"completed summary row is invalid: {directory}")
            fragment_path = directory / f"{row['scene_id']}.json"
            try:
                identity = _fragment_identity(fragment_path, directory)
            except OSError as error:
                raise _audit_progress_error(
                    f"completed fragment is missing: {row['scene_id']}") from error
            if row["fragment"] != identity:
                raise _audit_progress_error(
                    f"completed fragment identity changed: {row['scene_id']}")
        attempts = progress.get("attempts")
        if not isinstance(attempts, list):
            raise _audit_progress_error(f"shard attempts are invalid: {directory}")
        sequence_by_scene = {scene_id: 0 for scene_id in scenes}
        latest = {}
        for attempt_index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                raise _audit_progress_error(f"attempt is invalid: {directory}")
            scene_id = attempt.get("scene_id")
            if scene_id not in scenes:
                raise _audit_progress_error(
                    f"attempt scene is invalid: {directory}")
            try:
                _attempt_id(scene_id, attempt.get("attempt_id"),
                            sequence_by_scene[scene_id])
            except ValueError as error:
                raise _audit_progress_error(str(error)) from error
            sequence_by_scene[scene_id] += 1
            status = attempt.get("status")
            if status == "completed":
                if (frozenset(attempt) != {
                        "attempt_id", "scene_id", "status", "returncode"} or
                        attempt.get("returncode") != 0):
                    raise _audit_progress_error(
                        f"completed attempt is invalid: {directory}")
                completed_attempts.add(scene_id)
            elif status == "failed":
                required = {
                    "attempt_id", "scene_id", "status", "returncode", "error"}
                if (not required.issubset(attempt) or
                        set(attempt).difference(required | {"child_failure"}) or
                        not str(attempt.get("error") or "")):
                    raise _audit_progress_error(
                        f"failed attempt is invalid: {directory}")
            else:
                raise _audit_progress_error(
                    f"attempt status is invalid: {directory}")
            latest[scene_id] = attempt
            if attempt.get("status") == "failed":
                failures.setdefault(scene_id, []).append((
                    directory.name, attempt_index, attempt))
        expected_failed = []
        for scene_id in scenes:
            attempt = latest.get(scene_id)
            if scene_id not in completed_ids and attempt and \
                    attempt["status"] == "failed":
                expected_failed.append({
                    "scene_id": scene_id,
                    "attempt_id": attempt["attempt_id"],
                    "returncode": attempt["returncode"],
                    "error": attempt["error"],
                    **({"child_failure": attempt["child_failure"]}
                       if "child_failure" in attempt else {}),
                })
        if (progress.get("failed_scenes") != expected_failed or
                progress.get("complete") is not (completed_ids == scenes)):
            raise _audit_progress_error(
                f"failed or completion summary is inconsistent: {directory}")
        identities.append({
            "shard": directory.name,
            "directory": str(directory),
            "progress": _audit_file_identity(progress_path),
        })
    if set(assigned) != set(catalog):
        raise _audit_progress_error("shard assignments do not cover catalog")
    return (assigned, failures, completed_attempts,
            sorted(identities, key=lambda value: value["shard"]))


def _assemble_catalog_audit(args) -> int:
    """Assemble authority only from terminal audited catalog outcomes.

    A present, independently validated fragment accepts a scene.  A missing
    fragment can only exclude its scene after two isolated child attempts
    preserve the same non-transient typed failure.  Every other history is an
    unresolved authority error, never a source-manifest selection decision.
    """
    data_root = Path(args.data_root).resolve(strict=True)
    install, catalog, install_path = _load_audited_install(args, data_root)
    assigned, failed_attempts, completed_attempts, shard_identities = _load_audit_shards(
        data_root, catalog, args.fragment_dir)
    accepted = []
    excluded = []
    entries = []
    unresolved = []
    for scene_id in catalog:
        directory = assigned[scene_id]
        fragment_path = directory / f"{scene_id}.json"
        if fragment_path.is_file():
            try:
                fragment = _load_current_audit_fragment(
                    fragment_path, scene_id, data_root)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                unresolved.append({
                    "scene_id": scene_id,
                    "reason": "invalid_authority_fragment",
                    "detail": f"{type(error).__name__}: {error}",
                })
                continue
            identity = _fragment_identity(fragment_path, directory)
            accepted.append({
                "scene_id": scene_id,
                "terminal": "accepted_authority_fragment",
                "fragment": {"shard": directory.name, **identity},
            })
            entries.append(b1k_source_builder.build_scene_entry(
                data_root, scene_id, fragment["scene_authority"]))
            continue
        history = failed_attempts.get(scene_id, [])
        if scene_id in completed_attempts:
            unresolved.append({
                "scene_id": scene_id,
                "reason": "completed_attempt_without_fragment",
            })
            continue
        typed = []
        try:
            for shard_name, attempt_index, attempt in history:
                error_type, error = _typed_failure(scene_id, attempt)
                typed.append((shard_name, attempt_index, error_type, error))
        except ValueError as error:
            unresolved.append({
                "scene_id": scene_id,
                "reason": "missing_typed_failure",
                "detail": str(error),
            })
            continue
        outcomes = {(error_type, error) for _, _, error_type, error in typed}
        if len(typed) < 2:
            unresolved.append({
                "scene_id": scene_id,
                "reason": "insufficient_failure_attempts",
                "attempt_count": len(typed),
            })
            continue
        if len(outcomes) != 1:
            unresolved.append({
                "scene_id": scene_id,
                "reason": "divergent_typed_failures",
                "failures": [
                    {"error_type": error_type, "error": error}
                    for _, _, error_type, error in typed],
            })
            continue
        error_type, error = next(iter(outcomes))
        category = b1k_authority_audit._failure_category(error)
        if _is_transient_failure(error_type, error):
            unresolved.append({
                "scene_id": scene_id,
                "reason": "transient_typed_failure",
                "failure": {"error_type": error_type, "error": error},
            })
            continue
        if (error_type not in _EXCLUDABLE_FAILURE_TYPES or
                category not in _EXCLUDABLE_FAILURE_CATEGORIES):
            unresolved.append({
                "scene_id": scene_id,
                "reason": "unallowlisted_typed_failure",
                "failure": {
                    "error_type": error_type,
                    "error": error,
                    "category": category,
                },
            })
            continue
        excluded.append({
            "scene_id": scene_id,
            "terminal": "excluded_deterministic_failure",
            "failure": {
                "error_type": error_type,
                "error": error,
                "category": category,
            },
            "attempts": [
                {"shard": shard_name, "attempt_index": attempt_index}
                for shard_name, attempt_index, _error_type, _error in typed],
        })
    if unresolved:
        raise ValueError("B1K catalog authority audit unresolved: " +
                         json.dumps(unresolved, sort_keys=True))
    versions = install["versions"]
    manifest = b1k_source_builder.build_source_manifest(
        entries, installed_scene_ids=catalog,
        selection_scope=b1k_source_builder.CATALOG_AUTHORITY_AUDIT_SELECTION_SCOPE,
        simulator={
            "omnigibson": versions["omnigibson"],
            "bddl": versions["bddl"],
            "isaac_sim": versions["isaac_sim"],
            "torch": versions["torch"],
            "source_tag": versions["source_tag"],
            "source_commit": versions["source_commit"],
        },
        asset_versions={
            "behavior-1k-assets": versions["behavior-1k-assets"],
            "omnigibson-robot-assets":
                versions["omnigibson-robot-assets"],
        })
    audit = {
        "schema": b1k_source_builder.CATALOG_AUTHORITY_AUDIT_SCHEMA,
        "selection_scope": b1k_source_builder.CATALOG_AUTHORITY_AUDIT_SELECTION_SCOPE,
        "installed_scene_ids": catalog,
        "accepted": accepted,
        "excluded": excluded,
        "source_identities": {
            "install_manifest": _audit_file_identity(install_path),
            "shards": shard_identities,
        },
    }
    output = Path(args.output).resolve()
    audit_output = Path(args.audit_output).resolve()
    if output == audit_output:
        raise ValueError("B1K catalog audit and source outputs must differ")
    _write(output, manifest)
    _write(audit_output, audit)
    print(json.dumps({
        "installed_catalog_count": len(catalog),
        "accepted_scene_count": len(accepted),
        "excluded_scene_count": len(excluded),
        "output": str(output),
        "audit_output": str(audit_output),
    }, sort_keys=True))
    return 0


def _plan_shards(args) -> int:
    data_root = Path(args.data_root).resolve(strict=True)
    plan = b1k_source_builder.build_shard_plan(
        b1k_source_builder.installed_scene_ids(data_root),
        data_root=data_root, output_root=Path(args.output_root),
        gpu_ids=args.gpu_ids)
    _write(args.output, plan)
    for shard in plan["shards"]:
        print(shard["command"])
    return 0


def _plan_collection_shards(args) -> int:
    data_root = Path(args.data_root).resolve(strict=True)
    plan = b1k_source_builder.build_collection_shard_plan(
        b1k_source_builder.installed_scene_ids(data_root),
        data_root=data_root,
        source_manifest=Path(args.source_manifest),
        output_root=Path(args.output_root),
        gpu_ids=args.gpu_ids,
        code_revision=args.code_revision,
        collect_args=args.collect_args,
        scene_timeout_s=args.scene_timeout_s,
    )
    _write(args.output, plan)
    for shard in plan["shards"]:
        print(shard["command"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-install")
    verify.add_argument("--data-root", required=True)
    verify.add_argument("--source-root", required=True)
    verify.add_argument("--output", required=True)
    verify.set_defaults(run=_verify_install)

    derive = commands.add_parser("derive-scene")
    derive.add_argument("--data-root", required=True)
    derive.add_argument("--scene", required=True)
    derive.add_argument("--output", required=True)
    derive.add_argument("--attempt-id")
    derive.set_defaults(run=_derive_scene)

    shard = commands.add_parser("derive-shard")
    shard.add_argument("--data-root", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--resume", action="store_true")
    shard.add_argument(
        "--scene-timeout-s", type=int,
        default=b1k_source_builder.DEFAULT_SCENE_TIMEOUT_S)
    shard.add_argument("--scenes", nargs="+", required=True)
    shard.set_defaults(run=_derive_shard)

    assemble = commands.add_parser("assemble")
    assemble.add_argument("--data-root", required=True)
    assemble.add_argument("--install-manifest", required=True)
    assemble.add_argument(
        "--expected-install-manifest-sha256", required=True)
    assemble.add_argument(
        "--fragment-dir", action="append", required=True)
    assemble.add_argument("--selection-scope", required=True)
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--scenes", nargs="+", required=True)
    assemble.set_defaults(run=_assemble)

    catalog_audit = commands.add_parser("assemble-catalog-audit")
    catalog_audit.add_argument("--data-root", required=True)
    catalog_audit.add_argument("--install-manifest", required=True)
    catalog_audit.add_argument(
        "--expected-install-manifest-sha256", required=True)
    catalog_audit.add_argument(
        "--fragment-dir", action="append", required=True)
    catalog_audit.add_argument("--output", required=True)
    catalog_audit.add_argument("--audit-output", required=True)
    catalog_audit.set_defaults(run=_assemble_catalog_audit)

    plan = commands.add_parser("plan-shards")
    plan.add_argument("--data-root", required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    plan.add_argument("--output", required=True)
    plan.set_defaults(run=_plan_shards)

    collection_plan = commands.add_parser("plan-collection-shards")
    collection_plan.add_argument("--data-root", required=True)
    collection_plan.add_argument("--source-manifest", required=True)
    collection_plan.add_argument("--output-root", required=True)
    collection_plan.add_argument(
        "--gpu-ids", nargs=4, type=int, required=True)
    collection_plan.add_argument("--code-revision", required=True)
    collection_plan.add_argument(
        "--scene-timeout-s", type=int,
        default=b1k_source_builder.DEFAULT_COLLECTION_SCENE_TIMEOUT_S)
    collection_plan.add_argument("--output", required=True)
    collection_plan.add_argument(
        "--collect-args", nargs=argparse.REMAINDER, default=[])
    collection_plan.set_defaults(run=_plan_collection_shards)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
