"""Authenticated canary evidence and deterministic collection capacity."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping

from pipeline import (
    a1_common_support, background_authorities, background_canary, benchmark,
    candidate_preview, config, dataset_contracts, gate_authority, io_utils,
    record,
)


PROFILE_SCHEMA = "egoconseq.collection-capacity-profile.v3"
EVIDENCE_SCHEMA = "egoconseq.collection-capacity-evidence.v1"
CONTROLLER_SCHEMA = "egoconseq.capacity-canary-controller.v1"
CANARY_PUBLICATION_SCHEMA = "egoconseq.capacity-canary-publication.v1"
CANARY_PUBLICATION_NAME = "publication.json"
DATASETS = dataset_contracts.main_collection_datasets()
TASKS = tuple(benchmark.ABC_CANDIDATE_TASK_IDS)
_QA_FILES = (
    "public/items.jsonl",
    "private/answers.jsonl",
    "private/atoms.jsonl",
    "private/record_contexts.jsonl",
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _controller_paths(job: Mapping) -> tuple[Path, Path]:
    output = Path(str(job["output_dir"])).resolve()
    return (
        output / "capacity_controller_manifest.json",
        output / "controller_events.jsonl",
    )


def initialize_controller_events(
        manifest: Mapping, job: Mapping, *, time_unix: float,
        parent_manifest_path: Path = None) -> None:
    """Start one digest-bound timing trace for a controller scene job."""
    controller_path, events_path = _controller_paths(job)
    controller_path.parent.mkdir(parents=True, exist_ok=True)
    binding = job["transaction_binding"]
    body = {
        "schema": CONTROLLER_SCHEMA,
        "parent_manifest_sha256": manifest["sha256"],
        "dataset": job["dataset"],
        "scene_id": job["scenes"][0],
        "job_id": job["job_id"],
        "collection_shard_id": binding["collection_shard_id"],
        "revision": binding["revision"],
        "source_authority": dict(binding["source_authority"]),
    }
    if parent_manifest_path is not None:
        path = Path(parent_manifest_path).resolve()
        parent = gate_authority.load_authority_manifest(path)
        background_canary.validate_manifest(parent)
        if dict(parent) != dict(manifest):
            raise ValueError("capacity canary parent identity differs")
        body["parent_manifest"] = _identity(path)
    controller = {**body, "sha256": _canonical_sha256(body)}
    io_utils.atomic_write_json(
        controller_path, controller, allow_nan=False, durable=True)
    events_path.unlink(missing_ok=True)
    record_controller_event(
        manifest, job, "controller_scene_started", time_unix=time_unix)


def record_controller_event(
        manifest: Mapping, job: Mapping, event: str, *,
        time_unix: float) -> None:
    """Append one manifest/job-bound timing event for later profiling."""
    if event not in {
            "controller_scene_started", "backend_ready",
            "controller_scene_finished"}:
        raise ValueError("background capacity event is invalid")
    controller_path, events_path = _controller_paths(job)
    controller = gate_authority.load_authority_manifest(controller_path)
    binding = job["transaction_binding"]
    if controller.get("parent_manifest_sha256") != manifest["sha256"]:
        raise ValueError("background capacity controller manifest differs")
    if events_path.is_file():
        existing = io_utils.read_jsonl(events_path, require_dict=True)
        matches = [row for row in existing if row.get("event") == event]
        if len(matches) > 1:
            raise ValueError("background capacity event is duplicated")
        if matches:
            return
    value = {
        "event": event,
        "dataset": job["dataset"],
        "scene_id": job["scenes"][0],
        "time_unix": float(time_unix),
        "controller_manifest_sha256": controller["sha256"],
        "job_id": job["job_id"],
        "collection_shard_id": binding["collection_shard_id"],
        "revision": binding["revision"],
        "source_authority_sha256": binding["source_authority"]["sha256"],
    }
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def backend_ready_time(
        paths: list[Path], *, started_time: float = None):
    """Return the latest backend-ready event for the current attempt."""
    values = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "backend_ready" not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "backend_ready":
                    continue
                value = float(event["time_unix"])
                if started_time is None or value >= float(started_time):
                    values.append(value)
    return max(values) if values else None


def _read_jsonl(path: Path, *, label: str) -> list[dict]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"capacity evidence {label} is unreadable") from error
    try:
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            rows = []
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"capacity evidence {label} line {line_number} "
                        "is invalid") from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"capacity evidence {label} rows must be objects")
                rows.append(row)
            return rows
    except BaseException:
        # fdopen owns the descriptor after successful construction.
        raise


def _read_json(path: Path, *, label: str) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"capacity evidence {label} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"capacity evidence {label} must be one JSON object")
    return value


def _positive_int(value, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"capacity evidence {label} must be positive")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"capacity evidence {label} must be positive") from error
    if result != value or result <= 0:
        raise ValueError(f"capacity evidence {label} must be positive")
    return result


def _identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": io_utils.sha256_file(path)}


def candidate_publication_files(root: Path) -> dict[str, str]:
    """Digest the files protected by one canary publication manifest."""
    root = Path(root).resolve()
    return {
        path.relative_to(root).as_posix(): io_utils.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != CANARY_PUBLICATION_NAME
    }


def validate_canary_publication(
        artifact: Path, *, parent_manifest_sha256: str, job_id: str,
        source: Mapping, identity: Mapping = None,
        ) -> tuple[Path, Path, dict]:
    """Authenticate an already compiler-validated canary publication.

    The expensive deterministic compiler projection runs before publication.
    Reopening the durable publication only needs to recheck its exact source
    binding and the hashes of the bytes that were atomically published.
    """
    artifact = Path(artifact).resolve()
    root = artifact.parent
    if artifact != root / "candidate_qa":
        raise ValueError("capacity canary artifact path is invalid")
    report = root / "candidate_qa_report"
    manifest_path = root / CANARY_PUBLICATION_NAME
    value = gate_authority.load_authority_manifest(manifest_path)
    body = {key: item for key, item in value.items() if key != "sha256"}
    expected = {
        "schema": CANARY_PUBLICATION_SCHEMA,
        "parent_manifest_sha256": str(parent_manifest_sha256),
        "job_id": str(job_id),
        "source": dict(source),
        "artifact": "candidate_qa",
        "report": "candidate_qa_report",
        "files": candidate_publication_files(root),
    }
    if body != expected or value.get("sha256") != _canonical_sha256(body):
        raise ValueError("capacity canary publication binding differs")
    actual_identity = {
        "path": str(manifest_path),
        "sha256": io_utils.sha256_file(manifest_path),
        "content_sha256": value["sha256"],
    }
    if identity is not None and dict(identity) != actual_identity:
        raise ValueError("capacity canary publication identity differs")
    if not (report / "index.html").is_file():
        raise ValueError("capacity canary report is incomplete")
    return artifact, report, actual_identity


def _resolve_item_contexts(
        artifact: Path) -> tuple[dict[str, int], set[str], set[str]]:
    items = {str(row.get("id") or ""): row for row in _read_jsonl(
        artifact / _QA_FILES[0], label="compiled public items")}
    answers = {str(row.get("id") or ""): row for row in _read_jsonl(
        artifact / _QA_FILES[1], label="compiled private answers")}
    atoms = {str(row.get("id") or ""): row for row in _read_jsonl(
        artifact / _QA_FILES[2], label="compiled private atoms")}
    contexts = {str(row.get("record_sha256") or ""): row.get("context")
                for row in _read_jsonl(
                    artifact / _QA_FILES[3],
                    label="compiled private record contexts")}
    if (not items or "" in items or len(items) != len(_read_jsonl(
            artifact / _QA_FILES[0], label="compiled public items")) or
            set(items) != set(answers)):
        raise ValueError("capacity evidence compiled QA IDs are invalid")
    counts = {task_id: 0 for task_id in TASKS}
    record_digests = set()
    frame_digests = set()
    for item_id, item in items.items():
        try:
            atom = atoms[str(answers[item_id]["atom_ref"])]
            digest = str(atom["record_sha256"])
            context = contexts[digest]
            task_id = str(item["task_id"])
            input_asset = answers[item_id]["input_asset"]
            frame_digest = str(
                input_asset.get("raw_sha256") or
                input_asset.get("sha256") or "")
        except (KeyError, TypeError) as error:
            raise ValueError(
                "capacity evidence compiled QA source binding is missing") \
                from error
        if (task_id not in counts or not isinstance(context, Mapping) or
                not frame_digest):
            raise ValueError("capacity evidence compiled QA route is invalid")
        counts[task_id] += 1
        record_digests.add(digest)
        frame_digests.add(frame_digest)
    return counts, record_digests, frame_digests


def _validate_candidate_artifact(
        artifact: Path, records_path: Path, run_meta_path: Path) -> None:
    source = {
        "path": str(records_path),
        "records_sha256": io_utils.sha256_file(records_path),
        "run_meta_sha256": io_utils.sha256_file(run_meta_path),
    }
    authority = background_authorities.resolve_artifact_source_authority(
        artifact, [source])
    candidate_preview.validate_preview_artifact(
        artifact, expected_source_authority=authority)


def _authenticate_canary(
        dataset: str, scene_id: str, value: Mapping,
        controller: Mapping, source_authority) -> dict:
    parent_path = Path(str(value["parent_manifest_path"])).resolve()
    parent = background_canary.load(parent_path)
    if controller.get("parent_manifest_sha256") != parent["sha256"]:
        raise ValueError("capacity canary parent digest differs")
    if controller.get("parent_manifest") != _identity(parent_path):
        raise ValueError("capacity canary parent file identity differs")
    jobs = [job for job in parent["jobs"]
            if job.get("dataset") == dataset and
            job.get("scenes") == [scene_id]]
    if len(jobs) != 1:
        raise ValueError("capacity canary parent job binding differs")
    job = jobs[0]
    binding = job["transaction_binding"]
    expected = {
        "dataset": dataset,
        "scene_id": scene_id,
        "job_id": job["job_id"],
        "collection_shard_id": binding["collection_shard_id"],
        "revision": binding["revision"],
        "source_authority": binding["source_authority"],
    }
    if any(controller.get(key) != item for key, item in expected.items()):
        raise ValueError("capacity canary controller job binding differs")
    job_root = Path(job["output_dir"]).resolve()
    scene_root = job_root / scene_id if dataset == "b1k" else job_root
    expected_paths = {
        "records_path": scene_root / "records.jsonl",
        "run_meta_path": scene_root / "run_meta.json",
        "funnel_path": scene_root / "collection_funnel.json",
        "controller_events_path": job_root / "controller_events.jsonl",
        "controller_manifest_path":
            job_root / "capacity_controller_manifest.json",
        "compiled_qa_path":
            job_root / "capacity_candidate" / "candidate_qa",
    }
    if any(Path(str(value[name])).resolve() != expected_path
           for name, expected_path in expected_paths.items()):
        raise ValueError("capacity canary evidence path differs from job")
    authority_path = Path(str(value["source_authority_path"])).resolve()
    if authority_path != Path(parent["output_root"]).resolve() / \
            "capacity-source-authority.json":
        raise ValueError("capacity canary source authority path differs")
    records_path = Path(str(value["records_path"])).resolve()
    sources = {row["path"]: row for row in source_authority.sources}
    if str(records_path) not in sources:
        raise ValueError("capacity canary source authority omits records")
    publication_source = {
        "path": str(records_path),
        "records_sha256": io_utils.sha256_file(records_path),
        "run_meta_sha256": io_utils.sha256_file(
            expected_paths["run_meta_path"]),
    }
    _artifact, _report, publication = validate_canary_publication(
        expected_paths["compiled_qa_path"],
        parent_manifest_sha256=parent["sha256"], job_id=job["job_id"],
        source=publication_source)
    return {
        "parent_manifest": _identity(parent_path),
        "source_authority": _identity(authority_path),
        "candidate_publication": publication,
    }


def _source_route(context: Mapping) -> tuple[str, str]:
    source = context.get("source") or {}
    contract = context.get("collection_contract") or {}
    datasets = {str(value) for value in (
        source.get("source_dataset"), contract.get("source_dataset"))
        if value}
    scenes = {str(value) for value in (
        source.get("scene_id"), contract.get("scene_id")) if value}
    if len(datasets) != 1 or len(scenes) != 1:
        raise ValueError("capacity evidence compiled QA route is invalid")
    return datasets.pop(), scenes.pop()


def _measure_canary(dataset: str, value: Mapping, source_authority) -> dict:
    allowed = {
        "scene_id", "records_path", "run_meta_path", "funnel_path",
        "controller_events_path", "controller_manifest_path",
        "compiled_qa_path", "parent_manifest_path", "source_authority_path",
    }
    if not isinstance(value, Mapping) or set(value) != allowed:
        raise ValueError(
            "capacity evidence canary must contain only source paths")
    scene_id = str(value.get("scene_id") or "").strip()
    if not scene_id:
        raise ValueError("capacity evidence canary scene ID is invalid")
    records_path = Path(str(value["records_path"])).resolve()
    run_meta_path = Path(str(value["run_meta_path"])).resolve()
    funnel_path = Path(str(value["funnel_path"])).resolve()
    events_path = Path(str(value["controller_events_path"])).resolve()
    controller_path = Path(str(value["controller_manifest_path"])).resolve()
    artifact = Path(str(value["compiled_qa_path"])).resolve()
    controller = _read_json(
        controller_path, label="canary controller manifest")
    controller_body = {
        key: item for key, item in controller.items() if key != "sha256"}
    if (controller.get("schema") != CONTROLLER_SCHEMA or
            controller.get("sha256") != _canonical_sha256(controller_body) or
            controller.get("dataset") != dataset or
            controller.get("scene_id") != scene_id or
            _HEX64.fullmatch(str(
                controller.get("parent_manifest_sha256") or "")) is None or
            not str(controller.get("job_id") or "") or
            not str(controller.get("collection_shard_id") or "") or
            _HEX40.fullmatch(str(controller.get("revision") or "")) is None or
            not str((controller.get("source_authority") or {}).get(
                "path") or "") or
            _HEX64.fullmatch(str((controller.get("source_authority") or {}).get(
                "sha256") or "")) is None):
        raise ValueError("capacity evidence controller manifest differs")
    authority_identities = _authenticate_canary(
        dataset, scene_id, value, controller, source_authority)
    records = _read_jsonl(records_path, label="records")
    if not records:
        raise ValueError("capacity profile accepted records must be positive")
    if any(not a1_common_support.has_uniform_v3_ordinary_provenance(row)
           for row in records):
        raise ValueError("capacity evidence records require proposal-v3")
    record_digests = {record.canonical_atom_sha256(row) for row in records}
    run_meta = _read_json(run_meta_path, label="run metadata")
    source_catalog = run_meta.get("source_catalog") or {}
    if (source_catalog.get("datasets") != [dataset] or
            source_catalog.get("scene_ids") != [scene_id] or
            source_catalog.get("manifest_sha256") != [
                (controller.get("source_authority") or {}).get("sha256")]):
        raise ValueError("capacity evidence run metadata route differs")
    params = run_meta.get("params") or {}
    ordinary_actions_per_pose = _positive_int(
        params.get("ordinary_actions_per_pose"),
        label="ordinary actions per pose")
    if ordinary_actions_per_pose not in \
            config.ACTION_CANDIDATE_ORDINARY_LADDER:
        raise ValueError("capacity evidence action budget is invalid")
    source_key = dataset_contracts.source_path_key(dataset)
    if (params.get("backend") != dataset or params.get("scenes") != [scene_id] or
            params.get("collection_shard_id") !=
            controller.get("collection_shard_id") or
            params.get("code_revision") != controller.get("revision") or
            params.get(source_key) !=
            (controller.get("source_authority") or {}).get("path") or
            run_meta.get("code_revision") != controller.get("revision") or
            run_meta.get("code_dirty") is not False):
        raise ValueError("capacity evidence run metadata binding differs")
    attempts = _positive_int(
        (run_meta.get("stats") or {}).get("pose_attempts"),
        label="pose attempts")
    if len(records) > attempts:
        raise ValueError(
            "capacity profile accepted records exceed pose attempts")
    funnel = _read_json(funnel_path, label="collection funnel")
    rows = [row for row in (funnel.get("per_scene") or [])
            if isinstance(row, Mapping) and row.get("scene_id") == scene_id]
    if len(rows) != 1 or funnel.get("backend") != dataset:
        raise ValueError("capacity evidence funnel route differs")
    funnel_attempts = _positive_int(
        (rows[0].get("stage_counts") or {}).get("pose_attempts"),
        label="funnel pose attempts")
    if attempts != funnel_attempts:
        raise ValueError("capacity evidence pose attempt counters differ")
    events = _read_jsonl(events_path, label="controller events")
    by_name = {}
    for event in events:
        if event.get("dataset") != dataset or event.get("scene_id") != scene_id:
            raise ValueError("capacity evidence controller event route differs")
        name = str(event.get("event") or "")
        if (event.get("controller_manifest_sha256") !=
                controller.get("sha256") or
                event.get("job_id") != controller.get("job_id") or
                event.get("collection_shard_id") !=
                controller.get("collection_shard_id") or
                event.get("revision") != controller.get("revision") or
                event.get("source_authority_sha256") !=
                (controller.get("source_authority") or {}).get("sha256")):
            raise ValueError("capacity evidence controller event binding differs")
        if name in by_name:
            raise ValueError("capacity evidence controller events duplicate")
        by_name[name] = event
    expected_events = {
        "controller_scene_started", "backend_ready",
        "controller_scene_finished",
    }
    if set(by_name) != expected_events:
        raise ValueError("capacity evidence controller events are incomplete")
    try:
        started = float(by_name["controller_scene_started"]["time_unix"])
        ready = float(by_name["backend_ready"]["time_unix"])
        finished = float(by_name["controller_scene_finished"]["time_unix"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("capacity evidence controller times are invalid") \
            from error
    if not all(math.isfinite(value) for value in (started, ready, finished)) or \
            not started < ready < finished:
        raise ValueError("capacity evidence controller times are invalid")
    task_counts, qa_record_digests, frame_digests = \
        _resolve_item_contexts(artifact)
    contexts = _read_jsonl(
        artifact / _QA_FILES[3], label="compiled private record contexts")
    for context_row in contexts:
        route = _source_route(context_row.get("context") or {})
        if route != (dataset, scene_id):
            raise ValueError("capacity evidence compiled QA route differs")
    if not qa_record_digests.issubset(record_digests):
        raise ValueError("capacity evidence compiled QA records differ")
    sources = {
        "records": _identity(records_path),
        "run_meta": _identity(run_meta_path),
        "funnel": _identity(funnel_path),
        "controller_events": _identity(events_path),
        "controller_manifest": _identity(controller_path),
        "compiled_qa": {
            relative: _identity(artifact / relative)
            for relative in _QA_FILES
        },
        **authority_identities,
    }
    return {
        "scene_id": scene_id,
        "initialization_s": ready - started,
        "pose_attempts": attempts,
        "accepted_records": len(records),
        "accepted_unique_frames": len(frame_digests),
        "ordinary_actions_per_pose": ordinary_actions_per_pose,
        "pose_attempt_s": (finished - ready) / attempts,
        "task_qa_counts": task_counts,
        "measurement_source": sources,
    }


def _derive_dataset(
        dataset: str, catalog_count: int, canaries: list[dict]) -> dict:
    dataset = str(dataset)
    if dataset not in DATASETS:
        raise ValueError("capacity profile dataset is invalid")
    accepted_total = sum(row["accepted_records"] for row in canaries)
    action_budgets = {
        int(row["ordinary_actions_per_pose"]) for row in canaries}
    if len(action_budgets) != 1:
        raise ValueError("capacity canary action budgets differ")
    ordinary_actions_per_pose = action_budgets.pop()
    unique_frame_total = sum(
        row["accepted_unique_frames"] for row in canaries)
    attempts_total = sum(row["pose_attempts"] for row in canaries)
    task_totals = {task: sum(row["task_qa_counts"][task]
                             for row in canaries) for task in TASKS}
    task_yields = {task: task_totals[task] / accepted_total for task in TASKS}
    if any(value <= 0.0 for value in task_yields.values()):
        raise ValueError("capacity profile has zero task yield")
    total_yield = sum(task_totals.values()) / accepted_total
    frame_yield = unique_frame_total / accepted_total
    acceptance = accepted_total / attempts_total
    qa_records_needed = max(
        math.ceil(config.BACKGROUND_MIN_TOTAL_ITEMS / total_yield),
        max(math.ceil(config.BACKGROUND_MIN_ITEMS_PER_TASK / task_yields[task])
            for task in TASKS))
    unique_frames_needed = \
        config.BACKGROUND_MIN_UNIQUE_FRAMES_BY_DATASET[dataset]
    records_needed = max(
        math.ceil(unique_frames_needed / frame_yield), qa_records_needed)
    # Canary yields remain diagnostics and determine how many catalog passes
    # are likely required. They must not turn one low-yield scene into tens of
    # thousands of brute-force pose draws on every scene in the dataset.
    records_per_scene = config.BACKGROUND_CANARY_ACCEPTED_RECORDS
    pose_attempt_cap = \
        config.BACKGROUND_CANARY_POSE_ATTEMPT_CAP_BY_DATASET[dataset]
    initialization_s = max(row["initialization_s"] for row in canaries)
    pose_attempt_s = max(row["pose_attempt_s"] for row in canaries)
    wallclock = config.BACKGROUND_SCENE_WALLCLOCK_S_BY_DATASET[dataset]
    return {
        "catalog_scene_count": catalog_count,
        "ordinary_actions_per_pose": ordinary_actions_per_pose,
        "canary_scenes": canaries,
        "pooled_total_yield_per_record": total_yield,
        "pooled_task_yield_per_record": task_yields,
        "pooled_unique_frame_yield_per_record": frame_yield,
        "pooled_pose_acceptance_rate": acceptance,
        "max_initialization_s": initialization_s,
        "max_pose_attempt_s": pose_attempt_s,
        "records_needed": records_needed,
        "qa_records_needed": qa_records_needed,
        "unique_frames_needed": unique_frames_needed,
        "records_per_scene": records_per_scene,
        "pose_attempt_cap": pose_attempt_cap,
        "scene_wallclock_s": wallclock,
    }


def _authenticate_evidence_parent(value: Mapping):
    parent_path = Path(str(value["parent_manifest_path"])).resolve()
    authority_path = Path(str(value["source_authority_path"])).resolve()
    parent = background_canary.load(parent_path)
    expected_authority = Path(parent["output_root"]).resolve() / \
        "capacity-source-authority.json"
    if authority_path != expected_authority:
        raise ValueError("capacity canary source authority path differs")
    background_canary.validate_source_authority(authority_path, parent)
    authority = gate_authority.resolve_preview_source_authority_from_manifest(
        authority_path, root=Path(parent["output_root"]))
    return parent, authority


def _parent_source_authorities(parent: Mapping) -> dict[str, dict[str, str]]:
    return {
        dataset: {
            "path": str(parent["paths"][
                dataset_contracts.source_path_key(dataset)]),
            "sha256": str(parent["source_manifest_sha256"][dataset]),
        }
        for dataset in DATASETS
    }


def build_profile(value: Mapping) -> dict:
    """Build capacity solely from authenticated, scene-bound evidence."""
    if not isinstance(value, Mapping) or value.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("capacity evidence schema is invalid")
    if set(value) != {
            "schema", "parent_manifest_path", "source_authority_path",
            "datasets"}:
        raise ValueError("capacity evidence top-level paths are invalid")
    parent, source_authority = _authenticate_evidence_parent(value)
    datasets = value.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(DATASETS):
        raise ValueError("capacity evidence datasets are invalid")
    derived = {}
    for dataset in DATASETS:
        row = datasets[dataset]
        if not isinstance(row, Mapping) or set(row) != {
                "catalog_scene_count", "canary_scenes"}:
            raise ValueError("capacity evidence dataset row is invalid")
        catalog_count = _positive_int(
            row["catalog_scene_count"], label="catalog scene count")
        if catalog_count != len(parent["scene_catalog"][dataset]):
            raise ValueError(
                "capacity evidence catalog count differs from canary parent")
        canaries = row["canary_scenes"]
        if not isinstance(canaries, list) or len(canaries) != 2:
            raise ValueError("capacity evidence requires exactly two canary scenes")
        if (any(canary.get("parent_manifest_path") !=
                value["parent_manifest_path"] or
                canary.get("source_authority_path") !=
                value["source_authority_path"] for canary in canaries) or
                [canary.get("scene_id") for canary in canaries] !=
                parent["selected_scenes"][dataset]):
            raise ValueError("capacity evidence differs from canary parent")
        measured = [
            _measure_canary(dataset, canary, source_authority)
            for canary in canaries
        ]
        if len({canary["scene_id"] for canary in measured}) != 2:
            raise ValueError("capacity evidence canary scenes must be distinct")
        derived[dataset] = _derive_dataset(dataset, catalog_count, measured)
    action_budgets = {
        row["ordinary_actions_per_pose"] for row in derived.values()}
    if len(action_budgets) != 1:
        raise ValueError("capacity dataset action budgets differ")
    body = {
        "schema": PROFILE_SCHEMA,
        "canary_revision": parent["revision"],
        "ordinary_actions_per_pose": action_budgets.pop(),
        "source_authorities": _parent_source_authorities(parent),
        "datasets": derived,
    }
    return {**body, "sha256": _canonical_sha256(body)}


def _evidence_from_profile(profile: Mapping) -> dict:
    datasets = {}
    for dataset in DATASETS:
        row = (profile.get("datasets") or {}).get(dataset) or {}
        canaries = []
        for canary in row.get("canary_scenes") or []:
            source = canary.get("measurement_source") or {}
            compiled = source.get("compiled_qa") or {}
            first = compiled.get(_QA_FILES[0]) or {}
            artifact = Path(str(first.get("path") or "")).parent.parent
            canaries.append({
                "scene_id": canary.get("scene_id"),
                "records_path": (source.get("records") or {}).get("path"),
                "run_meta_path": (source.get("run_meta") or {}).get("path"),
                "funnel_path": (source.get("funnel") or {}).get("path"),
                "controller_events_path": (
                    source.get("controller_events") or {}).get("path"),
                "controller_manifest_path": (
                    source.get("controller_manifest") or {}).get("path"),
                "compiled_qa_path": str(artifact),
                "parent_manifest_path": (
                    source.get("parent_manifest") or {}).get("path"),
                "source_authority_path": (
                    source.get("source_authority") or {}).get("path"),
            })
        datasets[dataset] = {
            "catalog_scene_count": row.get("catalog_scene_count"),
            "canary_scenes": canaries,
        }
    first = next(iter(datasets.values()))["canary_scenes"][0]
    return {
        "schema": EVIDENCE_SCHEMA,
        "parent_manifest_path": first["parent_manifest_path"],
        "source_authority_path": first["source_authority_path"],
        "datasets": datasets,
    }


def validate_profile(value: Mapping) -> None:
    """Reopen every evidence input and require the exact recomputed profile."""
    if not isinstance(value, Mapping) or value.get("schema") != PROFILE_SCHEMA:
        raise ValueError("capacity profile schema is invalid")
    expected = build_profile(_evidence_from_profile(value))
    if dict(value) != expected:
        raise ValueError("capacity profile digest or derivation is invalid")
