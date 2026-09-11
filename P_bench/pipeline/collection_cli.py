"""Collect structured consequences under body and sensor interventions.

Each sampled world pose is reused across every requested camera height/FOV.
Radius-conditioned physical rollouts are cached across those observation-only
interventions; goal and future-view labels are recomputed from each rendering.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from pipeline import (
    abc1_record, action_proposal, config, dataset_contracts, scene_partitions,
    validate,
)
from pipeline.actions import (
    actions_to_dicts,
    balanced_action_pool,
    Forward,
    parse_actions,
    Turn,
    validate_physics_actions,
)
from pipeline.record import ORACLE_CONTRACT_VERSION
from pipeline.io_utils import (
    atomic_write_binary,
    atomic_write_json,
    atomic_write_text,
    fsync_directory,
    read_jsonl,
)


def action_seqs(file_path):
    with open(file_path) as handle:
        data = json.load(handle)
    return [parse_actions(item["actions"]) for item in data["sequences"]]


def sensor_intervention(heights, fovs):
    """Describe observation-only sensor changes; physics is height-invariant."""
    changed = []
    if len(heights) > 1:
        changed.append("nominal_camera_offset_m")
    if len(fovs) > 1:
        changed.extend(["hfov_deg", "vfov_deg"])
    kind = "sensor_profile" if changed else "base"
    return kind, changed


def validate_formal_fov_profiles(fovs):
    """Validate dynamic HFOV choices against the formal 4:3 calibration."""
    profiles = [tuple(map(float, pair)) for pair in fovs]
    for hfov, vfov in profiles:
        config.render_resolution(hfov, vfov)
    return profiles


def sensor_tag(height, hfov, vfov):
    return (f"{config.height_tag(height)}-"
            f"f{int(round(hfov * 10)):04d}x{int(round(vfov * 10)):04d}")


def pose_group_id(
        scene_id: str, pose_index: int,
        collection_shard_id: str | None = None) -> str:
    shard = f"-{collection_shard_id}" if collection_shard_id else ""
    return f"{scene_id}{shard}-p{int(pose_index):03d}-observation"


def frame_id(
        scene_id: str, pose_index: int, collection_shard_id: str | None,
        sensor_tag: str) -> str:
    shard = f"-{collection_shard_id}" if collection_shard_id else ""
    return f"F-{scene_id}{shard}-p{int(pose_index):03d}-{sensor_tag}"


def candidate_action_pools(args, rng, stats, half_fov_deg):
    if args.action_mode == "file":
        if not args.action_file:
            raise ValueError("--action-file is required with --action-mode file")
        pools = collections.defaultdict(list)
        seen = set()
        for index, actions in enumerate(action_seqs(args.action_file)):
            if len(actions) not in set(args.lengths):
                raise ValueError(
                    f"file action {index} has length {len(actions)} outside --lengths")
            validate_physics_actions(actions)
            key = tuple(
                ("turn", float(action.deg)) if isinstance(action, Turn)
                else ("forward", float(action.m))
                for action in actions)
            if key in seen:
                raise ValueError(f"duplicate action program at file index {index}")
            seen.add(key)
            pools[len(actions)].append((f"file-{index:03d}", actions))
        return dict(pools)
    pool = balanced_action_pool(
        rng, args.lengths, args.keep_per_length * args.pool_factor,
        half_fov_deg=half_fov_deg)
    candidates = {}
    for length in args.lengths:
        stats[f"pool_L{length}"] += len(pool[length])
        candidates[int(length)] = [
            (f"L{int(length)}-c{index:03d}", actions)
            for index, actions in enumerate(pool[length])
        ]
    return candidates


def close_sessions(sessions):
    seen = set()
    for sim in sessions:
        if id(sim) not in seen:
            sim.close()
            seen.add(id(sim))


def _spool_dir(path: Path) -> Path:
    return path.parent / f".{path.name}.groups"


def _atomic_write(path: Path, payload: str) -> None:
    atomic_write_text(path, payload, durable=True)


def _collection_contract(expected_siblings, run_contract) -> dict:
    return {
        "oracle_contract_version": ORACLE_CONTRACT_VERSION,
        "expected_siblings": int(expected_siblings),
        "run": run_contract or {},
    }


def _write_spool_contract(directory: Path, contract: dict) -> None:
    _atomic_write(
        directory / "contract.json",
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n")


def load_spool_run_contract(records_path: Path) -> dict:
    """Load the immutable run contract bound to a records spool."""
    contract_path = _spool_dir(Path(records_path)) / "contract.json"
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("records spool contract is unreadable") from error
    if not isinstance(contract, dict) or \
            contract.get("oracle_contract_version") != \
            ORACLE_CONTRACT_VERSION or \
            not isinstance(contract.get("run"), dict):
        raise ValueError("records spool contract is invalid")
    return dict(contract["run"])


def _load_spool_records(spool: Path, expected_siblings: int) -> tuple[str, list]:
    # The spool's run contract pins the schema. Full schema/oracle validation
    # still runs before publication.
    records = read_jsonl(spool, require_dict=True)
    group_ids = {_record_group_id(record) for record in records}
    frame_ids = {str(record.get("frame_id")) for record in records}
    if (len(records) != int(expected_siblings) or len(frame_ids) != len(records) or
            len(group_ids) != 1 or not next(iter(group_ids), "")):
        group_id = next(iter(group_ids), spool.stem)
        raise ValueError(
            f"partial intervention group in resume data: {group_id}="
            f"{len(frame_ids)}/{int(expected_siblings)}")
    return next(iter(group_ids)), records


def _rebuild_records_from_spools(path: Path, spools) -> None:
    def write(output):
        for spool in spools:
            with spool.open("rb") as source:
                shutil.copyfileobj(source, output)

    atomic_write_binary(path, write, durable=True)


def _prepare_records_output_with_contract(
    path, *, expected_siblings, resume, overwrite, run_contract,
):
    path = Path(path)
    directory = _spool_dir(path)
    contract_path = directory / "contract.json"
    expected_contract = _collection_contract(expected_siblings, run_contract)
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if overwrite:
        path.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            retired = directory.with_name(
                f"{directory.name}.overwrite-{os.getpid()}-{time.time_ns()}")
            os.replace(directory, retired)
            fsync_directory(directory.parent)
        _atomic_write(path, "")
        directory.mkdir(parents=True)
        fsync_directory(directory.parent)
        _write_spool_contract(directory, expected_contract)
        for retired in directory.parent.glob(
                f"{directory.name}.overwrite-*"):
            shutil.rmtree(retired, ignore_errors=True)
        fsync_directory(directory.parent)
        return set(), 0
    exists = ((path.exists() and path.stat().st_size) or directory.exists())
    if exists and not resume:
        raise FileExistsError(
            f"{path} already exists; pass --resume or --overwrite")
    if resume and not exists:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        directory.mkdir(parents=True, exist_ok=False)
        _write_spool_contract(directory, expected_contract)
        return set(), 0
    if resume:
        if not contract_path.exists():
            raise ValueError(
                "unsupported oracle contract in resume data; recollect with "
                f"{ORACLE_CONTRACT_VERSION}")
        stored_contract = json.loads(contract_path.read_text())
        stored_revision = (stored_contract.get("run") or {}).get("code_revision")
        expected_revision = (
            expected_contract.get("run") or {}).get("code_revision")
        if stored_revision != expected_revision:
            raise ValueError(
                f"resume code revision {expected_revision!r} differs from "
                f"stored {stored_revision!r}; cross-revision resume is forbidden")
        if stored_contract != expected_contract:
            raise ValueError("collection contract differs from resume data")
        spools = sorted(directory.glob("*.jsonl"))
        completed = set()
        total = 0
        for spool in spools:
            group_id, records = _load_spool_records(spool, expected_siblings)
            if group_id in completed:
                raise ValueError(f"duplicate intervention group spool: {group_id}")
            completed.add(group_id)
            total += len(records)
        _rebuild_records_from_spools(path, spools)
        return completed, total
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    directory.mkdir(parents=True, exist_ok=False)
    _write_spool_contract(directory, expected_contract)
    return set(), 0


def prepare_records_output(
    path, *, expected_siblings, resume, overwrite, run_contract=None,
):
    """Prepare transactional group storage and return completed group ids."""
    return _prepare_records_output_with_contract(
        path, expected_siblings=expected_siblings, resume=resume,
        overwrite=overwrite, run_contract=run_contract)


def validate_records_before_spool(
        records, *, validation_context, asset_root=None) -> None:
    """Fail closed on per-record invariants before durable group storage."""
    failures = []
    for record_value in records:
        errors = validate.validate_record_local(
            record_value, context=validation_context,
            asset_root=asset_root)
        if errors:
            failures.extend(errors)
    failures.extend(validate.validate_intervention_groups(records))
    if failures:
        raise ValueError(
            "record validation failed before spool:\n" + "\n".join(failures))


def append_compact_records(path, records, *, progress_path=None) -> bool:
    """Append committed rows under one lock; preserve all previous full rows."""
    records = list(records)
    payload = b"".join(json.dumps(record, separators=(",", ":"),
                                allow_nan=False).encode() + b"\n"
                       for record in records)
    with open(path, "r+b") as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        end = output.seek(0, os.SEEK_END)
        # Only a process interrupted during its final write leaves a tail.
        while end:
            start = max(0, end - 65536)
            output.seek(start)
            chunk = output.read(end - start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                end = start + newline + 1
                break
            end = start
        output.truncate(end)
        progress = None
        if progress_path is not None:
            progress = json.loads(Path(progress_path).read_text())
            output.seek(progress["byte_offset"])
            progress["record_count"] += output.read().count(b"\n")
            progress["byte_offset"] = end
            if progress["record_count"] + len(records) > progress["target_records"]:
                atomic_write_json(progress_path, progress, durable=True)
                return False
        output.seek(end)
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
        if progress is not None:
            progress["record_count"] += len(records)
            progress["byte_offset"] = output.tell()
            atomic_write_json(progress_path, progress, durable=True)
    return True


def append_record_group(path, records, *, order_key, expected_siblings) -> None:
    """Durably spool one complete sibling group before exposing it in JSONL."""
    path = Path(path)
    records = list(records)
    group_ids = {_record_group_id(record) for record in records}
    frame_ids = {str(record.get("frame_id")) for record in records}
    if (len(records) != int(expected_siblings) or len(frame_ids) != len(records) or
            len(group_ids) != 1 or not next(iter(group_ids), "")):
        group_id = next(iter(group_ids), "unknown")
        raise ValueError(
            f"partial intervention group: {group_id}="
            f"{len(frame_ids)}/{int(expected_siblings)}")
    if any(not abc1_record.is_compact(record) and
           record.get("oracle_contract_version") != ORACLE_CONTRACT_VERSION
           for record in records):
        raise ValueError("record oracle contract does not match collection contract")
    contract_path = _spool_dir(path) / "contract.json"
    if not contract_path.exists():
        raise ValueError("record group storage was not prepared")
    contract = json.loads(contract_path.read_text())
    if (contract.get("oracle_contract_version") != ORACLE_CONTRACT_VERSION or
            int(contract.get("expected_siblings", -1)) != int(expected_siblings)):
        raise ValueError("record group does not match collection contract")
    group_id = next(iter(group_ids))
    digest = hashlib.sha1(group_id.encode()).hexdigest()[:16]
    first, second = (int(value) for value in order_key)
    spool = _spool_dir(path) / f"{first:06d}-{second:06d}-{digest}.jsonl"
    payload = "".join(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"),
            allow_nan=False) + "\n"
        for record in records)
    if spool.exists():
        if spool.read_text() != payload:
            raise ValueError(f"intervention group changed during resume: {group_id}")
        _rebuild_records_from_spools(
            path, sorted(_spool_dir(path).glob("*.jsonl")))
        return
    _atomic_write(spool, payload)
    spools = sorted(_spool_dir(path).glob("*.jsonl"))
    if spools[-1] != spool:
        _rebuild_records_from_spools(path, spools)
        return
    with path.open("a") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _record_group_id(record: dict) -> str:
    return str(record.get("collection_group_id") or
               (record.get("intervention") or {}).get("group_id") or "")


def _safe_publication_ready(full_by_radius) -> bool:
    if not full_by_radius:
        return False
    for physical in full_by_radius.values():
        clearance = physical.get("minimum_clearance_m")
        if (physical.get("collision") is not False or clearance is None or
                float(clearance) < config.BENCH_SAFE_CLEARANCE_M - 1e-9):
            return False
    return True


def full_geometry_publication_label(actions, full_by_radius) -> str | None:
    """Classify a candidate before spending work on sensor-sibling depth."""
    if not full_by_radius:
        return None
    collisions = {
        physical.get("collision") for physical in full_by_radius.values()}
    if collisions == {False}:
        return "safe" if _safe_publication_ready(full_by_radius) else None
    if collisions != {True}:
        return None
    nominal_forward_m = float(sum(
        action.m for action in actions if isinstance(action, Forward)))
    if nominal_forward_m <= 0.0:
        return None
    for physical in full_by_radius.values():
        contact_arc = physical.get("first_contact_arc_m")
        if contact_arc is None:
            return None
        if (nominal_forward_m - float(contact_arc) <
                config.BENCH_COLLISION_REMAINING_M - 1e-9):
            return None
    return "collision"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """The collection CLI. Extracted so its defaults are inspectable: the
    threshold pilot must reproduce this population, and a default that lives
    in two places is how the two drift apart."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenes", nargs="+")
    source.add_argument("--auto-scenes", action="store_true")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument(
        "--poses-per-scene", type=int, default=20,
        help=("persisted-pose target per scene; witness modes use a bounded "
              "multi-pose retry budget to fill it"))
    parser.add_argument(
        "--pose-candidates-per-scene", type=int,
        default=config.POSE_CANDIDATES_PER_SCENE,
        help=("independent position-and-heading candidates inspected per "
              "scene before concluding that a witness is unavailable"))
    parser.add_argument(
        "--record-idle-stop-s", type=_positive_float,
        help=("ordinary collector-side capacity stop after this many seconds "
              "without a newly accepted record; disabled when omitted"))
    parser.add_argument(
        "--scene-wallclock-stop-s", type=_positive_float,
        help=("ordinary collector-side active-scene wallclock stop measured "
              "after backend_ready; disabled when omitted"))
    parser.add_argument(
        "--collection-shard-id",
        help="stable identifier included in frame and intervention ids")
    parser.add_argument(
        "--pose-exclusions",
        help="JSON map of accepted source poses to avoid repeating")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radii", type=float, nargs="+", default=list(config.RADII_M))
    parser.add_argument("--camera-heights", type=float, nargs="+",
                        default=list(config.BENCH_CAMERA_HEIGHTS_M))
    parser.add_argument("--fov", type=float, nargs=2, action="append",
                        metavar=("HFOV", "VFOV"),
                        help="repeat for controlled FOV interventions")
    parser.add_argument(
        "--proposal-pairs-per-length", type=int,
        default=action_proposal.PAIRS_PER_LENGTH_DEFAULT,
        help=("depth-conditioned safe/collision pairs materialised per action "
              "length per pose"))
    parser.add_argument(
        "--proposal-natural-per-length", type=int,
        default=action_proposal.NATURAL_PER_LENGTH_DEFAULT,
        help=("dynamic-natural programs drawn per action length from this "
              "pose's initial depth budget, alongside boundary pairs"))
    parser.add_argument(
        "--ordinary-actions-per-pose", type=int,
        choices=config.ACTION_CANDIDATE_ORDINARY_LADDER,
        default=config.ACTION_CANDIDATE_ORDINARY_PER_POSE,
        help="stable ordinary publication budget selected by the canary")
    parser.add_argument(
        "--oracle-evaluation", choices=("perturbed", "nominal"), default="perturbed",
        help="nominal evaluates only the saved pose with the same dual oracles")
    parser.add_argument(
        "--action-mode", choices=["balanced", "file"], default="balanced",
        help="candidate source only; every output still uses the formal 1-6/3-3 selector")
    parser.add_argument("--action-file", help="candidate programs for --action-mode file")
    parser.add_argument("--lengths", type=int, nargs="+", default=None)
    parser.add_argument(
        "--keep-per-length", type=int, default=config.KEEP_PER_LENGTH,
        help="formal groups per length; current contract requires 1")
    parser.add_argument("--pool-factor", type=int, default=config.POOL_FACTOR)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument(
        "--semantic-query-workers", type=_positive_int, default=1,
        help=("scipy workers used for large semantic nearest-neighbour "
              "queries; execution-only and excluded from the run contract"))
    parser.add_argument("--debug-images", action="store_true")
    parser.add_argument("--debug-outcomes-per-frame", type=int, default=4)
    parser.add_argument("--out", required=True)
    output_policy = parser.add_mutually_exclusive_group()
    output_policy.add_argument("--resume", action="store_true",
                               help="resume from complete transactional groups")
    output_policy.add_argument("--overwrite", action="store_true",
                               help="explicitly replace an existing records file")
    output_policy.add_argument(
        "--append-records", metavar="JSONL",
        help="append into existing compact records; --out holds progress only")
    parser.add_argument(
        "--code-revision",
        help="Git commit the controller pinned this run to; recorded for "
             "provenance and enforced across --resume")
    parser.add_argument(
        "--allow-dirty-code", action="store_true",
        help="record that collection uses uncommitted local code changes")
    parser.add_argument(
        "--backend", choices=dataset_contracts.main_collection_datasets(),
        default="r2r",
        help=("R2R uses an MP3D episode whitelist; B1K and GS use explicit "
              "authenticated source manifests"))
    parser.add_argument(
        "--collection-mode", choices=config.CLI_COLLECTION_MODES, default="main",
        help="explicit current physical/search protocol for this shard")
    parser.add_argument(
        "--benchmark-partition",
        choices=scene_partitions.COLLECTABLE_PARTITIONS,
        default="train_seen",
        help="frozen benchmark scene partition selected for this shard")
    parser.add_argument(
        "--source-split", choices=dataset_contracts.OFFICIAL_SOURCE_SPLITS,
        default="train",
        help="official source catalog split consumed by this shard")
    parser.add_argument(
        "--r2r-episodes", "--r2r-train-episodes", dest="r2r_episodes",
        default=config.R2R_TRAIN_EPISODES,
        help="R2R episode manifest (legacy train-specific name is accepted)")
    parser.add_argument("--mp3d-root", default=config.MP3D_ROOT)
    parser.add_argument("--b1k-data-root")
    parser.add_argument("--b1k-source-manifest")
    parser.add_argument("--gs-data-root", default=config.GS_ROOT)
    parser.add_argument(
        "--gs-source-manifest", default=config.GS_TRAIN_MANIFEST)
    parser.add_argument(
        "--b1k-supervisor-contract-sha256", help=argparse.SUPPRESS)
    return parser
