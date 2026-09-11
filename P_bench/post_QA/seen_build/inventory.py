"""Scene-grouped inventory of immutable record inputs."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

from pipeline import a2, abc1_record, fixed_pose_actions, io_utils
from post_QA.seen_build import catalog, spec


PLAN_SCHEMA = "egoconseq.seen-build-plan.v2"
DEFAULT_LEGACY_ROOT = Path(__file__).resolve().parents[2] / \
    "data" / "metadata" / "train" / "seen"


def sources_from_manifest(records_root: Path) -> list[tuple[str, Path]]:
    """Resolve the record files declared by a source manifest."""
    return [(source.dataset, source.records_path)
            for source in catalog.load(records_root)]


def _source_entries(sources) -> list[tuple[str, Path]]:
    if isinstance(sources, Mapping):
        return [(str(dataset), Path(path))
                for dataset, path in sources.items()]
    return [(str(dataset), Path(path)) for dataset, path in sources]


def _common_fields(record: dict) -> dict:
    if abc1_record.is_compact(record):
        cases = list(record.get("cases") or [])
        radii = list(record.get("body_radii_m") or sorted({
            float(case["body_radius_m"]) for case in cases
        }))
        groups = list(dict.fromkeys(
            str(case["group_id"]) for case in cases))
        return {
            "scene_id": str(record["scene_id"]),
            "frame_id": str(record["frame_id"]),
            "schema_version": abc1_record.SCHEMA_VERSION,
            "pose": record["pose"],
            "body_radius_m": radii[0],
            "camera_height_m": float(
                record["sensor"]["nominal_camera_offset_m"]),
            "hfov_deg": float(record["sensor"]["hfov_deg"]),
            "vfov_deg": float(record["sensor"]["vfov_deg"]),
            "has_turn_first": any(
                case.get("starts_with") == "turn" for case in cases),
            "has_noncompliant_initial_turn": False,
            "c1_source_candidate": len({
                str(case["group_id"]) for case in cases
                if case.get("collision") is False and case.get("completed")
            }) >= 4,
            "action_group_capacity": int(record["action_group_capacity"]),
            "action_group_count": len(groups),
        }
    selection = record.get("selection") or {}
    outcomes = record.get("outcomes") or []
    radii = sorted(map(float, selection.get("required_radii_m") or {
        float(outcome["body"]["radius_m"])
        for outcome in outcomes if outcome.get("body")
    }))
    groups = selection.get("action_group_ids") or list(dict.fromkeys(
        str(outcome["action_group_id"]) for outcome in outcomes))
    selected = set(map(str, groups))
    first_by_group = {}
    for outcome in outcomes:
        group = str(outcome["action_group_id"])
        if group in selected:
            first_by_group.setdefault(group, outcome)
    turn_groups = [
        group for group, outcome in first_by_group.items()
        if fixed_pose_actions.compliant_turn_first(
            outcome.get("actions") or [])
    ]
    invalid_turn_groups = [
        group for group, outcome in first_by_group.items()
        if fixed_pose_actions.starts_with_turn(outcome.get("actions") or []) and
        not fixed_pose_actions.compliant_turn_first(
            outcome.get("actions") or [])
    ]
    safe_groups = {
        str(outcome["action_group_id"])
        for outcome in outcomes
        if str(outcome.get("action_group_id")) in selected and
        (outcome.get("physical") or {}).get("collision") is False and
        (outcome.get("execution") or {}).get("completed") is True
    }
    sensor = record["sensor"]
    frame_id = str(record["frame_id"])
    radius_index = int(hashlib.sha256(
        frame_id.encode()).hexdigest()[:8], 16) % len(radii)
    return {
        "scene_id": str(record["scene_id"]),
        "frame_id": frame_id,
        "schema_version": str(record["schema_version"]),
        "pose": record["pose"],
        "body_radius_m": radii[radius_index],
        "camera_height_m": float(sensor["nominal_camera_offset_m"]),
        "hfov_deg": float(sensor["hfov_deg"]),
        "vfov_deg": float(sensor["vfov_deg"]),
        "action_group_capacity": int(
            record.get("action_group_capacity") or len(groups) + 1),
        "action_group_count": len(groups),
        "has_turn_first": bool(turn_groups),
        "has_noncompliant_initial_turn": bool(invalid_turn_groups),
        "c1_source_candidate": len(safe_groups) >= 4,
    }


RECORD_ADAPTERS: dict[str, Callable[[dict], dict]] = {
    "conseq.v11": _common_fields,
    "conseq.v18": _common_fields,
    abc1_record.SCHEMA_VERSION: _common_fields,
}


def _source_digest(path: Path) -> str:
    meta_path = Path(path).parent / "run_meta.json"
    if meta_path.is_file():
        digest = json.loads(meta_path.read_text(encoding="utf-8")).get(
            "records_sha256")
        if digest:
            return str(digest)
    return io_utils.sha256_file(path)


def _scan_source(
        dataset: str, path: Path, *, source_records_sha256: str,
        progress=None) -> list[dict]:
    rows = []
    meta_path = Path(path).with_name("run_meta.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    params = metadata.get("params") or {}
    manifest_key = {"gs": "gs_source_manifest", "b1k": "b1k_source_manifest",
                    "r2r": "r2r_episodes"}[dataset]
    scene_source = metadata.get("scene_source") or {
        "split": metadata.get("source_split", "train"),
        "manifest": params.get(manifest_key),
    }
    with Path(path).open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            payload = line.rstrip(b"\r\n")
            if not payload:
                continue
            record = json.loads(payload)
            fields = RECORD_ADAPTERS[str(record["schema_version"])](record)
            record_uid = (
                str(record["record_uid"])
                if abc1_record.is_compact(record) else
                abc1_record.historical_record_uid(
                    dataset, source_records_sha256, offset))
            rows.append({
                "kind": "record",
                "dataset": str(dataset),
                "record_uid": record_uid,
                "source_path": str(Path(path).resolve()),
                "byte_offset": offset,
                "source_record_sha256": hashlib.sha256(payload).hexdigest(),
                "scene_source": scene_source,
                **({"observation_profile": record["observation_profile"]}
                   if record.get("observation_profile") else {}),
                **fields,
            })
            if progress is not None and len(rows) % 1000 == 0:
                progress(dataset, len(rows))
    return rows


def _scene_workers(rows: Iterable[dict]) -> dict[tuple[str, str], str]:
    counts = Counter(
        (str(row["dataset"]), str(row["scene_id"])) for row in rows)
    result = {}
    for dataset in sorted({key[0] for key in counts}):
        scenes = [(scene, counts[(dataset, scene)])
                  for ds, scene in counts if ds == dataset]
        if dataset != "b1k":
            result.update({(dataset, scene): dataset for scene, _ in scenes})
            continue
        loads = [0, 0]
        for scene, count in sorted(scenes, key=lambda item: (-item[1], item[0])):
            worker = min(range(2), key=lambda index: (loads[index], index))
            result[(dataset, scene)] = f"b1k-{worker}"
            loads[worker] += count
    return result


def _dataset_order(dataset: str) -> tuple[int, str]:
    try:
        return spec.DATASETS.index(dataset), dataset
    except ValueError:
        return len(spec.DATASETS), dataset


def _weighted_lengths(count: int) -> list[int]:
    """Spend the C1 budget where long programs are actually scarce."""
    weights = {2: 80, 4: 120, 6: 1100}
    targets = {
        length: int(count) * weight // sum(weights.values())
        for length, weight in weights.items()
    }
    for length in (6, 4, 2)[:int(count) - sum(targets.values())]:
        targets[length] += 1
    used = Counter()
    schedule = []
    while len(schedule) < int(count):
        length = min(
            (value for value in (2, 4, 6) if used[value] < targets[value]),
            key=lambda value: (used[value] / targets[value], value))
        used[length] += 1
        schedule.append(length)
    return schedule


def _shortfall_rows(report: Mapping | None, dataset: str,
                    task: str) -> list[dict]:
    if report is None:
        return []
    return [
        row for row in report.get("slots") or []
        if row.get("slot", [None, None])[:2] == [dataset, task] and
        int(row.get("shortfall") or 0) > 0
    ]


def _assign_c1_attempts(
        rows: list[dict], count: int,
        repair_report: Mapping | None = None) -> Counter:
    assigned = Counter()
    for row in rows:
        row["collect_c1"] = False
        row["c1_length"] = None
    for dataset in spec.DATASETS:
        by_scene = defaultdict(list)
        for row in rows:
            if (row["dataset"] == dataset and
                    row["c1_source_candidate"] and
                    int(row["action_group_capacity"]) >= 6):
                by_scene[row["scene_id"]].append(row)
        queues = {
            scene: deque(sorted(
                values, key=lambda row: row["source_record_sha256"]))
            for scene, values in by_scene.items()
        }
        shortages = _shortfall_rows(repair_report, dataset, "C1")
        if repair_report is not None and not shortages:
            continue
        if shortages:
            lengths = []
            retries = {2: 2, 4: 3, 6: 15}
            for shortage in shortages:
                length = int(shortage["slot"][2])
                lengths.extend(
                    [length] * int(shortage["shortfall"]) * retries[length])
        else:
            lengths = _weighted_lengths(count)
        chosen = []
        while queues and len(chosen) < len(lengths):
            for scene in sorted(tuple(queues)):
                chosen.append(queues[scene].popleft())
                if not queues[scene]:
                    del queues[scene]
                if len(chosen) == len(lengths):
                    break
        for row, length in zip(chosen, lengths):
            row["collect_c1"] = True
            row["c1_length"] = length
            assigned[dataset] += 1
    return assigned


def _scene_spread(rows: Iterable[dict]) -> list[dict]:
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_id"]].append(row)
    queues = {
        scene: deque(sorted(
            values, key=lambda row: row["source_record_sha256"]))
        for scene, values in by_scene.items()
    }
    result = []
    while queues:
        for scene in sorted(tuple(queues)):
            result.append(queues[scene].popleft())
            if not queues[scene]:
                del queues[scene]
    return result


def _assign_a2_attempts(
        rows: list[dict], factor: int,
        repair_report: Mapping | None = None) -> Counter:
    assigned = Counter()
    eligible = ((repair_report or {}).get("eligible_record_uids"))
    eligible = None if eligible is None else set(eligible)
    for row in rows:
        row["collect_a2"] = False
        row["a2_length"] = None
        row["a2_starts_with"] = None
        row["a2_cell"] = None
    for dataset in spec.DATASETS:
        slots = []
        shortages = _shortfall_rows(repair_report, dataset, "A2")
        if repair_report is not None and not shortages:
            continue
        if shortages:
            retries = {"forward": 3, "turn": 10}
            for shortage in shortages:
                _ds, _task, length, start, ordinal, rank = shortage["slot"]
                multiplier = retries[start] * ({4: 1, 5: 2, 6: 5}[int(length)]
                                               if start == "turn" else 1)
                cell = a2.A2Cell(int(ordinal), str(rank))
                slots.extend(
                    (int(length), str(start), cell)
                    for _ in range(int(shortage["shortfall"]) * multiplier))
        else:
            for length, starts in spec.A2_START_TOTALS[dataset].items():
                for start, count in starts.items():
                    if not count:
                        continue
                    multiplier = int(factor)
                    if start == "turn":
                        multiplier *= {4: 5, 5: 10, 6: 25}[int(length)]
                    base = a2.balanced_cells(
                        length, count, starts_with=start,
                        rotation=spec.DATASETS.index(dataset) + length)
                    slots.extend(
                        (length, start, cell)
                        for cell in base for _ in range(multiplier))
        slots = [value for _index, value in sorted(
            enumerate(slots), key=lambda item: hashlib.sha256(
                repr((dataset, item[0], item[1])).encode()).hexdigest())]
        candidates = _scene_spread(
            row for row in rows
            if row["dataset"] == dataset and not row["collect_c1"] and
            (eligible is None or row["record_uid"] in eligible))
        for row, (length, start, cell) in zip(candidates, slots):
            row["collect_a2"] = True
            row["a2_length"] = length
            row["a2_starts_with"] = start
            row["a2_cell"] = {
                "forward_ordinal_1based": cell.forward_ordinal_1based,
                "distance_rank": cell.distance_rank,
                "collision_action_index_1based": a2.collision_action_index(
                    cell, length=length, starts_with=start),
            }
            assigned[dataset] += 1
    return assigned


def _assign_general_turn_repairs(
        rows: list[dict], repair_report: Mapping | None) -> Counter:
    assigned = Counter()
    for row in rows:
        row["force_turn"] = False
    if repair_report is None:
        return assigned
    for dataset in spec.DATASETS:
        missing = sum(
            int(row["shortfall"])
            for task in ("A1", "A3", "A4", "B1", "B2")
            for row in _shortfall_rows(repair_report, dataset, task)
            if "turn" in row["slot"])
        candidates = _scene_spread(
            row for row in rows
            if row["dataset"] == dataset and not row["collect_c1"] and
            not row["collect_a2"])
        for row in candidates[:missing * 5]:
            row["force_turn"] = True
            assigned[dataset] += 1
    return assigned


def write_plan(
        sources, output_path: Path, *,
        split: str = "train/seen",
        c1_attempts_per_dataset: int = 1300,
        a2_attempt_factor: int = 2,
        repair_report: Mapping | None = None, progress=None) -> dict:
    """Scan each source once and write records contiguously by scene."""
    source_entries = _source_entries(sources)
    source_digests = {
        (dataset, str(path.resolve())): _source_digest(path)
        for dataset, path in source_entries
    }
    rows = [
        row
        for dataset, path in source_entries
        for row in _scan_source(
            str(dataset), Path(path),
            source_records_sha256=source_digests[
                (dataset, str(path.resolve()))],
            progress=progress)
    ]
    workers = _scene_workers(rows)
    for row in rows:
        row["worker"] = workers[(row["dataset"], row["scene_id"])]
    c1_attempts = _assign_c1_attempts(
        rows, c1_attempts_per_dataset, repair_report)
    a2_attempts = _assign_a2_attempts(
        rows, a2_attempt_factor, repair_report)
    turn_attempts = _assign_general_turn_repairs(rows, repair_report)
    rows.sort(key=lambda row: (
        _dataset_order(row["dataset"]), row["scene_id"],
        row["byte_offset"]))
    dataset_counts = Counter(row["dataset"] for row in rows)
    summary = {
        "records": len(rows),
        "scenes": len(workers),
        "datasets": dict(sorted(dataset_counts.items())),
    }
    plan_identity = {
        "schema": PLAN_SCHEMA,
        "split": split,
        "sources": [
            {
                "dataset": str(dataset),
                "path": str(Path(path).resolve()),
                "records_sha256": source_digests[
                    (dataset, str(Path(path).resolve()))],
            }
            for dataset, path in sorted(source_entries)
        ],
        "c1_attempts_per_dataset": int(c1_attempts_per_dataset),
        "a2_attempt_factor": int(a2_attempt_factor),
        "repair_shortfalls": [] if repair_report is None else [
            {"slot": row["slot"], "shortfall": row["shortfall"]}
            for row in repair_report.get("slots") or []
            if int(row.get("shortfall") or 0) > 0
        ],
        "eligible_record_uids": (repair_report or {}).get("eligible_record_uids"),
    }
    plan_id = hashlib.sha256(json.dumps(
        plan_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for index, row in enumerate(rows):
        row["plan_id"] = plan_id
        row["plan_row_id"] = f"row-{index:06d}"
    header = {
        "kind": "header",
        "schema": PLAN_SCHEMA,
        "split": split,
        "repair_only": repair_report is not None,
        "plan_id": plan_id,
        **summary,
        "sources": plan_identity["sources"],
        "c1_attempts": dict(sorted(c1_attempts.items())),
        "a2_attempts": dict(sorted(a2_attempts.items())),
        "turn_attempts": dict(sorted(turn_attempts.items())),
    }

    def write(stream):
        for value in (header, *rows):
            stream.write((json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                allow_nan=False) + "\n").encode("utf-8"))

    io_utils.atomic_write_binary(output_path, write)
    return summary


def _surface_plan_id(source_entries, source_digests, seed: int) -> tuple[str, dict]:
    identity = {
        "schema": PLAN_SCHEMA,
        "mode": "surfaces",
        "seed": int(seed),
        "sources": [{
            "dataset": str(dataset),
            "path": str(Path(path).resolve()),
            "records_sha256": source_digests[
                (str(dataset), str(Path(path).resolve()))],
        } for dataset, path in sorted(source_entries)],
    }
    plan_id = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan_id, identity


def _legacy_b1k_profiles(legacy_root: Path) -> dict[str, dict]:
    """Index the original B1K render profile once by immutable record UID."""
    root = Path(legacy_root).resolve()
    if not root.exists():
        return {}
    sources = (sources_from_manifest(root)
               if (root / "manifest.json").is_file() else
               [("b1k", root / "b1k" / "records.jsonl")])
    profiles = {}
    for dataset, path in sources:
        if str(dataset) != "b1k" or not Path(path).is_file():
            continue
        digest = _source_digest(Path(path))
        with Path(path).open("rb") as stream:
            while True:
                offset = stream.tell()
                payload = stream.readline()
                if not payload:
                    break
                if not payload.strip():
                    continue
                record = json.loads(payload)
                contract = record.get("collection_contract") or {}
                version = contract.get("version")
                profile_sha256 = contract.get("observation_profile_sha256")
                if version and profile_sha256:
                    uid = abc1_record.historical_record_uid(
                        "b1k", digest, offset)
                    profiles[uid] = {
                        "collection_contract_version": str(version),
                        "sha256": str(profile_sha256),
                    }
    return profiles


def write_surface_plan(
        sources, output_path: Path, *, seed: int = 20260906,
        legacy_root: Path = DEFAULT_LEGACY_ROOT, progress=None) -> dict:
    """Write one immutable all-record surface-refinement plan."""
    source_entries = _source_entries(sources)
    source_digests = {
        (str(dataset), str(path.resolve())): _source_digest(path)
        for dataset, path in source_entries
    }
    plan_id, identity = _surface_plan_id(
        source_entries, source_digests, seed)
    output_path = Path(output_path)
    if output_path.exists():
        header, rows = read_plan(output_path)
        if header.get("mode") != "surfaces" or header.get("plan_id") != plan_id:
            raise FileExistsError(
                f"refusing to replace immutable plan: {output_path}")
        return {
            "records": len(rows),
            "scenes": len({(row["dataset"], row["scene_id"])
                           for row in rows}),
            "datasets": dict(sorted(Counter(
                str(row["dataset"]) for row in rows).items())),
        }

    profiles = None
    rows = []
    for dataset, path in source_entries:
        dataset = str(dataset)
        digest = source_digests[(dataset, str(path.resolve()))]
        dataset_rows = []
        with Path(path).open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                payload = line.rstrip(b"\r\n")
                if not payload:
                    continue
                source_record = json.loads(payload)
                if not abc1_record.is_compact(source_record):
                    raise ValueError(
                        "surface plans require compact record inputs")
                uid = str(source_record["record_uid"])
                radii = sorted(map(
                    float, source_record.get("body_radii_m") or []))
                if not radii:
                    raise ValueError(
                        f"surface record has no body radius: {uid}")
                row = {
                    "kind": "record", "dataset": dataset,
                    "record_uid": uid,
                    "source_path": str(Path(path).resolve()),
                    "byte_offset": offset,
                    "source_record_sha256": hashlib.sha256(payload).hexdigest(),
                    "scene_id": str(source_record["scene_id"]),
                    "frame_id": str(source_record["frame_id"]),
                    "body_radius_m": radii[
                        _seed_index(seed, uid, len(radii))],
                }
                if dataset == "b1k" and source_record.get(
                        "observation_profile") is not None:
                    row["observation_profile"] = source_record[
                        "observation_profile"]
                dataset_rows.append(row)
                if progress is not None and len(dataset_rows) % 1000 == 0:
                    progress(dataset, len(dataset_rows))
        if dataset == "b1k" and any(
                "observation_profile" not in row for row in dataset_rows):
            if profiles is None:
                profiles = _legacy_b1k_profiles(legacy_root)
            for row in dataset_rows:
                if "observation_profile" not in row:
                    profile = profiles.get(str(row["record_uid"]))
                    if profile is None:
                        raise ValueError(
                            "B1K observation profile is unavailable: "
                            f"{row['record_uid']}")
                    row["observation_profile"] = profile
        rows.extend(dataset_rows)
    workers = _scene_workers(rows)
    for row in rows:
        row["worker"] = workers[(row["dataset"], row["scene_id"])]
    rows.sort(key=lambda row: (
        _dataset_order(row["dataset"]), row["scene_id"], row["byte_offset"]))
    for index, row in enumerate(rows):
        row["plan_id"] = plan_id
        row["plan_row_id"] = f"row-{index:06d}"
    summary = {
        "records": len(rows), "scenes": len(workers),
        "datasets": dict(sorted(Counter(
            str(row["dataset"]) for row in rows).items())),
    }
    header = {
        "kind": "header", "schema": PLAN_SCHEMA, "mode": "surfaces",
        "repair_only": False, "plan_id": plan_id, "seed": int(seed),
        **summary, "sources": identity["sources"],
    }

    def write(stream):
        for value in (header, *rows):
            stream.write((json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                allow_nan=False) + "\n").encode("utf-8"))

    io_utils.atomic_write_binary(output_path, write)
    return summary


def _seed_index(seed: int, record_uid: str, size: int) -> int:
    digest = hashlib.sha256(
        f"{int(seed)}:surface-radius:{record_uid}".encode()).hexdigest()
    return int(digest[:16], 16) % int(size)


def read_plan(path: Path) -> tuple[dict, list[dict]]:
    values = io_utils.read_jsonl(path, require_dict=True)
    if not values or values[0].get("schema") != PLAN_SCHEMA:
        raise ValueError("not a seen-build plan")
    header, rows = values[0], values[1:]
    return header, rows


def records_by_scene(rows: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["scene_id"]))].append(row)
    return dict(grouped)
