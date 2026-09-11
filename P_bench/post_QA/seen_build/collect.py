"""Run one scene at a time and append one terminal row per record."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from pipeline import abc1_record, config, fixed_pose_actions, io_utils
from post_QA.seen_build import inventory


RESULT_SCHEMA = "egoconseq.seen-build-result.v2"
DEFAULT_MP3D_ROOT = Path(
    "/home/zhangshan/syp/datasets/scene_datasets/mp3d")
DEFAULT_GS_ROOT = Path("/home/zhangshan/syp/datasets/gs")
DEFAULT_GS_MANIFEST = DEFAULT_GS_ROOT / "splits/train.json"
DEFAULT_B1K_MANIFEST = Path(
    "/home/zhangshan/syp/datasets/behavior-1k-v3.9.1/"
    "pbench-abc1-task4/catalog-authority-audit-v2-20260812/"
    "audit-derived-source-manifest.json")


def _load_records(rows: list[dict]) -> list[tuple[dict, dict]]:
    streams = {}
    result = []
    try:
        for row in rows:
            path = str(row["source_path"])
            stream = streams.get(path)
            if stream is None:
                stream = Path(path).open("rb")
                streams[path] = stream
            stream.seek(int(row["byte_offset"]))
            payload = stream.readline().rstrip(b"\r\n")
            record = json.loads(payload)
            if str(record["frame_id"]) != str(row["frame_id"]):
                raise ValueError(f"plan offset changed: {row['record_uid']}")
            if hashlib.sha256(payload).hexdigest() != str(
                    row["source_record_sha256"]):
                raise ValueError(f"source record changed: {row['record_uid']}")
            result.append((row, record))
    finally:
        for stream in streams.values():
            stream.close()
    return result


def run_worker(
        plan_path: Path, worker: str, output_path: Path, *,
        open_scene: Callable, process_record: Callable) -> dict:
    """Process every assigned record before closing and advancing a scene."""
    header, plan_rows = inventory.read_plan(plan_path)
    rows = [row for row in plan_rows if row["worker"] == str(worker)]
    output_path = Path(output_path)
    plan_id = str(header["plan_id"])
    planned = {str(row["plan_row_id"]): row for row in rows}
    completed = set()
    for result in io_utils.read_jsonl(
            output_path, missing_ok=True, require_dict=True):
        row_id = str(result.get("plan_row_id"))
        row = planned.get(row_id)
        if (row is not None and result.get("schema") == RESULT_SCHEMA and
                result.get("plan_id") == plan_id and
                result.get("record_uid") == row["record_uid"] and
                result.get("source_record_sha256") ==
                row["source_record_sha256"]):
            completed.add(row_id)
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[(str(row["dataset"]), str(row["scene_id"]))].append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", buffering=1) as stream:
        for scene_index, ((dataset, scene_id), scene_rows) in enumerate(
                by_scene.items(), 1):
            remaining = [row for row in scene_rows
                         if str(row["plan_row_id"]) not in completed]
            if not remaining:
                continue
            print(
                f"[collect:{worker}] scene {scene_index}/{len(by_scene)} "
                f"{dataset}/{scene_id}: {len(remaining)} records",
                flush=True)
            active = [row for row in remaining if not header.get("repair_only")
                      or any(row.get(key) for key in
                             ("collect_c1", "collect_a2", "force_turn"))]
            records = _load_records(active if header.get("repair_only") else scene_rows)
            session = open_scene(dataset, scene_id, records) if active else None
            try:
                records_by_row = {
                    str(source_row["plan_row_id"]): record
                    for source_row, record in records}
                for row in remaining:
                    if header.get("repair_only") and not any(
                            row.get(key) for key in
                            ("collect_c1", "collect_a2", "force_turn")):
                        result = {"mode": "reused"}
                    else:
                        result = process_record(
                            session, {**row, "repair_only": header.get("repair_only", False)},
                            records_by_row[str(row["plan_row_id"])])
                    value = {
                        "schema": RESULT_SCHEMA,
                        "plan_id": plan_id,
                        "plan_row_id": str(row["plan_row_id"]),
                        "record_uid": str(row["record_uid"]),
                        "source_record_sha256": str(
                            row["source_record_sha256"]),
                        "dataset": dataset,
                        "scene_id": scene_id,
                        "frame_id": str(row["frame_id"]),
                        "status": "complete" if result is not None else
                                  "unavailable",
                    }
                    if result is not None:
                        value.update(result)
                    stream.write(json.dumps(
                        value, sort_keys=True, separators=(",", ":"),
                        allow_nan=False) + "\n")
                    completed.add(str(row["plan_row_id"]))
            finally:
                if session is not None:
                    session.close()
            print(
                f"[collect:{worker}] completed {len(completed)}/{len(rows)}",
                flush=True)
    return {
        "worker": str(worker),
        "planned": len(rows),
        "completed": len(completed.intersection(
            str(row["plan_row_id"]) for row in rows)),
    }


def _source_assets(source: dict):
    from pipeline.scene_pool import SourceAssetIdentity

    return tuple(SourceAssetIdentity(
        role=str(value["role"]), byte_size=int(value["bytes"]),
        sha256=str(value["sha256"]))
        for value in source["source_assets"])


def _scene_spec(dataset: str, record: dict, *, gs_root: Path,
                gs_manifest: Path, b1k_manifest: Path,
                b1k_entries: dict, b1k_manifest_sha256: str):
    from pipeline.scene_pool import SceneSpec

    source = record["source"]
    scene_id = str(record["scene_id"])
    assets = _source_assets(source)
    if dataset == "b1k":
        scene_path = str(Path(record["scene_glb"]).resolve())
        return SceneSpec(
            scene_id=scene_id, source_dataset="b1k",
            official_split="train", scene_path=scene_path,
            navmesh_path=scene_path, semantic_path=scene_path,
            semantic_metadata_path=None,
            semantic_format=str(source["semantic_format"]),
            scene_dataset_config=None,
            provenance_path=str(Path(b1k_manifest).resolve()),
            provenance_sha256=b1k_manifest_sha256,
            source_assets=assets,
            source_assets_sha256=str(source["source_assets_sha256"]),
            split_authority="project_defined",
            b1k_scene_authority=dict(
                b1k_entries[scene_id]["scene_authority"]))
    directory = Path(gs_root).resolve() / "train" / scene_id
    return SceneSpec(
        scene_id=scene_id, source_dataset="gs", official_split="train",
        scene_path=str(Path(record["scene_glb"]).resolve()),
        navmesh_path=str((directory / "scene.navmesh").resolve()),
        semantic_path=str((directory / "labels.json").resolve()),
        semantic_metadata_path=None,
        semantic_format=str(source["semantic_format"]),
        scene_dataset_config=None,
        provenance_path=str(Path(gs_manifest).resolve()),
        provenance_sha256=str(source["source_manifest_sha256"]),
        source_assets=assets,
        source_assets_sha256=str(source["source_assets_sha256"]),
        collision_authority_path=str(
            (directory / "scene.collision.npz").resolve()))


def make_scene_opener(
        *, mp3d_root: Path = DEFAULT_MP3D_ROOT,
        gs_root: Path = DEFAULT_GS_ROOT,
        gs_manifest: Path = DEFAULT_GS_MANIFEST,
        b1k_manifest: Path = DEFAULT_B1K_MANIFEST):
    """Create the one scene loader shared by all collection workers."""
    b1k_cache = {}

    def compact_scene(dataset: str, scene_id: str, row: dict):
        from pipeline import dataset_contracts
        from pipeline.scene_pool import (
            SceneSpec, discover_b1k_train_scenes,
            discover_gs_scenes)

        source = row.get("scene_source") or {}
        source_split = source.get("split", "train")

        if dataset == "r2r":
            root = Path(mp3d_root).resolve()
            directory = root / scene_id
            manifest = Path(source.get("manifest") or config.R2R_TRAIN_EPISODES).resolve()
            return SceneSpec(
                scene_id=scene_id, source_dataset="r2r",
                official_split=source_split,
                scene_path=str(directory / f"{scene_id}.glb"),
                navmesh_path=str(directory / f"{scene_id}.navmesh"),
                semantic_path=str(directory / f"{scene_id}_semantic.ply"),
                semantic_metadata_path=str(directory / f"{scene_id}.house"),
                semantic_format=dataset_contracts.dataset_source_contract(
                    "r2r").semantic_format,
                scene_dataset_config=str(
                    root / "mp3d_annotated_basis.scene_dataset_config.json"),
                provenance_path=str(manifest),
                provenance_sha256=io_utils.sha256_file(manifest))
        if dataset == "gs":
            return discover_gs_scenes(
                gs_root, source.get("manifest") or gs_manifest,
                requested=(scene_id,), source_split=source_split)[0]
        if "specs" not in b1k_cache:
            manifest = Path(source.get("manifest") or b1k_manifest).resolve()
            b1k_root = manifest.parents[2]
            b1k_cache["specs"] = {
                value.scene_id: value for value in discover_b1k_train_scenes(
                    b1k_root, manifest)}
        return b1k_cache["specs"][scene_id]

    def prepare_compact(record: dict, scene, row: dict) -> None:
        record["scene_glb"] = scene.scene_path
        record["source"] = scene.provenance()
        # This legacy oracle field is plane-relative, unlike the public
        # camera-to-ground height. Keep the two definitions separate.
        plane = record["floor_plane"]
        record["camera_height_above_visible_floor_m"] = (
            plane["normal_local"][1] * record["sensor"]["nominal_camera_offset_m"]
            + plane["offset_m"])
        if scene.source_dataset == "b1k":
            from pipeline import record as record_fields
            profile = row.get("observation_profile") or {}
            version = profile.get("collection_contract_version") or \
                record_fields.B1K_OFFICIAL_RENDER_COLLECTION_CONTRACT_VERSION
            record["collection_contract"] = {
                "version": str(version),
                "observation_profile_sha256": profile.get("sha256"),
                "c1_render_mode": record_fields.B1K_C1_RENDER_MODE_BY_CONTRACT[
                    str(version)],
            }

    def open_scene(dataset: str, _scene_id: str, rows_and_records):
        records = [record for _row, record in rows_and_records]
        first = records[0]
        heights = sorted({float(record["sensor"][
            "nominal_camera_offset_m"]) for record in records})
        fovs = sorted({(
            float(record["sensor"]["hfov_deg"]),
            float(record["sensor"]["vfov_deg"])) for record in records})
        if abc1_record.is_compact(first):
            scene_spec = compact_scene(dataset, str(first["scene_id"]), rows_and_records[0][0])
            for row, record in rows_and_records:
                prepare_compact(record, scene_spec, row)
            first = records[0]
        else:
            scene_spec = None
        if dataset == "r2r":
            from pipeline.sim import SimSession

            return SimSession(
                str(Path(first["scene_glb"]).resolve()),
                scene_dataset_cfg=str((Path(mp3d_root) /
                    "mp3d_annotated_basis.scene_dataset_config.json").resolve()),
                semantic_format=str(first["source"]["semantic_format"]),
                source_dataset="r2r", official_split=str(first["source"]["official_split"]),
                heights=heights, fovs=fovs, semantic_query_workers=1,
                gpu_device_id=int(os.environ.get(
                    "PBENCH_HABITAT_GPU_DEVICE_ID", "0")))
        if dataset == "b1k" and not b1k_cache:
            document = json.loads(
                Path(b1k_manifest).read_text(encoding="utf-8"))
            b1k_cache["entries"] = {
                str(value["scene_id"]): value
                for value in document["scenes"]}
            b1k_cache["sha256"] = io_utils.sha256_file(b1k_manifest)
        if scene_spec is None:
            scene_spec = _scene_spec(
                dataset, first, gs_root=gs_root, gs_manifest=gs_manifest,
                b1k_manifest=b1k_manifest,
                b1k_entries=b1k_cache.get("entries", {}),
                b1k_manifest_sha256=str(b1k_cache.get("sha256", "")))
        if dataset == "gs":
            from pipeline.gs_sim import GsSimSession

            return GsSimSession(scene_spec, heights=heights)
        from pipeline.b1k_sim import B1KSimSession

        contract = (first.get("collection_contract") or {}).get("version")
        return B1KSimSession(
            scene_spec, heights=heights, fovs=fovs,
            contract_version=contract)

    return open_scene


def _build_frame(sim, record: dict):
    from pipeline.frame import build_frame
    from pipeline import record as record_fields
    from pipeline.floor_plane import FloorPlaneEstimate

    pose = record["pose"]
    sensor = record["sensor"]
    return build_frame(
        sim, pose["position"], float(pose["yaw_rad"]),
        frame_id=str(record["frame_id"]),
        scene_id=str(record["scene_id"]),
        scene_glb=str(record["scene_glb"]),
        floor_plane=(
            FloorPlaneEstimate.from_json(record["floor_plane"])
            if abc1_record.is_compact(record) else
            record_fields.require_floor_plane(record)),
        cam_h=float(sensor["nominal_camera_offset_m"]),
        hfov=float(sensor["hfov_deg"]), vfov=float(sensor["vfov_deg"]))


def _seed(seed: int, record_id: str) -> int:
    return int(hashlib.sha256(
        f"{int(seed)}:{record_id}".encode()).hexdigest()[:16], 16)


def _render_c1_assets(
        sim, frame, record: dict, outcomes: list[dict], *,
        stage_root: Path) -> list[dict] | None:
    from pipeline import collection_assets, future_view_selection
    from pipeline.collection_support import (
        terminal_rgb_batch_renderer, terminal_rgb_renderer)

    cache = {}
    dataset = str(record["source"]["source_dataset"])
    transaction = None
    if dataset == "b1k":
        transaction = str((record.get(
            "collection_contract") or {}).get("c1_render_mode") or "")
        poses = [tuple(
            future_view_selection._terminal_checkpoint(outcome)[key]
            for key in ("x", "z", "heading_deg"))
            for outcome in outcomes]
        terminal_rgb_batch_renderer(
            sim, frame, cache,
            render_transaction=transaction)(poses)
    counts = collection_assets.attach_terminal_rgb_assets(
        stage_root, frame, outcomes, cache, source=record["source"],
        collection_contract=record.get("collection_contract"),
        terminal_renderer=(None if dataset == "b1k" else
                           terminal_rgb_renderer(sim, frame, cache)),
        eligible_outcome_ids={str(outcome["outcome_id"])
                              for outcome in outcomes},
        render_transaction=transaction)
    if counts["materialized"] != len(outcomes):
        return None
    return [{
        "path": str(outcome["terminal_rgb_asset"]["path"]),
        "source": str((stage_root / str(
            outcome["terminal_rgb_asset"]["path"])).resolve()),
    } for outcome in outcomes]


def _collect_c1_family(
        sim, frame, record: dict, row: dict, *, stage_root: Path,
        seed: int, max_programs: int,
        max_full_attempts: int) -> dict | None:
    from pipeline import action_proposal, c1_counterfactual
    from pipeline import future_view_selection

    radius = float(row["body_radius_m"])
    sim.recompute_navmesh(radius, height=config.GROUND_ORACLE_HEIGHT_M)
    nav = sim.nav(record["pose"]["position"], float(record["pose"]["yaw_rad"]))
    proxy = action_proposal.FrameDepthProxy(frame, radius)
    candidates = fixed_pose_actions.proxy_shortlist(
        proxy, half_fov_deg=float(record["sensor"]["hfov_deg"]) / 2.0,
        seed=seed, max_programs=max_programs,
        shortlist=max_full_attempts,
        desired_collision=False, lengths=(int(row["c1_length"]),))
    existing = {
        str(value.get("action_group_id") or value.get("group_id"))
        for value in ((record.get("outcomes") or []) +
                      (record.get("cases") or []))
    }
    for query_actions, query_verdict in candidates:
        query = fixed_pose_actions.certify_program(
            frame, record, query_actions, nav=nav,
            radius_m=radius, proxy_verdict=query_verdict)
        if query is None:
            continue
        outcomes = list(query["outcomes"])
        provenance = dict(query["provenance"])
        terminal_keys = {
            future_view_selection.terminal_render_cache_key(outcomes[0])}
        for index, neighbor in enumerate(
                c1_counterfactual.counterfactual_neighbors(query_actions)):
            if neighbor.tag in existing or neighbor.tag in provenance:
                continue
            verdict = fixed_pose_actions.proxy_verdict(
                proxy, neighbor.actions,
                half_fov_deg=float(record["sensor"]["hfov_deg"]) / 2.0,
                desired_collision=False)
            if verdict is None:
                continue
            delta = fixed_pose_actions.certify_program(
                frame, record, neighbor.actions, nav=nav,
                radius_m=radius, proxy_verdict=verdict)
            if delta is None:
                continue
            outcome = delta["outcomes"][0]
            key = future_view_selection.terminal_render_cache_key(outcome)
            if key in terminal_keys:
                continue
            terminal_keys.add(key)
            outcomes.append(outcome)
            provenance[neighbor.tag] = c1_counterfactual.neighbor_provenance(
                neighbor, query_actions, index=index)
            if len(outcomes) == 4:
                break
        if len(outcomes) != 4:
            continue
        staged_assets = _render_c1_assets(
            sim, frame, record, outcomes, stage_root=stage_root)
        if staged_assets is not None:
            return {
                "outcomes": outcomes, "provenance": provenance,
                "staged_assets": staged_assets,
            }
    return None


def _collect_a2_group(
        sim, frame, record: dict, *, row: dict, seed: int,
        max_programs: int, max_full_attempts: int) -> dict | None:
    from pipeline import a2, action_proposal

    radius = float(row["body_radius_m"])
    sim.recompute_navmesh(radius, height=config.GROUND_ORACLE_HEIGHT_M)
    nav = sim.nav(record["pose"]["position"], float(record["pose"]["yaw_rad"]))
    proxy = action_proposal.FrameDepthProxy(frame, radius)
    cell_value = row["a2_cell"]
    cell = a2.A2Cell(
        int(cell_value["forward_ordinal_1based"]),
        str(cell_value["distance_rank"]))
    attempts = 0
    for proposal in a2.collision_proposals(
            proxy, length=int(row["a2_length"]), cell=cell,
            seed=seed, half_fov_deg=float(record["sensor"]["hfov_deg"]) / 2.0,
            max_programs=max_programs,
            starts_with=str(row["a2_starts_with"])):
        verdict = fixed_pose_actions.proxy_verdict(
            proxy, proposal.collision_actions,
            half_fov_deg=float(record["sensor"]["hfov_deg"]) / 2.0,
            desired_collision=True)
        if verdict is None:
            continue
        attempts += 1
        delta = fixed_pose_actions.certify_program(
            frame, record, proposal.collision_actions, nav=nav,
            radius_m=radius, proxy_verdict=verdict)
        if delta is not None:
            for metadata in delta["provenance"].values():
                metadata["variant"] = action_proposal.A2_RANK_COLLISION_VARIANT
            return delta
        if attempts >= int(max_full_attempts):
            break
    return None


def make_record_processor(
        *, seed: int, max_programs: int,
        max_full_attempts: int,
        stage_root: Path | None = None):
    """Create the fixed-pose per-record collector used by every dataset."""
    def process(session, row: dict, record: dict):
        record_uid = str(row.get("record_uid") or row.get("record_id"))
        if (not row.get("collect_c1") and not row.get("collect_a2") and
                not row.get("force_turn") and
                fixed_pose_actions.record_has_turn_first(record)):
            return {"mode": "reused"}
        frame = _build_frame(session, record)
        if row.get("collect_c1"):
            c1 = _collect_c1_family(
                session, frame, record, row,
                stage_root=Path(stage_root),
                seed=_seed(seed, f"c1:{record_uid}"),
                max_programs=max_programs,
                max_full_attempts=max_full_attempts)
            if c1 is not None:
                return {"mode": "c1", **c1}
        if row.get("collect_a2"):
            a2_delta = _collect_a2_group(
                session, frame, record, row=row,
                seed=_seed(seed, f"a2:{record_uid}"),
                max_programs=max_programs,
                max_full_attempts=max_full_attempts)
            if a2_delta is not None:
                return {"mode": "a2", **a2_delta}
        if row.get("repair_only") and not row.get("force_turn"):
            return {"mode": "reused"}
        if (not row.get("force_turn") and
                fixed_pose_actions.record_has_turn_first(record)):
            return {"mode": "reused"}
        delta = fixed_pose_actions.collect_turn_group(
            session, frame, record,
            seed=_seed(seed, record_uid),
            max_programs=max_programs,
            max_full_attempts=max_full_attempts,
            radius_m=float(row["body_radius_m"]))
        if delta is None:
            if fixed_pose_actions.record_has_noncompliant_initial_turn(record):
                return {"mode": "pruned", "outcomes": [], "provenance": {}}
            return None
        return {"mode": "added", **delta}

    return process


def make_surface_processor(
        *, seed: int, max_programs: int, max_full_attempts: int):
    """Create a fixed-pose surface target and safe-case collector."""
    from pipeline import surface_points

    def process(session, row: dict, record: dict):
        compact = abc1_record.normalize(record)
        record_uid = str(row["record_uid"])
        frame = _build_frame(session, compact)
        target = surface_points.select_target(
            frame, record_uid=record_uid, seed=int(seed),
            existing=compact.get("surface_point_target"))
        result = {
            "mode": "surfaces", "surface_point_target": target,
            "cases": [],
        }
        if row.get("observation_profile") is not None:
            result["observation_profile"] = row["observation_profile"]
        if target is None:
            result["surface_status"] = "no_target"
            return result
        if any(case.get("collision") is False and
               case.get("completed") is True
               for case in compact.get("cases") or []):
            result["surface_status"] = "existing_safe"
            return result
        delta = fixed_pose_actions.collect_safe_group(
            session, frame, compact,
            seed=_seed(seed, f"surface-safe:{record_uid}"),
            max_programs=max_programs,
            max_full_attempts=max_full_attempts,
            radius_m=float(row["body_radius_m"]))
        if delta is None:
            result["surface_status"] = "no_safe"
            return result
        outcomes = list(delta.get("outcomes") or [])
        if not outcomes:
            result["surface_status"] = "no_safe"
            return result
        case_record = dict(compact)
        case_record.pop("surface_point_target", None)
        case = abc1_record.compact_case(
            case_record, outcomes[0], dataset=str(row["dataset"]),
            record_uid=record_uid,
            provenance=delta.get("provenance") or {})
        if (case is None or case.get("collision") is not False or
                case.get("completed") is not True):
            result["surface_status"] = "no_safe"
            return result
        for task in ("A4", "B1", "B2"):
            (case.get("task_outputs") or {}).pop(task, None)
        result["cases"] = [case]
        result["surface_status"] = "added_safe"
        return result

    return process


def collect_records(
        plan_path: Path, worker: str, output_path: Path, *,
        seed: int = 20260904, max_programs: int = 500,
        max_full_attempts: int = 4,
        mp3d_root: Path = DEFAULT_MP3D_ROOT,
        gs_root: Path = DEFAULT_GS_ROOT,
        gs_manifest: Path = DEFAULT_GS_MANIFEST,
        b1k_manifest: Path = DEFAULT_B1K_MANIFEST) -> dict:
    header, _rows = inventory.read_plan(plan_path)
    processor = (
        make_surface_processor(
            seed=seed, max_programs=max_programs,
            max_full_attempts=max_full_attempts)
        if header.get("mode") == "surfaces" else
        make_record_processor(
            seed=seed, max_programs=max_programs,
            max_full_attempts=max_full_attempts,
            stage_root=Path(output_path).parent / "tmp" / str(worker)))
    return run_worker(
        plan_path, worker, output_path,
        open_scene=make_scene_opener(
            mp3d_root=mp3d_root, gs_root=gs_root,
            gs_manifest=gs_manifest, b1k_manifest=b1k_manifest),
        process_record=processor)
