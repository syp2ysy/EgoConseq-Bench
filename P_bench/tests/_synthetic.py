"""Shared synthetic Frame builder for pipeline tests (no Habitat)."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pipeline import config, perception, record as record_fields, rollout
from pipeline import actions as action_geometry
from pipeline.floor_plane import FloorPlaneEstimate, FloorPlaneFitResult
from pipeline.frame import Frame, SensorProfile

# A level floor through the agent root: the pose-local plane a synthetic
# fixture implies when it places geometry at y=0.
LEVEL_FLOOR = FloorPlaneEstimate(normal_local=(0.0, 1.0, 0.0), offset_m=0.0)

# The calibration a fixture's level floor would have produced: a clean fit with
# diagnostics comfortably inside every gate, so a test that fails does so for
# the reason it is about and not because the synthetic floor looks marginal.
LEVEL_FLOOR_FIT = FloorPlaneFitResult(
    estimate=LEVEL_FLOOR,
    rejection_reasons=(),
    seed_y_m=0.0,
    candidate_band_m=(-0.10, 0.10),
    support_count=4000,
    support_extent_x_m=3.0,
    support_extent_z_m=3.0,
    support_cell_count=900,
    support_area_m2=9.0,
    inlier_count=3800,
    inlier_ratio=0.95,
    inlier_cell_count=900,
    inlier_area_m2=9.0,
    residual_rmse_m=0.002,
    residual_p95_m=0.004,
    tilt_deg=0.0,
)


CAPACITY_TASKS = (
    "A1_collision",
    "A2_collision_step_grounding",
    "A3_contact_object",
    "B1_endpoint_distance",
    "B2_endpoint_direction",
    "C1_future_view_selection",
)


def capacity_evidence(root: Path, *, counts=None) -> dict:
    """Create source-bound two-canary evidence for controller tests."""
    counts = counts or {"r2r": 8, "gs": 8, "b1k": 12}
    pose_times = {"r2r": 1.0, "gs": 2.0, "b1k": 3.0}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parent_path = root / "capacity-canary-parent.json"
    source_authority_path = root / "capacity-source-authority.json"
    parent_path.write_text(json.dumps({"schema": "synthetic-capacity-parent"}))
    source_authority_path.write_text(json.dumps({
        "schema": "synthetic-capacity-source-authority"}))
    datasets = {}
    record = {"selection": {"proposal_provenance": {"ordinary": {
        "protocol": "depth-conditioned-action-bank-v3",
        "variant": "natural_dynamic",
    }}}}
    record_digest = hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii")).hexdigest()
    for dataset, catalog_count in counts.items():
        canaries = []
        for suffix in ("0", "1"):
            scene_id = f"{dataset}-{suffix}"
            scene_root = Path(root) / dataset / scene_id
            artifact = scene_root / "candidate_qa"
            (artifact / "public").mkdir(parents=True)
            (artifact / "private").mkdir()
            records = scene_root / "records.jsonl"
            records.write_text((json.dumps(record) + "\n") * 10)
            source_path = {
                "r2r": "/data/r2r/train.json.gz",
                "gs": "/data/gs/train.json",
                "b1k": "/data/b1k/source.json",
            }[dataset]
            source_sha = {"r2r": "4", "gs": "5", "b1k": "3"}[dataset] * 64
            revision = "a" * 40
            shard_id = f"canary-{dataset}-{suffix}"
            controller = scene_root / "controller_manifest.json"
            controller_body = {
                "schema": "egoconseq.capacity-canary-controller.v1",
                "parent_manifest_sha256": "9" * 64,
                "dataset": dataset,
                "scene_id": scene_id,
                "job_id": shard_id,
                "collection_shard_id": shard_id,
                "revision": revision,
                "source_authority": {
                    "path": source_path, "sha256": source_sha},
            }
            controller_sha = hashlib.sha256(json.dumps(
                controller_body, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True).encode("ascii")).hexdigest()
            controller.write_text(json.dumps({
                **controller_body, "sha256": controller_sha}))
            run_meta = scene_root / "run_meta.json"
            run_meta.write_text(json.dumps({
                "stats": {"pose_attempts": 20},
                "params": {
                    "backend": dataset, "scenes": [scene_id],
                    "collection_shard_id": shard_id,
                    "code_revision": revision,
                    "ordinary_actions_per_pose": 36,
                    {"r2r": "r2r_train_episodes",
                     "gs": "gs_source_manifest",
                     "b1k": "b1k_source_manifest"}[dataset]: source_path,
                },
                "code_revision": revision,
                "code_dirty": False,
                "source_catalog": {
                    "datasets": [dataset], "scene_ids": [scene_id],
                    "manifest_sha256": [source_sha],
                },
            }))
            funnel = scene_root / "collection_funnel.json"
            funnel.write_text(json.dumps({
                "backend": dataset,
                "per_scene": [{
                    "scene_id": scene_id,
                    "stage_counts": {"pose_attempts": 20},
                }],
            }))
            events = scene_root / "controller_events.jsonl"
            event_binding = {
                "dataset": dataset, "scene_id": scene_id,
                "controller_manifest_sha256": controller_sha,
                "job_id": shard_id, "collection_shard_id": shard_id,
                "revision": revision,
                "source_authority_sha256": source_sha,
            }
            events.write_text("".join(json.dumps({
                **event_binding, "event": name, "time_unix": timestamp,
            }) + "\n" for name, timestamp in (
                ("controller_scene_started", 100.0),
                ("backend_ready", 110.0),
                ("controller_scene_finished",
                 110.0 + 20 * pose_times[dataset]),
            )))
            items = []
            answers = []
            atoms = [{"id": "atom-0", "record_sha256": record_digest}]
            for task_id in CAPACITY_TASKS:
                for index in range(10):
                    item_id = f"{task_id}-{index}"
                    items.append({"id": item_id, "task_id": task_id})
                    answers.append({
                        "id": item_id,
                        "atom_ref": "atom-0",
                        "input_asset": {
                            "sha256": f"marked-{task_id}-{index}",
                            "raw_sha256": f"frame-{index}",
                        },
                    })
            contexts = [{
                "record_sha256": record_digest,
                "context": {
                    "source": {
                        "source_dataset": dataset, "scene_id": scene_id},
                    "collection_contract": {
                        "source_dataset": dataset, "scene_id": scene_id},
                },
            }]
            for path, rows in (
                    (artifact / "public" / "items.jsonl", items),
                    (artifact / "private" / "answers.jsonl", answers),
                    (artifact / "private" / "atoms.jsonl", atoms),
                    (artifact / "private" / "record_contexts.jsonl", contexts)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            canaries.append({
                "scene_id": scene_id,
                "records_path": str(records),
                "run_meta_path": str(run_meta),
                "funnel_path": str(funnel),
                "controller_events_path": str(events),
                "controller_manifest_path": str(controller),
                "compiled_qa_path": str(artifact),
                "parent_manifest_path": str(parent_path),
                "source_authority_path": str(source_authority_path),
            })
        datasets[dataset] = {
            "catalog_scene_count": catalog_count,
            "canary_scenes": canaries,
        }
    return {
        "schema": "egoconseq.collection-capacity-evidence.v1",
        "parent_manifest_path": str(parent_path),
        "source_authority_path": str(source_authority_path),
        "datasets": datasets,
    }


def source_provenance(
        scene_id="scene", *, dataset="r2r", semantic_format=None) -> dict:
    """Return one internally consistent synthetic raw-source identity."""
    formats = {
        "r2r": "mp3d_ply",
        "gs": "gs_bbox",
    }
    roles = ["scene", "navmesh", "semantic"]
    if dataset == "gs":
        roles.append("collision_authority")
    else:
        roles.extend(["semantic_metadata", "scene_dataset_config"])
    assets = []
    for role in roles:
        payload = f"{dataset}:{scene_id}:{role}".encode("utf-8")
        assets.append({
            "role": role,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    identity = {
        "version": "egoconseq.source-assets.v1",
        "assets": assets,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return {
        "scene_id": str(scene_id),
        "source_dataset": str(dataset),
        "official_split": "train",
        "semantic_format": semantic_format or formats[dataset],
        "source_manifest": f"/datasets/{dataset}/train.json",
        "source_manifest_sha256": hashlib.sha256(
            f"{dataset}:manifest".encode("utf-8")).hexdigest(),
        "source_asset_identity_version": identity["version"],
        "source_assets": assets,
        "source_assets_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def clean_record() -> dict:
    """Return one current synthetic consequence record for CLI/oracle tests."""
    from pipeline import consequence
    from pipeline.actions import Forward
    from pipeline.geometry import Disc
    from tests.test_pl_rollout import HalfPlaneNav, _SurfaceIndex

    frame = make_frame()
    frame.objects[0]["mask_area_px"] = 30
    frame.semantic_index = _SurfaceIndex()
    outcome = consequence.judge(
        frame, Disc(0.25), [Forward(2.0)], nav=HalfPlaneNav(0.8),
        target_ids=[7])
    outcome["outcome_id"] = "o"
    return record_fields.build_record(
        frame, [outcome], image_path="img/f.png",
        floor_calibration=LEVEL_FLOOR_FIT)

K = config.intrinsics()
H, W = 480, 640


def make_object(xz, iid, cat):
    xz = np.asarray(xz, float)
    cx, cz = float(xz[:, 0].mean()), float(xz[:, 1].mean())
    d = np.hypot(xz[:, 0], xz[:, 1]); ni = int(np.argmin(d))
    return {
        "instance_id": iid, "category": cat, "is_structural": config.is_structural(cat),
        "mask_area_px": xz.shape[0], "centroid_px": [320.0, 240.0],
        "ground_xy_centroid": [cx, cz], "ground_xy_nearest": [float(xz[ni, 0]), float(xz[ni, 1])],
        "bearing_deg": math.degrees(math.atan2(cx, cz)),
        "dist_centroid_m": math.hypot(cx, cz), "dist_nearest_m": float(d[ni]),
        "n_points_raw": xz.shape[0], "_points_xz": xz,
    }


def make_frame():
    """Ground-supported wall at z=1 and target cluster at (0,2)."""
    xs = np.linspace(-0.5, 0.5, 60)
    wall = np.stack([xs, np.full_like(xs, 0.15), np.full_like(xs, 1.0)], axis=1)
    tgt_xz = np.array([[0, 2], [0.05, 2], [-0.05, 2], [0, 2.05], [0, 1.95]], float)
    tgt = np.stack([tgt_xz[:, 0], np.full(len(tgt_xz), 0.15), tgt_xz[:, 1]], axis=1)

    pts = np.vstack([wall, tgt])
    sem = np.array([5] * len(wall) + [7] * len(tgt), dtype=np.int64)
    vf = perception.VoxelField(pts)
    depth = np.full((H, W), 3.0, dtype=np.float32)

    return Frame(
        frame_id="F-test", scene_id="synthetic", scene_glb="/dev/null",
        position=np.zeros(3), yaw_rad=0.0, K=K, floor_plane=LEVEL_FLOOR,
        rgb=np.zeros((H, W, 3), np.uint8), depth=depth,
        pts=pts, pts_uv=np.zeros((len(pts), 2), np.int64), pts_sem=sem, vf=vf,
        id_to_cat={5: "wall", 7: "chair"},
        objects=[make_object(tgt_xz, 7, "chair")],
        quality={"valid_depth_ratio": 1.0, "dist_to_obstacle_m": 2.0, "visible_floor_ratio": 0.3},
    )


def make_overhang_frame(cam_h, slab_y=1.0):
    """Overhang slab at height ``slab_y`` spanning the path ahead (z~1), + ground.

    ``vf`` uses the fixed ground-support band, so the camera height never
    changes whether the slab is treated as an obstacle.
    """
    xs = np.linspace(-0.4, 0.4, 17)
    zs = np.linspace(0.9, 1.1, 5)
    gx, gz = np.meshgrid(xs, zs)
    slab = np.stack([gx.ravel(), np.full(gx.size, slab_y), gz.ravel()], axis=1)
    ground = np.stack([xs, np.zeros_like(xs), np.full_like(xs, 1.0)], axis=1)
    pts = np.vstack([slab, ground])
    sem = np.array([11] * slab.shape[0] + [3] * ground.shape[0], dtype=np.int64)
    obs = perception.obstacle_mask(
        pts, LEVEL_FLOOR, band=config.GROUND_OBSTACLE_BAND_M)
    vf = perception.VoxelField(pts[obs])
    return Frame(
        frame_id="F-overhang", scene_id="synthetic", scene_glb="/dev/null",
        position=np.zeros(3), yaw_rad=0.0, K=K, floor_plane=LEVEL_FLOOR,
        rgb=np.zeros((H, W, 3), np.uint8), depth=np.full((H, W), 3.0, dtype=np.float32),
        pts=pts, pts_uv=np.zeros((len(pts), 2), np.int64), pts_sem=sem, vf=vf,
        id_to_cat={11: "table", 3: "floor"}, objects=[],
        sensor=SensorProfile.from_values(cam_h, config.HFOV_DEG, config.VFOV_DEG),
    )


# Canonical conseq.v10 outcome/record fixtures shared by candidate and
# validator tests. Kept here so suites do not import fixtures from one another.

def _outcome(*, collision=False, progress=1.0, visible=True, radius=0.2,
             seq_len=2):
    contact_arc = 0.4 if collision else None
    visible_target_px = int(round(0.02 * W * H))
    actions = [action_geometry.Turn(30.0), action_geometry.Forward(1.5)]
    nominal_pose = action_geometry.pose_after(actions)
    realized_pose = (
        action_geometry.pose_at_arc(actions, contact_arc)
        if collision else nominal_pose)
    return {
        "outcome_id": "o1", "seq_len": seq_len,
        "body": {"shape": "disc", "radius_m": radius},
        "actions": [{"type": "turn", "deg": 30.0},
                    {"type": "forward", "m": 1.5}],
        "execution": {
            "completed": not collision,
            "stop_reason": "collision" if collision else "completed",
            "execution_regime": (
                "contact_truncated" if collision else "completed_clear"),
            "executed_forward_fraction": progress,
            "animation_stop_time_fraction": progress,
            "nominal_forward_m": 1.5,
            "executed_forward_m": contact_arc if collision else 1.5,
            "executed_turn_deg": 30.0,
            "executed_forward_after_turn_m": contact_arc if collision else 1.5,
            "stop_arc_m": contact_arc,
            "nominal_pose": {
                "x": nominal_pose[0], "z": nominal_pose[1],
                "heading_deg": nominal_pose[2],
            },
            "realized_pose": {
                "x": realized_pose[0], "z": realized_pose[1],
                "heading_deg": realized_pose[2],
            },
        },
        "physical": {
            "authority": "test_geometry",
            "collision": collision, "minimum_clearance_m": 0.3,
            "first_contact_arc_m": contact_arc,
            "contact": ({
                "category": "wall", "instance_id": 3, "unattributed": False,
                "depth_mask_attribution": {"category": "wall", "instance_id": 3,
                                           "unattributed": False},
                "full_geometry_attribution": {"category": "wall", "instance_id": 3,
                                              "unattributed": False},
            } if collision else None),
        },
        "depth_physical": {
            "authority": "depth",
            "collision": collision,
            "first_contact_arc_m": (
                progress * 1.5 if collision else None),
        },
        "future_view": {
            "status": "computed", "objects_entering_view": [],
            "objects_leaving_view": [9],
            "initial_depth_instance_set_agrees": True,
            "initial_depth_target_reprojection_agrees": True,
            "initial_depth_reprojection_agrees": True,
            "checkpoints": [{"camera_centered_instance": 7,
                "visible_instances": [7, 9],
                "instance_pixel_counts": {
                    7: visible_target_px, 9: 100},
                "structural_instance_ids": [], "targets": [{
                "target_instance_id": 7, "visible": True,
                "pixel_count": visible_target_px,
                "image_area_ratio": 0.02, "bbox_xyxy": [1, 2, 10, 20],
                "center_xy": [5, 10], "median_depth_m": 1.4,
            }], "target_reprojection_checks": [{
                "target_instance_id": 7, "agrees": True,
                "actual_visible": True, "predicted_visible": True,
                "center_error_normalized": 0.0, "area_error_ratio": 0.0,
            }]}, {"camera_centered_instance": 7,
                "visible_instances": [7] if visible else [],
                "instance_pixel_counts": {
                    "7": visible_target_px if visible else 0, "9": 0},
                "structural_instance_ids": [],
                "targets": [{
                "target_instance_id": 7, "visible": visible,
                "pixel_count": visible_target_px if visible else 0,
                "image_area_ratio": 0.02 if visible else 0.0,
                "bbox_xyxy": [1, 2, 10, 20] if visible else None,
                "center_xy": [5, 10] if visible else None,
                "median_depth_m": 1.4 if visible else None,
            }], "target_reprojection_checks": [{
                "target_instance_id": 7, "agrees": True,
                "actual_visible": visible, "predicted_visible": visible,
                "center_error_normalized": 0.0 if visible else None,
                "area_error_ratio": 0.0,
            }]}],
        },
        "evidence": {
            "physical": {"status": "sufficient", "coverage": 1.0,
                         "coverage_protocol": rollout.EVIDENCE_PROTOCOL_VERSION},
            "future_view": {"status": "sufficient"},
        },
        "oracle_consensus": {
            "accepted": True,
            "verdict": "agree_collision" if collision else "agree_safe",
            "reason": "accepted",
            "full_authority": "test_geometry",
            "depth_authority": "depth",
            "full_collision": collision,
            "depth_collision": collision,
            "full_contact_arc_m": contact_arc,
            "depth_contact_arc_m": (
                progress * 1.5 if collision else None),
            "contact_arc_difference_m": (
                abs(contact_arc - progress * 1.5) if collision else None),
            "contact_tolerance_m": 0.30,
            "corridor_coverage": 1.0,
            "coverage_min": 0.90,
        },
    }


def _record(outcome):
    intrinsics = config.intrinsics(110.0, 70.0)
    stored_outcome = copy.deepcopy(outcome)
    rec = {
        "schema_version": record_fields.SCHEMA_VERSION,
        "oracle_contract_version": record_fields.ORACLE_CONTRACT_VERSION,
        "substrate": {
            "base_rollout_key_version":
                record_fields.BASE_ROLLOUT_KEY_VERSION,
        },
        "frame_id": "f", "scene_id": "s",
        "observation_id": "observation-f",
        "image_path": "img/f.png",
        "source": source_provenance("s", dataset="r2r"),
        "camera_height_above_visible_floor_m": 0.8,
        "sensor": {"nominal_camera_offset_m": 0.8, "hfov_deg": 110,
                   "vfov_deg": 70, "resolution": [640, 480],
                   "focal_x_px": float(intrinsics[0, 0]),
                   "focal_y_px": float(intrinsics[1, 1])},
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "objects": [{"instance_id": 7, "category": "chair",
                     "is_structural": False},
                    {"instance_id": 9, "category": "lamp",
                     "is_structural": False}],
        "category_inventory": {"chair": 1, "lamp": 1},
        "outcomes": [stored_outcome],
        "intervention": {"group_id": "sensor-g", "type": "fov"},
    }
    stored_outcome["base_rollout_key"] = \
        record_fields.stored_base_rollout_key(rec, stored_outcome)
    return rec
