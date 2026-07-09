"""Shared synthetic Frame builder for pipeline tests (no Habitat)."""

import math

import numpy as np

from pipeline import config, perception
from pipeline.frame import Frame

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
    """Wall at z=1 (x in [-0.5,0.5]) + target cluster at (0,2), y=0.5 (in band)."""
    xs = np.linspace(-0.5, 0.5, 60)
    wall = np.stack([xs, np.full_like(xs, 0.5), np.full_like(xs, 1.0)], axis=1)
    tgt_xz = np.array([[0, 2], [0.05, 2], [-0.05, 2], [0, 2.05], [0, 1.95]], float)
    tgt = np.stack([tgt_xz[:, 0], np.full(len(tgt_xz), 0.5), tgt_xz[:, 1]], axis=1)

    pts = np.vstack([wall, tgt])
    sem = np.array([5] * len(wall) + [7] * len(tgt), dtype=np.int64)
    vf = perception.VoxelField(pts)
    depth = np.full((H, W), 3.0, dtype=np.float32)

    return Frame(
        frame_id="F-test", scene_id="synthetic", scene_glb="/dev/null",
        position=np.zeros(3), yaw_rad=0.0, K=K, floor_y=0.0,
        rgb=np.zeros((H, W, 3), np.uint8), depth=depth, semantic=None,
        pts=pts, pts_uv=np.zeros((len(pts), 2), np.int64), pts_sem=sem, vf=vf,
        id_to_cat={5: "wall", 7: "chair"},
        objects=[make_object(tgt_xz, 7, "chair")],
        quality={"valid_depth_ratio": 1.0, "dist_to_obstacle_m": 2.0, "visible_floor_ratio": 0.3},
    )
