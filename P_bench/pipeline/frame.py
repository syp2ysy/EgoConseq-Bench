"""Frame: all per-frame perception evidence, built once and reused by judge.

build_frame is duck-typed on `sim` (needs .render, .instance_category_map,
.dist_to_obstacle) and does NOT import Habitat, so a Frame can be constructed
directly from arrays in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pipeline import config, perception, objects


@dataclass
class Frame:
    frame_id: str
    scene_id: str
    scene_glb: str
    position: np.ndarray          # world xyz (3,)
    yaw_rad: float
    K: np.ndarray
    floor_y: float

    rgb: np.ndarray               # (H,W,3) uint8
    depth: np.ndarray             # (H,W) float32
    semantic: Optional[np.ndarray]  # unused in this build (sensor broken); kept for viz

    pts: np.ndarray               # (N,3) ground xyz
    pts_uv: np.ndarray            # (N,2) source pixel (u,v)
    pts_sem: np.ndarray           # (N,)  instance id per point
    vf: perception.VoxelField     # obstacle occupancy field

    id_to_cat: Dict[int, str]
    objects: List[dict]
    quality: dict = field(default_factory=dict)

    @property
    def category_inventory(self) -> Dict[str, int]:
        inv: Dict[str, int] = {}
        for o in self.objects:
            inv[o["category"]] = inv.get(o["category"], 0) + 1
        return inv


def build_frame(sim, position, yaw: float, *,
                frame_id: str, scene_id: str, scene_glb: str) -> Frame:
    """Render one observation and derive all geometric evidence (single unproject).

    Per-point instance ids come from the offline SemanticIndex (nearest
    labelled semantic surface in world frame), since the semantic sensor is
    unusable in this build.
    """
    rgb, depth, K = sim.render(position, yaw)

    pts_cam, uv = perception.unproject(depth, K)
    pts = perception.to_agent_ground(pts_cam)
    world = perception.world_from_local(pts, position, yaw)
    sem = sim.assign_instances(world).astype(np.int64)

    floor_y = perception.estimate_floor_height(pts)
    obs_mask = perception.obstacle_mask(pts, floor_y)
    vf = perception.VoxelField(pts[obs_mask])

    id_to_cat = sim.id_to_cat
    objs = objects.extract_objects(pts, uv, sem, id_to_cat)

    valid = np.isfinite(depth) & (depth > 0)
    floor_pts = np.abs(pts[:, 1] - floor_y) <= 0.10
    quality = {
        "valid_depth_ratio": float(valid.mean()),
        "dist_to_obstacle_m": float(sim.dist_to_obstacle(position)),
        "visible_floor_ratio": float(floor_pts.sum()) / float(depth.size),
    }

    return Frame(
        frame_id=frame_id, scene_id=scene_id, scene_glb=scene_glb,
        position=np.asarray(position, dtype=np.float64), yaw_rad=float(yaw),
        K=np.asarray(K, dtype=np.float64), floor_y=floor_y,
        rgb=rgb, depth=np.asarray(depth), semantic=None,
        pts=pts, pts_uv=uv, pts_sem=sem, vf=vf,
        id_to_cat=id_to_cat, objects=objs, quality=quality,
    )
