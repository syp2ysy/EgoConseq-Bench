"""GS-scene backend: same interface as sim.SimSession, but renders with gsplat.

A scene dir holds `scene.gs.ply` (gaussians, Habitat Y-up frame), `scene.navmesh`
(pre-baked, loaded standalone by habitat's PathFinder), and `labels.json`
(InteriorGS 3D bboxes -> gs_semantic). Only rendering + semantics differ from
`sim.py`; navmesh/pose-sampling reuse the pathfinder API. The pure-numpy pipeline
(frame/consequence/record/...) is unchanged.

Needs gsplat's CUDA ext on PATH: run with
    PATH=<env>/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import numpy as np
import habitat_sim

from pipeline import config, gs_render, gs_semantic
from pipeline.sim import Nav, sample_pose  # navmesh + pose-sampling reused


def discover_gs_scenes(root: str = None):
    """Scene dirs under GS_ROOT that have gaussians + navmesh + labels."""
    root = root or config.GS_ROOT
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if (os.path.exists(os.path.join(d, "scene.gs.ply")) and
                os.path.exists(os.path.join(d, "scene.navmesh")) and
                os.path.exists(os.path.join(d, "labels.json"))):
            out.append(d)
    return out


class GsSimSession:
    """Renders one GS scene (gsplat) + pathfinder + bbox semantics."""

    def __init__(self, scene_dir: str, *, device: str = "cuda"):
        self.scene_dir = scene_dir
        self.scene_glb = os.path.join(scene_dir, "scene.gs.ply")
        self.scene_id = os.path.basename(scene_dir.rstrip("/"))

        self._gs = gs_render.load_gs(self.scene_glb, device=device)
        self._K = config.intrinsics()
        self._hw = config.hw()

        self._pf = habitat_sim.PathFinder()
        self._pf.load_nav_mesh(os.path.join(scene_dir, "scene.navmesh"))
        if not self._pf.is_loaded:
            raise RuntimeError(f"navmesh failed to load: {scene_dir}")

        means = self._gs["means"].detach().cpu().numpy()
        self._sem = gs_semantic.load_bbox_index(
            os.path.join(scene_dir, "labels.json"), align_points=means)

    # -- rendering --
    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def id_to_cat(self) -> Dict[int, str]:
        return self._sem.id_to_cat

    def render(self, position, yaw: float,
               cam_h: float = config.CAMERA_HEIGHT_M) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb, depth = gs_render.render(self._gs, position, yaw, self._K, self._hw, cam_h=cam_h)
        return rgb, depth, self._K

    def assign_instances(self, world_points: np.ndarray) -> np.ndarray:
        return self._sem.assign(world_points)

    def instance_category_map(self) -> Dict[int, str]:
        return self._sem.id_to_cat

    # -- navmesh (pre-baked; not per-radius recomputable without a mesh) --
    def recompute_navmesh(self, radius: float, height: float = config.CYLINDER_HEIGHT_M) -> None:
        pass  # GS ships one baked navmesh; collision GT is depth-based (radius-independent)

    @property
    def pathfinder(self):
        return self._pf

    def dist_to_obstacle(self, position) -> float:
        return float(self._pf.distance_to_closest_obstacle(
            np.array(position, dtype=np.float32)))

    def nav(self, position, yaw: float) -> Nav:
        return Nav(self._pf, position, yaw)

    # -- sampling (GS depth is all-valid, so no min_valid_depth gate) --
    def sample_random_pose(self, rng, radii, yaws=None, max_tries=400):
        return sample_pose(self._pf, self.render, rng, radii,
                           min_floor=0.02, yaws=yaws, max_tries=max_tries)

    def close(self) -> None:
        self._gs = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
