"""Habitat-sim wrapper (the only Habitat-dependent module).

Renders RGB + Depth (the semantic sensor is unusable in this build — see
pipeline/semantic.py), exposes the pathfinder for navmesh cross-checks, and
delegates per-point instance labels to an offline-decoded SemanticIndex.
"""

from __future__ import annotations

import glob
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import quaternion  # bundled with habitat-sim
import habitat_sim

from pipeline import config, perception, semantic
from pipeline import actions as A
from pipeline.scene_pool import discover_semantic_scenes  # re-export (pure)


class Nav:
    """Navmesh cross-check bound to a pose (uses the pathfinder's current mesh)."""

    def __init__(self, pathfinder, pos, yaw: float):
        self._pf = pathfinder
        self._pos = np.asarray(pos, dtype=np.float64)
        self._yaw = float(yaw)

    def _world(self, x: float, z: float) -> np.ndarray:
        return perception.world_from_local(np.array([[x, 0.0, z]]), self._pos, self._yaw)[0]

    def d_safe(self, acts, radius: float,
               step: float = config.MARCH_STEP_M,
               max_y_delta: float = config.NAV_Y_DELTA_M) -> float:
        """Arc to first non-navigable sample along the action path (else total forward)."""
        total = A.total_forward_m(acts)
        for i, (x, z, _h, arc) in enumerate(A.sample_path(acts, step)):
            if i == 0:
                continue
            if not self._pf.is_navigable(self._world(x, z), max_y_delta=max_y_delta):
                return arc
        return total

    def geodesic(self, a_xz, b_xz) -> Optional[float]:
        wa = self._pf.snap_point(self._world(a_xz[0], a_xz[1]))
        wb = self._pf.snap_point(self._world(b_xz[0], b_xz[1]))
        if not (np.all(np.isfinite(wa)) and np.all(np.isfinite(wb))):
            return None
        sp = habitat_sim.ShortestPath()
        sp.requested_start = wa
        sp.requested_end = wb
        if not self._pf.find_path(sp):
            return None
        g = sp.geodesic_distance
        return float(g) if np.isfinite(g) else None


def sample_pose(pf, render, rng, radii, *, min_floor=0.05, min_valid_depth=None,
                yaws=None, max_tries=2000):
    """Draw a quality-filtered (position, yaw); None on failure. Shared by both
    backends. `render(pos, yaw) -> (rgb, depth, K)`; `min_valid_depth` gates HM3D
    depth holes (GS depth is always valid, so pass None)."""
    min_clear = max(radii) + 0.1
    if yaws is None:
        yaws = [math.radians(a) for a in range(0, 360, 45)]
    for _ in range(max_tries):
        pos = pf.get_random_navigable_point()
        if not np.all(np.isfinite(pos)):
            continue
        if pf.distance_to_closest_obstacle(pos) < min_clear:
            continue
        yaw = float(rng.choice(yaws))
        _, depth, K = render(pos, yaw)
        if min_valid_depth is not None and \
                float((np.isfinite(depth) & (depth > 0)).mean()) < min_valid_depth:
            continue
        pts = perception.to_agent_ground(perception.unproject(depth, K)[0])
        floor_y = perception.estimate_floor_height(pts)
        vfr = float((np.abs(pts[:, 1] - floor_y) <= 0.10).sum()) / float(depth.size)
        if vfr < min_floor:
            continue
        return np.array(pos, dtype=np.float64), yaw
    return None


class SimSession:
    """Renders one HM3D scene; owns the pathfinder and semantic index."""

    def __init__(self, scene_glb: str, *, scene_dataset_cfg: Optional[str] = None,
                 heights=None):
        self.scene_glb = scene_glb
        self.scene_id = os.path.basename(os.path.dirname(scene_glb))  # e.g. 00800-TEEsavR23oF
        # one color+depth sensor per requested camera height (default: just 1.5 m).
        self._heights = [float(h) for h in (heights or [config.CAMERA_HEIGHT_M])]

        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = scene_glb
        if scene_dataset_cfg:
            backend.scene_dataset_config_file = scene_dataset_cfg

        hw = config.hw()
        specs = []
        for h in self._heights:
            for pfx, stype in (("color", habitat_sim.SensorType.COLOR),
                               ("depth", habitat_sim.SensorType.DEPTH)):
                s = habitat_sim.CameraSensorSpec()
                s.uuid = f"{pfx}_{config.height_tag(h)}"; s.sensor_type = stype
                s.resolution = hw
                s.position = [0.0, h, 0.0]
                s.hfov = config.HFOV_DEG
                specs.append(s)
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = specs

        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
        self._K = config.intrinsics()
        self.recompute_navmesh(min(config.RADII_M))  # populate pathfinder
        self._sem = semantic.load_semantic_index(scene_glb)

    # -- rendering --
    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def id_to_cat(self) -> Dict[int, str]:
        return self._sem.id_to_cat

    def render(self, position, yaw: float,
               cam_h: float = config.CAMERA_HEIGHT_M) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        st = habitat_sim.AgentState()
        st.position = np.array(position, dtype=np.float32)
        st.rotation = quaternion.from_rotation_vector(np.array([0.0, yaw, 0.0]))
        self._sim.get_agent(0).set_state(st)
        obs = self._sim.get_sensor_observations()
        tag = config.height_tag(min(self._heights, key=lambda h: abs(h - cam_h)))  # nearest sensor
        rgb = obs[f"color_{tag}"][..., :3].astype(np.uint8)
        depth = obs[f"depth_{tag}"].astype(np.float32)
        return rgb, depth, self._K

    def assign_instances(self, world_points: np.ndarray) -> np.ndarray:
        return self._sem.assign(world_points)

    def instance_category_map(self) -> Dict[int, str]:
        return self._sem.id_to_cat

    # -- navmesh --
    def recompute_navmesh(self, radius: float, height: float = config.CYLINDER_HEIGHT_M) -> None:
        s = habitat_sim.NavMeshSettings()
        s.set_defaults()
        s.agent_radius = radius
        s.agent_height = height
        self._sim.recompute_navmesh(self._sim.pathfinder, s)

    @property
    def pathfinder(self):
        return self._sim.pathfinder

    def dist_to_obstacle(self, position) -> float:
        return float(self._sim.pathfinder.distance_to_closest_obstacle(
            np.array(position, dtype=np.float32)))

    def nav(self, position, yaw: float) -> Nav:
        return Nav(self._sim.pathfinder, position, yaw)

    # -- sampling --
    def sample_random_pose(self, rng, radii, yaws=None, max_tries=2000):
        return sample_pose(self._sim.pathfinder, self.render, rng, radii,
                           min_floor=0.05, min_valid_depth=0.85,
                           yaws=yaws, max_tries=max_tries)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
