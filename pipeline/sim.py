"""Habitat-sim wrapper (the only Habitat-dependent module).

Renders RGB + Depth (the semantic sensor is unusable in this build — see
pipeline/semantic.py), exposes the pathfinder for navmesh cross-checks, and
delegates per-point instance labels to an offline-decoded SemanticIndex.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Optional

import numpy as np
import quaternion  # bundled with habitat-sim
import habitat_sim
from PIL import Image

from pipeline import config, semantic
from pipeline.frame import RenderObservation, SensorProfile


def load_scene_semantic_index(
        scene_glb: str, semantic_format: str, *,
        query_workers: int = 1):
    """Dispatch semantic decoding without changing downstream Frame semantics."""
    if semantic_format == "mp3d_ply":
        return semantic.load_mp3d_semantic_index(
            scene_glb, query_workers=query_workers)
    raise ValueError(f"unsupported scene semantic format: {semantic_format}")


def sensor_uuid(kind: str, height: float, hfov: float, vfov: float) -> str:
    """Stable Habitat sensor name for one height and angular profile."""
    fov = f"{int(round(float(hfov) * 10)):04d}x{int(round(float(vfov) * 10)):04d}"
    return f"{kind}_{config.height_tag(height)}_f{fov}"


def render_profile_set(heights, fovs):
    """Exact ``(height, hfov, vfov)`` triples a session must be able to render.
    Habitat can only render sensors it registered: every height with every FOV,
    sorted so sensor registration order does not depend on set iteration.
    """
    return tuple(sorted({(float(height), float(hfov), float(vfov))
                         for hfov, vfov in fovs for height in heights}))


class SimSession:
    """Renders one R2R/MP3D scene and owns its semantic index."""
    def __init__(self, scene_glb: str, *, scene_dataset_cfg: Optional[str] = None,
                 semantic_format: str = "mp3d_ply",
                 source_dataset: str = "r2r",
                 official_split: Optional[str] = None,
                 heights=None, hfov: float = config.HFOV_DEG,
                 vfov: float = config.VFOV_DEG, fovs=None,
                 semantic_query_workers: int = 1,
                 gpu_device_id: int = 0):
        self.scene_glb = scene_glb
        self.scene_id = os.path.basename(os.path.dirname(scene_glb))  # e.g. 00800-TEEsavR23oF
        self.source_dataset = str(source_dataset)
        self.official_split = official_split
        # one color+depth sensor per requested camera height (default: just 1.5 m).
        self._heights = [float(h) for h in (heights or [config.CAMERA_HEIGHT_M])]
        self._fovs = [tuple(map(float, pair)) for pair in (
            fovs if fovs is not None else [(hfov, vfov)])]
        if not self._fovs:
            raise ValueError("at least one FOV profile is required")
        self._hfov, self._vfov = self._fovs[0]
        self._render_profiles = render_profile_set(self._heights, self._fovs)
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = scene_glb
        backend.gpu_device_id = int(gpu_device_id)
        if scene_dataset_cfg:
            backend.scene_dataset_config_file = scene_dataset_cfg
        specs = []
        sensor_names = set()
        for h, profile_hfov, profile_vfov in self._render_profiles:
            hw = config.render_hw(profile_hfov, profile_vfov)
            for pfx, stype in (("color", habitat_sim.SensorType.COLOR),
                               ("depth", habitat_sim.SensorType.DEPTH)):
                s = habitat_sim.CameraSensorSpec()
                s.uuid = sensor_uuid(pfx, h, profile_hfov, profile_vfov)
                if s.uuid in sensor_names:
                    raise ValueError(f"duplicate sensor profile: {s.uuid}")
                sensor_names.add(s.uuid)
                s.sensor_type = stype
                s.resolution = hw
                s.position = [0.0, h, 0.0]
                s.hfov = profile_hfov
                specs.append(s)
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = specs
        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
        self._navmesh_cache_dir = tempfile.TemporaryDirectory(
            prefix="egoconseq-navmesh-")
        self._navmesh_cache = {}
        self._active_navmesh_key = None
        self._K = config.intrinsics(self._hfov, self._vfov)
        self.recompute_navmesh(min(config.RADII_M))  # populate pathfinder
        self._sem = load_scene_semantic_index(
            scene_glb, semantic_format,
            query_workers=semantic_query_workers)
    # -- rendering --
    @property
    def K(self) -> np.ndarray:
        return self._K
    @property
    def id_to_cat(self) -> Dict[int, str]:
        return self._sem.id_to_cat
    @property
    def semantic_index(self):
        return self._sem
    def render(self, position, yaw: float,
               cam_h: float = config.CAMERA_HEIGHT_M,
               hfov: Optional[float] = None, vfov: Optional[float] = None
               ) -> RenderObservation:
        hfov = self._hfov if hfov is None else float(hfov)
        vfov = self._vfov if vfov is None else float(vfov)
        # Match the whole triple: a height and a FOV that were each registered
        # separately are not a profile this session renders.
        matches = [profile for profile in self._render_profiles
                   if abs(profile[0] - float(cam_h)) <= 1e-6 and
                   abs(profile[1] - hfov) <= 1e-6 and
                   abs(profile[2] - vfov) <= 1e-6]
        if not matches:
            raise ValueError(
                f"sensor profile {cam_h} m / {hfov}x{vfov} deg is not "
                f"configured; available={self._render_profiles}")
        st = habitat_sim.AgentState()
        st.position = np.array(position, dtype=np.float32)
        st.rotation = quaternion.from_rotation_vector(np.array([0.0, yaw, 0.0]))
        self._sim.get_agent(0).set_state(st)
        actual_height = matches[0][0]
        obs = {}
        for prefix in ('color', 'depth'):
            name = sensor_uuid(prefix, actual_height, hfov, vfov)
            sensor = self._sim._sensors[name]
            sensor.draw_observation()
            obs[name] = sensor.get_observation()
        rgb = obs[sensor_uuid("color", actual_height, hfov, vfov)][..., :3].astype(np.uint8)
        depth = obs[sensor_uuid("depth", actual_height, hfov, vfov)].astype(np.float32)
        width, height = config.resolution()
        if rgb.shape[:2] != (height, width):
            rgb = np.asarray(Image.fromarray(rgb).resize(
                (width, height), resample=Image.Resampling.BILINEAR))
            depth = np.asarray(Image.fromarray(depth).resize(
                (width, height), resample=Image.Resampling.NEAREST), dtype=np.float32)
        K = config.intrinsics(hfov, vfov)
        sensor = SensorProfile.from_values(actual_height, hfov, vfov)
        return RenderObservation(
            rgb=rgb, depth=depth, K=K, sensor=sensor,
            position=np.asarray(position, dtype=np.float64),
            yaw_rad=float(yaw))
    def assign_instances(self, world_points: np.ndarray) -> np.ndarray:
        return self._sem.assign(world_points)

    def support_ground(self, position):
        """Measure the support mesh directly beneath a native navigation pose.

        This build has no Bullet. A private downward depth camera measures the
        rendered Habitat mesh; NavMesh height alone includes voxelization error.
        Nearby furniture planes and unsupported poses are withheld.
        """
        name = "fine_native_support_depth"
        if name not in self._sim._sensors:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = name
            spec.sensor_type = habitat_sim.SensorType.DEPTH
            spec.resolution = [1, 1]
            spec.position = [0., .15, 0.]
            spec.orientation = [-math.pi / 2, 0., 0.]
            spec.hfov = 1.
            spec.near = .001
            self._sim.add_sensor(spec)
        agent = self._sim.get_agent(0)
        previous = agent.get_state()
        point = np.asarray(position, dtype=np.float64)
        snap = np.asarray(self.pathfinder.snap_point(point), dtype=np.float64)
        if (not np.isfinite(snap).all() or
                np.linalg.norm((snap - point)[[0, 2]]) > .001 or
                abs(snap[1] - point[1]) > .1):
            raise ValueError("floor support has no nearby native navigation surface")
        try:
            state = habitat_sim.AgentState()
            state.position = snap.astype(np.float32)
            agent.set_state(state)
            sensor = self._sim._sensors[name]
            sensor.draw_observation()
            depth = float(sensor.get_observation()[0, 0])
            origin = agent.get_state().sensor_states[name].position
        finally:
            agent.set_state(previous, reset_sensors=False, infer_sensor_states=False)
        ground_y = float(origin[1] - depth)
        if not math.isfinite(depth) or depth <= 0 or abs(ground_y - snap[1]) > .1:
            raise ValueError("floor depth disagrees with the native support level")
        grounded = np.array([point[0], ground_y, point[2]])
        instance = int(self.assign_instances(grounded[None, :])[0])
        category = str(self.id_to_cat.get(instance, ""))
        if category.casefold() not in {"floor", "flooring", "ground", "carpet", "rug", "mat"}:
            raise ValueError("floor support is not a source-labelled floor surface")
        return {"ground_y": ground_y, "agent_position": grounded.tolist(),
                "method": "habitat_mesh_downward_depth",
                "evidence": {"sensor_world": origin.tolist(), "depth_m": depth,
                             "navmesh_position": snap.tolist(), "instance_id": instance,
                             "source_category": category}}

    def sensor_pose_evidence(self, position, yaw, cam_h, hfov, vfov):
        state = self._sim.get_agent(0).get_state()
        sensor = state.sensor_states[sensor_uuid("depth", cam_h, hfov, vfov)]
        expected = np.asarray(position, dtype=np.float64) + [0., cam_h, 0.]
        rotation = quaternion.from_rotation_vector([0., float(yaw), 0.])
        if (not np.allclose(sensor.position, expected, rtol=0., atol=1e-5) or
                abs(float(np.dot(quaternion.as_float_array(sensor.rotation),
                                 quaternion.as_float_array(rotation)))) < 1 - 1e-6):
            raise ValueError("camera native pose differs from requested ground-relative pose")
        return {"position_world": sensor.position.tolist(), "ground_y": float(position[1]),
                "camera_height_m": float(sensor.position[1] - position[1]),
                "method": "habitat_sensor_state_readback"}


    # -- navmesh --
    def recompute_navmesh(
        self, radius: float, height: float = config.GROUND_ORACLE_HEIGHT_M,
    ) -> None:
        key = (
            round(float(radius), 9), round(float(height), 9),
            round(float(config.NAVMESH_CELL_SIZE_M), 9),
            round(float(config.NAVMESH_CELL_HEIGHT_M), 9),
            round(float(config.NAVMESH_MAX_CLIMB_M), 9),
        )
        if getattr(self, "_active_navmesh_key", None) == key:
            return
        cache = getattr(self, "_navmesh_cache", None)
        if cache is None:
            cache = self._navmesh_cache = {}
        cached = cache.get(key)
        if cached is not None:
            loaded = self._sim.pathfinder.load_nav_mesh(str(cached))
            if loaded is False:
                raise RuntimeError(f"cannot load cached navmesh: {cached}")
            self._active_navmesh_key = key
            return
        s = habitat_sim.NavMeshSettings()
        s.set_defaults()
        s.cell_size = config.NAVMESH_CELL_SIZE_M
        s.cell_height = config.NAVMESH_CELL_HEIGHT_M
        s.agent_max_climb = config.NAVMESH_MAX_CLIMB_M
        # Recast rounds the agent up to whole cells, and these settings are
        # float32: passing a value meant to be an exact multiple reads a shade
        # high and gains a whole cell. Aim at the middle of the intended cell so
        # the rounding cannot drift either way.
        s.agent_radius = config.navmesh_agent_radius(radius)
        s.agent_height = config.navmesh_agent_height(height)
        rebuilt = self._sim.recompute_navmesh(self._sim.pathfinder, s)
        if rebuilt is False:
            raise RuntimeError(
                f"cannot build navmesh for radius={radius}, height={height}")
        directory_value = getattr(self, "_navmesh_cache_dir", None)
        directory = Path(
            directory_value
            if isinstance(directory_value, (str, os.PathLike))
            else directory_value.name)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            "-".join(f"{part:.9f}" for part in key) + ".navmesh")
        saved = self._sim.pathfinder.save_nav_mesh(str(path))
        if saved is False:
            raise RuntimeError(f"cannot cache navmesh: {path}")
        cache[key] = path
        self._active_navmesh_key = key
    @property
    def pathfinder(self):
        return self._sim.pathfinder
    def dist_to_obstacle(self, position) -> float:
        return float(self._sim.pathfinder.distance_to_closest_obstacle(
            np.array(position, dtype=np.float32)))
    def close(self) -> None:
        self._sim.close()
        directory = getattr(self, "_navmesh_cache_dir", None)
        if hasattr(directory, "cleanup"):
            directory.cleanup()
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()
