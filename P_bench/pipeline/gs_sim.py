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
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, Tuple

import numpy as np
import habitat_sim
import torch
from pipeline import (
    config, dataset_contracts, gs_collision, gs_semantic, gs_source, perception,
    pose_calibration, scene_pool,
)
from pipeline.frame import (
    RenderObservation, SensorProfile, validate_render_observation,
)
from pipeline.scene_pool import SceneSpec
from pipeline.sim import (  # navmesh + pose-sampling reused
    Nav, pose_observations)


def _ensure_active_env_ninja(executable: str = None) -> None:
    if shutil.which("ninja") is not None:
        return
    env_bin = Path(executable or sys.executable).resolve().parent
    ninja = env_bin / "ninja"
    if not (ninja.is_file() and os.access(ninja, os.X_OK)):
        return
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = (
        str(env_bin) if not current else
        str(env_bin) + os.pathsep + current)


_ensure_active_env_ninja()

from gsplat import rasterization


def load_gs(path: str, device: str = "cuda") -> dict:
    """Parse a 3DGS ``.ply`` into the tensors required by gsplat."""
    source = gs_source.read_gaussian_source(path)

    def tensor(value):
        return torch.from_numpy(
            np.ascontiguousarray(value)).float().to(device)
    return {
        "means": tensor(source["means"]),
        "quats": tensor(source["quats"]),
        "scales": tensor(source["scales"]),
        "opacities": tensor(source["opacities"]),
        "colors": tensor(source["colors"]),
        "device": device,
    }


def _viewmat(position, yaw: float, cam_h: float) -> np.ndarray:
    """Return world-to-camera transform for the project OpenCV convention."""
    cosine, sine = math.cos(yaw), math.sin(yaw)
    right = np.array([cosine, 0.0, -sine])
    down = np.array([0.0, -1.0, 0.0])
    forward = np.array([-sine, 0.0, -cosine])
    camera_to_world = np.stack([right, down, forward], axis=1)
    camera = (
        np.asarray(position, np.float64) +
        np.array([0.0, cam_h, 0.0]))
    world_to_camera = camera_to_world.T
    view = np.eye(4)
    view[:3, :3] = world_to_camera
    view[:3, 3] = -world_to_camera @ camera
    return view


def render_gs(
        gs: dict, position, yaw: float, intrinsics: np.ndarray, hw,
        cam_h: float = config.CAMERA_HEIGHT_M,
        near: float = config.GS_RENDER_NEAR_M,
        far: float = config.GS_RENDER_FAR_M) -> Tuple[np.ndarray, np.ndarray]:
    """Render RGB and expected depth with the standard session contract."""
    height, width = int(hw[0]), int(hw[1])
    device = gs["device"]
    view = torch.from_numpy(
        _viewmat(position, yaw, cam_h)[None]).float().to(device)
    matrices = torch.from_numpy(
        np.asarray(intrinsics, np.float64)[None]).float().to(device)
    output, _alpha, _metadata = rasterization(
        gs["means"], gs["quats"], gs["scales"], gs["opacities"],
        gs["colors"], view, matrices, width, height, render_mode="RGB+ED",
        near_plane=near, far_plane=far, radius_clip=0.0)
    image = output[0]
    rgb = (image[..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
    depth = image[..., 3].contiguous().cpu().numpy().astype(np.float32)
    return rgb, depth


class _GroundAlignedPathfinder:
    """Use the navmesh for XZ proposals and collision source for root Y."""

    def __init__(self, pathfinder, collision: gs_collision.GsCollisionAuthority):
        self._pathfinder = pathfinder
        self._collision = collision

    def get_random_navigable_point(self) -> np.ndarray:
        raw = np.asarray(
            self._pathfinder.get_random_navigable_point(), dtype=np.float64)
        if raw.shape != (3,) or not np.all(np.isfinite(raw)):
            return raw
        ground_y = self._collision.ground_y(raw)
        if ground_y is None:
            return raw
        aligned = raw.copy()
        aligned[1] = float(ground_y)
        return aligned

    def distance_to_closest_obstacle(self, position) -> float:
        return float(self._pathfinder.distance_to_closest_obstacle(
            np.asarray(position, dtype=np.float32)))


class GsSimSession:
    """Renders one GS scene (gsplat) + pathfinder + bbox semantics."""
    def __init__(self, scene: SceneSpec, *, device: str = "cuda", heights=None,
                 hfov: float = config.HFOV_DEG, vfov: float = config.VFOV_DEG):
        if not isinstance(scene, SceneSpec):
            raise TypeError("GS session requires a verified SceneSpec")
        source = scene_pool.verify_scene_source_asset_paths(scene, (
            ("scene", scene.scene_path),
            ("navmesh", scene.navmesh_path),
            ("semantic", scene.semantic_path),
            ("collision_authority", scene.collision_authority_path),
        ))
        collision_asset = next(
            row for row in source["source_assets"]
            if row["role"] == "collision_authority")
        semantic_asset = next(
            row for row in source["source_assets"]
            if row["role"] == "semantic")
        self._collision = gs_collision.load_collision_artifact(
            scene.collision_authority_path,
            expected_file_sha256=collision_asset["sha256"])
        if self._collision.scene_id != scene.scene_id:
            raise ValueError(
                "GS collision artifact scene identity differs")
        embedded = self._collision.binding_atom()["source_assets"]
        source_prefix = source["source_assets"][:3]
        if embedded[:3] != source_prefix:
            raise ValueError(
                "GS collision authority is bound to different scene assets")
        self.collision_authority_binding = \
            dataset_contracts.resolve_gs_collision_binding(source)
        if (self._collision.geometry_authority_sha256 !=
                self.collision_authority_binding.
                collision_authority_sha256):
            raise ValueError(
                "GS collision runtime disagrees with its source binding")
        self._semantic_source_sha256 = semantic_asset["sha256"]
        self.scene_dir = str(Path(scene.scene_path).parent)
        self.scene_glb = scene.scene_path
        self.scene_id = scene.scene_id
        self.source_dataset = "gs"
        self.official_split = scene.official_split
        self._gs = load_gs(self.scene_glb, device=device)
        self._heights = [float(value) for value in (heights or [config.CAMERA_HEIGHT_M])]
        self._hfov, self._vfov = float(hfov), float(vfov)
        self._K = config.intrinsics(self._hfov, self._vfov)
        self._hw = config.hw()
        self._pf = habitat_sim.PathFinder()
        self._pf.load_nav_mesh(scene.navmesh_path)
        if not self._pf.is_loaded:
            raise RuntimeError(f"navmesh failed to load: {self.scene_dir}")
        self._sem = gs_semantic.load_bbox_index(scene.semantic_path)
        self._sampling_pf = _GroundAlignedPathfinder(
            self._pf, self._collision)
        self._nav_radius = min(config.RADII_M)
        self._nav_height = config.GROUND_ORACLE_HEIGHT_M
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
               cam_h: float = None, hfov: float = None, vfov: float = None
               ) -> RenderObservation:
        # gsplat takes an arbitrary K + size, so any FOV renders continuously (no rebuild).
        cam_h = self._heights[0] if cam_h is None else float(cam_h)
        hfov = self._hfov if hfov is None else float(hfov)
        vfov = self._vfov if vfov is None else float(vfov)
        K = config.intrinsics(hfov, vfov)
        hw = config.hw()
        rgb, depth = render_gs(
            self._gs, position, yaw, K, hw, cam_h=cam_h)
        sensor = SensorProfile.from_values(cam_h, hfov, vfov)
        return RenderObservation(
            rgb=rgb, depth=depth, K=K, sensor=sensor,
            position=np.asarray(position, dtype=np.float64),
            yaw_rad=float(yaw))
    def assign_instances(self, world_points: np.ndarray) -> np.ndarray:
        return self._sem.assign(world_points)
    def frame_semantic_index(
            self, world_points: np.ndarray, instance_ids: np.ndarray):
        """Bind GS semantics to this frame's initial visible depth points."""
        return self._sem.visible_depth_view(
            world_points, instance_ids,
            geometry_authority_sha256=self.collision_authority_binding.
                authority_sha256,
            semantic_source_sha256=self._semantic_source_sha256)
    # -- navmesh (pre-baked; not per-radius recomputable without a mesh) --
    def recompute_navmesh(
            self, radius: float,
            height: float = config.GROUND_ORACLE_HEIGHT_M) -> None:
        if abs(float(height) - config.GROUND_ORACLE_HEIGHT_M) > 1e-9:
            raise ValueError("ground-disc oracle height is fixed")
        self._nav_radius = float(radius)
        self._nav_height = config.GROUND_ORACLE_HEIGHT_M
    @property
    def pathfinder(self):
        return self._pf
    def dist_to_obstacle(self, position) -> float:
        return float(self._collision.bind(
            position, 0.0, radius_m=min(config.RADII_M)).clearance(
                (0.0, 0.0, 0.0)))
    def nav(self, position, yaw: float) -> Nav:
        return self._collision.bind(
            position, yaw, radius_m=self._nav_radius)
    def proposal_nav(self, position, yaw: float, *, radius_m: float) -> Nav:
        """Return the pre-baked navmesh as a non-authoritative proposal ranker.
        The GS physical GT remains the official collision geometry returned
        by :meth:`nav`; this pathfinder may prioritize work but may never label
        a record.
        """
        return Nav(
            self._pf,
            position,
            yaw,
            radius_m=float(radius_m),
            authority="gs_navmesh_proposal",
        )
    # -- sampling (GS depth is all-valid, so no min_valid_depth gate) --
    # Every pose gate this backend applies, written once, so the collector and
    # the threshold pilot cannot reach the floor fit through different ones.
    POSE_SAMPLING = {
        "min_floor": 0.02,
        "min_valid_depth": None,     # splat depth has no holes
        "obstacle_margin_m": None,   # falls back to max(radii) + 0.1
        "max_tries": 400,
    }
    def _pose_valid(self, radii, pose_valid):
        """Source-collision navigability, composed with any caller check.

        The Habitat navmesh proposes poses; the official SAGE-3D collision
        body is the only full-geometry authority allowed to accept them.
        """
        radius = max(float(value) for value in radii)
        def valid(position, yaw):
            geometry_valid = self._collision.bind(
                position, yaw, radius_m=radius).is_navigable((0.0, 0.0, 0.0))
            return bool(
                geometry_valid and
                (pose_valid is None or pose_valid(position, yaw)))
        return valid
    def pose_observations(self, rng, radii, *, yaws=None, pose_valid=None,
                          max_tries=None, on_reject=None):
        """Candidates that reach the floor fit, under this backend's gates."""
        spec = self.POSE_SAMPLING
        return pose_observations(
            self._sampling_pf, self.render, rng, radii,
            min_valid_depth=spec["min_valid_depth"],
            obstacle_margin_m=spec["obstacle_margin_m"],
            max_tries=spec["max_tries"] if max_tries is None else max_tries,
            yaws=yaws, pose_valid=self._pose_valid(radii, pose_valid),
            on_reject=on_reject)

    def _calibration_from_observation(
            self, position, yaw: float, rendered: RenderObservation,
            points_local: np.ndarray, *, on_reject=None):
        ground_y = self._collision.ground_y(position)
        if (ground_y is None or
                abs(float(position[1]) - float(ground_y)) > 1e-6):
            if on_reject is not None:
                on_reject("ground_reference_mismatch")
            return None
        fit = pose_calibration.gs_floor_reference_calibration()
        ratio = pose_calibration.visible_floor_ratio(
            fit.estimate, points_local, rendered.depth.size)
        if ratio < float(self.POSE_SAMPLING["min_floor"]):
            if on_reject is not None:
                on_reject("low_visible_floor_ratio")
            return None
        return pose_calibration.PoseCalibration(
            position=np.asarray(position, dtype=np.float64),
            yaw_rad=float(yaw),
            reference_profile=rendered.sensor,
            reference_observation=rendered,
            canonical_floor_fit=fit,
        )

    def calibration_from_pose(self, position, yaw: float):
        """Rebuild a pooled GS pose under the current source authority."""
        position = np.asarray(position, dtype=np.float64)
        height, hfov, vfov = config.calibration_profile()
        rendered = self.render(position, yaw, height, hfov, vfov)
        validate_render_observation(
            rendered, position=position, yaw=yaw,
            cam_h=height, hfov=hfov, vfov=vfov)
        points_local = perception.to_agent_ground(
            perception.unproject(rendered.depth, rendered.K)[0],
            camera_height=rendered.sensor.nominal_camera_offset_m)
        calibration = self._calibration_from_observation(
            position, yaw, rendered, points_local)
        if calibration is None:
            raise ValueError("pooled GS pose no longer passes floor reference")
        return calibration

    def sample_random_pose(
            self, rng, radii, yaws=None, max_tries=None,
            pose_valid=None, on_reject=None):
        spec = self.POSE_SAMPLING
        for position, yaw, rendered, points_local in self.pose_observations(
                rng, radii, yaws=yaws, pose_valid=pose_valid,
                max_tries=(spec["max_tries"] if max_tries is None
                           else max_tries),
                on_reject=on_reject):
            calibration = self._calibration_from_observation(
                position, yaw, rendered, points_local, on_reject=on_reject)
            if calibration is not None:
                return calibration
        return None
    def close(self) -> None:
        self._gs = None
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()
