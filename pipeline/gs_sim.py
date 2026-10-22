"""GS-scene backend: same interface as sim.SimSession, but renders with gsplat.

A scene dir holds `scene.gs.ply` (gaussians, Habitat Y-up frame), `scene.navmesh`
(pre-baked, loaded standalone by habitat's PathFinder), and `labels.json`
(InteriorGS 3D bboxes -> gs_semantic). Native collision execution rebuilds
radius-specific Habitat NavMeshes from the complete official SAGE-3D collision
USD. The supplied NavMesh remains a pose proposal source. The pure-numpy pipeline
(frame/consequence/record/...) is unchanged.

Needs gsplat's CUDA ext on PATH: run with
    PATH=<env>/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, Tuple

import numpy as np
import habitat_sim
import torch
from pipeline import config, dataset_contracts, gs_collision, gs_semantic, gs_source, scene_pool
from pipeline.frame import RenderObservation, SensorProfile
from pipeline.scene_pool import SceneSpec


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


def _rasterize_gs(
        gs: dict, position, yaw: float, intrinsics: np.ndarray, hw,
        *, cam_h: float, near: float, far: float,
        render_mode: str) -> torch.Tensor:
    height, width = int(hw[0]), int(hw[1])
    device = gs["device"]
    gs.pop("_last_camera", None)
    view = torch.from_numpy(
        _viewmat(position, yaw, cam_h)[None]).float().to(device)
    matrices = torch.from_numpy(
        np.asarray(intrinsics, np.float64)[None]).float().to(device)
    output, _alpha, _metadata = rasterization(
        gs["means"], gs["quats"], gs["scales"], gs["opacities"],
        gs["colors"], view, matrices, width, height, render_mode=render_mode,
        near_plane=near, far_plane=far, radius_clip=0.0)
    gs["_last_camera"] = {
        "view_matrix": view[0].detach().cpu().numpy().copy(),
        "intrinsics": matrices[0].detach().cpu().numpy().copy(),
        "position": np.asarray(position, dtype=np.float64).copy(),
        "yaw": float(yaw), "cam_h": float(cam_h),
        "hw": [height, width], "render_mode": render_mode,
    }
    return output[0]


def render_gs(
        gs: dict, position, yaw: float, intrinsics: np.ndarray, hw,
        cam_h: float = config.CAMERA_HEIGHT_M,
        near: float = config.GS_RENDER_NEAR_M,
        far: float = config.GS_RENDER_FAR_M) -> Tuple[np.ndarray, np.ndarray]:
    """Render RGB and expected depth with the standard session contract."""
    image = _rasterize_gs(
        gs, position, yaw, intrinsics, hw, cam_h=cam_h,
        near=near, far=far, render_mode="RGB+ED")
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
        self._pf = habitat_sim.PathFinder()
        self._pf.load_nav_mesh(scene.navmesh_path)
        if not self._pf.is_loaded:
            raise RuntimeError(f"navmesh failed to load: {self.scene_dir}")
        self._sem = gs_semantic.load_bbox_index(scene.semantic_path)
        self._sampling_pf = _GroundAlignedPathfinder(
            self._pf, self._collision)
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
    # -- proposal geometry and native execution --
    @property
    def pathfinder(self):
        return self._pf


    def support_ground(self, position) -> dict:
        """Reference camera offsets to authenticated source support surfaces."""
        import shapely

        point = np.asarray(position, dtype=np.float64)
        ground_y = self._collision.ground_y(point)
        if ground_y is None:
            raise ValueError("GS pose has no authenticated source support floor")
        supported = any(
            abs(float(level.ground_y_m) - float(ground_y)) < 1e-6 and
            shapely.covers(level.support, shapely.Point(float(point[0]), float(point[2])))
            for level in self._collision._ground_levels)
        if not supported:
            raise ValueError("GS pose lies outside the source support floor surface")
        aligned = point.copy()
        aligned[1] = float(ground_y)
        return {
            "ground_y": float(ground_y), "agent_position": aligned.tolist(),
            "method": "authenticated_sage3d_source_floor",
            "evidence": self._collision.binding_atom(),
        }


    def sensor_pose_evidence(self, position, yaw, cam_h, hfov, vfov) -> dict:
        """Read camera position from the exact view matrix sent to gsplat."""
        camera_state = self._gs.get("_last_camera")
        expected_intrinsics = config.intrinsics(hfov, vfov).astype(np.float32)
        if (camera_state is None or
                not np.allclose(camera_state["position"], position, rtol=0, atol=1e-7) or
                not math.isclose(camera_state["yaw"], float(yaw), rel_tol=0, abs_tol=1e-12) or
                not math.isclose(camera_state["cam_h"], float(cam_h), rel_tol=0, abs_tol=1e-9) or
                not np.array_equal(camera_state["intrinsics"], expected_intrinsics)):
            raise ValueError("GS camera evidence requires a matching completed render")
        view = camera_state["view_matrix"]
        camera_to_world = np.linalg.inv(view.astype(np.float64))
        camera = camera_to_world[:3, 3]
        return {
            "position_world": camera.tolist(),
            "ground_y": float(position[1]),
            "camera_height_m": float(camera[1] - float(position[1])),
            "method": "gsplat_rendered_view_matrix_inverse",
            "view_matrix_world_to_camera": view.astype(float).tolist(),
            "intrinsics": camera_state["intrinsics"].astype(float).tolist(),
            "resolution_hw": camera_state["hw"],
            "render_mode": camera_state["render_mode"],
            "hfov_deg": float(hfov), "vfov_deg": float(vfov),
        }

    def sample_navigable_point(self):
        return self._sampling_pf.get_random_navigable_point()
    def dist_to_obstacle(self, position) -> float:
        return float(self._collision.bind(
            position, 0.0, radius_m=min(config.RADII_M)).clearance(
                (0.0, 0.0, 0.0)))
    def close(self) -> None:
        self._gs = None
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()
