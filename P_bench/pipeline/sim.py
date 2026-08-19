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

from pipeline import config, floor_plane, perception, pose_calibration, semantic
from pipeline.frame import (
    RenderObservation, SensorProfile, validate_render_observation)
from pipeline.geometry import GeometryQuery


NAVMESH_LATERAL_SNAP_MAX_M = config.NAVMESH_LATERAL_SNAP_MAX_M
NAVMESH_SNAP_EPSILON_M = config.NAVMESH_SNAP_EPSILON_M


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


class Nav:
    """Radius-conditioned navmesh oracle bound to a local SE(2) origin."""
    def __init__(self, pathfinder, pos, yaw: float, *, radius_m=None,
                 authority="navmesh"):
        self._pf = pathfinder
        self._pos = np.asarray(pos, dtype=np.float64)
        self._yaw = float(yaw)
        self.radius_m = radius_m
        self.authority = authority
    def _world(self, x: float, z: float) -> np.ndarray:
        return perception.world_from_local(np.array([[x, 0.0, z]]), self._pos, self._yaw)[0]
    def is_navigable(self, pose, max_y_delta: float = config.NAV_Y_DELTA_M) -> bool:
        return self.query_pose(
            pose, max_y_delta=max_y_delta).navigable
    def clearance(self, pose) -> float:
        return max(0.0, self.query_pose(pose).clearance_m)
    def _rejection_source(self, point, max_y_delta: float) -> str | None:
        """Name the non-navigable causes that are not a lateral collision.

        Habitat rejects a point when its nearest polygon lies farther than
        ``NAVMESH_LATERAL_SNAP_MAX_M`` horizontally or more than
        ``max_y_delta`` vertically, and ``snap_point`` projects through the
        same query. This navmesh is rebuilt per body radius, so a centre that
        still projects laterally onto it cannot overlap an obstacle at that
        radius: the walkable surface beneath it sits at another height -- a
        stair tread, a ledge, a floor below a mezzanine. A missing projection
        means no navmesh at all within habitat's pick extent: a hole, not a
        wall. Anything the two clauses cannot explain is withheld rather than
        published as contact.
        """
        snap = getattr(self._pf, "snap_point", None)
        if snap is None:
            return None
        snapped = np.asarray(
            snap(np.asarray(point, dtype=np.float32)), dtype=np.float64)
        if not np.isfinite(snapped).all():
            return "unsupported_floor"
        lateral = float(np.linalg.norm(snapped[[0, 2]] - point[[0, 2]]))
        if lateral >= NAVMESH_LATERAL_SNAP_MAX_M - NAVMESH_SNAP_EPSILON_M:
            return None
        if abs(float(snapped[1]) - float(point[1])) > (
                float(max_y_delta) + NAVMESH_SNAP_EPSILON_M):
            return "unsupported_floor"
        return "navmesh_boundary"

    def query_pose(
            self, pose, max_y_delta: float = config.NAV_Y_DELTA_M
    ) -> GeometryQuery:
        point = np.asarray(
            self._world(float(pose[0]), float(pose[1])), dtype=np.float32)
        navigable = bool(
            self._pf.is_navigable(point, max_y_delta=max_y_delta))
        clearance = (
            float(self._pf.distance_to_closest_obstacle(point))
            if navigable else 0.0)
        return GeometryQuery(
            navigable=navigable,
            clearance_m=clearance,
            geometry_source=(
                None if navigable
                else self._rejection_source(point, max_y_delta)),
        )
    def query_many(
            self, poses, max_y_delta: float = config.NAV_Y_DELTA_M
    ) -> list[GeometryQuery]:
        return [
            self.query_pose(pose, max_y_delta=max_y_delta)
            for pose in poses
        ]
    def closest_obstacle(self, pose) -> dict:
        point = self._world(float(pose[0]), float(pose[1]))
        hit = self._pf.closest_obstacle_surface_point(point)
        boundary = np.asarray(hit.hit_pos, dtype=np.float64)
        normal = np.asarray(hit.hit_normal, dtype=np.float64)
        world_normal = (
            normal.tolist()
            if normal.shape == (3,) and np.isfinite(normal).all()
            else None
        )
        delta_xz = boundary[[0, 2]] - point[[0, 2]]
        norm = float(np.linalg.norm(delta_xz))
        surface = None
        protocol = None
        reach = (float(self.radius_m) - norm
                 if self.radius_m is not None else float("nan"))
        if (self.radius_m is not None and
                math.isfinite(float(self.radius_m)) and
                float(self.radius_m) > 0.0 and
                np.isfinite(boundary).all() and
                math.isfinite(norm) and
                1e-9 < norm <= config.CONTACT_NAVMESH_BOUNDARY_MAX_M and
                0.0 < reach < float(self.radius_m)):
            surface = point.copy()
            # ``point`` is the first non-navigable disc centre and ``boundary``
            # is the closest point back on the radius-expanded navmesh, so the
            # obstacle lies on the opposite side of the centre. The centre has
            # already crossed the boundary by ``norm``, and the disc touches the
            # obstacle at the boundary, so the surface sits ``radius - norm``
            # away rather than a full radius.
            surface[[0, 2]] -= delta_xz / norm * reach
            surface[1] = boundary[1]
            protocol = "radius_extrapolated_navmesh_boundary_v2"
        return {
            "world_point": (
                np.asarray(surface, dtype=float).tolist()
                if surface is not None else None),
            "world_normal": world_normal,
            "distance_m": reach if surface is not None else None,
            "configuration_boundary_world_point": boundary.tolist(),
            "configuration_boundary_distance_m": float(hit.hit_dist),
            "surface_protocol": protocol,
        }
    def rebase(self, pose):
        point = self._world(float(pose[0]), float(pose[1]))
        yaw = self._yaw - math.radians(float(pose[2]))
        return Nav(self._pf, point, yaw, radius_m=self.radius_m,
                   authority=self.authority)


def render_profile_set(heights, fovs):
    """Exact ``(height, hfov, vfov)`` triples a session must be able to render.
    Habitat can only render sensors it registered, and ``sample_pose`` always
    asks for the calibration profile, so a run publishing neither 1.5 m nor
    79 deg would fail on every pose. The calibration profile joins the RENDER
    set only -- the published lists are left untouched, so the sibling set a run
    emits is unchanged.
    A triple set, not a height list crossed with a FOV list: the cross terms are
    profiles nobody renders, and each one costs a registered colour and depth
    sensor at full resolution. Two published heights and one FOV needed three
    profiles and the product asked Habitat for six.
    Sorted, so sensor registration order does not depend on set iteration.
    """
    profiles = {(float(height), float(hfov), float(vfov))
                for hfov, vfov in ((pair[0], pair[1]) for pair in fovs)
                for height in heights}
    profiles.add(config.calibration_profile())
    return tuple(sorted(profiles))


def pose_observations(pf, render, rng, radii, *, min_valid_depth=None,
                      yaws=None, max_tries=2000, pose_valid=None,
                      obstacle_margin_m=None, on_reject=None):
    """Yield every candidate observation that reaches the floor fit.
    The cut is deliberate and load-bearing: everything a pose must pass *before*
    its floor is judged lives here, and nothing of the floor judgement does. The
    threshold pilot consumes this generator directly, so the population it
    measures floor-gate yield on is the population the collector actually sees.
    A pilot that drew its own poses would answer a different question and the
    yield curve would be quietly wrong.
    Yields ``(position, yaw, rendered, points_local)``. ``render(pos, yaw,
    height, hfov, vfov)`` returns a ``RenderObservation``; ``min_valid_depth``
    gates Habitat mesh depth holes (GS depth is always valid, so pass None).
    The observation is rendered at ``config.calibration_profile()`` -- taken from
    the frozen constant rather than the published list, so the published set
    cannot recalibrate every scene -- and handed back rather than discarded, so
    the same view is not drawn twice.
    ``on_reject`` receives the reason a candidate was discarded, so a collector
    can report why a scene yielded nothing instead of only that it did.
    """
    min_clear = (max(radii) + 0.1 if obstacle_margin_m is None
                 else float(obstacle_margin_m))
    if yaws is None:
        yaws = [math.radians(a) for a in range(0, 360, 45)]
    height, hfov, vfov = config.calibration_profile()
    def reject(reason):
        if on_reject is not None:
            on_reject(reason)
    for _ in range(max_tries):
        pos = pf.get_random_navigable_point()
        if not np.all(np.isfinite(pos)):
            reject("non_finite_navigable_point")
            continue
        if pf.distance_to_closest_obstacle(pos) < min_clear:
            reject("too_close_to_obstacle")
            continue
        yaw = float(rng.choice(yaws))
        if pose_valid is not None and not pose_valid(pos, yaw):
            reject("pose_rejected_by_backend")
            continue
        rendered = render(pos, yaw, height, hfov, vfov)
        # Validate before unprojecting: a wrong K would fit a plausible plane
        # from geometry that was never checked.
        validate_render_observation(
            rendered, position=pos, yaw=yaw,
            cam_h=height, hfov=hfov, vfov=vfov)
        depth, K = rendered.depth, rendered.K
        if min_valid_depth is not None and \
                float((np.isfinite(depth) & (depth > 0)).mean()) < min_valid_depth:
            reject("low_valid_depth_ratio")
            continue
        yield pos, yaw, rendered, perception.to_agent_ground(
            perception.unproject(depth, K)[0],
            camera_height=float(rendered.sensor.nominal_camera_offset_m),
        )


def sample_pose(pf, render, rng, radii, *, min_floor=0.05, min_valid_depth=None,
                yaws=None, max_tries=2000, pose_valid=None,
                obstacle_margin_m=None, on_reject=None):
    """Draw a pose that has a usable canonical floor plane; ``None`` on failure.
    The floor judgement, and only that: every earlier gate lives in
    :func:`pose_observations`, which this consumes.
    """
    def reject(reason):
        if on_reject is not None:
            on_reject(reason)
    for pos, yaw, rendered, pts in pose_observations(
            pf, render, rng, radii, min_valid_depth=min_valid_depth, yaws=yaws,
            max_tries=max_tries, pose_valid=pose_valid,
            obstacle_margin_m=obstacle_margin_m, on_reject=on_reject):
        fit = floor_plane.fit_floor_plane(pts)
        if not fit.fitted:
            for reason in fit.rejection_reasons:
                reject(reason)
            continue
        # Shared with the threshold pilot, which reports this ratio without
        # gating on it, so the two cannot measure different quantities.
        vfr = pose_calibration.visible_floor_ratio(
            fit.estimate, pts, rendered.depth.size)
        if vfr < min_floor:
            reject("low_visible_floor_ratio")
            continue
        return pose_calibration.PoseCalibration(
            position=np.array(pos, dtype=np.float64), yaw_rad=yaw,
            reference_profile=rendered.sensor,
            reference_observation=rendered,
            canonical_floor_fit=fit)
    return None


class SimSession:
    """Renders one R2R/MP3D scene and owns its semantic index."""
    def __init__(self, scene_glb: str, *, scene_dataset_cfg: Optional[str] = None,
                 semantic_format: str = "mp3d_ply",
                 source_dataset: str = "r2r",
                 official_split: Optional[str] = None,
                 heights=None, hfov: float = config.HFOV_DEG,
                 vfov: float = config.VFOV_DEG, fovs=None,
                 semantic_query_workers: int = 1):
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
        # Register the calibration profile privately: sample_pose always needs
        # it, while the published lists above stay exactly what a run emits.
        self._render_profiles = render_profile_set(self._heights, self._fovs)
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = scene_glb
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
        obs = self._sim.get_sensor_observations()
        actual_height = matches[0][0]
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
            self._nav_radius = float(radius)
            self._nav_height = float(height)
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
        self._nav_radius = float(radius)
        self._nav_height = float(height)
    @property
    def pathfinder(self):
        return self._sim.pathfinder
    def dist_to_obstacle(self, position) -> float:
        return float(self._sim.pathfinder.distance_to_closest_obstacle(
            np.array(position, dtype=np.float32)))
    def nav(self, position, yaw: float) -> Nav:
        return Nav(self._sim.pathfinder, position, yaw,
                   radius_m=self._nav_radius, authority="navmesh")
    def proposal_nav(self, position, yaw: float, *, radius_m: float) -> Nav:
        """Return a radius-conditioned navmesh used only to rank proposals."""
        self.recompute_navmesh(
            float(radius_m), height=config.GROUND_ORACLE_HEIGHT_M)
        return Nav(
            self._sim.pathfinder,
            position,
            yaw,
            radius_m=float(radius_m),
            authority="navmesh_proposal",
        )
    # -- sampling --
    # Every pose gate this backend applies, written once. Habitat and GS genuinely
    # differ -- depth-hole ratio, clearance margin, retry budget, visible-floor
    # floor -- which is exactly why neither may be spelled out at a call site:
    # two copies is how the collector and the threshold pilot drift apart.
    POSE_SAMPLING = {
        "min_floor": 0.05,
        "min_valid_depth": 0.85,   # Habitat meshes can have depth holes
        "obstacle_margin_m": config.BENCH_SAFE_CLEARANCE_M,
        "max_tries": 2000,
    }
    def pose_observations(self, rng, radii, *, yaws=None, pose_valid=None,
                          max_tries=None, on_reject=None):
        """Candidates that reach the floor fit, under this backend's gates."""
        spec = self.POSE_SAMPLING
        return pose_observations(
            self._sim.pathfinder, self.render, rng, radii,
            min_valid_depth=spec["min_valid_depth"],
            obstacle_margin_m=spec["obstacle_margin_m"],
            max_tries=spec["max_tries"] if max_tries is None else max_tries,
            yaws=yaws, pose_valid=pose_valid, on_reject=on_reject)
    def sample_random_pose(
            self, rng, radii, yaws=None, max_tries=None,
            pose_valid=None, on_reject=None):
        # The renderer is passed unbound: sample_pose supplies the frozen
        # calibration profile, so this session's published heights and FOV
        # cannot move the calibration.
        spec = self.POSE_SAMPLING
        return sample_pose(
            self._sim.pathfinder, self.render, rng, radii,
            min_floor=spec["min_floor"],
            min_valid_depth=spec["min_valid_depth"],
            obstacle_margin_m=spec["obstacle_margin_m"],
            max_tries=spec["max_tries"] if max_tries is None else max_tries,
            yaws=yaws, pose_valid=pose_valid, on_reject=on_reject)
    def close(self) -> None:
        self._sim.close()
        directory = getattr(self, "_navmesh_cache_dir", None)
        if hasattr(directory, "cleanup"):
            directory.cleanup()
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()
