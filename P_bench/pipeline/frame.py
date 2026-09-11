"""Frame: all per-frame perception evidence, built once and reused by judge.

build_frame is duck-typed on `sim` (needs .render, .assign_instances,
.id_to_cat, and .dist_to_obstacle) and does NOT import Habitat, so a Frame can be constructed
directly from arrays in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import numpy as np

from pipeline import config, perception, objects
from pipeline import floor_plane as floor_plane_module


@dataclass(frozen=True)
class SensorProfile:
    # The sensor's mount offset above the agent root -- the extrinsic Habitat
    # was configured with, not a measured height above any floor. Named apart
    # from the public camera-to-ground height measured against source geometry.
    nominal_camera_offset_m: float
    hfov_deg: float
    vfov_deg: float
    focal_x_px: float
    focal_y_px: float
    width_px: int
    height_px: int
    @classmethod
    def from_values(cls, nominal_camera_offset_m: float, hfov_deg: float,
                    vfov_deg: float):
        width, height = config.resolution()
        K = config.intrinsics(hfov_deg, vfov_deg)
        return cls(float(nominal_camera_offset_m), float(hfov_deg),
                   float(vfov_deg),
                   float(K[0, 0]), float(K[1, 1]), int(width), int(height))
    def to_dict(self) -> dict:
        return {
            "nominal_camera_offset_m": self.nominal_camera_offset_m,
            "hfov_deg": self.hfov_deg,
            "vfov_deg": self.vfov_deg,
            "focal_x_px": self.focal_x_px,
            "focal_y_px": self.focal_y_px,
            "resolution": [self.width_px, self.height_px],
        }


@dataclass(frozen=True)
class RenderObservation:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    sensor: SensorProfile
    # The pose this was taken from. Mandatory: an observation that does not
    # know where it was taken cannot be cached safely, and an optional pose
    # makes the cache guard unfalsifiable -- the validator has nothing to
    # compare, so a reused observation with the right profile silently
    # describes a different place.
    position: np.ndarray
    yaw_rad: float
    # Optional source-native, pixel-aligned stable instance IDs.  Backends
    # with an authenticated renderer mask provide ``(H, W)`` IDs here so
    # frame construction does not reconstruct semantics from every depth
    # point.  Zero is background or an unmapped source identity.
    semantic_instance_ids: Optional[np.ndarray] = None

    def __post_init__(self):
        position = np.asarray(self.position, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(
                f"observation pose position must be three finite numbers: "
                f"{self.position!r}")
        yaw = float(self.yaw_rad)
        if not math.isfinite(yaw):
            raise ValueError(f"observation pose yaw must be finite: {yaw!r}")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "yaw_rad", yaw)
        semantic_ids = self.semantic_instance_ids
        if semantic_ids is not None:
            values = np.asarray(semantic_ids)
            if (values.shape != np.asarray(self.depth).shape or
                    not np.issubdtype(values.dtype, np.integer) or
                    np.any(values < 0)):
                raise ValueError(
                    "semantic instance IDs must be nonnegative integral HxW")
            object.__setattr__(
                self, "semantic_instance_ids",
                np.ascontiguousarray(values, dtype=np.int64))


@dataclass(frozen=True)
class TerminalRGBObservation:
    """A source-bound endpoint render without reconstructed scene semantics."""
    scene_id: str
    scene_glb: str
    position: np.ndarray
    yaw_rad: float
    K: np.ndarray
    rgb: np.ndarray
    sensor: SensorProfile


@dataclass
class Frame:
    frame_id: str
    scene_id: str
    scene_glb: str
    position: np.ndarray          # world xyz (3,)
    yaw_rad: float
    K: np.ndarray
    # The canonical floor plane for this physical pose, fitted once at the
    # frozen calibration profile and shared by every sibling. Never estimated
    # from this observation: a per-image estimate lets the FOV decide where the
    # floor is, and through it which targets are eligible and whether the frame
    # is accepted at all.
    floor_plane: floor_plane_module.FloorPlaneEstimate
    rgb: np.ndarray               # (H,W,3) uint8
    depth: np.ndarray             # (H,W) float32
    pts: np.ndarray               # (N,3) ground xyz
    pts_uv: np.ndarray            # (N,2) source pixel (u,v)
    pts_sem: np.ndarray           # (N,)  instance id per point
    vf: perception.VoxelField     # obstacle occupancy field
    id_to_cat: Dict[int, str]
    objects: List[dict]
    # Machine labels remain in ``id_to_cat``.  Source-native raw labels live
    # only here and are used for structural/specific/contact predicates.
    id_to_predicate_cat: Dict[int, str] = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    sensor: SensorProfile = field(default_factory=lambda: SensorProfile.from_values(
        config.CAMERA_HEIGHT_M, config.HFOV_DEG, config.VFOV_DEG))
    semantic_index: object = None
    @property
    def category_inventory(self) -> Dict[str, int]:
        inv: Dict[str, int] = {}
        for o in self.objects:
            inv[o["category"]] = inv.get(o["category"], 0) + 1
        return inv
    @property
    def camera_height_above_visible_floor_m(self) -> float:
        """Legacy plane-relative distance for depth/oracle processing, not QA.
        ``n . c + d`` for the camera centre ``c = (0, nominal_offset, 0)``,
        which degenerates to ``nominal_offset - floor_y`` on a level floor.
        Public relative camera height uses the configured nominal offset.
        """
        return self.floor_plane.height_above(
            (0.0, self.sensor.nominal_camera_offset_m, 0.0))

    def predicate_category(self, instance_id: int) -> str:
        """Return the source-native category used only by semantic gates."""
        return self.id_to_predicate_cat.get(
            int(instance_id), self.id_to_cat.get(int(instance_id), "unknown"))


# Pose agreement is exact-tolerance, not relative: at scene scale a relative
# tolerance admits millimetres of drift, and a millimetre is a different place.
POSE_POSITION_TOL_M = 1e-6
POSE_YAW_TOL_RAD = 1e-6


def validate_render_observation(rendered, *, position, yaw,
                                cam_h: float, hfov: float, vfov: float) -> None:
    """Check an observation against the request, before anything unprojects it.
    Shared by ``build_frame`` and ``sim.sample_pose`` so a cached observation and
    a fresh one face the same contract, and so nothing is unprojected with
    intrinsics that were never checked: a renderer whose ``sensor`` fields are
    right but whose ``K`` is wrong would otherwise yield a plausible floor plane
    fitted from the wrong geometry.
    """
    if not isinstance(rendered, RenderObservation):
        raise TypeError("sim.render must return RenderObservation")
    sensor = rendered.sensor
    if abs(sensor.nominal_camera_offset_m - float(cam_h)) > 1e-6:
        raise ValueError(
            f"camera height mismatch: requested {cam_h}, "
            f"rendered {sensor.nominal_camera_offset_m}")
    if abs(sensor.hfov_deg - float(hfov)) > 1e-6 or abs(sensor.vfov_deg - float(vfov)) > 1e-6:
        raise ValueError("camera FOV mismatch between request and rendered observation")
    rgb, depth, K = rendered.rgb, rendered.depth, np.asarray(rendered.K)
    if rgb.shape[:2] != (sensor.height_px, sensor.width_px) or depth.shape != rgb.shape[:2]:
        raise ValueError("rendered array shape does not match SensorProfile")
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("rendered intrinsics must be a finite 3x3 matrix")
    expected = np.asarray(config.intrinsics(hfov, vfov), dtype=np.float64)
    if not np.allclose(K, expected, rtol=0.0, atol=1e-6):
        raise ValueError(
            "rendered intrinsics disagree with the declared FOV and resolution")
    requested = np.asarray(position, dtype=np.float64)
    if requested.shape != (3,) or not np.all(np.isfinite(requested)):
        raise ValueError(
            f"requested pose position must be three finite numbers: {position!r}")
    if not math.isfinite(float(yaw)):
        raise ValueError(f"requested pose yaw must be finite: {yaw!r}")
    if not np.allclose(rendered.position, requested,
                       rtol=0.0, atol=POSE_POSITION_TOL_M):
        raise ValueError(
            "rendered observation was taken at a different pose position")
    # Wrap-aware: yaw reaches this from a quaternion on one side and from an
    # unwrapped accumulator on the other, so a whole turn is the same heading.
    if abs(math.remainder(float(rendered.yaw_rad) - float(yaw), 2.0 * math.pi)) \
            > POSE_YAW_TOL_RAD:
        raise ValueError(
            "rendered observation was taken at a different pose yaw")


def build_frame(sim, position, yaw: float, *,
                frame_id: str, scene_id: str, scene_glb: str,
                floor_plane: floor_plane_module.FloorPlaneEstimate,
                cam_h: float = config.CAMERA_HEIGHT_M,
                hfov: float = config.HFOV_DEG,
                vfov: float = config.VFOV_DEG,
                rendered: "RenderObservation" = None) -> Frame:
    """Render one observation and derive all geometric evidence (single unproject).
    `cam_h` = the sensor's nominal mount offset above the agent root.
    `hfov`/`vfov` = the camera's horizontal/vertical field of view (deg); they
    thread through sim.render (which returns the matching K + image size), so a
    different lens yields a genuinely different egocentric image and visible
    geometry. Per-point ids come from the SemanticIndex.
    `floor_plane` is the canonical plane for this physical pose, supplied by the
    caller and mandatory. It is not estimated here: every sibling of a pose must
    answer floor-relative questions against the same surface, or the FOV decides
    the target set and the acceptance verdict through the floor estimate.
    `rendered` supplies an already-rendered observation -- the pose calibration
    render, when the profile being built is the calibration profile -- so the
    same view is not drawn twice. It is a cache, not a bypass: every profile,
    resolution and shape check below still runs against it.
    """
    if not isinstance(floor_plane, floor_plane_module.FloorPlaneEstimate):
        raise TypeError(
            "build_frame needs the pose's canonical FloorPlaneEstimate")
    if rendered is None:
        rendered = sim.render(position, yaw, cam_h, hfov, vfov)
    validate_render_observation(
        rendered, position=position, yaw=yaw,
        cam_h=cam_h, hfov=hfov, vfov=vfov)
    rgb, depth, K, sensor = rendered.rgb, rendered.depth, rendered.K, rendered.sensor
    pts_cam, uv = perception.unproject(depth, K)
    pts = perception.to_agent_ground(
        pts_cam, camera_height=sensor.nominal_camera_offset_m)
    world = perception.world_from_local(pts, position, yaw)
    if rendered.semantic_instance_ids is None:
        sem = sim.assign_instances(world).astype(np.int64)
    else:
        sem = rendered.semantic_instance_ids[uv[:, 1], uv[:, 0]].astype(
            np.int64, copy=False)
    # Collision evidence is a fixed ground-plane projection.  Camera height
    # changes the observation but never changes the physical body or labels.
    obs_mask = perception.obstacle_mask(
        pts, floor_plane, band=config.GROUND_OBSTACLE_BAND_M)
    vf = perception.VoxelField(pts[obs_mask])
    id_to_cat = sim.id_to_cat
    id_to_predicate_cat = getattr(sim, "id_to_predicate_cat", id_to_cat)

    objs = objects.extract_objects(
        pts, uv, sem, id_to_cat,
        predicate_categories=id_to_predicate_cat,
        resolution=(sensor.width_px, sensor.height_px))
    semantic_index = getattr(sim, "semantic_index", None)
    frame_semantic_index = getattr(sim, "frame_semantic_index", None)
    if callable(frame_semantic_index):
        semantic_index = frame_semantic_index(world, sem)
    valid = np.isfinite(depth) & (depth > 0)
    floor_pts = np.abs(floor_plane.height_above_points(pts)) <= 0.10
    quality = {
        "valid_depth_ratio": float(valid.mean()),
        "dist_to_obstacle_m": float(sim.dist_to_obstacle(position)),
        "visible_floor_ratio": float(floor_pts.sum()) / float(depth.size),
    }
    return Frame(
        frame_id=frame_id, scene_id=scene_id, scene_glb=scene_glb,
        position=np.asarray(position, dtype=np.float64), yaw_rad=float(yaw),
        K=np.asarray(K, dtype=np.float64), floor_plane=floor_plane,
        rgb=rgb, depth=np.asarray(depth),
        pts=pts, pts_uv=uv, pts_sem=sem, vf=vf,
        id_to_cat=id_to_cat, objects=objs,
        id_to_predicate_cat=id_to_predicate_cat, quality=quality,
        sensor=sensor,
        semantic_index=semantic_index,
    )


def _future_world_pose(base: Frame, local_pose) -> tuple[np.ndarray, float]:
    values = (
        (local_pose["x"], local_pose["z"], local_pose["heading_deg"])
        if isinstance(local_pose, dict) else local_pose)
    x, z, heading_deg = (float(v) for v in values)
    position = perception.world_from_local(
        np.array([[x, 0.0, z]], dtype=np.float64),
        base.position, base.yaw_rad)[0]
    return position, base.yaw_rad - math.radians(heading_deg)


def build_terminal_rgb_observation(
        sim, base: Frame, local_pose) -> TerminalRGBObservation:
    """Render only native endpoint RGB; do not unproject or assign semantics."""
    position, yaw = _future_world_pose(base, local_pose)
    sensor = base.sensor
    rgb_renderer = getattr(sim, "render_rgb", None)
    if callable(rgb_renderer):
        rgb = np.asarray(rgb_renderer(
            position, yaw,
            sensor.nominal_camera_offset_m,
            sensor.hfov_deg,
            sensor.vfov_deg,
        ))
        if (rgb.shape != (sensor.height_px, sensor.width_px, 3) or
                rgb.dtype != np.uint8):
            raise ValueError("terminal RGB renderer returned an invalid image")
        return TerminalRGBObservation(
            scene_id=base.scene_id,
            scene_glb=base.scene_glb,
            position=np.asarray(position, dtype=np.float64),
            yaw_rad=float(yaw),
            K=config.intrinsics(sensor.hfov_deg, sensor.vfov_deg),
            rgb=rgb,
            sensor=sensor,
        )
    rendered = sim.render(
        position, yaw,
        sensor.nominal_camera_offset_m,
        sensor.hfov_deg,
        sensor.vfov_deg,
    )
    validate_render_observation(
        rendered, position=position, yaw=yaw,
        cam_h=sensor.nominal_camera_offset_m,
        hfov=sensor.hfov_deg,
        vfov=sensor.vfov_deg,
    )
    return TerminalRGBObservation(
        scene_id=base.scene_id,
        scene_glb=base.scene_glb,
        position=rendered.position,
        yaw_rad=rendered.yaw_rad,
        K=np.asarray(rendered.K),
        rgb=rendered.rgb,
        sensor=rendered.sensor,
    )


def build_future_frame(sim, base: Frame, local_pose, *, checkpoint_index: int) -> Frame:
    """Render a checkpoint pose expressed in the base Frame's local SE(2) frame.
    The checkpoint inherits the base pose's canonical plane, transformed into
    its own local frame. It does not re-fit: the reference surface an action is
    judged against is decided once, at the pose the action starts from.
    """
    position, yaw = _future_world_pose(base, local_pose)
    sensor = base.sensor
    return build_frame(
        sim, position, yaw,
        frame_id=f"{base.frame_id}-k{checkpoint_index}",
        scene_id=base.scene_id,
        scene_glb=base.scene_glb,
        floor_plane=base.floor_plane.transformed(
            from_position=base.position, from_yaw_rad=base.yaw_rad,
            to_position=position, to_yaw_rad=yaw),
        cam_h=sensor.nominal_camera_offset_m,
        hfov=sensor.hfov_deg,
        vfov=sensor.vfov_deg,
    )
