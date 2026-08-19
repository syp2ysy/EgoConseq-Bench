"""One physical pose's canonical floor calibration (P0-4 stage 1).

The canonical floor plane is established **once per physical pose** and shared
by every sensor sibling. R2R/B1K fit it from a fixed reference profile. GS uses
the level floor frame bound by its official collision authority because
expected-depth surface noise is not a stable plane fit; RGB-D still certifies
every swept corridor.

Habitat-free: this holds the render output rather than producing it, so the whole
calibration path is testable without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from pipeline import config
from pipeline.actions import wrap_deg as _wrap_deg
from pipeline.floor_plane import FloorPlaneEstimate, FloorPlaneFitResult
from pipeline.frame import RenderObservation, SensorProfile


def gs_floor_reference_calibration() -> FloorPlaneFitResult:
    """Return the level local plane established by the GS source authority.

    Fit diagnostics are deliberately empty rather than fabricated. The GS
    record separately stores the exact source-bound floor-reference atom that
    justifies this plane.
    """
    return FloorPlaneFitResult(
        estimate=FloorPlaneEstimate(
            normal_local=(0.0, 1.0, 0.0), offset_m=0.0),
        rejection_reasons=(),
        seed_y_m=None,
        candidate_band_m=None,
        support_count=0,
        support_extent_x_m=0.0,
        support_extent_z_m=0.0,
        support_cell_count=0,
        support_area_m2=0.0,
        tilt_deg=0.0,
        plane_y_at_origin_m=0.0,
    )


@dataclass(frozen=True)
class PoseCalibration:
    """A pose with a usable canonical floor plane and its reference render.

    A rejected calibration cannot be wrapped in one of these. This prevents a
    pose whose floor could not be established from reaching a Frame through an
    implicit fallback height.
    """
    position: np.ndarray
    yaw_rad: float
    reference_profile: SensorProfile
    reference_observation: RenderObservation
    canonical_floor_fit: FloorPlaneFitResult
    def __post_init__(self):
        if not isinstance(self.canonical_floor_fit, FloorPlaneFitResult):
            raise ValueError("canonical_floor_fit must be a FloorPlaneFitResult")
        if not self.canonical_floor_fit.fitted:
            raise ValueError(
                "a pose calibration requires a fitted canonical floor plane; "
                f"got {list(self.canonical_floor_fit.rejection_reasons)}")
        height, hfov, vfov = config.calibration_profile()
        profile = self.reference_profile
        if (abs(profile.nominal_camera_offset_m - height) > 1e-6 or
                abs(profile.hfov_deg - hfov) > 1e-6 or
                abs(profile.vfov_deg - vfov) > 1e-6):
            raise ValueError(
                "canonical floor plane must come from the calibration profile "
                f"({height} m / {hfov} deg), got "
                f"{profile.nominal_camera_offset_m} m / {profile.hfov_deg} deg")
        if self.reference_observation.sensor != profile:
            raise ValueError(
                "reference observation does not match the reference profile")
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64))
        object.__setattr__(self, "yaw_rad", float(self.yaw_rad))
    @property
    def canonical_plane(self) -> FloorPlaneEstimate:
        """The plane every sibling of this pose shares, in the pose-local frame."""
        return self.canonical_floor_fit.estimate


def pose_sampling_radii(radii):
    """Use only primary benchmark bodies when selecting observation poses."""
    values = sorted({float(radius) for radius in radii})
    if not values or values[0] <= 0:
        raise ValueError("pose sampling requires positive primary body radii")
    return values


def derived_pose_seed(base_seed, scene_id, pose_index, stream) -> int:
    payload = (
        f"{int(base_seed)}\0{scene_id}\0{int(pose_index)}\0{stream}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def prepare_pose_sampling(
    sampler, radii, *, base_seed, scene_id, pose_index,
) -> np.random.Generator:
    """Restore canonical geometry and seed independent pose-local RNG streams.
    Shared by the collector and the threshold pilot. Both must rebake the
    navmesh at the largest primary radius and seed the pathfinder identically,
    or they draw from different pose distributions and the pilot's floor-gate
    yield does not describe the collection it is meant to calibrate.
    """
    sample_radii = pose_sampling_radii(radii)
    sampler.recompute_navmesh(
        max(sample_radii), height=config.GROUND_ORACLE_HEIGHT_M)
    sampler.pathfinder.seed(
        derived_pose_seed(base_seed, scene_id, pose_index, "pathfinder"))
    return np.random.default_rng(
        derived_pose_seed(base_seed, scene_id, pose_index, "numpy"))


def pose_has_publication_clearance(sampler, position, yaw_rad: float) -> bool:
    """Require the start pose to support the published safe-action margin."""
    clearance = sampler.nav(position, yaw_rad).clearance((0.0, 0.0, 0.0))
    return (
        clearance is not None
        and float(clearance) >= config.BENCH_SAFE_CLEARANCE_M - 1e-9
    )


# Half-thickness of the slab counted as "floor" when scoring an image's visible
# floor composition. Shared so the collector's gate and the pilot's report
# measure the same quantity.
VISIBLE_FLOOR_BAND_M = 0.10


def visible_floor_ratio(plane, points_local, pixel_count) -> float:
    """Fraction of an image whose points lie on the canonical plane.
    Measured against the plane rather than a scalar height, so a tilted or
    offset floor is not misjudged. Shared by ``sim.sample_pose``, which gates on
    it, and by the threshold pilot, which reports it: the pilot deliberately
    does not apply the gate (that would condition its sample on the very
    thresholds it exists to measure) but must not therefore conflate "the plane
    fitted" with "the collector accepted the pose".
    """
    count = int(pixel_count)
    if count <= 0:
        raise ValueError("visible floor ratio needs a positive pixel count")
    distance = np.abs(plane.height_above_points(points_local))
    return float((distance <= VISIBLE_FLOOR_BAND_M).sum()) / float(count)


def pose_is_diverse(position, yaw_rad: float, exclusions, *,
                    min_position_m: float, min_yaw_deg: float) -> bool:
    """Reject a pose only when position and heading both repeat a prior pose."""
    candidate = np.asarray(position, dtype=np.float64)
    for prior in exclusions:
        prior_position = np.asarray(prior["position"], dtype=np.float64)
        distance = float(np.linalg.norm(candidate - prior_position))
        yaw_delta = abs(_wrap_deg(np.rad2deg(
            float(yaw_rad) - float(prior["yaw_rad"]))))
        if (distance < float(min_position_m) and
                yaw_delta < float(min_yaw_deg)):
            return False
    return True


def load_pose_exclusions(path) -> dict:
    """Read the collector's pose-exclusion payload, or {} when absent."""
    if not path:
        return {}
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "egoconseq.pose_exclusions.v1":
        raise ValueError("unsupported pose exclusion schema")
    scenes = payload.get("scenes")
    if not isinstance(scenes, dict):
        raise ValueError("pose exclusions must contain a scenes object")
    return {str(scene_id): list(values) for scene_id, values in scenes.items()}
