"""Pick the one public sensor/body setting a pose publishes under.

Enumerating every camera height, FOV and body radius for a single pose yields
many near-duplicate questions of the same viewpoint. Sampling one setting per
pose spreads the benchmark over more scenes instead, at the cost of the paired
do(body)/do(FOV) comparison the cross product used to provide.

The draw deliberately does not touch the pose sampler's RNG. That stream is
consumed once per rejected candidate pose, so reading from it would make the
published radius depend on how many poses happened to be rejected first.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence, Tuple

from pipeline import config


SETTING_SAMPLING_POLICY = "one_sensor_one_body_per_pose.v1"
_STREAM = "public-setting-v1"


@dataclass(frozen=True)
class PublicSetting:
    """The single sensor profile and body a pose is published with."""

    nominal_camera_height_m: float
    fov_index: int
    fov: Tuple[float, float]
    radius_m: float


def _axis_index(seed: int, scene_id: str, pose_index: int,
                axis: str, modulus: int) -> int:
    """Index one option grid from a stream that only the pose identity feeds."""
    digest = hashlib.sha256(
        f"{int(seed)}:{scene_id}:{int(pose_index)}:{_STREAM}:{axis}"
        .encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(modulus)


def selection_errors(prefix: str, selection: dict, outcomes,
                     required_radii: Sequence[float],
                     expected_policy: str | None) -> list:
    """Check a selection's one-setting claim against the run that made it.

    ``expected_policy`` comes from hash-verified run metadata, never from the
    record, so dropping the marker cannot buy a record the older and weaker
    multi-radius contract. Runs collected before the policy existed declare no
    expectation, and their records must not claim one either.
    """
    claimed = selection.get("setting_sampling_policy")
    if claimed != expected_policy:
        return [f"{prefix} setting_sampling_policy is {claimed!r} but the "
                f"run declares {expected_policy!r}"]
    if claimed is None:
        return []
    if claimed != SETTING_SAMPLING_POLICY:
        return [f"{prefix} unsupported setting sampling policy {claimed!r}"]
    if len(required_radii) != 1:
        return [f"{prefix} one-setting selection must declare exactly one "
                f"body radius, found {list(required_radii)}"]
    published = float(required_radii[0])
    stray = sorted({
        round(float((outcome.get("body") or {}).get("radius_m", -1)), 6)
        for outcome in outcomes} - {published})
    if stray:
        return [f"{prefix} outcome does not use the published body radius "
                f"{published}: {stray}"]
    return []


def intervention_descriptor(setting, grid_type: str, grid_changed, *,
                            setting_policy: str | None = None):
    """Describe what actually varies across a pose's published records.

    A run that enumerates the height x FOV grid publishes several records per
    pose and really does intervene on the sensor. A sampled pose publishes one,
    so nothing varies and the honest descriptor is the base case -- the same
    answer `collection_cli.sensor_intervention` gives for a one-entry grid.
    """
    if setting is None:
        return grid_type, grid_changed
    return "base", []


def sample_public_setting(
        *, seed: int, scene_id: str, pose_index: int,
        heights: Sequence[float], fovs: Sequence[Sequence[float]],
        radii: Sequence[float]) -> PublicSetting:
    """Draw this pose's camera height, FOV pair and body radius."""
    if not heights or not fovs or not radii:
        raise ValueError("every setting axis needs at least one option")
    fov_index = _axis_index(seed, scene_id, pose_index, "fov", len(fovs))
    return PublicSetting(
        nominal_camera_height_m=float(
            heights[_axis_index(seed, scene_id, pose_index,
                                "height", len(heights))]),
        fov_index=fov_index,
        fov=tuple(float(value) for value in fovs[fov_index]),
        radius_m=float(
            radii[_axis_index(seed, scene_id, pose_index,
                              "radius", len(radii))]),
    )
