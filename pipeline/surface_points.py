"""Visible instance target points; no extra renders."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
from scipy.ndimage import distance_transform_edt

from pipeline import config, perception


MAX_VALIDATED_PIXELS = 24  # per instance: bounds ray casts when an object's depth disagrees with its mesh


def select_target(frame, *, record_uid: str, seed: int = 20260906,
                  existing: dict | None = None, min_distance_m: float = 0.,
                  point_validator=None) -> dict | None:
    """Sample one visible object uniformly, then up to three separated pixels."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        f"{seed}:{record_uid}".encode()).digest()[:8], "big"))
    width, height = frame.sensor.width_px, frame.sensor.height_px
    marker = int(config.TARGET_POINT_MARKER_RADIUS_PX)
    candidates = {}
    for instance in np.unique(frame.pts_sem):
        instance = int(instance)
        if instance <= 0:
            continue
        category = frame.predicate_category(instance)
        if (config.is_structural(category) or
                not config.is_specific_semantic_category(category) or
                any(token in category.casefold()
                    for token in config.SURFACE_TARGET_EXCLUDED_MATERIAL_TOKENS)):
            continue
        indices = np.flatnonzero(frame.pts_sem == instance)
        uv = frame.pts_uv[indices]
        x0, y0 = uv.min(axis=0)
        x1, y1 = uv.max(axis=0)
        mask = np.zeros((y1 - y0 + 3, x1 - x0 + 3), dtype=bool)
        mask[uv[:, 1] - y0 + 1, uv[:, 0] - x0 + 1] = True
        interior = distance_transform_edt(mask)[uv[:, 1] - y0 + 1, uv[:, 0] - x0 + 1]
        keep = ((interior > 2) & (uv[:, 0] >= marker) & (uv[:, 0] < width - marker)
                & (uv[:, 1] >= marker) & (uv[:, 1] < height - marker))
        if np.any(keep):
            candidates[instance] = indices[keep]
    camera = (np.asarray(frame.position, dtype=float) + [0., frame.sensor.nominal_camera_offset_m, 0.]
              if min_distance_m > 0 else None)

    def fill(instance, points):
        """Add up to three separated pixels of one instance; each kept point passes every check."""
        validated = 0
        for index in rng.permutation(candidates[instance]):
            if len(points) >= 3 or validated >= MAX_VALIDATED_PIXELS:
                break
            pixel = frame.pts_uv[index]
            if any(np.sum((pixel - np.asarray(point["pixel_xy_px"])) ** 2)
                   < (2 * marker) ** 2 for point in points):
                continue
            world = perception.world_from_local(
                frame.pts[index:index + 1], frame.position, frame.yaw_rad)[0]
            if point_validator is not None:
                validated += 1
                world = point_validator(pixel, world)
                if world is None:
                    continue
            if camera is not None and float(np.linalg.norm(np.asarray(world) - camera)) < min_distance_m:
                continue
            points.append({"point_id": f"p{len(points) + 1}",
                           "pixel_xy_px": pixel.tolist(), "world_xyz_m": np.asarray(world).tolist()})
        return points

    if existing is not None:
        target = copy.deepcopy(existing)
        instance = int(target["instance_id"])
        if instance in candidates:
            fill(instance, target["points"])
        return target
    if not candidates:
        return None
    first = int(rng.choice(sorted(candidates)))
    for instance in [first] + [int(i) for i in rng.permutation(sorted(candidates)) if int(i) != first]:
        points = fill(instance, [])
        if points:  # an instance whose pixels all fail validation is skipped, not kept empty
            return {"instance_id": instance, "category": frame.predicate_category(instance), "points": points}
    return None
