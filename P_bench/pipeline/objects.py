"""Visible-object extraction and contact attribution (pure numpy).

Both operate on the SINGLE ground point cloud produced once by build_frame,
sliced by semantic instance id. No second backprojection.

Bearing/distance are measured from the current pose (origin (0,0), heading 0):
    bearing_deg = deg(atan2(x, z))   # +x right, +z forward; 0 = straight ahead
    dist        = hypot(x, z)
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline import config, perception

_bearing_dist = perception.bearing_dist   # (x, z) -> (bearing deg, distance)

INITIAL_GROUND_SUPPORT_SCHEMA = (
    "initial-depth-instance-ground-support-grid.v1")
INITIAL_GROUND_SUPPORT_FRAME = "initial_robot_local_xz"


def _support_sha256(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def initial_ground_support_grid(
        points_local: np.ndarray, floor_plane, *,
        cell_size_m: float = config.INITIAL_VISIBLE_SUPPORT_GRID_CELL_M,
        support_band_m=config.GROUND_OBSTACLE_BAND_M) -> dict:
    """Compress one visible instance's ground support into canonical cells."""
    points = np.asarray(points_local, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"visible support points must have shape (N,3), got {points.shape}")
    cell_size = float(cell_size_m)
    if not math.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("visible support cell size must be positive")
    mask = ground_support_mask(points, floor_plane, support_band_m)
    support = points[mask][:, [0, 2]]
    cells = sorted({
        (int(math.floor(float(x) / cell_size)),
         int(math.floor(float(z) / cell_size)))
        for x, z in support
    })
    value = {
        "schema": INITIAL_GROUND_SUPPORT_SCHEMA,
        "frame": INITIAL_GROUND_SUPPORT_FRAME,
        "cell_size_m": cell_size,
        "ground_band_m": [float(value) for value in support_band_m],
        "raw_point_count": int(len(support)),
        "cells_ix_iz": [[int(ix), int(iz)] for ix, iz in cells],
        "cell_count": int(len(cells)),
    }
    return {**value, "sha256": _support_sha256(value)}


def initial_ground_support_errors(value: dict) -> list[str]:
    """Validate and re-hash a persisted initial support grid."""
    expected_keys = {
        "schema", "frame", "cell_size_m", "ground_band_m",
        "raw_point_count", "cells_ix_iz", "cell_count", "sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return ["initial ground support fields are invalid"]
    errors = []
    if value.get("schema") != INITIAL_GROUND_SUPPORT_SCHEMA:
        errors.append("initial ground support schema is invalid")
    if value.get("frame") != INITIAL_GROUND_SUPPORT_FRAME:
        errors.append("initial ground support frame is invalid")
    try:
        cell_size = float(value["cell_size_m"])
        band = [float(item) for item in value["ground_band_m"]]
        raw_cells = value["cells_ix_iz"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return errors + ["initial ground support values are invalid"]
    raw_count = value["raw_point_count"]
    cell_count = value["cell_count"]
    if (not isinstance(raw_count, int) or isinstance(raw_count, bool) or
            not isinstance(cell_count, int) or isinstance(cell_count, bool) or
            not isinstance(raw_cells, list) or
            any(not isinstance(item, list) or len(item) != 2 or
                any(not isinstance(index, int) or isinstance(index, bool)
                    for index in item)
                for item in raw_cells)):
        return errors + ["initial ground support values are invalid"]
    cells = [(item[0], item[1]) for item in raw_cells]
    if (not math.isfinite(cell_size) or
            cell_size != config.INITIAL_VISIBLE_SUPPORT_GRID_CELL_M):
        errors.append("initial ground support cell size is invalid")
    if band != [float(item) for item in config.GROUND_OBSTACLE_BAND_M]:
        errors.append("initial ground support band is invalid")
    if (raw_count < 0 or cell_count < 0 or raw_count < cell_count or
            cells != sorted(set(cells)) or cell_count != len(cells)):
        errors.append("initial ground support cells are invalid")
    unhashed = {key: value[key] for key in expected_keys - {"sha256"}}
    if value.get("sha256") != _support_sha256(unhashed):
        errors.append("initial ground support sha256 is invalid")
    return errors


def ground_support_mask(
    points_local: np.ndarray, floor_plane,
    support_band_m=config.GROUND_OBSTACLE_BAND_M,
) -> np.ndarray:
    """Select finite local points in the configured near-floor support band.
    Measured against the pose's canonical plane, so every sibling of a pose
    agrees on which objects are floor-supported and therefore eligible.
    """
    low, high = (float(value) for value in support_band_m)
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low <= high):
        raise ValueError(f"invalid target support band: {support_band_m!r}")
    points = np.asarray(points_local, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    height = floor_plane.height_above_points(np.where(finite[:, None], points, 0.0))
    return finite & (height >= low) & (height <= high)


def target_ground_support_points(
    points_world: np.ndarray, frame,
    support_band_m=config.GROUND_OBSTACLE_BAND_M,
) -> np.ndarray:
    """Transform full-scene target samples and retain ground support points."""
    points = np.asarray(points_world, dtype=np.float64)
    dx = points[:, 0] - float(frame.position[0])
    dz = points[:, 2] - float(frame.position[2])
    cosine, sine = math.cos(frame.yaw_rad), math.sin(frame.yaw_rad)
    local = np.stack([
        dx * cosine - dz * sine,
        points[:, 1] - float(frame.position[1]),
        -dx * sine - dz * cosine,
    ], axis=1)
    return local[ground_support_mask(local, frame.floor_plane, support_band_m)]


def eligible_target_ids(frame) -> list[int]:
    """Return unique visible objects supported by full and visible geometry."""
    counts = frame.category_inventory
    points = np.asarray(frame.pts, dtype=np.float64)
    labels = np.asarray(frame.pts_sem)
    visible_support = ground_support_mask(points, frame.floor_plane)
    semantic_index = getattr(frame, "semantic_index", None)
    eligible = []
    for obj in frame.objects:
        instance_id = int(obj["instance_id"])
        predicate_category = obj.get(
            "_predicate_category", obj["category"])
        if (obj["is_structural"] or
                not config.is_specific_semantic_category(
                    predicate_category) or
                counts.get(obj["category"], 0) != 1 or
                obj.get("mask_area_px", 0) < config.TARGET_VISIBLE_MIN_PX or
                int(np.count_nonzero(
                    (labels == instance_id) & visible_support)) <
                config.TARGET_GROUND_SUPPORT_MIN_POINTS or
                semantic_index is None or
                not hasattr(semantic_index, "instance_points")):
            continue
        full_points = np.asarray(
            semantic_index.instance_points(instance_id), dtype=np.float64)
        if (len(full_points) and
                len(target_ground_support_points(full_points, frame)) >=
                config.TARGET_GROUND_SUPPORT_MIN_POINTS):
            eligible.append(instance_id)
    return eligible


def initial_visible_entity_inventory(record: dict) -> list[dict]:
    """Build deterministic, outcome-blind choices from the initial frame.

    A category name is sufficient when only one usable instance has that
    category.  Duplicate categories receive consecutive numbered-dot markers;
    no bbox, outline, mask tint, or answer-dependent filtering is introduced.
    """
    values = []
    seen = set()
    for raw in record.get("objects") or []:
        try:
            instance_id = int(raw.get("instance_id"))
            category = str(raw.get("category") or "").strip()
            center = np.asarray(raw.get("centroid_px"), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if (instance_id <= 0 or instance_id in seen or not category or
                category.lower() == "unknown" or center.shape != (2,) or
                not np.all(np.isfinite(center))):
            continue
        seen.add(instance_id)
        values.append((category, instance_id, center))
    values.sort(key=lambda value: (value[0].casefold(), value[1]))
    counts = Counter(category.casefold() for category, _iid, _center in values)
    ordinals = Counter()
    result = []
    for category, instance_id, center in values:
        key = category.casefold()
        duplicate = counts[key] > 1
        ordinals[key] += 1
        number = int(ordinals[key])
        result.append({
            "choice_id": f"entity_{instance_id}",
            "instance_id": instance_id,
            "category": category,
            "name": f"{category} {number}" if duplicate else category,
            "marker": ({
                "kind": "numbered_dot",
                "number": number,
                "center_xy": [float(center[0]), float(center[1])],
            } if duplicate else None),
        })
    return result


def a3_category_evidence(record: Dict, certificate: Dict) -> tuple:
    """Return one perturbation-stable visible contact category and choices."""
    inventory = initial_visible_entity_inventory(record)
    by_id = {value["instance_id"]: value for value in inventory}
    categories = []
    for row in certificate.get("rows") or []:
        consensus = row.get("consensus") or {}
        full_id = consensus.get("full_contact_instance_id")
        depth_id = consensus.get("depth_contact_instance_id")
        if (not isinstance(full_id, int) or isinstance(full_id, bool) or
                not isinstance(depth_id, int) or isinstance(depth_id, bool) or
                full_id <= 0 or full_id != depth_id):
            return None, None, "contact_attribution_disagreement"
        entity = by_id.get(full_id)
        if entity is None:
            return None, None, "contact_instance_not_initially_visible"
        category = str(entity.get("category") or "").strip()
        if not config.is_contact_obstacle_category(category):
            return None, None, "contact_category_ineligible"
        full_attr = (row.get("physical", {}).get("contact") or {}).get(
            "full_geometry_attribution") or {}
        depth_attr = (row.get("depth_physical", {}).get("contact") or {}).get(
            "depth_mask_attribution") or {}
        attributed = {
            str(full_attr.get("category") or "").strip().casefold(),
            str(depth_attr.get("category") or "").strip().casefold(),
            category.casefold(),
        }
        if len(attributed) != 1:
            return None, None, "contact_category_oracle_disagreement"
        categories.append(category)
    if not categories or len({value.casefold() for value in categories}) != 1:
        return None, None, "contact_category_unstable"
    canonical = categories[0]
    unique = {}
    for entity in inventory:
        category = str(entity.get("category") or "").strip()
        if config.is_contact_obstacle_category(category):
            unique.setdefault(category.casefold(), category)
    choices = [
        {"id": value, "text": value.replace("_", " ")}
        for _key, value in sorted(unique.items())
    ]
    if canonical not in {value["id"] for value in choices}:
        return None, None, "contact_category_not_initially_visible"
    if len(choices) < 2:
        return None, None, "insufficient_visible_contact_categories"
    return canonical, choices, None


def _b_target_rejection(raw: dict, *, width: int, height: int) -> str | None:
    category = str(raw.get("category") or "").strip()
    predicate_category = str(
        raw.get("_predicate_category") or category).strip()
    normalized = " ".join(predicate_category.casefold().split())
    specific = raw.get("_predicate_is_specific")
    if not isinstance(specific, bool):
        specific = config.is_specific_semantic_category(predicate_category)
    excluded_material = raw.get("_predicate_excluded_material")
    if not isinstance(excluded_material, bool):
        excluded_material = any(
            token in normalized
            for token in config.B_TARGET_EXCLUDED_MATERIAL_TOKENS)
    if (raw.get("is_structural") is not False or
            not specific or excluded_material):
        return "target_category_ineligible"
    mask_area = raw.get("mask_area_px")
    depth_backed = raw.get("depth_backed_px")
    if (not isinstance(mask_area, int) or isinstance(mask_area, bool) or
            mask_area < 0 or not isinstance(depth_backed, int) or
            isinstance(depth_backed, bool) or depth_backed < 0):
        return "depth_backed_area_missing"
    minimum_pixels = math.ceil(
        config.B_TARGET_MIN_DEPTH_BACKED_AREA_RATIO * width * height)
    if depth_backed < minimum_pixels:
        return "depth_backed_area_too_small"
    raw_bbox = raw.get("bbox_xyxy_px")
    if (not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4 or
            any(not isinstance(value, int) or isinstance(value, bool)
                for value in raw_bbox)):
        return "bbox_missing"
    bbox = np.asarray(raw_bbox, dtype=np.int64)
    if (bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= width or
            bbox[3] >= height or bbox[0] > bbox[2] or bbox[1] > bbox[3]):
        return "bbox_out_of_raster"
    bbox_width = float(bbox[2] - bbox[0] + 1.0)
    bbox_height = float(bbox[3] - bbox[1] + 1.0)
    if (bbox_width < config.B_TARGET_MIN_BBOX_WIDTH_RATIO * width or
            bbox_height < config.B_TARGET_MIN_BBOX_HEIGHT_RATIO * height):
        return "bbox_too_small"
    bbox_area = int(bbox_width * bbox_height)
    if not 0 <= depth_backed <= mask_area <= bbox_area:
        return "depth_backed_mask_bbox_inconsistent"
    try:
        centroid = np.asarray(raw.get("centroid_px"), dtype=np.float64)
    except (TypeError, ValueError):
        return "centroid_invalid"
    if (centroid.shape != (2,) or not np.all(np.isfinite(centroid)) or
            not 0 <= centroid[0] < width or
            not 0 <= centroid[1] < height or
            not bbox[0] <= centroid[0] <= bbox[2] or
            not bbox[1] <= centroid[1] <= bbox[3]):
        return "centroid_invalid"
    try:
        distance = float(raw.get("dist_nearest_m"))
    except (TypeError, ValueError):
        return "visible_distance_missing"
    low, high = config.B_TARGET_VISIBLE_DISTANCE_RANGE_M
    if not math.isfinite(distance) or not low <= distance <= high:
        return "visible_distance_out_of_range"
    return None


def select_b_target(record: dict) -> dict:
    """Select one deterministic B target using only serialized s0 facts."""
    sensor = record.get("sensor") if isinstance(record, dict) else None
    resolution = sensor.get("resolution") if isinstance(sensor, dict) else None
    if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or
            any(not isinstance(value, int) or isinstance(value, bool)
                for value in resolution)):
        return {"eligible": False, "reason": "sensor_resolution_invalid"}
    width, height = resolution
    if width <= 0 or height <= 0:
        return {"eligible": False, "reason": "sensor_resolution_invalid"}
    inventory = {
        value["instance_id"]: value
        for value in initial_visible_entity_inventory(record)
    }
    accepted = []
    rejections = []
    for raw in record.get("objects") or []:
        reason = _b_target_rejection(raw, width=width, height=height)
        if reason is not None:
            rejections.append(reason)
            continue
        try:
            instance_id = int(raw["instance_id"])
            entity = inventory[instance_id]
            bbox = [float(value) for value in raw["bbox_xyxy_px"]]
            depth_backed = int(raw["depth_backed_px"])
            distance = float(raw["dist_nearest_m"])
        except (KeyError, TypeError, ValueError):
            rejections.append("initial_visible_identity_invalid")
            continue
        bbox_area = (bbox[2] - bbox[0] + 1.0) * (bbox[3] - bbox[1] + 1.0)
        quality = (
            -depth_backed, -bbox_area, abs(distance - 3.0),
            entity["category"].casefold(), instance_id,
        )
        accepted.append((quality, raw, entity, bbox, distance, depth_backed))
    if not accepted:
        return {
            "eligible": False,
            "reason": rejections[0] if rejections else "no_visible_target",
        }
    _quality, raw, entity, bbox, distance, depth_backed = min(accepted)
    return {
        "eligible": True,
        "selection_protocol": config.B_TARGET_SELECTION_PROTOCOL,
        "instance_id": entity["instance_id"],
        "category": entity["category"],
        "name": entity["name"],
        "marker": entity["marker"],
        "depth_backed_px": depth_backed,
        "bbox_xyxy_px": bbox,
        "visible_distance_m": distance,
    }


def extract_objects(
    pts_ground: np.ndarray,
    uv: np.ndarray,
    sem: np.ndarray,
    id_to_cat: Dict[int, str],
    *,
    predicate_categories: Optional[Dict[int, str]] = None,
    min_area_px: int = config.OBJ_MIN_AREA_PX,
    min_valid: int = config.OBJ_MIN_VALID_DEPTH,
    max_points: int = config.OBJ_MAX_POINTS,
) -> List[dict]:
    """Extract one record per visible instance, from per-point assignments.
    Parameters
    ----------
    pts_ground, uv, sem : valid-pixel arrays from build_frame
        pts_ground[i] is the ground xyz of pixel uv[i]; sem[i] its instance id
        (assigned by nearest labelled semantic surface; 0 = unlabelled).
    id_to_cat : instance id -> raw source category string.
    ``mask_area_px`` is the count of assigned (valid-depth) pixels for the
    instance; ``centroid_px`` the mean source pixel of those points.
    """
    xz = np.asarray(pts_ground, dtype=np.float64)[:, [0, 2]]
    uv = np.asarray(uv)
    out: List[dict] = []
    for iid in np.unique(sem):
        iid = int(iid)
        if iid == 0:
            continue
        pmask = sem == iid
        n = int(pmask.sum())
        if n < min_area_px or n < min_valid:
            continue
        obj_uv = uv[pmask]
        cpx = (float(obj_uv[:, 0].mean()), float(obj_uv[:, 1].mean()))
        # Subsample first, then derive every statistic from the same points so
        # stored nearest/centroid values match later relation recomputation.
        pts = xz[pmask]
        if pts.shape[0] > max_points:
            sel = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
            pts = pts[sel]
        cx, cz = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        d = np.hypot(pts[:, 0], pts[:, 1])
        ni = int(np.argmin(d))
        nx, nz = float(pts[ni, 0]), float(pts[ni, 1])
        bearing, dist_c = _bearing_dist(cx, cz)
        dist_n = float(d[ni])
        category = id_to_cat.get(iid, "unknown") or "unknown"
        predicate_category = (
            predicate_categories.get(iid, category)
            if predicate_categories is not None else category)
        predicate_category = predicate_category or "unknown"
        out.append({
            "instance_id": iid,
            "category": category,
            "is_structural": config.is_structural(predicate_category),
            "mask_area_px": n,
            "depth_backed_px": n,
            "bbox_xyxy_px": [
                int(obj_uv[:, 0].min()), int(obj_uv[:, 1].min()),
                int(obj_uv[:, 0].max()), int(obj_uv[:, 1].max()),
            ],
            "centroid_px": cpx,
            "ground_xy_centroid": [cx, cz],
            "ground_xy_nearest": [nx, nz],
            "bearing_deg": bearing,
            "dist_centroid_m": dist_c,
            "dist_nearest_m": dist_n,
            "n_points_raw": n,
            "_predicate_category": predicate_category,
            "_points_xz": pts,  # transient; stripped before serialization
        })
    out.sort(key=lambda o: o["instance_id"])
    return out


def attribute_contact(
    pts_ground: np.ndarray,
    sem: np.ndarray,
    center_xz: Tuple[float, float],
    radius: float,
    id_to_cat: Dict[int, str],
    *,
    predicate_categories: Optional[Dict[int, str]] = None,
    margin: float = config.ATTR_MARGIN_M,
    height_band: Optional[Tuple[float, float]] = None,
    floor_plane=None,
) -> dict:
    """Majority-vote the semantic id of raw points near a contact disk.
    Queries points within (radius + margin) of center; relaxes once (x2 margin)
    if empty. Ties broken by lowest instance id. Unlabelled (id 0) points are
    ignored. Returns unattributed=True if no labelled point is found.
    When ``height_band`` is given, only points whose height above the pose's
    canonical ``floor_plane`` lies in the fixed ground-support band, so floor
    points and overhead geometry cannot be attributed as a ground-disc contact
    object. ``floor_plane`` is then required: defaulting it to a level floor at
    the origin would attribute against a surface nobody measured.
    """
    pts = np.asarray(pts_ground, dtype=np.float64)
    xz = pts[:, [0, 2]]
    sem = np.asarray(sem)
    cx, cz = center_xz
    d = np.hypot(xz[:, 0] - cx, xz[:, 1] - cz)
    labelled = sem != 0
    if height_band is not None:
        if floor_plane is None:
            raise ValueError(
                "a height band needs the pose's canonical floor plane")
        above = floor_plane.height_above_points(pts)
        labelled = labelled & (above >= height_band[0]) & (above <= height_band[1])
        obstacle_ids = {
            int(instance_id) for instance_id in np.unique(sem)
            if int(instance_id) != 0 and
            config.is_contact_obstacle_category(
                (predicate_categories or {}).get(
                    int(instance_id), id_to_cat.get(
                        int(instance_id), "unknown")))
        }
        labelled = labelled & np.isin(sem, list(obstacle_ids))
    for r in (radius + margin, radius + 2 * margin):
        within = (d <= r) & labelled
        ids = sem[within]
        if ids.size:
            counts = Counter(int(i) for i in ids)
            total = sum(counts.values())
            # winner = highest count, tie -> lowest id
            winner = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            cat = id_to_cat.get(winner, "unknown") or "unknown"
            return {
                "instance_id": winner,
                "category": cat,
                "vote_fraction": counts[winner] / total,
                "votes": {str(k): v for k, v in sorted(counts.items())},
                "unattributed": False,
            }
    return {"instance_id": None, "category": None,
            "vote_fraction": 0.0, "votes": {}, "unattributed": True}
