"""Visible-object extraction and contact attribution (pure numpy).

Both operate on the SINGLE ground point cloud produced once by build_frame,
sliced by semantic instance id. No second backprojection.

Bearing/distance are measured from the current pose (origin (0,0), heading 0):
    bearing_deg = deg(atan2(x, z))   # +x right, +z forward; 0 = straight ahead
    dist        = hypot(x, z)
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline import config, perception

_bearing_dist = perception.bearing_dist   # (x, z) -> (bearing deg, distance)


def initial_visible_entity_inventory(record: dict) -> list[dict]:
    """Return deterministic visible instance/category identities."""
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
        values.append({"instance_id": instance_id, "category": category})
    return sorted(
        values,
        key=lambda value: (value["category"].casefold(), value["instance_id"]),
    )


def a3_category_evidence(
        record: Dict, certificate: Dict, *,
        inventory: List[dict] | None = None) -> tuple:
    """Return one perturbation-stable visible contact category and choices."""
    inventory = (initial_visible_entity_inventory(record)
                 if inventory is None else list(inventory))
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



def extract_objects(
    pts_ground: np.ndarray,
    uv: np.ndarray,
    sem: np.ndarray,
    id_to_cat: Dict[int, str],
    *,
    predicate_categories: Optional[Dict[int, str]] = None,
    resolution: Optional[Tuple[int, int]] = None,
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
    points_ground = np.asarray(pts_ground, dtype=np.float64)
    xz = points_ground[:, [0, 2]]
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
        structural = config.is_structural(predicate_category)
        value = {
            "instance_id": iid,
            "category": category,
            "source_category": predicate_category,
            "is_structural": structural,
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
        }
        out.append(value)
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
