"""Visible-object extraction and contact attribution (pure numpy).

Both operate on the SINGLE ground point cloud produced once by build_frame,
sliced by semantic instance id. No second backprojection.

Bearing/distance are measured from the current pose (origin (0,0), heading 0):
    bearing_deg = deg(atan2(x, z))   # +x right, +z forward; 0 = straight ahead
    dist        = hypot(x, z)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline import config, perception

_bearing_dist = perception.bearing_dist   # (x, z) -> (bearing deg, distance)


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


