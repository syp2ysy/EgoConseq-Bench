"""Visible-object extraction and contact attribution (pure numpy).

Both operate on the SINGLE ground point cloud produced once by build_frame,
sliced by semantic instance id. No second backprojection.

Bearing/distance are measured from the current pose (origin (0,0), heading 0):
    bearing_deg = deg(atan2(x, z))   # +x right, +z forward; 0 = straight ahead
    dist        = hypot(x, z)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline import config


def _bearing_dist(x: float, z: float) -> Tuple[float, float]:
    return math.degrees(math.atan2(x, z)), math.hypot(x, z)


def extract_objects(
    pts_ground: np.ndarray,
    uv: np.ndarray,
    sem: np.ndarray,
    id_to_cat: Dict[int, str],
    *,
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
    id_to_cat : instance id -> raw HM3D category string.

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

        # Subsample FIRST, then derive all stats from the SAME points so the
        # stored nearest/centroid exactly match what object_relations recomputes
        # from _points_xz (otherwise pure-turn deltas are non-zero -> V1/V2).
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
        out.append({
            "instance_id": iid,
            "category": category,
            "is_structural": config.is_structural(category),
            "mask_area_px": n,
            "centroid_px": cpx,
            "ground_xy_centroid": [cx, cz],
            "ground_xy_nearest": [nx, nz],
            "bearing_deg": bearing,
            "dist_centroid_m": dist_c,
            "dist_nearest_m": dist_n,
            "n_points_raw": n,
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
    margin: float = config.ATTR_MARGIN_M,
) -> dict:
    """Majority-vote the semantic id of raw points near a contact disk.

    Queries points within (radius + margin) of center; relaxes once (x2 margin)
    if empty. Ties broken by lowest instance id. Unlabelled (id 0) points are
    ignored. Returns unattributed=True if no labelled point is found.
    """
    xz = np.asarray(pts_ground, dtype=np.float64)[:, [0, 2]]
    sem = np.asarray(sem)
    cx, cz = center_xz
    d = np.hypot(xz[:, 0] - cx, xz[:, 1] - cz)

    for r in (radius + margin, radius + 2 * margin):
        within = (d <= r) & (sem != 0)
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
