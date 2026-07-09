"""judge(frame, body, actions, nav) -> Consequence (dict of all GT elements).

Pure geometry over a prebuilt Frame. `nav` is duck-typed and optional:
    nav.d_safe(actions, radius) -> arc (m) to first non-navigable, or d_max
    nav.geodesic(a_local_xz, b_local_xz) -> float | None
When nav is None, nav_check and all geodesic fields are null. No Habitat import.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from pipeline import config, actions as A, objects as OBJ, perception


FOV_HALF = config.FOV_HALF_DEG


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _in_frame(uv, shape) -> bool:
    if uv is None:
        return False
    H, W = shape
    u, v = uv
    return 0 <= u < W and 0 <= v < H


def _in_cone(x: float, z: float) -> bool:
    return z > 1e-9 and abs(math.degrees(math.atan2(x, z))) <= FOV_HALF


def _bearing_dist(x: float, z: float) -> Tuple[float, float]:
    return math.degrees(math.atan2(x, z)), math.hypot(x, z)


# --------------------------------------------------------------------------
# collision
# --------------------------------------------------------------------------

def march_collision(vf, path, radius, min_support=config.MIN_SUPPORT_VOXELS):
    """First contact along path. Returns (collided, arc, (x,z)) or (False, None, None)."""
    for i, (x, z, _h, arc) in enumerate(path):
        if i == 0:
            continue
        if vf.support_count((x, z), radius) >= min_support:
            return True, arc, (x, z)
    return False, None, None


# --------------------------------------------------------------------------
# view exit
# --------------------------------------------------------------------------

def compute_view_exit(acts, frame, pose_full) -> dict:
    ex, ez, _heading = pose_full
    step = config.MARCH_STEP_M

    # path first-exit arc
    exit_arc = None
    for i, (x, z, _h, arc) in enumerate(A.sample_path(acts, step)):
        if i == 0:
            continue
        if not _in_cone(x, z):
            exit_arc = arc
            break

    end_bearing, end_dist = _bearing_dist(ex, ez)
    if end_dist < 1e-9:                       # in-place end: convention
        end_in_fov, end_bearing = True, 0.0
    else:
        end_in_fov = _in_cone(ex, ez)

    pt = (ex, frame.floor_y + 0.5, ez)
    uv = perception.project_ground(pt, frame.K)
    in_frame = _in_frame(uv, frame.depth.shape)
    end_visible = False
    if in_frame and ez > 0:
        u, v = int(round(uv[0])), int(round(uv[1]))
        u = min(u, frame.depth.shape[1] - 1)
        v = min(v, frame.depth.shape[0] - 1)
        end_visible = float(frame.depth[v, u]) + config.END_VISIBLE_DEPTH_TOL_M >= ez

    return {
        "path_exit_arc_m": exit_arc,
        "end_in_fov": bool(end_in_fov),
        "end_bearing_deg": end_bearing,
        "end_dist_m": end_dist,
        "end_pixel": list(uv) if uv is not None else None,
        "end_pixel_in_frame": bool(in_frame),
        "end_visible": bool(end_visible),
        "end_heading_offset_deg": A.wrap_deg(A.net_turn_deg(acts)),
    }


# --------------------------------------------------------------------------
# object relations
# --------------------------------------------------------------------------

def _relation_at(obj, endpos, heading_deg, nav, geodesic) -> dict:
    px, pz = endpos
    h = math.radians(heading_deg)
    cos_h, sin_h = math.cos(h), math.sin(h)

    def transform(tx, tz):
        dx, dz = tx - px, tz - pz
        xp = dx * cos_h - dz * sin_h
        zp = dx * sin_h + dz * cos_h
        return xp, zp

    cx, cz = obj["ground_xy_centroid"]
    xp, zp = transform(cx, cz)
    bearing = math.degrees(math.atan2(xp, zp))
    in_fov = zp > 1e-9 and abs(bearing) <= FOV_HALF
    dist_c = math.hypot(cx - px, cz - pz)

    pts = obj["_points_xz"]
    dn = np.hypot(pts[:, 0] - px, pts[:, 1] - pz)
    dist_n = float(dn.min())

    do_geo = nav is not None and geodesic
    geo = nav.geodesic((0.0, 0.0), (cx, cz)) if do_geo else None
    geo_end = nav.geodesic(endpos, (cx, cz)) if do_geo else None

    return {"bearing_deg": bearing, "dist_centroid_m": dist_c,
            "dist_nearest_m": dist_n, "dist_geodesic_m": geo_end,
            "in_fov": bool(in_fov), "_geo_before": geo}


def object_relations(frame, pose_exec, pose_full, nav, geodesic=True) -> List[dict]:
    ex_pos = (pose_exec[0], pose_exec[1])
    fu_pos = (pose_full[0], pose_full[1])
    rels = []
    for obj in frame.objects:
        exec_r = _relation_at(obj, ex_pos, pose_exec[2], nav, geodesic)
        full_r = _relation_at(obj, fu_pos, pose_full[2], nav, geodesic)
        geo_before = full_r.pop("_geo_before")
        exec_r.pop("_geo_before")
        d_geo = None
        if geo_before is not None and full_r["dist_geodesic_m"] is not None:
            d_geo = full_r["dist_geodesic_m"] - geo_before
        rels.append({
            "instance_id": obj["instance_id"], "category": obj["category"],
            "exec": exec_r, "full": full_r,
            "delta_full": {
                "d_centroid_m": full_r["dist_centroid_m"] - obj["dist_centroid_m"],
                "d_nearest_m": full_r["dist_nearest_m"] - obj["dist_nearest_m"],
                "d_geodesic_m": d_geo,
            },
        })
    return rels


# --------------------------------------------------------------------------
# visibility of the swept corridor
# --------------------------------------------------------------------------

def compute_visibility(acts, frame) -> dict:
    path = A.sample_path(acts, config.MARCH_STEP_M)
    n, seen = 0, 0
    for i, (x, z, _h, _arc) in enumerate(path):
        if i == 0:
            continue
        n += 1
        uv = perception.project_ground((x, frame.floor_y + 0.5, z), frame.K)
        if _in_frame(uv, frame.depth.shape):
            u, v = int(round(uv[0])), int(round(uv[1]))
            u = min(u, frame.depth.shape[1] - 1); v = min(v, frame.depth.shape[0] - 1)
            if np.isfinite(frame.depth[v, u]) and frame.depth[v, u] > 0:
                seen += 1
    vdr = frame.quality.get("valid_depth_ratio", float(
        (np.isfinite(frame.depth) & (frame.depth > 0)).mean()))
    ratio = (seen / n) if n else 1.0
    return {"visible_sweep_ratio": ratio,
            "valid_depth_ratio": vdr,
            "depth_hole_ratio": 1.0 - vdr,
            "occlusion_free_ratio": ratio}


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------

def judge(frame, body, acts, nav=None, geodesic=True) -> dict:
    radius = body.radius_m
    path = A.sample_path(acts, config.MARCH_STEP_M)

    collided, arc, contact_xz = march_collision(frame.vf, path, radius)
    pose_full = A.pose_after(acts)
    pose_exec = A.pose_at_arc(acts, arc) if collided else pose_full
    contact_ai, contact_local = A.contact_action_index(acts, arc) if collided else (None, None)

    # contact block
    if collided:
        attr = OBJ.attribute_contact(frame.pts, frame.pts_sem, contact_xz, radius, frame.id_to_cat)
        cx, cz = contact_xz
        pt3d = (cx, frame.floor_y + 0.5, cz)
        uv = perception.project_ground(pt3d, frame.K)
        contact = {
            "xy": [cx, cz], "point_3d": list(pt3d),
            "pixel": list(uv) if uv is not None else None,
            "pixel_in_frame": _in_frame(uv, frame.depth.shape),
            **attr,
        }
    else:
        contact = None

    view_exit = compute_view_exit(acts, frame, pose_full)
    rels = object_relations(frame, pose_exec, pose_full, nav, geodesic)
    vis = compute_visibility(acts, frame)

    # nav cross-check (mesh)
    nav_check = None
    if nav is not None:
        d_nav = nav.d_safe(acts, radius)
        d_depth = arc if collided else config.D_MAX_M
        nav_check = _nav_verdict(frame, acts, d_nav, d_depth, radius)

    return {
        "body": body.to_dict(),
        "actions": A.actions_to_dicts(acts),
        "total_forward_m": A.total_forward_m(acts),
        "net_turn_deg": A.net_turn_deg(acts),
        "pose_end_full": {"x": pose_full[0], "z": pose_full[1], "heading_deg": pose_full[2]},
        "collided": bool(collided),
        "first_contact_arc_m": arc,
        "contact_action_index": contact_ai,
        "contact_action_local_arc_m": contact_local,
        "pose_end_exec": {"x": pose_exec[0], "z": pose_exec[1], "heading_deg": pose_exec[2]},
        "contact": contact,
        "view_exit": view_exit,
        "nav_check": nav_check,
        "visibility": vis,
        "object_relations": rels,
    }


def _forward_cover(frame) -> float:
    """Valid-depth fraction in the central-forward image window (where a wall /
    furniture obstacle in the path would appear). Low => depth hole => can't be
    sure depth didn't miss a real obstacle. Artifact-free (image-space, no ground
    projection of close/low points)."""
    H, W = frame.depth.shape
    r0, r1 = int(0.25 * H), int(0.90 * H)     # eye-level down to lower band, skip ceiling
    c0, c1 = int(0.30 * W), int(0.70 * W)     # central columns
    win = frame.depth[r0:r1, c0:c1]
    if win.size == 0:
        return 1.0
    return float((np.isfinite(win) & (win > 0)).mean())


def _nav_verdict(frame, acts, d_nav, d_depth, radius, tol=config.NAV_AGREE_TOL_M) -> dict:
    """Depth-vs-navmesh taxonomy for QA (GT collision stays depth-based).

    Uses artifact-free signals: the frame's own obstacle voxel field (same frame
    as march_collision) and forward-cone valid-depth coverage -- NOT the old
    pixel-projection of a floor point that spuriously fell below the frame.

    diff = d_depth - d_nav:
      |diff| <= tol                    -> keep_agree       (green) navmesh ~= depth
      diff > 0  & forward view covered -> keep_depth        (green) navmesh conservative, trust depth
      diff > 0  & forward depth hole   -> review_depth_hole (yellow) possible miss -> human check
      diff < 0  & solid depth obstacle -> keep_visible      (green) depth saw a real obstacle
      diff < 0  & weak/isolated support-> discard_noise     (red) depth noise -> drop
    """
    diff = d_depth - d_nav
    fwd_cover = _forward_cover(frame)
    support = None

    if A.total_forward_m(acts) <= 1e-9:             # pure rotation: no translation, trivially safe
        verdict, keep = "keep_agree", True
    elif abs(diff) <= tol:
        verdict, keep = "keep_agree", True
    elif diff > 0:                                  # navmesh stops earlier than depth
        if fwd_cover >= config.FWD_COVER_MIN:
            verdict, keep = "keep_depth", True       # depth observed clear -> trust depth
        else:
            verdict, keep = "review_depth_hole", False
    else:                                           # depth stops earlier (implies collided)
        hit = A.pose_at_arc(acts, d_depth)
        support = int(frame.vf.support_count((hit[0], hit[1]), radius))
        if support >= config.NAV_STRONG_SUPPORT:
            verdict, keep = "keep_visible", True
        else:
            verdict, keep = "discard_noise", False

    return {"d_nav_m": float(d_nav), "d_depth_m": float(d_depth), "diff_m": float(diff),
            "fwd_cover": float(fwd_cover), "support_at_hit": support,
            "verdict": verdict, "keep": bool(keep)}
