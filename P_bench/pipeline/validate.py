"""Invariant checks V1-V8 over serialized records (pure functions).

validate_record(rec) -> list[str] of violation messages (empty = clean).
Recomputes poses/bearings from the stored actions and compares to stored
values; checks physical inequalities (triangle bound, FOV, schema ranges).
"""

from __future__ import annotations

import math
from typing import List

from pipeline import actions as A, config

FOV = config.FOV_HALF_DEG
_ANG = 1e-4      # recompute tolerance (deg)
_POS = 1e-4      # recompute tolerance (m)
_INEQ = 1e-3     # inequality slack (m)


def _bd(x, z):
    return math.degrees(math.atan2(x, z)), math.hypot(x, z)


def validate_record(rec: dict) -> List[str]:
    v: List[str] = []
    fid = rec.get("frame_id", "?")
    objs = {o["instance_id"]: o for o in rec.get("objects", [])}

    # V5: frame objects self-consistent
    for o in rec.get("objects", []):
        cx, cz = o["ground_xy_centroid"]
        b, d = _bd(cx, cz)
        if abs(A.wrap_deg(b - o["bearing_deg"])) > _ANG:
            v.append(f"[{fid}] V5 obj{o['instance_id']} bearing recompute {b:.4f}!={o['bearing_deg']:.4f}")
        if abs(d - o["dist_centroid_m"]) > _POS:
            v.append(f"[{fid}] V5 obj{o['instance_id']} dist recompute mismatch")

    for oc in rec.get("outcomes", []):
        oid = oc.get("outcome_id", "?")
        acts = A.parse_actions(oc["actions"])
        tf = oc["total_forward_m"]
        nt = oc["net_turn_deg"]

        # V4 pose_end_full
        ex, ez, eh = A.pose_after(acts)
        pf = oc["pose_end_full"]
        if abs(ex - pf["x"]) > _POS or abs(ez - pf["z"]) > _POS or abs(A.wrap_deg(eh - pf["heading_deg"])) > _ANG:
            v.append(f"[{fid}:{oid}] V4 pose_end_full recompute mismatch")

        # V3 truncation consistency
        collided = oc["collided"]; arc = oc["first_contact_arc_m"]; contact = oc["contact"]
        if collided != (arc is not None) or collided != (contact is not None):
            v.append(f"[{fid}:{oid}] V3 collided/arc/contact inconsistency")
        pe = oc["pose_end_exec"]
        if collided and arc is not None:
            tx, tz, th = A.pose_at_arc(acts, arc)
            if abs(tx - pe["x"]) > _POS or abs(tz - pe["z"]) > _POS or abs(A.wrap_deg(th - pe["heading_deg"])) > _ANG:
                v.append(f"[{fid}:{oid}] V3 pose_end_exec != pose_at_arc(arc)")
        else:
            if pe != pf:
                v.append(f"[{fid}:{oid}] V3 exec!=full when not collided")

        # V9 contact_action_index consistency (new-schema outcomes only)
        if "contact_action_index" in oc:
            cai = oc["contact_action_index"]
            cla = oc.get("contact_action_local_arc_m")
            if collided and arc is not None:
                ci, la = A.contact_action_index(acts, arc)
                if cai != ci:
                    v.append(f"[{fid}:{oid}] V9 contact_action_index {cai}!={ci}")
                if ci is not None and not isinstance(acts[ci], A.Forward):
                    v.append(f"[{fid}:{oid}] V9 contact action is not a forward")
                if cla is not None and (cla < -_POS or (ci is not None and cla > acts[ci].m + _INEQ)):
                    v.append(f"[{fid}:{oid}] V9 local arc out of leg range")
            elif cai is not None:
                v.append(f"[{fid}:{oid}] V9 non-collided outcome has contact_action_index")

        # V2 pure turn
        if abs(tf) < 1e-12:
            if collided:
                v.append(f"[{fid}:{oid}] V2 pure-turn collided")
            for r in oc["object_relations"]:
                if abs(r["delta_full"]["d_centroid_m"]) > _INEQ or abs(r["delta_full"]["d_nearest_m"]) > _INEQ:
                    v.append(f"[{fid}:{oid}] V2 pure-turn distance changed")
                before = objs.get(r["instance_id"])
                if before is not None:
                    expect = A.wrap_deg(before["bearing_deg"] - nt)
                    if abs(A.wrap_deg(expect - r["full"]["bearing_deg"])) > 1e-2:
                        v.append(f"[{fid}:{oid}] V2 bearing_after != wrap(before - net_turn)")

        # V1 triangle inequality + V6 in_fov
        for r in oc["object_relations"]:
            before = objs.get(r["instance_id"])
            if before is not None:
                for key_r, key_o in (("dist_centroid_m", "dist_centroid_m"),
                                     ("dist_nearest_m", "dist_nearest_m")):
                    da = abs(r["full"][key_r] - before[key_o])
                    if da > tf + _INEQ:
                        v.append(f"[{fid}:{oid}] V1 |d_after-d_before|={da:.3f} > tf+eps for obj{r['instance_id']}")
                gb, ga = before.get("dist_geodesic_m"), r["full"]["dist_geodesic_m"]
                # geodesic lower bound: geo >= euclid - eps
                if ga is not None and ga + _INEQ < r["full"]["dist_centroid_m"] - 0.5:
                    v.append(f"[{fid}:{oid}] V1 geodesic < euclid for obj{r['instance_id']}")
            for phase in ("exec", "full"):
                p = r[phase]
                if p["in_fov"] and abs(p["bearing_deg"]) > FOV + _ANG:
                    v.append(f"[{fid}:{oid}] V6 in_fov but |bearing|>FOV ({phase} obj{r['instance_id']})")

        # V7 schema
        for r in oc["object_relations"]:
            for phase in ("exec", "full"):
                p = r[phase]
                if not (-180.0 - _ANG <= p["bearing_deg"] <= 180.0 + _ANG):
                    v.append(f"[{fid}:{oid}] V7 bearing out of range")
                if p["dist_centroid_m"] < -_POS or p["dist_nearest_m"] < -_POS:
                    v.append(f"[{fid}:{oid}] V7 negative distance")
        nc = oc.get("nav_check")
        if nc is not None and nc["d_nav_m"] > tf + config.D_MAX_M + _INEQ:
            v.append(f"[{fid}:{oid}] V7 d_nav absurdly large")
        for k in ("visible_sweep_ratio", "valid_depth_ratio", "depth_hole_ratio", "occlusion_free_ratio"):
            val = oc["visibility"][k]
            if not (-_POS <= val <= 1.0 + _POS):
                v.append(f"[{fid}:{oid}] V7 ratio {k}={val} out of [0,1]")
        if contact is not None:
            if contact["unattributed"] and contact["instance_id"] is not None:
                v.append(f"[{fid}:{oid}] V7 unattributed but has instance_id")

        # V8 view_exit
        ve = oc["view_exit"]
        rb, rd = _bd(pf["x"], pf["z"])
        if rd < 1e-9:
            if not ve["end_in_fov"]:
                v.append(f"[{fid}:{oid}] V8 in-place end must be in_fov")
        else:
            expect_fov = (pf["z"] > 1e-9) and abs(rb) <= FOV + _ANG
            if bool(ve["end_in_fov"]) != bool(expect_fov):
                v.append(f"[{fid}:{oid}] V8 end_in_fov inconsistent with bearing/dist")
            if abs(A.wrap_deg(rb - ve["end_bearing_deg"])) > _ANG or abs(rd - ve["end_dist_m"]) > _POS:
                v.append(f"[{fid}:{oid}] V8 end_bearing/dist recompute mismatch")
        if ve["path_exit_arc_m"] is not None and ve["path_exit_arc_m"] > tf + _INEQ:
            v.append(f"[{fid}:{oid}] V8 path_exit_arc > total_forward")
        if abs(A.wrap_deg(A.wrap_deg(nt) - ve["end_heading_offset_deg"])) > _ANG:
            v.append(f"[{fid}:{oid}] V8 end_heading_offset != wrap(net_turn)")
        if ve["end_visible"] and not ve["end_pixel_in_frame"]:
            v.append(f"[{fid}:{oid}] V8 end_visible but not in frame")

    return v


def validate_file(path: str):
    from pipeline import record as R
    total = 0
    violations: List[str] = []
    for rec in R.read_records(path):
        total += 1
        violations.extend(validate_record(rec))
    return total, violations
