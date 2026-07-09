"""Debug visualisations: object overlay (RGB) + top-down evidence map.

Pure matplotlib/numpy over a Frame + one judged outcome. No Habitat.
"""

from __future__ import annotations

import math

import numpy as np

from pipeline import config, perception
from pipeline import actions as A


# --------------------------------------------------------------------------
# reasoning trace (pure; step-by-step derivation of one outcome for review UI)
# --------------------------------------------------------------------------

_CIRCLED = "①②③④⑤⑥⑦⑧⑨"
_VERDICT_WHY = {
    "keep_agree": "navmesh ≈ depth",
    "keep_depth": "navmesh conservative → trust depth",
    "keep_visible": "depth saw a real obstacle navmesh missed",
    "review_depth_hole": "forward depth hole → possible miss → human check",
    "discard_noise": "depth noise / isolated support",
}


def _step_no(n: int) -> str:
    return _CIRCLED[n - 1] if 1 <= n <= len(_CIRCLED) else f"{n}."


def _xz(x, z) -> str:
    return f"({x:+.2f}, {z:+.2f})"


def reasoning_trace(outcome: dict, objects=None) -> list:
    """Human-readable step-by-step geometric derivation of one outcome.

    Pure: reads only the serialized outcome (+ optional frame `objects` for the
    'before' target distance). Mirrors the geometry judge() computed.
    """
    id2obj = {o["instance_id"]: o for o in (objects or [])}
    r = outcome["body"]["radius_m"]
    lines = [f"Body: cylinder r={r:.2f} m. Camera 1.5 m, HFOV 79° (view cone ±39.5°). "
             f"Start at footprint origin, heading 0° (facing forward, +z)."]

    heading = 0.0
    for n, a in enumerate(outcome["actions"], 1):
        if a["type"] == "turn":
            d = a["deg"]
            heading += d
            side = "left" if d < 0 else "right"
            lines.append(f"{_step_no(n)} turn {side} {abs(d):g}° → heading {heading:+.0f}°.")
        else:
            lines.append(f"{_step_no(n)} forward {a['m']:g} m along heading {heading:+.0f}° "
                         f"(body sweeps the corridor).")

    if outcome["collided"]:
        arc = outcome["first_contact_arc_m"]
        c = outcome.get("contact") or {}
        hit = (f'class "{c["category"]}" (vote {c.get("vote_fraction", 0) * 100:.0f}%)'
               if c and not c.get("unattributed") else "an unattributed obstacle")
        at = f" at {_xz(*c['xy'])}" if c.get("xy") else ""
        ci = outcome.get("contact_action_index")
        step = ""
        if ci is not None and ci < len(outcome["actions"]):
            a = outcome["actions"][ci]
            desc = f"forward {a['m']:g} m" if a["type"] == "forward" else f"turn {a['deg']:g}°"
            step = f" during action {_step_no(ci + 1)} ({desc})"
            la = outcome.get("contact_action_local_arc_m")
            if la is not None:
                step += f", {la:.2f} m into it"
        lines.append(f"→ COLLISION{step} (cumulative arc {arc:.2f} m): "
                     f"footprint (r={r:.2f}) overlaps {hit}{at}.")
    else:
        lines.append("→ no contact along the swept path; the body reaches the end freely.")

    pe, pf = outcome["pose_end_exec"], outcome["pose_end_full"]
    tail = (f" Counterfactual full-execution end: {_xz(pf['x'], pf['z'])}."
            if outcome["collided"] else "")
    lines.append(f"End (realised): {_xz(pe['x'], pe['z'])} heading {pe['heading_deg']:+.0f}°.{tail}")

    # target: the hit object, else nearest non-structural relation
    rels = outcome.get("object_relations", [])
    # distances to every (non-structural) object after the move, nearest first
    which = "exec" if outcome["collided"] else "full"
    c = outcome.get("contact")
    cid = c["instance_id"] if (c and not c.get("unattributed")) else None
    tgts = [x for x in rels if x.get(which) and not config.is_structural(x["category"])]
    tgts.sort(key=lambda x: x[which]["dist_centroid_m"])
    if tgts:
        lines.append("Distances to objects after move (centroid, nearest→farthest):")
        for x in tgts:
            after = x[which]
            obj = id2obj.get(x["instance_id"])
            before = obj["dist_centroid_m"] if obj else after["dist_centroid_m"]
            dc = after["dist_centroid_m"] - before
            trend = "closer" if dc < -1e-3 else ("farther" if dc > 1e-3 else "same")
            mark = "  ← contact" if x["instance_id"] == cid else ""
            lines.append(f"· {x['category']}: {after['dist_centroid_m']:.2f} m "
                         f"(Δ{dc:+.2f} {trend}; nearest {after['dist_nearest_m']:.2f} m; "
                         f"{'in view' if after['in_fov'] else 'out of view'}){mark}")

    ve = outcome["view_exit"]
    vtxt = ("path stays inside the view cone" if ve["path_exit_arc_m"] is None
            else f"path leaves the cone at arc {ve['path_exit_arc_m']:.2f} m")
    lines.append(f"View: {vtxt}; end pose in FOV: {'yes' if ve['end_in_fov'] else 'no'}.")

    nv = outcome.get("nav_check")
    if nv:
        why = _VERDICT_WHY.get(nv["verdict"], "")
        lines.append(f"Cross-check (nav vs depth): {nv['verdict']} — {why} "
                     f"(d_nav {nv['d_nav_m']:.2f} vs d_depth {nv['d_depth_m']:.2f} m).")
    return lines


def _color(iid):
    rng = np.random.default_rng(int(iid) * 9973 + 1)
    return rng.integers(60, 255, size=3)


def object_overlay(frame) -> np.ndarray:
    """RGB with each visible instance tinted by its id; returns (H,W,3) uint8."""
    H, W = frame.depth.shape
    inst = np.zeros((H, W), np.int64)
    inst[frame.pts_uv[:, 1], frame.pts_uv[:, 0]] = frame.pts_sem
    out = frame.rgb.copy()
    for o in frame.objects:
        m = inst == o["instance_id"]
        out[m] = (0.45 * out[m] + 0.55 * _color(o["instance_id"])).astype(np.uint8)
    return out


def save_evidence(frame, outcome, path, target_instance=None):
    """RGB overlay + top-down evidence for one outcome -> PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    acts = A.parse_actions(outcome["actions"])
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))

    # -- left: object overlay + contact pixel --
    ax[0].imshow(object_overlay(frame)); ax[0].axis("off")
    ax[0].set_title(f"objects (cov={np.mean(frame.pts_sem>0):.2f})")
    c = outcome["contact"]
    if c and c["pixel"] and c["pixel_in_frame"]:
        ax[0].plot(c["pixel"][0], c["pixel"][1], "o", ms=16, mfc="none", mec="red", mew=2)
        ax[0].text(c["pixel"][0], c["pixel"][1] - 12, f"hit: {c['category']}",
                   color="red", fontsize=9, ha="center")

    # -- right: top-down (x=right, z=forward) --
    a = ax[1]
    om = perception.obstacle_mask(frame.pts, frame.floor_y)
    ob = frame.pts[om][:, [0, 2]]
    if ob.shape[0] > 6000:
        ob = ob[np.linspace(0, ob.shape[0] - 1, 6000).astype(int)]
    a.scatter(ob[:, 0], ob[:, 1], s=1, c="0.7", label="obstacles")

    # view cone +-FOV_HALF
    R = 4.0
    for s in (+1, -1):
        ang = math.radians(s * config.FOV_HALF_DEG)
        a.plot([0, R * math.sin(ang)], [0, R * math.cos(ang)], "b--", lw=1)

    # path + endpoint
    samples = A.sample_path(acts, config.MARCH_STEP_M)
    px = [p[0] for p in samples]; pz = [p[1] for p in samples]
    a.plot(px, pz, "g-", lw=2, label="path")
    pf = outcome["pose_end_full"]; h = math.radians(pf["heading_deg"])
    a.arrow(pf["x"], pf["z"], 0.4 * math.sin(h), 0.4 * math.cos(h),
            head_width=0.12, color="green")

    # body swept corridor: chassis footprint (radius) along the path
    r = outcome["body"]["radius_m"]
    step = max(1, int(round(r / config.MARCH_STEP_M)))   # ~1 circle per radius of travel
    for i in range(0, len(samples), step):
        a.add_patch(Circle((samples[i][0], samples[i][1]), r,
                           fill=False, ec="green", lw=0.6, alpha=0.35))
    a.add_patch(Circle((0, 0), r, fill=False, ec="blue", lw=1.5,
                       label=f"body r={r:.2f}m"))         # start footprint
    a.add_patch(Circle((pf["x"], pf["z"]), r, fill=False, ec="green", lw=1.5))  # end footprint

    # contact
    if c:
        a.plot(c["xy"][0], c["xy"][1], "rx", ms=12, mew=3, label="contact")
        a.add_patch(Circle((c["xy"][0], c["xy"][1]), r, fill=False, ec="red", lw=1.5))

    # objects + before/after distance line to target
    for o in frame.objects:
        if o["is_structural"]:
            continue
        gx, gz = o["ground_xy_centroid"]
        a.plot(gx, gz, "k.", ms=4)
        a.text(gx, gz, o["category"], fontsize=6, color="black")
        if target_instance is not None and o["instance_id"] == target_instance:
            a.plot([0, gx], [0, gz], "c--", lw=1)                 # before
            a.plot([pf["x"], gx], [pf["z"], gz], "c-", lw=1.5)    # after

    a.plot(0, 0, "b^", ms=10)
    xs = [0, pf["x"]] + px; zs = [0, pf["z"]] + pz   # ensure body circles stay in view
    a.set_xlim(min(a.get_xlim()[0], min(xs) - r), max(a.get_xlim()[1], max(xs) + r))
    a.set_ylim(min(a.get_ylim()[0], min(zs) - r), max(a.get_ylim()[1], max(zs) + r))
    a.set_aspect("equal"); a.grid(alpha=0.3); a.set_xlabel("x (right)"); a.set_ylabel("z (fwd)")
    a.set_title(f"topdown  collided={outcome['collided']} arc={outcome['first_contact_arc_m']}")
    a.legend(loc="upper right", fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()
