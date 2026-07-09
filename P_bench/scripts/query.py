"""Manual single query: scene (+optional pose) + actions JSON -> full consequence.

    MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python scripts/query.py \
        --scene .../TEEsavR23oF.basis.glb --seed 7 --radius 0.25 \
        --actions '[{"type":"turn","deg":-30},{"type":"forward","m":1.2}]' \
        --out data/conseq/query_demo
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import config, viz
from pipeline.actions import parse_actions
from pipeline.body import Cylinder
from pipeline.consequence import judge
from pipeline.frame import build_frame
from pipeline.sim import SimSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--actions", required=True, help="JSON list of {type,deg|m}")
    ap.add_argument("--radius", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose", type=float, nargs=3, default=None, help="world x y z")
    ap.add_argument("--yaw", type=float, default=None, help="radians")
    ap.add_argument("--target", type=int, default=None, help="instance id to draw distance line")
    ap.add_argument("--geodesic", action="store_true")
    ap.add_argument("--out", default="data/conseq/query_demo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    acts = parse_actions(json.loads(args.actions))
    sim = SimSession(args.scene)
    sim.recompute_navmesh(args.radius)

    if args.pose is not None and args.yaw is not None:
        pos, yaw = np.array(args.pose, dtype=np.float64), float(args.yaw)
    else:
        pose = sim.sample_random_pose(np.random.default_rng(args.seed), [args.radius])
        if pose is None:
            print("pose sampling failed"); return
        pos, yaw = pose

    fr = build_frame(sim, pos, yaw, frame_id=f"Q-{sim.scene_id}",
                     scene_id=sim.scene_id, scene_glb=args.scene)
    nav = sim.nav(pos, yaw)
    oc = judge(fr, Cylinder(radius_m=args.radius), acts, nav=nav, geodesic=args.geodesic)

    print(f"scene={sim.scene_id} yaw={np.degrees(yaw):.0f} n_obj={len(fr.objects)}")
    print("actions:", oc["actions"])
    print(f"collided={oc['collided']} arc={oc['first_contact_arc_m']}")
    if oc["contact"]:
        print("contact:", {k: oc["contact"][k] for k in ("category", "instance_id", "vote_fraction", "unattributed")})
    print("pose_end_full:", {k: round(v, 3) for k, v in oc["pose_end_full"].items()})
    print("view_exit:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in oc["view_exit"].items()})
    print("nav_check:", None if not oc["nav_check"] else {k: oc["nav_check"][k] for k in ("d_nav_m", "verdict", "keep")})
    print("\nclosest targets after action (by centroid dist):")
    rels = sorted(oc["object_relations"], key=lambda r: r["full"]["dist_centroid_m"])
    for r in rels[:6]:
        f = r["full"]; d = r["delta_full"]
        print(f"  {r['category']:<14} dist={f['dist_centroid_m']:.2f} near={f['dist_nearest_m']:.2f} "
              f"bearing={f['bearing_deg']:+.0f} in_fov={f['in_fov']} d_centroid={d['d_centroid_m']:+.2f}")

    tgt = args.target
    if tgt is None and oc["contact"] and not oc["contact"]["unattributed"]:
        tgt = oc["contact"]["instance_id"]
    viz.save_evidence(fr, oc, os.path.join(args.out, "evidence.png"), target_instance=tgt)
    json.dump(oc, open(os.path.join(args.out, "outcome.json"), "w"),
              indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o))
    print(f"\nevidence -> {args.out}/evidence.png ; outcome.json")
    sim.close()


if __name__ == "__main__":
    main()
