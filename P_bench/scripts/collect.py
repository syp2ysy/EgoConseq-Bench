"""Batch collection: random poses x action seqs x bodies -> records.jsonl.

    MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python scripts/collect.py \
        --auto-scenes --poses-per-scene 20 --radii 0.15 0.20 0.25 \
        --out data/conseq/v2                 # HM3D (default backend)
    # ... --backend gs --gs-root /path/to/gs   # 3DGS scenes (gsplat)
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import config, action_gen, validate
from pipeline.actions import parse_actions
from pipeline.body import Cylinder
from pipeline.consequence import judge
from pipeline.frame import build_frame
from pipeline.record import build_record, append_record
from pipeline.sim import SimSession, discover_semantic_scenes


def action_seqs(file_path):
    """--action-mode file: load explicit sequences from JSON."""
    data = json.load(open(file_path))
    return [parse_actions(s["actions"]) for s in data["sequences"]]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenes", nargs="+")
    src.add_argument("--auto-scenes", action="store_true")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--poses-per-scene", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--radii", type=float, nargs="+", default=list(config.RADII_M))
    ap.add_argument("--action-mode", choices=["fov_len", "file"], default="fov_len")
    ap.add_argument("--action-file", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=list(config.GEN_LENGTHS))
    ap.add_argument("--keep-per-length", type=int, default=config.KEEP_PER_LENGTH)
    ap.add_argument("--pool-factor", type=int, default=config.POOL_FACTOR)
    ap.add_argument("--min-objects", type=int, default=1)
    ap.add_argument("--geodesic", action="store_true", help="compute per-relation geodesic (slow)")
    ap.add_argument("--save-arrays", action="store_true")
    ap.add_argument("--debug-images", action="store_true")
    ap.add_argument("--debug-outcomes-per-frame", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--backend", choices=["hm3d", "gs"], default="hm3d")
    ap.add_argument("--gs-root", default=config.GS_ROOT)
    args = ap.parse_args()

    if args.backend == "gs":
        from pipeline.gs_sim import GsSimSession as Session, discover_gs_scenes
        scenes = discover_gs_scenes(args.gs_root) if args.auto_scenes else args.scenes
    else:
        from pipeline.sim import SimSession as Session
        scenes = discover_semantic_scenes() if args.auto_scenes else args.scenes
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    os.makedirs(args.out, exist_ok=True)
    img_dir = os.path.join(args.out, "img"); os.makedirs(img_dir, exist_ok=True)
    arr_dir = os.path.join(args.out, "arr"); os.makedirs(arr_dir, exist_ok=True)
    records_path = os.path.join(args.out, "records.jsonl")
    open(records_path, "w").close()  # truncate

    rng = np.random.default_rng(args.seed)
    stats = collections.Counter()
    skip = collections.Counter()
    t0 = time.time()

    for si, scene in enumerate(scenes):
        sim = Session(scene)
        print(f"[{si+1}/{len(scenes)}] {sim.scene_id}")

        # sample poses (using the largest-radius navmesh)
        sim.recompute_navmesh(max(args.radii))
        frames = []
        for p in range(args.poses_per_scene):
            pose = sim.sample_random_pose(rng, args.radii)
            if pose is None:
                skip["sample_fail"] += 1; continue
            pos, yaw = pose
            fr = build_frame(sim, pos, yaw, frame_id=f"F-{sim.scene_id}-p{p:03d}",
                             scene_id=sim.scene_id, scene_glb=scene)
            n_ns = sum(0 if o["is_structural"] else 1 for o in fr.objects)
            if n_ns < args.min_objects:
                skip["few_objects"] += 1; continue
            frames.append(fr)
            Image.fromarray(fr.rgb).save(os.path.join(img_dir, fr.frame_id + ".png"))
            if args.save_arrays:
                np.save(os.path.join(arr_dir, fr.frame_id + "_depth.npy"), fr.depth)

        # action source: fov_len -> length-bucketed in-FOV pool; else flat seq list
        keepN = args.keep_per_length
        if args.action_mode == "fov_len":
            pool = action_gen.fov_bucketed_pool(rng, args.lengths, keepN * args.pool_factor)
            for L in args.lengths:
                stats[f"pool_L{L}"] += len(pool[L])
        else:
            seqs = action_seqs(args.action_file)

        # judge: outer loop over radius (navmesh recompute is expensive)
        outcomes = {fr.frame_id: [] for fr in frames}
        for r in args.radii:
            sim.recompute_navmesh(r)
            body = Cylinder(radius_m=r)
            rtag = f"b{int(r*100):03d}"
            for fr in frames:
                nav = sim.nav(fr.position, fr.yaw_rad)
                if args.action_mode == "fov_len":
                    for L in args.lengths:
                        kept = 0
                        for acts in pool[L]:
                            if kept >= keepN:
                                break
                            oc = judge(fr, body, acts, nav=nav, geodesic=args.geodesic)
                            verdict = oc["nav_check"]["verdict"] if oc["nav_check"] else "keep_agree"
                            stats[f"verdict_{verdict}"] += 1
                            if verdict == "discard_noise":       # red -> drop
                                stats["dropped_red"] += 1
                                continue
                            oc["seq_len"] = L
                            if verdict == "review_depth_hole":   # yellow -> keep + flag
                                oc["human_check"] = True
                                stats["flagged_yellow"] += 1
                            oc["outcome_id"] = f"{fr.frame_id}-{rtag}-L{L}-{kept:02d}"
                            outcomes[fr.frame_id].append(oc)
                            kept += 1
                            stats["outcomes"] += 1
                            stats["collided"] += int(oc["collided"])
                        if kept < keepN:
                            stats[f"shortfall_L{L}"] += 1
                else:
                    for ai, acts in enumerate(seqs):
                        oc = judge(fr, body, acts, nav=nav, geodesic=args.geodesic)
                        oc["outcome_id"] = f"{fr.frame_id}-{rtag}-a{ai:02d}"
                        outcomes[fr.frame_id].append(oc)
                        stats["outcomes"] += 1
                        stats["collided"] += int(oc["collided"])
                        if oc["nav_check"]:
                            stats[f"verdict_{oc['nav_check']['verdict']}"] += 1

        if args.debug_images:
            from pipeline import viz
            dbg = os.path.join(args.out, "debug"); os.makedirs(dbg, exist_ok=True)
            for fr in frames:
                Image.fromarray(viz.object_overlay(fr)).save(
                    os.path.join(dbg, f"overlay_{fr.frame_id}.png"))
                for oc in outcomes[fr.frame_id][:args.debug_outcomes_per_frame]:
                    tgt = (oc["contact"]["instance_id"]
                           if oc["contact"] and not oc["contact"]["unattributed"] else None)
                    viz.save_evidence(fr, oc, os.path.join(dbg, f"ev_{oc['outcome_id']}.png"),
                                      target_instance=tgt)

        for fr in frames:
            rec = build_record(fr, outcomes[fr.frame_id],
                               image_path=os.path.join("img", fr.frame_id + ".png"),
                               depth_path=(os.path.join("arr", fr.frame_id + "_depth.npy")
                                           if args.save_arrays else None))
            append_record(records_path, rec)
            stats["frames"] += 1
        sim.close()

    meta = {"scenes": len(scenes), "params": vars(args), "stats": dict(stats),
            "skipped": dict(skip), "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(args.out, "run_meta.json"), "w"), indent=2)
    print("stats:", dict(stats), "skipped:", dict(skip),
          f"collide_rate={stats['collided']/max(stats['outcomes'],1):.2f}",
          f"time={meta['seconds']}s")

    if not args.no_validate:
        total, viol = validate.validate_file(records_path)
        print(f"validate: {total} records, {len(viol)} violations")
        for v in viol[:20]:
            print("  ", v)
        if viol:
            sys.exit(1)


if __name__ == "__main__":
    main()
