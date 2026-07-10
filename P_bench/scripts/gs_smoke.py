"""Render smoke test for a GS scene: sample a navmesh pose -> gsplat RGB+depth.

    PATH=$ENV/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
    MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python scripts/gs_smoke.py \
        --scene /home/zhangshan/syp/datasets/gs/interior_0733_841584 --out /tmp
"""
import argparse
import os
import sys

import numpy as np
import habitat_sim
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config, gs_render, perception


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="/tmp")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    print("loading gaussians ...", flush=True)
    gs = gs_render.load_gs(os.path.join(args.scene, "scene.gs.ply"))
    print("  N gaussians:", gs["means"].shape[0], flush=True)

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(os.path.join(args.scene, "scene.navmesh"))
    pos = None
    for _ in range(1000):
        p = pf.get_random_navigable_point()
        if np.all(np.isfinite(p)) and pf.distance_to_closest_obstacle(p) > 0.6:
            pos = p
            break
    print("  pose:", np.round(pos, 2), flush=True)

    K = config.intrinsics()
    H, W = config.hw()
    best = None
    for yaw_deg in (0, 45, 90, 135, 180, 225, 270, 315):
        yaw = np.radians(yaw_deg)
        rgb, depth = gs_render.render(gs, pos, yaw, K, (H, W))
        valid = np.isfinite(depth) & (depth > 0)
        pts, _ = perception.unproject(depth, K)
        pg = perception.to_agent_ground(pts)
        fy = perception.estimate_floor_height(pg)
        fr = float((np.abs(pg[:, 1] - fy) <= 0.1).sum()) / depth.size
        dv = depth[valid]
        print(f"  yaw={yaw_deg:3d}: valid={valid.mean():.2f} "
              f"depth[min/med/max]={dv.min():.2f}/{np.median(dv):.2f}/{dv.max():.2f} "
              f"floor={fr:.2f}", flush=True)
        if best is None or fr > best[0]:
            best = (fr, yaw_deg, rgb, depth)

    fr, yaw_deg, rgb, depth = best
    Image.fromarray(rgb).save(os.path.join(args.out, "gs_rgb.png"))
    dd = depth.copy(); dd[~np.isfinite(dd)] = 0; dd = np.clip(dd, 0, 8) / 8
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(os.path.join(args.out, "gs_depth.png"), dd, cmap="turbo")
    print(f"SAVED rgb+depth (best yaw={yaw_deg}, floor={fr:.2f})", flush=True)


if __name__ == "__main__":
    main()
