"""P0 smoke: build a Frame on a real HM3D scene and eye-check object masks.

Verifies the offline semantic decode + depth-aligned instance assignment:
prints semantic coverage / objects, and saves an RGB + object-overlay PNG.

    MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python scripts/smoke_frame.py \
        --scene .../TEEsavR23oF.basis.glb --poses 2 --out data/conseq/smoke_p0
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import config
from pipeline.sim import SimSession
from pipeline.frame import build_frame


def _color(iid):
    rng = np.random.default_rng(iid * 9973 + 1)
    return rng.integers(60, 255, size=3)


def save_overlay(frame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = frame.depth.shape
    inst_img = np.zeros((H, W), np.int64)
    inst_img[frame.pts_uv[:, 1], frame.pts_uv[:, 0]] = frame.pts_sem

    overlay = frame.rgb.copy()
    for o in frame.objects:
        m = inst_img == o["instance_id"]
        overlay[m] = (0.45 * overlay[m] + 0.55 * _color(o["instance_id"])).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].imshow(frame.rgb); ax[0].set_title("RGB"); ax[0].axis("off")
    ax[1].imshow(overlay); ax[1].axis("off")
    ax[1].set_title(f"objects (cov={np.mean(frame.pts_sem>0):.2f})")
    for o in frame.objects:
        u, v = o["centroid_px"]
        ax[1].text(u, v, f"{o['category']}\n{o['dist_centroid_m']:.1f}m {o['bearing_deg']:+.0f}°",
                   color="white", fontsize=7, ha="center", va="center",
                   bbox=dict(facecolor="black", alpha=0.5, pad=1))
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--poses", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/conseq/smoke_p0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sim = SimSession(args.scene)
    print(f"scene {sim.scene_id}: {len(sim.id_to_cat)} annotated instances in palette")

    for p in range(args.poses):
        pose = sim.sample_random_pose(rng, config.RADII_M)
        if pose is None:
            print(f"pose {p}: sampling failed"); continue
        pos, yaw = pose
        fr = build_frame(sim, pos, yaw, frame_id=f"F-{sim.scene_id}-p{p:03d}",
                         scene_id=sim.scene_id, scene_glb=args.scene)
        cov = float(np.mean(fr.pts_sem > 0))
        print(f"\npose {p}: yaw={np.degrees(yaw):.0f} coverage={cov:.2f} "
              f"n_obj={len(fr.objects)} n_nonstruct={fr.n_objects if False else sum(0 if o['is_structural'] else 1 for o in fr.objects)}")
        print("  inventory:", fr.category_inventory)
        for o in sorted(fr.objects, key=lambda o: o["dist_centroid_m"])[:6]:
            print(f"    {o['category']:<14} dist={o['dist_centroid_m']:.2f} "
                  f"near={o['dist_nearest_m']:.2f} bearing={o['bearing_deg']:+.0f} "
                  f"area={o['mask_area_px']} struct={o['is_structural']}")
        save_overlay(fr, os.path.join(args.out, f"overlay_p{p:03d}.png"))
    sim.close()
    print(f"\noverlays -> {args.out}")


if __name__ == "__main__":
    main()
