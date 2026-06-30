"""Smoke gate for Task 10: Sweep oracle overlay + P1 oracle smoke gate.

For each of N navigable poses in the target HM3D val scene, runs the full
oracle pipeline and saves:
  - data/demo/p1_overlay_{i}.png  : RGB with swept corridor + contact marker
  - data/demo/p1_topdown_{i}.png  : top-down matplotlib scatter of obstacle
                                     cloud, swept path, and contact circle

Usage
-----
  cd /home/zhangshan/syp/myvln/P_bench
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \\
    python scripts/smoke_oracle.py
"""

from __future__ import annotations

import math
import os
import random
import sys

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless; must be set before pyplot import
import matplotlib.pyplot as plt
from PIL import Image

# Make sure the package root is importable when running as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from egoconseq import config
from egoconseq.sim.habitat_env import EgoConseqSim
from egoconseq.oracle.pointcloud import (
    backproject,
    to_agent_ground,
    estimate_floor_height,
    remove_floor,
)
from egoconseq.oracle.voxel import VoxelField
from egoconseq.oracle.sweep import d_safe_visible, attach_contact_projection
from egoconseq.geometry import swept_path
from egoconseq.oracle.overlay import compute_corridor_pixels, draw_sweep_overlay

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCENE_GLB = (
    "/home/zhangshan/syp/datasets/versioned_data/"
    "hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
)
OUT_DIR = os.path.join(_REPO_ROOT, "data", "demo")

N_POSES = 4
MAX_TRIES_PER_POSE = 30
VALID_THRESHOLD = 0.85
RADIUS = 0.25
TURN_DEG = 0.0  # forward march


# ---------------------------------------------------------------------------
# Top-down plot helper
# ---------------------------------------------------------------------------

def save_topdown(
    out_path: str,
    obstacle_pts: np.ndarray,
    path_samples: list,
    d_safe: float,
    contact_xy,
    radius: float,
    step: float = config.MARCH_STEP_M,
    title: str = "",
) -> None:
    """Save a top-down (x horizontal, z forward/vertical) scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 7))

    # Obstacle cloud (x, z)
    if obstacle_pts.shape[0] > 0:
        # Subsample for speed if very large
        n = obstacle_pts.shape[0]
        idx = np.random.choice(n, min(n, 8000), replace=False)
        ax.scatter(
            obstacle_pts[idx, 0], obstacle_pts[idx, 2],
            s=1, c="steelblue", alpha=0.4, label="obstacle pts",
        )

    # Swept path centerline (only up to d_safe)
    cx_list, cz_list = [], []
    for i, (x, z, _h) in enumerate(path_samples):
        arc = i * step
        if arc > d_safe + 1e-9:
            break
        cx_list.append(x)
        cz_list.append(z)

    if cx_list:
        ax.plot(cx_list, cz_list, "y-", linewidth=2, label="swept path")

    # Contact disk / end-of-path circle
    if contact_xy is not None:
        cx, cz = contact_xy
        circle = plt.Circle((cx, cz), radius, color="red", fill=False, linewidth=2, label="contact disk")
        ax.add_patch(circle)
        ax.plot(cx, cz, "rx", markersize=10, markeredgewidth=2)
    else:
        # No contact → draw circle at end of swept path
        if cx_list:
            circle = plt.Circle(
                (cx_list[-1], cz_list[-1]), radius,
                color="orange", fill=False, linewidth=2, label="path end (no contact)"
            )
            ax.add_patch(circle)

    # Agent at origin
    ax.plot(0, 0, "g^", markersize=12, label="agent")
    agent_circle = plt.Circle((0, 0), radius, color="green", fill=False, linewidth=1, linestyle="--")
    ax.add_patch(agent_circle)

    ax.set_xlabel("x (right)  [m]")
    ax.set_ylabel("z (forward)  [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[smoke_oracle] Loading scene: {SCENE_GLB}")
    sim = EgoConseqSim(SCENE_GLB)

    try:
        poses_accepted: list = []
        attempt_total = 0

        while len(poses_accepted) < N_POSES and attempt_total < N_POSES * MAX_TRIES_PER_POSE:
            attempt_total += 1
            cand_pos = sim.pathfinder.get_random_navigable_point()
            cand_yaw = random.uniform(0.0, 2 * math.pi)
            cand_rgb, cand_depth, cand_K, _ = sim.render(cand_pos, cand_yaw)
            valid_ratio = (
                np.isfinite(cand_depth)
                & (cand_depth > 0)
                & (cand_depth <= config.D_MAX_M * 2)
            ).mean()
            if valid_ratio >= VALID_THRESHOLD:
                poses_accepted.append((cand_pos, cand_yaw, cand_rgb, cand_depth, cand_K))
                print(
                    f"  [pose {len(poses_accepted)}] accepted at attempt {attempt_total}"
                    f"  valid={valid_ratio:.4f}"
                )
            else:
                print(
                    f"  [attempt {attempt_total}] pos y={cand_pos[1]:.2f}"
                    f"  valid={valid_ratio:.4f} < {VALID_THRESHOLD} — retrying…"
                )

        if len(poses_accepted) < N_POSES:
            print(
                f"[smoke_oracle] WARNING: only {len(poses_accepted)}/{N_POSES} poses "
                f"accepted after {attempt_total} attempts. Proceeding with what we have."
            )

        if not poses_accepted:
            print("[smoke_oracle] BLOCKED: no valid poses found.")
            sys.exit(2)

        # ------------------------------------------------------------------ #
        # Per-pose processing                                                  #
        # ------------------------------------------------------------------ #
        overlay_paths = []
        topdown_paths = []

        for i, (pos, yaw, rgb, depth, K) in enumerate(poses_accepted):
            print(f"\n=== Pose {i} ===")
            print(f"  pos   : {pos}  yaw={math.degrees(yaw):.1f}°")

            # --- Full oracle chain ---
            pts_cam = backproject(depth, K)
            pts_ground = to_agent_ground(pts_cam, camera_height=config.CAMERA_HEIGHT_M)
            floor_y = estimate_floor_height(pts_ground)
            pts_obs = remove_floor(pts_ground, floor_y, band=config.OBSTACLE_BAND_M)
            vf = VoxelField(pts_obs)
            result = d_safe_visible(vf, radius=RADIUS, turn_deg=TURN_DEG)
            result = attach_contact_projection(result, floor_y=floor_y, K=K,
                                               camera_height=config.CAMERA_HEIGHT_M)

            print(f"  floor_y      : {floor_y:.4f} m")
            print(f"  d_safe       : {result.d_safe:.4f} m")
            print(f"  contact_xy   : {result.contact_xy}")
            print(f"  contact_pixel: {result.contact_pixel}")
            print(f"  #obstacle pts: {pts_obs.shape[0]}")

            # --- Corridor pixels ---
            path_samples = swept_path(TURN_DEG, forward_m=result.d_safe, step=config.MARCH_STEP_M)
            # Also generate samples up to d_safe for the corridor projection
            # (swept_path already caps at forward_m=d_safe)
            # For a level camera at 1.5m, floor-level points (y_cam ≈ -1.5)
            # are only visible at z > ~2m.  Project the corridor at body height
            # (1.0m above floor, y_cam ≈ -0.5) so the corridor is visible in the
            # middle of the image for obstacles at z > ~0.8m.
            corridor = compute_corridor_pixels(
                path_samples,
                floor_y=floor_y,
                K=K,
                radius=RADIUS,
                d_safe=result.d_safe,
                step=config.MARCH_STEP_M,
                camera_height=config.CAMERA_HEIGHT_M,
                display_height_above_floor=1.0,  # body-height for camera visibility
            )

            # --- Overlay PNG ---
            overlay_img = draw_sweep_overlay(
                rgb,
                path_pixels=corridor,
                contact_pixel=result.contact_pixel,
            )
            ov_path = os.path.join(OUT_DIR, f"p1_overlay_{i}.png")
            overlay_img.save(ov_path)
            overlay_paths.append(ov_path)
            print(f"  -> overlay saved: {ov_path}")

            # --- Top-down PNG ---
            td_path = os.path.join(OUT_DIR, f"p1_topdown_{i}.png")
            # For the full topdown we want all path samples up to d_safe
            all_path_samples = swept_path(TURN_DEG, forward_m=result.d_safe, step=config.MARCH_STEP_M)
            save_topdown(
                td_path,
                obstacle_pts=pts_obs,
                path_samples=all_path_samples,
                d_safe=result.d_safe,
                contact_xy=result.contact_xy,
                radius=RADIUS,
                step=config.MARCH_STEP_M,
                title=f"Pose {i}  |  d_safe={result.d_safe:.3f} m",
            )
            topdown_paths.append(td_path)
            print(f"  -> topdown saved: {td_path}")

        print("\n=== All files written ===")
        for p in overlay_paths + topdown_paths:
            print(f"  {p}")

        print("\n[smoke_oracle] SMOKE GATE PASSED")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
