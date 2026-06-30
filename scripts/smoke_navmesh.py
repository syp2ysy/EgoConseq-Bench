"""Smoke gate for Task 12: Per-radius navmesh d_safe cross-check.

For a single navigable pose in the HM3D val scene:
  1. For r in {0.10, 0.25, 0.40}: recompute navmesh, compute d_safe_navmesh.
  2. Assert the three values are monotonic non-increasing in r
     (a wider body contacts no later than a narrower one).
  3. Compare navmesh d_safe (r=0.25, turn=0) side-by-side with the depth-
     oracle d_safe_visible (r=0.25, turn=0) for the same pose.

Usage
-----
  cd /home/zhangshan/syp/myvln/P_bench
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \\
    python scripts/smoke_navmesh.py
"""

from __future__ import annotations

import math
import os
import random
import sys

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from egoconseq import config
from egoconseq.sim.habitat_env import EgoConseqSim
from egoconseq.sim.navmesh import recompute_navmesh, d_safe_navmesh

# Depth-oracle imports
from egoconseq.oracle.pointcloud import (
    backproject,
    to_agent_ground,
    estimate_floor_height,
    remove_floor,
)
from egoconseq.oracle.voxel import VoxelField
from egoconseq.oracle.sweep import d_safe_visible

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCENE_GLB = (
    "/home/zhangshan/syp/datasets/versioned_data/"
    "hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
)
MAX_TRIES = 30
VALID_THRESHOLD = 0.85
TURN_DEG = 0.0  # straight ahead
RADII = config.RADII_M          # (0.10, 0.25, 0.40)
COMPARE_RADIUS = 0.25           # radius used for cross-oracle comparison


def main() -> None:
    print(f"[smoke_navmesh] Loading scene: {SCENE_GLB}")
    sim_wrapper = EgoConseqSim(SCENE_GLB)

    try:
        # ------------------------------------------------------------------
        # Sample a valid navigable pose (same retry pattern as smoke_render)
        # ------------------------------------------------------------------
        pos = yaw = rgb = depth = K = None
        for attempt in range(MAX_TRIES):
            cand_pos = sim_wrapper.pathfinder.get_random_navigable_point()
            cand_yaw = random.uniform(0.0, 2 * math.pi)
            cand_rgb, cand_depth, cand_K, _ = sim_wrapper.render(cand_pos, cand_yaw)
            valid_ratio = (
                np.isfinite(cand_depth)
                & (cand_depth > 0)
                & (cand_depth <= config.D_MAX_M * 2)
            ).mean()
            if valid_ratio >= VALID_THRESHOLD:
                pos, yaw = cand_pos, cand_yaw
                rgb, depth, K = cand_rgb, cand_depth, cand_K
                print(
                    f"[smoke_navmesh] Pose accepted on attempt {attempt + 1}"
                    f"  valid={valid_ratio:.4f}"
                )
                break
            else:
                print(
                    f"[smoke_navmesh] Attempt {attempt + 1}: y={cand_pos[1]:.2f}"
                    f"  valid={valid_ratio:.4f} < {VALID_THRESHOLD} — retrying…"
                )

        if pos is None:
            print(
                f"[smoke_navmesh] BLOCKED: no valid pose after {MAX_TRIES} attempts."
            )
            sys.exit(2)

        print(f"[smoke_navmesh] Pose: pos={pos}  yaw={math.degrees(yaw):.1f}°")

        # ------------------------------------------------------------------
        # Part 1: Per-radius navmesh d_safe (monotonicity test)
        # ------------------------------------------------------------------
        print("\n--- Part 1: per-radius navmesh d_safe (turn=0°) ---")
        raw_sim = sim_wrapper.sim  # habitat_sim.Simulator
        d_values = []
        for r in RADII:
            recompute_navmesh(raw_sim, radius=r, height=config.CYLINDER_HEIGHT_M)
            d = d_safe_navmesh(
                sim_wrapper.pathfinder,
                pos=pos,
                yaw=yaw,
                turn_deg=TURN_DEG,
                d_max=config.D_MAX_M,
                step=config.MARCH_STEP_M,
            )
            d_values.append(d)
            print(f"  r={r:.2f} m  ->  d_safe_navmesh = {d:.4f} m")

        # Assert monotonic non-increasing: d(r1) >= d(r2) when r1 < r2
        monotonic = all(d_values[i] >= d_values[i + 1] for i in range(len(d_values) - 1))
        if monotonic:
            print("\n  [PASS] Monotonic non-increasing in r: "
                  f"{d_values[0]:.4f} >= {d_values[1]:.4f} >= {d_values[2]:.4f}")
        else:
            print(
                f"\n  [FAIL] NOT monotonic: {d_values}  "
                "(a wider body should contact no later than a narrower one)"
            )
            sys.exit(1)

        # ------------------------------------------------------------------
        # Part 2: Cross-oracle comparison (navmesh vs depth oracle, r=0.25)
        # ------------------------------------------------------------------
        print("\n--- Part 2: navmesh vs depth-oracle cross-check (r=0.25, turn=0°) ---")

        # Recompute navmesh for COMPARE_RADIUS (in case Part 1 left it at 0.40)
        recompute_navmesh(raw_sim, radius=COMPARE_RADIUS, height=config.CYLINDER_HEIGHT_M)
        d_nav = d_safe_navmesh(
            sim_wrapper.pathfinder,
            pos=pos,
            yaw=yaw,
            turn_deg=TURN_DEG,
            d_max=config.D_MAX_M,
            step=config.MARCH_STEP_M,
        )

        # Depth oracle (does NOT need navmesh; uses the render from the same pose)
        pts_cam = backproject(depth, K)
        pts_ground = to_agent_ground(pts_cam, camera_height=config.CAMERA_HEIGHT_M)
        floor_y = estimate_floor_height(pts_ground)
        pts_obs = remove_floor(pts_ground, floor_y, band=config.OBSTACLE_BAND_M)
        vf = VoxelField(pts_obs)
        result_vis = d_safe_visible(vf, radius=COMPARE_RADIUS, turn_deg=TURN_DEG)
        d_vis = result_vis.d_safe

        print(f"  d_safe_navmesh (r={COMPARE_RADIUS})  = {d_nav:.4f} m")
        print(f"  d_safe_visible (r={COMPARE_RADIUS})  = {d_vis:.4f} m")
        print(
            f"  ratio navmesh/visible = {d_nav / d_vis:.3f}"
            if d_vis > 0 else "  (d_vis=0, ratio undefined)"
        )
        print(
            "  (Note: not required to be equal — different sensing modalities; "
            "both should be in the same ballpark for open-scene forward march.)"
        )

        print("\n[smoke_navmesh] SMOKE GATE PASSED")

    finally:
        sim_wrapper.close()


if __name__ == "__main__":
    main()
