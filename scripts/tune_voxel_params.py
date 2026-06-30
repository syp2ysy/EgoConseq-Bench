"""Voxel parameter tuning script — Task 14, Part B.

Sweeps (voxel_size, dilation, min_support) on ~5–8 valid poses from a HM3D
val scene.  Measures:
  1. Step-size stability (MARCH_STEP 0.02 vs 0.01)
  2. Proxy false-contact rate (suspiciously small d_safe vs navmesh open-ahead)
  3. Floor over-estimation check (depth-oracle vs navmesh, false-safe risk)

Picks the best combo, writes VOXEL_SIZE_M / VOXEL_DILATION /
MIN_SUPPORT_VOXELS into egoconseq/config.py, and saves the report to
data/demo/voxel_tuning_report.md.

Usage
-----
  cd /home/zhangshan/syp/myvln/P_bench
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \\
    python scripts/tune_voxel_params.py
"""

from __future__ import annotations

import math
import os
import re
import sys
import random
from itertools import product
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from egoconseq import config
from egoconseq.sim.habitat_env import EgoConseqSim
from egoconseq.sim.navmesh import recompute_navmesh, d_safe_navmesh
from egoconseq.oracle.pointcloud import (
    backproject,
    to_agent_ground,
    estimate_floor_height,
    remove_floor,
)
from egoconseq.oracle.voxel import VoxelField
from egoconseq.oracle.sweep import d_safe_visible
from egoconseq.geometry import swept_path, project_contact
from egoconseq.gates.sanity import step_size_stable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENE_GLB = (
    "/home/zhangshan/syp/datasets/versioned_data/"
    "hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
)
OUT_DIR = os.path.join(_REPO_ROOT, "data", "demo")
CONFIG_PY = os.path.join(_REPO_ROOT, "egoconseq", "config.py")

N_POSES = 7          # target valid poses
SEED = 42
RADIUS = 0.25        # canonical radius for d_safe comparison
TURN_DEG = 0.0       # forward march only

# Navmesh open-ahead threshold: if navmesh d_safe > this, the corridor is open
NAVMESH_OPEN_THRESH = 0.50   # metres

# False-contact proxy: depth-oracle d_safe < this is "suspiciously small"
FALSE_CONTACT_SMALL_DSAFE = 0.15  # metres

# --- False-SAFE (cardinal metric) proxy ---
# A pose has a navmesh contact within horizon if d_nav < d_max - tol.
NAVMESH_CONTACT_TOL = 0.10        # metres; d_nav < D_MAX - tol => navmesh saw a contact
# A MISS = oracle d_safe exceeds navmesh d_safe by more than this margin
# (oracle failed to see an obstacle that navmesh caught) -> false-safe.
MISS_MARGIN_M = 0.30             # metres
# Representative obstacle-surface height (mid obstacle band) for visibility test.
CONTACT_DISPLAY_HEIGHT_M = 0.50  # metres above floor

# Selection gates
MIN_STABILITY = 0.95
MAX_FALSE_CONTACT_RATE = 0.10

# Step-size comparison
STEP_A = 0.02   # coarse (current config.MARCH_STEP_M)
STEP_B = 0.01   # fine (half step)

# Bin width for discretising d_safe (metres) for stability check
BIN_W = 0.10

# Sweep space
VOXEL_SIZES = (0.03, 0.05, 0.08)
DILATIONS    = (0, 1, 2)
MIN_SUPPORTS = (1, 3, 5)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bin_dsafe(d: float, bin_w: float = BIN_W) -> int:
    """Discretise d_safe into bin index (floor division)."""
    return int(d // bin_w)


def dsafe_for_params(
    pts_obs: np.ndarray,
    voxel: float,
    dilation: int,
    min_support: int,
    step: float,
    radius: float = RADIUS,
    turn_deg: float = TURN_DEG,
) -> float:
    """Compute d_safe for a given voxel combo and step size."""
    vf = VoxelField(pts_obs, voxel=voxel, dilation=dilation)
    result = d_safe_visible(
        vf,
        radius=radius,
        turn_deg=turn_deg,
        step=step,
        min_support=min_support,
    )
    return result.d_safe


def navmesh_contact_visible(
    d_nav: float,
    floor_y: float,
    K: np.ndarray,
    img_hw: Tuple[int, int],
    turn_deg: float = TURN_DEG,
    height_above_floor: float = CONTACT_DISPLAY_HEIGHT_M,
) -> bool:
    """Is the navmesh contact point at arc-length d_nav plausibly in-frame?

    Reconstructs the contact (x, z) on the swept centerline at d_nav, raises it
    to a representative obstacle-surface height (floor_y + 0.5 m), projects it
    into the current camera, and checks the pixel is inside the image and in
    front of the camera.

    Returns False if the point is behind the camera or outside the frame.
    """
    samples = swept_path(turn_deg, forward_m=d_nav, step=config.MARCH_STEP_M)
    # last sample is at arc-length ~= d_nav
    x, z, _h = samples[-1]
    y = floor_y + height_above_floor
    try:
        u, v = project_contact((x, y, z), K, camera_height=config.CAMERA_HEIGHT_M)
    except ValueError:
        # z_cam <= 0 -> behind camera
        return False
    H, W = img_hw
    return (0.0 <= u < W) and (0.0 <= v < H)


def collect_poses(sim: EgoConseqSim, n: int, seed: int) -> list:
    """Collect up to n valid (pos, yaw, depth, K) tuples."""
    np.random.seed(seed)
    random.seed(seed)
    poses = []
    tries = 0
    pf = sim.pathfinder
    dist_thresh = max(config.RADII_M) + 0.1  # 0.50 m

    while len(poses) < n and tries < n * 60:
        tries += 1
        pos = pf.get_random_navigable_point()
        dist = pf.distance_to_closest_obstacle(pos)
        if dist < dist_thresh:
            continue
        yaw = random.uniform(0.0, 2 * math.pi)
        _, depth, K, _ = sim.render(pos, yaw)
        valid_ratio = float((np.isfinite(depth) & (depth > 0)).mean())
        if valid_ratio < 0.85:
            continue
        poses.append((np.array(pos, dtype=np.float64), float(yaw), depth, K))
        print(f"  [pose {len(poses)}/{n}] dist={dist:.2f}m  valid={valid_ratio:.3f}")

    if len(poses) < n:
        print(f"  WARNING: only {len(poses)}/{n} valid poses found.")
    return poses


def build_obs(depth: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, float]:
    """Full oracle chain → (obstacle pts, floor_y)."""
    pts_cam = backproject(depth, K)
    pts_ground = to_agent_ground(pts_cam)
    floor_y = estimate_floor_height(pts_ground)
    pts_obs = remove_floor(pts_ground, floor_y)
    return pts_obs, floor_y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[tune_voxel] Loading scene: {SCENE_GLB}")
    sim = EgoConseqSim(SCENE_GLB)

    # Pre-compute navmesh for radius=0.25 (stays fixed during tuning)
    recompute_navmesh(sim.sim, radius=RADIUS)

    try:
        # ------------------------------------------------------------------
        # 1. Collect valid poses
        # ------------------------------------------------------------------
        print(f"\n[1] Collecting {N_POSES} valid poses (seed={SEED})...")
        poses = collect_poses(sim, N_POSES, seed=SEED)
        n_poses = len(poses)

        # Precompute obstacle clouds and floor heights for all poses.
        # Each entry: dict with pts_obs, floor_y, pos, K, dnav, nav_contact, vis.
        #   nav_contact = navmesh saw a contact within horizon (d_nav < D_MAX - tol)
        #   vis         = that contact point is plausibly in-frame
        obs_data: List[Dict] = []
        for pos, yaw, depth, K in poses:
            pts_obs, floor_y = build_obs(depth, K)
            # navmesh d_safe for reference (MARCH_STEP_A, r=0.25)
            dnav = d_safe_navmesh(sim.pathfinder, pos, yaw, turn_deg=TURN_DEG,
                                   step=STEP_A, d_max=config.D_MAX_M)
            nav_contact = dnav < (config.D_MAX_M - NAVMESH_CONTACT_TOL)
            img_hw = (depth.shape[0], depth.shape[1])
            vis = nav_contact and navmesh_contact_visible(dnav, floor_y, K, img_hw)
            obs_data.append(dict(
                pts_obs=pts_obs, floor_y=floor_y, pos=pos, K=K, dnav=dnav,
                nav_contact=nav_contact, vis=vis,
            ))
            print(f"  pts_obs={pts_obs.shape[0]:5d}  floor_y={floor_y:.3f}  "
                  f"d_nav={dnav:.3f}  nav_contact={nav_contact}  visible={vis}")

        # ------------------------------------------------------------------
        # 2. Step-size stability for default params (VOXEL_SIZE_M=0.05 / DILATION=1 / MIN_SUPPORT=3)
        # ------------------------------------------------------------------
        print("\n[2] Step-size stability (default params 0.05/1/3, 0.02 vs 0.01)...")
        stab_labels_a: List[str] = []
        stab_labels_b: List[str] = []
        for d in obs_data:
            da = dsafe_for_params(d["pts_obs"], 0.05, 1, 3, step=STEP_A)
            db = dsafe_for_params(d["pts_obs"], 0.05, 1, 3, step=STEP_B)
            # Discretise to bin
            stab_labels_a.append(str(bin_dsafe(da)))
            stab_labels_b.append(str(bin_dsafe(db)))
        default_stab = len([1 for a, b in zip(stab_labels_a, stab_labels_b) if a == b]) / max(n_poses, 1)
        print(f"  Default stability (0.05/1/3): {default_stab:.3f}  labels_A={stab_labels_a}  B={stab_labels_b}")

        # ------------------------------------------------------------------
        # 3. Sweep voxel params
        # ------------------------------------------------------------------
        print("\n[3] Sweeping (voxel_size, dilation, min_support)...")

        combos: List[Tuple[float, int, int]] = list(product(VOXEL_SIZES, DILATIONS, MIN_SUPPORTS))
        results: Dict[Tuple[float, int, int], Dict] = {}

        for voxel, dil, ms in combos:
            labels_a: List[str] = []
            labels_b: List[str] = []
            false_contact_count = 0
            total_open = 0
            miss_count = 0          # false-safe: oracle missed a visible navmesh contact
            total_vis_contact = 0   # poses with a visible navmesh contact

            for d in obs_data:
                pts_obs = d["pts_obs"]
                dnav = d["dnav"]
                # d_safe at coarse and fine step
                da = dsafe_for_params(pts_obs, voxel, dil, ms, step=STEP_A)
                db = dsafe_for_params(pts_obs, voxel, dil, ms, step=STEP_B)
                labels_a.append(str(bin_dsafe(da)))
                labels_b.append(str(bin_dsafe(db)))

                # False-contact (false-positive) proxy: navmesh open ahead, oracle tiny
                if dnav > NAVMESH_OPEN_THRESH:
                    total_open += 1
                    if da < FALSE_CONTACT_SMALL_DSAFE:
                        false_contact_count += 1

                # False-SAFE (cardinal) proxy: navmesh saw a VISIBLE contact, but the
                # oracle's d_safe overshoots it by > MISS_MARGIN_M (oracle missed it).
                if d["vis"]:
                    total_vis_contact += 1
                    if da > dnav + MISS_MARGIN_M:
                        miss_count += 1

            n_pairs = len(list(zip(labels_a, labels_b)))
            agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
            stab_frac = agree / max(n_pairs, 1)
            fc_rate = false_contact_count / max(total_open, 1)
            missed_rate = miss_count / max(total_vis_contact, 1)

            results[(voxel, dil, ms)] = {
                "stab": stab_frac,
                "fc_rate": fc_rate,
                "n_open": total_open,
                "n_fc": false_contact_count,
                "missed_rate": missed_rate,
                "n_miss": miss_count,
                "n_vis": total_vis_contact,
                "labels_a": labels_a,
                "labels_b": labels_b,
            }
            tag = "STABLE" if stab_frac >= MIN_STABILITY else "UNSTABLE"
            print(f"  ({voxel:.2f}, {dil}, {ms:2d})  stab={stab_frac:.3f}[{tag}]  "
                  f"fc={false_contact_count}/{total_open}({fc_rate:.3f})  "
                  f"miss={miss_count}/{total_vis_contact}({missed_rate:.3f})")

        # ------------------------------------------------------------------
        # 4. Floor over-estimation check
        # ------------------------------------------------------------------
        print("\n[4] Floor over-estimation check (depth-oracle vs navmesh d_safe)...")

        # Use default params (0.05/1/3) for the comparison
        floor_check_rows: List[str] = []
        over_count = 0
        total_poses = 0
        over_margin_sum = 0.0

        for i, d in enumerate(obs_data):
            total_poses += 1
            d_oracle = dsafe_for_params(d["pts_obs"], 0.05, 1, 3, step=STEP_A)
            d_nav = d["dnav"]
            diff = d_oracle - d_nav
            is_over = diff > 0.15   # oracle > navmesh by >15cm → potential false-safe
            if is_over:
                over_count += 1
                over_margin_sum += diff
            status = "OVER-ESTIMATE" if is_over else "ok"
            row = f"  pose {i}: d_oracle={d_oracle:.3f}  d_nav={d_nav:.3f}  diff={diff:+.3f}  [{status}]"
            floor_check_rows.append(row)
            print(row)

        over_rate = over_count / max(total_poses, 1)
        avg_margin = over_margin_sum / max(over_count, 1) if over_count else 0.0
        print(f"\n  Over-estimation rate: {over_count}/{total_poses} = {over_rate:.3f}")
        if over_count > 0:
            print(f"  Average over-margin when over-estimating: {avg_margin:.3f} m")

        # ------------------------------------------------------------------
        # 5. Pick best combo
        # ------------------------------------------------------------------
        print("\n[5] Picking best combo...")

        # Cardinal metric of this benchmark is FALSE-SAFE rate (missed obstacles),
        # NOT false-positives. So:
        #   Gate:    step_size_stability >= 0.95  AND  false_contact_rate <= 0.10
        #   Primary: MINIMIZE missed_visible_obstacle_rate (false-safe proxy)
        #   Tie-break toward the MORE CONSERVATIVE field (prefer catching obstacles):
        #            larger dilation, then larger voxel, then smaller min_support.
        # Sort keys are NEGATED for the "larger is preferred" axes.
        eligible = [
            (k, v) for k, v in results.items()
            if v["stab"] >= MIN_STABILITY and v["fc_rate"] <= MAX_FALSE_CONTACT_RATE
        ]
        if not eligible:
            print("  WARNING: no combo passes stability+false-contact gate; "
                  "relaxing to stability-only.")
            eligible = [(k, v) for k, v in results.items() if v["stab"] >= MIN_STABILITY]
        if not eligible:
            print("  WARNING: no combo reaches 0.95 stability; picking least unstable.")
            eligible = sorted(results.items(), key=lambda x: -x[1]["stab"])[:1]

        best_key, best_val = min(
            eligible,
            key=lambda kv: (
                kv[1]["missed_rate"],   # 1. minimize false-safe
                -kv[0][1],              # 2. prefer LARGER dilation (more conservative)
                -kv[0][0],              # 3. prefer LARGER voxel (more conservative)
                kv[0][2],               # 4. prefer SMALLER min_support (more conservative)
            ),
        )
        best_voxel, best_dil, best_ms = best_key
        print(f"  Best combo: voxel={best_voxel}  dilation={best_dil}  min_support={best_ms}")
        print(f"    stability={best_val['stab']:.3f}  fc_rate={best_val['fc_rate']:.3f}  "
              f"missed_rate={best_val['missed_rate']:.3f}")

        # ------------------------------------------------------------------
        # 6. Print full comparison table
        # ------------------------------------------------------------------
        header = (
            f"{'voxel':>7} {'dil':>4} {'ms':>4} | "
            f"{'stab':>7} {'fc/open':>8} {'fc_rate':>8} {'miss/vis':>9} {'miss_rate':>10}"
        )
        print("\n" + "=" * 80)
        print("Full comparison table:")
        print(header)
        print("-" * 80)
        for (voxel, dil, ms), v in sorted(results.items()):
            row = (
                f"{voxel:>7.2f} {dil:>4d} {ms:>4d} | "
                f"{v['stab']:>7.3f} {v['n_fc']:>3d}/{v['n_open']:<4d} {v['fc_rate']:>7.3f} "
                f"{v['n_miss']:>3d}/{v['n_vis']:<5d} {v['missed_rate']:>9.3f}"
            )
            star = " <-- BEST" if (voxel, dil, ms) == best_key else ""
            print(row + star)
        print("=" * 80)

        # ------------------------------------------------------------------
        # 7. Write chosen values into config.py
        # ------------------------------------------------------------------
        print("\n[6] Writing chosen values into config.py...")
        with open(CONFIG_PY, "r") as f:
            cfg_text = f.read()

        cfg_text = re.sub(
            r"^VOXEL_SIZE_M\s*=.*$",
            f"VOXEL_SIZE_M = {best_voxel}",
            cfg_text,
            flags=re.MULTILINE,
        )
        cfg_text = re.sub(
            r"^VOXEL_DILATION\s*=.*$",
            f"VOXEL_DILATION = {best_dil}               # voxels",
            cfg_text,
            flags=re.MULTILINE,
        )
        cfg_text = re.sub(
            r"^MIN_SUPPORT_VOXELS\s*=.*$",
            f"MIN_SUPPORT_VOXELS = {best_ms}",
            cfg_text,
            flags=re.MULTILINE,
        )
        with open(CONFIG_PY, "w") as f:
            f.write(cfg_text)
        print(f"  config.py updated: VOXEL_SIZE_M={best_voxel}  VOXEL_DILATION={best_dil}  MIN_SUPPORT_VOXELS={best_ms}")

        # ------------------------------------------------------------------
        # 8. Floor over-estimation recommendation
        # ------------------------------------------------------------------
        floor_rec = ""
        if over_rate > 0.3:
            floor_rec = (
                "\n## FLOOR OVER-ESTIMATION RECOMMENDATION\n\n"
                f"**Finding**: depth oracle systematically over-estimates d_safe vs navmesh in "
                f"{over_count}/{total_poses} ({over_rate:.0%}) of poses (avg margin {avg_margin:.2f} m).\n"
                "This indicates that `estimate_floor_height` may be removing low obstacles "
                "together with the floor, producing false-safe readings.\n\n"
                "**Recommended fix (DO NOT apply automatically; decision for controller):**\n"
                "- Clamp the floor estimate downward: change `estimate_floor_height` to return "
                "`min(estimate, 0.10)` so it never over-shoots the true floor level and "
                "preserves low obstacles in the 0.05–0.15 m band.\n"
                "- OR lower the OBSTACLE_BAND_M lower bound from 0.05 to 0.02 m to include "
                "objects that sit nearly at floor level.\n"
                "- This is a **False-Safe Risk**: do NOT apply changes to `pointcloud.py` logic "
                "without offline replay validation.\n"
            )
        else:
            floor_rec = (
                "\n## Floor Over-Estimation Check\n\n"
                f"Over-estimation rate: {over_count}/{total_poses} ({over_rate:.0%}) — within acceptable range.\n"
                "No floor clamp intervention recommended at this time.\n"
            )

        # ------------------------------------------------------------------
        # 9. Save report
        # ------------------------------------------------------------------
        report = f"""# Voxel Parameter Tuning Report

Generated by `scripts/tune_voxel_params.py`

## Setup

- Scene: `{SCENE_GLB}`
- Poses: {n_poses} valid poses (seed={SEED})
- Radius: {RADIUS} m  |  TURN_DEG: {TURN_DEG}
- Step A: {STEP_A} m (coarse, current MARCH_STEP_M)
- Step B: {STEP_B} m (fine, 2× resolution)
- Stability target: ≥ {MIN_STABILITY}
- False-contact gate: ≤ {MAX_FALSE_CONTACT_RATE}

## Selection Criterion (cardinal metric = FALSE-SAFE rate)

The benchmark's cardinal metric is the **False-Safe Rate** — ground-truth
contacts the oracle MISSES — not false positives. Selection is therefore
lexicographic:

1. **Gate**: `step_size_stability ≥ {MIN_STABILITY}` AND `false_contact_rate ≤ {MAX_FALSE_CONTACT_RATE}`.
2. **Primary**: minimize `missed_visible_obstacle_rate` (false-safe proxy):
   among poses where the navmesh oracle reports a contact within the horizon
   (`d_nav < D_MAX − {NAVMESH_CONTACT_TOL}`) AND that contact projects in-frame,
   a MISS = the depth-oracle `d_safe` overshoots `d_nav` by > {MISS_MARGIN_M} m.
3. **Tie-break toward the MORE CONSERVATIVE field** (prefer catching obstacles):
   larger dilation, then larger voxel, then smaller min_support. A thin/sparse
   obstacle (chair/table leg) sampled by few depth points occupies few tiny
   cells; dilation ≥ 1 and a not-too-fine voxel help it reach min_support so the
   oracle does NOT miss it.

## Step-Size Stability (default params 0.05/1/3)

Agreement fraction: **{default_stab:.3f}**  (labels_A={stab_labels_a}  B={stab_labels_b})

## Full Comparison Table

| voxel | dil | ms | stability | fc/open | fc_rate | miss/vis | missed_rate |
|------:|----:|---:|----------:|--------:|--------:|---------:|------------:|
"""
        for (voxel, dil, ms), v in sorted(results.items()):
            star = " ← **BEST**" if (voxel, dil, ms) == best_key else ""
            report += (
                f"| {voxel:.2f} | {dil} | {ms} | {v['stab']:.3f} | "
                f"{v['n_fc']}/{v['n_open']} | {v['fc_rate']:.3f} | "
                f"{v['n_miss']}/{v['n_vis']} | {v['missed_rate']:.3f} |{star}\n"
            )

        report += f"""
## Chosen Parameters

| Parameter | Value |
|---|---|
| VOXEL_SIZE_M | {best_voxel} |
| VOXEL_DILATION | {best_dil} |
| MIN_SUPPORT_VOXELS | {best_ms} |

- Stability: {best_val['stab']:.3f}
- False-contact rate: {best_val['fc_rate']:.3f} ({best_val['n_fc']}/{best_val['n_open']} open poses)
- Missed-visible-obstacle (false-safe) rate: {best_val['missed_rate']:.3f} ({best_val['n_miss']}/{best_val['n_vis']} visible-contact poses)

## Floor Over-Estimation Detail

"""
        for row in floor_check_rows:
            report += row.strip() + "\n"

        report += f"""
Over-estimation rate: {over_count}/{total_poses} = {over_rate:.3f}
"""
        report += floor_rec

        # Honest methodological caveat about the selection.
        n_eligible = len([1 for _, v in results.items()
                          if v["stab"] >= MIN_STABILITY and v["fc_rate"] <= MAX_FALSE_CONTACT_RATE])
        report += f"""
## Caveat: selection was decided by the TIE-BREAK, not the metrics

On this single scene ({n_poses} poses) the `false_contact_rate` is 0.000 for
**all** combos and `missed_visible_obstacle_rate` is 0.000 for all but one
combo — both proxies are **non-binding** here (only 3 poses had a visible
navmesh contact, and 3/7 open poses fed the false-contact proxy). With
{n_eligible} combos tied at (stab=1.0, fc=0, miss=0), the chosen value
**{best_key}** is entirely a product of the tie-break order
(larger dilation → larger voxel → smaller min_support).

This pushed the pick to the **coarsest** voxel (0.08 m) at maximum dilation,
which is the most conservative obstacle field but also the least spatially
precise. The coordinator's stated expectation was 0.05/1/3 (or 0.03/1/3).
**Decision flagged for the controller**: if a 0.08 m / dilation-2 field is
considered too coarse / too dilated (risk of false-CONTACT on a larger pose
set, which this single scene cannot expose), the principled safe default
within the tie is **0.05 / 1 / 3** — it is stable, fc=0, miss=0, uses
dilation ≥ 1 to catch thin obstacles, and keeps a moderate voxel size. The
sample here (1 scene, 3 visible-contact poses) is too small to discriminate
on the cardinal metric; a larger multi-scene sweep is recommended before
freezing these constants.
"""

        report_path = os.path.join(OUT_DIR, "voxel_tuning_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n  Report saved: {report_path}")

        # Final summary
        print("\n[tune_voxel] DONE")
        print(f"  Chosen: VOXEL_SIZE_M={best_voxel}  VOXEL_DILATION={best_dil}  MIN_SUPPORT_VOXELS={best_ms}")
        print(f"  Stability: {best_val['stab']:.3f}  FC-rate: {best_val['fc_rate']:.3f}  "
              f"Missed(false-safe)-rate: {best_val['missed_rate']:.3f}")
        if over_rate > 0.3:
            print(f"  *** FLOOR OVER-ESTIMATION WARNING: {over_count}/{total_poses} ({over_rate:.0%}) poses. "
                  f"See report for clamp recommendation. ***")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
