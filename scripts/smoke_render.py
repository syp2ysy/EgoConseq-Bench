"""Smoke gate for Task 5: habitat_env render().

Loads the first HM3D val scene, samples a navigable point, renders
RGB + Depth, saves PNGs under data/demo/, and prints depth statistics.

Usage
-----
  cd /home/zhangshan/syp/myvln/P_bench
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \\
    python scripts/smoke_render.py
"""

from __future__ import annotations

import os
import sys
import math
import random

# Silence habitat / Magnum output before any import triggers them.
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np
from PIL import Image

# Make sure the package root is importable when running as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from egoconseq import config
from egoconseq.sim.habitat_env import EgoConseqSim

# ---------------------------------------------------------------------------
# Scene to use
# ---------------------------------------------------------------------------
SCENE_GLB = (
    "/home/zhangshan/syp/datasets/versioned_data/"
    "hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(_REPO_ROOT, "data", "demo")
OUT_RGB = os.path.join(OUT_DIR, "smoke_rgb.png")
OUT_DEPTH = os.path.join(OUT_DIR, "smoke_depth.png")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[smoke_render] Loading scene: {SCENE_GLB}")
    sim = EgoConseqSim(SCENE_GLB)

    try:
        # Sample a navigable position.
        # HM3D can have mesh holes on upper floors; try up to MAX_TRIES points
        # and use the first one where the rendered valid-depth ratio >= 0.8.
        MAX_TRIES = 20
        VALID_THRESHOLD = 0.8
        rgb = depth = K = pos = yaw = None
        for attempt in range(MAX_TRIES):
            cand_pos = sim.pathfinder.get_random_navigable_point()
            cand_yaw = random.uniform(0.0, 2 * math.pi)
            cand_rgb, cand_depth, cand_K, _ = sim.render(cand_pos, cand_yaw)
            cand_valid = (
                np.isfinite(cand_depth)
                & (cand_depth > 0)
                & (cand_depth <= config.D_MAX_M * 2)
            ).mean()
            if cand_valid >= VALID_THRESHOLD:
                rgb, depth, K = cand_rgb, cand_depth, cand_K
                pos, yaw = cand_pos, cand_yaw
                print(
                    f"[smoke_render] Point accepted on attempt {attempt + 1}  "
                    f"(valid={cand_valid:.4f})"
                )
                break
            else:
                print(
                    f"[smoke_render] Attempt {attempt + 1}: pos y={cand_pos[1]:.2f}  "
                    f"valid={cand_valid:.4f} < {VALID_THRESHOLD} — retrying…"
                )

        if rgb is None:
            print(
                f"[smoke_render] BLOCKED: could not find a navigable point with "
                f"valid-depth ratio >= {VALID_THRESHOLD} in {MAX_TRIES} attempts."
            )
            sys.exit(2)

        print(f"[smoke_render] Sampled position : {pos}")
        print(f"[smoke_render] Sampled yaw (rad): {yaw:.4f}  ({math.degrees(yaw):.1f}°)")

        # (rgb, depth, K already set in the sampling loop above)
        # ---- Depth statistics ----
        valid_mask = np.isfinite(depth) & (depth > 0) & (depth <= config.D_MAX_M * 2)
        valid_ratio = valid_mask.mean()
        d_min = float(depth[valid_mask].min()) if valid_mask.any() else float("nan")
        d_max = float(depth[valid_mask].max()) if valid_mask.any() else float("nan")
        d_mean = float(depth[valid_mask].mean()) if valid_mask.any() else float("nan")

        print(f"[smoke_render] Depth min  : {d_min:.3f} m")
        print(f"[smoke_render] Depth max  : {d_max:.3f} m")
        print(f"[smoke_render] Depth mean : {d_mean:.3f} m")
        print(f"[smoke_render] Valid-depth ratio (0, {config.D_MAX_M*2:.1f}m]: {valid_ratio:.4f}")

        # ---- Save RGB ----
        Image.fromarray(rgb).save(OUT_RGB)
        print(f"[smoke_render] Saved RGB  -> {OUT_RGB}")

        # ---- Save Depth (normalized 0–255 for visual inspection) ----
        d_vis = depth.copy()
        d_vis = np.clip(d_vis, 0, config.D_MAX_M * 2)
        d_vis = (d_vis / (config.D_MAX_M * 2) * 255).astype(np.uint8)
        Image.fromarray(d_vis).save(OUT_DEPTH)
        print(f"[smoke_render] Saved Depth -> {OUT_DEPTH}")

        # ---- Gate check ----
        if valid_ratio > 0.8:
            print("[smoke_render] SMOKE GATE PASSED  (valid-ratio > 0.8)")
        else:
            print(
                f"[smoke_render] WARNING: valid-ratio {valid_ratio:.4f} < 0.8  "
                "— gate NOT passed; check scene / navmesh."
            )
            sys.exit(1)

    finally:
        sim.close()


if __name__ == "__main__":
    main()
