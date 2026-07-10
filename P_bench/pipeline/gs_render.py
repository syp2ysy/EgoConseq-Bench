"""gsplat renderer for 3DGS (`.gs.ply`) scenes — RGB + depth at a Habitat pose.

Camera convention matches `pipeline/perception`/`sim.py`:
  world is Y-up; at yaw=0 the camera looks toward world -Z, +x = right, +y = up.
  OpenCV camera axes (for gsplat) are +x right, +y down, +z forward(=viewing dir).
So camera-to-world columns [right, down, forward] with the agent yaw about +Y.
Rendering depends only on the gaussians (same frame as the scene's `.navmesh`),
so it is independent of the InteriorGS bbox alignment (that is `gs_semantic`'s job).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from plyfile import PlyData
from gsplat import rasterization

from pipeline import config

_SH_C0 = 0.28209479177387814     # SH degree-0 basis -> approx albedo from f_dc


def load_gs(path: str, device: str = "cuda") -> dict:
    """Parse a 3DGS `.ply` into gsplat tensors (DC-only colour for the POC)."""
    v = PlyData.read(path)["vertex"]
    col = lambda n: np.asarray(v[n], np.float32)
    xyz = np.stack([col("x"), col("y"), col("z")], 1)
    scales = np.stack([col(f"scale_{i}") for i in range(3)], 1)
    quats = np.stack([col(f"rot_{i}") for i in range(4)], 1)     # (w,x,y,z), gsplat normalizes
    fdc = np.stack([col(f"f_dc_{i}") for i in range(3)], 1)
    rgb = np.clip(0.5 + _SH_C0 * fdc, 0.0, 1.0)                  # view-independent albedo
    tt = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
    return {"means": tt(xyz), "quats": tt(quats),
            "scales": torch.exp(tt(scales)), "opacities": torch.sigmoid(tt(col("opacity"))),
            "colors": tt(rgb), "device": device}


def _viewmat(position, yaw: float, cam_h: float) -> np.ndarray:
    """4x4 world->camera for OpenCV axes, matching perception.world_from_local."""
    c, s = math.cos(yaw), math.sin(yaw)
    right = np.array([c, 0.0, -s])          # agent +x in world
    down = np.array([0.0, -1.0, 0.0])       # camera +y = down
    forward = np.array([-s, 0.0, -c])       # viewing dir (agent -z) in world
    R_c2w = np.stack([right, down, forward], axis=1)   # columns
    cam = np.asarray(position, np.float64) + np.array([0.0, cam_h, 0.0])
    R_w2c = R_c2w.T
    vm = np.eye(4)
    vm[:3, :3] = R_w2c
    vm[:3, 3] = -R_w2c @ cam
    return vm


def render(gs: dict, position, yaw: float, K: np.ndarray, hw,
           cam_h: float = config.CAMERA_HEIGHT_M,
           near: float = 0.05, far: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """Render (rgb uint8 HxWx3, depth float32 HxW) — contract matches sim.render."""
    H, W = int(hw[0]), int(hw[1])
    dev = gs["device"]
    vm = torch.from_numpy(_viewmat(position, yaw, cam_h)[None]).float().to(dev)
    Ks = torch.from_numpy(np.asarray(K, np.float64)[None]).float().to(dev)
    out, _alpha, _meta = rasterization(
        gs["means"], gs["quats"], gs["scales"], gs["opacities"], gs["colors"],
        vm, Ks, W, H, render_mode="RGB+ED", near_plane=near, far_plane=far,
        radius_clip=0.0,
    )
    img = out[0]                                   # (H, W, 4): RGB + expected depth
    rgb = (img[..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
    depth = img[..., 3].contiguous().cpu().numpy().astype(np.float32)
    return rgb, depth
