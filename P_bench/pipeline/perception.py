"""Depth -> ground point cloud + occupancy field (pure numpy, no Habitat).

Copied and cleaned from the validated egoconseq math. Single source of truth
for all pixel<->3D geometry in the pipeline.

Frames
------
Camera frame: +z forward (z == depth), +x right, +y up.
Agent ground frame: origin at footprint centre ON THE FLOOR, +z forward,
    +x right, +y up. Level camera at height h => y_ground = y_cam + h.

Pixel convention: u = column, v = row (grows DOWN). backproject uses
    x = (u-cx)/fx*d ; y = (cy-v)/fy*d ; z = d
so project_ground below is its exact inverse.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from pipeline import config


def _valid_mask(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float64)
    return np.isfinite(d) & (d > 0.0)


def unproject(depth: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Depth image -> (camera-frame points (N,3), pixel uv (N,2 int)).

    Only pixels with finite depth > 0 are kept; uv[i] = (u, v) is the source
    pixel of pts_cam[i], so callers can slice any per-pixel array (e.g.
    semantic) with ``arr[uv[:,1], uv[:,0]]`` in matching order.
    """
    depth = np.asarray(depth, dtype=np.float64)
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    us, vs = np.meshgrid(np.arange(W), np.arange(H))  # (H,W) int
    valid = _valid_mask(depth)

    u = us[valid].astype(np.float64)
    v = vs[valid].astype(np.float64)
    d = depth[valid]

    x = (u - cx) / fx * d
    y = (cy - v) / fy * d
    z = d
    pts_cam = np.stack([x, y, z], axis=1)
    uv = np.stack([us[valid], vs[valid]], axis=1).astype(np.int64)
    return pts_cam, uv


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame points only (convenience; see :func:`unproject`)."""
    return unproject(depth, K)[0]


def to_agent_ground(pts_cam: np.ndarray,
                    camera_height: float = config.CAMERA_HEIGHT_M) -> np.ndarray:
    """Camera-frame -> agent ground frame (level camera: y += camera_height)."""
    pts = np.asarray(pts_cam, dtype=np.float64).copy()
    pts[:, 1] += camera_height
    return pts


def project_ground(point_3d_ground, K: np.ndarray,
                   camera_height: float = config.CAMERA_HEIGHT_M
                   ) -> Optional[Tuple[float, float]]:
    """Ground-frame 3D point -> pixel (u, v); None if at/behind the camera.

    Exact inverse of unproject + to_agent_ground for a level camera.
    """
    x_g, y_g, z_g = float(point_3d_ground[0]), float(point_3d_ground[1]), float(point_3d_ground[2])
    z_cam = z_g
    if z_cam <= 0:
        return None
    y_cam = y_g - camera_height
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * x_g / z_cam + cx
    v = cy - fy * y_cam / z_cam
    return (float(u), float(v))


def world_from_local(pts_ground: np.ndarray, pos, yaw: float) -> np.ndarray:
    """Agent-ground-frame points -> Habitat world frame.

    Local axes: +x right, +y up (height above floor), +z forward. Agent root
    `pos` is at floor level; yaw is rotation about +Y (yaw=0 faces world -Z).
        right   = ( cos y, 0, -sin y)
        up      = ( 0,     1,  0)
        forward = (-sin y, 0, -cos y)
    """
    p = np.asarray(pts_ground, dtype=np.float64)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    cy, sy = math.cos(yaw), math.sin(yaw)
    wx = pos[0] + x * cy - z * sy
    wy = pos[1] + y
    wz = pos[2] - x * sy - z * cy
    return np.stack([wx, wy, wz], axis=1)


def estimate_floor_height(pts_ground: np.ndarray) -> float:
    """Robust floor y in the ground frame (mode of a tight [-0.25,0.30] window).

    Returns the constructional floor level 0.0 when the floor is not densely
    visible (<200 in-window points), rather than chasing furniture/artifacts.
    """
    y = np.asarray(pts_ground, dtype=np.float64)[:, 1]
    lo, hi, bin_w, min_support = -0.25, 0.30, 0.05, 200
    window = y[(y >= lo) & (y <= hi)]
    if window.size < min_support:
        return 0.0
    edges = np.arange(lo, hi + bin_w, bin_w)
    counts, edges = np.histogram(window, bins=edges)
    k = int(np.argmax(counts))
    return float((edges[k] + edges[k + 1]) / 2.0)


def obstacle_mask(pts_ground: np.ndarray, floor_y: float,
                  band: Tuple[float, float] = config.OBSTACLE_BAND_M) -> np.ndarray:
    """Boolean mask selecting obstacle-band points (height above floor in band)."""
    above = np.asarray(pts_ground, dtype=np.float64)[:, 1] - floor_y
    low, high = band
    return (above >= low) & (above <= high)


class VoxelField:
    """2-D (x, z) occupancy grid with binary dilation and radius support query.

    Build from obstacle-band ground points (y ignored). ``support_count`` returns
    the number of occupied dilated voxel centres within a footprint disk.
    """

    _MARGIN_VOXELS = 2

    def __init__(self, pts_ground: np.ndarray,
                 voxel: float = config.VOXEL_SIZE_M,
                 dilation: int = config.VOXEL_DILATION) -> None:
        self._voxel = float(voxel)
        pts = np.asarray(pts_ground, dtype=np.float64)
        if pts.shape[0] == 0:
            self._tree = None
            return

        xz = pts[:, [0, 2]]
        margin = self._MARGIN_VOXELS * voxel
        origin = xz.min(axis=0) - margin
        extent = xz.max(axis=0) + margin - origin
        n_cells = np.maximum(np.ceil(extent / voxel).astype(int), 1)

        idx = np.clip(np.floor((xz - origin) / voxel).astype(int), 0, n_cells - 1)
        grid = np.zeros(n_cells, dtype=bool)
        grid[idx[:, 0], idx[:, 1]] = True
        if dilation > 0:
            grid = binary_dilation(grid, iterations=dilation)

        occ_ix, occ_iz = np.nonzero(grid)
        centers = (np.stack([occ_ix, occ_iz], axis=1).astype(np.float64) + 0.5) * voxel + origin
        self._tree = cKDTree(centers) if centers.shape[0] else None

    def support_count(self, center: Tuple[float, float], radius: float) -> int:
        if self._tree is None:
            return 0
        return len(self._tree.query_ball_point([center[0], center[1]], radius))
