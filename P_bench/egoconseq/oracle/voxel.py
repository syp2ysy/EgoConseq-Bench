"""Obstacle voxel field: 2-D occupancy grid on the (x, z) ground plane.

Canonical ground frame (project-wide):
    +x = right, +z = forward, +y = up, origin at agent footprint on the floor.

VoxelField works on the **2D ground projection (x, z)** — the y-axis (height)
is ignored here; callers are expected to have already applied floor removal
(e.g. pointcloud.remove_floor) before building a VoxelField.

Usage
-----
    pts = remove_floor(to_agent_ground(backproject(depth, K)), floor_y)
    vf  = VoxelField(pts, voxel=config.VOXEL_SIZE_M, dilation=config.VOXEL_DILATION)
    hits = vf.support_count(center=(0.0, 0.5), radius=0.1)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from egoconseq import config


class VoxelField:
    """2-D occupancy grid (x-z plane) with binary dilation and radius query.

    Parameters
    ----------
    pts : (N, 3) array_like
        Obstacle points in the agent ground frame (x, y, z).
        The y column is ignored; the field works on (x, z).
    voxel : float
        Voxel side length in metres.  Defaults to ``config.VOXEL_SIZE_M``.
    dilation : int
        Number of dilation iterations applied via
        ``scipy.ndimage.binary_dilation``.  Each iteration expands the
        occupied region by one voxel in all 4 cardinal directions (and
        diagonals, using the default 3×3 structuring element).
        Defaults to ``config.VOXEL_DILATION``.
    """

    # One cell of padding added around the bounding box so that edge voxels
    # can be dilated into their neighbours without boundary effects.
    _MARGIN_VOXELS: int = 2

    def __init__(
        self,
        pts: np.ndarray,
        voxel: float = config.VOXEL_SIZE_M,
        dilation: int = config.VOXEL_DILATION,
    ) -> None:
        self._voxel = float(voxel)
        pts = np.asarray(pts, dtype=np.float64)

        # Empty point cloud → no occupied voxels; queries always return 0.
        if pts.shape[0] == 0:
            self._tree: cKDTree | None = None
            return

        # Ground-plane projection: keep x (col 0) and z (col 2).
        xz = pts[:, [0, 2]]   # (N, 2)

        # Build grid origin with a small margin.
        margin = self._MARGIN_VOXELS * voxel
        origin = xz.min(axis=0) - margin          # (2,) world coord of cell (0,0)
        extent = xz.max(axis=0) + margin - origin  # world size of the grid

        # Number of cells along each axis (at least 1).
        n_cells = np.maximum(np.ceil(extent / voxel).astype(int), 1)  # (2,)

        # Voxel indices for every point.
        idx = np.floor((xz - origin) / voxel).astype(int)   # (N, 2)
        # Clamp to grid (should not be needed with margin, but defensive).
        idx = np.clip(idx, 0, n_cells - 1)

        # Build binary occupancy grid shape (nx, nz).
        grid = np.zeros(n_cells, dtype=bool)
        grid[idx[:, 0], idx[:, 1]] = True

        # Dilate.
        if dilation > 0:
            grid = binary_dilation(grid, iterations=dilation)

        # Collect world (x, z) centre of every occupied cell.
        occ_ix, occ_iz = np.nonzero(grid)          # 1-D index arrays
        occ_centers = (
            np.stack([occ_ix, occ_iz], axis=1).astype(np.float64) + 0.5
        ) * voxel + origin   # (M, 2) world coords

        if occ_centers.shape[0] == 0:
            self._tree = None
        else:
            self._tree = cKDTree(occ_centers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def support_count(self, center: tuple[float, float], radius: float) -> int:
        """Count occupied (dilated) voxels whose centre lies within *radius*.

        Parameters
        ----------
        center : (x, z) float tuple
            Query point on the 2-D ground plane.
        radius : float
            Search radius in metres (footprint disk radius).

        Returns
        -------
        int
            Number of occupied voxel centres within the disk.  Always ≥ 0;
            returns 0 when the field is empty or no voxels lie within *radius*.
        """
        if self._tree is None:
            return 0
        hits = self._tree.query_ball_point([center[0], center[1]], radius)
        return len(hits)
