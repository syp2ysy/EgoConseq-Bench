"""Perception math tests (pure numpy, no Habitat)."""

import numpy as np

from pipeline import config, perception
from tests._synthetic import LEVEL_FLOOR


K = config.intrinsics()


def test_unproject_roundtrip_recovers_pixels():
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    pts_cam, uv = perception.unproject(depth, K)
    pts_g = perception.to_agent_ground(pts_cam)
    # sample a few points, project back
    for i in np.linspace(0, pts_g.shape[0] - 1, 25).astype(int):
        res = perception.project_ground(pts_g[i], K, camera_height=config.CAMERA_HEIGHT_M)
        assert res is not None
        u, v = res
        assert abs(u - uv[i, 0]) < 1e-6 and abs(v - uv[i, 1]) < 1e-6


def test_project_ground_behind_camera_returns_none():
    assert perception.project_ground((0.0, 1.5, -1.0), K,
                                     camera_height=config.CAMERA_HEIGHT_M) is None


def test_perception_no_longer_offers_a_per_image_floor_estimate():
    # The scalar histogram estimator is gone, not deprecated: while it existed,
    # any consumer could reach for a floor of its own and disagree with the
    # pose's canonical plane. Its replacement is tested in test_pl_floor_plane.
    assert not hasattr(perception, "estimate_floor_height")


def test_obstacle_mask_band():
    pts = np.array([[0, 0.0, 1], [0, 0.15, 1], [0, 0.5, 1], [0, -0.5, 1]], float)
    m = perception.obstacle_mask(pts, LEVEL_FLOOR)
    assert list(m) == [False, True, False, False]


def test_obstacle_mask_band_top_excludes_overhang():
    # Explicit bands remain available for deterministic geometry diagnostics.
    pts = np.array([[0, 0.3, 1], [0, 0.7, 1]], float)
    short = perception.obstacle_mask(pts, LEVEL_FLOOR, band=(0.05, 0.5))
    tall = perception.obstacle_mask(pts, LEVEL_FLOOR, band=(0.05, 1.0))
    assert list(short) == [True, False]
    assert list(tall) == [True, True]


def test_voxel_support_hits_wall_and_misses_far():
    # A wall of points at z=1.0, x in [-0.3, 0.3], y in band.
    xs = np.linspace(-0.3, 0.3, 40)
    pts = np.stack([xs, np.full_like(xs, 0.5), np.full_like(xs, 1.0)], axis=1)
    vf = perception.VoxelField(pts)
    assert vf.support_count((0.0, 1.0), 0.1) >= config.MIN_SUPPORT_VOXELS
    assert vf.support_count((0.0, 3.0), 0.1) == 0


def test_voxel_empty_field():
    vf = perception.VoxelField(np.zeros((0, 3)))
    assert vf.support_count((0.0, 0.0), 1.0) == 0
