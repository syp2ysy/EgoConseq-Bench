"""Perception math tests (pure numpy, no Habitat)."""

import numpy as np

from pipeline import config, perception


K = config.intrinsics()


def test_unproject_roundtrip_recovers_pixels():
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    pts_cam, uv = perception.unproject(depth, K)
    pts_g = perception.to_agent_ground(pts_cam)
    # sample a few points, project back
    for i in np.linspace(0, pts_g.shape[0] - 1, 25).astype(int):
        res = perception.project_ground(pts_g[i], K)
        assert res is not None
        u, v = res
        assert abs(u - uv[i, 0]) < 1e-6 and abs(v - uv[i, 1]) < 1e-6


def test_project_ground_behind_camera_returns_none():
    assert perception.project_ground((0.0, 1.5, -1.0), K) is None


def test_estimate_floor_dense_floor():
    floor = np.zeros((500, 3))          # y = 0
    floor[:, 0] = np.random.uniform(-2, 2, 500)
    floor[:, 2] = np.random.uniform(0.5, 4, 500)
    furniture = np.zeros((300, 3))
    furniture[:, 1] = 0.6
    pts = np.vstack([floor, furniture])
    assert abs(perception.estimate_floor_height(pts)) < 0.06


def test_estimate_floor_sparse_returns_zero():
    pts = np.zeros((50, 3)); pts[:, 1] = 0.02
    assert perception.estimate_floor_height(pts) == 0.0


def test_obstacle_mask_band():
    pts = np.array([[0, 0.0, 1], [0, 0.5, 1], [0, 2.0, 1], [0, -0.5, 1]], float)
    m = perception.obstacle_mask(pts, floor_y=0.0)
    assert list(m) == [False, True, False, False]


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
