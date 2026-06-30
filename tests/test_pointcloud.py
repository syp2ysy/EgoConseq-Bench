import numpy as np
from egoconseq.oracle.pointcloud import backproject, to_agent_ground, estimate_floor_height, remove_floor


def test_backproject_center_ray_distance():
    H, W = 4, 4
    depth = np.full((H, W), 2.0, np.float32)
    K = np.array([[100, 0, W/2], [0, 100, H/2], [0, 0, 1]], float)
    pts = backproject(depth, K)                       # (N,3) camera frame, +z forward
    assert np.isclose(pts[:, 2].min(), 2.0, atol=1e-5)  # z == depth


def test_backproject_y_axis_points_up():
    # Camera +y must point UP. Image row v grows DOWN, so a TOP-row pixel
    # (v=0, physically higher) must yield a LARGER camera-frame y than a
    # BOTTOM-row pixel (v=H-1, physically lower).
    H, W = 4, 4
    depth = np.full((H, W), 2.0, np.float32)
    K = np.array([[100, 0, 2], [0, 100, 2], [0, 0, 1]], float)
    # all depths valid -> N == H*W, row-major (v, u) layout preserved
    pts = backproject(depth, K).reshape(H, W, 3)  # index [v, u]
    u = W // 2  # near cx column
    y_top = pts[0, u, 1]         # top row, v=0
    y_bottom = pts[H - 1, u, 1]  # bottom row, v=H-1
    assert y_top > y_bottom


def test_to_agent_ground_level_camera():
    # straight-ahead camera point at depth d -> ground (0, camera_height, d)
    cam = np.array([[0.0, 0.0, 2.0]])
    g = to_agent_ground(cam, camera_height=1.5, pitch=0.0)
    assert np.allclose(g[0], [0.0, 1.5, 2.0], atol=1e-6)
    # a floor point (camera-frame y = -1.5) -> ground y ~ 0
    floorpt = np.array([[0.0, -1.5, 2.0]])
    g2 = to_agent_ground(floorpt, camera_height=1.5, pitch=0.0)
    assert abs(g2[0][1]) < 1e-6


def test_estimate_floor_robust_to_furniture_and_outliers():
    # Dense floor cluster at y~0, an even DENSER furniture surface at ~0.6 m
    # (couch/bed seat), and a handful of sub-floor artifacts at y=-3.0.
    # The tight-window mode must lock onto the floor (~0): the furniture is
    # outside the [-0.25, 0.30] window so it cannot out-vote the floor.
    rng = np.random.default_rng(0)
    floor = np.zeros((500, 3))
    floor[:, 1] = rng.uniform(-0.02, 0.02, 500)
    furniture = np.zeros((2000, 3))
    furniture[:, 1] = rng.uniform(0.55, 0.65, 2000)
    outliers = np.zeros((10, 3))
    outliers[:, 1] = -3.0
    pts = np.vstack([floor, furniture, outliers])
    fh = estimate_floor_height(pts)
    assert abs(fh) < 0.1


def test_estimate_floor_no_floor_falls_back_to_zero():
    # No floor visible — only a dense furniture surface at ~0.6 m. The tight
    # window has < min_support points, so we return the constructional floor
    # level 0.0 rather than chasing the furniture.
    rng = np.random.default_rng(1)
    furniture = np.zeros((2000, 3))
    furniture[:, 1] = rng.uniform(0.55, 0.65, 2000)
    fh = estimate_floor_height(furniture)
    assert fh == 0.0


def test_floor_removed_keeps_wall():
    # GROUND-FRAME points: floor at y≈0, wall points at y in [0.05,1.5]
    floor = np.array([[x * 0.1, 0.0, 1.0] for x in range(20)])
    wall = np.array([[0.0, h, 2.0] for h in np.linspace(0.1, 1.4, 20)])
    pts = np.vstack([floor, wall])
    fh = estimate_floor_height(pts)
    kept = remove_floor(pts, fh, band=(0.05, 1.5))
    assert len(kept) == len(wall)                     # floor gone, wall kept
