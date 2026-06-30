"""Tests for d_safe_visible swept-cylinder oracle (Task 8) and
attach_contact_projection O6 substrate (Task 9).

TDD: this file is written before the implementation exists.
"""

import numpy as np
from egoconseq.oracle.voxel import VoxelField
from egoconseq.oracle.sweep import d_safe_visible, attach_contact_projection
from egoconseq import config


def _make_K(W=640, H=480, hfov_deg=79):
    fx = W / (2 * np.tan(np.deg2rad(hfov_deg / 2)))
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=float)


def test_wall_ahead_gives_finite_contact():
    wall = np.array([[x, 0.3, 1.0] for x in np.linspace(-1, 1, 200)])   # wall at z=1.0
    vf = VoxelField(wall, voxel=0.05, dilation=1)
    res = d_safe_visible(vf, radius=0.10, turn_deg=0, d_max=5.0,
                         step=0.02, min_support=1)
    assert 0.7 < res.d_safe < 1.1
    assert res.contact_xy is not None


def test_open_space_returns_dmax():
    vf = VoxelField(np.zeros((0, 3)), voxel=0.05, dilation=0)
    res = d_safe_visible(vf, radius=0.10, turn_deg=0, d_max=5.0, step=0.02, min_support=1)
    assert res.d_safe >= 5.0 and res.contact_xy is None


def test_larger_radius_contacts_no_later():
    # monotonicity sanity at the unit level
    wall = np.array([[x, 0.3, 1.0] for x in np.linspace(-1, 1, 200)])
    vf = VoxelField(wall, voxel=0.05, dilation=1)
    small = d_safe_visible(vf, 0.10, 0, 5.0, 0.02, 1).d_safe
    large = d_safe_visible(vf, 0.40, 0, 5.0, 0.02, 1).d_safe
    assert large <= small + 1e-6


# ---------------------------------------------------------------------------
# Task 9: attach_contact_projection (O6 substrate)
# ---------------------------------------------------------------------------

def test_attach_contact_projection_fills_fields():
    """attach_contact_projection should populate contact_3d and contact_pixel
    when contact_xy is not None."""
    wall = np.array([[x, 0.3, 1.0] for x in np.linspace(-1, 1, 200)])
    vf = VoxelField(wall, voxel=0.05, dilation=1)
    res = d_safe_visible(vf, radius=0.10, turn_deg=0, d_max=5.0,
                         step=0.02, min_support=1)
    assert res.contact_xy is not None
    # Before attach: both None
    assert res.contact_3d is None
    assert res.contact_pixel is None

    K = _make_K()
    floor_y = 0.0
    res2 = attach_contact_projection(res, floor_y=floor_y, K=K)

    assert res2.contact_3d is not None
    assert res2.contact_pixel is not None
    # contact_3d y = floor_y + 0.5
    assert abs(res2.contact_3d[1] - (floor_y + 0.5)) < 1e-9
    # contact_3d x, z match contact_xy
    assert abs(res2.contact_3d[0] - res.contact_xy[0]) < 1e-9
    assert abs(res2.contact_3d[2] - res.contact_xy[1]) < 1e-9


def test_attach_contact_projection_no_contact_stays_none():
    """attach_contact_projection on an open-space result leaves fields as None."""
    vf = VoxelField(np.zeros((0, 3)), voxel=0.05, dilation=0)
    res = d_safe_visible(vf, radius=0.10, turn_deg=0, d_max=5.0, step=0.02, min_support=1)
    assert res.contact_xy is None

    K = _make_K()
    res2 = attach_contact_projection(res, floor_y=0.0, K=K)
    assert res2.contact_3d is None
    assert res2.contact_pixel is None


def test_attach_contact_projection_ahead_projects_near_cx():
    """Contact directly ahead (x=0) should project near u ≈ cx."""
    wall = np.array([[0.0, 0.3, 1.0] for _ in range(50)])  # thin wall, x=0
    vf = VoxelField(wall, voxel=0.05, dilation=1)
    res = d_safe_visible(vf, radius=0.10, turn_deg=0, d_max=5.0,
                         step=0.02, min_support=1)
    assert res.contact_xy is not None

    K = _make_K()
    cx = K[0, 2]
    res2 = attach_contact_projection(res, floor_y=0.0, K=K)

    u, v = res2.contact_pixel
    assert abs(u - cx) < 5.0, f"Contact ahead should project near cx={cx}, got u={u}"
    # contact_3d y = 0.5 m, camera is at 1.5 m  → y_cam = 0.5 - 1.5 = -1.0 (below camera)
    # backproject formula: y = (cy - v)/fy*d  →  v > cy when y_cam < 0
    # So the contact projects into the LOWER half of the image (v > cy).
    cy = K[1, 2]
    assert v > cy, f"Obstacle below camera ({0.5} m < {config.CAMERA_HEIGHT_M} m) should project to lower half, v={v} cy={cy}"
