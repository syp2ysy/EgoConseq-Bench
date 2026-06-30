"""Tests for d_safe_visible swept-cylinder oracle (Task 8).

TDD: this file is written before the implementation exists.
"""

import numpy as np
from egoconseq.oracle.voxel import VoxelField
from egoconseq.oracle.sweep import d_safe_visible


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
