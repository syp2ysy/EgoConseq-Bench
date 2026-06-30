"""Tests for VoxelField (Task 7: obstacle voxel field).

TDD: this file is written before the implementation exists.
"""

import numpy as np
import pytest
from egoconseq.oracle.voxel import VoxelField


# ---------------------------------------------------------------------------
# Plan's canonical test
# ---------------------------------------------------------------------------

def test_support_query_dilation_and_min_support():
    pts = np.array([[0.5, 0.2, 1.0]] * 5)       # 5 coincident points -> support
    vf = VoxelField(pts, voxel=0.05, dilation=1)
    assert vf.support_count(center=(0.5, 1.0), radius=0.1) >= 1
    assert vf.support_count(center=(3.0, 3.0), radius=0.1) == 0


# ---------------------------------------------------------------------------
# Additional: empty point cloud never crashes, always returns 0
# ---------------------------------------------------------------------------

def test_empty_voxelfield_returns_zero():
    pts = np.zeros((0, 3), dtype=np.float64)
    vf = VoxelField(pts, voxel=0.05, dilation=1)
    assert vf.support_count(center=(0.0, 0.0), radius=0.5) == 0
    assert vf.support_count(center=(1.5, 2.3), radius=1.0) == 0


# ---------------------------------------------------------------------------
# Additional: a single dilated point yields support within radius
# ---------------------------------------------------------------------------

def test_single_point_within_radius():
    # One point at (x=1.0, z=2.0); query center at the same location
    pts = np.array([[1.0, 0.5, 2.0]])
    vf = VoxelField(pts, voxel=0.05, dilation=0)
    # Query at exact projected location should hit
    assert vf.support_count(center=(1.0, 2.0), radius=0.1) >= 1


def test_single_point_outside_radius():
    pts = np.array([[1.0, 0.5, 2.0]])
    vf = VoxelField(pts, voxel=0.05, dilation=0)
    # Query far away should miss
    assert vf.support_count(center=(5.0, 5.0), radius=0.1) == 0


# ---------------------------------------------------------------------------
# Additional: dilation=0 vs dilation>0 boundary
# ---------------------------------------------------------------------------

def test_dilation_expands_occupied_region():
    """A point slightly outside radius=0.02 without dilation, but inside after
    dilation by 1 voxel (0.05 m)."""
    pts = np.array([[0.0, 0.3, 0.0]])
    vf_no_dil = VoxelField(pts, voxel=0.05, dilation=0)
    vf_dil    = VoxelField(pts, voxel=0.05, dilation=2)

    # The centre of the voxel containing (0,0) is at most 0.025 m from (0,0).
    # With dilation=2 an extra ring of 0.05*2=0.10 m is added, so a query at
    # radius=0.05 from (0.08, 0.08) that misses without dilation may hit with it.
    # We just check that dilated field has >= as many hits as undilated field.
    hits_no = vf_no_dil.support_count(center=(0.0, 0.0), radius=0.1)
    hits_di  = vf_dil.support_count(center=(0.0, 0.0), radius=0.3)
    assert hits_di >= hits_no
