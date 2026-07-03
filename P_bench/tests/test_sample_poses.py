"""Unit tests for egoconseq.pipeline.sample_poses.is_valid_start.

These tests are PURE LOGIC — no Habitat import required.
The pure predicate `is_valid_start` only operates on already-computed floats.
"""

from egoconseq.pipeline.sample_poses import is_valid_start


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------

def test_rejects_low_valid_depth():
    """valid_depth_ratio=0.5 is well below the 0.85 threshold — must reject."""
    ok, why = is_valid_start(valid_depth_ratio=0.5, dist_to_obstacle=2.0, visible_floor_ratio=0.3)
    assert not ok
    assert why  # non-empty reason string


def test_rejects_too_close_to_wall():
    """dist_to_obstacle=0.2 is smaller than MIN_CLEARANCE_M — must reject."""
    ok, why = is_valid_start(valid_depth_ratio=0.95, dist_to_obstacle=0.2, visible_floor_ratio=0.3)
    assert not ok
    assert why


def test_rejects_no_visible_floor():
    """visible_floor_ratio=0.0 is below 0.05 — must reject."""
    ok, why = is_valid_start(valid_depth_ratio=0.95, dist_to_obstacle=2.0, visible_floor_ratio=0.0)
    assert not ok
    assert why


def test_rejects_visible_floor_just_below_threshold():
    """visible_floor_ratio=0.04 is just below 0.05 — must reject."""
    ok, why = is_valid_start(valid_depth_ratio=0.95, dist_to_obstacle=2.0, visible_floor_ratio=0.04)
    assert not ok


def test_rejects_valid_depth_exactly_at_threshold():
    """valid_depth_ratio=0.85 is the threshold boundary — boundary is exclusive (< 0.85 rejects).
    At exactly 0.85 it passes the depth check."""
    ok, why = is_valid_start(valid_depth_ratio=0.85, dist_to_obstacle=2.0, visible_floor_ratio=0.3)
    # 0.85 == threshold, not strictly less, so depth check passes
    assert ok


# ---------------------------------------------------------------------------
# Accept case
# ---------------------------------------------------------------------------

def test_accepts_good_pose():
    """All metrics well above thresholds — must accept."""
    ok, why = is_valid_start(valid_depth_ratio=0.95, dist_to_obstacle=2.0, visible_floor_ratio=0.3)
    assert ok
    assert why == ""  # empty reason when accepted


# ---------------------------------------------------------------------------
# Default visible_floor_ratio (None means "skip floor check")
# ---------------------------------------------------------------------------

def test_accepts_without_floor_ratio():
    """When visible_floor_ratio is not provided (None), floor check is skipped."""
    ok, why = is_valid_start(valid_depth_ratio=0.95, dist_to_obstacle=2.0)
    assert ok


def test_rejects_depth_without_floor_ratio():
    """Floor check skipped but depth still fails."""
    ok, why = is_valid_start(valid_depth_ratio=0.5, dist_to_obstacle=2.0)
    assert not ok
