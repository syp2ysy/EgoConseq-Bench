"""Tests for egoconseq.gates.sanity — Task 14 Part A (TDD)."""

from egoconseq.gates.sanity import step_size_stable, monotonic_in_radius


# ---------------------------------------------------------------------------
# step_size_stable
# ---------------------------------------------------------------------------

def test_step_size_stable_true():
    a = ["c", "nc", "c", "c"]
    b = ["c", "nc", "c", "c"]
    assert step_size_stable(a, b, thresh=0.95)


def test_step_size_unstable():
    a = ["c"] * 100
    b = ["nc"] * 10 + ["c"] * 90   # 90% agreement < 0.95
    assert not step_size_stable(a, b, thresh=0.95)


def test_step_size_stable_exact_thresh():
    """Exactly at threshold → True (>= not >)."""
    # 19/20 = 0.95 exactly
    a = ["c"] * 20
    b = ["c"] * 19 + ["nc"] * 1
    assert step_size_stable(a, b, thresh=0.95)


def test_step_size_stable_below_thresh():
    """One below exact threshold → False."""
    # 18/20 = 0.90 < 0.95
    a = ["c"] * 20
    b = ["c"] * 18 + ["nc"] * 2
    assert not step_size_stable(a, b, thresh=0.95)


def test_step_size_stable_floats():
    """Works with numeric labels (binned d_safe)."""
    a = [0, 1, 2, 1, 0]
    b = [0, 1, 2, 1, 0]
    assert step_size_stable(a, b, thresh=1.0)


def test_step_size_stable_mismatched_len():
    """Different-length lists → agreement over common prefix (zip semantics)."""
    a = ["c", "c", "c"]
    b = ["c", "c"]  # only 2 pairs; both match → agreement 1.0
    assert step_size_stable(a, b, thresh=0.95)


# ---------------------------------------------------------------------------
# monotonic_in_radius
# ---------------------------------------------------------------------------

def test_monotonic_true():
    assert monotonic_in_radius({0.10: 2.0, 0.25: 1.5, 0.40: 1.0})


def test_monotonic_false():
    # increasing d_safe as radius grows → violation
    assert not monotonic_in_radius({0.10: 1.0, 0.25: 1.5, 0.40: 2.0})


def test_monotonic_flat():
    """Flat (all equal) → still non-increasing → True."""
    assert monotonic_in_radius({0.10: 1.0, 0.25: 1.0, 0.40: 1.0})


def test_monotonic_tolerance():
    """Tiny upward wiggle within tolerance → True."""
    assert monotonic_in_radius({0.10: 2.0, 0.25: 2.0 + 5e-7, 0.40: 1.5})


def test_monotonic_single():
    """Single entry → trivially True."""
    assert monotonic_in_radius({0.25: 3.0})


def test_monotonic_unsorted_keys():
    """Keys out of order — function sorts by radius key."""
    # 0.40→1.0, 0.10→2.0, 0.25→1.5 out-of-order; after sort: 2.0,1.5,1.0 → True
    assert monotonic_in_radius({0.40: 1.0, 0.10: 2.0, 0.25: 1.5})
