"""Sanity gate functions for EgoConseq-Bench (Task 14).

Two pure-logic predicates, no Habitat dependency:

  step_size_stable(labels_a, labels_b, thresh) → bool
      Fraction of element-wise equal entries across two label sequences
      must reach *thresh*.  Labels may be strings (contact/nc) or any
      equality-comparable type (e.g. binned d_safe integers).

  monotonic_in_radius(dsafe_by_radius) → bool
      True iff d_safe values are non-increasing as the body radius grows
      (larger body → smaller or equal safe distance, with a tiny tolerance).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence


# ---------------------------------------------------------------------------
# 1. Step-size stability
# ---------------------------------------------------------------------------

def step_size_stable(
    labels_a: Sequence[Any],
    labels_b: Sequence[Any],
    thresh: float = 0.95,
) -> bool:
    """Check whether two label sequences agree on at least *thresh* fraction.

    Intended use: discretise d_safe values at MARCH_STEP 0.02 m and 0.01 m
    (e.g. contact/no-contact at a threshold H, or bin index at 0.1 m
    boundaries) and verify that the coarser and finer step sizes produce the
    same label ≥ 95% of the time.

    Parameters
    ----------
    labels_a, labels_b : sequence of any equality-comparable elements
        The two label sequences.  Agreement is computed over the common prefix
        (zip semantics) — different lengths are allowed.
    thresh : float
        Minimum required agreement fraction (inclusive).  Default 0.95.

    Returns
    -------
    bool
        True iff (number of matching pairs) / (number of pairs) >= thresh.
        Returns True if the sequences share zero pairs (trivially stable).
    """
    pairs = list(zip(labels_a, labels_b))
    if not pairs:
        return True
    matches = sum(1 for a, b in pairs if a == b)
    return matches / len(pairs) >= thresh


# ---------------------------------------------------------------------------
# 2. Monotonicity in radius
# ---------------------------------------------------------------------------

def monotonic_in_radius(
    dsafe_by_radius: Dict[float, float],
    tol: float = 1e-6,
) -> bool:
    """Check that d_safe is non-increasing as body radius increases.

    Physically: a wider body sees contact sooner (or at the same distance) as
    radius grows.  Violations indicate a mis-calibrated voxel oracle that
    reports MORE clearance for a wider body — a false-safe risk.

    Parameters
    ----------
    dsafe_by_radius : dict {radius (float) → d_safe (float)}
        d_safe values keyed by agent radius (metres).  Keys are sorted
        ascending before checking.
    tol : float
        Upward tolerance: a tiny increase up to *tol* is allowed to
        accommodate floating-point rounding.  Default 1e-6.

    Returns
    -------
    bool
        True iff for all consecutive (r_i, r_j) with r_i < r_j:
            dsafe[r_j] <= dsafe[r_i] + tol
        (i.e. d_safe is allowed to decrease or stay the same, but not to
        increase beyond the tolerance).
    """
    sorted_radii = sorted(dsafe_by_radius.keys())
    for i in range(1, len(sorted_radii)):
        r_prev = sorted_radii[i - 1]
        r_curr = sorted_radii[i]
        if dsafe_by_radius[r_curr] > dsafe_by_radius[r_prev] + tol:
            return False
    return True
