"""Stratified pose sampling for EgoConseq-Bench.

Architecture
------------
This module has TWO distinct parts with different testability:

1. **Pure-logic predicate `is_valid_start(...)`** (UNIT-TESTABLE, no Habitat)
   Takes already-computed scalar metrics and returns (ok, reason).
   Thresholds:
     - valid_depth_ratio  >= 0.85   (from config: gate threshold × 0.94 ≈ conservative)
     - dist_to_obstacle   >= MIN_CLEARANCE_M  (ensures largest body fits)
     - visible_floor_ratio >= 0.05  (enough floor visible for grounding)

2. **Integration sampler `sample_poses(sim, n, yaws, seed)`** (requires Habitat)
   Uses sim.pathfinder + sim.render(), calls is_valid_start, yields
   accepted (position, yaw) pairs.

MIN_CLEARANCE_M
---------------
Defined as max(config.RADII_M) + 0.1 = 0.40 + 0.1 = 0.50 m.
This guarantees the largest body cylinder (r=0.40) can be placed at the pose
without intersecting a wall, which is required for the counterfactual
(we swap bodies of different radii at the same start pose).
"""

from __future__ import annotations

import math
import random
from typing import Generator, Iterable, Optional, Tuple

import numpy as np

from egoconseq import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum clearance to nearest obstacle.
#: = max(RADII_M) + 0.1 m safety margin.
#: Ensures the LARGEST body radius (0.40 m) fits at the pose —
#: which in turn guarantees every smaller radius also fits (counterfactual).
MIN_CLEARANCE_M: float = max(config.RADII_M) + 0.1  # 0.50 m

#: Valid depth fraction required.  Slightly below GATE_VALID_DEPTH_RATIO (0.9)
#: to give sampling headroom while still filtering out depth-vanished scenes.
_MIN_VALID_DEPTH_RATIO: float = 0.85

#: Minimum fraction of depth pixels that correspond to floor pixels.
_MIN_VISIBLE_FLOOR_RATIO: float = 0.05

# Default yaw angles to evaluate per candidate point (in radians).
_DEFAULT_YAWS: Tuple[float, ...] = tuple(
    math.radians(a) for a in (0, 45, 90, 135, 180, 225, 270, 315)
)


# ---------------------------------------------------------------------------
# Pure-logic predicate (unit-testable, no Habitat)
# ---------------------------------------------------------------------------

def is_valid_start(
    valid_depth_ratio: float,
    dist_to_obstacle: float,
    visible_floor_ratio: Optional[float] = None,
    *,
    min_clearance: float = MIN_CLEARANCE_M,
    min_valid_depth: float = _MIN_VALID_DEPTH_RATIO,
    min_floor_ratio: float = _MIN_VISIBLE_FLOOR_RATIO,
) -> Tuple[bool, str]:
    """Decide whether a candidate (position, yaw) is a valid start pose.

    All inputs are pre-computed scalars — **no Habitat import required**.

    Parameters
    ----------
    valid_depth_ratio : float
        Fraction of depth pixels that are finite and > 0.
        Reject if < min_valid_depth (default 0.85).
    dist_to_obstacle : float
        Distance in metres from the candidate position to the nearest obstacle,
        as returned by pathfinder.distance_to_closest_obstacle(point).
        Reject if < min_clearance (default MIN_CLEARANCE_M = 0.50 m).
    visible_floor_ratio : float | None
        Fraction of depth pixels that project onto the navigable floor plane.
        If None, the floor check is skipped.
        Reject if provided and < min_floor_ratio (default 0.05).
    min_clearance : float
        Override for the clearance threshold (keyword-only).
    min_valid_depth : float
        Override for the valid depth threshold (keyword-only).
    min_floor_ratio : float
        Override for the visible floor threshold (keyword-only).

    Returns
    -------
    (ok, reason) : (bool, str)
        ok=True and reason="" on acceptance.
        ok=False and reason=<human-readable explanation> on rejection.
    """
    if valid_depth_ratio < min_valid_depth:
        return (
            False,
            f"valid_depth_ratio {valid_depth_ratio:.3f} < {min_valid_depth:.3f} "
            "(too many vanishing/invalid depth pixels)",
        )

    if dist_to_obstacle < min_clearance:
        return (
            False,
            f"dist_to_obstacle {dist_to_obstacle:.3f} m < {min_clearance:.3f} m "
            "(too close to wall; largest body would not fit)",
        )

    if visible_floor_ratio is not None and visible_floor_ratio < min_floor_ratio:
        return (
            False,
            f"visible_floor_ratio {visible_floor_ratio:.3f} < {min_floor_ratio:.3f} "
            "(insufficient floor visible for grounding)",
        )

    return (True, "")


# ---------------------------------------------------------------------------
# Integration sampler (requires Habitat — NOT imported at module level)
# ---------------------------------------------------------------------------

def sample_poses(
    sim,  # EgoConseqSim — typed loosely to avoid Habitat import at module level
    n: int,
    yaws: Iterable[float] = _DEFAULT_YAWS,
    seed: Optional[int] = None,
    max_tries: int = 10_000,
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """Sample up to *n* valid (position, yaw) start poses from *sim*.

    A pose is **valid** iff it passes `is_valid_start` for every yaw tested
    at that position (equivalently: the position itself passes the clearance
    check, which covers ALL radii, and each yaw passes depth/floor checks).

    In practice we:
      1. Draw a random navigable point from sim.pathfinder.
      2. Check dist_to_obstacle >= MIN_CLEARANCE_M (position-level, covers all
         body sizes — the counterfactual requirement).
      3. For each candidate yaw, render depth, compute metrics, call
         is_valid_start.
      4. Yield accepted (position, yaw) pairs until *n* poses collected.

    Parameters
    ----------
    sim : EgoConseqSim
        Habitat simulator wrapper exposing .pathfinder and .render().
    n : int
        Number of valid (position, yaw) pairs to collect.
    yaws : iterable of float
        Yaw angles in radians to evaluate per candidate position.
    seed : int | None
        Optional random seed for reproducibility (seeds numpy + Python random).
    max_tries : int
        Hard cap on candidate positions to avoid infinite loops.

    Yields
    ------
    (position, yaw) : (np.ndarray shape (3,), float)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    yaws = list(yaws)
    collected = 0
    tries = 0

    pathfinder = sim.pathfinder

    while collected < n and tries < max_tries:
        tries += 1

        # --- 1. Draw a random navigable point ---
        pos = pathfinder.get_random_navigable_point()

        # --- 2. Position-level clearance check (covers all radii) ---
        dist = pathfinder.distance_to_closest_obstacle(pos)
        if dist < MIN_CLEARANCE_M:
            continue

        # --- 3. Per-yaw depth + floor check ---
        for yaw in yaws:
            _, depth, _, _ = sim.render(pos, yaw)

            # Valid depth ratio: finite pixels with depth > 0
            valid_mask = np.isfinite(depth) & (depth > 0.0)
            vdr = float(valid_mask.mean())

            # Visible floor ratio: pixels whose projected world-Y is near floor
            # (simplified: pixels with depth < 4 m and valid — full floor
            #  segmentation needs semantic; here we use a depth proxy)
            floor_mask = valid_mask & (depth < 4.0)
            vfr = float(floor_mask.mean())

            ok, _reason = is_valid_start(
                valid_depth_ratio=vdr,
                dist_to_obstacle=dist,
                visible_floor_ratio=vfr,
            )
            if ok:
                yield (np.array(pos, dtype=np.float32), float(yaw))
                collected += 1
                if collected >= n:
                    return


# ---------------------------------------------------------------------------
# Quick smoke (optional, guarded)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke the pure predicate only — no Habitat needed.
    cases = [
        dict(valid_depth_ratio=0.5,  dist_to_obstacle=2.0,  visible_floor_ratio=0.3),
        dict(valid_depth_ratio=0.95, dist_to_obstacle=0.2,  visible_floor_ratio=0.3),
        dict(valid_depth_ratio=0.95, dist_to_obstacle=2.0,  visible_floor_ratio=0.0),
        dict(valid_depth_ratio=0.95, dist_to_obstacle=2.0,  visible_floor_ratio=0.3),
    ]
    for kw in cases:
        ok, why = is_valid_start(**kw)
        tag = "ACCEPT" if ok else "REJECT"
        print(f"[{tag}] {kw}  →  {why!r}")
