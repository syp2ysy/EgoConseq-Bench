"""Prefix-conditioned depth reach probes.

The single definition of reach, shared by the sampler and by the measurement
tools that scored it. ``action_proposal`` calls into this module rather than
keeping its own copy so the distribution a pilot measures and the distribution
the collector draws from cannot drift apart.

These are sampling hints, never ground truth. Records store no depth array, so
nothing downstream can recompute a reach; the formal collision label continues
to come from the depth and full-geometry consensus alone, and reach reaches the
artifacts only through the sampler contract digest.

Two things are deliberate.

``REACH_PROBE_MAX_M`` is a sensing horizon, not a publication ceiling. A probe
is a corridor query against one depth image, never an action, and the two
numbers are independent: which distances may be *published* is
:data:`pipeline.config.GEN_FORWARDS_M`, which distances may be *measured* is
this. They were equal while the vocabulary stopped at 3 m, and reading one off
the other made "the boundary is past the longest publishable leg"
indistinguishable from "the boundary is exactly at it".

Reach is read per prefix, not per pose. A template may turn before it walks, and
the free distance after a 45 degree turn is not the free distance ahead of the
initial heading -- a single pose-level number would be wrong for every template
that does not start with a Forward.
"""

from __future__ import annotations

from typing import Optional, Sequence

from pipeline import actions as A, config, rollout
from pipeline.actions import Action, Forward


# Probes are a measurement, not a publishable distance: this bound exists so a
# fully open corridor reports "open" instead of saturating at the longest leg
# the generator vocabulary happens to contain.
REACH_PROBE_MAX_M = 6.0
REACH_GRID_STEP_M = 0.5

# ``free_distance_m`` returns ``None`` for two different worlds today. Naming
# them apart is the whole point of the tri-state: an unobserved boundary must
# not be reported with the same confidence as an observed one.
STATE_COLLISION = "collision_at_arc"
STATE_OPEN = "open_through_probe"
STATE_INSUFFICIENT = "insufficient_depth_evidence"

# ``program`` scores the whole prefix+leg corridor from the pose origin;
# ``leg`` scores only the new leg, read from the prefix endpoint. The formal
# definition is chosen in step 4 from the measured gap between the two, so both
# are implemented and neither is the silent default.
MODE_PROGRAM = "program"
MODE_LEG = "leg"

_EPS = 1e-9


def probe_grid(max_m: float = REACH_PROBE_MAX_M) -> tuple[float, ...]:
    """Ascending 0.5 m probe distances up to *max_m*."""
    count = int(round(float(max_m) / REACH_GRID_STEP_M))
    return tuple(REACH_GRID_STEP_M * step for step in range(1, count + 1))


def _base_pose(prefix: Sequence[Action]) -> tuple[float, float, float]:
    if not prefix:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) for value in A.pose_after(list(prefix)))


def _coverage(frame, radius: float, prefix: Sequence[Action], distance: float,
              *, mode: str) -> float:
    leg = [Forward(float(distance))]
    if mode == MODE_LEG:
        return rollout.corridor_coverage(
            frame, leg, float(radius), base_pose=_base_pose(prefix))
    if mode == MODE_PROGRAM:
        return rollout.corridor_coverage(
            frame, list(prefix) + leg, float(radius))
    raise ValueError(f"unknown reach mode: {mode!r}")


def prefix_supported_reach_m(
        frame, radius: float, prefix: Sequence[Action] = (), *,
        mode: str = MODE_LEG,
        max_m: float = REACH_PROBE_MAX_M,
        coverage_min: float = config.EVIDENCE_COVERAGE_MIN) -> float:
    """Largest grid distance whose corridor is still covered by this depth.

    Scanned from short to long and stopped at the first shortfall on purpose:
    coverage is not guaranteed monotone in distance, and a probe that skipped a
    gap to find a covered far segment would certify a corridor the depth image
    never saw.
    """
    supported = 0.0
    for distance in probe_grid(max_m):
        if _coverage(frame, radius, prefix, distance,
                     mode=mode) + _EPS < float(coverage_min):
            break
        supported = float(distance)
    return supported


def free_distance_state(
        frame, radius: float, prefix: Sequence[Action] = (), *,
        max_m: float = REACH_PROBE_MAX_M,
        coverage_min: float = config.EVIDENCE_COVERAGE_MIN,
        tolerance_m: float = config.ORACLE_CONTACT_TOL_M) -> dict:
    """Classify the corridor ahead of *prefix* into the three named states.

    ``collision_at_arc`` requires the contact arc's own corridor to be covered:
    a boundary seen through a hole in the depth is not an observed boundary, and
    reporting it as one is exactly the confusion this function exists to remove.
    """
    base = _base_pose(prefix)
    verdict = rollout.view_collision_rollout(
        frame, [Forward(float(max_m))], float(radius), base_pose=base)
    if verdict.get("collision") is True:
        arc = float(verdict["first_contact_arc_m"])
        coverage = rollout.corridor_coverage(
            frame, [Forward(arc + float(tolerance_m))], float(radius),
            base_pose=base)
        state = (STATE_COLLISION if coverage + _EPS >= float(coverage_min)
                 else STATE_INSUFFICIENT)
        return {"state": state, "first_contact_arc_m": arc,
                "coverage": float(coverage)}
    coverage = rollout.corridor_coverage(
        frame, [Forward(float(max_m))], float(radius), base_pose=base)
    state = (STATE_OPEN if coverage + _EPS >= float(coverage_min)
             else STATE_INSUFFICIENT)
    return {"state": state, "first_contact_arc_m": None,
            "coverage": float(coverage)}


def floor_grid_cap(reach_m: float, remaining_forward_legs: int) -> Optional[float]:
    """Fair share of *reach_m* for one leg, quantised down to the grid.

    Imported by the pilot to score the cap rule before it is wired in; the
    sampler keeps its own copy of ``floor_grid`` and this stays a measurement.
    """
    legs = int(remaining_forward_legs)
    if legs <= 0:
        raise ValueError("remaining_forward_legs must be positive")
    share = float(reach_m) / legs
    feasible = [g for g in config.GEN_FORWARDS_M if g <= share + _EPS]
    return float(feasible[-1]) if feasible else None
