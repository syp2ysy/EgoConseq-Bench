"""Prefix-conditioned depth reach probes.

These were measurement helpers when the pilot scored the rule; the sampler now
draws from them, so the probe bound, the scan order and the tri-state verdict
are production contracts and are pinned here rather than in the pilot script.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import action_proposal as AP
from pipeline import config
from pipeline import perception
from pipeline import reach_probe
from pipeline import rollout
from pipeline.actions import Forward, Turn
from tests import _synthetic


def _wall_frame(distance_m):
    """One frontal wall and a flat 3.0 m depth everywhere else."""
    frame = _synthetic.make_frame()
    xs = np.linspace(-2.0, 2.0, 400)
    wall = np.stack(
        [xs, np.full_like(xs, 0.15), np.full_like(xs, distance_m)], axis=1)
    object.__setattr__(frame, "vf", perception.VoxelField(wall))
    return frame


# --------------------------------------------------------------------------
# The probe grid
# --------------------------------------------------------------------------

def test_probe_bound_covers_the_publication_vocabulary():
    """A probe is a corridor query, never an action.

    The probe must reach at least the longest publishable leg, or that leg could
    never be certified from the initial view. It is not *derived* from the
    vocabulary: v1 read ``max(GEN_FORWARDS_M)`` directly, which is why a boundary
    at 3.5 m was indistinguishable from open space.
    """
    assert reach_probe.REACH_PROBE_MAX_M >= max(config.GEN_FORWARDS_M)
    grid = reach_probe.probe_grid()
    assert grid[0] == reach_probe.REACH_GRID_STEP_M
    assert grid[-1] == reach_probe.REACH_PROBE_MAX_M
    assert list(grid) == sorted(grid)
    assert all(round(value / reach_probe.REACH_GRID_STEP_M, 9) ==
               int(round(value / reach_probe.REACH_GRID_STEP_M))
               for value in grid)
    assert reach_probe.probe_grid(2.0) == (0.5, 1.0, 1.5, 2.0)


def test_the_longest_publishable_leg_saturates_the_probe():
    """v2 set both numbers to 6.0, so the top leg has no certified headroom.

    Pinned rather than hidden. At the ceiling the probe cannot report a boundary
    beyond the leg, so ``free_distance_m`` returns ``None`` and ``feasible_grid``
    admits 6.0 with no ``BENCH_SAFE_CLEARANCE_M`` reserve -- the same posture v1
    had at 3.0, now at longer range. It is sound because full geometry still
    certifies every published leg, but a shorter vocabulary ceiling than the
    probe bound would be the stronger contract.
    """
    assert max(config.GEN_FORWARDS_M) == reach_probe.REACH_PROBE_MAX_M
    assert AP.feasible_grid(None)[-1] == reach_probe.REACH_PROBE_MAX_M


def test_the_v1_vocabulary_is_a_subset_of_v2():
    """Frozen v1 programs stay inside the published grid with no adapter."""
    assert set(config.GEN_FORWARDS_M_V1) <= set(config.GEN_FORWARDS_M_V2)
    assert config.GEN_FORWARDS_M == config.GEN_FORWARDS_M_V2
    assert max(config.GEN_FORWARDS_M_V1) == 3.0
    assert all(abs(value / 0.5 - round(value / 0.5)) < 1e-9
               for value in config.GEN_FORWARDS_M_V2)


def test_reach_stops_at_the_first_shortfall_instead_of_skipping_a_gap(
        monkeypatch):
    """Coverage is not monotone in distance, so the scan may not resume.

    A probe that skipped an uncovered 1.5 m and certified a covered 3.0 m would
    report a corridor the depth image never saw the middle of.
    """
    seen = []

    def coverage(frame, radius, prefix, distance, *, mode):
        seen.append(distance)
        return 0.0 if abs(distance - 1.5) < 1e-9 else 1.0

    monkeypatch.setattr(reach_probe, "_coverage", coverage)
    reach = reach_probe.prefix_supported_reach_m(object(), 0.2, [])
    assert reach == 1.0
    assert seen == [0.5, 1.0, 1.5]


def test_reach_is_measured_from_where_the_prefix_ends():
    frame = _wall_frame(2.5)
    at_origin = reach_probe.prefix_supported_reach_m(frame, 0.2, [])
    after_a_metre = reach_probe.prefix_supported_reach_m(
        frame, 0.2, [Forward(1.0)])
    assert at_origin == 3.0
    assert after_a_metre == 2.0
    # A heading with no depth behind it supports nothing, whatever the corridor
    # ahead of the initial heading measured.
    assert reach_probe.prefix_supported_reach_m(
        frame, 0.2, [Turn(90.0)]) == 0.0


def test_neither_reach_mode_is_a_silent_default():
    """``leg`` and ``program`` score different corridors; an unnamed mode is a
    bug, not a fallback."""
    frame = _wall_frame(2.5)
    assert reach_probe.MODE_LEG != reach_probe.MODE_PROGRAM
    for mode in (reach_probe.MODE_LEG, reach_probe.MODE_PROGRAM):
        assert reach_probe.prefix_supported_reach_m(
            frame, 0.2, [Forward(1.0)], mode=mode) == 2.0
    with pytest.raises(ValueError):
        reach_probe.prefix_supported_reach_m(
            frame, 0.2, [], mode="whole-programme")


# --------------------------------------------------------------------------
# The tri-state verdict
# --------------------------------------------------------------------------

def test_an_observed_boundary_reports_its_arc():
    state = reach_probe.free_distance_state(_wall_frame(2.5), 0.2, [])
    assert state["state"] == reach_probe.STATE_COLLISION
    assert state["first_contact_arc_m"] == pytest.approx(2.5 - 0.2, abs=0.15)


def test_a_boundary_past_the_evidence_is_not_reported_as_open():
    """The distinction the tri-state exists for.

    ``free_distance_m`` returns ``None`` for both of these, and treating an
    unobserved corridor as an open one is what would let a colliding leg reach
    past its budget toward a boundary nothing saw.
    """
    far = reach_probe.free_distance_state(_wall_frame(8.0), 0.2, [])
    assert far["state"] == reach_probe.STATE_INSUFFICIENT
    assert far["first_contact_arc_m"] is None

    near_open = reach_probe.free_distance_state(
        _wall_frame(8.0), 0.2, [], max_m=2.0)
    assert near_open["state"] == reach_probe.STATE_OPEN
    assert near_open["first_contact_arc_m"] is None


def test_a_contact_seen_through_a_hole_is_not_an_observed_boundary(monkeypatch):
    monkeypatch.setattr(rollout, "corridor_coverage",
                        lambda *args, **kwargs: 0.0)
    state = reach_probe.free_distance_state(_wall_frame(2.5), 0.2, [])
    assert state["state"] == reach_probe.STATE_INSUFFICIENT
    assert state["first_contact_arc_m"] is not None
    assert state["coverage"] == 0.0


# --------------------------------------------------------------------------
# The cap rule the pilot scored and the sampler now draws from
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reach,legs,expected", [
    (5.0, 3, 1.5), (3.5, 2, 1.5), (2.0, 1, 2.0), (5.0, 1, 5.0),
    (99.0, 1, 6.0), (0.3, 1, None), (1.0, 3, None),
])
def test_floor_grid_cap_matches_the_worked_example(reach, legs, expected):
    assert reach_probe.floor_grid_cap(reach, legs) == expected


def test_the_pilot_and_the_sampler_cannot_drift_apart():
    """Same rule, two call sites: a pilot that scored a different cap than the
    collector draws from would be measuring a distribution nobody samples."""
    for reach in (0.0, 0.3, 1.0, 2.0, 3.5, 5.0, 6.0):
        for legs in (1, 2, 3):
            assert reach_probe.floor_grid_cap(reach, legs) == \
                AP.leg_cap_m(reach, legs)
    for legs in (0, -1):
        with pytest.raises(ValueError):
            reach_probe.floor_grid_cap(5.0, legs)
