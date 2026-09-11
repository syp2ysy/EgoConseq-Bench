"""Depth-conditioned action proposal: per-pose materialisation of templates.

A template carries structure only -- how many primitives, which are Turns, what
the turn angles are, and which Forward leg is the boundary target. It carries no
distance. Every Forward distance is read off the pose's own initial depth, on the
frozen ``GEN_FORWARDS_M`` grid, so the same template produces different distances
in different scenes. The aim is to weaken the label shortcut a scene-independent
global action bank creates; whether it is actually weakened is a pilot
measurement, not a property this module can claim.

Each template materialises a *pair*: one safe and one colliding program that
share prefix, turns and suffix and differ only at the target leg. The suffix is
materialised along the safe branch, because that is the branch that is actually
executed -- the colliding branch stops at the target leg and its remainder is
nominal only.

This layer never consults navmesh, the physical oracle, consensus, or a
published label. It does use a depth-side collision verdict, which is what the
swept-disc query over the initial depth *is*; that proxy is the point. Final
labels still come from the unchanged full/depth/consensus stack.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional, Sequence, Tuple

from pipeline import (
    action_control_catalog, actions as A, config, reach_probe, rollout,
)
from pipeline.actions import Action, Forward, Turn


# Two frozen contracts that live side by side; neither is derived from the
# other and there is deliberately no adapter between them. Under v1 every
# Forward was read from its own ``free_distance_m`` probe with no cross-leg
# budget, so an L6 program could nominally travel three full grid steps per leg
# off a single depth image. v2 bounds each leg by its own prefix's reach, re-read
# after every Turn, and prefers -- but no longer requires -- an even share of it.
# The v1 constant stays because frozen records were written under it and must
# keep validating against their own field set -- rewriting the constant in place
# would leave those records unverifiable.
PROPOSAL_PROTOCOL_V1 = "depth-conditioned-pair-v1"
PROPOSAL_PROTOCOL_V2 = "depth-conditioned-pair-v2"
PROPOSAL_PROTOCOL_V3 = "depth-conditioned-action-bank-v3"
PROPOSAL_PROTOCOL_V4 = "depth-conditioned-action-bank-v4"
PROPOSAL_PROTOCOL_V5 = "depth-conditioned-action-bank-v5"
PROPOSAL_PROTOCOL_VERSION = PROPOSAL_PROTOCOL_V5

# The two action-sampling policies a run may commit to. They are mutually
# exclusive: a file-driven run performs no depth proposal at all and must not
# borrow the name of one.
# Legacy runs did not persist their policy separately, so this value remains
# the exact inference for those frozen run contracts. New v3 runs write their
# policy explicitly and never reinterpret the old name.
DEPTH_CONDITIONED_POLICY = "depth_conditioned_action_proposal_v1"
DEPTH_CONDITIONED_POLICY_V3 = "depth_conditioned_action_bank_v3"
DEPTH_CONDITIONED_POLICY_V4 = "depth_conditioned_action_bank_v4"
DEPTH_CONDITIONED_POLICY_V5 = "depth_conditioned_action_bank_v5"
EXPLICIT_ACTION_FILE_POLICY = "explicit_action_file_v1"
_POLICY_BY_ACTION_MODE = {
    "balanced": DEPTH_CONDITIONED_POLICY,
    "file": EXPLICIT_ACTION_FILE_POLICY,
}


def expected_policy_for_action_mode(action_mode) -> Optional[str]:
    """Policy a run in *action_mode* must declare, or None if unrecognised.

    Unknown modes are unrecognised rather than silently accepted.
    """
    base = _POLICY_BY_ACTION_MODE.get(str(action_mode or ""))
    return base


def policy_for_new_collection(action_mode) -> Optional[str]:
    """Policy written by a newly started collector.

    ``expected_policy_for_action_mode`` is deliberately the legacy inference
    for metadata that predates an explicit policy field. Keeping the functions
    separate prevents a v3 rollout from silently rewriting old records.
    """
    if str(action_mode or "") == "balanced":
        return DEPTH_CONDITIONED_POLICY_V5
    return expected_policy_for_action_mode(action_mode)


def declared_policy_matches_action_mode(action_mode, declared) -> bool:
    """Return whether an explicit frozen/current policy matches its mode."""
    if str(action_mode or "") == "balanced":
        return declared in {
            DEPTH_CONDITIONED_POLICY_V3,
            DEPTH_CONDITIONED_POLICY_V4,
            DEPTH_CONDITIONED_POLICY_V5,
        }
    return declared == expected_policy_for_action_mode(action_mode)

# The generator vocabulary is already a 0.5 m grid, and is already hashed into
# the collection config digest. Materialising onto it keeps every published
# distance interpretable and keeps the private action match key (0.25 m buckets)
# meaningful.
GRID_M: Tuple[float, ...] = tuple(float(value) for value in config.GEN_FORWARDS_M)
# How far the depth probe looks, which is a sensing property. It read
# ``max(GRID_M)`` while the vocabulary stopped at 3 m, which made a boundary at
# 3.5 m indistinguishable from open space: the probe could not report anything
# the vocabulary could not say. Publication ceilings belong to
# ``GLOBAL_MAX_FORWARD_M`` below, and the two are now free to differ.
PROBE_DISTANCE_M = reach_probe.REACH_PROBE_MAX_M
# The longest leg the vocabulary can express at all. Every pose is bounded well
# below this by its own depth; this only stops a prefix with an unusually long
# corridor from proposing a distance no published record may carry.
GLOBAL_MAX_FORWARD_M = max(GRID_M)

# Two independent constraints bound how far past the depth boundary a colliding
# target leg must reach. See ``local_guard_m``.
_PUBLICATION_GUARD_M = (
    config.BENCH_COLLISION_REMAINING_M + config.ORACLE_CONTACT_TOL_M)
_A2_GUARD_M = config.A2_ACTION_BOUNDARY_MARGIN_M + config.ORACLE_CONTACT_TOL_M

# Only the suffix cost the target position makes unavoidable is allowed on top
# of the target leg's own overshoot budget. Without this an early target leg
# followed by two long safe legs would recreate the multi-metre nominal
# overshoot this sampler exists to remove.
_REMAINING_BUDGET_BASE_M = 1.30
_REMAINING_BUDGET_PER_SUFFIX_LEG_M = 0.50

# Per-pose budget. Kept here rather than in ``config`` on purpose: these are
# sampler parameters, already bound into ``action_sampler_contract_sha256``, and
# adding them to the config digest would invalidate every resumable collection
# and make a same-input comparison against the previous sampler impossible.
PAIRS_PER_LENGTH_DEFAULT = 12
NATURAL_PER_LENGTH_DEFAULT = 40
NATURAL_DYNAMIC_VARIANT = "natural_dynamic"
A1_CONTROL_VARIANT = "a1_control"
ACTION_REFRESH_VARIANT = "action_refresh"
A2_RANK_COLLISION_VARIANT = "a2_rank_collision"

_PROVENANCE_FIELDS_V1 = (
    "protocol", "template_id", "target_forward_leg_number",
    "target_action_index", "a2_capable", "variant", "b_j_m",
    "suffix_forward_m", "local_guard_m", "remaining_proxy_m",
    "safe_proxy_gap_m", "proxy_coverage",
)

# v2 publishes the same authoritative fields. Reach and the per-leg cap are
# sampling hints derived from a depth array records do not store, so no
# validator could ever recompute them; carrying them here would add provenance
# nothing is able to check. They stay in run-level diagnostics instead.
_PROVENANCE_FIELDS_V2 = _PROVENANCE_FIELDS_V1
_PROVENANCE_FIELDS_V3 = _PROVENANCE_FIELDS_V2
_PROVENANCE_FIELDS_V4 = _PROVENANCE_FIELDS_V3
_PROVENANCE_FIELDS_V5 = _PROVENANCE_FIELDS_V4

PROVENANCE_FIELDS_BY_PROTOCOL = {
    PROPOSAL_PROTOCOL_V1: _PROVENANCE_FIELDS_V1,
    PROPOSAL_PROTOCOL_V2: _PROVENANCE_FIELDS_V2,
    PROPOSAL_PROTOCOL_V3: _PROVENANCE_FIELDS_V3,
    PROPOSAL_PROTOCOL_V4: _PROVENANCE_FIELDS_V4,
    PROPOSAL_PROTOCOL_V5: _PROVENANCE_FIELDS_V5,
}
PROVENANCE_FIELDS = PROVENANCE_FIELDS_BY_PROTOCOL[PROPOSAL_PROTOCOL_VERSION]

_EPS = 1e-9

# Length 1 admits exactly one structure, so a saturated slot must terminate the
# draw rather than spin against a key space it has already exhausted.
_DRAW_RETRY_LIMIT = config.ACTION_REJECTION_FACTOR


# --------------------------------------------------------------------------
# Grid quantisation
# --------------------------------------------------------------------------

def floor_grid(value: float) -> Optional[float]:
    """Largest grid distance not exceeding *value*."""
    feasible = [g for g in GRID_M if g <= float(value) + _EPS]
    return feasible[-1] if feasible else None


def ceil_grid(value: float) -> Optional[float]:
    """Smallest grid distance at least *value*."""
    feasible = [g for g in GRID_M if g >= float(value) - _EPS]
    return feasible[0] if feasible else None


def feasible_grid(free_distance_m: Optional[float]) -> Tuple[float, ...]:
    """Grid distances that keep the body clear of the measured boundary.

    ``None`` means the probe reached its full ``PROBE_DISTANCE_M`` without
    contact, so the boundary is beyond the grid and no point is excluded.
    Substituting the probe distance instead would apply the clearance reserve to
    a boundary that was never observed and cap the leg one grid step short.
    """
    if free_distance_m is None:
        return GRID_M
    limit = float(free_distance_m) - config.BENCH_SAFE_CLEARANCE_M
    return tuple(g for g in GRID_M if g <= limit + _EPS)


def pick_distance(free_distance_m: Optional[float], rank: int, *,
                  cap_m: Optional[float] = None,
                  prefer_max_m: Optional[float] = None) -> Optional[float]:
    """Choose one feasible distance by deterministic rank, longest first.

    Ranking over the whole feasible set rather than a fixed set of backoffs keeps
    legs off the clearance boundary; a bank whose every leg sits at the minimum
    reserve fails the navmesh clearance gate in bulk.

    *cap_m* filters the feasible set rather than replacing it, which is what
    makes ``local_cap`` structurally incapable of admitting a distance the
    boundary already excluded: the cap can only ever tighten.

    *prefer_max_m* reorders rather than filters. It is the fair share of the
    prefix's reach, and a template whose one long leg the depth genuinely
    supports is not worth dropping over an average -- so it narrows the draw
    only while something under it remains, and steps aside otherwise.
    """
    feasible = feasible_grid(free_distance_m)
    if cap_m is not None:
        feasible = tuple(g for g in feasible if g <= float(cap_m) + _EPS)
    if not feasible:
        return None
    if prefer_max_m is not None:
        preferred = tuple(
            g for g in feasible if g <= float(prefer_max_m) + _EPS)
        if preferred:
            feasible = preferred
    return feasible[-1 - (int(rank) % len(feasible))]


def local_cap_m(reach_m: float) -> Optional[float]:
    """Longest publishable leg this prefix's own depth evidence supports.

    A hard bound: past it the leg would be certified against geometry the
    initial view never covered. ``None`` means this prefix cannot support even
    the shortest grid step, so the template is dropped -- not the pose.

    Reach is re-read from each prefix, so a template that turns away from the
    wall it was about to hit is measured against the new heading's corridor
    rather than the old one's remaining budget.
    """
    return floor_grid(min(float(reach_m), GLOBAL_MAX_FORWARD_M))


def leg_cap_m(reach_m: float, remaining_forward_legs: int) -> Optional[float]:
    """One Forward leg's fair share of the prefix's reach, on the grid.

    A *preference*, not a bound -- see ``pick_distance``. Spreading the reach
    evenly stops an L6 program from spending all of it on the first leg, but a
    leg that exceeds the average is not thereby unsupported: on a straight run
    the prefix's own reach shortens as the body advances, and after a turn the
    new corridor is read in full. Deleting such a template measured the average
    rather than the scene. The hard bound is ``local_cap_m``.
    """
    legs = int(remaining_forward_legs)
    if legs <= 0:
        raise ValueError("remaining_forward_legs must be positive")
    return floor_grid(float(reach_m) / legs)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def local_guard_m(suffix_forward_m: float, a2_capable: bool) -> float:
    """How far past the depth boundary the colliding target leg must reach.

    The navmesh may see contact up to ``ORACLE_CONTACT_TOL_M`` later than depth,
    which is exactly what consensus tolerates. Two constraints follow:

    * publication margin compares the *whole* nominal forward sum against the
      navmesh contact arc, so an already-long suffix pays part of the bill;
    * A2's action index is only stable when the contact sits at least
      ``A2_ACTION_BOUNDARY_MARGIN_M`` from both ends of its leg, which the suffix
      cannot pay for. Programs with a single Forward have no A2 question to keep
      stable, so that term is dropped rather than costing them samples.
    """
    publication = _PUBLICATION_GUARD_M - float(suffix_forward_m)
    return max(publication, _A2_GUARD_M if a2_capable else 0.0, 0.0)


def remaining_budget_cap_m(suffix_leg_count: int) -> float:
    """Upper bound on nominal action left unexecuted after first contact."""
    return (_REMAINING_BUDGET_BASE_M +
            _REMAINING_BUDGET_PER_SUFFIX_LEG_M * int(suffix_leg_count))


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

def action_pattern(length: int, starts_with: str) -> Tuple[str, ...]:
    """Primitive kinds of an alternating program of *length* primitives."""
    if starts_with not in {"turn", "forward"}:
        raise ValueError(f"unknown template start {starts_with!r}")
    order = ("turn", "forward") if starts_with == "turn" else ("forward", "turn")
    return tuple(order[index % 2] for index in range(int(length)))


@dataclass(frozen=True)
class Template:
    """Structure without distance. Distances come from the pose's depth."""

    template_id: str
    length: int
    starts_with: str
    turn_angles: Tuple[float, ...]
    target_forward_leg_number: int
    distance_ranks: Tuple[int, ...]

    def __post_init__(self) -> None:
        pattern = self.action_pattern
        turns = sum(1 for kind in pattern if kind == "turn")
        if len(self.turn_angles) != turns:
            raise ValueError("template turn angles do not match its structure")
        if not 1 <= int(self.target_forward_leg_number) <= self.forward_count:
            raise ValueError("template target leg is outside its Forward legs")
        if len(self.distance_ranks) != self.forward_count - 1:
            raise ValueError(
                "template needs one distance rank per non-target Forward")

    @property
    def action_pattern(self) -> Tuple[str, ...]:
        return action_pattern(self.length, self.starts_with)

    @property
    def forward_action_indices(self) -> Tuple[int, ...]:
        return tuple(index for index, kind in enumerate(self.action_pattern)
                     if kind == "forward")

    @property
    def forward_count(self) -> int:
        return len(self.forward_action_indices)

    @property
    def target_action_index(self) -> int:
        """Index into ``actions``, which is not the Forward leg number.

        ``actions.contact_action_index`` and ``consensus._arc_action_index``
        both speak primitive indices; conflating the two silently mislabels
        every template whose program starts with a Turn.
        """
        return self.forward_action_indices[
            int(self.target_forward_leg_number) - 1]

    @property
    def a2_capable(self) -> bool:
        """A single-Forward program's A2 answer is forced, so it is not a question."""
        return self.forward_count >= 2


def build_template_bank(rng, *, lengths=tuple(config.GEN_LENGTHS),
                        per_length: int = config.MAIN_ACTION_PROPOSAL_PER_LENGTH,
                        turns=tuple(config.GEN_TURNS_DEG),
                        initial_turns=tuple(config.INITIAL_TURNS_DEG)) -> list:
    """Draw the run-level structural vocabulary, once, with no distances.

    The target leg is cycled rather than drawn so every Forward position gets
    attempted equally often. How often each position *succeeds* is a property of
    what one depth image can see and is reported, not assumed.
    """
    bank = []
    for length in lengths:
        starts = [start for start in ("forward", "turn")
                  if _forward_count(length, start)]
        seen = set()
        produced = []
        targets_drawn = {start: 0 for start in starts}
        stale = 0
        while len(produced) < int(per_length) and stale < _DRAW_RETRY_LIMIT:
            # Both the start form and the target leg advance only on
            # acceptance, so a rejected draw retries the same slot and the
            # quotas stay balanced however often the draw collides.
            start = starts[len(produced) % len(starts)]
            forwards = _forward_count(length, start)
            target = targets_drawn[start] % forwards + 1
            turn_count = length - forwards
            angles = tuple(
                float(rng.choice(
                    initial_turns if start == "turn" and index == 0
                    else turns))
                for index in range(turn_count))
            ranks = tuple(
                int(rng.integers(0, len(GRID_M)))
                for _ in range(forwards - 1))
            key = (start, angles, target, ranks)
            if key in seen:
                stale += 1
                continue
            seen.add(key)
            stale = 0
            targets_drawn[start] += 1
            produced.append((start, angles, target, ranks))
        for index, (start, angles, target, ranks) in enumerate(produced):
            bank.append(Template(
                template_id=f"T{int(length)}-{index:02d}",
                length=int(length), starts_with=start, turn_angles=angles,
                target_forward_leg_number=target, distance_ranks=ranks))
    return bank


def _forward_count(length: int, starts_with: str) -> int:
    return sum(1 for kind in action_pattern(length, starts_with)
               if kind == "forward")


def action_sampler_contract_sha256(templates: Iterable[Template],
                                   natural_candidates: Iterable[tuple], *,
                                   pairs_per_length: int,
                                   natural_per_length: int,
                                   ordinary_actions_per_pose: int =
                                   config.ACTION_CANDIDATE_ORDINARY_PER_POSE,
                                   ) -> str:
    """Bind the proposal rule, its control vocabulary, and the depth oracle.

    A rule digest alone would stay constant while a changed march step or
    support threshold silently moved every boundary the sampler measures, and
    with it the whole distribution. Controls are up to half of each pose's bank,
    so a contract blind to them would let that half change unrecorded.
    """
    payload = {
        "natural_control_bank": [
            {"tag": str(tag),
             "actions": A.actions_to_dicts(list(actions))}
            for tag, actions in natural_candidates
        ],
        "protocol": PROPOSAL_PROTOCOL_VERSION,
        "pose_action_allocation": {
            "ordinary_attempts_per_pose": (
                int(ordinary_actions_per_pose) +
                config.C1_NEIGHBOR_SLOTS_PER_POSE),
            "ordinary_actions_per_pose": (
                int(ordinary_actions_per_pose)),
            "c1_queries_per_pose": config.C1_QUERIES_PER_POSE,
            "c1_neighbors_per_query": config.C1_NEIGHBORS_PER_QUERY,
            "retention": "progressive-stability-fill.v1",
            "c1_reservation": "certified-clear-query-families.v1",
        },
        "dynamic_natural": {
            "variant": NATURAL_DYNAMIC_VARIANT,
            "seed_namespace": "natural-dynamic-v1",
            "pose_budget": "initial_prefix_supported_reach_m",
            "per_leg_cap": "floor_grid(pose_budget/forward_count)",
            "distance_strata": dict(
                config.NATURAL_DYNAMIC_STRATUM_WEIGHTS),
            "post_draw_screen": "proxy_natural_coverage",
        },
        "a1_control": {
            "variant": A1_CONTROL_VARIANT,
            "catalog_contract": action_control_catalog.CONTRACT,
            "catalog_sha256": action_control_catalog.CATALOG_SHA256,
            "anchors_per_pose": action_control_catalog.ANCHORS_PER_POSE,
            "rotation": "five-consecutive-over-thirteen.v1",
            "post_draw_screen": "proxy_natural_coverage",
        },
        "grid_m": list(GRID_M),
        "probe_distance_m": PROBE_DISTANCE_M,
        # v2 only. The per-leg cap is a sampling rule, not a published field,
        # so the contract digest is the only place that can record which reach
        # definition drew the bank -- a moved probe bound or coverage floor
        # samples a different distribution from an unchanged template set.
        "reach_probe_max_m": reach_probe.REACH_PROBE_MAX_M,
        "reach_grid_step_m": reach_probe.REACH_GRID_STEP_M,
        "reach_coverage_min": config.EVIDENCE_COVERAGE_MIN,
        "reach_mode": reach_probe.MODE_LEG,
        # Which arms the budget above binds. Extending it from the paired arm to
        # the control arm changed the sampled distribution without changing a
        # single number, so a digest built only from parameter values would hash
        # a budgeted control bank identically to an unbudgeted one.
        "prefix_budget_arms": ["paired", "natural_dynamic", "a1_control"],
        "proposal_fov_screen": "swept-disc-near-field.v1",
        "global_max_forward_m": GLOBAL_MAX_FORWARD_M,
        # The colliding target leg is admitted by an observed boundary rather
        # than by an overshoot allowance on top of the reach cap, so the digest
        # records the state it demands instead of a metre value.
        "collision_boundary_state": reach_probe.STATE_COLLISION,
        "publication_guard_m": _PUBLICATION_GUARD_M,
        "a2_guard_m": _A2_GUARD_M,
        "remaining_budget_base_m": _REMAINING_BUDGET_BASE_M,
        "remaining_budget_per_suffix_leg_m": (
            _REMAINING_BUDGET_PER_SUFFIX_LEG_M),
        "safe_clearance_m": config.BENCH_SAFE_CLEARANCE_M,
        "pairs_per_length": int(pairs_per_length),
        "natural_per_length": int(natural_per_length),
        "oracle": {
            "contract_version": _oracle_contract_version(),
            "evidence_protocol": rollout.EVIDENCE_PROTOCOL_VERSION,
            "march_step_m": config.MARCH_STEP_M,
            "min_support_voxels": config.MIN_SUPPORT_VOXELS,
            "voxel_size_m": config.VOXEL_SIZE_M,
            "voxel_dilation": config.VOXEL_DILATION,
            "ground_obstacle_band_m": list(config.GROUND_OBSTACLE_BAND_M),
            "coverage_min": config.EVIDENCE_COVERAGE_MIN,
            "corridor_lateral_samples": (
                config.EVIDENCE_CORRIDOR_LATERAL_SAMPLES),
        },
        "templates": [
            {
                "template_id": template.template_id,
                "length": int(template.length),
                "starts_with": template.starts_with,
                "turn_angles": [float(a) for a in template.turn_angles],
                "target_forward_leg_number": int(
                    template.target_forward_leg_number),
                "distance_ranks": [int(r) for r in template.distance_ranks],
            }
            for template in templates
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _oracle_contract_version() -> str:
    # Imported lazily so the proposal layer keeps a geometry-only import graph.
    from pipeline import record as REC
    return REC.ORACLE_CONTRACT_VERSION


# --------------------------------------------------------------------------
# Depth proxy
# --------------------------------------------------------------------------

class FrameDepthProxy:
    """Swept-disc and corridor queries against one pose's initial depth.

    Reuses the same primitives the formal depth authority uses, so the proposal
    prior cannot drift away from the frozen oracle contract.
    """

    def __init__(self, frame, radius: float) -> None:
        self._frame = frame
        self._radius = float(radius)
        # Templates share prefixes heavily -- every one of them starts from the
        # pose origin -- and a reach scan costs a dozen corridor queries. The
        # answer depends only on (frame, radius, base pose), all fixed here, so
        # memoising it is exact rather than an approximation.
        self._reach_by_pose: dict = {}
        self._state_by_pose: dict = {}
        self._free_by_pose: dict = {}

    @property
    def radius_m(self) -> float:
        return self._radius

    @staticmethod
    def _pose_key(base_pose) -> tuple:
        return tuple(round(float(value), 9) for value in base_pose)

    def prefix_supported_reach_m(self, prefix: Sequence[Action]) -> float:
        """Longest grid Forward off *prefix* still covered by this depth."""
        key = self._pose_key(A.pose_after(list(prefix)))
        if key not in self._reach_by_pose:
            self._reach_by_pose[key] = reach_probe.prefix_supported_reach_m(
                self._frame, self._radius, list(prefix),
                mode=reach_probe.MODE_LEG)
        return self._reach_by_pose[key]

    def free_distance_state(self, prefix: Sequence[Action]) -> dict:
        """Tri-state corridor verdict ahead of *prefix*.

        ``free_distance_m`` collapses "probed to the end without contact" and
        "the depth image never saw out there" into one ``None``. Telling them
        apart is what lets the colliding target leg reach past its fair share
        only when the boundary it is aiming at was actually observed.
        """
        key = self._pose_key(A.pose_after(list(prefix)))
        if key not in self._state_by_pose:
            self._state_by_pose[key] = reach_probe.free_distance_state(
                self._frame, self._radius, list(prefix))
        return self._state_by_pose[key]

    def free_distance_m(self, base_pose) -> Optional[float]:
        """Longitudinal free distance ahead of *base_pose*, or None past reach."""
        pose = tuple(float(value) for value in base_pose)
        key = self._pose_key(pose)
        if key not in self._free_by_pose:
            verdict = rollout.view_collision_rollout(
                self._frame, [Forward(PROBE_DISTANCE_M)], self._radius,
                base_pose=pose)
            self._free_by_pose[key] = (
                float(verdict["first_contact_arc_m"])
                if verdict.get("collision") is True else None)
        return self._free_by_pose[key]

    def rollout(self, actions: Sequence[Action]) -> dict:
        return rollout.view_collision_rollout(
            self._frame, list(actions), self._radius)

    def coverage(self, actions: Sequence[Action], max_arc_m=None) -> float:
        return rollout.corridor_coverage(
            self._frame, list(actions), self._radius, max_arc_m=max_arc_m)


# --------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Materialized:
    actions: Tuple[Action, ...]
    variant: str
    provenance: dict


@dataclass(frozen=True)
class ProposalPair:
    template_id: str
    safe: Materialized
    collision: Materialized


def _tally(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + int(amount)


def _count(rejections, reason):
    if rejections is not None:
        rejections[reason] = rejections.get(reason, 0) + 1
    return None


def _segment_forward_count(pattern) -> int:
    return sum(1 for kind in pattern if kind == "forward")


def _materialize_segment(pattern, prior, proxy, turns, ranks, *,
                         remaining_forward_legs: int):
    """Append one alternating segment, reading every Forward off the depth.

    *remaining_forward_legs* counts every Forward the whole template still owes,
    including the target leg and the ones this segment has not reached yet, so a
    prefix prefers not to spend the depth evidence its own suffix will need.
    That is a preference; the bound is each prefix's own reach, re-read after
    every Turn. The rank stream is drawn once per Forward whatever the outcome:
    a cap that skipped the draw would desynchronise every later leg of the same
    template and make the bank depend on which legs happened to be capped.
    """
    produced = []
    remaining = int(remaining_forward_legs)
    for kind in pattern:
        if kind == "turn":
            produced.append(Turn(float(next(turns))))
            continue
        prefix = list(prior) + produced
        rank = next(ranks)
        reach = proxy.prefix_supported_reach_m(prefix)
        cap = local_cap_m(reach)
        if cap is None:
            return None
        distance = pick_distance(
            proxy.free_distance_m(A.pose_after(prefix)), rank, cap_m=cap,
            prefer_max_m=leg_cap_m(reach, remaining))
        if distance is None:
            return None
        produced.append(Forward(distance))
        remaining -= 1
    return produced


def materialize_pair(template: Template, proxy, *, half_fov_deg: float,
                     rejections=None) -> Optional[ProposalPair]:
    """Materialise one template against one pose, or explain why it cannot be.

    Ordering is strictly ``b_j -> m_safe -> suffix -> S_j -> guard -> m_collision``
    and there is no backtracking: a template that cannot be placed is dropped and
    counted, never repaired by relaxing a bound.
    """
    pattern = template.action_pattern
    target_index = template.target_action_index
    turns = iter(template.turn_angles)
    ranks = iter(template.distance_ranks)

    suffix_pattern = pattern[target_index + 1:]
    suffix_leg_budget = _segment_forward_count(suffix_pattern)
    # The target leg is owed by every segment, so the prefix is charged for it
    # and for the suffix as well as for its own legs.
    target_leg_budget = 1 + suffix_leg_budget

    prefix = _materialize_segment(
        pattern[:target_index], [], proxy, turns, ranks,
        remaining_forward_legs=(
            _segment_forward_count(pattern[:target_index]) +
            target_leg_budget))
    if prefix is None:
        return _count(rejections, "template_infeasible_prefix")

    boundary = proxy.free_distance_m(A.pose_after(prefix))
    if boundary is None:
        # Not "the space ahead is open" -- one depth image simply has no voxels
        # out there, so no collision variant can be certified from it.
        return _count(rejections, "bracket_unbounded")
    # Safe-grid first, deliberately. A pair needs both sides, so a boundary too
    # near to leave room for a safe variant dies here whatever A2 would say.
    # Testing A2 first would attribute those templates to the A2 band in the
    # funnel and hide the constraint that actually bound them.
    safe_target = floor_grid(boundary - config.BENCH_SAFE_CLEARANCE_M)
    if safe_target is None:
        return _count(rejections, "bracket_grid_exhausted_safe")
    target_cap = local_cap_m(proxy.prefix_supported_reach_m(prefix))
    if target_cap is None:
        # Not "this template asked for too much" -- the prefix's own depth
        # supports no publishable leg at all, so no target distance exists.
        return _count(rejections, "prefix_reach_supports_no_leg")
    if safe_target > target_cap + _EPS:
        safe_target = target_cap

    if template.a2_capable and boundary < _A2_GUARD_M - _EPS:
        # Contact would land inside A2's near-end boundary band. Single-Forward
        # programs skip this: they have no A2 index to destabilise and should
        # keep contributing A1/A3/B samples. Under the frozen constants the
        # safe-grid floor above already implies this bound, so the counter stays
        # zero until the grid gains a finer first step.
        return _count(rejections, "a2_near_boundary_infeasible")

    if (proxy.free_distance_state(prefix).get("state") !=
            reach_probe.STATE_COLLISION):
        return _count(rejections, "collision_boundary_unobserved")

    safe_head = list(prefix) + [Forward(safe_target)]
    suffix = _materialize_segment(
        suffix_pattern, safe_head, proxy, turns, ranks,
        remaining_forward_legs=suffix_leg_budget or 1)
    if suffix is None:
        return _count(rejections, "template_infeasible_suffix")

    suffix_forward_m = A.total_forward_m(suffix)
    suffix_leg_count = sum(1 for value in suffix if isinstance(value, Forward))
    guard = local_guard_m(suffix_forward_m, template.a2_capable)
    collision_target = ceil_grid(boundary + guard)
    if collision_target is None:
        return _count(rejections, "bracket_grid_exhausted_collision")
    # The colliding leg exists to cross the boundary reach itself stops at, so
    # holding it to that reach could never be satisfied -- the previous rule
    # bought its way out with a one-grid-step allowance, which measured the
    # allowance rather than the scene. What the leg must not do is aim at a hole
    # in the depth image. ``STATE_COLLISION`` is exactly that guarantee: the
    # corridor up to contact plus tolerance is covered, and the executed part of
    # this leg terminates at contact. The nominal remainder past it stays bounded
    # by the remaining-budget cap below. Either way this drops the template,
    # never the pose.
    remaining = (collision_target - boundary) + suffix_forward_m
    if remaining > remaining_budget_cap_m(suffix_leg_count) + _EPS:
        return _count(rejections, "template_remaining_budget_exhausted")

    safe_actions = safe_head + suffix
    collision_actions = list(prefix) + [Forward(collision_target)] + suffix

    safe_coverage = _verify_safe(
        safe_actions, proxy, half_fov_deg=half_fov_deg, rejections=rejections)
    if safe_coverage is None:
        return None
    collision = _verify_collision(
        collision_actions, proxy, half_fov_deg=half_fov_deg,
        target_action_index=target_index, rejections=rejections)
    if collision is None:
        return None

    def provenance(variant, coverage):
        return {
            "protocol": PROPOSAL_PROTOCOL_VERSION,
            "template_id": template.template_id,
            "target_forward_leg_number": int(
                template.target_forward_leg_number),
            "target_action_index": int(target_index),
            "a2_capable": bool(template.a2_capable),
            "variant": variant,
            "b_j_m": float(boundary),
            "suffix_forward_m": float(suffix_forward_m),
            "local_guard_m": float(guard),
            "remaining_proxy_m": float(remaining),
            # Longitudinal only. The publication gate reads navmesh clearance,
            # which is omnidirectional along the whole path, so this cannot be
            # called the safe clearance.
            "safe_proxy_gap_m": float(boundary - safe_target),
            "proxy_coverage": float(coverage),
        }

    return ProposalPair(
        template_id=template.template_id,
        safe=Materialized(
            actions=tuple(safe_actions), variant="safe",
            provenance=provenance("safe", safe_coverage)),
        collision=Materialized(
            actions=tuple(collision_actions), variant="collision",
            provenance=provenance("collision", collision)),
    )


# --------------------------------------------------------------------------
# Per-pose bank
# --------------------------------------------------------------------------



@dataclass(frozen=True)
class Candidate:
    """One materialised program offered to the formal oracle."""

    tag: str
    length: int
    actions: Tuple[Action, ...]
    variant: str
    provenance: dict

    @property
    def order_key(self) -> tuple:
        return (int(self.length), str(self.provenance["template_id"]),
                str(self.variant))


# Re-exported so the tag digest has exactly one definition, in the module that
# owns action canonicalisation.
canonical_actions_sha256 = A.canonical_actions_sha256


def candidate_tag(actions: Sequence[Action]) -> str:
    """An opaque, content-addressed group id.

    This value reaches ``action_group_id``, then ``outcome_id``, then the
    published QA's ``oracle_ref``. A tag spelling out ``safe`` / ``collision``
    or naming its template would hand the answer to anyone reading the artifact,
    so both stay in the private provenance and never in the identifier.
    """
    return (f"L{len([a for a in actions])}-"
            f"a{canonical_actions_sha256(actions)[:12]}")


def _natural_stratum_targets(target: int) -> dict[str, int]:
    weights = dict(config.NATURAL_DYNAMIC_STRATUM_WEIGHTS)
    if tuple(weights) != ("short", "mid", "near") or any(
            not isinstance(value, int) or value <= 0
            for value in weights.values()):
        raise ValueError("natural distance-stratum weights are invalid")
    total_weight = sum(weights.values())
    counts = {
        name: int(target) * weight // total_weight
        for name, weight in weights.items()
    }
    remainder = int(target) - sum(counts.values())
    for name in ("near", "mid", "short"):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1
    return counts


def _natural_stratum_grid(values, stratum: str) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid:
        return ()
    if len(grid) == 1:
        return grid if stratum == "near" else ()
    if len(grid) == 2:
        if stratum == "short":
            return grid[:1]
        if stratum == "mid":
            return ()
        if stratum == "near":
            return grid[1:]
        raise ValueError(f"unknown natural distance stratum: {stratum}")
    first = max(1, (len(grid) + 2) // 3)
    second = max(first + 1, (2 * len(grid) + 2) // 3)
    second = min(second, len(grid))
    if stratum == "short":
        return grid[:first]
    if stratum == "mid":
        return grid[first:second]
    if stratum == "near":
        return grid[second:]
    raise ValueError(f"unknown natural distance stratum: {stratum}")


def _natural_start_space(
        length: int, start: str, feasible, stratum: str) -> tuple:
    pattern = action_pattern(length, start)
    forwards = _natural_stratum_grid(feasible, stratum)
    if not forwards:
        return ()
    domains = [
        forwards if kind == "forward" else (
            config.INITIAL_TURNS_DEG
            if index == 0 else config.GEN_TURNS_DEG)
        for index, kind in enumerate(pattern)
    ]
    return tuple(tuple(
        Forward(float(value)) if kind == "forward" else Turn(float(value))
        for kind, value in zip(pattern, values)
    ) for values in itertools.product(*domains))


@lru_cache(maxsize=None)
def _natural_fov_mask(
        length: int, start: str, feasible_count: int, stratum: str,
        half_fov_deg: float) -> tuple[bool, ...]:
    """FOV decisions aligned with one start-pattern's program order."""
    space = _natural_start_space(
        length, start, GRID_M[:feasible_count], stratum)
    return tuple(A.inside_initial_fov(
        actions, half_fov_deg) for actions in space)


def _natural_program_space_and_fov_mask(
        length: int, starts, reach_m: float, stratum: str, *,
        half_fov_deg: float, use_cache: bool) -> tuple[list, list[bool]]:
    programs, mask = [], []
    for start in starts:
        pattern = action_pattern(length, start)
        forward_count = sum(kind == "forward" for kind in pattern)
        cap = floor_grid(reach_m / forward_count)
        if cap is None:
            continue
        feasible = tuple(value for value in GRID_M if value <= cap + _EPS)
        start_space = _natural_start_space(
            length, start, feasible, stratum)
        if use_cache:
            start_mask = _natural_fov_mask(
                length, start, len(feasible), stratum,
                float(half_fov_deg))
        else:
            start_mask = tuple(A.inside_initial_fov(
                actions, float(half_fov_deg))
                for actions in start_space)
        assert len(start_mask) == len(start_space)
        programs.extend(start_space)
        mask.extend(start_mask)
    return programs, mask


def build_dynamic_natural_bank(
        rng, proxy, *, half_fov_deg: float,
        lengths: Iterable[int] = config.GEN_LENGTHS,
        per_length: int = NATURAL_PER_LENGTH_DEFAULT,
        rejections=None, stats=None, _use_fov_cache: bool = True) -> dict:
    """Draw one pose's natural programs from one depth-supported budget.

    The initial frame, public body radius and FOV determine the programme-wide
    budget.  Every Forward shares the same fair cap derived from that budget;
    the action is then frozen and the ordinary per-prefix evidence screen may
    only accept or reject it.  In particular this path never asks where a
    boundary is and never edits a leg after observing a proxy collision.
    """
    reach_m = float(proxy.prefix_supported_reach_m([]))
    result = {}
    target = int(per_length)
    if target < 0:
        raise ValueError("dynamic natural budget must be nonnegative")
    for raw_length in lengths:
        length = int(raw_length)
        starts = [start for start in ("forward", "turn")
                  if _forward_count(length, start)]
        accepted = []
        seen = set()
        for stratum, stratum_target in _natural_stratum_targets(target).items():
            start_targets = {starts[0]: int(stratum_target)}
            if len(starts) == 2:
                turn_target = int(stratum_target) // 2
                start_targets = {
                    "forward": int(stratum_target) - turn_target,
                    "turn": turn_target,
                }
            stratum_accepted = 0
            for start in starts:
                start_target = start_targets[start]
                if not start_target:
                    continue
                space, fov_mask = _natural_program_space_and_fov_mask(
                    length, (start,), reach_m, stratum,
                    half_fov_deg=float(half_fov_deg),
                    use_cache=bool(_use_fov_cache))
                if not space:
                    _count(rejections, "natural_budget_no_leg")
                    continue
                start_accepted = 0
                for raw_index in rng.permutation(len(space)):
                    actions = space[int(raw_index)]
                    key = A._action_key(actions)
                    if key in seen:
                        continue
                    seen.add(key)
                    if not fov_mask[int(raw_index)]:
                        _count(rejections, "natural_dynamic_out_of_view")
                        continue
                    coverage = proxy_natural_coverage(
                        actions, proxy, half_fov_deg=half_fov_deg,
                        rejections=rejections)
                    if coverage is None:
                        continue
                    tag = candidate_tag(actions)
                    accepted.append(Candidate(
                        tag=tag, length=length, actions=tuple(actions),
                        variant=NATURAL_DYNAMIC_VARIANT,
                        provenance={
                            "protocol": PROPOSAL_PROTOCOL_VERSION,
                            "template_id": f"N{length}-{tag.split('-', 1)[1]}",
                            "variant": NATURAL_DYNAMIC_VARIANT,
                            "starts_with": start,
                            "natural_distance_stratum": stratum,
                            "proxy_coverage": float(coverage),
                        }))
                    start_accepted += 1
                    stratum_accepted += 1
                    _tally(stats, "proposal_natural_dynamic_materialized")
                    _tally(stats,
                           f"proposal_natural_dynamic.{start}.{stratum}.L{length}")
                    if start_accepted == start_target:
                        break
                if start_accepted < start_target:
                    _tally(
                        stats,
                        f"proposal_natural_dynamic_shortfall."
                        f"{start}.{stratum}.L{length}",
                        start_target - start_accepted)
            if stratum_accepted < stratum_target:
                _tally(
                    stats,
                    f"proposal_natural_dynamic_shortfall.{stratum}.L{length}",
                    stratum_target - stratum_accepted)
        if accepted:
            result[length] = sorted(accepted, key=lambda row: row.order_key)
        if len(accepted) < target:
            _tally(stats, f"proposal_natural_dynamic_shortfall.L{length}",
                   target - len(accepted))
    return result


def build_control_bank(
        anchors, proxy, *, half_fov_deg: float,
        rejections=None, stats=None) -> dict:
    """Screen immutable cross-pose controls without changing their actions.

    Controls exist to observe the same public action under different scenes and
    poses.  Any boundary-conditioned repair would destroy that intervention,
    so depth may only admit or reject each committed action verbatim.
    """
    result = defaultdict(list)
    for anchor in anchors:
        actions = tuple(anchor.actions)
        coverage = proxy_natural_coverage(
            actions, proxy, half_fov_deg=half_fov_deg,
            rejections=rejections)
        if coverage is None:
            _tally(stats, "proposal_a1_control_rejected")
            continue
        candidate = Candidate(
            tag=candidate_tag(actions),
            length=len(actions),
            actions=actions,
            variant=A1_CONTROL_VARIANT,
            provenance={
                "protocol": PROPOSAL_PROTOCOL_VERSION,
                "template_id": str(anchor.tag),
                "variant": A1_CONTROL_VARIANT,
                "proxy_coverage": float(coverage),
                "control_catalog_sha256":
                    action_control_catalog.CATALOG_SHA256,
            },
        )
        result[candidate.length].append(candidate)
        _tally(stats, f"proposal_bank.{A1_CONTROL_VARIANT}.L{candidate.length}")
    return {
        length: sorted(candidates, key=lambda row: row.order_key)
        for length, candidates in sorted(result.items())
    }


def build_pose_bank(templates: Iterable[Template], proxy, *,
                    half_fov_deg: float, pairs_per_length: int,
                    rejections=None, stats=None) -> dict:
    """Materialise one pose's complete pre-oracle candidate bank.

    Pairs are atomic: if either side repeats a program already in the bank the
    whole pair is dropped, because a lone half is not the controlled contrast
    the pair exists to provide. Dynamic natural actions are built separately.
    """
    by_length = defaultdict(list)
    for template in templates:
        by_length[int(template.length)].append(template)
    bank = {}
    for length in sorted(by_length):
        accepted = []
        seen = set()
        pairs = 0
        for index, template in enumerate(by_length[length]):
            if pairs >= int(pairs_per_length):
                break
            _tally(stats, "proposal_templates_attempted")
            pair = materialize_pair(
                template, proxy, half_fov_deg=half_fov_deg,
                rejections=rejections)
            if pair is None:
                continue
            keys = [A._action_key(list(side.actions))
                    for side in (pair.safe, pair.collision)]
            if any(key in seen for key in keys) or keys[0] == keys[1]:
                _count(rejections, "pair_duplicate")
                continue
            seen.update(keys)
            pairs += 1
            _tally(stats, "proposal_pairs_materialized")
            for side in (pair.safe, pair.collision):
                accepted.append(Candidate(
                    tag=candidate_tag(side.actions), length=length,
                    actions=side.actions, variant=side.variant,
                    provenance=side.provenance))
        for candidate in accepted:
            _tally(stats, f"proposal_bank.{candidate.variant}.L{length}")
        if not pairs:
            _tally(stats, f"proposal_zero_pair_length.L{length}")
        if accepted:
            bank[length] = sorted(accepted, key=lambda c: c.order_key)
    if not any(bank.values()):
        _tally(stats, "proposal_zero_pair_poses")
    return bank


def action_bank_manifest(bank: dict) -> list:
    """The canonical, recomputable record of what this pose offered.

    Persisting this alongside its digest is what makes the digest an
    authentication rather than a fingerprint: a reader can rebuild the hash from
    the manifest instead of having to trust that the run was rerun identically.
    """
    return [
        {
            "length": int(length), "tag": candidate.tag,
            "variant": candidate.variant,
            "template_id": str(candidate.provenance["template_id"]),
            "actions": A.actions_to_dicts(list(candidate.actions)),
        }
        for length in sorted(bank)
        for candidate in sorted(bank[length], key=lambda c: c.order_key)
    ]


def manifest_order_key(entry: dict) -> tuple:
    return (int(entry["length"]), str(entry["template_id"]),
            str(entry["variant"]))


def action_bank_manifest_sha256(manifest) -> str:
    """Digest one canonical bank manifest, rejecting a non-canonical order."""
    entries = list(manifest)
    keys = [manifest_order_key(entry) for entry in entries]
    if keys != sorted(keys):
        raise ValueError("action bank manifest is not in canonical order")
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def materialized_action_bank_sha256(bank: dict) -> str:
    """Authenticate the whole pre-oracle bank this pose actually generated.

    Hashing only the programs that survived selection would certify the result
    rather than the sampler; the digest has to cover what was offered.
    """
    return action_bank_manifest_sha256(action_bank_manifest(bank))


def _verify_safe(actions, proxy, *, half_fov_deg, rejections):
    """Certify a safe variant over its whole path, as the formal gate does."""
    if proxy.rollout(actions).get("collision") is not False:
        return _count(rejections, "proxy_safe_collides")
    if not A.inside_initial_fov(
            actions, float(half_fov_deg), max_arc_m=None):
        return _count(rejections, "proxy_safe_out_of_view")
    coverage = float(proxy.coverage(actions, None))
    if coverage < config.EVIDENCE_COVERAGE_MIN - _EPS:
        return _count(rejections, "proxy_safe_coverage")
    return coverage


def proxy_safe_coverage(actions, proxy, *, half_fov_deg, rejections=None):
    """The frozen depth-proxy safe screen, named for callers outside this module.

    Depth no-collision, whole path inside the initial FOV, whole-path corridor
    coverage at or above ``EVIDENCE_COVERAGE_MIN``.  Deliberately *not* a safety
    certificate: the publication margin is navmesh omnidirectional clearance,
    which depth cannot compute, so a program passing here is only worth spending
    full-geometry work on.  Returns the coverage, or ``None`` with the reason
    tallied into ``rejections``.
    """
    return _verify_safe(
        actions, proxy, half_fov_deg=half_fov_deg, rejections=rejections)


def _executed_forward_legs(actions, contact_arc_m=None):
    """Every Forward leg the program walks in full, with the prefix it starts from.

    A program that stops at contact never finishes the leg the contact falls in,
    and never begins the ones after it, so neither is offered a budget: the part
    of the path that actually runs is bounded by the contact-arc coverage rule
    instead. Scanning stops at the first such leg rather than skipping it,
    because everything past it is equally unwalked.
    """
    legs = []
    prefix: list = []
    arc = 0.0
    for action in actions:
        if isinstance(action, Forward):
            metres = float(action.m)
            if (contact_arc_m is not None and
                    arc + metres > float(contact_arc_m) + _EPS):
                return legs
            legs.append((list(prefix), metres))
            arc += metres
        prefix.append(action)
    return legs


def proxy_natural_coverage(actions, proxy, *, half_fov_deg, rejections=None):
    """The label-agnostic depth screen for a natural control.

    A control exists precisely because nobody knows yet whether it collides, so
    neither pair screen fits: the safe one rejects contact and the collision one
    demands it. What they share is label-free -- the corridor the formal gate
    will judge, which is the whole path when nothing is hit and stops one
    contact tolerance past first contact when something is. That is what this
    measures, so a control is screened on the same evidence as a pair without
    being told apart from one.

    That evidence is then read a second time per leg. Whole-path coverage is a
    mean and a mean hides its worst leg, so a control whose last Forward walks
    into unobserved space could pass an aggregate the paired arm never had to
    take: the paired arm builds each leg under ``local_cap_m`` of that leg's own
    prefix and cannot propose one past it. Applying the same bound here is what
    makes the two arms sample the same budget rather than two different ones.
    It stays label-blind by bounding only the legs the program walks in full --
    reach stops close to where an obstacle is, so budgeting the contact leg
    would reject a control for colliding.

    Until this existed, controls reached full geometry with no depth screen at
    all and no ``proxy_coverage`` in their provenance. Returns the coverage, or
    ``None`` with the reason tallied into *rejections*.
    """
    verdict = proxy.rollout(actions)
    contact = (float(verdict["first_contact_arc_m"])
               if verdict.get("collision") is True else None)
    if not A.inside_initial_fov(
            actions, float(half_fov_deg), max_arc_m=contact):
        return _count(rejections, "proxy_natural_out_of_view")
    max_arc = (None if contact is None
               else contact + config.ORACLE_CONTACT_TOL_M)
    coverage = float(proxy.coverage(actions, max_arc))
    if coverage < config.EVIDENCE_COVERAGE_MIN - _EPS:
        return _count(rejections, "proxy_natural_coverage")
    for prefix, metres in _executed_forward_legs(actions, contact):
        cap = local_cap_m(proxy.prefix_supported_reach_m(prefix))
        if cap is None or metres > cap + _EPS:
            return _count(rejections, "proxy_natural_over_prefix_reach")
    return coverage


def _verify_collision(actions, proxy, *, half_fov_deg, target_action_index,
                      rejections):
    """Certify a colliding variant over its executed prefix only.

    The nominal remainder never runs, and the formal corridor gate truncates at
    first contact for exactly that reason. Requiring the whole nominal path to be
    visible here would be stricter than the gate this proposal feeds.
    """
    verdict = proxy.rollout(actions)
    if verdict.get("collision") is not True:
        return _count(rejections, "proxy_collision_missing")
    if verdict.get("contact_action_index") != target_action_index:
        return _count(rejections, "proxy_collision_wrong_leg")
    contact_arc = float(verdict["first_contact_arc_m"])
    if not A.inside_initial_fov(
            actions, float(half_fov_deg), max_arc_m=contact_arc):
        return _count(rejections, "proxy_collision_out_of_view")
    coverage = float(proxy.coverage(
        actions, contact_arc + config.ORACLE_CONTACT_TOL_M))
    if coverage < config.EVIDENCE_COVERAGE_MIN - _EPS:
        return _count(rejections, "proxy_collision_coverage")
    return coverage
