"""Rank-conditioned action design for A2 collision grounding."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import random
from typing import Iterable, Iterator, Sequence

from pipeline import actions as A, config


PROTOCOL = "rank-conditioned-a2-v1"
RANKS_BY_FORWARD_COUNT = {
    2: ("shortest", "longest"),
    3: ("shortest", "middle", "longest"),
}


@dataclass(frozen=True, order=True)
class A2Cell:
    forward_ordinal_1based: int
    distance_rank: str


@dataclass(frozen=True)
class A2Proposal:
    collision_actions: tuple[A.Action, ...]
    cell: A2Cell
    collision_action_index_1based: int
    proxy_contact_action_local_arc_m: float
    proxy_collision_coverage: float


def forward_distances(actions: Sequence[A.Action]) -> tuple[float, ...]:
    return tuple(
        float(action.m) for action in actions
        if isinstance(action, A.Forward))


def action_pattern(length: int, starts_with: str) -> tuple[str, ...]:
    if starts_with not in {"forward", "turn"}:
        raise ValueError("A2 program must start with Forward or Turn")
    return tuple(
        starts_with if index % 2 == 0 else
        ("turn" if starts_with == "forward" else "forward")
        for index in range(int(length)))


def collision_action_index(
        cell: A2Cell, *, length: int,
        starts_with: str = "forward") -> int:
    """Return the primitive index of the requested Forward in the pattern."""
    forwards = [
        index for index, kind in enumerate(
            action_pattern(length, starts_with), 1)
        if kind == "forward"
    ]
    ordinal = int(cell.forward_ordinal_1based)
    if ordinal not in range(1, len(forwards) + 1):
        raise ValueError("A2 Forward ordinal is outside the action pattern")
    return forwards[ordinal - 1]


def action_cell(
        actions: Sequence[A.Action], *,
        collision_action_index_1based: int) -> A2Cell:
    parsed = list(actions)
    forwards = [
        (index, float(action.m))
        for index, action in enumerate(parsed, 1)
        if isinstance(action, A.Forward)
    ]
    ranks = RANKS_BY_FORWARD_COUNT.get(len(forwards))
    if ranks is None:
        raise ValueError("A2 requires exactly two or three Forward actions")
    distances = [distance for _index, distance in forwards]
    indices = [index for index, _distance in forwards]
    try:
        ordinal = indices.index(int(collision_action_index_1based)) + 1
    except ValueError as error:
        raise ValueError("A2 collision action must be a Forward") from error
    target = distances[ordinal - 1]
    if distances.count(target) != 1:
        raise ValueError("A2 collision Forward distance must be distinct")
    rank_index = sum(value < target for value in distances)
    return A2Cell(ordinal, ranks[rank_index])


def cells_for_length(
        length: int, *, starts_with: str = "forward") -> tuple[A2Cell, ...]:
    if int(length) not in {3, 4, 5, 6}:
        raise ValueError("A2 action length must be L3-L6")
    if starts_with not in {"forward", "turn"}:
        raise ValueError("A2 program must start with Forward or Turn")
    pattern = action_pattern(length, starts_with)
    forward_count = pattern.count("forward")
    ranks = RANKS_BY_FORWARD_COUNT.get(forward_count)
    if ranks is None:
        return ()
    return tuple(
        A2Cell(ordinal, rank)
        for ordinal in range(1, forward_count + 1)
        for rank in ranks
    )


def balanced_cells(
        length: int, total: int, *, starts_with: str = "forward",
        rotation: int = 0) -> tuple[A2Cell, ...]:
    """Return joint ordinal/rank slots whose counts differ by at most one."""
    cells = cells_for_length(length, starts_with=starts_with)
    count = int(total)
    if count < 0:
        raise ValueError("A2 cell total must be nonnegative")
    if not cells:
        return ()
    base, remainder = divmod(count, len(cells))
    rotated = cells[int(rotation) % len(cells):] + \
        cells[:int(rotation) % len(cells)]
    return tuple(list(cells) * base + list(rotated[:remainder]))


def candidate_programs(
        *, length: int, cell: A2Cell, seed: int,
        starts_with: str = "forward",
        forward_grid: Iterable[float] = config.GEN_FORWARDS_M,
        turn_grid: Iterable[float] = config.GEN_TURNS_DEG,
        initial_turn_grid: Iterable[float] = config.INITIAL_TURNS_DEG,
        ) -> Iterator[tuple[A.Action, ...]]:
    """Enumerate each physically distinct pre-collision prefix once."""
    if starts_with not in {"forward", "turn"}:
        raise ValueError("A2 program must start with Forward or Turn")
    pattern = action_pattern(length, starts_with)
    forward_count = pattern.count("forward")
    ranks = RANKS_BY_FORWARD_COUNT.get(forward_count)
    if (ranks is None or
            cell.forward_ordinal_1based not in range(1, forward_count + 1) or
            cell.distance_rank not in ranks):
        raise ValueError("A2 cell does not belong to the action length")
    forward_values = tuple(float(value) for value in forward_grid)
    turn_values = tuple(float(value) for value in turn_grid)
    initial_turn_values = tuple(float(value) for value in initial_turn_grid)
    target_index = collision_action_index(
        cell, length=length, starts_with=starts_with) - 1
    prefix = pattern[:target_index + 1]
    prefix_forward_count = prefix.count("forward")
    desired_less = ranks.index(cell.distance_rank)
    profiles = []
    for values in itertools.product(
            forward_values, repeat=prefix_forward_count):
        target = values[-1]
        before = values[:-1]
        if target in before:
            continue
        less_before = sum(value < target for value in before)
        greater_before = len(before) - less_before
        tail_count = forward_count - prefix_forward_count
        less_after = desired_less - less_before
        greater_after = (
            forward_count - 1 - desired_less - greater_before)
        if (less_after < 0 or greater_after < 0 or
                less_after + greater_after != tail_count):
            continue
        lower = tuple(value for value in forward_values if value < target)
        higher = tuple(value for value in forward_values if value > target)
        if (less_after and not lower) or (greater_after and not higher):
            continue
        profiles.append((values, lower, higher, less_after, greater_after))
    turn_domains = [
        initial_turn_values if index == 0 else turn_values
        for index, kind in enumerate(prefix) if kind == "turn"
    ]
    turns = list(itertools.product(*turn_domains))
    rng = random.Random(int(seed))
    rng.shuffle(profiles)
    rng.shuffle(turns)
    total = len(profiles) * len(turns)
    if total == 0:
        return
    start = rng.randrange(total)
    step = rng.randrange(1, total + 1)
    while math.gcd(step, total) != 1:
        step = step % total + 1
    for offset in range(total):
        flat = (start + offset * step) % total
        profile, lower, higher, less_after, greater_after = \
            profiles[flat % len(profiles)]
        angles = turns[flat // len(profiles)]
        tail = [
            lower[(flat + index) % len(lower)]
            for index in range(less_after)
        ] + [
            higher[(flat + index) % len(higher)]
            for index in range(greater_after)
        ]
        random.Random(int(seed) ^ flat).shuffle(tail)
        actions = []
        forward_index = turn_index = tail_index = 0
        for index, kind in enumerate(pattern):
            if kind == "forward":
                if index <= target_index:
                    actions.append(A.Forward(profile[forward_index]))
                    forward_index += 1
                else:
                    actions.append(A.Forward(tail[tail_index]))
                    tail_index += 1
            else:
                if index <= target_index:
                    actions.append(A.Turn(angles[turn_index]))
                    turn_index += 1
                else:
                    domain = initial_turn_values if index == 0 else turn_values
                    actions.append(A.Turn(domain[(flat + index) % len(domain)]))
        yield tuple(actions)


def propose_collision(
        proxy, *, length: int, cell: A2Cell, seed: int,
        half_fov_deg: float, max_programs: int | None = None,
        starts_with: str = "forward",
        ) -> A2Proposal | None:
    """Return the first proxy-certified collision program for one cell."""
    return next(collision_proposals(
        proxy, length=length, cell=cell, seed=seed,
        half_fov_deg=half_fov_deg, max_programs=max_programs,
        starts_with=starts_with), None)


def collision_proposals(
        proxy, *, length: int, cell: A2Cell, seed: int,
        half_fov_deg: float, max_programs: int | None = None,
        starts_with: str = "forward",
        ) -> Iterator[A2Proposal]:
    """Yield proxy-certified programs so the full oracle can continue."""
    target_index = collision_action_index(
        cell, length=length, starts_with=starts_with) - 1
    margin = float(config.A2_ACTION_BOUNDARY_MARGIN_M)
    programs = candidate_programs(
        length=length, cell=cell, seed=seed, starts_with=starts_with)
    if max_programs is not None:
        programs = itertools.islice(programs, int(max_programs))
    for actions in programs:
        prefix = tuple(actions[:target_index + 1])
        collision = proxy.rollout(prefix)
        valid = (
            collision.get("collision") is True and
            collision.get("contact_action_index") == target_index)
        if valid:
            local = float(collision["contact_action_local_arc_m"])
            target_distance = float(actions[target_index].m)
            contact_arc = float(collision["first_contact_arc_m"])
            valid = (
                local >= margin and
                target_distance - local >= margin and
                A.inside_initial_fov(
                    prefix, float(half_fov_deg), max_arc_m=contact_arc))
        if valid:
            collision_coverage = float(proxy.coverage(
                prefix, contact_arc + config.ORACLE_CONTACT_TOL_M))
            valid = collision_coverage >= config.EVIDENCE_COVERAGE_MIN
        if not valid:
            continue
        local = float(collision["contact_action_local_arc_m"])
        yield A2Proposal(
            collision_actions=tuple(actions), cell=cell,
            collision_action_index_1based=target_index + 1,
            proxy_contact_action_local_arc_m=local,
            proxy_collision_coverage=collision_coverage)


def cell_counts(rows: Iterable[tuple[int, A2Cell]]) -> dict[str, int]:
    """Small JSON-ready counter shared by collection and visualization."""
    counts = Counter(f"L{length}:{cell.forward_ordinal_1based}:{cell.distance_rank}"
                     for length, cell in rows)
    return dict(sorted(counts.items()))


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _accepted_collision(outcome: dict) -> bool:
    return bool(
        (outcome.get("physical") or {}).get("collision") is True and
        (outcome.get("depth_physical") or {}).get("collision") is True and
        (outcome.get("oracle_consensus") or {}).get("accepted") is True and
        (outcome.get("execution") or {}).get("completed") is False)


def build_design_certificate(
        collision_outcome: dict, *,
        collision_action_index_1based: int) -> dict:
    """Bind a dual-oracle collision to its ordinal/rank cell."""
    if not _accepted_collision(collision_outcome):
        raise ValueError("collision outcome is not dual-oracle accepted")
    collision_actions = A.parse_actions(collision_outcome.get("actions") or [])
    target = int(collision_action_index_1based) - 1
    if not 0 <= target < len(collision_actions):
        raise ValueError("collision action index is outside the program")
    if any(
            (collision_outcome.get(key) or {}).get("contact_action_index") !=
            target for key in ("physical", "depth_physical")):
        raise ValueError("dual oracles disagree with the collision action")
    target_distance = float(collision_actions[target].m)
    margins = [min(
        float(collision_outcome[key]["contact_action_local_arc_m"]),
        target_distance - float(
            collision_outcome[key]["contact_action_local_arc_m"]))
        for key in ("physical", "depth_physical")]
    stability = collision_outcome.get("shared_oracle_stability") or {}
    summary = stability.get("summary") or {}
    nominal = (
        stability.get("version") == "nominal-oracle.v1" and
        summary.get("evaluation") == "nominal")
    if (not nominal and
            min(margins) < float(config.A2_ACTION_BOUNDARY_MARGIN_M)):
        raise ValueError("collision is too close to an action boundary")
    if not (
            summary.get("collision") is True and
            (nominal or (
                summary.get("collision_label_stable") is True and
                summary.get("original_action_index_stable") is True)) and
            summary.get("original_action_index") == target + 1):
        raise ValueError("collision action is not stable")
    cell = action_cell(
        collision_actions,
        collision_action_index_1based=collision_action_index_1based)
    value = {
        "protocol": PROTOCOL,
        "collision_action_index_1based": int(
            collision_action_index_1based),
        "cell": {
            "forward_ordinal_1based": cell.forward_ordinal_1based,
            "distance_rank": cell.distance_rank,
        },
        "forward_distances_m": list(forward_distances(collision_actions)),
        "collision_actions_sha256": A.canonical_actions_sha256(
            collision_actions),
        "minimum_action_boundary_margin_m": min(margins),
    }
    return {**value, "sha256": _sha256(value)}


def validate_design_certificate(
        collision_outcome: dict, certificate: dict) -> list[str]:
    """Return concise structural mismatches for one stored A2 certificate."""
    try:
        stored = dict(certificate)
        digest = stored.pop("sha256")
        if digest != _sha256(stored):
            return ["sha256"]
        rebuilt = build_design_certificate(
            collision_outcome,
            collision_action_index_1based=stored[
                "collision_action_index_1based"])
    except (KeyError, TypeError, ValueError) as error:
        return [str(error)]
    return [] if rebuilt == certificate else ["content"]
