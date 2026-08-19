"""Label-blind action selection and private matched do(action) units.

This module is simulator-free so the collection policy can be tested without
Habitat.  Public records contain the selected action groups, while ``pair_id``
and the matching key remain private record metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from pipeline import actions as A
from pipeline import config


def scalar_match_bucket(value: float, width: float) -> int:
    """Return the shared nearest-width bucket for a finite scalar."""
    if not math.isfinite(float(value)):
        raise ValueError("action matching bucket value must be finite")
    if width <= 0.0:
        raise ValueError("action matching bucket width must be positive")
    return int(round(float(value) / float(width)))


def action_match_key(actions: Sequence[A.Action]) -> dict:
    """Return the preregistered nuisance-control key for one action program."""
    parsed = list(actions)
    A.validate_alternating_actions(parsed)
    if not parsed or not any(isinstance(action, A.Forward) for action in parsed):
        raise ValueError("matched action program must contain a forward primitive")
    for action in parsed:
        value = float(action.deg if isinstance(action, A.Turn) else action.m)
        if not math.isfinite(value):
            raise ValueError("matched action values must be finite")
        if isinstance(action, A.Forward) and value <= 0.0:
            raise ValueError("matched forward distances must be positive")
    turns = [float(action.deg) for action in parsed
             if isinstance(action, A.Turn)]
    forwards = [float(action.m) for action in parsed
                if isinstance(action, A.Forward)]
    return {
        "total_forward_bucket": scalar_match_bucket(
            sum(forwards), config.ACTION_MATCH_DISTANCE_BUCKET_M),
        "primitive_count": len(parsed),
        "total_turn_bucket": scalar_match_bucket(
            sum(abs(value) for value in turns),
            config.ACTION_MATCH_TURN_BUCKET_DEG),
        "net_turn_bucket": scalar_match_bucket(
            sum(turns), config.ACTION_MATCH_TURN_BUCKET_DEG),
        "turn_direction_sequence": [
            1 if value > 0.0 else -1 if value < 0.0 else 0
            for value in turns
        ],
        "forward_distance_profile": [
            scalar_match_bucket(value, config.ACTION_MATCH_DISTANCE_BUCKET_M)
            for value in forwards
        ],
    }


def candidate_sort_key(
    pose_seed: int,
    group_id: str,
    actions: Sequence[A.Action],
) -> str:
    payload = {
        "pose_seed": int(pose_seed),
        "group_id": str(group_id),
        "actions": A.actions_to_dicts(list(actions)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_natural_action_groups(
    pools: Mapping[int, Sequence[tuple[str, Sequence[A.Action]]]],
    group_labels: Mapping[str, str],
    *,
    pose_seed: int,
    min_actions: int = config.ACTION_CANDIDATE_MIN_PER_POSE,
    max_actions: int = config.ACTION_CANDIDATE_MAX_PER_POSE,
    reserved_tags: Sequence[str] = (),
) -> list[tuple[str, Sequence[A.Action]]]:
    """Select a variable-size candidate set without consulting outcome labels.
    Membership in ``group_labels`` marks candidates whose physical precheck
    completed.  Label *values* are intentionally never read.  Reserved C1
    members are kept first, then every certified action length gets one seat
    before the remaining deterministic hash order is filled.
    """
    minimum = int(min_actions)
    maximum = int(max_actions)
    if minimum < 1 or maximum < minimum:
        raise ValueError("natural action count bounds are invalid")
    candidates = [
        (str(group_id), actions, int(length))
        for length in sorted(pools)
        for group_id, actions in pools[length]
        if str(group_id) in group_labels
    ]
    if not candidates:
        return []
    by_tag = {tag: (actions, length)
              for tag, actions, length in candidates}
    reserved = []
    for raw_tag in reserved_tags:
        tag = str(raw_tag)
        # A reserved program rejected by certification is absent from
        # ``group_labels`` and simply costs one distractor, never the pose.
        if tag in by_tag and tag not in reserved:
            reserved.append(tag)
    if len(reserved) > maximum:
        raise ValueError("certified reserved actions exceed selection budget")
    covered_lengths = {by_tag[tag][1] for tag in reserved}
    length_floor = []
    for length in sorted({length for _tag, _actions, length in candidates}):
        if length in covered_lengths:
            continue
        choices = sorted(
            [(tag, actions) for tag, actions, candidate_length in candidates
             if candidate_length == length and tag not in reserved],
            key=lambda item: candidate_sort_key(
                int(pose_seed), item[0], item[1]),
        )
        if choices:
            length_floor.append(choices[0][0])
            covered_lengths.add(length)
    required = reserved + length_floor
    if len(required) > maximum:
        raise ValueError(
            "certified action-length floor exceeds selection budget")
    budget = max(len(required), natural_candidate_budget(
        len(candidates), pose_seed=pose_seed,
        min_actions=minimum, max_actions=maximum))
    required_set = set(required)
    remainder = sorted(
        [(tag, actions) for tag, actions, _length in candidates
         if tag not in required_set],
        key=lambda item: candidate_sort_key(
            int(pose_seed), item[0], item[1]),
    )
    return [(tag, by_tag[tag][0]) for tag in required] + \
        remainder[:budget - len(required)]


def retain_certified_action_groups(
    pools: Mapping[int, Sequence[tuple[str, Sequence[A.Action]]]],
    group_labels: Mapping[str, str],
    *,
    maximum: int = config.ACTION_CANDIDATE_MAX_PER_POSE,
) -> list[tuple[str, Sequence[A.Action]]]:
    """Retain every certified action in deterministic bank order.

    Proposal v4 pays geometry only for the bounded ordinary shortlist plus its
    bounded C1 neighbours, so a second random crop would discard work the pose
    has already certified. Membership in ``group_labels`` is the sole gate;
    label values are never consulted.
    """
    limit = int(maximum)
    if limit < 1:
        raise ValueError("certified action limit must be positive")
    selected = [
        (str(tag), actions)
        for length in sorted(pools)
        for tag, actions in pools[length]
        if str(tag) in group_labels
    ]
    if len(selected) > limit:
        raise ValueError("certified actions exceed selection budget")
    return selected


# The order cells are served in. Nothing here reads an outcome label: a
# ``variant`` and the natural distance stratum are fixed before geometry runs.
_VARIANT_ORDER = (
    "safe", "collision", "natural_dynamic", "a1_control", "natural",
)
_STRATUM_ORDER = ("short", "mid", "near", "na")


def _cell_rank(cell: tuple[int, str, str]) -> tuple[int, int, int, int, str]:
    length, variant, stratum = cell
    try:
        position = _VARIANT_ORDER.index(variant)
    except ValueError:
        position = len(_VARIANT_ORDER)
    try:
        stratum_position = _STRATUM_ORDER.index(stratum)
    except ValueError:
        stratum_position = len(_STRATUM_ORDER)
    natural_rank = 0 if variant == "natural_dynamic" else 1
    return (
        natural_rank, int(length), stratum_position, position, str(variant))


def _candidate_cell(length: int, tag: str, provenance: Mapping) -> tuple:
    row = (provenance or {}).get(str(tag), {}) or {}
    variant = str(row.get("variant") or "file")
    stratum = (
        str(row.get("natural_distance_stratum") or "unspecified")
        if variant == "natural_dynamic" else "na")
    return int(length), variant, stratum


def _cell_tag_order(
        cell: tuple[int, str, str], tags: Sequence[str], *,
        known: Mapping[str, tuple[int, Sequence[A.Action]]],
        pose_seed: int) -> list[str]:
    by_start = {"f": [], "t": [], "other": []}
    for tag in tags:
        actions = known[tag][1]
        start = (
            "f" if actions and isinstance(actions[0], A.Forward)
            else "t" if actions and isinstance(actions[0], A.Turn)
            else "other")
        by_start[start].append(tag)
    for values in by_start.values():
        values.sort(key=lambda tag: candidate_sort_key(
            int(pose_seed), tag, known[tag][1]))
    parity = int(hashlib.sha256(
        f"{int(pose_seed)}\0{cell}".encode()).hexdigest(), 16) % 2
    first, second = (("f", "t") if parity == 0 else ("t", "f"))
    ordered = []
    depth = 0
    while (depth < len(by_start[first]) or depth < len(by_start[second])):
        if depth < len(by_start[first]):
            ordered.append(by_start[first][depth])
        if depth < len(by_start[second]):
            ordered.append(by_start[second][depth])
        depth += 1
    return ordered + by_start["other"]


def stratified_action_order(
        pools: Mapping[int, Sequence[tuple[str, Sequence[A.Action]]]],
        provenance: Mapping[str, Mapping], *, pose_seed: int,
        forced_tags: Sequence[str] = ()) -> list[str]:
    """Order proposed programs without access to any physical label."""
    known = {
        str(tag): (int(length), actions)
        for length in pools for tag, actions in pools[length]
    }
    reserved = []
    for raw_tag in forced_tags:
        tag = str(raw_tag)
        if tag not in known:
            raise ValueError(
                f"shortlist forced reservation is not in the bank: {tag}")
        if tag not in reserved:
            reserved.append(tag)
    cells = defaultdict(list)
    for tag, (length, _actions) in known.items():
        cells[_candidate_cell(length, tag, provenance)].append(tag)
    cell_tags = {
        cell: _cell_tag_order(
            cell, tags, known=known, pose_seed=int(pose_seed))
        for cell, tags in cells.items()
    }
    ordered = list(reserved)
    taken = set(ordered)
    cells_in_order = sorted(cells, key=_cell_rank)
    depth = 0
    while any(len(cell_tags[cell]) > depth for cell in cells_in_order):
        for cell in cells_in_order:
            if len(cell_tags[cell]) <= depth:
                continue
            tag = cell_tags[cell][depth]
            if tag not in taken:
                ordered.append(tag)
                taken.add(tag)
        depth += 1
    return ordered


def retain_stratified_action_groups(
        pools: Mapping[int, Sequence[tuple[str, Sequence[A.Action]]]],
        provenance: Mapping[str, Mapping], *, stable_tags: Iterable[str],
        pose_seed: int, maximum: int,
        forced_tags: Sequence[str] = ()) -> list[tuple[str, Sequence[A.Action]]]:
    """Crop stable ordinary groups with the same label-blind cell order."""
    limit = int(maximum)
    if limit < 1:
        raise ValueError("stable action limit must be positive")
    stable = {str(tag) for tag in stable_tags}
    forced = tuple(str(tag) for tag in forced_tags)
    if not set(forced) <= stable:
        raise ValueError("forced stable action is unavailable")
    known = {
        str(tag): actions
        for candidates in pools.values() for tag, actions in candidates
    }
    ordered = stratified_action_order(
        pools, provenance, pose_seed=int(pose_seed), forced_tags=forced)
    return [(tag, known[tag]) for tag in ordered if tag in stable][:limit]


def _tally(stats, key: str) -> None:
    if stats is not None:
        stats[key] += 1


def shortlist_action_bank(
    pools: Mapping[int, Sequence[tuple[str, Sequence[A.Action]]]],
    provenance: Mapping[str, Mapping],
    *,
    pose_seed: int,
    budget: int = config.ACTION_CANDIDATE_MAX_PER_POSE,
    forced_tags: Sequence[str] = (),
    stats=None,
) -> dict[int, list[tuple[str, Sequence[A.Action]]]]:
    """Cut the pose's proposal bank down to what full geometry will pay for.

    Certification is the expensive half of a pose -- every candidate costs a
    publication label plus two variants times three radii of rollout, coverage
    and consensus -- and it used to run over the entire bank, several hundred
    programs, before anything trimmed it.  Trimming first is the whole point,
    so this returns a pools-shaped bank and the caller certifies only that.

    Two properties make the trim safe rather than merely cheap.

    It is label-blind by construction, not by discipline: ``group_labels`` is
    the *product* of certification and does not exist yet at this point, so
    there is no label here to read even by accident.  What it stratifies on is
    ``(length, variant)`` -- both fixed when the program was proposed.

    And it is stratified rather than uniform.  A flat hash order over a bank
    whose lengths are unevenly populated routinely left whole lengths with no
    survivors, which is what starved L5/L6.  Cells are served round-robin, so
    the first round *is* the "every non-empty cell gets one" floor and no
    separate pass is needed to guarantee it.

    Frozen family tags are admitted before the round-robin because they cannot
    be substituted: a frozen member that misses the shortlist never gets a
    certificate and takes its whole family down.  C1 neighbours are different:
    they are derived only after this first pass identifies a certified-clear
    query, then consume the explicitly reserved second-pass budget.
    """
    limit = int(budget)
    if limit < 1:
        raise ValueError("shortlist budget must be positive")
    known = {str(tag): (length, actions)
             for length in pools
             for tag, actions in pools[length]}

    reserved = list(dict.fromkeys(str(tag) for tag in forced_tags))
    if len(reserved) > limit:
        raise ValueError(
            "shortlist reservations exceed the candidate budget")

    cells: dict[tuple[int, str, str], list[str]] = defaultdict(list)
    for length in sorted(pools):
        for tag, actions in pools[length]:
            tag = str(tag)
            cell = _candidate_cell(length, tag, provenance)
            cells[cell].append(tag)
            _, variant, stratum = cell
            name = variant if stratum == "na" else f"{variant}.{stratum}"
            _tally(stats, f"shortlist_offered.{name}.L{length}")
    order = stratified_action_order(
        pools, provenance, pose_seed=int(pose_seed),
        forced_tags=reserved)
    admitted = order[:limit]
    taken = set(admitted)

    for cell in sorted(cells, key=_cell_rank):
        length, variant, stratum = cell
        name = variant if stratum == "na" else f"{variant}.{stratum}"
        count = sum(1 for tag in cells[cell] if tag in taken)
        if count:
            for _ in range(count):
                _tally(stats, f"shortlist_admitted.{name}.L{length}")
        else:
            _tally(stats, f"shortlist_cell_starved.{name}.L{length}")

    return {
        length: [(tag, actions) for tag, actions in pools[length]
                 if str(tag) in taken]
        for length in sorted(pools)
        if any(str(tag) in taken for tag, _actions in pools[length])
    }


def natural_candidate_budget(
    available: int,
    *,
    pose_seed: int,
    min_actions: int = config.ACTION_CANDIDATE_MIN_PER_POSE,
    max_actions: int = config.ACTION_CANDIDATE_MAX_PER_POSE,
) -> int:
    """Return the deterministic variable budget before label observation."""
    minimum = int(min_actions)
    maximum = int(max_actions)
    if minimum < 1 or maximum < minimum:
        raise ValueError("natural action count bounds are invalid")
    span = maximum - minimum + 1
    return min(max(0, int(available)),
               minimum + int(pose_seed) % span)


def _pair_id(match_key: dict, safe_group_id: str,
             collision_group_id: str) -> str:
    payload = {
        "contract": "matched-action-unit.v1",
        "match_key": match_key,
        "safe_group_id": str(safe_group_id),
        "collision_group_id": str(collision_group_id),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def matched_action_units(candidates: Iterable[dict]) -> list[dict]:
    """Pair safe/collision programs with equal nuisance-control keys.
    Pairing is opportunistic: unmatched actions remain valid candidates and the
    number of pairs is allowed to vary by pose.
    """
    buckets: dict[str, dict[str, list[tuple[str, dict]]]] = defaultdict(
        lambda: {"safe": [], "collision": []})
    for candidate in candidates:
        label = candidate.get("label")
        if label not in {"safe", "collision"}:
            continue
        group_id = str(candidate["group_id"])
        key = action_match_key(candidate["actions"])
        serialized = json.dumps(key, sort_keys=True, separators=(",", ":"))
        buckets[serialized][label].append((group_id, key))
    units = []
    for serialized in sorted(buckets):
        members = buckets[serialized]
        safe = sorted(members["safe"], key=lambda value: value[0])
        collision = sorted(members["collision"], key=lambda value: value[0])
        for (safe_id, key), (collision_id, _other_key) in zip(safe, collision):
            units.append({
                "pair_id": _pair_id(key, safe_id, collision_id),
                "safe_group_id": safe_id,
                "collision_group_id": collision_id,
                "match_key": key,
            })
    return units


def expected_same_label_pair_id(
        match_key: dict, left_group_id: str, right_group_id: str, *,
        label: str) -> str:
    """Recompute a private same-label pair identifier."""
    payload = {
        "contract": "matched-same-label-action-unit.v1",
        "label": str(label),
        "match_key": match_key,
        "left_group_id": str(left_group_id),
        "right_group_id": str(right_group_id),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def matched_same_label_units(
        candidates: Iterable[dict], *, label: str) -> list[dict]:
    """Pair actions with the same label and identical nuisance controls."""
    required_label = str(label)
    if required_label not in {"safe", "collision"}:
        raise ValueError("same-label action matching requires safe or collision")
    buckets: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for candidate in candidates:
        if candidate.get("label") != required_label:
            continue
        group_id = str(candidate["group_id"])
        key = action_match_key(candidate["actions"])
        serialized = json.dumps(key, sort_keys=True, separators=(",", ":"))
        buckets[serialized].append((group_id, key))
    units = []
    for serialized in sorted(buckets):
        members = sorted(buckets[serialized], key=lambda value: value[0])
        for index in range(0, len(members) - 1, 2):
            (left_id, key), (right_id, _other_key) = members[index:index + 2]
            units.append({
                "pair_id": expected_same_label_pair_id(
                    key, left_id, right_id, label=required_label),
                "left_group_id": left_id,
                "right_group_id": right_id,
                "label": required_label,
                "match_key": key,
            })
    return units


def expected_pair_id(match_key: dict, safe_group_id: str,
                     collision_group_id: str) -> str:
    """Recompute a stored private identifier at a validation boundary."""
    return _pair_id(match_key, safe_group_id, collision_group_id)


def classify_action_group(
        sibling_acceptance, radius_collisions, required_siblings: int):
    """Return safe/collision/radius_mixed after the atomic consensus gate."""
    if (len(sibling_acceptance) != int(required_siblings) or
            not all(sibling_acceptance)):
        return None
    labels = {bool(value) for value in radius_collisions.values()}
    if len(labels) != 1:
        return "radius_mixed"
    return "collision" if labels.pop() else "safe"
