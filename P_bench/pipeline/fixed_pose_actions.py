"""Generate and install action groups without changing record geometry."""

from __future__ import annotations

from collections import Counter
import copy
import math
import random
from typing import Iterable, Mapping, Sequence

from pipeline import abc1_record, action_proposal, actions as A, config


def parse_actions(values: Iterable[object]) -> tuple[A.Action, ...]:
    return tuple(A.parse_actions(values))


def starts_with_turn(actions: Sequence[object]) -> bool:
    if not actions:
        return False
    first = actions[0]
    return isinstance(first, A.Turn) or (
        isinstance(first, Mapping) and first.get("type") == "turn")


def compliant_turn_first(actions: Sequence[object]) -> bool:
    if not starts_with_turn(actions):
        return False
    first = actions[0]
    degrees = first.deg if isinstance(first, A.Turn) else first.get("deg")
    return float(degrees) in set(map(float, config.INITIAL_TURNS_DEG))


def record_has_turn_first(record: Mapping[str, object]) -> bool:
    if abc1_record.is_compact(record):
        return any(
            case.get("starts_with") == "turn"
            for case in record.get("cases") or [])
    selected = set(map(str, (record.get("selection") or {}).get(
        "action_group_ids") or []))
    return any(
        (not selected or str(outcome.get("action_group_id")) in selected) and
        compliant_turn_first(outcome.get("actions") or [])
        for outcome in record.get("outcomes") or [])


def record_has_noncompliant_initial_turn(
        record: Mapping[str, object]) -> bool:
    if abc1_record.is_compact(record):
        return any(
            starts_with_turn(case.get("actions") or []) and
            not compliant_turn_first(case.get("actions") or [])
            for case in record.get("cases") or [])
    selected = set(map(str, (record.get("selection") or {}).get(
        "action_group_ids") or []))
    return any(
        (not selected or str(outcome.get("action_group_id")) in selected) and
        starts_with_turn(outcome.get("actions") or []) and
        not compliant_turn_first(outcome.get("actions") or [])
        for outcome in record.get("outcomes") or [])


def candidate_programs(
        *, length: int, seed: int, limit: int, starts_with: str = "turn",
        forwards: Iterable[float] = config.GEN_FORWARDS_M,
        initial_turns: Iterable[float] = config.INITIAL_TURNS_DEG,
        later_turns: Iterable[float] = config.GEN_TURNS_DEG,
        ) -> Iterable[tuple[A.Action, ...]]:
    """Yield a deterministic bounded sample of alternating paths."""
    pattern = action_proposal.action_pattern(int(length), starts_with)
    domains = []
    for index, kind in enumerate(pattern):
        if kind == "forward":
            domains.append(tuple(float(value) for value in forwards))
        else:
            turns = initial_turns if index == 0 else later_turns
            domains.append(tuple(float(value) for value in turns))
    total = math.prod(map(len, domains))
    rng = random.Random(int(seed))
    start = rng.randrange(total)
    step = rng.randrange(1, total + 1)
    while math.gcd(step, total) != 1:
        step = step % total + 1
    for offset in range(min(int(limit), total)):
        flat = (start + offset * step) % total
        values = []
        for domain in reversed(domains):
            values.append(domain[flat % len(domain)])
            flat //= len(domain)
        values.reverse()
        yield tuple(
            A.Turn(value) if kind == "turn" else A.Forward(value)
            for kind, value in zip(pattern, values))


def proxy_shortlist(
        proxy, *, half_fov_deg: float, seed: int,
        max_programs: int, shortlist: int,
        desired_collision: bool | None = None,
        lengths: Iterable[int] = range(2, 7),
        starts: Iterable[str] = ("turn",),
        ) -> list[tuple[tuple[A.Action, ...], dict]]:
    """Find a few depth-proxy candidates under one finite search budget."""
    streams = [iter(candidate_programs(
        length=int(length), seed=int(seed) + int(length),
        limit=int(max_programs), starts_with=start))
        for length in lengths for start in starts
        if length > 1 or start == "forward"]
    rotation = int(seed) % len(streams) if streams else 0
    streams = streams[rotation:] + streams[:rotation]
    accepted = []
    attempted = 0
    while streams and attempted < int(max_programs):
        remaining = []
        for stream in streams:
            if attempted >= int(max_programs):
                break
            try:
                actions = next(stream)
            except StopIteration:
                continue
            remaining.append(stream)
            attempted += 1
            verdict = proxy_verdict(
                proxy, actions, half_fov_deg=half_fov_deg,
                desired_collision=desired_collision)
            if verdict is None:
                continue
            accepted.append((actions, verdict))
            if len(accepted) >= int(shortlist):
                return accepted
        streams = remaining
    return accepted


def proxy_verdict(
        proxy, actions: Sequence[A.Action], *, half_fov_deg: float,
        desired_collision: bool | None = None) -> dict | None:
    if (desired_collision is False and
            not A.inside_initial_fov(actions, float(half_fov_deg))):
        return None
    verdict = proxy.rollout(actions)
    collision = verdict.get("collision")
    if (not isinstance(collision, bool) or
            (desired_collision is not None and
             collision is not desired_collision)):
        return None
    contact_arc = float(verdict["first_contact_arc_m"]) if collision else None
    if not A.inside_initial_fov(
            actions, float(half_fov_deg), max_arc_m=contact_arc):
        return None
    coverage = float(proxy.coverage(
        actions, None if contact_arc is None else
        contact_arc + config.ORACLE_CONTACT_TOL_M))
    if coverage < config.EVIDENCE_COVERAGE_MIN:
        return None
    return {**verdict, "_proxy_coverage": coverage}


def _accepted_outcome(outcome: Mapping[str, object], collision: bool) -> bool:
    return bool(
        (outcome.get("physical") or {}).get("collision") is collision and
        (outcome.get("depth_physical") or {}).get("collision") is collision and
        (outcome.get("oracle_consensus") or {}).get("accepted") is True and
        (outcome.get("execution") or {}).get("completed") is not collision)


def nominal_certificate(outcome: dict) -> dict:
    from pipeline import record as record_fields

    physical = outcome["physical"]
    consensus = copy.deepcopy(outcome["oracle_consensus"])
    contact_index = physical.get("contact_action_index")
    full_id = consensus.get("full_contact_instance_id")
    depth_id = consensus.get("depth_contact_instance_id")
    value = {
        "version": "nominal-oracle.v1",
        "rows": [{
            "perturbation_id": "nominal",
            "transform": {"x_m": 0.0, "z_m": 0.0, "yaw_deg": 0.0},
            "physical": copy.deepcopy(physical),
            "depth_physical": copy.deepcopy(outcome["depth_physical"]),
            "consensus": consensus,
            "corridor_coverage": float(
                outcome["evidence"]["physical"]["coverage"]),
        }],
        "summary": {
            "evaluation": "nominal",
            "collision": bool(physical["collision"]),
            "original_action_index": (
                None if contact_index is None else int(contact_index) + 1),
            "contact_instance_id": full_id if full_id == depth_id else None,
        },
    }
    value["sha256"] = record_fields.canonical_atom_sha256(value)
    return value


def _distance_stratum(actions: Sequence[A.Action]) -> str:
    values = [float(action.m) for action in actions
              if isinstance(action, A.Forward)]
    mean = sum(values) / len(values)
    return "short" if mean <= 1.0 else ("mid" if mean <= 2.0 else "near")


def certify_program(
        frame, record: dict, actions: Sequence[A.Action], *, nav,
        radius_m: float, proxy_verdict: Mapping[str, object],
        judge_fn=None) -> dict | None:
    """Run the nominal physical/depth oracle and return one record delta."""
    from pipeline import consequence
    from pipeline.geometry import Disc
    from pipeline import outcome as outcome_fields
    from pipeline import record as record_fields

    radius = float(radius_m)
    if judge_fn is None:
        judge_fn = consequence.judge
    collision = bool(proxy_verdict["collision"])
    require_contact_witness = (
        (record.get("source") or {}).get("source_dataset") in {"r2r", "b1k"})
    outcome = judge_fn(
        frame, Disc(radius_m=radius), actions, nav=nav,
        require_contact_instance_witness=require_contact_witness)
    if not _accepted_outcome(outcome, collision):
        return None
    outcome["shared_oracle_stability"] = nominal_certificate(outcome)
    if collision:
        from pipeline import a2

        contact_index = outcome["physical"].get("contact_action_index")
        if isinstance(contact_index, int) and not isinstance(
                contact_index, bool):
            try:
                outcome["a2_design"] = a2.build_design_certificate(
                    outcome, collision_action_index_1based=contact_index + 1)
            except ValueError:
                pass
    group = action_proposal.candidate_tag(actions)
    label = "collision" if collision else "safe"
    outcome["seq_len"] = len(actions)
    outcome["action_group_id"] = group
    outcome["action_group_label"] = label
    outcome["outcome_id"] = f"b{int(round(radius * 100)):03d}-{group}"
    outcome["execution"]["execution_regime"] = \
        outcome_fields.derive_execution_regime(outcome)
    outcome["base_rollout_key"] = \
        record_fields.stored_base_rollout_key(record, outcome)
    provenance = {
        "protocol": action_proposal.PROPOSAL_PROTOCOL_V5,
        "template_id": f"N{len(actions)}-{group.split('-', 1)[-1]}",
        "variant": action_proposal.ACTION_REFRESH_VARIANT,
        "natural_distance_stratum": _distance_stratum(actions),
        "proxy_coverage": float(proxy_verdict["_proxy_coverage"]),
    }
    return {"outcomes": [outcome], "provenance": {group: provenance}}


def collect_turn_group(
        sim, frame, record: dict, *, seed: int,
        max_programs: int, max_full_attempts: int,
        radius_m: float,
        proxy=None, judge_fn=None,
        desired_collision: bool | None = None,
        lengths: Iterable[int] | None = None) -> dict | None:
    """Collect one nominally certified Turn-first group at a fixed record pose."""
    radius = float(radius_m)
    sim.recompute_navmesh(radius, height=config.GROUND_ORACLE_HEIGHT_M)
    pose = record["pose"]
    nav = sim.nav(pose["position"], float(pose["yaw_rad"]))
    if proxy is None:
        proxy = action_proposal.FrameDepthProxy(frame, radius)
    desired = bool(int(seed) & 1) if desired_collision is None else \
        bool(desired_collision)
    length_order = list(lengths or (
        (4, 5, 6) if desired else (2, 3, 4, 5, 6)))
    rotation = int(seed) % len(length_order)
    length_order = length_order[rotation:] + length_order[:rotation]
    candidates = proxy_shortlist(
        proxy, half_fov_deg=float(record["sensor"]["hfov_deg"]) / 2.0,
        seed=seed, max_programs=max_programs,
        shortlist=max(1, int(max_full_attempts)),
        lengths=length_order, desired_collision=desired_collision)
    candidates.sort(key=lambda row: row[1]["collision"] is not desired)
    for actions, proxy_verdict in candidates[:int(max_full_attempts)]:
        result = certify_program(
            frame, record, actions, nav=nav,
            radius_m=radius, proxy_verdict=proxy_verdict,
            judge_fn=judge_fn)
        if result is not None:
            return result
    return None


def collect_safe_group(
        sim, frame, record: dict, *, seed: int, radius_m: float,
        max_programs: int = 500, max_full_attempts: int = 4,
        proxy=None, judge_fn=None) -> dict | None:
    """Try both starts at the unchanged pose, returning at most one safe group."""
    radius = float(radius_m)
    sim.recompute_navmesh(radius, height=config.GROUND_ORACLE_HEIGHT_M)
    pose = record["pose"]
    nav = sim.nav(pose["position"], float(pose["yaw_rad"]))
    if proxy is None:
        proxy = action_proposal.FrameDepthProxy(frame, radius)
    candidates = proxy_shortlist(
        proxy, half_fov_deg=float(record["sensor"]["hfov_deg"]) / 2,
        seed=seed, max_programs=max_programs, shortlist=max_full_attempts,
        desired_collision=False, lengths=range(1, 7),
        starts=("forward", "turn"))
    for actions, verdict in candidates:
        result = certify_program(
            frame, record, actions, nav=nav, radius_m=radius,
            proxy_verdict=verdict, judge_fn=judge_fn)
        if result is not None:
            return result
    return None


def _turn_group_ids(record: Mapping[str, object]) -> set[str]:
    return {
        str(outcome["action_group_id"])
        for outcome in record.get("outcomes") or []
        if compliant_turn_first(outcome.get("actions") or [])
    }


def _invalid_initial_turn_group_ids(record: Mapping[str, object]) -> set[str]:
    return {
        str(outcome["action_group_id"])
        for outcome in record.get("outcomes") or []
        if starts_with_turn(outcome.get("actions") or []) and
        not compliant_turn_first(outcome.get("actions") or [])
    }


def install_groups(
        record: dict, outcomes: Iterable[dict],
        provenance: Mapping[str, dict], *,
        protected_group_ids: Iterable[str] = ()) -> dict | None:
    """Install required groups, preserving protected groups within old+1 slots."""
    updated = copy.deepcopy(record)
    selection = updated["selection"]
    original_groups = list(dict.fromkeys(map(
        str, selection.get("action_group_ids") or [])))
    invalid = _invalid_initial_turn_group_ids(updated)
    old_groups = [group for group in original_groups if group not in invalid]
    new_outcomes = [copy.deepcopy(outcome) for outcome in outcomes]
    new_groups = list(dict.fromkeys(
        str(outcome["action_group_id"]) for outcome in new_outcomes))
    protected = set(map(str, protected_group_ids)) - invalid
    required = protected | set(new_groups)
    capacity = len(original_groups) + 1
    if len(required) > capacity:
        return None

    existing_turn = _turn_group_ids(updated)
    kept_old = [group for group in old_groups if group in protected]
    for group in old_groups:
        if (group not in required and group in existing_turn and
                len(kept_old) + len(new_groups) < capacity):
            kept_old.append(group)
    for group in old_groups:
        if (group not in required and group not in kept_old and
                len(kept_old) + len(new_groups) < capacity):
            kept_old.append(group)
    selected_groups = kept_old + [
        group for group in new_groups if group not in kept_old]
    selected = set(selected_groups)

    updated["outcomes"] = [
        outcome for outcome in updated.get("outcomes") or []
        if str(outcome.get("action_group_id")) in selected and
        str(outcome.get("action_group_id")) not in new_groups
    ] + new_outcomes
    labels = {
        str(group): str(label)
        for group, label in (selection.get("action_group_labels") or {}).items()
        if str(group) in selected
    }
    for outcome in new_outcomes:
        labels[str(outcome["action_group_id"])] = str(
            outcome["action_group_label"])
    selection["action_group_ids"] = selected_groups
    selection["action_group_labels"] = labels

    old_bank = {
        str(value["tag"]): value
        for value in selection.get("materialized_action_bank") or []
        if str(value.get("tag")) in selected
    }
    first_new = {}
    for outcome in new_outcomes:
        first_new.setdefault(str(outcome["action_group_id"]), outcome)
    for group, outcome in first_new.items():
        metadata = provenance[group]
        old_bank[group] = {
            "tag": group,
            "length": len(outcome["actions"]),
            "actions": copy.deepcopy(outcome["actions"]),
            "variant": str(metadata.get("variant") or "fixed_pose_turn"),
            "template_id": str(metadata.get("template_id") or group),
        }
    bank = sorted(old_bank.values(), key=action_proposal.manifest_order_key)
    selection["materialized_action_bank"] = bank
    selection["materialized_action_bank_sha256"] = \
        action_proposal.action_bank_manifest_sha256(bank)
    old_provenance = selection.get("proposal_provenance") or {}
    selection["proposal_provenance"] = {
        group: copy.deepcopy(
            provenance[group] if group in provenance else old_provenance[group])
        for group in selected_groups
    }
    selection["observed_lengths"] = sorted({
        int(outcome["seq_len"]) for outcome in updated["outcomes"]})
    selection["observed_label_counts"] = dict(Counter(
        labels[group] for group in selected_groups))
    selection["candidate_budget"] = len(selected_groups)
    selection["matched_action_units"] = [
        unit for unit in selection.get("matched_action_units") or []
        if {str(unit.get("safe_group_id")),
            str(unit.get("collision_group_id"))} <= selected
    ]
    selection["matched_safe_action_units"] = [
        unit for unit in selection.get("matched_safe_action_units") or []
        if {str(unit.get("left_group_id")),
            str(unit.get("right_group_id"))} <= selected
    ]
    return updated
