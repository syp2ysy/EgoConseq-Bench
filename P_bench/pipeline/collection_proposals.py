"""Candidate geometry precomputation and per-pose action evaluation."""

from __future__ import annotations



from pipeline import config
from pipeline import rollout
from pipeline.geometry import EXCLUDED_GEOMETRY_SOURCES


def precompute_full_geometry_candidates(
        sim, frame, pools, radii, stats):
    """Evaluate all actions while rebuilding each radius geometry only once."""
    result = {}
    for raw_length in sorted(pools):
        for action_tag, _actions in pools[raw_length]:
            if action_tag in result:
                raise ValueError(
                    f"duplicate action tag in candidate bank: {action_tag}")
            result[action_tag] = {}
    for raw_radius in radii:
        radius = float(raw_radius)
        sim.recompute_navmesh(
            radius, height=config.GROUND_ORACLE_HEIGHT_M)
        nav = sim.nav(frame.position, frame.yaw_rad)
        for raw_length in sorted(pools):
            for action_tag, actions in pools[raw_length]:
                result[action_tag][radius] = (
                    rollout.physical_collision_precheck(nav, actions))
                stats["physical_prechecks"] += 1
    return result

def record_candidate_stage(
        stats, stage: str, label: str, length: int, *,
        reason: str | None = None) -> None:
    """Record where a fixed-vocabulary action candidate enters or exits."""
    parts = [
        "candidate", str(stage), str(label), f"L{int(length)}"]
    if reason is not None:
        parts.append(str(reason))
    stats[".".join(parts)] += 1

def structured_outcome_disposition(outcome: dict) -> str:
    """Classify one structured outcome at the persistence boundary.

    An oracle disagreement rejects the whole action specification: intervention
    siblings must preserve identical action and body membership, so a partial
    keep would publish a pose whose siblings disagree.
    """
    physical = outcome.get("physical") or {}
    source = (
        physical.get("collision_source") or
        physical.get("contact_source") or
        (physical.get("contact") or {}).get("source"))
    if source in EXCLUDED_GEOMETRY_SOURCES:
        return "reject_spec"
    consensus = outcome.get("oracle_consensus") or {}
    if consensus.get("accepted") is True:
        return "keep"
    return "reject_spec"

def pose_attempt_budget(
        requested: int, *,
        pose_candidates_per_scene: int | None = None) -> int:
    """Return pose attempts without conflating candidates and witnesses."""
    count = int(requested)
    if count < 0:
        raise ValueError("requested pose count cannot be negative")
    if pose_candidates_per_scene is None:
        return count
    candidate_count = int(pose_candidates_per_scene)
    if candidate_count <= 0:
        raise ValueError("pose candidate count must be positive")
    return max(count, candidate_count)


def pose_candidate_draws_per_attempt(
        pose_candidates_per_scene: int | None) -> int | None:
    """Bind an explicit candidate budget to individual backend pose draws.

    Without a candidate budget, one requested witness keeps the backend's
    historical bounded search for a usable pose.  With a controller candidate
    budget, however, every outer attempt is already one candidate: nesting the
    backend's 400/2000-draw default under it would hide up to millions of draws
    from both the funnel counter and the capacity profile.
    """
    if pose_candidates_per_scene is None:
        return None
    candidate_count = int(pose_candidates_per_scene)
    if candidate_count <= 0:
        raise ValueError("pose candidate count must be positive")
    return 1

def evaluate_selected_action_groups(selected, evaluate_spec, *, radii,
                                    skipped,
                                    minimum=config.ACTION_CANDIDATE_MIN_PER_POSE):
    """Evaluate selected groups, dropping only what actually disagreed.

    A depth/mesh disagreement is a statement about one action, and for the
    natural route the compiler treats every outcome independently -- so
    rejecting the whole pose threw away a dozen sound programs to punish one,
    and did it most often at exactly the lengths that are hardest to fill.
    The pose is still rejected if too few survive to be a candidate set.

    The active pipeline has no atomic family route: each action is an
    independently certified candidate.
    """
    surviving = []
    accepted = {}
    for action_tag, actions in selected:
        spec = {
            "action_tag": action_tag,
            "actions": actions,
            "radii": list(radii),
            "type": "main",
        }
        result = evaluate_spec(spec)
        if result is None:
            skipped["structured_main_group_dropped"] += 1
            continue
        surviving.append((action_tag, actions))
        accepted[action_tag] = result
    if len(surviving) < int(minimum):
        skipped["structured_main_survivor_shortfall"] += 1
        return None
    return surviving, accepted
