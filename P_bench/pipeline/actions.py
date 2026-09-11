"""Action primitives and pure path geometry (no Habitat, no numpy state).

Ground frame convention (project-wide):
    origin = footprint centre on the floor, +z = forward, +x = right, +y = up.
    heading h (rad): forward direction at heading h is (sin h, cos h) in (x, z);
    h = 0 faces +z. A positive Turn (deg > 0) = right turn (toward +x).

An ActionSeq is any list of Turn / Forward. Turns are in-place (zero arc length;
the footprint disk is rotation-invariant so they add no swept area). Forward legs
march along the current heading.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, TypeVar, Union

import numpy as np

from pipeline import config


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    deg: float
    def to_dict(self) -> dict:
        return {"type": "turn", "deg": float(self.deg)}


@dataclass(frozen=True)
class Forward:
    m: float
    def to_dict(self) -> dict:
        return {"type": "forward", "m": float(self.m)}


Action = Union[Turn, Forward]
ActionSeq = List[Action]


@dataclass(frozen=True)
class ForwardLegLocation:
    """Metric location within the ordered Forward legs of an action program."""
    action_index: int
    forward_leg_number: int
    cumulative_before_leg_m: float
    distance_into_leg_m: float
    leg_length_m: float


def parse_actions(raw: Sequence[dict]) -> ActionSeq:
    """Parse a list of {"type":"turn","deg":..} / {"type":"forward","m":..} dicts."""
    out: ActionSeq = []
    for a in raw:
        t = a["type"]
        if t == "turn":
            out.append(Turn(float(a["deg"])))
        elif t == "forward":
            out.append(Forward(float(a["m"])))
        else:
            raise ValueError(f"unknown action type: {t!r}")
    return out


def actions_to_dicts(actions: Sequence[Action]) -> List[dict]:
    return [a.to_dict() for a in actions]


# --------------------------------------------------------------------------
# Canonical action digests
#
# Two digests of the same program exist and they are NOT interchangeable.  Each
# has exactly one definition, in the layer that owns it:
#
#   actions.canonical_actions_sha256   hashes the bare list.  Its first twelve
#                                      hex digits are the candidate tag, which
#                                      reaches action_group_id -> outcome_id ->
#                                      the published QA's oracle_ref.
#   record.action_program_sha256       hashes {"actions": [...]} as a record
#                                      atom.  That is the value published as
#                                      terminal_rgb_asset.binding.action_sha256
#                                      and the key C1 families and the QA choice
#                                      descriptors join on.
#
# Merging them would move one published identifier or the other, so they stay
# distinct until a deliberate re-freeze.  The record-atom one cannot live here:
# it needs the record layer's json normaliser, and pipeline.record imports this
# module, so defining it here would create the import cycle that
# tests/test_pl_module_structure.py forbids.
# --------------------------------------------------------------------------

def canonical_actions_sha256(actions: Sequence[Action]) -> str:
    """Digest of the bare canonical action list (candidate-tag domain)."""
    encoded = json.dumps(
        actions_to_dicts(list(actions)), sort_keys=True,
        separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------------
# Scalar summaries
# --------------------------------------------------------------------------

def wrap_deg(d: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    r = (d + 180.0) % 360.0 - 180.0
    return 180.0 if r == -180.0 else r


def planar_distance_m(x_m: float, z_m: float) -> float:
    """Ground-plane distance from the origin, identical in every interpreter.

    ``math.hypot`` looks like the obvious call and is the wrong one for any
    number that reaches a content hash. It is implemented inside CPython and
    its last bit changed in 3.10, so the same two coordinates give
    1.3989663259659064 under 3.9 and ...66 under 3.11. The B1K route straddles
    exactly that boundary -- the collector runs under the simulator's
    interpreter, its supervisor validates under the pipeline's -- so a family
    hashed on one side could not be re-derived on the other, and a shard whose
    data was entirely sound failed its own completeness check.

    Multiplication, addition and ``sqrt`` are correctly rounded by IEEE-754, so
    this agrees bit for bit everywhere a conforming double exists.
    """
    x = float(x_m)
    z = float(z_m)
    return math.sqrt(x * x + z * z)


HORIZONTAL_DIRECTION_LABELS = (
    "front", "front-right", "right", "rear-right",
    "rear", "rear-left", "left", "front-left",
)
VERTICAL_DIRECTION_LABELS = ("above", "level", "below")
DIRECTION_CONVENTION = (
    "Use the camera frame at the queried moment. Front, right, rear, and left "
    "each span ±7.5° around their corresponding horizontal axes. Front-right, "
    "rear-right, rear-left, and front-left cover the intervals between these "
    "ranges. Elevation greater than 7.5° is above; less than −7.5° is below; "
    "otherwise it is at camera level."
)


def horizontal_direction(azimuth_deg: float) -> str:
    """Classify camera-centred azimuth into eight horizontal directions.

    Axis-aligned directions span 15 degrees total (plus or minus 7.5 degrees);
    each intervening diagonal direction spans 75 degrees.
    """
    azimuth = wrap_deg(float(azimuth_deg))
    if abs(azimuth) <= 7.5:
        direction = "front"
    elif abs(azimuth - 90.0) <= 7.5:
        direction = "right"
    elif abs(azimuth + 90.0) <= 7.5:
        direction = "left"
    elif abs(azimuth) >= 172.5:
        direction = "rear"
    elif 0.0 < azimuth < 90.0:
        direction = "front-right"
    elif 90.0 < azimuth < 180.0:
        direction = "rear-right"
    elif -90.0 < azimuth < 0.0:
        direction = "front-left"
    else:
        direction = "rear-left"
    return direction


def vertical_direction(elevation_deg: float) -> str:
    """Classify elevation relative to the camera-level plane."""
    elevation = float(elevation_deg)
    if elevation > 7.5:
        return "above"
    if elevation < -7.5:
        return "below"
    return "level"


def spatial_direction_key(horizontal: str, vertical: str) -> str:
    """Compose a compact internal key; level is the unqualified direction."""
    if horizontal not in HORIZONTAL_DIRECTION_LABELS:
        raise ValueError(f"unknown horizontal direction: {horizontal}")
    if vertical not in VERTICAL_DIRECTION_LABELS:
        raise ValueError(f"unknown vertical direction: {vertical}")
    return horizontal if vertical == "level" else f"{horizontal}|{vertical}"


def net_turn_deg(actions: Sequence[Action]) -> float:
    """Raw sum of turn degrees (may exceed +/-180; use wrap_deg for heading)."""
    return float(sum(a.deg for a in actions if isinstance(a, Turn)))


def total_forward_m(actions: Sequence[Action]) -> float:
    return float(sum(a.m for a in actions if isinstance(a, Forward)))


# --------------------------------------------------------------------------
# Poses
# --------------------------------------------------------------------------

def _primitive_duration_s(action: Action) -> float:
    if isinstance(action, Turn):
        return abs(float(action.deg)) / float(config.ANGULAR_SPEED_DEG_S)
    return abs(float(action.m)) / float(config.LINEAR_SPEED_M_S)


def _action_amounts_at_progress(
    actions: Sequence[Action], progress: float,
) -> List[float]:
    durations = [_primitive_duration_s(action) for action in actions]
    total_time_s = sum(durations)
    remaining = min(1.0, max(0.0, float(progress))) * total_time_s
    amounts = []
    for duration in durations:
        if duration <= 0.0:
            amount = 1.0 if remaining > 0.0 else 0.0
        elif remaining >= duration:
            amount = 1.0
            remaining -= duration
        else:
            amount = remaining / duration
            remaining = 0.0
        amounts.append(float(amount))
    return amounts


def _fold(actions: Sequence[Action], arc_limit: float) -> Tuple[float, float, float, float]:
    """Walk the path, stopping when cumulative forward arc reaches arc_limit.
    Returns (x, z, heading_rad, arc). When arc_limit is reached mid- (or at
    end-of) a forward leg, we stop immediately without applying any later
    action (the agent has physically stopped there).
    """
    x = z = h = arc = 0.0
    for a in actions:
        if isinstance(a, Turn):
            h += math.radians(a.deg)
        else:  # Forward
            remaining = arc_limit - arc
            if a.m >= remaining:  # reaches / passes the limit inside this leg
                x += remaining * math.sin(h)
                z += remaining * math.cos(h)
                return x, z, h, arc_limit
            x += a.m * math.sin(h)
            z += a.m * math.cos(h)
            arc += a.m
    return x, z, h, arc


def pose_after(actions: Sequence[Action]) -> Tuple[float, float, float]:
    """Full-execution end pose (x, z, heading_deg wrapped to (-180,180])."""
    x, z, h, _ = _fold(actions, math.inf)
    return x, z, wrap_deg(math.degrees(h))


def pose_at_arc(actions: Sequence[Action], arc: float) -> Tuple[float, float, float]:
    """End pose truncated at cumulative forward arc-length `arc`."""
    x, z, h, _ = _fold(actions, float(arc))
    return x, z, wrap_deg(math.degrees(h))


def pose_at_progress(actions: Sequence[Action], progress: float) -> Tuple[float, float, float]:
    """Pose at normalized elapsed time under the configured constant speeds."""
    x = z = h = 0.0
    for action, amount in zip(actions, _action_amounts_at_progress(actions, progress)):
        if amount <= 0:
            break
        if isinstance(action, Turn):
            h += math.radians(action.deg * amount)
        else:
            d = action.m * amount
            x += d * math.sin(h)
            z += d * math.cos(h)
    return x, z, wrap_deg(math.degrees(h))


def arc_at_progress(actions: Sequence[Action], progress: float) -> float:
    """Forward arc executed at normalized physical elapsed time."""
    return float(sum(
        action.m * amount
        for action, amount in zip(
            actions, _action_amounts_at_progress(actions, progress))
        if isinstance(action, Forward)
    ))


def program_progress_at_contact(actions: Sequence[Action], action_index: int,
                                local_arc_m: float) -> float:
    """Map a Forward contact to normalized physical elapsed time."""
    if not actions or not (0 <= action_index < len(actions)):
        raise ValueError("contact action index out of range")
    action = actions[action_index]
    if not isinstance(action, Forward) or action.m <= 0:
        raise ValueError("contact must lie in a positive Forward primitive")
    local_arc = min(float(action.m), max(0.0, float(local_arc_m)))
    elapsed = sum(_primitive_duration_s(value) for value in actions[:action_index])
    elapsed += local_arc / float(config.LINEAR_SPEED_M_S)
    total = sum(_primitive_duration_s(value) for value in actions)
    return float(elapsed / total)


def inside_initial_fov(actions: Sequence[Action], half_fov_deg: float, *,
                       max_arc_m: float = None) -> bool:
    """Whether the executed centreline stays inside the initial view cone.

    ``max_arc_m`` limits the check to the prefix actually executed. A colliding
    program never runs its nominal remainder, so requiring the full nominal path
    to stay in view would reject candidates the physical oracle would accept --
    the formal corridor gate truncates at first contact for the same reason.

    Body radius is intentionally absent here.  The physical and depth oracles
    use the real disc radius, while this predicate asks only whether the path
    being reasoned about remains in the initial camera view.  Corridor coverage
    separately checks whether that view contains enough evidence for the body.
    """
    half_fov = float(half_fov_deg)
    for x, z, _heading, arc in sample_path(actions, config.MARCH_STEP_M):
        if arc <= 0:
            continue
        if max_arc_m is not None and float(arc) > float(max_arc_m) + 1e-9:
            break
        if z <= 0 or abs(math.degrees(math.atan2(x, z))) > half_fov + 1e-9:
            return False
    return True


def _random_action(rng: np.random.Generator, previous: Action = None, *,
                   turns=tuple(config.GEN_TURNS_DEG),
                   forwards=tuple(config.GEN_FORWARDS_M),
                   initial_turns=tuple(config.INITIAL_TURNS_DEG)) -> Action:
    choose_turn = (bool(rng.integers(0, 2)) if previous is None
                   else isinstance(previous, Forward))
    if not choose_turn:
        return Forward(float(rng.choice(forwards)))
    return Turn(float(rng.choice(
        initial_turns if previous is None else turns)))


def validate_alternating_actions(actions: Sequence[Action]) -> None:
    """Raise when adjacent primitives have the same action type."""
    for index, (previous, current) in enumerate(zip(actions, actions[1:]), start=1):
        if isinstance(previous, Turn) == isinstance(current, Turn):
            raise ValueError(
                f"actions {index} and {index + 1} must alternate Turn/Forward")


def validate_physics_actions(
    actions: Sequence[Action], *,
    turns=tuple(config.GEN_TURNS_DEG),
    forwards=tuple(config.GEN_FORWARDS_M),
) -> None:
    """Validate one physical program against the registered main vocabulary."""
    validate_alternating_actions(actions)
    allowed_turns = {float(value) for value in turns}
    allowed_forwards = {float(value) for value in forwards}
    if not any(isinstance(action, Forward) for action in actions):
        raise ValueError("physics action program must contain a forward primitive")
    for action in actions:
        if isinstance(action, Turn):
            value = float(action.deg)
            if not math.isfinite(value) or value not in allowed_turns:
                raise ValueError(f"turn value {value!r} is outside the main vocabulary")
        else:
            value = float(action.m)
            if (not math.isfinite(value) or value <= 0.0 or
                    value not in allowed_forwards):
                raise ValueError(
                    f"forward value {value!r} is outside the positive main vocabulary")


def _action_key(actions: Sequence[Action]) -> tuple:
    return tuple(("turn", action.deg) if isinstance(action, Turn)
                 else ("forward", action.m) for action in actions)


def balanced_action_pool(rng: np.random.Generator,
                         lengths=config.GEN_LENGTHS,
                         pool_per_length: int = None,
                         half_fov_deg: float = config.HFOV_DEG / 2.0,
                         max_attempts: int = None, *,
                         turns=tuple(config.GEN_TURNS_DEG),
                         forwards=tuple(config.GEN_FORWARDS_M),
                         initial_turns=tuple(config.INITIAL_TURNS_DEG),
                         require_initial_fov: bool = True) -> Dict[int, List[ActionSeq]]:
    """Randomly draw benchmark programs and keep paths in the initial FOV.
    Adjacent primitives strictly alternate between Turn and Forward. Duplicate
    and out-of-view programs are rejected.
    """
    if pool_per_length is None:
        pool_per_length = config.KEEP_PER_LENGTH * config.POOL_FACTOR
    pools: Dict[int, List[ActionSeq]] = {}
    for length in lengths:
        candidates: List[ActionSeq] = []
        seen = set()
        attempts = 0
        limit = max_attempts or max(
            1000, int(pool_per_length) * config.ACTION_REJECTION_FACTOR)
        while len(candidates) < int(pool_per_length) and attempts < limit:
            attempts += 1
            seq: ActionSeq = []
            for _ in range(int(length)):
                seq.append(_random_action(
                    rng, seq[-1] if seq else None,
                    turns=turns, forwards=forwards,
                    initial_turns=initial_turns))
            validate_alternating_actions(seq)
            key = _action_key(seq)
            if (key in seen or not any(isinstance(a, Forward) for a in seq) or
                    (require_initial_fov and
                     not inside_initial_fov(seq, float(half_fov_deg)))):
                continue
            seen.add(key)
            candidates.append(seq)
        pools[int(length)] = candidates
    return pools


_T = TypeVar("_T")


# --------------------------------------------------------------------------
# Path sampling (for collision march & view-exit)
# --------------------------------------------------------------------------

def sample_path(actions: Sequence[Action], step: float) -> List[Tuple[float, float, float, float]]:
    """Dense centreline samples of the path.
    Returns a list of (x, z, heading_rad, arc) tuples. Sample 0 is always the
    origin (0, 0, 0, 0). Each forward leg emits samples at k*step plus the
    exact leg endpoint. Turns emit no sample (zero arc length).
    """
    samples: List[Tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)]
    x = z = h = arc = 0.0
    for a in actions:
        if isinstance(a, Turn):
            h += math.radians(a.deg)
            continue
        m = a.m
        if m <= 0:
            continue
        # arc offsets within this leg: step, 2*step, ... (< m), then exact m.
        offs: List[float] = []
        k = 1
        while k * step < m - 1e-9:
            offs.append(k * step)
            k += 1
        offs.append(m)
        sh, ch = math.sin(h), math.cos(h)
        for d in offs:
            samples.append((x + d * sh, z + d * ch, h, arc + d))
        x += m * sh
        z += m * ch
        arc += m
    return samples


def contact_action_index(actions: Sequence[Action], arc: float):
    """Which action a collision at cumulative forward `arc` happens during.
    Turns consume no arc, so a collision always lands inside a Forward leg.
    Returns (idx, local_arc) — `idx` is the 0-based index in `actions` of that
    Forward, `local_arc` how far into it (m). Returns (None, None) if `arc` is
    past the total forward (should not happen for a real contact).
    """
    cum = 0.0
    for i, a in enumerate(actions):
        if isinstance(a, Turn):
            continue
        if arc <= cum + a.m + 1e-9:
            return i, float(arc - cum)
        cum += a.m
    return None, None


def forward_leg_location(actions: Sequence[Action],
                         arc_m: float) -> ForwardLegLocation:
    """Locate a cumulative forward arc in a program's 1-based Forward stages.
    An arc exactly at a Forward endpoint belongs to that closing leg. Turns
    consume no arc and therefore never define a termination stage.
    """
    arc = float(arc_m)
    total = total_forward_m(actions)
    if not math.isfinite(arc) or arc < 0.0 or arc > total + 1e-9:
        raise ValueError(
            f"forward arc {arc_m!r} is outside [0, {total:g}]")
    cumulative = 0.0
    leg_number = 0
    for action_index, action in enumerate(actions):
        if isinstance(action, Turn):
            continue
        if action.m <= 0.0:
            raise ValueError("forward legs must have positive length")
        leg_number += 1
        endpoint = cumulative + float(action.m)
        if arc <= endpoint + 1e-9:
            local = min(float(action.m), max(0.0, arc - cumulative))
            return ForwardLegLocation(
                action_index=int(action_index),
                forward_leg_number=int(leg_number),
                cumulative_before_leg_m=float(cumulative),
                distance_into_leg_m=float(local),
                leg_length_m=float(action.m),
            )
        cumulative = endpoint
    raise ValueError("forward arc cannot be located in an empty action program")
