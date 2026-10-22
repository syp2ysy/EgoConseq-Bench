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

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, TypeVar, Union




# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    deg: float


@dataclass(frozen=True)
class Forward:
    m: float


Action = Union[Turn, Forward]
ActionSeq = List[Action]


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


# --------------------------------------------------------------------------
# Scalar summaries
# --------------------------------------------------------------------------

def wrap_deg(d: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    r = (d + 180.0) % 360.0 - 180.0
    return 180.0 if r == -180.0 else r


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


# --------------------------------------------------------------------------
# Poses
# --------------------------------------------------------------------------


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


