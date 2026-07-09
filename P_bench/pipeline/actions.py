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
from typing import List, Sequence, Tuple, Union


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
# Scalar summaries
# --------------------------------------------------------------------------

def wrap_deg(d: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    r = (d + 180.0) % 360.0 - 180.0
    return 180.0 if r == -180.0 else r


def net_turn_deg(actions: Sequence[Action]) -> float:
    """Raw sum of turn degrees (may exceed +/-180; use wrap_deg for heading)."""
    return float(sum(a.deg for a in actions if isinstance(a, Turn)))


def total_forward_m(actions: Sequence[Action]) -> float:
    return float(sum(a.m for a in actions if isinstance(a, Forward)))


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
