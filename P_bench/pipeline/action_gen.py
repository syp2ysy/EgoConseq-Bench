"""Action-sequence generation: length-bucketed, in-FOV, no-degenerate pools."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from pipeline import config, perception
from pipeline.actions import Turn, Forward, ActionSeq, sample_path

_FOV_HALF = config.FOV_HALF_DEG
_in_cone = perception.in_cone     # (x, z) -> bool, within horizontal FOV cone


# --------------------------------------------------------------------------
# in-FOV, length-bucketed generation
# --------------------------------------------------------------------------

def path_stays_in_fov(acts: ActionSeq) -> bool:
    """True iff every non-origin swept-centreline sample stays in the FOV cone.

    Equivalent to consequence.compute_view_exit's `path_exit_arc_m is None`, but
    frame-independent (pure geometry) so it can pre-filter before any Habitat call.
    Pure-turn sequences emit no forward samples, so they are trivially in-FOV.
    """
    for i, (x, z, _h, _arc) in enumerate(sample_path(acts, config.MARCH_STEP_M)):
        if i == 0:
            continue
        if not _in_cone(x, z):
            return False
    return True


def _seq_key(acts: ActionSeq):
    return tuple(("T", a.deg) if isinstance(a, Turn) else ("F", a.m) for a in acts)


def _valid_combo(acts: ActionSeq) -> bool:
    """No two adjacent Forwards (redundant longer move); no two adjacent same-sign
    Turns (redundant bigger turn -> consecutive turns must alternate direction)."""
    for a, b in zip(acts, acts[1:]):
        if isinstance(a, Forward) and isinstance(b, Forward):
            return False
        if isinstance(a, Turn) and isinstance(b, Turn) and (a.deg > 0) == (b.deg > 0):
            return False
    return True


def _gen_candidate(rng: np.random.Generator, length: int) -> ActionSeq:
    """One length-`length` sequence, biased to satisfy the combo constraints:
    turns keep |heading| <= FOV_HALF and differ in sign from the previous turn;
    a forward never follows a forward. Dead ends break early (rejected downstream)."""
    heading = 0.0
    acts: ActionSeq = []
    prev = None            # 'T' / 'F'
    last_sign = 0
    for _ in range(length):
        allowed = [d for d in config.GEN_TURNS_DEG
                   if abs(heading + d) <= _FOV_HALF
                   and not (prev == "T" and (d > 0) == (last_sign > 0))]
        forward_ok = prev != "F"
        want_turn = bool(rng.integers(0, 2))
        if (want_turn or not forward_ok) and allowed:
            d = float(rng.choice(allowed))
            heading += d
            acts.append(Turn(d)); prev, last_sign = "T", d
        elif forward_ok:
            acts.append(Forward(float(rng.choice(config.GEN_FORWARDS_M)))); prev, last_sign = "F", 0
        else:
            break          # forward blocked and no legal turn -> incomplete, rejected
    return acts


def fov_bucketed_pool(rng: np.random.Generator,
                      lengths=config.GEN_LENGTHS,
                      pool_per_length: int = None) -> Dict[int, List[ActionSeq]]:
    """For each length L, rejection-sample de-duplicated in-FOV sequences.

    Constraints: correct primitive count L; length 1 = a single Turn or Forward;
    length >=2 requires >=1 Forward AND >=1 Turn (no degenerate pure runs); the
    whole swept centreline stays in the FOV cone (`path_stays_in_fov`).
    Length-1 has only len(GEN_TURNS_DEG)+len(GEN_FORWARDS_M) distinct sequences,
    so pool[1] may be smaller than requested (that is fine).
    """
    if pool_per_length is None:
        pool_per_length = config.KEEP_PER_LENGTH * config.POOL_FACTOR
    pool: Dict[int, List[ActionSeq]] = {}
    for L in lengths:
        seen = set()
        seqs: List[ActionSeq] = []
        max_attempts = pool_per_length * 300 + 2000
        for _ in range(max_attempts):
            if len(seqs) >= pool_per_length:
                break
            acts = _gen_candidate(rng, L)
            if len(acts) != L:
                continue
            n_fwd = sum(isinstance(a, Forward) for a in acts)
            n_turn = sum(isinstance(a, Turn) for a in acts)
            if L >= 2 and (n_fwd < 1 or n_turn < 1):
                continue
            if not _valid_combo(acts):
                continue
            if not path_stays_in_fov(acts):
                continue
            key = _seq_key(acts)
            if key in seen:
                continue
            seen.add(key)
            seqs.append(acts)
        pool[L] = seqs
    return pool
