"""A4 direction questions at an in-progress Forward checkpoint."""

from __future__ import annotations

import hashlib

from pipeline import actions


FRACTIONS = (0.25, 0.5, 0.75)


def _digest(*parts) -> str:
    return hashlib.sha256(
        ":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def build_forward_checkpoint(program, *, forward_stage: int,
                             fraction: float) -> dict:
    """Return one strictly interior point of a 1-based Forward stage."""
    if fraction not in FRACTIONS:
        raise ValueError("checkpoint fraction must be 0.25, 0.5, or 0.75")
    if (not isinstance(forward_stage, int) or
            isinstance(forward_stage, bool) or forward_stage <= 0):
        raise ValueError("forward_stage must be a positive integer")
    cumulative = 0.0
    stage = 0
    for primitive_index, action in enumerate(program):
        if not isinstance(action, actions.Forward):
            continue
        stage += 1
        if stage == forward_stage:
            arc = cumulative + fraction * action.m
            x, z, heading = actions.pose_at_arc(program, arc)
            return {
                "action_index": primitive_index + 1,
                "forward_stage": stage,
                "fraction": fraction,
                "arc_m": arc,
                "pose": {"x": x, "z": z, "heading_deg": heading},
            }
        cumulative += action.m
    raise ValueError("forward_stage is outside the action program")


def select_forward_checkpoint(program, *, seed: str) -> dict:
    """Choose one Forward and one interior fraction without reading GT."""
    stages = sum(isinstance(action, actions.Forward) for action in program)
    if not stages:
        raise ValueError("checkpoint direction requires a Forward action")
    digest = bytes.fromhex(_digest(seed))
    stage = int.from_bytes(digest[:8], "big") % stages + 1
    fraction = FRACTIONS[int.from_bytes(digest[8:16], "big") % len(FRACTIONS)]
    return build_forward_checkpoint(
        program, forward_stage=stage, fraction=fraction)
