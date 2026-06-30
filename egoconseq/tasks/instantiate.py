"""
egoconseq.tasks.instantiate
============================
Factory helpers for EgoConseq-Bench O5 (footprint counterfactual).

Key design invariants
---------------------
RED LINE (design §6):
    The action horizon for O5 is a FIXED physical path in METRES, identical
    for BOTH body sizes.  It is stored as horizon_m in the action dict, with
    horizon_reference="metric_fixed".  The horizon must NEVER be derived from
    body-width (e.g. "N body-widths") — doing so would confound the causal
    claim that only width changes.

Margin rule (per-body):
    A case is accepted only when EACH body is clearly on one side of the
    decision boundary H:

        abs(d_safe_X - H) >= 0.5 * (2 * r_X)   for X ∈ {small, large}

    This equals r_X — one body-radius — as the minimum distance from d_safe
    to H.  Cases where either body sits ambiguously close to the boundary
    are rejected (make_o5 returns None).

Monotonicity invariant:
    By physical geometry, d_safe_large <= d_safe_small (a wider body
    encounters obstacles sooner or at the same point as a narrower one).
    If d_safe_large > d_safe_small, the caller has made an error; o5_label
    returns the sentinel "INVALID" so the pipeline can flag and discard.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from egoconseq.manifest import Case


# ── label function ────────────────────────────────────────────────────────────

def o5_label(
    d_safe_small: float,
    d_safe_large: float,
    horizon_m: float,
) -> str:
    """Assign the O5 class label by comparing EACH d_safe to the shared horizon.

    Parameters
    ----------
    d_safe_small:
        Safe travel distance (m) for the narrow body before first contact.
    d_safe_large:
        Safe travel distance (m) for the wide body before first contact.
    horizon_m:
        The shared fixed metric path length both bodies travel.

    Returns
    -------
    "both"
        Both d_safe >= horizon_m → neither body contacts an obstacle.
    "small_only"
        d_safe_small >= horizon_m but d_safe_large < horizon_m → only the
        small body passes; the large body contacts (this is the flip case).
    "neither"
        Both d_safe < horizon_m → both bodies contact an obstacle.
    "INVALID"
        d_safe_large > d_safe_small, which violates the physical monotonicity
        invariant (a wider body cannot have more clearance in the same scene).
        Callers should treat this as a data error and discard the case.
    """
    # Monotonicity check: large body must never have MORE clearance than small.
    if d_safe_large > d_safe_small:
        return "INVALID"

    small_passes = d_safe_small >= horizon_m
    large_passes = d_safe_large >= horizon_m

    if small_passes and large_passes:
        return "both"
    elif small_passes and not large_passes:
        return "small_only"
    else:
        # Neither passes (large_passes being True while small_passes is False
        # is excluded by the monotonicity guard above).
        return "neither"


# ── factory function ──────────────────────────────────────────────────────────

def make_o5(
    d_safe_small: float,
    d_safe_large: float,
    horizon_m: float,
    r_small: float,
    r_large: float,
    **meta: Any,
) -> Optional[Case]:
    """Build an O5 Case, or return None if any quality gate fails.

    Parameters
    ----------
    d_safe_small:
        Safe travel distance (m) for the narrow body.
    d_safe_large:
        Safe travel distance (m) for the wide body.
    horizon_m:
        The shared fixed forward distance (metres) — RED LINE: this is the
        SAME physical path for both bodies (design §6).
    r_small:
        Body radius (m) of the narrow robot.  Width = 2 * r_small.
    r_large:
        Body radius (m) of the wide robot.  Width = 2 * r_large.
    **meta:
        Optional metadata forwarded to Case fields:
        group_id, scene_id, episode_id, image_path, question, pose,
        sensor_profile, geometry_tag, etc.

    Returns
    -------
    Case | None
        A fully populated Case on success, or None when:
        - o5_label returns "INVALID" (monotonicity violation), OR
        - The per-body margin rule is violated for either body.

    Margin rule (per-body)
    ----------------------
        abs(d_safe_X - H) >= 0.5 * (2 * r_X)   ≡   abs(d_safe_X - H) >= r_X

    This ensures each body is unambiguously on its correct side of H.
    """
    # ── 1. Compute label (also checks monotonicity) ───────────────────────────
    label = o5_label(d_safe_small, d_safe_large, horizon_m)
    if label == "INVALID":
        return None

    # ── 2. Per-body margin check (each body must clear its own radius from H) ──
    margin_small = abs(d_safe_small - horizon_m)
    margin_large = abs(d_safe_large - horizon_m)
    threshold_small = 0.5 * (2 * r_small)   # == r_small
    threshold_large = 0.5 * (2 * r_large)   # == r_large

    if margin_small < threshold_small or margin_large < threshold_large:
        return None

    # ── 3. Extract well-known meta fields ─────────────────────────────────────
    group_id    = meta.pop("group_id",      None)
    scene_id    = meta.pop("scene_id",      None)
    episode_id  = meta.pop("episode_id",    None)
    image_path  = meta.pop("image_path",    None)
    question    = meta.pop("question",      None)
    pose        = meta.pop("pose",          [])
    sensor_profile = meta.pop("sensor_profile", {})
    geometry_tag   = meta.pop("geometry_tag",   "narrow-gap")
    # Remaining kwargs are ignored (YAGNI)

    case_id = meta.pop("case_id", f"O5-{uuid.uuid4().hex[:8]}")

    # ── 4. Build the Case ─────────────────────────────────────────────────────
    case = Case(
        case_id=case_id,
        operation_id="O5",
        readout_tag="pair_flip",
        group_id=group_id,
        scene_id=scene_id,
        episode_id=episode_id,
        image_path=image_path,
        question=question,
        pose=pose,
        sensor_profile=sensor_profile,
        # ── body: carry both radii ────────────────────────────────────────────
        body={
            "radius_small_m": r_small,
            "radius_large_m": r_large,
        },
        # ── action: FIXED metric horizon (RED LINE §6) ────────────────────────
        action={
            "type": "forward",
            "horizon_m": horizon_m,
            "horizon_reference": "metric_fixed",
        },
        # ── answer schema ─────────────────────────────────────────────────────
        answer={
            "answer_type": "pair_flip",
            "options": ["both", "small_only", "neither"],
            "label": label,
        },
        # ── tags ──────────────────────────────────────────────────────────────
        tags={
            "geometry_tag": geometry_tag,
        },
    )
    return case
