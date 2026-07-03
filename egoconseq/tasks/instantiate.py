"""
egoconseq.tasks.instantiate
============================
Factory helpers for EgoConseq-Bench O5 (footprint counterfactual).

New O5 design (single-robot binary contact + counterfactual grouping)
----------------------------------------------------------------------
Each case describes exactly ONE cylindrical-chassis robot with a given
radius.  The question is binary: "会接触 / 不会接触".

The counterfactual is NOT inside one question — it is constructed ACROSS
cases: same image + same pose + same forward horizon H, multiple cases
with DIFFERENT chassis diameters, tied by a shared ``group_id``.
The flip metric is measured across the group.

Key design invariants
---------------------
RED LINE (design §6):
    The action horizon for O5 is a FIXED physical path in METRES.
    Stored as horizon_m in the action dict with
    horizon_reference="metric_fixed".  The horizon must NEVER be derived
    from body-width (e.g. "N body-widths") — doing so would confound the
    causal claim that only diameter changes.

Margin rule (single body):
    A case is accepted only when the body is clearly on one side of the
    decision boundary H:

        abs(d_safe_m - H) >= radius_m

    Cases where the body sits ambiguously close to the boundary are
    rejected (make_o5_case returns None).

GT evidence (anti-bug invariant):
    The rule "contact iff d_safe < H" together with the raw values
    (d_safe_m, horizon_m) is stored in tags["gt_evidence"] so any
    reviewer can independently verify the label without the pipeline code.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from egoconseq.manifest import Case
from egoconseq.tasks.prompts import o5_prompt


# ── label function ────────────────────────────────────────────────────────────

def o5_label(d_safe_m: float, horizon_m: float) -> str:
    """Assign binary contact label for a single robot.

    Parameters
    ----------
    d_safe_m:
        Safe travel distance (m) before first contact with any obstacle.
    horizon_m:
        The fixed forward path length the robot attempts to travel.

    Returns
    -------
    "contact"
        d_safe_m < horizon_m: the robot will hit an obstacle before
        reaching the horizon.
    "no_contact"
        d_safe_m >= horizon_m: the robot clears the horizon safely.
    """
    if d_safe_m < horizon_m:
        return "contact"
    return "no_contact"


# ── factory function ──────────────────────────────────────────────────────────

def make_o5_case(
    d_safe_m: float,
    radius_m: float,
    horizon_m: float,
    group_id: str,
    **meta: Any,
) -> Optional[Case]:
    """Build a single-robot O5 Case, or return None if the margin gate fails.

    Parameters
    ----------
    d_safe_m:
        Safe travel distance (m) for this robot before first contact.
    radius_m:
        Body radius (m) of the cylindrical-chassis robot.
        Diameter = 2 * radius_m.
    horizon_m:
        The fixed forward distance (metres) — RED LINE: must be a fixed
        metric path (design §6), not derived from body width.
    group_id:
        Shared identifier for the counterfactual group (same image/pose/H,
        different diameters).
    **meta:
        Optional metadata forwarded to Case fields:
        scene_id, episode_id, image_path, question, pose,
        sensor_profile, geometry_tag, case_id, etc.

    Returns
    -------
    Case | None
        A fully populated Case on success, or None when the margin rule
        is violated.

    Margin rule
    -----------
        abs(d_safe_m - horizon_m) >= radius_m

    This ensures the robot is unambiguously on its correct side of H by
    at least its own footprint radius.

    GT evidence
    -----------
    Stored in tags["gt_evidence"] = {
        "d_safe_m": d_safe_m,
        "horizon_m": horizon_m,
        "rule": "contact iff d_safe < H"
    }
    so a reviewer can independently verify the label.
    """
    # ── 1. Margin check ───────────────────────────────────────────────────────
    if abs(d_safe_m - horizon_m) < radius_m:
        return None

    # ── 2. Compute binary label ───────────────────────────────────────────────
    label = o5_label(d_safe_m, horizon_m)

    # ── 3. Extract well-known meta fields ─────────────────────────────────────
    scene_id       = meta.pop("scene_id",       None)
    episode_id     = meta.pop("episode_id",     None)
    image_path     = meta.pop("image_path",     None)
    question       = meta.pop("question",       None)
    pose           = meta.pop("pose",           [])
    sensor_profile = meta.pop("sensor_profile", {})
    geometry_tag   = meta.pop("geometry_tag",   "narrow-gap")
    case_id        = meta.pop("case_id",        f"O5-{uuid.uuid4().hex[:8]}")
    # Remaining kwargs are silently ignored (YAGNI)

    # ── 4. Build the question if not provided ─────────────────────────────────
    if question is None:
        question = o5_prompt(2 * radius_m, horizon_m)

    # ── 5. Build the Case ─────────────────────────────────────────────────────
    case = Case(
        case_id=case_id,
        operation_id="O5",
        readout_tag="binary_contact",
        group_id=group_id,
        scene_id=scene_id,
        episode_id=episode_id,
        image_path=image_path,
        question=question,
        pose=pose,
        sensor_profile=sensor_profile,
        # ── body: single-robot cylindrical chassis ────────────────────────────
        body={
            "radius_m": radius_m,
            "diameter_m": 2 * radius_m,
        },
        # ── action: FIXED metric horizon (RED LINE §6) ────────────────────────
        action={
            "type": "forward",
            "horizon_m": horizon_m,
            "horizon_reference": "metric_fixed",
        },
        # ── answer schema: binary_contact ─────────────────────────────────────
        answer={
            "answer_type": "binary_contact",
            "options": ["contact", "no_contact"],
            "label": label,
        },
        # ── GT safe distance (evidence anchor) ───────────────────────────────
        d_safe_visible_m=d_safe_m,
        # ── tags: GT evidence + geometry tag ─────────────────────────────────
        tags={
            "geometry_tag": geometry_tag,
            "gt_evidence": {
                "d_safe_m": d_safe_m,
                "horizon_m": horizon_m,
                "rule": "contact iff d_safe < H",
            },
        },
    )
    return case


# ── group flip detection ──────────────────────────────────────────────────────

def o5_group_has_flip(cases_in_group: List[Case]) -> bool:
    """Return True iff the group contains at least one label flip.

    A flip means the group has BOTH "contact" and "no_contact" GT labels —
    i.e. some chassis sizes contact an obstacle and some don't, producing a
    genuine counterfactual pair.

    By physical monotonicity, larger-radius bodies encounter obstacles
    sooner (smaller d_safe), so larger diameters contact first.

    Parameters
    ----------
    cases_in_group:
        List of Case objects sharing the same group_id (same image/pose/H,
        different radii).

    Returns
    -------
    bool
        True iff GT labels are NOT all identical.
    """
    if len(cases_in_group) <= 1:
        return False

    labels = {c.answer.get("label") for c in cases_in_group if c.answer}
    # A flip exists iff there are at least 2 distinct labels
    return len(labels) >= 2
