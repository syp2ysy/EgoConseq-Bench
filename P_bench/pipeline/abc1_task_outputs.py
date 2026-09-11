"""Freeze certified collision-task outputs; surface tasks share surface_points."""

from __future__ import annotations

from typing import Iterable, Mapping

from pipeline import a1_common_support
from pipeline import a2
from pipeline import objects


def build_task_outputs(
        record: Mapping[str, object], outcome: Mapping[str, object], *,
        record_uid: str, proposal_protocol: str | None,
        proposal_variant: str | None,
        allowed_tasks: Iterable[str]) -> dict[str, dict]:
    """Return only the case-level ABC task outputs this outcome supports."""
    allowed = set(allowed_tasks)
    collision = bool(outcome["physical"]["collision"])
    result = {}
    if ("A1" in allowed and
            a1_common_support.v3_source_rejection_for(
                proposal_protocol, proposal_variant) is None):
        result["A1"] = {
            "answer": "collision" if collision else "no_collision"}

    if collision:
        contact_index = outcome["physical"].get("contact_action_index")
        if "A2" in allowed and isinstance(contact_index, int) and not isinstance(
                contact_index, bool):
            try:
                certificate = a2.build_design_certificate(
                    dict(outcome),
                    collision_action_index_1based=contact_index + 1)
            except (KeyError, TypeError, ValueError):
                certificate = None
            if certificate is not None:
                cell = certificate["cell"]
                result["A2"] = {
                    "collision_action_index_1based": int(
                        certificate["collision_action_index_1based"]),
                    "forward_ordinal_1based": int(
                        cell["forward_ordinal_1based"]),
                    "distance_rank": str(cell["distance_rank"]),
                    "minimum_action_boundary_margin_m": float(
                        certificate["minimum_action_boundary_margin_m"]),
                }
        if "A3" in allowed:
            inventory = record.get("visible_entities")
            category, choices, reason = objects.a3_category_evidence(
                dict(record), dict(outcome["shared_oracle_stability"]),
                inventory=(list(inventory)
                           if isinstance(inventory, list) else None))
            contact = outcome["physical"].get("contact") or {}
            if reason is None and contact.get("instance_id") is not None:
                result["A3"] = {
                    "instance_id": int(contact["instance_id"]),
                    "category": str(category),
                    "choices": choices,
                }
                if contact.get("source_category"):
                    result["A3"]["source_category"] = str(contact["source_category"])
                if contact.get("obstacle_identity"):
                    result["A3"]["contact_component"] = str(contact["obstacle_identity"])
    return result
