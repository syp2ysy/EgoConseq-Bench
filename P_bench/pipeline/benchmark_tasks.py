"""Question registry access, eligibility, evidence, and answer derivation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List

from pipeline import actions as action_geometry
from pipeline import c1_counterfactual
from pipeline import capability_contracts
from pipeline import config
from pipeline import gs_semantic
from pipeline import objects as object_fields
from pipeline import outcome as outcome_fields
from pipeline import record as record_fields
from pipeline import viz
from pipeline.benchmark import (
    A_TASK_QUESTIONS, B_TASK_QUESTIONS, PUBLIC_HEIGHT_DECIMALS, Eligibility,
    SharedVisibleSpaceEvidence,
    shared_visible_space_certificate as _a_shared_certificate,
)


_CAPABILITY_TASK = {
    "A1_collision": "A1",
    "A2_collision_step_grounding": "A2",
    "A3_contact_object": "A3",
    "B1_endpoint_distance": "B1",
    "B2_endpoint_direction": "B2",
}


def _capability_envelope_rejection(
        task_id: str, record: Dict) -> Eligibility | None:
    """Consume the exact GS v18 capability snapshot before task GT."""
    if record.get("schema_version") != record_fields.V18_SCHEMA_VERSION:
        return None
    source = record.get("source") or {}
    dataset = source.get("source_dataset")
    try:
        expected = capability_contracts.validate_snapshot(
            record.get("authority_surface_capability"), dataset)
        task = _CAPABILITY_TASK[task_id]
    except (KeyError, TypeError, ValueError):
        return Eligibility(False, "capability_unavailable")
    if expected["task_statuses"][task].get("status") != "available":
        return Eligibility(False, "capability_unavailable")
    if dataset == "gs":
        try:
            available = gs_semantic.scene_task_available(
                record.get("gs_scene_capability"), source, task)
        except (KeyError, TypeError, ValueError):
            return Eligibility(False, "capability_unavailable")
        if not available:
            return Eligibility(False, "capability_unavailable")
    return None


def a_candidate_eligibility(
        task_id: str, record: Dict, outcome: Dict, *,
        shared_evidence: SharedVisibleSpaceEvidence | None = None
        ) -> Eligibility:
    """Fail-closed v16 eligibility over the shared A certificate."""
    if task_id not in A_TASK_QUESTIONS:
        return Eligibility(False, "unknown_a_task")
    capability_rejection = _capability_envelope_rejection(task_id, record)
    if capability_rejection is not None:
        return capability_rejection
    certificate, reason = (
        _a_shared_certificate(record, outcome) if shared_evidence is None
        else shared_evidence.require(record, outcome))
    if certificate is None:
        return Eligibility(False, reason)
    summary = certificate["summary"]
    margins = {"shared_certificate_sha256": certificate["sha256"]}
    if task_id == "A1_collision":
        return Eligibility(True, margins=margins)
    if summary["collision"] is not True:
        return Eligibility(False, "collision_required")
    if task_id == "A2_collision_step_grounding":
        actions = action_geometry.parse_actions(outcome.get("actions") or [])
        forward_count = sum(
            isinstance(action, action_geometry.Forward) for action in actions)
        if forward_count < 2:
            return Eligibility(False, "multiple_forward_actions_required")
        index = summary.get("original_action_index")
        nominal = (
            certificate.get("version") == "nominal-oracle.v1" and
            summary.get("evaluation") == "nominal")
        if ((not nominal and
             summary.get("original_action_index_stable") is not True) or
                not isinstance(index, int) or isinstance(index, bool) or
                not 1 <= index <= len(actions) or
                not isinstance(actions[index - 1], action_geometry.Forward)):
            return Eligibility(False, "collision_action_index_unstable")
        margins["original_action_index"] = int(index)
        return Eligibility(True, margins=margins)
    category, choices, reason = object_fields.a3_category_evidence(
        record, certificate)
    if reason is not None:
        return Eligibility(False, reason)
    margins["contact_category"] = category
    margins["visible_category_choice_count"] = len(choices)
    return Eligibility(True, margins=margins)


@dataclass(frozen=True)
class ACandidateEvidence:
    """One task eligibility result bound to its record and outcome objects."""

    task_id: str
    record_identity: int
    outcome_identity: int
    eligibility: Eligibility

    def require(self, task_id: str, record: Dict, outcome: Dict) -> Eligibility:
        if (self.task_id != task_id or self.record_identity != id(record) or
                self.outcome_identity != id(outcome)):
            raise ValueError("A candidate evidence belongs to another task")
        return self.eligibility


def build_a_candidate_evidence(
        task_id: str, record: Dict, outcome: Dict, *,
        shared_evidence: SharedVisibleSpaceEvidence | None = None
        ) -> ACandidateEvidence:
    """Bind one A eligibility computation for projection reuse."""
    return ACandidateEvidence(
        task_id, id(record), id(outcome),
        a_candidate_eligibility(
            task_id, record, outcome, shared_evidence=shared_evidence))


def _a_bound_eligibility(
        task_id: str, record: Dict, outcome: Dict,
        evidence: ACandidateEvidence | None) -> Eligibility:
    if evidence is None:
        return a_candidate_eligibility(task_id, record, outcome)
    if not isinstance(evidence, ACandidateEvidence):
        raise TypeError("evidence must be ACandidateEvidence")
    return evidence.require(task_id, record, outcome)


def a_candidate_answer(
        task_id: str, record: Dict, outcome: Dict, *,
        evidence: ACandidateEvidence | None = None) -> tuple[str, list]:
    """Rederive one Closed Exact answer without reading simulator substeps."""
    eligibility = _a_bound_eligibility(
        task_id, record, outcome, evidence)
    if not eligibility.eligible:
        raise ValueError(f"ineligible {task_id}: {eligibility.reason}")
    summary = outcome["shared_oracle_stability"]["summary"]
    if task_id == "A1_collision":
        answer = "collision" if summary["collision"] else "no_collision"
        return answer, ["collision", "no_collision"]
    if task_id == "A2_collision_step_grounding":
        actions = action_geometry.parse_actions(outcome.get("actions") or [])
        choices = [
            f"action_{index}"
            for index, action in enumerate(actions, 1)
            if isinstance(action, action_geometry.Forward)
        ]
        return f"action_{summary['original_action_index']}", choices
    answer, choices, reason = object_fields.a3_category_evidence(
        record, outcome["shared_oracle_stability"])
    if reason is not None:
        raise ValueError(f"ineligible A3_contact_object: {reason}")
    return answer, choices


def build_a_candidate_projection(*, task_id: str, record: Dict,
                                 outcome: Dict, image: str,
                                 stable_id, expected_raw_image_sha256: str,
                                 asset_dir=None,
                                 authenticated_rgb=None,
                                 evidence: ACandidateEvidence | None = None
                                 ) -> tuple[dict, dict]:
    """Assemble A using the raw digest as the authoritative relocation key."""
    sensor = record.get("sensor")
    if not isinstance(sensor, dict):
        raise ValueError("record sensor resolution is invalid")
    raw_asset = viz.authenticate_raw_rgb_image(
        image, expected_sha256=expected_raw_image_sha256,
        expected_resolution=sensor.get("resolution"),
        authenticated=authenticated_rgb)
    eligibility = _a_bound_eligibility(
        task_id, record, outcome, evidence)
    if not eligibility.eligible:
        raise ValueError(f"ineligible {task_id}: {eligibility.reason}")
    canonical_answer, raw_choices = a_candidate_answer(
        task_id, record, outcome, evidence=evidence)
    choices = [dict(value) if isinstance(value, dict) else
               {"id": str(value), "text": str(value)} for value in raw_choices]
    certificate = outcome["shared_oracle_stability"]
    oracle_ref = {
        "frame_id": record["frame_id"], "outcome_id": outcome.get("outcome_id"),
        "base_rollout_key": outcome.get("base_rollout_key"),
        "shared_certificate_sha256": certificate["sha256"],
    }
    item_id = "a-" + stable_id(
        record.get("observation_id") or record["frame_id"],
        outcome.get("outcome_id"), task_id)
    model_input = {
        "initial_rgb": str(image),
        "initial_rgb_sha256": raw_asset.sha256,
        "camera_height_above_visible_floor_m": round(
            float(record["camera_height_above_visible_floor_m"]),
            PUBLIC_HEIGHT_DECIMALS),
        "hfov_deg": float(record["sensor"]["hfov_deg"]),
        "vfov_deg": float(record["sensor"]["vfov_deg"]),
        "body_radius_m": float(outcome["body"]["radius_m"]),
        "actions": list(outcome.get("actions") or []),
    }
    contact_category = None
    input_asset = {
        "marked": False,
        "path": str(image),
        "sha256": raw_asset.sha256,
        "raw_path": str(image),
        "raw_sha256": model_input["initial_rgb_sha256"],
        "binding": "sha256_authoritative_relocated",
        "record_image_path": record.get("image_path"),
    }
    if task_id == "A3_contact_object":
        contact_category = canonical_answer
    item = {
        "id": item_id, "result_head": "A", "task_id": task_id,
        "question": A_TASK_QUESTIONS[task_id], "answer_format": "closed_exact",
        "model_input": model_input, "choices": choices, "oracle_ref": oracle_ref,
    }
    private = {
        "id": item_id, "task_id": task_id,
        "canonical_answer": canonical_answer, "oracle_ref": dict(oracle_ref),
        "contact_category": contact_category,
        "input_asset": input_asset,
    }
    return item, private



def c_candidate_selection(
        record: Dict, outcome: Dict, *,
        asset_root, selection_index=None) -> tuple[dict | None, str]:
    """Build the sole C1 option family from certified action neighbours."""
    return c1_counterfactual.candidate_selection_or_reason(
        record, outcome, asset_root=asset_root,
        selection_index=selection_index)
