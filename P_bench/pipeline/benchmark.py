"""Pure item construction for the embodied future benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pipeline import consensus as consensus_fields
from pipeline import capability_contracts
from pipeline import config
from pipeline import dataset_contracts
from pipeline import gs_semantic
from pipeline import record as record_fields
from pipeline import rollout


EVIDENCE_PROTOCOL_VERSION = rollout.EVIDENCE_PROTOCOL_VERSION
OBSERVATION_SPEC = {
    "modality": "monocular_rgb",
    "projection": "pinhole",
    "resolution_px": list(config.resolution()),
    "pixel_aspect_ratio": 1.0,
    "per_item_calibration": [
        "camera_height_above_visible_floor_m",
        "hfov_deg",
        "vfov_deg",
    ],
    "camera_height_semantics": "optical_center_height_above_floor",
}
# Machine-readable public geometry/execution contract.  Per-item numeric
# values live in model_input; this declares what those values mean so question
# paraphrases cannot silently change the reference point or coordinate frame.
PUBLIC_INPUT_CONTRACT = {
    "schema": "egoconseq.public-input-contract.v1",
    "body": {
        "shape": "centered_planar_disc",
        "radius_semantics": "ground_plane_footprint_radius_m",
        "rotation_pivot": "disc_center",
        "height_modelled": False,
        "collision_scope": "horizontal_ground_plane_only",
    },
    "camera": {
        "projection": "pinhole",
        "resolution_px": list(config.resolution()),
        "principal_point": "image_center",
        "optical_center_horizontal_offset_m": [0.0, 0.0],
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "yaw_alignment": "robot_heading",
        "height_semantics": "optical_center_height_above_visible_floor_m",
        "intrinsics_from": ["resolution_px", "hfov_deg", "vfov_deg"],
    },
    "action_program": {
        "frame": "robot_local_x_right_z_forward",
        "positive_turn": "right",
        "forward_direction": "current_heading",
        "composition": "ordered_metric_se2",
        "turn_execution": "in_place",
        "forward_collision_check": "continuous_swept_disc",
        "collision_response": "stop_at_first_contact",
    },
    "relations": {
        "B1_endpoint_distance": (
            "robot_center_to_target_ground_support_nearest_distance; "
            "the item task_metadata declares the authenticated geometry "
            "protocol"),
        "B2_endpoint_direction": (
            "target_reference_centroid_terminal_bearing; the item "
            "task_metadata declares the authenticated geometry protocol"),
    },
}
# The published height is a continuous fit rounded for legibility, not a
# quantised measurement: the private record keeps full precision.
PUBLIC_HEIGHT_DECIMALS = 2

A_TASK_QUESTIONS = {
    "A1_collision":
        "Will the robot collide with anything while executing all the actions?",
    "A2_collision_step_grounding":
        "The robot will collide while executing the actions. During which "
        "action does the collision occur?",
    "A3_contact_object":
        "The robot will collide while executing the actions. What category "
        "of visible object or surface will it contact first?",
}
B_TASK_QUESTIONS = {
    "B1_endpoint_distance":
        "After safely completing all the actions, how far will the robot be "
        "from {target}?",
    "B2_endpoint_direction":
        "After safely completing all the actions, where will {target} be "
        "relative to the robot’s final facing direction?",
}
C_TASK_QUESTIONS = {
    "C1_future_view_selection":
        "After safely completing all the actions, which image shows the "
        "robot’s final view?",
}
ABC_CANDIDATE_TASK_IDS = (
    *A_TASK_QUESTIONS,
    *B_TASK_QUESTIONS,
    *C_TASK_QUESTIONS,
)


def r2r_semantic_source_sha256(source: dict) -> str:
    """Validate strict source provenance and return its semantic PLY hash."""
    return record_fields.r2r_semantic_source_sha256(source)


def _stability_shortfall_reason(certificate: Dict) -> str:
    """Name which of the two ways a label failed to become declarable.

    ``collision_label_stable`` is the conjunction of "every perturbation row
    was accepted" and "every accepted row agreed", and reporting only the
    conjunction reads as if the label flipped. It usually did not: a row whose
    consensus rejected for thin depth evidence never produced a label to
    disagree with. The publication gate is unchanged -- both still reject.
    """
    rows = certificate.get("rows") or []
    if any((row.get("consensus") or {}).get("accepted") is not True
           for row in rows):
        return "stability_evidence_incomplete"
    return "stability_label_disagreement"


def shared_visible_space_certificate(
        rec: Dict, outcome: Dict) -> tuple[dict | None, str]:
    """Authenticate the stored visible-space certificate shared by A and C."""
    source = rec.get("source")
    if not isinstance(source, dict):
        return None, "strict_r2r_main_contract_required"
    contract = rec.get("collection_contract") or {}
    dataset = source.get("source_dataset")
    if dataset == "r2r":
        contract_reason = "strict_r2r_main_contract_required"
    elif dataset == "b1k":
        contract_reason = "strict_b1k_main_contract_required"
    elif dataset == "gs":
        if (rec.get("schema_version") != record_fields.V18_SCHEMA_VERSION or
                "collection_contract" in rec):
            return None, "shared_oracle_stability_invalid"
        try:
            gs_binding = dataset_contracts.resolve_gs_collision_binding(source)
            expected_atom = dataset_contracts.gs_collision_binding_atom(
                gs_binding)
            identity_binding = record_fields.authority_binding(source)
            require_contact_instance_witness = \
                gs_semantic.scene_task_available(
                    rec.get("gs_scene_capability"), source, "A3")
        except (KeyError, StopIteration, TypeError, ValueError):
            return None, "shared_oracle_stability_invalid"
        if rec.get("gs_collision_authority") != expected_atom:
            return None, "shared_oracle_stability_invalid"
        certificate = outcome.get("shared_oracle_stability")
        if not isinstance(certificate, dict):
            return None, "shared_oracle_stability_missing"
        physical_values = [outcome.get("physical"), *[
            row.get("physical") if isinstance(row, dict) else None
            for row in certificate.get("rows") or []
        ]]
        if any(
                not isinstance(physical, dict) or
                physical.get("authority") != "gs_collision_mesh" or
                physical.get("geometry_authority_sha256") !=
                gs_binding.collision_authority_sha256
                for physical in physical_values):
            return None, "shared_oracle_stability_invalid"
        try:
            rebuilt = consensus_fields.build_a_stability_certificate(
                outcome.get("actions") or [], certificate.get("rows") or [],
                authority_binding=identity_binding,
                require_contact_instance_witness=
                    require_contact_instance_witness)
        except (KeyError, TypeError, ValueError):
            return None, "shared_oracle_stability_invalid"
        if any(certificate.get(key) != rebuilt.get(key) for key in (
                "version", "perturbation_asset_sha256", "rows", "summary",
                "contact_instance_witness_required", "sha256")):
            return None, "shared_oracle_stability_invalid"
        if consensus_fields.a_stability_certificate_mismatches(
                outcome.get("actions") or [], certificate, outcome,
                authority_binding=identity_binding,
                require_contact_instance_witness=
                    require_contact_instance_witness):
            return None, "shared_oracle_stability_invalid"
        summary = certificate.get("summary") or {}
        if (summary.get("collision_label_stable") is not True or
                not isinstance(summary.get("collision"), bool)):
            return None, _stability_shortfall_reason(certificate)
        return certificate, "eligible"
    else:
        return None, "unsupported_source_dataset"
    try:
        expected_contract = record_fields.collection_contract(
            source, "main", record_schema_version=rec.get(
                "schema_version", record_fields.SCHEMA_VERSION))
        authority_binding = record_fields.authority_binding(source)
    except (StopIteration, TypeError, ValueError):
        return None, contract_reason
    if contract != expected_contract:
        return None, contract_reason
    certificate = outcome.get("shared_oracle_stability")
    if not isinstance(certificate, dict):
        return None, "shared_oracle_stability_missing"
    try:
        rebuilt = consensus_fields.build_a_stability_certificate(
            outcome.get("actions") or [], certificate.get("rows") or [],
            authority_binding=authority_binding)
    except (KeyError, TypeError, ValueError):
        return None, "shared_oracle_stability_invalid"
    if any(certificate.get(key) != rebuilt.get(key) for key in (
            "version", "perturbation_asset_sha256", "rows", "summary",
            "sha256")):
        return None, "shared_oracle_stability_invalid"
    if consensus_fields.a_stability_certificate_mismatches(
            outcome.get("actions") or [], certificate, outcome,
            authority_binding=authority_binding):
        return None, "shared_oracle_stability_invalid"
    summary = certificate.get("summary") or {}
    if (summary.get("collision_label_stable") is not True or
            not isinstance(summary.get("collision"), bool)):
        return None, _stability_shortfall_reason(certificate)
    return certificate, "eligible"


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str = "eligible"
    margins: Dict[str, object] = None


@dataclass(frozen=True)
class SharedVisibleSpaceEvidence:
    """One authenticated stability certificate bound to its source objects."""

    record_identity: int
    outcome_identity: int
    certificate: dict | None
    reason: str

    def require(self, rec: Dict, outcome: Dict) -> tuple[dict | None, str]:
        if (self.record_identity != id(rec) or
                self.outcome_identity != id(outcome)):
            raise ValueError("shared evidence belongs to another outcome")
        return self.certificate, self.reason


def build_shared_visible_space_evidence(
        rec: Dict, outcome: Dict, *, source_validated: bool = False
        ) -> SharedVisibleSpaceEvidence:
    """Authenticate shared A/B/C evidence once for record compilation."""
    if source_validated:
        certificate = outcome.get("shared_oracle_stability")
        if not isinstance(certificate, dict):
            certificate, reason = None, "shared_oracle_stability_missing"
        else:
            summary = certificate.get("summary") or {}
            if (summary.get("collision_label_stable") is not True or
                    not isinstance(summary.get("collision"), bool)):
                certificate, reason = (
                    None, _stability_shortfall_reason(certificate))
            else:
                reason = "eligible"
    else:
        certificate, reason = shared_visible_space_certificate(rec, outcome)
    return SharedVisibleSpaceEvidence(
        id(rec), id(outcome), certificate, reason)
