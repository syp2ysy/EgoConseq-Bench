"""Question registry access, eligibility, evidence, and answer derivation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
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
        expected_fields = capability_contracts.snapshot_fields(dataset)
        expected = {
            **expected_fields,
            "sha256": record_fields.canonical_atom_sha256(expected_fields),
        }
        task = _CAPABILITY_TASK[task_id]
    except (KeyError, TypeError, ValueError):
        return Eligibility(False, "capability_unavailable")
    if record.get("authority_surface_capability") != expected:
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
        if (summary.get("original_action_index_stable") is not True or
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


def _b_target(record: Dict) -> tuple[dict | None, str]:
    return record_fields.authenticated_b_target(record)


def _numbered_dot_markers(record: Dict, selection: Dict) -> list:
    """The disambiguation dots one B question would have to draw.

    Shared by the eligibility check and the builder on purpose: when they
    computed the marker set separately, eligibility could pass a question the
    builder then failed to render.
    """
    key = str(selection.get("category") or "").casefold()
    return [
        value["marker"]
        for value in object_fields.initial_visible_entity_inventory(record)
        if str(value["category"]).casefold() == key and
        value["marker"] is not None]


def _b1_metric_choices(
        precise_distance_m: float, *, seed: str) -> tuple[str, list, dict]:
    precise = float(precise_distance_m)
    if not math.isfinite(precise) or precise < config.B_ENDPOINT_DISTANCE_MIN_M:
        raise ValueError("B1 precise endpoint distance is invalid")
    quantum = Decimal(1).scaleb(-config.B_DISTANCE_CHOICE_DECIMALS)
    displayed_decimal = Decimal(str(precise)).quantize(
        quantum, rounding=ROUND_HALF_EVEN)
    displayed = float(displayed_decimal)
    display_error = abs(displayed - precise)
    if display_error >= config.B_DISTANCE_DISPLAY_ERROR_TOLERANCE_M:
        raise ValueError("B1 display quantization error is too large")
    separation = float(config.B_DISTANCE_CHOICE_MIN_SEPARATION_M)
    plausible_lower_bound = float(config.B_ENDPOINT_DISTANCE_MIN_M)
    plausibility_tolerance = float(
        config.B_DISTANCE_DISPLAY_ERROR_TOLERANCE_M)
    rank_digest = hashlib.sha256(
        f"{seed}:truth-numeric-rank".encode("utf-8")).digest()
    # Sampling the sign count first makes the truth's sorted numeric rank a
    # uniform causal choice.  The old "pick any three of +/-1..3" procedure
    # put it in a middle rank with 90% probability.  For a low truth, sample
    # uniformly only from ranks that can be realised without a negative
    # distance.  QA randomisation must not delete an otherwise valid Task-GT;
    # the compiled artifact reports the resulting empirical rank baseline.
    negative_steps = [
        step for step in range(1, 4)
        if displayed - step * separation + plausibility_tolerance >=
        plausible_lower_bound
    ]
    positive_steps = list(range(1, 4))
    maximum_negative_count = min(3, len(negative_steps))
    negative_count = (
        int.from_bytes(rank_digest[:8], "big") %
        (maximum_negative_count + 1))
    positive_count = 3 - negative_count

    def choose_steps(values, count, side):
        ranked = sorted(values, key=lambda step: hashlib.sha256(
            f"{seed}:distractor:{side}:{step}".encode("utf-8")).hexdigest())
        return ranked[:count]

    selected_negative = choose_steps(
        negative_steps, negative_count, "negative")
    selected_positive = choose_steps(
        positive_steps, positive_count, "positive")
    alternatives = [
        displayed - step * separation for step in selected_negative
    ] + [
        displayed + step * separation for step in selected_positive
    ]
    values = [displayed, *alternatives]
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("B1 cannot construct four distinct metric choices")
    if any(value + plausibility_tolerance < plausible_lower_bound
           for value in values):
        raise ValueError("B1 choice is below the plausible distance bound")
    values.sort(key=lambda value: hashlib.sha256(
        f"{seed}:position:{value:.12f}".encode("ascii")).hexdigest())
    choices = [{
        "id": f"choice_{index}",
        "text": f"{value:.{config.B_DISTANCE_CHOICE_DECIMALS}f} m",
        "value_m": float(value),
    } for index, value in enumerate(values, 1)]
    answer = next(
        choice["id"] for choice in choices
        if choice["value_m"] == displayed)
    certificate = {
        "protocol": config.B_DISTANCE_CHOICE_PROTOCOL,
        "precise_distance_m": precise,
        "displayed_distance_m": displayed,
        "display_decimals": config.B_DISTANCE_CHOICE_DECIMALS,
        "display_quantization_error_m": display_error,
        "display_error_tolerance_m":
            config.B_DISTANCE_DISPLAY_ERROR_TOLERANCE_M,
        "minimum_separation_m": separation,
        "plausible_lower_bound_m": plausible_lower_bound,
        "plausibility_tolerance_m": plausibility_tolerance,
        "truth_numeric_rank_1based": int(negative_count + 1),
        "feasible_truth_numeric_ranks_1based": list(
            range(1, maximum_negative_count + 2)),
        "negative_choice_count": int(negative_count),
        "positive_choice_count": int(positive_count),
        "choice_values_m": [choice["value_m"] for choice in choices],
        "canonical_answer": answer,
    }
    certificate["sha256"] = record_fields.canonical_atom_sha256(certificate)
    return answer, choices, certificate


@dataclass(frozen=True)
class BRecordEvidence:
    """The fixed s0 target authenticated once for every outcome in a record."""

    record_identity: int
    target: dict | None
    reason: str

    def require(self, record: Dict) -> tuple[dict | None, str]:
        if self.record_identity != id(record):
            raise ValueError("B record evidence belongs to another record")
        return self.target, self.reason


def build_b_record_evidence(
        record: Dict, *, source_validated: bool = False) -> BRecordEvidence:
    if source_validated:
        target = record.get("b_target")
        reason = "eligible" if isinstance(target, dict) else "b_target_invalid"
    else:
        target, reason = _b_target(record)
    return BRecordEvidence(id(record), target, reason)


@dataclass(frozen=True)
class BOutcomeEvidence:
    """Task-independent B evidence authenticated once for one outcome."""

    record_identity: int
    outcome_identity: int
    rejection: str | None
    margins: dict

    def require(self, record: Dict, outcome: Dict) -> None:
        if (self.record_identity != id(record) or
                self.outcome_identity != id(outcome)):
            raise ValueError("B outcome evidence belongs to another outcome")


def build_b_outcome_evidence(
        record: Dict, outcome: Dict, *,
        shared_evidence: SharedVisibleSpaceEvidence | None = None,
        record_evidence: BRecordEvidence | None = None,
        source_validated: bool = False) -> BOutcomeEvidence:
    """Authenticate the evidence shared by B1 and B2 exactly once."""
    def rejected(reason: str) -> BOutcomeEvidence:
        return BOutcomeEvidence(id(record), id(outcome), reason, {})

    certificate, reason = (
        _a_shared_certificate(record, outcome) if shared_evidence is None
        else shared_evidence.require(record, outcome))
    if certificate is None:
        return rejected(reason)
    if certificate["summary"].get("collision") is not False:
        return rejected("completed_clear_required")
    if not outcome_fields.is_completed_clear(outcome):
        return rejected("completed_clear_required")
    if record_evidence is None:
        target, reason = _b_target(record)
    else:
        target, reason = record_evidence.require(record)
    if target is None:
        return rejected(reason)
    selection = target.get("selection") or {}
    if selection.get("marker") is not None:
        # The record only guarantees the centroid is inside the raster; the dot
        # additionally needs a whole radius of margin. Refuse here rather than
        # letting the builder raise mid-compile.
        resolution = (record.get("sensor") or {}).get("resolution") or ()
        if len(resolution) != 2:
            return rejected("b_marker_resolution_missing")
        _placement, marker_reason = viz.numbered_dot_placement(
            _numbered_dot_markers(record, selection),
            width=resolution[0], height=resolution[1])
        if marker_reason is not None:
            return rejected("b_numbered_dot_unrenderable")
    relation = outcome.get("b_endpoint_relation")
    if (not isinstance(relation, dict) or
            (not source_validated and
             not record_fields.b_endpoint_relation_atom_valid(
                 relation, pose=record["pose"], outcome=outcome,
                 b_target=target))):
        return rejected("b_endpoint_relation_invalid")
    margins = {
        "shared_certificate_sha256": certificate["sha256"],
        "target_sha256": target["sha256"],
        "target_geometry_sha256": target["geometry"]["sha256"],
        "endpoint_relation_sha256": relation["sha256"],
    }
    return BOutcomeEvidence(id(record), id(outcome), None, margins)


def b_candidate_eligibility(
        task_id: str, record: Dict, outcome: Dict, *,
        evidence: BOutcomeEvidence | None = None) -> Eligibility:
    """Fail closed over one authenticated clear rollout and fixed s0 target."""
    if task_id not in B_TASK_QUESTIONS:
        return Eligibility(False, "unknown_b_task")
    capability_rejection = _capability_envelope_rejection(task_id, record)
    if capability_rejection is not None:
        return capability_rejection
    evidence = evidence or build_b_outcome_evidence(record, outcome)
    if not isinstance(evidence, BOutcomeEvidence):
        raise TypeError("evidence must be BOutcomeEvidence")
    evidence.require(record, outcome)
    if evidence.rejection is not None:
        return Eligibility(False, evidence.rejection)
    relation = outcome["b_endpoint_relation"]
    margins = dict(evidence.margins)
    if task_id == "B1_endpoint_distance":
        after = float(relation["distance_after_m"])
        before = float(relation["distance_before_m"])
        minimum_change = max(
            config.B_DISTANCE_CHANGE_MIN_M,
            config.B_DISTANCE_CHANGE_MIN_RATIO * before)
        if (not math.isfinite(after) or
                after < config.B_ENDPOINT_DISTANCE_MIN_M):
            return Eligibility(False, "endpoint_distance_nontrivial_required")
        if abs(after - before) < minimum_change:
            return Eligibility(False, "endpoint_distance_change_too_small")
        seed = ":".join(map(str, (
            record.get("observation_id") or record.get("frame_id"),
            outcome.get("outcome_id"), task_id)))
        try:
            _answer, _choices, choice_certificate = _b1_metric_choices(
                after, seed=seed)
        except ValueError:
            return Eligibility(False, "b1_metric_choices_invalid")
        margins.update({
            "minimum_distance_change_m": float(minimum_change),
            "display_quantization_error_m":
                choice_certificate["display_quantization_error_m"],
            "display_error_tolerance_m":
                choice_certificate["display_error_tolerance_m"],
            "choice_certificate_sha256": choice_certificate["sha256"],
        })
        return Eligibility(True, margins=margins)
    if relation.get("direction_status") == "undefined_centroid_range":
        return Eligibility(False, "centroid_anchor_range_too_small")
    if relation.get("direction_status") != "computed":
        return Eligibility(False, "b_endpoint_relation_invalid")
    try:
        direction, boundary_margin = record_fields.b_direction_with_margin(
            relation["bearing_after_deg"])
    except (TypeError, ValueError):
        return Eligibility(False, "bearing_sector_boundary_margin_failed")
    if direction != relation.get("direction"):
        return Eligibility(False, "b_endpoint_relation_invalid")
    margins["sector_boundary_margin_deg"] = boundary_margin
    return Eligibility(True, margins=margins)


def b_candidate_answer(
        task_id: str, record: Dict, outcome: Dict, *,
        evidence: BOutcomeEvidence | None = None
        ) -> tuple[str, list, dict | None]:
    eligibility = b_candidate_eligibility(
        task_id, record, outcome, evidence=evidence)
    if not eligibility.eligible:
        raise ValueError(f"ineligible {task_id}: {eligibility.reason}")
    relation = outcome["b_endpoint_relation"]
    if task_id == "B1_endpoint_distance":
        seed = ":".join(map(str, (
            record.get("observation_id") or record.get("frame_id"),
            outcome.get("outcome_id"), task_id)))
        return _b1_metric_choices(relation["distance_after_m"], seed=seed)
    choices = [
        {"id": value, "text": value}
        for value in ("front", "left", "right", "rear")]
    return str(relation["direction"]), choices, None


def build_b_candidate_projection(*, task_id: str, record: Dict,
                                 outcome: Dict, image: str, stable_id,
                                 expected_raw_image_sha256: str,
                                 asset_dir=None, authenticated_rgb=None,
                                 numbered_dot_cache=None,
                                 evidence: BOutcomeEvidence | None = None
                                 ) -> tuple[dict, dict]:
    sensor = record.get("sensor")
    if not isinstance(sensor, dict):
        raise ValueError("record sensor resolution is invalid")
    raw_asset = viz.authenticate_raw_rgb_image(
        image, expected_sha256=expected_raw_image_sha256,
        expected_resolution=sensor.get("resolution"),
        authenticated=authenticated_rgb)
    eligibility = b_candidate_eligibility(
        task_id, record, outcome, evidence=evidence)
    if not eligibility.eligible:
        raise ValueError(f"ineligible {task_id}: {eligibility.reason}")
    canonical_answer, choices, choice_certificate = b_candidate_answer(
        task_id, record, outcome, evidence=evidence)
    target = record["b_target"]
    selection = target["selection"]
    geometry = target["geometry"]
    relation = outcome["b_endpoint_relation"]
    oracle_ref = {
        "frame_id": record["frame_id"],
        "outcome_id": outcome.get("outcome_id"),
        "base_rollout_key": outcome.get("base_rollout_key"),
        "shared_certificate_sha256":
            outcome["shared_oracle_stability"]["sha256"],
        "target_sha256": target["sha256"],
        "target_geometry_sha256": target["geometry"]["sha256"],
        "endpoint_relation_sha256": relation["sha256"],
    }
    item_id = "b-" + stable_id(
        record.get("observation_id") or record["frame_id"],
        outcome.get("outcome_id"), task_id)
    model_input = {
        "initial_rgb": str(image),
        "initial_rgb_sha256": raw_asset.sha256,
        "camera_height_above_visible_floor_m": round(
            float(record["camera_height_above_visible_floor_m"]),
            PUBLIC_HEIGHT_DECIMALS),
        "hfov_deg": float(sensor["hfov_deg"]),
        "vfov_deg": float(sensor["vfov_deg"]),
        "body_radius_m": float(outcome["body"]["radius_m"]),
        "actions": list(outcome.get("actions") or []),
        "target": selection["name"],
    }
    input_asset = {
        "marked": False, "path": str(image), "sha256": raw_asset.sha256,
        "raw_path": str(image), "raw_sha256": raw_asset.sha256,
        "binding": "sha256_authoritative_relocated",
        "record_image_path": record.get("image_path"),
    }
    if selection.get("marker") is not None:
        if asset_dir is None:
            raise ValueError("B numbered-dot target requires an asset directory")
        input_asset = viz.materialize_numbered_dot_image(
            raw_asset, _numbered_dot_markers(record, selection),
            asset_dir, item_id, encoded_cache=numbered_dot_cache)
        input_asset.update({
            "binding": "sha256_authoritative_relocated",
            "record_image_path": record.get("image_path"),
        })
        model_input["initial_rgb"] = input_asset["path"]
        model_input["initial_rgb_sha256"] = input_asset["sha256"]
    item = {
        "id": item_id, "result_head": "B", "task_id": task_id,
        "question": B_TASK_QUESTIONS[task_id].format(
            target=selection["name"]),
        "answer_format": "closed_exact", "model_input": model_input,
        "choices": choices, "oracle_ref": oracle_ref,
        "task_metadata": {
            "target_geometry_protocol": str(
                geometry[
                    "ground_support" if task_id == "B1_endpoint_distance"
                    else "reference_centroid"]["protocol"]),
        },
    }
    private = {
        "id": item_id, "task_id": task_id,
        "canonical_answer": canonical_answer,
        "oracle_ref": dict(oracle_ref),
        "target_instance_id": int(selection["instance_id"]),
        "precise_distance_m": float(relation["distance_after_m"]),
        "precise_bearing_deg": (
            None if relation["bearing_after_deg"] is None
            else float(relation["bearing_after_deg"])),
        "choice_certificate": choice_certificate,
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
