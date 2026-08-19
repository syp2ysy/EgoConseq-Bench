"""Frozen disposition reasons allowed in candidate ``report.json``.

This authority is intentionally scoped to QA construction and report routing.
Simulator diagnostics and collection-funnel reasons remain owned by their
respective contracts.
"""

from __future__ import annotations


QA_REPORT_REASONS = frozenset({
    # Candidate routing and family publication.
    "a1_control_exact_unmatched",
    "a1_common_support_unmatched",
    "a1_natural_balance_unmatched",
    "a1_nonpublication_source",
    "capability_unavailable",
    "family_incomplete",
    "family_leg_answers_a1_a2_only",
    "ineligible",
    "not_a_family_leg",
    "other_family_kind",
    # Shared visible-space and A construction.
    "collision_required",
    "collision_action_index_unstable",
    "contact_attribution_disagreement",
    "contact_category_ineligible",
    "contact_category_not_initially_visible",
    "contact_category_oracle_disagreement",
    "contact_category_unstable",
    "contact_instance_not_initially_visible",
    "insufficient_visible_contact_categories",
    "multiple_forward_actions_required",
    "shared_oracle_stability_invalid",
    "shared_oracle_stability_missing",
    "stability_evidence_incomplete",
    "stability_label_disagreement",
    "strict_b1k_main_contract_required",
    "strict_r2r_main_contract_required",
    "unknown_a_task",
    # B construction.
    "b1_metric_choices_invalid",
    "b_endpoint_relation_invalid",
    "b_marker_resolution_missing",
    "b_numbered_dot_unrenderable",
    "b_target_invalid",
    "bearing_sector_boundary_margin_failed",
    "centroid_anchor_range_too_small",
    "completed_clear_required",
    "endpoint_distance_change_too_small",
    "endpoint_distance_nontrivial_required",
    "unknown_b_task",
    # C eligibility, terminal assets, and selector disposition.
    "counterfactual_distractor_shortfall",
    "counterfactual_query_not_eligible",
    "counterfactual_selection_invalid",
    "terminal_cache_miss",
    "terminal_checkpoint_incomplete",
    "terminal_checkpoint_missing",
    "terminal_checkpoint_pose_invalid",
    "terminal_encoding_invalid",
    "terminal_pose_disagreement",
    "terminal_provenance_invalid",
    "terminal_publication_invalid",
    "terminal_rgb_asset_invalid",
})


def require_report_reason(reason: object) -> str:
    """Return one frozen reason or fail at the report disposition boundary."""
    value = str(reason)
    if value not in QA_REPORT_REASONS:
        raise ValueError(f"unknown QA report reason: {value}")
    return value


def validate_report_rejections(
        rejections: object, *, task_ids: tuple[str, ...]) -> None:
    """Validate report reason names and positive integer occurrence counts."""
    if not isinstance(rejections, dict) or set(rejections) != set(task_ids):
        raise ValueError("QA report rejection task catalog is invalid")
    for task_id in task_ids:
        bucket = rejections[task_id]
        if not isinstance(bucket, dict):
            raise ValueError("QA report rejection bucket is invalid")
        for reason, count in bucket.items():
            require_report_reason(reason)
            if (isinstance(count, bool) or not isinstance(count, int) or
                    count <= 0):
                raise ValueError("QA report rejection count is invalid")
