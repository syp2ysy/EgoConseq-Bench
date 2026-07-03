"""Unit tests for O5 non-vision baselines.

New O5 schema:
- each case is one robot with one diameter;
- labels are binary: ``contact`` / ``no_contact``;
- counterfactual behavior is measured across cases sharing a group_id.
"""

from __future__ import annotations

import math

from egoconseq.eval.baselines import (
    blind_text_only,
    evaluate_all_baselines,
    majority_baseline,
    radius_only,
    random_baseline,
)


def make_group(*cases):
    return [{"radius_m": r, "gt": gt, "pred": pred} for r, gt, pred in cases]


GROUPS = [
    make_group(
        (0.10, "no_contact", "no_contact"),
        (0.40, "contact", "contact"),
    ),
    make_group(
        (0.10, "no_contact", "no_contact"),
        (0.40, "no_contact", "no_contact"),
    ),
    make_group(
        (0.10, "contact", "contact"),
        (0.40, "contact", "contact"),
    ),
]


def test_random_baseline_binary_labels():
    preds = random_baseline(100, seed=0)
    assert len(preds) == 100
    assert set(preds) <= {"contact", "no_contact"}


def test_random_baseline_reproducible():
    assert random_baseline(20, seed=7) == random_baseline(20, seed=7)


def test_majority_baseline_binary_mode():
    gt = ["contact", "contact", "no_contact"]
    preds = majority_baseline(gt)
    assert preds == ["contact", "contact", "contact"]


def test_blind_text_only_defaults_to_no_contact():
    preds = blind_text_only(4)
    assert preds == ["no_contact"] * 4


def test_radius_only_predicts_by_radius_threshold():
    cases = [
        {"radius_m": 0.10},
        {"radius_m": 0.25},
        {"radius_m": 0.40},
    ]
    preds = radius_only(cases, threshold_radius_m=0.25)
    assert preds == ["no_contact", "contact", "contact"]


def test_evaluate_all_baselines_returns_group_metrics():
    results = evaluate_all_baselines(GROUPS, seed=0)
    assert set(results) == {
        "random",
        "majority",
        "blind_text_only",
        "radius_only",
        "geometry_oracle",
    }
    for metrics in results.values():
        assert "case_accuracy" in metrics
        assert "false_safe_rate" in metrics
        assert "flip_groups" in metrics
        assert "correct_flip_rate" in metrics
        assert "invariance_error" in metrics


def test_geometry_oracle_is_perfect_on_groups():
    results = evaluate_all_baselines(GROUPS, seed=0)
    oracle = results["geometry_oracle"]
    assert oracle["case_accuracy"] == 1.0
    assert oracle["false_safe_rate"] == 0.0
    assert oracle["flip_groups"] == 1
    assert oracle["correct_flip_rate"] == 1.0
    assert oracle["invariance_error"] == 0.0


def test_blind_text_only_has_invariance_error_on_flip_group():
    results = evaluate_all_baselines(GROUPS, seed=0)
    assert results["blind_text_only"]["invariance_error"] == 1.0
    assert results["blind_text_only"]["correct_flip_rate"] == 0.0


def test_empty_groups_return_nan_metrics():
    results = evaluate_all_baselines([], seed=0)
    for metrics in results.values():
        assert math.isnan(metrics["case_accuracy"])
        assert math.isnan(metrics["false_safe_rate"])
        assert metrics["flip_groups"] == 0
        assert math.isnan(metrics["correct_flip_rate"])
        assert math.isnan(metrics["invariance_error"])
