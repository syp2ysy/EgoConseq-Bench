"""Unit tests for egoconseq.eval.baselines (non-vision shortcut baselines).

These tests do NOT require Habitat or any vision model — all baselines are
purely algorithmic.
"""

from __future__ import annotations

import math
import pytest

from egoconseq.eval.baselines import (
    blind_text_only,
    evaluate_all_baselines,
    majority_baseline,
    radius_only,
    random_baseline,
)
from egoconseq.eval.metrics import o5_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LABELS_BALANCED = (
    ["both"] * 10
    + ["small_only"] * 10
    + ["neither"] * 10
)

LABELS_SMALL_ONLY_HEAVY = (
    ["small_only"] * 20
    + ["both"] * 5
    + ["neither"] * 5
)

LABELS_BOTH_ONLY = ["both"] * 15


# ---------------------------------------------------------------------------
# random_baseline
# ---------------------------------------------------------------------------

class TestRandomBaseline:
    def test_output_length(self):
        preds = random_baseline(20, seed=0)
        assert len(preds) == 20

    def test_all_valid_labels(self):
        valid = {"both", "small_only", "neither"}
        preds = random_baseline(100, seed=42)
        assert all(p in valid for p in preds)

    def test_reproducible_with_seed(self):
        p1 = random_baseline(50, seed=7)
        p2 = random_baseline(50, seed=7)
        assert p1 == p2

    def test_different_seeds_differ(self):
        p1 = random_baseline(50, seed=1)
        p2 = random_baseline(50, seed=2)
        assert p1 != p2

    def test_all_three_labels_appear(self):
        """With 300 draws, all three labels should appear."""
        preds = random_baseline(300, seed=0)
        unique = set(preds)
        assert "both" in unique
        assert "small_only" in unique
        assert "neither" in unique

    def test_zero_n(self):
        assert random_baseline(0) == []

    def test_narrow_band_flip_acc_approximately_chance(self):
        """Over a balanced gt set, random should yield ~1/3 flip acc."""
        gt = LABELS_BALANCED
        accs = []
        for seed in range(20):
            preds = random_baseline(len(gt), seed=seed)
            m = o5_metrics(gt, preds)
            accs.append(m["narrow_band_flip_acc"])
        avg = sum(accs) / len(accs)
        # Expected ≈ 1/3; allow generous band [0.1, 0.6] for small N
        assert 0.1 <= avg <= 0.6, f"avg narrow_band_flip_acc = {avg:.3f}, expected ~0.33"


# ---------------------------------------------------------------------------
# majority_baseline
# ---------------------------------------------------------------------------

class TestMajorityBaseline:
    def test_output_length(self):
        preds = majority_baseline(LABELS_BALANCED)
        assert len(preds) == len(LABELS_BALANCED)

    def test_empty_input(self):
        assert majority_baseline([]) == []

    def test_predicts_majority_label(self):
        preds = majority_baseline(LABELS_SMALL_ONLY_HEAVY)
        assert all(p == "small_only" for p in preds)

    def test_predicts_majority_label_balanced(self):
        # "both" appears 10× in LABELS_BALANCED; tie broken alphabetically.
        # Actually all three appear equally (10 each). Alphabetical tie-break:
        # "both" < "neither" < "small_only" → "both" wins.
        preds = majority_baseline(LABELS_BALANCED)
        # All three labels appear equally; ensure all preds are the same label.
        assert len(set(preds)) == 1

    def test_all_same_label(self):
        preds = majority_baseline(LABELS_BOTH_ONLY)
        assert all(p == "both" for p in preds)

    def test_narrow_band_flip_acc_is_zero_when_majority_is_not_small_only(self):
        """Majority = 'both' → flip acc on small_only GT = 0."""
        gt = ["both"] * 20 + ["small_only"] * 5
        preds = majority_baseline(gt)
        m = o5_metrics(gt, preds)
        assert m["narrow_band_flip_acc"] == 0.0

    def test_narrow_band_flip_acc_is_one_when_majority_is_small_only(self):
        """Majority = 'small_only' → predicts small_only for every case."""
        gt = ["small_only"] * 20 + ["both"] * 1
        preds = majority_baseline(gt)
        m = o5_metrics(gt, preds)
        assert m["narrow_band_flip_acc"] == 1.0


# ---------------------------------------------------------------------------
# blind_text_only
# ---------------------------------------------------------------------------

class TestBlindTextOnly:
    def test_output_length(self):
        preds = blind_text_only(15)
        assert len(preds) == 15

    def test_default_label_is_both(self):
        preds = blind_text_only(10)
        assert all(p == "both" for p in preds)

    def test_custom_fixed_label(self):
        preds = blind_text_only(10, fixed_label="neither")
        assert all(p == "neither" for p in preds)

    def test_zero_n(self):
        assert blind_text_only(0) == []

    def test_narrow_band_flip_acc_is_zero(self):
        """Always predicting 'both' → flip acc on small_only GT = 0."""
        gt = LABELS_BALANCED
        preds = blind_text_only(len(gt))
        m = o5_metrics(gt, preds)
        assert m["narrow_band_flip_acc"] == 0.0

    def test_overall_accuracy_matches_fraction_of_predicted_label(self):
        """Accuracy = fraction of GT that equals the fixed label."""
        gt = LABELS_BALANCED  # 10 "both", 10 "small_only", 10 "neither"
        preds = blind_text_only(len(gt), fixed_label="both")
        m = o5_metrics(gt, preds)
        expected_acc = 10 / 30
        assert abs(m["accuracy"] - expected_acc) < 1e-9


# ---------------------------------------------------------------------------
# radius_only
# ---------------------------------------------------------------------------

class TestRadiusOnly:
    def test_output_length(self):
        preds = radius_only(12)
        assert len(preds) == 12

    def test_default_label_is_small_only(self):
        preds = radius_only(10)
        assert all(p == "small_only" for p in preds)

    def test_custom_fixed_label(self):
        preds = radius_only(10, fixed_label="both")
        assert all(p == "both" for p in preds)

    def test_zero_n(self):
        assert radius_only(0) == []

    def test_narrow_band_flip_acc_is_one_when_all_gt_is_small_only(self):
        """If all GT = 'small_only', always predicting small_only gives flip=1."""
        gt = ["small_only"] * 20
        preds = radius_only(20)
        m = o5_metrics(gt, preds)
        assert m["narrow_band_flip_acc"] == 1.0

    def test_narrow_band_flip_acc_is_one_on_small_only_subset(self):
        """radius_only gets flip_acc=1 on small_only cases, but overall acc is lower."""
        gt = LABELS_BALANCED  # 10 each
        preds = radius_only(len(gt))
        m = o5_metrics(gt, preds)
        # All small_only GT cases predicted correctly
        assert m["narrow_band_flip_acc"] == 1.0
        # Overall accuracy = 10/30 (only small_only GT cases correct)
        assert abs(m["accuracy"] - 10 / 30) < 1e-9


# ---------------------------------------------------------------------------
# evaluate_all_baselines
# ---------------------------------------------------------------------------

class TestEvaluateAllBaselines:
    def test_returns_all_baselines(self):
        results = evaluate_all_baselines(LABELS_BALANCED, seed=0)
        expected_keys = {
            "random", "majority", "blind_text_only",
            "radius_only", "geometry_oracle",
        }
        assert expected_keys == set(results.keys())

    def test_geometry_oracle_is_perfect(self):
        """Geometry oracle must achieve accuracy = 1.0 and flip_acc = 1.0."""
        results = evaluate_all_baselines(LABELS_BALANCED, seed=0)
        oracle = results["geometry_oracle"]
        assert oracle["accuracy"] == 1.0
        assert oracle["narrow_band_flip_acc"] == 1.0
        assert oracle["invariance_error"] == 0.0

    def test_blind_text_only_flip_acc_is_zero(self):
        """blind_text_only always predicts 'both' → flip_acc = 0 on small_only GT."""
        results = evaluate_all_baselines(LABELS_BALANCED, seed=0)
        assert results["blind_text_only"]["narrow_band_flip_acc"] == 0.0

    def test_all_metrics_are_finite(self):
        """With a balanced label set, no metric should be NaN."""
        results = evaluate_all_baselines(LABELS_BALANCED, seed=0)
        for name, m in results.items():
            for metric_name, val in m.items():
                assert not math.isnan(val), (
                    f"baseline '{name}' metric '{metric_name}' is NaN"
                )

    def test_non_vision_baselines_narrow_band_flip_acc_not_high(self):
        """Random / majority / blind / radius should NOT achieve flip_acc > 0.5
        except by structural coincidence (radius_only always picks small_only).
        On a BALANCED set, check that at least random and blind are below 0.5."""
        results = evaluate_all_baselines(LABELS_BALANCED, seed=0)
        # random: ~ 1/3
        assert results["random"]["narrow_band_flip_acc"] < 0.6
        # blind_text_only: 0 (always "both")
        assert results["blind_text_only"]["narrow_band_flip_acc"] == 0.0

    def test_empty_gt_returns_nan_metrics(self):
        results = evaluate_all_baselines([], seed=0)
        for name, m in results.items():
            assert math.isnan(m["accuracy"]), (
                f"baseline '{name}' accuracy should be NaN for empty GT"
            )
