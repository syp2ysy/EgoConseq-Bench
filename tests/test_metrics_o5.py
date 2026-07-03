"""
Tests for egoconseq.eval.metrics: o5_metrics() — new group-based binary contact metrics.

Operationalization:
- Input: groups = list of groups; each group = list of dicts
  {"radius_m": float, "gt": "contact"|"no_contact", "pred": "contact"|"no_contact"}
- case_accuracy: fraction of all cases correct.
- false_safe_rate: among gt=="contact", fraction predicted "no_contact" (cardinal safety error).
- flip_groups: count of groups whose GT labels are NOT all equal.
- correct_flip_rate: among flip_groups, fraction where ALL cases in the group are
  predicted correctly (pred matches gt for every case in the group).
- invariance_error: among flip_groups, fraction where model gives the SAME prediction
  to ALL cases in the group (failed to react to chassis size).
- NaN when denominator is zero.
"""
import math
import pytest
from egoconseq.eval.metrics import o5_metrics


# ── Helper: build a group ─────────────────────────────────────────────────────

def make_group(*cases):
    """Build a group list from (radius_m, gt, pred) tuples."""
    return [{"radius_m": r, "gt": gt, "pred": pred} for r, gt, pred in cases]


# ── Basic correctness ─────────────────────────────────────────────────────────

def test_all_correct_no_flip():
    """All predictions correct, no flip groups."""
    groups = [
        make_group((0.10, "no_contact", "no_contact"), (0.40, "contact", "contact")),
    ]
    # group has flip (no_contact + contact)
    m = o5_metrics(groups)
    assert m["case_accuracy"] == pytest.approx(1.0)
    assert m["false_safe_rate"] == pytest.approx(0.0)
    assert m["flip_groups"] == 1
    assert m["correct_flip_rate"] == pytest.approx(1.0)
    assert m["invariance_error"] == pytest.approx(0.0)


def test_correct_flip_group():
    """One flip group, all cases correctly predicted → correct_flip_rate=1.0."""
    groups = [
        make_group(
            (0.10, "no_contact", "no_contact"),
            (0.40, "contact", "contact"),
        )
    ]
    m = o5_metrics(groups)
    assert m["correct_flip_rate"] == pytest.approx(1.0)
    assert m["invariance_error"] == pytest.approx(0.0)


def test_invariance_error_model_same_pred():
    """Model predicts same label for all in a flip group → invariance_error=1.0."""
    groups = [
        make_group(
            (0.10, "no_contact", "no_contact"),   # gt=no_contact, pred=no_contact ✓
            (0.40, "contact", "no_contact"),       # gt=contact, pred=no_contact ✗ + invariance
        )
    ]
    m = o5_metrics(groups)
    assert m["flip_groups"] == 1
    # all predictions are "no_contact" → model gave same pred to both → invariance_error=1.0
    assert m["invariance_error"] == pytest.approx(1.0)
    # group not fully correct → correct_flip_rate=0.0
    assert m["correct_flip_rate"] == pytest.approx(0.0)


def test_false_safe_rate():
    """false_safe_rate: gt=contact predicted no_contact."""
    # 1 contact case predicted no_contact, 1 contact case predicted correctly
    groups = [
        make_group((0.40, "contact", "no_contact")),   # false safe
        make_group((0.40, "contact", "contact")),      # correct
    ]
    m = o5_metrics(groups)
    assert m["false_safe_rate"] == pytest.approx(0.5)


def test_false_safe_rate_zero_when_all_contact_correct():
    """false_safe_rate=0 when all contact cases predicted correctly."""
    groups = [
        make_group((0.40, "contact", "contact")),
        make_group((0.10, "no_contact", "no_contact")),
    ]
    m = o5_metrics(groups)
    assert m["false_safe_rate"] == pytest.approx(0.0)


# ── Nan handling ──────────────────────────────────────────────────────────────

def test_no_flip_groups_nan_rates():
    """When no flip groups, correct_flip_rate and invariance_error are NaN."""
    groups = [
        make_group((0.10, "no_contact", "no_contact")),
        make_group((0.40, "no_contact", "no_contact")),
    ]
    m = o5_metrics(groups)
    assert m["flip_groups"] == 0
    assert math.isnan(m["correct_flip_rate"])
    assert math.isnan(m["invariance_error"])


def test_no_contact_cases_false_safe_nan():
    """When no gt=='contact' cases, false_safe_rate is NaN."""
    groups = [
        make_group((0.10, "no_contact", "no_contact")),
    ]
    m = o5_metrics(groups)
    assert math.isnan(m["false_safe_rate"])


def test_empty_groups():
    """Empty input must return dict with expected keys and NaN values."""
    m = o5_metrics([])
    assert "case_accuracy" in m
    assert "false_safe_rate" in m
    assert "flip_groups" in m
    assert "correct_flip_rate" in m
    assert "invariance_error" in m
    assert math.isnan(m["case_accuracy"])
    assert math.isnan(m["false_safe_rate"])
    assert m["flip_groups"] == 0
    assert math.isnan(m["correct_flip_rate"])
    assert math.isnan(m["invariance_error"])


# ── Multi-group scenarios ─────────────────────────────────────────────────────

def test_two_flip_groups_one_correct_one_invariant():
    """Two flip groups: one correct, one invariant → correct_flip_rate=0.5, invariance_error=0.5."""
    groups = [
        # Group 1: correctly predicted flip
        make_group(
            (0.10, "no_contact", "no_contact"),
            (0.40, "contact", "contact"),
        ),
        # Group 2: model predicts same for all (invariance error)
        make_group(
            (0.10, "no_contact", "contact"),    # wrong
            (0.40, "contact", "contact"),       # correct but same pred as above
        ),
    ]
    m = o5_metrics(groups)
    assert m["flip_groups"] == 2
    assert m["correct_flip_rate"] == pytest.approx(0.5)
    assert m["invariance_error"] == pytest.approx(0.5)


def test_case_accuracy_mixed():
    """case_accuracy across multiple groups."""
    groups = [
        make_group(
            (0.10, "no_contact", "no_contact"),   # correct
            (0.40, "contact", "no_contact"),       # wrong
        ),
        make_group(
            (0.10, "no_contact", "no_contact"),   # correct
        ),
    ]
    m = o5_metrics(groups)
    # 2/3 cases correct
    assert m["case_accuracy"] == pytest.approx(2.0 / 3.0)


def test_non_flip_group_not_counted_in_flip_metrics():
    """Groups with all same GT labels don't count toward flip_groups."""
    groups = [
        make_group((0.10, "no_contact", "contact")),   # all no_contact GT → no flip
        make_group((0.40, "contact", "contact")),      # all contact GT → no flip
    ]
    m = o5_metrics(groups)
    assert m["flip_groups"] == 0
    assert math.isnan(m["correct_flip_rate"])
    assert math.isnan(m["invariance_error"])
