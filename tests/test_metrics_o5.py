"""
Tests for egoconseq.eval.metrics: o5_metrics()
TDD: tests must FAIL before implementation, PASS after.

Operationalization of metrics:
- narrow_band_flip_acc:
    Accuracy over the GT `small_only` subset only.
    = #{pred == "small_only" | GT == "small_only"} / #{GT == "small_only"}

- embodiment_sensitivity:
    Among GT `small_only` cases, fraction where the model acknowledges a
    width-based difference (i.e. pred == "small_only").
    Operationally: same as correct_flip_rate (pred == "small_only" over GT
    small_only).  A model that always says "both" or "neither" has
    sensitivity = 0.

- correct_flip_rate:
    Among GT `small_only`, fraction predicted exactly "small_only".
    = #{pred=="small_only" | GT=="small_only"} / #{GT=="small_only"}
    (same fraction as narrow_band_flip_acc but named semantically)

- invariance_error:
    Among GT `small_only`, fraction where pred is "both" or "neither"
    (model treats both bodies identically — fails to discriminate).
    = #{pred in {"both","neither"} | GT=="small_only"} / #{GT=="small_only"}
    Note: invariance_error + embodiment_sensitivity == 1.0 by construction.

- accuracy:
    Standard 3-way overall accuracy = #{pred == gt} / N.
"""
import pytest
from egoconseq.eval.metrics import o5_metrics


# ── Minimal smoke test (given in the task spec) ───────────────────────────────

def test_o5_flip_metrics():
    """Spec-given minimal test: keys exist and values are in [0,1]."""
    gt   = ["small_only", "small_only", "both"]
    pred = ["small_only", "both",       "both"]
    m = o5_metrics(gt, pred)
    assert 0 <= m["narrow_band_flip_acc"] <= 1
    assert "embodiment_sensitivity" in m
    assert "correct_flip_rate" in m
    assert "invariance_error" in m
    assert "accuracy" in m


# ── All-correct case ──────────────────────────────────────────────────────────

def test_all_correct():
    """All predictions match GT → all metrics at maximum."""
    gt   = ["small_only", "both", "neither", "small_only"]
    pred = ["small_only", "both", "neither", "small_only"]
    m = o5_metrics(gt, pred)
    assert m["accuracy"] == 1.0
    assert m["narrow_band_flip_acc"] == 1.0
    assert m["embodiment_sensitivity"] == 1.0
    assert m["correct_flip_rate"] == 1.0
    assert m["invariance_error"] == 0.0


# ── All-invariant case (model always says "both") ─────────────────────────────

def test_all_invariant_both():
    """Model says 'both' for everything → invariance_error == 1.0 over small_only."""
    gt   = ["small_only", "small_only", "neither"]
    pred = ["both",       "both",       "both"]
    m = o5_metrics(gt, pred)
    assert m["invariance_error"] == 1.0
    assert m["correct_flip_rate"] == 0.0
    assert m["embodiment_sensitivity"] == 0.0
    assert m["narrow_band_flip_acc"] == 0.0


def test_all_invariant_neither():
    """Model says 'neither' for everything → invariance_error == 1.0 over small_only."""
    gt   = ["small_only", "small_only", "both"]
    pred = ["neither",    "neither",    "neither"]
    m = o5_metrics(gt, pred)
    assert m["invariance_error"] == 1.0
    assert m["embodiment_sensitivity"] == 0.0


# ── Numeric sanity checks on the spec example ─────────────────────────────────

def test_spec_example_numeric():
    """
    gt   = ["small_only", "small_only", "both"]
    pred = ["small_only", "both",       "both"]

    GT small_only cases: indices 0, 1
      - idx 0: pred "small_only" → correct
      - idx 1: pred "both"       → invariance error

    narrow_band_flip_acc = 1/2 = 0.5
    correct_flip_rate    = 1/2 = 0.5
    embodiment_sensitivity = 1/2 = 0.5
    invariance_error     = 1/2 = 0.5
    accuracy (3-way overall): idx0 correct, idx1 wrong, idx2 correct → 2/3
    """
    gt   = ["small_only", "small_only", "both"]
    pred = ["small_only", "both",       "both"]
    m = o5_metrics(gt, pred)
    assert m["narrow_band_flip_acc"] == pytest.approx(0.5)
    assert m["correct_flip_rate"]    == pytest.approx(0.5)
    assert m["embodiment_sensitivity"] == pytest.approx(0.5)
    assert m["invariance_error"]     == pytest.approx(0.5)
    assert m["accuracy"]             == pytest.approx(2/3)


# ── Edge: no small_only in GT ─────────────────────────────────────────────────

def test_no_small_only_in_gt():
    """
    When there are no 'small_only' GT cases, band metrics should be NaN or None
    (division by zero — implementation should handle gracefully, not crash).
    """
    gt   = ["both", "neither", "both"]
    pred = ["both", "neither", "both"]
    m = o5_metrics(gt, pred)
    assert "narrow_band_flip_acc" in m
    # Value should be NaN or None (implementation-defined; must not crash)
    val = m["narrow_band_flip_acc"]
    import math
    assert val is None or (isinstance(val, float) and math.isnan(val)), (
        f"Expected NaN/None when no small_only GT cases, got {val!r}"
    )


# ── Edge: empty lists ─────────────────────────────────────────────────────────

def test_empty_lists():
    """Empty inputs must not crash and return a dict with the expected keys."""
    m = o5_metrics([], [])
    assert "accuracy" in m
    assert "narrow_band_flip_acc" in m
    assert "invariance_error" in m


# ── invariance_error + embodiment_sensitivity sum to 1.0 ──────────────────────

def test_invariance_plus_sensitivity_sums_to_one():
    """invariance_error + embodiment_sensitivity == 1.0 (complementary over small_only)."""
    gt   = ["small_only", "small_only", "small_only", "both"]
    pred = ["small_only", "both",       "neither",    "small_only"]
    m = o5_metrics(gt, pred)
    import math
    ie = m["invariance_error"]
    es = m["embodiment_sensitivity"]
    assert not math.isnan(ie) and not math.isnan(es)
    assert ie + es == pytest.approx(1.0), (
        f"invariance_error ({ie}) + embodiment_sensitivity ({es}) != 1.0"
    )


# ── Partial flip correctness ───────────────────────────────────────────────────

def test_partial_flip():
    """Two small_only GT, one correctly predicted, one predicted 'neither'."""
    gt   = ["small_only", "small_only"]
    pred = ["small_only", "neither"]
    m = o5_metrics(gt, pred)
    assert m["correct_flip_rate"] == pytest.approx(0.5)
    # 'neither' is an invariance error (model treated them the same)
    assert m["invariance_error"] == pytest.approx(0.5)
    assert m["embodiment_sensitivity"] == pytest.approx(0.5)
    assert m["narrow_band_flip_acc"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.5)
