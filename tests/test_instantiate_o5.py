"""
Tests for egoconseq.tasks.instantiate: o5_label() and make_o5()
TDD: tests must FAIL before implementation, PASS after.

Design RED LINE (§6): the action horizon for O5 is a FIXED physical path in
METRES, identical for both bodies.  'horizon_m' is that shared metric path;
each body's d_safe is compared to the SAME H (not to its own body-width).

Margin rule (per-body):
  abs(d_safe_X - H) >= 0.5 * (2 * r_X)
  i.e. the clearance from the decision boundary H must be at least one body
  radius for each body separately.  Cases too close to the boundary are
  rejected (make_o5 returns None).

Monotonicity invariant: a LARGE body cannot have MORE clearance than a SMALL
body through the same geometry, so d_safe_large > d_safe_small should be
physically impossible (INVALID sentinel / ValueError).
"""
import pytest
from egoconseq.tasks.instantiate import make_o5, o5_label


# ── o5_label tests ────────────────────────────────────────────────────────────

class TestO5Label:
    def test_small_only(self):
        """small passes (d_safe_small >= H), large contacts (d_safe_large < H)."""
        assert o5_label(1.2, 0.6, 1.0) == "small_only"

    def test_both_pass(self):
        """Both d_safe >= H → neither contacts → label 'both'."""
        assert o5_label(1.2, 1.1, 1.0) == "both"

    def test_neither_passes(self):
        """Both d_safe < H → both contact → label 'neither'."""
        assert o5_label(0.5, 0.4, 1.0) == "neither"

    def test_exact_boundary_large_passes(self):
        """d_safe == H counts as passes (>=)."""
        assert o5_label(1.0, 1.0, 1.0) == "both"

    def test_exact_boundary_small_only(self):
        """d_safe_small == H (passes), d_safe_large just below."""
        assert o5_label(1.0, 0.99, 1.0) == "small_only"

    def test_invalid_large_gt_small(self):
        """d_safe_large > d_safe_small is a monotonicity violation → INVALID or ValueError."""
        # Large body cannot have MORE clearance than small body in same geometry.
        result = o5_label(0.6, 1.2, 1.0)
        # Implementation may return "INVALID" or raise; both are acceptable.
        assert result == "INVALID", (
            f"Expected 'INVALID' for monotonicity violation, got {result!r}"
        )


# ── make_o5: fixed-metric horizon tests (RED LINE §6) ─────────────────────────

class TestO5FixedMetricHorizon:
    """Verify that horizon_reference is 'metric_fixed' and horizon_m is stored — not body-width derived."""

    def test_horizon_reference_is_metric_fixed(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1"
        )
        assert case is not None, "Expected a Case, got None"
        assert case.action["horizon_reference"] == "metric_fixed"

    def test_horizon_m_stored_correctly(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1"
        )
        assert case is not None
        assert case.action["horizon_m"] == 1.0

    def test_action_type_is_forward(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1"
        )
        assert case is not None
        assert case.action["type"] == "forward"


# ── make_o5: label propagation ────────────────────────────────────────────────

class TestO5LabelPropagation:
    def test_label_small_only(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1"
        )
        assert case is not None
        assert case.answer["label"] == "small_only"
        assert o5_label(1.2, 0.6, 1.0) == "small_only"

    def test_label_both(self):
        # d_safe_small=1.5, H=1.0, r_small=0.10: margin=0.5 >= 0.10 ✓
        # d_safe_large=1.5, H=1.0, r_large=0.40: margin=0.5 >= 0.40 ✓
        # o5_label check: both pass, so label="both"
        case = make_o5(
            d_safe_small=1.5, d_safe_large=1.5,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g2"
        )
        assert case is not None
        assert case.answer["label"] == "both"
        assert o5_label(1.2, 1.1, 1.0) == "both"  # pure label logic, no margin

    def test_label_neither(self):
        # d_safe_small=0.5 < H=1.0 → neither passes
        # margin check: abs(0.5 - 1.0)=0.5 >= 0.5*(2*0.10)=0.10 ✓
        # margin check: abs(0.4 - 1.0)=0.6 >= 0.5*(2*0.40)=0.40 ✓
        case = make_o5(
            d_safe_small=0.5, d_safe_large=0.4,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g3"
        )
        assert case is not None
        assert case.answer["label"] == "neither"
        assert o5_label(0.5, 0.4, 1.0) == "neither"


# ── make_o5: answer schema ────────────────────────────────────────────────────

class TestO5AnswerSchema:
    def test_answer_type_pair_flip(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1"
        )
        assert case is not None
        assert case.answer["answer_type"] == "pair_flip"

    def test_answer_options_three_values(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1"
        )
        assert case is not None
        opts = case.answer["options"]
        assert set(opts) == {"both", "small_only", "neither"}

    def test_operation_id_and_readout_tag(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1"
        )
        assert case is not None
        assert case.operation_id == "O5"
        assert case.readout_tag == "pair_flip"


# ── make_o5: body carries both radii ─────────────────────────────────────────

class TestO5BodySchema:
    def test_body_has_both_radii(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1"
        )
        assert case is not None
        assert case.body["radius_small_m"] == 0.10
        assert case.body["radius_large_m"] == 0.40

    def test_group_id_propagated(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="my_group"
        )
        assert case is not None
        assert case.group_id == "my_group"

    def test_geometry_tag_default(self):
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1"
        )
        assert case is not None
        assert case.tags.get("geometry_tag") == "narrow-gap"


# ── make_o5: margin-reject rule ───────────────────────────────────────────────

class TestO5MarginReject:
    """
    Margin rule: abs(d_safe_X - H) >= 0.5 * (2 * r_X) for each body.
    If EITHER body violates the margin, make_o5 returns None.
    """

    def test_reject_small_too_close_to_horizon(self):
        """
        d_safe_small=1.05, H=1.0, r_small=0.10
        abs(1.05-1.0)=0.05 < 0.5*(2*0.10)=0.10 → reject
        """
        case = make_o5(
            d_safe_small=1.05, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g_reject"
        )
        assert case is None, (
            f"Expected None (small too close to margin), got {case}"
        )

    def test_reject_large_too_close_to_horizon(self):
        """
        d_safe_large=0.7, H=1.0, r_large=0.40
        abs(0.7-1.0)=0.30 < 0.5*(2*0.40)=0.40 → reject
        """
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.7,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g_reject2"
        )
        assert case is None, (
            f"Expected None (large too close to margin), got {case}"
        )

    def test_accept_both_margins_satisfied(self):
        """
        d_safe_small=1.2, H=1.0, r_small=0.10
        abs(1.2-1.0)=0.20 >= 0.10 ✓
        d_safe_large=0.6, H=1.0, r_large=0.40
        abs(0.6-1.0)=0.40 >= 0.40 ✓  (exactly at threshold → accept)
        """
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g_accept"
        )
        assert case is not None, "Expected a Case when both margins satisfied"

    def test_reject_both_margins_violated(self):
        """Both bodies are ambiguously close to H → reject."""
        case = make_o5(
            d_safe_small=1.01, d_safe_large=0.99,
            horizon_m=1.0,
            r_small=0.10, r_large=0.10,
            group_id="g_reject3"
        )
        assert case is None

    def test_accept_exactly_at_margin_both(self):
        """
        d_safe_small=1.10, H=1.0, r_small=0.10
        abs(0.10) >= 0.10 ✓ (exactly)
        d_safe_large=0.60, H=1.0, r_large=0.40
        abs(0.40) >= 0.40 ✓ (exactly)
        """
        case = make_o5(
            d_safe_small=1.10, d_safe_large=0.60,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g_exact"
        )
        assert case is not None


# ── make_o5: monotonicity-violation handling ──────────────────────────────────

class TestO5MonotonicityViolation:
    def test_invalid_large_gt_small_returns_invalid_or_raises(self):
        """d_safe_large > d_safe_small is physically impossible; o5_label must return INVALID."""
        result = o5_label(d_safe_small=0.6, d_safe_large=1.2, horizon_m=1.0)
        assert result == "INVALID"

    def test_make_o5_invalid_label_returns_none(self):
        """When o5_label returns INVALID, make_o5 must return None (not crash)."""
        case = make_o5(
            d_safe_small=0.6, d_safe_large=1.2,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g_invalid"
        )
        assert case is None, (
            f"Expected None for monotonicity violation, got {case}"
        )


# ── make_o5: meta kwargs propagation ─────────────────────────────────────────

class TestO5MetaKwargs:
    def test_extra_meta_stored(self):
        """Additional kwargs passed via **meta must be accessible on the case."""
        case = make_o5(
            d_safe_small=1.2, d_safe_large=0.6,
            horizon_m=1.0,
            r_small=0.10, r_large=0.40,
            group_id="g1",
            scene_id="sc001",
            episode_id="ep042",
            image_path="/data/img.jpg",
        )
        assert case is not None
        assert case.scene_id == "sc001"
        assert case.episode_id == "ep042"
        assert case.image_path == "/data/img.jpg"
