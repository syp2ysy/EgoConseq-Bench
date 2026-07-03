"""
Tests for egoconseq.tasks.instantiate: o5_label(), make_o5_case(), o5_group_has_flip()
TDD: new single-robot binary contact design.

Design:
- ONE robot per case, cylindrical chassis, given radius_m.
- Binary label: "contact" if d_safe_m < horizon_m else "no_contact".
- GT evidence stored in tags["gt_evidence"].
- Margin rule: |d_safe_m - horizon_m| >= radius_m (reject if not).
- o5_group_has_flip: True iff group has BOTH "contact" and "no_contact" labels.
"""
import pytest
from egoconseq.tasks.instantiate import o5_label, make_o5_case, o5_group_has_flip


# ── o5_label tests ────────────────────────────────────────────────────────────

def test_o5_binary_label():
    """Basic binary contact labelling."""
    assert o5_label(0.6, 1.0) == "contact"      # 0.6 < 1.0
    assert o5_label(1.4, 1.0) == "no_contact"   # 1.4 >= 1.0


def test_o5_label_exact_boundary():
    """d_safe == horizon → no_contact (>= boundary)."""
    assert o5_label(1.0, 1.0) == "no_contact"


def test_o5_label_just_below_horizon():
    """d_safe just below horizon → contact."""
    assert o5_label(0.99, 1.0) == "contact"


# ── make_o5_case: evidence storage ───────────────────────────────────────────

def test_make_case_stores_evidence_and_fixed_metric():
    """GT evidence must be stored and horizon_reference must be metric_fixed."""
    c = make_o5_case(d_safe_m=0.6, radius_m=0.40, horizon_m=1.0, group_id="g1")
    assert c is not None, "Expected a Case, got None"
    assert c.action["horizon_reference"] == "metric_fixed"
    assert c.answer["label"] == "contact"
    assert c.d_safe_visible_m == 0.6
    # evidence retrievable for the "why"
    ev = c.tags.get("gt_evidence")
    assert ev is not None, "gt_evidence not in tags"
    assert ev["d_safe_m"] == 0.6
    assert ev["horizon_m"] == 1.0
    assert ev["rule"] == "contact iff d_safe < H"
    # diameter stored in body
    assert c.body["diameter_m"] == pytest.approx(0.8)


def test_make_case_body_has_radius_and_diameter():
    """body must carry both radius_m and diameter_m."""
    c = make_o5_case(d_safe_m=0.6, radius_m=0.30, horizon_m=1.0, group_id="g1")
    assert c is not None
    assert c.body["radius_m"] == pytest.approx(0.30)
    assert c.body["diameter_m"] == pytest.approx(0.60)


def test_make_case_answer_schema():
    """answer dict must have answer_type, options, label."""
    c = make_o5_case(d_safe_m=0.6, radius_m=0.40, horizon_m=1.0, group_id="g1")
    assert c is not None
    assert c.answer["answer_type"] == "binary_contact"
    assert set(c.answer["options"]) == {"contact", "no_contact"}
    assert c.answer["label"] == "contact"


def test_make_case_operation_and_readout():
    """operation_id and readout_tag must match spec."""
    c = make_o5_case(d_safe_m=0.6, radius_m=0.40, horizon_m=1.0, group_id="g1")
    assert c is not None
    assert c.operation_id == "O5"
    assert c.readout_tag == "binary_contact"


def test_make_case_group_id_propagated():
    """group_id must be stored on the Case."""
    c = make_o5_case(d_safe_m=0.6, radius_m=0.40, horizon_m=1.0, group_id="my_group")
    assert c is not None
    assert c.group_id == "my_group"


def test_make_case_no_contact_label():
    """no_contact case: d_safe > horizon, margin satisfied."""
    # d_safe=1.5, horizon=1.0, radius=0.40: margin=0.5 >= 0.40 ✓
    c = make_o5_case(d_safe_m=1.5, radius_m=0.40, horizon_m=1.0, group_id="g1")
    assert c is not None
    assert c.answer["label"] == "no_contact"


# ── make_o5_case: margin rule ─────────────────────────────────────────────────

def test_margin_reject():
    """Case too close to boundary must be rejected."""
    # |0.95 - 1.0| = 0.05 < radius 0.40 → reject
    assert make_o5_case(0.95, 0.40, 1.0, "g") is None


def test_margin_accept_exactly_at_threshold():
    """Exactly at margin threshold should be accepted."""
    # |0.6 - 1.0| = 0.40 == radius 0.40 → accept
    c = make_o5_case(d_safe_m=0.6, radius_m=0.40, horizon_m=1.0, group_id="g")
    assert c is not None


def test_margin_reject_no_contact_side():
    """no_contact case too close to boundary must be rejected."""
    # d_safe=1.05, horizon=1.0, radius=0.10: |0.05| < 0.10 → reject
    assert make_o5_case(1.05, 0.10, 1.0, "g") is None


def test_margin_accept_no_contact_clear():
    """no_contact case clearly above horizon by >= radius → accept."""
    # d_safe=1.5, horizon=1.0, radius=0.10: |0.5| >= 0.10 ✓
    c = make_o5_case(d_safe_m=1.5, radius_m=0.10, horizon_m=1.0, group_id="g")
    assert c is not None
    assert c.answer["label"] == "no_contact"


# ── make_o5_case: meta kwargs propagation ─────────────────────────────────────

def test_meta_kwargs_propagated():
    """Extra kwargs (scene_id, episode_id, image_path) must be stored."""
    c = make_o5_case(
        d_safe_m=0.6, radius_m=0.40, horizon_m=1.0, group_id="g1",
        scene_id="sc001", episode_id="ep042", image_path="/data/img.jpg",
    )
    assert c is not None
    assert c.scene_id == "sc001"
    assert c.episode_id == "ep042"
    assert c.image_path == "/data/img.jpg"


# ── o5_group_has_flip ─────────────────────────────────────────────────────────

def test_group_flip_detection():
    """Group with both contact and no_contact → has_flip=True."""
    small = make_o5_case(1.4, 0.10, 1.0, "g")   # no_contact (1.4 >= 1.0, margin 0.4>=0.10 ✓)
    large = make_o5_case(0.5, 0.40, 1.0, "g")   # contact (0.5 < 1.0, margin 0.5>=0.40 ✓)
    assert small is not None
    assert large is not None
    assert o5_group_has_flip([small, large]) is True


def test_group_no_flip_all_same():
    """Group where all cases have same label → has_flip=False."""
    c1 = make_o5_case(1.4, 0.10, 1.0, "g")   # no_contact
    c2 = make_o5_case(1.6, 0.15, 1.0, "g")   # no_contact
    assert c1 is not None
    assert c2 is not None
    assert o5_group_has_flip([c1, c2]) is False


def test_group_flip_single_case():
    """Single case group → has_flip=False (no variation possible)."""
    c = make_o5_case(0.5, 0.40, 1.0, "g")   # contact
    assert c is not None
    assert o5_group_has_flip([c]) is False


def test_group_flip_empty():
    """Empty group → has_flip=False."""
    assert o5_group_has_flip([]) is False
