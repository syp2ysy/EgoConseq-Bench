import pytest

from pipeline.consensus import oracle_consensus
from pipeline.action_sampling import (
    classify_action_group,
)


def _physical(collision, arc=None, authority="navmesh"):
    return {
        "authority": authority,
        "collision": collision,
        "first_contact_arc_m": arc,
    }


def test_consensus_accepts_two_safe_oracles_with_coverage():
    got = oracle_consensus(_physical(False), _physical(False, authority="depth"), 0.95)
    assert got["accepted"] is True
    assert got["verdict"] == "agree_safe"
    assert got["contact_arc_difference_m"] is None


def test_consensus_accepts_close_contact_arcs_at_boundary():
    got = oracle_consensus(
        _physical(True, 1.0), _physical(True, 1.3, "depth"), 1.0,
        contact_tolerance_m=0.3,
    )
    assert got["accepted"] is True
    assert got["contact_arc_difference_m"] == pytest.approx(0.3)


def test_consensus_rejects_state_arc_and_coverage_disagreement():
    state = oracle_consensus(_physical(True, 1.0), _physical(False, authority="depth"), 1.0)
    arc = oracle_consensus(_physical(True, 1.0), _physical(True, 1.31, "depth"), 1.0)
    coverage = oracle_consensus(_physical(False), _physical(False, authority="depth"), 0.89)
    assert (state["accepted"], state["reason"]) == (False, "collision_state_mismatch")
    assert (arc["accepted"], arc["reason"]) == (False, "contact_arc_mismatch")
    assert (coverage["accepted"], coverage["reason"]) == (False, "insufficient_depth_coverage")


def test_consensus_rejects_unavailable_full_geometry():
    got = oracle_consensus(
        _physical(None, authority="unavailable"),
        _physical(False, authority="depth"),
        1.0,
    )
    assert got["accepted"] is False
    assert got["reason"] == "full_geometry_unavailable"


def test_action_group_label_requires_every_sibling():
    assert classify_action_group(
        [True], {0.15: False}, 1) == "safe"
    assert classify_action_group(
        [True], {0.15: True}, 1) == "collision"
    assert classify_action_group(
        [], {0.15: False}, 1) is None
    assert classify_action_group(
        [False], {0.15: False}, 1) is None
