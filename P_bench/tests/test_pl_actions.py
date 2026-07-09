"""Pure-geometry action tests (no Habitat)."""

import math

import pytest

from pipeline.actions import (
    Turn, Forward, parse_actions, actions_to_dicts,
    wrap_deg, net_turn_deg, total_forward_m,
    pose_after, pose_at_arc, sample_path, contact_action_index,
)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_contact_action_index_multi_forward():
    acts = [Turn(15), Forward(1.5), Forward(1.5)]     # arc spans: F0=[0,1.5], F1=[1.5,3]
    idx, local = contact_action_index(acts, 1.78)
    assert idx == 2 and _close(local, 0.28, 1e-6)     # forward at index 2, 0.28 m in


def test_contact_action_index_first_leg():
    idx, local = contact_action_index([Forward(1.0)], 0.6)
    assert idx == 0 and _close(local, 0.6)


def test_contact_action_index_after_turns():
    acts = [Turn(30), Forward(0.5), Turn(-30), Forward(1.0)]   # F0=[0,0.5], F1=[0.5,1.5]
    idx, local = contact_action_index(acts, 1.2)
    assert idx == 3 and _close(local, 0.7)             # turns consume no arc


def test_contact_action_index_beyond_total_is_none():
    idx, local = contact_action_index([Forward(1.0)], 5.0)
    assert idx is None and local is None


# --- pose_after known answers -------------------------------------------

def test_pose_after_forward():
    x, z, hd = pose_after([Forward(1.0)])
    assert _close(x, 0.0) and _close(z, 1.0) and _close(hd, 0.0)


def test_pose_after_right_then_forward():
    x, z, hd = pose_after([Turn(90), Forward(1.0)])
    assert _close(x, 1.0, 1e-9) and _close(z, 0.0, 1e-9) and _close(hd, 90.0)


def test_pose_after_square_back_to_axis():
    x, z, hd = pose_after([Turn(90), Forward(1.0), Turn(-90), Forward(1.0)])
    assert _close(x, 1.0, 1e-9) and _close(z, 1.0, 1e-9) and _close(hd, 0.0)


def test_pose_after_left_turn():
    x, z, hd = pose_after([Turn(-90), Forward(2.0)])
    assert _close(x, -2.0, 1e-9) and _close(z, 0.0, 1e-9) and _close(hd, -90.0)


# --- scalar summaries ---------------------------------------------------

def test_net_turn_and_total_forward():
    acts = [Turn(-30), Forward(1.2), Turn(15), Forward(0.5)]
    assert _close(net_turn_deg(acts), -15.0)
    assert _close(total_forward_m(acts), 1.7)


@pytest.mark.parametrize("d,expect", [
    (0, 0), (30, 30), (-30, -30), (180, 180), (-180, 180),
    (190, -170), (360, 0), (540, 180), (-540, 180),
])
def test_wrap_deg(d, expect):
    assert _close(wrap_deg(d), expect)


# --- pose_at_arc --------------------------------------------------------

def test_pose_at_arc_mid_leg():
    x, z, hd = pose_at_arc([Forward(1.0)], 0.5)
    assert _close(x, 0.0) and _close(z, 0.5) and _close(hd, 0.0)


def test_pose_at_arc_after_turn():
    x, z, hd = pose_at_arc([Turn(90), Forward(2.0)], 0.5)
    assert _close(x, 0.5, 1e-9) and _close(z, 0.0, 1e-9) and _close(hd, 90.0)


def test_pose_at_arc_boundary_does_not_apply_later_turn():
    # Stops exactly at end of first forward -> the R(90) is never applied.
    x, z, hd = pose_at_arc([Forward(1.0), Turn(90), Forward(1.0)], 1.0)
    assert _close(x, 0.0) and _close(z, 1.0) and _close(hd, 0.0)


def test_pose_at_arc_beyond_total_equals_pose_after():
    acts = [Turn(30), Forward(1.0), Turn(-15), Forward(0.5)]
    assert pose_at_arc(acts, 99.0) == pose_after(acts)


# --- sample_path --------------------------------------------------------

def test_sample_path_origin_and_exact_endpoint():
    s = sample_path([Forward(1.0)], step=0.02)
    assert s[0] == (0.0, 0.0, 0.0, 0.0)
    xe, ze, he, arce = s[-1]
    assert _close(xe, 0.0) and _close(ze, 1.0, 1e-9) and _close(arce, 1.0, 1e-9)
    # monotone arc
    arcs = [a for *_, a in s]
    assert all(arcs[i] < arcs[i + 1] + 1e-12 for i in range(len(arcs) - 1))


def test_sample_path_noninteger_leg_hits_exact_endpoint():
    s = sample_path([Forward(0.03)], step=0.02)
    # samples: origin, 0.02, 0.03(exact)
    assert _close(s[-1][3], 0.03)
    assert _close(s[-1][1], 0.03, 1e-9)


def test_sample_path_turn_adds_no_sample():
    only_turn = sample_path([Turn(45), Turn(-10)], step=0.02)
    assert only_turn == [(0.0, 0.0, 0.0, 0.0)]


def test_sample_path_multi_leg_arc_accumulates():
    s = sample_path([Turn(90), Forward(0.5), Turn(-90), Forward(0.5)], step=0.02)
    assert _close(s[-1][3], 1.0, 1e-9)   # total arc
    assert _close(s[-1][0], 0.5, 1e-9) and _close(s[-1][1], 0.5, 1e-9)


def test_parse_and_roundtrip():
    raw = [{"type": "turn", "deg": -30}, {"type": "forward", "m": 1.2}]
    acts = parse_actions(raw)
    assert acts == [Turn(-30.0), Forward(1.2)]
    assert actions_to_dicts(acts) == [{"type": "turn", "deg": -30.0},
                                      {"type": "forward", "m": 1.2}]
