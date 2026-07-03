"""
TDD tests for depth-vs-navmesh disagreement taxonomy (design §4).

Taxonomy rules (tol=0.3 by default):
  agree         |d_depth - d_nav| <= tol  -> keep_agree,  keep=True
  navmesh_earlier  d_nav < d_depth - tol
    contact_visible=False  -> discard_hidden,      keep=False
    contact_visible=True   -> review_depth_miss,   keep=False
  depth_earlier    d_depth < d_nav - tol
    contact_visible=True   -> keep_visible,        keep=True
    contact_visible=False  -> discard_noise,       keep=False
"""

import pytest
from egoconseq.oracle.disagreement import classify


# ─── plan's canonical four-way taxonomy test ──────────────────────────────────

def test_taxonomy():
    # navmesh earlier + contact NOT visible -> hidden geometry, discard
    assert classify(d_depth=5.0, d_nav=1.0, contact_visible=False).verdict == "discard_hidden"
    # navmesh earlier + visible -> depth miss
    assert classify(5.0, 1.0, contact_visible=True).verdict == "review_depth_miss"
    # depth earlier + no clear visible obstacle -> noise
    assert classify(1.0, 5.0, contact_visible=False).verdict == "discard_noise"
    # depth earlier + clear visible obstacle -> keep visible obstacle
    assert classify(1.0, 5.0, contact_visible=True).verdict == "keep_visible"
    # agree -> high confidence
    assert classify(1.45, 1.5, contact_visible=True).verdict == "keep_agree"


# ─── keep-bool correctness for every verdict ──────────────────────────────────

def test_keep_flags():
    assert classify(1.45, 1.5, contact_visible=True).keep is True,   "keep_agree should keep"
    assert classify(1.0,  5.0, contact_visible=True).keep is True,   "keep_visible should keep"
    assert classify(5.0,  1.0, contact_visible=False).keep is False,  "discard_hidden should not keep"
    assert classify(5.0,  1.0, contact_visible=True).keep is False,   "review_depth_miss should not keep"
    assert classify(1.0,  5.0, contact_visible=False).keep is False,  "discard_noise should not keep"


# ─── boundary: just under tol → agree; just over → disagree ──────────────────

def test_boundary_agree_vs_disagree():
    # Use values that are exactly representable in IEEE-754 to avoid
    # floating-point representation error at the boundary.
    # tol=0.25 is a power-of-two fraction and is exactly representable.
    tol = 0.25

    # |d_depth - d_nav| == tol exactly  -> agree  (<=)
    # 1.0 - 0.75 = 0.25 exactly in IEEE-754
    v = classify(d_depth=1.0, d_nav=0.75, contact_visible=False, tol=tol)
    assert v.verdict == "keep_agree", f"expected keep_agree at exact tol, got {v.verdict}"

    v_sym = classify(d_depth=0.75, d_nav=1.0, contact_visible=False, tol=tol)
    assert v_sym.verdict == "keep_agree", f"expected keep_agree (symmetric), got {v_sym.verdict}"

    # |d_depth - d_nav| just over tol -> disagree
    # Use a gap that is definitely larger than tol=0.25
    # d_depth=1.0, d_nav=0.5 -> diff=0.5 > 0.25 -> navmesh_earlier, not visible -> discard_hidden
    v2 = classify(d_depth=1.0, d_nav=0.5, contact_visible=False, tol=tol)
    assert v2.verdict == "discard_hidden", f"expected discard_hidden past tol, got {v2.verdict}"

    # depth_earlier past tol: d_depth=0.5, d_nav=1.0 -> diff=-0.5 < -0.25 -> discard_noise
    v3 = classify(d_depth=0.5, d_nav=1.0, contact_visible=False, tol=tol)
    assert v3.verdict == "discard_noise", f"expected discard_noise past tol, got {v3.verdict}"


# ─── custom tol parameter ─────────────────────────────────────────────────────

def test_custom_tol():
    # with tol=1.0, d_depth=5.0 d_nav=1.0 diff=4.0 > 1.0 -> navmesh earlier
    r = classify(5.0, 1.0, contact_visible=False, tol=1.0)
    assert r.verdict == "discard_hidden"

    # with tol=5.0, same pair is within tol -> agree
    r2 = classify(5.0, 1.0, contact_visible=False, tol=5.0)
    assert r2.verdict == "keep_agree"


# ─── pure function: no mutable state leakage ─────────────────────────────────

def test_pure_repeated_calls():
    r1 = classify(1.0, 5.0, True)
    r2 = classify(1.0, 5.0, True)
    assert r1.verdict == r2.verdict
    assert r1.keep == r2.keep
