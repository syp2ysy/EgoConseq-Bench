"""Tests for G1/G2 visibility gates — pure numpy, no Habitat.

Tests (in TDD order):
  1. test_ratios_on_clean_corridor          — from plan spec
  2. test_g2_rejects_holey_no_contact       — from plan spec
  3. test_out_of_frame_reduces_visible_sweep_ratio
  4. test_occlusion_free_ratio_detects_occluder
  5. test_passes_g1_boundary               — passes_g1 threshold boundary
  6. test_all_out_of_frame                 — edge: all pixels OOF
  7. test_depth_indexing_row_col           — depth[v,u] order check
"""

import numpy as np
import pytest

from egoconseq.gates.visibility import corridor_visibility, passes_g1, passes_g2


# ──────────────────────────────────────────────────────────────────
# 1. Clean corridor → all ratios near 1.0
# ──────────────────────────────────────────────────────────────────

def test_ratios_on_clean_corridor():
    """All pixels valid depth, all in-frame → ratios > 0.95."""
    depth = np.full((480, 640), 3.0, np.float32)          # all-valid depth
    corridor_px = [(320, v) for v in range(240, 460)]     # (u, v) in-frame
    r = corridor_visibility(depth, corridor_px, d_max=5.0)
    assert r["valid_depth_ratio"] > 0.95
    assert r["depth_hole_ratio"] < 0.05
    assert r["visible_sweep_ratio"] > 0.95


# ──────────────────────────────────────────────────────────────────
# 2. Depth holes down corridor → G2 rejects no-contact label
# ──────────────────────────────────────────────────────────────────

def test_g2_rejects_holey_no_contact():
    """A block of zero-depth pixels through the corridor makes G2 reject."""
    depth = np.full((480, 640), 3.0, np.float32)
    depth[240:460, 300:340] = 0.0                         # invalid/hole at col 320
    corridor_px = [(320, v) for v in range(240, 460)]
    r = corridor_visibility(depth, corridor_px, d_max=5.0)
    assert not passes_g2(r), "G2 must reject when depth_hole_ratio is high"


# ──────────────────────────────────────────────────────────────────
# 3. Out-of-frame pixels reduce visible_sweep_ratio
# ──────────────────────────────────────────────────────────────────

def test_out_of_frame_reduces_visible_sweep_ratio():
    """Half the corridor pixels are OOF → visible_sweep_ratio ≈ 0.5."""
    depth = np.full((480, 640), 3.0, np.float32)
    # 100 in-frame pixels (u=320, v=0..99) + 100 out-of-frame (u=-1, v=0..99)
    in_frame = [(320, v) for v in range(100)]
    out_frame = [(-1, v) for v in range(100)]           # u < 0 → OOF
    corridor_px = in_frame + out_frame
    r = corridor_visibility(depth, corridor_px, d_max=5.0)
    assert abs(r["visible_sweep_ratio"] - 0.5) < 0.02, (
        f"Expected ~0.5, got {r['visible_sweep_ratio']}"
    )
    # In-frame depth ratios should be computed only over the 100 in-frame pixels
    assert r["valid_depth_ratio"] > 0.95
    assert r["depth_hole_ratio"] < 0.05


# ──────────────────────────────────────────────────────────────────
# 4. occlusion_free_ratio detects an occluder
# ──────────────────────────────────────────────────────────────────

def test_occlusion_free_ratio_detects_occluder():
    """When rendered depth << sample_distance, the point is occluded."""
    depth = np.full((480, 640), 1.0, np.float32)   # wall at 1 m
    # Three corridor samples at distance 3 m — all behind the 1-m wall
    corridor_px = [(320, 240), (320, 250), (320, 260)]
    sample_distances = [3.0, 3.0, 3.0]             # queried points are at 3 m
    r = corridor_visibility(
        depth, corridor_px, d_max=5.0, sample_distances=sample_distances
    )
    # depth=1.0 < sample_distance=3.0 → occluded → occlusion_free_ratio = 0
    assert r["occlusion_free_ratio"] < 0.05, (
        f"Expected ~0, got {r['occlusion_free_ratio']}"
    )


def test_occlusion_free_ratio_is_1_without_distances():
    """If sample_distances is None, occlusion_free_ratio is 1.0 (not assessed)."""
    depth = np.full((480, 640), 1.0, np.float32)
    corridor_px = [(320, 240), (320, 250)]
    r = corridor_visibility(depth, corridor_px, d_max=5.0, sample_distances=None)
    assert r["occlusion_free_ratio"] == 1.0


# ──────────────────────────────────────────────────────────────────
# 5. passes_g1 boundary
# ──────────────────────────────────────────────────────────────────

def test_passes_g1_boundary():
    """passes_g1 threshold is GATE_VISIBLE_SWEEP_RATIO=0.7; 0.70 → pass, 0.69 → fail."""
    from egoconseq import config

    # Exactly at threshold → should pass (≥ threshold)
    r_at = {"visible_sweep_ratio": config.GATE_VISIBLE_SWEEP_RATIO}
    assert passes_g1(r_at), "visible_sweep_ratio == threshold should PASS G1"

    # Just below threshold → should fail
    r_below = {"visible_sweep_ratio": config.GATE_VISIBLE_SWEEP_RATIO - 0.01}
    assert not passes_g1(r_below), "visible_sweep_ratio < threshold should FAIL G1"


# ──────────────────────────────────────────────────────────────────
# 6. All pixels out-of-frame
# ──────────────────────────────────────────────────────────────────

def test_all_out_of_frame():
    """All corridor pixels OOF → visible_sweep_ratio=0; depth ratios are 0/1 (no data)."""
    depth = np.full((480, 640), 3.0, np.float32)
    corridor_px = [(-1, v) for v in range(50)]          # all OOF
    r = corridor_visibility(depth, corridor_px, d_max=5.0)
    assert r["visible_sweep_ratio"] == 0.0
    # With no in-frame pixels, valid_depth_ratio=1.0 and depth_hole_ratio=0.0
    # (trivially true — no evidence of holes). G2 must FAIL because visible_sweep_ratio=0.
    assert not passes_g1(r), "No in-frame pixels → G1 must fail"
    assert not passes_g2(r), "No in-frame pixels → G2 must fail"


# ──────────────────────────────────────────────────────────────────
# 7. depth indexing: depth[v, u] (row=v, col=u)
# ──────────────────────────────────────────────────────────────────

def test_depth_indexing_row_col():
    """Confirm indexing is depth[v, u] — near-edge pixel in corner."""
    depth = np.zeros((480, 640), np.float32)
    # Set depth=4.0 only at row=479 (v), col=639 (u) — bottom-right corner pixel
    depth[479, 639] = 4.0
    # A single corridor pixel pointing to that corner
    corridor_px = [(639, 479)]  # (u=639, v=479) → depth[479, 639] = 4.0, valid
    r = corridor_visibility(depth, corridor_px, d_max=5.0)
    assert r["in_frame_count"] == 1
    assert r["valid_depth_ratio"] > 0.99, (
        f"Corner pixel at depth[479,639] should be valid; got {r['valid_depth_ratio']}"
    )

    # Now a pixel that is OUTSIDE the frame on the right edge
    corridor_px_oof = [(640, 479)]  # u=640 >= W=640 → OOF
    r2 = corridor_visibility(depth, corridor_px_oof, d_max=5.0)
    assert r2["visible_sweep_ratio"] == 0.0
    assert r2["in_frame_count"] == 0
