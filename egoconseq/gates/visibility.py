"""Visibility gates G1 and G2 for EgoConseq-Bench (Task 14b).

Spec §5:
  G1 — coarse sweep coverage: enough of the swept-corridor footprint must
       project inside the camera frame to trust the depth reading at all.
  G2 — hard no-contact evidence gate: even in-frame pixels may have depth holes
       (sensor dropout, reflective surfaces, sky) or may be occluded by a nearer
       object.  G2 **must** be satisfied before accepting a no-contact label in
       P2+ annotation; without it, "can't see ≠ nothing there" and the label is
       silently corrupted.

Public API
----------
corridor_visibility(depth, corridor_pixels, d_max, sample_distances=None) → dict
    Compute all visibility metrics for a swept-corridor footprint.

passes_g1(metrics) → bool
    True iff visible_sweep_ratio ≥ GATE_VISIBLE_SWEEP_RATIO (0.7).

passes_g2(metrics) → bool
    True iff valid_depth_ratio ≥ GATE_VALID_DEPTH_RATIO (0.9)
           AND depth_hole_ratio ≤ GATE_DEPTH_HOLE_RATIO (0.05)
           AND occlusion_free_ratio ≥ OCCLUSION_FREE_THRESHOLD (0.9)
           AND in_frame_count > 0.

Notes
-----
- ``corridor_pixels`` is a list of (u, v) integer pixel coordinates where
  u = column in [0, W) and v = row in [0, H).  Some entries may be OOF
  (u < 0, u >= W, v < 0, or v >= H); those are excluded from in-frame depth
  computations.
- depth is indexed as depth[v, u]  (NumPy row-major: row = v, col = u).
- ``sample_distances`` is an optional parallel list of per-pixel along-ray 3-D
  distances (metres) for the queried corridor surface points.  When provided,
  ``occlusion_free_ratio`` = fraction of in-frame pixels where
  ``depth[v,u] + OCCLUSION_TOL_M >= sample_distance`` (rendered depth is at
  least as far as the queried point, i.e. nothing nearer occludes it).  When
  ``sample_distances`` is None, ``occlusion_free_ratio`` is set to 1.0 and
  occlusion is considered "not assessed".
- A pixel depth is considered valid (not a hole) if it is nonzero, finite, and
  ≤ d_max + DEPTH_MARGIN_M.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from egoconseq import config

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Extra margin above d_max within which depth is still considered "valid".
#: Handles cases where the surface is measured slightly beyond the nominal cap.
DEPTH_MARGIN_M: float = 1.0

#: Tolerance for occlusion check: a sample at distance d is considered
#: visible (not occluded) if depth[v,u] >= sample_distance - OCCLUSION_TOL_M.
#: 0.2 m gives one voxel width of slack for depth noise.
OCCLUSION_TOL_M: float = 0.2

#: Minimum required occlusion_free_ratio for G2 to pass.
OCCLUSION_FREE_THRESHOLD: float = 0.9


# ---------------------------------------------------------------------------
# Main metric function
# ---------------------------------------------------------------------------

def corridor_visibility(
    depth: np.ndarray,
    corridor_pixels: List[Tuple[int, int]],
    d_max: float,
    sample_distances: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Compute visibility metrics for a swept-corridor footprint.

    Parameters
    ----------
    depth : np.ndarray, shape (H, W), dtype float32
        Rendered depth image.  ``depth[v, u]`` is the metric depth (metres) at
        pixel column u, row v.
    corridor_pixels : list of (u, v) int tuples
        Projected swept-corridor footprint sample points.
        u = column index (0-based), v = row index (0-based).
        May include out-of-frame coordinates.
    d_max : float
        Nominal maximum sensing distance (metres).  Depths beyond
        ``d_max + DEPTH_MARGIN_M`` are treated as invalid/sentinel.
    sample_distances : list of float or None
        Optional per-pixel along-ray 3-D distances (same length as
        corridor_pixels).  When provided, used to compute
        ``occlusion_free_ratio``.  If None, occlusion_free_ratio = 1.0
        (not assessed; caller is responsible for interpreting this).

    Returns
    -------
    dict with keys:
        visible_sweep_ratio   — fraction of corridor_pixels inside the frame.
        valid_depth_ratio     — fraction of in-frame pixels with valid depth.
        depth_hole_ratio      — fraction of in-frame pixels with invalid depth.
        occlusion_free_ratio  — fraction of in-frame pixels not occluded by
                                a nearer surface (1.0 if sample_distances=None).
        in_frame_count        — number of in-frame pixels (int, stored as float).
        total_count           — total number of corridor_pixels (int, stored as float).
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D (H, W), got shape {depth.shape}")

    H, W = depth.shape
    n_total = len(corridor_pixels)

    if n_total == 0:
        return {
            "visible_sweep_ratio": 0.0,
            "valid_depth_ratio": 1.0,   # trivially true — no in-frame evidence
            "depth_hole_ratio": 0.0,
            "occlusion_free_ratio": 1.0,
            "in_frame_count": 0.0,
            "total_count": 0.0,
        }

    # ------------------------------------------------------------------
    # 1. Partition in-frame vs out-of-frame
    # ------------------------------------------------------------------
    in_frame_mask = np.array(
        [(0 <= u < W) and (0 <= v < H) for u, v in corridor_pixels],
        dtype=bool,
    )
    n_in_frame = int(in_frame_mask.sum())
    visible_sweep_ratio = n_in_frame / n_total

    # ------------------------------------------------------------------
    # 2. Depth validity for in-frame pixels
    # ------------------------------------------------------------------
    if n_in_frame == 0:
        return {
            "visible_sweep_ratio": visible_sweep_ratio,
            "valid_depth_ratio": 1.0,   # no in-frame data → trivially "no holes"
            "depth_hole_ratio": 0.0,
            "occlusion_free_ratio": 1.0,
            "in_frame_count": 0.0,
            "total_count": float(n_total),
        }

    # Gather depth values at in-frame pixels; depth is indexed depth[v, u].
    in_frame_px = [corridor_pixels[i] for i in range(n_total) if in_frame_mask[i]]
    us = np.array([p[0] for p in in_frame_px], dtype=np.intp)
    vs = np.array([p[1] for p in in_frame_px], dtype=np.intp)
    d_vals = depth[vs, us]  # shape (n_in_frame,)

    d_cap = d_max + DEPTH_MARGIN_M
    valid_mask = (d_vals > 0.0) & np.isfinite(d_vals) & (d_vals <= d_cap)
    n_valid = int(valid_mask.sum())
    n_hole = n_in_frame - n_valid

    valid_depth_ratio = n_valid / n_in_frame
    depth_hole_ratio = n_hole / n_in_frame

    # ------------------------------------------------------------------
    # 3. Occlusion-free ratio
    # ------------------------------------------------------------------
    if sample_distances is None:
        # Not assessed: treat as fully visible.
        occlusion_free_ratio = 1.0
    else:
        if len(sample_distances) != n_total:
            raise ValueError(
                f"sample_distances length ({len(sample_distances)}) must equal "
                f"len(corridor_pixels) ({n_total})"
            )
        # Extract sample distances for in-frame pixels
        sd_arr = np.array(sample_distances, dtype=np.float64)
        sd_in_frame = sd_arr[in_frame_mask]
        # A pixel is occlusion-free if rendered depth >= sample_distance - tol
        # i.e. the surface is not hidden behind a nearer obstacle.
        occ_free_mask = d_vals + OCCLUSION_TOL_M >= sd_in_frame
        occlusion_free_ratio = float(occ_free_mask.sum()) / n_in_frame

    return {
        "visible_sweep_ratio": float(visible_sweep_ratio),
        "valid_depth_ratio": float(valid_depth_ratio),
        "depth_hole_ratio": float(depth_hole_ratio),
        "occlusion_free_ratio": float(occlusion_free_ratio),
        "in_frame_count": float(n_in_frame),
        "total_count": float(n_total),
    }


# ---------------------------------------------------------------------------
# Gate predicates
# ---------------------------------------------------------------------------

def passes_g1(metrics: Dict[str, float]) -> bool:
    """G1: coarse sweep-coverage gate.

    True iff ``visible_sweep_ratio >= config.GATE_VISIBLE_SWEEP_RATIO`` (0.7).

    Apply before any depth analysis — if most of the corridor projects outside
    the frame, the depth signal is too sparse to trust.
    """
    return metrics["visible_sweep_ratio"] >= config.GATE_VISIBLE_SWEEP_RATIO


def passes_g2(metrics: Dict[str, float]) -> bool:
    """G2: hard no-contact evidence gate.

    True iff ALL of the following hold:
      1. valid_depth_ratio >= config.GATE_VALID_DEPTH_RATIO (0.9)
      2. depth_hole_ratio  <= config.GATE_DEPTH_HOLE_RATIO  (0.05)
      3. occlusion_free_ratio >= OCCLUSION_FREE_THRESHOLD   (0.9)
      4. in_frame_count > 0  (at least one in-frame pixel exists)

    **Apply G2 only to no-contact labels** (O5/O4/O1/O3 outcomes in P2+).
    For contact labels, use a different check: the contact pixel must be
    in-frame and depth-consistent (handled in Task 9).

    Failing G2 means the no-contact annotation is unreliable — likely corrupted
    by depth holes, sensor dropout, or occlusion by a nearer surface.
    """
    return (
        metrics["in_frame_count"] > 0
        and metrics["valid_depth_ratio"] >= config.GATE_VALID_DEPTH_RATIO
        and metrics["depth_hole_ratio"] <= config.GATE_DEPTH_HOLE_RATIO
        and metrics["occlusion_free_ratio"] >= OCCLUSION_FREE_THRESHOLD
    )
