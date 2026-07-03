"""
Depth-vs-navmesh disagreement taxonomy (design §4).

The depth-corridor oracle (d_depth) DEFINES the label; the navmesh distance
(d_nav) is an independent cross-check / hidden-geometry detector — NOT a
second ground truth.  This module decides whether to keep/discard/review a
sample based on how the two signals relate.

Taxonomy (tol in metres, default 0.3):
  agree           |d_depth - d_nav| <= tol
                  -> keep_agree   (keep=True)  high-confidence sample

  navmesh_earlier  d_nav < d_depth - tol   (navmesh detects contact first)
    contact_visible=False  -> discard_hidden      (keep=False)
        geometry that is present in the navmesh is NOT visible in the frame
        → single-image-unfair, drop (requires hidden geometry)
    contact_visible=True   -> review_depth_miss   (keep=False)
        depth oracle MISSED a visible obstacle → under-detection, human review

  depth_earlier   d_depth < d_nav - tol   (depth oracle detects contact first)
    contact_visible=True   -> keep_visible        (keep=True)
        real visible obstacle (e.g. furniture) ignored by navmesh floor
        abstraction → legitimately keep
    contact_visible=False  -> discard_noise       (keep=False)
        likely point-cloud noise with no visible support → discard
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    """Result of the disagreement taxonomy classifier.

    Attributes
    ----------
    verdict : str
        One of: ``"keep_agree"``, ``"keep_visible"``, ``"discard_hidden"``,
        ``"review_depth_miss"``, ``"discard_noise"``.
    keep : bool
        Whether the sample should be retained in the benchmark.
    """

    verdict: str
    keep: bool


def classify(
    d_depth: float,
    d_nav: float,
    contact_visible: bool,
    tol: float = 0.3,
) -> Verdict:
    """Classify a depth-vs-navmesh pair according to the §4 taxonomy.

    Parameters
    ----------
    d_depth : float
        Distance to contact as measured by the depth-corridor oracle (metres).
    d_nav : float
        Distance to contact as measured by the navmesh cross-check (metres).
    contact_visible : bool
        Whether the relevant contact point is visible in the current frame
        (caller-computed; see module docstring for semantics).
    tol : float, optional
        Agreement tolerance in metres (default 0.3).

    Returns
    -------
    Verdict
        Frozen dataclass with ``verdict`` (str) and ``keep`` (bool).
    """
    diff = d_depth - d_nav  # positive → navmesh earlier (d_nav < d_depth); negative → depth earlier

    if abs(diff) <= tol:
        # Both sensors agree within tolerance → high-confidence sample
        return Verdict(verdict="keep_agree", keep=True)

    if diff > 0:
        # navmesh_earlier: d_nav < d_depth - tol  (i.e. diff > tol)
        # Navmesh sees a contact the depth oracle didn't.
        if not contact_visible:
            # Hidden geometry — not fair to label from a single image
            return Verdict(verdict="discard_hidden", keep=False)
        else:
            # Depth oracle missed a visible obstacle → review / discard
            return Verdict(verdict="review_depth_miss", keep=False)

    # depth_earlier: d_depth < d_nav - tol  (i.e. diff < -tol)
    # Depth oracle sees a contact navmesh doesn't.
    if contact_visible:
        # Real visible obstacle (furniture etc.) navmesh's floor abstraction
        # ignores — legitimately a consequence the agent should predict
        return Verdict(verdict="keep_visible", keep=True)
    else:
        # No visible support → likely point-cloud noise
        return Verdict(verdict="discard_noise", keep=False)
