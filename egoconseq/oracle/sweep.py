"""Swept-cylinder oracle: d_safe_visible.

Canonical ground frame: +x = right, +z = forward, +y = up.
The agent footprint is a disk of radius *radius* centred at each sample
along the path returned by ``geometry.swept_path``.

Arc-length convention
---------------------
``swept_path`` returns samples [0, 1, ..., n] where sample *i* corresponds
to arc length ``i * step``.  Sample 0 is the origin (agent at rest) — we
skip it in the contact search because the agent is *already* at the origin.
Contact is declared at the FIRST sample i >= 1 whose ``support_count``
reaches ``min_support``.  The reported ``d_safe`` is that arc length
``i * step``.  If no contact is found within d_max the result is
``d_safe = d_max, contact_xy = None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from egoconseq import config
from egoconseq import geometry
from egoconseq.oracle.voxel import VoxelField


@dataclass
class SweepResult:
    """Result of a swept-cylinder forward march.

    Attributes
    ----------
    d_safe : float
        Arc length (metres) to the first contact, or *d_max* if none.
    contact_xy : tuple[float, float] | None
        (x, z) of the contacting disk centre in agent-local ground frame,
        or ``None`` if no contact within d_max.
    """

    d_safe: float
    contact_xy: Optional[Tuple[float, float]]


def d_safe_visible(
    vf: VoxelField,
    radius: float,
    turn_deg: float,
    d_max: float = config.D_MAX_M,
    step: float = config.MARCH_STEP_M,
    min_support: int = config.MIN_SUPPORT_VOXELS,
) -> SweepResult:
    """March a swept cylinder along the path and return the first contact.

    Parameters
    ----------
    vf : VoxelField
        Obstacle voxel field built from the current depth frame.
    radius : float
        Footprint disk radius in metres.
    turn_deg : float
        In-place yaw change before marching forward (degrees, + = right).
    d_max : float
        Maximum arc length to march (metres).
    step : float
        Sample spacing along the path (metres).
    min_support : int
        Minimum number of occupied voxels within *radius* to declare contact.

    Returns
    -------
    SweepResult
        ``d_safe`` = arc-length to first contact (or d_max); ``contact_xy``
        = (x, z) of that sample or ``None``.
    """
    path = geometry.swept_path(turn_deg, forward_m=d_max, step=step)

    # path[0] is the origin (i=0, arc-length=0); start checking from index 1.
    for i, (x, z, _heading) in enumerate(path):
        if i == 0:
            # Skip the current footprint — agent is already here.
            continue
        if vf.support_count(center=(x, z), radius=radius) >= min_support:
            arc_len = i * step
            return SweepResult(d_safe=arc_len, contact_xy=(x, z))

    # No contact found within d_max.
    return SweepResult(d_safe=d_max, contact_xy=None)
