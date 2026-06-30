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

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from egoconseq import config
from egoconseq import geometry
from egoconseq.oracle.voxel import VoxelField

# Height above floor_y used as the representative contact point for O6 pixel
# projection (mid obstacle band).  Documented choice: obstacles are detected in
# the band [0.05, 1.5] m above the floor (config.OBSTACLE_BAND_M); we use
# 0.5 m as a mid-band representative height that lies well within the band and
# is visible for typical objects (chairs, tables, walls).
_CONTACT_DISPLAY_HEIGHT_ABOVE_FLOOR = 0.5  # metres


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
    contact_3d : tuple[float, float, float] | None
        3D ground-frame point used for O6 pixel projection:
        (x, floor_y + 0.5, z).  ``None`` if contact_xy is None OR if
        ``attach_contact_projection`` has not been called yet.
    contact_pixel : tuple[float, float] | None
        (u, v) pixel coordinates of the contact point projected into the
        current camera image.  ``None`` if contact_xy is None OR if
        ``attach_contact_projection`` has not been called yet.
    """

    d_safe: float
    contact_xy: Optional[Tuple[float, float]]
    contact_3d: Optional[Tuple[float, float, float]] = field(default=None)
    contact_pixel: Optional[Tuple[float, float]] = field(default=None)


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


def attach_contact_projection(
    result: SweepResult,
    floor_y: float,
    K: np.ndarray,
    camera_height: float = config.CAMERA_HEIGHT_M,
    pitch: float = 0.0,
) -> SweepResult:
    """Fill ``result.contact_3d`` and ``result.contact_pixel`` for O6 substrate.

    This is a post-processing step; it does NOT change the core sweep result
    (``d_safe``, ``contact_xy``).  Existing callers of ``d_safe_visible`` are
    unaffected — call this function only when the pixel projection is needed.

    Contact-height choice
    ---------------------
    The sweep oracle reports contact at the ground-plane disk centre (y = floor_y).
    Obstacles are detected in the band [0.05, 1.5] m above the floor
    (``config.OBSTACLE_BAND_M``).  We project the contact point raised to
    ``floor_y + 0.5`` (mid-band, ~0.5 m above floor) as a representative
    visible surface height.  This is documented in ``_CONTACT_DISPLAY_HEIGHT_ABOVE_FLOOR``.

    Parameters
    ----------
    result : SweepResult
        Output of ``d_safe_visible``.
    floor_y : float
        Floor y-coordinate in the agent ground frame (from ``estimate_floor_height``
        or 0.0 if not available).
    K : (3, 3) array
        Pinhole camera intrinsic matrix.
    camera_height : float
        Camera height above the floor in metres.
    pitch : float
        Camera pitch (radians); currently only 0 is supported.

    Returns
    -------
    SweepResult
        The same ``result`` object with ``contact_3d`` and ``contact_pixel``
        filled in (or both remaining ``None`` if ``contact_xy`` is None).
    """
    if result.contact_xy is None:
        return result  # no contact → leave fields as None

    x, z = result.contact_xy
    y_contact = floor_y + _CONTACT_DISPLAY_HEIGHT_ABOVE_FLOOR

    point_3d = (x, y_contact, z)
    result.contact_3d = point_3d
    result.contact_pixel = geometry.project_contact(
        point_3d, K, camera_height=camera_height, pitch=pitch
    )
    return result
