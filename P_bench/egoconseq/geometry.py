import numpy as np
from typing import Tuple

from egoconseq import config


def swept_path(turn_deg: float, forward_m: float, step: float):
    """Return list of (x, z, heading_rad) centerline samples in agent-local ground frame.

    Turn is in-place (footprint unchanged), then translate forward along new heading.
    +z = forward, +x = right.

    Parameters
    ----------
    turn_deg  : in-place yaw change in degrees (positive = right)
    forward_m : distance to march forward along the new heading (metres)
    step      : sample spacing in metres (MARCH_STEP_M)

    Returns
    -------
    list of (x, z, heading_rad) tuples; first sample is always the origin (0, 0, h).
    """
    h = np.deg2rad(turn_deg)
    n = max(1, int(round(forward_m / step)))
    out = [(0.0, 0.0, h)]
    for i in range(1, n + 1):
        d = i * step
        out.append((d * np.sin(h), d * np.cos(h), h))
    return out


def project_contact(
    point_3d_ground: Tuple[float, float, float],
    K: np.ndarray,
    camera_height: float = config.CAMERA_HEIGHT_M,
    pitch: float = 0.0,
) -> Tuple[float, float]:
    """Project a ground-frame 3D point to image pixel (u, v).

    This is the inverse of the ``backproject`` + ``to_agent_ground`` pipeline
    in ``pointcloud.py``.  The v-sign convention is consistent with that
    pipeline so that the round-trip
        pixel → backproject → to_agent_ground → project_contact → pixel
    recovers the original pixel within rounding error.

    Coordinate conventions
    ----------------------
    Agent-local ground frame: origin at footprint centre ON THE FLOOR,
        +z = forward, +x = right, +y = up (metres).
    Camera frame: origin at camera centre, +z = forward, +x = right,
        +y = up.  For a level camera at height *camera_height* the
        conversion is simply:
            x_cam = x_ground
            y_cam = y_ground - camera_height    (inverse of to_agent_ground)
            z_cam = z_ground

    Pinhole projection (consistent with backproject's unprojection):
        backproject:  x = (u - cx) / fx * d
                      y = (cy - v) / fy * d      ← v grows DOWN, y points UP
                      z = d
        Inverting:    u = fx * x_cam / z_cam + cx
                      v = cy - fy * y_cam / z_cam  ← same sign as backproject

    Parameters
    ----------
    point_3d_ground : (x, y, z) tuple
        3D point in the agent-local ground frame (metres).
    K : (3, 3) array
        Pinhole intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    camera_height : float
        Height of the camera centre above the floor in metres.
        Default: ``config.CAMERA_HEIGHT_M`` (1.5 m).
    pitch : float
        Camera pitch in radians (positive = looking down).
        Currently only pitch=0 is supported (demo cameras are level).

    Returns
    -------
    (u, v) : tuple of float
        Pixel coordinates (float, not rounded) in the image plane.
        To get integer pixel indices use ``int(round(u))``, ``int(round(v))``.

    Notes
    -----
    Contact-height choice for O6 substrate: the sweep oracle reports contact
    at the ground-plane disk centre.  For display purposes the caller should
    raise the y-coordinate to a representative obstacle-surface height, e.g.
        y_contact = floor_y + 0.5   # mid obstacle band (~0.5 m above floor)
    before calling ``project_contact``.  This function does NOT apply that
    offset — it projects whatever 3D point is passed in.
    """
    x_g, y_g, z_g = float(point_3d_ground[0]), float(point_3d_ground[1]), float(point_3d_ground[2])

    if pitch != 0.0:
        raise NotImplementedError("Non-zero pitch not yet implemented; demo cameras are level.")

    # Ground frame → camera frame (inverse of to_agent_ground for pitch=0)
    x_cam = x_g
    y_cam = y_g - camera_height
    z_cam = z_g

    if z_cam <= 0:
        raise ValueError(f"Point is at or behind the camera (z_cam={z_cam:.4f}). "
                         "project_contact requires z > 0.")

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Pinhole projection — v formula mirrors backproject's y = (cy-v)/fy*d
    u = fx * x_cam / z_cam + cx
    v = cy - fy * y_cam / z_cam

    return (u, v)
