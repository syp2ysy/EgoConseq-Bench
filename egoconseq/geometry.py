import numpy as np


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
