"""Depth → point cloud utilities (pure numpy, no Habitat required).

Canonical frame (project-wide):
    agent-local ground frame, origin at footprint centre ON THE FLOOR,
    +z = forward, +x = right, +y = up, metres.

Camera frame:
    +z = forward (z == depth value), +x = right, +y = up.
    For a level camera sitting at height h above the floor the only transform
    to ground frame is a y-translation: y_ground = y_cam + h.
"""

import numpy as np
from egoconseq import config


# ---------------------------------------------------------------------------
# 1. Backprojection: depth image → camera-frame 3-D points
# ---------------------------------------------------------------------------

def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject a depth image into camera-frame 3-D points.

    Parameters
    ----------
    depth : (H, W) float32/float64
        Metric depth in metres. Pixels with depth <= 0 or non-finite are dropped.
    K : (3, 3) array
        Pinhole intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].

    Returns
    -------
    pts : (N, 3) float64
        Camera-frame points.  +z = forward (z == depth value), +x = right,
        +y = up.  For pixel (u, v) at depth d:
            x = (u - cx) / fx * d
            y = (cy - v) / fy * d      # image row v grows DOWN, negate so +y is up
            z = d
    """
    depth = np.asarray(depth, dtype=np.float64)
    H, W = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # pixel grid
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))  # (H,W) each

    d = depth.ravel()
    valid = np.isfinite(d) & (d > 0)

    u = us.ravel()[valid]
    v = vs.ravel()[valid]
    d = d[valid]

    x = (u - cx) / fx * d
    y = (cy - v) / fy * d   # image row v grows DOWN; negate so camera +y points up
    z = d

    return np.stack([x, y, z], axis=1)  # (N, 3)


# ---------------------------------------------------------------------------
# 2. Camera → agent ground frame
# ---------------------------------------------------------------------------

def to_agent_ground(
    pts_cam: np.ndarray,
    camera_height: float = config.CAMERA_HEIGHT_M,
    pitch: float = 0.0,
) -> np.ndarray:
    """Transform camera-frame points to the agent ground frame.

    Agent ground frame: origin at footprint centre ON THE FLOOR,
    +z = forward, +x = right, +y = up.

    For a level camera (pitch = 0) the camera is at height *camera_height*
    above the floor and its axes are aligned with the ground frame.
    Therefore the only transform is a y-translation:
        y_ground = y_cam + camera_height

    A point at camera-frame y = -camera_height (i.e. directly on the floor
    in front of the agent) maps to y_ground = 0, as expected.

    Parameters
    ----------
    pts_cam : (N, 3) array
        Camera-frame points (+z forward, +x right, +y up).
    camera_height : float
        Height of the camera centre above the floor in metres.
    pitch : float
        Camera pitch in radians (positive = looking down).
        Currently treated as identity (reserved for future use).
        Demo cameras are level.

    Returns
    -------
    pts_ground : (N, 3) float64
        Agent-ground-frame points.
    """
    pts = np.asarray(pts_cam, dtype=np.float64).copy()

    if pitch == 0.0:
        pts[:, 1] += camera_height
    else:
        # Simple rotation around the x-axis by -pitch (camera tilted downward)
        # then translate y by camera_height.
        c, s = np.cos(-pitch), np.sin(-pitch)
        y = pts[:, 1] * c - pts[:, 2] * s
        z = pts[:, 1] * s + pts[:, 2] * c
        pts[:, 1] = y + camera_height
        pts[:, 2] = z

    return pts


# ---------------------------------------------------------------------------
# 3. Floor estimation
# ---------------------------------------------------------------------------

def estimate_floor_height(pts: np.ndarray) -> float:
    """Robustly estimate the floor y-coordinate in the ground frame.

    We exploit the strong construction prior: the floor is at ground-y ≈ 0
    (the agent stands on the floor and ``to_agent_ground`` adds the camera
    height; real floor-ahead measures ≈ 0.0–0.22), while furniture seating
    surfaces (couch / bed / table) sit at ≥ 0.4 m. Naive estimators fail:

    * a raw low percentile gets dragged far negative by a sub-floor artifact
      tail (far pixels / mesh holes projecting metres below the floor);
    * a wide-window or global histogram mode can be out-voted by a dense
      furniture surface (~0.6 m) or a far wall.

    So we histogram only points inside a TIGHT floor window (y ∈ [-0.25, +0.30],
    which excludes furniture surfaces at ≥ 0.4 m) with ~0.05 m bins and return
    the centre of the most populated bin (the floor mode) — but only when the
    floor is densely visible (>= 200 in-window points). Otherwise we return the
    constructional floor level 0.0 rather than a percentile (which would chase
    furniture or artifacts).

    Parameters
    ----------
    pts : (N, 3) array
        Agent-ground-frame points.

    Returns
    -------
    floor_y : float
        Estimated floor height (≈ 0 for properly transformed ground-frame
        input).
    """
    y = np.asarray(pts, dtype=np.float64)[:, 1]

    lo, hi, bin_w = -0.25, 0.30, 0.05
    min_support = 200
    window = y[(y >= lo) & (y <= hi)]
    if window.size < min_support:
        return 0.0  # constructional floor level

    edges = np.arange(lo, hi + bin_w, bin_w)
    counts, edges = np.histogram(window, bins=edges)
    k = int(np.argmax(counts))
    return float((edges[k] + edges[k + 1]) / 2.0)


# ---------------------------------------------------------------------------
# 4. Floor removal
# ---------------------------------------------------------------------------

def remove_floor(
    pts: np.ndarray,
    floor_y: float,
    band: tuple = config.OBSTACLE_BAND_M,
) -> np.ndarray:
    """Keep only points whose height above the floor is in [band[0], band[1]].

    Parameters
    ----------
    pts : (N, 3) array
        Agent-ground-frame points.
    floor_y : float
        Floor y-coordinate as returned by :func:`estimate_floor_height`.
    band : (low, high) tuple
        Height interval (metres above floor) to keep.  Default is
        ``config.OBSTACLE_BAND_M = (0.05, 1.5)``.

    Returns
    -------
    kept : (M, 3) float64
        Filtered point cloud.
    """
    pts = np.asarray(pts, dtype=np.float64)
    above = pts[:, 1] - floor_y
    low, high = band
    mask = (above >= low) & (above <= high)
    return pts[mask]
