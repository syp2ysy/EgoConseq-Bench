"""Per-radius navmesh d_safe cross-check.

This module provides two functions:

  recompute_navmesh(sim, radius, height)
      Recompute the pathfinder's navmesh with the given agent cylinder.

  d_safe_navmesh(pathfinder, pos, yaw, turn_deg, d_max, step)
      March the swept-centerline path in world coordinates and return the
      arc-length to the first non-navigable sample.

World-frame transform
---------------------
habitat_env.render() uses::

    agent_state.rotation = quaternion.from_rotation_vector([0, yaw, 0])

which is a rotation of *yaw* radians about +Y.  The agent's local frame
has +z = forward, +x = right.  With yaw = 0 the agent looks along -Z_world
(Habitat default).

The world-frame offset for a local point (x_local, z_local) is therefore::

    right_world   = ( cos(yaw),  0, -sin(yaw))
    forward_world = (-sin(yaw),  0, -cos(yaw))

    world_offset  = x_local * right_world + z_local * forward_world
                  = (x_local*cos(yaw) - z_local*sin(yaw),
                     0,
                     -x_local*sin(yaw) - z_local*cos(yaw))

Verify at yaw=0: right_world=(1,0,0), forward_world=(0,0,-1) — consistent
with Habitat convention.  The navmesh march uses the same geometry as the
rendered camera forward direction.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import habitat_sim

from egoconseq import config
from egoconseq import geometry as geom


def recompute_navmesh(
    sim: habitat_sim.Simulator,
    radius: float,
    height: float = config.CYLINDER_HEIGHT_M,
) -> None:
    """Recompute the simulator's navmesh with the given agent cylinder.

    Mutates ``sim.pathfinder`` in place.  Call this before querying
    ``d_safe_navmesh`` for the desired radius.

    Parameters
    ----------
    sim : habitat_sim.Simulator
        The raw habitat_sim.Simulator object (accessible via
        ``EgoConseqSim._sim`` or a property you add to that class).
    radius : float
        Agent footprint radius in metres.
    height : float
        Agent cylinder height in metres.  Defaults to
        ``config.CYLINDER_HEIGHT_M`` (1.5 m) — do NOT hardcode.
    """
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = radius
    navmesh_settings.agent_height = height
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)


def _local_to_world(
    x_local: float,
    z_local: float,
    pos: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """Convert an agent-local ground point to world coordinates.

    Parameters
    ----------
    x_local : float
        Local +x (right) offset in metres.
    z_local : float
        Local +z (forward) offset in metres.
    pos : array-like, shape (3,)
        World position of the agent root.
    yaw : float
        Yaw in radians (rotation about +Y axis, 0 = looking along -Z_world).

    Returns
    -------
    np.ndarray, shape (3,)
        World-frame [X, Y, Z] position.
    """
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    # right_world   = ( cos(yaw), 0, -sin(yaw))
    # forward_world = (-sin(yaw), 0, -cos(yaw))
    dx = x_local * cos_y - z_local * sin_y
    dz = -x_local * sin_y - z_local * cos_y
    return np.array(
        [pos[0] + dx, pos[1], pos[2] + dz],
        dtype=np.float64,
    )


def d_safe_navmesh(
    pathfinder: habitat_sim.PathFinder,
    pos: np.ndarray,
    yaw: float,
    turn_deg: float,
    d_max: float = config.D_MAX_M,
    step: float = config.MARCH_STEP_M,
    max_y_delta: float = 0.5,
) -> float:
    """March the swept centerline in world coordinates; return arc-length to
    first non-navigable point (or *d_max* if all navigable).

    Uses the same yaw → world-frame transform as ``habitat_env.render()``
    so the navmesh march direction agrees with the rendered camera forward.

    Parameters
    ----------
    pathfinder : habitat_sim.PathFinder
        The pathfinder belonging to the simulator (already recomputed for the
        desired radius via ``recompute_navmesh``).
    pos : array-like, shape (3,)
        World-frame XYZ position of the agent root.
    yaw : float
        Yaw angle in radians, same convention as ``EgoConseqSim.render()``.
    turn_deg : float
        In-place yaw change before marching forward (degrees, + = right),
        consistent with ``geometry.swept_path``.
    d_max : float
        Maximum arc length to march in metres.
    step : float
        Sample spacing along the path in metres.
    max_y_delta : float
        ``is_navigable`` Y-snap tolerance in metres.  0.5 m is a sensible
        default for indoor scenes.

    Returns
    -------
    float
        Arc-length in metres to the first non-navigable sample, or *d_max*
        if all samples are navigable.
    """
    pos = np.asarray(pos, dtype=np.float64)
    path_samples = geom.swept_path(turn_deg, forward_m=d_max, step=step)

    # path_samples[0] is the origin (arc-length 0); skip it.
    for i, (x_local, z_local, _heading) in enumerate(path_samples):
        if i == 0:
            continue
        world_pt = _local_to_world(x_local, z_local, pos, yaw)
        if not pathfinder.is_navigable(world_pt, max_y_delta=max_y_delta):
            arc_len = i * step
            return arc_len

    return float(d_max)
