"""Habitat-sim wrapper for EgoConseq-Bench.

Provides EgoConseqSim: configure a habitat_sim.Simulator with RGB + Depth
cameras at the design-spec settings, and a .render(pos, yaw) method that
returns (rgb, depth, K, agent_state).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import quaternion  # numpy-quaternion, bundled with habitat-sim

import habitat_sim

from egoconseq import config


def _build_intrinsics() -> np.ndarray:
    """Return 3×3 camera intrinsic matrix K from HFOV and RESOLUTION=(W,H)."""
    W, H = config.RESOLUTION
    fx = (W / 2.0) / math.tan(math.radians(config.HFOV_DEG) / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K


class EgoConseqSim:
    """Thin habitat_sim wrapper for EgoConseq-Bench rendering.

    Parameters
    ----------
    scene_glb : str
        Absolute path to the scene .glb file.

    Usage
    -----
    sim = EgoConseqSim(scene_glb)
    rgb, depth, K, agent_state = sim.render(position, yaw)
    navpoint = sim.pathfinder.get_random_navigable_point()
    sim.close()
    """

    def __init__(self, scene_glb: str) -> None:
        # --- Backend ---
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_glb
        backend_cfg.scene_dataset_config_file = config.HM3D_SCENE_DATASET_CFG

        # --- Sensors ---
        hw = config.hw()  # [H, W]

        color_spec = habitat_sim.CameraSensorSpec()
        color_spec.uuid = "color"
        color_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_spec.resolution = hw
        color_spec.position = [0.0, config.CAMERA_HEIGHT_M, 0.0]
        color_spec.hfov = config.HFOV_DEG

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = hw
        depth_spec.position = [0.0, config.CAMERA_HEIGHT_M, 0.0]
        depth_spec.hfov = config.HFOV_DEG

        # --- Agent ---
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [color_spec, depth_spec]
        # Leave height (1.5 m) and radius (0.1 m) at habitat defaults.

        # --- Simulator ---
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)

        # Precompute intrinsics (constant for the lifetime of this sim).
        self._K: np.ndarray = _build_intrinsics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def sim(self) -> habitat_sim.Simulator:
        """The underlying habitat_sim.Simulator (needed by navmesh utils)."""
        return self._sim

    @property
    def pathfinder(self) -> habitat_sim.PathFinder:
        return self._sim.pathfinder

    @property
    def K(self) -> np.ndarray:
        """Camera intrinsic matrix (3×3, float64)."""
        return self._K

    def render(
        self,
        position: np.ndarray,
        yaw: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, habitat_sim.AgentState]:
        """Render an observation from a given world position and yaw.

        Parameters
        ----------
        position : array-like, shape (3,)
            World-frame XYZ position of the agent root (floor level).
        yaw : float
            Yaw angle in radians, rotation around the +Y (up) axis.
            0 = looking along −Z (Habitat default forward).

        Returns
        -------
        rgb   : np.ndarray, shape (H, W, 3), uint8
        depth : np.ndarray, shape (H, W),    float32, metres
        K     : np.ndarray, shape (3, 3),    float64, camera intrinsics
        agent_state : habitat_sim.AgentState  (position + rotation set)
        """
        # Build agent state.
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position, dtype=np.float32)
        # Habitat rotation convention: quaternion around +Y axis.
        agent_state.rotation = quaternion.from_rotation_vector(
            np.array([0.0, yaw, 0.0], dtype=np.float64)
        )

        self._sim.get_agent(0).set_state(agent_state)
        obs = self._sim.get_sensor_observations()

        rgb = obs["color"][..., :3].astype(np.uint8)          # drop alpha
        depth = obs["depth"].astype(np.float32)                # metres

        return rgb, depth, self._K, agent_state

    def close(self) -> None:
        """Release the simulator."""
        self._sim.close()

    # Context-manager support.
    def __enter__(self) -> "EgoConseqSim":
        return self

    def __exit__(self, *_) -> None:
        self.close()
