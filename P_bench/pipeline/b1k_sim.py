"""OmniGibson-backed B1K session with an injectable, lazy runtime boundary.

Importing this module never imports OmniGibson, Isaac Sim, or Habitat.  The
real runtime is loaded only when a session is opened without an injected
facade; tests use the same session logic with a small in-memory facade.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

import numpy as np

from pipeline import (
    b1k_geometry, b1k_observation, b1k_semantic, config, floor_plane, perception,
    pose_calibration, record,
)
from pipeline.frame import (
    RenderObservation, SensorProfile, TerminalRGBObservation,
    validate_render_observation,
)
from pipeline.scene_pool import SceneSpec


B1K_SENSOR_MODALITIES = ("rgb", "depth_linear")
B1K_OBSERVATION_MODALITIES = b1k_observation.OBSERVATION_MODALITIES


class NoFiniteDepthObservation(ValueError):
    """The renderer returned no usable positive finite depth sample."""


_ACTIVE_OG = None
_RUNTIME_NEEDS_SHUTDOWN = False


def _to_numpy(value) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    return np.asarray(value)


def _orientation_distance_rad(left, right) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("B1K object orientation must be a quaternion")
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cosine = min(1.0, abs(float(np.dot(first, second))))
    return 2.0 * math.acos(cosine)


def _physical_state_errors(
        expected: dict, actual: dict, *, context: str = "reset") -> dict:
    """Measure reset pose errors; image repeatability is not a hard gate."""
    expected_poses = expected.get("object_poses") or {}
    actual_poses = actual.get("object_poses") or {}
    if set(expected_poses) != set(actual_poses):
        raise RuntimeError("B1K reset changed the physical object set")
    position_errors = []
    orientation_errors = []
    for identity in sorted(expected_poses):
        expected_position, expected_orientation = expected_poses[identity]
        actual_position, actual_orientation = actual_poses[identity]
        position_errors.append(float(np.max(np.abs(
            np.asarray(expected_position, dtype=np.float64) -
            np.asarray(actual_position, dtype=np.float64)))))
        orientation_errors.append(_orientation_distance_rad(
            expected_orientation, actual_orientation))
    expected_joints = expected.get("joint_positions_rad") or {}
    actual_joints = actual.get("joint_positions_rad") or {}
    if set(expected_joints) != set(actual_joints):
        raise RuntimeError("B1K reset changed the physical joint set")
    joint_errors = []
    for identity in sorted(expected_joints):
        expected_positions = np.asarray(
            expected_joints[identity], dtype=np.float64)
        actual_positions = np.asarray(
            actual_joints[identity], dtype=np.float64)
        if expected_positions.shape != actual_positions.shape:
            joint_context = (
                "joint reset" if context == "reset" else f"{context} joint")
            raise RuntimeError(
                f"B1K {joint_context} drift for {identity}: joint vector "
                f"shape {actual_positions.shape} != "
                f"{expected_positions.shape}")
        difference = expected_positions - actual_positions
        joint_errors.append(
            float(np.max(np.abs(difference))) if difference.size else 0.0)
    return {
        "max_object_position_error_m": max(position_errors, default=0.0),
        "max_object_orientation_error_rad": max(
            orientation_errors, default=0.0),
        "max_joint_position_error_rad": max(joint_errors, default=0.0),
    }


def _assert_valid_physical_state(state: dict, *, context: str) -> None:
    if not isinstance(state, Mapping):
        raise RuntimeError(f"B1K {context} has invalid physical state")
    poses = state.get("object_poses")
    joints = state.get("joint_positions_rad")
    if not isinstance(poses, Mapping) or not poses:
        raise RuntimeError(f"B1K {context} has invalid physical state")
    if not isinstance(joints, Mapping):
        raise RuntimeError(f"B1K {context} has invalid physical state")
    for identity, raw_pose in poses.items():
        if not isinstance(raw_pose, (tuple, list)) or len(raw_pose) != 2:
            raise RuntimeError(
                f"B1K {context} has invalid physical state for {identity}")
        position = np.asarray(raw_pose[0], dtype=np.float64)
        orientation = np.asarray(raw_pose[1], dtype=np.float64)
        if (position.shape != (3,) or orientation.shape != (4,) or
                not np.isfinite(position).all() or
                not np.isfinite(orientation).all() or
                np.linalg.norm(orientation) == 0.0):
            raise RuntimeError(
                f"B1K {context} has invalid physical state for {identity}")
    for identity, raw_positions in joints.items():
        positions = np.asarray(raw_positions, dtype=np.float64)
        if positions.ndim != 1 or not np.isfinite(positions).all():
            raise RuntimeError(
                f"B1K {context} has invalid physical state for {identity}")


def _assert_physical_state_close(
        expected: dict, actual: dict, *, context: str = "reset") -> None:
    _assert_valid_physical_state(expected, context=context)
    _assert_valid_physical_state(actual, context=context)
    pose_context = "object reset" if context == "reset" else context
    joint_context = (
        "joint reset" if context == "reset" else f"{context} joint")
    expected_poses = expected.get("object_poses") or {}
    actual_poses = actual.get("object_poses") or {}
    if set(expected_poses) != set(actual_poses):
        raise RuntimeError(f"B1K {context} changed the physical object set")
    for identity in sorted(expected_poses):
        expected_position, expected_orientation = expected_poses[identity]
        actual_position, actual_orientation = actual_poses[identity]
        position_error = float(np.max(np.abs(
            np.asarray(expected_position, dtype=np.float64) -
            np.asarray(actual_position, dtype=np.float64))))
        orientation_error = _orientation_distance_rad(
            expected_orientation, actual_orientation)
        if position_error > config.B1K_RESET_POSITION_TOL_M:
            metadata = (actual.get("object_reset_metadata") or {}).get(
                identity, {})
            raise RuntimeError(
                f"B1K {pose_context} position drift for {identity}: "
                f"{position_error:.9g} m > "
                f"{config.B1K_RESET_POSITION_TOL_M:.9g} m; "
                f"metadata={metadata}")
        if orientation_error > config.B1K_RESET_ORIENTATION_TOL_RAD:
            metadata = (actual.get("object_reset_metadata") or {}).get(
                identity, {})
            raise RuntimeError(
                f"B1K {pose_context} orientation drift for {identity}: "
                f"{orientation_error:.9g} rad > "
                f"{config.B1K_RESET_ORIENTATION_TOL_RAD:.9g} rad; "
                f"metadata={metadata}")
    expected_joints = expected.get("joint_positions_rad") or {}
    actual_joints = actual.get("joint_positions_rad") or {}
    if set(expected_joints) != set(actual_joints):
        raise RuntimeError(f"B1K {context} changed the physical joint set")
    for identity in sorted(expected_joints):
        expected_positions = np.asarray(
            expected_joints[identity], dtype=np.float64)
        actual_positions = np.asarray(
            actual_joints[identity], dtype=np.float64)
        if expected_positions.shape != actual_positions.shape:
            raise RuntimeError(
                f"B1K {joint_context} drift for {identity}: joint vector "
                f"shape {actual_positions.shape} != "
                f"{expected_positions.shape}")
        difference = expected_positions - actual_positions
        error = float(np.max(np.abs(difference))) if difference.size else 0.0
        if error > config.B1K_RESET_ORIENTATION_TOL_RAD:
            raise RuntimeError(
                f"B1K {joint_context} drift for {identity}: "
                f"{error:.9g} rad > "
                f"{config.B1K_RESET_ORIENTATION_TOL_RAD:.9g} rad")


def _canonical_physical_state_sha256(state: dict) -> str:
    """Digest one captured B1K physical state in stable identity order."""
    poses = state.get("object_poses") or {}
    joints = state.get("joint_positions_rad") or {}
    object_rows = []
    for identity in sorted(poses):
        position, orientation = poses[identity]
        position = np.asarray(position, dtype=np.float64)
        orientation = np.asarray(orientation, dtype=np.float64)
        if (position.shape != (3,) or orientation.shape != (4,) or
                not np.isfinite(position).all() or
                not np.isfinite(orientation).all()):
            raise ValueError("B1K canonical object state is invalid")
        object_rows.append({
            "prim_identity": str(identity),
            "position_m": position.tolist(),
            "orientation_xyzw": orientation.tolist(),
        })
    joint_rows = []
    for identity in sorted(joints):
        positions = np.asarray(joints[identity], dtype=np.float64)
        if positions.ndim != 1 or not np.isfinite(positions).all():
            raise ValueError("B1K canonical joint state is invalid")
        joint_rows.append({
            "joint_identity": str(identity),
            "positions_rad": positions.tolist(),
        })
    return record.canonical_atom_sha256({
        "schema": "b1k-canonical-physical-state.v1",
        "object_poses": object_rows,
        "joint_positions": joint_rows,
    })


def _canonical_replay(scene, runtime) -> dict:
    """Derive one fixed snapshot, then verify three equivalent reloads.

    The initial scene dump is loaded once to establish ``state_1``.  Its
    serialized output is then frozen: states 2--4 each reload that exact same
    byte sequence.  Re-dumping after every load would measure a chained
    numerical round trip that normal collection never performs.
    """
    state_0 = runtime.capture_physical_state(scene)
    _assert_valid_physical_state(
        state_0, context="canonical replay state_0")
    seed_serialized = runtime.dump_state(scene)
    states = {}
    triangle_digests = {}
    canonical_inputs = None
    runtime.load_state(scene, seed_serialized)
    states["state_1"] = runtime.capture_physical_state(scene)
    _assert_valid_physical_state(
        states["state_1"], context="canonical replay state_1")
    state_1_inputs = runtime.authority_inputs(scene)
    precanonical_triangle_digest = \
        b1k_semantic.derive_scene_authority_atom(
            floor_components=state_1_inputs[0],
            collision_components=state_1_inputs[1],
            instances=state_1_inputs[2])["sha256"]
    del state_1_inputs
    canonical_serialized = runtime.dump_state(scene)
    for index in range(2, 5):
        runtime.load_state(scene, canonical_serialized)
        state_name = f"state_{index}"
        states[state_name] = runtime.capture_physical_state(scene)
        inputs = runtime.authority_inputs(scene)
        triangle_digests[state_name] = \
            b1k_semantic.derive_scene_authority_atom(
                floor_components=inputs[0],
                collision_components=inputs[1],
                instances=inputs[2])["sha256"]
        if index == 2:
            canonical_inputs = inputs
        del inputs
    for left, right in ((2, 3), (2, 4), (3, 4)):
        _assert_physical_state_close(
            states[f"state_{left}"], states[f"state_{right}"],
            context=f"canonical replay state_{left} -> state_{right}")
    if len(set(triangle_digests.values())) != 1:
        formatted = ", ".join(
            f"{key}={value}"
            for key, value in triangle_digests.items())
        raise RuntimeError(
            f"B1K canonical replay triangle authority drift: {formatted}")
    canonical_state = states["state_2"]
    canonical_state_sha256 = _canonical_physical_state_sha256(canonical_state)
    return {
        "state_0": state_0,
        "canonical_state": canonical_state,
        "canonical_serialized": canonical_serialized,
        "canonical_inputs": canonical_inputs,
        "report": {
            "protocol": b1k_semantic.B1K_CANONICAL_REPLAY_PROTOCOL,
            "canonical_state": "state_2",
            "canonical_state_sha256": canonical_state_sha256,
            "state_1_to_state_2_errors": _physical_state_errors(
                states["state_1"], states["state_2"]),
            "state_2_to_state_3_errors": _physical_state_errors(
                states["state_2"], states["state_3"]),
            "state_2_to_state_4_errors": _physical_state_errors(
                states["state_2"], states["state_4"]),
            "state_3_to_state_4_errors": _physical_state_errors(
                states["state_3"], states["state_4"]),
            "precanonical_triangle_authority_sha256":
                precanonical_triangle_digest,
            "triangle_authority_sha256_by_state": triangle_digests,
        },
    }


def bootstrap_scene_authority(
        scene_id: str, scene_json_path, *, runtime=None, fovs=None) -> dict:
    """Derive one loaded-scene authority before a trusted manifest exists.

    This is the sole bootstrap boundary: it uses the same runtime extraction
    and pure authority builder as :class:`B1KSimSession`, then builds and
    queries all frozen C-spaces.  The normal session subsequently requires the
    resulting digest and rederives it before collection.
    """
    selected_runtime = runtime if runtime is not None else load_b1k_runtime()
    profiles = [tuple(map(float, value)) for value in (
        fovs or config.BENCH_FOVS_DEG)]
    scene = selected_runtime.open_scene(
        scene_id=str(scene_id), scene_json_path=str(scene_json_path))
    sensors = []
    try:
        hfov, vfov = profiles[0]
        width, height = config.render_resolution(hfov, vfov)
        requests = [{
            "width": width,
            "height": height,
            "hfov_deg": hfov,
            "vfov_deg": vfov,
            "modalities": B1K_SENSOR_MODALITIES,
        }]
        sensors = list(selected_runtime.create_vision_sensors(scene, requests))
        if len(sensors) != 1:
            raise RuntimeError("B1K runtime returned the wrong sensor count")
        replay = _canonical_replay(scene, selected_runtime)
        floor_components, collision_components, instances = \
            replay["canonical_inputs"]
        replay_report = replay["report"]
        geometry, _semantic, authority = b1k_semantic.build_b1k_authorities(
            floor_components=floor_components,
            collision_components=collision_components,
            instances=instances,
            canonical_replay_protocol=replay_report["protocol"],
            canonical_state_sha256=
                replay_report["canonical_state_sha256"])
        reset_errors = _physical_state_errors(
            replay["state_0"], replay["canonical_state"],
            context="canonical replay state_0 -> state_2")
        rng = np.random.default_rng(0)
        cspace_queries = {}
        for radius in geometry.radii_m:
            position = geometry.sample_position(rng, radius_m=radius)
            query = geometry.bind(
                position, 0.0, radius_m=radius).query_pose((0.0, 0.0, 0.0))
            if not query.navigable:
                raise RuntimeError(
                    f"B1K radius {radius} C-space sample is not queryable")
            cspace_queries[str(float(radius))] = "navigable"
        return {
            "schema": "b1k-scene-authority-bootstrap.v2",
            "scene_id": str(scene_id),
            "scene_authority": authority,
            "geometry": {
                "floor_component_count": len(floor_components),
                "collision_component_count": len(collision_components),
                "runtime_instance_count": len(instances),
                "collision_triangle_count": sum(
                    len(value.triangles) for value in collision_components),
                "cspace_query_by_radius_m": cspace_queries,
            },
            "reset_pose_errors": reset_errors,
            "canonical_replay": replay_report,
        }
    finally:
        selected_runtime.close(scene, sensors)


class _B1KPathfinder:
    """The deterministic sampling subset used by pose collection."""

    def __init__(self, session):
        self._session = session
        self._rng = np.random.default_rng(0)

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def get_random_navigable_point(self) -> np.ndarray:
        return self._session._geometry.sample_position(
            self._rng, radius_m=self._session._nav_radius)

    def distance_to_closest_obstacle(self, position) -> float:
        return self._session.dist_to_obstacle(position)


class B1KSimSession:
    """One base InteractiveTraversableScene with external FOV sensors."""

    POSE_SAMPLING = {
        "min_floor": 0.05,
        "min_valid_depth": None,
        "obstacle_margin_m": None,
        "max_tries": 400,
    }

    def __init__(self, scene: SceneSpec, *, heights=None, fovs=None,
                 runtime=None):
        if not isinstance(scene, SceneSpec) or scene.source_dataset != "b1k":
            raise TypeError("B1KSimSession requires a B1K SceneSpec")
        self.scene_id = scene.scene_id
        self.scene_glb = scene.scene_path
        self.source_dataset = "b1k"
        self.official_split = "train"
        self._heights = [float(value) for value in (
            heights or [config.CAMERA_HEIGHT_M])]
        self._fovs = [tuple(map(float, value)) for value in (
            fovs or [(config.HFOV_DEG, config.VFOV_DEG)])]
        if not self._fovs or len(set(self._fovs)) != len(self._fovs):
            raise ValueError("B1K session FOV profiles must be unique")
        self._hfov, self._vfov = self._fovs[0]
        self._runtime = runtime if runtime is not None else load_b1k_runtime()
        self._scene = None
        self._sensors = []
        try:
            self._scene = self._runtime.open_scene(
                scene_id=self.scene_id, scene_json_path=self.scene_glb)
            hfov, vfov = self._fovs[0]
            width, height = config.render_resolution(hfov, vfov)
            requests = [{
                "width": width, "height": height,
                "hfov_deg": hfov, "vfov_deg": vfov,
                "modalities": B1K_SENSOR_MODALITIES,
            }]
            self._sensors = list(self._runtime.create_vision_sensors(
                self._scene, requests))
            if len(self._sensors) != 1:
                raise RuntimeError("B1K runtime returned the wrong sensor count")
            self._sensor = self._sensors[0]
            expected_atom = scene.b1k_scene_authority or {}
            replay_binding = b1k_semantic.canonical_replay_binding(
                expected_atom)
            versioned_replay = replay_binding is not None
            if versioned_replay:
                replay = _canonical_replay(self._scene, self._runtime)
                floor_components, collision_components, instances = \
                    replay["canonical_inputs"]
                self._initial_physical_state = replay["canonical_state"]
                self._serialized_state = replay["canonical_serialized"]
                replay_kwargs = {
                    "canonical_replay_protocol":
                        replay["report"]["protocol"],
                    "canonical_state_sha256":
                        replay["report"]["canonical_state_sha256"],
                }
            else:
                floor_components, collision_components, instances = \
                    self._runtime.authority_inputs(self._scene)
                replay_kwargs = {}
            self._floor_components = tuple(floor_components)
            self._collision_components = tuple(collision_components)
            self._geometry, self._semantic, self.scene_authority_atom = \
                b1k_semantic.build_b1k_authorities(
                    floor_components=floor_components,
                    collision_components=collision_components,
                    instances=instances, **replay_kwargs)
            visual_identities = \
                self._runtime.visual_prim_to_instance_identity(self._scene)
            self._visual_prim_to_instance_id = {
                str(visual_prim): self._semantic.instance_id_for_prim(
                    str(instance_prim))
                for visual_prim, instance_prim in visual_identities.items()
            }
            expected = expected_atom.get("sha256")
            if self.scene_authority_atom["sha256"] != expected:
                raise ValueError(
                    "runtime B1K scene authority differs from source manifest")
            self.scene_authority_sha256 = self.scene_authority_atom["sha256"]
            if not versioned_replay:
                self._initial_physical_state = \
                    self._runtime.capture_physical_state(self._scene)
                self._serialized_state = self._runtime.dump_state(self._scene)
                self._runtime.load_state(self._scene, self._serialized_state)
                _assert_physical_state_close(
                    self._initial_physical_state,
                    self._runtime.capture_physical_state(self._scene))
            self._reset_physical_state()
            self._runtime.enable_native_instance_sensor(
                self._scene, self._sensor)
            self._runtime.assert_observation_sync(self._sensor)
            self._nav_radius = min(config.RADII_M)
            self._pathfinder = _B1KPathfinder(self)
            self._last_rgb = None
            self.reset_rgb_change_ratio = None
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def _reset_physical_state(self) -> None:
        self._runtime.load_state(self._scene, self._serialized_state)
        actual = self._runtime.capture_physical_state(self._scene)
        self.last_reset_pose_errors = _physical_state_errors(
            self._initial_physical_state, actual)
        _assert_physical_state_close(
            self._initial_physical_state,
            actual)

    @property
    def id_to_cat(self):
        return self._semantic.id_to_cat

    @property
    def id_to_predicate_cat(self):
        return self._semantic.id_to_predicate_cat

    @property
    def semantic_index(self):
        return self._semantic

    @property
    def pathfinder(self):
        return self._pathfinder

    def recompute_navmesh(
            self, radius: float,
            height: float = config.GROUND_ORACLE_HEIGHT_M) -> None:
        if abs(float(height) - config.GROUND_ORACLE_HEIGHT_M) > 1e-9:
            raise ValueError("ground-disc oracle height is fixed")
        value = float(radius)
        if value not in self._geometry.radii_m:
            raise ValueError("B1K radius is outside the frozen grid")
        self._nav_radius = value

    def nav(self, position, yaw: float):
        return self._geometry.bind(
            position, yaw, radius_m=self._nav_radius)

    def proposal_nav(self, position, yaw: float, *, radius_m: float):
        return self._geometry.bind(position, yaw, radius_m=float(radius_m))

    def dist_to_obstacle(self, position) -> float:
        query = self._geometry.bind(
            position, 0.0, radius_m=self._nav_radius).query_pose(
                (0.0, 0.0, 0.0))
        return max(0.0, float(query.clearance_m))

    def assign_instances(self, world_points: np.ndarray) -> np.ndarray:
        return self._semantic.assign(world_points)

    @staticmethod
    def _validated_modalities(
            observation, info, *, width: int, height: int,
            visual_prim_to_instance_id):
        if not isinstance(observation, dict) or any(
                name not in observation for name in
                B1K_OBSERVATION_MODALITIES):
            raise ValueError("B1K sensor omitted a required modality")
        rgb = _to_numpy(observation["rgb"])
        if rgb.shape == (height, width, 4):
            rgb = rgb[..., :3]
        if rgb.shape != (height, width, 3):
            raise ValueError("B1K RGB shape is invalid")
        if np.issubdtype(rgb.dtype, np.floating):
            if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                raise ValueError("B1K RGB content is invalid")
            rgb = np.rint(rgb * 255.0).astype(np.uint8)
        elif rgb.dtype != np.uint8:
            raise ValueError("B1K RGB dtype is invalid")
        depth = _to_numpy(observation["depth_linear"]).astype(
            np.float32, copy=True)
        if depth.shape != (height, width):
            raise ValueError("B1K linear depth content is invalid")
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid_depth):
            raise NoFiniteDepthObservation(
                "B1K linear depth contains no finite hit")
        depth[~valid_depth] = 0.0
        if not isinstance(info, Mapping):
            raise ValueError("B1K sensor omitted modality metadata")
        instance_info = info.get("seg_instance_id")
        if not isinstance(instance_info, Mapping):
            raise ValueError("B1K sensor omitted native instance metadata")
        semantic_ids, diagnostics = \
            b1k_observation.remap_native_instance_mask(
                _to_numpy(observation["seg_instance_id"]),
                id_to_prim_path=instance_info,
                visual_prim_to_instance_id=visual_prim_to_instance_id)
        if semantic_ids.shape != (height, width):
            raise ValueError("B1K native instance mask shape is invalid")
        return (
            np.ascontiguousarray(rgb), np.ascontiguousarray(depth),
            semantic_ids, diagnostics,
        )

    def render(self, position, yaw: float, cam_h: float = None,
               hfov: float = None, vfov: float = None) -> RenderObservation:
        camera_height = self._heights[0] if cam_h is None else float(cam_h)
        selected_hfov = self._hfov if hfov is None else float(hfov)
        selected_vfov = self._vfov if vfov is None else float(vfov)
        width, height = config.render_resolution(
            selected_hfov, selected_vfov)
        if (selected_hfov, selected_vfov) not in self._fovs:
            raise ValueError("requested B1K FOV was not opened by this session")
        sensor = self._sensor
        root = np.asarray(position, dtype=np.float64)
        camera = root + np.array([0.0, camera_height, 0.0])
        self._runtime.set_sensor_pose(
            sensor,
            position_og=b1k_geometry.pbench_to_og_xyz(camera),
            yaw_rad=b1k_geometry.og_yaw_to_pbench_yaw_rad(yaw))
        self._runtime.configure_vision_sensor(
            sensor, width=width, height=height,
            hfov_deg=selected_hfov, vfov_deg=selected_vfov)
        raw_observation, raw_info = self._runtime.render(sensor)
        rgb, depth, semantic_ids, semantic_diagnostics = \
            self._validated_modalities(
                raw_observation, raw_info, width=width, height=height,
                visual_prim_to_instance_id=
                    self._visual_prim_to_instance_id)
        self.last_native_instance_diagnostics = semantic_diagnostics
        if self._last_rgb is not None and self._last_rgb.shape == rgb.shape:
            self.reset_rgb_change_ratio = float(np.mean(
                np.any(self._last_rgb != rgb, axis=2)))
        self._last_rgb = rgb.copy()
        observation = RenderObservation(
            rgb=rgb,
            depth=depth,
            K=config.intrinsics(selected_hfov, selected_vfov),
            sensor=SensorProfile.from_values(
                camera_height, selected_hfov, selected_vfov),
            position=root,
            yaw_rad=float(yaw),
            semantic_instance_ids=semantic_ids,
        )
        validate_render_observation(
            observation, position=root, yaw=yaw,
            cam_h=camera_height, hfov=selected_hfov, vfov=selected_vfov)
        return observation

    @staticmethod
    def _validated_rgb(rgb, *, width: int, height: int) -> np.ndarray:
        values = _to_numpy(rgb)
        if values.shape == (height, width, 4):
            values = values[..., :3]
        if values.shape != (height, width, 3):
            raise ValueError("B1K terminal RGB shape is invalid")
        if np.issubdtype(values.dtype, np.floating):
            if (not np.isfinite(values).all() or values.min() < 0.0 or
                    values.max() > 1.0):
                raise ValueError("B1K terminal RGB content is invalid")
            values = np.rint(values * 255.0).astype(np.uint8)
        elif values.dtype != np.uint8:
            raise ValueError("B1K terminal RGB dtype is invalid")
        return np.ascontiguousarray(values)

    def render_terminal_rgb_batch(self, base, local_poses):
        """Render one C1 query bank with simultaneous temporary cameras.

        The renderer's RGB is source-authentic but not pose-pure under a
        sequential temporal history.  Temporary cameras are therefore created
        only for this call, share one accumulation reset and one frozen render
        count, and are destroyed before ordinary collection resumes.
        """
        if (base.scene_id != self.scene_id or
                str(base.scene_glb) != str(self.scene_glb)):
            raise ValueError("B1K terminal batch base frame is not session-bound")
        sensor = base.sensor
        width, height = config.render_resolution(
            sensor.hfov_deg, sensor.vfov_deg)
        requests = []
        world_poses = []
        for raw_pose in local_poses:
            x_m, z_m, heading_deg = (float(value) for value in raw_pose)
            root = perception.world_from_local(
                np.array([[x_m, 0.0, z_m]], dtype=np.float64),
                base.position, base.yaw_rad)[0]
            yaw = base.yaw_rad - math.radians(heading_deg)
            camera = root + np.array([
                0.0, sensor.nominal_camera_offset_m, 0.0])
            requests.append({
                "width": width,
                "height": height,
                "hfov_deg": sensor.hfov_deg,
                "vfov_deg": sensor.vfov_deg,
                "position_og": b1k_geometry.pbench_to_og_xyz(camera),
                "yaw_rad": b1k_geometry.og_yaw_to_pbench_yaw_rad(yaw),
            })
            world_poses.append((root, yaw))
        raw_images = list(self._runtime.render_rgb_batch(
            self._scene, requests))
        if len(raw_images) != len(requests):
            raise RuntimeError("B1K terminal RGB batch returned the wrong size")
        return [
            TerminalRGBObservation(
                scene_id=base.scene_id,
                scene_glb=base.scene_glb,
                position=root,
                yaw_rad=float(yaw),
                K=np.asarray(base.K, dtype=np.float64),
                rgb=self._validated_rgb(image, width=width, height=height),
                sensor=sensor,
            )
            for image, (root, yaw) in zip(raw_images, world_poses)
        ]

    def diagnostic_geometry_components(self):
        """Return loaded triangle inputs for non-gating Task-4 probes."""
        return {
            "floor": tuple(self._floor_components),
            "collision": tuple(self._collision_components),
            "visual": tuple(self._runtime.visual_components(self._scene)),
        }

    def sample_random_pose(
            self, rng, radii, yaws=None, max_tries=None,
            pose_valid=None, on_reject=None):
        radius = max(float(value) for value in radii)
        tries = int(self.POSE_SAMPLING["max_tries"] if max_tries is None
                    else max_tries)
        headings = (list(yaws) if yaws is not None else
                    [math.radians(value) for value in range(0, 360, 45)])
        height, hfov, vfov = config.calibration_profile()
        for _ in range(tries):
            position = self._geometry.sample_position(
                rng, radius_m=radius,
                minimum_clearance_m=config.BENCH_SAFE_CLEARANCE_M)
            yaw = float(rng.choice(headings))
            if pose_valid is not None and not pose_valid(position, yaw):
                if on_reject is not None:
                    on_reject("pose_rejected_by_backend")
                continue
            try:
                rendered = self.render(position, yaw, height, hfov, vfov)
            except NoFiniteDepthObservation:
                if on_reject is not None:
                    on_reject("low_valid_depth")
                continue
            points = perception.to_agent_ground(
                perception.unproject(rendered.depth, rendered.K)[0],
                camera_height=height)
            fit = floor_plane.fit_floor_plane(points)
            if not fit.fitted:
                if on_reject is not None:
                    for reason in fit.rejection_reasons:
                        on_reject(reason)
                continue
            ratio = pose_calibration.visible_floor_ratio(
                fit.estimate, points, rendered.depth.size)
            if ratio < self.POSE_SAMPLING["min_floor"]:
                if on_reject is not None:
                    on_reject("low_visible_floor_ratio")
                continue
            return pose_calibration.PoseCalibration(
                position=np.asarray(position, dtype=np.float64),
                yaw_rad=yaw,
                reference_profile=rendered.sensor,
                reference_observation=rendered,
                canonical_floor_fit=fit)
        return None

    def close(self) -> None:
        scene = getattr(self, "_scene", None)
        sensors = getattr(self, "_sensors", [])
        if scene is not None:
            try:
                if hasattr(self, "_initial_physical_state"):
                    _assert_physical_state_close(
                        self._initial_physical_state,
                        self._runtime.capture_physical_state(scene),
                        context="session close")
            finally:
                self._scene = None
                self._sensors = []
                self._runtime.close(scene, sensors)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _OmniGibsonRuntime:
    """Thin dynamic facade; constructed only by ``load_b1k_runtime``."""

    def __init__(self, og, Environment, object_taxonomy, mesh_converter):
        self._og = og
        self._Environment = Environment
        self._object_taxonomy = object_taxonomy
        self._mesh_converter = mesh_converter
        self._observation_profile = b1k_observation.load_frozen_profile()
        self._profile_atom = self._observation_profile["profile_atom"]
        self._settings_api = None
        self._temporary_sensor_serial = 0
        if callable(getattr(self._og, "launch", None)):
            self._og.launch()
            from omnigibson import lazy

            self._settings_api = lazy.carb.settings.get_settings()
            expected = b1k_observation.profile_setting_writes(
                self._profile_atom["settings"])
            for key, value in expected.items():
                self._settings_api.set(key, value)
            self._assert_observation_profile()

    def _assert_observation_profile(self) -> None:
        if self._settings_api is None:
            return
        expected = b1k_observation.profile_setting_writes(
            self._profile_atom["settings"])
        actual = {key: self._settings_api.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError("B1K renderer settings differ from v5 profile")

    def open_scene(self, *, scene_id, scene_json_path):
        return {
            "scene_id": str(scene_id),
            "scene_json_path": str(Path(scene_json_path).resolve()),
            "environment": None,
        }

    def create_vision_sensors(self, scene, requests):
        if len(requests) != 1:
            raise ValueError(
                "B1K runtime requires one shared external vision sensor")
        external_sensors = []
        for index, request in enumerate(requests):
            aperture = 20.955
            focal = aperture / (2.0 * math.tan(
                math.radians(float(request["hfov_deg"])) / 2.0))
            external_sensors.append({
                "sensor_type": "VisionSensor",
                "name": f"pbench_vision_sensor_{index}",
                "relative_prim_path": f"/pbench_vision_sensor_{index}",
                "modalities": list(request["modalities"]),
                "include_in_obs": False,
                "sensor_kwargs": {
                    "image_width": int(request["width"]),
                    "image_height": int(request["height"]),
                    "focal_length": focal,
                    "horizontal_aperture": aperture,
                },
            })
        config_value = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene["scene_id"],
                "scene_file": scene["scene_json_path"],
                "include_robots": False,
            },
            "robots": [],
            "task": {"type": "DummyTask"},
            "env": {"external_sensors": external_sensors},
        }
        environment = self._Environment(configs=config_value)
        scene["environment"] = environment
        self._assert_observation_profile()
        return [
            environment._external_sensors[f"pbench_vision_sensor_{index}"]
            for index in range(len(external_sensors))]

    @staticmethod
    def _environment(scene):
        environment = scene.get("environment")
        if environment is None:
            raise RuntimeError("B1K environment has not been opened")
        return environment

    def _world_triangles(self, mesh) -> np.ndarray:
        runtime_points = mesh.points
        points = _to_numpy(runtime_points).astype(np.float64, copy=False)
        faces = getattr(mesh, "faces", None)
        if faces is None:
            if self._mesh_converter is None:
                raise RuntimeError(
                    f"B1K collision shape has no triangle faces: "
                    f"{mesh.prim_path}")
            converted = self._mesh_converter(mesh.prim)
            points = np.asarray(converted.vertices, dtype=np.float64)
            faces = np.asarray(converted.faces)
        else:
            faces = _to_numpy(faces)
        faces = np.asarray(faces)
        if (points.ndim != 2 or points.shape[1:] != (3,) or
                faces.ndim != 2 or faces.shape[1:] != (3,) or
                not np.issubdtype(faces.dtype, np.integer)):
            raise RuntimeError(
                f"B1K collision mesh topology is invalid: {mesh.prim_path}")
        if not len(faces):
            return np.empty((0, 3, 3), dtype=np.float64)
        if faces.min() < 0 or faces.max() >= len(points):
            raise RuntimeError(
                f"B1K collision mesh indices are invalid: {mesh.prim_path}")
        tensor_factory = getattr(runtime_points, "new_tensor", None)
        transform_input = (
            tensor_factory(points) if callable(tensor_factory) else points)
        world_og = _to_numpy(
            mesh.transform_local_points_to_world(transform_input))
        if world_og.shape != points.shape or not np.isfinite(world_og).all():
            raise RuntimeError(
                f"B1K collision mesh transform is invalid: {mesh.prim_path}")
        world = b1k_geometry.og_to_pbench_xyz(world_og)
        return np.ascontiguousarray(world[faces.astype(np.int64)])

    def authority_inputs(self, scene):
        environment = self._environment(scene)
        floors = []
        collisions = []
        instances = []
        for obj in sorted(
                environment.scene.objects,
                key=lambda value: str(value.prim_path)):
            object_triangles = []
            for link_name, link in sorted(obj.links.items()):
                del link_name
                for mesh_name, mesh in sorted(link.collision_meshes.items()):
                    del mesh_name
                    triangles = self._world_triangles(mesh)
                    if not len(triangles):
                        continue
                    component = b1k_geometry.TriangleComponent(
                        str(mesh.prim_path), triangles)
                    collisions.append(component)
                    object_triangles.append(triangles)
                    if str(obj.category) == "floors":
                        edges_a = triangles[:, 1] - triangles[:, 0]
                        edges_b = triangles[:, 2] - triangles[:, 0]
                        normals = np.cross(edges_a, edges_b)
                        lengths = np.linalg.norm(normals, axis=1)
                        upward = np.divide(
                            normals[:, 1], lengths,
                            out=np.zeros_like(lengths), where=lengths > 0.0,
                        ) > config.B1K_FLOOR_UP_NORMAL_MIN
                        if np.any(upward):
                            floors.append(b1k_geometry.TriangleComponent(
                                str(mesh.prim_path), triangles[upward]))
            if not object_triangles:
                continue
            synset = self._object_taxonomy.get_synset_from_category(
                str(obj.category))
            if not isinstance(synset, str) or not synset:
                raise RuntimeError(
                    f"B1K category has no frozen synset: {obj.category!r}")
            instances.append(b1k_semantic.RuntimeInstanceSpec(
                prim_identity=str(obj.prim_path),
                raw_category=str(obj.category),
                model=str(obj.model),
                synset_label=synset,
                triangles=np.concatenate(object_triangles, axis=0),
            ))
        if not floors:
            raise RuntimeError(
                "loaded B1K scene contains no floor collision triangles")
        if not collisions or not instances:
            raise RuntimeError(
                "loaded B1K scene contains no collision instance triangles")
        return floors, collisions, instances

    def visual_components(self, scene):
        """Extract visual triangles only from the already loaded OG scene."""
        environment = self._environment(scene)
        components = []
        for obj in sorted(
                environment.scene.objects,
                key=lambda value: str(value.prim_path)):
            for _link_name, link in sorted(obj.links.items()):
                for _mesh_name, mesh in sorted(
                        getattr(link, "visual_meshes", {}).items()):
                    triangles = self._world_triangles(mesh)
                    if len(triangles):
                        components.append(b1k_geometry.TriangleComponent(
                            str(mesh.prim_path), triangles))
        return components

    def visual_prim_to_instance_identity(self, scene) -> dict[str, str]:
        """Bind renderer visual prim paths to frozen object identities."""
        environment = self._environment(scene)
        result = {}
        for obj in sorted(
                environment.scene.objects,
                key=lambda value: str(value.prim_path)):
            for _link_name, link in sorted(obj.links.items()):
                for _mesh_name, mesh in sorted(
                        getattr(link, "visual_meshes", {}).items()):
                    visual = str(mesh.prim_path)
                    identity = str(obj.prim_path)
                    previous = result.setdefault(visual, identity)
                    if previous != identity:
                        raise RuntimeError(
                            "B1K visual prim belongs to multiple instances")
        if not result:
            raise RuntimeError("loaded B1K scene has no visual instance prims")
        return result

    def enable_native_instance_sensor(self, scene, sensor) -> None:
        """Attach the raw visual-prim mask after the scene is initialized."""
        del scene
        from omnigibson import lazy

        annotator = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(
            "instance_id_segmentation")
        with self._og.sim.editing_usd():
            annotator.attach([sensor._render_product])
        sensor._pbench_raw_instance_annotator = annotator
        # The raw visual-prim annotator is render-only.  Do not call sim.step:
        # even one physics step moves unfixed objects and violates the frozen
        # 1e-6 m scene-state contract.

    @staticmethod
    def _native_instance_data(sensor) -> tuple[np.ndarray, dict]:
        annotator = getattr(
            sensor, "_pbench_raw_instance_annotator", None)
        if annotator is None:
            raise RuntimeError("B1K raw instance annotator is not attached")
        raw = annotator.get_data()
        if not isinstance(raw, dict) or "data" not in raw:
            raise RuntimeError("B1K raw instance annotator omitted data")
        values = _to_numpy(raw["data"])
        labels = dict((raw.get("info") or {}).get("idToLabels") or {})
        labels.setdefault("0", "background")
        return values, labels

    def _read_observation(self, sensor) -> tuple[dict, dict]:
        observation, info = sensor.get_obs()
        values, labels = self._native_instance_data(sensor)
        return {
            **observation,
            "seg_instance_id": values,
        }, {
            **dict(info),
            "seg_instance_id": labels,
        }

    def assert_observation_sync(self, sensor) -> None:
        """Canary the frozen N/N+1 exact geometry synchronization."""
        count = int(self._profile_atom["sync_render_count"])
        for _index in range(count):
            self._og.sim.render()
        first, _first_info = self._read_observation(sensor)
        self._og.sim.render()
        second, _second_info = self._read_observation(sensor)
        for name in b1k_observation.GEOMETRY_MODALITIES:
            if not np.array_equal(
                    _to_numpy(first[name]), _to_numpy(second[name])):
                raise RuntimeError(
                    f"B1K {name} did not synchronize at frozen N/N+1")

    def capture_physical_state(self, scene):
        environment = self._environment(scene)
        poses = {}
        joints = {}
        metadata = {}
        for obj in sorted(
                environment.scene.objects,
                key=lambda value: str(value.prim_path)):
            position, orientation = obj.get_position_orientation()
            poses[str(obj.prim_path)] = (
                _to_numpy(position).astype(np.float64),
                _to_numpy(orientation).astype(np.float64))
            metadata[str(obj.prim_path)] = {
                "category": str(getattr(obj, "category", "") or ""),
                "fixed_base": bool(getattr(obj, "fixed_base", False)),
                "kinematic_only": bool(
                    getattr(obj, "kinematic_only", False)),
            }
            joint_count = int(getattr(obj, "n_joints", 0))
            if joint_count:
                joints[str(obj.prim_path)] = _to_numpy(
                    obj.get_joint_positions()).astype(np.float64)
        return {
            "object_poses": poses,
            "joint_positions_rad": joints,
            "object_reset_metadata": metadata,
        }

    def dump_state(self, scene):
        del scene
        return self._og.sim.dump_state(serialized=True)

    def load_state(self, scene, serialized):
        del scene
        self._og.sim.load_state(serialized, serialized=True)

    def set_sensor_pose(self, sensor, *, position_og, yaw_rad):
        half_yaw = float(yaw_rad) / 2.0
        diagonal = math.sqrt(0.5)
        orientation = np.array([
            diagonal * math.cos(half_yaw),
            diagonal * math.sin(half_yaw),
            diagonal * math.sin(half_yaw),
            diagonal * math.cos(half_yaw),
        ], dtype=np.float64)
        sensor.set_position_orientation(
            position=position_og, orientation=orientation, frame="world")

    @staticmethod
    def configure_vision_sensor(
            sensor, *, width, height, hfov_deg, vfov_deg):
        """Switch one persistent render product between calibrated optics."""
        config.render_resolution(hfov_deg, vfov_deg)
        if (int(sensor.image_width) != int(width) or
                int(sensor.image_height) != int(height)):
            raise RuntimeError(
                "B1K optical switching cannot recreate a render product")
        aperture = 20.955
        focal = aperture / (2.0 * math.tan(
            math.radians(float(hfov_deg)) / 2.0))
        if np.float32(sensor.horizontal_aperture) != np.float32(aperture):
            sensor.horizontal_aperture = aperture
        if np.float32(sensor.focal_length) != np.float32(focal):
            sensor.focal_length = focal

    def render(self, sensor):
        for _index in range(int(self._profile_atom["sync_render_count"])):
            self._og.sim.render()
        return self._read_observation(sensor)

    def render_rgb_batch(self, scene, requests):
        """Render temporary, simultaneous RGB products and remove them."""
        if not requests:
            return []
        from omnigibson import lazy
        from omnigibson.sensors import VisionSensor, create_sensor

        environment = self._environment(scene)
        before = set(VisionSensor.SENSORS)
        physical_state_before = self.capture_physical_state(scene)
        sensors = []
        self._temporary_sensor_serial += 1
        serial = self._temporary_sensor_serial
        try:
            for index, request in enumerate(requests):
                aperture = 20.955
                focal = aperture / (2.0 * math.tan(math.radians(
                    float(request["hfov_deg"])) / 2.0))
                name = f"pbench_c1_{serial}_{index}"
                sensor = create_sensor(
                    sensor_type="VisionSensor",
                    relative_prim_path=f"/{name}",
                    name=name,
                    modalities=("rgb",),
                    sensor_kwargs={
                        "image_width": int(request["width"]),
                        "image_height": int(request["height"]),
                        "focal_length": focal,
                        "horizontal_aperture": aperture,
                    },
                )
                sensor.load(environment.scene)
                sensor.initialize()
                self.set_sensor_pose(
                    sensor,
                    position_og=np.asarray(
                        request["position_og"], dtype=np.float64),
                    yaw_rad=float(request["yaw_rad"]))
                sensors.append(sensor)
            # Sensor construction itself renders several frames at unequal
            # ages.  Reset once after every product and pose exists, then all
            # products accumulate the same frozen number of frames.
            lazy.omni.usd.get_context().reset_renderer_accumulation()
            for _index in range(int(self._profile_atom["sync_render_count"])):
                self._og.sim.render()
            result = []
            for sensor in sensors:
                observation, _info = sensor.get_obs()
                if "rgb" not in observation:
                    raise RuntimeError("temporary B1K sensor omitted RGB")
                result.append(_to_numpy(observation["rgb"]).copy())
            _assert_physical_state_close(
                physical_state_before,
                self.capture_physical_state(scene),
                context="temporary C1 render transaction")
            return result
        finally:
            for sensor in reversed(sensors):
                sensor.remove()
            if sensors:
                # Removing a render product invalidates Isaac's physics view
                # even though no physical state changed. Rebuild handles
                # without stepping physics so later poses and the session-close
                # audit can continue to read joint state.
                self._og.sim.update_handles()
                _assert_physical_state_close(
                    physical_state_before,
                    self.capture_physical_state(scene),
                    context="temporary C1 render cleanup")
            if set(VisionSensor.SENSORS) != before:
                raise RuntimeError(
                    "temporary B1K render products were not fully removed")

    def close(self, scene, sensors):
        environment = scene.get("environment")
        for sensor in tuple(sensors):
            annotator = getattr(
                sensor, "_pbench_raw_instance_annotator", None)
            if annotator is not None:
                with self._og.sim.editing_usd():
                    annotator.detach(sensor._render_product)
                del sensor._pbench_raw_instance_annotator
            sensor.remove()
        if environment is not None:
            environment._external_sensors.clear()
        self._og.clear()


def load_b1k_runtime():
    """Load OmniGibson only at the explicit runtime trust boundary."""
    try:
        import omnigibson as og
        from bddl.object_taxonomy import ObjectTaxonomy
        from omnigibson.envs import Environment
        from omnigibson.utils.usd_utils import \
            mesh_prim_shape_to_trimesh_mesh
    except ImportError as error:
        raise RuntimeError(
            "B1K backend requires the pinned OmniGibson runtime") from error
    global _ACTIVE_OG, _RUNTIME_NEEDS_SHUTDOWN
    _ACTIVE_OG = og
    _RUNTIME_NEEDS_SHUTDOWN = True
    return _OmniGibsonRuntime(
        og, Environment, ObjectTaxonomy(),
        mesh_converter=mesh_prim_shape_to_trimesh_mesh)


def shutdown_b1k_runtime() -> None:
    """Shutdown the lazily loaded process runtime at most once per run."""
    global _RUNTIME_NEEDS_SHUTDOWN
    if not _RUNTIME_NEEDS_SHUTDOWN:
        return
    _RUNTIME_NEEDS_SHUTDOWN = False
    _ACTIVE_OG.shutdown(due_to_signal=True)
