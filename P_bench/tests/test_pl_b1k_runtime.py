"""Runnable BEHAVIOR-1K session and source catalog boundaries."""

from __future__ import annotations

import hashlib
import json
import copy
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pipeline import (
    actions, b1k_geometry, b1k_semantic, benchmark_tasks, config, consensus,
    consequence, frame as frame_module, record, scene_pool, source_manifest,
    validate,
)
from pipeline.collection_cli import build_parser
from pipeline.collection_runtime import _open_collection_sessions
from pipeline.geometry import Disc
from tests._synthetic import LEVEL_FLOOR, LEVEL_FLOOR_FIT, make_frame
from tests.test_pl_rollout import HalfPlaneNav


def _rectangle(x0, x1, z0, z1, *, y):
    return np.array([
        [[x0, y, z0], [x1, y, z0], [x1, y, z1]],
        [[x0, y, z0], [x1, y, z1], [x0, y, z1]],
    ], dtype=np.float64)


def _authority_inputs():
    floor_triangles = _rectangle(-4.0, 4.0, -4.0, 4.0, y=0.0)
    floor = [b1k_geometry.TriangleComponent(
        "/World/floor", floor_triangles[:, [0, 2, 1]])]
    obstacle = [b1k_geometry.TriangleComponent(
        "/World/table/collision",
        _rectangle(-0.2, 0.2, -2.1, -1.9, y=0.2))]
    instances = [b1k_semantic.RuntimeInstanceSpec(
        prim_identity="/World/table",
        raw_category="breakfast_table",
        model="abc123",
        synset_label="table.n.02",
        triangles=_rectangle(-0.8, 0.8, -2.1, -1.9, y=0.2),
    )]
    return floor, obstacle, instances


def _identity(root: Path, relative: str) -> dict:
    path = root / relative
    payload = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_manifest(
        tmp_path: Path, *, count: int = 3,
        canonical_replay: bool = True) -> Path:
    floor, obstacle, instances = _authority_inputs()
    replay_kwargs = {}
    if canonical_replay:
        from pipeline.b1k_sim import _canonical_physical_state_sha256

        replay_kwargs = {
            "canonical_replay_protocol":
                b1k_semantic.B1K_CANONICAL_REPLAY_PROTOCOL,
            "canonical_state_sha256": _canonical_physical_state_sha256({
                "object_poses": {
                    "/World/table": (
                        np.array([0.0, 2.0, 0.0]),
                        np.array([0.0, 0.0, 0.0, 1.0])),
                },
                "joint_positions_rad": {
                    "/World/table/joint": np.array([0.25]),
                },
            }),
        }
    _geometry, _semantics, authority = b1k_semantic.build_b1k_authorities(
        floor_components=floor,
        collision_components=obstacle,
        instances=instances,
        **replay_kwargs,
    )
    scenes = []
    for index in range(count):
        files = {
            "scene_json": f"scene-{index}.json",
            "layout": f"layout-{index}.json",
            "encrypted": f"asset-{index}.encrypted.usd",
            "initial": f"initial-{index}.json",
        }
        for label, relative in files.items():
            (tmp_path / relative).write_bytes(
                f"{label}:{index}".encode("ascii"))
        scenes.append({
            "scene_id": f"scene-{index}",
            "scene_json": _identity(tmp_path, files["scene_json"]),
            "layouts": [_identity(tmp_path, files["layout"])],
            "encrypted_assets": [
                _identity(tmp_path, files["encrypted"])],
            "initial_state": _identity(tmp_path, files["initial"]),
            "scene_authority": authority,
        })
    manifest = tmp_path / "b1k-source.json"
    manifest.write_text(json.dumps({
        "schema_version": "egoconseq.b1k-source-manifest.v1",
        "dataset": "b1k",
        "official_split": "train",
        "split_authority": "project_defined",
        "simulator": {
            "omnigibson": "3.9.1",
            "isaac_sim": "5.1.0",
        },
        "asset_versions": {"behavior-1k-assets": "2025.1.0"},
        "scenes": scenes,
    }, sort_keys=True))
    return manifest


def test_b1k_discovery_authenticates_actual_installed_catalog(tmp_path):
    """Catches hard-coded scene counts and unhashed B1K source inputs."""
    manifest = _write_manifest(tmp_path)

    scenes = scene_pool.discover_b1k_train_scenes(tmp_path, manifest)

    assert [scene.scene_id for scene in scenes] == [
        "scene-0", "scene-1", "scene-2"]
    source = scenes[0].provenance()
    assert source["source_dataset"] == "b1k"
    assert source["official_split"] == "train"
    assert source["split_authority"] == "project_defined"
    assert source["semantic_format"] == "omnigibson_instance"
    assert [value["role"] for value in source["source_assets"]] == [
        "scene", "scene_authority"]

    encrypted = tmp_path / "asset-0.encrypted.usd"
    encrypted.write_bytes(b"substituted")
    with pytest.raises(scene_pool.SceneCatalogError, match="digest changed"):
        scene_pool.discover_b1k_train_scenes(tmp_path, manifest)


def test_b1k_scene_job_authenticates_only_its_requested_assets(tmp_path):
    manifest = _write_manifest(tmp_path)
    (tmp_path / "asset-1.encrypted.usd").unlink()
    scenes = scene_pool.discover_b1k_train_scenes(
        tmp_path, manifest, requested=["scene-0"])
    assert [scene.scene_id for scene in scenes] == ["scene-0"]
    (tmp_path / "asset-0.encrypted.usd").write_bytes(b"substituted")
    with pytest.raises(scene_pool.SceneCatalogError, match="digest changed"):
        scene_pool.discover_b1k_train_scenes(
            tmp_path, manifest, requested=["scene-0"])


def test_b1k_discovery_refuses_an_incomplete_install(tmp_path):
    """Catches treating a one-scene partial install as a valid catalog."""
    manifest = _write_manifest(tmp_path, count=2)

    with pytest.raises(scene_pool.SceneCatalogError, match="at least 3"):
        scene_pool.discover_b1k_train_scenes(tmp_path, manifest)


def test_b1k_discovery_rejects_partial_canonical_replay_binding(tmp_path):
    """Catches admitting a hash-valid authority with only one replay key."""
    manifest = _write_manifest(tmp_path, canonical_replay=False)
    payload = json.loads(manifest.read_text())
    authority = payload["scenes"][0]["scene_authority"]
    authority["canonical_state_sha256"] = "a" * 64
    authority["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in authority.items() if key != "sha256"
    })
    manifest.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(
            scene_pool.SceneCatalogError, match="canonical replay fields"):
        scene_pool.discover_b1k_train_scenes(tmp_path, manifest)


class _FakeRuntime:
    def __init__(self):
        self.open_request = None
        self.sensor_requests = None
        self.optical_requests = []
        self.active_optics = None
        self.pose_request = None
        self._state = {
            "object_poses": {
                "/World/table": (
                    np.array([0.0, 2.0, 0.0]),
                    np.array([0.0, 0.0, 0.0, 1.0])),
            },
            "joint_positions_rad": {
                "/World/table/joint": np.array([0.25]),
            },
        }
        self.loads = 0
        self.native_instance_enable_count = 0
        self.observation_sync_count = 0
        self.rgb_batch_requests = []

    def open_scene(self, *, scene_id, scene_json_path):
        self.open_request = (scene_id, scene_json_path)
        return object()

    def create_vision_sensors(self, scene, requests):
        self.sensor_requests = list(requests)
        self.created_sensors = [object() for _request in self.sensor_requests]
        return self.created_sensors

    def authority_inputs(self, scene):
        return _authority_inputs()

    @staticmethod
    def visual_prim_to_instance_identity(scene):
        del scene
        return {"/World/table/visual": "/World/table"}

    def enable_native_instance_sensor(self, scene, sensor):
        del scene
        assert sensor is self.created_sensors[0]
        self.native_instance_enable_count += 1

    def assert_observation_sync(self, sensor):
        assert sensor is self.created_sensors[0]
        self.observation_sync_count += 1

    def capture_physical_state(self, scene):
        return {
            "object_poses": {
                key: (position.copy(), orientation.copy())
                for key, (position, orientation)
                in self._state["object_poses"].items()
            },
            "joint_positions_rad": {
                key: value.copy()
                for key, value in self._state["joint_positions_rad"].items()
            },
        }

    def dump_state(self, scene):
        return b"serialized-state"

    def load_state(self, scene, serialized):
        assert serialized == b"serialized-state"
        self.loads += 1

    def set_sensor_pose(self, sensor, *, position_og, yaw_rad):
        self.pose_request = (np.asarray(position_og), float(yaw_rad))

    def configure_vision_sensor(
            self, sensor, *, width, height, hfov_deg, vfov_deg):
        assert sensor is self.created_sensors[0]
        self.active_optics = {
            "width": int(width), "height": int(height),
            "hfov_deg": float(hfov_deg), "vfov_deg": float(vfov_deg),
        }
        self.optical_requests.append(self.active_optics.copy())

    def render(self, sensor):
        assert sensor is self.created_sensors[0]
        width = self.active_optics["width"]
        height = self.active_optics["height"]
        rgb = np.zeros((height, width, 4), dtype=np.uint8)
        rgb[..., 0] = 31
        depth = np.full((height, width), 2.0, np.float32)
        depth[0, 0] = np.inf
        depth[0, 1] = -1.0
        instance = np.ones((height, width), dtype=np.int32)
        return {
            "rgb": rgb,
            "depth_linear": depth,
            "seg_instance_id": instance,
        }, {
            "seg_instance_id": {1: "/World/table/visual"},
        }

    def render_rgb_batch(
            self, scene, requests, *, render_transaction=None):
        del scene
        self.last_render_transaction = render_transaction
        self.rgb_batch_requests.append(copy.deepcopy(list(requests)))
        result = []
        for index, request in enumerate(requests, 1):
            result.append(np.full(
                (int(request["height"]), int(request["width"]), 3),
                30 + index, dtype=np.uint8))
        return result

    def close(self, scene, sensors):
        pass

    @property
    def _sensors(self):
        return getattr(self, "created_sensors", [])


class _CanonicalReplayRuntime(_FakeRuntime):
    """Apply one requested object offset after each dump/load replay."""

    def __init__(self, replay_offsets_m, *, triangle_drift_at=None):
        super().__init__()
        self._replay_offsets_m = tuple(float(value)
                                       for value in replay_offsets_m)
        self._triangle_drift_at = triangle_drift_at

    def load_state(self, scene, serialized):
        super().load_state(scene, serialized)
        offset = self._replay_offsets_m[min(
            self.loads - 1, len(self._replay_offsets_m) - 1)]
        self._state["object_poses"]["/World/table"][0][0] = offset

    def authority_inputs(self, scene):
        floor, collisions, instances = _authority_inputs()
        if self._triangle_drift_at == self.loads:
            collisions = [b1k_geometry.TriangleComponent(
                collisions[0].identity,
                collisions[0].triangles + np.array([0.01, 0.0, 0.0]))]
        return floor, collisions, instances


class _FixedSnapshotReplayRuntime(_FakeRuntime):
    """Expose whether replay reloads one snapshot or chains new dumps."""

    def __init__(self):
        super().__init__()
        self.dump_count = 0
        self.loaded_serialized = []
        self.authority_input_count = 0

    def authority_inputs(self, scene):
        self.authority_input_count += 1
        return super().authority_inputs(scene)

    def dump_state(self, scene):
        del scene
        value = f"snapshot-{self.dump_count}".encode("ascii")
        self.dump_count += 1
        return value

    def load_state(self, scene, serialized):
        del scene
        self.loaded_serialized.append(serialized)
        offset_by_snapshot = {
            b"snapshot-0": 0.0,
            b"snapshot-1": 0.25e-6,
            b"snapshot-2": 0.50e-6,
        }
        self._state["object_poses"]["/World/table"][0][0] = \
            offset_by_snapshot[serialized]
        self.loads += 1


class _InvalidCanonicalReplayRuntime(_CanonicalReplayRuntime):
    def __init__(self, invalid_kind):
        super().__init__([0.0, 0.0, 0.0])
        self._invalid_kind = invalid_kind

    def load_state(self, scene, serialized):
        super().load_state(scene, serialized)
        if self.loads != 2:
            return
        if self._invalid_kind == "position_nan":
            self._state["object_poses"]["/World/table"][0][0] = np.nan
        elif self._invalid_kind == "orientation_zero":
            self._state["object_poses"]["/World/table"][1][:] = 0.0
        elif self._invalid_kind == "joint_nan":
            self._state["joint_positions_rad"][
                "/World/table/joint"][0] = np.nan


class _MalformedCanonicalReplayRuntime(_CanonicalReplayRuntime):
    def __init__(self, malformed_kind):
        super().__init__([0.0, 0.0, 0.0])
        self._malformed_kind = malformed_kind

    def capture_physical_state(self, scene):
        state = super().capture_physical_state(scene)
        if self.loads != 2:
            return state
        if self._malformed_kind == "missing_object_poses":
            state.pop("object_poses")
        elif self._malformed_kind == "list_object_poses":
            state["object_poses"] = []
        elif self._malformed_kind == "empty_object_poses":
            state["object_poses"] = {}
        elif self._malformed_kind == "missing_joint_positions":
            state.pop("joint_positions_rad")
        elif self._malformed_kind == "list_joint_positions":
            state["joint_positions_rad"] = []
        elif self._malformed_kind == "broadcast_joint_shape":
            state["joint_positions_rad"]["/World/table/joint"] = np.array(
                [0.25, 0.25])
        return state


class _FirstReplayJointShapeRuntime(_CanonicalReplayRuntime):
    def __init__(self):
        super().__init__([0.0, 0.0, 0.0])

    def load_state(self, scene, serialized):
        super().load_state(scene, serialized)
        if self.loads == 1:
            self._state["joint_positions_rad"]["/World/table/joint"] = \
                np.array([0.25, 0.25])


class _B1KSafeNav(HalfPlaneNav):
    authority = "b1k_geometry"


def test_b1k_session_rederives_versioned_canonical_replay_authority(tmp_path):
    """Catches bootstrap authority that normal collection cannot rederive."""
    from pipeline.b1k_sim import B1KSimSession, bootstrap_scene_authority

    manifest = _write_manifest(tmp_path)
    bootstrap = bootstrap_scene_authority(
        "scene-0", tmp_path / "scene-0.json",
        runtime=_CanonicalReplayRuntime([4e-6, 4e-6, 4e-6]))
    payload = json.loads(manifest.read_text())
    payload["scenes"][0]["scene_authority"] = bootstrap["scene_authority"]
    manifest.write_text(json.dumps(payload, sort_keys=True))
    scene = scene_pool.discover_b1k_train_scenes(tmp_path, manifest)[0]

    with B1KSimSession(
            scene,
            runtime=_CanonicalReplayRuntime([4e-6, 4e-6, 4e-6])) as session:
        assert session.scene_authority_atom == bootstrap["scene_authority"]
        assert session.last_reset_pose_errors[
            "max_object_position_error_m"] == 0.0


def test_b1k_session_requires_a_versioned_canonical_snapshot(tmp_path):
    """Normal collection must use the already-audited source authority."""
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path, canonical_replay=False))[0]

    with pytest.raises(ValueError, match="canonical replay binding"):
        B1KSimSession(scene, runtime=_FakeRuntime())


def test_b1k_session_reuses_the_frozen_canonical_snapshot(tmp_path):
    """Collection loads the canonical snapshot without replaying its audit."""
    from pipeline.b1k_sim import B1KSimSession, bootstrap_scene_authority

    manifest = _write_manifest(tmp_path)
    bootstrap = bootstrap_scene_authority(
        "scene-0", tmp_path / "scene-0.json",
        runtime=_FixedSnapshotReplayRuntime())
    payload = json.loads(manifest.read_text())
    payload["scenes"][0]["scene_authority"] = bootstrap["scene_authority"]
    manifest.write_text(json.dumps(payload, sort_keys=True))
    scene = scene_pool.discover_b1k_train_scenes(tmp_path, manifest)[0]
    runtime = _FixedSnapshotReplayRuntime()

    with B1KSimSession(scene, runtime=runtime) as session:
        session.render([0.0, 0.0, 0.0], 0.0)
        session.render([0.0, 0.0, 0.0], 0.0)

    assert runtime.loaded_serialized == [b"snapshot-0", b"snapshot-1"]
    assert runtime.dump_count == 2
    assert runtime.authority_input_count == 1


def test_b1k_session_rejects_partial_canonical_replay_binding(tmp_path):
    """Catches session protocol detection keying on only one replay field."""
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    authority = dict(scene.b1k_scene_authority)
    authority.pop("canonical_replay_protocol")
    authority["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in authority.items() if key != "sha256"
    })

    with pytest.raises(ValueError, match="canonical replay fields"):
        B1KSimSession(
            replace(scene, b1k_scene_authority=authority),
            runtime=_FakeRuntime())


def test_b1k_session_uses_lazy_fake_runtime_and_derived_authorities(tmp_path):
    """Catches bypassing reset, modality, pose, or derived-authority logic."""
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    runtime = _FakeRuntime()

    with B1KSimSession(scene, runtime=runtime) as session:
        observation = session.render(
            [1.0, 0.0, -2.0], 0.5,
            config.CAMERA_HEIGHT_M, config.HFOV_DEG, config.VFOV_DEG)

        assert runtime.sensor_requests == [{
            "width": config.resolution()[0],
            "height": config.resolution()[1],
            "hfov_deg": config.HFOV_DEG,
            "vfov_deg": config.VFOV_DEG,
            "modalities": ("rgb", "depth_linear"),
        }]
        assert runtime.optical_requests == [{
            "width": config.resolution()[0],
            "height": config.resolution()[1],
            "hfov_deg": config.HFOV_DEG,
            "vfov_deg": config.VFOV_DEG,
        }]
        assert runtime.loads == 2
        assert runtime.native_instance_enable_count == 1
        assert runtime.observation_sync_count == 1
        assert runtime.pose_request[0] == pytest.approx(
            b1k_geometry.pbench_to_og_xyz(
                [1.0, config.CAMERA_HEIGHT_M, -2.0]))
        assert runtime.pose_request[1] == 0.5
        assert observation.rgb.shape == (
            config.resolution()[1], config.resolution()[0], 3)
        assert observation.depth[0, :2].tolist() == [0.0, 0.0]
        assert observation.depth[1, 0] == 2.0
        assert np.all(observation.semantic_instance_ids == 1)
        assert session.last_reset_pose_errors == {
            "max_object_position_error_m": 0.0,
            "max_object_orientation_error_rad": 0.0,
            "max_joint_position_error_rad": 0.0,
        }
        assert session.id_to_cat == {1: "table.n.02"}
        assert session.scene_authority_sha256 == next(
            asset.sha256 for asset in scene.source_assets
            if asset.role == "scene_authority")
        # Sparse contact probes still use the exact triangle authority.
        assert session.assign_instances(np.array([
            [0.0, 0.2, -2.0]])).tolist() == [1]


def test_b1k_semantic_assign_uses_indexed_exact_surface_query(monkeypatch):
    floor, obstacle, instances = _authority_inputs()
    _geometry, semantics, _authority = b1k_semantic.build_b1k_authorities(
        floor_components=floor,
        collision_components=obstacle,
        instances=instances,
    )
    monkeypatch.setattr(
        b1k_semantic, "_point_triangle_distances_m",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("brute point-triangle matrix must not run")))

    assigned = semantics.assign(np.array([
        [0.0, 0.2, -2.0],
        [3.0, 0.2, -2.0],
    ]))

    assert assigned.tolist() == [1, 0]
    assert semantics.last_assign_diagnostics == {
        "query_point_count": 2,
        "aabb_candidate_count": 1,
        "indexed_surface_query_count": 1,
    }


def test_b1k_build_frame_uses_native_semantics_without_dense_assignment(
        tmp_path):
    """Catches reintroducing all-depth-point triangle assignment."""
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    runtime = _FakeRuntime()

    with B1KSimSession(scene, runtime=runtime) as session:
        session.assign_instances = lambda _points: (_ for _ in ()).throw(
            AssertionError("dense semantic assignment must not run"))
        built = frame_module.build_frame(
            session, [0.0, 0.0, 0.0], 0.0,
            frame_id="b1k-indexed-semantics", scene_id=scene.scene_id,
            scene_glb=scene.scene_path, floor_plane=LEVEL_FLOOR,
        )

    assert set(np.unique(built.pts_sem)).issubset({0, 1})
    assert np.count_nonzero(built.pts_sem == 1) > 0


def test_b1k_render_does_not_reload_the_scene_per_observation(tmp_path):
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    runtime = _FakeRuntime()

    with B1KSimSession(scene, runtime=runtime) as session:
        loads_after_open = runtime.loads
        session.render([0.0, 0.0, 0.0], 0.0)
        session.render([0.5, 0.0, 0.0], 0.25)
        assert runtime.loads == loads_after_open


def test_b1k_render_uses_current_frame_native_instance_prim_paths(tmp_path):
    from pipeline.b1k_sim import B1KSimSession

    class NativeMaskRuntime(_FakeRuntime):
        @staticmethod
        def visual_prim_to_instance_identity(_scene):
            return {
                "/World/table/base_link/visuals": "/World/table",
            }

        def render(self, sensor):
            observation, _old_info = super().render(sensor)
            height, width = observation["depth_linear"].shape
            raw_ids = np.full((height, width), 7, dtype=np.int32)
            raw_ids[0, 0] = 0
            observation.update({
                "seg_instance_id": raw_ids,
            })
            return observation, {
                "seg_instance_id": {
                    0: "BACKGROUND",
                    7: "/World/table/base_link/visuals",
                },
            }

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    runtime = NativeMaskRuntime()

    with B1KSimSession(scene, runtime=runtime) as session:
        rendered = session.render([0.0, 0.0, 0.0], 0.0)

    assert runtime.sensor_requests[0]["modalities"] == (
        "rgb", "depth_linear")
    assert rendered.semantic_instance_ids[0, 0] == 0
    assert np.all(rendered.semantic_instance_ids[1:, :] == 1)


def test_b1k_c1_renders_real_terminal_views_in_one_simultaneous_batch(tmp_path):
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    runtime = _FakeRuntime()

    with B1KSimSession(scene, runtime=runtime) as session:
        base = frame_module.build_frame(
            session, [1.0, 0.0, -2.0], 0.25,
            frame_id="b1k-c1-base", scene_id=scene.scene_id,
            scene_glb=scene.scene_path, floor_plane=LEVEL_FLOOR)
        rendered = session.render_terminal_rgb_batch(base, [
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 15.0),
            (0.5, -0.5, -15.0),
            (1.0, -0.5, 30.0),
        ])

    assert len(runtime.rgb_batch_requests) == 1
    assert len(runtime.rgb_batch_requests[0]) == 4
    assert len(rendered) == 4
    assert [int(value.rgb[0, 0, 0]) for value in rendered] == [31, 32, 33, 34]
    assert all(value.sensor == base.sensor for value in rendered)
    assert np.array_equal(rendered[0].position, base.position)
    assert rendered[0].yaw_rad == base.yaw_rad


def test_b1k_session_dispatches_the_stored_legacy_c1_transaction(tmp_path):
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    runtime = _FakeRuntime()

    with B1KSimSession(
            scene, runtime=runtime,
            contract_version=record.B1K_V5_COLLECTION_CONTRACT_VERSION
            ) as session:
        base = frame_module.build_frame(
            session, [1.0, 0.0, -2.0], 0.25,
            frame_id="b1k-c1-v5", scene_id=scene.scene_id,
            scene_glb=scene.scene_path, floor_plane=LEVEL_FLOOR)
        session.render_terminal_rgb_batch(base, [(0.0, 0.0, 0.0)])

    assert runtime.last_render_transaction == record.B1K_V5_C1_RENDER_MODE


def test_real_runtime_dispatches_legacy_and_persistent_c1_transactions(
        monkeypatch):
    from pipeline.b1k_sim import _OmniGibsonRuntime

    runtime = _OmniGibsonRuntime(
        SimpleNamespace(sim=SimpleNamespace()), None, None,
        mesh_converter=None,
        contract_version=record.B1K_V5_COLLECTION_CONTRACT_VERSION)
    calls = []
    monkeypatch.setattr(
        runtime, "_render_rgb_batch_temporary",
        lambda _scene, _requests: calls.append("temporary") or ["v5"])
    monkeypatch.setattr(
        runtime, "_render_rgb_batch_persistent",
        lambda _scene, _requests: calls.append("persistent") or ["v6"])

    assert runtime.render_rgb_batch(
        {}, [{}], render_transaction=record.B1K_V5_C1_RENDER_MODE) == ["v5"]
    assert runtime.render_rgb_batch(
        {}, [{}], render_transaction=record.B1K_C1_RENDER_MODE) == ["v6"]
    assert calls == ["temporary", "persistent"]


def test_real_runtime_reuses_c1_render_products_across_batches(monkeypatch):
    """C1 batches keep one stable render graph and refresh handles once."""
    from pipeline.b1k_sim import _OmniGibsonRuntime

    created = []
    resets = []
    handle_refreshes = []
    physics_valid = {"value": True}

    class Sensor:
        def __init__(self, name, request):
            self.name = name
            self.prim_path = f"/{name}"
            self.image_width = int(request["image_width"])
            self.image_height = int(request["image_height"])
            self.horizontal_aperture = request["horizontal_aperture"]
            self.focal_length = request["focal_length"]
            self.modalities = {"rgb"}
            self.removed = False

        def load(self, _scene):
            pass

        def initialize(self):
            physics_valid["value"] = False

        def set_position_orientation(self, **_values):
            pass

        def get_obs(self):
            assert "rgb" in self.modalities
            index = created.index(self) + 1
            return {"rgb": np.full(
                (self.image_height, self.image_width, 3), index,
                dtype=np.uint8)}, {}

        def remove(self):
            self.removed = True

    def create_sensor(*, name, sensor_kwargs, **_kwargs):
        sensor = Sensor(name, sensor_kwargs)
        created.append(sensor)
        return sensor

    def update_handles():
        physics_valid["value"] = True
        handle_refreshes.append(True)

    sim = SimpleNamespace(render=lambda: None, update_handles=update_handles)
    og = SimpleNamespace(sim=sim, clear=lambda: None)
    lazy = SimpleNamespace(omni=SimpleNamespace(usd=SimpleNamespace(
        get_context=lambda: SimpleNamespace(
            reset_renderer_accumulation=lambda: resets.append(True)))))
    monkeypatch.setitem(
        sys.modules, "omnigibson", SimpleNamespace(lazy=lazy))
    monkeypatch.setitem(sys.modules, "omnigibson.sensors", SimpleNamespace(
        create_sensor=create_sensor))

    runtime = _OmniGibsonRuntime(og, None, None, mesh_converter=None)

    def capture_physical_state(_scene):
        assert physics_valid["value"]
        return {
            "object_poses": {"/World/floor": (
                np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))},
            "joint_positions_rad": {},
        }

    runtime.capture_physical_state = capture_physical_state
    scene = {"environment": SimpleNamespace(scene=object(),
                                              _external_sensors={})}
    request = {
        "width": 8, "height": 6,
        "hfov_deg": config.HFOV_DEG, "vfov_deg": config.VFOV_DEG,
        "position_og": np.zeros(3), "yaw_rad": 0.0,
    }

    first = runtime.render_rgb_batch(scene, [request, request])
    second = runtime.render_rgb_batch(scene, [request])

    assert len(created) == 2
    assert len(first) == 2 and len(second) == 1
    assert resets == [True, True]
    runtime.capture_physical_state(scene)
    assert handle_refreshes == [True]
    assert all(sensor.modalities == {"rgb"} for sensor in created)
    assert not any(sensor.removed for sensor in created)

    runtime.close(scene, [])
    assert all(sensor.removed for sensor in created)


def test_b1k_session_opens_both_default_fovs_in_one_scene(tmp_path):
    """Catches constructing a second OmniGibson singleton environment."""
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    runtime = _FakeRuntime()

    with B1KSimSession(
            scene, fovs=config.BENCH_FOVS_DEG, runtime=runtime) as session:
        observations = [
            session.render([0.0, 0.0, 0.0], 0.0, hfov=hfov, vfov=vfov)
            for hfov, vfov in config.BENCH_FOVS_DEG
        ]

    assert runtime.open_request == (scene.scene_id, scene.scene_path)
    assert len(runtime.sensor_requests) == 1
    assert (
        runtime.sensor_requests[0]["hfov_deg"],
        runtime.sensor_requests[0]["vfov_deg"],
    ) == config.BENCH_FOVS_DEG[0]
    assert [
        (request["hfov_deg"], request["vfov_deg"])
        for request in runtime.optical_requests
    ] == list(config.BENCH_FOVS_DEG)
    assert [observation.sensor.hfov_deg for observation in observations] == [
        value[0] for value in config.BENCH_FOVS_DEG]


class _RuntimeMesh:
    def __init__(self, prim_path, triangles):
        self.prim_path = prim_path
        values = np.asarray(triangles, dtype=np.float64)
        self.points = values.reshape(-1, 3)
        self.faces = np.arange(self.points.shape[0]).reshape(-1, 3)

    def transform_local_points_to_world(self, points):
        return np.asarray(points)


class _RuntimeObject:
    def __init__(self, prim_path, category, model, triangles):
        self.prim_path = prim_path
        self.category = category
        self.model = model
        link = SimpleNamespace(collision_meshes={
            "collision": _RuntimeMesh(
                f"{prim_path}/collision", triangles),
        })
        self.links = {"base_link": link}


def test_real_runtime_config_and_loaded_triangle_extraction():
    """Catches fake-only hooks and invalid external-sensor constructor args."""
    from pipeline.b1k_sim import _OmniGibsonRuntime

    floor_slab = np.concatenate([
        _rectangle(-2.0, 2.0, -2.0, 2.0, y=0.0)[:, [0, 2, 1]],
        _rectangle(-2.0, 2.0, -2.0, 2.0, y=-0.3),
    ])
    objects = [
        _RuntimeObject(
            "/World/floor", "floors", "floor-model",
            b1k_geometry.pbench_to_og_xyz(floor_slab)),
        _RuntimeObject(
            "/World/table", "breakfast_table", "table-model",
            _rectangle(-0.2, 0.2, -2.1, -1.9, y=0.2)),
    ]
    captured = {}

    class Sensor:
        def __init__(self):
            self.image_width = 640
            self.image_height = 480
            self._horizontal_aperture = 20.955
            self._focal_length = 20.955 / (2.0 * np.tan(
                np.radians(config.BENCH_FOVS_DEG[0][0]) / 2.0))
            self.aperture_writes = 0
            self.focal_writes = 0

        @property
        def horizontal_aperture(self):
            return self._horizontal_aperture

        @horizontal_aperture.setter
        def horizontal_aperture(self, value):
            self.aperture_writes += 1
            self._horizontal_aperture = value

        @property
        def focal_length(self):
            return self._focal_length

        @focal_length.setter
        def focal_length(self, value):
            self.focal_writes += 1
            self._focal_length = value

        def set_position_orientation(self, **values):
            captured["sensor_pose"] = values

    sensor = Sensor()

    class Environment:
        def __init__(self, *, configs):
            captured["config"] = configs
            self.scene = SimpleNamespace(objects=objects)
            self._external_sensors = {
                "pbench_vision_sensor_0": sensor,
            }

    class Taxonomy:
        @staticmethod
        def get_synset_from_category(category):
            return {
                "floors": "floor.n.01",
                "breakfast_table": "breakfast_table.n.01",
            }[category]

    runtime = _OmniGibsonRuntime(
        None, Environment, Taxonomy(), mesh_converter=None)
    scene = runtime.open_scene(
        scene_id="Beechwood_0_int", scene_json_path=__file__)
    hfov, vfov = config.BENCH_FOVS_DEG[0]
    sensors = runtime.create_vision_sensors(scene, [{
        "width": 640, "height": 480, "hfov_deg": hfov,
        "vfov_deg": vfov,
        "modalities": ("rgb", "depth_linear")}])
    assert "external_sensors" not in captured["config"]
    assert captured["config"]["scene"]["include_robots"] is False
    assert len(captured["config"]["env"]["external_sensors"]) == 1
    sensor_config = captured["config"]["env"]["external_sensors"][0]
    assert "image_width" not in sensor_config
    assert sensor_config["sensor_kwargs"]["image_width"] == 640
    assert sensor_config["sensor_kwargs"]["image_height"] == 480
    assert set(sensor_config["modalities"]) == {"rgb", "depth_linear"}
    for index, (expected_hfov, expected_vfov) in enumerate(
            config.BENCH_FOVS_DEG):
        runtime.configure_vision_sensor(
            sensors[0], width=640, height=480,
            hfov_deg=expected_hfov, vfov_deg=expected_vfov)
        assert sensor.horizontal_aperture == pytest.approx(20.955)
        assert sensor.focal_length == pytest.approx(
            20.955 / (2.0 * np.tan(np.radians(expected_hfov) / 2.0)))
        assert sensor.aperture_writes == 0
        assert sensor.focal_writes == index
    runtime.set_sensor_pose(
        sensors[0], position_og=np.array([1.0, 2.0, 3.0]), yaw_rad=0.0)
    assert captured["sensor_pose"]["orientation"] == pytest.approx([
        np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])

    floors, collisions, instances = runtime.authority_inputs(scene)
    assert [value.identity for value in floors] == [
        "/World/floor/collision"]
    assert len(floors[0].triangles) == 2
    assert [value.identity for value in collisions] == [
        "/World/floor/collision", "/World/table/collision"]
    assert len(collisions[0].triangles) == 4
    assert [value.prim_identity for value in instances] == [
        "/World/floor", "/World/table"]
    assert [value.synset_label for value in instances] == [
        "floor.n.01", "breakfast_table.n.01"]


def test_collection_runtime_configures_rt2_and_viewer_before_launch(
        monkeypatch):
    """Catches launching Kit before RT2 registration or viewer suppression."""
    from pipeline.b1k_sim import _launch_omnigibson_for_collection

    class SimulationApp:
        DEFAULT_LAUNCHER_CONFIG = {
            "extra_args": ["--keep-existing"],
        }

    monkeypatch.setitem(
        sys.modules, "isaacsim",
        SimpleNamespace(SimulationApp=SimulationApp))
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    gm = SimpleNamespace(RENDER_VIEWER_CAMERA=True)
    launch_values = []
    og = SimpleNamespace(
        gm=gm,
        launch=lambda: launch_values.append({
            "viewer": gm.RENDER_VIEWER_CAMERA,
            "extra_args": list(
                SimulationApp.DEFAULT_LAUNCHER_CONFIG["extra_args"]),
            "eula": os.environ.get("OMNI_KIT_ACCEPT_EULA"),
        }),
    )

    _launch_omnigibson_for_collection(og)

    assert launch_values == [{
        "viewer": False,
        "eula": "YES",
        "extra_args": [
            "--keep-existing",
            "--/persistent/rtx/modes/rt/enabled=true",
            "--/persistent/rtx/modes/rt2/enabled=true",
        ],
    }]


def test_real_runtime_visual_probe_extracts_only_loaded_meshes():
    """Catches probing encrypted USDs instead of the loaded OG scene."""
    from pipeline.b1k_sim import _OmniGibsonRuntime

    visual = _RuntimeMesh(
        "/World/table/visual",
        _rectangle(-0.3, 0.3, -2.2, -1.8, y=0.2))
    obj = _RuntimeObject(
        "/World/table", "breakfast_table", "table-model",
        _rectangle(-0.2, 0.2, -2.1, -1.9, y=0.2))
    next(iter(obj.links.values())).visual_meshes = {"visual": visual}
    environment = SimpleNamespace(scene=SimpleNamespace(objects=[obj]))
    runtime = _OmniGibsonRuntime(
        None, None, None, mesh_converter=None)
    scene = {"environment": environment}

    components = runtime.visual_components(scene)

    assert [value.identity for value in components] == [
        "/World/table/visual"]


def test_real_runtime_detaches_external_sensor_before_omnigibson_clear():
    """Catches SyntheticData graphs surviving into Simulator.stop()."""
    from pipeline.b1k_sim import _OmniGibsonRuntime

    events = []

    class Sensor:
        def remove(self):
            events.append("sensor.remove")

    og = SimpleNamespace(clear=lambda: events.append("og.clear"))
    runtime = _OmniGibsonRuntime(
        og, None, None, mesh_converter=None)

    runtime.close({}, [Sensor()])

    assert events == ["sensor.remove", "og.clear"]


def test_session_close_releases_runtime_when_final_state_check_fails(
        monkeypatch):
    """Catches losing OG cleanup after a final-state audit failure."""
    from pipeline import b1k_sim

    events = []

    class Runtime:
        def capture_physical_state(self, _scene):
            return {"actual": True}

        def close(self, scene, sensors):
            events.append((scene, tuple(sensors)))

    session = object.__new__(b1k_sim.B1KSimSession)
    session._scene = object()
    session._sensors = [object()]
    session._runtime = Runtime()
    session._initial_physical_state = {"expected": True}
    monkeypatch.setattr(
        b1k_sim, "_assert_physical_state_close",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("final drift")))

    with pytest.raises(RuntimeError, match="final drift"):
        session.close()

    assert len(events) == 1
    assert session._scene is None
    assert session._sensors == []


def test_fake_runtime_reaches_record_and_source_bound_b1_b2(tmp_path):
    """Catches a runnable adapter that still drops both B benchmark heads."""
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    session = B1KSimSession(scene, runtime=_FakeRuntime())
    try:
        position = np.zeros(3, dtype=np.float64)
        rendered = session.render(
            position, 0.0, config.CAMERA_HEIGHT_M,
            config.HFOV_DEG, config.VFOV_DEG)
        assert np.count_nonzero(rendered.depth == 2.0) > 0
        frame = make_frame()
        frame.frame_id = "F-b1k-e2e"
        frame.scene_id = scene.scene_id
        frame.scene_glb = scene.scene_path
        frame.semantic_index = session.semantic_index
        frame.id_to_cat = session.id_to_cat
        frame.objects[0]["instance_id"] = 1
        frame.objects[0]["category"] = "table.n.02"
        frame.objects[0].update({
            "is_structural": False,
            "mask_area_px": 4000,
            "depth_backed_px": 4000,
            "bbox_xyxy_px": [200, 150, 439, 329],
            "centroid_px": [320.0, 240.0],
            "dist_nearest_m": 2.0,
            "surface_anchor": {
                "protocol": "b1k-rendered-instance-depth.v1",
                "pixel_xy_px": [320, 240],
                "initial_robot_xyz_m": [0.0, 0.0, 2.0],
            },
        })
        program = [actions.Forward(0.5)]
        outcome = consequence.judge(
            frame, Disc(radius_m=0.2), program,
            nav=_B1KSafeNav(10.0),
            require_contact_instance_witness=True)
        rows = []
        for perturbation in consensus.R2R_A_STABILITY_PERTURBATIONS:
            rows.append({
                "perturbation_id": perturbation["id"],
                "transform": {
                    key: perturbation[key]
                    for key in ("x_m", "z_m", "yaw_deg")
                },
                "physical": copy.deepcopy(outcome["physical"]),
                "depth_physical": copy.deepcopy(outcome["depth_physical"]),
                "corridor_coverage": 1.0,
            })
        outcome["shared_oracle_stability"] = \
            consensus.build_a_stability_certificate(program, rows)
        source = scene.provenance()
        contract = record.collection_contract(source, "main")
        image_path = tmp_path / "initial.png"
        Image.fromarray(frame.rgb).save(image_path)
        rec = record.build_record(
            frame, [outcome], image_path=image_path.name,
            floor_calibration=LEVEL_FLOOR_FIT,
            source_provenance=source,
            collection_contract=contract)
        from pipeline import abc1_record, surface_points

        rec["outcomes"][0].update(action_group_id="safe", outcome_id="b020-safe")
        compact = abc1_record.from_legacy(
            rec, dataset="b1k", source_records_sha256="a" * 64, byte_offset=0)
        compact["surface_point_target"] = {
            "instance_id": 1, "category": "table",
            "points": [{"point_id": "p1", "pixel_xy_px": [320, 240],
                        "world_xyz_m": [0.0, 0.0, -2.0]}],
        }
        surface_points.refresh_outputs(compact)
        assert {"B1", "B2"} <= compact["cases"][0]["task_outputs"].keys()
        context = source_manifest.b1k_v16_registered_validation_context(
            [source], collection_mode="main",
            expected_schema_version=record.SCHEMA_VERSION,
            expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION,
            authority_sha256="f" * 64,
            scene_authority_resolver=lambda _scene_id:
                session.scene_authority_sha256)
        errors = validate.validate_record_source_bound(
            rec, context=context, asset_root=tmp_path)
        assert errors == [], errors
    finally:
        session.close()


def test_b1k_cli_and_session_dispatch_do_not_change_r2r(monkeypatch, tmp_path):
    """Catches parsing B1K flags but still opening Habitat's session."""
    parser = build_parser()
    args = parser.parse_args([
        "--backend", "b1k",
        "--b1k-data-root", str(tmp_path),
        "--b1k-source-manifest", str(tmp_path / "manifest.json"),
        "--auto-scenes",
        "--out", str(tmp_path / "out"),
    ])
    assert args.backend == "b1k"

    scene = SimpleNamespace(scene_id="test")
    opened = object()
    fovs = list(config.BENCH_FOVS_DEG)
    observed = []

    def open_once(selected, heights, fovs):
        observed.append((selected, heights, fovs))
        return opened

    monkeypatch.setattr("pipeline.b1k_sim.B1KSimSession", open_once)
    assert _open_collection_sessions(
        args, scene, [config.CAMERA_HEIGHT_M], fovs) == [opened, opened]
    assert observed == [(scene, [config.CAMERA_HEIGHT_M], fovs)]


def test_b1k_pose_sampling_uses_the_shared_floor_visibility_gate():
    """Catches weakening B1K pose discovery below the R2R floor gate."""
    from pipeline.b1k_sim import B1KSimSession

    assert B1KSimSession.POSE_SAMPLING["min_floor"] == 0.05


def test_b1k_partial_session_construction_clears_scene_once(tmp_path):
    """Catches leaking a singleton scene when external sensor setup fails."""
    from pipeline.b1k_sim import B1KSimSession

    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]

    class BrokenRuntime(_FakeRuntime):
        def __init__(self):
            super().__init__()
            self.closes = 0

        def create_vision_sensors(self, scene, requests):
            raise RuntimeError("sensor construction failed")

        def close(self, scene, sensors):
            self.closes += 1

    runtime = BrokenRuntime()
    with pytest.raises(RuntimeError, match="sensor construction failed"):
        B1KSimSession(scene, runtime=runtime)
    assert runtime.closes == 1


@pytest.mark.parametrize("result, expected_calls", [(0, ["shutdown"]), (7, [])])
def test_b1k_collection_preserves_cli_status_before_process_shutdown(
        monkeypatch, result, expected_calls):
    from pipeline import b1k_sim, collection_runtime

    calls = []
    monkeypatch.setattr(
        collection_runtime, "_run_collection", lambda _args, _parser: result)
    monkeypatch.setattr(
        b1k_sim, "shutdown_b1k_runtime", lambda: calls.append("shutdown"))

    assert collection_runtime.run_collection(
        SimpleNamespace(backend="b1k"), object()) == result
    assert calls == expected_calls


def test_b1k_failed_cli_uses_result_as_hard_process_status():
    """Catches Isaac teardown replacing a failed funnel with exit 0 or SIGSEGV."""
    from pipeline.collection_support import preserve_b1k_cli_status

    exits = []
    preserve_b1k_cli_status(
        SimpleNamespace(backend="b1k"), 7,
        hard_exit=lambda status: exits.append(status))
    assert exits == [7]
    assert preserve_b1k_cli_status(
        SimpleNamespace(backend="r2r"), 7,
        hard_exit=lambda status: exits.append(status)) == 7


@pytest.mark.parametrize("shutdown_error", [
    RuntimeError("shutdown failed"), SystemExit(0)])
def test_b1k_collection_preserves_nonzero_result_when_shutdown_fails(
        monkeypatch, shutdown_error):
    from pipeline import b1k_sim, collection_runtime

    monkeypatch.setattr(
        collection_runtime, "_run_collection", lambda _args, _parser: 7)
    monkeypatch.setattr(
        b1k_sim, "shutdown_b1k_runtime",
        lambda: (_ for _ in ()).throw(shutdown_error))

    assert collection_runtime.run_collection(
        SimpleNamespace(backend="b1k"), object()) == 7


def test_b1k_success_does_not_hide_nonzero_shutdown_exit(monkeypatch):
    from pipeline import b1k_sim, collection_runtime

    monkeypatch.setattr(
        collection_runtime, "_run_collection", lambda _args, _parser: 0)
    monkeypatch.setattr(
        b1k_sim, "shutdown_b1k_runtime",
        lambda: (_ for _ in ()).throw(SystemExit(9)))

    with pytest.raises(SystemExit) as captured:
        collection_runtime.run_collection(
            SimpleNamespace(backend="b1k"), object())
    assert captured.value.code == 9


@pytest.mark.parametrize("shutdown_error", [
    RuntimeError("shutdown failed"), SystemExit(0)])
def test_b1k_collection_preserves_failure_when_shutdown_also_fails(
        monkeypatch, shutdown_error):
    from pipeline import b1k_sim, collection_runtime

    class CollectionFailure(RuntimeError):
        pass

    def fail_collection(_args, _parser):
        raise CollectionFailure("collection failed")

    def fail_shutdown():
        raise shutdown_error

    monkeypatch.setattr(collection_runtime, "_run_collection", fail_collection)
    monkeypatch.setattr(b1k_sim, "shutdown_b1k_runtime", fail_shutdown)

    with pytest.raises(CollectionFailure, match="collection failed"):
        collection_runtime.run_collection(
            SimpleNamespace(backend="b1k"), object())


def test_scene_cleanup_preserves_collection_failure_and_closes_once():
    from pipeline.collection_support import finish_scene_cleanup

    class CollectionFailure(RuntimeError):
        pass

    events = []

    def close():
        events.append("close")
        raise RuntimeError("clear failed")

    def finish():
        events.append("finish")
        raise RuntimeError("funnel finish failed")

    with pytest.raises(CollectionFailure, match="collection failed"):
        try:
            raise CollectionFailure("collection failed")
        finally:
            finish_scene_cleanup(close, finish)
    assert events == ["close", "finish"]


def test_strict_collector_oracle_dispatch_includes_b1k_main_only():
    from pipeline.collection_runtime import (
        contact_instance_witness_required, strict_shared_oracle_required,
        v16_r2r_shared_oracle_required,
    )

    assert strict_shared_oracle_required("main", "r2r") is True
    assert strict_shared_oracle_required("main", "b1k") is True
    assert strict_shared_oracle_required("candidate", "b1k") is False
    assert strict_shared_oracle_required("main", "gs") is True
    assert contact_instance_witness_required("main", "r2r") is True
    assert contact_instance_witness_required("main", "b1k") is True
    assert contact_instance_witness_required(
        "main", "gs", semantic_certified=True) is True
    assert contact_instance_witness_required(
        "main", "gs", semantic_certified=False) is False
    # The named legacy predicate remains byte-for-byte R2R-only.
    assert v16_r2r_shared_oracle_required("main", "b1k") is False


def test_b1k_validation_context_binds_trusted_run_action_policy(tmp_path):
    """Catches trusting a record's self-declared selection policy."""
    from pipeline import action_proposal
    from pipeline.collection_support import b1k_run_validation_context

    scenes = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))
    args = SimpleNamespace(
        action_mode="balanced", collection_mode="main",
        setting_sampling_policy=None)

    context = b1k_run_validation_context(
        args, [scene.provenance() for scene in scenes], scenes,
        record_schema=record.SCHEMA_VERSION,
        authority_sha256="f" * 64)

    assert context.expected_action_sampling_policy == \
        action_proposal.DEPTH_CONDITIONED_POLICY
