"""B1K observation profile and instance-mask contracts."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import b1k_observation, record


def test_frozen_profile_uses_official_omnigibson_renderer_defaults():
    value = b1k_observation.load_frozen_profile()
    atom = value["profile_atom"]

    assert value["schema"] == "b1k-observation-profile.v1"
    assert value["sync_render_count"] == 6
    assert value["c1_render_mode"] == \
        "persistent-counterfactual-bank-batch.v2"
    assert atom["renderer_authority"] == {
        "implementation": "omnigibson-official-defaults",
        "source_commit": "26f2c7ef7b9cf96bd0414f81e1e751e493762779",
        "source_tag": "v3.9.1",
    }
    assert atom["reset_accumulation"] is True
    assert "settings" not in atom
    assert "settings_readback" not in atom
    assert atom["modalities"] == [
        "rgb", "depth_linear", "seg_instance_id"]
    assert atom["sha256"] == record.B1K_OBSERVATION_PROFILE_SHA256


@pytest.mark.parametrize(
    ("contract_version", "render_mode"),
    [
        (
            record.B1K_V5_COLLECTION_CONTRACT_VERSION,
            record.B1K_V5_C1_RENDER_MODE,
        ),
        (
            record.B1K_V16_COLLECTION_CONTRACT_VERSION,
            record.B1K_C1_RENDER_MODE,
        ),
    ],
)
def test_frozen_profile_dispatches_legacy_contract_versions(
        contract_version, render_mode):
    value = b1k_observation.load_frozen_profile(
        contract_version=contract_version)
    atom = value["profile_atom"]

    assert value["schema"] == "b1k-render-profile-probe.v1"
    assert value["c1_render_mode"] == render_mode
    assert atom["sha256"] == \
        record.B1K_LEGACY_OBSERVATION_PROFILE_SHA256
    assert atom["reset_accumulation"] is False
    assert atom["antialiasing"] == "FXAA"
    assert atom["settings"]["/rtx/post/aa/op"] == 2


def test_frozen_profile_rejects_a_self_consistent_unpublished_atom(tmp_path):
    value = json.loads(b1k_observation.PROFILE_ASSET.read_text())
    value["sync_render_count"] += 1
    value["profile_atom"]["sync_render_count"] += 1
    value["profile_atom"]["sha256"] = \
        b1k_observation.profile_atom_sha256(value["profile_atom"])
    path = tmp_path / "substituted-profile.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="published digest"):
        b1k_observation.load_frozen_profile(path)


def test_frozen_profile_requires_the_published_c1_batch_mode(tmp_path):
    value = json.loads(b1k_observation.PROFILE_ASSET.read_text())
    value["c1_render_mode"] = "sequential-single-sensor.v1"
    path = tmp_path / "sequential-profile.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="published C1 render mode"):
        b1k_observation.load_frozen_profile(path)


def test_runtime_does_not_override_official_renderer_settings(monkeypatch):
    from pipeline.b1k_sim import _OmniGibsonRuntime

    writes = []

    class Settings:
        def __init__(self):
            self.values = {}

        def set(self, key, value):
            writes.append((key, value))
            self.values[key] = value

        def get(self, key):
            return self.values.get(key)

    settings = Settings()
    lazy = SimpleNamespace(carb=SimpleNamespace(settings=SimpleNamespace(
        get_settings=lambda: settings)))
    monkeypatch.setitem(
        sys.modules, "omnigibson", SimpleNamespace(lazy=lazy))
    gm = SimpleNamespace(RENDER_VIEWER_CAMERA=True)
    og = SimpleNamespace(gm=gm, launch=lambda: None)

    _OmniGibsonRuntime(og, None, None, mesh_converter=None)

    assert writes == []


def test_runtime_resets_accumulation_before_main_observation(monkeypatch):
    from pipeline.b1k_sim import _OmniGibsonRuntime

    events = []
    lazy = SimpleNamespace(omni=SimpleNamespace(usd=SimpleNamespace(
        get_context=lambda: SimpleNamespace(
            reset_renderer_accumulation=lambda: events.append("reset")))))
    monkeypatch.setitem(
        sys.modules, "omnigibson", SimpleNamespace(lazy=lazy))
    og = SimpleNamespace(
        sim=SimpleNamespace(render=lambda: events.append("render")))
    runtime = _OmniGibsonRuntime(og, None, None, mesh_converter=None)
    runtime._read_observation = lambda _sensor: ({"rgb": object()}, {})

    runtime.render(object())

    assert events == ["reset"] + ["render"] * 6


def test_legacy_runtime_restores_profile_settings_without_reset(monkeypatch):
    from pipeline.b1k_sim import _OmniGibsonRuntime

    writes = []

    class Settings:
        def __init__(self):
            self.values = {}

        def set(self, key, value):
            writes.append((key, value))
            self.values[key] = value

        def get(self, key):
            return self.values.get(key)

    events = []
    settings = Settings()
    lazy = SimpleNamespace(
        carb=SimpleNamespace(settings=SimpleNamespace(
            get_settings=lambda: settings)),
        omni=SimpleNamespace(usd=SimpleNamespace(
            get_context=lambda: SimpleNamespace(
                reset_renderer_accumulation=lambda: events.append("reset")))),
    )
    monkeypatch.setitem(
        sys.modules, "omnigibson", SimpleNamespace(lazy=lazy))
    og = SimpleNamespace(
        launch=lambda: events.append("launch"),
        sim=SimpleNamespace(render=lambda: events.append("render")))

    runtime = _OmniGibsonRuntime(
        og, None, None, mesh_converter=None,
        contract_version=record.B1K_V5_COLLECTION_CONTRACT_VERSION)
    runtime._read_observation = lambda _sensor: ({"rgb": object()}, {})
    runtime.render(object())

    assert events == ["launch"] + ["render"] * 6
    assert dict(writes)["/rtx/rendermode"] == "RaytracedLighting"
    assert dict(writes)["/rtx-defaults/rendermode"] == "RaytracedLighting"
    assert dict(writes)["/rtx/post/aa/op"] == 2


def test_legacy_runtime_restores_profile_settings_after_scene_open(monkeypatch):
    """OmniGibson scene construction may restore its renderer defaults."""
    from pipeline.b1k_sim import _OmniGibsonRuntime

    class Settings:
        def __init__(self):
            self.values = {}

        def set(self, key, value):
            self.values[key] = value

        def get(self, key):
            return self.values.get(key)

    settings = Settings()
    lazy = SimpleNamespace(carb=SimpleNamespace(settings=SimpleNamespace(
        get_settings=lambda: settings)))
    monkeypatch.setitem(
        sys.modules, "omnigibson", SimpleNamespace(lazy=lazy))

    expected = b1k_observation.profile_setting_writes(
        b1k_observation.load_frozen_profile(
            contract_version=record.B1K_V5_COLLECTION_CONTRACT_VERSION
        )["profile_atom"]["settings"])

    class Environment:
        def __init__(self, *, configs):
            for key in expected:
                settings.set(key, "runtime-default")
            self._external_sensors = {
                "pbench_vision_sensor_0": object(),
            }

    runtime = _OmniGibsonRuntime(
        SimpleNamespace(launch=lambda: None), Environment, None,
        mesh_converter=None,
        contract_version=record.B1K_V5_COLLECTION_CONTRACT_VERSION)
    scene = runtime.open_scene(
        scene_id="synthetic", scene_json_path=__file__)

    sensors = runtime.create_vision_sensors(scene, [{
        "width": 640,
        "height": 480,
        "hfov_deg": 79.0,
        "vfov_deg": 63.453048374758716,
        "modalities": ("rgb", "depth_linear", "seg_instance_id"),
    }])

    assert len(sensors) == 1
    assert {key: settings.get(key) for key in expected} == expected


def test_remap_native_instance_mask_uses_per_frame_prim_paths():
    raw = np.array([[7, 9, 11], [7, 9, 0]], dtype=np.int32)
    remapped, diagnostics = b1k_observation.remap_native_instance_mask(
        raw,
        id_to_prim_path={
            0: "BACKGROUND",
            7: "/World/scene_0/table/base_link/visuals",
            9: "/World/scene_0/chair/base_link/visuals",
            11: "unlabelled",
        },
        visual_prim_to_instance_id={
            "/World/scene_0/table/base_link/visuals": 4,
            "/World/scene_0/chair/base_link/visuals": 8,
        },
    )

    assert remapped.tolist() == [[4, 8, 0], [4, 8, 0]]
    assert diagnostics == {
        "observed_raw_instance_count": 4,
        "mapped_source_instance_count": 2,
        "unmapped_pixel_ratio": pytest.approx(1 / 3),
    }


def test_remap_native_instance_mask_refuses_missing_frame_metadata():
    with pytest.raises(ValueError, match="metadata omits observed IDs"):
        b1k_observation.remap_native_instance_mask(
            np.array([[1, 2]], dtype=np.int32),
            id_to_prim_path={1: "/World/table/visual"},
            visual_prim_to_instance_id={"/World/table/visual": 1},
        )
