"""Frozen B1K renderer-profile and observation transaction contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline import b1k_observation, record


def test_frozen_profile_is_self_authenticated_and_uses_raw_instance_mask():
    value = b1k_observation.load_frozen_profile()

    assert value["selected_profile_id"] == "aa-op-2-reset-0"
    assert value["sync_render_count"] == 6
    assert value["rgb_pose_pure"] is False
    assert value["c1_render_mode"] == \
        "temporary-counterfactual-bank-batch.v1"
    assert value["profile_atom"]["modalities"] == [
        "rgb", "depth_linear", "seg_instance_id"]
    assert value["profile_atom"]["sha256"] == \
        record.B1K_OBSERVATION_PROFILE_SHA256


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


def test_profile_writes_persistent_active_and_defaults_keys():
    writes = b1k_observation.profile_setting_writes({
        "/rtx/rendermode": "RaytracedLighting",
        "/rtx/post/aa/op": 2,
        "/rtx-transient/post/aa/limitedOps": False,
    })

    assert writes == {
        "/rtx/rendermode": "RaytracedLighting",
        "/rtx-defaults/rendermode": "RaytracedLighting",
        "/rtx/post/aa/op": 2,
        "/rtx-defaults/post/aa/op": 2,
        "/rtx-transient/post/aa/limitedOps": False,
    }


def test_sync_render_count_adds_one_safety_frame_and_enforces_cost():
    result = b1k_observation.derive_sync_render_count(
        first_stable_renders=[3, 5, 4, 5],
        worst_render_seconds=0.10,
    )

    assert result == 6

    with pytest.raises(ValueError, match="transaction exceeds"):
        b1k_observation.derive_sync_render_count(
            first_stable_renders=[8], worst_render_seconds=0.12)


def test_profile_atom_binds_semantics_resolution_and_readbacks():
    atom = b1k_observation.build_profile_atom(
        settings={
            "/rtx/rendermode": "RaytracedLighting",
            "/rtx/post/aa/op": 2,
        },
        settings_readback={
            "/rtx/rendermode": "RaytracedLighting",
            "/rtx-defaults/rendermode": "RaytracedLighting",
            "/rtx/post/aa/op": 2,
            "/rtx-defaults/post/aa/op": 2,
        },
        antialiasing_name="FXAA",
        renderer_input_resolution=(640, 480),
        renderer_output_resolution=(640, 480),
        sync_render_count=6,
        modalities=("rgb", "depth_linear", "seg_instance_id"),
        runtime_identity={"omnigibson": "3.9.1", "isaac_sim": "5.1.0"},
        reset_accumulation=True,
    )

    assert atom["protocol"] == "b1k-observation.v5"
    assert atom["antialiasing"] == "FXAA"
    assert atom["renderer_input_resolution"] == [640, 480]
    assert atom["renderer_output_resolution"] == [640, 480]
    assert atom["sync_render_count"] == 6
    assert atom["sha256"] == b1k_observation.profile_atom_sha256(atom)


def test_profile_atom_refuses_neural_upscaling():
    with pytest.raises(ValueError, match="input and output resolution"):
        b1k_observation.build_profile_atom(
            settings={"/rtx/rendermode": "RaytracedLighting"},
            settings_readback={
                "/rtx/rendermode": "RaytracedLighting",
                "/rtx-defaults/rendermode": "RaytracedLighting",
            },
            antialiasing_name="DLSS",
            renderer_input_resolution=(320, 240),
            renderer_output_resolution=(640, 480),
            sync_render_count=6,
            modalities=("rgb", "depth_linear"),
            runtime_identity={"omnigibson": "3.9.1"},
            reset_accumulation=False,
        )


def _passing_probe_row(*, rgb_pose_pure: bool = False) -> dict:
    return {
        "profile_id": "fxaa",
        "antialiasing": "FXAA",
        "settings": {
            "/rtx/rendermode": "RaytracedLighting",
            "/rtx/post/aa/op": 2,
            "/rtx/post/scaling/staticRatio": 1.0,
        },
        "settings_readback": b1k_observation.profile_setting_writes({
            "/rtx/rendermode": "RaytracedLighting",
            "/rtx/post/aa/op": 2,
            "/rtx/post/scaling/staticRatio": 1.0,
        }),
        "renderer_input_resolution": [640, 480],
        "renderer_output_resolution": [640, 480],
        "first_stable_render": {
            "depth_linear": 3,
            "seg_instance_id": 4,
        },
        "current_pose_differs": {
            "depth_linear": True,
            "seg_instance_id": True,
        },
        "history_independent": {
            "rgb": rgb_pose_pure,
            "depth_linear": True,
            "seg_instance_id": True,
        },
        "worst_render_seconds": 0.1,
        "reset_accumulation": True,
    }


def test_select_profile_freezes_geometry_modalities_and_routes_c1_batch():
    report = b1k_observation.select_profile_from_probe(
        [_passing_probe_row()],
        runtime_identity={"omnigibson": "3.9.1", "isaac_sim": "5.1.0"},
    )

    assert report["sync_render_count"] == 5
    assert report["rgb_pose_pure"] is False
    assert report["c1_render_mode"] == \
        "temporary-counterfactual-bank-batch.v1"
    assert report["profile_atom"]["protocol"] == "b1k-observation.v5"


def test_select_profile_prefers_measured_quality_rank_before_speed():
    fxaa = _passing_probe_row(rgb_pose_pure=True)
    fxaa["quality_rank"] = 0
    off = _passing_probe_row(rgb_pose_pure=True)
    off.update({
        "profile_id": "aa-off",
        "antialiasing": "OFF",
        "quality_rank": 1,
        "worst_render_seconds": 0.01,
    })

    selected = b1k_observation.select_profile_from_probe(
        [off, fxaa], runtime_identity={"omnigibson": "3.9.1"})

    assert selected["selected_profile_id"] == "fxaa"


def test_select_profile_refuses_stale_segmentation_or_unknown_resolution():
    stale = _passing_probe_row()
    stale["history_independent"]["seg_instance_id"] = False
    with pytest.raises(ValueError, match="no B1K renderer profile"):
        b1k_observation.select_profile_from_probe(
            [stale], runtime_identity={"omnigibson": "3.9.1"})

    upscaled = _passing_probe_row()
    upscaled["renderer_input_resolution"] = None
    with pytest.raises(ValueError, match="no B1K renderer profile"):
        b1k_observation.select_profile_from_probe(
            [upscaled], runtime_identity={"omnigibson": "3.9.1"})


def test_first_stable_render_requires_the_remaining_suffix_to_be_exact():
    frames = [
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([2], dtype=np.int32),
    ]

    assert b1k_observation.first_stable_render(frames) == 4
    assert b1k_observation.first_stable_render(frames[:3]) == 2
    assert b1k_observation.first_stable_render(
        [np.array([0]), np.array([1])]) is None


def test_array_sha256_binds_dtype_shape_and_bytes():
    first = np.array([[1, 2]], dtype=np.int16)
    same_bytes_new_shape = first.reshape(2, 1)
    same_values_new_dtype = first.astype(np.int32)

    assert b1k_observation.array_sha256(first) != \
        b1k_observation.array_sha256(same_bytes_new_shape)
    assert b1k_observation.array_sha256(first) != \
        b1k_observation.array_sha256(same_values_new_dtype)


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
