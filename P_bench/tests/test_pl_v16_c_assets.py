"""Native terminal RGB authority for the frozen C1 candidate task."""

import dataclasses
import hashlib
import importlib
import io
import math

import numpy as np
import pytest
from PIL import Image

from pipeline import config, frame as frame_module, perception, record
from tests._synthetic import make_frame, source_provenance
from tests.test_pl_v16_a_candidates import _case as _a_case


def _c_module():
    return importlib.import_module("pipeline.future_view_selection")


def _clear_outcome(*, outcome_id="b020-main-a"):
    _rec, outcome = _a_case(collision=False)
    outcome["outcome_id"] = outcome_id
    endpoint = {"x": 0.25, "z": 1.0, "heading_deg": 30.0}
    outcome["execution"] = {
        "completed": True,
        "stop_reason": "completed",
        "nominal_forward_m": 1.0,
        "executed_forward_m": 1.0,
        "realized_pose": dict(endpoint),
    }
    outcome["checkpoints"] = [
        {
            "requested_progress": 0.0,
            "realized_progress": 0.0,
            "pose": {"x": 0.0, "z": 0.0, "heading_deg": 0.0},
        },
        {
            "requested_progress": 1.0,
            "realized_progress": 1.0,
            "pose": endpoint,
        },
    ]
    outcome["future_view"] = {
        "status": "computed",
        "objects_entering_view": [],
        "initial_depth_reprojection_agrees": True,
    }
    outcome["evidence"] = {
        **(outcome.get("evidence") or {}),
        "future_view": {"status": "sufficient", "source_frame": "initial"},
    }
    return outcome


def _terminal_frame(base, outcome, *, value=77):
    pose = outcome["checkpoints"][-1]["pose"]
    position = perception.world_from_local(
        np.asarray([[pose["x"], 0.0, pose["z"]]], dtype=np.float64),
        base.position, base.yaw_rad,
    )[0]
    yaw = base.yaw_rad - math.radians(pose["heading_deg"])
    return dataclasses.replace(
        base,
        frame_id=base.frame_id + "-terminal",
        position=position,
        yaw_rad=yaw,
        rgb=np.full_like(base.rgb, value, dtype=np.uint8),
    )


def _materialize(tmp_path, *, mutate_terminal=None, cache=True):
    module = _c_module()
    rec, _unused = _a_case(collision=False)
    base = make_frame()
    base.frame_id = rec["frame_id"]
    base.scene_id = rec["source"]["scene_id"]
    outcome = _clear_outcome()
    terminal = _terminal_frame(base, outcome)
    if mutate_terminal is not None:
        terminal = mutate_terminal(terminal)
    key = module.terminal_render_cache_key(outcome)
    render_cache = {key: terminal} if cache else {}
    contract = record.r2r_v16_collection_contract(rec["source"], "main")
    atom = module.materialize_terminal_rgb_asset(
        tmp_path,
        base_frame=base,
        outcome=outcome,
        render_cache=render_cache,
        source=rec["source"],
        collection_contract=contract,
    )
    return atom, base, outcome, terminal, rec


def test_native_png_is_byte_stable_lossless_and_native(tmp_path):
    """Catches thumbnails, lossy encoding, or encode/decode pixel drift."""
    module = _c_module()
    rgb = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)

    first = module.encode_native_rgb_png(rgb, expected_resolution=(24, 16))
    second = module.encode_native_rgb_png(
        rgb.copy(), expected_resolution=(24, 16))

    assert first == second
    assert first.mode == "RGB"
    assert (first.width_px, first.height_px) == (24, 16)
    assert hashlib.sha256(first.png_bytes).hexdigest() == first.png_sha256
    with Image.open(io.BytesIO(first.png_bytes)) as opened:
        opened.load()
        assert opened.mode == "RGB"
        assert opened.size == (24, 16)
        assert np.array_equal(np.asarray(opened), rgb)


@pytest.mark.parametrize(
    "rgb,resolution",
    [
        (np.zeros((16, 24, 3), dtype=np.float32), (24, 16)),
        (np.zeros((16, 24, 4), dtype=np.uint8), (24, 16)),
        (np.zeros((16, 24, 3), dtype=np.uint8), (12, 8)),
    ],
)
def test_native_png_rejects_wrong_dtype_shape_or_resolution(rgb, resolution):
    """Catches implicit conversion or resizing entering the C1 GT path."""
    with pytest.raises(ValueError):
        _c_module().encode_native_rgb_png(
            rgb, expected_resolution=resolution)


def test_gs_terminal_rgb_is_bound_without_a_legacy_collection_contract(
        tmp_path):
    module = _c_module()
    base = make_frame()
    base.scene_id = "synthetic-gs"
    source = source_provenance(base.scene_id, dataset="gs")
    outcome = _clear_outcome()
    terminal = _terminal_frame(base, outcome)

    atom = module.materialize_terminal_rgb_asset(
        tmp_path,
        base_frame=base,
        outcome=outcome,
        render_cache={module.terminal_render_cache_key(outcome): terminal},
        source=source,
        collection_contract=None,
    )
    outcome["terminal_rgb_asset"] = atom
    outcome["base_rollout_key"] = atom["binding"]["base_rollout_key"]
    rec = {
        "schema_version": record.V18_SCHEMA_VERSION,
        "frame_id": base.frame_id,
        "scene_id": base.scene_id,
        "pose": {
            "position": base.position.tolist(),
            "yaw_rad": base.yaw_rad,
        },
        "sensor": base.sensor.to_dict(),
        "source": source,
        "outcomes": [outcome],
    }

    assert atom["source"]["source_dataset"] == "gs"
    assert atom["renderer"] == {
        "protocol": "gsplat-rgb-ed.v1",
        "backend": "gsplat",
        "color_format": "uint8_rgb",
        "source_scene_sha256": source["source_assets"][0]["sha256"],
        "collision_authority_sha256": atom["source"][
            "collision_authority_sha256"],
        "source_bundle_authority_sha256": atom["source"][
            "source_bundle_authority_sha256"],
        "near_m": config.GS_RENDER_NEAR_M,
        "far_m": config.GS_RENDER_FAR_M,
    }
    assert module.validate_terminal_rgb_asset(
        rec, outcome, asset_root=tmp_path).png_sha256 == atom["png_sha256"]


def test_terminal_asset_binds_cached_true_pose_sensor_source_and_bytes(tmp_path):
    """Catches a correct-looking PNG detached from the certified endpoint."""
    atom, base, outcome, terminal, rec = _materialize(tmp_path)
    asset = tmp_path / atom["path"]

    assert asset.is_file()
    assert asset.suffix == ".png"
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == atom["png_sha256"]
    assert atom["schema"] == "terminal-rgb-asset.v1"
    assert atom["binding"]["frame_id"] == base.frame_id
    assert atom["binding"]["outcome_id"] == outcome["outcome_id"]
    assert atom["binding"]["base_rollout_key"] == \
        record.base_rollout_key(base, outcome)
    assert atom["binding"]["action_sha256"] == record.canonical_atom_sha256({
        "actions": outcome["actions"],
    })
    assert atom["terminal_pose"]["world_position"] == \
        pytest.approx(terminal.position.tolist())
    assert atom["terminal_pose"]["world_yaw_rad"] == \
        pytest.approx(terminal.yaw_rad)
    assert atom["sensor"] == base.sensor.to_dict()
    assert atom["source"]["source_assets_sha256"] == \
        rec["source"]["source_assets_sha256"]
    assert atom["renderer"]["protocol"] == "habitat-sim-rgb.v1"
    assert atom["sha256"] == record.canonical_atom_sha256({
        key: value for key, value in atom.items() if key != "sha256"
    })


def test_terminal_asset_accepts_rgb_only_endpoint_render(tmp_path):
    """Natural C1 candidates must not rebuild million-point semantics."""
    module = _c_module()
    rec, _unused = _a_case(collision=False)
    base = make_frame()
    base.frame_id = rec["frame_id"]
    base.scene_id = rec["source"]["scene_id"]
    outcome = _clear_outcome()
    expected = _terminal_frame(base, outcome, value=91)

    class RGBOnlySim:
        def render(self, position, yaw, cam_h, hfov, vfov):
            assert np.array_equal(position, expected.position)
            assert yaw == pytest.approx(expected.yaw_rad)
            assert cam_h == base.sensor.nominal_camera_offset_m
            assert hfov == base.sensor.hfov_deg
            assert vfov == base.sensor.vfov_deg
            return frame_module.RenderObservation(
                rgb=expected.rgb,
                depth=np.ones_like(base.depth),
                K=base.K,
                sensor=base.sensor,
                position=expected.position,
                yaw_rad=expected.yaw_rad,
            )

        def assign_instances(self, _points):
            pytest.fail("terminal RGB path reconstructed semantic instances")

    terminal = frame_module.build_terminal_rgb_observation(
        RGBOnlySim(), base, outcome["checkpoints"][-1]["pose"])
    atom = module.materialize_terminal_rgb_asset(
        tmp_path,
        base_frame=base,
        outcome=outcome,
        render_cache={module.terminal_render_cache_key(outcome): terminal},
        source=rec["source"],
        collection_contract=record.r2r_v16_collection_contract(
            rec["source"], "main"),
    )

    assert terminal.rgb is expected.rgb
    assert atom["terminal_pose"]["world_position"] == pytest.approx(
        expected.position.tolist())
    assert atom["terminal_pose"]["world_yaw_rad"] == pytest.approx(
        expected.yaw_rad)


def test_terminal_asset_missing_cached_frame_fails_closed(tmp_path):
    """Catches the legacy review rerender path being used as a fallback."""
    with pytest.raises(ValueError, match="cached terminal RGB is missing"):
        _materialize(tmp_path, cache=False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: dataclasses.replace(
            value, position=value.position + np.array([0.01, 0.0, 0.0])),
         "pose position"),
        (lambda value: dataclasses.replace(
            value, position=value.position + np.array([4e-7, 0.0, 0.0])),
         "pose position"),
        (lambda value: dataclasses.replace(value, yaw_rad=value.yaw_rad + 0.01),
         "pose yaw"),
        (lambda value: dataclasses.replace(
            value, yaw_rad=value.yaw_rad + 4e-7),
         "pose yaw"),
        (lambda value: dataclasses.replace(
            value,
            sensor=dataclasses.replace(value.sensor, hfov_deg=110.0)),
         "sensor"),
        (lambda value: dataclasses.replace(
            value, rgb=value.rgb.astype(np.float32)),
         "uint8"),
    ],
)
def test_terminal_asset_rejects_rounded_key_pose_sensor_or_rgb_mismatch(
        tmp_path, mutate, message):
    """Catches a round(6) cache hit authorizing the wrong rendered Frame."""
    with pytest.raises(ValueError, match=message):
        _materialize(tmp_path, mutate_terminal=mutate)


def test_terminal_asset_existing_byte_conflict_is_not_overwritten(tmp_path):
    """Catches content-addressed paths silently replacing prior bytes."""
    atom, _base, _outcome, _terminal, _rec = _materialize(tmp_path)
    path = tmp_path / atom["path"]
    retried, *_unused = _materialize(tmp_path)
    assert retried == atom
    path.write_bytes(b"conflict")

    with pytest.raises(ValueError, match="conflicting terminal RGB asset"):
        _materialize(tmp_path)
    assert path.read_bytes() == b"conflict"


def test_terminal_asset_rejects_in_directory_final_component_symlink(tmp_path):
    """Catches resolve() hiding a final-component symlink inside asset_dir."""
    atom, _base, _outcome, _terminal, _rec = _materialize(tmp_path)
    expected = tmp_path / atom["path"]
    payload = expected.read_bytes()
    expected.unlink()
    alternate = expected.with_name("alternate.png")
    alternate.write_bytes(payload)
    expected.symlink_to(alternate.name)

    with pytest.raises(ValueError, match="conflicting terminal RGB asset"):
        _materialize(tmp_path)
