"""What a single-setting pose actually writes, end of pipeline.

The pure sampler tests prove the draw is deterministic and the validator tests
prove a bad record is rejected. Neither notices if the collector labels an
honest record dishonestly: `sensor_intervention` is computed once per run from
the full height x FOV grid, so a pose that publishes one sampled profile would
still be stamped `type: sensor_profile, changed_fields: [height, hfov, vfov]`
and read as a paired do(FOV) intervention that never happened.

These tests drive `_persist_pose_group` directly because that is the only place
the descriptor and the sampled setting meet.
"""

import collections
from types import SimpleNamespace

import numpy as np

from pipeline import collection_runtime
from pipeline.pose_setting import PublicSetting, SETTING_SAMPLING_POLICY


LEVEL_FLOOR_FIT = {
    "normal": [0.0, 1.0, 0.0], "offset": 0.0, "inlier_ratio": 1.0,
    "residual_rms_m": 0.0, "support": 1024,
}
SETTING = PublicSetting(
    nominal_camera_height_m=1.0, fov_index=1, fov=(110.0, 70.0),
    radius_m=0.2)


def _persist(monkeypatch, tmp_path, *, setting, intervention_type,
             changed_fields, setting_policy=SETTING_SAMPLING_POLICY,
             active_radii=None):
    """Run one pose through persistence and return the record it built."""
    built = {}

    monkeypatch.setattr(
        collection_runtime.Image, "fromarray",
        lambda _array: SimpleNamespace(save=lambda _path: None))
    monkeypatch.setattr(
        collection_runtime, "validate_records_before_spool",
        lambda _records, **_kwargs: None)
    monkeypatch.setattr(
        collection_runtime, "append_record_group",
        lambda *_args, **_kwargs: None)

    def build_record(*_args, **kwargs):
        built.update(kwargs)
        return {"outcomes": []}

    monkeypatch.setattr(collection_runtime, "build_record", build_record)
    frame = SimpleNamespace(
        frame_id="frame",
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32),
    )
    state = {
        "selected_ids": [], "group_labels": {}, "selected": [],
        "diagnostics": [], "accepted_outcomes": {},
        "variants": [(object(), frame)], "render_caches": {"frame": {}},
        "pools": {}, "selection_seed": 1,
        "required_siblings": 1,
        "calibration": SimpleNamespace(canonical_floor_fit=LEVEL_FLOOR_FIT),
        "position": np.array([0.0, 0.0, 0.0]), "yaw": 0.0,
        "intervention_group_id": "group",
        "scene": SimpleNamespace(provenance=lambda: {
            "source_dataset": "gs",
        }), "scene_id": "scene",
        "active_radii": (list(active_radii) if active_radii is not None else
                         ([setting.radius_m] if setting else
                          [0.15, 0.2, 0.25])),
        "setting": setting,
        "proposal_provenance": {},
        "bank_manifest": None,
    }
    assert collection_runtime._persist_pose_group(
        state,
        args=SimpleNamespace(
            debug_images=False, save_arrays=False, out=str(tmp_path),
            collection_mode="diagnostic", radii=(0.15, 0.20, 0.25),
            action_mode="balanced",
            setting_sampling_policy=setting_policy),
        image_dir=str(tmp_path / "img"), array_dir=str(tmp_path / "arr"),
        records_path=tmp_path / "records.jsonl",
        scene_index=0, pose_index=0, sampler_contract_hash="bank",
        intervention_type=intervention_type, changed_fields=changed_fields,
        funnel=SimpleNamespace(record_pose=lambda *_args: None),
        stats=collections.Counter(),
    ) is True
    return built


def test_a_sampled_setting_is_not_published_as_a_sensor_intervention(
        monkeypatch, tmp_path):
    # The run-level descriptor still describes the full grid; the pose must
    # override it, because this pose has no sensor sibling to contrast with.
    built = _persist(
        monkeypatch, tmp_path, setting=SETTING,
        intervention_type="sensor_profile",
        changed_fields=["nominal_camera_offset_m", "hfov_deg", "vfov_deg"])

    assert built["intervention"]["type"] == "base"
    assert built["intervention"]["changed_fields"] == []


def test_the_pose_still_publishes_only_the_sampled_body_radius(
        monkeypatch, tmp_path):
    built = _persist(
        monkeypatch, tmp_path, setting=SETTING,
        intervention_type="sensor_profile", changed_fields=["hfov_deg"])

    assert built["selection"]["required_radii_m"] == [SETTING.radius_m]


def test_the_sampled_descriptor_agrees_with_the_sensor_taxonomy():
    # pose_setting hardcodes the base case rather than importing the taxonomy.
    # This is what stops the two from drifting apart.
    from pipeline.collection_cli import sensor_intervention
    from pipeline import pose_setting

    assert pose_setting.intervention_descriptor(
        SETTING, "sensor_profile", ["hfov_deg"]) == sensor_intervention(
            [SETTING.nominal_camera_height_m], [SETTING.fov])


def test_a_run_without_sampling_keeps_its_grid_descriptor(
        monkeypatch, tmp_path):
    # Guards the override from firing on runs that really do enumerate the
    # grid, which is what makes the first assertion meaningful.
    built = _persist(
        monkeypatch, tmp_path, setting=None,
        intervention_type="sensor_profile",
        changed_fields=["nominal_camera_offset_m"])

    assert built["intervention"]["type"] == "sensor_profile"
    assert built["intervention"]["changed_fields"] == [
        "nominal_camera_offset_m"]
