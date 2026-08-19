"""The published sensor/body setting must be a pure function of the pose id.

Drawing it from the pose sampler's RNG would make the configuration depend on
how many candidate poses happened to be rejected first, so a rejection that
changes nothing about the accepted pose would silently move the body radius or
the camera height. These tests pin the sampler to its own hash stream.
"""

import collections

from pipeline import config
from pipeline.pose_setting import sample_public_setting


HEIGHTS = list(config.BENCH_CAMERA_HEIGHTS_M)
FOVS = [tuple(pair) for pair in config.BENCH_FOVS_DEG]
RADII = list(config.RADII_M)


def _sample(scene_id="scene-a", pose_index=0, seed=42):
    return sample_public_setting(
        seed=seed, scene_id=scene_id, pose_index=pose_index,
        heights=HEIGHTS, fovs=FOVS, radii=RADII)


def test_setting_comes_from_the_frozen_option_grids():
    setting = _sample()

    assert setting.nominal_camera_height_m in HEIGHTS
    assert setting.fov in FOVS
    assert setting.radius_m in RADII
    assert FOVS[setting.fov_index] == setting.fov


def test_setting_is_reproducible_for_the_same_pose_identity():
    assert _sample(pose_index=7) == _sample(pose_index=7)
    assert _sample(scene_id="scene-b") == _sample(scene_id="scene-b")


def test_setting_depends_on_seed_and_scene():
    # pose_index is covered by the spread test below: two neighbouring indexes
    # may legitimately draw the same setting, so asserting inequality there
    # would be flaky rather than strict.
    base = _sample(scene_id="scene-a", pose_index=3, seed=42)

    assert _sample(scene_id="scene-z", pose_index=3, seed=42) != base
    assert _sample(scene_id="scene-a", pose_index=3, seed=43) != base


def test_settings_spread_over_every_option_rather_than_collapsing():
    settings = [_sample(pose_index=index) for index in range(60)]
    heights = collections.Counter(s.nominal_camera_height_m for s in settings)
    fovs = collections.Counter(s.fov for s in settings)
    radii = collections.Counter(s.radius_m for s in settings)

    assert set(heights) == set(HEIGHTS)
    assert set(fovs) == set(FOVS)
    assert set(radii) == set(RADII)


def test_sampler_takes_no_random_state_so_pose_retries_cannot_move_it():
    # A structural guarantee, not a behavioural one: if the signature ever
    # grows an rng parameter, the rejection-count coupling is back.
    import inspect

    parameters = inspect.signature(sample_public_setting).parameters

    assert set(parameters) == {
        "seed", "scene_id", "pose_index", "heights", "fovs", "radii"}
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
               for parameter in parameters.values())
