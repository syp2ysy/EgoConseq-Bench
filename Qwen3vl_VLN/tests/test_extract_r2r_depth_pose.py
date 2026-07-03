import math

import numpy as np

from scripts.extract_r2r_depth_pose import (
    build_camera_params,
    coerce_hfov_for_habitat_config,
    depth_to_uint8,
    episode_outputs_complete,
    filter_pending_annotations,
    select_annotations,
    shard_annotations,
)


class _FakeRotation:
    def __init__(self, real, imag):
        self.real = real
        self.imag = np.array(imag, dtype=np.float32)


class _FakeSensorState:
    def __init__(self):
        self.position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.rotation = _FakeRotation(1.0, [0.0, 0.0, 0.0])


def test_depth_to_uint8_matches_streamvln_png_encoding():
    depth = np.array([[-0.2, 0.0, 0.5, 1.0, 1.2]], dtype=np.float32)

    encoded = depth_to_uint8(depth)

    assert encoded.dtype == np.uint8
    assert encoded.tolist() == [[0, 0, 127, 255, 255]]


def test_build_camera_params_uses_streamvln_intrinsics_and_pose_format():
    params = build_camera_params(
        _FakeSensorState(),
        width=640,
        height=480,
        hfov=79.0,
        min_depth=0.0,
        max_depth=10.0,
        agent_height=1.5,
        sensor_height_offset=1.25,
    )

    expected_focal = 640 / (2 * math.tan(math.radians(79.0) / 2))
    assert params["intrinsics"]["width"] == 640
    assert params["intrinsics"]["height"] == 480
    assert params["intrinsics"]["fx"] == expected_focal
    assert params["intrinsics"]["fy"] == expected_focal
    assert params["intrinsics"]["cx"] == 320.0
    assert params["intrinsics"]["cy"] == 240.0
    assert params["intrinsics"]["hfov"] == 79.0
    assert params["intrinsics"]["near"] == 0.0
    assert params["intrinsics"]["far"] == 10.0
    assert params["intrinsics"]["agent_height"] == 1.5
    assert params["intrinsics"]["sensor_height_offset"] == 1.25
    assert params["extrinsics"]["position"] == [1.0, 2.0, 3.0]
    assert params["extrinsics"]["rotation_quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert params["extrinsics"]["rotation_matrix"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert params["extrinsics"]["camera_to_world"] == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_select_annotations_filters_ids_in_requested_order():
    annotations = [
        {"id": 7, "video": "images/a"},
        {"id": 3, "video": "images/b"},
        {"id": 5, "video": "images/c"},
    ]

    selected = select_annotations(annotations, episode_ids=[5, 7], limit=None)

    assert [item["id"] for item in selected] == [5, 7]


def test_coerce_hfov_for_habitat_config_keeps_integer_hfov_as_int():
    assert coerce_hfov_for_habitat_config(79.0) == 79
    assert isinstance(coerce_hfov_for_habitat_config(79.0), int)
    assert coerce_hfov_for_habitat_config(79.5) == 79.5


def test_shard_annotations_splits_by_annotation_order():
    annotations = [{"id": idx} for idx in range(8)]

    assert [item["id"] for item in shard_annotations(annotations, shard_index=0, num_shards=3)] == [0, 3, 6]
    assert [item["id"] for item in shard_annotations(annotations, shard_index=1, num_shards=3)] == [1, 4, 7]
    assert [item["id"] for item in shard_annotations(annotations, shard_index=2, num_shards=3)] == [2, 5]


def test_episode_outputs_complete_requires_exact_depth_and_pose_files(tmp_path):
    output_dir = tmp_path / "episode"
    (output_dir / "depth").mkdir(parents=True)
    (output_dir / "pose").mkdir()
    for idx in range(1, 3):
        (output_dir / "depth" / f"{idx:03d}.png").write_bytes(b"png")
        (output_dir / "pose" / f"{idx:03d}.json").write_text("{}\n")

    assert episode_outputs_complete(output_dir, frame_count=2)

    (output_dir / "pose" / "002.json").unlink()

    assert not episode_outputs_complete(output_dir, frame_count=2)


def test_filter_pending_annotations_skips_complete_episodes(tmp_path):
    image_root = tmp_path / "images"
    complete = image_root / "complete"
    pending = image_root / "pending"
    for output_dir in (complete, pending):
        (output_dir / "depth").mkdir(parents=True)
        (output_dir / "pose").mkdir()
    (complete / "depth" / "001.png").write_bytes(b"png")
    (complete / "pose" / "001.json").write_text("{}\n")

    annotations = [
        {"id": 1, "video": "images/complete", "actions": [-1]},
        {"id": 2, "video": "images/pending", "actions": [-1]},
    ]

    pending_annotations, skipped_count = filter_pending_annotations(annotations, image_root)

    assert [item["id"] for item in pending_annotations] == [2]
    assert skipped_count == 1
