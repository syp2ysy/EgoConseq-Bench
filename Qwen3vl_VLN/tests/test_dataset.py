import json
from pathlib import Path

from PIL import Image

from vln_baseline.dataset import (
    StreamVLNImitationDataset,
    build_navida_messages,
    uniform_sample_indices,
)


def _write_rgb_sequence(root: Path, video: str, count: int) -> None:
    rgb_dir = root / video / "rgb"
    rgb_dir.mkdir(parents=True)
    for idx in range(count):
        image = Image.new("RGB", (8, 6), (idx * 20 % 255, 0, 0))
        image.save(rgb_dir / f"{idx:03d}.jpg")


def _write_depth_sequence(root: Path, video: str, count: int) -> None:
    depth_dir = root / video / "depth"
    depth_dir.mkdir(parents=True)
    for idx in range(count):
        image = Image.new("L", (8, 6), idx + 1)
        image.save(depth_dir / f"{idx:03d}.png")


def test_uniform_sample_indices_keeps_ends():
    assert uniform_sample_indices(10, 4) == [0, 3, 6, 9]
    assert uniform_sample_indices(3, 8) == [0, 1, 2]
    assert uniform_sample_indices(0, 8) == []


def test_navida_prompt_tail_preserves_sentence_boundary_after_current_observation():
    image = Image.new("RGB", (8, 6))

    messages = build_navida_messages("go to the chair", [image], image)

    assert messages[1]["content"][-1]["text"].startswith(". Your assigned task")


def test_dataset_builds_single_turn_navida_style_sample(tmp_path):
    image_root = tmp_path / "images"
    _write_rgb_sequence(image_root, "episode_1", 8)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "video": "episode_1",
                    "instructions": ["go to the chair"],
                    "actions": [-1, 1, 1, 1, 1, 2, 2, 0],
                }
            ]
        )
    )

    dataset = StreamVLNImitationDataset(
        annotation_path=str(annotation_path),
        image_root=str(image_root),
        history_images=3,
        max_action_segments=3,
        stop_upsample_ratio=1,
    )

    sample = dataset[0]
    assert sample["instruction"] == "go to the chair"
    assert sample["answer"] == "forward 75 cm, forward 25 cm, turn left 30 degree"
    assert len(sample["messages"]) == 3
    assert sample["messages"][0]["role"] == "system"
    assert sample["messages"][1]["role"] == "user"
    assert sample["messages"][2]["role"] == "assistant"
    assert "forward 75 cm" in sample["messages"][2]["content"][0]["text"]
    roles = [message["role"] for message in sample["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert len(dataset) == 2


def test_dataset_stop_sample_uses_last_available_frame(tmp_path):
    image_root = tmp_path / "images"
    _write_rgb_sequence(image_root, "episode_2", 3)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "video": "images/episode_2",
                    "instructions": "stop there",
                    "actions": [-1, 1, 2, 0],
                }
            ]
        )
    )

    dataset = StreamVLNImitationDataset(
        annotation_path=str(annotation_path),
        image_root=str(image_root),
        max_action_segments=2,
        stop_upsample_ratio=1,
    )

    stop_sample = dataset[1]
    assert stop_sample["answer"] == "stop"
    assert stop_sample["current_frame"].endswith("002.jpg")


def test_dataset_supports_navida_random_chunking_with_seeded_no_merge(tmp_path):
    image_root = tmp_path / "images"
    _write_rgb_sequence(image_root, "episode_3", 6)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "video": "episode_3",
                    "instructions": ["go"],
                    "actions": [-1, 1, 1, 1, 1, 2],
                }
            ]
        )
    )

    dataset = StreamVLNImitationDataset(
        annotation_path=str(annotation_path),
        image_root=str(image_root),
        max_action_segments=3,
        stop_upsample_ratio=2,
        chunking_strategy="navida_random",
        chunking_merge_probability=0.0,
        chunking_seed=7,
    )

    assert [dataset[i]["answer"] for i in range(len(dataset))] == [
        "forward 25 cm, forward 25 cm, forward 25 cm",
        "forward 25 cm, turn left 15 degree, stop",
        "forward 25 cm, turn left 15 degree, stop",
    ]


def test_dataset_returns_depth_paths_in_prompt_image_order(tmp_path):
    image_root = tmp_path / "images"
    _write_rgb_sequence(image_root, "episode_4", 5)
    _write_depth_sequence(image_root, "episode_4", 5)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "video": "images/episode_4",
                    "instructions": ["go"],
                    "actions": [-1, 1, 1, 1, 0],
                }
            ]
        )
    )

    dataset = StreamVLNImitationDataset(
        annotation_path=str(annotation_path),
        image_root=str(image_root),
        history_images=2,
        max_action_segments=1,
        stop_upsample_ratio=1,
        depth_enable=True,
    )

    first = dataset[0]
    second = dataset[1]

    assert [Path(path).name for path in first["depth_paths"]] == ["000.png", "000.png"]
    assert [Path(path).name for path in second["depth_paths"]] == ["000.png", "002.png", "003.png"]
