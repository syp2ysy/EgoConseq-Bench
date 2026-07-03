import numpy as np
import torch
from PIL import Image

from vln_baseline.collator import QwenVLNCollator, build_assistant_labels


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.ids = {
            "<|im_start|>": 10,
            "<|im_end|>": 11,
            "assistant": 12,
            "\n": 13,
        }

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def encode(self, text, add_special_tokens=False):
        return [self.ids[text]]


def test_build_assistant_labels_masks_non_assistant_tokens():
    input_ids = torch.tensor(
        [
            [
                10,
                99,
                13,
                30,
                11,
                10,
                12,
                13,
                41,
                42,
                11,
                13,
                0,
            ]
        ]
    )

    labels = build_assistant_labels(input_ids, FakeTokenizer())

    assert labels.tolist()[0] == [
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        41,
        42,
        11,
        13,
        -100,
    ]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.tokenizer.image_token = "<image>"
        self.tokenizer.ids["<image>"] = 88
        self.calls = {}

    def apply_chat_template(self, message, tokenize=False, add_generation_prompt=False):
        return "prompt"

    def __call__(self, text, images, padding, return_tensors):
        self.calls["images"] = images
        input_ids = torch.tensor(
            [
                [1, 88, 88, 88, 88, 2, 88, 88, 88, 88],
                [3, 88, 88, 88, 88, 0, 0, 0, 0, 0],
            ]
        )
        return {
            "input_ids": input_ids,
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4], [1, 4, 4]]),
        }


def _write_depth(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 3), value).save(path)


def test_load_depth_decodes_16bit_png_with_65535_scale(tmp_path):
    path = tmp_path / "depth16.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((2, 2), 32768, dtype=np.uint16)).save(path)
    collator = QwenVLNCollator(
        FakeProcessor(),
        depth_enable=True,
        depth_max_meters=10.0,
        spatial_merge_size=2,
    )

    depth, mask = collator._load_depth(str(path))

    assert torch.isclose(depth[0, 0, 0], torch.tensor(32768 / 65535 * 10.0), atol=1e-4)
    assert mask.all()


def test_load_depth_resizes_to_image_size_when_configured(tmp_path):
    path = tmp_path / "depth.png"
    _write_depth(path, 64)
    collator = QwenVLNCollator(
        FakeProcessor(),
        depth_enable=True,
        depth_max_meters=10.0,
        spatial_merge_size=2,
        image_size=(2, 2),
        depth_resize_mode="nearest",
    )

    depth, mask = collator._load_depth(str(path))

    assert depth.shape == (1, 2, 2)
    assert mask.shape == (1, 2, 2)
    assert torch.isclose(depth[0, 0, 0], torch.tensor(64 / 255 * 10.0))
    assert mask.all()


def test_collator_flattens_depths_and_image_token_spans_in_processor_order(tmp_path):
    paths = [tmp_path / f"{idx}.png" for idx in range(3)]
    for idx, path in enumerate(paths, start=1):
        _write_depth(path, idx * 10)
    processor = FakeProcessor()
    collator = QwenVLNCollator(
        processor,
        depth_enable=True,
        depth_max_meters=10.0,
        spatial_merge_size=2,
    )

    batch = collator(
        [
            {"messages": [{"role": "user", "content": []}], "depth_paths": [str(paths[0]), str(paths[1])]},
            {"messages": [{"role": "user", "content": []}], "depth_paths": [str(paths[2])]},
        ]
    )

    assert batch["depth_gt"].shape == (3, 1, 3, 4)
    assert batch["depth_mask"].shape == (3, 1, 3, 4)
    assert torch.isclose(batch["depth_gt"][0, 0, 0, 0], torch.tensor(10 / 255 * 10.0))
    assert batch["image_token_spans"].tolist() == [[0, 1, 5], [0, 6, 10], [1, 1, 5]]
    assert batch["image_batch_indices"].tolist() == [0, 0, 1]
    assert batch["current_image_indices"].tolist() == [1, 2]


def test_collator_rejects_depth_count_mismatch(tmp_path):
    path = tmp_path / "0.png"
    _write_depth(path, 10)
    collator = QwenVLNCollator(
        FakeProcessor(),
        depth_enable=True,
        depth_max_meters=10.0,
        spatial_merge_size=2,
    )

    try:
        collator(
            [
                {"messages": [{"role": "user", "content": []}], "depth_paths": [str(path)]},
                {"messages": [{"role": "user", "content": []}], "depth_paths": [str(path)]},
            ]
        )
    except ValueError as exc:
        assert "depth count" in str(exc)
    else:
        raise AssertionError("Expected depth count mismatch to raise")
