import torch

from vln_baseline.collator import build_assistant_labels


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
