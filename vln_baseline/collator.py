"""Encode navigation conversations and supervise assistant action tokens."""

from typing import Any, Dict, List

import torch

IGNORE_INDEX = -100


def build_assistant_labels(input_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    pad_token_id = tokenizer.pad_token_id
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]

    for batch_idx in range(input_ids.shape[0]):
        ids = input_ids[batch_idx].tolist()
        seq_len = len(ids)
        i = 0
        while i < seq_len:
            if ids[i] != im_start_id:
                i += 1
                continue
            assistant_pos = None
            for j in range(i + 1, min(i + 6, seq_len)):
                if ids[j] == assistant_token_id:
                    assistant_pos = j
                    break
            if assistant_pos is None:
                i += 1
                continue
            content_start = assistant_pos + 1
            while content_start < seq_len and ids[content_start] == newline_token_id:
                content_start += 1
            content_end = content_start
            while content_end < seq_len and ids[content_end] != im_end_id:
                content_end += 1
            if content_end > content_start:
                end_pos = min(content_end + 2, seq_len)
                labels[batch_idx, content_start:end_pos] = input_ids[batch_idx, content_start:end_pos]
            i = content_end + 1

    labels[input_ids == pad_token_id] = IGNORE_INDEX
    image_token = getattr(tokenizer, "image_token", None)
    if image_token is not None:
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        labels[input_ids == image_token_id] = IGNORE_INDEX
    return labels


class QwenVLNCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages = [example["messages"] for example in examples]
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            for message in messages
        ]
        image_inputs = []
        for message in messages:
            images = []
            for turn in message:
                for part in turn["content"]:
                    if part["type"] == "image":
                        images.append(part["image"])
            image_inputs.append(images)

        batch = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = build_assistant_labels(batch["input_ids"], self.processor.tokenizer)
        return batch
