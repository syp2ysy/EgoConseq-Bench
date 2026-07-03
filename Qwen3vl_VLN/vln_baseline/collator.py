from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

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
    def __init__(
        self,
        processor: Any,
        depth_enable: bool = False,
        depth_max_meters: float = 10.0,
        spatial_merge_size: int = 2,
        image_size: Optional[Tuple[int, int]] = None,
        depth_resize_mode: str = "nearest",
    ):
        self.processor = processor
        self.depth_enable = depth_enable
        self.depth_max_meters = depth_max_meters
        self.spatial_merge_size = spatial_merge_size
        self.image_size = image_size
        self.depth_resize_mode = depth_resize_mode

    def _depth_resample(self):
        resampling = getattr(Image, "Resampling", Image)
        mode = self.depth_resize_mode.lower()
        if mode == "nearest":
            return resampling.NEAREST
        if mode == "bilinear":
            return resampling.BILINEAR
        raise ValueError(f"Unsupported depth_resize_mode={self.depth_resize_mode!r}")

    def _image_token_id(self) -> int:
        tokenizer = self.processor.tokenizer
        if hasattr(tokenizer, "image_token_id"):
            return int(tokenizer.image_token_id)
        image_token = getattr(tokenizer, "image_token", None)
        if image_token is not None:
            return int(tokenizer.convert_tokens_to_ids(image_token))
        for token in ("<|image_pad|>", "<image>"):
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if token_id is not None:
                return int(token_id)
        raise ValueError("Could not resolve Qwen image token id from tokenizer")

    @staticmethod
    def _contiguous_true_spans(mask: torch.Tensor) -> List[Tuple[int, int]]:
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not indices:
            return []
        spans = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            spans.append((start, prev + 1))
            start = prev = idx
        spans.append((start, prev + 1))
        return spans

    def _image_token_spans(self, input_ids: torch.Tensor) -> torch.Tensor:
        image_token_id = self._image_token_id()
        spans = []
        for batch_idx in range(input_ids.shape[0]):
            row_spans = self._contiguous_true_spans(input_ids[batch_idx] == image_token_id)
            spans.extend((batch_idx, start, end) for start, end in row_spans)
        return torch.tensor(spans, dtype=torch.long)

    def _load_depth(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(path)
        raw_depth = np.asarray(image)
        scale = (
            65535.0
            if raw_depth.dtype == np.uint16
            or image.mode in {"I;16", "I;16L", "I;16B", "I;16N"}
            or raw_depth.max(initial=0) > 255
            else 255.0
        )
        if self.image_size is not None:
            if raw_depth.ndim == 3:
                image = Image.fromarray(raw_depth[..., 0])
            image = image.resize(self.image_size, self._depth_resample())
            raw_depth = np.asarray(image)
        depth = raw_depth.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth_tensor = torch.from_numpy(depth).unsqueeze(0) / scale * float(self.depth_max_meters)
        depth_mask = (depth_tensor > 0.0) & (depth_tensor < float(self.depth_max_meters))
        return depth_tensor, depth_mask

    def _depth_batch(self, examples: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth_tensors = []
        mask_tensors = []
        batch_indices = []
        current_indices = []
        flat_idx = 0
        for batch_idx, example in enumerate(examples):
            paths = list(example.get("depth_paths", []))
            if not paths:
                raise ValueError("depth_enable=True requires each example to provide depth_paths")
            for path in paths:
                depth, mask = self._load_depth(path)
                depth_tensors.append(depth)
                mask_tensors.append(mask)
                batch_indices.append(batch_idx)
            current_indices.append(flat_idx + len(paths) - 1)
            flat_idx += len(paths)
        return (
            torch.stack(depth_tensors, dim=0),
            torch.stack(mask_tensors, dim=0),
            torch.tensor(batch_indices, dtype=torch.long),
            torch.tensor(current_indices, dtype=torch.long),
        )

    def _validate_depth_alignment(
        self,
        *,
        spans: torch.Tensor,
        image_grid_thw: torch.Tensor,
        depth_count: int,
    ) -> None:
        if spans.shape[0] != image_grid_thw.shape[0]:
            raise ValueError(
                "image token span count must match image_grid_thw rows: "
                f"spans={spans.shape[0]}, image_grid_thw={image_grid_thw.shape[0]}"
            )
        if spans.shape[0] != depth_count:
            raise ValueError(f"depth count must match image spans: depth count={depth_count}, spans={spans.shape[0]}")
        for idx, (batch_idx, start, end) in enumerate(spans.tolist()):
            t, h, w = [int(value) for value in image_grid_thw[idx].tolist()]
            expected = t * h * w // (self.spatial_merge_size**2)
            actual = end - start
            if actual != expected:
                raise ValueError(
                    "image token span length does not match image_grid_thw: "
                    f"image={idx}, batch={batch_idx}, span={actual}, expected={expected}, grid={(t, h, w)}"
                )

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
        if self.depth_enable:
            depth_gt, depth_mask, image_batch_indices, current_image_indices = self._depth_batch(examples)
            image_token_spans = self._image_token_spans(batch["input_ids"])
            self._validate_depth_alignment(
                spans=image_token_spans,
                image_grid_thw=batch["image_grid_thw"],
                depth_count=depth_gt.shape[0],
            )
            batch["depth_gt"] = depth_gt
            batch["depth_mask"] = depth_mask
            batch["image_token_spans"] = image_token_spans
            batch["image_batch_indices"] = image_batch_indices
            batch["current_image_indices"] = current_image_indices
        return batch
