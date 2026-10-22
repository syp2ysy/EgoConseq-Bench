#!/usr/bin/env python3
"""SpatioLM inference using its official spatial encoder and chat interface."""

from pathlib import Path

from common import record_generation, run


# Image preprocessing from lmms-eval, as used by the official SpatioLM example.
# Source revision: 33991bb4b8f258e33e57843d99bda8e39980730b.
# Kept here to avoid requiring the unrelated evaluation framework at runtime.
# Adopted from lmms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval
# Copyright (c) 2025 LMMs-Lab
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    import torch

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def chunk_geometry_queries(model, chunk_size=128):
    """Run the author's eager attention in independent query rows to bound memory."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

    def eager_chunks(module, query, key, value, attention_mask, **kwargs):
        if module.training or query.shape[-2] <= chunk_size:
            return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
        outputs = []
        for start in range(0, query.shape[-2], chunk_size):
            stop = start + chunk_size
            mask = attention_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., start:stop, :]
            outputs.append(eager_attention_forward(
                module, query[..., start:stop, :], key, value, mask, **kwargs)[0])
        # Geometry Qwen3DecoderLayer discards the attention weights.
        return torch.cat(outputs, dim=1), None

    ALL_ATTENTION_FUNCTIONS.register("spatiolm_eager_chunks", eager_chunks)
    for layer in model.language_model.model.vision_3r_cond.layers:
        for attention in (layer.frame_layer.self_attn, layer.global_layer.self_attn):
            attention.config._attn_implementation = "spatiolm_eager_chunks"


def create_predictor(args):
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    from spatiolm.models import InternVL3RChatModel

    settings = getattr(args, "official_settings", {})
    generation = {"do_sample": False, "num_beams": 1,
                  **settings.get("generation", {}), "max_new_tokens": args.max_new_tokens}
    torch.manual_seed(args.seed)
    model = InternVL3RChatModel.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    chunk_geometry_queries(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    image_size = model.config.force_image_size or model.config.vision_config.image_size

    def predict(batch):
        results = []
        for _, root, item in batch:
            model.system_message = "\n\n".join(
                m["content"] for m in item["messages"] if m["role"] == "system")
            question = "\n".join(m["content"] for m in item["messages"] if m["role"] == "user")
            if question.count("<image>") != len(item["images"]):
                raise ValueError("Image placeholders must match separate input images")
            patches = []
            for path in item["images"]:
                with Image.open(root / path) as image:
                    patches.append(load_image(image.convert("RGB"), input_size=image_size))
            pixels = torch.cat(patches).to(device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode(), record_generation(model) as completion:
                answer = model.chat(tokenizer, pixels, question,
                    dict(generation),
                    num_patches_list=[len(patch) for patch in patches])
            results.append({"answer": answer, **completion})
        return results
    checkpoint = Path(args.model)
    revision = getattr(model.config, "_commit_hash", None)
    if checkpoint.parent.name == "snapshots":
        revision = checkpoint.name
    return predict, {"model_revision": revision, "model_path": str(checkpoint),
                     "dtype": str(model.dtype), "resolved_generation": generation,
                     "geometry_attention": {"implementation": "query_chunked_native_eager",
                                            "query_chunk_size": 128},
                     "image_processor": {"implementation": "lmms_eval.models.simple.internvl2.load_image",
                                         "input_size": image_size, "max_num": 6,
                                         "use_thumbnail": True, "multi_image_mode": "separate_images"}}


if __name__ == "__main__":
    run(create_predictor, model="xiaomi-research/SpatioLM-Understanding-InternVL3.5", backend="spatiolm")
