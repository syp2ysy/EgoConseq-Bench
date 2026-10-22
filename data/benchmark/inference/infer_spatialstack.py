#!/usr/bin/env python3
"""SpatialStack QA with the official Qwen3.5 geometry model and image inputs.

Runtime: vllm_serve, with vendor/spatialstack_deps and vendor/spatialstack/src
on PYTHONPATH. The isolated dependency directory contains Transformers 5.3.0.
"""

from common import build_messages, run


def geometry_inputs(images, image_grid_thw):
    """Official scripts/inference/infer.py geometry resizing (VGGT patch size 14)."""
    import numpy as np
    import torch
    from PIL import Image

    tensors = []
    for image, grid in zip(images, image_grid_thw):
        _, height, width = (int(value) for value in grid.tolist())
        resized = image.resize((width * 14, height * 14), Image.Resampling.BICUBIC)
        tensors.append(torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0)
    height = max(tensor.shape[1] for tensor in tensors)
    width = max(tensor.shape[2] for tensor in tensors)
    padded = []
    for tensor in tensors:
        dh, dw = height - tensor.shape[1], width - tensor.shape[2]
        padded.append(torch.nn.functional.pad(
            tensor, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), value=1.0))
    return torch.stack(padded)


def create_predictor(args):
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor
    from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

    config = AutoConfig.from_pretrained(args.model)
    if not getattr(config, "use_geometry_encoder", False):
        raise ValueError("SpatialStack QA requires a geometry-enabled checkpoint")
    # Geometry weights are restored from the released checkpoint by this class.
    # Its official --no-flash-attn2 path permits stock attention on this runtime.
    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=1605632, min_pixels=256 * 28 * 28, padding_side="left")
    processor.tokenizer.padding_side = "left"
    eos = processor.tokenizer.eos_token_id
    generation = {
        "eos_token_id": eos, "pad_token_id": processor.tokenizer.pad_token_id,
        "do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        **getattr(args, "official_settings", {}).get("generation", {}),
    }
    torch.manual_seed(generation.pop("seed", args.seed))
    generation.pop("max_tokens", None)
    generation["max_new_tokens"] = args.max_new_tokens

    def predict(batch):
        results = []
        for _, root, item in batch:
            messages = build_messages(item, root)
            images = []
            for message in messages:
                for block in message["content"]:
                    if block["type"] == "image":
                        with Image.open(block["image"]) as image:
                            images.append(image.convert("RGB"))
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **args.chat_template_kwargs)
            inputs = processor(text=[text], images=images, videos=None, padding=True, return_tensors="pt")
            geometry = geometry_inputs(images, inputs["image_grid_thw"]).to(model.device)
            inputs = inputs.to(model.device)
            inputs["geometry_encoder_inputs"] = [geometry]
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation)
            tokens = generated[:, inputs.input_ids.shape[1]:]
            answer = processor.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            results.append({"answer": answer,
                            "finish_reason": "stop" if eos in tokens[0].tolist() else "length"})
        return results

    return predict, {
        "model_revision": getattr(config, "_commit_hash", None),
        "model_path": args.model,
        "dtype": "bfloat16",
        "image_processor": processor.image_processor.to_dict(),
        "geometry_preprocessing": "official Qwen3.5 image_grid_thw × VGGT patch_size=14",
        "official_source": "https://github.com/jzh15/SpatialStack/tree/b84ebaa5451df42a275077772e7a0a6c66e74453",
    }


if __name__ == "__main__":
    run(create_predictor, model="Journey9ni/SpatialStack-Qwen3.5-4B", backend="spatialstack")
