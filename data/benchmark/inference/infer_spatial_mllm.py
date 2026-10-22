#!/usr/bin/env python3
"""Spatial-MLLM text QA with its official spatial encoder and distinct images.

Runtime: qwen3vl with vendor/spatial_mllm_deps, vendor/spatial_mllm and
vendor/spatial_mllm/src/qwenvl/external on PYTHONPATH (Transformers 4.51.3).
"""

from common import build_messages, run


def create_predictor(args):
    import numpy as np
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import Qwen2_5_VLProcessor
    from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration

    if args.thinking != "auto":
        raise ValueError("Spatial-MLLM has no official enable_thinking switch")
    config = SpatialMLLMConfig.from_pretrained(args.model)
    # The installed FlashAttention rotary kernel requires torch.library.wrap_triton
    # (absent in torch 2.5). Use Transformers' supported SDPA implementation.
    model = SpatialMLLMForConditionalGeneration.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa").eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    generation = {
        "max_new_tokens": args.max_new_tokens, "do_sample": True,
        "temperature": 0.1, "top_p": 0.001, "use_cache": True,
        **getattr(args, "official_settings", {}).get("generation", {}),
    }
    torch.manual_seed(generation.pop("seed", args.seed))
    generation.pop("max_tokens", None)
    generation["max_new_tokens"] = args.max_new_tokens
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)

    def predict(batch):
        results = []
        for _, root, item in batch:
            messages = build_messages(item, root)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            if videos:
                raise ValueError("Saved EgoConseq QA must use distinct images")
            inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device)
            # Match the image processor's temporal padding: each still image is
            # repeated to fill one temporal patch. Keep a separate TCHW tensor
            # per image so the spatial connector never pairs different choices.
            inputs["image_tchw"] = [
                (torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
                 .float().unsqueeze(0).repeat(processor.image_processor.temporal_patch_size, 1, 1, 1)
                 / 255.0).to(model.device)
                for image in images
            ]
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation)
            tokens = generated[:, inputs.input_ids.shape[1]:]
            answer = processor.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            results.append({"answer": answer,
                            "finish_reason": "stop" if eos_ids.intersection(tokens[0].tolist()) else "length"})
        return results

    return predict, {
        "model_revision": getattr(config, "_commit_hash", None),
        "model_path": args.model,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "image_processor": processor.image_processor.to_dict(),
        "spatial_image_layout": "each independent image repeated to fill its own temporal patch",
        "official_generation": generation,
        "official_source": "https://github.com/THU-SI/Spatial-MLLM",
    }


if __name__ == "__main__":
    run(create_predictor, model="Diankun/Spatial-MLLM-v1.1-Instruct-820K", backend="spatial_mllm")
