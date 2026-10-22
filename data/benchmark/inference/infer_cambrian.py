#!/usr/bin/env python3
"""Cambrian-S 3B/7B using the official image processor and Qwen conversation template."""

from copy import copy
from pathlib import Path
import subprocess

from common import run


def create_predictor(args):
    import cambrian
    import torch
    import transformers
    from PIL import Image
    from cambrian.conversation import conv_templates
    from cambrian.mm_utils import process_images, tokenizer_image_token
    from cambrian.model.builder import load_pretrained_model

    tokenizer, model, processors, _ = load_pretrained_model(
        args.model, None, Path(args.model).name, device_map="cuda", attn_implementation="sdpa")
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)

    def predict(batch):
        results = []
        for _, root, item in batch:
            conv = conv_templates["qwen_2"].copy()
            for message in item["messages"]:
                if message["role"] == "system":
                    conv.system = "<|im_start|>system\n" + message["content"]
                elif message["role"] == "user":
                    conv.append_message(conv.roles[0], message["content"])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            if prompt.count("<image>") != len(item["images"]):
                raise ValueError(f"Image placeholders disagree with images: {item['id']}")
            images = []
            for path in item["images"]:
                with Image.open(root / path) as image:
                    images.append(image.convert("RGB"))
            # The official processor switches multi-image inputs to pad in place.
            # Copy the config so subsequent single-image questions retain anyres.
            tensors, sizes = process_images(images, processors, copy(model.config), use_pad=len(images) > 1)
            inputs = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
            with torch.inference_mode():
                tokens = model.generate(
                    inputs=inputs, images=[tensor.to(device="cuda", dtype=model.dtype) for tensor in tensors],
                    image_sizes=sizes, do_sample=False, num_beams=1, use_cache=True,
                    max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0,
                    repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)[0]
            # inputs_embeds generation can prefix BOS, which is also an EOS ID.
            # Only the last token determines whether generation stopped normally.
            results.append({"answer": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                            "finish_reason": "stop" if tokens[-1].item() in eos_ids else "length"})
        return results

    revision = subprocess.check_output(
        ["git", "-C", str(Path(cambrian.__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip()
    download = Path(args.model) / ".cache/huggingface/download/config.json.metadata"
    model_revision = download.read_text().splitlines()[0] if download.exists() else model.config._commit_hash
    return predict, {"model_revision": model_revision, "implementation_revision": revision,
                     "transformers_version": transformers.__version__,
                     "image_processor": {"single_image": model.config.image_aspect_ratio,
                                         "multiple_images": "pad", "si_token_len": model.config.si_token_len,
                                         "anyres_max_subimages": model.config.anyres_max_subimages}}


if __name__ == "__main__":
    run(create_predictor, model="nyu-visionx/Cambrian-S-3B", backend="cambrian")
