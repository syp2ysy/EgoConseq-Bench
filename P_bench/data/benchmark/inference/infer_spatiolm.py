#!/usr/bin/env python3
"""SpatioLM inference using its official spatial encoder and chat interface."""

from common import run


def create_predictor(args):
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    from lmms_eval.models.simple.internvl2 import load_image
    from spatiolm.models import InternVL3RChatModel

    model = InternVL3RChatModel.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    image_size = model.config.force_image_size or model.config.vision_config.image_size

    def predict(batch):
        results = []
        for _, root, item in batch:
            model.system_message = next(m["content"] for m in item["messages"] if m["role"] == "system")
            question = "\n".join(m["content"] for m in item["messages"] if m["role"] == "user")
            patches = []
            for path in item["images"]:
                with Image.open(root / path) as image:
                    patches.append(load_image(image.convert("RGB"), input_size=image_size))
            pixels = torch.cat(patches).to(device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                answer = model.chat(tokenizer, pixels, question,
                    {"max_new_tokens": args.max_new_tokens, "do_sample": False},
                    num_patches_list=[len(patch) for patch in patches])
            # Official chat() returns only decoded text, not a token stop reason.
            results.append({"answer": answer, "finish_reason": "unknown"})
        return results
    return predict, {"model_revision": getattr(model.config, "_commit_hash", None),
                     "image_processor": {"implementation": "lmms_eval.models.simple.internvl2.load_image",
                                         "input_size": image_size}}


if __name__ == "__main__":
    run(create_predictor, model="xiaomi-research/SpatioLM-Understanding-InternVL3.5", backend="spatiolm")
