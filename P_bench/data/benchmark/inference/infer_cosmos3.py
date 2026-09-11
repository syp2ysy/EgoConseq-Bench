#!/usr/bin/env python3
"""Cosmos3-Edge Reasoner: saved images to text, without the Generator tower."""

from pathlib import Path

from common import build_messages, run


def create_predictor(args):
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    checkpoint = args.model if Path(args.model).is_dir() else snapshot_download(args.model, allow_patterns=[
        "*.json", "*.jinja", "transformer/*.safetensors", "vision_encoder/*.safetensors"])
    processor = AutoProcessor.from_pretrained(checkpoint)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="auto").eval()
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)

    def predict(batch):
        messages = [build_messages(item, root) for _, root, item in batch]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", processor_kwargs={"padding": True},
            **args.chat_template_kwargs).to(model.device, torch.bfloat16)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        tokens = generated[:, inputs.input_ids.shape[1]:]
        answers = processor.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [{"answer": answer,
                 "finish_reason": "stop" if eos_ids.intersection(output.tolist()) else "length"}
                for answer, output in zip(answers, tokens)]

    return predict, {"model_revision": Path(checkpoint).name,
                     "transformers_version": transformers.__version__,
                     "torch_version": torch.__version__,
                     "image_processor": processor.image_processor.to_dict()}


if __name__ == "__main__":
    run(create_predictor, model="nvidia/Cosmos3-Edge", backend="transformers")
