#!/usr/bin/env python3
"""Qwen3-VL text-answer inference on the saved benchmark."""

from common import build_messages, run


def create_predictor(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="auto").eval()
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)

    def predict(batch):
        messages = [build_messages(item, root) for _, root, item in batch]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
            **args.chat_template_kwargs).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        tokens = generated[:, inputs.input_ids.shape[1]:]
        answers = processor.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [{"answer": answer,
                 "finish_reason": "stop" if eos_ids.intersection(output.tolist()) else "length"}
                for answer, output in zip(answers, tokens)]

    return predict, {"model_revision": getattr(model.config, "_commit_hash", None),
                     "image_processor": processor.image_processor.to_dict()}


if __name__ == "__main__":
    run(create_predictor, model="Qwen/Qwen3-VL-4B-Instruct", backend="transformers")
