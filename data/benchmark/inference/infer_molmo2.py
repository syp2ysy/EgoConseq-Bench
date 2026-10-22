#!/usr/bin/env python3
"""Molmo2 inference with the official Transformers processor and precision."""

from pathlib import Path

from common import build_messages, completion_metadata, run


def molmo_messages(item, image_root):
    """Keep system instructions as text because Molmo2 has no system role."""
    messages = build_messages(item, image_root)
    system = "\n\n".join(
        block["text"] for message in messages if message["role"] == "system"
        for block in message["content"] if block["type"] == "text")
    messages = [message for message in messages if message["role"] != "system"]
    if not messages or messages[0]["role"] != "user":
        raise ValueError("Molmo2 requires a user message")
    if system:
        messages[0]["content"].insert(0, {"type": "text", "text": system + "\n\n"})
    return messages


def create_predictor(args):
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if args.thinking != "auto":
        raise ValueError("Molmo2's official template has no thinking switch; use --thinking auto")
    settings = getattr(args, "official_settings", {})
    generation = {**settings.get("generation", {}), "max_new_tokens": args.max_new_tokens}
    torch.manual_seed(args.seed)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto",
        device_map="balanced" if torch.cuda.device_count() > 1 else {"": "cuda:0"}).eval()
    input_device = next(model.get_input_embeddings().parameters()).device
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, padding_side="left", use_fast=False)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])

    def predict(batch):
        results = []
        for _, root, item in batch:
            inputs = processor.apply_chat_template(
                molmo_messages(item, root), tokenize=True,
                add_generation_prompt=True, return_tensors="pt",
                return_dict=True, padding=True)
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            # Official weights stay in their saved dtype (ER: float32), with
            # bfloat16 autocast. Avoid retaining another full set of cast weights
            # on a 24 GiB card; this does not alter the arithmetic precision.
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, cache_enabled=False):
                output = model.generate(**inputs, **generation)
            tokens = output[0, inputs["input_ids"].shape[1]:]
            results.append({"answer": processor.decode(tokens, skip_special_tokens=True),
                            **completion_metadata(tokens.tolist(), eos_ids, args.max_new_tokens)})
        return results

    checkpoint = Path(args.model)
    revision = getattr(model.config, "_commit_hash", None)
    if checkpoint.parent.name == "snapshots":
        revision = checkpoint.name
    return predict, {
        "model_path": str(checkpoint), "model_revision": revision,
        "dtype": str(model.dtype), "autocast_dtype": "torch.bfloat16",
        "autocast_cache_enabled": False, "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device_map": {key: str(value) for key, value in model.hf_device_map.items()},
        "resolved_generation": {
            "do_sample": model.generation_config.do_sample,
            "num_beams": model.generation_config.num_beams, **generation},
        "image_processor": {
            "implementation": type(processor.image_processor).__name__,
            "settings": processor.image_processor.to_dict(),
            "multi_image_mode": "separate_images",
            "system_message_handling": "prepended_verbatim_to_first_user_message",
        },
    }


if __name__ == "__main__":
    run(create_predictor, model="allenai/Molmo2-ER", backend="molmo2")
