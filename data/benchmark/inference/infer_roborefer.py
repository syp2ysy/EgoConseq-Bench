#!/usr/bin/env python3
"""RGB-only text QA through RoboRefer's official generate_content interface.

Runtime: qwen3vl plus vendor/roborefer_deps, vendor/roborefer, and
vendor/roborefer_s2wrapper on PYTHONPATH. The official loader uses FP16.
"""

from common import build_messages, record_generation, run


def create_predictor(args):
    import llava
    import torch
    from llava import conversation as clib
    from llava.media import Image

    if args.thinking != "auto":
        raise ValueError("RoboRefer has no official enable_thinking switch")
    model = llava.load(args.model, device_map="cuda:0").eval()
    clib.default_conversation = clib.conv_templates["auto"].copy()
    generation = model.default_generation_config
    settings = dict(getattr(args, "official_settings", {}).get("generation", {}))
    torch.manual_seed(settings.pop("seed", args.seed))
    settings.pop("max_tokens", None)
    for key, value in settings.items():
        setattr(generation, key, value)
    generation.max_new_tokens = args.max_new_tokens
    original_template = model.tokenizer.apply_chat_template

    def predict(batch):
        results = []
        for _, root, item in batch:
            messages = build_messages(item, root)
            systems = [message for message in messages if message["role"] == "system"]
            users = [message for message in messages if message["role"] == "user"]
            if len(users) != 1:
                raise ValueError("RoboRefer generate_content requires one saved user turn")
            prompt = [Image(block["image"]) if block["type"] == "image" else block["text"]
                      for block in users[0]["content"]]
            system_messages = [
                {"role": "system", "content": "".join(block["text"] for block in message["content"])}
                for message in systems
            ]

            def saved_system_template(conversation, **kwargs):
                return original_template(system_messages + conversation, **kwargs)

            # The official helper accepts one user turn. Its tokenizer hook lets
            # us retain the benchmark system message without changing user text.
            model.tokenizer.apply_chat_template = saved_system_template
            try:
                with record_generation(model) as completion:
                    answer = model.generate_content(prompt, generation_config=generation)
            finally:
                model.tokenizer.apply_chat_template = original_template
            results.append({"answer": answer, **completion})
        return results

    return predict, {
        "model_revision": getattr(model.config, "_commit_hash", None),
        "model_path": args.model,
        "dtype": "float16",
        "image_processor": {
            "implementation": "RoboRefer llava.generate_content / dynamic_process_images_and_prompt",
            "image_aspect_ratio": model.config.image_aspect_ratio,
            "enable_depth_input": False,
        },
        "official_source": "https://github.com/Zhoues/RoboRefer/tree/d97a995ad28376720a4c8beb64915c58ed16c844",
    }


if __name__ == "__main__":
    run(create_predictor, model="Zhoues/RoboRefer-2B-SFT", backend="roborefer")
