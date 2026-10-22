#!/usr/bin/env python3
"""RxBrain VQA using official greedy decoding and the saved QA conversation.

Runtime: qwen3vl plus vendor/rxbrain_transformers/src, vendor/rxbrain_deps,
and vendor/rxbrain on PYTHONPATH. The Transformers fork is pinned by RxBrain.
"""

from common import build_messages, completion_metadata, run


class ConversationProcessor:
    """Retain saved system/user turns inside the official answer() decoder."""

    def __init__(self, processor, messages):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        # The native template requires text strings for system turns, while
        # user turns retain the interleaved image/text blocks from the benchmark.
        self.messages = [
            {**message, "content": "".join(block["text"] for block in message["content"])}
            if message["role"] == "system" else message
            for message in messages
        ]

    def apply_chat_template(self, _official_messages, **kwargs):
        tokenize = kwargs.pop("tokenize", False)
        return_dict = kwargs.pop("return_dict", False)
        prompt = self.processor.apply_chat_template(
            self.messages + [{"role": "assistant", "content": ""}], tokenize=False, **kwargs)
        if not tokenize:
            return prompt
        # The pinned processor's tokenize=True collector only accepts block
        # lists, whereas its system template requires strings. Keep both native
        # stages, passing the original image order to the processor explicitly.
        images = [block["image"] for message in self.messages if message["role"] == "user"
                  for block in message["content"] if block["type"] == "image"]
        token_kwargs = {}
        if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
            token_kwargs["add_special_tokens"] = False
        inputs = self.processor(text=prompt, images=[images], **token_kwargs)
        return inputs if return_dict else inputs["input_ids"]


def create_predictor(args):
    import torch
    from transformers.models.hunyuan_vl_mot import HunYuanVLMoTProcessor
    from model import UnifiedMoTForConditionalGeneration, maybe_init_generation_path
    from inference.vqa import answer

    if args.thinking != "auto":
        raise ValueError("RxBrain official answer() has no enable_thinking switch")
    generation = getattr(args, "official_settings", {}).get("generation", {})
    if generation.get("do_sample", False):
        raise ValueError("RxBrain official answer() uses greedy decoding")
    torch.manual_seed(generation.get("seed", args.seed))
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    processor = HunYuanVLMoTProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = UnifiedMoTForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    maybe_init_generation_path(model, model_load_path=args.model)
    model.to(device).eval()

    def predict(batch):
        results = []
        for _, root, item in batch:
            messages = build_messages(item, root)
            adapted_processor = ConversationProcessor(processor, messages)
            paths = [str((root / path).resolve()) for path in item["images"]]
            question = "\n".join(message["content"] for message in item["messages"]
                                 if message["role"] == "user")
            completion = {}
            original_decode = processor.tokenizer.decode

            def decode(token_ids, *decode_args, **decode_kwargs):
                # Official answer() omits EOS and otherwise runs to the budget.
                completion.update(completion_metadata(
                    token_ids, model.config.eos_token_id, args.max_new_tokens, eos_omitted=True))
                return original_decode(token_ids, *decode_args, **decode_kwargs)

            processor.tokenizer.decode = decode
            try:
                text = answer(model, adapted_processor, paths, question, device, dtype,
                              args.max_new_tokens)
            finally:
                processor.tokenizer.decode = original_decode
            if not completion:
                raise RuntimeError("No RxBrain completion tokens were recorded")
            results.append({"answer": text, **completion})
        return results

    return predict, {
        "model_revision": getattr(model.config, "_commit_hash", None),
        "model_path": args.model,
        "dtype": "bfloat16",
        "image_processor": processor.image_processor.to_dict(),
        "official_source": "https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0/tree/7d35a9c70634e11dbfd18d94e95635150a035b33",
        "transformers_revision": "9293856c419762ebf98fbe2bd9440f9ce7069f1a",
    }


if __name__ == "__main__":
    run(create_predictor, model="tencent/Hy-Embodied-RxBrain-1.0", backend="rxbrain")
