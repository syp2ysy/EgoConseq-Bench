import sys
import types

import torch

from vln_baseline.model import create_model_and_processor


def test_create_model_and_processor_enables_mistral_regex_fix(monkeypatch):
    calls = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["model"] = {"args": args, "kwargs": kwargs}
            return cls()

        def gradient_checkpointing_enable(self):
            calls["gradient_checkpointing"] = True

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["processor"] = {"args": args, "kwargs": kwargs}
            return cls()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3VLForConditionalGeneration = FakeModel
    fake_transformers.Qwen3VLProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    create_model_and_processor(
        model_name="local-qwen3vl",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        gradient_checkpointing=True,
    )

    assert calls["processor"]["kwargs"]["fix_mistral_regex"] is True
