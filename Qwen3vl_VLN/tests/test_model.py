import sys
import types

import torch
import pytest

from vln_baseline.model import (
    create_model_and_processor,
    validate_trainable_budget,
    validate_trainable_prefixes,
)


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


def test_validate_trainable_budget_rejects_too_many_trainable_parameters():
    with pytest.raises(ValueError, match="trainable parameter budget"):
        validate_trainable_budget(
            total_parameters=4_000_000_000,
            trainable_parameters=10_000_000,
            max_trainable_parameters=5_000_000,
        )


def test_validate_trainable_budget_rejects_too_large_trainable_fraction():
    with pytest.raises(ValueError, match="trainable parameter fraction"):
        validate_trainable_budget(
            total_parameters=4_000_000_000,
            trainable_parameters=80_000_000,
            max_trainable_fraction=0.01,
        )


def test_validate_trainable_budget_accepts_lightweight_module():
    validate_trainable_budget(
        total_parameters=4_000_000_000,
        trainable_parameters=4_000_000,
        max_trainable_parameters=5_000_000,
        max_trainable_fraction=0.01,
    )


def test_validate_trainable_prefixes_accepts_allowed_visual_merger_prefix():
    validate_trainable_prefixes(
        ["visual.merger.mlp.0.weight", "visual.merger.mlp.2.bias"],
        allowed_prefixes=["visual.merger."],
    )
