"""Load native Qwen3-VL and select parameters for navigation fine-tuning."""

from typing import Any, List, Optional, Tuple

import torch


def create_model_and_processor(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    gradient_checkpointing: bool = True,
) -> Tuple[Any, Any]:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    processor = Qwen3VLProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    return model, processor


def _vision_modules(model: Any) -> List[Any]:
    modules = []
    seen = set()
    containers = [model, getattr(model, "model", None)]
    for container in containers:
        if container is None:
            continue
        for name in ("visual", "vision_model", "visual_model"):
            module = getattr(container, name, None)
            if module is not None and id(module) not in seen:
                modules.append(module)
                seen.add(id(module))
    return modules


def freeze_vision_encoder(model: Any, train_visual_merger: bool = True) -> None:
    for module in _vision_modules(model):
        for param in module.parameters():
            param.requires_grad = False
        if train_visual_merger:
            merger = getattr(module, "merger", None)
            if merger is not None:
                for param in merger.parameters():
                    param.requires_grad = True


def count_parameters(model: Any) -> Tuple[int, int]:
    total = sum(getattr(param, "ds_numel", param.numel()) for param in model.parameters())
    trainable = sum(getattr(param, "ds_numel", param.numel()) for param in model.parameters() if param.requires_grad)
    return total, trainable


def trainable_parameter_names(model: Any, max_names: Optional[int] = None) -> List[str]:
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if max_names is None:
        return names
    return names[:max_names]
