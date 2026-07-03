import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .depth_head import DepthReadoutHead


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


def attach_depth_head(
    model: Any,
    *,
    input_dim: int,
    hidden_dim: int = 512,
    max_depth: float = 10.0,
    patch_size: int = 1,
) -> None:
    model.depth_head = DepthReadoutHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_depth=max_depth,
        patch_size=patch_size,
    )


def _unwrap_model(model: Any) -> Any:
    while hasattr(model, "module"):
        model = model.module
    return model


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


def freeze_all_but_depth_head(model: Any) -> None:
    base = _unwrap_model(model)
    if not hasattr(base, "depth_head"):
        raise ValueError("depth_stage=1 requires model.depth_head to be attached before freezing")
    for param in base.parameters():
        param.requires_grad = False
    for param in base.depth_head.parameters():
        param.requires_grad = True


def _load_checkpoint_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint file did not contain a state dict: {path}")
    return state


def _load_sharded_state(directory: Path, index_name: str) -> Dict[str, torch.Tensor]:
    index_path = directory / index_name
    with index_path.open("r") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    shards = sorted({filename for key, filename in weight_map.items() if key.startswith("depth_head.")})
    state: Dict[str, torch.Tensor] = {}
    for shard in shards:
        shard_state = _load_checkpoint_file(directory / shard)
        state.update({key: value for key, value in shard_state.items() if key.startswith("depth_head.")})
    return state


def _load_model_state(path: str) -> Dict[str, torch.Tensor]:
    source = Path(path)
    if source.is_file():
        return _load_checkpoint_file(source)
    if not source.is_dir():
        raise FileNotFoundError(f"depth head checkpoint path does not exist: {path}")

    for name in ("depth_head.safetensors", "depth_head.pt", "model.safetensors", "pytorch_model.bin"):
        candidate = source / name
        if candidate.exists():
            return _load_checkpoint_file(candidate)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (source / name).exists():
            return _load_sharded_state(source, name)
    raise FileNotFoundError(f"could not find model.safetensors or pytorch_model.bin under {path}")


def load_depth_head_state(model: Any, src_dir: str) -> List[str]:
    base = _unwrap_model(model)
    if not hasattr(base, "depth_head"):
        raise ValueError("cannot load depth head before model.depth_head is attached")

    state = _load_model_state(src_dir)
    depth_state: Dict[str, torch.Tensor] = {}
    loaded_keys = []
    for key, value in state.items():
        stripped = None
        for prefix in ("depth_head.", "module.depth_head."):
            if key.startswith(prefix):
                stripped = key[len(prefix) :]
                break
        if stripped is None:
            continue
        depth_state[stripped] = value
        loaded_keys.append(key)
    if not depth_state:
        expected_keys = set(base.depth_head.state_dict().keys())
        if set(state.keys()) == expected_keys:
            depth_state = state
            loaded_keys = [f"depth_head.{key}" for key in state.keys()]
        else:
            raise ValueError(f"no depth_head.* weights found in {src_dir}")
    base.depth_head.load_state_dict(depth_state, strict=True)
    return sorted(loaded_keys)


def _clear_zero_active_sub_modules(params: List[torch.nn.Parameter]) -> None:
    for param in params:
        active = getattr(param, "ds_active_sub_modules", None)
        if active:
            active.clear()


def save_depth_head_state(model: Any, output_dir: str) -> Optional[Path]:
    base = _unwrap_model(model)
    if not hasattr(base, "depth_head"):
        raise ValueError("cannot save depth head before model.depth_head is attached")

    rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
    state: Dict[str, torch.Tensor] = {}
    try:
        import deepspeed
    except Exception:
        deepspeed = None

    named_parameters = list(base.depth_head.named_parameters())
    if deepspeed is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        # Stage1 saves immediately before process exit. DeepSpeed can still
        # report the manually-called depth_head as an active submodule after
        # early stopping, which makes GatheredParameters fail when it partitions
        # on context exit. Clear only the head params we are about to save.
        params = [param for _, param in named_parameters]
        _clear_zero_active_sub_modules(params)
        with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
            if rank == 0:
                for name, param in named_parameters:
                    state[f"depth_head.{name}"] = param.detach().cpu().clone()
    elif rank == 0:
        for name, param in named_parameters:
            state[f"depth_head.{name}"] = param.detach().cpu().clone()
    for name, buffer in base.depth_head.named_buffers():
        if rank == 0:
            state[f"depth_head.{name}"] = buffer.detach().cpu().clone()

    saved_path: Optional[Path] = None
    if rank == 0:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file

            saved_path = path / "depth_head.safetensors"
            save_file(state, str(saved_path))
        except Exception:
            saved_path = path / "depth_head.pt"
            torch.save(state, saved_path)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return saved_path


def count_parameters(model: Any) -> Tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def trainable_parameter_names(model: Any, max_names: Optional[int] = None) -> List[str]:
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if max_names is None:
        return names
    return names[:max_names]


def validate_trainable_prefixes(trainable_names: List[str], allowed_prefixes: List[str]) -> None:
    unexpected = [
        name for name in trainable_names if not any(name.startswith(prefix) for prefix in allowed_prefixes)
    ]
    if unexpected:
        raise ValueError(f"unexpected trainable parameters: {unexpected[:20]}")


def validate_trainable_budget(
    total_parameters: int,
    trainable_parameters: int,
    max_trainable_parameters: Optional[int] = None,
    max_trainable_fraction: Optional[float] = None,
) -> None:
    if max_trainable_parameters is not None and trainable_parameters > max_trainable_parameters:
        raise ValueError(
            "trainable parameter budget exceeded: "
            f"trainable={trainable_parameters:,}, max={max_trainable_parameters:,}"
        )

    if max_trainable_fraction is not None:
        fraction = trainable_parameters / total_parameters if total_parameters else 0.0
        if fraction > max_trainable_fraction:
            raise ValueError(
                "trainable parameter fraction exceeded: "
                f"trainable_fraction={fraction:.6f}, max={max_trainable_fraction:.6f}"
            )
