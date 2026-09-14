"""Fine-tune Qwen3-VL on single-turn RGB navigation conversations."""

import argparse
import json
from typing import Any, Dict, List

import torch
import yaml

from vln_baseline.collator import QwenVLNCollator
from vln_baseline.dataset import StreamVLNImitationDataset
from vln_baseline.model import (
    count_parameters,
    create_model_and_processor,
    freeze_vision_encoder,
)


CONFIG_KEYS = {
    "model_name", "output_dir", "annotation_path", "image_root",
    "history_images", "max_action_segments", "stop_upsample_ratio", "image_size",
    "chunking_strategy", "chunking_seed", "chunking_merge_probability", "max_samples",
    "seed", "num_train_epochs", "max_steps", "learning_rate", "lr_scheduler_type",
    "per_device_train_batch_size", "gradient_accumulation_steps", "bf16",
    "attn_implementation", "gradient_checkpointing", "dataloader_num_workers",
    "dataloader_pin_memory", "logging_steps", "save_strategy", "save_steps",
    "eval_strategy", "eval_steps",
    "save_total_limit", "report_to", "deepspeed", "freeze_vision_encoder",
    "train_visual_merger", "resume_from_checkpoint", "max_grad_norm",
}


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    raise TypeError(f"Expected list or comma-separated string, got {type(value)}")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unsupported training config options: {sorted(unknown)}")
    return config


def validate_zero2_config(path: str) -> None:
    with open(path, "r") as f:
        config = json.load(f)
    stage = config.get("zero_optimization", {}).get("stage")
    if stage != 2:
        raise ValueError(f"Expected DeepSpeed ZeRO-2, got stage={stage} in {path}")


def configure_model_for_training(model: torch.nn.Module, cfg: Dict[str, Any]) -> None:
    if cfg.get("freeze_vision_encoder", True):
        freeze_vision_encoder(model, train_visual_merger=bool(cfg.get("train_visual_merger", True)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_r2r.yaml")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dry_run_batch_only", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    for key in ("model_name", "output_dir", "max_samples", "resume_from_checkpoint"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    from transformers import Qwen3VLProcessor, Trainer, TrainingArguments, set_seed

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    model_name = cfg.get("model_name", "Qwen/Qwen3-VL-4B-Instruct")
    dataset = StreamVLNImitationDataset(
        annotation_path=_as_list(cfg["annotation_path"]),
        image_root=_as_list(cfg["image_root"]),
        history_images=int(cfg.get("history_images", 8)),
        max_action_segments=int(cfg.get("max_action_segments", 3)),
        stop_upsample_ratio=int(cfg.get("stop_upsample_ratio", 2)),
        image_size=tuple(cfg.get("image_size", [308, 252])),
        chunking_strategy=cfg.get("chunking_strategy", "navida_random"),
        chunking_seed=int(cfg.get("chunking_seed", seed)),
        chunking_merge_probability=float(cfg.get("chunking_merge_probability", 0.7)),
        max_samples=cfg.get("max_samples"),
    )
    print(f"Training samples: {len(dataset):,} from {len(dataset.episodes):,} trajectories")

    if args.dry_run_batch_only:
        processor = Qwen3VLProcessor.from_pretrained(model_name, fix_mistral_regex=True)
        batch = QwenVLNCollator(processor)([dataset[0]])
        print({key: tuple(value.shape) for key, value in batch.items()})
        return

    output_dir = cfg.get("output_dir", "outputs/qwen3vl_r2r")
    deepspeed_config = cfg.get("deepspeed")
    if deepspeed_config:
        validate_zero2_config(deepspeed_config)
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=int(cfg.get("max_steps", -1)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=bool(cfg.get("bf16", True)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        eval_strategy=cfg.get("eval_strategy", "no"),
        eval_steps=int(cfg.get("eval_steps", 100)),
        save_strategy=cfg.get("save_strategy", "no"),
        save_steps=int(cfg.get("save_steps", 12000)),
        save_total_limit=cfg.get("save_total_limit"),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(cfg.get("dataloader_pin_memory", True)),
        report_to=cfg.get("report_to", "tensorboard"),
        logging_nan_inf_filter=False,
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        remove_unused_columns=False,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        deepspeed=deepspeed_config,
    )
    model, processor = create_model_and_processor(
        model_name=model_name,
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )
    configure_model_for_training(model, cfg)
    total, trainable = count_parameters(model)
    print(f"Parameters: total={total:,}, trainable={trainable:,}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=QwenVLNCollator(processor),
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
