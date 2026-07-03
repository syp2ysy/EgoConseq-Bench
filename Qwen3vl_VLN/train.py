import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
try:
    from transformers import TrainerCallback
except ModuleNotFoundError:
    class TrainerCallback:  # type: ignore[no-redef]
        def on_init_end(self, args, state, control, **kwargs):
            return control

from vln_baseline.collator import QwenVLNCollator
from vln_baseline.dataset import StreamVLNImitationDataset
from vln_baseline.model import (
    attach_depth_head,
    count_parameters,
    create_model_and_processor,
    freeze_all_but_depth_head,
    freeze_vision_encoder,
    load_depth_head_state,
    save_depth_head_state,
    trainable_parameter_names,
    validate_trainable_budget,
    validate_trainable_prefixes,
)
from vln_baseline.losses import SILogLoss


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    raise TypeError(f"Expected list or comma-separated string, got {type(value)}")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def validate_zero3_config(path: str) -> None:
    with open(path, "r") as f:
        config = json.load(f)
    stage = config.get("zero_optimization", {}).get("stage")
    if stage != 3:
        raise ValueError(f"Only DeepSpeed ZeRO-3 is allowed for this run, got stage={stage} in {path}")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def assert_finite_scalar(value: torch.Tensor, name: str) -> None:
    if not is_finite_scalar(value):
        raise RuntimeError(f"Non-finite {name} encountered during training")


def is_finite_scalar(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach().float()).all().item())


def distributed_mean_scalar(value: torch.Tensor) -> float:
    scalar = value.detach().float().reshape(1)
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(scalar.cpu().item())

    backend = torch.distributed.get_backend()
    if backend == "nccl" and scalar.device.type != "cuda":
        scalar = scalar.to(torch.device("cuda", torch.cuda.current_device()))
    else:
        scalar = scalar.clone()
    torch.distributed.all_reduce(scalar, op=torch.distributed.ReduceOp.SUM)
    scalar /= torch.distributed.get_world_size()
    return float(scalar.cpu().item())


def get_spatial_merge_size(model: torch.nn.Module) -> int:
    base = unwrap_model(model)
    vision_config = getattr(getattr(base, "config", None), "vision_config", None)
    if vision_config is not None and hasattr(vision_config, "spatial_merge_size"):
        return int(vision_config.spatial_merge_size)
    visual = getattr(getattr(base, "model", base), "visual", None)
    if visual is not None and hasattr(visual, "spatial_merge_size"):
        return int(visual.spatial_merge_size)
    return 2


def get_language_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    base = unwrap_model(model)
    layers = base.model.language_model.layers
    if layer_idx < 0:
        layer_idx += len(layers)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"depth_tap_layer={layer_idx} outside language model layer range 0..{len(layers) - 1}")
    return layers[layer_idx]


def extract_visual_feature_maps(
    hidden_states: torch.Tensor,
    *,
    image_token_spans: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    features = []
    for image_idx, (batch_idx, start, end) in enumerate(image_token_spans.tolist()):
        t, h, w = [int(value) for value in image_grid_thw[image_idx].tolist()]
        grid_h = h // spatial_merge_size
        grid_w = w // spatial_merge_size
        expected = t * grid_h * grid_w
        tokens = hidden_states[batch_idx, start:end, :]
        if tokens.shape[0] != expected:
            raise ValueError(
                "image token span length does not match image_grid_thw during depth extraction: "
                f"image={image_idx}, span={tokens.shape[0]}, expected={expected}, grid={(t, h, w)}"
            )
        if t != 1:
            raise ValueError(f"depth supervision expects image grids with t=1, got t={t} for image {image_idx}")
        features.append(tokens.view(grid_h, grid_w, hidden_states.shape[-1]).permute(2, 0, 1))
    return torch.stack(features, dim=0)


class DepthAuxTrainerMixin:
    depth_enable: bool = False
    depth_stage: int = 2
    depth_weight: float = 0.0
    depth_weight_schedule: str = "constant"
    depth_weight_final: float = 0.0
    depth_weight_hold_ratio: float = 0.0
    depth_warmup_steps: int = 0
    depth_tap_layer: int = -1
    depth_spatial_merge_size: int = 2
    depth_supervise_frames: str = "all"
    resume_model_only: bool = False
    depth_loss_fn: SILogLoss

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is not None and getattr(self, "resume_model_only", False):
            if getattr(self.args, "process_index", 0) == 0:
                print(f"Skipping optimizer/scheduler state load for model-only resume from {checkpoint}")
            return
        return super()._load_optimizer_and_scheduler(checkpoint)

    def _depth_weight_for_step(self) -> float:
        step = float(getattr(self.state, "global_step", 0) + 1)
        weight = float(self.depth_weight)
        schedule = str(getattr(self, "depth_weight_schedule", "constant") or "constant").lower()
        if schedule in {"constant", "none"}:
            pass
        elif schedule == "plateau_cosine_decay":
            max_steps = float(
                getattr(self.state, "max_steps", 0)
                or getattr(getattr(self, "args", None), "max_steps", 0)
                or 0
            )
            if max_steps > 0:
                progress = min(1.0, max(0.0, step / max_steps))
                hold_ratio = min(1.0, max(0.0, float(getattr(self, "depth_weight_hold_ratio", 0.0))))
                final_weight = float(getattr(self, "depth_weight_final", 0.0))
                if progress > hold_ratio and hold_ratio < 1.0:
                    decay_progress = min(1.0, max(0.0, (progress - hold_ratio) / (1.0 - hold_ratio)))
                    weight = final_weight + (weight - final_weight) * 0.5 * (
                        1.0 + math.cos(math.pi * decay_progress)
                    )
        else:
            raise ValueError(f"Unsupported depth_weight_schedule={schedule!r}")
        if self.depth_warmup_steps > 0:
            weight *= min(1.0, step / float(self.depth_warmup_steps))
        return weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not getattr(self, "depth_enable", False) or "depth_gt" not in inputs:
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

        depth_gt = inputs.pop("depth_gt")
        depth_mask = inputs.pop("depth_mask")
        image_token_spans = inputs.pop("image_token_spans")
        inputs.pop("image_batch_indices", None)
        current_image_indices = inputs.pop("current_image_indices", None)
        image_grid_thw = inputs["image_grid_thw"]
        depth_stage = int(getattr(self, "depth_stage", 2))

        captured: Dict[str, torch.Tensor] = {}

        def capture_hidden(_module, _module_inputs, module_output):
            captured["hidden"] = module_output[0] if isinstance(module_output, (tuple, list)) else module_output

        handle = get_language_layer(model, self.depth_tap_layer).register_forward_hook(capture_hidden)
        try:
            if depth_stage == 1:
                model_inputs = dict(inputs)
                model_inputs.pop("labels", None)
                with torch.no_grad():
                    outputs = model(**model_inputs)
            else:
                outputs = model(**inputs)
        finally:
            handle.remove()

        if "hidden" not in captured:
            raise RuntimeError(f"Depth tap layer {self.depth_tap_layer} hook did not capture hidden states")

        base = unwrap_model(model)
        features = extract_visual_feature_maps(
            captured["hidden"],
            image_token_spans=image_token_spans,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=self.depth_spatial_merge_size,
        )
        supervise_frames = getattr(self, "depth_supervise_frames", "all")
        if supervise_frames == "current":
            if current_image_indices is None:
                raise ValueError("depth_supervise_frames='current' requires current_image_indices in the batch")
            feature_indices = current_image_indices.to(device=features.device, dtype=torch.long)
            depth_indices = current_image_indices.to(device=depth_gt.device, dtype=torch.long)
            features = features.index_select(0, feature_indices)
            depth_gt = depth_gt.index_select(0, depth_indices)
            depth_mask = depth_mask.index_select(0, depth_indices.to(device=depth_mask.device))
            expected = int(inputs["input_ids"].shape[0])
            if features.shape[0] != expected:
                raise ValueError(
                    "current-only depth supervision must select exactly one image per sample: "
                    f"selected={features.shape[0]}, batch={expected}"
                )
        elif supervise_frames not in {"all", "all_frames"}:
            raise ValueError(f"Unsupported depth_supervise_frames={supervise_frames!r}")
        if depth_stage == 1:
            features = features.detach()
        depth_pred = base.depth_head(features)
        depth_loss = self.depth_loss_fn(depth_pred, depth_gt, depth_mask)
        if depth_stage == 1:
            assert_finite_scalar(depth_loss, "loss_depth")
            loss = depth_loss
            assert_finite_scalar(loss, "loss_total")
            if hasattr(self, "log"):
                loss_depth_log = distributed_mean_scalar(depth_loss)
                self.log(
                    {
                        "loss_depth": loss_depth_log,
                        "depth_stage": 1,
                    }
                )
            return (loss, outputs) if return_outputs else loss

        action_loss = outputs.loss
        assert_finite_scalar(action_loss, "loss_action")
        depth_loss_skipped = 0.0
        if not is_finite_scalar(depth_loss):
            depth_loss = action_loss.detach().new_zeros(())
            depth_loss_skipped = 1.0
        depth_weight = self._depth_weight_for_step()
        loss = action_loss + depth_weight * depth_loss
        assert_finite_scalar(loss, "loss_total")
        if hasattr(self, "log"):
            loss_action_log = distributed_mean_scalar(action_loss)
            loss_depth_log = distributed_mean_scalar(depth_loss)
            self.log(
                {
                    "loss_action": loss_action_log,
                    "loss_depth": loss_depth_log,
                    "depth_loss_skipped": depth_loss_skipped,
                    "depth_weight": float(depth_weight),
                    "depth_stage": 2,
                }
            )
        return (loss, outputs) if return_outputs else loss


class DepthPlateauStopCallback(TrainerCallback):
    def __init__(
        self,
        *,
        ema_beta: float = 0.9,
        patience: int = 50,
        min_delta: float = 0.005,
        min_steps: int = 100,
    ):
        self.ema_beta = ema_beta
        self.patience = patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.ema: Optional[float] = None
        self.best: Optional[float] = None
        self.bad_logs = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss_depth" not in logs:
            return control
        value = float(logs["loss_depth"])
        self.ema = value if self.ema is None else self.ema_beta * self.ema + (1.0 - self.ema_beta) * value
        if self.best is None or self.ema < self.best - self.min_delta:
            self.best = self.ema
            self.bad_logs = 0
        elif state.global_step < self.min_steps:
            self.bad_logs = 0
        else:
            self.bad_logs += 1
        if state.global_step >= self.min_steps and self.bad_logs >= self.patience:
            control.should_training_stop = True
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL StreamVLN imitation training")
    parser.add_argument("--config", type=str, default="config/train_streamvln.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_model_only", action="store_true")
    parser.add_argument("--dry_run_batch_only", action="store_true")
    parser.add_argument("--depth_weight", type=float, default=None)
    parser.add_argument("--depth_weight_schedule", type=str, default=None)
    parser.add_argument("--depth_weight_final", type=float, default=None)
    parser.add_argument("--depth_weight_hold_ratio", type=float, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def configure_model_for_training(model: torch.nn.Module, cfg: Dict[str, Any]) -> Optional[List[str]]:
    depth_stage = int(cfg.get("depth_stage", 2 if cfg.get("depth_enable", False) else 0))
    if depth_stage == 1:
        freeze_all_but_depth_head(model)
        return list(cfg.get("allowed_trainable_prefixes", ["depth_head."]))

    init_depth_head_from = cfg.get("init_depth_head_from")
    if depth_stage == 2 and init_depth_head_from:
        loaded_keys = load_depth_head_state(model, str(init_depth_head_from))
        print(f"Loaded depth head weights from {init_depth_head_from}: {len(loaded_keys)} tensors")

    if cfg.get("freeze_vision_encoder", True):
        freeze_vision_encoder(model, train_visual_merger=bool(cfg.get("train_visual_merger", True)))
    return cfg.get("allowed_trainable_prefixes")


def main() -> None:
    from transformers import Trainer, TrainingArguments, set_seed

    args = parse_args()
    cfg = load_config(args.config)
    if args.model_name is not None:
        cfg["model_name"] = args.model_name
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.depth_weight is not None:
        cfg["depth_weight"] = args.depth_weight
    if args.depth_weight_schedule is not None:
        cfg["depth_weight_schedule"] = args.depth_weight_schedule
    if args.depth_weight_final is not None:
        cfg["depth_weight_final"] = args.depth_weight_final
    if args.depth_weight_hold_ratio is not None:
        cfg["depth_weight_hold_ratio"] = args.depth_weight_hold_ratio

    set_seed(int(cfg.get("seed", 41)))

    annotation_path = _as_list(cfg["annotation_path"])
    image_root = _as_list(cfg["image_root"])
    image_size = tuple(cfg.get("image_size", [308, 252]))

    dataset = StreamVLNImitationDataset(
        annotation_path=annotation_path,
        image_root=image_root,
        history_images=int(cfg.get("history_images", 8)),
        max_action_segments=int(cfg.get("max_action_segments", 3)),
        stop_upsample_ratio=int(cfg.get("stop_upsample_ratio", 5)),
        image_size=image_size,
        chunking_strategy=cfg.get("chunking_strategy", "greedy"),
        chunking_seed=int(cfg.get("chunking_seed", cfg.get("seed", 41))),
        chunking_merge_probability=float(cfg.get("chunking_merge_probability", 0.7)),
        max_samples=cfg.get("max_samples"),
        depth_root=cfg.get("depth_root"),
        depth_enable=bool(cfg.get("depth_enable", False)),
    )

    if args.dry_run_batch_only:
        from transformers import Qwen3VLProcessor

        processor = Qwen3VLProcessor.from_pretrained(
            cfg.get("model_name", "Qwen/Qwen3-VL-4B-Instruct"),
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        collator = QwenVLNCollator(
            processor,
            depth_enable=bool(cfg.get("depth_enable", False)),
            depth_max_meters=float(cfg.get("depth_max_meters", 10.0)),
            spatial_merge_size=int(cfg.get("spatial_merge_size", 2)),
            image_size=image_size,
            depth_resize_mode=cfg.get("depth_resize_mode", "nearest"),
        )
        batch = collator([dataset[0]])
        print({key: tuple(value.shape) for key, value in batch.items() if hasattr(value, "shape")})
        return

    model, processor = create_model_and_processor(
        model_name=cfg.get("model_name", "Qwen/Qwen3-VL-4B-Instruct"),
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )
    depth_enable = bool(cfg.get("depth_enable", False))
    if depth_enable:
        hidden_size = int(model.config.text_config.hidden_size)
        attach_depth_head(
            model,
            input_dim=hidden_size,
            hidden_dim=int(cfg.get("depth_head_hidden_dim", 512)),
            max_depth=float(cfg.get("depth_max_meters", 10.0)),
            patch_size=int(cfg.get("depth_head_patch", 1)),
        )
    allowed_trainable_prefixes = configure_model_for_training(model, cfg)
    total, trainable = count_parameters(model)
    print(f"Parameters: total={total:,}, trainable={trainable:,}, trainable%={100 * trainable / total:.2f}")
    all_trainable_names = trainable_parameter_names(model)
    names = all_trainable_names[: int(cfg.get("trainable_name_log_limit", 80))]
    print(f"Trainable parameter names ({len(names)} shown): {names}")
    if allowed_trainable_prefixes:
        validate_trainable_prefixes(all_trainable_names, list(allowed_trainable_prefixes))
    validate_trainable_budget(
        total_parameters=total,
        trainable_parameters=trainable,
        max_trainable_parameters=cfg.get("max_trainable_parameters"),
        max_trainable_fraction=cfg.get("max_trainable_fraction"),
    )

    spatial_merge_size = get_spatial_merge_size(model)
    collator = QwenVLNCollator(
        processor,
        depth_enable=depth_enable,
        depth_max_meters=float(cfg.get("depth_max_meters", 10.0)),
        spatial_merge_size=spatial_merge_size,
        image_size=image_size,
        depth_resize_mode=cfg.get("depth_resize_mode", "nearest"),
    )

    output_dir = cfg.get("output_dir", "outputs/qwen3vl_streamvln_imitation")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    deepspeed_config = cfg.get("deepspeed", "scripts/zero3.json")
    validate_zero3_config(deepspeed_config)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=bool(cfg.get("bf16", True)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(cfg.get("dataloader_pin_memory", True)),
        report_to=cfg.get("report_to", "none"),
        logging_nan_inf_filter=bool(cfg.get("logging_nan_inf_filter", False)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        remove_unused_columns=False,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        deepspeed=deepspeed_config,
    )

    resume_model_only = bool(args.resume_model_only or cfg.get("resume_model_only", False))
    TrainerClass = type("DepthAuxTrainer", (DepthAuxTrainerMixin, Trainer), {}) if depth_enable or resume_model_only else Trainer
    callbacks = []
    depth_stage = int(cfg.get("depth_stage", 2 if depth_enable else 0))
    if depth_enable and depth_stage == 1 and bool(cfg.get("depth_plateau_enable", True)):
        callbacks.append(
            DepthPlateauStopCallback(
                ema_beta=float(cfg.get("depth_plateau_ema_beta", 0.9)),
                patience=int(cfg.get("depth_plateau_patience", 50)),
                min_delta=float(cfg.get("depth_plateau_min_delta", 0.005)),
                min_steps=int(cfg.get("depth_plateau_min_steps", 100)),
            )
        )

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    if depth_enable:
        trainer.depth_enable = True
        trainer.depth_stage = depth_stage
        trainer.depth_weight = float(cfg.get("depth_weight", 0.5))
        trainer.depth_weight_schedule = cfg.get("depth_weight_schedule", "constant")
        trainer.depth_weight_final = float(cfg.get("depth_weight_final", 0.0))
        trainer.depth_weight_hold_ratio = float(cfg.get("depth_weight_hold_ratio", 0.0))
        trainer.depth_warmup_steps = int(cfg.get("depth_warmup_steps", 0))
        trainer.depth_tap_layer = int(cfg.get("depth_tap_layer", -1))
        trainer.depth_spatial_merge_size = spatial_merge_size
        trainer.depth_supervise_frames = cfg.get("depth_supervise_frames", "all")
        trainer.depth_loss_fn = SILogLoss(
            alpha=float(cfg.get("depth_silog_alpha", 10.0)),
            variance_focus=float(cfg.get("depth_silog_variance_focus", 0.85)),
            min_valid_pixels=int(cfg.get("depth_min_valid_pixels", 16)),
        )
    if resume_model_only:
        trainer.resume_model_only = True
    resume_from_checkpoint = args.resume_from_checkpoint or cfg.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if depth_enable and depth_stage == 1:
        saved_path = save_depth_head_state(trainer.model, output_dir)
        if getattr(trainer.args, "process_index", 0) == 0:
            print(f"Saved Stage1 depth head to {saved_path}")
            processor.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
        if getattr(trainer.args, "process_index", 0) == 0:
            processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
