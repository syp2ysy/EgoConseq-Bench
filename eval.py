import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from vln_baseline.actions import parse_and_expand_actions, sanitize_metric
from vln_baseline.dataset import build_navida_messages, uniform_sample_indices
from vln_baseline.model import create_model_and_processor


class SingleTurnNavigator:
    def __init__(
        self,
        model_path: str,
        history_images: int = 8,
        execute_action_segments: int = 2,
        max_action_history: int = 200,
        generation_max_new_tokens: int = 64,
        decode_strategy: str = "sample",
        temperature: float = 0.2,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        image_size=(308, 252),
        seed: int = 41,
    ):
        self.model_path = model_path
        self.history_images = history_images
        self.execute_action_segments = execute_action_segments
        self.max_action_history = max_action_history
        self.generation_max_new_tokens = generation_max_new_tokens
        self.decode_strategy = decode_strategy
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.image_size = tuple(image_size)
        self.rng = random.Random(seed)
        self.model, self.processor = create_model_and_processor(
            model_name=model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            gradient_checkpointing=False,
        )
        self.model.cuda()
        self.model.eval()
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = 501760
        self.reset()

    def reset(self):
        self.rgb_list: List[Image.Image] = []
        self.pending_actions: List[int] = []
        self.used_fallback = False
        self.last_response = ""
        self.last_step: Dict[str, Any] = {}
        self.last_history_indices: List[int] = []

    def _resize_rgb(self, rgb: np.ndarray) -> Image.Image:
        return Image.fromarray(rgb.astype("uint8")).convert("RGB").resize(self.image_size)

    def _append_observation(self, rgb: np.ndarray) -> Image.Image:
        current_image = self._resize_rgb(rgb)
        self.rgb_list.append(current_image)
        if self.max_action_history > 0 and len(self.rgb_list) > self.max_action_history:
            self.rgb_list = self.rgb_list[-self.max_action_history :]
        return current_image

    def _history_for_prompt(self) -> List[Image.Image]:
        if len(self.rgb_list) <= 1:
            self.last_history_indices = [0]
            return [self.rgb_list[-1]]
        history = self.rgb_list[:-1]
        indices = uniform_sample_indices(len(history), self.history_images)
        self.last_history_indices = indices
        return [history[i] for i in indices]

    def _generation_kwargs(self) -> Dict[str, Any]:
        decode_strategy = getattr(self, "decode_strategy", "sample")
        kwargs = {
            "do_sample": decode_strategy == "sample",
            "max_new_tokens": self.generation_max_new_tokens,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if decode_strategy == "sample":
            kwargs.update(
                {
                    "temperature": getattr(self, "temperature", 0.2),
                    "top_p": getattr(self, "top_p", 1.0),
                    "repetition_penalty": getattr(self, "repetition_penalty", 1.05),
                }
            )
        return kwargs

    def act(self, rgb: np.ndarray, instruction: str) -> int:
        current_image = self._append_observation(rgb)
        if self.pending_actions:
            action_id = self.pending_actions.pop(0)
            self.last_step = {
                "source": "pending",
                "response": self.last_response,
                "action": action_id,
                "pending_after": list(self.pending_actions),
                "history_indices": list(self.last_history_indices),
            }
            return action_id

        messages = build_navida_messages(
            instruction=instruction,
            history_images=self._history_for_prompt(),
            current_image=current_image,
            answer=None,
        )
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = []
        for turn in messages:
            for part in turn["content"]:
                if part["type"] == "image":
                    images.append(part["image"])
        inputs = self.processor(text=[text], images=[images], return_tensors="pt", padding=True).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self._generation_kwargs())
        input_len = inputs["input_ids"].shape[1]
        response = self.processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0].strip()
        self.last_response = response
        actions, used_fallback = parse_and_expand_actions(response, self.execute_action_segments, self.rng)
        self.used_fallback = used_fallback
        self.pending_actions.extend(actions)
        action_id = self.pending_actions.pop(0)
        self.last_step = {
            "source": "model",
            "response": response,
            "expanded_actions": actions,
            "action": action_id,
            "pending_after": list(self.pending_actions),
            "used_fallback": used_fallback,
            "history_indices": list(self.last_history_indices),
        }
        return action_id


def _episode_instruction(episode: Any) -> str:
    instruction = episode.instruction
    if hasattr(instruction, "instruction_text"):
        return instruction.instruction_text
    return instruction["instruction_text"]


def _episode_sort_key(episode: Any):
    episode_id = str(getattr(episode, "episode_id", ""))
    try:
        episode_id_key = (0, int(episode_id))
    except ValueError:
        episode_id_key = (1, episode_id)
    return (episode_id_key, str(getattr(episode, "scene_id", "")), _episode_instruction(episode))


def _deterministic_split_dataset(dataset: Any, split_num: int, split_id: int):
    if split_num <= 1:
        return dataset
    if split_id < 0 or split_id >= split_num:
        raise ValueError(f"split_id must be in [0, {split_num}), got {split_id}")

    episodes = sorted(dataset.episodes, key=_episode_sort_key)
    dataset.episodes = episodes[split_id::split_num]
    return dataset


def _step_env(env: Any, action_id: int):
    return env.step(action_id)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    import habitat
    from habitat_baselines.config.default import get_config
    import vln_baseline.habitat_extensions  # noqa: F401

    config = get_config(args.config)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_id

    dataset = habitat.datasets.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    if args.split_num > 1:
        dataset = _deterministic_split_dataset(dataset, args.split_num, args.split_id)
    env = habitat.Env(config.habitat, dataset)
    navigator = SingleTurnNavigator(
        model_path=args.model_path,
        history_images=args.history_images,
        execute_action_segments=args.execute_action_segments,
        max_action_history=args.max_action_history,
        generation_max_new_tokens=args.generation_max_new_tokens,
        decode_strategy=args.decode_strategy,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    output_dir = Path(args.output)
    log_dir = output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    metrics: List[Dict[str, Any]] = []
    max_episodes = args.max_episodes or len(env.episodes)

    for _ in range(max_episodes):
        observations = env.reset()
        navigator.reset()
        episode = env.current_episode
        instruction = _episode_instruction(episode)
        episode_id = str(episode.episode_id)
        step_id = 0
        unchanged_distance_steps = 0
        last_distance = None
        fallback_count = 0
        start_time = time.time()
        trace_steps: List[Dict[str, Any]] = []

        while not env.episode_over:
            info = env.get_metrics()
            distance = info.get("distance_to_goal")
            if distance == last_distance:
                unchanged_distance_steps += 1
            else:
                unchanged_distance_steps = 0
                last_distance = distance

            if unchanged_distance_steps > args.early_stop_rotation or step_id > args.early_stop_steps:
                action_id = 0
                if args.save_traces:
                    trace_steps.append(
                        {
                            "step": step_id,
                            "distance_to_goal": sanitize_metric(float(distance)) if distance is not None else None,
                            "source": "early_stop",
                            "action": action_id,
                        }
                    )
            else:
                action_id = navigator.act(observations["rgb"], instruction)
                fallback_count += int(navigator.used_fallback)
                navigator.used_fallback = False
                if args.save_traces:
                    trace_steps.append(
                        {
                            "step": step_id,
                            "distance_to_goal": sanitize_metric(float(distance)) if distance is not None else None,
                            **navigator.last_step,
                        }
                    )

            observations = _step_env(env, action_id)
            step_id += 1

        info = env.get_metrics()
        result = {
            "id": episode_id,
            "scene_id": str(getattr(episode, "scene_id", "")),
            "split_id": args.split_id,
            "split_num": args.split_num,
            "instruction": instruction,
            "success": sanitize_metric(float(info.get("success", 0.0))),
            "spl": sanitize_metric(float(info.get("spl", 0.0))),
            "distance_to_goal": sanitize_metric(float(info.get("distance_to_goal", 0.0))),
            "oracle_success": sanitize_metric(float(info.get("oracle_success", 0.0))),
            "oracle_navigation_error": sanitize_metric(float(info.get("oracle_navigation_error", 0.0))),
            "path_length": sanitize_metric(float(info.get("path_length", 0.0))),
            "num_steps": step_id,
            "fallback_count": fallback_count,
            "elapsed_seconds": time.time() - start_time,
        }
        with open(log_dir / f"stats_{episode_id}.json", "w") as f:
            json.dump(result, f, indent=2)
        if args.save_traces:
            with open(log_dir / f"trace_{episode_id}.json", "w") as f:
                json.dump(trace_steps, f, indent=2)
        metrics.append(result)

    env.close()
    final = {
        "split": args.split,
        "num_episodes": len(metrics),
        "success_rate": float(np.mean([m["success"] for m in metrics])) if metrics else 0.0,
        "oracle_success": float(np.mean([m["oracle_success"] for m in metrics])) if metrics else 0.0,
        "spl": float(np.mean([m["spl"] for m in metrics])) if metrics else 0.0,
        "navigation_error": float(np.mean([m["distance_to_goal"] for m in metrics])) if metrics else 0.0,
        "oracle_navigation_error": float(np.mean([m["oracle_navigation_error"] for m in metrics])) if metrics else 0.0,
    }
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2))
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL StreamVLN imitation evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", default="config/vln_r2r.yaml")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--output", default="results/qwen3vl_streamvln_imitation")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--history_images", type=int, default=8)
    parser.add_argument("--execute_action_segments", type=int, default=2)
    parser.add_argument("--max_action_history", type=int, default=200)
    parser.add_argument("--generation_max_new_tokens", type=int, default=64)
    parser.add_argument("--decode_strategy", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--save_traces", action="store_true")
    parser.add_argument("--split_num", type=int, default=1)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--early_stop_rotation", type=int, default=25)
    parser.add_argument("--early_stop_steps", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
