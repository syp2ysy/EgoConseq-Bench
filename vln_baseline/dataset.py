import json
import os
import random
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from zipfile import ZipFile

from PIL import Image
from torch.utils.data import Dataset

from .actions import STOP, build_navida_random_action_chunks, format_action_segments


SYSTEM_PROMPT = "You are a helpful assistant."

USER_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{instruction}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)


def build_navida_messages(
    instruction: str,
    history_images: List[Image.Image],
    current_image: Image.Image,
    answer: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations",
        }
    ]
    user_content.extend({"type": "image", "image": image} for image in history_images)
    user_content.append({"type": "text", "text": "and an image of the current observation"})
    user_content.append({"type": "image", "image": current_image})
    prompt_tail = USER_PROMPT_TEMPLATE.format(instruction=instruction).split("current observation", 1)[1]
    user_content.append({"type": "text", "text": prompt_tail})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def uniform_sample_indices(length: int, max_count: int) -> List[int]:
    if length <= 0 or max_count <= 0:
        return []
    if length <= max_count:
        return list(range(length))
    if max_count == 1:
        return [length - 1]
    return [round(i * (length - 1) / (max_count - 1)) for i in range(max_count)]


class StreamVLNImitationDataset(Dataset):
    def __init__(
        self,
        annotation_path: Union[str, Sequence[str]],
        image_root: Union[str, Sequence[str]],
        history_images: int = 8,
        max_action_segments: int = 3,
        stop_upsample_ratio: int = 5,
        image_size: Tuple[int, int] = (308, 252),
        chunking_strategy: str = "greedy",
        chunking_seed: int = 41,
        chunking_merge_probability: float = 0.7,
        max_samples: Optional[int] = None,
    ):
        self.history_images = history_images
        if max_action_segments < 1:
            raise ValueError("max_action_segments must be positive")
        self.max_action_segments = max_action_segments
        self.stop_upsample_ratio = stop_upsample_ratio
        self.image_size = image_size
        self.chunking_strategy = chunking_strategy
        self.chunking_seed = chunking_seed
        self.chunking_merge_probability = chunking_merge_probability
        self.sample_answers: Dict[Tuple[int, int, int], str] = {}
        self.sample_consumed: Dict[Tuple[int, int, int], int] = {}

        annotation_paths = [annotation_path] if isinstance(annotation_path, str) else list(annotation_path)
        image_roots = [image_root] if isinstance(image_root, str) else list(image_root)
        if len(image_roots) == 1 and len(annotation_paths) > 1:
            image_roots = image_roots * len(annotation_paths)
        if len(annotation_paths) != len(image_roots):
            raise ValueError("annotation_path and image_root must have the same length")
        self.episodes: List[Dict[str, Any]] = []
        self.image_roots: List[str] = []
        for ann_path, root in zip(annotation_paths, image_roots):
            with open(ann_path, "r") as f:
                loaded = json.load(f)
            self.episodes.extend(loaded)
            self.image_roots.extend([root] * len(loaded))

        self.samples = self._build_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _build_samples(self) -> List[Tuple[int, int, int]]:
        samples: List[Tuple[int, int, int]] = []
        stop_samples: List[Tuple[int, int, int]] = []
        rng = random.Random(self.chunking_seed)
        for ep_idx, episode in enumerate(self.episodes):
            actions = self._episode_actions(episode)
            if not actions:
                continue
            instructions = episode.get("instructions", ["Navigate to the goal."])
            if not isinstance(instructions, list):
                instructions = [instructions]
            for ins_idx in range(len(instructions)):
                if self.chunking_strategy == "navida_random":
                    chunks = build_navida_random_action_chunks(
                        actions,
                        max_segments=self.max_action_segments,
                        merge_probability=self.chunking_merge_probability,
                        rng=rng,
                    )
                    for step_idx, answer, consumed in chunks:
                        sample = (ep_idx, ins_idx, step_idx)
                        samples.append(sample)
                        self.sample_answers[sample] = answer
                        self.sample_consumed[sample] = consumed
                        if STOP in actions[step_idx : step_idx + consumed]:
                            stop_samples.append(sample)
                    continue
                if self.chunking_strategy != "greedy":
                    raise ValueError(f"Unsupported chunking_strategy: {self.chunking_strategy}")

                step_idx = 0
                while step_idx < len(actions):
                    answer, consumed = format_action_segments(actions, step_idx, self.max_action_segments)
                    if consumed <= 0:
                        raise ValueError("Action segment builder consumed no actions")
                    sample = (ep_idx, ins_idx, step_idx)
                    samples.append(sample)
                    self.sample_answers[sample] = answer
                    self.sample_consumed[sample] = consumed
                    if STOP in actions[step_idx : step_idx + consumed]:
                        stop_samples.append(sample)
                    step_idx += consumed
        if self.stop_upsample_ratio > 1 and stop_samples:
            for _ in range(self.stop_upsample_ratio - 1):
                samples.extend(stop_samples)
        return samples

    def _episode_actions(self, episode: Dict[str, Any]) -> List[int]:
        raw_actions = list(episode["actions"])
        actions = raw_actions[1:] if raw_actions and raw_actions[0] == -1 else raw_actions
        if not actions or actions[-1] != STOP:
            actions = actions + [STOP]
        return actions

    def __len__(self) -> int:
        return len(self.samples)

    def _rgb_dir(self, episode: Dict[str, Any], ep_idx: int) -> Path:
        video_path = episode["video"]
        if video_path.startswith("images/"):
            video_path = video_path[len("images/") :]
        return Path(self.image_roots[ep_idx]) / video_path / "rgb"

    def _frame_paths(
        self, episode: Dict[str, Any], ep_idx: int, archive: Optional[ZipFile] = None
    ) -> List[str]:
        rgb_dir = self._rgb_dir(episode, ep_idx)
        if archive is not None:
            # Keep logical frame paths and the original lexicographic frame order.
            names = sorted(
                name[len("rgb/") :] for name in archive.namelist()
                if name.startswith("rgb/") and "/" not in name[len("rgb/") :] and not name.endswith("/")
            )
            return [str(rgb_dir / name) for name in names]
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        return [str(rgb_dir / name) for name in sorted(os.listdir(rgb_dir))]

    def _load_image(self, path: str, archive: Optional[ZipFile] = None) -> Image.Image:
        source = BytesIO(archive.read(f"rgb/{Path(path).name}")) if archive is not None else path
        with Image.open(source) as image:
            return image.convert("RGB").resize(self.image_size)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, ins_idx, step_idx = self.samples[idx]
        episode = self.episodes[ep_idx]
        instructions = episode.get("instructions", ["Navigate to the goal."])
        if not isinstance(instructions, list):
            instructions = [instructions]
        instruction = instructions[ins_idx]
        actions = self._episode_actions(episode)
        archive_path = Path(str(self._rgb_dir(episode, ep_idx).parent) + ".zip")
        # One handle per sample, closed before returning; safe across DataLoader workers.
        with (ZipFile(archive_path) if archive_path.is_file() else nullcontext()) as archive:
            frame_paths = self._frame_paths(episode, ep_idx, archive)
            current_frame_idx = min(step_idx, len(frame_paths) - 1)
            history_indices = uniform_sample_indices(current_frame_idx, self.history_images)
            history_frames = [frame_paths[i] for i in history_indices]
            current_frame = frame_paths[current_frame_idx]

            history_pil = [self._load_image(path, archive) for path in history_frames]
            current_pil = self._load_image(current_frame, archive)
        history_for_prompt = history_pil if history_pil else [current_pil]

        sample_key = (ep_idx, ins_idx, step_idx)
        answer = self.sample_answers.get(sample_key)
        if answer is None:
            answer, _ = format_action_segments(actions, step_idx, self.max_action_segments)
        messages = build_navida_messages(
            instruction,
            history_for_prompt,
            current_pil,
            answer,
        )
        sample = {
            "messages": messages,
            "instruction": instruction,
            "answer": answer,
            "history_frames": history_frames,
            "current_frame": current_frame,
        }
        return sample
