#!/usr/bin/env python3
"""Render ScaleVLN trajectories to StreamVLN-style RGB/depth/pose folders."""

import argparse
import copy
import gzip
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_r2r_depth_pose import (  # noqa: E402
    build_camera_params,
    coerce_hfov_for_habitat_config,
    depth_to_uint8,
)


DEFAULT_SCALEVLN_ROOT = Path("/home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data/ScaleVLN")
DEFAULT_ANNOTATION_PATH = DEFAULT_SCALEVLN_ROOT / "annotations.json"
DEFAULT_HABITAT_DATA_PATH = DEFAULT_SCALEVLN_ROOT / "scalevln_subset_150k.json.gz"
DEFAULT_IMAGE_ROOT = DEFAULT_SCALEVLN_ROOT / "images"
DEFAULT_SUBSET_ANNOTATION_PATH = DEFAULT_SCALEVLN_ROOT / "annotations_10k.json"
DEFAULT_CONFIG_PATH = Path("/home/zhangshan/syp/StreamVLN-R2RCE-Test/StreamVLN/config/vln_r2r.yaml")
DEFAULT_SCENES_DIR = Path("/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2")


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def load_habitat_dataset(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def output_dir_name(annotation: Dict[str, Any]) -> str:
    return Path(annotation["video"]).name


def select_annotations(
    annotations: Sequence[Dict[str, Any]],
    *,
    episode_ids: Optional[Sequence[int]],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    if episode_ids:
        by_id = {int(annotation["id"]): annotation for annotation in annotations}
        missing = [episode_id for episode_id in episode_ids if episode_id not in by_id]
        if missing:
            raise KeyError(f"Episode ids not found in annotations: {missing}")
        selected = [by_id[episode_id] for episode_id in episode_ids]
    else:
        selected = list(annotations)
    if limit is not None:
        selected = selected[:limit]
    return selected


def shard_annotations(
    annotations: Sequence[Dict[str, Any]], *, shard_index: int, num_shards: int
) -> List[Dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return list(annotations)[shard_index::num_shards]


def _expected_names(frame_count: int, suffix: str) -> set[str]:
    return {f"{idx:03d}.{suffix}" for idx in range(1, frame_count + 1)}


def episode_outputs_complete(output_dir: Path, frame_count: int) -> bool:
    expected_rgb = _expected_names(frame_count, "jpg")
    expected_depth = _expected_names(frame_count, "png")
    expected_pose = _expected_names(frame_count, "json")
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    pose_dir = output_dir / "pose"
    if not rgb_dir.is_dir() or not depth_dir.is_dir() or not pose_dir.is_dir():
        return False
    return (
        {path.name for path in rgb_dir.glob("*.jpg")} == expected_rgb
        and {path.name for path in depth_dir.glob("*.png")} == expected_depth
        and {path.name for path in pose_dir.glob("*.json")} == expected_pose
    )


def make_annotation_subset(annotations: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(list(annotations), f, indent=2)
        f.write("\n")


def _scene_candidate_relpaths(scene_id: str) -> List[Path]:
    scene_path = Path(scene_id)
    candidates = [scene_path]
    parts = scene_path.parts
    if parts and parts[0] == "hm3d":
        rest = Path(*parts[1:])
        candidates.extend(
            [
                Path("hm3d") / "train" / rest,
                Path("hm3d") / "val" / rest,
                Path("train") / rest,
                Path("val") / rest,
            ]
        )
    return list(dict.fromkeys(candidates))


def resolve_scene_id(scene_id: str, scenes_dir: Path) -> str:
    checked = []
    for candidate in _scene_candidate_relpaths(scene_id):
        scene_path = scenes_dir / candidate
        checked.append(str(scene_path))
        if scene_path.is_file():
            return candidate.as_posix()
    raise FileNotFoundError(
        "Could not resolve ScaleVLN scene file for "
        f"{scene_id!r} under {scenes_dir}. Checked: {checked[:8]}"
    )


def _episode_id_map(episodes: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(episode["episode_id"]): episode for episode in episodes}


def write_habitat_subset(
    raw_dataset: Dict[str, Any],
    selected_annotations: Sequence[Dict[str, Any]],
    *,
    scenes_dir: Path,
    output_path: Path,
) -> None:
    raw_by_id = _episode_id_map(raw_dataset["episodes"])
    subset_episodes = []
    for annotation in selected_annotations:
        episode_id = int(annotation["id"])
        if episode_id not in raw_by_id:
            raise KeyError(f"Episode {episode_id} not found in Habitat ScaleVLN dataset")
        episode = copy.deepcopy(raw_by_id[episode_id])
        episode["scene_id"] = resolve_scene_id(episode["scene_id"], scenes_dir)
        subset_episodes.append(episode)

    subset = {
        "episodes": subset_episodes,
        "instruction_vocab": raw_dataset.get("instruction_vocab", {"word_list": []}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt") as f:
        json.dump(subset, f)


def filter_pending_annotations(
    annotations: Sequence[Dict[str, Any]], image_root: Path
) -> tuple[List[Dict[str, Any]], int]:
    pending = []
    skipped = 0
    for annotation in annotations:
        output_dir = image_root / output_dir_name(annotation)
        if episode_outputs_complete(output_dir, frame_count=len(annotation["actions"])):
            skipped += 1
            continue
        pending.append(annotation)
    return pending, skipped


def _configure_env(args: argparse.Namespace, data_path: Path) -> Any:
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("MAGNUM_LOG", "quiet")

    import habitat
    from habitat.config import read_write
    from habitat_baselines.config.default import get_config as get_habitat_config
    from omegaconf import open_dict

    import vln_baseline.habitat_extensions  # noqa: F401

    config = get_habitat_config(str(args.config_path))
    with read_write(config):
        config.habitat.dataset.type = "R2RVLN-v1"
        config.habitat.dataset.split = args.split
        config.habitat.dataset.data_path = str(data_path)
        config.habitat.dataset.scenes_dir = str(args.scenes_dir)
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_device_id
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        sensors.rgb_sensor.width = args.width
        sensors.rgb_sensor.height = args.height
        sensors.rgb_sensor.hfov = coerce_hfov_for_habitat_config(args.hfov)
        sensors.depth_sensor.width = args.width
        sensors.depth_sensor.height = args.height
        sensors.depth_sensor.hfov = coerce_hfov_for_habitat_config(args.hfov)
        sensors.depth_sensor.min_depth = args.min_depth
        sensors.depth_sensor.max_depth = args.max_depth
        if "normalize_depth" in sensors.depth_sensor:
            sensors.depth_sensor.normalize_depth = True
        if "oracle_success" in config.habitat.task.measurements:
            with open_dict(config.habitat.task.measurements.oracle_success):
                config.habitat.task.measurements.oracle_success.success_distance = 3.0
    return habitat.Env(config=config)


def _habitat_episode_map(env: Any) -> Dict[int, Any]:
    return {int(episode.episode_id): episode for episode in env.episodes}


def _prepare_output_dirs(output_dir: Path, *, overwrite: bool) -> None:
    if overwrite:
        for subdir in ("rgb", "depth", "pose"):
            shutil.rmtree(output_dir / subdir, ignore_errors=True)
    else:
        partial = [output_dir / subdir for subdir in ("rgb", "depth", "pose") if (output_dir / subdir).exists()]
        if partial:
            for path in partial:
                shutil.rmtree(path)
    for subdir in ("rgb", "depth", "pose"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def _save_frame(
    *,
    observation: Dict[str, Any],
    sensor_state: Any,
    frame_idx: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    frame_stem = f"{frame_idx:03d}"
    Image.fromarray(observation["rgb"]).convert("RGB").save(output_dir / "rgb" / f"{frame_stem}.jpg")
    Image.fromarray(depth_to_uint8(observation["depth"]), mode="L").save(
        output_dir / "depth" / f"{frame_stem}.png"
    )
    camera_params = build_camera_params(
        sensor_state,
        width=args.width,
        height=args.height,
        hfov=args.hfov,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        agent_height=args.agent_height,
        sensor_height_offset=args.sensor_height_offset,
    )
    with (output_dir / "pose" / f"{frame_stem}.json").open("w") as f:
        json.dump(camera_params, f, indent=2)
        f.write("\n")


def extract_episode(
    *,
    env: Any,
    habitat_episode: Any,
    annotation: Dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    actions = list(annotation["actions"])
    if not actions or actions[0] != -1:
        raise ValueError(f"Episode {annotation['id']} actions must start with -1")
    if args.skip_complete and episode_outputs_complete(output_dir, len(actions)):
        return {
            "episode_id": int(annotation["id"]),
            "episode_dir": str(output_dir),
            "frames": len(actions),
            "status": "skipped_complete",
        }

    _prepare_output_dirs(output_dir, overwrite=args.overwrite)
    env.current_episode = habitat_episode
    observation = env.reset()
    for frame_idx, action in enumerate(actions):
        if frame_idx > 0:
            observation = env.step(action)
        sensor_state = env.sim.get_agent_state().sensor_states["rgb"]
        _save_frame(
            observation=observation,
            sensor_state=sensor_state,
            frame_idx=frame_idx + 1,
            output_dir=output_dir,
            args=args,
        )
    return {
        "episode_id": int(annotation["id"]),
        "episode_dir": str(output_dir),
        "frames": len(actions),
        "status": "written",
    }


def extract_scalevln(args: argparse.Namespace) -> List[Dict[str, Any]]:
    annotations = load_json(args.annotation_path)
    selected = select_annotations(annotations, episode_ids=args.episode_ids, limit=args.limit)
    selected = shard_annotations(selected, shard_index=args.shard_index, num_shards=args.num_shards)
    if args.skip_complete:
        selected, skipped = filter_pending_annotations(selected, args.image_root)
        print(
            json.dumps(
                {
                    "event": "skip_complete_summary",
                    "skipped_complete_episodes": skipped,
                    "pending_episodes": len(selected),
                },
                sort_keys=True,
            )
        )
    if not selected:
        return []

    make_annotation_subset(selected, args.subset_annotation_path)
    raw_dataset = load_habitat_dataset(args.habitat_data_path)
    with tempfile.TemporaryDirectory(prefix="scalevln_habitat_subset_") as tmpdir:
        temp_data_path = Path(tmpdir) / "train.json.gz"
        write_habitat_subset(raw_dataset, selected, scenes_dir=args.scenes_dir, output_path=temp_data_path)
        env = _configure_env(args, temp_data_path)
        episode_by_id = _habitat_episode_map(env)
        results = []
        try:
            for index, annotation in enumerate(selected, start=1):
                episode_id = int(annotation["id"])
                if episode_id not in episode_by_id:
                    raise KeyError(f"Episode {episode_id} not loaded by Habitat")
                output_dir = args.image_root / output_dir_name(annotation)
                result = extract_episode(
                    env=env,
                    habitat_episode=episode_by_id[episode_id],
                    annotation=annotation,
                    output_dir=output_dir,
                    args=args,
                )
                result["index"] = index
                result["total"] = len(selected)
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
        finally:
            env.close()
    return results


def _parse_episode_ids(values: Optional[Iterable[str]]) -> Optional[List[int]]:
    if not values:
        return None
    return [int(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation_path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--habitat_data_path", type=Path, default=DEFAULT_HABITAT_DATA_PATH)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--subset_annotation_path", type=Path, default=DEFAULT_SUBSET_ANNOTATION_PATH)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scenes_dir", type=Path, default=DEFAULT_SCENES_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode_ids", nargs="*", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--gpu_device_id", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--min_depth", type=float, default=0.0)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--agent_height", type=float, default=1.5)
    parser.add_argument("--sensor_height_offset", type=float, default=1.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_complete", action="store_true")
    args = parser.parse_args()
    args.episode_ids = _parse_episode_ids(args.episode_ids)
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    return args


if __name__ == "__main__":
    extract_scalevln(parse_args())
