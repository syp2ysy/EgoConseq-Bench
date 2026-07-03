#!/usr/bin/env python3
"""Replay StreamVLN-style R2R/RxR trajectories and save depth plus camera pose."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ANNOTATION_PATH = Path(
    "/home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data/R2R/annotations_v1-3.json"
)
DEFAULT_IMAGE_ROOT = Path("/home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data/R2R/images")
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "vln_r2r.yaml"
DEFAULT_DATA_PATH = (
    "/home/zhangshan/syp/StreamVLN-R2RCE-Test/data/datasets/r2r/{split}/{split}.json.gz"
)
DEFAULT_SCENES_DIR = "/home/zhangshan/syp/StreamVLN-R2RCE-Test/data/scene_datasets/"


def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """Match the existing StreamVLN depth PNG encoding: normalized depth * 255."""
    depth_array = np.asarray(depth)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(depth_array, 0.0, 1.0) * 255).astype(np.uint8)


def coerce_hfov_for_habitat_config(hfov: float) -> float:
    return int(hfov) if float(hfov).is_integer() else hfov


def _quaternion_wxyz(rotation: Any) -> List[float]:
    if hasattr(rotation, "real") and hasattr(rotation, "imag"):
        imag = np.asarray(rotation.imag, dtype=np.float64)
        return [float(rotation.real), float(imag[0]), float(imag[1]), float(imag[2])]
    if all(hasattr(rotation, attr) for attr in ("w", "x", "y", "z")):
        return [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)]
    values = list(rotation)
    if len(values) != 4:
        raise ValueError(f"Expected a 4-value quaternion, got {values}")
    return [float(value) for value in values]


def _rotation_matrix_from_wxyz(quaternion_wxyz: Sequence[float]) -> List[List[float]]:
    w, x, y, z = quaternion_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ],
        [
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ],
        [
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
    ]


def build_camera_params(
    sensor_state: Any,
    *,
    width: int,
    height: int,
    hfov: float,
    min_depth: float,
    max_depth: float,
    agent_height: float,
    sensor_height_offset: float,
) -> Dict[str, Any]:
    focal = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    position = [float(value) for value in np.asarray(sensor_state.position).tolist()]
    quaternion_wxyz = _quaternion_wxyz(sensor_state.rotation)
    rotation_matrix = _rotation_matrix_from_wxyz(quaternion_wxyz)
    camera_to_world = [
        rotation_matrix[0] + [position[0]],
        rotation_matrix[1] + [position[1]],
        rotation_matrix[2] + [position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": focal,
            "fy": focal,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "hfov": hfov,
            "near": min_depth,
            "far": max_depth,
            "agent_height": agent_height,
            "sensor_height_offset": sensor_height_offset,
        },
        "extrinsics": {
            "position": position,
            "rotation_quaternion": quaternion_wxyz,
            "rotation_matrix": rotation_matrix,
            "camera_to_world": camera_to_world,
        },
    }


def load_annotations(annotation_path: Path) -> List[Dict[str, Any]]:
    with annotation_path.open("r") as f:
        return json.load(f)


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
            raise KeyError(f"Episode ids not found in annotation file: {missing}")
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


def _episode_dir_name(annotation: Dict[str, Any]) -> str:
    return Path(annotation["video"]).name


def episode_outputs_complete(output_dir: Path, frame_count: int) -> bool:
    expected_depth = {f"{idx:03d}.png" for idx in range(1, frame_count + 1)}
    expected_pose = {f"{idx:03d}.json" for idx in range(1, frame_count + 1)}
    depth_dir = output_dir / "depth"
    pose_dir = output_dir / "pose"
    if not depth_dir.is_dir() or not pose_dir.is_dir():
        return False
    return (
        {path.name for path in depth_dir.glob("*.png")} == expected_depth
        and {path.name for path in pose_dir.glob("*.json")} == expected_pose
    )


def filter_pending_annotations(
    annotations: Sequence[Dict[str, Any]], image_root: Path
) -> tuple[List[Dict[str, Any]], int]:
    pending = []
    skipped = 0
    for annotation in annotations:
        output_dir = image_root / _episode_dir_name(annotation)
        if episode_outputs_complete(output_dir, frame_count=len(annotation["actions"])):
            skipped += 1
            continue
        pending.append(annotation)
    return pending, skipped


def _episode_id_map(env: Any) -> Dict[int, Any]:
    return {int(episode.episode_id): episode for episode in env.episodes}


def _configure_env(args: argparse.Namespace) -> Any:
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("MAGNUM_LOG", "quiet")

    import habitat
    from habitat.config import read_write
    from habitat_baselines.config.default import get_config as get_habitat_config
    from omegaconf import open_dict

    import vln_baseline.habitat_extensions  # noqa: F401

    config = get_habitat_config(str(args.config_path))
    with read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.dataset.data_path = args.data_path
        config.habitat.dataset.scenes_dir = args.scenes_dir
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


def _mean_abs_rgb_diff(rendered_rgb: np.ndarray, existing_rgb_path: Path) -> float:
    existing = np.asarray(Image.open(existing_rgb_path).convert("RGB"), dtype=np.float32)
    rendered = np.asarray(rendered_rgb, dtype=np.float32)
    if existing.shape != rendered.shape:
        return float("inf")
    return float(np.mean(np.abs(existing - rendered)))


def _save_frame(
    *,
    observation: Dict[str, Any],
    sensor_state: Any,
    frame_idx: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> float:
    frame_stem = f"{frame_idx:03d}"
    depth_path = output_dir / "depth" / f"{frame_stem}.png"
    pose_path = output_dir / "pose" / f"{frame_stem}.json"
    rgb_path = output_dir / "rgb" / f"{frame_stem}.jpg"

    Image.fromarray(depth_to_uint8(observation["depth"]), mode="L").save(depth_path)
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
    with pose_path.open("w") as f:
        json.dump(camera_params, f, indent=2)
        f.write("\n")

    return _mean_abs_rgb_diff(observation["rgb"], rgb_path)


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
    rgb_dir = output_dir / "rgb"
    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if len(rgb_files) != len(actions):
        raise ValueError(
            f"Episode {annotation['id']} has {len(rgb_files)} RGB frames but {len(actions)} actions"
        )

    depth_dir = output_dir / "depth"
    pose_dir = output_dir / "pose"
    depth_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)
    if not args.overwrite and (any(depth_dir.iterdir()) or any(pose_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing depth/pose files in {output_dir}")

    env.current_episode = habitat_episode
    observation = env.reset()
    rgb_diffs = []

    for frame_idx, action in enumerate(actions):
        if frame_idx > 0:
            observation = env.step(action)
        sensor_state = env.sim.get_agent_state().sensor_states["rgb"]
        rgb_diffs.append(
            _save_frame(
                observation=observation,
                sensor_state=sensor_state,
                frame_idx=frame_idx + 1,
                output_dir=output_dir,
                args=args,
            )
        )

    return {
        "episode_id": int(annotation["id"]),
        "episode_dir": str(output_dir),
        "frames": len(actions),
        "rgb_mean_abs_diff_mean": float(np.mean(rgb_diffs)),
        "rgb_mean_abs_diff_max": float(np.max(rgb_diffs)),
    }


def extract_depth_pose(args: argparse.Namespace) -> List[Dict[str, Any]]:
    annotations = load_annotations(args.annotation_path)
    selected = select_annotations(annotations, episode_ids=args.episode_ids, limit=args.limit)
    selected = shard_annotations(selected, shard_index=args.shard_index, num_shards=args.num_shards)
    if args.skip_complete:
        selected, skipped_count = filter_pending_annotations(selected, args.image_root)
        print(
            json.dumps(
                {
                    "event": "skip_complete_summary",
                    "skipped_complete_episodes": skipped_count,
                    "pending_episodes": len(selected),
                },
                sort_keys=True,
            )
        )
        if not selected:
            return []
    env = _configure_env(args)
    episode_by_id = _episode_id_map(env)
    results = []
    try:
        for annotation in selected:
            episode_id = int(annotation["id"])
            if episode_id not in episode_by_id:
                raise KeyError(f"Episode {episode_id} not found in Habitat {args.split} split")
            output_dir = args.image_root / _episode_dir_name(annotation)
            result = extract_episode(
                env=env,
                habitat_episode=episode_by_id[episode_id],
                annotation=annotation,
                output_dir=output_dir,
                args=args,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True))
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
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--scenes_dir", default=DEFAULT_SCENES_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode_ids", nargs="*", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--gpu_device_id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--min_depth", type=float, default=0.0)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--agent_height", type=float, default=1.5)
    parser.add_argument("--sensor_height_offset", type=float, default=1.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip_complete",
        action="store_true",
        help="Skip episodes whose depth and pose files exactly match the expected frame count.",
    )
    args = parser.parse_args()
    args.episode_ids = _parse_episode_ids(args.episode_ids)
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    return args


if __name__ == "__main__":
    extract_depth_pose(parse_args())
