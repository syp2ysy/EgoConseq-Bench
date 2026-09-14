import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _mean(metrics: List[Dict[str, Any]], key: str) -> float:
    return float(np.mean([float(item.get(key, 0.0)) for item in metrics])) if metrics else 0.0


def _count_ge(metrics: List[Dict[str, Any]], key: str, threshold: float) -> int:
    return int(sum(float(item.get(key, 0.0)) >= threshold for item in metrics))


def _duplicates(items: List[Tuple[str, ...]]) -> List[List[str]]:
    counts = Counter(items)
    return [list(item) for item, count in sorted(counts.items()) if count > 1]


def aggregate_split_metrics(
    output_dir: Path,
    split: str,
    expected_splits: Optional[int] = None,
    expected_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    metrics: List[Dict[str, Any]] = []
    split_counts: Dict[str, int] = {}

    for split_dir in sorted(output_dir.glob("split_*")):
        log_dir = split_dir / "log"
        stats = []
        if log_dir.exists():
            for stats_path in sorted(log_dir.glob("stats_*.json")):
                with open(stats_path, "r") as f:
                    stats.append(json.load(f))
        if stats:
            split_counts[split_dir.name] = len(stats)
            metrics.extend(stats)

    episode_ids = [str(item.get("id", "")) for item in metrics]
    episode_scene_pairs = [(str(item.get("id", "")), str(item.get("scene_id", ""))) for item in metrics]
    episode_instruction_pairs = [
        (str(item.get("id", "")), str(item.get("instruction", ""))) for item in metrics
    ]
    missing_split_dirs: List[str] = []
    if expected_splits is not None:
        missing_split_dirs = [
            f"split_{split_id}"
            for split_id in range(expected_splits)
            if split_counts.get(f"split_{split_id}", 0) == 0
        ]
    episode_count_matches_expected = None
    if expected_episodes is not None:
        episode_count_matches_expected = len(metrics) == expected_episodes
    duplicate_episode_ids = _duplicates([(episode_id,) for episode_id in episode_ids])
    duplicate_episode_scene_pairs = _duplicates(episode_scene_pairs)
    duplicate_episode_instruction_pairs = _duplicates(episode_instruction_pairs)
    distributed_valid = (
        not missing_split_dirs
        and not duplicate_episode_instruction_pairs
        and (episode_count_matches_expected is not False)
    )

    return {
        "split": split,
        "num_episodes": len(metrics),
        "split_counts": split_counts,
        "expected_splits": expected_splits,
        "missing_split_dirs": missing_split_dirs,
        "expected_episodes": expected_episodes,
        "episode_count_matches_expected": episode_count_matches_expected,
        "unique_episode_ids": len(set(episode_ids)),
        "unique_episode_id_scene_pairs": len(set(episode_scene_pairs)),
        "unique_episode_id_instruction_pairs": len(set(episode_instruction_pairs)),
        "duplicate_episode_ids": duplicate_episode_ids,
        "duplicate_episode_id_scene_pairs": duplicate_episode_scene_pairs,
        "duplicate_episode_id_instruction_pairs": duplicate_episode_instruction_pairs,
        "distributed_valid": distributed_valid,
        "success_rate": _mean(metrics, "success"),
        "oracle_success": _mean(metrics, "oracle_success"),
        "spl": _mean(metrics, "spl"),
        "navigation_error": _mean(metrics, "distance_to_goal"),
        "oracle_navigation_error": _mean(metrics, "oracle_navigation_error"),
        "avg_path_length": _mean(metrics, "path_length"),
        "avg_num_steps": _mean(metrics, "num_steps"),
        "episodes_ge_400_steps": _count_ge(metrics, "num_steps", 400),
        "avg_elapsed_seconds": _mean(metrics, "elapsed_seconds"),
        "fallback_total": int(sum(int(item.get("fallback_count", 0)) for item in metrics)),
    }


def write_aggregate_metrics(
    output_dir: Path,
    split: str,
    expected_splits: Optional[int] = None,
    expected_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    metrics = aggregate_split_metrics(
        Path(output_dir),
        split,
        expected_splits=expected_splits,
        expected_episodes=expected_episodes,
    )
    with open(Path(output_dir) / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
