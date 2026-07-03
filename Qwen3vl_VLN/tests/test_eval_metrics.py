import json
import subprocess
import sys
from pathlib import Path

from vln_baseline.eval_metrics import aggregate_split_metrics


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _write_stat(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_aggregate_split_metrics_uses_stats_id_and_distance_fields(tmp_path: Path):
    _write_stat(
        tmp_path / "split_0" / "log" / "stats_1.json",
        {
            "id": "1",
            "scene_id": "scene",
            "instruction": "go",
            "success": 1.0,
            "spl": 0.5,
            "distance_to_goal": 2.0,
            "oracle_success": 1.0,
            "oracle_navigation_error": 1.0,
            "ndtw": 0.1,
            "sdtw": 0.1,
            "path_length": 4.0,
            "num_steps": 10,
            "elapsed_seconds": 3.0,
            "fallback_count": 1,
        },
    )
    _write_stat(
        tmp_path / "split_1" / "log" / "stats_2.json",
        {
            "id": "2",
            "scene_id": "scene",
            "instruction": "turn",
            "success": 0.0,
            "spl": 0.0,
            "distance_to_goal": 6.0,
            "oracle_success": 0.0,
            "oracle_navigation_error": 5.0,
            "ndtw": 0.0,
            "sdtw": 0.0,
            "path_length": 8.0,
            "num_steps": 20,
            "elapsed_seconds": 5.0,
            "fallback_count": 0,
        },
    )

    metrics = aggregate_split_metrics(tmp_path, split="val_unseen")

    assert metrics["num_episodes"] == 2
    assert metrics["unique_episode_ids"] == 2
    assert metrics["success_rate"] == 0.5
    assert metrics["navigation_error"] == 4.0
    assert metrics["split_counts"] == {"split_0": 1, "split_1": 1}
    assert metrics["fallback_total"] == 1


def test_aggregate_split_metrics_flags_duplicate_and_missing_distributed_outputs(tmp_path: Path):
    payload = {
        "id": "1",
        "scene_id": "scene",
        "instruction": "go",
        "success": 1.0,
        "spl": 0.5,
        "distance_to_goal": 2.0,
        "oracle_success": 1.0,
        "oracle_navigation_error": 1.0,
        "ndtw": 0.1,
        "sdtw": 0.1,
        "path_length": 4.0,
        "num_steps": 10,
        "elapsed_seconds": 3.0,
        "fallback_count": 0,
    }
    _write_stat(tmp_path / "split_0" / "log" / "stats_1.json", payload)
    _write_stat(tmp_path / "split_1" / "log" / "stats_1.json", payload)

    metrics = aggregate_split_metrics(
        tmp_path,
        split="val_unseen",
        expected_splits=4,
        expected_episodes=1839,
    )

    assert metrics["duplicate_episode_id_instruction_pairs"] == [["1", "go"]]
    assert metrics["missing_split_dirs"] == ["split_2", "split_3"]
    assert metrics["expected_episodes"] == 1839
    assert metrics["episode_count_matches_expected"] is False
    assert metrics["distributed_valid"] is False


def test_aggregate_split_metrics_marks_distributed_output_valid_when_complete(tmp_path: Path):
    for split_id in range(4):
        _write_stat(
            tmp_path / f"split_{split_id}" / "log" / f"stats_{split_id}.json",
            {
                "id": str(split_id),
                "scene_id": "scene",
                "instruction": f"go {split_id}",
                "success": 1.0,
                "spl": 0.5,
                "distance_to_goal": 2.0,
                "oracle_success": 1.0,
                "oracle_navigation_error": 1.0,
                "ndtw": 0.1,
                "sdtw": 0.1,
                "path_length": 4.0,
                "num_steps": 10,
                "elapsed_seconds": 3.0,
                "fallback_count": 0,
            },
        )

    metrics = aggregate_split_metrics(
        tmp_path,
        split="val_unseen",
        expected_splits=4,
        expected_episodes=4,
    )

    assert metrics["duplicate_episode_id_instruction_pairs"] == []
    assert metrics["missing_split_dirs"] == []
    assert metrics["episode_count_matches_expected"] is True
    assert metrics["distributed_valid"] is True


def test_aggregate_cli_require_valid_rejects_invalid_distributed_output(tmp_path: Path):
    _write_stat(
        tmp_path / "split_0" / "log" / "stats_1.json",
        {
            "id": "1",
            "scene_id": "scene",
            "instruction": "go",
            "success": 1.0,
            "spl": 0.5,
            "distance_to_goal": 2.0,
            "oracle_success": 1.0,
            "oracle_navigation_error": 1.0,
            "ndtw": 0.1,
            "sdtw": 0.1,
            "path_length": 4.0,
            "num_steps": 10,
            "elapsed_seconds": 3.0,
            "fallback_count": 0,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "aggregate_eval_results.py"),
            str(tmp_path),
            "--split",
            "val_unseen",
            "--expected-splits",
            "4",
            "--expected-episodes",
            "1839",
            "--require-valid",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "distributed_valid=false" in result.stderr


def test_aggregate_cli_require_valid_accepts_valid_distributed_output(tmp_path: Path):
    for split_id in range(4):
        _write_stat(
            tmp_path / f"split_{split_id}" / "log" / f"stats_{split_id}.json",
            {
                "id": str(split_id),
                "scene_id": "scene",
                "instruction": f"go {split_id}",
                "success": 1.0,
                "spl": 0.5,
                "distance_to_goal": 2.0,
                "oracle_success": 1.0,
                "oracle_navigation_error": 1.0,
                "ndtw": 0.1,
                "sdtw": 0.1,
                "path_length": 4.0,
                "num_steps": 10,
                "elapsed_seconds": 3.0,
                "fallback_count": 0,
            },
        )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "aggregate_eval_results.py"),
            str(tmp_path),
            "--split",
            "val_unseen",
            "--expected-splits",
            "4",
            "--expected-episodes",
            "4",
            "--require-valid",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert '"distributed_valid": true' in result.stdout
