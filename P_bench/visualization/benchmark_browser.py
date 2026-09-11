"""Static browser built only from public QA and its structured private index."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Mapping

from pipeline import actions


TEMPLATE_PATH = Path(__file__).with_name("benchmark_browser.html")
TASK_DEFINITIONS = {
    "A1": {
        "family": "A",
        "name": "Collision",
        "name_zh": "碰撞判断",
        "description_zh": "结合初始视图、圆形机体半径和完整动作序列，判断执行过程中是否会碰撞。测的是给定动作的可通行性，不是重新规划或主动避障。",
        "capability": "Predict whether the action sequence causes a collision.",
        "answer": "Collision or no collision",
    },
    "A2": {
        "family": "A",
        "name": "First collision action",
        "name_zh": "首次碰撞动作定位",
        "description_zh": "已知序列会发生碰撞，判断首次接触发生在第几个动作。序号按包含 Forward 和 Turn 的完整列表计算；测动作级时序定位，不是精确碰撞时间。",
        "capability": "Identify the action during which the first collision begins.",
        "answer": "Action index",
    },
    "A3": {
        "family": "A",
        "name": "First-contact category",
        "name_zh": "首次接触类别识别",
        "description_zh": "结合运动路径和可见目标的语义，判断首先碰到什么类别，例如 chair 或 wall。答案是类别而非具体实例 ID；仅使用 B1K 和 R2R。",
        "capability": (
            "For B1K and R2R/MP3D, name the initially visible object or "
            "surface contacted first. GS does not publish A3 labels."
        ),
        "answer": "Open category name",
    },
    "A4": {
        "family": "B",
        "name": "In-progress target direction",
        "name_zh": "中途目标方向",
        "description_zh": "在指定 Forward 完成题目给定百分比时，推演当时的相机位置与朝向，判断固定标记点的相对方向。必须使用这个中途时刻，而不是完整序列的终点。",
        "capability": (
            "Predict the camera-centred 3D direction to an initially visible "
            "target while a specified Forward action is in progress."
        ),
        "answer": (
            "Eight horizontal directions plus above, camera level, or below"
        ),
        "ground_truth_basis": (
            "Azimuth and elevation from the camera optical center at the "
            "selected checkpoint to the fixed surface point at the dot's center."
        ),
    },
    "B1": {
        "family": "B",
        "name": "Endpoint target distance",
        "name_zh": "终点目标距离",
        "description_zh": "安全执行完整动作序列后，估计相机光心到标记点的三维直线距离，单位为米。不是运动路程、水平距离或到物体中心的距离；仅使用 B1K 和 R2R。",
        "capability": (
            "Estimate the three-dimensional straight-line distance from the "
            "endpoint onboard camera optical center to the marked surface "
            "point."
        ),
        "answer": "Distance in meters",
    },
    "B2": {
        "family": "B",
        "name": "Endpoint target direction",
        "name_zh": "终点目标方向",
        "description_zh": "安全执行完整序列后，判断固定标记点相对终点相机的位置，包含最后一次转向带来的朝向变化。与 A4 使用同一方向定义，区别是查询时刻。",
        "capability": (
            "Locate a marked target in 3D relative to the camera optical "
            "center, viewing direction, and camera level at the endpoint."
        ),
        "answer": (
            "Eight horizontal directions plus above, camera level, or below"
        ),
    },
    "B3": {
        "family": "B",
        "name": "In-progress target distance",
        "name_zh": "中途目标距离",
        "description_zh": "在指定 Forward 完成题目给定百分比时，估计当时相机光心到固定标记点的三维直线距离。与 A4 查询同类中途时刻，与 B1 使用同一距离定义；仅使用 B1K 和 R2R。",
        "capability": "Estimate camera-to-point 3D distance partway through a specified Forward action. B1K and R2R only.",
        "answer": "Distance in meters",
        "ground_truth_basis": "3D straight-line distance from the camera optical center at the selected checkpoint to the fixed surface point at the dot's center.",
    },
    "C1": {
        "family": "C",
        "name": "Future-view selection",
        "name_zh": "终点视图辨别",
        "description_zh": "根据初始视图和安全动作序列，从 A–D 四张候选图中选出执行完后的相机视图。需要理解自身平移、转向和视角变化；测视图辨别，不是生成未来图像。",
        "capability": "Select the true camera observation after executing all actions.",
        "answer": "Candidate label A, B, C, or D",
    },
}


def _message(item: Mapping[str, object], role: str) -> str:
    messages = item.get("messages") or []
    matches = [
        str(message.get("content") or "")
        for message in messages if message.get("role") == role
    ]
    return matches[-1] if matches else ""


def _question(prompt: str) -> str:
    marker = "Question:"
    text = prompt.split(marker, 1)[1].strip() if marker in prompt else prompt
    context = [block for block in prompt.split(marker, 1)[0].split("\n\n")
               if block.startswith(("Query moment:", "Measure the 3D straight-line distance"))]
    return "\n\n".join([*context, text.split("\n\nCandidate images:", 1)[0].strip()])


def _number(value: object) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _action_sequence(actions) -> str:
    lines = []
    for index, action in enumerate(actions, 1):
        if action["type"] == "forward":
            text = f"forward {_number(action['m'])} m"
        else:
            degrees = float(action["deg"])
            side = "right" if degrees >= 0 else "left"
            text = f"turn {side} {_number(abs(degrees))} degree"
        lines.append(f"({index}) {text}")
    return "\n".join(lines)


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _case_item(item: Mapping[str, object], index: Mapping[str, object]) -> dict:
    prompt = _message(item, "user")
    source = index.get("source") or {}
    usage = index.get("usage") or {}
    action_rows = usage["actions"]
    relation = (usage.get("target_point") or {}).get("relation") or {}
    direction = (actions.spatial_direction_key(
        relation["horizontal_direction"], relation["vertical_direction"])
        if item["task_id"] in {"A4", "B2"} and relation else None)
    ground_truth_evidence = (
        usage.get("category_recheck") or usage.get("target_point")
    )
    marked_sha = (
        ground_truth_evidence.get("marked_image_sha256")
        if isinstance(ground_truth_evidence, Mapping) else None
    )
    raw_checkpoint = usage.get("checkpoint")
    checkpoint = None
    if isinstance(raw_checkpoint, Mapping):
        checkpoint = {
            "action_index": int(raw_checkpoint["action_index_1based"]),
            "forward_ordinal": int(
                raw_checkpoint["forward_ordinal_1based"]),
            "fraction": float(raw_checkpoint["fraction"]),
        }
    raw_a2 = usage.get("a2_design")
    a2_design = None
    if isinstance(raw_a2, Mapping):
        cell = raw_a2["cell"]
        a2_design = {
            "collision_action_index": int(
                usage["collision_action_index_1based"]),
            "forward_ordinal": int(cell["forward_ordinal_1based"]),
            "distance_rank": str(cell["distance_rank"]),
        }
    return {
        "id": str(item["id"]),
        "task_id": str(item["task_id"]),
        "dataset": str(item["dataset"]),
        "scene_id": str(index.get("scene_id") or "unknown"),
        "images": [str(path) for path in item.get("images") or []],
        "image_version": str(
            marked_sha or index.get("initial_image_sha256") or ""),
        "system": _message(item, "system"),
        "prompt": prompt,
        "question": _question(prompt),
        "answer": _message(item, "assistant"),
        "direction": direction,
        "action_sequence": _action_sequence(action_rows),
        "action_length": int(usage["action_length"]),
        "starts_with": action_rows[0]["type"],
        "checkpoint": checkpoint,
        "a2_design": a2_design,
        "ground_truth_evidence": ground_truth_evidence,
        "image_resolution_px": usage["resolution_px"],
        "robot": {
            "radius_m": float(usage["body_radius_m"]),
        },
        "camera": {
            "height_m": float(usage["camera_height_m"]),
            "hfov_deg": float(usage["hfov_deg"]),
            "vfov_deg": float(usage["vfov_deg"]),
        },
        "provenance": {
            "record_id": index.get("record_uid") or index.get("record_id"),
            "record_sha256": (
                index.get("source_record_sha256") or
                index.get("record_sha256")
            ),
            "initial_image_sha256": index.get("initial_image_sha256"),
            "records_path": source.get("records_path"),
            "byte_offset": source.get("byte_offset"),
            "outcome_id": usage.get("outcome_id"),
        },
    }


def _payload(benchmark_dir: Path, record_index_path: Path,
             output_dir: Path, *, qa_filename="qa.json") -> dict:
    qa_items = json.loads(
        (benchmark_dir / qa_filename).read_text(encoding="utf-8")
    )
    record_index = json.loads(
        record_index_path.read_text(encoding="utf-8")
    )
    index_by_id = {
        str(row["item_id"]): row for row in record_index["items"]
    }
    cases = [
        _case_item(item, index_by_id[str(item["id"])])
        for item in qa_items
    ]
    image_base = Path(os.path.relpath(benchmark_dir, output_dir)).as_posix().rstrip("/") + "/"
    return {
        **_case_statistics(cases),
        "schema": "egoconseq.benchmark-case-browser.v1",
        "benchmark": record_index.get("benchmark", benchmark_dir.name),
        "image_base": image_base,
        "qa_links": [{"label": "QA JSON", "path": image_base + qa_filename}],
    }


def _case_statistics(cases: list[dict]) -> dict:
    task_dataset = defaultdict(Counter)
    task_lengths, task_starts, answer_counts, a1_starts = (defaultdict(Counter) for _ in range(4))
    for case in cases:
        task = case["task_id"]
        task_dataset[task][case["dataset"]] += 1
        task_lengths[task][f"L{case['action_length']}"] += 1
        task_starts[task][case["starts_with"]] += 1
        if task in {"A1", "A3", "C1"}:
            answer_counts[task][case["answer"]] += 1
        if task == "A1":
            a1_starts[case["starts_with"]][case["answer"]] += 1

    task_definitions = {
        task_id: {
            **definition,
            "count": sum(case["task_id"] == task_id for case in cases),
            "action_lengths": sorted({
                case["action_length"] for case in cases
                if case["task_id"] == task_id
            }),
        }
        for task_id, definition in TASK_DEFINITIONS.items()
    }
    checkpoints = {task: [case["checkpoint"] for case in cases
                         if case["task_id"] == task and case["checkpoint"] is not None]
                   for task in ("A4", "B3")}
    a2_designs = [
        case["a2_design"] for case in cases
        if case["task_id"] == "A2" and case["a2_design"] is not None
    ]
    return {
        "summary": {
            "qa_items": len(cases),
            "records": len({case["provenance"]["record_id"] for case in cases}),
            "question_image_references": sum(
                bool(case["images"]) for case in cases),
            "unique_initial_observations": len({
                case["provenance"]["initial_image_sha256"]
                for case in cases
                if case["provenance"]["initial_image_sha256"]
            }),
            "c1_candidates": sum(
                max(0, len(case["images"]) - 1) for case in cases
                if case["task_id"] == "C1"
            ),
            "image_references": sum(len(case["images"]) for case in cases),
            "datasets": len({case["dataset"] for case in cases}),
            "scenes": len({
                (case["dataset"], case["scene_id"]) for case in cases
            }),
            "tasks": len({case["task_id"] for case in cases}),
        },
        "counts": {
            "task_lengths": dict(task_lengths),
            "task_starts": dict(task_starts),
            "answers": dict(answer_counts),
            "a1_starts": dict(a1_starts),
            "parameters": {
                "radius": _counts(_number(case["robot"]["radius_m"]) for case in cases),
                "height": _counts(_number(case["camera"]["height_m"]) for case in cases),
                "fov": _counts(f"{_number(case['camera']['hfov_deg'])}° / {_number(case['camera']['vfov_deg'])}°" for case in cases),
            },
            "directions": {task: {
                actions.spatial_direction_key(h, v): sum(
                    case["task_id"] == task and case["direction"] == actions.spatial_direction_key(h, v)
                    for case in cases)
                for h in actions.HORIZONTAL_DIRECTION_LABELS
                for v in actions.VERTICAL_DIRECTION_LABELS
            } for task in ("A4", "B2")},
            "tasks": _counts(case["task_id"] for case in cases),
            "datasets": _counts(case["dataset"] for case in cases),
            "scenes": _counts(
                f"{case['dataset']}/{case['scene_id']}" for case in cases
            ),
            "action_lengths": _counts(
                f"L{case['action_length']}" for case in cases
            ),
            **{f"{task.lower()}_checkpoints": {
                "fractions": _counts(
                    f"{checkpoint['fraction'] * 100:g}%"
                    for checkpoint in values
                ),
                "action_indices": _counts(
                    f"Action {checkpoint['action_index']}"
                    for checkpoint in values
                ),
                "forward_ordinals": _counts(
                    f"Forward {checkpoint['forward_ordinal']}"
                    for checkpoint in values
                ),
            } for task, values in checkpoints.items()},
            "a2_collision_design": {
                "action_indices": _counts(
                    f"Action {value['collision_action_index']}" for value in a2_designs
                ),
                "forward_ordinals": _counts(
                    f"Forward {value['forward_ordinal']}"
                    for value in a2_designs
                ),
                "distance_ranks": _counts(
                    value["distance_rank"].title()
                    for value in a2_designs
                ),
                "joint_cells": _counts(
                    f"Forward {value['forward_ordinal']} · "
                    f"{value['distance_rank']}"
                    for value in a2_designs
                ),
            },
            "task_dataset": {
                task_id: dict(sorted(counts.items()))
                for task_id, counts in sorted(task_dataset.items())
            },
        },
        "taxonomy": task_definitions,
        "direction_axes": {"horizontal": actions.HORIZONTAL_DIRECTION_LABELS,
                           "vertical": actions.VERTICAL_DIRECTION_LABELS},
        "items": cases,
    }


def build_benchmark_browser(
        benchmark_dir: Path, record_index_path: Path,
        output_dir: Path, *, selection_report=None) -> dict[str, Path]:
    """One HTML with embedded metadata; usable directly without a web server."""
    benchmark_dir = Path(benchmark_dir)
    record_index_path = Path(record_index_path)
    output_dir = Path(output_dir)
    payload = _payload(benchmark_dir, record_index_path, output_dir)
    payload["selection"] = selection_report or {}
    return _write_browser(payload, output_dir)


def build_combined_benchmark_browser(
        root: Path, output_dir: Path | None = None) -> dict[str, Path]:
    """Join published splits for browsing only; reuse their QA, indices and images."""
    root = Path(root)
    output_dir = Path(output_dir) if output_dir is not None else root
    cases, split_views, qa_links = [], {}, []
    for split in ("seen", "unseen"):
        metadata = root / "metadata" / split
        view = _payload(root / "benchmark" / split, metadata / "record_index.json",
                        output_dir, qa_filename="QA.json")
        image_base = view.pop("image_base")
        for case in view.pop("items"):
            case.update(split=split, key=f"{split}/{case['id']}")
            case["images"] = [image_base + path for path in case["images"]]
            cases.append(case)
        view["label"] = split.title()
        view["qa_links"][0]["label"] = f"{split.title()} QA JSON"
        view["selection"] = json.loads((metadata / "report.json").read_text(encoding="utf-8"))
        split_views[split] = view
        qa_links.extend(view["qa_links"])
    return _write_browser({
        **_case_statistics(cases),
        "schema": "egoconseq.benchmark-case-browser.v1",
        "benchmark": "egoconseq-seen-unseen",
        "image_base": "", "qa_links": qa_links,
        "splits": split_views, "selection": {},
    }, output_dir)


def _write_browser(payload: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html_path = output_dir / "index.html"
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        '<!-- BENCHMARK_DATA -->',
        '<script id="benchmark-data" type="application/json">' + embedded + '</script>')
    html_path.write_text(html, encoding="utf-8")
    return {"html": html_path}
