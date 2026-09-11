import json
from pathlib import Path

from visualization.benchmark_browser import (
    build_benchmark_browser, build_combined_benchmark_browser,
)


TASKS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1")


def test_card_question_keeps_the_separate_query_moment_and_distance_definition():
    from visualization.benchmark_browser import _question

    prompt = ("Configuration: ...\n\n"
              "Query moment: After earlier actions and 50% of the forward distance in action 3.\n\n"
              "Measure the 3D straight-line distance from your camera's optical center to the marked point.\n\n"
              "Question: At this moment, how far is the marked point from your camera?\n\n"
              "Answer in meters.")
    visible = _question(prompt)
    assert "50%" in visible and "action 3" in visible
    assert "3D straight-line distance" in visible and "optical center" in visible
    assert "how far" in visible and "Configuration" not in visible


def test_combined_browser_keeps_split_bindings_and_statistics(tmp_path):
    roots = {split: tmp_path / "benchmark" / split for split in ("seen", "unseen")}
    originals = {}
    for split, root in roots.items():
        root.mkdir(parents=True)
        metadata = tmp_path / "metadata" / split
        metadata.mkdir(parents=True)
        qa = [{"id": "shared-id", "task_id": "A1", "dataset": "r2r",
               "images": ["images/initial.png"], "messages": [
                   {"role": "user", "content": f"Question: {split}?"},
                   {"role": "assistant", "content": split}]}]
        index = {"items": [{
            "item_id": "shared-id", "scene_id": split,
            "record_uid": f"record-{split}", "initial_image_sha256": split,
            "source": {"records_path": f"/{split}/records.jsonl", "byte_offset": 0},
            "usage": {"actions": ([{"type": "turn", "deg": 15}] if split == "unseen" else []) + [{"type": "forward", "m": 0.5}],
                      "action_length": 2 if split == "unseen" else 1, "body_radius_m": 0.2,
                      "camera_height_m": 1.5 if split == "unseen" else 0.5, "hfov_deg": 79, "vfov_deg": 63,
                      "resolution_px": [640, 480]},
        }]}
        for path, value in ((root / "QA.json", qa),
                            (metadata / "record_index.json", index),
                            (metadata / "report.json", {"visual_selection": {"model": split}})):
            path.write_text(json.dumps(value))
            originals[path] = path.read_bytes()

    output = tmp_path
    result = build_combined_benchmark_browser(tmp_path)
    html = result["html"].read_text()
    payload = json.loads(html.split('<script id="benchmark-data" type="application/json">')[1].split('</script>')[0])
    assert payload["summary"]["qa_items"] == 2
    assert payload["summary"]["scenes"] == 2
    assert payload["counts"]["tasks"] == {"A1": 2}
    assert payload["counts"]["task_starts"] == {"A1": {"forward": 1, "turn": 1}}
    assert payload["counts"]["task_lengths"] == {"A1": {"L1": 1, "L2": 1}}
    assert payload["counts"]["parameters"]["height"] == {"0.5": 1, "1.5": 1}
    assert payload["counts"]["answers"]["A1"] == {"seen": 1, "unseen": 1}
    assert payload["summary"]["records"] == 2
    assert payload["selection"] == {}  # No invented cross-split similarity audit.
    assert payload["image_base"] == ""
    assert len({case["key"] for case in payload["items"]}) == 2
    for case in payload["items"]:
        split = case["split"]
        assert case["id"] == "shared-id"
        assert case["answer"] == split
        assert case["provenance"]["record_id"] == f"record-{split}"
        assert (output / case["images"][0]).resolve() == roots[split] / "images/initial.png"
        scope = payload["splits"][split]
        assert scope["summary"]["qa_items"] == 1
        assert scope["counts"]["scenes"] == {f"r2r/{split}": 1}
        assert scope["taxonomy"]["A1"]["count"] == 1
        assert scope["counts"]["task_starts"]["A1"] == {"turn" if split == "unseen" else "forward": 1}
        assert scope["counts"]["answers"]["A1"] == {split: 1}
        assert scope["selection"]["visual_selection"]["model"] == split
        assert (output / scope["qa_links"][0]["path"]).resolve() == roots[split] / "QA.json"
        assert "items" not in scope  # Embed cases once, not in every split view.
    assert all(path.read_bytes() == value for path, value in originals.items())
    assert set(tmp_path.rglob("*.*")) == {*originals, result["html"]}


def _question(task_id: str, action_count: int) -> str:
    actions = ", ".join(
        f"({index}) forward 0.5 m" for index in range(1, action_count + 1)
    )
    convention = (
        "\n\n3D direction convention:\nUse the camera optical center."
        if task_id in {"A4", "B2"} else ""
    )
    return (
        "<image>\nRobot and camera setup:\n"
        "- Ground-plane footprint for navigation and collision checking: "
        "circle, radius 0.2 m\n"
        "- Camera optical-center height above ground: 1 m\n"
        "- Horizontal field of view (HFOV): 79°\n"
        "- Vertical field of view (VFOV): 63.45°\n\n"
        f"Action sequence (execute in order):\n{actions}{convention}\n\n"
        f"Question: Test question for {task_id}?"
    )


def test_build_benchmark_browser_uses_indexed_record_fov_without_images(
        tmp_path: Path):
    benchmark = tmp_path / "seen_8400"
    benchmark.mkdir()
    records_path = tmp_path / "records.jsonl"
    record_lines = []
    byte_offset = 0
    items = []
    index_items = []
    for number, task_id in enumerate(TASKS, 1):
        item_id = f"item-{task_id.lower()}"
        images = [f"images/egocentric/{item_id}.png"]
        if task_id == "C1":
            images.extend(
                f"images/c1/{item_id}_{label}.png" for label in "ABCD"
            )
        items.append({
            "id": item_id,
            "task_id": task_id,
            "dataset": "gs" if number % 2 else "r2r",
            "images": images,
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": _question(task_id, number)},
                {"role": "assistant", "content": f"Answer {task_id}"},
            ],
        })
        usage = {
            "outcome_id": f"outcome-{number}",
            "actions": [
                {"type": "forward", "m": 0.5}
                for _ in range(number)
            ],
            "action_length": number,
            "starts_with": "forward",
            "body_radius_m": 0.2,
            "camera_height_m": 1.0,
            "hfov_deg": 110.0,
            "vfov_deg": 93.93,
            "resolution_px": [640, 480],
        }
        if task_id == "A2":
            usage["collision_action_index_1based"] = 3
            usage["a2_design"] = {
                "cell": {
                    "forward_ordinal_1based": 2,
                    "distance_rank": "shortest",
                },
            }
        if task_id == "A3":
            usage["category_recheck"] = {
                "machine": "chair.n.01", "raw": "chair",
                "canonical": "chair", "aliases": ["seat"],
                "instance_id": 12, "bbox_xyxy_px": [10, 20, 30, 40],
                "mask_area_px": 256,
            }
        if task_id == "A4":
            usage["checkpoint"] = {
                "action_index_1based": 3,
                "forward_ordinal_1based": 2,
                "fraction": 0.5,
            }
            usage["target_point"] = {
                "marked_image_sha256": "marked-image-sha-a4",
                "relation": {"horizontal_direction": "front-left", "vertical_direction": "above"},
            }
        if task_id == "B2":
            usage["target_point"] = {
                "relation": {"horizontal_direction": "rear", "vertical_direction": "below"},
            }
        index_items.append({
            "item_id": item_id,
            "task_id": task_id,
            "dataset": items[-1]["dataset"],
            "scene_id": "scene-one" if number < 5 else "scene-two",
            "record_id": f"record-{number}",
            "record_sha256": f"sha-{number}",
            "initial_image_sha256": f"image-sha-{number}",
            "source": {
                "records_path": str(records_path),
                "byte_offset": byte_offset,
            },
            "usage": usage,
        })
        line = (json.dumps({
            "sensor": {
                "hfov_deg": 110.0, "vfov_deg": 93.93,
                "resolution": [640, 480],
            },
        }) + "\n").encode()
        record_lines.append(line)
        byte_offset += len(line)
    record_bytes = b"".join(record_lines)
    records_path.write_bytes(record_bytes)
    staged_records = tmp_path / "staged-records"
    for dataset in ("gs", "r2r"):
        path = staged_records / dataset / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(record_bytes)
    for entry in index_items:
        entry["source"]["records_path"] = str(tmp_path / "not-published.jsonl")
    (benchmark / "qa.json").write_text(json.dumps(items), encoding="utf-8")
    index_path = tmp_path / "record_index.json"
    index_path.write_text(json.dumps({"items": index_items}), encoding="utf-8")

    output = tmp_path / "reports" / "seen_8400"
    paths = build_benchmark_browser(benchmark, index_path, output)

    assert set(paths) == {"html"}
    assert all(path.is_file() for path in paths.values())
    html = paths["html"].read_text(encoding="utf-8")
    payload = json.loads(html.split('<script id="benchmark-data" type="application/json">')[1].split('</script>')[0])
    assert payload["schema"] == "egoconseq.benchmark-case-browser.v1"
    assert payload["summary"] == {
        "qa_items": 8,
        "records": 8,
        "question_image_references": 8,
        "unique_initial_observations": 8,
        "c1_candidates": 4,
        "image_references": 12,
        "datasets": 2,
        "scenes": 4,
        "tasks": 8,
    }
    assert payload["counts"]["tasks"] == {task: 1 for task in TASKS}
    assert {
        family: [task for task, definition in payload["taxonomy"].items()
                 if definition["family"] == family]
        for family in ("A", "B", "C")
    } == {"A": ["A1", "A2", "A3"],
          "B": ["A4", "B1", "B2", "B3"], "C": ["C1"]}
    for case, original in zip(payload["items"], items):
        assert case["prompt"] == original["messages"][1]["content"]
        assert case["answer"] == original["messages"][2]["content"]
    assert payload["items"][3]["direction"] == "front-left|above"
    assert payload["counts"]["directions"]["A4"]["front-left|above"] == 1
    assert payload["counts"]["directions"]["A4"]["rear|below"] == 0
    assert payload["counts"]["directions"]["B2"]["rear|below"] == 1
    assert len(payload["counts"]["directions"]["A4"]) == 24
    assert payload["counts"]["action_lengths"] == {
        f"L{length}": 1 for length in range(1, 9)
    }
    assert payload["counts"]["a4_checkpoints"] == {
        "fractions": {"50%": 1},
        "action_indices": {"Action 3": 1},
        "forward_ordinals": {"Forward 2": 1},
    }
    assert payload["counts"]["a2_collision_design"] == {
        "action_indices": {"Action 3": 1},
        "forward_ordinals": {"Forward 2": 1},
        "distance_ranks": {"Shortest": 1},
        "joint_cells": {"Forward 2 · shortest": 1},
    }
    assert payload["items"][0]["robot"] == {"radius_m": 0.2}
    assert payload["items"][0]["camera"] == {
        "height_m": 1.0,
        "hfov_deg": 110.0,
        "vfov_deg": 93.93,
    }
    assert payload["items"][0]["image_resolution_px"] == [640, 480]
    assert payload["items"][0]["scene_id"] == "scene-one"
    assert payload["items"][0]["action_sequence"] == "(1) forward 0.5 m"
    assert "direction convention" not in payload["items"][3][
        "action_sequence"].casefold()
    assert payload["items"][3]["checkpoint"] == {
        "action_index": 3,
        "forward_ordinal": 2,
        "fraction": 0.5,
    }
    assert payload["items"][0]["image_version"] == "image-sha-1"
    assert payload["items"][3]["image_version"] == "marked-image-sha-a4"
    evidence = payload["items"][2]["ground_truth_evidence"]
    assert evidence["raw"] == "chair"
    assert evidence["aliases"] == ["seat"]
    assert payload["items"][-1]["images"][1].endswith("_A.png")
    assert payload["image_base"] == "../../seen_8400/"
    assert all(
        case["provenance"]["records_path"].endswith("not-published.jsonl")
        for case in payload["items"])
