import json
from collections import Counter

from pipeline import io_utils
from post_QA import templates
from post_QA.seen_build import sft
from tests.test_pl_qa_generation import _inputs


def test_benchmark_ablations_delete_only_saved_configuration_lines(tmp_path):
    from copy import deepcopy
    from post_QA.seen_build import prompts

    before = "Your initial view:\n<image>\n"
    lines = ["- Collision footprint radius: 0.2 m",
             "- Camera optical-center height relative to the local ground reference: 1.5 m",
             "- Horizontal field of view: 79°", "- Vertical field of view: 63.45°"]
    after = "Actions:\n(1) forward 1.5 m\n\nQuestion: Keep the saved wording."
    row = {"id": "same-id", "dataset": "r2r", "task_id": "C1",
           "images": [f"images/{i}.png" for i in range(5)], "extra": "unchanged",
           "messages": [{"role": "system", "content": "Keep this system prompt."},
                        {"role": "user", "content": before + "Configuration:\n" + "\n".join(lines) + "\n\n" + after},
                        {"role": "assistant", "content": "B"}]}
    original = json.dumps([row])
    for split in ("seen", "unseen"):
        folder = tmp_path / "benchmark" / split
        folder.mkdir(parents=True)
        (folder / "QA.json").write_text(original)

    prompts.export_benchmark_views(tmp_path)

    for split in ("seen", "unseen"):
        folder = tmp_path / "benchmark" / split
        for name, kept in (("no_radius", lines[1:]),
                           ("no_height", [lines[0], *lines[2:]]),
                           ("no_fov", lines[:2]), ("no_parameters", [])):
            expected = deepcopy(row)
            configuration = "Configuration:\n" + "\n".join(kept) + "\n\n" if kept else ""
            expected["messages"][1]["content"] = before + configuration + after
            assert json.loads((folder / f"QA_{name}.json").read_text()) == [expected]
        assert (folder / "QA.json").read_text() == original
        assert len(list(folder.iterdir())) == 5


def test_refresh_updates_both_splits_and_sft_without_touching_inputs(tmp_path):
    benchmark, training = tmp_path / "benchmark", tmp_path / "sft"
    frozen = {"splits": {}, "items": [{"record_uid": "holdout"}], "total": 160}
    originals = {}
    for split, name in (("seen", "seen"), ("unseen", "unseen")):
        root = benchmark / "benchmark" / name
        metadata = benchmark / "metadata" / name
        qa, entries = [], []
        for task in templates.TASK_IDS:
            for i in range(10):
                uid = f"{task}-{i}"
                qa.append({"id": uid, "task_id": task, "dataset": "r2r",
                           "images": ["image.png"] * (5 if task == "C1" else 1),
                           "messages": [{"role": "system", "content": "old system"},
                                        {"role": "user", "content": "old question"},
                                        {"role": "assistant", "content": "A" if task == "C1" else "GT"}]})
                usage = {"actions": [{"type": "forward", "m": 0.5},
                                     {"type": "turn", "deg": -30},
                                     {"type": "forward", "m": 1.0}],
                         "action_length": 3, "body_radius_m": .2, "camera_height_m": 1.0,
                         "hfov_deg": 110, "vfov_deg": 93.93, "resolution_px": [640, 480]}
                if task in {"A4", "B3"}:
                    usage["checkpoint"] = {"action_index_1based": 3, "fraction": .5,
                                           "forward_ordinal_1based": 2}
                entries.append({"item_id": uid, "task_id": task, "record_uid": f"{split}-{uid}",
                                "scene_id": "scene", "usage": usage,
                                "source": {"records_path": "absent.jsonl", "byte_offset": i}})
        for path, value in ((root / "QA.json", qa), (metadata / "record_index.json", {"items": entries}),
                            (metadata / "report.json", {})):
            path.parent.mkdir(parents=True, exist_ok=True)
            io_utils.atomic_write_json(path, value)
        image = root / "image.png"
        image.write_bytes(b"no image decoding required")
        originals[split] = (qa, entries, image.read_bytes())
        frozen["splits"][split] = {"qa_sha256": io_utils.sha256_file(root / "QA.json")}
    io_utils.atomic_write_json(benchmark / "metadata/frozen.json", frozen)
    rows = [{"id": f"sft-{i}", "task_id": "C1", "dataset": "r2r", "record_uid": f"sft-record-{i}",
             "template_id": "C1_01", "images": ["image.png"] * 5, "inputs": _inputs(),
             "answer": "ABCD"[i % 4]} for i in range(40)]
    (training / "metadata").mkdir(parents=True)
    (training / "json").mkdir()
    (training / "metadata/selection.jsonl").write_text("".join(sft._json(row) for row in rows))
    io_utils.atomic_write_json(training / "metadata/summary.json", {"prompts": {}, "distributions": {}})
    sft.export_view(training)
    sft.export_view(training, hide_params=("height",))
    summary_path = training / "metadata/summary.json"
    summary = json.loads(summary_path.read_text())
    summary.pop("image_path_base")
    summary["unreadable_images"] = [{"path": str(tmp_path / "missing.png"),
                                     "kind": "initial", "record_uid": "unreadable", "error": "OSError"}]
    io_utils.atomic_write_json(summary_path, summary)

    from post_QA.seen_build import prompts
    prompts.refresh(benchmark, training, seed=19)

    for split, name in (("seen", "seen"), ("unseen", "unseen")):
        root = benchmark / "benchmark" / name
        metadata = benchmark / "metadata" / name
        qa = json.loads((root / "QA.json").read_text())
        entries = json.loads((metadata / "record_index.json").read_text())["items"]
        old_qa, old_entries, pixels = originals[split]
        assert (root / "image.png").read_bytes() == pixels
        counts = Counter(r["usage"].pop("template_id") for r in entries)
        assert len(counts) == 80 and set(counts.values()) == {1}
        assert entries == old_entries
        for old, new in zip(old_qa, qa):
            assert (old["id"], old["images"], old["messages"][-1]) == (new["id"], new["images"], new["messages"][-1])
            assert new["messages"][0]["content"].startswith("You are a mobile robot")
            if new["task_id"] in {"A4", "B3"}:
                assert "50% of the forward distance in action 3" in new["messages"][1]["content"]
        for variant, removed in (
                ("no_radius", ("- Collision footprint radius:",)),
                ("no_height", ("- Camera optical-center height",)),
                ("no_fov", ("- Horizontal field of view:", "- Vertical field of view:")),
                ("no_parameters", ("- Collision footprint radius:", "- Camera optical-center height",
                                   "- Horizontal field of view:", "- Vertical field of view:"))):
            variant_rows = json.loads((root / f"QA_{variant}.json").read_text())
            assert len(variant_rows) == len(qa)
            for full, row in zip(qa, variant_rows):
                assert (row["id"], row["images"], row["messages"][0], row["messages"][-1]) == (
                    full["id"], full["images"], full["messages"][0], full["messages"][-1])
                expected = "\n".join(line for line in full["messages"][1]["content"].split("\n")
                                     if not line.startswith(removed))
                if variant == "no_parameters":
                    expected = expected.replace("Configuration:\n\n", "")
                assert row["messages"][1]["content"] == expected
        sealed = json.loads((benchmark / "metadata/frozen.json").read_text())
        assert sealed["splits"][split]["qa_sha256"] == io_utils.sha256_file(root / "QA.json")
    selected = [json.loads(line) for line in (training / "metadata/selection.jsonl").read_text().splitlines()]
    assert set(Counter(r["template_id"] for r in selected).values()) == {4}
    assert len({r["template_id"] for r in selected}) == 10
    for old, new in zip(rows, selected):
        old.pop("template_id"); new.pop("template_id")
        assert old == new
    full = [json.loads(line) for line in (training / "json/full.jsonl").read_text().splitlines()]
    hidden = [json.loads(line) for line in (training / "json/no_height.jsonl").read_text().splitlines()]
    for a, b in zip(full, hidden):
        assert a["images"] == b["images"] and a["messages"][-1] == b["messages"][-1]
        assert "Camera optical-center height" in a["messages"][1]["content"]
        assert "Camera optical-center height" not in b["messages"][1]["content"]
    assert "old question" not in (benchmark / "index.html").read_text()
    # A wording-only refresh must retain template assignments even with a new seed.
    unchanged = [benchmark / "benchmark" / name / "QA.json"
                 for name in ("seen", "unseen")]
    unchanged += [training / "metadata/selection.jsonl", training / "json/full.jsonl",
                  training / "json/no_height.jsonl"]
    previous = {path: path.read_bytes() for path in unchanged}
    prompts.refresh(benchmark, training, seed=20, keep_templates=True)
    assert all(path.read_bytes() == content for path, content in previous.items())
    summary = json.loads(summary_path.read_text())
    assert summary["image_path_base"] == "."
    assert not summary["unreadable_images"][0]["path"].startswith("/")
    assert not list(benchmark.glob(".prompts-*"))
