import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
import pytest

from pipeline import abc1_record, benchmark_candidates
from post_QA import templates
from post_QA.seen_build import sft
from tests.test_pl_seen_compact_artifact import _record


@pytest.mark.parametrize("legacy_height", [False, True])
def test_height_sync_preserves_sources_qa_gt_and_images(tmp_path, legacy_height):
    from pipeline import candidate_sources, config, io_utils
    from post_QA.seen_build import artifact, catalog, freeze, parameters

    benchmark = tmp_path / "benchmark"
    roots = []
    for split, name in (("seen", "seen"), ("unseen", "unseen")):
        root = tmp_path / split
        assets = root / "b1k"
        record = abc1_record.normalize(_record(assets))
        record.update(dataset="b1k", record_uid=f"{split}-heldout")
        record["sensor"].update(camera_height_above_floor_m=-0.04,
                                vfov_deg=config.vfov_for_hfov(79))
        path = assets / "records.jsonl"
        records = [record]
        path.write_text("".join(json.dumps(r)+"\n" for r in records))
        (assets/"run_meta.json").write_text(json.dumps({"record_count": len(records)}))
        (root/"manifest.json").write_text(json.dumps({"split": split, "datasets": [{
            "dataset": "b1k", "records_path": "b1k/records.jsonl"}]}))
        if split == "seen":
            extra = root / "b1k-extra"
            _record(extra)
            extra_path = extra / "records.jsonl"
            extra_path.write_text(json.dumps(dict(record, record_uid="training"))+"\n")
            (extra/"run_meta.json").write_text(json.dumps({"record_count": 1}))
            catalog.write([{"path": str(p), "dataset": "b1k"} for p in (path, extra_path)],
                          root/"manifest.json", split=split)
        candidate = replace(benchmark_candidates.project_record(
            dataset="b1k", source_path=str(path), byte_offset=0,
            record_sha256=candidate_sources.canonical_sha256(record), record=record,
            allowed_tasks=("A1",))[0], template_id="A1_01")
        folder = benchmark/"benchmark"/name
        metadata = benchmark/"metadata"/name
        folder.mkdir(parents=True)
        item, entry = artifact.materialize_item(candidate, catalog.load(root)[0], record,
                                               folder, tmp_path/"marked")
        item["messages"][1]["content"] = item["messages"][1]["content"].replace(
            "Camera optical-center height relative to the local ground reference: 1.0 m",
            ("Camera optical-center height above ground: 1.02 m" if legacy_height else
             "Camera optical-center height relative to the local ground reference: 1.02 m"))
        entry["usage"]["camera_height_m"] = 1.02
        io_utils.atomic_write_json(folder/"QA.json", [item])
        artifact.write_index([entry], folder/"QA.json", metadata/"record_index.json")
        (metadata/"report.json").write_text("{}")
        roots.append(root)
    freeze.freeze(benchmark)
    training = tmp_path/"sft"
    sft.compile_records(roots[0], training, exclude_index=benchmark/"metadata/frozen.json", workers=1)
    selection_path = training/"metadata/selection.jsonl"
    selection = [json.loads(line) for line in selection_path.read_text().splitlines()]
    for row in selection:
        row["inputs"]["camera"]["optical_center_height_m"] = 1.02
    selection_path.write_text("".join(json.dumps(row)+"\n" for row in selection))
    sft.export_view(training)
    sft.export_view(training, hide_params=("height",))
    before = [json.loads(line) for line in (training/"json/full.jsonl").read_text().splitlines()]
    image_hashes = {p: io_utils.sha256_file(p) for p in training.glob("images/**/*.png")}
    sources = [s for root in roots for s in catalog.load(root)]
    source_hashes = {s.records_path: io_utils.sha256_file(s.records_path) for s in sources}
    report = parameters.repair(*roots, benchmark, training)
    after = [json.loads(line) for line in (training/"json/full.jsonl").read_text().splitlines()]
    assert report["records_unchanged"]
    assert source_hashes == {path: io_utils.sha256_file(path) for path in source_hashes}
    assert [(r["id"],r["images"],r["messages"][-1]) for r in before] == [
        (r["id"],r["images"],r["messages"][-1]) for r in after]
    assert all("Camera optical-center height relative to the local ground reference: 1.0 m"
               in r["messages"][1]["content"] for r in after)
    assert "Camera optical-center height" not in (training/"json/no_height.jsonl").read_text()
    for name in ("seen", "unseen"):
        rows = json.loads((benchmark/"benchmark"/name/"QA.json").read_text())
        assert "Camera optical-center height relative to the local ground reference: 1.0 m" in rows[0]["messages"][1]["content"]
    assert {p: io_utils.sha256_file(p) for p in image_hashes} == image_hashes
    sources = catalog.load(roots[0])
    assert report["datasets"]["seen/b1k"]["records"] == 2
    for source in sources:
        assert source.records_sha256 == io_utils.sha256_file(source.records_path)
    for line in (training/"metadata/selection.jsonl").read_text().splitlines():
        row = json.loads(line)
        source = sources[row["source_index"]]
        with source.records_path.open("rb") as stream:
            stream.seek(row["byte_offset"])
            record = json.loads(stream.readline())
        assert record["record_uid"] == row["record_uid"]
        assert record["sensor"]["nominal_camera_offset_m"] == 1.0
        assert row["inputs"]["robot"]["radius_m"] == 0.2


def test_sft_covers_partial_records_excludes_holdout_and_shares_images(tmp_path):
    source_root = tmp_path / "source"
    record = abc1_record.normalize(_record(source_root))
    record["c1_families"].append({
        "query_case_id": "safe-1", "member_case_ids": [f"safe-{i}" for i in range(4)]})
    collision = record["cases"][-1]
    record["cases"].append(dict(
        collision, case_id="collision-first", group_id="collision-first",
        actions=[{"type": "forward", "m": 4.0}, {"type": "turn", "deg": 15.0},
                 {"type": "forward", "m": 1.0}],
        task_outputs={"A2": {"collision_action_index_1based": 1,
                             "forward_ordinal_1based": 1, "distance_rank": "longest"}}))
    partial = dict(record, record_uid="partial", cases=[collision],
                   c1_families=[])
    held_out = dict(record, record_uid="benchmark")
    records_path = source_root / "records.jsonl"
    records_path.write_text("".join(json.dumps(r) + "\n"
                                    for r in (record, partial, held_out)))
    (source_root / "manifest.json").write_text(json.dumps({"datasets": [{
        "dataset": "r2r", "records_path": "records.jsonl"}]}))
    exclude = tmp_path / "exclude.json"
    exclude.write_text(json.dumps({"items": [{"record_uid": "benchmark"}]}))
    original = records_path.read_bytes()
    root = tmp_path / "sft"

    report = sft.compile_records(source_root, root, exclude_index=exclude, workers=1)
    selected = [json.loads(line) for line in
                (root / "metadata/selection.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (root / "json/full.jsonl").read_text().splitlines()]

    assert {r["record_uid"] for r in selected} == {"r2r-record", "partial"}
    assert report["excluded_records"] == 1
    assert report["covered_records"] == 2
    assert {r["task_id"] for r in rows} == {"A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1"}
    assert max(Counter((r["record_uid"], r["task_id"]) for r in selected).values()) <= 2
    marked = [r["images"][0] for r in rows if r["task_id"] in {"A4", "B1", "B2", "B3"}]
    assert len(set(marked)) == 1
    for row in selected:
        if row["task_id"] == "B3":
            checkpoint = row["inputs"]["checkpoint"]
            forward = row["inputs"]["actions"][checkpoint["action_index"] - 1]
            distance = (4 + (4 - forward["meters"] * checkpoint["fraction"]) ** 2) ** .5
            assert abs(float(row["answer"].split()[0]) - distance) <= .005
    for row in rows:
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert row["messages"][1]["content"].count("<image>") == len(row["images"])
        assert all(not Path(p).is_absolute() and (root / p).is_file() for p in row["images"])
        assert "world_xyz" not in row["messages"][1]["content"]
    assert records_path.read_bytes() == original
    assert set(p.name for p in root.iterdir()) == {"images", "json", "metadata"}
    assert not list(tmp_path.glob(".sft.*"))

    sft.export_view(root, hide_params=("height", "radius", "fov"))
    hidden = [json.loads(line) for line in
              (root / "json/no_parameters.jsonl").read_text().splitlines()]
    assert len(rows) == len(hidden)
    for full, ablated in zip(rows, hidden):
        assert full["id"] == ablated["id"]
        assert full["images"] == ablated["images"]
        assert full["messages"][2] == ablated["messages"][2]
        question = ablated["messages"][1]["content"]
        assert "field of view" not in question
        assert "Camera optical-center height" not in question and "radius" not in question
        assert "Configuration:" not in question and "forward" in question
        assert "circular" in ablated["messages"][0]["content"]
        if full["task_id"] == "A4":
            assert "50%" in question
        if full["task_id"] == "B3":
            assert "3D straight-line distance" in question
            assert "%" in question

    moved = tmp_path / "moved-sft"
    root.rename(moved)
    for filename in ("full.jsonl", "no_parameters.jsonl"):
        with (moved / "json" / filename).open() as stream:
            for line in stream:
                assert all((moved / path).is_file() for path in json.loads(line)["images"])


def test_sft_keeps_distinct_b3_checkpoints_without_duplicating_a_question(tmp_path):
    from post_QA.seen_build import sft_selection

    record = abc1_record.normalize(_record(tmp_path))
    base = benchmark_candidates.project_record(
        dataset="r2r", source_path="records.jsonl", byte_offset=0,
        record_sha256="", record=record, allowed_tasks=("B3",))[0]
    rows = [replace(base, checkpoint_fraction=fraction, item_id=f"b3-{fraction}")
            for fraction in (.25, .75)]
    selected, _ = sft_selection.select([rows + rows], seed=7)
    assert {r.checkpoint_fraction for r in selected} == {.25, .75}
    assert len(selected) == 2


def test_sft_selection_cannot_fill_a2_with_only_longest_forward(tmp_path):
    from post_QA.seen_build import sft_selection

    record = abc1_record.normalize(_record(tmp_path))
    base = next(c for c in benchmark_candidates.project_record(
        dataset="r2r", source_path="records.jsonl", byte_offset=0,
        record_sha256="", record=record, allowed_tasks=("A1",))
        if c.answer_bucket == "collision")
    records = []
    for i in range(30):
        a1 = replace(base, record_id=f"r{i}", item_id=f"a1-{i}")
        safe = replace(a1, item_id=f"safe-{i}", answer_bucket="no_collision",
                       action_signature="safe")
        a2 = replace(a1, item_id=f"a2-{i}", task_id="A2", a2_ordinal=1,
                     a2_rank="shortest" if i < 3 else "longest")
        records.append([a1, safe, a2])
    selected, _ = sft_selection.select(records, seed=7)

    assert {r.record_id for r in selected} == {f"r{i}" for i in range(30)}
    ranks = Counter(r.a2_rank for r in selected if r.task_id == "A2")
    assert ranks["shortest"] == ranks["longest"] == 3
    labels = Counter(r.answer_bucket for r in selected if r.task_id == "A1")
    assert abs(labels["collision"] - labels["no_collision"]) <= 1


def test_hiding_one_parameter_does_not_hide_others_or_action_units():
    inputs = {"camera": {"optical_center_height_m": 1.5, "hfov_deg": 79,
                         "vfov_deg": 63.45}, "robot": {"radius_m": .2},
              "actions": [{"index": 1, "text": "forward 1.5 m"}]}
    question = templates.render_question(
        "A1", inputs, templates.QUESTION_TEMPLATES["A1"][0],
        hide_params=("height",))
    assert "Camera optical-center height" not in question
    assert "radius: 0.2 m" in question
    assert "Horizontal field of view" in question and "Vertical field of view" in question
    assert "forward 1.5 m" in question


def test_ablation_copies_saved_full_without_rerendering_or_selection(tmp_path):
    root = tmp_path / "sft"
    (root / "json").mkdir(parents=True)
    before = "Your initial view:\n<image>\n"
    radius = "- Collision footprint radius: 0.2 m"
    height = "- Camera optical-center height relative to the local ground reference: 1.5 m"
    hfov = "- Horizontal field of view: 79°"
    vfov = "- Vertical field of view: 63.45°"
    config = "Configuration:\n" + "\n".join((radius, height, hfov, vfov)) + "\n\n"
    after = "Actions:\n(1) forward 1.5 m\n\nQuestion: Keep this saved wording exactly.\n"
    row = {"id": "frozen-qa", "task_id": "A1", "dataset": "r2r",
           "images": ["images/view.png"], "extra_metadata": "preserve me",
           "messages": [{"role": "system", "content": "Keep this saved system exactly."},
                        {"role": "user", "content": before + config + after},
                        {"role": "assistant", "content": "No collision."}]}
    path = root / "json/full.jsonl"
    original = json.dumps(row) + "\n"
    path.write_text(original)
    for hidden, name, kept in (
        (("height",), "no_height", (radius, hfov, vfov)),
        (("radius",), "no_radius", (height, hfov, vfov)),
        (("fov",), "no_fov", (radius, height)),
        (("height", "radius", "fov"), "no_parameters", ()),
    ):
        sft.export_view(root, hide_params=hidden)
        actual = json.loads((root / f"json/{name}.jsonl").read_text())
        expected = dict(row, messages=[dict(message) for message in row["messages"]])
        configuration = "Configuration:\n" + "\n".join(kept) + "\n\n" if kept else ""
        expected["messages"][1]["content"] = before + configuration + after
        assert actual == expected
    assert path.read_text() == original
    assert set(root.iterdir()) == {root / "json"}


def test_sft_rank_balance_accounts_for_shared_record_capacity(tmp_path):
    from post_QA.seen_build import sft_selection

    record = abc1_record.normalize(_record(tmp_path))
    base = benchmark_candidates.project_record(
        dataset="r2r", source_path="records.jsonl", byte_offset=0,
        record_sha256="", record=record, allowed_tasks=("A1",))[0]
    records = []
    groups = ((3, "forward"), (4, "forward"), (4, "turn"), (5, "turn"))
    for index in range(41):
        rows = [replace(base, record_id=f"r{index}", item_id=f"a1-{index}-{answer}",
                        answer_bucket=answer, action_signature=answer)
                for answer in ("collision", "no_collision")]
        for length, start in groups:
            rank = "shortest" if index == 0 else "longest"
            rows.append(replace(
                base, record_id=f"r{index}", item_id=f"a2-{index}-{length}-{start}",
                task_id="A2", action_length=length, starts_with=start,
                a2_ordinal=1, a2_rank=rank, action_signature=f"{length}-{start}"))
        records.append(rows)
    selected, _ = sft_selection.select(records, seed=20260906)
    assert len({row.record_id for row in selected}) == 41
    for length, start in groups:
        ranks = Counter(row.a2_rank for row in selected
                        if (row.task_id, row.action_length, row.starts_with) ==
                        ("A2", length, start))
        assert ranks["shortest"] == ranks["longest"]


@pytest.mark.parametrize("failure", ["broken", "dark"])
def test_bad_images_do_not_abort_sft_or_export_broken_c1_options(tmp_path, failure):
    from PIL import Image

    source = tmp_path / "source"
    valid = abc1_record.normalize(_record(source))
    broken = dict(valid, record_uid="bad-image", image_path="bad.png")
    if failure == "broken":
        (source / "bad.png").write_bytes(b"not a PNG")
        (source / "terminal/0.png").write_bytes(b"broken terminal")
    else:
        for path in (source / "bad.png", source / "terminal/0.png"):
            Image.new("RGB", (32, 24), (2, 3, 4)).save(path)
    (source / "records.jsonl").write_text(json.dumps(valid) + "\n" + json.dumps(broken) + "\n")
    (source / "manifest.json").write_text(json.dumps({"datasets": [{
        "dataset": "r2r", "records_path": "records.jsonl"}]}))
    exclude = tmp_path / "excluded.json"
    exclude.write_text('{"items": []}')

    report = sft.compile_records(source, tmp_path / "sft", exclude_index=exclude, workers=1)
    rows = [json.loads(line) for line in
            (tmp_path / "sft/metadata/selection.jsonl").read_text().splitlines()]
    assert report["covered_records"] == 1
    assert {row["record_uid"] for row in rows} == {"r2r-record"}
    assert "C1" not in {row["task_id"] for row in rows}
    assert {row["kind"] for row in report["unreadable_images"]} == {"initial", "terminal"}
    assert all(not Path(row["path"]).is_absolute() for row in report["unreadable_images"])
    assert {p.relative_to(tmp_path / "sft").as_posix()
            for p in (tmp_path / "sft/images").rglob("*.png")} == {
                p for row in rows for p in row["images"]}


def test_cleanup_removes_dark_c1_only_and_updates_all_exported_category_names(tmp_path):
    from PIL import Image
    from pipeline import io_utils
    from post_QA.seen_build import artifact, catalog, freeze, sft_cleanup

    source = tmp_path / "source"
    record = abc1_record.normalize(_record(source))
    record["cases"][-1]["task_outputs"]["A3"].update(category="lighting", source_category="lighting")
    records_path = source / "records.jsonl"
    records_path.write_text(json.dumps(record) + "\n")
    (source / "manifest.json").write_text(json.dumps({"datasets": [{
        "dataset": "r2r", "records_path": "records.jsonl"}]}))
    benchmark = tmp_path / "benchmark"
    for split, folder in (("seen", "seen"), ("unseen", "unseen")):
        candidate = replace(benchmark_candidates.project_record(
            dataset="r2r", source_path=str(records_path), byte_offset=0,
            record_sha256="", record=record, allowed_tasks=("A3",))[0],
            template_id="A3_01", item_id=split + "-a3")
        target = benchmark / "benchmark" / folder
        metadata = benchmark / "metadata" / folder
        item, entry = artifact.materialize_item(candidate, catalog.load(source)[0], record,
                                                target, tmp_path / "marked")
        item["messages"][-1]["content"] = "lighting"
        entry.update(record_uid=split + "-heldout")
        entry["usage"]["contact_category"] = "lighting"
        # Source identity is looked up only for exported A3 cases; use a distinct heldout record.
        heldout = dict(record, record_uid=split + "-heldout")
        heldout_path = tmp_path / (split + ".jsonl")
        heldout_path.write_text(json.dumps(heldout) + "\n")
        entry["source"]["records_path"] = str(heldout_path)
        io_utils.atomic_write_json(target / "QA.json", [item])
        artifact.write_index([entry], target / "QA.json", metadata / "record_index.json")
        io_utils.atomic_write_json(metadata / "report.json", {"answer_buckets": {"r2r/A3/lighting": 1}})
    freeze.freeze(benchmark)
    training = tmp_path / "sft"
    sft.compile_records(source, training, exclude_index=benchmark / "metadata/frozen.json", workers=1)
    meta_path = training / "metadata/selection.jsonl"
    rows = [json.loads(line) for line in meta_path.read_text().splitlines()]
    c1 = next(r for r in rows if r["task_id"] == "C1")
    dark = training / c1["images"][1]
    dark.unlink()  # The export is a hardlink: never overwrite its source image.
    Image.new("RGB", (32, 24), (2, 3, 4)).save(dark)
    for row in rows:
        if row["task_id"] == "A3":row["answer"] = row["answer_bucket"] = "lighting"
    meta_path.write_text("".join(sft._json(r) for r in rows))
    sft.export_view(training)
    for hidden in (("height",), ("radius",), ("fov",), ("height", "radius", "fov")):
        sft.export_view(training, hide_params=hidden)
    before = {p.name: [json.loads(line) for line in p.read_text().splitlines()]
              for p in (training / "json").glob("*.jsonl")}
    source_bytes = records_path.read_bytes()
    report = sft_cleanup.clean(training, benchmark, workers=1)
    assert report["removed_qa"] == 1
    assert report["dark_images"] == {c1["images"][1]: 1.0}
    assert not dark.exists()
    assert (source / "terminal/0.png").exists()
    assert records_path.read_bytes() == source_bytes
    for name, old in before.items():
        expected = [r for r in old if r["id"] != c1["id"]]
        for r in expected:
            if r["task_id"] == "A3":r["messages"][-1]["content"] = "light fixture"
        assert [json.loads(line) for line in (training / "json" / name).read_text().splitlines()] == expected
    remaining = [json.loads(line) for line in meta_path.read_text().splitlines()]
    assert "A4" in {r["task_id"] for r in remaining}
    assert {p.relative_to(training).as_posix() for p in (training / "images").rglob("*.png")} == {
        p for r in remaining for p in r["images"]}
    assert json.loads((training / "metadata/summary.json").read_text())["qa_items"] == len(rows) - 1
    for folder in ("seen", "unseen"):
        q = json.loads((benchmark / "benchmark" / folder / "QA.json").read_text())
        assert q[0]["messages"][-1]["content"] == "light fixture"
        assert "light fixture" in (benchmark / "index.html").read_text()
    assert not list(training.parent.glob(".sft-clean-*"))
