"""A3 repair must never rebuild a record or change other task outputs."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest


def test_a3_patch_preserves_every_other_record_field():
    from post_QA.seen_build.a3_repair import apply_updates

    record = {"record_uid": "one", "pose": {"position": [1, 2, 3]},
              "sensor": {"hfov_deg": 79}, "image_path": "original.png",
              "cases": [{"case_id": "c", "actions": [{"type": "forward", "m": 2}],
                         "collision": True, "contact": {"category": "cabinet.n.03"},
                         "task_outputs": {"A1": {"answer": "collision"},
                                          "A3": {"instance_id": 7, "category": "cabinet.n.03"}}}]}
    before = copy.deepcopy(record)
    updates = {"c": {"instance_id": 7, "category": "cabinet.n.03", "source_category": "locker"}}
    result = apply_updates(record, updates)
    assert result["cases"][0]["task_outputs"]["A3"]["source_category"] == "locker"
    result["cases"][0]["task_outputs"]["A3"] = before["cases"][0]["task_outputs"]["A3"]
    assert result == before
    assert record == before


def test_interrupted_review_retry_keeps_its_targets(tmp_path, monkeypatch):
    from post_QA.seen_build import a3_repair

    status = tmp_path / "status.json"
    status.write_text(json.dumps({"phase": "replaying", "targets": {"record": ["case"]}}))
    monkeypatch.setattr(a3_repair, "plan", lambda *args, **kwargs: [])
    def interrupted(*args, **kwargs):
        raise RuntimeError("worker interrupted")
    monkeypatch.setattr(a3_repair, "run_jobs", interrupted)
    args = SimpleNamespace(work=tmp_path, seen_records=tmp_path, unseen_records=tmp_path,
                           gpus=[0], b1k_python="python", b1k_data_root=tmp_path)
    with pytest.raises(RuntimeError, match="worker interrupted"):
        a3_repair.run(args)
    assert json.loads(status.read_text())["targets"] == {"record": ["case"]}


@pytest.mark.parametrize("browser_failure", [False, True])
def test_publish_keeps_prompts_other_tasks_and_unsorted_record_bindings(tmp_path, monkeypatch, browser_failure):
    from pipeline.candidate_sources import canonical_sha256
    from post_QA.seen_build import a3_repair
    from visualization import benchmark_browser

    benchmark, training, work = tmp_path / "benchmark", tmp_path / "sft", tmp_path / "work"
    work.mkdir()
    roots, jobs, snapshots = {}, [], {}

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def item(uid, task, answer):
        return {"id": uid, "task_id": task, "dataset": "b1k", "images": ["images/one.png"],
                "messages": [{"role": "system", "content": "untouched system"},
                             {"role": "user", "content": "untouched question/config"},
                             {"role": "assistant", "content": answer}]}

    frozen = {"items": [], "splits": {"seen": {}, "unseen": {}}}
    for split, folder in (("seen", "seen"), ("unseen", "unseen")):
        root = roots[split] = tmp_path / split
        # Keep insertion order deliberately noncanonical, as in actual historical records.
        rows = [{"record_uid": split + "-" + kind, "scene_id": "one", "dataset": "b1k",
                 "pose": {"position": [0, 0, 0]}, "visible_entities": [],
                 "cases": [{"case_id": "c", "collision": True, "actions": [{"type": "forward", "m": 1}],
                            "task_outputs": {"A1": {"answer": "collision"},
                                             "A3": {"instance_id": 7, "category": "cabinet.n.03"}}}]}
                for kind in ("fixed", "unchanged", "training")]
        path = root / "b1k/records.jsonl"
        path.parent.mkdir(parents=True)
        raw_lines = [(json.dumps(row) + "\n").encode() for row in rows]
        path.write_bytes(b"".join(raw_lines))
        snapshots[path] = path.read_bytes()
        write(root / "b1k/run_meta.json", {"record_count": 3})
        write(root / "manifest.json", {"datasets": [{"dataset": "b1k"}]})
        job = work / f"{split}.job.json"
        planned, results = [], []
        for row, line in zip(rows, raw_lines):
            digest = hashlib.sha256(line).hexdigest()
            planned.append({"record_uid": row["record_uid"], "source_record_sha256": digest})
            output = {**row["cases"][0]["task_outputs"]["A3"], "source_category": "locker"}
            results.append({"record_uid": row["record_uid"], "source_record_sha256": digest,
                            "updates": {} if row["record_uid"].endswith("unchanged") else {"c": output}, "issues": []})
        write(job, {"rows": planned, "scene_id": "one", "split": split})
        job.with_suffix(".results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
        jobs.append(job)
        qa = [item(split + "-fixed", "A3", "cabinet"), item(split + "-unchanged", "A1", "collision")]
        entries = [{"item_id": q["id"], "record_uid": q["id"], "task_id": q["task_id"],
                    "dataset": "b1k", "scene_id": "one", "source": {},
                    "usage": {"outcome_id": "c", "actions": [{"type": "forward", "m": 1}],
                              "action_length": 1, "starts_with": "forward", "body_radius_m": .2,
                              "camera_height_m": 1., "hfov_deg": 79., "vfov_deg": 63.5,
                              "resolution_px": [640, 480]}} for q in qa]
        write(benchmark / "benchmark" / folder / "QA.json", qa)
        write(benchmark / "metadata" / folder / "record_index.json", {"items": entries})
        write(benchmark / "metadata" / folder / "report.json", {"answer_buckets": {"b1k/A3/cabinet": 1}})
        frozen["items"].extend({"record_uid": q["id"]} for q in qa)
    write(benchmark / "metadata/frozen.json", frozen)
    meta = {"id": "train-item", "record_uid": "seen-training", "case_id": "c", "task_id": "A3",
            "dataset": "b1k", "byte_offset": 1, "answer": "cabinet", "answer_bucket": "cabinet",
            "action_length": 1, "starts_with": "forward", "template_id": "A3_01", "images": ["images/one.png"]}
    path = training / "metadata/selection.jsonl"
    write(path, meta)
    write(training / "metadata/summary.json", {"prompts": {}, "distributions": {}, "sources": []})
    for name in ("full", "no_height", "no_radius", "no_fov", "no_parameters"):
        value = item("train-item", "A3", "cabinet")
        value["messages"][1]["content"] += name
        write(training / "json" / f"{name}.jsonl", value)
    if browser_failure:
        def fail(*args, **kwargs):
            raise RuntimeError("browser failed before publish")
        monkeypatch.setattr(benchmark_browser, "build_combined_benchmark_browser", fail)
    args = SimpleNamespace(work=work, benchmark_root=benchmark, sft_root=training,
                           seen_records=roots["seen"], unseen_records=roots["unseen"])
    if browser_failure:
        with pytest.raises(RuntimeError, match="browser failed"):
            a3_repair.publish(args, jobs)
        for path, payload in snapshots.items():
            assert path.read_bytes() == payload
        assert json.loads((training / "json/full.jsonl").read_text())["messages"][-1]["content"] == "cabinet"
        return
    report = a3_repair.publish(args, jobs)
    assert report["sft"]["a3_answers_changed"] == 1
    for split, folder in (("seen", "seen"), ("unseen", "unseen")):
        qa = json.loads((benchmark / "benchmark" / folder / "QA.json").read_text())
        assert qa[0]["messages"][-1]["content"] == "locker"
        assert qa[0]["messages"][1]["content"] == "untouched question/config"
        assert qa[1] == item(split + "-unchanged", "A1", "collision")
        index = json.loads((benchmark / "metadata" / folder / "record_index.json").read_text())
        with (roots[split] / "b1k/records.jsonl").open("rb") as stream:
            for entry in index["items"]:
                stream.seek(entry["source"]["byte_offset"])
                record = json.loads(stream.readline())
                assert record["record_uid"] == entry["record_uid"]
                assert entry["record_sha256"] == canonical_sha256(record)
    for name in ("full", "no_height", "no_radius", "no_fov", "no_parameters"):
        value = json.loads((training / "json" / f"{name}.jsonl").read_text())
        assert value["messages"][-1]["content"] == "locker"
        assert value["messages"][1]["content"] == "untouched question/config" + name
    assert not list(work.glob("*.job.json"))
    retry = a3_repair.plan(roots["seen"], roots["unseen"], work,
                           targets={"seen-fixed": {"c"}})
    assert len(retry) == 1
    retry_row = json.loads(retry[0].read_text())["rows"][0]
    assert retry_row["record_uid"] == "seen-fixed"
    assert retry_row["case_ids"] == ["c"]
    with (roots["seen"] / "b1k/records.jsonl").open("rb") as stream:
        stream.seek(retry_row["byte_offset"])
        assert hashlib.sha256(stream.readline()).hexdigest() == retry_row["source_record_sha256"]

    # Withdraw only the audited A3 outputs, not other QA from the same record.
    audit_path = work.parent / "a3_category_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["review"] = [
        {"record_uid": uid, "case_id": "c", "reason": "structural_asset_needs_review"}
        for uid in ("seen-fixed", "seen-training")]
    write(audit_path, audit)
    keep = item("keep-training", "A1", "collision")
    keep_meta = {**meta, "id": keep["id"], "task_id": "A1", "answer": "collision"}
    with (training / "metadata/selection.jsonl").open("a") as stream:
        stream.write(json.dumps(keep_meta) + "\n")
    for path in (training / "json").glob("*.jsonl"):
        with path.open("a") as stream:
            stream.write(json.dumps(keep) + "\n")
    qa_path = benchmark / "benchmark/seen/QA.json"
    qa = json.loads(qa_path.read_text())
    qa[0]["images"] = ["images/withdrawn.png"]
    write(qa_path, qa)
    image_dir = qa_path.parent / "images"
    image_dir.mkdir()
    (image_dir / "one.png").write_bytes(b"retained image")
    (image_dir / "withdrawn.png").write_bytes(b"withdrawn image")
    before = [json.loads(line) for line in (roots["seen"] / "b1k/records.jsonl").read_text().splitlines()]
    report = a3_repair.publish(args, [], drop_review=True)
    after = [json.loads(line) for line in (roots["seen"] / "b1k/records.jsonl").read_text().splitlines()]
    for row in before:
        if row["record_uid"] in {"seen-fixed", "seen-training"}:
            del row["cases"][0]["task_outputs"]["A3"]
    assert after == before
    assert report["removed_cases"] == 2
    assert report["review_cases"] == 0
    assert json.loads((benchmark / "benchmark/seen/QA.json").read_text()) == [
        item("seen-unchanged", "A1", "collision")]
    assert len(json.loads((benchmark / "benchmark/unseen/QA.json").read_text())) == 2
    assert json.loads((benchmark / "metadata/seen/record_index.json").read_text())["record_count"] == 1
    frozen = json.loads((benchmark / "metadata/frozen.json").read_text())
    assert frozen["total"] == 3
    assert frozen["splits"]["seen"]["qa_count"] == 1
    assert "seen-fixed" in {row["record_uid"] for row in frozen["items"]}
    for path in (training / "json").glob("*.jsonl"):
        assert [json.loads(line) for line in path.read_text().splitlines()] == [keep]
    summary = json.loads((training / "metadata/summary.json").read_text())
    assert summary["qa_items"] == 1
    assert summary["task_totals"] == {"A1": 1}
    assert '"id":"seen-fixed"' not in (benchmark / "index.html").read_text()
    assert not (image_dir / "withdrawn.png").exists()
    assert (image_dir / "one.png").read_bytes() == b"retained image"
    assert a3_repair.publish(args, [], drop_review=True)["removed_cases"] == 2
