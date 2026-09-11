"""Unseen launch plans must preserve scene ownership and evaluation splits."""

import json
from pathlib import Path
from types import SimpleNamespace

def test_jobs_keep_scenes_on_one_worker_and_preserve_source_splits(tmp_path):
    from scripts import collect_unseen_records as launcher
    data = tmp_path / "sources"
    episodes = data / "r2r_vlnce_v1-3/test/test.json.gz"
    import gzip

    episodes.parent.mkdir(parents=True)
    with gzip.open(episodes, "wt") as stream:
        json.dump({"episodes": [{"scene_id": "mp3d/test-a/test-a.glb"},
                                {"scene_id": "mp3d/test-b/test-b.glb"}]}, stream)
    b1k_manifest = tmp_path / "b1k.json"
    b1k_manifest.write_text(json.dumps({"scenes": [
        {"scene_id": scene} for scene in
        ("Merom_0_int", "office_bike", "restaurant_asian", "gates_bedroom")]}))
    gs_manifest = data / "gs/splits/val.json"
    gs_manifest.parent.mkdir(parents=True)
    gs_manifest.write_text(json.dumps({"scenes": [
        {"scene_id": "gs-val", "split": "val"},
        {"scene_id": "gs-train", "split": "train"}]}))
    args = SimpleNamespace(root=tmp_path / "unseen", data_root=data,
                           b1k_source_manifest=b1k_manifest,
                           python=Path("/python"), b1k_python=Path("/b1k-python"),
                           gpus=[0, 1, 2, 3],
                           targets=[30, 20, 10], seed=7)

    jobs = launcher.build_jobs(args)

    assert [job["dataset"] for job in jobs] == ["b1k", "b1k", "r2r", "gs"]
    b1k_scenes = [scene for job in jobs[:2] for scene in job["scenes"]]
    assert sorted(b1k_scenes) == ["Merom_0_int", "office_bike", "restaurant_asian"]
    assert len(set(b1k_scenes)) == len(b1k_scenes)
    assert jobs[2]["scenes"] == ["test-a", "test-b"]
    assert jobs[3]["scenes"] == ["gs-val"]
    for job, split in zip(jobs, ("train", "train", "test", "val")):
        command = job["command"]
        assert command[command.index("--source-split") + 1] == split
        assert command[command.index("--benchmark-partition") + 1] == "test_unseen"
        assert "--append-records" not in command
        assert Path(job["output_dir"]).is_relative_to(args.root)
        assert command[command.index("--oracle-evaluation") + 1] == "nominal"
    assert "CUDA_VISIBLE_DEVICES" not in jobs[0]["environment"]
    assert jobs[0]["environment"]["OMNIGIBSON_GPU_ID"] == "0"
    assert jobs[2]["environment"]["CUDA_VISIBLE_DEVICES"] == "2"


def test_final_catalog_references_shards_without_copying_records(tmp_path):
    from scripts import collect_unseen_records as launcher
    jobs = []
    for dataset, worker in (("b1k", "b1k-0"), ("b1k", "b1k-1"),
                            ("r2r", "r2r"), ("gs", "gs")):
        worker_dir = tmp_path / dataset / worker
        out = worker_dir / "scene"
        out.mkdir(parents=True)
        (out / "records.jsonl").write_text('{"dataset":"' + dataset + '"}\n')
        (out / "run_meta.json").write_text(json.dumps({"dataset": dataset, "record_count": 1}))
        (worker_dir / "collection.json").write_text(json.dumps({
            "scenes": {"scene": {"status": "completed"}}}))
        jobs.append({"dataset": dataset, "output_dir": str(worker_dir), "scenes": ["scene"]})

    stale = tmp_path / "b1k/b1k-0/stale"
    stale.mkdir()
    (stale / "records.jsonl").write_text('{}\n')
    (stale / "run_meta.json").write_text('{"dataset":"b1k","record_count":1}')

    value = launcher.finish_catalog(tmp_path, jobs)

    assert value["split"] == "test/unseen"
    assert value["record_count"] == 4
    assert len(value["datasets"][0]["shards"]) == 2
    assert len(list(tmp_path.rglob("records.jsonl"))) == 5
