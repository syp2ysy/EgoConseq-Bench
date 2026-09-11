import json
from pathlib import Path

from pipeline import collection_cli, io_utils


def test_expansion_appends_to_existing_catalog_and_resumes_without_duplicates(
        tmp_path, monkeypatch):
    from pipeline import collection_expansion as expansion

    root = tmp_path / "current"
    dataset = root / "b1k"
    dataset.mkdir(parents=True)
    records = dataset / "records.jsonl"
    old = {"dataset": "b1k", "scene_id": "gates_bedroom", "record_uid": "old",
           "pose": {"position": [0, 0, 0], "yaw_rad": 0}, "cases": []}
    original = json.dumps(old).encode() + b"\n"
    records.write_bytes(original)
    (dataset / "run_meta.json").write_text('{"record_count":1}')
    gs = {"dataset": "gs", "record_count": 20, "records_sha256": "unchanged"}
    (root / "manifest.json").write_text(json.dumps({
        "record_count": 21, "datasets": [
            {"dataset": "b1k", "record_count": 1, "records_path": "b1k/records.jsonl"},
            gs]}))
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"scenes": [
        {"scene_id": scene} for scene in
        ("gates_bedroom", "office_large", "hall_conference_large", "office_bike")]}))
    opened = []

    def collect(gpu, job, *, root, **kwargs):
        scene = job["scene_id"]
        opened.append(scene)
        assert scene != "office_bike"  # Test/unseen must never enter this run.
        for i in range(20):
            record = {"dataset": "b1k", "scene_id": scene,
                      "record_uid": f"{scene}-{i}", "cases": [],
                      "pose": {"position": [2 * (i + 1), 0, 0], "yaw_rad": 0}}
            if not collection_cli.append_compact_records(
                    root / "b1k/records.jsonl", [record],
                    progress_path=root / "b1k/expansion.json"):
                break
        return 0

    monkeypatch.setattr(expansion, "collect_scene", collect)
    options = dict(root=root, target_records=9, gpus=[0, 1, 2, 3],
                   python=Path("python"), data_root=tmp_path,
                   source_manifest=manifest, seed=123)
    result = expansion.run_expansion(**options)
    assert result["phase"] == "complete"
    assert result["record_count"] == 9
    assert records.read_bytes().startswith(original)
    assert len(opened) == len(set(opened))
    assert list(root.rglob("records.jsonl")) == [records]
    final = json.loads((root / "manifest.json").read_text())
    assert final["record_count"] == 29
    assert final["datasets"][1] == gs
    assert final["datasets"][0]["records_sha256"] == io_utils.sha256_file(records)
    assert json.loads((dataset / "run_meta.json").read_text())["record_count"] == 9
    before = records.read_bytes()
    assert expansion.run_expansion(**options)["phase"] == "complete"
    assert records.read_bytes() == before


def test_expansion_seal_keeps_disjoint_scene_byte_ranges(tmp_path):
    from pipeline import collection_expansion as expansion

    path = tmp_path / "records.jsonl"
    rows = [{"scene_id": scene} for scene in ("a", "b", "a")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with path.open("rb") as stream:
        counts, ranges, digest = expansion.scan_records(stream)
    assert counts == {"a": 2, "b": 1}
    assert [row["scene_id"] for row in ranges] == ["a", "b", "a"]
    with path.open("rb") as stream:
        for row in ranges:
            stream.seek(row["start_byte"])
            assert json.loads(stream.read(row["end_byte"] - row["start_byte"]))[
                "scene_id"] == row["scene_id"]
    assert digest == io_utils.sha256_file(path)


def test_worker_retains_run_lock_after_coordinator_closes_it(tmp_path, monkeypatch):
    import fcntl
    import subprocess
    import sys
    import pytest
    from pipeline import collection_expansion as expansion

    (tmp_path / "b1k").mkdir()
    lock_path = tmp_path / "b1k/expansion.lock"
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    descriptor = lock.fileno()

    def run(_command, **kwargs):
        process = subprocess.Popen([
            sys.executable, "-c",
            "import os,sys; os.fstat(int(sys.argv[1])); print('ready',flush=True); "
            "sys.stdin.read(1)", str(descriptor)],
            pass_fds=kwargs.get("pass_fds", ()), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert process.stdout.readline().strip() == "ready"
            lock.close()
            with lock_path.open("a") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            process.communicate("x", timeout=5)
            lock.close()
        return subprocess.CompletedProcess([], process.returncode)

    monkeypatch.setattr(expansion.subprocess, "run", run)
    assert expansion.collect_scene(
        0, {"scene_id": "room", "shard_id": "run-room", "poses": 1},
        root=tmp_path, python=Path(sys.executable), data_root=tmp_path,
        source_manifest=tmp_path / "source.json", seed=1, lock_fd=descriptor) == 0
    with lock_path.open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
