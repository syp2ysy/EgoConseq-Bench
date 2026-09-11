"""Official non-train source selection and shard provenance."""

import collections
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from pipeline import (
    collection_cli, collection_funnel, collection_runtime,
    future_view_selection, record, scene_pool,
)
from tests._synthetic import source_provenance


def _write_mp3d_scene(root: Path, scene_id: str) -> None:
    directory = root / scene_id
    directory.mkdir(parents=True)
    for suffix in (".glb", ".navmesh", ".house", "_semantic.ply"):
        (directory / f"{scene_id}{suffix}").write_bytes(b"")


def _write_gs_scene(root: Path, relative: str) -> None:
    directory = root / relative
    directory.mkdir(parents=True)
    for name in (
            "scene.gs.ply", "scene.navmesh", "labels.json",
            "scene.collision.npz"):
        (directory / name).write_bytes(b"[]" if name == "labels.json" else b"")


def test_official_nontrain_manifests_preserve_their_source_splits(tmp_path):
    mp3d = tmp_path / "mp3d"
    mp3d.mkdir()
    (mp3d / "mp3d_annotated_basis.scene_dataset_config.json").write_text("{}")
    _write_mp3d_scene(mp3d, "TEST")
    r2r_manifest = tmp_path / "test.json.gz"
    with gzip.open(r2r_manifest, "wt") as stream:
        json.dump({"episodes": [{"scene_id": "mp3d/TEST/TEST.glb"}]}, stream)

    gs_root = tmp_path / "gs"
    _write_gs_scene(gs_root, "val/interior_val")
    gs_manifest = tmp_path / "val.json"
    gs_manifest.write_text(json.dumps({
        "schema_version": scene_pool.SCENE_MANIFEST_VERSION,
        "dataset": "gs",
        "scenes": [{
            "scene_id": "interior_val", "split": "val",
            "path": "val/interior_val",
        }],
    }))

    r2r = scene_pool.discover_r2r_scenes(
        r2r_manifest, mp3d, source_split="test")
    gs = scene_pool.discover_gs_scenes(
        gs_root, gs_manifest, source_split="val")

    assert [(value.scene_id, value.official_split) for value in r2r] == [
        ("TEST", "test")]
    assert [(value.scene_id, value.official_split) for value in gs] == [
        ("interior_val", "val")]


def test_cli_exposes_source_split_and_one_r2r_episode_destination(tmp_path):
    parser = collection_cli.build_parser()
    current = parser.parse_args([
        "--auto-scenes", "--out", str(tmp_path / "current"),
        "--source-split", "test", "--r2r-episodes", "test.json.gz",
    ])
    legacy = parser.parse_args([
        "--auto-scenes", "--out", str(tmp_path / "legacy"),
        "--r2r-train-episodes", "train.json.gz",
    ])

    assert current.source_split == "test"
    assert current.r2r_episodes == "test.json.gz"
    assert legacy.r2r_episodes == "train.json.gz"
    assert legacy.source_split == "train"


def test_nontrain_source_bindings_remain_hash_bound():
    r2r = source_provenance("r2r-test", dataset="r2r")
    r2r["official_split"] = "test"
    contract = record.r2r_v16_collection_contract(r2r, "main")

    gs = source_provenance("gs-val", dataset="gs")
    gs["official_split"] = "val"
    terminal_binding = future_view_selection._source_binding(
        gs, scene_id="gs-val")

    assert contract["official_split"] == "test"
    assert terminal_binding["official_split"] == "val"


def test_unseen_finalize_labels_run_meta_and_shard_catalog(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("")
    args = SimpleNamespace(
        out=str(tmp_path), backend="gs", source_split="val",
        benchmark_partition="test_unseen", code_revision="abc123",
        allow_dirty_code=False,
    )
    scene = SimpleNamespace(
        scene_id="interior_val", source_dataset="gs",
        official_split="val", provenance_sha256="a" * 64,
    )

    class Funnel:
        path = tmp_path / "collection_funnel.json"

        def complete(self):
            self.path.write_text(json.dumps({"status": "completed"}))

    Funnel.path.write_text(json.dumps({"status": "running"}))
    run_contract = {"sampling_provenance": {}}

    assert collection_runtime._finalize_collection_run(
        args=args, scenes=[scene], started=0.0,
        stats=collections.Counter(), skipped=collections.Counter(),
        records_path=str(records_path), funnel=Funnel(),
        existing_records=0, completed_groups=set(),
        run_contract=run_contract) == 0

    metadata = json.loads((tmp_path / "run_meta.json").read_text())
    catalog = json.loads((tmp_path / "manifest.json").read_text())
    assert metadata["source_split"] == "val"
    assert metadata["benchmark_partition"] == "test_unseen"
    assert metadata["source_catalog"]["official_splits"] == ["val"]
    assert catalog["split"] == "test_unseen"
    assert catalog["source_split"] == "val"
    assert catalog["benchmark_partition"] == "test_unseen"
