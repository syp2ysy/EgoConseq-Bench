"""Scene discovery logic (pure; tmp fake dirs, no Habitat)."""

import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from pipeline import scene_pool
from pipeline.scene_pool import (
    SceneCatalogError,
    deterministic_scene_order,
    discover_gs_train_scenes,
    discover_r2r_train_scenes,
    resolve_scene_subset,
)


def _write_mp3d_scene(root, scene_id):
    directory = root / scene_id
    directory.mkdir(parents=True)
    for suffix in (".glb", ".navmesh", ".house", "_semantic.ply"):
        (directory / f"{scene_id}{suffix}").write_text("")


def _write_r2r(path, episodes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump({"episodes": episodes, "instruction_vocab": {}}, handle)


def _write_mp3d_catalog(root, *scene_ids):
    root.mkdir()
    (root / "mp3d_annotated_basis.scene_dataset_config.json").write_text("{}")
    for scene_id in scene_ids:
        _write_mp3d_scene(root, scene_id)


def test_r2r_manifest_parse_and_hash_use_the_same_pinned_bytes(
        tmp_path, monkeypatch):
    mp3d = tmp_path / "mp3d"
    _write_mp3d_catalog(mp3d, "AAA", "BBB")
    manifest = tmp_path / "train.json"
    old_raw = json.dumps({
        "episodes": [{"scene_id": "mp3d/AAA/AAA.glb"}],
    }).encode()
    new_raw = json.dumps({
        "episodes": [{"scene_id": "mp3d/BBB/BBB.glb"}],
    }).encode()
    manifest.write_bytes(old_raw)
    original_loads = scene_pool.json.loads
    swapped = False

    def loads_then_swap(value, *args, **kwargs):
        nonlocal swapped
        payload = original_loads(value, *args, **kwargs)
        if not swapped:
            manifest.write_bytes(new_raw)
            swapped = True
        return payload

    monkeypatch.setattr(scene_pool.json, "loads", loads_then_swap)

    scenes = discover_r2r_train_scenes(manifest, mp3d)

    assert swapped
    assert [scene.scene_id for scene in scenes] == ["AAA"]
    assert scenes[0].provenance_sha256 == hashlib.sha256(old_raw).hexdigest()


def test_r2r_malformed_gzip_closes_pinned_descriptor(tmp_path, monkeypatch):
    manifest = tmp_path / "train.json.gz"
    manifest.write_bytes(b"not a gzip stream")
    original_open = scene_pool.os.open
    original_close = scene_pool.os.close
    opened = []
    closed = []

    def tracked_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor):
        closed.append(descriptor)
        return original_close(descriptor)

    monkeypatch.setattr(scene_pool.os, "open", tracked_open)
    monkeypatch.setattr(scene_pool.os, "close", tracked_close)

    with pytest.raises(SceneCatalogError, match="cannot decode R2R manifest"):
        discover_r2r_train_scenes(manifest, tmp_path / "mp3d")

    assert opened and opened[-1] in closed


@pytest.mark.parametrize(
    ("exception_type", "exception_args"), [
        (KeyboardInterrupt, ("cancelled",)), (SystemExit, (29,)),
    ])
def test_r2r_manifest_read_closes_before_base_exception(
        tmp_path, monkeypatch, exception_type, exception_args):
    manifest = tmp_path / "train.json.gz"
    manifest.write_bytes(gzip.compress(b'{"episodes":[]}'))
    original_open = scene_pool.os.open
    original_close = scene_pool.os.close
    opened = []
    closed = []

    def tracked_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor):
        closed.append(descriptor)
        return original_close(descriptor)

    def interrupt(*_args, **_kwargs):
        raise exception_type(*exception_args)

    monkeypatch.setattr(scene_pool.os, "open", tracked_open)
    monkeypatch.setattr(scene_pool.os, "close", tracked_close)
    monkeypatch.setattr(scene_pool.os, "read", interrupt)

    with pytest.raises(exception_type) as caught:
        discover_r2r_train_scenes(manifest, tmp_path / "mp3d")

    assert caught.value.args == exception_args
    assert opened and opened[-1] in closed


def test_r2r_train_catalog_deduplicates_scenes_and_ignores_trajectory_fields(
        tmp_path):
    mp3d = tmp_path / "mp3d"
    mp3d.mkdir()
    (mp3d / "mp3d_annotated_basis.scene_dataset_config.json").write_text("{}")
    _write_mp3d_scene(mp3d, "AAA")
    _write_mp3d_scene(mp3d, "BBB")
    first = tmp_path / "r2r" / "train" / "train.json.gz"
    second = tmp_path / "r2r_changed" / "train" / "train.json.gz"
    base = [
        {"scene_id": "mp3d/AAA/AAA.glb", "reference_path": [1, 2],
         "start_position": [0, 0, 0], "goals": [{"position": [1, 0, 1]}]},
        {"scene_id": "mp3d/AAA/AAA.glb", "reference_path": [3, 4]},
        {"scene_id": "mp3d/BBB/BBB.glb", "instruction": {"text": "go"}},
    ]
    changed = [
        {**episode, "reference_path": ["changed"],
         "start_position": [99, 99, 99], "goals": []}
        for episode in base
    ]
    _write_r2r(first, base)
    _write_r2r(second, changed)

    original = discover_r2r_train_scenes(first, mp3d)
    modified = discover_r2r_train_scenes(second, mp3d)

    assert [scene.scene_id for scene in original] == ["AAA", "BBB"]
    assert [scene.scene_id for scene in modified] == ["AAA", "BBB"]
    assert all(scene.official_split == "train" for scene in original)
    assert all(scene.semantic_format == "mp3d_ply" for scene in original)


def _write_gs_scene(root, scene_id):
    directory = root / scene_id
    directory.mkdir(parents=True)
    for name in (
            "scene.gs.ply", "scene.navmesh", "labels.json",
            "scene.collision.npz"):
        (directory / name).write_text("[]" if name == "labels.json" else "")


def test_gs_catalog_requires_explicit_train_manifest(tmp_path):
    root = tmp_path / "gs"
    _write_gs_scene(root, "train_scene")
    _write_gs_scene(root, "test_scene")
    manifest = root / "splits.json"
    manifest.write_text(json.dumps({
        "schema_version": "egoconseq.scene_manifest.v1",
        "dataset": "gs",
        "scenes": [
            {"scene_id": "train_scene", "split": "train",
             "path": "train_scene"},
            {"scene_id": "test_scene", "split": "test",
             "path": "test_scene"},
        ],
    }))

    scenes = discover_gs_train_scenes(root, manifest)

    assert [scene.scene_id for scene in scenes] == ["train_scene"]
    assert scenes[0].official_split == "train"
    assert scenes[0].semantic_format == "gs_bbox"


def test_gs_source_asset_identity_follows_replaced_symlink_target(tmp_path):
    root = tmp_path / "gs"
    _write_gs_scene(root, "train_scene")
    directory = root / "train_scene"
    scene_path = directory / "scene.gs.ply"
    scene_path.unlink()
    first_target = directory / "geometry-a.ply"
    second_target = directory / "geometry-b.ply"
    first_target.write_bytes(b"geometry-a")
    second_target.write_bytes(b"geometry-b")
    stat = first_target.stat()
    os.utime(
        second_target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    scene_path.symlink_to(first_target.name)
    manifest = root / "splits.json"
    manifest.write_text(json.dumps({
        "schema_version": "egoconseq.scene_manifest.v1",
        "dataset": "gs",
        "scenes": [
            {"scene_id": "train_scene", "split": "train",
             "path": "train_scene"},
        ],
    }))
    original_scene = discover_gs_train_scenes(root, manifest)[0]
    assert Path(original_scene.scene_path) == first_target.resolve()
    assert scene_pool.verify_scene_source_asset_paths(
        original_scene, (
            ("scene", original_scene.scene_path),
            ("navmesh", original_scene.navmesh_path),
            ("semantic", original_scene.semantic_path),
        ))["source_dataset"] == "gs"
    original = original_scene.provenance()

    scene_path.unlink()
    scene_path.symlink_to(second_target.name)
    changed = discover_gs_train_scenes(root, manifest)[0].provenance()

    assert changed["source_assets_sha256"] != \
        original["source_assets_sha256"]
    original_scene = next(
        asset for asset in original["source_assets"]
        if asset["role"] == "scene")
    changed_scene = next(
        asset for asset in changed["source_assets"]
        if asset["role"] == "scene")
    assert original_scene["bytes"] == changed_scene["bytes"]
    assert original_scene["sha256"] != changed_scene["sha256"]


def test_source_asset_provenance_rejects_an_internally_tampered_digest(
        tmp_path):
    root = tmp_path / "gs"
    _write_gs_scene(root, "train_scene")
    manifest = root / "splits.json"
    manifest.write_text(json.dumps({
        "schema_version": "egoconseq.scene_manifest.v1",
        "dataset": "gs",
        "scenes": [
            {"scene_id": "train_scene", "split": "train",
             "path": "train_scene"},
        ],
    }))
    source = discover_gs_train_scenes(root, manifest)[0].provenance()
    source["source_assets"][0]["sha256"] = "0" * 64

    with pytest.raises(SceneCatalogError, match="source asset digest"):
        scene_pool.validate_source_asset_provenance(source)


def test_gs_catalog_rejects_unknown_split_and_missing_manifest(tmp_path):
    root = tmp_path / "gs"
    _write_gs_scene(root, "scene")
    with pytest.raises(SceneCatalogError, match="manifest"):
        discover_gs_train_scenes(root, root / "missing.json")

    manifest = root / "splits.json"
    manifest.write_text(json.dumps({
        "schema_version": "egoconseq.scene_manifest.v1",
        "dataset": "gs",
        "scenes": [
            {"scene_id": "scene", "split": "mystery", "path": "scene"},
        ],
    }))
    with pytest.raises(SceneCatalogError, match="split"):
        discover_gs_train_scenes(root, manifest)


def test_explicit_scene_subset_must_belong_to_verified_train_catalog(tmp_path):
    root = tmp_path / "gs"
    _write_gs_scene(root, "first")
    _write_gs_scene(root, "second")
    manifest = root / "splits.json"
    manifest.write_text(json.dumps({
        "schema_version": "egoconseq.scene_manifest.v1",
        "dataset": "gs",
        "scenes": [
            {"scene_id": "first", "split": "train", "path": "first"},
            {"scene_id": "second", "split": "train", "path": "second"},
        ],
    }))
    catalog = discover_gs_train_scenes(root, manifest)

    subset = resolve_scene_subset(catalog, ["second", str(root / "first")])
    assert [scene.scene_id for scene in subset] == ["second", "first"]
    with pytest.raises(SceneCatalogError, match="not in the verified train"):
        resolve_scene_subset(catalog, ["heldout"])


def test_gs_explicit_discovery_requires_only_requested_scene_assets(tmp_path):
    root = tmp_path / "gs"
    _write_gs_scene(root, "ready")
    _write_gs_scene(root, "not_preprocessed")
    (root / "not_preprocessed" / "scene.collision.npz").unlink()
    manifest = root / "splits.json"
    manifest.write_text(json.dumps({
        "schema_version": "egoconseq.scene_manifest.v1",
        "dataset": "gs",
        "scenes": [
            {"scene_id": "ready", "split": "train", "path": "ready"},
            {"scene_id": "not_preprocessed", "split": "train",
             "path": "not_preprocessed"},
        ],
    }))

    selected = discover_gs_train_scenes(
        root, manifest, requested=["ready"])

    assert [scene.scene_id for scene in selected] == ["ready"]
    with pytest.raises(SceneCatalogError, match="not in the verified train"):
        discover_gs_train_scenes(root, manifest, requested=["heldout"])
    with pytest.raises(SceneCatalogError, match="not_preprocessed"):
        discover_gs_train_scenes(root, manifest)


def test_scene_order_is_seeded_deterministic_and_not_lexical(tmp_path):
    root = tmp_path / "gs"
    for scene_id in ("a", "b", "c", "d", "e"):
        _write_gs_scene(root, scene_id)
    manifest = root / "splits.json"
    manifest.write_text(json.dumps({
        "schema_version": "egoconseq.scene_manifest.v1",
        "dataset": "gs",
        "scenes": [
            {"scene_id": scene_id, "split": "train", "path": scene_id}
            for scene_id in ("a", "b", "c", "d", "e")
        ],
    }))
    catalog = discover_gs_train_scenes(root, manifest)

    first = deterministic_scene_order(catalog, seed=17)
    again = deterministic_scene_order(reversed(catalog), seed=17)
    changed = deterministic_scene_order(catalog, seed=18)

    assert [scene.scene_id for scene in first] == [
        scene.scene_id for scene in again]
    assert [scene.scene_id for scene in first] != sorted(
        scene.scene_id for scene in first)
    assert [scene.scene_id for scene in first] != [
        scene.scene_id for scene in changed]
