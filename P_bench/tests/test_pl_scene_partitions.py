"""Frozen benchmark scene partitions and train collection guard."""

from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import pytest

from pipeline import collection_cli, collection_support, scene_partitions


@dataclass(frozen=True)
class _Scene:
    scene_id: str
    source_dataset: str


def test_default_partition_asset_enumerates_complete_collection_catalogs():
    partitions = scene_partitions.load()

    assert partitions.counts("gs") == {
        "excluded_damaged": 1,
        "test_unseen": 21,
        "train_seen": 42,
    }
    assert partitions.counts("r2r") == {
        "test_unseen": 29,
        "train_seen": 61,
    }
    assert partitions.counts("b1k") == {
        "excluded_unavailable": 1,
        "test_unseen": 10,
        "train_seen": 40,
    }
    assert partitions.partition("gs", "interior_0289_840757") == \
        "train_seen"
    assert partitions.partition("gs", "interior_0505_839970") == \
        "excluded_damaged"
    assert partitions.partition("gs", "interior_0516_840045") == \
        "test_unseen"
    assert partitions.partition("r2r", "2t7WUuJeko7") == "test_unseen"
    assert partitions.sha256 == hashlib.sha256(
        scene_partitions.DEFAULT_PATH.read_bytes()).hexdigest()


def test_select_catalog_keeps_only_requested_partition():
    partitions = scene_partitions.from_rows({
        "gs": {
            "train": {"partition": "train_seen", "family": "train"},
            "test": {"partition": "test_unseen", "family": "test"},
            "bad": {"partition": "excluded_damaged", "family": "bad"},
        },
    })
    catalog = [
        _Scene("train", "gs"),
        _Scene("test", "gs"),
        _Scene("bad", "gs"),
    ]

    selected = partitions.select_catalog(
        "gs", catalog, benchmark_partition="train_seen")

    assert [scene.scene_id for scene in selected] == ["train"]


def test_select_catalog_rejects_unassigned_scene():
    partitions = scene_partitions.from_rows({
        "gs": {
            "known": {"partition": "train_seen", "family": "known"},
        },
    })

    with pytest.raises(
            ValueError, match="scene partition has no gs scene 'unknown'"):
        partitions.select_catalog(
            "gs", [_Scene("known", "gs"), _Scene("unknown", "gs")],
            benchmark_partition="train_seen")


def test_partition_rows_reject_family_crossing_train_and_test():
    with pytest.raises(ValueError, match="family .* crosses partitions"):
        scene_partitions.from_rows({
            "b1k": {
                "floor_a": {"partition": "train_seen", "family": "house"},
                "floor_b": {"partition": "test_unseen", "family": "house"},
            },
        })


def test_collection_cli_defaults_to_train_seen_partition(tmp_path):
    args = collection_cli.build_parser().parse_args([
        "--scenes", "scene", "--out", str(tmp_path / "out"),
    ])

    assert args.benchmark_partition == "train_seen"


def test_collection_discovery_filters_catalog_before_auto_scene_order(
        monkeypatch):
    partitions = scene_partitions.from_rows({
        "gs": {
            "train": {"partition": "train_seen", "family": "train"},
            "test": {"partition": "test_unseen", "family": "test"},
            "bad": {"partition": "excluded_damaged", "family": "bad"},
        },
    })
    catalog = [
        _Scene("train", "gs"),
        _Scene("test", "gs"),
        _Scene("bad", "gs"),
    ]
    monkeypatch.setattr(
        collection_support, "discover_gs_train_scenes",
        lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(
        collection_support.scene_partitions, "load", lambda: partitions)
    args = SimpleNamespace(
        backend="gs", gs_data_root="root", gs_source_manifest="manifest",
        auto_scenes=True, scenes=None, seed=7, max_scenes=None,
        benchmark_partition="train_seen",
    )

    selected = collection_support.discover_collection_scenes(args)

    assert [scene.scene_id for scene in selected] == ["train"]
