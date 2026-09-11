"""Persistent collection-funnel provenance and interruption tests."""

import collections
import json

import pytest

from pipeline import collection_funnel
from pipeline.collection_funnel import CollectionFunnel


def test_collection_funnel_is_self_describing_and_updates_per_scene(tmp_path):
    path = tmp_path / "collection_funnel.json"
    stats = collections.Counter({"candidate": 3})
    skipped = collections.Counter({"no_floor": 1})
    funnel = CollectionFunnel(
        path,
        record_schema_version="conseq.v10",
        oracle_contract_version="ground-disc-visible-v8",
        code_revision="abc123",
        code_dirty=False,
        config_sha256="c" * 64,
        run_contract={"params": {"seed": 7}},
        backend="r2r",
        mode="main",
    )
    funnel.begin_scene("scene-a", stats, skipped)
    stats.update({"candidate": 2, "witness": 1})
    skipped.update({"no_floor": 2})
    funnel.record_searched_pose("scene-a", (1.1, 0.0, -0.4), 0.0)
    funnel.record_searched_pose("scene-a", (2.1, 0.0, -0.4), 1.6)
    funnel.record_pose("scene-a", (1.2, 0.0, -0.4), 0.8)
    funnel.finish_scene("scene-a", stats, skipped, status="completed")

    value = json.loads(path.read_text())
    assert value["schema_version"] == "egoconseq.collection-funnel.v3"
    assert value["record_schema_version"] == "conseq.v10"
    assert value["oracle_contract_version"] == "ground-disc-visible-v8"
    assert value["code_revision"] == "abc123"
    assert value["config_sha256"] == "c" * 64
    assert len(value["run_contract_sha256"]) == 64
    assert value["status"] == "running"
    assert value["scene_counts"] == {
        "completed": 1, "failed": 0, "interrupted": 0, "total": 1}
    assert value["stage_counts"] == {"candidate": 5, "witness": 1}
    assert value["skip_reasons"] == {"no_floor": 3}
    assert value["per_scene"][0]["stage_counts"] == {
        "candidate": 2, "witness": 1}
    assert value["per_scene"][0]["skip_reasons"] == {"no_floor": 2}
    assert value["position_heading_distribution"]["accepted_poses"] == 1
    assert value["position_heading_distribution"]["heading_bins_deg"]
    assert value["searched_position_heading_distribution"] == {
        "distinct_position_cells": 2,
        "heading_bins_deg": {"0": 1, "90": 1},
        "position_cells": ["1:-1", "2:-1"],
        "searched_poses": 2,
    }
    assert value["per_scene"][0][
        "searched_position_heading_distribution"]["searched_poses"] == 2


def test_collection_funnel_interruption_is_durable_and_blocks_completion(
        tmp_path):
    path = tmp_path / "collection_funnel.json"
    funnel = CollectionFunnel(
        path,
        record_schema_version="conseq.v10",
        oracle_contract_version="ground-disc-visible-v8",
        code_revision="abc123",
        code_dirty=False,
        config_sha256="c" * 64,
        run_contract={"params": {"seed": 7}},
        backend="gs",
        mode="main",
    )

    CollectionFunnel.mark_interrupted(path, reason="keyboard_interrupt")

    value = json.loads(path.read_text())
    assert value["status"] == "interrupted"
    assert value["interruption_reason"] == "keyboard_interrupt"


def test_collection_funnel_resume_preserves_prior_scene_counts(tmp_path):
    path = tmp_path / "collection_funnel.json"
    contract = {"params": {"seed": 7}}
    first = CollectionFunnel(
        path,
        record_schema_version="conseq.v10",
        oracle_contract_version="ground-disc-visible-v8",
        code_revision="abc123",
        code_dirty=False,
        config_sha256="c" * 64,
        run_contract=contract,
        backend="r2r",
        mode="main",
    )
    stats = collections.Counter({"candidate": 2})
    skipped = collections.Counter({"no_floor": 1})
    first.begin_scene("scene-a", {}, {})
    first.record_pose("scene-a", (1.2, 0.0, -0.4), 0.8)
    first.finish_scene(
        "scene-a", stats, skipped, status="interrupted")
    first.mark_interrupted(path, reason="keyboard_interrupt")

    resumed = CollectionFunnel(
        path,
        record_schema_version="conseq.v10",
        oracle_contract_version="ground-disc-visible-v8",
        code_revision="abc123",
        code_dirty=False,
        config_sha256="c" * 64,
        run_contract=contract,
        backend="r2r",
        mode="main",
    )
    restored_stats = collections.Counter()
    restored_skipped = collections.Counter()
    resumed.restore_counters(restored_stats, restored_skipped)
    assert restored_stats == {"candidate": 2}
    assert restored_skipped == {"no_floor": 1}
    resumed.begin_scene("scene-a", restored_stats, restored_skipped)
    restored_stats["candidate"] += 3
    resumed.record_pose("scene-a", (2.2, 0.0, -0.4), 1.6)
    resumed.finish_scene(
        "scene-a", restored_stats, restored_skipped, status="completed")

    value = json.loads(path.read_text())
    assert value["stage_counts"] == {"candidate": 5}
    assert value["per_scene"][0]["stage_counts"] == {"candidate": 5}
    assert value["per_scene"][0][
        "position_heading_distribution"]["accepted_poses"] == 2


def test_collection_funnel_restore_is_idempotent_on_nonempty_counters(
        tmp_path):
    """Restoring persisted counters must overwrite, never add.

    The collector rebuilds its deterministic action pools (``pool_L*``) before
    ``restore_counters`` runs, so on resume the
    live Counter is never empty. Additive restore semantics would grow those
    keys by one full pool per resume.
    """
    path = tmp_path / "collection_funnel.json"
    kwargs = {
        "record_schema_version": "conseq.v10",
        "oracle_contract_version": "ground-disc-visible-v8",
        "code_revision": "abc123",
        "code_dirty": False,
        "config_sha256": "c" * 64,
        "run_contract": {"params": {"seed": 7}},
        "backend": "r2r",
        "mode": "main",
    }
    first = CollectionFunnel(path, **kwargs)
    stats = collections.Counter({"pool_L3": 40, "candidate": 2})
    skipped = collections.Counter({"no_floor": 1})
    first.begin_scene("scene-a", {}, {})
    first.finish_scene("scene-a", stats, skipped, status="interrupted")

    resumed = CollectionFunnel(path, **kwargs)
    # The resumed process has already rebuilt the same pool and produced one
    # key the persisted file has never seen.
    live_stats = collections.Counter({"pool_L3": 40, "process_only": 1})
    live_skipped = collections.Counter({"pose_rejected": 4})
    resumed.restore_counters(live_stats, live_skipped)
    resumed.restore_counters(live_stats, live_skipped)

    assert live_stats == {
        "pool_L3": 40, "candidate": 2, "process_only": 1}
    assert live_skipped == {"no_floor": 1, "pose_rejected": 4}


@pytest.mark.parametrize(
    ("method_name", "status", "reason_field"),
    [
        ("mark_interrupted", "interrupted", "interruption_reason"),
        ("mark_failed", "failed", "failure_reason"),
    ],
)
def test_collection_funnel_terminal_status_is_durable(
        tmp_path, monkeypatch, method_name, status, reason_field):
    path = tmp_path / "collection_funnel.json"
    path.write_text('{"status":"running"}')
    calls = []

    def write_json(target, value, **kwargs):
        calls.append((target, value, kwargs))

    monkeypatch.setattr(collection_funnel, "atomic_write_json", write_json)

    getattr(CollectionFunnel, method_name)(path, reason="operator stop")

    assert calls == [(
        path,
        {
            "status": status,
            reason_field: "operator stop",
        },
        {
            "allow_nan": False,
            "durable": True,
        },
    )]


@pytest.mark.parametrize(
    ("override", "expected_field"),
    [
        ({"record_schema_version": "conseq.v8"}, "record_schema_version"),
        (
            {"oracle_contract_version": "ground-disc-visible-v6"},
            "oracle_contract_version",
        ),
        ({"code_revision": "def456"}, "code_revision"),
        ({"code_dirty": True}, "code_dirty"),
        ({"config_sha256": "d" * 64}, "config_sha256"),
        ({"backend": "b1k"}, "backend"),
        ({"mode": "other"}, "collection_mode"),
    ],
)
def test_collection_funnel_resume_rejects_provenance_changes(
        tmp_path, override, expected_field):
    path = tmp_path / "collection_funnel.json"
    kwargs = {
        "record_schema_version": "conseq.v10",
        "oracle_contract_version": "ground-disc-visible-v8",
        "code_revision": "abc123",
        "code_dirty": False,
        "config_sha256": "c" * 64,
        "run_contract": {"params": {"seed": 7}},
        "backend": "r2r",
        "mode": "main",
    }
    CollectionFunnel(path, **kwargs)

    with pytest.raises(ValueError, match=expected_field):
        CollectionFunnel(path, **(kwargs | override))


@pytest.mark.parametrize("name", [
    "SURFACE_TARGET_EXCLUDED_MATERIAL_TOKENS",
    "NON_CONTACT_GROUND_CATEGORIES",
    "NON_SPECIFIC_SEMANTIC_CATEGORIES",
    "STRUCTURAL_CATEGORIES",
])
def test_category_frozensets_move_the_hashed_config_summary(monkeypatch, name):
    """Catches a semantics-bearing constant escaping the reuse guard.

    These four decide A3 contact attribution, category specificity and B
    target exclusion.  A JSON encoder that cannot take a frozenset must not
    make them invisible: a stale pose pool would then pass its config check
    while the labels it feeds were computed under different rules.
    """
    from pipeline import config

    baseline = collection_funnel.config_sha256(config)
    monkeypatch.setattr(
        config, name, frozenset(set(getattr(config, name)) | {"__probe__"}))

    assert collection_funnel.config_sha256(config) != baseline


def test_an_unhashable_config_constant_fails_loudly(monkeypatch):
    """Catches a future constant being dropped instead of raising."""
    from pipeline import config

    monkeypatch.setattr(config, "STRUCTURAL_CATEGORIES", object(),
                        raising=False)

    with pytest.raises(TypeError, match="STRUCTURAL_CATEGORIES"):
        collection_funnel.config_sha256(config)
