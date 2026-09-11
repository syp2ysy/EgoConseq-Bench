"""Collection output resume and overwrite policy."""

import collections
import copy
import pytest
import numpy as np
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pipeline import (
    abc1_record, action_control_catalog, action_sampling, collection_cli, collection_funnel,
    collection_runtime,
    collection_support, config, record, consequence,
    pose_setting, rollout, semantic, source_manifest, validate,
)
from pipeline import sim as sim_module
from pipeline.actions import Forward, Turn
from pipeline.scene_pool import SceneSpec, SourceAssetIdentity
from tests import _collection_api as collect
from pipeline.collection_cli import append_record_group, prepare_records_output
from pipeline.pose_calibration import load_pose_exclusions, pose_is_diverse
from tests._synthetic import (
    LEVEL_FLOOR,
    LEVEL_FLOOR_FIT,
    make_frame,
    source_provenance,
)


def test_pose_exclusions_read_compact_records_without_a_checkpoint(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({
        "dataset": "b1k", "scene_id": "room", "record_uid": "old",
        "pose": {"position": [1.0, 0.0, 2.0], "yaw_rad": 0.0},
    }) + "\n")
    poses = load_pose_exclusions(path)["room"]
    assert not pose_is_diverse([1.1, 0.0, 2.0], 0.1, poses)
    assert pose_is_diverse([2.0, 0.0, 2.0], 0.0, poses)


def _append_expansion_rows(path, worker, progress=None):
    for index in range(5):
        value = {"record_uid": f"{worker}-{index}", "padding": "x" * 10000}
        collection_cli.append_compact_records(path, [value], progress_path=progress)


def test_shared_append_preserves_old_bytes_and_concurrent_records(tmp_path):
    import multiprocessing

    path = tmp_path / "records.jsonl"
    original = b'{"record_uid": "old", "unmodified": true}\n'
    path.write_bytes(original)
    # Call once locally so a missing implementation fails at its actual API.
    collection_cli.append_compact_records(path, [{"record_uid": "first"}])
    workers = [multiprocessing.get_context("fork").Process(
        target=_append_expansion_rows, args=(path, index)) for index in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert path.read_bytes().startswith(original)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len({row["record_uid"] for row in rows}) == 22
    assert list(tmp_path.iterdir()) == [path]


def test_shared_append_recovers_only_an_uncommitted_partial_tail(tmp_path):
    path = tmp_path / "records.jsonl"
    original = b'{"record_uid":"old"}\n'
    path.write_bytes(original + b'{"record_uid":"interrupted')
    collection_cli.append_compact_records(path, [{"record_uid": "new"}])
    assert path.read_bytes().startswith(original)
    assert [json.loads(line)["record_uid"] for line in
            path.read_text().splitlines()] == ["old", "new"]


def test_shared_append_enforces_one_total_across_workers(tmp_path):
    import multiprocessing

    path = tmp_path / "records.jsonl"
    original = b'{"record_uid":"old"}\n'
    path.write_bytes(original)
    progress = tmp_path / "expansion.json"
    progress.write_text(json.dumps({"record_count": 1, "target_records": 7,
                                    "byte_offset": len(original)}))
    collection_cli.append_compact_records(path, [], progress_path=progress)
    workers = [multiprocessing.get_context("fork").Process(
        target=_append_expansion_rows, args=(path, i, progress)) for i in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert len(path.read_text().splitlines()) == 7
    state = json.loads(progress.read_text())
    assert state["record_count"] == 7
    assert state["byte_offset"] == path.stat().st_size


def test_append_collection_keeps_the_shared_dataset_and_metadata(tmp_path, monkeypatch):
    root = tmp_path / "b1k"
    root.mkdir()
    records = root / "records.jsonl"
    original = json.dumps({"record_uid": "old", "scene_id": "room",
                           "pose": {"position": [0, 0, 0], "yaw_rad": 0.0}}) + "\n"
    records.write_text(original)
    (root / "run_meta.json").write_text('{"record_count":1}')
    (root / "expansion.json").write_text(json.dumps({
        "record_count": 1, "target_records": 2,
        "byte_offset": records.stat().st_size}))
    parser = collection_cli.build_parser()
    args = parser.parse_args([
        "--scenes", "room", "--out", str(tmp_path / "progress"),
        "--append-records", str(records), "--collection-shard-id", "new-run"])
    monkeypatch.setattr(collection_runtime, "discover_collection_scenes",
                        lambda _args: ["room"])

    def scene(**kwargs):
        assert not pose_is_diverse([0, 0, 0], 0, kwargs["pose_exclusions"]["room"])
        image = Path(kwargs["image_dir"]) / "new.png"
        image.write_bytes(b"image")
        collection_cli.append_compact_records(kwargs["records_path"], [{
            "record_uid": "new", "scene_id": "room", "image_path": "img/new.png",
            "pose": {"position": [2, 0, 0], "yaw_rad": 0.0},
        }], progress_path=root / "expansion.json")

    monkeypatch.setattr(collection_runtime, "_collect_scene", scene)
    assert collection_runtime._run_collection(args, parser) == 0
    assert records.read_text().startswith(original)
    assert len(records.read_text().splitlines()) == 2
    assert (root / "img/new.png").exists()
    assert json.loads((root / "run_meta.json").read_text()) == {"record_count": 1}
    assert list(tmp_path.rglob("records.jsonl")) == [records]


def _source_scene_spec(scene_id="scene/example"):
    source = source_provenance(scene_id, dataset="r2r")
    return SceneSpec(
        scene_id=scene_id,
        source_dataset="r2r",
        official_split="train",
        scene_path="/datasets/mp3d/scene.glb",
        navmesh_path="/datasets/mp3d/scene.navmesh",
        semantic_path="/datasets/mp3d/scene_semantic.ply",
        semantic_metadata_path="/datasets/mp3d/scene.house",
        semantic_format="mp3d_ply",
        scene_dataset_config="/datasets/mp3d/runtime.json",
        provenance_path=source["source_manifest"],
        provenance_sha256=source["source_manifest_sha256"],
        source_assets=tuple(SourceAssetIdentity(
            role=asset["role"],
            byte_size=asset["bytes"],
            sha256=asset["sha256"],
        ) for asset in source["source_assets"]),
        source_assets_sha256=source["source_assets_sha256"],
    )


def test_balanced_pose_bank_combines_pairs_dynamic_natural_and_controls(
        monkeypatch):
    class Proxy:
        def rollout(self, _actions):
            return {"collision": False, "first_contact_arc_m": None}

        def coverage(self, _actions, _max_arc_m=None):
            return 1.0

        def prefix_supported_reach_m(self, _prefix):
            return 6.0

    paired = collection_runtime.action_proposal.Candidate(
        tag="paired", length=1, actions=(Forward(0.5),), variant="safe",
        provenance={
            "protocol": collection_runtime.action_proposal.
                PROPOSAL_PROTOCOL_VERSION,
            "template_id": "T1-00", "variant": "safe",
        })
    dynamic = collection_runtime.action_proposal.Candidate(
        tag="dynamic", length=1, actions=(Forward(1.0),),
        variant=collection_runtime.action_proposal.NATURAL_DYNAMIC_VARIANT,
        provenance={
            "protocol": collection_runtime.action_proposal.
                PROPOSAL_PROTOCOL_VERSION,
            "template_id": "N1-dynamic",
            "variant": collection_runtime.action_proposal.
                NATURAL_DYNAMIC_VARIANT,
        })
    monkeypatch.setattr(
        collection_runtime.action_proposal, "FrameDepthProxy",
        lambda *_args: Proxy())
    monkeypatch.setattr(
        collection_runtime.action_proposal, "build_pose_bank",
        lambda *_args, **_kwargs: {1: [paired]})
    monkeypatch.setattr(
        collection_runtime.action_proposal, "build_dynamic_natural_bank",
        lambda *_args, **_kwargs: {1: [dynamic]})

    pools, provenance, manifest, control_tags = \
        collection_runtime._materialize_pose_action_bank(
            collection_runtime.ProposalBank([], {}),
            SimpleNamespace(sensor=SimpleNamespace(hfov_deg=180.0)),
            0.2,
            args=SimpleNamespace(
                action_mode="balanced", seed=7, backend="r2r",
                lengths=(1, 2, 3, 4, 5, 6),
                proposal_pairs_per_length=12,
                proposal_natural_per_length=40),
            scene_id="scene-a", pose_index=3, include_controls=True,
            stats=collections.Counter(), skipped=collections.Counter())

    assert set(provenance[tag]["variant"] for tag in provenance) == {
        "safe", collection_runtime.action_proposal.NATURAL_DYNAMIC_VARIANT,
        collection_runtime.action_proposal.A1_CONTROL_VARIANT,
    }
    assert len(control_tags) == action_control_catalog.ANCHORS_PER_POSE
    assert set(control_tags).issubset({tag for rows in pools.values()
                                      for tag, _actions in rows})
    assert all(row["variant"] != "natural" for row in manifest)


def test_dynamic_natural_overrides_same_action_from_paired_arm(monkeypatch):
    """One physical program appears once and keeps its A1-safe provenance."""
    class Proxy:
        def prefix_supported_reach_m(self, _prefix):
            return 6.0

    actions = (Forward(1.5),)
    tag = collection_runtime.action_proposal.candidate_tag(actions)
    paired = collection_runtime.action_proposal.Candidate(
        tag=tag, length=1, actions=actions, variant="safe",
        provenance={
            "protocol": collection_runtime.action_proposal.
                PROPOSAL_PROTOCOL_VERSION,
            "template_id": "T1-00", "variant": "safe",
        })
    dynamic = collection_runtime.action_proposal.Candidate(
        tag=tag, length=1, actions=actions,
        variant=collection_runtime.action_proposal.NATURAL_DYNAMIC_VARIANT,
        provenance={
            "protocol": collection_runtime.action_proposal.
                PROPOSAL_PROTOCOL_VERSION,
            "template_id": "N1-duplicate",
            "variant": collection_runtime.action_proposal.
                NATURAL_DYNAMIC_VARIANT,
        })
    monkeypatch.setattr(
        collection_runtime.action_proposal, "FrameDepthProxy",
        lambda *_args: Proxy())
    monkeypatch.setattr(
        collection_runtime.action_proposal, "build_pose_bank",
        lambda *_args, **_kwargs: {1: [paired]})
    monkeypatch.setattr(
        collection_runtime.action_proposal, "build_dynamic_natural_bank",
        lambda *_args, **_kwargs: {1: [dynamic]})

    pools, provenance, manifest, _control_tags = \
        collection_runtime._materialize_pose_action_bank(
            collection_runtime.ProposalBank([], {}),
            SimpleNamespace(sensor=SimpleNamespace(hfov_deg=180.0)),
            0.2,
            args=SimpleNamespace(
                action_mode="balanced", seed=7, backend="r2r", lengths=(1,),
                proposal_pairs_per_length=1,
                proposal_natural_per_length=1),
            scene_id="scene-a", pose_index=0, include_controls=False,
            stats=collections.Counter(), skipped=collections.Counter())

    assert [row[0] for row in pools[1]] == [tag]
    assert provenance[tag]["variant"] == \
        collection_runtime.action_proposal.NATURAL_DYNAMIC_VARIANT
    assert [row["variant"] for row in manifest] == [
        collection_runtime.action_proposal.NATURAL_DYNAMIC_VARIANT]


def test_balanced_run_bank_does_not_build_a_scene_independent_natural_pool(
        monkeypatch):
    monkeypatch.setattr(
        collection_runtime, "candidate_action_pools",
        lambda *_args, **_kwargs: pytest.fail(
            "balanced v3 natural actions must be drawn from each pose depth"))
    monkeypatch.setattr(
        collection_runtime.action_proposal, "build_template_bank",
        lambda *_args, **_kwargs: [])

    bank, digest = collection_runtime._build_collection_action_banks(
        SimpleNamespace(
            action_mode="balanced", seed=3,
            lengths=(1, 2, 3, 4, 5, 6),
            proposal_pairs_per_length=12,
            proposal_natural_per_length=40),
        [(79.0, 63.453)], collections.Counter())

    assert bank.natural_pools == {}
    assert len(digest) == 64


def test_new_collection_defaults_pin_the_v4_action_policy():
    args = SimpleNamespace(
        lengths=[1, 2, 3, 4, 5, 6], collection_mode="main",
        action_mode="balanced")

    collection_runtime._resolve_collection_mode_defaults(args, object())

    assert args.action_sampling_policy == \
        collection_runtime.action_proposal.DEPTH_CONDITIONED_POLICY_V5


def test_capacity_stop_cli_is_opt_in_and_frozen_into_run_contract():
    parser = collection_cli.build_parser()
    manual = parser.parse_args([
        "--scenes", "scene-a", "--out", "/tmp/manual"])
    background = parser.parse_args([
        "--scenes", "scene-a", "--record-idle-stop-s", "120",
        "--scene-wallclock-stop-s", "600", "--out", "/tmp/background"])

    assert manual.record_idle_stop_s is None
    assert manual.scene_wallclock_stop_s is None
    contract = collection_runtime.collection_run_contract(
        background, ["scene-a"], [1.0], [(79.0, 63.453)])
    assert contract["params"]["record_idle_stop_s"] == 120.0
    assert contract["params"]["scene_wallclock_stop_s"] == 600.0


def test_ordinary_action_budget_cli_accepts_only_canary_ladder_values():
    parser = collection_cli.build_parser()
    default = parser.parse_args([
        "--scenes", "scene-a", "--out", "/tmp/default"])
    compact = parser.parse_args([
        "--scenes", "scene-a", "--ordinary-actions-per-pose", "24",
        "--out", "/tmp/compact"])

    assert default.ordinary_actions_per_pose == 36
    assert compact.ordinary_actions_per_pose == 24
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--scenes", "scene-a", "--ordinary-actions-per-pose", "20",
            "--out", "/tmp/invalid"])


@pytest.mark.parametrize(
    ("extra_argv", "expected_fragment"),
    [
        (["--lengths", "1", "2"],
         "formal collection requires --lengths 1 2 3 4 5 6 exactly"),
        (["--keep-per-length", "0"],
         "--keep-per-length must be positive"),
        (["--collection-shard-id", "bad id"],
         "--collection-shard-id must contain only letters"),
        (["--pose-exclusions", "{tmp}/missing.json"],
         "No such file"),
    ],
)
def test_collect_invalid_arguments_use_argparse_error(
        monkeypatch, tmp_path, capsys, extra_argv, expected_fragment):
    """Every CLI validation path must exit 2 with usage text, not a
    traceback (the parser used to be a NameError at each of these sites)."""
    argv = [
        "collect.py",
        "--auto-scenes",
        "--out", str(tmp_path / "out"),
        "--lengths", "1", "2", "3", "4", "5", "6",
    ] + [value.format(tmp=tmp_path) for value in extra_argv]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as stopped:
        collect.main()

    assert stopped.value.code == 2
    assert expected_fragment in capsys.readouterr().err


def test_collect_argparse_exit_never_stamps_a_previous_funnel(
        monkeypatch, tmp_path):
    """An argparse rejection happens before any collection: it must not
    mark the previous run's crash audit trail as failed."""
    out = tmp_path / "out"
    out.mkdir()
    funnel_path = out / "collection_funnel.json"
    funnel_path.write_text(json.dumps({"status": "completed"}))
    monkeypatch.setattr(sys, "argv", [
        "collect.py",
        "--auto-scenes",
        "--out", str(out),
        "--lengths", "1", "2",
    ])

    with pytest.raises(SystemExit) as stopped:
        collect.main()

    assert stopped.value.code == 2
    assert json.loads(funnel_path.read_text()) == {"status": "completed"}


def test_main_collection_defaults_resolve(tmp_path):
    parser = collect.build_parser()
    main = parser.parse_args([
        "--auto-scenes", "--out", str(tmp_path / "main"),
        "--collection-mode", "main",
    ])
    assert main.lengths is None
    collect._validate_collection_args(main, parser)

    assert main.lengths == list(config.GEN_LENGTHS)


def test_collection_cli_rejects_retired_c1_suffix_mode(tmp_path):
    """Catches accidentally restoring the second, initial-visible C1 route."""
    parser = collect.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--auto-scenes", "--out", str(tmp_path / "strict-c1"),
            "--c1-suffix-candidates",
        ])


def test_semantic_query_workers_are_positive_execution_only_parameter(
        tmp_path):
    parser = collect.build_parser()
    args = parser.parse_args([
        "--auto-scenes", "--out", str(tmp_path / "workers"),
        "--semantic-query-workers", "8",
    ])
    assert args.semantic_query_workers == 8

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--auto-scenes", "--out", str(tmp_path / "bad-workers"),
            "--semantic-query-workers", "0",
        ])


def test_semantic_query_workers_do_not_change_collection_contract():
    common = {
        "out": "ignored", "overwrite": False, "resume": True,
        "debug_images": False, "debug_outcomes_per_frame": 4,
        "action_file": None,
    }
    serial = collect.collection_run_contract(
        SimpleNamespace(**common, semantic_query_workers=1),
        ["scene"], [1.0], [(79.0, config.vfov_for_hfov(79.0))])
    parallel = collect.collection_run_contract(
        SimpleNamespace(**common, semantic_query_workers=8),
        ["scene"], [1.0], [(79.0, config.vfov_for_hfov(79.0))])

    assert serial == parallel
    assert "semantic_query_workers" not in serial["params"]


def test_execution_only_geometry_tuning_does_not_change_collection_config(
        monkeypatch):
    baseline = collection_funnel.config_sha256(config)

    monkeypatch.setattr(
        config, "SEMANTIC_ASSIGN_PARALLEL_MIN_POINTS", 123_456)
    monkeypatch.setattr(
        config, "A3_FACE_AABB_PRUNE_SLACK_M", 1e-6)
    monkeypatch.setattr(
        config, "MP3D_AABB_MADVISE_INTERVAL_CHUNKS", 17)

    assert collection_funnel.config_sha256(config) == baseline


def test_collection_session_receives_semantic_query_workers(monkeypatch):
    observed = {}
    sentinel = object()

    def build_session(scene_path, **kwargs):
        observed.update(scene_path=scene_path, **kwargs)
        return sentinel

    monkeypatch.setattr(sim_module, "SimSession", build_session)
    scene = SimpleNamespace(
        scene_path="scene.glb", scene_dataset_config="dataset.json",
        semantic_format="mp3d_ply", source_dataset="r2r",
        official_split="train")
    args = SimpleNamespace(backend="r2r", semantic_query_workers=8)

    sessions = collection_runtime._open_collection_sessions(
        args, scene, [1.0], [(79.0, 63.45)])

    assert sessions == [sentinel]
    assert observed["semantic_query_workers"] == 8


def test_gs_main_route_selects_v18_and_opens_the_verified_session(
        monkeypatch, tmp_path):
    parser = collect.build_parser()
    args = parser.parse_args([
        "--backend", "gs",
        "--gs-data-root", str(tmp_path / "gs"),
        "--gs-source-manifest", str(tmp_path / "train.json"),
        "--auto-scenes", "--out", str(tmp_path / "out"),
    ])
    scene = SimpleNamespace(scene_id="interior_0007_840137")
    opened = object()
    observed = []

    def open_once(selected, *, heights):
        observed.append((selected, heights))
        return opened

    monkeypatch.setattr("pipeline.gs_sim.GsSimSession", open_once)

    assert collection_runtime.collection_record_schema(args) == \
        record.V18_SCHEMA_VERSION
    assert collection_runtime._open_collection_sessions(
        args, scene, [config.CAMERA_HEIGHT_M],
        list(config.BENCH_FOVS_DEG)) == [opened, opened]
    assert observed == [(scene, [config.CAMERA_HEIGHT_M])]


def test_gs_collection_discovers_only_the_registered_manifest(
        monkeypatch, tmp_path):
    sentinel = [dataclasses.replace(
        _source_scene_spec("gs/scene"), source_dataset="gs")]
    observed = []

    def discover(root, manifest, *, requested=None):
        observed.append((root, manifest, requested))
        return sentinel

    monkeypatch.setattr(collection_support, "discover_gs_train_scenes", discover)
    monkeypatch.setattr(
        collection_support.scene_partitions, "load",
        lambda: collection_support.scene_partitions.from_rows({
            "gs": {
                "gs/scene": {
                    "partition": "train_seen",
                    "family": "gs/scene",
                },
            },
        }, sha256="a" * 64))
    args = SimpleNamespace(
        backend="gs", gs_data_root=str(tmp_path / "gs"),
        gs_source_manifest=str(tmp_path / "train.json"),
        auto_scenes=True, scenes=None, max_scenes=None, seed=7,
    )
    monkeypatch.setattr(
        collection_support, "deterministic_scene_order",
        lambda catalog, seed: list(catalog))

    assert collection_support.discover_collection_scenes(args) == sentinel
    assert observed == [(args.gs_data_root, args.gs_source_manifest, None)]


def test_active_collection_cli_exposes_registered_main_backends(tmp_path):
    parser = collect.build_parser()
    args = parser.parse_args([
        "--auto-scenes", "--out", str(tmp_path / "main")])
    help_text = parser.format_help()

    assert args.backend == "r2r"
    assert args.collection_mode == "main"
    assert args.gs_data_root == config.GS_ROOT
    assert args.gs_source_manifest == config.GS_TRAIN_MANIFEST
    assert "--gs-data-root" in help_text
    assert "--hm3d-root" not in help_text


def test_main_collection_builds_the_existing_deterministic_random_action_bank(
        tmp_path):
    parser = collect.build_parser()
    args = parser.parse_args([
        "--auto-scenes",
        "--out", str(tmp_path / "main"),
        "--collection-mode", "main",
        "--lengths", "1",
        "--pool-factor", "1",
    ])
    fovs = tuple(config.BENCH_FOVS_DEG)

    first_stats = collections.Counter()
    first_main, first_hash = (
        collect._build_collection_action_banks(args, fovs, first_stats))
    second_stats = collections.Counter()
    second_main, second_hash = (
        collect._build_collection_action_banks(args, fovs, second_stats))

    assert first_hash == second_hash
    assert first_stats == second_stats
    assert first_main == second_main


def test_structured_evaluator_factory_preserves_main_outcome_annotations(
        monkeypatch):
    class Sim:
        source_dataset = "r2r"

        def recompute_navmesh(self, radius, height):
            self.radius = radius
            self.height = height

        def nav(self, position, yaw):
            return object()

    frame = SimpleNamespace(
        frame_id="frame", position=[0.0, 0.0, 0.0], yaw_rad=0.0)
    physical = {
        "physical": {"collision": False},
        "execution": {
            "realized_pose": {"x": 0.0, "z": 1.0, "heading_deg": 0.0},
            "nominal_pose": {"x": 0.0, "z": 1.0, "heading_deg": 0.0},
        },
    }
    path_trace = object()
    seen_path_traces = []
    monkeypatch.setattr(
        collect.rollout, "physical_rollout",
        lambda *_args, **kwargs: (
            seen_path_traces.append(kwargs.get("path_trace")) or physical))
    assert not hasattr(collect.rollout, "terminal_options")
    monkeypatch.setattr(
        collect._runtime, "judge",
        lambda *_args, **_kwargs: {
            "physical": {"collision": False},
            "execution": {"completed": True},
        })
    monkeypatch.setattr(
        collect._runtime, "structured_outcome_disposition",
        lambda *_args, **_kwargs: "keep")
    monkeypatch.setattr(
        collect._runtime, "strict_shared_oracle_required",
        lambda *_args, **_kwargs: False)

    stats = collections.Counter()
    skipped = collections.Counter()
    evaluate = collect._make_structured_spec_evaluator(
        args=SimpleNamespace(collection_mode="main"),
        variants=[(Sim(), frame)],
        group_labels={"action": "safe"},
        precheck_cache={"action": {
            ("frame", 0.2): {"physical_path_trace": path_trace},
        }},
        stats=stats,
        skipped=skipped,
        pending_a_certificates=[],
    )
    result = evaluate({
        "action_tag": "action",
        "actions": [Forward(1.0)],
        "radii": [0.2],
        "type": "main",
        "_physical_by_radius": {0.2: physical},
    })

    assert list(result) == ["frame"]
    assert result["frame"][0]["outcome_id"] == "b020-action"
    assert result["frame"][0]["action_group_id"] == "action"
    assert result["frame"][0]["action_group_label"] == "safe"
    assert stats == {
        "physical_rollouts": 1,
        "evaluated_outcomes": 1,
    }
    assert skipped == {}
    assert seen_path_traces == [path_trace]


def _outcome():
    return {"outcome_id": "o", "checkpoints": []}


def _record(outcome):
    return {
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "frame_id": "frame",
        "outcomes": [outcome],
    }


def _write_group(path, count):
    prepare_records_output(
        path, expected_siblings=count, resume=False, overwrite=True,
        run_contract={"seed": 42})
    records = []
    for index in range(count):
        value = _record(_outcome())
        value["oracle_contract_version"] = record.ORACLE_CONTRACT_VERSION
        value["frame_id"] = f"frame-{index}"
        value["intervention"] = {
            "group_id": "scene-p000-observation", "type": "sensor_profile"}
        records.append(value)
    append_record_group(path, records, order_key=(0, 0), expected_siblings=count)


def _prepare(path, *, expected_siblings, resume=False, overwrite=False,
             run_contract=None):
    return prepare_records_output(
        path, expected_siblings=expected_siblings, resume=resume,
        overwrite=overwrite, run_contract=run_contract or {"seed": 42})


def test_resume_skips_only_complete_sensor_groups(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_group(path, 2)

    completed, existing = _prepare(
        path, expected_siblings=2, resume=True)

    assert completed == {"scene-p000-observation"}
    assert existing == 2


def test_resume_on_new_output_initializes_current_contract(tmp_path):
    path = tmp_path / "records.jsonl"

    completed, existing = _prepare(
        path, expected_siblings=2, resume=True)

    assert completed == set()
    assert existing == 0
    assert path.read_text() == ""
    assert (tmp_path / ".records.jsonl.groups" / "contract.json").exists()


def test_resume_rejects_partially_written_sensor_group(tmp_path):
    path = tmp_path / "records.jsonl"
    value = _record(_outcome())
    value["oracle_contract_version"] = record.ORACLE_CONTRACT_VERSION
    value["frame_id"] = "frame-0"
    value["intervention"] = {
        "group_id": "scene-p000-observation", "type": "sensor_profile"}

    with pytest.raises(ValueError, match="partial intervention group"):
        append_record_group(
            path, [value], order_key=(0, 0), expected_siblings=2)


def test_pre_spool_guard_rejects_invalid_current_record():
    value = record.build_record(make_frame(), [], image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT)
    value["targets"] = [{"instance_id": 999, "category": "missing"}]

    with pytest.raises(ValueError, match="record validation failed before spool"):
        collect.validate_records_before_spool(
            [value], validation_context=
            source_manifest.LEGACY_RECORD_VALIDATION_CONTEXT)




def test_existing_output_requires_resume_or_overwrite(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_group(path, 1)

    with pytest.raises(FileExistsError, match="--resume or --overwrite"):
        _prepare(path, expected_siblings=1)


def test_explicit_overwrite_truncates_records(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_group(path, 1)

    completed, existing = _prepare(
        path, expected_siblings=1, overwrite=True)

    assert completed == set()
    assert existing == 0
    assert path.read_text() == ""


def test_collection_atomic_write_uses_shared_durable_primitive(
        tmp_path, monkeypatch):
    path = tmp_path / "contract.json"
    calls = []

    def write_text(target, payload, **kwargs):
        calls.append((target, payload, kwargs))

    monkeypatch.setattr(collect._cli, "atomic_write_text", write_text)

    collect._cli._atomic_write(path, "payload\n")

    assert calls == [(path, "payload\n", {"durable": True})]


def test_interrupted_overwrite_cannot_resurrect_old_spools(tmp_path, monkeypatch):
    path = tmp_path / "records.jsonl"
    _write_group(path, 1)
    original_atomic_write = collect._atomic_write

    def interrupt_master_reset(target, payload):
        if target == path:
            raise RuntimeError("interrupted reset")
        return original_atomic_write(target, payload)

    monkeypatch.setattr(
        collect._cli, "_atomic_write", interrupt_master_reset)
    with pytest.raises(RuntimeError, match="interrupted reset"):
        _prepare(path, expected_siblings=1, overwrite=True)
    monkeypatch.setattr(collect._cli, "_atomic_write", original_atomic_write)

    assert not collect._spool_dir(path).exists()
    with pytest.raises(ValueError, match="oracle contract"):
        _prepare(path, expected_siblings=1, resume=True)


def test_resume_rebuilds_records_from_complete_group_spools(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_group(path, 2)
    with path.open("a") as handle:
        handle.write('{"truncated":')

    completed, existing = _prepare(path, expected_siblings=2, resume=True)

    assert completed == {"scene-p000-observation"}
    assert existing == 2
    loaded = list(record.read_records(path))
    assert [value["frame_id"] for value in loaded] == ["frame-0", "frame-1"]


def test_group_spools_keep_public_records_in_canonical_order(tmp_path):
    path = tmp_path / "records.jsonl"
    _prepare(path, expected_siblings=1, overwrite=True)
    for pose_index in (1, 0):
        value = _record(_outcome())
        value["frame_id"] = f"frame-{pose_index}"
        value["intervention"] = {
            "group_id": f"scene-p{pose_index:03d}-observation",
            "type": "sensor_profile",
        }
        append_record_group(
            path, [value], order_key=(0, pose_index), expected_siblings=1)

    assert [value["frame_id"] for value in record.read_records(path)] == [
        "frame-0", "frame-1"]


def test_resume_rejects_old_oracle_contract_without_spools(tmp_path):
    path = tmp_path / "records.jsonl"
    value = _record(_outcome())
    value["intervention"] = {"group_id": "old", "type": "sensor_profile"}
    path.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="oracle contract"):
        _prepare(path, expected_siblings=1, resume=True)


def test_resume_rejects_changed_run_contract(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_group(path, 2)

    with pytest.raises(ValueError, match="collection contract"):
        _prepare(
            path, expected_siblings=2, resume=True,
            run_contract={"seed": 99})


def test_pose_sampling_uses_primary_bodies_not_counterfactual_maximum():
    assert collect._pose_sampling_radii([0.15, 0.2, 0.25]) == [0.15, 0.2, 0.25]


def _calibration_cloud(floor_y=0.0):
    """A dense flat patch the canonical fitter accepts."""
    axis = np.linspace(-1.5, 1.5, 61)
    xs, zs = np.meshgrid(axis, axis)
    return np.column_stack([
        xs.ravel(), np.full(xs.size, floor_y), zs.ravel()])


def _calibration_render(camera_height_m):
    """A real RenderObservation, so it faces the same validation as production."""
    _height, hfov, vfov = config.calibration_profile()
    sensor = sim_module.SensorProfile.from_values(camera_height_m, hfov, vfov)
    return sim_module.RenderObservation(
        rgb=np.zeros((sensor.height_px, sensor.width_px, 3), dtype=np.uint8),
        depth=np.ones((sensor.height_px, sensor.width_px), dtype=np.float32),
        K=config.intrinsics(hfov, vfov),
        sensor=sensor,
        position=np.zeros(3),
        yaw_rad=0.0,
    )


class _AnywherePathfinder:
    def get_random_navigable_point(self):
        return np.zeros(3)

    def distance_to_closest_obstacle(self, _position):
        return 1.0


def test_sample_pose_unprojects_at_rendered_camera_height(monkeypatch):
    # Guards against assuming the renderer honoured the requested height: the
    # cloud must be lifted by what came back, not by what was asked for.
    rendered = _calibration_render(config.CAMERA_HEIGHT_M)
    seen = []
    monkeypatch.setattr(
        sim_module.perception, "unproject",
        lambda _depth, _k: (_calibration_cloud(), np.zeros((3721, 2))))
    monkeypatch.setattr(
        sim_module.perception, "to_agent_ground",
        lambda points, camera_height: seen.append(camera_height) or points)

    calibration = sim_module.sample_pose(
        _AnywherePathfinder(), lambda *_args: rendered,
        np.random.default_rng(0), [0.2], yaws=[0.0], max_tries=1,
        min_floor=0.0)

    assert calibration is not None
    assert seen == [config.CAMERA_HEIGHT_M]


def test_sample_pose_requests_the_frozen_calibration_profile(monkeypatch):
    requests = []
    rendered = _calibration_render(config.CAMERA_HEIGHT_M)
    monkeypatch.setattr(
        sim_module.perception, "unproject",
        lambda _depth, _k: (_calibration_cloud(), np.zeros((3721, 2))))
    monkeypatch.setattr(
        sim_module.perception, "to_agent_ground",
        lambda points, camera_height: points)

    def render(position, yaw, cam_h, hfov, vfov):
        requests.append((cam_h, hfov, vfov))
        return rendered

    sim_module.sample_pose(
        _AnywherePathfinder(), render, np.random.default_rng(0), [0.2],
        yaws=[0.0], max_tries=1, min_floor=0.0)

    assert requests == [config.calibration_profile()]


def test_a_render_that_ignores_the_requested_profile_is_refused(monkeypatch):
    # Not a scene property but a programming error, so it must surface rather
    # than quietly calibrate the pose against the wrong camera.
    rendered = _calibration_render(config.CAMERA_HEIGHT_M + 0.7)
    monkeypatch.setattr(
        sim_module.perception, "unproject",
        lambda _depth, _k: (_calibration_cloud(), np.zeros((3721, 2))))
    monkeypatch.setattr(
        sim_module.perception, "to_agent_ground",
        lambda points, camera_height: points)

    with pytest.raises(ValueError, match="camera height mismatch"):
        sim_module.sample_pose(
            _AnywherePathfinder(), lambda *_args: rendered,
            np.random.default_rng(0), [0.2], yaws=[0.0], max_tries=1,
            min_floor=0.0)


def test_pose_rng_and_pathfinder_seed_are_independent_per_pose():
    class Pathfinder:
        def __init__(self):
            self.seeds = []

        def seed(self, value):
            self.seeds.append(value)

    class Sampler:
        def __init__(self):
            self.pathfinder = Pathfinder()
            self.navmesh_calls = []

        def recompute_navmesh(self, radius, height):
            self.navmesh_calls.append((radius, height))

    sampler = Sampler()
    first = collect._prepare_pose_sampling(
        sampler, [0.15, 0.20, 0.25], base_seed=42,
        scene_id="scene-a", pose_index=3)
    first_values = first.integers(0, 2**31, size=8).tolist()
    # Consuming an unrelated pose stream must not affect this pose's replay.
    collect._prepare_pose_sampling(
        sampler, [0.15, 0.20, 0.25], base_seed=42,
        scene_id="scene-a", pose_index=2).random(1000)
    replay = collect._prepare_pose_sampling(
        sampler, [0.15, 0.20, 0.25], base_seed=42,
        scene_id="scene-a", pose_index=3)

    assert replay.integers(0, 2**31, size=8).tolist() == first_values
    assert sampler.navmesh_calls == [
        (0.25, config.GROUND_ORACLE_HEIGHT_M),
        (0.25, config.GROUND_ORACLE_HEIGHT_M),
        (0.25, config.GROUND_ORACLE_HEIGHT_M),
    ]
    assert sampler.pathfinder.seeds[0] == sampler.pathfinder.seeds[-1]
    assert sampler.pathfinder.seeds[0] != sampler.pathfinder.seeds[1]


def test_pose_diversity_rejects_only_spatially_and_angularly_near_duplicates():
    exclusions = [{
        "position": [0.0, 0.0, 0.0],
        "yaw_rad": 0.0,
    }]

    assert config.POSE_DIVERSITY_POSITION_M == 0.75
    assert config.POSE_DIVERSITY_YAW_DEG == 45.0
    assert not collect._pose_is_diverse(
        [0.74, 0.0, 0.0], np.deg2rad(44.0), exclusions)
    assert collect._pose_is_diverse(
        [0.75, 0.0, 0.0], np.deg2rad(44.0), exclusions)
    assert collect._pose_is_diverse(
        [0.0, 0.0, 0.0], np.deg2rad(45.0), exclusions)
    assert collect._pose_is_diverse(
        [0.74, 0.0, 0.0], np.deg2rad(90.0), exclusions)


def test_pose_publication_clearance_matches_safe_action_margin():
    class FakeNav:
        def __init__(self, clearance):
            self._clearance = clearance

        def clearance(self, pose):
            assert pose == (0.0, 0.0, 0.0)
            return self._clearance

    class FakeSampler:
        def __init__(self, clearance):
            self._clearance = clearance

        def nav(self, _position, _yaw):
            return FakeNav(self._clearance)

    assert collect._pose_has_publication_clearance(
        FakeSampler(config.BENCH_SAFE_CLEARANCE_M), [0.0, 0.0, 0.0], 0.0)
    assert not collect._pose_has_publication_clearance(
        FakeSampler(config.BENCH_SAFE_CLEARANCE_M - 0.01),
        [0.0, 0.0, 0.0], 0.0)


def test_collection_shard_id_makes_pose_and_frame_ids_unique_across_rounds():
    assert collect.pose_group_id(
        "scene", 0, "main-r00") == "scene-main-r00-p000-observation"
    assert collect.frame_id(
        "scene", 0, "main-r01", "h150") == "F-scene-main-r01-p000-h150"


def test_removed_min_complete_lengths_option_is_rejected(tmp_path):
    parser = collect.build_parser()
    args = parser.parse_args(["--auto-scenes", "--out", str(tmp_path)])
    collect._validate_collection_args(args, parser)
    assert not hasattr(args, "min_complete_lengths")

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--auto-scenes", "--out", str(tmp_path),
            "--min-complete-lengths", "2",
        ])


def _coverage_record(
        *, intervention_id, group_id, length):
    selection = {"action_group_ids": [group_id]}
    return {
        "intervention": {"group_id": intervention_id},
        "selection": selection,
        "outcomes": [{
            "action_group_id": group_id,
            "actions": [
                {"type": "forward", "m": 0.5}
                for _index in range(length)
            ],
        }],
    }


def test_finalize_accepts_a_source_valid_record_without_length_coverage(
        tmp_path):
    row = {
        "schema_version": abc1_record.SCHEMA_VERSION,
        "record_uid": "r2r-partial", "dataset": "r2r",
        "cases": [{
            "case_id": "safe", "group_id": "safe",
            "actions": [{"type": "forward", "m": 0.5}],
            "starts_with": "forward", "collision": False,
        }],
    }
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(row) + "\n")
    args = SimpleNamespace(
        out=str(tmp_path), code_revision="abc123", allow_dirty_code=False,
        keep_per_length=1)

    class Funnel:
        value = {"run_contract_sha256": "unused"}

        def __init__(self):
            self.completed = False

        def complete(self):
            self.completed = True

    funnel = Funnel()
    funnel.path = tmp_path / "funnel.json"
    funnel.path.write_text(json.dumps({"status": "running"}))

    result = collection_runtime._finalize_collection_run(
        args=args, scenes=[], started=0.0,
        stats=collections.Counter(), skipped=collections.Counter(),
        records_path=str(records_path), funnel=funnel,
        existing_records=0, completed_groups=set(),
        run_contract={"sampling_provenance": {}})

    assert result == 0
    assert funnel.completed is True
    metadata = json.loads((tmp_path / "run_meta.json").read_text())
    assert "formal_action_length_coverage" not in metadata


def test_pose_selection_retains_all_certified_candidates():
    pools = {1: [
        (f"tag-{index}", [Forward(0.5 + 0.5 * index)])
        for index in range(4)
    ]}
    state = {
        "variants": [],
        "pools": pools,
        "group_labels": {tag: "safe" for tag, _actions in pools[1]},
        "precheck_cache": {},
        "proposal_provenance": {},
        "required_siblings": 0,
        "calibration": None,
        "position": None,
        "yaw": 0.0,
        "base_frame": None,
        "family_candidates": (),
    }

    selected = collection_runtime._select_pose_candidates(
        state,
        args=SimpleNamespace(setting_sampling_policy=None, seed=0),
        scene_id="scene", pose_index=0,
        stats=collections.Counter(), skipped=collections.Counter())

    assert selected["selected_ids"] == [
        "tag-0", "tag-1", "tag-2", "tag-3"]


@pytest.mark.parametrize("evaluation", ["perturbed", "nominal", "rejected"])
def test_pose_evaluation_keeps_a_stable_partial_action_bank(monkeypatch, evaluation):
    outcome = {"shared_oracle_stability": {"summary": {
        "collision_label_stable": evaluation == "perturbed",
    }}}
    state = {
        "variants": [],
        "pools": {1: [("safe", [Forward(0.5)])]},
        "group_labels": {"safe": "safe"},
        "precheck_cache": {},
        "proposal_provenance": {"safe": {"variant": "natural_dynamic"}},
        "active_radii": [0.2],
        "selection_seed": 1,
        "selected": [("safe", [Forward(0.5)])],
        "c1_families": (),
    }
    monkeypatch.setattr(
        collection_runtime, "_make_structured_spec_evaluator",
        lambda **_kwargs: (lambda _spec: {"frame": [outcome]}))
    monkeypatch.setattr(
        collection_runtime, "finalize_a_stability_certificates",
        lambda _pending: None)

    selected = collection_runtime._evaluate_pose_candidates(
        state, args=SimpleNamespace(ordinary_actions_per_pose=1,
                                   oracle_evaluation=evaluation),
        stats=collections.Counter(), skipped=collections.Counter())

    if evaluation == "rejected":
        assert selected is None
        return
    assert selected is not None
    assert selected["selected_ids"] == ["safe"]


def test_nominal_collection_reuses_dual_oracles_without_pose_rerollouts(monkeypatch):
    physical = {"collision": False, "contact_action_index": None}
    outcome = {"physical": physical, "depth_physical": dict(physical),
               "oracle_consensus": {"accepted": True},
               "evidence": {"physical": {"coverage": 1.0}}}
    sim = SimpleNamespace(source_dataset="b1k", nav=lambda *_args: None,
                          recompute_navmesh=lambda *_args, **_kwargs: None)
    frame = SimpleNamespace(frame_id="frame", position=[0, 0, 0], yaw_rad=0)
    monkeypatch.setattr(collection_runtime.rollout, "physical_rollout",
                        lambda *_args, **_kwargs: physical)
    monkeypatch.setattr(collection_runtime, "judge", lambda *_args, **_kwargs: outcome)

    def perturb(*_args, **_kwargs):
        raise AssertionError("nominal collection must not perturb the saved pose")

    monkeypatch.setattr(collection_runtime, "collect_a_stability_rows", perturb)
    evaluator = collection_runtime._make_structured_spec_evaluator(
        args=SimpleNamespace(collection_mode="main", oracle_evaluation="nominal"),
        variants=[(sim, frame)], group_labels={"safe": "safe"},
        precheck_cache={}, stats=collections.Counter(), skipped=collections.Counter(),
        pending_a_certificates=[])
    program = [Forward(1.0)]
    result = evaluator({"action_tag": "safe", "actions": program, "radii": [0.2]})
    published = result["frame"][0]
    assert published["shared_oracle_stability"]["version"] == "nominal-oracle.v1"
    assert validate._a_stability_validation_errors(published, program, "test") == []


def test_full_geometry_exclusion_reports_its_source():
    stats = collections.Counter()
    skipped = collections.Counter()

    label = collection_runtime._precheck_action_candidate(
        "action", [Forward(0.5)],
        {0.2: {
            "collision": None,
            "collision_source": "unsupported_floor",
            "minimum_clearance_m": None,
        }},
        variants=[], active_radii=[0.2], required_siblings=1,
        proposal_provenance={"action": {"variant": "natural_dynamic"}},
        stats=stats, skipped=skipped, group_labels={}, precheck_cache={})

    assert label is None
    assert skipped == collections.Counter({
        "full_geometry_excluded.unsupported_floor": 1})


def test_partial_frame_evidence_keeps_later_valid_action_at_same_pose(
        monkeypatch):
    calibration = SimpleNamespace(
        position=np.array([0.0, 0.0, 0.0]),
        yaw_rad=0.0,
        canonical_plane=LEVEL_FLOOR,
        reference_observation=None,
    )
    sampler = SimpleNamespace(
        sample_random_pose=lambda *_args, **_kwargs: calibration,
    )
    frame = SimpleNamespace(
        frame_id="frame", objects=[],
        quality={"valid_depth_ratio": 0.1, "visible_floor_ratio": 0.0},
        camera_height_above_visible_floor_m=1.0,
        sensor=SimpleNamespace(to_dict=lambda: {"resolution": [2, 2]}))
    monkeypatch.setattr(
        collection_runtime, "_prepare_pose_sampling",
        lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        collection_runtime, "build_frame", lambda *_args, **_kwargs: frame)
    monkeypatch.setattr(
        collection_runtime, "precompute_full_geometry_candidates",
        lambda *_args, **_kwargs: {
            "rejected": {
                0.15: {
                    "collision": False,
                    "minimum_clearance_m": 0.1,
                },
            },
            "accepted": {
                0.15: {
                    "collision": False,
                    "minimum_clearance_m": 0.5,
                },
            },
        })
    monkeypatch.setattr(
        collection_runtime.rollout,
        "view_collision_rollout",
        lambda *_args, **_kwargs: {"collision": False})
    monkeypatch.setattr(
        collection_runtime.rollout,
        "corridor_coverage",
        lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        collection_runtime,
        "oracle_consensus",
        lambda *_args, **_kwargs: {"accepted": True, "reason": "accepted"})
    monkeypatch.setattr(
        collection_runtime.action_sampling,
        "classify_action_group",
        lambda *_args, **_kwargs: "safe")
    stats = collections.Counter()
    skipped = collections.Counter()

    state = collection_runtime._prepare_pose_candidates(
        args=SimpleNamespace(
            seed=1,
            collection_shard_id="main",
            collection_mode="main",
            radii=(0.15,),
            # File mode passes its programs through untouched, which is how
            # this test injects the exact two candidates it needs.
            action_mode="file",
        ),
        sampler=sampler,
        sample_radii=(0.15,),
        scene_id="scene",
        pose_index=0,
        pose_tries_per_attempt=1,
        pose_exclusions={},
        sessions=[object()],
        fovs=[(79.0, 63.453)],
        heights=[1.0],
        scene=SimpleNamespace(scene_path="scene.glb"),
        proposal_bank=collection_runtime.ProposalBank(
            templates=[],
            natural_pools={
                1: [
                    ("rejected", [Forward(0.5)]),
                    ("accepted", [Forward(1.0)]),
                ],
            }),
        stats=stats,
        skipped=skipped,
        intervention_group_id="group",
    )

    assert state is not None
    assert state["group_labels"] == {"accepted": "safe"}
    assert skipped["safe_publication_margin"] == 1
    assert stats["candidate.accepted.safe.L1"] == 1


def test_persist_pose_group_drops_generic_target_metadata(monkeypatch, tmp_path):
    from pipeline import collection_runtime
    from pipeline.action_proposal import DEPTH_CONDITIONED_POLICY
    from tests.test_pl_b1k_contracts import _b1k_source

    captured = {}
    publication_events = []
    source = _b1k_source(scene_id="scene")

    class SavedImage:
        def save(self, _path):
            pass

    monkeypatch.setattr(
        collection_runtime.Image, "fromarray",
        lambda _array: SavedImage())
    monkeypatch.setattr(
        collection_runtime, "validate_records_before_spool",
        lambda _records, **kwargs: (
            captured.setdefault(
                "validation_context", kwargs["validation_context"]),
            publication_events.append("validate")))
    monkeypatch.setattr(
        collection_runtime, "append_record_group",
        lambda *_args, **_kwargs: publication_events.append("commit"))
    monkeypatch.setattr(
        collection_runtime.abc1_record, "from_collected",
        lambda record_value, **_kwargs: record_value)
    def build_record(*_args, **kwargs):
        captured.update(kwargs)
        return {"outcomes": []}

    monkeypatch.setattr(collection_runtime, "build_record", build_record)
    frame = SimpleNamespace(
        frame_id="frame",
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32),
    )
    funnel = SimpleNamespace(record_pose=lambda *_args: None)
    state = {
        "selected_ids": [],
        "group_labels": {},
        "selected": [],
        "diagnostics": [],
        "accepted_outcomes": {},
        "variants": [(SimpleNamespace(
            scene_id="scene", scene_authority_sha256="a" * 64), frame)],
            "render_caches": {"frame": {}},
            "pools": {}, "shortlist_size": 1,
        "selection_seed": 1,
        "required_siblings": 1,
        "calibration": SimpleNamespace(canonical_floor_fit=LEVEL_FLOOR_FIT),
        "position": np.array([0.0, 0.0, 0.0]),
        "yaw": 0.0,
        "intervention_group_id": "group",
        "scene": SimpleNamespace(provenance=lambda: source),
        "scene_id": "scene",
        "active_radii": [0.2],
        "setting": None,
        "proposal_provenance": {},
        "bank_manifest": None,
    }

    assert collection_runtime._persist_pose_group(
        state,
        args=SimpleNamespace(
            debug_images=False,
            save_arrays=False,
            out=str(tmp_path),
            collection_mode="main",
            action_mode="balanced",
            radii=(0.15, 0.20, 0.25),
            setting_sampling_policy=None,
            backend="b1k",
            b1k_data_root=str(tmp_path),
            b1k_source_manifest=str(tmp_path / "source.json"),
        ),
        image_dir=str(tmp_path / "img"),
        array_dir=str(tmp_path / "arr"),
        records_path=tmp_path / "records.jsonl",
        scene_index=0,
        pose_index=0,
        sampler_contract_hash="bank",
        intervention_type="base",
        changed_fields=[],
        funnel=funnel,
        stats=collections.Counter(),
    ) is True
    assert "target_ids" not in captured
    assert "review_evidence" not in captured
    assert publication_events == ["validate", "commit"]
    assert captured["validation_context"].expected_action_sampling_policy == \
        DEPTH_CONDITIONED_POLICY


def test_gs_persist_uses_v18_without_a_legacy_collection_contract(
        monkeypatch, tmp_path):
    captured = {}
    terminal_calls = []
    source = source_provenance("scene", dataset="gs")

    class SavedImage:
        def save(self, _path):
            pass

    monkeypatch.setattr(
        collection_runtime.Image, "fromarray", lambda _array: SavedImage())
    monkeypatch.setattr(
        collection_runtime.REC, "collection_contract",
        lambda *_args, **_kwargs: pytest.fail(
            "GS v18 attempted to build a legacy collection contract"))
    monkeypatch.setattr(
        collection_runtime, "validate_records_before_spool",
        lambda _records, **kwargs: captured.setdefault(
            "validation_context", kwargs["validation_context"]))
    monkeypatch.setattr(
        collection_runtime, "append_record_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        collection_runtime.abc1_record, "from_collected",
        lambda record_value, **_kwargs: record_value)
    def attach_terminal(*_args, **kwargs):
        terminal_calls.append(kwargs)
        return {"materialized": 0, "withheld": 0}
    monkeypatch.setattr(
        collection_runtime, "attach_terminal_rgb_assets", attach_terminal)

    def build_v18(*_args, **kwargs):
        captured["builder_kwargs"] = kwargs
        return {"outcomes": []}

    monkeypatch.setattr(collection_runtime.REC, "build_record_v18", build_v18)
    frame = SimpleNamespace(
        frame_id="frame", rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32))
    state = {
        "selected_ids": [], "group_labels": {}, "selected": [],
        "diagnostics": [], "accepted_outcomes": {},
        "variants": [(SimpleNamespace(scene_id="scene"), frame)],
            "render_caches": {"frame": {}}, "pools": {},
            "shortlist_size": 1, "selection_seed": 1,
        "required_siblings": 1,
        "calibration": SimpleNamespace(canonical_floor_fit=LEVEL_FLOOR_FIT),
        "position": np.zeros(3), "yaw": 0.0,
        "intervention_group_id": "group",
        "scene": SimpleNamespace(provenance=lambda: source),
        "scene_id": "scene", "active_radii": [0.2], "setting": None,
        "proposal_provenance": {}, "bank_manifest": None,
    }

    assert collection_runtime._persist_pose_group(
        state,
        args=SimpleNamespace(
            debug_images=False, save_arrays=False, out=str(tmp_path),
            collection_mode="main", action_mode="balanced",
            radii=(0.15, 0.20, 0.25), setting_sampling_policy=None,
            backend="gs"),
        image_dir=str(tmp_path / "img"), array_dir=str(tmp_path / "arr"),
        records_path=tmp_path / "records.jsonl", scene_index=0, pose_index=0,
        sampler_contract_hash="bank", intervention_type="base",
        changed_fields=[],
        funnel=SimpleNamespace(record_pose=lambda *_args: None),
        stats=collections.Counter()) is True
    assert captured["builder_kwargs"]["collection_contract"] is None
    assert captured["validation_context"].route == "gs_v18_registered"
    assert len(terminal_calls) == 1
    assert terminal_calls[0]["collection_contract"] is None
    assert terminal_calls[0]["eligible_outcome_ids"] == set()



@pytest.mark.parametrize(
    ("accepted", "evidence_status", "expected"),
    [
        (True, "sufficient", "keep"),
        (True, "insufficient", "keep"),
        (False, "insufficient", "reject_spec"),
        (False, "sufficient", "reject_spec"),
    ],
)
def test_structured_outcome_disposition_follows_the_oracle_alone(
        accepted, evidence_status, expected):
    # The evidence gap used to buy a directed spec a partial keep. With
    # directed collection gone, consensus is the only thing that decides.
    outcome = {
        "oracle_consensus": {"accepted": accepted},
        "evidence": {"physical": {"status": evidence_status}},
    }

    assert collect.structured_outcome_disposition(outcome) == expected


@pytest.mark.parametrize(
    "certificate",
    [None, {}, {"malformed": True}, {"summary": {"collision": True}}],
)
def test_structured_disposition_does_not_consume_deferred_certificate(
        certificate):
    outcome = {
        "physical": {"authority": "navmesh", "collision": False},
        "oracle_consensus": {"accepted": True},
    }
    if certificate is not None:
        outcome["shared_oracle_stability"] = certificate

    assert collect.structured_outcome_disposition(outcome) == "keep"


def test_structured_outcome_disposition_rejects_excluded_geometry():
    outcome = {
        "physical": {
            "authority": "gs_collision_mesh",
            "collision": None,
            "collision_source": "navmesh_boundary",
        },
        "oracle_consensus": {"accepted": True},
        "evidence": {"physical": {"status": "sufficient"}},
    }

    assert collect.structured_outcome_disposition(outcome) == "reject_spec"


def test_pose_attempt_budget_never_undercuts_the_candidate_pool():
    # One requested pose still searches the full candidate budget, so a scene
    # is not abandoned after a single rejected start.
    assert collect.pose_attempt_budget(2) == 2
    assert collect.pose_attempt_budget(
        1, pose_candidates_per_scene=100) == 100


def test_explicit_pose_candidate_budget_counts_individual_draws():
    # A controller budget of 800 means 800 raw pose draws, not 800 batches of
    # the backend's hidden 400/2000-draw default search.
    assert collect.pose_candidate_draws_per_attempt(None) is None
    assert collect.pose_candidate_draws_per_attempt(800) == 1


def test_collect_scene_applies_individual_pose_draw_budget(monkeypatch):
    draws = []
    stats = collections.Counter()
    session = SimpleNamespace(scene_id="scene")
    monkeypatch.setattr(
        collection_runtime, "_open_collection_sessions",
        lambda *_args, **_kwargs: [session])
    monkeypatch.setattr(
        collection_runtime, "emit_backend_ready",
        lambda **_kwargs: None)
    monkeypatch.setattr(
        collection_runtime, "_collect_pose",
        lambda **kwargs: draws.append(
            kwargs["pose_tries_per_attempt"]) or len(draws) == 2)
    monkeypatch.setattr(
        collection_runtime, "_finish_scene_cleanup",
        lambda *_args: None)

    collection_runtime._collect_scene(
        args=SimpleNamespace(
            backend="r2r", radii=(0.15,),
            poses_per_scene=1, pose_candidates_per_scene=2,
            collection_mode="main",
            collection_shard_id="shard"),
        scene=SimpleNamespace(scene_id="scene"), scene_index=0,
        scene_count=1, heights=[1.0], fovs=[(79.0, 63.453)],
        proposal_bank=object(), stats=stats,
        skipped=collections.Counter(), pose_exclusions={},
        completed_groups=set(), image_dir="img", array_dir="arr",
        records_path="records.jsonl", sampler_contract_hash="digest",
        intervention_type="base", changed_fields=[],
        funnel=SimpleNamespace(begin_scene=lambda *_args: None),
    )

    assert draws == [1, 1]
    assert stats["accepted_pose_attempt_index_sum"] == 2
    assert stats["accepted_pose_attempt_0001_1000"] == 1
    assert stats["pose_attempt_wallclock_us"] > 0


def test_collect_scene_stops_normally_after_record_idle_budget(monkeypatch):
    """Capacity exhaustion is an ordinary return, never KeyboardInterrupt."""
    draws = []
    stats = collections.Counter()
    skipped = collections.Counter()
    session = SimpleNamespace(scene_id="scene")
    monotonic_values = iter((0.0, 0.0, 1.0, 122.0))
    monkeypatch.setattr(
        collection_runtime, "_open_collection_sessions",
        lambda *_args, **_kwargs: [session])
    monkeypatch.setattr(
        collection_runtime, "emit_backend_ready", lambda **_kwargs: None)
    monkeypatch.setattr(
        collection_runtime.time, "monotonic",
        lambda: next(monotonic_values))
    monkeypatch.setattr(
        collection_runtime, "_collect_pose",
        lambda **_kwargs: draws.append(True) or True)
    monkeypatch.setattr(
        collection_runtime, "_finish_scene_cleanup",
        lambda *_args: None)

    stop = collection_runtime._collect_scene(
        args=SimpleNamespace(
            backend="r2r", radii=(0.15,),
            poses_per_scene=10, pose_candidates_per_scene=3,
            collection_mode="main",
            collection_shard_id="shard", record_idle_stop_s=120.0,
            scene_wallclock_stop_s=600.0),
        scene=SimpleNamespace(scene_id="scene"), scene_index=0,
        scene_count=1, heights=[1.0], fovs=[(79.0, 63.453)],
        proposal_bank=object(), stats=stats, skipped=skipped,
        pose_exclusions={}, completed_groups=set(), image_dir="img",
        array_dir="arr", records_path="records.jsonl",
        sampler_contract_hash="digest", intervention_type="base",
        changed_fields=[],
        funnel=SimpleNamespace(begin_scene=lambda *_args: None),
    )

    assert draws == [True]
    assert stop == {
        "reason": "inter_record_idle",
        "accepted_records": 1,
        "pose_attempts": 1,
        "active_seconds": 122.0,
        "record_idle_stop_s": 120.0,
        "scene_wallclock_stop_s": 600.0,
    }
    assert skipped["capacity_stop_inter_record_idle"] == 1


def test_capacity_stop_is_bound_into_run_meta_and_sealed_marker(
        tmp_path, monkeypatch):
    records = [
        _coverage_record(
            intervention_id=f"pose-L{length}", group_id=f"g-L{length}",
            length=length)
        for length in config.GEN_LENGTHS
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    args = SimpleNamespace(
        out=str(tmp_path), code_revision="abc123", allow_dirty_code=False,
        keep_per_length=1)
    funnel_path = tmp_path / "collection_funnel.json"
    funnel_path.write_text(json.dumps({"status": "running"}))

    class Funnel:
        value = {"run_contract_sha256": "unused"}
        path = funnel_path

        def complete(self):
            self.path.write_text(json.dumps({"status": "completed"}))

    stop = {
        "reason": "inter_record_idle", "accepted_records": 6,
        "pose_attempts": 7, "active_seconds": 121.0,
        "record_idle_stop_s": 120.0, "scene_wallclock_stop_s": 600.0,
    }
    monkeypatch.setattr(collection_runtime.time, "time", lambda: 200.0)

    result = collection_runtime._finalize_collection_run(
        args=args, scenes=[], started=100.0,
        stats=collections.Counter(), skipped=collections.Counter(),
        records_path=str(records_path), funnel=Funnel(),
        existing_records=0, completed_groups=set(),
        run_contract={"sampling_provenance": {}}, capacity_stop=stop)

    assert result == 0
    metadata = json.loads((tmp_path / "run_meta.json").read_text())
    assert metadata["capacity_stop"] == stop
    marker = json.loads(
        (tmp_path / "collection_finalization.json").read_text())
    assert marker["schema"] == "egoconseq.collection-finalization.v2"
    assert marker["status"] == "completed"
    assert marker["source_validation"] == "passed"
    assert marker["record_count"] == 6
    assert marker["capacity_stop_reason"] == "inter_record_idle"
    assert marker["records_sha256"] == collection_runtime.io_utils.sha256_file(
        records_path)
    assert marker["run_meta_sha256"] == collection_runtime.io_utils.sha256_file(
        tmp_path / "run_meta.json")
    assert marker["funnel_sha256"] == collection_runtime.io_utils.sha256_file(
        funnel_path)


def test_successful_pose_becomes_a_same_shard_diversity_exclusion(
        monkeypatch):
    from pipeline import collection_runtime

    state = {
        "position": np.array([1.0, 0.0, 2.0]),
        "yaw": 0.5,
    }
    monkeypatch.setattr(
        collection_runtime, "_prepare_pose_candidates",
        lambda **_kwargs: dict(state))
    monkeypatch.setattr(
        collection_runtime, "_select_pose_candidates",
        lambda value, **_kwargs: value)
    monkeypatch.setattr(
        collection_runtime, "_evaluate_pose_candidates",
        lambda value, **_kwargs: value)
    monkeypatch.setattr(
        collection_runtime, "_persist_pose_group",
        lambda value, **_kwargs: True)
    funnel = SimpleNamespace(record_searched_pose=lambda *_args: None)
    exclusions = {}

    accepted = collection_runtime._collect_pose(
        args=SimpleNamespace(
            collection_mode="main",
            collection_shard_id="main"),
        sampler=object(),
        sample_radii=(0.15,),
        scene_id="scene",
        pose_index=0,
        pose_tries_per_attempt=1,
        pose_exclusions=exclusions,
        sessions=[],
        fovs=[],
        heights=[],
        scene=object(),
        proposal_bank=collection_runtime.ProposalBank([], {}),
        stats=collections.Counter(),
        skipped=collections.Counter(),
        completed_groups=set(),
        image_dir="img",
        array_dir="arr",
        records_path="records.jsonl",
        scene_index=0,
        sampler_contract_hash="hash",
        intervention_type="base",
        changed_fields=[],
        funnel=funnel,
    )

    assert accepted is True
    assert exclusions == {
        "scene": [{
            "position": [1.0, 0.0, 2.0],
            "yaw_rad": 0.5,
        }],
    }


def test_resume_preloads_existing_record_poses_into_same_shard_exclusions(
        tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in [
        {"scene_id": "scene-a", "pose": {
            "position": [1.0, 0.0, 2.0], "yaw_rad": 0.5}},
        {"scene_id": "scene-b", "pose": {
            "position": [3.0, 0.0, 4.0], "yaw_rad": -0.5}},
    ]))
    exclusions = {"scene-a": [
        {"position": [1.0, 0.0, 2.0], "yaw_rad": 0.5},
    ]}

    collection_runtime._preload_resumed_pose_exclusions(
        records, exclusions)

    assert exclusions == {
        "scene-a": [
            {"position": [1.0, 0.0, 2.0], "yaw_rad": 0.5},
        ],
        "scene-b": [
            {"position": [3.0, 0.0, 4.0], "yaw_rad": -0.5},
        ],
    }


def test_safe_candidate_requires_publication_clearance_for_every_radius():
    full = {
        0.15: {"collision": False, "minimum_clearance_m": 0.45},
        0.20: {"collision": False, "minimum_clearance_m": 0.30},
        0.25: {"collision": False, "minimum_clearance_m": 0.31},
    }

    assert collect._safe_publication_ready(full)

    tight = copy.deepcopy(full)
    tight[0.20]["minimum_clearance_m"] = 0.299
    assert not collect._safe_publication_ready(tight)

    collision = copy.deepcopy(full)
    collision[0.25]["collision"] = True
    assert not collect._safe_publication_ready(collision)


def test_full_geometry_publication_label_filters_before_depth_rollout():
    safe = {
        0.15: {"collision": False, "minimum_clearance_m": 0.45},
        0.20: {"collision": False, "minimum_clearance_m": 0.40},
        0.25: {"collision": False, "minimum_clearance_m": 0.35},
    }
    collision = {
        0.15: {"collision": True, "first_contact_arc_m": 0.75},
        0.20: {"collision": True, "first_contact_arc_m": 0.72},
        0.25: {"collision": True, "first_contact_arc_m": 0.70},
    }

    assert collect.full_geometry_publication_label(
        [Forward(2.0)], safe) == "safe"
    assert collect.full_geometry_publication_label(
        [Forward(2.0)], collision) == "collision"

    near_old_half_boundary = copy.deepcopy(collision)
    near_old_half_boundary[0.25]["first_contact_arc_m"] = 1.05
    assert collect.full_geometry_publication_label(
        [Forward(2.0)], near_old_half_boundary) == "collision"

    collision[0.25]["first_contact_arc_m"] = 1.70
    assert collect.full_geometry_publication_label(
        [Forward(2.0)], collision) is None

    mixed = copy.deepcopy(safe)
    mixed[0.25] = {
        "collision": True, "first_contact_arc_m": 0.7,
        "minimum_clearance_m": 0.0,
    }
    assert collect.full_geometry_publication_label(
        [Forward(2.0)], mixed) is None


def test_run_contract_records_pose_independent_action_bank_hash():
    args = SimpleNamespace(
        out="ignored", overwrite=False, resume=True, debug_images=False,
        debug_outcomes_per_frame=4, action_file=None,
    )

    contract = collect.collection_run_contract(
        args, ["scene"], [1.0],
        [(79.0, config.vfov_for_hfov(79.0))],
        action_sampler_contract_sha256="a" * 64)

    assert contract["action_sampler_contract_sha256"] == "a" * 64


def test_run_contract_carries_each_resolved_scene_asset_identity():
    scene = _source_scene_spec()
    args = SimpleNamespace(
        collection_mode="main",
        out="ignored",
        overwrite=False,
        resume=True,
        debug_images=False,
        debug_outcomes_per_frame=4,
        action_file=None,
    )

    contract = collect.collection_run_contract(
        args,
        [scene],
        [1.0],
        [(79.0, config.vfov_for_hfov(79.0))],
        sampling_provenance=collect.collection_sampling_provenance(),
    )

    resolved = contract["resolved_scenes"][0]
    assert resolved["source_assets"] == scene.provenance()["source_assets"]
    assert resolved["source_assets_sha256"] == \
        scene.provenance()["source_assets_sha256"]


def test_run_contract_records_inline_pose_sampling_provenance():
    args = SimpleNamespace(
        collection_mode="main",
        out="ignored",
        overwrite=False,
        resume=True,
        debug_images=False,
        debug_outcomes_per_frame=4,
        action_file=None,
    )

    contract = collect.collection_run_contract(
        args,
        ["scene"],
        [1.0],
        [(79.0, config.vfov_for_hfov(79.0))],
        sampling_provenance=collect.collection_sampling_provenance(),
    )

    assert contract["sampling_provenance"] == {"pose_discovery": "inline"}
    assert contract["main_action_proposal"] == {
        "cap_per_length": config.MAIN_ACTION_PROPOSAL_PER_LENGTH,
        "ranking": "label-blind-stratified-shortlist.v4",
        "retention": "progressive-stability-fill.v1",
        "ordinary_attempts_per_pose": 48,
        "ordinary_actions_per_pose": 36,
        "c1_queries_per_pose": 2,
        "c1_neighbors_per_query": 6,
        "dynamic_natural": "pose-depth-budget.v1",
        "a1_control_catalog_sha256":
            action_control_catalog.CATALOG_SHA256,
    }


def test_run_meta_copies_sampling_provenance_from_run_contract(tmp_path):
    provenance = {"pose_discovery": "inline"}
    args = SimpleNamespace(
        out=str(tmp_path),
        code_revision="abc123",
        allow_dirty_code=False,
    )

    class Funnel:
        path = tmp_path / "collection_funnel.json"

        def complete(self):
            self.path.write_text(json.dumps({"status": "completed"}))

    (tmp_path / "records.jsonl").write_text("")
    Funnel.path.write_text(json.dumps({"status": "running"}))

    run_contract = {
        "sampling_provenance": provenance,
        "resolved_scenes": [{
            **source_provenance("scene/example", dataset="r2r"),
            "scene_path": "/datasets/mp3d/scene.glb",
        }],
        "candidate_rejection_scope": "action",
        "action_sampler_contract_sha256": "c" * 64,
        "main_action_proposal": {
            "cap_per_length": config.MAIN_ACTION_PROPOSAL_PER_LENGTH,
            "ranking": "deterministic candidate-bank prefix",
        },
    }
    result = collect._finalize_collection_run(
        args=args,
        scenes=[],
        started=0.0,
        stats=collections.Counter(),
        skipped=collections.Counter(),
        records_path=str(tmp_path / "records.jsonl"),
        funnel=Funnel(),
        existing_records=0,
        completed_groups=set(),
        run_contract=run_contract,
    )

    assert result == 0
    metadata = json.loads((tmp_path / "run_meta.json").read_text())
    assert metadata["record_schema_version"] == \
        collection_runtime.abc1_record.SCHEMA_VERSION
    assert metadata["oracle_contract_version"] == record.ORACLE_CONTRACT_VERSION
    assert metadata["run_contract_sha256"] == \
        collection_funnel.canonical_sha256(run_contract)
    assert metadata["sampling_provenance"] == provenance
    assert metadata["resolved_scenes"] == run_contract["resolved_scenes"]
    assert metadata["candidate_rejection_scope"] == "action"
    assert metadata["action_sampler_contract_sha256"] == "c" * 64
    assert metadata["main_action_proposal"] == run_contract[
        "main_action_proposal"]
    assert not (tmp_path / "a3_prune_report.json").exists()


def test_run_contract_pins_revision_outside_params():
    # (B3 ③, collection half) The pinned revision and dirty flag ride the
    # contract as top-level keys, never as resume-sensitive params.
    args = SimpleNamespace(
        out="ignored", overwrite=False, resume=True, debug_images=False,
        debug_outcomes_per_frame=4, action_file=None,
        code_revision="sha-one", allow_dirty_code=True,
    )

    contract = collect.collection_run_contract(
        args, ["scene"], [1.0], [(79.0, config.vfov_for_hfov(79.0))])

    assert contract["code_revision"] == "sha-one"
    assert contract["code_dirty"] is True
    assert "code_revision" not in contract["params"]
    assert "allow_dirty_code" not in contract["params"]


def test_resume_rejects_cross_code_revision(tmp_path):
    # (B3 ④) Resuming a run pinned to a different commit is forbidden even when
    # every other contract field matches.
    path = tmp_path / "records.jsonl"
    _prepare(
        path, expected_siblings=2, overwrite=True,
        run_contract={"seed": 42, "code_revision": "sha-one"})

    with pytest.raises(ValueError, match="cross-revision resume is forbidden"):
        _prepare(
            path, expected_siblings=2, resume=True,
            run_contract={"seed": 42, "code_revision": "sha-two"})


def _file_args(path):
    return SimpleNamespace(
        action_mode="file", action_file=str(path),
        lengths=list(config.GEN_LENGTHS), keep_per_length=3, pool_factor=1)


def test_action_file_rejects_duplicate_programs(tmp_path):
    path = tmp_path / "actions.json"
    sequence = {"actions": [{"type": "forward", "m": 0.5}]}
    path.write_text(json.dumps({"sequences": [sequence, sequence]}))

    with pytest.raises(ValueError, match="duplicate action program"):
        collect.candidate_action_pools(
            _file_args(path), np.random.default_rng(0),
            collections.Counter(), half_fov_deg=39.5)


def test_action_file_rejects_values_outside_main_vocabulary(tmp_path):
    path = tmp_path / "actions.json"
    path.write_text(json.dumps({"sequences": [{
        "actions": [{"type": "forward", "m": -0.5}],
    }]}))

    with pytest.raises(ValueError, match="forward value"):
        collect.candidate_action_pools(
            _file_args(path), np.random.default_rng(0),
            collections.Counter(), half_fov_deg=39.5)


def test_full_geometry_candidate_precheck_rebuilds_once_per_radius():
    class FakeNav:
        authority = "test"

        def __init__(self, radius):
            self.radius = float(radius)
            self.batch_calls = 0

        def query_many(self, poses):
            self.batch_calls += 1
            return [SimpleNamespace(
                navigable=True,
                clearance_m=1.0 - self.radius,
                obstacle_index=None,
                geometry_source=None,
            ) for _pose in poses]

        def is_navigable(self, _pose):
            return True

        def clearance(self, _pose):
            return 1.0 - self.radius

    class FakeSim:
        source_dataset = "b1k"

        def __init__(self):
            self.radius = None
            self.recomputed = []
            self.navs = []

        def recompute_navmesh(self, radius, *, height):
            assert height == config.GROUND_ORACLE_HEIGHT_M
            self.radius = float(radius)
            self.recomputed.append(float(radius))

        def nav(self, _position, _yaw):
            nav = FakeNav(self.radius)
            self.navs.append(nav)
            return nav

    frame = SimpleNamespace(position=[0.0, 0.0, 0.0], yaw_rad=0.0)
    pools = {
        1: [("L1-a", [Forward(0.5)]), ("L1-b", [Forward(1.0)])],
        2: [("L2-a", [Turn(15.0), Forward(0.5)])],
    }
    stats = collections.Counter()
    sim = FakeSim()

    result = collect.precompute_full_geometry_candidates(
        sim, frame, pools, [0.15, 0.25, 0.35], stats)

    assert sim.recomputed == [0.15, 0.25, 0.35]
    assert [nav.batch_calls for nav in sim.navs] == [1, 1, 1]
    assert stats["physical_prechecks"] == 9
    assert set(result) == {"L1-a", "L1-b", "L2-a"}
    assert set(result["L1-a"]) == {0.15, 0.25, 0.35}
    assert all(not value["collision"]
               for by_radius in result.values()
               for value in by_radius.values())
    assert all(isinstance(value, rollout.PhysicalPathTrace)
               for by_radius in result.values()
               for value in by_radius.values())


@pytest.mark.parametrize("candidate_budget,valid", [(48, True), (49, False)])
def test_record_candidate_budget_accepts_48_and_rejects_49(
        candidate_budget, valid):
    rec = {
        "frame_id": "f",
        "selection": {
            "policy": collection_runtime.action_proposal.
                EXPLICIT_ACTION_FILE_POLICY,
            "action_group_ids": ["group"],
            "action_group_labels": {"group": "safe"},
            "candidate_budget": candidate_budget,
            "observed_lengths": [1],
            "observed_label_counts": {"safe": 1},
            "required_radii_m": [],
        },
        "outcomes": [{
            "action_group_id": "group",
            "action_group_label": "safe",
            "seq_len": 1,
            "actions": [{"type": "forward", "m": 0.5}],
            "body": {"radius_m": 0.2},
            "physical": {"collision": False},
        }],
    }

    errors = validate._validate_balanced_selection(rec)

    assert (not any("natural candidate budget is invalid" in error
                    for error in errors)) is valid


def test_candidate_stage_diagnostics_preserve_label_length_and_reason():
    stats = collections.Counter()

    collect.record_candidate_stage(
        stats, "full_ready", "collision", 4)
    collect.record_candidate_stage(
        stats, "depth_reject", "collision", 4,
        reason="insufficient_depth_coverage")
    collect.record_candidate_stage(
        stats, "depth_reject", "collision", 4,
        reason="insufficient_depth_coverage")

    assert stats == {
        "candidate.full_ready.collision.L4": 1,
        "candidate.depth_reject.collision.L4."
        "insufficient_depth_coverage": 2,
    }
