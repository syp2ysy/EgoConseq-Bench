"""Catalog revisits exclude only authenticated poses from earlier passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline import background_pose_exclusions


def _job(job_id: str, *, catalog_pass: int, output_root: Path,
         bind_exclusions: bool | None = None, dataset: str = "r2r",
         scene_id: str = "scene-a") -> dict:
    if bind_exclusions is None:
        bind_exclusions = catalog_pass > 0
    return {
        "job_id": job_id,
        "dataset": dataset,
        "catalog_pass": catalog_pass,
        "scenes": [scene_id],
        "pose_exclusions_path": (
            None if not bind_exclusions else
            str(output_root / "controller" / "pose_exclusions" /
                dataset / f"{scene_id}-pass{catalog_pass:02d}.json")),
    }


def test_revisit_pose_exclusions_come_from_source_valid_prior_records(tmp_path):
    first = _job("first", catalog_pass=0, output_root=tmp_path)
    revisit = _job("revisit", catalog_pass=1, output_root=tmp_path)
    records = tmp_path / "first" / "records.jsonl"
    records.parent.mkdir()
    rows = [
        {"scene_id": "scene-a", "pose": {
            "position": [1.0, 2.0, 3.0], "yaw_rad": 0.5}},
        {"scene_id": "other", "pose": {
            "position": [9.0, 9.0, 9.0], "yaw_rad": 1.0}},
        {"scene_id": "scene-a", "pose": {
            "position": [1.0, 2.0, 3.0], "yaw_rad": 0.5}},
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(records.read_bytes()).hexdigest()
    manifest = {"rounds": [
        {"jobs": [first]}, {"jobs": [revisit]},
    ]}
    state = {"jobs": {"first": {
        "catalog_status": "completed",
        "source_validation": {"sources": [{
            "path": str(records), "records_sha256": digest,
        }]},
    }}}

    path = background_pose_exclusions.materialize_for_job(
        manifest, state, revisit)

    assert path == Path(revisit["pose_exclusions_path"])
    assert json.loads(path.read_text()) == {
        "schema_version": "egoconseq.pose_exclusions.v1",
        "scenes": {"scene-a": [{
            "position": [1.0, 2.0, 3.0], "yaw_rad": 0.5,
        }]},
    }


def test_first_catalog_pass_has_no_pose_exclusion_side_effect(tmp_path):
    job = _job("first", catalog_pass=0, output_root=tmp_path)

    assert background_pose_exclusions.materialize_for_job(
        {"rounds": [{"jobs": [job]}]}, {"jobs": {}}, job) is None
    assert not (tmp_path / "controller" / "pose_exclusions").exists()


def test_first_catalog_pass_materializes_authenticated_seed(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({
        "schema_version": "egoconseq.pose_exclusions.v1",
        "scenes": {
            "scene-a": [{"position": [4.0, 0.0, 2.0], "yaw_rad": 1.5}],
        },
    }))
    job = _job(
        "first", catalog_pass=0, output_root=tmp_path,
        bind_exclusions=True)
    manifest = {
        "seed_pose_exclusions": {"r2r": {
            "path": str(seed),
            "sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
        }},
        "rounds": [{"jobs": [job]}],
    }

    path = background_pose_exclusions.materialize_for_job(
        manifest, {"jobs": {}}, job)

    assert json.loads(path.read_text()) == {
        "schema_version": "egoconseq.pose_exclusions.v1",
        "scenes": {"scene-a": [
            {"position": [4.0, 0.0, 2.0], "yaw_rad": 1.5},
        ]},
    }


def test_revisit_merges_seed_with_prior_authenticated_poses(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({
        "schema_version": "egoconseq.pose_exclusions.v1",
        "scenes": {"scene-a": [
            {"position": [4.0, 0.0, 2.0], "yaw_rad": 1.5},
        ]},
    }))
    first = _job(
        "first", catalog_pass=0, output_root=tmp_path,
        bind_exclusions=True)
    revisit = _job("revisit", catalog_pass=1, output_root=tmp_path)
    records = tmp_path / "first" / "records.jsonl"
    records.parent.mkdir()
    records.write_text(json.dumps({
        "scene_id": "scene-a",
        "pose": {"position": [1.0, 2.0, 3.0], "yaw_rad": 0.5},
    }) + "\n")
    manifest = {
        "seed_pose_exclusions": {"r2r": {
            "path": str(seed),
            "sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
        }},
        "rounds": [{"jobs": [first]}, {"jobs": [revisit]}],
    }
    state = {"jobs": {"first": {
        "durable_records": 1,
        "catalog_status": "completed",
        "source_validation": {"sources": [{
            "path": str(records),
            "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        }]},
    }}}

    path = background_pose_exclusions.materialize_for_job(
        manifest, state, revisit)

    assert json.loads(path.read_text())["scenes"]["scene-a"] == [
        {"position": [4.0, 0.0, 2.0], "yaw_rad": 1.5},
        {"position": [1.0, 2.0, 3.0], "yaw_rad": 0.5},
    ]


def test_continuous_revisit_reads_only_same_dataset_scene_from_state(tmp_path):
    revisit = _job(
        "gs-revisit", catalog_pass=1, output_root=tmp_path, dataset="gs")
    gs_records = tmp_path / "gs" / "records.jsonl"
    b1k_records = tmp_path / "b1k" / "records.jsonl"
    gs_records.parent.mkdir()
    b1k_records.parent.mkdir()
    gs_records.write_text(json.dumps({
        "scene_id": "scene-a",
        "pose": {"position": [1.0, 0.0, 2.0], "yaw_rad": 0.5},
    }) + "\n")
    b1k_records.write_text(json.dumps({
        "scene_id": "scene-a",
        "pose": {"position": [9.0, 0.0, 9.0], "yaw_rad": 1.5},
    }) + "\n")
    state = {"jobs": {
        "dynamic-gs-pass0": {
            "dataset": "gs", "scene_id": "scene-a", "catalog_pass": 0,
            "durable_records": 1, "catalog_status": "completed",
            "source_validation": {"sources": [{
                "path": str(gs_records),
                "records_sha256": hashlib.sha256(
                    gs_records.read_bytes()).hexdigest(),
            }]},
        },
        "dynamic-b1k-pass0": {
            "dataset": "b1k", "scene_id": "scene-a", "catalog_pass": 0,
            "durable_records": 1, "catalog_status": "completed",
            "source_validation": {"sources": [{
                "path": str(b1k_records),
                "records_sha256": hashlib.sha256(
                    b1k_records.read_bytes()).hexdigest(),
            }]},
        },
    }}

    path = background_pose_exclusions.materialize_for_job(
        {"rounds": []}, state, revisit)

    assert json.loads(path.read_text())["scenes"]["scene-a"] == [{
        "position": [1.0, 0.0, 2.0], "yaw_rad": 0.5,
    }]


def test_revisit_rejects_durable_nonreusable_prior_job(tmp_path):
    first = _job("first", catalog_pass=0, output_root=tmp_path)
    revisit = _job("revisit", catalog_pass=1, output_root=tmp_path)
    manifest = {"rounds": [{"jobs": [first]}, {"jobs": [revisit]}]}
    state = {"jobs": {"first": {
        "durable_records": 3,
        "catalog_status": "failed",
    }}}

    try:
        background_pose_exclusions.materialize_for_job(
            manifest, state, revisit)
    except ValueError as error:
        assert "durable records are not reusable" in str(error)
    else:
        raise AssertionError("durable non-reusable records were ignored")


def test_revisit_rejects_authenticated_source_without_scene_pose(tmp_path):
    first = _job("first", catalog_pass=0, output_root=tmp_path)
    revisit = _job("revisit", catalog_pass=1, output_root=tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "scene_id": "other",
        "pose": {"position": [1.0, 2.0, 3.0], "yaw_rad": 0.5},
    }) + "\n")
    manifest = {"rounds": [{"jobs": [first]}, {"jobs": [revisit]}]}
    state = {"jobs": {"first": {
        "durable_records": 1,
        "catalog_status": "completed",
        "source_validation": {"sources": [{
            "path": str(records),
            "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        }]},
    }}}

    try:
        background_pose_exclusions.materialize_for_job(
            manifest, state, revisit)
    except ValueError as error:
        assert "contains no pose for scene-a" in str(error)
    else:
        raise AssertionError("empty authenticated exclusion source was ignored")


def test_revisit_rejects_tampered_prior_record_source(tmp_path):
    first = _job("first", catalog_pass=0, output_root=tmp_path)
    revisit = _job("revisit", catalog_pass=1, output_root=tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text("{}\n")
    manifest = {"rounds": [{"jobs": [first]}, {"jobs": [revisit]}]}
    state = {"jobs": {"first": {
        "catalog_status": "completed",
        "source_validation": {"sources": [{
            "path": str(records), "records_sha256": "0" * 64,
        }]},
    }}}

    try:
        background_pose_exclusions.materialize_for_job(
            manifest, state, revisit)
    except ValueError as error:
        assert "identity differs" in str(error)
    else:
        raise AssertionError("tampered prior source was accepted")


def test_checkpoint_seed_stores_unique_poses_per_dataset_and_scene(tmp_path):
    records = tmp_path / "records.jsonl"
    rows = [
        {"scene_id": "scene-a", "source": {"source_dataset": "r2r"},
         "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0}},
        {"scene_id": "scene-a", "source": {"source_dataset": "r2r"},
         "pose": {"position": [0.2, 0.0, 0.0], "yaw_rad": 0.1}},
        {"scene_id": "scene-a", "source": {"source_dataset": "r2r"},
         "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0}},
        {"scene_id": "scene-a", "source": {"source_dataset": "gs"},
         "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0}},
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({
        "schema": "egoconseq.background-compile-checkpoint.v1",
        "sources": [{
            "path": str(records),
            "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        }],
    }))

    identities = background_pose_exclusions.build_seed_from_checkpoint(
        checkpoint,
        expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        output_dir=tmp_path / "seed")

    r2r = json.loads(Path(identities["r2r"]["path"]).read_text())
    assert r2r["scenes"]["scene-a"] == [
        {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        {"position": [0.2, 0.0, 0.0], "yaw_rad": 0.1},
    ]
    assert identities["r2r"]["representative_count"] == 2
    assert identities["r2r"]["record_count"] == 3
    gs = json.loads(Path(identities["gs"]["path"]).read_text())
    assert gs["scenes"]["scene-a"] == [
        {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
    ]
    assert identities["gs"]["representative_count"] == 1
    assert identities["gs"]["record_count"] == 1
    assert identities["b1k"]["representative_count"] == 0
    assert identities["b1k"]["record_count"] == 0
