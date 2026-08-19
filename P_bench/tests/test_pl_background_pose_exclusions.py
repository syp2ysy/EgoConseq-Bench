"""Catalog revisits exclude only authenticated poses from earlier passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline import background_pose_exclusions


def _job(job_id: str, *, catalog_pass: int, output_root: Path) -> dict:
    return {
        "job_id": job_id,
        "dataset": "r2r",
        "catalog_pass": catalog_pass,
        "scenes": ["scene-a"],
        "pose_exclusions_path": (
            None if catalog_pass == 0 else
            str(output_root / "controller" / "pose_exclusions" /
                "r2r" / "scene-a-pass01.json")),
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
