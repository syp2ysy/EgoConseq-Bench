"""Static authority-surface capability diagnostics without simulators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pipeline import (
    benchmark_tasks, capability_contracts, capability_matrix, record,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_CAPABILITY_MATRIX = ROOT / "scripts" / "report_capability_matrix.py"


def _legacy_gs_capability_snapshot() -> dict:
    body = {
        "schema": "egoconseq.authority-capability-snapshot.v1",
        "scope": "static_authority_surface_capability",
        "source_dataset": "gs",
        "main_collection_enabled": True,
        "method_surface": {
            "assign": True,
            "instance_points": True,
            "target_geometry_atom": True,
            "confirm_contact_instances": True,
            "instance_triangles": False,
        },
        "task_statuses": {
            task: {"status": "available"}
            for task in ("A1", "A2", "A3", "B1", "B2", "C1")
        },
        "certification": {
            "runtime_evidence": "not_assessed",
            "source_binding": "not_assessed",
            "formal_collection": "not_assessed",
        },
    }
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return {**body, "sha256": hashlib.sha256(encoded).hexdigest()}


def test_legacy_capability_snapshot_is_validated_by_its_own_schema():
    """Catches recomputing a frozen v1 record with the current v2 surface."""
    snapshot = _legacy_gs_capability_snapshot()

    validated = capability_contracts.validate_snapshot(snapshot, "gs")

    assert validated == snapshot


def test_legacy_capability_snapshot_still_fails_closed_when_tampered():
    snapshot = _legacy_gs_capability_snapshot()
    snapshot["method_surface"]["instance_triangles"] = True

    with pytest.raises(ValueError, match="snapshot"):
        capability_contracts.validate_snapshot(snapshot, "gs")


def test_gs_a2_eligibility_accepts_an_authenticated_legacy_snapshot(
        monkeypatch):
    """Catches the action refresh dying before it can inspect A2 evidence."""
    rec = {
        "schema_version": record.V18_SCHEMA_VERSION,
        "source": {"source_dataset": "gs"},
        "authority_surface_capability": _legacy_gs_capability_snapshot(),
        "gs_scene_capability": {},
    }
    monkeypatch.setattr(
        benchmark_tasks.gs_semantic, "scene_task_available",
        lambda _atom, _source, task: task == "A2")

    assert benchmark_tasks._capability_envelope_rejection(
        "A2_collision_step_grounding", rec) is None


def test_new_capability_snapshot_uses_a_new_schema():
    """Catches mutating the meaning of v1 in place again."""
    fields = capability_contracts.snapshot_fields("gs")

    assert fields["schema"] == "egoconseq.authority-capability-snapshot.v2"
    assert "target_geometry_atom" not in fields["method_surface"]


def test_frozen_authority_method_surface_and_task_statuses():
    """Catches a changed adapter surface being reported as collection-ready."""
    report = capability_matrix.build_capability_report()
    rows = {row["source_dataset"]: row for row in report["datasets"]}

    assert [row["source_dataset"] for row in report["datasets"]] == [
        "r2r", "b1k", "gs",
    ]
    assert rows["r2r"]["method_surface"] == {
        "assign": True,
        "instance_points": True,
        "confirm_contact_instances": True,
        "instance_triangles": True,
    }
    assert rows["b1k"]["method_surface"] == {
        "assign": True,
        "instance_points": True,
        "confirm_contact_instances": True,
        "instance_triangles": True,
    }
    assert rows["gs"]["method_surface"] == {
        "assign": True,
        "instance_points": True,
        "confirm_contact_instances": True,
        "instance_triangles": False,
    }

    assert rows["r2r"]["task_statuses"] == {
        "A1": "available",
        "A2": "available",
        "A3": "available",
        "B1": "available",
        "B2": "available",
        "C1": "available",
    }
    assert rows["b1k"]["task_statuses"]["B1"] == "available"
    assert rows["b1k"]["task_statuses"]["B2"] == "available"
    assert rows["gs"]["task_statuses"] == {
        task: "available" for task in ("A1", "A2", "A3", "B1", "B2", "C1")
    }


def test_scene_rows_are_explicit_sorted_and_keep_certification_separate():
    """Catches inventing a per-scene collection or evidence certification."""
    report = capability_matrix.build_capability_report(
        scene_identities=[("gs", "scene-b"), ("r2r", "scene-z"),
                          ("gs", "scene-a")])

    assert [(row["source_dataset"], row["scene_id"])
            for row in report["scenes"]] == [
                ("r2r", "scene-z"), ("gs", "scene-a"), ("gs", "scene-b"),
            ]
    assert report["scope"] == "static_authority_surface_capability"
    assert report["certification"] == {
        "runtime_evidence": "not_assessed",
        "source_binding": "not_assessed",
        "formal_collection": "not_assessed",
    }
    gs = report["scenes"][1]
    assert gs["main_collection_enabled"] is True
    assert gs["task_statuses"]["A3"] == "available"
    assert gs["task_statuses"]["B1"] == "available"


def test_discovered_scene_specs_expand_without_constructing_a_runtime():
    """Catches a scene report requiring simulator-backed authority objects."""
    identities = capability_matrix.scene_identities_from_specs([
        SimpleNamespace(source_dataset="gs", scene_id="scene-b"),
        SimpleNamespace(source_dataset="r2r", scene_id="scene-a"),
    ])

    assert identities == [("r2r", "scene-a"), ("gs", "scene-b")]


def test_cli_reports_explicit_scene_identities_as_static_capabilities():
    """Catches the report CLI depending on a simulator for explicit rows."""
    result = subprocess.run(
        [sys.executable, str(REPORT_CAPABILITY_MATRIX),
         "--scene", "gs:scene-b", "--scene", "r2r:scene-a"],
        cwd=ROOT, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert '"scope": "static_authority_surface_capability"' in result.stdout
    assert result.stdout.index('"scene_id": "scene-a"') < \
        result.stdout.index('"scene_id": "scene-b"')
