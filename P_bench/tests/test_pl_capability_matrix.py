"""Static authority-surface capability diagnostics without simulators."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from pipeline import capability_matrix


ROOT = Path(__file__).resolve().parents[1]
REPORT_CAPABILITY_MATRIX = ROOT / "scripts" / "report_capability_matrix.py"


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
        "target_geometry_atom": True,
        "confirm_contact_instances": True,
        "instance_triangles": True,
    }
    assert rows["b1k"]["method_surface"] == {
        "assign": True,
        "instance_points": True,
        "target_geometry_atom": True,
        "confirm_contact_instances": True,
        "instance_triangles": True,
    }
    assert rows["gs"]["method_surface"] == {
        "assign": True,
        "instance_points": True,
        "target_geometry_atom": True,
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
