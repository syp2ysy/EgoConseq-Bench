"""External authority revalidation for the background controller."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from pipeline import gate_authority, io_utils


def resolve_artifact_source_authority(
        artifact: Path, sources: Sequence[Mapping]
        ) -> gate_authority.ResolvedPreviewSourceAuthority:
    """Reconstruct the independent source authority bound by an artifact."""
    source_map = gate_authority.load_authority_manifest(
        Path(artifact) / "private" / "source_map.json")
    binding = source_map.get("source_authority") or {}
    bound_sources = binding.get("sources")
    if not isinstance(bound_sources, list) or len(bound_sources) != len(sources):
        raise ValueError("candidate artifact source authority count differs")
    return gate_authority.resolve_preview_source_authority(
        records_paths=[str(row["path"]) for row in sources],
        expected_records_sha256=[str(row["records_sha256"])
                                 for row in sources],
        expected_run_meta_sha256=[row.get("run_meta_sha256")
                                  for row in sources],
        authority_kind=str(binding.get("authority_kind") or ""),
        authority_id=str(binding.get("authority_id") or ""),
        source_locators=[row.get("locator") for row in bound_sources])


def _terminal_scene_ids(audit: Mapping) -> tuple[list[str], list[str]]:
    if audit.get("schema") != "b1k-catalog-authority-audit.v1":
        raise ValueError("B1K catalog audit schema is invalid")
    installed = audit.get("installed_scene_ids")
    accepted_rows = audit.get("accepted")
    excluded_rows = audit.get("excluded")
    if (not isinstance(installed, list) or not isinstance(accepted_rows, list) or
            not isinstance(excluded_rows, list)):
        raise ValueError("B1K catalog audit terminal partition is invalid")
    accepted = [str((row or {}).get("scene_id") or "")
                for row in accepted_rows if isinstance(row, Mapping)]
    excluded = [str((row or {}).get("scene_id") or "")
                for row in excluded_rows if isinstance(row, Mapping)]
    if (len(accepted) != len(accepted_rows) or
            len(excluded) != len(excluded_rows) or not accepted or
            len(set(accepted)) != len(accepted) or
            len(set(excluded)) != len(excluded) or
            set(accepted) & set(excluded) or
            set(accepted) | set(excluded) != set(installed)):
        raise ValueError("B1K catalog audit terminal partition is invalid")
    return accepted, excluded


def validate_external_authorities(manifest: Mapping) -> None:
    """Reopen B1K authorities and bind their bytes and terminal scene sets."""
    authorities = manifest["authorities"]
    audit_path = Path(manifest["paths"]["b1k_catalog_audit"])
    source_path = Path(manifest["paths"]["b1k_source_manifest"])
    audit = gate_authority.load_authority_manifest(audit_path)
    source = gate_authority.load_authority_manifest(source_path)
    if io_utils.sha256_file(audit_path) != \
            authorities["b1k_catalog_audit"]["sha256"]:
        raise ValueError("B1K catalog audit file digest differs")
    if io_utils.sha256_file(source_path) != \
            authorities["b1k_source_manifest"]["sha256"]:
        raise ValueError("B1K source manifest file digest differs")
    accepted, excluded = _terminal_scene_ids(audit)
    audit_binding = authorities["b1k_catalog_audit"]
    if (set(audit["installed_scene_ids"]) !=
            set(audit_binding["installed_scene_ids"]) or
            set(accepted) != set(audit_binding["accepted_scene_ids"]) or
            set(excluded) != set(audit_binding["excluded_scene_ids"])):
        raise ValueError("B1K catalog audit authority differs")
    source_rows = source.get("scenes")
    if not isinstance(source_rows, list):
        raise ValueError("B1K source manifest scenes are invalid")
    source_ids = [str((row or {}).get("scene_id") or "")
                  for row in source_rows if isinstance(row, Mapping)]
    if (len(source_ids) != len(source_rows) or not source_ids or
            len(source_ids) != len(set(source_ids))):
        raise ValueError("B1K source manifest scenes are invalid")
    if set(accepted) != set(source_ids):
        raise ValueError("B1K accepted and source scene sets differ")
    if set(source_ids) != set(
            authorities["b1k_source_manifest"]["accepted_scene_ids"]):
        raise ValueError("B1K source manifest authority differs")
