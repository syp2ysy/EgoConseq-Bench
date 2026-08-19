"""Resolve candidate-preview sources against an independent digest authority."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat

from pipeline import io_utils


PREVIEW_SOURCE_AUTHORITY_SCHEMA = "egoconseq.preview-source-authority.v3"
CLI_EXTERNAL_AUTHORITY_KIND = "cli_external"
COMMITTED_MANIFEST_AUTHORITY_KIND = "committed_manifest"
_AUTHORITY_KINDS = frozenset({
    CLI_EXTERNAL_AUTHORITY_KIND,
    COMMITTED_MANIFEST_AUTHORITY_KIND,
})


@dataclass(frozen=True)
class ResolvedPreviewSourceAuthority:
    """Independent authority for every source a preview validator may open."""

    sources: tuple[dict, ...]
    authority_kind: str
    authority_id: str
    source_locators: tuple[dict, ...]

    def binding(self) -> dict:
        """Return the exact binding persisted in a candidate source map."""
        return {
            "schema": PREVIEW_SOURCE_AUTHORITY_SCHEMA,
            "authority_kind": self.authority_kind,
            "authority_id": self.authority_id,
            "sources": [{
                "locator": dict(locator),
                "records_sha256": source["records_sha256"],
                "run_meta_sha256": source["run_meta_sha256"],
            } for source, locator in zip(self.sources, self.source_locators)],
        }


def lexical_absolute_path(path, *, root=None) -> Path:
    """Make *path* absolute without dereferencing any symlink component."""
    value = os.fspath(path)
    if root is not None and not os.path.isabs(value):
        value = os.path.join(os.fspath(root), value)
    return Path(os.path.abspath(value))


def _sha256(value, label: str) -> str:
    digest = str(value or "")
    if (len(digest) != 64 or
            any(character not in "0123456789abcdef"
                for character in digest)):
        raise ValueError(f"{label} digest is invalid")
    return digest


def _authority_identity(kind: str, identity: str) -> tuple[str, str]:
    if kind not in _AUTHORITY_KINDS:
        raise ValueError("preview source authority kind is invalid")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("preview source authority id is invalid")
    return kind, identity


def _manifest_relative_path(value, *, root, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} relative path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a contained relative path")
    return lexical_absolute_path(relative, root=root)


def _explicit_trusted_root(root) -> Path:
    if root is None:
        raise ValueError("committed manifest requires an explicit trusted root")
    return lexical_absolute_path(root)


def load_authority_manifest(manifest_path) -> dict:
    """Read one authoritative JSON manifest, refusing a final symlink."""
    path = lexical_absolute_path(manifest_path)
    descriptor = None
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("authority manifest is not a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read()
    except OSError as error:
        raise ValueError("authority manifest cannot be opened") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("authority manifest is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("authority manifest must contain a JSON object")
    return value


def resolve_preview_source_authority(
        *, records_paths, expected_records_sha256,
        expected_run_meta_sha256, authority_kind: str, authority_id: str,
        root=None, source_locators=None) -> ResolvedPreviewSourceAuthority:
    """Authenticate externally named record bytes and freeze their binding."""
    authority_kind, authority_id = _authority_identity(
        authority_kind, authority_id)
    paths = list(records_paths)
    record_digests = list(expected_records_sha256)
    run_meta_digests = list(expected_run_meta_sha256)
    if not paths or not (
            len(paths) == len(record_digests) == len(run_meta_digests)):
        raise ValueError("preview source authority counts differ")
    if source_locators is None:
        locators = tuple({
            "kind": "authority_id", "id": f"{authority_id}:source:{index}",
        } for index in range(len(paths)))
    else:
        locators = tuple(dict(value) for value in source_locators)
        if len(locators) != len(paths):
            raise ValueError("preview source locator count differs")
    sources = []
    for index, (path, record_digest, run_meta_digest) in enumerate(zip(
            paths, record_digests, run_meta_digests)):
        normalized = lexical_absolute_path(path, root=root)
        expected_record = _sha256(
            record_digest, f"preview records {index}")
        if io_utils.sha256_file(normalized) != expected_record:
            raise ValueError("preview source records digest changed")
        expected_run_meta = (
            None if run_meta_digest is None else
            _sha256(run_meta_digest, f"preview run metadata {index}"))
        sources.append({
            "path": str(normalized),
            "records_sha256": expected_record,
            "run_meta_sha256": expected_run_meta,
        })
    return ResolvedPreviewSourceAuthority(
        sources=tuple(sources),
        authority_kind=authority_kind,
        authority_id=authority_id,
        source_locators=locators,
    )


def resolve_preview_source_authority_from_inputs(
        *, records_paths, expected_run_meta_sha256,
        authority_id: str = "cli-external:preview-sources",
        root=None) -> ResolvedPreviewSourceAuthority:
    """Capture CLI-named record bytes before any artifact is written."""
    paths = [lexical_absolute_path(path, root=root) for path in records_paths]
    return resolve_preview_source_authority(
        records_paths=paths,
        expected_records_sha256=[io_utils.sha256_file(path) for path in paths],
        expected_run_meta_sha256=expected_run_meta_sha256,
        authority_kind=CLI_EXTERNAL_AUTHORITY_KIND,
        authority_id=authority_id)


def resolve_build_preview_source_authority(
        *, records_paths, expected_run_meta_sha256,
        source_authority_manifest=None, root=None
        ) -> ResolvedPreviewSourceAuthority:
    """Resolve and cross-check manual or committed preview build inputs."""
    if source_authority_manifest is None:
        return resolve_preview_source_authority_from_inputs(
            records_paths=records_paths,
            expected_run_meta_sha256=expected_run_meta_sha256,
            root=root)
    committed = resolve_preview_source_authority_from_manifest(
        source_authority_manifest, root=root)
    direct_sources = [
        str(lexical_absolute_path(path, root=root))
        for path in records_paths]
    if (direct_sources != [source["path"] for source in committed.sources] or
            list(expected_run_meta_sha256) != [
                source["run_meta_sha256"] for source in committed.sources]):
        raise ValueError("build arguments conflict with committed manifest")
    return committed


def resolve_preview_source_authority_from_manifest(
        manifest_path, *, root=None) -> ResolvedPreviewSourceAuthority:
    """Resolve records and run metadata from one committed manifest."""
    trusted_root = _explicit_trusted_root(root)
    value = load_authority_manifest(manifest_path)
    schema = value.get("schema")
    name = value.get("name")
    if not isinstance(schema, str) or not schema:
        raise ValueError("source authority manifest schema is invalid")
    if not isinstance(name, str) or not name:
        raise ValueError("source authority manifest name is invalid")
    inputs = value.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("source authority manifest inputs are invalid")
    paths = []
    source_locators = []
    records_digests = []
    run_meta_digests = []
    for entry in inputs:
        if not isinstance(entry, dict):
            raise ValueError("source authority manifest input is invalid")
        shard = entry.get("shard")
        shard_path = _manifest_relative_path(
            shard, root=trusted_root, label="input shard")
        paths.append(shard_path / "records.jsonl")
        source_locators.append({
            "kind": "authority_root_relative",
            "path": (Path(str(shard)) / "records.jsonl").as_posix(),
        })
        records_digests.append(entry.get("records.jsonl"))
        run_meta_digest = _sha256(
            entry.get("run_meta.json"), "preview run metadata")
        run_meta_path = shard_path / "run_meta.json"
        if io_utils.sha256_file(run_meta_path) != run_meta_digest:
            raise ValueError("preview run metadata digest changed")
        run_meta_digests.append(run_meta_digest)
    return resolve_preview_source_authority(
        records_paths=paths,
        expected_records_sha256=records_digests,
        expected_run_meta_sha256=run_meta_digests,
        authority_kind=COMMITTED_MANIFEST_AUTHORITY_KIND,
        authority_id=f"committed-manifest:{schema}:{name}",
        source_locators=source_locators)
