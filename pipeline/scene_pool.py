"""Pure scene-catalog discovery with official-split provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import List, Optional, Sequence

from pipeline import b1k_semantic, dataset_contracts
from pipeline.io_utils import sha256_file


SCENE_MANIFEST_VERSION = "egoconseq.scene_manifest.v1"
B1K_SOURCE_MANIFEST_VERSION = "egoconseq.b1k-source-manifest.v1"
SOURCE_ASSET_IDENTITY_VERSION = dataset_contracts.SOURCE_ASSET_IDENTITY_VERSION
_SOURCE_ASSET_ROLE_ORDER = {
    role: index for index, role in enumerate((
        "scene",
        "navmesh",
        "semantic",
        "collision_authority",
        "semantic_metadata",
        "scene_dataset_config",
        "scene_authority",
    ))
}


class SceneCatalogError(ValueError):
    """Raised when a trusted scene catalog cannot be verified."""


@dataclass(frozen=True)
class SourceAssetIdentity:
    """Content identity for one simulator input, independent of its location."""

    role: str
    byte_size: int
    sha256: str

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "bytes": int(self.byte_size),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SceneSpec:
    """One simulator scene with enough provenance to audit its source split."""
    scene_id: str
    source_dataset: str
    official_split: str
    scene_path: str
    navmesh_path: str
    semantic_path: str
    semantic_metadata_path: Optional[str]
    semantic_format: str
    scene_dataset_config: Optional[str]
    provenance_sha256: str
    source_assets: Optional[tuple[SourceAssetIdentity, ...]] = None
    source_assets_sha256: Optional[str] = None
    split_authority: Optional[str] = None
    b1k_scene_authority: Optional[dict] = None
    collision_authority_path: Optional[str] = None

    def _ensure_source_asset_identity(self) -> None:
        if self.source_assets is not None and \
                self.source_assets_sha256 is not None:
            return
        if self.source_assets is not None or \
                self.source_assets_sha256 is not None:
            raise SceneCatalogError(
                "scene source asset identity is only partially initialized")
        assets, digest = _source_asset_identity(
            scene=self.scene_path,
            navmesh=self.navmesh_path,
            semantic=self.semantic_path,
            collision_authority=self.collision_authority_path,
            semantic_metadata=self.semantic_metadata_path,
            scene_dataset_config=self.scene_dataset_config,
        )
        object.__setattr__(self, "source_assets", assets)
        object.__setattr__(self, "source_assets_sha256", digest)

    def provenance(self) -> dict:
        self._ensure_source_asset_identity()
        value = {
            "scene_id": self.scene_id,
            "source_dataset": self.source_dataset,
            "official_split": self.official_split,
            "semantic_format": self.semantic_format,
            "source_manifest_sha256": self.provenance_sha256,
            "source_asset_identity_version": SOURCE_ASSET_IDENTITY_VERSION,
            "source_assets": [
                asset.to_dict() for asset in self.source_assets],
            "source_assets_sha256": self.source_assets_sha256,
        }
        if self.split_authority is not None:
            value["split_authority"] = self.split_authority
        return value


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_asset_identity(**role_paths) -> tuple[
        tuple[SourceAssetIdentity, ...], str]:
    assets = []
    for role, path in role_paths.items():
        if path is None:
            continue
        resolved = Path(path).resolve(strict=True)
        assets.append(SourceAssetIdentity(
            role=str(role),
            byte_size=int(resolved.stat().st_size),
            sha256=sha256_file(resolved),
        ))
    assets.sort(key=lambda value: _SOURCE_ASSET_ROLE_ORDER[value.role])
    serialized = [asset.to_dict() for asset in assets]
    return tuple(assets), _canonical_sha256({
        "version": SOURCE_ASSET_IDENTITY_VERSION,
        "assets": serialized,
    })


def _authoritative_source_file_identity(
        role: str, path: os.PathLike) -> SourceAssetIdentity:
    """Hash one exact regular-file path once, refusing its final symlink."""
    descriptor = None
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise SceneCatalogError(
                f"scene source {role!r} is not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return SourceAssetIdentity(
            role=role, byte_size=int(source_stat.st_size),
            sha256=digest.hexdigest())
    except OSError as error:
        raise SceneCatalogError(
            f"scene source {role!r} cannot be opened without following a "
            f"symlink: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def verify_scene_source_asset_paths(
        scene: SceneSpec,
        role_paths: Sequence[tuple[str, os.PathLike]]) -> dict:
    """Match cached identities to the exact paths a simulator will consume.

    This is the single simulator-session trust-boundary assertion.  The local,
    offline threat model does not require repeated stat/inode checks after it.
    """
    if not isinstance(scene, SceneSpec):
        raise TypeError("scene source verification requires a SceneSpec")
    roles = tuple(str(role) for role, _path in role_paths)
    observed = tuple(
        _authoritative_source_file_identity(str(role), path)
        for role, path in role_paths)
    observed_sha256 = _canonical_sha256({
        "version": SOURCE_ASSET_IDENTITY_VERSION,
        "assets": [asset.to_dict() for asset in observed],
    })
    if (scene.source_assets is None and
            scene.source_assets_sha256 is None):
        object.__setattr__(scene, "source_assets", observed)
        object.__setattr__(scene, "source_assets_sha256", observed_sha256)
    elif (scene.source_assets is None or
          scene.source_assets_sha256 is None):
        raise SceneCatalogError(
            "scene source asset identity is only partially initialized")
    expected = tuple(scene.source_assets or ())
    if tuple(asset.role for asset in expected) != roles:
        raise SceneCatalogError(
            "scene source identity roles do not match consumed paths")
    if observed != expected:
        raise SceneCatalogError(
            "scene source path identity or digest does not match provenance")
    source = scene.provenance()
    if source.get("source_assets_sha256") != observed_sha256:
        raise SceneCatalogError(
            "scene source aggregate identity does not match provenance")
    return source


def validate_source_asset_provenance(source: dict) -> str:
    """Validate and return the content digest of persisted simulator inputs."""
    try:
        return dataset_contracts.validate_source_asset_provenance(source)
    except ValueError as error:
        raise SceneCatalogError(str(error)) from error


def _require_files(scene_id: str, paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SceneCatalogError(
            f"scene {scene_id} is missing required semantic/geometry assets: "
            + ", ".join(missing))


def _b1k_input_identity(root: Path, raw, *, label: str) -> dict:
    if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
        raise SceneCatalogError(f"B1K {label} identity is invalid")
    relative = Path(str(raw["path"]))
    if relative.is_absolute():
        raise SceneCatalogError(f"B1K {label} path must be relative")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise SceneCatalogError(
            f"B1K {label} path escapes data root") from error
    if not resolved.is_file():
        raise SceneCatalogError(f"B1K {label} is not a file: {resolved}")
    actual_size = int(resolved.stat().st_size)
    actual_sha256 = sha256_file(resolved)
    if (isinstance(raw["bytes"], bool) or
            raw["bytes"] != actual_size or
            raw["sha256"] != actual_sha256):
        raise SceneCatalogError(f"B1K {label} digest changed: {resolved}")
    return {
        "path": relative.as_posix(),
        "bytes": actual_size,
        "sha256": actual_sha256,
    }


def _read_pinned_b1k_manifest(path: os.PathLike) -> tuple[dict, str]:
    """Read, hash, and parse one B1K manifest from a single descriptor."""
    resolved = Path(path).resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SceneCatalogError(
                f"B1K source manifest is not a regular file: {resolved}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SceneCatalogError(
            f"cannot decode B1K source manifest {resolved}: {error}") \
            from error
    if not isinstance(payload, dict):
        raise SceneCatalogError("B1K source manifest must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def discover_b1k_train_scenes(
        root: os.PathLike, manifest: os.PathLike, *, requested=None) -> List[SceneSpec]:
    """Resolve the project-defined B1K train catalog and authenticate inputs.

    Encrypted USD content is hashed as an opaque source file here.  Runtime
    collision triangles are extracted only after OmniGibson loads the scene;
    their independently derived authority atom is checked by ``B1KSimSession``.
    """
    root_path = Path(root).resolve(strict=True)
    manifest_path = Path(manifest).resolve(strict=True)
    try:
        payload, manifest_sha256 = _read_pinned_b1k_manifest(manifest_path)
    except OSError as error:
        raise SceneCatalogError(
            f"cannot decode B1K source manifest {manifest_path}: {error}") \
            from error
    expected_header = {
        "schema_version": B1K_SOURCE_MANIFEST_VERSION,
        "dataset": "b1k",
        "official_split": "train",
        "split_authority": "project_defined",
    }
    if any(payload.get(key) != value
           for key, value in expected_header.items()):
        raise SceneCatalogError("B1K source manifest header is invalid")
    simulator = payload.get("simulator")
    asset_versions = payload.get("asset_versions")
    if (not isinstance(simulator, dict) or not simulator or
            not isinstance(asset_versions, dict) or not asset_versions):
        raise SceneCatalogError(
            "B1K source manifest lacks simulator or asset versions")
    entries = payload.get("scenes")
    if not isinstance(entries, list):
        raise SceneCatalogError("B1K source manifest has no scenes list")
    if len(entries) < 3:
        raise SceneCatalogError(
            "B1K installation must contain at least 3 declared scenes")
    seen = set()
    specs = []
    requested = None if requested is None else set(map(str, requested))
    for entry in entries:
        if not isinstance(entry, dict):
            raise SceneCatalogError("B1K scene entry must be an object")
        scene_id = str(entry.get("scene_id") or "").strip()
        if not scene_id or scene_id in seen:
            raise SceneCatalogError(
                f"B1K scene id is missing or duplicated: {scene_id!r}")
        seen.add(scene_id)
        if requested is not None and scene_id not in requested:
            continue
        scene_json = _b1k_input_identity(
            root_path, entry.get("scene_json"),
            label=f"scene {scene_id} JSON")
        initial_state = _b1k_input_identity(
            root_path, entry.get("initial_state"),
            label=f"scene {scene_id} initial state")
        layouts_raw = entry.get("layouts")
        encrypted_raw = entry.get("encrypted_assets")
        if (not isinstance(layouts_raw, list) or not layouts_raw or
                not isinstance(encrypted_raw, list) or not encrypted_raw):
            raise SceneCatalogError(
                f"B1K scene {scene_id} lacks layouts or encrypted assets")
        layouts = [
            _b1k_input_identity(
                root_path, value, label=f"scene {scene_id} layout")
            for value in layouts_raw
        ]
        encrypted = [
            _b1k_input_identity(
                root_path, value, label=f"scene {scene_id} encrypted asset")
            for value in encrypted_raw
        ]
        authority = entry.get("scene_authority")
        if (not isinstance(authority, dict) or
                authority.get("schema") !=
                "b1k-derived-scene-authority.v1"):
            raise SceneCatalogError(
                f"B1K scene {scene_id} authority atom is invalid")
        unhashed_authority = {
            key: value for key, value in authority.items() if key != "sha256"}
        if authority.get("sha256") != _canonical_sha256(unhashed_authority):
            raise SceneCatalogError(
                f"B1K scene {scene_id} authority digest is invalid")
        try:
            b1k_semantic.canonical_replay_binding(authority)
        except ValueError as error:
            raise SceneCatalogError(
                f"B1K scene {scene_id} {error}") from error
        scene_value = {
            "schema": "b1k-scene-source-inputs.v1",
            "scene_id": scene_id,
            "simulator": simulator,
            "asset_versions": asset_versions,
            "scene_json": scene_json,
            "layouts": layouts,
            "encrypted_assets": encrypted,
            "initial_state": initial_state,
        }
        scene_payload = json.dumps(
            scene_value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("ascii")
        authority_payload = json.dumps(
            authority, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("ascii")
        source_assets = (
            SourceAssetIdentity(
                role="scene", byte_size=len(scene_payload),
                sha256=hashlib.sha256(scene_payload).hexdigest()),
            SourceAssetIdentity(
                role="scene_authority", byte_size=len(authority_payload),
                sha256=str(authority["sha256"])),
        )
        source_assets_sha256 = _canonical_sha256({
            "version": SOURCE_ASSET_IDENTITY_VERSION,
            "assets": [value.to_dict() for value in source_assets],
        })
        scene_path = str((root_path / scene_json["path"]).resolve())
        specs.append(SceneSpec(
            scene_id=scene_id,
            source_dataset="b1k",
            official_split="train",
            scene_path=scene_path,
            navmesh_path=scene_path,
            semantic_path=scene_path,
            semantic_metadata_path=None,
            semantic_format=dataset_contracts.dataset_source_contract(
                "b1k").semantic_format,
            scene_dataset_config=None,
            provenance_sha256=manifest_sha256,
            source_assets=source_assets,
            source_assets_sha256=source_assets_sha256,
            split_authority="project_defined",
            b1k_scene_authority=dict(authority),
        ))
    return sorted(specs, key=lambda value: value.scene_id)


def discover_gs_scenes(
        root: os.PathLike, manifest: os.PathLike, *,
        requested: Optional[Sequence[os.PathLike]] = None,
        source_split: str = "train",
        ) -> List[SceneSpec]:
    """Resolve explicitly declared GS split scenes and their local assets.

    ``requested`` narrows asset validation without weakening the manifest
    boundary: every requested key must still name a unique split entry.  This
    lets one-scene smoke runs proceed while collision artifacts for the rest of
    the authenticated catalog are still being preprocessed.  Full-catalog
    discovery continues to fail closed if any selected asset is absent.
    """
    source_split = str(source_split)
    if source_split not in dataset_contracts.OFFICIAL_SOURCE_SPLITS:
        raise SceneCatalogError(
            f"unsupported GS source split: {source_split!r}")
    root = Path(root).resolve()
    manifest_path = Path(manifest).resolve()
    if not manifest_path.is_file():
        raise SceneCatalogError(
            f"GS {source_split} manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != SCENE_MANIFEST_VERSION:
        raise SceneCatalogError(
            "GS manifest has unsupported schema_version")
    if payload.get("dataset") != "gs":
        raise SceneCatalogError("GS manifest dataset must be 'gs'")
    entries = payload.get("scenes")
    if not isinstance(entries, list):
        raise SceneCatalogError("GS manifest has no scenes list")
    allowed_splits = set(dataset_contracts.OFFICIAL_SOURCE_SPLITS)
    requested_keys = None if requested is None else {
        str(value).strip() for value in requested}
    if requested_keys is not None and (
            not requested_keys or "" in requested_keys):
        raise SceneCatalogError("requested GS scenes are empty")
    matched_requested = set()
    seen = set()
    specs = []
    manifest_hash = sha256_file(manifest_path)
    for entry in entries:
        split = str(entry.get("split", ""))
        if split not in allowed_splits:
            raise SceneCatalogError(
                f"GS scene {entry.get('scene_id')!r} has unknown split {split!r}")
        scene_id = str(entry.get("scene_id", "")).strip()
        if not scene_id:
            raise SceneCatalogError("GS scene manifest entry has no scene_id")
        if scene_id in seen:
            raise SceneCatalogError(f"duplicate GS scene_id: {scene_id}")
        seen.add(scene_id)
        if split != source_split:
            continue
        raw_path = Path(str(entry.get("path", scene_id)))
        if raw_path.is_absolute():
            directory = raw_path.resolve()
        else:
            directory = (root / raw_path).resolve()
            try:
                directory.relative_to(root)
            except ValueError as error:
                raise SceneCatalogError(
                    f"GS scene path escapes dataset root: {raw_path}") from error
        entry_keys = {
            scene_id,
            str(raw_path),
            str(directory),
            str(directory / "scene.gs.ply"),
        }
        if requested_keys is not None:
            matched = entry_keys & requested_keys
            if not matched:
                continue
            matched_requested.update(matched)
        scene_path = directory / "scene.gs.ply"
        navmesh = directory / "scene.navmesh"
        labels = directory / "labels.json"
        collision_authority = directory / "scene.collision.npz"
        _require_files(
            scene_id, (scene_path, navmesh, labels, collision_authority))
        # The downloaded GS layout may use manifest-controlled links into its
        # immutable official asset store.  Resolve those links once during
        # discovery; the simulator trust boundary subsequently opens these
        # exact regular-file targets with O_NOFOLLOW and verifies their hashes.
        scene_path = scene_path.resolve(strict=True)
        navmesh = navmesh.resolve(strict=True)
        labels = labels.resolve(strict=True)
        collision_authority = collision_authority.resolve(strict=True)
        specs.append(SceneSpec(
            scene_id=scene_id,
            source_dataset="gs",
            official_split=source_split,
            scene_path=str(scene_path),
            navmesh_path=str(navmesh),
            semantic_path=str(labels),
            semantic_metadata_path=None,
            semantic_format=dataset_contracts.dataset_source_contract(
                "gs").semantic_format,
            scene_dataset_config=None,
            provenance_sha256=manifest_hash,
            collision_authority_path=str(collision_authority),
        ))
    if requested_keys is not None:
        missing = sorted(requested_keys - matched_requested)
        if missing:
            raise SceneCatalogError(
                f"requested GS scene is not in the verified {source_split} "
                "catalog: "
                + ", ".join(missing))
    if not specs:
        raise SceneCatalogError(
            f"GS manifest contains no complete {source_split} scenes: "
            f"{manifest_path}")
    return sorted(specs, key=lambda value: value.scene_id)


