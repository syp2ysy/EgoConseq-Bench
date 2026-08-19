"""Pure scene-catalog discovery with official-split provenance."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import glob
import hashlib
import gzip
import json
import os
from pathlib import Path
import stat
from typing import List, Optional, Sequence
import zlib

from pipeline import b1k_semantic, config, dataset_contracts
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
_TRUSTED_R2R_SCENE_CACHE = OrderedDict()


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
    provenance_path: str
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


def _read_pinned_r2r_manifest(path: os.PathLike) -> tuple[dict, str]:
    """Read, hash, and parse one snapshot of an R2R manifest."""
    resolved = Path(path).resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SceneCatalogError(
                f"R2R manifest is not a regular file: {resolved}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        serialized = gzip.decompress(raw) if resolved.suffix == ".gz" else raw
        payload = json.loads(serialized)
    except (EOFError, OSError, UnicodeDecodeError, zlib.error,
            json.JSONDecodeError) as error:
        raise SceneCatalogError(
            f"cannot decode R2R manifest {resolved}: {error}") from error
    return payload, digest


def _r2r_manifest_scene_ids(payload: dict) -> List[str]:
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list):
        raise SceneCatalogError("R2R train manifest has no episodes list")
    scene_ids = sorted({
        Path(str(episode.get("scene_id", ""))).stem
        for episode in episodes
        if isinstance(episode, dict) and episode.get("scene_id")
    })
    if not scene_ids:
        raise SceneCatalogError("R2R train manifest contains no scene ids")
    return scene_ids


def discover_r2r_train_scenes(
        episodes_path: os.PathLike, mp3d_root: os.PathLike) -> List[SceneSpec]:
    """Use R2R train only as an MP3D scene whitelist; ignore trajectories."""
    episodes_path = Path(episodes_path).resolve()
    mp3d_root = Path(mp3d_root).resolve()
    try:
        payload, manifest_hash = _read_pinned_r2r_manifest(episodes_path)
    except OSError as error:
        raise SceneCatalogError(
            f"R2R train episode manifest is unavailable: {episodes_path}") \
            from error
    scene_ids = _r2r_manifest_scene_ids(payload)
    scene_config = (
        mp3d_root / "mp3d_annotated_basis.scene_dataset_config.json")
    if not scene_config.is_file():
        raise SceneCatalogError(
            f"MP3D scene dataset config does not exist: {scene_config}")
    specs = []
    missing = []
    for scene_id in scene_ids:
        directory = mp3d_root / scene_id
        scene_path = directory / f"{scene_id}.glb"
        navmesh = directory / f"{scene_id}.navmesh"
        semantic_mesh = directory / f"{scene_id}_semantic.ply"
        semantic_metadata = directory / f"{scene_id}.house"
        required = (scene_path, navmesh, semantic_mesh, semantic_metadata)
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing.append(f"{scene_id}: " + ", ".join(absent))
            continue
        specs.append(SceneSpec(
            scene_id=scene_id,
            source_dataset="r2r",
            official_split="train",
            scene_path=str(scene_path),
            navmesh_path=str(navmesh),
            semantic_path=str(semantic_mesh),
            semantic_metadata_path=str(semantic_metadata),
            semantic_format=dataset_contracts.dataset_source_contract(
                "r2r").semantic_format,
            scene_dataset_config=str(scene_config),
            provenance_path=str(episodes_path),
            provenance_sha256=manifest_hash,
        ))
    if missing:
        preview = "; ".join(missing[:20])
        suffix = (
            f"; ... and {len(missing) - 20} more" if len(missing) > 20 else "")
        raise SceneCatalogError(
            f"R2R train references unavailable MP3D assets: {preview}{suffix}")
    return specs


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
        root: os.PathLike, manifest: os.PathLike) -> List[SceneSpec]:
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
    for entry in entries:
        if not isinstance(entry, dict):
            raise SceneCatalogError("B1K scene entry must be an object")
        scene_id = str(entry.get("scene_id") or "").strip()
        if not scene_id or scene_id in seen:
            raise SceneCatalogError(
                f"B1K scene id is missing or duplicated: {scene_id!r}")
        seen.add(scene_id)
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
            provenance_path=str(manifest_path),
            provenance_sha256=manifest_sha256,
            source_assets=source_assets,
            source_assets_sha256=source_assets_sha256,
            split_authority="project_defined",
            b1k_scene_authority=dict(authority),
        ))
    return sorted(specs, key=lambda value: value.scene_id)


def resolve_trusted_r2r_scene(
        scene_id: str, *,
        episodes_path: os.PathLike = config.R2R_TRAIN_EPISODES,
        mp3d_root: os.PathLike = config.MP3D_ROOT) -> SceneSpec:
    """Resolve one scene only through the configured local R2R authority."""
    requested = str(scene_id)
    matches = [
        spec for spec in discover_r2r_train_scenes(episodes_path, mp3d_root)
        if spec.scene_id == requested
    ]
    if len(matches) != 1:
        raise SceneCatalogError(
            f"trusted R2R catalog has {len(matches)} matches for "
            f"scene {requested!r}")
    return matches[0]


def _source_asset_sha256(source: dict, role: str) -> str:
    for asset in source.get("source_assets") or []:
        if isinstance(asset, dict) and asset.get("role") == role:
            return str(asset.get("sha256") or "")
    raise ValueError(f"source authority has no {role!r} asset")


def _authenticate_trusted_scene(
        spec: SceneSpec, expected_manifest_sha256: str) -> dict:
    manifest, manifest_sha256 = _read_pinned_r2r_manifest(
        spec.provenance_path)
    if (manifest_sha256 != spec.provenance_sha256 or
            manifest_sha256 != expected_manifest_sha256):
        raise SceneCatalogError("trusted R2R manifest digest changed")
    if spec.scene_id not in _r2r_manifest_scene_ids(manifest):
        raise SceneCatalogError(
            f"trusted R2R manifest does not authorize scene {spec.scene_id!r}")
    assets, source_assets_sha256 = _source_asset_identity(
        scene=spec.scene_path,
        navmesh=spec.navmesh_path,
        semantic=spec.semantic_path,
        semantic_metadata=spec.semantic_metadata_path,
        scene_dataset_config=spec.scene_dataset_config,
    )
    provenance = {
        "scene_id": spec.scene_id,
        "source_dataset": spec.source_dataset,
        "official_split": spec.official_split,
        "semantic_format": spec.semantic_format,
        "source_manifest_sha256": spec.provenance_sha256,
        "source_asset_identity_version": SOURCE_ASSET_IDENTITY_VERSION,
        "source_assets": [asset.to_dict() for asset in assets],
        "source_assets_sha256": source_assets_sha256,
    }
    return provenance


def _source_identity(source: dict) -> dict:
    """Remove the local manifest locator, which is not an authority field."""
    return {
        key: value for key, value in source.items()
        if key != "source_manifest"
    }


def _cache_trusted_r2r_scene(key: tuple, value: dict) -> None:
    maximum = int(config.TRUSTED_R2R_SCENE_CACHE_MAX_SCENES)
    if maximum <= 0:
        raise SceneCatalogError("trusted R2R scene cache is disabled")
    while len(_TRUSTED_R2R_SCENE_CACHE) >= maximum:
        _TRUSTED_R2R_SCENE_CACHE.popitem(last=False)
    _TRUSTED_R2R_SCENE_CACHE[key] = value


def _trusted_r2r_scene_for_context(rec: dict, context):
    source = rec.get("source") or {}
    scene_id = str(rec.get("scene_id") or "")
    expected = context.expected_collection_contracts.get(scene_id)
    if not isinstance(expected, dict):
        raise SceneCatalogError(
            "trusted validation context has no R2R scene contract")
    if (source.get("source_manifest_sha256") !=
            expected.get("source_manifest_sha256") or
            source.get("source_assets_sha256") !=
            expected.get("source_assets_sha256")):
        raise SceneCatalogError(
            "record source binding disagrees with expected collection "
            "contract")
    try:
        episodes_path = str(Path(
            context.r2r_train_episodes).resolve(strict=True))
        mp3d_root = str(Path(context.mp3d_root).resolve(strict=True))
    except (OSError, TypeError) as error:
        raise SceneCatalogError(
            f"trusted R2R roots are unavailable: {error}") from error
    key = (
        episodes_path, mp3d_root, scene_id,
        str(expected.get("source_manifest_sha256") or ""),
        str(expected.get("source_assets_sha256") or ""),
    )
    cached = _TRUSTED_R2R_SCENE_CACHE.get(key)
    if cached is not None:
        if _source_identity(cached["provenance"]) != _source_identity(source):
            raise SceneCatalogError(
                "cached trusted R2R provenance disagrees with record binding")
        _TRUSTED_R2R_SCENE_CACHE.move_to_end(key)
        return cached["spec"]
    spec = resolve_trusted_r2r_scene(
        scene_id, episodes_path=episodes_path, mp3d_root=mp3d_root)
    provenance = _authenticate_trusted_scene(
        spec, str(expected.get("source_manifest_sha256") or ""))
    if _source_identity(provenance) != _source_identity(source):
        raise SceneCatalogError(
            "trusted R2R source provenance disagrees with record binding")
    _cache_trusted_r2r_scene(key, {
        "spec": spec, "provenance": provenance,
    })
    return spec


def trusted_r2r_source_binding_errors(rec: dict, context) -> List[str]:
    """Authenticate every registered R2R source asset for one record."""
    fid = rec.get("frame_id", "?")
    if context.route != "r2r_v16_registered":
        return [f"[{fid}] source-bound validation requires registered "
                "R2R authority"]
    try:
        _trusted_r2r_scene_for_context(rec, context)
    except (KeyError, MemoryError, OSError, TypeError, ValueError) as error:
        return [f"[{fid}] trusted R2R source unavailable: {error}"]
    return []


def trusted_r2r_b_source_errors(rec: dict, context) -> List[str]:
    """Rederive B atoms from configured local R2R/MP3D source authority."""
    if context.route != "r2r_v16_registered" or rec.get("b_target") is None:
        return []
    from pipeline import outcome as outcome_fields, record as record_fields
    from pipeline import semantic
    from pipeline.geometry import points_to_ground_support_distances_m

    errors = []
    fid = rec.get("frame_id", "?")
    source = rec.get("source") or {}
    try:
        spec = _trusted_r2r_scene_for_context(rec, context)
        semantic_sha = _source_asset_sha256(source, "semantic")
        index = semantic.load_mp3d_target_authority(
            spec.scene_path, expected_semantic_ply_sha256=semantic_sha,
            expected_house_sha256=_source_asset_sha256(
                source, "semantic_metadata"))
        stored_target = rec.get("b_target")
        if not isinstance(stored_target, dict):
            raise ValueError("B target is not an object")
        selection = stored_target.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("B target selection is not an object")
        instance_id = selection.get("instance_id")
        if (not isinstance(instance_id, int) or
                isinstance(instance_id, bool) or instance_id <= 0):
            raise ValueError("B target instance is invalid")
        geometry = index.target_geometry_atom(
            instance_id, record_fields.require_floor_plane(rec),
            expected_semantic_ply_sha256=semantic_sha,
            pose=rec.get("pose") or {})
        if stored_target.get("geometry") != geometry:
            return [f"[{fid}] trusted B target geometry disagrees with record"]
        rebuilt_target = record_fields.build_b_target_atom(
            selection=selection, geometry=geometry,
            pose=rec.get("pose") or {})
        if stored_target != rebuilt_target:
            return [f"[{fid}] trusted B target atom disagrees with source"]
        relation_outcomes = [
            outcome for outcome in rec.get("outcomes") or []
            if outcome.get("b_endpoint_relation") is not None
        ]
        endpoint_world_xz = []
        for outcome in relation_outcomes:
            endpoint = outcome_fields.realized_pose(outcome)
            endpoint_world_xz.append(
                record_fields.local_ground_xz_to_world(
                    rec.get("pose") or {},
                    (endpoint["x"], endpoint["z"])))
        distances = (
            points_to_ground_support_distances_m(
                endpoint_world_xz, geometry["ground_support"])
            if endpoint_world_xz else [])
        for outcome, distance_after_m in zip(
                relation_outcomes, distances):
            stored_relation = outcome["b_endpoint_relation"]
            rebuilt = \
                record_fields.build_b_endpoint_relation_from_validated_target(
                pose=rec.get("pose") or {}, outcome=outcome,
                b_target=rebuilt_target,
                distance_after_m=float(distance_after_m))
            if stored_relation != rebuilt:
                errors.append(
                    f"[{fid}:{outcome.get('outcome_id', '?')}] trusted B "
                    "endpoint relation disagrees with source")
    except (KeyError, MemoryError, OSError, TypeError, ValueError) as error:
        return [f"[{fid}] trusted R2R source unavailable: {error}"]
    return errors


def trusted_r2r_a3_source_errors(rec: dict, context) -> List[str]:
    """Rederive exact A3 identities from authenticated MP3D PLY/house."""
    if context.route != "r2r_v16_registered":
        return []
    from pipeline import consensus, objects as object_fields, semantic

    fid = rec.get("frame_id", "?")
    source = rec.get("source") or {}
    try:
        spec = _trusted_r2r_scene_for_context(rec, context)
        semantic_sha = _source_asset_sha256(source, "semantic")
        authority = semantic.load_mp3d_target_authority(
            spec.scene_path,
            expected_semantic_ply_sha256=semantic_sha,
            expected_house_sha256=_source_asset_sha256(
                source, "semantic_metadata"))
        errors = []
        for outcome in rec.get("outcomes") or []:
            certificate = outcome.get("shared_oracle_stability") or {}
            summary = certificate.get("summary") or {}
            _category, _choices, category_reason = \
                object_fields.a3_category_evidence(rec, certificate)
            if (summary.get("collision_label_stable") is not True or
                    summary.get("collision") is not True or
                    category_reason is not None):
                continue
            rows = certificate.get("rows") or []
            prefix = f"[{fid}:{outcome.get('outcome_id', '?')}]"
            mismatches = consensus.a_stability_certificate_mismatches(
                outcome.get("actions") or [], certificate, outcome,
                authority_binding=dataset_contracts.AuthorityBinding(
                    source_dataset="r2r",
                    identity_schema="mp3d-contact-face-identity.v1",
                    authoritative_source_role="semantic",
                    source_sha256=semantic_sha,
                ))
            if mismatches:
                errors.append(
                    f"{prefix} trusted A3 certificate source binding "
                    f"disagrees: {', '.join(mismatches)}")
            requests = []
            stored_identities = []
            for row in rows:
                physical = row.get("physical") or {}
                if physical.get("collision") is not True:
                    continue
                row_consensus = row.get("consensus") or {}
                instance_id = row_consensus.get("full_contact_instance_id")
                world_point = (physical.get("contact") or {}).get(
                    "world_point")
                if (not isinstance(instance_id, int) or
                        isinstance(instance_id, bool) or instance_id <= 0 or
                        world_point is None):
                    raise ValueError(
                        "A3 contact replay inputs are incomplete")
                requests.append((instance_id, world_point))
                stored_identities.append(row.get("exact_contact_identity"))
            replayed = authority.confirm_contact_instances(requests)
            for stored, rebuilt in zip(stored_identities, replayed):
                if stored != rebuilt:
                    errors.append(
                        f"{prefix} trusted A3 contact identity disagrees "
                        "with source")
                    break
        return errors
    except (KeyError, MemoryError, OSError, TypeError, ValueError) as error:
        return [f"[{fid}] trusted R2R source unavailable: {error}"]


def discover_gs_train_scenes(
        root: os.PathLike, manifest: os.PathLike, *,
        requested: Optional[Sequence[os.PathLike]] = None,
        ) -> List[SceneSpec]:
    """Resolve explicitly declared GS train scenes and their local assets.

    ``requested`` narrows asset validation without weakening the manifest
    boundary: every requested key must still name a unique train entry.  This
    lets one-scene smoke runs proceed while collision artifacts for the rest of
    the authenticated catalog are still being preprocessed.  Full-catalog
    discovery continues to fail closed if any train asset is absent.
    """
    root = Path(root).resolve()
    manifest_path = Path(manifest).resolve()
    if not manifest_path.is_file():
        raise SceneCatalogError(
            f"GS train manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != SCENE_MANIFEST_VERSION:
        raise SceneCatalogError(
            "GS train manifest has unsupported schema_version")
    if payload.get("dataset") != "gs":
        raise SceneCatalogError("GS train manifest dataset must be 'gs'")
    entries = payload.get("scenes")
    if not isinstance(entries, list):
        raise SceneCatalogError("GS train manifest has no scenes list")
    allowed_splits = {"train", "val", "test"}
    requested_keys = None if requested is None else {
        str(value).strip() for value in requested}
    if requested_keys is not None and (
            not requested_keys or "" in requested_keys):
        raise SceneCatalogError("requested GS train scenes are empty")
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
        if split != "train":
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
            official_split="train",
            scene_path=str(scene_path),
            navmesh_path=str(navmesh),
            semantic_path=str(labels),
            semantic_metadata_path=None,
            semantic_format=dataset_contracts.dataset_source_contract(
                "gs").semantic_format,
            scene_dataset_config=None,
            provenance_path=str(manifest_path),
            provenance_sha256=manifest_hash,
            collision_authority_path=str(collision_authority),
        ))
    if requested_keys is not None:
        missing = sorted(requested_keys - matched_requested)
        if missing:
            raise SceneCatalogError(
                "requested GS scene is not in the verified train catalog: "
                + ", ".join(missing))
    if not specs:
        raise SceneCatalogError(
            f"GS manifest contains no complete train scenes: {manifest_path}")
    return sorted(specs, key=lambda value: value.scene_id)


def resolve_scene_subset(
        catalog: Sequence[SceneSpec],
        requested: Sequence[os.PathLike]) -> List[SceneSpec]:
    """Resolve explicit IDs/paths without permitting a non-train scene."""
    by_key = {}
    for scene in catalog:
        scene_path = Path(scene.scene_path).resolve()
        keys = {
            scene.scene_id,
            str(scene_path),
            str(scene_path.parent),
        }
        for key in keys:
            existing = by_key.get(key)
            if existing is not None and existing != scene:
                raise SceneCatalogError(
                    f"ambiguous verified scene key {key!r}")
            by_key[key] = scene
    resolved = []
    seen = set()
    unknown = []
    for value in requested:
        raw = str(value)
        candidates = [raw]
        path = Path(raw).expanduser()
        if path.exists() or path.is_absolute() or os.sep in raw:
            candidates.insert(0, str(path.resolve()))
        scene = next(
            (by_key[key] for key in candidates if key in by_key), None)
        if scene is None:
            unknown.append(raw)
            continue
        if scene.scene_id in seen:
            raise SceneCatalogError(
                f"duplicate requested scene {scene.scene_id!r}")
        seen.add(scene.scene_id)
        resolved.append(scene)
    if unknown:
        raise SceneCatalogError(
            "requested scenes are not in the verified train catalog: "
            + ", ".join(unknown))
    return resolved


def deterministic_scene_order(
        catalog: Sequence[SceneSpec], *, seed: int) -> List[SceneSpec]:
    """Return a source-stratified seeded order independent of filesystem order."""
    def rank(scene: SceneSpec) -> str:
        payload = (
            f"{int(seed)}\0{scene.source_dataset}\0{scene.scene_id}").encode()
        return hashlib.sha256(payload).hexdigest()
    return sorted(catalog, key=lambda scene: (rank(scene), scene.scene_id))
