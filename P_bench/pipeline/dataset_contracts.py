"""Immutable source contracts shared across dataset trust boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping

from pipeline import config


SOURCE_ASSET_IDENTITY_VERSION = "egoconseq.source-assets.v1"
GS_OFFICIAL_COORDINATE_BINDING_SCHEMA = \
    "gs-official-coordinate-binding.v1"
OFFICIAL_SOURCE_SPLITS = ("train", "val", "val_unseen", "test")


@dataclass(frozen=True)
class DatasetSourceContract:
    """Static simulator-input contract for one source dataset."""

    source_dataset: str
    required_asset_roles: tuple[str, ...]
    semantic_format: str
    main_collection_enabled: bool
    collection_source_param: str


@dataclass(frozen=True)
class AuthorityBinding:
    """Typed identity authority authenticated from one source asset."""

    source_dataset: str
    identity_schema: str
    authoritative_source_role: str
    source_sha256: str
    semantic_source_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
                "source_dataset", "identity_schema",
                "authoritative_source_role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"authority binding {name} is invalid")
        if (not isinstance(self.source_sha256, str) or
                not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256)):
            raise ValueError("authority binding source sha256 is invalid")
        if (self.semantic_source_sha256 is not None and
                (not isinstance(self.semantic_source_sha256, str) or
                 not re.fullmatch(
                     r"[0-9a-f]{64}", self.semantic_source_sha256))):
            raise ValueError(
                "authority binding semantic source sha256 is invalid")


@dataclass(frozen=True)
class GsCollisionAuthorityBinding:
    """GS renderer/labels plus official collision authority identity."""

    source_dataset: str
    identity_schema: str
    scene_id: str
    source_assets: tuple[tuple[str, int, str], ...]
    protocol_atom: dict
    protocol_sha256: str
    collision_authority_sha256: str
    authority_sha256: str

    def __post_init__(self) -> None:
        if self.source_dataset != "gs":
            raise ValueError("GS collision binding dataset must be 'gs'")
        if self.identity_schema != "gs-collision-runtime-authority.v1":
            raise ValueError("GS collision binding schema is invalid")
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("GS collision binding scene id is invalid")
        if tuple(value[0] for value in self.source_assets) != (
                "scene", "navmesh", "semantic", "collision_authority"):
            raise ValueError("GS collision binding asset roles are invalid")
        for role, byte_size, digest in self.source_assets:
            if (not isinstance(role, str) or not role or
                    isinstance(byte_size, bool) or
                    not isinstance(byte_size, int) or byte_size < 0 or
                    not isinstance(digest, str) or
                    not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError("GS collision binding asset is invalid")
        for name in (
                "protocol_sha256", "collision_authority_sha256",
                "authority_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or
                    not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError(f"GS collision binding {name} is invalid")


DATASET_SOURCE_CONTRACTS: Mapping[str, DatasetSourceContract] = MappingProxyType(
    {
        "r2r": DatasetSourceContract(
            source_dataset="r2r",
            required_asset_roles=(
                "scene", "navmesh", "semantic", "semantic_metadata",
                "scene_dataset_config",
            ),
            semantic_format="mp3d_ply",
            main_collection_enabled=True,
            collection_source_param="r2r_train_episodes",
        ),
        "gs": DatasetSourceContract(
            source_dataset="gs",
            required_asset_roles=(
                "scene", "navmesh", "semantic", "collision_authority"),
            semantic_format="gs_bbox",
            main_collection_enabled=True,
            collection_source_param="gs_source_manifest",
        ),
        "b1k": DatasetSourceContract(
            source_dataset="b1k",
            required_asset_roles=("scene", "scene_authority"),
            semantic_format="omnigibson_instance",
            main_collection_enabled=True,
            collection_source_param="b1k_source_manifest",
        ),
    }
)

_IDENTITY_BINDINGS = MappingProxyType({
    "r2r": ("mp3d-contact-face-identity.v1", "semantic"),
    "b1k": ("b1k-contact-triangle-identity.v1", "scene_authority"),
    "gs": ("gs-visible-contact-instance-identity.v1", "source_bundle"),
})
_COLLECTION_GPU_BY_DATASET = MappingProxyType({
    "gs": 0,
    "b1k": 1,
    "r2r": 2,
})


def dataset_source_contract(source_dataset: str) -> DatasetSourceContract:
    """Return one registered dataset contract; unknown names fail closed."""
    if not isinstance(source_dataset, str) or not source_dataset:
        raise ValueError(
            f"source asset dataset is unsupported: {source_dataset!r}")
    try:
        return DATASET_SOURCE_CONTRACTS[source_dataset]
    except KeyError as error:
        raise ValueError(
            f"source asset dataset is unsupported: {source_dataset!r}") \
            from error


def main_collection_datasets() -> tuple[str, ...]:
    """Return registered backends whose main adapter is certified."""
    return tuple(
        name for name, contract in DATASET_SOURCE_CONTRACTS.items()
        if contract.main_collection_enabled)


def collection_gpu_id(source_dataset: str) -> int:
    """Return the fixed collection GPU assigned to one dataset."""
    dataset_source_contract(source_dataset)
    return _COLLECTION_GPU_BY_DATASET[source_dataset]


def source_path_key(source_dataset: str) -> str:
    """Return the controller path key for one registered dataset."""
    return dataset_source_contract(source_dataset).collection_source_param


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_source_asset_provenance(source: dict) -> str:
    """Validate registered source roles, format, fields, and aggregate hash."""
    if not isinstance(source, dict):
        raise ValueError("source asset provenance must be an object")
    if source.get("source_asset_identity_version") != \
            SOURCE_ASSET_IDENTITY_VERSION:
        raise ValueError("source asset identity version is not current")
    contract = dataset_source_contract(source.get("source_dataset"))
    if source.get("semantic_format") != contract.semantic_format:
        raise ValueError(
            "source semantic format does not match the backend contract")
    assets = source.get("source_assets")
    if not isinstance(assets, list):
        raise ValueError("source assets must be a list")
    observed_roles = tuple(
        asset.get("role") if isinstance(asset, dict) else None
        for asset in assets)
    if observed_roles != contract.required_asset_roles:
        raise ValueError(
            "source asset roles do not match the backend contract")
    for asset in assets:
        if set(asset) != {"role", "bytes", "sha256"}:
            raise ValueError("source asset fields are invalid")
        byte_size = asset["bytes"]
        if (isinstance(byte_size, bool) or
                not isinstance(byte_size, int) or byte_size < 0):
            raise ValueError("source asset byte size is invalid")
        digest = asset["sha256"]
        if (not isinstance(digest, str) or
                not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("source asset sha256 is invalid")
    expected = _canonical_sha256({
        "version": SOURCE_ASSET_IDENTITY_VERSION,
        "assets": assets,
    })
    if source.get("source_assets_sha256") != expected:
        raise ValueError("source asset digest does not match its assets")
    return expected


def resolve_authority_binding(source: dict) -> AuthorityBinding:
    """Authenticate and type the exact A3 identity source for one dataset."""
    validate_source_asset_provenance(source)
    dataset = dataset_source_contract(
        source.get("source_dataset")).source_dataset
    try:
        identity_schema, source_role = _IDENTITY_BINDINGS[dataset]
    except KeyError as error:
        raise ValueError(
            f"{dataset} identity binding is not certified") from error
    semantic_source_sha256 = None
    if dataset == "gs":
        source_sha256 = resolve_gs_collision_binding(source).authority_sha256
        semantic_source_sha256 = next(
            asset["sha256"] for asset in source["source_assets"]
            if asset["role"] == "semantic")
    else:
        source_asset = next(
            asset for asset in source["source_assets"]
            if asset["role"] == source_role)
        source_sha256 = source_asset["sha256"]
    return AuthorityBinding(
        source_dataset=dataset,
        identity_schema=identity_schema,
        authoritative_source_role=source_role,
        source_sha256=source_sha256,
        semantic_source_sha256=semantic_source_sha256,
    )


def _gs_collision_protocol_atom() -> dict:
    return {
        "schema": "gs-collision-protocol-stack.v1",
        "renderer": {
            "protocol": "gsplat-rgb-ed.v1",
            "near_m": float(config.GS_RENDER_NEAR_M),
            "far_m": float(config.GS_RENDER_FAR_M),
            "role": "observation_depth_and_future_rgb",
        },
        "geometry": {
            "protocol": "sage3d-source-collision-footprint.v1",
            "body_model": "radius-conditioned-planar-disc",
            "source": "official_sage3d_collision_usd",
        },
        "depth_consensus": {
            "protocol": "swept_floor_v6",
            "source": "initial_rgbd",
        },
        "semantic_identity": {
            "alignment_certificate":
                GS_OFFICIAL_COORDINATE_BINDING_SCHEMA,
            "assignment": "unique-transformed-bbox-containment.v1",
            "contact_witness": "initial-visible-depth-instance.v1",
        },
        "target_geometry": {
            "protocol": "initial-visible-depth-anchor.v1",
            "scope": "visible_anchor_not_complete_object_surface",
        },
        "navmesh": {
            "source_bound": True,
            "proposal_only": True,
            "label_authority": False,
        },
        "authority_strength": "collision-mesh-plus-initial-depth-consensus",
    }


def resolve_gs_collision_binding(source: dict) -> GsCollisionAuthorityBinding:
    """Authenticate the GS bundle used by new collision-backed collection."""
    validate_source_asset_provenance(source)
    if source.get("source_dataset") != "gs":
        raise ValueError("GS collision binding requires a GS source")
    scene_id = source.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("GS collision binding scene id is invalid")
    assets = tuple(
        (asset["role"], int(asset["bytes"]), asset["sha256"])
        for asset in source["source_assets"])
    collision_sha256 = next(
        digest for role, _byte_size, digest in assets
        if role == "collision_authority")
    protocol_atom = _gs_collision_protocol_atom()
    protocol_sha256 = _canonical_sha256(protocol_atom)
    body = {
        "schema": "gs-collision-runtime-authority.v1",
        "source_dataset": "gs",
        "scene_id": scene_id,
        "source_assets": [list(value) for value in assets],
        "protocol_sha256": protocol_sha256,
        "collision_authority_sha256": collision_sha256,
    }
    return GsCollisionAuthorityBinding(
        source_dataset="gs",
        identity_schema=body["schema"],
        scene_id=scene_id,
        source_assets=assets,
        protocol_atom=protocol_atom,
        protocol_sha256=protocol_sha256,
        collision_authority_sha256=collision_sha256,
        authority_sha256=_canonical_sha256(body),
    )


def gs_collision_binding_atom(binding: GsCollisionAuthorityBinding) -> dict:
    if not isinstance(binding, GsCollisionAuthorityBinding):
        raise TypeError("GS collision binding has the wrong type")
    return {
        "schema": binding.identity_schema,
        "source_dataset": binding.source_dataset,
        "scene_id": binding.scene_id,
        "source_assets": [
            {"role": role, "bytes": int(byte_size), "sha256": digest}
            for role, byte_size, digest in binding.source_assets
        ],
        "protocol_atom": json.loads(json.dumps(
            binding.protocol_atom, sort_keys=True)),
        "protocol_sha256": binding.protocol_sha256,
        "collision_authority_sha256":
            binding.collision_authority_sha256,
        "sha256": binding.authority_sha256,
    }


def gs_collision_floor_reference_atom(
        binding: GsCollisionAuthorityBinding, *, ground_y_m: float) -> dict:
    """Bind a level GS floor to the official collision artifact."""
    if not isinstance(binding, GsCollisionAuthorityBinding):
        raise TypeError("GS collision floor binding has the wrong type")
    ground_y = float(ground_y_m)
    if not math.isfinite(ground_y):
        raise ValueError("GS collision floor height must be finite")
    body = {
        "schema": "gs-collision-floor-reference.v1",
        "collision_authority_sha256":
            binding.collision_authority_sha256,
        "binding_sha256": binding.authority_sha256,
        "ground_y_m": ground_y,
        "plane_local": {
            "normal_local": [0.0, 1.0, 0.0],
            "offset_m": 0.0,
        },
    }
    return {**body, "sha256": _canonical_sha256(body)}
