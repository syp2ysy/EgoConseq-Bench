"""Build strict record-validation contexts from trusted run metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from pipeline import action_proposal, config, dataset_contracts, record as REC

_HEX = frozenset("0123456789abcdef")


def load_object_identity(path: Path, *, label: str) -> tuple[dict, str]:
    """Load one explicit JSON authority file and return its byte identity."""
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validation_sha256(value, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(char not in _HEX for char in digest):
        raise ValueError(f"{label} sha256 is invalid")
    return digest


@dataclass(frozen=True)
class RecordValidationContext:
    """Trusted route metadata supplied independently of a record payload."""

    route: str
    expected_schema_version: str
    expected_oracle_contract_version: str
    expected_collection_contracts: Mapping[str, dict]
    authority_sha256: Optional[str] = None
    r2r_train_episodes: Optional[str] = None
    mp3d_root: Optional[str] = None
    b1k_scene_authority_resolver: Optional[Callable[[str], str]] = None
    gs_scene_source_resolver: Optional[Callable[[str], dict]] = None
    gs_scene_capability_resolver: Optional[Callable[[str], dict]] = None
    # The policies the run committed to, read from hash-verified metadata. A
    # record cannot weaken its own contract by dropping the marker, and it
    # cannot claim to have been depth-conditioned when the run was driven by an
    # action file.
    expected_setting_sampling_policy: Optional[str] = None
    expected_action_sampling_policy: Optional[str] = None

    def __post_init__(self) -> None:
        if self.expected_schema_version not in {
                REC.SCHEMA_VERSION, REC.V18_SCHEMA_VERSION}:
            raise ValueError("validation context schema version is unsupported")
        if self.expected_oracle_contract_version != REC.ORACLE_CONTRACT_VERSION:
            raise ValueError(
                "validation context oracle contract version is unsupported")
        if self.route not in {
                "legacy", "r2r_v16_registered", "b1k_v16_registered",
                "gs_v18_registered"}:
            raise ValueError(f"unsupported record validation route {self.route!r}")
        if self.route == "legacy" and self.expected_collection_contracts:
            raise ValueError(
                "legacy validation context cannot require collection contracts")
        if (self.route == "r2r_v16_registered" and
                not self.expected_collection_contracts):
            raise ValueError(
                "registered R2R v16 context requires collection contracts")
        if self.route == "r2r_v16_registered":
            _validation_sha256(self.authority_sha256, "authority")
            if not self.r2r_train_episodes or not self.mp3d_root:
                raise ValueError(
                    "registered R2R validation context requires trusted roots")
        if self.route == "b1k_v16_registered":
            if not self.expected_collection_contracts:
                raise ValueError(
                    "registered B1K context requires collection contracts")
            if self.expected_schema_version != REC.SCHEMA_VERSION:
                raise ValueError(
                    "registered B1K context requires conseq.v11")
            _validation_sha256(self.authority_sha256, "authority")
            if not callable(self.b1k_scene_authority_resolver):
                raise ValueError(
                    "registered B1K context requires a scene authority resolver")
        if self.route == "gs_v18_registered":
            if self.expected_collection_contracts:
                raise ValueError(
                    "registered GS v18 context forbids legacy contracts")
            if self.expected_schema_version != REC.V18_SCHEMA_VERSION:
                raise ValueError("registered GS context requires conseq.v18")
            _validation_sha256(self.authority_sha256, "authority")
            if not callable(self.gs_scene_source_resolver):
                raise ValueError(
                    "registered GS context requires a scene source resolver")
            if not callable(self.gs_scene_capability_resolver):
                raise ValueError(
                    "registered GS context requires a scene capability resolver")


LEGACY_RECORD_VALIDATION_CONTEXT = RecordValidationContext(
    route="legacy",
    expected_schema_version=REC.SCHEMA_VERSION,
    expected_oracle_contract_version=REC.ORACLE_CONTRACT_VERSION,
    expected_collection_contracts={},
)


def r2r_v16_registered_validation_context(
        sources: Iterable[dict], *, collection_mode: str,
        expected_schema_version: str,
        expected_oracle_contract_version: str,
        authority_sha256: str,
        r2r_train_episodes=None, mp3d_root=None,
        setting_sampling_policy=None,
        action_sampling_policy=None,
        ) -> RecordValidationContext:
    """Build strict context from trusted metadata, never record fields."""
    contracts = {}
    for source in sources:
        scene_id = str(source.get("scene_id") or "")
        if not scene_id:
            raise ValueError("registered source is missing scene_id")
        contract = REC.collection_contract(
            source, collection_mode,
            record_schema_version=expected_schema_version)
        previous = contracts.setdefault(scene_id, contract)
        if previous != contract:
            raise ValueError(f"conflicting provenance for scene {scene_id!r}")
    return RecordValidationContext(
        route="r2r_v16_registered",
        expected_schema_version=expected_schema_version,
        expected_oracle_contract_version=expected_oracle_contract_version,
        expected_collection_contracts=contracts,
        authority_sha256=authority_sha256,
        r2r_train_episodes=str(
            config.R2R_TRAIN_EPISODES if r2r_train_episodes is None
            else r2r_train_episodes),
        mp3d_root=str(config.MP3D_ROOT if mp3d_root is None else mp3d_root),
        expected_setting_sampling_policy=setting_sampling_policy,
        expected_action_sampling_policy=action_sampling_policy,
    )


def b1k_v16_registered_validation_context(
        sources: Iterable[dict], *, collection_mode: str,
        expected_schema_version: str,
        expected_oracle_contract_version: str,
        authority_sha256: str,
        scene_authority_resolver: Callable[[str], str],
        setting_sampling_policy=None,
        action_sampling_policy=None,
        ) -> RecordValidationContext:
    """Build strict B1K context with an external scene-authority resolver."""
    contracts = {}
    for source in sources:
        scene_id = str(source.get("scene_id") or "")
        if not scene_id:
            raise ValueError("registered source is missing scene_id")
        contract = REC.collection_contract(
            source, collection_mode,
            record_schema_version=expected_schema_version)
        previous = contracts.setdefault(scene_id, contract)
        if previous != contract:
            raise ValueError(f"conflicting provenance for scene {scene_id!r}")
    return RecordValidationContext(
        route="b1k_v16_registered",
        expected_schema_version=expected_schema_version,
        expected_oracle_contract_version=expected_oracle_contract_version,
        expected_collection_contracts=contracts,
        authority_sha256=authority_sha256,
        b1k_scene_authority_resolver=scene_authority_resolver,
        expected_setting_sampling_policy=setting_sampling_policy,
        expected_action_sampling_policy=action_sampling_policy,
    )


def gs_v18_registered_validation_context(
        sources: Iterable[dict], *, authority_sha256: str,
        scene_capabilities: Mapping[str, dict],
        setting_sampling_policy=None,
        action_sampling_policy=None) -> RecordValidationContext:
    """Build a v18 context from exact GS collision-backed source bundles."""
    trusted = {}
    for source in sources:
        dataset_contracts.validate_source_asset_provenance(source)
        if source.get("source_dataset") != "gs":
            raise ValueError("registered GS context received a non-GS source")
        scene_id = str(source.get("scene_id") or "")
        if not scene_id:
            raise ValueError("registered GS source is missing scene_id")
        canonical = json.loads(json.dumps(source, sort_keys=True))
        previous = trusted.setdefault(scene_id, canonical)
        if previous != canonical:
            raise ValueError(f"conflicting GS provenance for scene {scene_id!r}")

    def resolve_scene_source(scene_id: str) -> dict:
        try:
            return json.loads(json.dumps(trusted[str(scene_id)], sort_keys=True))
        except KeyError as error:
            raise ValueError(
                "GS scene source is absent from the trusted catalog") from error

    capabilities = {}
    for scene_id, source in trusted.items():
        try:
            capability = scene_capabilities[scene_id]
        except KeyError as error:
            raise ValueError(
                f"GS scene capability is missing for {scene_id!r}") from error
        from pipeline import gs_semantic
        gs_semantic.validate_scene_capability_atom(capability, source)
        capabilities[scene_id] = json.loads(json.dumps(
            capability, sort_keys=True))

    def resolve_scene_capability(scene_id: str) -> dict:
        try:
            return json.loads(json.dumps(
                capabilities[str(scene_id)], sort_keys=True))
        except KeyError as error:
            raise ValueError(
                "GS scene capability is absent from the trusted catalog") \
                from error

    return RecordValidationContext(
        route="gs_v18_registered",
        expected_schema_version=REC.V18_SCHEMA_VERSION,
        expected_oracle_contract_version=REC.ORACLE_CONTRACT_VERSION,
        expected_collection_contracts={},
        authority_sha256=authority_sha256,
        gs_scene_source_resolver=resolve_scene_source,
        gs_scene_capability_resolver=resolve_scene_capability,
        expected_setting_sampling_policy=setting_sampling_policy,
        expected_action_sampling_policy=action_sampling_policy,
    )


def _action_sampling_policy_from_params(params: dict) -> Optional[str]:
    """Read an explicit new policy or infer the exact frozen legacy one."""
    declared = params.get("action_sampling_policy")
    if declared is None:
        return action_proposal.expected_policy_for_action_mode(
            params.get("action_mode"))
    if not action_proposal.declared_policy_matches_action_mode(
            params.get("action_mode"), declared):
        raise ValueError(
            "run metadata action-sampling policy disagrees with its mode")
    return str(declared)


def r2r_v16_context_from_run_contract(
        run_contract: dict, *, expected_run_contract_sha256: str,
        expected_schema_version: str,
        expected_oracle_contract_version: str) -> RecordValidationContext:
    """Verify a trusted run-contract hash and derive strict context."""
    actual_sha256 = _canonical_sha256(run_contract)
    if actual_sha256 != expected_run_contract_sha256:
        raise ValueError("run contract hash does not match trusted metadata")
    params = run_contract.get("params") or {}
    resolved_scenes = run_contract.get("resolved_scenes")
    if not isinstance(resolved_scenes, list):
        raise ValueError("run contract resolved_scenes must be a list")
    if not all(isinstance(source, dict) for source in resolved_scenes):
        raise ValueError("registered R2R v16 run lacks source provenance")
    episodes, mp3d_root = _r2r_roots_from_params(params)
    return r2r_v16_registered_validation_context(
        resolved_scenes,
        collection_mode=params.get("collection_mode"),
        expected_schema_version=expected_schema_version,
        expected_oracle_contract_version=expected_oracle_contract_version,
        authority_sha256=expected_run_contract_sha256,
        r2r_train_episodes=episodes,
        mp3d_root=mp3d_root,
        setting_sampling_policy=params.get("setting_sampling_policy"),
        action_sampling_policy=_action_sampling_policy_from_params(params),
    )


def r2r_v16_context_from_run_meta(
        run_meta: dict, *, authority_sha256: str) -> RecordValidationContext:
    """Validate explicit run metadata and derive its strict context."""
    _validation_sha256(run_meta.get("run_contract_sha256"), "run contract")
    params = run_meta.get("params") or {}
    resolved_scenes = run_meta.get("resolved_scenes")
    if not isinstance(resolved_scenes, list):
        raise ValueError("run metadata resolved_scenes must be a list")
    if not all(isinstance(source, dict) for source in resolved_scenes):
        raise ValueError("registered R2R v16 run lacks source provenance")
    episodes, mp3d_root = _r2r_roots_from_params(params)
    return r2r_v16_registered_validation_context(
        resolved_scenes,
        collection_mode=params.get("collection_mode"),
        expected_schema_version=run_meta.get("record_schema_version"),
        expected_oracle_contract_version=run_meta.get(
            "oracle_contract_version"),
        authority_sha256=authority_sha256,
        r2r_train_episodes=episodes,
        mp3d_root=mp3d_root,
        setting_sampling_policy=params.get("setting_sampling_policy"),
        action_sampling_policy=_action_sampling_policy_from_params(params),
    )


def b1k_v16_context_from_run_meta(
        run_meta: dict, *, authority_sha256: str) -> RecordValidationContext:
    """Authenticate B1K run metadata against its pinned source manifest."""
    from pipeline import scene_pool

    _validation_sha256(run_meta.get("run_contract_sha256"), "run contract")
    params = run_meta.get("params") or {}
    if params.get("backend") != "b1k":
        raise ValueError("run metadata is not a B1K source")
    data_root = params.get("b1k_data_root")
    manifest_path = params.get("b1k_source_manifest")
    if (not isinstance(data_root, str) or not data_root.strip() or
            not isinstance(manifest_path, str) or not manifest_path.strip()):
        raise ValueError(
            "run metadata lacks hash-authorized trusted B1K roots")
    resolved_scenes = run_meta.get("resolved_scenes")
    if (not isinstance(resolved_scenes, list) or
            not all(isinstance(source, dict) for source in resolved_scenes)):
        raise ValueError("registered B1K v16 run lacks source provenance")
    try:
        catalog = scene_pool.discover_b1k_train_scenes(
            data_root, manifest_path)
    except scene_pool.SceneCatalogError as error:
        raise ValueError(
            f"trusted B1K catalog is unavailable: {error}") from error
    catalog_by_id = {scene.scene_id: scene for scene in catalog}
    authorities = {}
    for source in resolved_scenes:
        scene_id = str(source.get("scene_id") or "")
        scene = catalog_by_id.get(scene_id)
        if scene is None:
            raise ValueError(
                f"trusted B1K catalog has no scene {scene_id!r}")
        expected = {**scene.provenance(), "scene_path": scene.scene_path}
        if source != expected:
            raise ValueError(
                f"trusted B1K provenance differs for scene {scene_id!r}")
        authorities[scene_id] = str(
            (scene.b1k_scene_authority or {}).get("sha256") or "")

    def resolve_scene_authority(scene_id: str) -> str:
        try:
            return authorities[str(scene_id)]
        except KeyError as error:
            raise ValueError(
                "B1K scene authority is absent from the trusted catalog") \
                from error

    return b1k_v16_registered_validation_context(
        resolved_scenes,
        collection_mode=params.get("collection_mode"),
        expected_schema_version=run_meta.get("record_schema_version"),
        expected_oracle_contract_version=run_meta.get(
            "oracle_contract_version"),
        authority_sha256=authority_sha256,
        scene_authority_resolver=resolve_scene_authority,
        setting_sampling_policy=params.get("setting_sampling_policy"),
        action_sampling_policy=_action_sampling_policy_from_params(params),
    )


def gs_v18_context_from_run_meta(
        run_meta: dict, *, authority_sha256: str) -> RecordValidationContext:
    """Authenticate one GS v18 run against its registered source manifest."""
    from pipeline import scene_pool

    _validation_sha256(authority_sha256, "authority")
    _validation_sha256(run_meta.get("run_contract_sha256"), "run contract")
    params = run_meta.get("params") or {}
    if params.get("backend") != "gs":
        raise ValueError("run metadata is not a GS source")
    if run_meta.get("record_schema_version") != REC.V18_SCHEMA_VERSION:
        raise ValueError("registered GS run metadata requires conseq.v18")
    data_root = params.get("gs_data_root")
    manifest_path = params.get("gs_source_manifest")
    if (not isinstance(data_root, str) or not data_root.strip() or
            not isinstance(manifest_path, str) or not manifest_path.strip()):
        raise ValueError("run metadata lacks trusted GS roots")
    resolved_scenes = run_meta.get("resolved_scenes")
    if (not isinstance(resolved_scenes, list) or
            not all(isinstance(source, dict) for source in resolved_scenes)):
        raise ValueError("registered GS v18 run lacks source provenance")
    requested_scene_ids = [
        str(source.get("scene_id") or "") for source in resolved_scenes]
    if (not requested_scene_ids or any(not value for value in requested_scene_ids)
            or len(requested_scene_ids) != len(set(requested_scene_ids))):
        raise ValueError("registered GS v18 run has invalid scene identities")
    try:
        catalog = scene_pool.discover_gs_train_scenes(
            data_root, manifest_path, requested=requested_scene_ids)
    except scene_pool.SceneCatalogError as error:
        raise ValueError(f"trusted GS catalog is unavailable: {error}") \
            from error
    catalog_by_id = {scene.scene_id: scene for scene in catalog}
    trusted_sources = []
    trusted_capabilities = {}
    for declared in resolved_scenes:
        scene_id = str(declared.get("scene_id") or "")
        scene = catalog_by_id.get(scene_id)
        if scene is None:
            raise ValueError(f"trusted GS catalog has no scene {scene_id!r}")
        try:
            trusted = scene_pool.verify_scene_source_asset_paths(scene, (
                ("scene", scene.scene_path),
                ("navmesh", scene.navmesh_path),
                ("semantic", scene.semantic_path),
                ("collision_authority",
                 scene.collision_authority_path),
            ))
        except scene_pool.SceneCatalogError as error:
            raise ValueError(f"trusted GS source is unavailable: {error}") \
                from error
        expected = {**trusted, "scene_path": scene.scene_path}
        if declared != expected:
            raise ValueError(
                f"trusted GS provenance differs for scene {scene_id!r}")
        trusted_sources.append(trusted)
        from pipeline import gs_semantic
        trusted_capabilities[scene_id] = \
            gs_semantic.derive_scene_capability_from_sources(
                trusted, scene.semantic_path)
    return gs_v18_registered_validation_context(
        trusted_sources,
        authority_sha256=authority_sha256,
        scene_capabilities=trusted_capabilities,
        setting_sampling_policy=params.get("setting_sampling_policy"),
        action_sampling_policy=_action_sampling_policy_from_params(params),
    )


def _r2r_roots_from_params(params: dict) -> tuple[str, str]:
    episodes = params.get("r2r_train_episodes")
    mp3d_root = params.get("mp3d_root")
    if (not isinstance(episodes, str) or not episodes.strip() or
            not isinstance(mp3d_root, str) or not mp3d_root.strip()):
        raise ValueError(
            "run metadata lacks hash-authorized trusted R2R roots")
    return episodes, mp3d_root
