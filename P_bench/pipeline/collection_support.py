"""Shared collection contracts, scene discovery, and terminal rendering."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

from pipeline import (
    action_control_catalog, action_proposal, capability_contracts,
    collection_funnel, config, dataset_contracts, gs_semantic,
    scene_partitions, source_manifest,
)
from pipeline import record as REC
from pipeline.frame import build_terminal_rgb_observation
from pipeline.io_utils import sha256_file
from pipeline.scene_pool import (
    SceneCatalogError,
    SceneSpec,
    deterministic_scene_order,
    discover_b1k_train_scenes,
    discover_gs_train_scenes,
    discover_r2r_train_scenes,
    resolve_scene_subset,
)


def prepared_pose_state(
        *, variants, pools, group_labels, precheck_cache,
        target_ids_by_frame, proposal_provenance, required_siblings,
        calibration, position, yaw, base_frame,
        active_radii, setting, bank_manifest, intervention_group_id,
        scene, scene_id, c1_families=None,
        control_tags=None, shortlist_size=None) -> dict:
    """Build the explicit state handed across collection pose phases."""
    state = {
        "variants": variants, "pools": pools,
        "group_labels": group_labels, "precheck_cache": precheck_cache,
        "target_ids_by_frame": target_ids_by_frame,
        "proposal_provenance": proposal_provenance,
        "required_siblings": required_siblings,
        "calibration": calibration, "position": position, "yaw": yaw,
        "base_frame": base_frame,
        "active_radii": active_radii, "setting": setting,
        "bank_manifest": bank_manifest,
        "intervention_group_id": intervention_group_id,
        "scene": scene, "scene_id": scene_id,
    }
    if c1_families is not None:
        state["c1_families"] = tuple(c1_families)
    if control_tags is not None:
        state["control_tags"] = tuple(str(tag) for tag in control_tags)
    if shortlist_size is not None:
        # Publication budget after the wider, private stability reserve has
        # been consumed. Validators deliberately bound records, not internal
        # candidates that never reached a record.
        state["shortlist_size"] = int(shortlist_size)
    return state


def strict_shared_oracle_required(
        collection_mode: str, source_dataset: str) -> bool:
    """Whether a registered main backend requires the shared A certificate."""
    contract = dataset_contracts.dataset_source_contract(source_dataset)
    return bool(
        str(collection_mode) == "main" and
        contract.main_collection_enabled)


def contact_instance_witness_required(
        collection_mode: str, source_dataset: str, *,
        semantic_certified=None) -> bool:
    """Require semantic contact identity only where A3 is available."""
    if not strict_shared_oracle_required(collection_mode, source_dataset):
        return False
    if source_dataset == "gs":
        return semantic_certified is True
    return capability_contracts.task_available(source_dataset, "A3")


def collection_record_schema(args) -> str:
    """Return the explicitly requested record schema for this run."""
    if getattr(args, "backend", "r2r") == "gs":
        return REC.V18_SCHEMA_VERSION
    return REC.SCHEMA_VERSION


def open_collection_sessions(args, scene, heights, fovs):
    """Open one backend scene, sharing it across every sensor profile."""
    contract = dataset_contracts.dataset_source_contract(args.backend)
    if not contract.main_collection_enabled:
        raise SceneCatalogError(
            f"collection backend is unsupported: {args.backend!r}")
    if args.backend == "b1k":
        from pipeline.b1k_sim import B1KSimSession

        shared = B1KSimSession(scene, heights=heights, fovs=fovs)
    elif args.backend == "r2r":
        from pipeline.sim import SimSession

        shared = SimSession(
            scene.scene_path,
            scene_dataset_cfg=scene.scene_dataset_config,
            semantic_format=scene.semantic_format,
            source_dataset=scene.source_dataset,
            official_split=scene.official_split,
            heights=heights, fovs=fovs,
            semantic_query_workers=args.semantic_query_workers)
    elif args.backend == "gs":
        from pipeline.gs_sim import GsSimSession

        shared = GsSimSession(scene, heights=heights)
    else:
        raise SceneCatalogError(
            f"collection backend is unsupported: {args.backend!r}")
    return [shared for _ in fovs]


def run_with_backend_shutdown(args, parser, collector):
    """Run a collector with process-final B1K cleanup and intact status."""
    if getattr(args, "backend", "r2r") != "b1k":
        return collector(args, parser)
    from pipeline.b1k_sim import shutdown_b1k_runtime

    result = collector(args, parser)
    # OmniGibson's SimulationApp.close() terminates the host process with a
    # zero status on this pinned runtime.  A failed collector has already
    # removed its sensor and cleared its scene in the scene-level finally; let
    # the CLI receive its nonzero result before process teardown can erase it.
    if result:
        return result
    try:
        shutdown_b1k_runtime()
    except SystemExit as error:
        # SimulationApp.close() may exit zero; an existing failure still wins.
        if not result and error.code not in (None, 0):
            raise
    except BaseException:
        if not result:
            raise
    return result


def preserve_b1k_cli_status(args, result: int, *, hard_exit=None) -> int:
    """Exit failed B1K runs before Isaac interpreter teardown can mask them."""
    status = int(result)
    if getattr(args, "backend", "r2r") == "b1k" and status:
        sys.stdout.flush()
        sys.stderr.flush()
        (os._exit if hard_exit is None else hard_exit)(status)
    return status


def finish_scene_cleanup(close, finish) -> None:
    """Attempt both cleanup steps without replacing an active failure."""
    active_error = sys.exc_info()[1]
    cleanup_error = None
    for callback in (close, finish):
        try:
            callback()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
    if active_error is None and cleanup_error is not None:
        raise cleanup_error


def b1k_registered_validation_context(
        sources, *, collection_mode: str, record_schema: str,
        authority_sha256: str, scene_authorities: dict,
        setting_sampling_policy=None, action_sampling_policy=None):
    """Build a strict B1K context from independently opened authorities."""
    authorities = dict(scene_authorities)

    def resolve_scene_authority(scene_id):
        try:
            return authorities[str(scene_id)]
        except KeyError as error:
            raise ValueError(
                "B1K scene authority is absent from the trusted catalog") \
                from error

    return source_manifest.b1k_v16_registered_validation_context(
        sources, collection_mode=collection_mode,
        expected_schema_version=record_schema,
        expected_oracle_contract_version=REC.ORACLE_CONTRACT_VERSION,
        authority_sha256=authority_sha256,
        scene_authority_resolver=resolve_scene_authority,
        setting_sampling_policy=setting_sampling_policy,
        action_sampling_policy=action_sampling_policy)


def trusted_action_sampling_policy(args):
    """Derive the expected record policy exclusively from trusted run args."""
    declared = getattr(args, "action_sampling_policy", None)
    if declared is None:
        # Frozen v1/v2 metadata predates an explicit policy field.
        return action_proposal.expected_policy_for_action_mode(
            getattr(args, "action_mode", None))
    if not action_proposal.declared_policy_matches_action_mode(
            getattr(args, "action_mode", None), declared):
        raise ValueError("run action-sampling policy disagrees with its mode")
    return str(declared)


def b1k_run_validation_context(
        args, sources, scenes, *, record_schema: str,
        authority_sha256: str):
    """Return a strict final B1K context, or ``None`` for another route."""
    sources = list(sources)
    if not (sources and all(
            strict_shared_oracle_required(
                args.collection_mode, source.get("source_dataset", "")) and
            source.get("source_dataset") == "b1k"
            for source in sources)):
        return None
    scene_authorities = {
        scene.scene_id: str(
            (scene.b1k_scene_authority or {}).get("sha256") or "")
        for scene in scenes
    }
    return b1k_registered_validation_context(
        sources, collection_mode=args.collection_mode,
        record_schema=record_schema, authority_sha256=authority_sha256,
        scene_authorities=scene_authorities,
        setting_sampling_policy=args.setting_sampling_policy,
        action_sampling_policy=trusted_action_sampling_policy(args))


def final_validation_context(
        args, sources, scenes, *, record_schema: str,
        authority_sha256: str, run_contract: dict):
    """Resolve the source-bound validator used after one complete run."""
    sources = list(sources)
    if record_schema == REC.V18_SCHEMA_VERSION:
        scene_by_id = {scene.scene_id: scene for scene in scenes}
        capabilities = {
            source["scene_id"]:
                gs_semantic.derive_scene_capability_from_sources(
                    source,
                    scene_by_id[source["scene_id"]].semantic_path)
            for source in sources
        }
        return source_manifest.gs_v18_registered_validation_context(
            sources, authority_sha256=authority_sha256,
            scene_capabilities=capabilities,
            setting_sampling_policy=args.setting_sampling_policy,
            action_sampling_policy=trusted_action_sampling_policy(args))
    if sources and all(
            source.get("source_dataset") == "r2r" and
            strict_shared_oracle_required(
                args.collection_mode, source.get("source_dataset", ""))
            for source in sources):
        return source_manifest.r2r_v16_context_from_run_contract(
            run_contract,
            expected_run_contract_sha256=authority_sha256,
            expected_schema_version=record_schema,
            expected_oracle_contract_version=REC.ORACLE_CONTRACT_VERSION)
    b1k_context = b1k_run_validation_context(
        args, sources, scenes, record_schema=record_schema,
        authority_sha256=authority_sha256)
    return b1k_context or dataclasses.replace(
        source_manifest.LEGACY_RECORD_VALIDATION_CONTEXT,
        expected_setting_sampling_policy=args.setting_sampling_policy)


def pose_validation_context(
        args, source: dict, variants, *, record_schema: str):
    """Resolve the source-bound validator before one pose reaches its spool."""
    if record_schema == REC.V18_SCHEMA_VERSION:
        capability = gs_semantic.scene_capability_atom(
            source,
            getattr(
                getattr(variants[0][0], "semantic_index", None),
                "alignment_certificate", None))
        return source_manifest.gs_v18_registered_validation_context(
            [source], authority_sha256=source["source_assets_sha256"],
            scene_capabilities={source["scene_id"]: capability},
            setting_sampling_policy=args.setting_sampling_policy,
            action_sampling_policy=trusted_action_sampling_policy(args))
    if not strict_shared_oracle_required(
            args.collection_mode, source.get("source_dataset", "")):
        return dataclasses.replace(
            source_manifest.LEGACY_RECORD_VALIDATION_CONTEXT,
            expected_setting_sampling_policy=args.setting_sampling_policy)
    if source.get("source_dataset") == "b1k":
        authorities = {
            str(sim.scene_id): str(sim.scene_authority_sha256)
            for sim, _frame in variants
        }
        return b1k_registered_validation_context(
            [source], collection_mode=args.collection_mode,
            record_schema=record_schema,
            authority_sha256=collection_funnel.canonical_sha256({
                "source": source,
                "collection_mode": args.collection_mode,
                "record_schema_version": record_schema,
                "oracle_contract_version": REC.ORACLE_CONTRACT_VERSION,
                "b1k_data_root": str(args.b1k_data_root),
                "b1k_source_manifest": str(args.b1k_source_manifest),
            }), scene_authorities=authorities,
            setting_sampling_policy=args.setting_sampling_policy,
            action_sampling_policy=trusted_action_sampling_policy(args))
    return source_manifest.r2r_v16_registered_validation_context(
        [source], collection_mode=args.collection_mode,
        expected_schema_version=record_schema,
        expected_oracle_contract_version=REC.ORACLE_CONTRACT_VERSION,
        authority_sha256=collection_funnel.canonical_sha256({
            "source": source,
            "collection_mode": args.collection_mode,
            "record_schema_version": record_schema,
            "oracle_contract_version": REC.ORACLE_CONTRACT_VERSION,
            "r2r_train_episodes": str(args.r2r_train_episodes),
            "mp3d_root": str(args.mp3d_root),
        }), r2r_train_episodes=args.r2r_train_episodes,
        mp3d_root=args.mp3d_root,
        setting_sampling_policy=args.setting_sampling_policy)


def collection_sampling_provenance() -> dict:
    """Describe the sole production pose-discovery route."""
    return {
        "pose_discovery": "inline",
    }


def prune_backend_scoped_params(params: dict, backend: str) -> dict:
    """Drop the params a run of *backend* never reads.

    Run-meta params are backend-scoped on purpose: a b1k run must not record a
    GS source root it never opened. Supervisors rebuild the expected child
    policy from the same parser the child was launched with, so they have to
    prune identically -- otherwise the expected policy names keys the child
    deliberately never persisted and the readback fails on every child. Keeping
    the rule in one function is what stops the next backend from desyncing them
    again.
    """
    pruned = dict(params)
    backend = str(backend or "r2r")
    if backend != "b1k":
        pruned.pop("b1k_data_root", None)
        pruned.pop("b1k_source_manifest", None)
        pruned.pop("b1k_supervisor_contract_sha256", None)
    elif not pruned.get("b1k_supervisor_contract_sha256"):
        pruned.pop("b1k_supervisor_contract_sha256", None)
    if backend != "gs":
        pruned.pop("gs_data_root", None)
        pruned.pop("gs_source_manifest", None)
    return pruned


def collection_run_contract(
        args, scenes, heights, fovs, *, action_sampler_contract_sha256=None,
        sampling_provenance=None) -> dict:
    excluded = {
        "debug_images", "debug_outcomes_per_frame", "no_validate",
        "out", "overwrite", "resume", "code_revision", "allow_dirty_code",
        "semantic_query_workers",
    }
    params = prune_backend_scoped_params(
        {key: value for key, value in vars(args).items()
         if key not in excluded},
        getattr(args, "backend", "r2r"))
    action_file = params.get("action_file")
    if action_file:
        with open(action_file, "rb") as handle:
            params["action_file_sha256"] = hashlib.sha256(
                handle.read()).hexdigest()
    pose_exclusions = params.get("pose_exclusions")
    if pose_exclusions:
        with open(pose_exclusions, "rb") as handle:
            params["pose_exclusions_sha256"] = hashlib.sha256(
                handle.read()).hexdigest()
    if sampling_provenance is None:
        sampling_provenance = collection_sampling_provenance()
    resolved_scenes = []
    for scene in scenes:
        if isinstance(scene, SceneSpec):
            resolved_scenes.append({
                **scene.provenance(),
                "scene_path": scene.scene_path,
            })
        else:
            resolved_scenes.append(str(scene))
    payload = {
        "params": params,
        # Named in the contract so a run collected under a different setting
        # policy cannot be resumed into this one.
        "setting_sampling_policy": getattr(
            args, "setting_sampling_policy", None),
        "resolved_scenes": resolved_scenes,
        "camera_heights_m": [float(height) for height in heights],
        "fovs_deg": [[float(hfov), float(vfov)] for hfov, vfov in fovs],
        # Pin the run to one commit so --resume cannot mix code revisions and
        # so the release gate can reject artifacts built from a dirty tree.
        "code_revision": getattr(args, "code_revision", None),
        "code_dirty": bool(getattr(args, "allow_dirty_code", False)),
        "collection_funnel_schema_version": (
            collection_funnel.FUNNEL_SCHEMA_VERSION),
        "sampling_provenance": dict(sampling_provenance),
        "candidate_rejection_scope": "action",
    }
    if action_sampler_contract_sha256 is not None:
        # Binds the template vocabulary, the control bank, the budget, and
        # the depth oracle parameters the proposal queries -- a run whose
        # oracle moved samples a different distribution from an unchanged
        # rule, so the rule alone cannot identify the contract.
        payload["action_sampler_contract_sha256"] = str(
            action_sampler_contract_sha256)
    if getattr(args, "collection_mode", None) == "main":
        ordinary_limit = ordinary_actions_per_pose(args)
        payload["main_action_proposal"] = {
            "cap_per_length": int(config.MAIN_ACTION_PROPOSAL_PER_LENGTH),
            "ranking": "label-blind-stratified-shortlist.v4",
            "retention": "stable-ordinary-reserve.v1",
            "ordinary_attempts_per_pose": int(
                ordinary_limit + config.C1_NEIGHBOR_SLOTS_PER_POSE),
            "ordinary_actions_per_pose": ordinary_limit,
            "c1_queries_per_pose": int(config.C1_QUERIES_PER_POSE),
            "c1_neighbors_per_query": int(
                config.C1_NEIGHBORS_PER_QUERY),
            "dynamic_natural": "pose-depth-budget.v1",
            "a1_control_catalog_sha256":
                action_control_catalog.CATALOG_SHA256,
        }
    return json.loads(json.dumps(payload, sort_keys=True))


def ordinary_actions_per_pose(args) -> int:
    """Return the configured ordinary-action budget, defaulting to v4."""
    return int(getattr(
        args, "ordinary_actions_per_pose",
        config.ACTION_CANDIDATE_ORDINARY_PER_POSE))


def discover_collection_scenes(args) -> list[SceneSpec]:
    """Resolve one verified source-train catalog for formal collection."""
    contract = dataset_contracts.dataset_source_contract(args.backend)
    if not contract.main_collection_enabled:
        raise SceneCatalogError(
            f"collection backend is unsupported: {args.backend!r}")
    if args.backend == "r2r":
        catalog = discover_r2r_train_scenes(
            args.r2r_train_episodes, args.mp3d_root)
        empty_error = "no verified r2r train scenes selected"
    elif args.backend == "b1k":
        if not args.b1k_data_root or not args.b1k_source_manifest:
            raise SceneCatalogError(
                "B1K collection requires --b1k-data-root and "
                "--b1k-source-manifest")
        catalog = discover_b1k_train_scenes(
            args.b1k_data_root, args.b1k_source_manifest)
        empty_error = "no verified b1k train scenes selected"
    elif args.backend == "gs":
        if not args.gs_data_root or not args.gs_source_manifest:
            raise SceneCatalogError(
                "GS collection requires --gs-data-root and "
                "--gs-source-manifest")
        catalog = discover_gs_train_scenes(
            args.gs_data_root, args.gs_source_manifest,
            requested=None if args.auto_scenes else (args.scenes or []))
        empty_error = "no verified gs train scenes selected"
    else:
        raise SceneCatalogError(
            f"collection backend is unsupported: {args.backend!r}")
    try:
        partitions = scene_partitions.load()
        catalog = partitions.select_catalog(
            args.backend, catalog,
            benchmark_partition=getattr(
                args, "benchmark_partition", "train_seen"))
        args.scene_partitions_sha256 = partitions.sha256
    except ValueError as error:
        raise SceneCatalogError(str(error)) from error
    if args.auto_scenes:
        scenes = deterministic_scene_order(catalog, seed=args.seed)
    else:
        scenes = resolve_scene_subset(catalog, args.scenes or [])
    if args.max_scenes is not None:
        scenes = scenes[:int(args.max_scenes)]
    if not scenes:
        raise SceneCatalogError(empty_error)
    return scenes


def terminal_rgb_renderer(sim, frame, cache):
    """Return a cached endpoint RGB without depth unprojection or semantics."""
    def render(pose):
        key = tuple(round(float(value), 6) for value in pose)
        if key not in cache:
            cache[key] = build_terminal_rgb_observation(sim, frame, pose)
        return cache[key]

    return render


def terminal_rgb_batch_renderer(sim, frame, cache):
    """Return the fail-closed B1K simultaneous terminal renderer."""
    def render(poses):
        missing = []
        for pose in poses:
            values = tuple(float(value) for value in pose)
            key = tuple(round(value, 6) for value in values)
            if key not in cache:
                missing.append(values)
        if not missing:
            return
        batch_method = getattr(sim, "render_terminal_rgb_batch", None)
        if not callable(batch_method):
            raise RuntimeError(
                "B1K C1 requires a simultaneous batch renderer")
        rendered = list(batch_method(frame, missing))
        if len(rendered) != len(missing):
            raise RuntimeError("terminal RGB batch returned the wrong size")
        for pose, observation in zip(missing, rendered):
            cache[tuple(round(value, 6) for value in pose)] = observation

    return render
