"""Runtime orchestration for scene, pose, and rollout collection."""

from __future__ import annotations

import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import NamedTuple

import numpy as np
from PIL import Image

from pipeline import (
    abc1_record, action_control_catalog, action_proposal, action_sampling,
    actions as A,
    c1_counterfactual, collection_assets,
    collection_closeout, collection_funnel, collection_setup, config,
    io_utils, consequence as consequence_module,
    fixed_pose_actions, rollout, semantic as semantic_module, validate,
)
from pipeline import record as REC
from pipeline.geometry import Disc
from pipeline.consensus import R2R_A_STABILITY_PERTURBATIONS, oracle_consensus
from pipeline.consequence import (
    collect_a_stability_rows, deferred_a_stability_certificate,
    finalize_a_stability_certificates, judge,
)
from pipeline.frame import build_frame
from pipeline import pose_setting
from pipeline.pose_setting import sample_public_setting
from pipeline.pose_calibration import (
    derived_pose_seed as _derived_pose_seed,
    pose_has_publication_clearance as _pose_has_publication_clearance,
    pose_is_diverse as _pose_is_diverse,
    pose_sampling_radii as _pose_sampling_radii,
    prepare_pose_sampling as _prepare_pose_sampling,
)
from pipeline.record import build_record
from pipeline.scene_pool import SceneCatalogError
from pipeline.collection_cli import (
    append_compact_records, append_record_group, candidate_action_pools, close_sessions,
    frame_id as make_frame_id,
    full_geometry_publication_label, pose_group_id, prepare_records_output,
    sensor_tag, validate_records_before_spool,
    sensor_intervention,
)
from pipeline.collection_proposals import (
    pose_candidate_draws_per_attempt,
    precompute_full_geometry_candidates, record_candidate_stage,
    structured_outcome_disposition,
    pose_attempt_budget as pose_attempt_budget_for,
)
from pipeline.collection_support import (
    collection_record_schema,
    collection_run_contract, collection_sampling_provenance,
    contact_instance_witness_required,
    discover_collection_scenes,
    finish_scene_cleanup as _finish_scene_cleanup,
    open_collection_sessions as _open_backend_sessions,
    ordinary_actions_per_pose as _ordinary_actions_per_pose,
    pose_validation_context as _pose_validation_context,
    prune_backend_scoped_params,
    prepared_pose_state as _prepared_pose_state,
    run_with_backend_shutdown as _run_with_backend_shutdown,
    strict_shared_oracle_required,
    terminal_rgb_batch_renderer,
    terminal_rgb_renderer, trusted_action_sampling_policy,
)


def v16_r2r_shared_oracle_required(
        collection_mode: str, source_dataset: str) -> bool:
    """Whether this path uses the frozen fail-closed R2R ABC certificate."""
    return (
        str(source_dataset).lower() == "r2r" and
        str(collection_mode) == "main"
    )


def exact_contact_identity_required(*, c1_only: bool) -> bool:
    """Whether this route can publish A3 and therefore needs exact identity."""
    return not bool(c1_only)


attach_terminal_rgb_assets = collection_assets.attach_terminal_rgb_assets


_resolve_collection_mode_defaults = \
    collection_setup.resolve_collection_mode_defaults
active_radii_for_setting = collection_setup.active_radii_for_setting
_validate_collection_args = collection_setup.validate_collection_args


class ProposalBank(NamedTuple):
    """The run-level vocabulary: structures to place, controls to keep."""

    templates: list
    natural_pools: dict


def _namespaced_seed(seed, namespace) -> int:
    return int(hashlib.sha256(
        f"{int(seed)}:{namespace}".encode()).hexdigest()[:16], 16)


def _build_collection_action_banks(args, fovs, stats):
    """Draw the run-level vocabulary that is genuinely scene-independent.

    V3 natural programs are conditioned on each pose's initial RGB-D and must
    not be drawn here. File mode remains a literal pass-through for its frozen
    external action list.
    """
    limit = int(config.MAIN_ACTION_PROPOSAL_PER_LENGTH)
    natural_pools = {}
    if args.action_mode == "file":
        natural_pools = candidate_action_pools(
            args, np.random.default_rng(
                _namespaced_seed(args.seed, "explicit-action-file")),
            stats, min(hfov for hfov, _vfov in fovs) / 2.0)
        natural_pools = {
            int(length): list(candidates)
            for length, candidates in natural_pools.items()
        }
    templates = (
        [] if args.action_mode == "file" else
        action_proposal.build_template_bank(
            np.random.default_rng(
                _namespaced_seed(args.seed, "template-bank")),
            lengths=tuple(args.lengths), per_length=limit))
    bank = ProposalBank(templates=templates, natural_pools=natural_pools)
    contract_hash = action_proposal.action_sampler_contract_sha256(
        templates,
        [(tag, actions) for candidates in natural_pools.values()
         for tag, actions in candidates],
        pairs_per_length=int(args.proposal_pairs_per_length),
        natural_per_length=int(args.proposal_natural_per_length),
        ordinary_actions_per_pose=_ordinary_actions_per_pose(args))
    return bank, contract_hash


def _variant_of(provenance, tag) -> str:
    """Which proposal family a program came from.

    File-driven runs carry no provenance, so their programs report as
    ``file`` rather than being silently counted as depth-conditioned.
    """
    entry = (provenance or {}).get(tag)
    return str(entry.get("variant")) if entry else "file"


def _materialize_pose_action_bank(
        bank, base_frame, radius, *, args, stats, skipped,
        scene_id, pose_index, include_controls):
    """Turn the run-level vocabulary into this pose's own candidate bank.

    A file-driven run has nothing to condition on -- its programs came from
    disk -- so it passes its pools through untouched and carries no proposal
    provenance.

    The returned manifest authenticates every action offered to the oracle.
    """
    if args.action_mode == "file":
        return dict(bank.natural_pools), {}, None, ()
    templates = list(bank.templates)
    proxy = action_proposal.FrameDepthProxy(base_frame, radius)
    half_fov_deg = float(base_frame.sensor.hfov_deg) / 2.0
    posed = action_proposal.build_pose_bank(
        templates, proxy,
        half_fov_deg=half_fov_deg,
        pairs_per_length=int(args.proposal_pairs_per_length),
        rejections=skipped, stats=stats)

    def overlay(candidates):
        """Replace identical lower-priority programs with one source atom."""
        incoming = [candidate for rows in candidates.values()
                    for candidate in rows]
        tags = [candidate.tag for candidate in incoming]
        if len(tags) != len(set(tags)):
            raise ValueError("duplicate action tag within proposal source")
        incoming_tags = set(tags)
        for length, rows in list(posed.items()):
            retained = [candidate for candidate in rows
                        if candidate.tag not in incoming_tags]
            stats["proposal_source_duplicate_overridden"] += \
                len(rows) - len(retained)
            posed[length] = retained
        for candidate in incoming:
            posed.setdefault(int(candidate.length), []).append(candidate)

    dynamic = action_proposal.build_dynamic_natural_bank(
        np.random.default_rng(_derived_pose_seed(
            args.seed, scene_id, pose_index, "natural-dynamic-v1")),
        proxy,
        half_fov_deg=half_fov_deg,
        lengths=tuple(args.lengths),
        per_length=int(args.proposal_natural_per_length),
        rejections=skipped, stats=stats)
    # Natural programs are sampled without consulting a label, whereas a
    # paired program changes its target leg to manufacture a label.  If both
    # arms draw the same physical action, keep exactly one natural member so
    # the A1 compiler does not discard a healthy natural example merely
    # because the paired arm happened to duplicate it.
    overlay(dynamic)
    controls = {}
    if include_controls:
        controls = action_proposal.build_control_bank(
            action_control_catalog.anchors_for_pose(
                dataset=getattr(args, "backend", "r2r"),
                scene_id=scene_id, pose_index=pose_index),
            proxy, half_fov_deg=half_fov_deg,
            rejections=skipped, stats=stats)
        # A control's intervention is its exact action.  In the unlikely event
        # another arm drew the same bytes, retain one bank member and assign it
        # to the control rather than publishing two contradictory provenances.
        control_tags = {
            candidate.tag
            for candidates in controls.values() for candidate in candidates
        }
        overlay(controls)
    else:
        control_tags = set()
    for length, candidates in posed.items():
        posed[length] = sorted(candidates, key=lambda c: c.order_key)
    pools = {
        length: [(candidate.tag, list(candidate.actions))
                 for candidate in candidates]
        for length, candidates in posed.items()
    }
    provenance = {
        candidate.tag: candidate.provenance
        for candidates in posed.values() for candidate in candidates
    }
    return (
        pools,
        provenance,
        action_proposal.action_bank_manifest(posed),
        tuple(sorted(control_tags)),
    )


def _precheck_action_candidate(
        action_tag, actions, full_by_radius, *, variants, active_radii,
        required_siblings, proposal_provenance, stats, skipped,
        group_labels, precheck_cache):
    """Apply the unchanged full/depth/consensus gate to one action program."""
    length = len(actions)
    full_label = full_geometry_publication_label(actions, full_by_radius)
    if full_label is None:
        collision_states = {
            value.get("collision") for value in full_by_radius.values()}
        if collision_states == {True}:
            skipped["collision_publication_margin"] += 1
            record_candidate_stage(
                stats, "full_reject", "collision", length,
                reason="publication_margin")
        elif collision_states == {False}:
            skipped["safe_publication_margin"] += 1
            record_candidate_stage(
                stats, "full_reject", "safe", length,
                reason="publication_margin")
        else:
            source = next((
                str(value.get("collision_source"))
                for value in full_by_radius.values()
                if value.get("collision") is None and
                value.get("collision_source")
            ), "unavailable")
            skipped[f"full_geometry_excluded.{source}"] += 1
            record_candidate_stage(
                stats, "full_reject", "unavailable", length,
                reason=source)
        return None
    record_candidate_stage(stats, "full_ready", full_label, length)
    depth_checks = []
    for sim, frame in variants:
        for radius in active_radii:
            depth_physical = rollout.view_collision_rollout(
                frame, actions, radius)
            coverage = rollout.corridor_coverage(
                frame, actions, radius,
                max_arc_m=rollout.realized_corridor_arc_m(
                    full_by_radius[float(radius)]))
            stats["depth_prechecks"] += 1
            if coverage < config.EVIDENCE_COVERAGE_MIN:
                skipped["insufficient_depth_coverage"] += 1
                record_candidate_stage(
                    stats, "depth_reject", full_label, length,
                    reason="insufficient_depth_coverage")
                return None
            depth_checks.append((
                frame.frame_id, float(radius), depth_physical, coverage))
    by_radius = collections.defaultdict(set)
    for _frame_id, radius, depth, _coverage in depth_checks:
        by_radius[float(radius)].add(bool(depth["collision"]))
    if any(len(values) != 1 for values in by_radius.values()):
        skipped["sensor_label_mixed"] += 1
        record_candidate_stage(
            stats, "sensor_reject", full_label, length,
            reason="sensor_label_mixed")
        return None
    radius_collisions = {
        float(radius): bool(full_by_radius[float(radius)]["collision"])
        for radius in active_radii}
    flags = []
    consensus_checks = {}
    consensus_rejected = False
    for frame_id, radius, depth_physical, coverage in depth_checks:
        precheck = oracle_consensus(
            full_by_radius[radius], depth_physical, coverage)
        stats["prechecked_outcomes"] += 1
        flags.append(bool(precheck["accepted"]))
        consensus_check = {
            "depth_physical": depth_physical,
            "coverage": float(coverage),
            "consensus": precheck,
        }
        path_trace = full_by_radius[radius]
        if isinstance(path_trace, rollout.PhysicalPathTrace):
            consensus_check["physical_path_trace"] = path_trace
        consensus_checks[(frame_id, radius)] = consensus_check
        if not precheck["accepted"]:
            if not consensus_rejected:
                skipped[precheck["reason"]] += 1
                record_candidate_stage(
                    stats, "consensus_reject", full_label, length,
                    reason=precheck["reason"])
            consensus_rejected = True
            break
    if consensus_rejected:
        record_candidate_stage(
            stats, "group_reject", full_label, length,
            reason="incomplete_or_label_mismatch")
        return None
    label = action_sampling.classify_action_group(
        flags, radius_collisions, required_siblings)
    if label is None or label != full_label:
        record_candidate_stage(
            stats, "group_reject", full_label, length,
            reason="incomplete_or_label_mismatch")
        return None
    group_labels[action_tag] = label
    stats[
        "proposal_accepted."
        f"{_variant_of(proposal_provenance, action_tag)}.L{length}"
    ] += 1
    stats[f"consensus_{label}_L{length}"] += 1
    record_candidate_stage(stats, "accepted", label, length)
    precheck_cache[action_tag] = consensus_checks
    return label


def _precheck_action_pool(
        pools, *, base_sim, base_frame, active_radii, variants,
        required_siblings, proposal_provenance, stats, skipped,
        group_labels, precheck_cache) -> None:
    if not pools:
        return
    for length, candidates in pools.items():
        stats[f"shortlist_entered_L{length}"] += len(candidates)
    full_cache = precompute_full_geometry_candidates(
        base_sim, base_frame, pools, active_radii, stats)
    for length in sorted(pools):
        for action_tag, actions in pools[length]:
            _precheck_action_candidate(
                action_tag, actions, full_cache[action_tag],
                variants=variants, active_radii=active_radii,
                required_siblings=required_siblings,
                proposal_provenance=proposal_provenance,
                stats=stats, skipped=skipped,
                group_labels=group_labels,
                precheck_cache=precheck_cache)



def _make_structured_spec_evaluator(
        *, args, variants,
        group_labels, precheck_cache, stats, skipped,
        pending_a_certificates):
    def evaluate_spec(spec):
        action_tag, actions = spec["action_tag"], spec["actions"]
        radii = spec["radii"]
        candidate_outcomes = collections.defaultdict(list)
        length = len(actions)
        physical_by_radius = {}
        base_sim, base_frame = variants[0]
        for radius in radii:
            base_sim.recompute_navmesh(
                radius, height=config.GROUND_ORACLE_HEIGHT_M)
            nav = base_sim.nav(base_frame.position, base_frame.yaw_rad)
            radius_key = round(float(radius), 6)
            physical = physical_by_radius.get(radius_key)
            if physical is None:
                cached_precheck = precheck_cache.get(action_tag, {}).get(
                    (base_frame.frame_id, radius_key), {})
                physical = rollout.physical_rollout(
                    nav, actions,
                    path_trace=cached_precheck.get("physical_path_trace"))
                physical_by_radius[radius_key] = physical
                stats["physical_rollouts"] += 1
        # Judge every sibling before paying for the seven SE(2) stability
        # rerollouts.  A spec is atomic across variant x radius, so one failed
        # sibling invalidates every certificate we could have built here.
        judged = []
        for sim, frame in variants:
            for radius in radii:
                strict_shared_oracle = strict_shared_oracle_required(
                    args.collection_mode,
                    getattr(sim, "source_dataset", ""),
                )
                require_instance_witness = contact_instance_witness_required(
                    args.collection_mode,
                    getattr(sim, "source_dataset", ""),
                    semantic_certified=getattr(
                        getattr(sim, "semantic_index", None),
                        "semantic_certified", None),
                )
                sim.recompute_navmesh(
                    radius, height=config.GROUND_ORACLE_HEIGHT_M)
                nav = sim.nav(frame.position, frame.yaw_rad)
                radius_key = round(float(radius), 6)
                outcome = judge(
                    frame,
                    Disc(radius_m=radius),
                    actions,
                    nav=nav,
                    cached_physical=physical_by_radius[radius_key],
                    cached_depth_physical=(
                        precheck_cache.get(action_tag, {}).get(
                            (frame.frame_id, radius_key), {}).get(
                                "depth_physical")),
                    cached_corridor_coverage=(
                        precheck_cache.get(action_tag, {}).get(
                            (frame.frame_id, radius_key), {}).get("coverage")),
                    cached_oracle_consensus=(
                        precheck_cache.get(action_tag, {}).get(
                            (frame.frame_id, radius_key), {}).get("consensus")),
                    require_contact_instance_witness=
                        require_instance_witness,
                )
                stats["evaluated_outcomes"] += 1
                disposition = structured_outcome_disposition(outcome)
                if disposition == "reject_spec":
                    # Excluded geometry and a genuine dual-oracle disagreement
                    # both land here. Keeping them apart is what makes a
                    # before/after funnel readable.
                    reject_source = (
                        (outcome.get("physical") or {}).get("collision_source")
                        or "oracle_disagreement")
                    skipped[f"structured_reject_{reject_source}"] += 1
                    skipped["structured_oracle_disagreement"] += 1
                    return None
                judged.append((
                    sim, frame, radius, outcome, strict_shared_oracle,
                    require_instance_witness))

        spec_pending_a_certificates = []
        for (sim, frame, radius, outcome, strict_shared_oracle,
             require_instance_witness) in judged:
            if strict_shared_oracle and getattr(args, "oracle_evaluation", "perturbed") == "nominal":
                outcome["shared_oracle_stability"] = fixed_pose_actions.nominal_certificate(outcome)
                stats["nominal_certificates"] += 1
            elif strict_shared_oracle:
                # Exact contact identity feeds only A3. Frozen A1/A2 legs and
                # C1-only records can never publish A3, so re-querying full
                # geometry for their seven perturbations buys no GT element
                # and is skipped fail-closed.
                sim.recompute_navmesh(
                    radius, height=config.GROUND_ORACLE_HEIGHT_M)
                rows = collect_a_stability_rows(
                    sim, frame, Disc(radius_m=radius), actions, outcome,
                    require_contact_instance_witness=
                        require_instance_witness)
                spec_pending_a_certificates.append(
                    deferred_a_stability_certificate(
                        frame, actions, rows, outcome,
                        exact_contact_identity=
                            require_instance_witness and
                            exact_contact_identity_required(
                                c1_only=False),
                        require_contact_instance_witness=
                            require_instance_witness))
                stats["a_stability_certificates"] += 1
                stats["a_stability_rerollouts"] += (
                    len(R2R_A_STABILITY_PERTURBATIONS) - 1)

            radius_tag = f"b{int(round(radius * 100)):03d}"
            outcome["seq_len"] = int(length)
            outcome["action_group_id"] = action_tag
            outcome["action_group_label"] = group_labels[action_tag]
            outcome["outcome_id"] = f"{radius_tag}-{action_tag}"
            candidate_outcomes[frame.frame_id].append(outcome)
        pending_a_certificates.extend(spec_pending_a_certificates)
        return candidate_outcomes or None
    return evaluate_spec

_open_collection_sessions = _open_backend_sessions
emit_backend_ready = collection_closeout.emit_backend_ready


def _finalize_collection_run(
        *, args, scenes, started, stats, skipped, records_path, funnel,
        existing_records, completed_groups, run_contract,
        capacity_stop=None):
    run_contract_sha256 = collection_funnel.canonical_sha256(run_contract)
    finalization = collection_closeout.begin_finalization(
        args.out, run_contract_sha256=run_contract_sha256,
        capacity_stop=capacity_stop)

    def seal_finalization(
            *, status: str, source_validation: str,
            record_count: int, records_sha256: str) -> dict:
        return collection_closeout.seal_finalization(
            args.out, finalization, records_path=records_path,
            run_meta_path=Path(args.out) / "run_meta.json",
            funnel_path=funnel.path,
            status=status,
            source_validation=source_validation,
            record_count=record_count,
            records_sha256=records_sha256)

    dataset = str(getattr(args, "backend", "r2r"))
    official_splits = sorted({scene.official_split for scene in scenes})
    source_split = str(getattr(
        args, "source_split",
        official_splits[0] if len(official_splits) == 1 else "train"))
    if official_splits and official_splits != [source_split]:
        raise ValueError(
            "run source split disagrees with resolved scene provenance")
    benchmark_partition = str(getattr(
        args, "benchmark_partition", "train_seen"))
    record_schema = abc1_record.SCHEMA_VERSION
    params = prune_backend_scoped_params(
        vars(args), getattr(args, "backend", "r2r"))
    metadata = {
        "scenes": len(scenes), "params": params, "stats": dict(stats),
        "skipped": dict(skipped), "seconds": round(time.time() - started, 1),
        "record_schema_version": record_schema,
        "source_split": source_split,
        "benchmark_partition": benchmark_partition,
        "surface_relation_schema": "surface-point-relation.v4",
        "oracle_contract_version": REC.ORACLE_CONTRACT_VERSION,
        "supported_tasks": {
            dataset: list(abc1_record.supported_tasks(dataset))
            for dataset in sorted({scene.source_dataset for scene in scenes})
        },
        "action_protocol": {
            "pattern": "alternating-forward-turn",
            "starts_with": ["forward", "turn"],
            "initial_turns_deg": sorted(abc1_record.INITIAL_TURNS_DEG),
        },
        "run_contract_sha256": run_contract_sha256,
        "code_revision": args.code_revision,
        "code_dirty": bool(args.allow_dirty_code),
        "sampling_provenance": dict(
            run_contract["sampling_provenance"]),
        "source_catalog": {
            "datasets": sorted({scene.source_dataset for scene in scenes}),
            "official_splits": official_splits,
            "source_split": source_split,
            "benchmark_partition": benchmark_partition,
            "scene_ids": [scene.scene_id for scene in scenes],
            "manifest_sha256": sorted({
                scene.provenance_sha256 for scene in scenes}),
        },
        "resume": {"existing_records": existing_records,
                   "completed_groups": len(completed_groups)},
    }
    if capacity_stop is not None:
        metadata["capacity_stop"] = dict(capacity_stop)
    for key in (
            "resolved_scenes",
            "candidate_rejection_scope",
            "action_sampler_contract_sha256",
            "main_action_proposal"):
        if key in run_contract:
            metadata[key] = json.loads(json.dumps(run_contract[key]))
    # Atomic and durable like the funnel: a torn run_meta.json after a crash
    # would desynchronise the two views of the same counters.
    records_digest = hashlib.sha256()
    record_count = 0
    with Path(records_path).open("rb") as stream:
        for line in stream:
            records_digest.update(line)
            if line.strip():
                record_count += 1
    records_sha256 = records_digest.hexdigest()
    metadata.update({
        "dataset": dataset,
        "record_count": record_count,
        "records_sha256": records_sha256,
    })
    io_utils.atomic_write_json(
        os.path.join(args.out, "run_meta.json"), metadata,
        sort_keys=False, durable=True)
    print("stats:", dict(stats), "skipped:", dict(skipped),
          f"collide_rate={stats['collided'] / max(stats['outcomes'], 1):.2f}",
          f"time={metadata['seconds']}s")
    funnel.complete()
    sealed = seal_finalization(
        status="completed", source_validation="passed",
        record_count=record_count, records_sha256=records_sha256)
    io_utils.atomic_write_json(Path(args.out) / "manifest.json", {
        "schema": "egoconseq.abc1-record-catalog.v1",
        "split": (
            "train" if benchmark_partition == "train_seen" else
            benchmark_partition),
        "source_split": source_split,
        "benchmark_partition": benchmark_partition,
        "record_count": record_count,
        "datasets": [{
            "dataset": dataset,
            "records_path": "records.jsonl",
            "records_sha256": records_sha256,
            "run_meta_sha256": sealed["run_meta_sha256"],
            "record_count": record_count,
        }],
    }, allow_nan=False, durable=True)
    return 0

def _resolve_pose_calibration(
        *, args, sampler, sample_radii, scene_id, pose_index,
        pose_tries_per_attempt, pose_exclusions, stats, skipped):
    prior_poses = pose_exclusions.get(scene_id, [])
    pose_rng = _prepare_pose_sampling(
        sampler, sample_radii, base_seed=args.seed,
        scene_id=scene_id, pose_index=pose_index)
    return sampler.sample_random_pose(
        pose_rng, sample_radii,
        max_tries=pose_tries_per_attempt,
        on_reject=lambda reason: skipped.__setitem__(
            f"pose_{reason}", skipped[f"pose_{reason}"] + 1),
        pose_valid=lambda position, yaw: (
            _pose_has_publication_clearance(sampler, position, yaw)
            and _pose_is_diverse(position, yaw, prior_poses)
        ))



def _prepare_pose_candidates(
        *, args, sampler, sample_radii, scene_id, pose_index,
        pose_tries_per_attempt, pose_exclusions, sessions, fovs, heights,
        scene, proposal_bank, stats, skipped, intervention_group_id):
    calibration = _resolve_pose_calibration(
        args=args, sampler=sampler, sample_radii=sample_radii,
        scene_id=scene_id, pose_index=pose_index,
        pose_tries_per_attempt=pose_tries_per_attempt,
        pose_exclusions=pose_exclusions, stats=stats, skipped=skipped)
    if calibration is None:
        skipped["sample_fail"] += 1
        return None
    position, yaw = calibration.position, calibration.yaw_rad
    calibration_profile = config.calibration_profile()
    # Pose discovery above certified the start with the full radius grid; the
    # published body is drawn only now, so every body shares one pose
    # population instead of each carrying its own selection bias.
    setting_policy = getattr(
        args, "setting_sampling_policy",
        pose_setting.SETTING_SAMPLING_POLICY)
    setting = sample_public_setting(
        seed=args.seed, scene_id=scene_id, pose_index=pose_index,
        heights=heights, fovs=fovs, radii=args.radii)
    sessions = [sessions[setting.fov_index]]
    fovs = [fovs[setting.fov_index]]
    heights = [setting.nominal_camera_height_m]
    active_radii = active_radii_for_setting(
        setting, setting_policy)
    variants = []
    for sim, (hfov, vfov) in zip(sessions, fovs):
        for height in heights:
            tag = sensor_tag(height, hfov, vfov)
            frame_id = make_frame_id(
                scene_id, pose_index, args.collection_shard_id, tag)
            # The calibration render IS this sibling when the
            # profiles coincide, so reuse it instead of drawing the
            # same view twice.
            reuse = (
                calibration.reference_observation
                if (height, hfov, vfov) == calibration_profile
                else None)
            frame = build_frame(
                sim, position, yaw, frame_id=frame_id,
                scene_id=scene_id, scene_glb=scene.scene_path,
                floor_plane=calibration.canonical_plane,
                cam_h=height, hfov=hfov, vfov=vfov,
                rendered=reuse,
            )
            variants.append((sim, frame))
    base_frame = variants[0][1]
    group_labels = {}
    precheck_cache = {}
    required_siblings = len(variants) * len(active_radii)
    base_sim, base_frame = variants[0]
    # The pose's own depth materialises its action bank.
    pools, proposal_provenance, bank_manifest, control_tags = \
        _materialize_pose_action_bank(
        proposal_bank, base_frame, active_radii[0], args=args,
        stats=stats, skipped=skipped,
        scene_id=scene_id, pose_index=pose_index,
        include_controls=True)
    if not pools:
        skipped["proposal_bank_empty"] += 1
        return None
    # Trim before certifying, not after.  Everything below this line costs a
    # publication label plus (variants x radii) rollouts per candidate, and it
    # used to run over the whole bank -- hundreds of programs -- so that a
    # later hash cut could keep a couple of dozen.  ``group_labels`` is still
    # empty here, which is exactly why the shortlist cannot read a label.
    # The queries themselves come from this label-blind ordinary shortlist;
    # only after certification says they completed clear do their additional
    # neighbours spend any of the bounded second-pass allocation.
    ordinary_limit = _ordinary_actions_per_pose(args)
    first_pass_budget = (
        ordinary_limit + config.C1_NEIGHBOR_SLOTS_PER_POSE)
    pools = action_sampling.shortlist_action_bank(
        pools, proposal_provenance,
        pose_seed=_derived_pose_seed(
            args.seed, scene_id, pose_index, "action-bank-shortlist"),
        budget=first_pass_budget,
        forced_tags=control_tags,
        stats=stats)
    _precheck_action_pool(
        pools, base_sim=base_sim, base_frame=base_frame,
        active_radii=active_radii, variants=variants,
        required_siblings=required_siblings,
        proposal_provenance=proposal_provenance,
        stats=stats, skipped=skipped,
        group_labels=group_labels, precheck_cache=precheck_cache)
    c1_families = ()
    if bank_manifest is not None:
        c1_families = c1_counterfactual.reserve_pose_slots(
                pools, proposal_provenance, bank_manifest,
                pose_seed=_derived_pose_seed(
                    args.seed, scene_id, pose_index,
                    "c1-counterfactual-query"),
                stats=stats, variant_of=_variant_of,
                group_labels=group_labels)
    action_bank_size = sum(len(candidates) for candidates in pools.values())
    certification_limit = (
        ordinary_limit + 2 * config.C1_NEIGHBOR_SLOTS_PER_POSE)
    if action_bank_size > certification_limit:
        raise ValueError("C1 reservation exceeds per-pose action budget")
    second_pass_tags = {
        tag for family in c1_families for tag in family.neighbor_tags}
    second_pass_pools = {
        length: [
            (tag, actions) for tag, actions in candidates
            if tag in second_pass_tags and tag not in group_labels
        ]
        for length, candidates in pools.items()
    }
    second_pass_pools = {
        length: candidates
        for length, candidates in second_pass_pools.items()
        if candidates
    }
    if second_pass_pools:
        _precheck_action_pool(
            second_pass_pools, base_sim=base_sim, base_frame=base_frame,
            active_radii=active_radii, variants=variants,
            required_siblings=required_siblings,
            proposal_provenance=proposal_provenance,
            stats=stats, skipped=skipped,
            group_labels=group_labels, precheck_cache=precheck_cache)
    return _prepared_pose_state(
        variants=variants, pools=pools, group_labels=group_labels,
        precheck_cache=precheck_cache,
        proposal_provenance=proposal_provenance,
        required_siblings=required_siblings,
        calibration=calibration, position=position, yaw=yaw,
        base_frame=base_frame,
        active_radii=active_radii, setting=setting,
        bank_manifest=bank_manifest,
        intervention_group_id=intervention_group_id,
        scene=scene, scene_id=scene_id,
        c1_families=c1_families,
        # ``candidate_budget`` is the publication budget, not the wider
        # stability reserve that is consumed before a record exists.
        shortlist_size=min(
            action_bank_size,
            ordinary_limit + config.C1_NEIGHBOR_SLOTS_PER_POSE))

def _select_pose_candidates(
        state: dict, *, args, scene_id, pose_index,
        stats, skipped):
    pools = state["pools"]
    group_labels = state["group_labels"]
    proposal_provenance = state.get("proposal_provenance") or {}
    accepted_labels = {"safe", "collision"}
    for length in sorted(pools):
        available = collections.Counter(
            group_labels.get(tag) for tag, _actions in pools[length])
        if not any(available[label] for label in accepted_labels):
            stats[f"formal_candidate_shortfall_L{length}"] += 1
    selection_seed = _derived_pose_seed(
        args.seed, scene_id, pose_index,
        "natural-action-candidates")
    selected = action_sampling.retain_certified_action_groups(
        pools, group_labels,
        maximum=(_ordinary_actions_per_pose(args) +
                 2 * config.C1_NEIGHBOR_SLOTS_PER_POSE))
    if len(selected or []) < config.ACTION_CANDIDATE_MIN_PER_POSE:
        selected = []
        skipped["natural_action_candidate_shortfall"] += 1
    if not selected:
        skipped["natural_action_set_infeasible"] += 1
        return None
    selected_ids = [tag for tag, _actions in selected]
    for tag, actions in selected:
        stats[
            f"proposal_selected.{_variant_of(proposal_provenance, tag)}"
            f".L{len(actions)}"
        ] += 1
    result = dict(state)
    result.update({
        "selected": selected,
        "selected_ids": selected_ids,
        "selection_seed": selection_seed,
    })
    return result

def _evaluate_pose_candidates(state: dict, *, args, stats, skipped):
    variants = state["variants"]
    group_labels = state["group_labels"]
    precheck_cache = state["precheck_cache"]
    pools = state["pools"]
    active_radii = state["active_radii"]
    proposal_provenance = state.get("proposal_provenance") or {}
    selection_seed = state["selection_seed"]
    selected_by_tag = dict(state["selected"])
    families = tuple(state.get("c1_families") or ())
    query_tags = tuple(dict.fromkeys(
        family.query_tag for family in families
        if family.query_tag in selected_by_tag))
    ordinary_limit = _ordinary_actions_per_pose(args)
    if len(query_tags) > ordinary_limit:
        raise ValueError("C1 queries exceed the ordinary publication cap")

    order = action_sampling.stratified_action_order(
        pools, proposal_provenance, pose_seed=selection_seed,
        forced_tags=query_tags)
    ordinary_candidates = [
        (tag, selected_by_tag[tag]) for tag in order
        if tag in selected_by_tag and
        ((proposal_provenance.get(tag) or {}).get("variant") !=
         c1_counterfactual.VARIANT)
    ]
    queries = [
        (tag, selected_by_tag[tag]) for tag in query_tags]
    query_set = set(query_tags)
    ordinary_candidates = [
        value for value in ordinary_candidates if value[0] not in query_set]

    render_caches = collections.defaultdict(dict)
    pending_a_certificates = []
    evaluate_spec = _make_structured_spec_evaluator(
        args=args,
        variants=variants,
        group_labels=group_labels,
        precheck_cache=precheck_cache,
        stats=stats,
        skipped=skipped,
        pending_a_certificates=pending_a_certificates,
    )

    nominal_survivors = 0

    def evaluate_stability_batch(candidates):
        nonlocal nominal_survivors
        if not candidates:
            return []
        evaluated = []
        for tag, actions in candidates:
            group = evaluate_spec({
                "action_tag": tag,
                "actions": actions,
                "radii": list(active_radii),
                "type": "main",
            })
            if group is None:
                skipped["structured_main_group_dropped"] += 1
                continue
            nominal_survivors += 1
            evaluated.append((tag, actions, group))
        if getattr(args, "oracle_evaluation", "perturbed") == "nominal":
            return evaluated  # evaluate_spec already applied the dual-oracle verdict.
        finalize_a_stability_certificates(pending_a_certificates)
        pending_a_certificates.clear()
        stable = []
        for tag, actions, group in evaluated:
            siblings = [
                outcome for values in group.values() for outcome in values]
            stable_flags = [
                ((outcome.get("shared_oracle_stability") or {}).get(
                    "summary") or {}).get(
                        "collision_label_stable") is True
                for outcome in siblings]
            if siblings and all(stable_flags):
                stable.append((tag, actions, group))
                continue
            skipped["stability_group_rejected"] += 1
            skipped["stability_sibling_collateral_outcomes"] += sum(
                stable_flags)
        return stable

    stable_ordinary = evaluate_stability_batch(queries)
    offset = 0
    while len(stable_ordinary) < ordinary_limit and \
            offset < len(ordinary_candidates):
        remaining = ordinary_limit - len(stable_ordinary)
        batch = ordinary_candidates[offset:offset + remaining]
        offset += len(batch)
        stable_ordinary.extend(evaluate_stability_batch(batch))

    if not stable_ordinary:
        if nominal_survivors == 0:
            skipped["structured_main_survivor_shortfall"] += 1
        skipped["stability_survivor_shortfall"] += 1
        skipped["stability_ordinary_survivor_shortfall"] += 1
        return None

    stable_query_tags = {
        tag for tag, _actions, _group in stable_ordinary
        if tag in query_set}
    c1_families = tuple(
        family for family in families
        if family.query_tag in stable_query_tags)
    c1_candidates = []
    claimed = set()
    for family in c1_families:
        for tag in family.neighbor_tags:
            if (tag in claimed or tag not in selected_by_tag or
                    (proposal_provenance.get(tag) or {}).get("variant") !=
                    c1_counterfactual.VARIANT):
                continue
            claimed.add(tag)
            c1_candidates.append((tag, selected_by_tag[tag]))
    stable_c1 = evaluate_stability_batch(
        c1_candidates[:config.C1_NEIGHBOR_SLOTS_PER_POSE])

    selected_groups = stable_ordinary + stable_c1
    selected = [(tag, actions) for tag, actions, _group in selected_groups]
    accepted_outcomes = {
        tag: group for tag, _actions, group in selected_groups}
    selected_ids = [tag for tag, _actions in selected]
    if len(selected) > ordinary_limit + config.C1_NEIGHBOR_SLOTS_PER_POSE:
        raise ValueError("stable actions exceed the publication budget")
    result = dict(state)
    result.update({
        "selected": selected,
        "selected_ids": selected_ids,
        "accepted_outcomes": accepted_outcomes,
        "render_caches": render_caches,
        "c1_families": c1_families,
    })
    return result

def _persist_pose_group(
        state: dict, *, args, image_dir, array_dir, records_path,
        scene_index, pose_index, sampler_contract_hash, intervention_type,
        changed_fields, funnel, stats):
    asset_root = (str(Path(args.append_records).parent)
                  if getattr(args, "append_records", None) else args.out)
    names = (
        "selected_ids group_labels selected accepted_outcomes "
        "variants render_caches pools "
        "required_siblings calibration position yaw intervention_group_id "
        "scene scene_id active_radii setting "
        "proposal_provenance bank_manifest"
    ).split()
    (
        selected_ids, group_labels, selected, accepted_outcomes,
        variants, render_caches, pools,
        required_siblings, calibration, position, yaw, intervention_group_id,
        scene, scene_id, active_radii, setting,
        proposal_provenance, bank_manifest,
    ) = map(state.__getitem__, names)
    selected_labels = {
        tag: group_labels[tag] for tag in selected_ids
    }
    matched_units = action_sampling.matched_action_units([
        {
            "group_id": tag,
            "label": group_labels[tag],
            "actions": actions,
        }
        for tag, actions in selected
    ])
    matched_safe_units = action_sampling.matched_same_label_units([
        {
            "group_id": tag,
            "label": group_labels[tag],
            "actions": actions,
        }
        for tag, actions in selected
    ], label="safe")
    outcomes = {frame.frame_id: [] for _sim, frame in variants}
    for action_tag in selected_ids:
        for frame_id, values in accepted_outcomes[action_tag].items():
            outcomes[frame_id].extend(values)
    if args.debug_images:
        from pipeline import viz
        debug_dir = os.path.join(args.out, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        for _sim, frame in variants:
            Image.fromarray(viz.object_overlay(frame)).save(
                os.path.join(debug_dir, f"overlay_{frame.frame_id}.png"))
            for outcome in outcomes[frame.frame_id][:args.debug_outcomes_per_frame]:
                contact = outcome["physical"].get("contact")
                target = (contact.get("instance_id") if contact and
                          not contact.get("unattributed") else None)
                viz.save_evidence(
                    frame, outcome,
                    os.path.join(debug_dir,
                                 f"ev_{frame.frame_id}-{outcome['outcome_id']}.png"),
                    target_instance=target,
                )
    source_provenance = scene.provenance()
    record_schema = collection_record_schema(args)
    strict_collection_contract = (
        REC.collection_contract(
            source_provenance, args.collection_mode,
            record_schema_version=record_schema)
        if (record_schema != REC.V18_SCHEMA_VERSION and
            strict_shared_oracle_required(
                args.collection_mode,
                source_provenance.get("source_dataset", "")))
        else None
    )
    rich_records = []
    for _sim, frame in variants:
        current_image_path = os.path.join("img", frame.frame_id + ".png")
        Image.fromarray(frame.rgb).save(
            os.path.join(asset_root, current_image_path))
        if (strict_collection_contract is not None or
                (record_schema == REC.V18_SCHEMA_VERSION and
                 source_provenance.get("source_dataset") == "gs" and
                 args.collection_mode == "main")):
            c1_families = state.get("c1_families", ())
            is_b1k = source_provenance.get("source_dataset") == "b1k"
            c1_outcome_ids = \
                collection_assets.counterfactual_terminal_outcome_ids(
                    accepted_outcomes, frame_id=frame.frame_id,
                    families=c1_families)
            authorized_outcome_ids = set() if is_b1k else c1_outcome_ids
            if is_b1k and c1_families:
                batch_count = \
                    collection_assets.preload_counterfactual_terminal_rgb(
                        accepted_outcomes, frame_id=frame.frame_id,
                        families=c1_families,
                        render_cache=render_caches[frame.frame_id],
                        terminal_batch_renderer=terminal_rgb_batch_renderer(
                            _sim, frame, render_caches[frame.frame_id]),
                        authorized_outcome_ids=authorized_outcome_ids)
                stats["c1_terminal_rgb_batch_members"] += batch_count
            terminal_counts = attach_terminal_rgb_assets(
                asset_root, frame, outcomes[frame.frame_id],
                render_caches[frame.frame_id], source=source_provenance,
                collection_contract=strict_collection_contract,
                terminal_renderer=(None if is_b1k else terminal_rgb_renderer(
                    _sim, frame, render_caches[frame.frame_id])),
                eligible_outcome_ids=authorized_outcome_ids,
                render_transaction=(
                    REC.B1K_C1_RENDER_MODE if is_b1k else None))
            stats["terminal_rgb_assets"] += terminal_counts["materialized"]
            stats["terminal_rgb_assets_withheld"] += terminal_counts["withheld"]
            if c1_families:
                c1_counterfactual.tally_neighbor_outcome_failures(
                    stats, c1_families, accepted_outcomes,
                    frame_id=frame.frame_id)
        if args.save_arrays:
            np.save(os.path.join(array_dir, frame.frame_id + "_depth.npy"),
                    frame.depth)
        published_type, published_changed = pose_setting.intervention_descriptor(
            setting, intervention_type, changed_fields,
            setting_policy=args.setting_sampling_policy)
        intervention = {
            "group_id": intervention_group_id,
            "type": published_type,
            "changed_fields": published_changed,
            "collection_mode": args.collection_mode,
        }
        selection = {
            "policy": trusted_action_sampling_policy(args),
            "action_group_ids": selected_ids,
            "action_group_labels": selected_labels,
            # The set the published actions were drawn from.  On the main path
            # that is the shortlist, decided before any label existed, so the
            # number says what it means.
            "candidate_budget": max(
                len(selected_ids), state["shortlist_size"]),
            "observed_lengths": sorted({
                len(actions) for _tag, actions in selected}),
            "observed_label_counts": dict(collections.Counter(
                selected_labels.values())),
            "matched_action_units": matched_units,
            "matched_safe_action_units": matched_safe_units,
            "requested_lengths": sorted(int(length) for length in pools),
            "required_radii_m": sorted(float(r) for r in active_radii),
            "setting_sampling_policy": (
                args.setting_sampling_policy if setting is not None else None),
            "required_siblings_per_action": required_siblings,
            "radius_label_policy": "all_radii_consistent",
            "action_pattern": "strict_turn_forward_alternation",
            **({} if bank_manifest is None else {
                # The manifest is kept beside its digest so a reader can
                # rebuild the hash instead of trusting that the run was
                # rerun identically, and it covers what the sampler
                # offered rather than only what survived selection.
                "materialized_action_bank": bank_manifest,
                "materialized_action_bank_sha256": (
                    action_proposal.action_bank_manifest_sha256(
                        bank_manifest)),
                "proposal_provenance": proposal_provenance,
            }),
        }
        record_builder = (
            REC.build_record_v18
            if record_schema == REC.V18_SCHEMA_VERSION else build_record)
        rec = record_builder(
            frame, outcomes[frame.frame_id],
            image_path=current_image_path,
            floor_calibration=calibration.canonical_floor_fit,
            depth_path=(os.path.join("arr", frame.frame_id + "_depth.npy")
                        if args.save_arrays else None),
            intervention=intervention,
            selection=selection,
            source_provenance=source_provenance,
            collection_contract=strict_collection_contract,
        )
        rich_records.append(rec)
    validation_context = _pose_validation_context(
        args, source_provenance, variants, record_schema=record_schema)
    validate_records_before_spool(
        rich_records, validation_context=validation_context,
        asset_root=asset_root)
    for rec in rich_records:
        stats["frames"] += 1
        stats["outcomes"] += len(rec["outcomes"])
        stats["collided"] += sum(
            outcome["physical"]["collision"] is True
            for outcome in rec["outcomes"])
    group_records = []
    for rec, (sim, frame) in zip(rich_records, variants):
        compact = abc1_record.from_collected(
            rec, dataset=str(source_provenance["source_dataset"]),
            shard_id=str(getattr(args, "collection_shard_id", None) or "main"),
            pose_index=pose_index, frame=frame)
        if compact.get("surface_point_target") and compact["body_radii_m"] and not any(
                not case["collision"] and case["completed"] for case in compact["cases"]):
            seed = int(hashlib.sha256(compact["record_uid"].encode()).hexdigest()[:16], 16)
            radii = compact["body_radii_m"]
            added = fixed_pose_actions.collect_safe_group(
                sim, frame, rec, seed=seed, radius_m=radii[seed % len(radii)])
            if added is not None:
                cases = [abc1_record.compact_case(
                    compact, outcome, dataset=compact["dataset"],
                    record_uid=compact["record_uid"], provenance=added["provenance"])
                    for outcome in added["outcomes"]]
                compact = abc1_record.install_surface_delta(compact, {"cases": cases})
        group_records.append(compact)
    if getattr(args, "append_records", None):
        if not append_compact_records(
                records_path, group_records,
                progress_path=Path(asset_root) / "expansion.json"):
            for compact in group_records:
                paths = {compact["image_path"]}
                paths.update(case["terminal_rgb_path"] for case in compact["cases"]
                             if case.get("terminal_rgb_path"))
                for relative in paths:
                    (Path(asset_root) / relative).unlink(missing_ok=True)
            return False
    else:
        append_record_group(
            records_path, group_records,
            order_key=(scene_index, pose_index),
            expected_siblings=len(variants))
    funnel.record_pose(scene_id, position, yaw)
    for action_tag, actions in selected:
        label = group_labels[action_tag]
        stats[f"selected_{label}_L{len(actions)}"] += 1
    stats["accepted_poses"] += 1
    return True




def _collect_pose(
        *, args, sampler, sample_radii, scene_id, pose_index,
        pose_tries_per_attempt, pose_exclusions, sessions, fovs, heights, scene,
        proposal_bank, stats,
        skipped, completed_groups, image_dir, array_dir, records_path,
        scene_index, sampler_contract_hash, intervention_type, changed_fields,
        funnel) -> bool:
    stats["pose_attempts"] += 1
    intervention_group_id = pose_group_id(
        scene_id, pose_index, args.collection_shard_id)
    if intervention_group_id in completed_groups:
        stats["resumed_pose_groups"] += 1
        return True
    state = _prepare_pose_candidates(
        args=args, sampler=sampler, sample_radii=sample_radii,
        scene_id=scene_id, pose_index=pose_index,
        pose_tries_per_attempt=pose_tries_per_attempt,
        pose_exclusions=pose_exclusions, sessions=sessions, fovs=fovs,
        heights=heights, scene=scene, proposal_bank=proposal_bank,
        stats=stats, skipped=skipped,
        intervention_group_id=intervention_group_id)
    if state is None:
        return False
    stats["search_ready_poses"] += 1
    funnel.record_searched_pose(
        scene_id, state["position"], state["yaw"])
    state = _select_pose_candidates(
        state, args=args, scene_id=scene_id, pose_index=pose_index,
        stats=stats, skipped=skipped)
    if state is None:
        return False
    state = _evaluate_pose_candidates(
        state, args=args, stats=stats, skipped=skipped)
    if state is None:
        return False
    accepted = _persist_pose_group(
        state, args=args, image_dir=image_dir, array_dir=array_dir,
        records_path=records_path, scene_index=scene_index,
        pose_index=pose_index, sampler_contract_hash=sampler_contract_hash,
        intervention_type=intervention_type, changed_fields=changed_fields,
        funnel=funnel, stats=stats)
    if accepted:
        pose_exclusions.setdefault(scene_id, []).append({
            "position": [float(value) for value in state["position"]],
            "yaw_rad": float(state["yaw"]),
        })
    return accepted


def _preload_resumed_pose_exclusions(records_path, pose_exclusions) -> None:
    """Make already persisted resume groups visible to this shard's sampler."""
    seen = {
        (str(scene_id), *row["position"], row["yaw_rad"])
        for scene_id, rows in pose_exclusions.items()
        for row in rows
    }
    for record in io_utils.read_jsonl(Path(records_path), require_dict=True):
        scene_id = str(record["scene_id"])
        pose = record["pose"]
        row = {
            "position": [float(value) for value in pose["position"]],
            "yaw_rad": float(pose["yaw_rad"]),
        }
        key = (scene_id, *row["position"], row["yaw_rad"])
        if key not in seen:
            seen.add(key)
            pose_exclusions.setdefault(scene_id, []).append(row)


def _collect_scene(
        *, args, scene, scene_index, scene_count, heights, fovs,
        proposal_bank, stats,
        skipped, pose_exclusions, completed_groups, image_dir, array_dir,
        records_path, sampler_contract_hash, intervention_type, changed_fields,
        funnel) -> dict | None:
    sessions = _open_collection_sessions(args, scene, heights, fovs)
    sampler = sessions[0]
    emit_backend_ready(
        backend=getattr(args, "backend", "r2r"),
        scene_id=sampler.scene_id)
    scene_started_mono = time.monotonic()
    last_record_mono = None
    print(f"[{scene_index + 1}/{scene_count}] {sampler.scene_id}")
    scene_id = scene.scene_id
    sample_radii = _pose_sampling_radii(args.radii)
    scene_status = "completed"
    funnel.begin_scene(scene_id, stats, skipped)
    try:
        accepted_scene_poses = 0
        pose_attempt_budget = pose_attempt_budget_for(
            args.poses_per_scene,
            pose_candidates_per_scene=args.pose_candidates_per_scene)
        pose_tries_per_attempt = pose_candidate_draws_per_attempt(
            args.pose_candidates_per_scene)
        completed_attempts = 0
        capacity_stop = None
        for attempt_index in range(pose_attempt_budget):
            if accepted_scene_poses >= int(args.poses_per_scene):
                break
            if getattr(args, "append_records", None):
                progress = json.loads((Path(args.append_records).parent /
                                       "expansion.json").read_text())
                if progress["record_count"] >= progress["target_records"]:
                    break
            capacity_stop = collection_closeout.capacity_stop_descriptor(
                args, scene_started_mono=scene_started_mono,
                last_record_mono=last_record_mono,
                accepted_records=accepted_scene_poses,
                pose_attempts=completed_attempts,
                now_mono=time.monotonic())
            if capacity_stop is not None:
                skipped[f"capacity_stop_{capacity_stop['reason']}"] += 1
                break
            pose_index = attempt_index
            attempt_started_ns = time.perf_counter_ns()
            accepted = bool(_collect_pose(
                args=args, sampler=sampler, sample_radii=sample_radii,
                scene_id=scene_id, pose_index=pose_index,
                pose_tries_per_attempt=pose_tries_per_attempt,
                pose_exclusions=pose_exclusions, sessions=sessions, fovs=fovs,
                heights=heights, scene=scene,
                proposal_bank=proposal_bank,
                stats=stats, skipped=skipped,
                completed_groups=completed_groups, image_dir=image_dir,
                array_dir=array_dir, records_path=records_path,
                scene_index=scene_index, sampler_contract_hash=sampler_contract_hash,
                intervention_type=intervention_type,
                changed_fields=changed_fields, funnel=funnel))
            stats["pose_attempt_wallclock_us"] += max(
                1, (time.perf_counter_ns() - attempt_started_ns) // 1000)
            completed_attempts += 1
            accepted_scene_poses += int(accepted)
            if accepted:
                last_record_mono = time.monotonic()
                attempt_number = attempt_index + 1
                stats["accepted_pose_attempt_index_sum"] += attempt_number
                lower = (attempt_index // 1000) * 1000 + 1
                stats[
                    f"accepted_pose_attempt_{lower:04d}_{lower + 999:04d}"
                ] += 1
        return capacity_stop
    except KeyboardInterrupt:
        scene_status = "interrupted"
        raise
    except BaseException:
        scene_status = "failed"
        raise
    finally:
        _finish_scene_cleanup(
            lambda: close_sessions(sessions),
            lambda: funnel.finish_scene(
                scene_id, stats, skipped, status=scene_status))
def _run_collection(args, parser):
    pose_exclusions, fovs, heights = _validate_collection_args(args, parser)
    record_schema = collection_record_schema(args)
    semantic_module._reset_mp3d_query_diagnostics()
    consequence_module._reset_a_stability_batch_diagnostics()
    intervention_type, changed_fields = sensor_intervention(heights, fovs)
    stats, skipped = collections.Counter(), collections.Counter()
    try:
        scenes = discover_collection_scenes(args)
    except SceneCatalogError as error:
        parser.error(str(error))
    (proposal_bank,
     sampler_contract_hash) = _build_collection_action_banks(args, fovs, stats)
    os.makedirs(args.out, exist_ok=True)
    appending = getattr(args, "append_records", None)
    asset_root = str(Path(appending).parent) if appending else args.out
    image_dir = os.path.join(asset_root, "img")
    array_dir = os.path.join(asset_root, "arr")
    os.makedirs(image_dir, exist_ok=True)
    if args.save_arrays:
        os.makedirs(array_dir, exist_ok=True)
    records_path = appending or os.path.join(args.out, "records.jsonl")
    run_contract = collection_run_contract(
        args, scenes, heights, fovs,
        action_sampler_contract_sha256=sampler_contract_hash,
        sampling_provenance=collection_sampling_provenance(),
    )
    # A main pose publishes exactly one sensor sibling, so demanding the full
    # height x FOV cross product here would make every resumed pose look as if
    # it were missing renders.
    expected_siblings = (
        1)
    if appending:
        completed_groups = set()
        existing_records = 0
        with open(records_path, "rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                record = json.loads(line)
                existing_records += 1
                if record.get("collection_group_id"):
                    completed_groups.add(record["collection_group_id"])
                pose_exclusions.setdefault(record["scene_id"], []).append(record["pose"])
    else:
        completed_groups, existing_records = prepare_records_output(
            records_path, expected_siblings=expected_siblings,
            resume=args.resume, overwrite=args.overwrite,
            run_contract=run_contract)
        if args.resume and existing_records:
            _preload_resumed_pose_exclusions(records_path, pose_exclusions)
    funnel_path = Path(args.out) / "collection_funnel.json"
    if args.overwrite:
        funnel_path.unlink(missing_ok=True)
    funnel = collection_funnel.CollectionFunnel(
        funnel_path,
        record_schema_version=record_schema,
        oracle_contract_version=REC.ORACLE_CONTRACT_VERSION,
        code_revision=args.code_revision,
        code_dirty=bool(args.allow_dirty_code),
        config_sha256=collection_funnel.config_sha256(config),
        run_contract=run_contract, backend=args.backend,
        mode=args.collection_mode)
    funnel.restore_counters(stats, skipped)
    started = time.time()
    capacity_stop = None
    for scene_index, scene in enumerate(scenes):
        scene_capacity_stop = _collect_scene(
            args=args, scene=scene, scene_index=scene_index,
            scene_count=len(scenes), heights=heights, fovs=fovs,
            proposal_bank=proposal_bank,
            stats=stats, skipped=skipped,
            pose_exclusions=pose_exclusions, completed_groups=completed_groups,
            image_dir=image_dir, array_dir=array_dir,
            records_path=records_path, sampler_contract_hash=sampler_contract_hash,
            intervention_type=intervention_type,
            changed_fields=changed_fields, funnel=funnel)
        if scene_capacity_stop is not None:
            if capacity_stop is not None:
                raise ValueError(
                    "collector-side capacity stops require one scene per run")
            capacity_stop = scene_capacity_stop
    if appending:
        funnel.complete()
        print(json.dumps({"appended": stats["frames"], "seconds": time.time() - started,
                          "stats": dict(stats), "skipped": dict(skipped)}), flush=True)
        return 0
    return _finalize_collection_run(
        args=args, scenes=scenes, started=started, stats=stats, skipped=skipped,
        records_path=records_path, funnel=funnel,
        existing_records=existing_records, completed_groups=completed_groups,
        run_contract=run_contract, capacity_stop=capacity_stop)


def run_collection(args, parser):
    """Run collection, shutting the process-wide B1K runtime exactly once."""
    return _run_with_backend_shutdown(args, parser, _run_collection)
