"""Single-registry and typed A3 source-binding regressions."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from pipeline import (
    b1k_semantic, benchmark, collection_cli, collection_support, consensus,
    dataset_contracts, gs_semantic, record, scene_pool, semantic,
    source_manifest, validate,
)
from tests._synthetic import LEVEL_FLOOR_FIT, make_frame


EXPECTED_CONTRACTS = {
    "r2r": (
        ("scene", "navmesh", "semantic", "semantic_metadata",
         "scene_dataset_config"),
        "mp3d_ply",
        True,
        "r2r_train_episodes",
    ),
    "gs": (
        ("scene", "navmesh", "semantic", "collision_authority"),
        "gs_bbox",
        True,
        "gs_source_manifest",
    ),
    "b1k": (
        ("scene", "scene_authority"),
        "omnigibson_instance",
        True,
        "b1k_source_manifest",
    ),
}


def _source(dataset: str) -> dict:
    roles, semantic_format, _enabled, _source_param = \
        EXPECTED_CONTRACTS[dataset]
    assets = [{
        "role": role,
        "bytes": len(role),
        "sha256": hashlib.sha256(
            f"{dataset}:{role}".encode("ascii")).hexdigest(),
    } for role in roles]
    identity = {
        "version": "egoconseq.source-assets.v1",
        "assets": assets,
    }
    return {
        "scene_id": "scene",
        "source_dataset": dataset,
        "official_split": "train",
        "semantic_format": semantic_format,
        "source_manifest_sha256": "a" * 64,
        "source_asset_identity_version": identity["version"],
        "source_assets": assets,
        "source_assets_sha256": hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("utf-8")).hexdigest(),
    }


def _r2r_identity(digest: str) -> dict:
    protocol = "mp3d-complete-face-instance-universe.v1"
    return {
        "authority": "mp3d_full_face_universe",
        "schema": "mp3d-contact-face-identity.v1",
        "confirmed": True,
        "reason": "confirmed",
        "instance_id": 1,
        "category": "chair",
        "streaming_instance_faces_sha256": "b" * 64,
        "contact_face_distance_m": 0.0,
        "runner_up_face_distance_m": 0.4,
        "face_distance_margin_m": 0.4,
        "global_query_protocol": protocol,
        "semantic_ply_sha256": digest,
        "global_universe_sha256": semantic.complete_face_universe_sha256(
            global_query_protocol=protocol,
            semantic_ply_sha256=digest),
        "global_winner_instance_id": 1,
        "global_runner_up_instance_id": 2,
    }


def _b1k_identity(digest: str) -> dict:
    protocol = "b1k-runtime-instance-triangle-universe.v1"
    return {
        "authority": "b1k_runtime_triangle_universe",
        "schema": "b1k-contact-triangle-identity.v1",
        "confirmed": True,
        "reason": "confirmed",
        "instance_id": 1,
        "category": "chair",
        "runtime_instance_triangles_sha256": "b" * 64,
        "contact_triangle_distance_m": 0.0,
        "runner_up_triangle_distance_m": 0.4,
        "triangle_distance_margin_m": 0.4,
        "global_query_protocol": protocol,
        "scene_authority_sha256": digest,
        "global_universe_sha256":
            b1k_semantic.complete_triangle_universe_sha256(
                global_query_protocol=protocol,
                scene_authority_sha256=digest),
        "global_winner_instance_id": 1,
        "global_runner_up_instance_id": 2,
    }


def test_one_registry_drives_source_roles_formats_and_main_cli_choices():
    """Catches independent role/format/backend allowlists drifting apart."""
    observed = {}
    for dataset in EXPECTED_CONTRACTS:
        contract = dataset_contracts.dataset_source_contract(dataset)
        observed[dataset] = (
            contract.required_asset_roles,
            contract.semantic_format,
            contract.main_collection_enabled,
            contract.collection_source_param,
        )
        assert dataset_contracts.source_path_key(dataset) == \
            EXPECTED_CONTRACTS[dataset][3]
        assert collection_support.strict_shared_oracle_required(
            "main", dataset) is contract.main_collection_enabled
        assert scene_pool.validate_source_asset_provenance(
            _source(dataset))
    assert observed == EXPECTED_CONTRACTS

    parser = collection_cli.build_parser()
    backend_action = next(
        action for action in parser._actions if action.dest == "backend")
    assert tuple(backend_action.choices) == ("r2r", "gs", "b1k")
    assert tuple(backend_action.choices) == \
        dataset_contracts.main_collection_datasets()


def test_collection_runtime_uses_the_shared_backend_param_pruner():
    """Prevents child run metadata from growing a second pruning rule."""
    from pipeline import collection_runtime

    assert collection_runtime.prune_backend_scoped_params is \
        collection_support.prune_backend_scoped_params


def _gs_catalog(tmp_path):
    from tests.test_pl_gs_authority import _write_alignment_sources

    root = tmp_path / "gs"
    directory = root / "interior_0007_840137"
    directory.mkdir(parents=True)
    _write_alignment_sources(directory)
    (directory / "scene.navmesh").write_bytes(b"navmesh")
    (directory / "scene.collision.npz").write_bytes(b"collision")
    manifest = root / "train.json"
    manifest.write_text(json.dumps({
        "schema_version": scene_pool.SCENE_MANIFEST_VERSION,
        "dataset": "gs",
        "scenes": [{
            "scene_id": "interior_0007_840137",
            "split": "train",
            "path": "interior_0007_840137",
        }],
    }))
    scene = scene_pool.discover_gs_train_scenes(root, manifest)[0]
    source = scene_pool.verify_scene_source_asset_paths(scene, (
        ("scene", scene.scene_path),
        ("navmesh", scene.navmesh_path),
        ("semantic", scene.semantic_path),
        ("collision_authority", scene.collision_authority_path),
    ))
    return root, manifest, scene, source


def _gs_run_meta(tmp_path):
    root, manifest, scene, source = _gs_catalog(tmp_path)
    return {
        "record_schema_version": record.V18_SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "run_contract_sha256": "c" * 64,
        "params": {
            "backend": "gs",
            "collection_mode": "main",
            "gs_data_root": str(root),
            "gs_source_manifest": str(manifest),
            "action_mode": "balanced",
            "setting_sampling_policy": "one_sensor_one_body_per_pose.v1",
        },
        "resolved_scenes": [{**source, "scene_path": scene.scene_path}],
    }, source


def _gs_semantic_a_case(
        *, collision: bool = False, semantic_certified: bool = True):
    from tests.test_pl_gs_authority import _alignment_certificate

    source = _source("gs")
    binding = dataset_contracts.resolve_gs_collision_binding(source)
    actions = [
        {"type": "forward", "m": 1.0},
        {"type": "turn", "deg": 15.0},
        {"type": "forward", "m": 1.0},
    ]
    rows = []
    for perturbation in consensus.R2R_A_STABILITY_PERTURBATIONS:
        arc = 1.5 if collision else None
        full_attribution = ({
            "unattributed": False,
            "instance_id": 7,
        } if collision else {})
        depth_attribution = ({
            "unattributed": False,
            "instance_id": 7,
        } if collision else {})
        rows.append({
            "perturbation_id": perturbation["id"],
            "transform": {
                "x_m": perturbation["x_m"],
                "z_m": perturbation["z_m"],
                "yaw_deg": perturbation["yaw_deg"],
            },
            "physical": {
                "authority": "gs_collision_mesh",
                "geometry_authority_sha256":
                    binding.collision_authority_sha256,
                "collision": collision,
                "first_contact_arc_m": arc,
                "contact": ({
                    "full_geometry_attribution": full_attribution,
                } if collision else None),
            },
            "depth_physical": {
                "authority": "depth",
                "collision": collision,
                "first_contact_arc_m": arc,
                "contact": ({
                    "depth_mask_attribution": depth_attribution,
                } if collision else None),
            },
            "corridor_coverage": 1.0,
        })
    certificate = consensus.build_a_stability_certificate(
        actions, rows,
        require_contact_instance_witness=semantic_certified)
    alignment = _alignment_certificate()
    alignment["accepted"] = semantic_certified
    alignment["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in alignment.items() if key != "sha256"
    })
    outcome = {
        "outcome_id": "gs-outcome",
        "body": {"shape": "disc", "radius_m": 0.2},
        "actions": actions,
        "physical": copy.deepcopy(rows[0]["physical"]),
        "depth_physical": copy.deepcopy(rows[0]["depth_physical"]),
        "evidence": {"physical": {"coverage": 1.0}},
        "oracle_consensus": copy.deepcopy(
            certificate["rows"][0]["consensus"]),
        "shared_oracle_stability": certificate,
    }
    record_value = {
        "schema_version": record.V18_SCHEMA_VERSION,
        "source": source,
        "gs_collision_authority":
            dataset_contracts.gs_collision_binding_atom(binding),
        "gs_scene_capability":
            gs_semantic.scene_capability_atom(source, alignment),
    }
    return record_value, outcome


def test_gs_v18_source_context_rederives_the_exact_registered_bundle(tmp_path):
    run_meta, source = _gs_run_meta(tmp_path)
    context = source_manifest.gs_v18_context_from_run_meta(
        run_meta, authority_sha256="d" * 64)

    assert context.route == "gs_v18_registered"
    assert context.expected_schema_version == record.V18_SCHEMA_VERSION
    assert context.gs_scene_source_resolver(source["scene_id"]) == source

    # The synthetic record below intentionally has no collection selection.
    # Validate source binding under the same trusted source set without
    # claiming the balanced policies carried by the run metadata fixture.
    record_context = source_manifest.gs_v18_registered_validation_context(
        [source], authority_sha256="d" * 64,
        scene_capabilities={
            source["scene_id"]:
                context.gs_scene_capability_resolver(source["scene_id"]),
        })

    frame = make_frame()
    frame.scene_id = source["scene_id"]
    frame.semantic_index = type("Semantic", (), {
        "alignment_certificate":
            context.gs_scene_capability_resolver(source["scene_id"])[
                "alignment_certificate"],
        "semantic_certified": True,
    })()
    candidate = record.build_record_v18(
        frame, [], image_path="img/f.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source)
    assert validate.validate_record_source_bound(
        candidate, context=record_context) == []

    downgraded = copy.deepcopy(candidate)
    downgraded["gs_scene_capability"] = \
        gs_semantic.scene_capability_atom(source, None)
    assert record.validate_record_v18(downgraded) == []
    assert any(
        "trusted GS scene capability differs" in error
        for error in validate.validate_record_source_bound(
            downgraded, context=record_context))

    candidate["source"]["source_manifest_sha256"] = "0" * 64
    assert any(
        "trusted GS source differs" in error
        for error in validate.validate_record_source_bound(
            candidate, context=record_context))


def test_gs_collision_protocol_binds_the_official_coordinate_transform(
        tmp_path):
    _root, _manifest, _scene, source = _gs_catalog(tmp_path)
    binding = dataset_contracts.resolve_gs_collision_binding(source)

    assert binding.protocol_atom["semantic_identity"][
        "alignment_certificate"] == gs_semantic.ALIGNMENT_CERTIFICATE_SCHEMA


def test_gs_v18_source_context_rejects_run_meta_source_substitution(tmp_path):
    run_meta, _source_value = _gs_run_meta(tmp_path)
    run_meta["resolved_scenes"][0]["source_assets"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="trusted GS provenance"):
        source_manifest.gs_v18_context_from_run_meta(
            run_meta, authority_sha256="d" * 64)


def test_registry_rejects_unknown_dataset_format_and_asset_role_mismatch():
    with pytest.raises(ValueError):
        dataset_contracts.dataset_source_contract("not-r2r")
    wrong_format = _source("r2r")
    wrong_format["semantic_format"] = "omnigibson_instance"
    with pytest.raises(scene_pool.SceneCatalogError, match="semantic format"):
        scene_pool.validate_source_asset_provenance(wrong_format)
    wrong_role = _source("r2r")
    wrong_role["source_assets"][2]["role"] = "scene_authority"
    with pytest.raises(scene_pool.SceneCatalogError, match="asset roles"):
        scene_pool.validate_source_asset_provenance(wrong_role)
    missing_role = _source("r2r")
    missing_role["source_assets"].pop()
    with pytest.raises(scene_pool.SceneCatalogError, match="asset roles"):
        scene_pool.validate_source_asset_provenance(missing_role)


@pytest.mark.parametrize(
    "alias", [None, "", "R2R", " r2r", "r2r ", "B1K", 1, True],
)
def test_registry_requires_an_exact_lowercase_string_key(alias):
    """Catches coercion, trimming, or case-folding at the trust boundary."""
    with pytest.raises(ValueError, match="dataset is unsupported"):
        dataset_contracts.dataset_source_contract(alias)


@pytest.mark.parametrize("dataset", ["unknown"])
def test_unsupported_dataset_never_acquires_r2r_benchmark_reason(dataset):
    certificate, reason = benchmark.shared_visible_space_certificate(
        {"source": {"source_dataset": dataset}}, {})

    assert certificate is None
    assert reason == "unsupported_source_dataset"


def test_gs_semantic_a_certificate_is_source_bound_and_reconstructed():
    record_value, outcome = _gs_semantic_a_case(collision=False)

    certificate, reason = benchmark.shared_visible_space_certificate(
        record_value, outcome)

    assert reason == "eligible"
    assert certificate == outcome["shared_oracle_stability"]
    assert "contact_instance_witness_required" not in certificate
    assert certificate["rows"][0]["consensus"][
        "contact_instance_witness_required"] is True


def test_gs_geometry_only_a_certificate_is_source_bound_and_reconstructed():
    record_value, outcome = _gs_semantic_a_case(
        collision=False, semantic_certified=False)

    certificate, reason = benchmark.shared_visible_space_certificate(
        record_value, outcome)

    assert reason == "eligible"
    assert certificate == outcome["shared_oracle_stability"]
    assert certificate["contact_instance_witness_required"] is False
    assert "contact_instance_witness_required" not in \
        certificate["rows"][0]["consensus"]


@pytest.mark.parametrize("tamper", ["record_atom", "stability_row"])
def test_gs_semantic_a_certificate_rejects_authority_tampering(tamper):
    record_value, outcome = _gs_semantic_a_case(collision=False)
    if tamper == "record_atom":
        record_value["gs_collision_authority"]["sha256"] = "0" * 64
    else:
        rows = outcome["shared_oracle_stability"]["rows"]
        rows[1]["physical"]["geometry_authority_sha256"] = "0" * 64
        # Reseal the certificate so rejection proves source binding, rather
        # than merely detecting a stale certificate hash.
        outcome["shared_oracle_stability"] = \
            consensus.build_a_stability_certificate(
                outcome["actions"], rows,
                require_contact_instance_witness=True)

    certificate, reason = benchmark.shared_visible_space_certificate(
        record_value, outcome)

    assert certificate is None
    assert reason == "shared_oracle_stability_invalid"


@pytest.mark.parametrize("dataset", ["gs", "unknown"])
def test_unsupported_dataset_never_acquires_r2r_validation_label(dataset):
    errors = []

    validate._validate_shared_oracle_certificate(
        "[frame:outcome]", {}, [],
        {"contact_instance_witness_required": True}, False, errors,
        source_dataset=dataset)

    assert errors == [
        "[frame:outcome] contact instance witness has unsupported source "
        f"dataset {dataset!r}",
    ]


@pytest.mark.parametrize(
    ("identity_dataset", "binding_dataset"),
    [("r2r", "b1k"), ("b1k", "r2r")],
)
def test_typed_authority_binding_rejects_cross_injection(
        identity_dataset, binding_dataset):
    """Catches a matching digest authorizing the other identity schema."""
    shared_digest = "c" * 64
    schemas = {
        "r2r": "mp3d-contact-face-identity.v1",
        "b1k": "b1k-contact-triangle-identity.v1",
    }
    roles = {"r2r": "semantic", "b1k": "scene_authority"}
    binding = dataset_contracts.AuthorityBinding(
        source_dataset=binding_dataset,
        identity_schema=schemas[binding_dataset],
        authoritative_source_role=roles[binding_dataset],
        source_sha256=shared_digest,
    )
    identity = (_r2r_identity(shared_digest)
                if identity_dataset == "r2r"
                else _b1k_identity(shared_digest))

    assert consensus.a3_exact_contact_identity_valid(
        identity, full_instance_id=1, depth_instance_id=1,
        authority_binding=binding) is False


@pytest.mark.parametrize("dataset", ["r2r", "b1k"])
def test_typed_authority_binding_requires_exact_digest_and_is_not_optional(
        dataset):
    source = _source(dataset)
    if dataset == "b1k":
        source["split_authority"] = "project_defined"
    binding = dataset_contracts.resolve_authority_binding(source)
    expected_role = "semantic" if dataset == "r2r" else "scene_authority"
    expected_schema = (
        "mp3d-contact-face-identity.v1" if dataset == "r2r" else
        "b1k-contact-triangle-identity.v1")
    expected_digest = next(
        asset["sha256"] for asset in source["source_assets"]
        if asset["role"] == expected_role)
    assert binding == dataset_contracts.AuthorityBinding(
        source_dataset=dataset,
        identity_schema=expected_schema,
        authoritative_source_role=expected_role,
        source_sha256=expected_digest,
    )
    identity = (_r2r_identity(binding.source_sha256)
                if dataset == "r2r"
                else _b1k_identity(binding.source_sha256))

    assert consensus.a3_exact_contact_identity_valid(
        identity, full_instance_id=1, depth_instance_id=1,
        authority_binding=binding) is True
    wrong_digest = dataset_contracts.AuthorityBinding(
        source_dataset=binding.source_dataset,
        identity_schema=binding.identity_schema,
        authoritative_source_role=binding.authoritative_source_role,
        source_sha256="0" * 64,
    )
    assert consensus.a3_exact_contact_identity_valid(
        identity, full_instance_id=1, depth_instance_id=1,
        authority_binding=wrong_digest) is False
    assert consensus.a3_exact_contact_identity_valid(
        identity, full_instance_id=1, depth_instance_id=1,
        authority_binding=None) is False


def test_authority_binding_resolution_fails_closed_for_uncertified_adapters():
    unsupported = _source("r2r")
    unsupported["source_dataset"] = "unknown"
    with pytest.raises(ValueError, match="dataset is unsupported"):
        dataset_contracts.resolve_authority_binding(unsupported)

    gs = dataset_contracts.resolve_authority_binding(_source("gs"))
    assert gs.source_dataset == "gs"
    assert gs.identity_schema == "gs-visible-contact-instance-identity.v1"
    assert gs.authoritative_source_role == "source_bundle"
    assert gs.semantic_source_sha256 == next(
        row["sha256"] for row in _source("gs")["source_assets"]
        if row["role"] == "semantic")
