"""Dataset-aware record contracts shared by the R2R and B1K routes."""

import copy
import hashlib

import pytest

from pipeline import (
    benchmark,
    collection_assets,
    consensus,
    future_view_selection,
    record,
    source_manifest,
    validate,
)
from tests._synthetic import LEVEL_FLOOR_FIT, make_frame, source_provenance
from tests.test_pl_v16_c_assets import _clear_outcome, _terminal_frame


def _b1k_source(scene_id="synthetic") -> dict:
    assets = [
        {
            "role": role,
            "bytes": len(role),
            "sha256": hashlib.sha256(role.encode("ascii")).hexdigest(),
        }
        for role in ("scene", "scene_authority")
    ]
    return {
        "scene_id": str(scene_id),
        "source_dataset": "b1k",
        "official_split": "train",
        "split_authority": "project_defined",
        "semantic_format": "omnigibson_instance",
        "source_manifest": "/datasets/b1k/train.json",
        "source_manifest_sha256": "a" * 64,
        "source_asset_identity_version": "egoconseq.source-assets.v1",
        "source_assets": assets,
        "source_assets_sha256": record.canonical_atom_sha256({
            "version": "egoconseq.source-assets.v1",
            "assets": assets,
        }),
    }


def _scene_authority_sha256(source: dict) -> str:
    return next(
        asset["sha256"] for asset in source["source_assets"]
        if asset["role"] == "scene_authority")


def _bind_b1k_stability(outcome: dict) -> dict:
    value = copy.deepcopy(outcome)
    rows = copy.deepcopy(value["shared_oracle_stability"]["rows"])
    for row in rows:
        row["physical"]["authority"] = "b1k_geometry"
    certificate = consensus.build_a_stability_certificate(
        value["actions"], rows)
    value["shared_oracle_stability"] = certificate
    value["physical"] = copy.deepcopy(certificate["rows"][0]["physical"])
    value["depth_physical"] = copy.deepcopy(
        certificate["rows"][0]["depth_physical"])
    value["oracle_consensus"] = copy.deepcopy(
        certificate["rows"][0]["consensus"])
    return value


def _build_b1k_b_record():
    source = _b1k_source()
    frame = make_frame()
    frame.objects = [{
        "instance_id": 7,
        "category": "chair",
        "is_structural": False,
        "mask_area_px": 4000,
        "depth_backed_px": 4000,
        "bbox_xyxy_px": [200, 150, 439, 329],
        "centroid_px": [320, 240],
        "dist_nearest_m": 2.0,
        "surface_anchor": {
            "protocol": "b1k-visible-depth-interior.v1",
            "pixel_xy_px": [320, 240],
            "initial_robot_xyz_m": [0.0, 0.5, -2.0],
        },
    }]
    outcome = _bind_b1k_stability(_clear_outcome(outcome_id="o-a"))
    contract = record.collection_contract(source, "main")
    rec = record.build_record(
        frame, [outcome], image_path="images/target.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source,
        collection_contract=contract)
    return rec, rec["outcomes"][0], None


def test_collection_contract_dispatches_b1k_v11_without_changing_r2r():
    source = _b1k_source()

    contract = record.collection_contract(source, "main")

    assert contract == {
        "version": "b1k-visible-space-abc1.v7",
        "record_schema_version": "conseq.v11",
        "oracle_contract_version": "ground-disc-visible-v8",
        "collection_mode": "main",
        "source_dataset": "b1k",
        "official_split": "train",
        "split_authority": "project_defined",
        "semantic_format": "omnigibson_instance",
        "source_manifest_sha256": "a" * 64,
        "source_assets_sha256": source["source_assets_sha256"],
        "scene_authority_sha256": _scene_authority_sha256(source),
        "renderer_instance_protocol":
            "b1k-observation.v5",
        "observation_profile_sha256":
            record.B1K_OBSERVATION_PROFILE_SHA256,
        "c1_render_mode": record.B1K_C1_RENDER_MODE,
        "sha256": contract["sha256"],
    }
    assert contract["sha256"] == record.canonical_atom_sha256({
        key: value for key, value in contract.items() if key != "sha256"
    })
    assert contract["c1_render_mode"] == \
        "persistent-counterfactual-bank-batch.v2"
    legacy = record.collection_contract(
        source, "main", contract_version="b1k-visible-space-abc1.v5")
    assert legacy["c1_render_mode"] == \
        "temporary-counterfactual-bank-batch.v1"
    legacy_v6 = record.collection_contract(
        source, "main", contract_version="b1k-visible-space-abc1.v6")
    assert legacy_v6["observation_profile_sha256"] == \
        record.B1K_LEGACY_OBSERVATION_PROFILE_SHA256
    assert contract["observation_profile_sha256"] != \
        legacy_v6["observation_profile_sha256"]
    r2r = source_provenance("r2r-scene", dataset="r2r")
    assert record.collection_contract(r2r, "main") == \
        record.r2r_v16_collection_contract(r2r, "main")


def test_b1k_v5_collection_contract_remains_readable():
    rec, outcome, _authority = _build_b1k_b_record()
    rec["collection_contract"] = record.collection_contract(
        rec["source"], "main",
        contract_version="b1k-visible-space-abc1.v5")

    context = source_manifest.b1k_v16_registered_validation_context(
        [rec["source"]], collection_mode="main",
        expected_schema_version=record.SCHEMA_VERSION,
        expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION,
        authority_sha256="f" * 64,
        scene_authority_resolver=lambda _scene_id: _scene_authority_sha256(
            rec["source"]))
    errors = validate.validate_record_source_bound(rec, context=context)
    certificate, reason = benchmark.shared_visible_space_certificate(
        rec, outcome)

    assert not [error for error in errors if "collection contract" in error]
    assert certificate == outcome["shared_oracle_stability"]
    assert reason == "eligible"


def test_b1k_source_format_and_physical_authority_validate():
    rec, _outcome, _authority = _build_b1k_b_record()

    errors = validate.validate_record_local(rec)

    assert not [
        error for error in errors
        if "unsupported source dataset" in error
        or "source semantic format" in error
        or "physical authority" in error
        or "source asset" in error
        or "collection contract" in error
    ]


def test_b1k_collision_source_must_match_its_geometry_authority():
    rec, outcome, _authority = _build_b1k_b_record()
    outcome["physical"].update({
        "collision": True,
        "collision_source": "navmesh",
    })

    errors = validate.validate_record_local(rec)

    assert (
        "[F-test:o-a] B1K geometry source does not match its physical "
        "collision state") in errors


def test_b1k_invalid_contract_validation_never_claims_it_is_r2r():
    rec, _outcome, _authority = _build_b1k_b_record()
    rec["collection_contract"]["version"] = "unknown-contract"

    errors = validate.validate_record_local(rec)

    assert any("collection contract version is unsupported" in error
               for error in errors)
    assert not [error for error in errors if "R2R v16" in error]


def test_b1k_source_bound_validation_uses_registered_scene_authority():
    rec, _outcome, _authority = _build_b1k_b_record()
    resolved = []

    def resolve_scene_authority(scene_id):
        resolved.append(scene_id)
        return _scene_authority_sha256(rec["source"])

    context = source_manifest.b1k_v16_registered_validation_context(
        [rec["source"]], collection_mode="main",
        expected_schema_version=record.SCHEMA_VERSION,
        expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION,
        authority_sha256="c" * 64,
        scene_authority_resolver=resolve_scene_authority)

    errors = validate.validate_record_source_bound(rec, context=context)

    assert resolved == [rec["scene_id"]]
    assert not [
        error for error in errors
        if "trusted R2R source" in error or "B1K scene authority" in error
    ]

    substituted = source_manifest.b1k_v16_registered_validation_context(
        [rec["source"]], collection_mode="main",
        expected_schema_version=record.SCHEMA_VERSION,
        expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION,
        authority_sha256="d" * 64,
        scene_authority_resolver=lambda _scene_id: "0" * 64)
    substituted_errors = validate.validate_record_source_bound(
        rec, context=substituted)
    assert "[F-test] B1K scene authority differs from trusted resolver" \
        in substituted_errors


def test_b1k_source_bound_validation_contains_unexpected_resolver_error():
    rec, _outcome, _authority = _build_b1k_b_record()

    def broken_resolver(_scene_id):
        raise AttributeError("resolver bug")

    context = source_manifest.b1k_v16_registered_validation_context(
        [rec["source"]], collection_mode="main",
        expected_schema_version=record.SCHEMA_VERSION,
        expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION,
        authority_sha256="e" * 64,
        scene_authority_resolver=broken_resolver)

    errors = validate.validate_record_source_bound(rec, context=context)

    assert "[F-test] trusted B1K scene authority unavailable: resolver bug" \
        in errors


@pytest.mark.parametrize(
    ("malformed", "expected_error"),
    [
        ("terminal_rgb_asset",
         "[F-test:o-a] trusted B1K terminal RGB atom is invalid"),
        ("source",
         "[F-test:o-a] trusted B1K terminal source authority is invalid"),
        ("renderer",
         "[F-test:o-a] trusted B1K renderer authority is invalid"),
    ],
)
def test_b1k_source_bound_validation_contains_malformed_nested_atom(
        malformed, expected_error):
    rec, outcome, _authority = _build_b1k_b_record()
    scene_sha256 = _scene_authority_sha256(rec["source"])
    if malformed == "terminal_rgb_asset":
        outcome["terminal_rgb_asset"] = "not-an-atom"
    else:
        outcome["terminal_rgb_asset"] = {
            "source": (
                "not-a-source" if malformed == "source" else
                {"scene_authority_sha256": scene_sha256}),
            "renderer": (
                "not-a-renderer" if malformed == "renderer" else
                {"source_scene_sha256": scene_sha256}),
        }
    context = source_manifest.b1k_v16_registered_validation_context(
        [rec["source"]], collection_mode="main",
        expected_schema_version=record.SCHEMA_VERSION,
        expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION,
        authority_sha256="f" * 64,
        scene_authority_resolver=lambda _scene_id: scene_sha256)

    errors = validate.validate_record_source_bound(rec, context=context)

    assert expected_error in errors


def test_b1k_c1_binds_scene_authority_and_omnigibson_renderer(tmp_path):
    source = _b1k_source()
    contract = record.collection_contract(source, "main")
    base = make_frame()
    outcome = _bind_b1k_stability(_clear_outcome())
    terminal = _terminal_frame(base, outcome)
    cache = {
        future_view_selection.terminal_render_cache_key(outcome): terminal,
    }

    atom = future_view_selection.materialize_terminal_rgb_asset(
        tmp_path, base_frame=base, outcome=outcome, render_cache=cache,
        source=source, collection_contract=contract,
        render_transaction=record.B1K_C1_RENDER_MODE)
    outcome["terminal_rgb_asset"] = atom
    outcome["base_rollout_key"] = atom["binding"]["base_rollout_key"]
    rec = {
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "frame_id": base.frame_id,
        "scene_id": base.scene_id,
        "pose": {
            "position": base.position.tolist(),
            "yaw_rad": base.yaw_rad,
        },
        "sensor": base.sensor.to_dict(),
        "source": source,
        "collection_contract": contract,
        "outcomes": [outcome],
    }

    assert atom["source"]["scene_authority_sha256"] == \
        _scene_authority_sha256(source)
    assert atom["renderer"] == {
        "protocol": "b1k-observation.v5",
        "backend": "omnigibson",
        "color_format": "uint8_rgb",
        "source_scene_sha256": _scene_authority_sha256(source),
        "observation_profile_sha256":
            record.B1K_OBSERVATION_PROFILE_SHA256,
        "c1_render_mode": record.B1K_C1_RENDER_MODE,
    }
    assert future_view_selection.counterfactual_c_outcome_eligibility(
        rec, outcome, asset_root=tmp_path) == (True, "eligible")


@pytest.mark.parametrize(
    ("contract_version", "render_transaction", "profile_sha256"),
    [
        (
            record.B1K_V5_COLLECTION_CONTRACT_VERSION,
            record.B1K_V5_C1_RENDER_MODE,
            record.B1K_LEGACY_OBSERVATION_PROFILE_SHA256,
        ),
        (
            record.B1K_V16_COLLECTION_CONTRACT_VERSION,
            record.B1K_C1_RENDER_MODE,
            record.B1K_LEGACY_OBSERVATION_PROFILE_SHA256,
        ),
    ],
)
def test_b1k_legacy_c1_materializes_under_its_stored_contract(
        tmp_path, contract_version, render_transaction, profile_sha256):
    source = _b1k_source()
    contract = record.collection_contract(
        source, "main", contract_version=contract_version)
    base = make_frame()
    outcome = _bind_b1k_stability(_clear_outcome())
    terminal = _terminal_frame(base, outcome)
    cache = {
        future_view_selection.terminal_render_cache_key(outcome): terminal,
    }

    atom = future_view_selection.materialize_terminal_rgb_asset(
        tmp_path, base_frame=base, outcome=outcome, render_cache=cache,
        source=source, collection_contract=contract,
        render_transaction=render_transaction)

    assert atom["renderer"]["observation_profile_sha256"] == \
        profile_sha256
    assert atom["renderer"]["c1_render_mode"] == render_transaction


@pytest.mark.parametrize(
    ("contract_version", "render_transaction"),
    [
        (
            record.B1K_V5_COLLECTION_CONTRACT_VERSION,
            record.B1K_V5_C1_RENDER_MODE,
        ),
        (
            record.B1K_V16_COLLECTION_CONTRACT_VERSION,
            record.B1K_C1_RENDER_MODE,
        ),
    ],
)
def test_b1k_terminal_assets_are_limited_to_the_simultaneous_batch(
        tmp_path, contract_version, render_transaction):
    source = _b1k_source()
    contract = record.collection_contract(
        source, "main", contract_version=contract_version)
    base = make_frame()
    selected = _bind_b1k_stability(_clear_outcome())
    incidental = copy.deepcopy(selected)
    incidental["outcome_id"] = "incidental-clear"
    terminal = _terminal_frame(base, selected)
    cache = {
        future_view_selection.terminal_render_cache_key(selected): terminal,
    }

    counts = collection_assets.attach_terminal_rgb_assets(
        tmp_path, base, [selected, incidental], cache,
        source=source, collection_contract=contract,
        eligible_outcome_ids={selected["outcome_id"]},
        render_transaction=render_transaction,
        terminal_renderer=lambda _pose: pytest.fail(
            "B1K v5 must not fall back to sequential terminal rendering"))

    assert counts == {"materialized": 1, "withheld": 1}
    assert selected["terminal_rgb_asset"]["renderer"]["c1_render_mode"] == \
        render_transaction
    assert "terminal_rgb_asset" not in incidental
    assert incidental["terminal_rgb_asset_withhold"] == \
        "terminal_render_transaction_unavailable"
    assert incidental["terminal_rgb_asset_withhold_authority"] == \
        "collection_runtime_attested"


def test_b1k_terminal_asset_withhold_exposes_ephemeral_diagnostics(tmp_path):
    source = _b1k_source()
    contract = record.collection_contract(source, "main")
    base = make_frame()
    selected = _bind_b1k_stability(_clear_outcome())
    failures = []

    counts = collection_assets.attach_terminal_rgb_assets(
        tmp_path, base, [selected], {}, source=source,
        collection_contract=contract,
        eligible_outcome_ids={selected["outcome_id"]},
        render_transaction=record.B1K_C1_RENDER_MODE,
        failure_diagnostics=failures)

    assert counts == {"materialized": 0, "withheld": 1}
    assert failures == [{
        "outcome_id": selected["outcome_id"],
        "reason": "terminal_cache_miss",
        "authority": "collection_runtime_attested",
        "detail": "cached terminal RGB is missing",
    }]
    assert "terminal_rgb_asset_diagnostic" not in selected


def test_b1k_terminal_asset_validator_rejects_non_batch_transaction(tmp_path):
    source = _b1k_source()
    contract = record.collection_contract(source, "main")
    base = make_frame()
    outcome = _bind_b1k_stability(_clear_outcome())
    terminal = _terminal_frame(base, outcome)
    cache = {
        future_view_selection.terminal_render_cache_key(outcome): terminal,
    }
    atom = future_view_selection.materialize_terminal_rgb_asset(
        tmp_path, base_frame=base, outcome=outcome, render_cache=cache,
        source=source, collection_contract=contract,
        render_transaction=record.B1K_C1_RENDER_MODE)
    outcome["terminal_rgb_asset"] = atom
    outcome["base_rollout_key"] = atom["binding"]["base_rollout_key"]
    rec = {
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "frame_id": base.frame_id,
        "scene_id": base.scene_id,
        "pose": {
            "position": base.position.tolist(),
            "yaw_rad": base.yaw_rad,
        },
        "sensor": base.sensor.to_dict(),
        "source": source,
        "collection_contract": contract,
        "outcomes": [outcome],
    }
    atom["renderer"]["c1_render_mode"] = "sequential-single-sensor.v1"
    atom["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in atom.items() if key != "sha256"
    })

    with pytest.raises(ValueError, match="renderer binding"):
        future_view_selection.validate_terminal_rgb_asset(
            rec, outcome, asset_root=tmp_path)
