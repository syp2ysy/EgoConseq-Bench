"""Adversarial source-rederivation tests for strict registered v16 B."""

import copy
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline import (
    action_proposal, config, record, scene_pool, semantic, source_manifest,
    validate,
)
from tests._synthetic import LEVEL_FLOOR_FIT, make_frame
from tests.test_pl_v16_b_candidates import _record_outcome
from tests.test_pl_v16_b_geometry import _write_target_scene


def _trusted_record(tmp_path, *, catalog_root=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene_dir = (catalog_root / "TARGET"
                 if catalog_root is not None else tmp_path)
    scene, semantic_ply = _write_target_scene(scene_dir)
    navmesh = scene_dir / "TARGET.navmesh"
    navmesh.write_bytes(b"navmesh")
    scene_config = (
        catalog_root if catalog_root is not None else tmp_path
    ) / "mp3d_annotated_basis.scene_dataset_config.json"
    scene_config.write_text("{}")
    episodes = tmp_path / "train.json"
    episodes.write_text(json.dumps({"episodes": [{"scene_id": "TARGET.glb"}]}))
    spec = scene_pool.SceneSpec(
        scene_id="TARGET", source_dataset="r2r", official_split="train",
        scene_path=str(scene), navmesh_path=str(navmesh),
        semantic_path=str(semantic_ply),
        semantic_metadata_path=str(scene_dir / "TARGET.house"),
        semantic_format="mp3d_ply",
        scene_dataset_config=str(scene_config),
        provenance_path=str(episodes),
        provenance_sha256=hashlib.sha256(episodes.read_bytes()).hexdigest(),
    )
    source = spec.provenance()
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    frame = make_frame()
    frame.scene_id = "TARGET"
    frame.scene_glb = str(scene)
    frame.semantic_index = index
    frame.id_to_cat = {1: "chair"}
    frame.objects = [{
        "instance_id": 1, "category": "chair", "is_structural": False,
        "mask_area_px": 4000, "depth_backed_px": 4000,
        "bbox_xyxy_px": [200, 150, 439, 329],
        "centroid_px": [320, 240], "dist_nearest_m": 2.0,
    }]
    rec = record.build_record(
        frame, [_record_outcome()], image_path="images/target.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source,
        collection_contract=record.r2r_v16_collection_contract(
            source, "main"))
    context = source_manifest.r2r_v16_registered_validation_context(
        [source], collection_mode="main",
        expected_schema_version=record.SCHEMA_VERSION,
        expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION,
        authority_sha256="a" * 64,
        r2r_train_episodes=episodes,
        mp3d_root=(catalog_root if catalog_root is not None else tmp_path))
    return rec, context, spec


def test_scene_provenance_does_not_persist_manifest_path(tmp_path):
    _rec, _context, spec = _trusted_record(tmp_path)

    provenance = spec.provenance()

    assert "source_manifest" not in provenance
    assert provenance["source_manifest_sha256"] == spec.provenance_sha256


def _context_for_spec(spec, authority="c"):
    return source_manifest.r2r_v16_registered_validation_context(
        [spec.provenance()], collection_mode="main",
        expected_schema_version=record.SCHEMA_VERSION,
        expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION,
        authority_sha256=authority * 64,
        r2r_train_episodes=spec.provenance_path,
        mp3d_root=Path(spec.scene_dataset_config).parent,
    )


def test_trusted_scene_cache_reuses_one_full_hash_across_temporary_contexts(
        tmp_path, monkeypatch):
    mp3d_root = tmp_path / "mp3d"
    rec, _context, spec = _trusted_record(
        tmp_path, catalog_root=mp3d_root)
    scene_pool._TRUSTED_R2R_SCENE_CACHE.clear()
    original_sha256 = scene_pool.sha256_file
    original_manifest_read = scene_pool._read_pinned_r2r_manifest
    hashes = Counter()
    manifest_reads = Counter()

    def counted_sha256(path):
        hashes[str(path)] += 1
        return original_sha256(path)

    def counted_manifest_read(path):
        manifest_reads[str(Path(path).resolve())] += 1
        return original_manifest_read(path)

    monkeypatch.setattr(scene_pool, "sha256_file", counted_sha256)
    monkeypatch.setattr(
        scene_pool, "_read_pinned_r2r_manifest", counted_manifest_read)

    for index in range(6):
        scene_pool._trusted_r2r_scene_for_context(
            rec, _context_for_spec(spec, authority=str(index)))

    expected_assets = [
        spec.scene_path, spec.navmesh_path, spec.semantic_path,
        spec.semantic_metadata_path, spec.scene_dataset_config,
    ]
    assert all(hashes[str(path)] == 1 for path in expected_assets)
    assert manifest_reads[str(Path(spec.provenance_path).resolve())] == 2
    assert len(scene_pool._TRUSTED_R2R_SCENE_CACHE) == 1


def test_trusted_scene_cache_separates_new_authorized_manifest(
        tmp_path, monkeypatch):
    mp3d_root = tmp_path / "mp3d"
    rec, _unused, spec = _trusted_record(tmp_path, catalog_root=mp3d_root)
    old_context = _context_for_spec(spec)
    scene_pool._TRUSTED_R2R_SCENE_CACHE.clear()
    scene_pool._trusted_r2r_scene_for_context(rec, old_context)
    episodes = Path(spec.provenance_path)
    episodes.write_text(json.dumps({
        "episodes": [{"scene_id": "TARGET.glb"}], "revision": 2,
    }))
    new_spec = replace(
        spec, provenance_sha256=hashlib.sha256(
            episodes.read_bytes()).hexdigest())
    new_rec = copy.deepcopy(rec)
    new_rec["source"] = new_spec.provenance()
    new_context = _context_for_spec(new_spec, authority="d")
    hashes = Counter()
    original_sha256 = scene_pool.sha256_file
    original_manifest_read = scene_pool._read_pinned_r2r_manifest
    manifest_reads = Counter()

    def counted_sha256(path):
        hashes[str(path)] += 1
        return original_sha256(path)

    def counted_manifest_read(path):
        manifest_reads[str(Path(path).resolve())] += 1
        return original_manifest_read(path)

    monkeypatch.setattr(scene_pool, "sha256_file", counted_sha256)
    monkeypatch.setattr(
        scene_pool, "_read_pinned_r2r_manifest", counted_manifest_read)

    scene_pool._trusted_r2r_scene_for_context(new_rec, new_context)

    assert manifest_reads[str(episodes.resolve())] == 2
    assert all(hashes[str(path)] == 1 for path in [
        spec.scene_path, spec.navmesh_path, spec.semantic_path,
        spec.semantic_metadata_path, spec.scene_dataset_config,
    ])
def test_trusted_scene_auth_rechecks_manifest_after_discovery(
        tmp_path, monkeypatch):
    mp3d_root = tmp_path / "mp3d"
    rec, context, spec = _trusted_record(tmp_path, catalog_root=mp3d_root)
    episodes = Path(spec.provenance_path)
    original_resolve = scene_pool.resolve_trusted_r2r_scene

    def resolve_then_replace(*args, **kwargs):
        resolved = original_resolve(*args, **kwargs)
        episodes.write_text(json.dumps({
            "episodes": [{"scene_id": "TARGET.glb"}], "revision": 2,
        }))
        return resolved

    monkeypatch.setattr(
        scene_pool, "resolve_trusted_r2r_scene", resolve_then_replace)
    scene_pool._TRUSTED_R2R_SCENE_CACHE.clear()

    with pytest.raises(
            scene_pool.SceneCatalogError,
            match="trusted R2R manifest digest changed"):
        scene_pool._trusted_r2r_scene_for_context(rec, context)


def test_trusted_scene_auth_rechecks_manifest_scene_whitelist(
        tmp_path, monkeypatch):
    mp3d_root = tmp_path / "mp3d"
    rec, _unused, spec = _trusted_record(tmp_path, catalog_root=mp3d_root)
    episodes = Path(spec.provenance_path)
    episodes.write_text(json.dumps({
        "episodes": [{"scene_id": "OTHER.glb"}],
    }))
    inconsistent = replace(
        spec, provenance_sha256=hashlib.sha256(
            episodes.read_bytes()).hexdigest())
    rec["source"] = inconsistent.provenance()
    context = _context_for_spec(inconsistent)
    monkeypatch.setattr(
        scene_pool, "resolve_trusted_r2r_scene",
        lambda *_args, **_kwargs: inconsistent)
    scene_pool._TRUSTED_R2R_SCENE_CACHE.clear()

    with pytest.raises(
            scene_pool.SceneCatalogError,
            match="does not authorize scene 'TARGET'"):
        scene_pool._trusted_r2r_scene_for_context(rec, context)


def test_trusted_scene_cache_hit_rejects_forged_source_payload(tmp_path):
    mp3d_root = tmp_path / "mp3d"
    rec, _unused, spec = _trusted_record(tmp_path, catalog_root=mp3d_root)
    context = _context_for_spec(spec)
    scene_pool._TRUSTED_R2R_SCENE_CACHE.clear()
    scene_pool._trusted_r2r_scene_for_context(rec, context)
    forged = copy.deepcopy(rec)
    forged["source"]["semantic_format"] = "forged-format"

    with pytest.raises(
            scene_pool.SceneCatalogError,
            match="cached trusted R2R provenance disagrees"):
        scene_pool._trusted_r2r_scene_for_context(forged, context)


@pytest.mark.parametrize("digest_field", [
    "source_manifest_sha256", "source_assets_sha256",
])
def test_forged_source_digest_cannot_force_scene_reauthentication(
        tmp_path, monkeypatch, digest_field):
    mp3d_root = tmp_path / "mp3d"
    rec, _unused, spec = _trusted_record(tmp_path, catalog_root=mp3d_root)
    context = _context_for_spec(spec)
    scene_pool._TRUSTED_R2R_SCENE_CACHE.clear()
    scene_pool._trusted_r2r_scene_for_context(rec, context)
    forged = copy.deepcopy(rec)
    forged["source"][digest_field] = "1" * 64
    hashes = []
    original_sha256 = scene_pool.sha256_file

    def counted_sha256(path):
        hashes.append(str(path))
        return original_sha256(path)

    monkeypatch.setattr(scene_pool, "sha256_file", counted_sha256)

    with pytest.raises(
            scene_pool.SceneCatalogError,
            match="record source binding disagrees with expected"):
        scene_pool._trusted_r2r_scene_for_context(forged, context)

    assert hashes == []
    assert len(scene_pool._TRUSTED_R2R_SCENE_CACHE) == 1


def test_trusted_scene_cache_has_hard_lru_scene_limit(tmp_path, monkeypatch):
    first_root = tmp_path / "first-mp3d"
    second_root = tmp_path / "second-mp3d"
    first, _unused, first_spec = _trusted_record(
        tmp_path / "first", catalog_root=first_root)
    second, _unused, second_spec = _trusted_record(
        tmp_path / "second", catalog_root=second_root)
    monkeypatch.setattr(config, "TRUSTED_R2R_SCENE_CACHE_MAX_SCENES", 1)
    scene_pool._TRUSTED_R2R_SCENE_CACHE.clear()

    scene_pool._trusted_r2r_scene_for_context(
        first, _context_for_spec(first_spec, authority="d"))
    scene_pool._trusted_r2r_scene_for_context(
        second, _context_for_spec(second_spec, authority="e"))

    assert len(scene_pool._TRUSTED_R2R_SCENE_CACHE) == 1
    only_key = next(iter(scene_pool._TRUSTED_R2R_SCENE_CACHE))
    assert only_key[1] == str(second_root.resolve())


def test_registered_validation_uses_context_nondefault_r2r_roots(tmp_path):
    mp3d_root = tmp_path / "nondefault-mp3d"
    rec, _context, spec = _trusted_record(
        tmp_path, catalog_root=mp3d_root)
    context = source_manifest.r2r_v16_registered_validation_context(
        [spec.provenance()], collection_mode="main",
        expected_schema_version=record.SCHEMA_VERSION,
        expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION,
        authority_sha256="b" * 64,
        r2r_train_episodes=spec.provenance_path,
        mp3d_root=mp3d_root,
    )

    errors = validate.validate_record_source_bound(rec, context=context)

    assert not any("trusted R2R source" in error for error in errors)


def test_run_contract_context_derives_hash_authorized_r2r_roots(tmp_path):
    episodes = tmp_path / "episodes.json.gz"
    mp3d_root = tmp_path / "mp3d"
    run_contract = {
        "params": {
            "collection_mode": "main",
            "r2r_train_episodes": str(episodes),
            "mp3d_root": str(mp3d_root),
        },
        "resolved_scenes": [copy.deepcopy(
            _trusted_record(tmp_path / "source")[0]["source"])],
    }
    digest = record.canonical_atom_sha256(run_contract)

    context = source_manifest.r2r_v16_context_from_run_contract(
        run_contract, expected_run_contract_sha256=digest,
        expected_schema_version=record.SCHEMA_VERSION,
        expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION)

    assert context.r2r_train_episodes == str(episodes)
    assert context.mp3d_root == str(mp3d_root)


def test_run_contract_context_rejects_missing_hash_authorized_roots(tmp_path):
    run_contract = {
        "params": {"collection_mode": "main"},
        "resolved_scenes": [copy.deepcopy(
            _trusted_record(tmp_path / "source")[0]["source"])],
    }

    with pytest.raises(ValueError, match="trusted R2R roots"):
        source_manifest.r2r_v16_context_from_run_contract(
            run_contract,
            expected_run_contract_sha256=record.canonical_atom_sha256(
                run_contract),
            expected_schema_version=record.SCHEMA_VERSION,
            expected_oracle_contract_version=record.ORACLE_CONTRACT_VERSION)


def test_run_meta_context_derives_hash_authorized_r2r_roots(tmp_path):
    episodes = tmp_path / "episodes.json.gz"
    mp3d_root = tmp_path / "mp3d"
    source = copy.deepcopy(
        _trusted_record(tmp_path / "source")[0]["source"])
    run_meta = {
        "run_contract_sha256": "d" * 64,
        "params": {
            "collection_mode": "main",
            "r2r_train_episodes": str(episodes),
            "mp3d_root": str(mp3d_root),
        },
        "resolved_scenes": [source],
        "record_schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
    }

    context = source_manifest.r2r_v16_context_from_run_meta(
        run_meta, authority_sha256="e" * 64)

    assert context.r2r_train_episodes == str(episodes)
    assert context.mp3d_root == str(mp3d_root)

    v3_meta = copy.deepcopy(run_meta)
    v3_meta["params"].update({
        "action_mode": "balanced",
        "action_sampling_policy":
            action_proposal.DEPTH_CONDITIONED_POLICY_V3,
    })
    v3_context = source_manifest.r2r_v16_context_from_run_meta(
        v3_meta, authority_sha256="e" * 64)
    assert v3_context.expected_action_sampling_policy == \
        action_proposal.DEPTH_CONDITIONED_POLICY_V3


def _coordinated_b_forgery(rec):
    forged = copy.deepcopy(rec)
    geometry = forged["b_target"]["geometry"]
    support = geometry["ground_support"]
    support["triangles_xz_m"][0][0][0] += 0.25
    support["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in support.items() if key != "sha256"})
    centroid = geometry["reference_centroid"]
    centroid["world_xyz_m"][0] += 0.25
    centroid["world_xz_m"][0] += 0.25
    centroid["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in centroid.items() if key != "sha256"})
    geometry_payload = {key: geometry[key] for key in (
        "schema", "instance_id", "category", "semantic_ply_sha256",
        "full_triangle_protocol", "full_triangle_count",
        "full_triangles_sha256", "ground_support", "reference_centroid",
    )}
    geometry["sha256"] = record.canonical_atom_sha256(geometry_payload)
    forged["b_target"] = record.build_b_target_atom(
        selection=forged["b_target"]["selection"], geometry=geometry,
        pose=forged["pose"])
    outcome = forged["outcomes"][0]
    outcome["b_endpoint_relation"] = record.build_b_endpoint_relation_atom(
        pose=forged["pose"], outcome=outcome,
        b_target=forged["b_target"])
    return forged


def test_registered_validation_rederives_b_from_trusted_local_source(
        tmp_path, monkeypatch):
    rec, context, spec = _trusted_record(tmp_path)
    monkeypatch.setattr(
        scene_pool, "resolve_trusted_r2r_scene",
        lambda scene_id, **_kwargs: spec, raising=False)

    clean_errors = validate.validate_record_source_bound(
        rec, context=context)
    forged_errors = validate.validate_record_source_bound(
        _coordinated_b_forgery(rec), context=context)

    assert not any("trusted B target geometry" in error
                   for error in clean_errors)
    assert any("trusted B target geometry" in error
               for error in forged_errors)


def test_registered_validation_batches_endpoint_support_queries(
        tmp_path, monkeypatch):
    rec, context, spec = _trusted_record(tmp_path)
    second = copy.deepcopy(rec["outcomes"][0])
    second["outcome_id"] = f'{second["outcome_id"]}-second'
    rec["outcomes"].append(second)
    monkeypatch.setattr(
        scene_pool, "resolve_trusted_r2r_scene",
        lambda scene_id, **_kwargs: spec, raising=False)
    scalar = semantic.point_to_ground_support_distance_m
    calls = 0

    def counted_scalar(*args, **kwargs):
        nonlocal calls
        calls += 1
        return scalar(*args, **kwargs)

    monkeypatch.setattr(
        semantic, "point_to_ground_support_distance_m", counted_scalar)

    assert scene_pool.trusted_r2r_b_source_errors(rec, context) == []
    assert calls == 1


def test_registered_validation_fails_closed_without_trusted_local_source(
        tmp_path, monkeypatch):
    rec, context, _spec = _trusted_record(tmp_path)

    def missing(*_args, **_kwargs):
        raise scene_pool.SceneCatalogError("missing trusted scene")

    monkeypatch.setattr(scene_pool, "resolve_trusted_r2r_scene", missing, raising=False)

    assert any("trusted R2R source unavailable" in error
               for error in validate.validate_record_source_bound(
                   rec, context=context))


def test_registered_validation_hashes_house_bytes_used_for_categories(
        tmp_path, monkeypatch):
    rec, context, spec = _trusted_record(tmp_path)
    monkeypatch.setattr(
        scene_pool, "resolve_trusted_r2r_scene",
        lambda scene_id, **_kwargs: spec, raising=False)
    with open(spec.semantic_metadata_path, "a") as handle:
        handle.write("# changed\n")

    assert any("trusted R2R source" in error
               for error in validate.validate_record_source_bound(
                   rec, context=context))


def test_registered_validation_fails_closed_on_authority_memory_error(
        tmp_path, monkeypatch):
    rec, context, spec = _trusted_record(tmp_path)
    monkeypatch.setattr(
        scene_pool, "resolve_trusted_r2r_scene",
        lambda scene_id, **_kwargs: spec)

    def exhausted(*_args, **_kwargs):
        raise MemoryError("authority allocation failed")

    monkeypatch.setattr(semantic, "load_mp3d_target_authority", exhausted)

    assert any("trusted R2R source unavailable" in error
               for error in validate.validate_record_source_bound(
                   rec, context=context))
