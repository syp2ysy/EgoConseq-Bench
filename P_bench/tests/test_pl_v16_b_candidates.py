"""Frozen v16 B target selection and Closed Exact projection tests."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from pipeline import (
    benchmark, benchmark_builders, benchmark_tasks, config, objects, record,
    semantic, viz,
)
from tests._synthetic import LEVEL_FLOOR_FIT, make_frame, source_provenance
from tests.test_pl_v16_a_candidates import _case as _a_case
from tests.test_pl_v16_b_geometry import _write_target_scene


def _target_record():
    return {
        "sensor": {"resolution": [100, 100]},
        "objects": [
            {
                "instance_id": 9,
                "category": "chair",
                "is_structural": False,
                "mask_area_px": 140,
                "depth_backed_px": 140,
                "bbox_xyxy_px": [60, 20, 79, 39],
                "centroid_px": [70, 30],
                "dist_nearest_m": 2.5,
            },
            {
                "instance_id": 7,
                "category": "chair",
                "is_structural": False,
                "mask_area_px": 200,
                "depth_backed_px": 200,
                "bbox_xyxy_px": [20, 20, 39, 39],
                "centroid_px": [30, 30],
                "dist_nearest_m": 2.0,
            },
            {
                "instance_id": 3,
                "category": "mirror",
                "is_structural": False,
                "mask_area_px": 900,
                "depth_backed_px": 900,
                "bbox_xyxy_px": [10, 10, 49, 49],
                "centroid_px": [30, 30],
                "dist_nearest_m": 2.0,
            },
        ],
    }


def test_b_target_selection_is_deterministic_outcome_blind_and_numbered():
    """Catches endpoint facts changing the one target fixed from s0."""
    first = _target_record()
    second = copy.deepcopy(first)
    second["objects"].reverse()
    first["outcomes"] = [{"physical": {"collision": True}}]
    second["outcomes"] = [{
        "physical": {"collision": False},
        "execution": {"realized_pose": {
            "x": 99.0, "z": 99.0, "heading_deg": 99.0}},
    }]

    selected = objects.select_b_target(first)

    assert selected == objects.select_b_target(second)
    assert selected["instance_id"] == 7
    assert selected["category"] == "chair"
    assert selected["name"] == "chair 1"
    assert selected["marker"] == {
        "kind": "numbered_dot", "number": 1,
        "center_xy": [30.0, 30.0],
    }
    assert selected["selection_protocol"] == "b-s0-target-selection.v1"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"mask_area_px": 99, "depth_backed_px": 99},
         "depth_backed_area_too_small"),
        ({"bbox_xyxy_px": [20, 20, 26, 39]}, "bbox_too_small"),
        ({"dist_nearest_m": 0.999}, "visible_distance_out_of_range"),
        ({"dist_nearest_m": 5.001}, "visible_distance_out_of_range"),
        ({"is_structural": True}, "target_category_ineligible"),
        ({"category": "glass door"}, "target_category_ineligible"),
    ],
)
def test_b_target_selection_fails_closed_at_preregistered_s0_gates(
        change, reason):
    """Catches small, shallow, structural, or unstable targets leaking in."""
    record = _target_record()
    record["objects"] = [{**record["objects"][1], **change}]

    result = objects.select_b_target(record)

    assert result == {"eligible": False, "reason": reason}


def test_b_target_selector_requires_explicit_depth_backed_pixel_count():
    """Catches mask pixels being mislabeled as valid-depth evidence."""
    record = _target_record()
    target = dict(record["objects"][1])
    target.pop("depth_backed_px")
    record["objects"] = [target]

    assert objects.select_b_target(record) == {
        "eligible": False, "reason": "depth_backed_area_missing"}


@pytest.mark.parametrize(
    "change",
    [
        {"bbox_xyxy_px": [-1, 10, 20, 30]},
        {"bbox_xyxy_px": [10, 10, 100, 30]},
        {"centroid_px": [101, 20]},
        {"centroid_px": [9, 20]},
        {"mask_area_px": 200.5},
        {"depth_backed_px": 201},
        {"mask_area_px": 401, "depth_backed_px": 200,
         "bbox_xyxy_px": [10, 10, 29, 29]},
    ],
)
def test_b_target_visible_facts_are_raster_bounded_and_consistent(change):
    """Catches impossible masks/bboxes/centroids, including non-square input."""
    rec = _target_record()
    rec["sensor"]["resolution"] = [100, 60]
    target = {**rec["objects"][1], **change}
    target.update({key: change.get(key, target[key]) for key in change})
    rec["objects"] = [target]

    assert objects.select_b_target(rec)["eligible"] is False


def _geometry():
    support = {
        "protocol": "full-triangle-floor-slab-xz-union.v1",
        "frame": "habitat_world_xz",
        "ground_band_m": [0.05, 0.30],
        "triangles_xz_m": [[[-0.2, -2.0], [0.0, -2.2], [0.2, -2.0]]],
        "segments_xz_m": [],
        "points_xz_m": [],
    }
    support["sha256"] = record.canonical_atom_sha256(support)
    centroid = {
        "protocol": "full-triangle-area-weighted-centroid.v1",
        "frame": "habitat_world_xyz",
        "world_xyz_m": [0.0, 0.5, -2.0],
        "world_xz_m": [0.0, -2.0],
    }
    centroid["sha256"] = record.canonical_atom_sha256(centroid)
    value = {
        "schema": "b-target-geometry.v1",
        "instance_id": 7,
        "category": "chair",
        "semantic_ply_sha256": next(
            value["sha256"] for value in _a_case(collision=False)[0][
                "source"]["source_assets"] if value["role"] == "semantic"),
        "full_triangle_protocol": "mp3d-instance-triangles.v1",
        "full_triangle_count": 12,
        "full_triangles_sha256": "f" * 64,
        "ground_support": support,
        "reference_centroid": centroid,
    }
    return {**value, "sha256": record.canonical_atom_sha256(value)}


def _b_case(*, duplicate=False, endpoint=(0.0, 1.0, 0.0)):
    rec, outcome = _a_case(collision=False, duplicate=duplicate)
    rec["pose"] = {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0}
    rec["objects"] = _target_record()["objects"][:2] if duplicate else [
        _target_record()["objects"][1]]
    rec["sensor"]["resolution"] = [100, 100]
    selected = objects.select_b_target(rec)
    rec["b_target"] = record.build_b_target_atom(
        selection=selected, geometry=_geometry(), pose=rec["pose"])
    x_m, z_m, heading_deg = endpoint
    outcome["execution"] = {
        "completed": True, "stop_reason": "completed",
        "nominal_forward_m": 2.0, "executed_forward_m": 2.0,
        "realized_pose": {
            "x": x_m, "z": z_m, "heading_deg": heading_deg,
        },
    }
    outcome["b_endpoint_relation"] = record.build_b_endpoint_relation_atom(
        pose=rec["pose"], outcome=outcome, b_target=rec["b_target"])
    return rec, outcome


def test_b_contract_has_only_two_frozen_questions():
    assert benchmark.B_TASK_QUESTIONS == {
        "B1_endpoint_distance":
            "After safely completing all the actions, how far will the robot "
            "be from {target}?",
        "B2_endpoint_direction":
            "After safely completing all the actions, where will {target} be "
            "relative to the robot’s final facing direction?",
    }


def test_b1_uses_exact_support_not_sampled_cache_and_has_four_choices():
    """Catches anything but exact source support changing B1 GT or choices."""
    rec, outcome = _b_case()

    eligibility = benchmark_tasks.b_candidate_eligibility(
        "B1_endpoint_distance", rec, outcome)
    answer, choices, certificate = benchmark_tasks.b_candidate_answer(
        "B1_endpoint_distance", rec, outcome)

    assert eligibility.eligible is True
    assert eligibility.margins["display_quantization_error_m"] == \
        certificate["display_quantization_error_m"]
    assert certificate["precise_distance_m"] == pytest.approx(1.0)
    assert certificate["display_quantization_error_m"] < \
        certificate["display_error_tolerance_m"]
    assert len(choices) == 4
    assert len({choice["id"] for choice in choices}) == 4
    values = sorted(choice["value_m"] for choice in choices)
    assert all(b - a >= certificate["minimum_separation_m"] - 1e-12
               for a, b in zip(values, values[1:]))
    assert all(abs(value - certificate["displayed_distance_m"]) <=
               3 * certificate["minimum_separation_m"] + 1e-12
               for value in values)
    assert answer in {choice["id"] for choice in choices}


def test_b1_samples_truth_numeric_rank_before_distractor_values():
    """Catches the old construction's 90% middle-rank blind shortcut."""
    counts = {rank: 0 for rank in range(1, 5)}
    for index in range(4000):
        answer, choices, certificate = benchmark_tasks._b1_metric_choices(
            2.0, seed=f"rank-audit-{index}")
        ordered = sorted(choices, key=lambda value: value["value_m"])
        rank = next(
            item_index for item_index, choice in enumerate(ordered, 1)
            if choice["id"] == answer)
        assert rank == certificate["truth_numeric_rank_1based"]
        assert certificate["negative_choice_count"] == rank - 1
        counts[rank] += 1

    # SHA-256 modulo-four selection is not forced to exact counts, but a
    # substantial drift would immediately resurrect a numeric-rank shortcut.
    assert all(900 <= count <= 1100 for count in counts.values())


def test_b1_samples_only_physically_feasible_ranks_for_low_distances():
    for index in range(100):
        _answer, choices, certificate = \
            benchmark_tasks._b1_metric_choices(
                0.10, seed=f"low-distance-{index}")
        assert all(choice["value_m"] >= 0.0 for choice in choices)
        assert certificate["feasible_truth_numeric_ranks_1based"] == [1]
        assert certificate["truth_numeric_rank_1based"] == 1


def test_b1_v3_allows_only_choices_plausible_with_display_tolerance():
    answer, choices, certificate = benchmark_tasks._b1_metric_choices(
        0.05, seed="lower-bound")

    assert answer in {choice["id"] for choice in choices}
    assert certificate["protocol"] == \
        "b1-metric-choices-rank-balanced.v3"
    assert certificate["plausible_lower_bound_m"] == 0.05
    assert certificate["plausibility_tolerance_m"] == 0.006
    assert all(
        choice["value_m"] + certificate["plausibility_tolerance_m"] >=
        certificate["plausible_lower_bound_m"]
        for choice in choices)

@pytest.mark.parametrize(
    ("bearing", "label"), [(0, "front"), (-90, "left"),
                            (90, "right"), (180, "rear")])
def test_b2_has_all_four_sectors(bearing, label):
    assert record.b_direction_with_margin(bearing) == (label, 45.0)


@pytest.mark.parametrize("bearing", [45, -45, 135, -135, 44.999, -59.999])
def test_b2_rejects_exact_or_near_sector_boundaries(bearing):
    with pytest.raises(ValueError, match="sector boundary"):
        record.b_direction_with_margin(bearing)


def test_b_projection_rejects_tampered_relation_and_target_identity():
    rec, outcome = _b_case()
    outcome["b_endpoint_relation"]["distance_after_m"] += 0.1
    assert benchmark_tasks.b_candidate_eligibility(
        "B1_endpoint_distance", rec, outcome).reason == \
        "b_endpoint_relation_invalid"

    rec, outcome = _b_case()
    rec["b_target"]["selection"]["instance_id"] = 9
    assert benchmark_tasks.b_candidate_eligibility(
        "B2_endpoint_direction", rec, outcome).reason == \
        "b_target_invalid"


def test_b_rejects_nominal_outcome_tamper_outside_stability_rows():
    """Catches a rebuilt certificate being accepted against a changed outcome."""
    rec, outcome = _b_case()
    outcome["depth_physical"]["collision"] = True

    result = benchmark_tasks.b_candidate_eligibility(
        "B1_endpoint_distance", rec, outcome)

    assert result.eligible is False
    assert result.reason == "shared_oracle_stability_invalid"


def test_b_rejects_a_duplicate_target_whose_marker_cannot_fit():
    """An edge marker cannot identify the numbered target without clipping."""
    rec, outcome = _b_case(duplicate=True)
    sibling = next(
        value for value in rec["objects"]
        if value["instance_id"] != rec["b_target"]["selection"]["instance_id"])
    sibling["bbox_xyxy_px"] = [0, 20, 19, 39]
    sibling["centroid_px"] = [5, 30]

    result = benchmark_tasks.b_candidate_eligibility(
        "B1_endpoint_distance", rec, outcome)

    assert result.eligible is False
    assert result.reason == "b_numbered_dot_unrenderable"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("extra",), 1),
        (("full_triangle_protocol",), "wrong"),
        (("full_triangles_sha256",), "A" * 64),
        (("ground_support", "frame"), "wrong"),
        (("ground_support", "ground_band_m"), [0.0, 0.3]),
        (("ground_support", "segments_xz_m"), [[1.0, 2.0, 3.0]]),
        (("reference_centroid", "world_xz_m"), [0.0]),
        (("reference_centroid", "world_xyz_m"), [0.0, float("nan"), 1.0]),
    ],
)
def test_b_geometry_validation_rejects_noncanonical_fields(path, value):
    rec, _outcome = _b_case()
    geometry = rec["b_target"]["geometry"]
    cursor = geometry
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    if path[0] in {"ground_support", "reference_centroid"}:
        child = geometry[path[0]]
        child["sha256"] = record.canonical_atom_sha256({
            key: item for key, item in child.items() if key != "sha256"})
    payload = {key: geometry.get(key) for key in (
        "schema", "instance_id", "category", "semantic_ply_sha256",
        "full_triangle_protocol", "full_triangle_count",
        "full_triangles_sha256", "ground_support", "reference_centroid",
    )}
    geometry["sha256"] = record.canonical_atom_sha256(payload)

    assert record._valid_b_geometry(geometry) is False


@pytest.mark.parametrize("offset", [0.0, 1e-8])
def test_centroid_degeneracy_preserves_b1_and_withholds_only_b2(
        offset, tmp_path):
    rec, outcome = _b_case()
    endpoint = outcome["b_endpoint_relation"]["endpoint_world_xz_m"]
    centroid = rec["b_target"]["geometry"]["reference_centroid"]
    centroid["world_xz_m"] = [endpoint[0] + offset, endpoint[1]]
    centroid["world_xyz_m"] = [endpoint[0] + offset, 0.5, endpoint[1]]
    centroid["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in centroid.items() if key != "sha256"})
    geometry = rec["b_target"]["geometry"]
    geometry["sha256"] = record.canonical_atom_sha256({
        key: geometry[key] for key in (
            "schema", "instance_id", "category", "semantic_ply_sha256",
            "full_triangle_protocol", "full_triangle_count",
            "full_triangles_sha256", "ground_support", "reference_centroid",
        )})
    rec["b_target"] = record.build_b_target_atom(
        selection=rec["b_target"]["selection"], geometry=geometry,
        pose=rec["pose"])
    relation = record.build_b_endpoint_relation_atom(
        pose=rec["pose"], outcome=outcome, b_target=rec["b_target"])
    outcome["b_endpoint_relation"] = relation

    assert relation["direction_status"] == "undefined_centroid_range"
    assert relation["centroid_range_m"] == pytest.approx(offset)
    assert relation["bearing_after_deg"] is None
    assert relation["direction"] is None
    assert relation["sector_boundary_margin_deg"] is None
    assert benchmark_tasks.b_candidate_eligibility(
        "B1_endpoint_distance", rec, outcome).eligible is True
    b2 = benchmark_tasks.b_candidate_eligibility(
        "B2_endpoint_direction", rec, outcome)
    assert b2.eligible is False
    assert b2.reason == "centroid_anchor_range_too_small"

    raw = tmp_path / "raw.png"
    Image.new("RGB", (100, 100), (1, 2, 3)).save(raw)
    _item, private = benchmark_builders.build_b_candidate(
        task_id="B1_endpoint_distance", record=rec, outcome=outcome,
        image=str(raw), expected_raw_image_sha256=hashlib.sha256(
            raw.read_bytes()).hexdigest())
    assert private["precise_bearing_deg"] is None


def test_closed_exact_b_builders_authenticate_rgb_and_render_target_dot(
        tmp_path):
    rec, outcome = _b_case(duplicate=True)
    raw = tmp_path / "raw.png"
    Image.new("RGB", (100, 100), (10, 20, 30)).save(raw)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()

    b1, private_b1 = benchmark_builders.build_b_candidate(
        task_id="B1_endpoint_distance", record=rec, outcome=outcome,
        image=str(raw), expected_raw_image_sha256=digest,
        asset_dir=tmp_path / "assets")
    b2, private_b2 = benchmark_builders.build_b_candidate(
        task_id="B2_endpoint_direction", record=rec, outcome=outcome,
        image=str(raw), expected_raw_image_sha256=digest,
        asset_dir=tmp_path / "assets")

    assert b1["question"].endswith("from chair 1?")
    assert b2["question"].startswith(
        "After safely completing all the actions, where will chair 1")
    assert len(b1["choices"]) == 4
    assert [choice["id"] for choice in b2["choices"]] == [
        "front", "left", "right", "rear"]
    assert b1["task_metadata"] == {
        "target_geometry_protocol":
            semantic.B_GROUND_SUPPORT_PROTOCOL,
    }
    assert b2["task_metadata"] == {
        "target_geometry_protocol":
            semantic.B_REFERENCE_CENTROID_PROTOCOL,
    }
    assert private_b1["precise_distance_m"] == pytest.approx(1.0)
    assert private_b2["precise_bearing_deg"] == pytest.approx(0.0)
    marked = Path(b1["model_input"]["initial_rgb"])
    assert marked != raw and marked.exists()
    assert Image.open(marked).getpixel((30, 30)) != (10, 20, 30)
    assert private_b1["input_asset"]["raw_sha256"] == digest
    assert private_b2["input_asset"]["raw_sha256"] == digest

    with pytest.raises(ValueError, match="digest"):
        benchmark_builders.build_b_candidate(
            task_id="B1_endpoint_distance", record=rec, outcome=outcome,
            image=str(raw), expected_raw_image_sha256="0" * 64,
            asset_dir=tmp_path / "assets")


def test_b_builders_reuse_one_authenticated_and_encoded_target_image(
        tmp_path, monkeypatch):
    rec, outcome = _b_case(duplicate=True)
    raw = tmp_path / "raw.png"
    Image.new("RGB", (100, 100), (10, 20, 30)).save(raw)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    authenticated = viz.authenticate_raw_rgb_image(
        raw, expected_sha256=digest, expected_resolution=[100, 100])
    encoded_cache = {}
    original_save = Image.Image.save
    encoded = 0
    compression_levels = []

    def counted_save(image, *args, **kwargs):
        nonlocal encoded
        if kwargs.get("format") == "PNG":
            encoded += 1
            compression_levels.append(kwargs.get("compress_level"))
        return original_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", counted_save)
    first, _private = benchmark_builders.build_b_candidate(
        task_id="B1_endpoint_distance", record=rec, outcome=outcome,
        image=str(raw), expected_raw_image_sha256=digest,
        asset_dir=tmp_path / "assets", authenticated_rgb=authenticated,
        numbered_dot_cache=encoded_cache)
    second, _private = benchmark_builders.build_b_candidate(
        task_id="B2_endpoint_direction", record=rec, outcome=outcome,
        image=str(raw), expected_raw_image_sha256=digest,
        asset_dir=tmp_path / "assets", authenticated_rgb=authenticated,
        numbered_dot_cache=encoded_cache)

    first_path = Path(first["model_input"]["initial_rgb"])
    second_path = Path(second["model_input"]["initial_rgb"])
    assert first_path != second_path
    assert first_path.read_bytes() == second_path.read_bytes()
    assert os.stat(first_path).st_ino == os.stat(second_path).st_ino
    assert encoded == 1
    assert compression_levels == [config.QA_MARKED_PNG_COMPRESSION_LEVEL]


def test_b_builders_reuse_one_authenticated_outcome_evidence(
        tmp_path, monkeypatch):
    rec, outcome = _b_case(duplicate=True)
    raw = tmp_path / "raw.png"
    Image.new("RGB", (100, 100), (10, 20, 30)).save(raw)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    evidence = benchmark_tasks.build_b_outcome_evidence(rec, outcome)

    def repeated_validation(*_args, **_kwargs):
        raise AssertionError("B geometry was authenticated more than once")

    monkeypatch.setattr(
        record, "authenticated_b_target", repeated_validation)
    monkeypatch.setattr(
        record, "b_endpoint_relation_atom_valid", repeated_validation)
    for task_id in ("B1_endpoint_distance", "B2_endpoint_direction"):
        assert benchmark_tasks.b_candidate_eligibility(
            task_id, rec, outcome, evidence=evidence).eligible is True
        benchmark_builders.build_b_candidate(
            task_id=task_id, record=rec, outcome=outcome,
            image=str(raw), expected_raw_image_sha256=digest,
            asset_dir=tmp_path / "assets", b_outcome_evidence=evidence)


def test_source_validated_b_evidence_does_not_replay_geometry(monkeypatch):
    rec, outcome = _b_case(duplicate=True)

    def repeated_validation(*_args, **_kwargs):
        raise AssertionError("source-validated B geometry was replayed")

    monkeypatch.setattr(
        record, "authenticated_b_target", repeated_validation)
    monkeypatch.setattr(
        record, "b_endpoint_relation_atom_valid", repeated_validation)
    record_evidence = benchmark_tasks.build_b_record_evidence(
        rec, source_validated=True)
    outcome_evidence = benchmark_tasks.build_b_outcome_evidence(
        rec, outcome, record_evidence=record_evidence,
        source_validated=True)

    assert record_evidence.target == rec["b_target"]
    assert outcome_evidence.rejection is None


def test_validated_target_endpoint_builder_skips_target_revalidation(
        monkeypatch):
    rec, outcome = _b_case(duplicate=True)
    expected = outcome["b_endpoint_relation"]

    def repeated_validation(*_args, **_kwargs):
        raise AssertionError("validated B target was revalidated")

    monkeypatch.setattr(record, "b_target_atom_valid", repeated_validation)
    rebuilt = record.build_b_endpoint_relation_from_validated_target(
        pose=rec["pose"], outcome=outcome, b_target=rec["b_target"])

    assert rebuilt == expected


def test_validated_target_endpoint_builder_accepts_batched_distance(
        monkeypatch):
    rec, outcome = _b_case(duplicate=True)
    expected = outcome["b_endpoint_relation"]

    def scalar_distance(*_args, **_kwargs):
        raise AssertionError("batched source distance was recomputed")

    monkeypatch.setattr(
        semantic, "point_to_ground_support_distance_m", scalar_distance)
    rebuilt = record.build_b_endpoint_relation_from_validated_target(
        pose=rec["pose"], outcome=outcome, b_target=rec["b_target"],
        distance_after_m=expected["distance_after_m"])

    assert rebuilt == expected


def test_unique_b_target_keeps_authenticated_raw_rgb(tmp_path):
    rec, outcome = _b_case(duplicate=False)
    raw = tmp_path / "raw.png"
    Image.new("RGB", (100, 100), (1, 2, 3)).save(raw)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()

    item, private = benchmark_builders.build_b_candidate(
        task_id="B2_endpoint_direction", record=rec, outcome=outcome,
        image=str(raw), expected_raw_image_sha256=digest,
        asset_dir=tmp_path / "assets")

    assert item["model_input"]["initial_rgb"] == str(raw)
    assert private["input_asset"]["marked"] is False


def _source_with_semantic_sha(source, path):
    value = copy.deepcopy(source)
    for asset in value["source_assets"]:
        if asset["role"] == "semantic":
            payload = path.read_bytes()
            asset["bytes"] = len(payload)
            asset["sha256"] = hashlib.sha256(payload).hexdigest()
    identity = {
        "version": value["source_asset_identity_version"],
        "assets": value["source_assets"],
    }
    value["source_assets_sha256"] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")).hexdigest()
    return value


def _record_outcome():
    _rec, outcome = _a_case(collision=False)
    outcome["execution"] = {
        "completed": True, "stop_reason": "completed",
        "nominal_forward_m": 2.0, "executed_forward_m": 2.0,
        "realized_pose": {"x": 0.0, "z": 1.0, "heading_deg": 0.0},
    }
    return outcome


def test_build_record_materializes_one_source_pinned_target_and_closed_b(
        tmp_path):
    """Proves a strict authenticated record can emit real Closed Exact B."""
    scene, semantic_ply = _write_target_scene(tmp_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    frame = make_frame()
    frame.scene_glb = str(scene)
    frame.semantic_index = index
    frame.id_to_cat = {1: "chair"}
    frame.objects = [{
        "instance_id": 1, "category": "chair", "is_structural": False,
        "mask_area_px": 4000, "depth_backed_px": 4000,
        "bbox_xyxy_px": [200, 150, 439, 329],
        "centroid_px": [320, 240], "dist_nearest_m": 2.0,
    }]
    source = _source_with_semantic_sha(
        source_provenance("TARGET", dataset="r2r"), semantic_ply)
    outcome = _record_outcome()

    rec = record.build_record(
        frame, [outcome], image_path="images/target.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source,
        collection_contract=record.r2r_v16_collection_contract(
            source, "main"))
    assert rec["b_target"]["selection"]["instance_id"] == 1
    assert rec["b_target"]["geometry"]["semantic_ply_sha256"] == \
        hashlib.sha256(semantic_ply.read_bytes()).hexdigest()
    stored = rec["outcomes"][0]
    assert stored["b_endpoint_relation"]["target_geometry_sha256"] == \
        rec["b_target"]["geometry"]["sha256"]
    assert benchmark_tasks.b_candidate_eligibility(
        "B1_endpoint_distance", rec, stored).eligible is True
    assert benchmark_tasks.b_candidate_eligibility(
        "B2_endpoint_direction", rec, stored).eligible is True
    raw = tmp_path / "raw.png"
    Image.new("RGB", (640, 480), (8, 9, 10)).save(raw)
    item, private = benchmark_builders.build_b_candidate(
        task_id="B1_endpoint_distance", record=rec, outcome=stored,
        image=str(raw), expected_raw_image_sha256=hashlib.sha256(
            raw.read_bytes()).hexdigest())
    assert item["answer_format"] == "closed_exact"
    assert len(item["choices"]) == 4
    assert private["canonical_answer"] in {
        choice["id"] for choice in item["choices"]}


def test_build_record_never_falls_back_after_selected_geometry_failure():
    """Catches exact failure causing an endpoint-aware second target query."""
    class FailingAuthority:
        def __init__(self):
            self.queries = []

        def target_geometry_atom(self, instance_id, *_args, **_kwargs):
            self.queries.append(instance_id)
            raise ValueError("selected geometry failed")

    frame = make_frame()
    authority = FailingAuthority()
    frame.semantic_index = authority
    frame.objects = [
        {
            "instance_id": 1, "category": "chair", "is_structural": False,
            "mask_area_px": 5000, "depth_backed_px": 5000,
            "bbox_xyxy_px": [100, 100, 299, 299],
            "centroid_px": [200, 200], "dist_nearest_m": 2.0,
        },
        {
            "instance_id": 2, "category": "table", "is_structural": False,
            "mask_area_px": 4000, "depth_backed_px": 4000,
            "bbox_xyxy_px": [300, 100, 499, 299],
            "centroid_px": [400, 200], "dist_nearest_m": 2.0,
        },
    ]
    source = source_provenance("TARGET", dataset="r2r")

    rec = record.build_record(
        frame, [_record_outcome()], image_path="images/target.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source,
        collection_contract=record.r2r_v16_collection_contract(
            source, "main"))
    assert authority.queries == [1]
    assert "b_target" not in rec
    assert rec["b_target_withhold"] == {
        "reason": "exact_target_geometry_failed",
        "selected_instance_id": 1,
    }
    assert "b_endpoint_relation" not in rec["outcomes"][0]


def test_build_record_does_not_materialize_b_target_for_other_modes():
    """Catches broadening B-target collection beyond strict R2R main."""
    frame = make_frame()
    frame.objects = _target_record()["objects"][:1]
    source = source_provenance("TARGET", dataset="r2r")
    contract = record.r2r_v16_collection_contract(source, "main")
    contract["collection_mode"] = "other"

    rec = record.build_record(
        frame, [], image_path="images/target.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source, collection_contract=contract)

    assert "b_target" not in rec
    assert "b_target_withhold" not in rec


def test_build_record_withholds_memory_error_without_trying_second_target():
    class ExhaustedAuthority:
        def __init__(self):
            self.queries = []

        def target_geometry_atom(self, instance_id, *_args, **_kwargs):
            self.queries.append(instance_id)
            raise MemoryError("mesh too large")

    frame = make_frame()
    authority = ExhaustedAuthority()
    frame.semantic_index = authority
    frame.objects = _target_record()["objects"][:2]
    # Adapt the synthetic facts to the frame's native 640x480 raster.
    for index, target in enumerate(frame.objects):
        target.update({
            "mask_area_px": 5000 - index * 500,
            "depth_backed_px": 5000 - index * 500,
            "bbox_xyxy_px": [100 + index * 250, 100,
                              299 + index * 250, 299],
            "centroid_px": [200 + index * 250, 200],
        })
    source = source_provenance("TARGET", dataset="r2r")

    rec = record.build_record(
        frame, [_record_outcome()], image_path="images/target.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source,
        collection_contract=record.r2r_v16_collection_contract(
            source, "main"))

    assert authority.queries == [9]
    assert rec["b_target_withhold"]["reason"] == \
        "exact_target_geometry_failed"
