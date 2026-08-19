"""GS source alignment and diagnostic-only Gaussian quality tests."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from pipeline import (
    config, consequence, dataset_contracts, gs_semantic, gs_source,
    source_manifest,
)
from tests._synthetic import source_provenance


def _box(center, half_extents):
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half_extents, dtype=np.float64)
    signs = np.asarray([
        [x, y, z]
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ])
    return center + signs * half


def _correct_transform(points):
    points = np.asarray(points, dtype=np.float64)
    return np.stack(
        [points[..., 0], points[..., 2], -points[..., 1]], axis=-1)


def _alignment_fixture():
    boxes = np.stack([
        _box([-2.0, -0.5, 0.0], [0.20, 0.35, 0.55]),
        _box([0.5, 1.5, 0.8], [0.30, 0.45, 0.20]),
        _box([2.2, -1.0, 1.7], [0.45, 0.20, 0.30]),
    ])
    means = []
    for box in boxes:
        lo, hi = box.min(axis=0), box.max(axis=0)
        fractions = np.asarray([
            [0.20, 0.25, 0.30],
            [0.35, 0.70, 0.55],
            [0.65, 0.40, 0.75],
            [0.80, 0.80, 0.20],
        ])
        means.append(_correct_transform(lo + fractions * (hi - lo)))
    return boxes, np.arange(1, 4), np.vstack(means), np.ones(12)


def _alignment_certificate():
    boxes, ids, means, opacities = _alignment_fixture()
    certificate = gs_semantic.build_alignment_certificate(boxes, ids)
    assert certificate["accepted"] is True
    return certificate


def _write_gaussian_ply(path, means, opacities, *, opacity_logits=None):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"),
        ("rot_2", "f4"), ("rot_3", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
    ]
    rows = np.zeros(len(means), dtype=dtype)
    rows["x"], rows["y"], rows["z"] = np.asarray(means).T
    for name in ("scale_0", "scale_1", "scale_2"):
        rows[name] = np.log(0.05)
    rows["rot_0"] = 1.0
    if opacity_logits is None:
        probability = np.clip(np.asarray(opacities), 1e-6, 1.0 - 1e-6)
        rows["opacity"] = np.log(probability / (1.0 - probability))
    else:
        rows["opacity"] = np.asarray(opacity_logits, dtype=np.float32)
    PlyData([PlyElement.describe(rows, "vertex")]).write(str(path))


def _write_alignment_sources(tmp_path):
    boxes, ids, means, opacities = _alignment_fixture()
    gaussian_path = tmp_path / "scene.gs.ply"
    labels_path = tmp_path / "labels.json"
    _write_gaussian_ply(gaussian_path, means, opacities)
    labels_path.write_text(json.dumps([
        {
            "ins_id": int(instance_id),
            "label": f"object_{instance_id}",
            "bounding_box": [
                {"x": float(x), "y": float(y), "z": float(z)}
                for x, y, z in box
            ],
        }
        for box, instance_id in zip(boxes, ids)
    ]))
    return gaussian_path, labels_path


def test_gs_source_reader_rederives_v18_scene_capability(tmp_path):
    gaussian_path, labels_path = _write_alignment_sources(tmp_path)
    source = source_provenance("gs-scene", dataset="gs")
    arrays = gs_source.read_gaussian_source(gaussian_path)
    capability = gs_semantic.derive_scene_capability_from_sources(
        source, labels_path)
    context = source_manifest.gs_v18_registered_validation_context(
        [source], authority_sha256="d" * 64,
        scene_capabilities={source["scene_id"]: capability})

    assert arrays["means"].shape == (12, 3)
    assert arrays["scales"] == pytest.approx(0.05)
    assert capability["alignment_certificate"]["accepted"] is True
    assert context.gs_scene_capability_resolver(source["scene_id"]) == \
        capability


def test_gs_source_reader_accepts_saturated_opacity_logits(tmp_path):
    path = tmp_path / "saturated-opacity.gs.ply"
    _write_gaussian_ply(
        path, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        [0.5, 0.5], opacity_logits=[np.inf, -np.inf])
    assert gs_source.read_gaussian_source(path)["opacities"].tolist() == \
        [1.0, 0.0]


def test_gs_root_has_an_environment_override(tmp_path):
    environment = dict(os.environ)
    environment["EGOCONSEQ_GS_ROOT"] = str(tmp_path / "alternate-gs")
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output([
        sys.executable, "-c",
        "from pipeline import config; print(config.GS_ROOT); "
        "print(config.GS_TRAIN_MANIFEST)",
    ], cwd=root, env=environment, text=True).splitlines()
    assert output == [
        str(tmp_path / "alternate-gs"),
        str(tmp_path / "alternate-gs" / "splits" / "train.json"),
    ]


def test_gs_scene_diagnostics_rank_without_excluding_sources(tmp_path):
    gaussian_path, labels_path = _write_alignment_sources(tmp_path)
    source = source_provenance("thin-accepted", dataset="gs")
    diagnostics = gs_semantic.scene_diagnostics(
        source, gaussian_path, labels_path)
    rows = [
        {**diagnostics, "scene_id": "dense-thinner",
         "support_radius_p90_m": 0.01,
         "fraction_above_quality_band": 0.0,
         "gaussian_count": 1_000_000},
        {**diagnostics, "scene_id": "thick", "alignment_accepted": True,
         "support_radius_p90_m": 0.40,
         "fraction_above_quality_band": 0.20},
        diagnostics,
    ]
    assert diagnostics["gaussian_count"] == 12
    assert diagnostics["support_radius_p50_m"] == pytest.approx(0.10)
    assert diagnostics["support_radius_p90_m"] == pytest.approx(0.10)
    assert diagnostics["fraction_above_quality_band"] == 0.0
    assert [row["scene_id"] for row in gs_source.rank_scene_diagnostics(
        rows)] == ["dense-thinner", "thin-accepted", "thick"]


def test_official_coordinate_binding_is_fixed_and_rejects_tampering():
    boxes, ids, means, opacities = _alignment_fixture()
    certificate = gs_semantic.build_alignment_certificate(boxes, ids)
    assert certificate["accepted"] is True
    assert certificate["matrix"] == [
        [1, 0, 0], [0, 0, 1], [0, -1, 0]]
    assert certificate["translation_m"] == [0.0, 0.0, 0.0]
    assert gs_semantic.validate_alignment_certificate(
        boxes, ids, certificate) is None
    changed = json.loads(json.dumps(certificate))
    changed["translation_m"][0] = 1.0
    with pytest.raises(ValueError, match="certificate"):
        gs_semantic.validate_alignment_certificate(boxes, ids, changed)


def test_bbox_assignment_and_loader_fail_closed(tmp_path):
    index = gs_semantic.BboxSemanticIndex(
        mins=np.array([[0, 0, 0], [0.5, 0, 0]], dtype=np.float64),
        maxs=np.array([[1, 1, 1], [1.5, 1, 1]], dtype=np.float64),
        ids=np.array([7, 8]), id_to_cat={7: "chair", 8: "table"},
        surface_points=np.array([
            [0.25, 0.5, 0.5], [0.75, 0.5, 0.5], [1.25, 0.5, 0.5]]))
    assert index.assign(np.array([
        [0.25, 0.5, 0.5], [0.75, 0.5, 0.5], [1.25, 0.5, 0.5],
    ])).tolist() == [7, 0, 8]

    labels = [{
        "ins_id": value,
        "label": f"object_{value}",
        "bounding_box": [
            {"x": float(x), "y": float(y), "z": float(z)}
            for x, y, z in _box([offset, 0, 0], [0.5, 0.5, 0.5])],
    } for value, offset in ((41, 0.0), (66, 3.0))]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    loaded = gs_semantic.load_bbox_index(str(path))
    assert loaded.id_to_cat == {41: "object_41", 66: "object_66"}
    assert loaded.semantic_certified is True
    assert loaded.alignment_certificate["coordinate_authority"] == \
        "habitat-gs-interiorgs-objectnav.v1"
    labels[1]["ins_id"] = 41
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="instance ids"):
        gs_semantic.load_bbox_index(str(path))


@pytest.mark.parametrize("invalid_id", [None, True, 0, -1, "", "3.0", "x"])
def test_bbox_loader_rejects_invalid_source_instance_ids(tmp_path, invalid_id):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{
        "ins_id": invalid_id, "label": "object",
        "bounding_box": [
            {"x": float(x), "y": float(y), "z": float(z)}
            for x, y, z in _box([0, 0, 0], [0.5, 0.5, 0.5])],
    }]))
    with pytest.raises(ValueError, match="instance ids"):
        gs_semantic.load_bbox_index(str(path))


def test_symmetric_scene_does_not_disable_official_semantics(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{
        "ins_id": 1, "label": "cube",
        "bounding_box": [
            {"x": float(x), "y": float(y), "z": float(z)}
            for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
    }]))
    symmetric = np.array([
        [x, y, z]
        for x in (-0.5, 0.5)
        for y in (-0.5, 0.5)
        for z in (-0.5, 0.5)])
    index = gs_semantic.load_bbox_index(str(path))
    assert index.alignment_certificate["accepted"] is True
    assert index.semantic_certified is True
    habitat_points = _correct_transform(symmetric)
    assert index.assign(habitat_points).tolist() == [1] * len(symmetric)


def test_visible_depth_runtime_uses_the_registered_gs_identity_binding():
    boxes, ids, _means, _opacities = _alignment_fixture()
    certificate = gs_semantic.build_alignment_certificate(boxes, ids)
    transformed = _correct_transform(boxes)
    base = gs_semantic.BboxSemanticIndex(
        transformed.min(axis=1), transformed.max(axis=1), ids,
        {int(value): f"object_{value}" for value in ids},
        alignment_certificate=certificate)
    view = base.visible_depth_view(
        np.asarray([[0.0, 0.0, 0.0]]), np.asarray([1]),
        geometry_authority_sha256="a" * 64,
        semantic_source_sha256="b" * 64)

    assert consequence._runtime_authority_binding(view) == \
        dataset_contracts.AuthorityBinding(
            source_dataset="gs",
            identity_schema=gs_semantic.VISIBLE_CONTACT_IDENTITY_SCHEMA,
            authoritative_source_role="source_bundle",
            source_sha256="a" * 64,
            semantic_source_sha256="b" * 64)
