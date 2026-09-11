"""GS-only v18 record contract backed by official collision geometry."""

from __future__ import annotations

import hashlib
import json

from pipeline import dataset_contracts, gs_semantic, record
from tests._synthetic import (
    LEVEL_FLOOR_FIT, make_frame, source_provenance,
)


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _v18_frame_and_source():
    frame = make_frame()
    source = source_provenance(frame.scene_id, dataset="gs")
    certificate = {
        "schema": gs_semantic.ALIGNMENT_CERTIFICATE_SCHEMA,
        "accepted": True,
    }
    certificate["sha256"] = _sha(certificate)
    base = gs_semantic.BboxSemanticIndex(
        mins=[[-1.0, -1.0, 0.5]], maxs=[[1.0, 1.0, 3.0]],
        ids=[7], id_to_cat={7: "chair"},
        alignment_certificate=certificate)
    binding = dataset_contracts.resolve_gs_collision_binding(source)
    semantic_sha256 = next(
        row["sha256"] for row in source["source_assets"]
        if row["role"] == "semantic")
    frame.semantic_index = base.visible_depth_view(
        frame.pts, frame.pts_sem,
        geometry_authority_sha256=binding.authority_sha256,
        semantic_source_sha256=semantic_sha256)
    return frame, source, binding


def test_v18_empty_record_round_trips_without_gaussian_authority():
    frame, source, binding = _v18_frame_and_source()

    candidate = record.build_record_v18(
        frame, [], image_path="img/f.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source)

    assert candidate["schema_version"] == "conseq.v18"
    assert candidate["gs_collision_authority"] == \
        dataset_contracts.gs_collision_binding_atom(binding)
    assert "gs_geometry_authority" not in candidate
    assert "collection_contract" not in candidate
    assert record.validate_record_v18(candidate) == []
    payload = json.dumps(candidate, sort_keys=True) + "\n"
    assert record.decode_v18_records(payload) == [candidate]


def test_v18_rejects_collision_authority_substitution():
    frame, source, _binding = _v18_frame_and_source()
    candidate = record.build_record_v18(
        frame, [], image_path="img/f.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source)

    candidate["gs_collision_authority"]["collision_authority_sha256"] = \
        "f" * 64

    assert any("collision authority differs" in error
               for error in record.validate_record_v18(candidate))


def test_conseq_v17_has_no_decoder_or_compatibility_route():
    try:
        record.decode_records_for_schema("", "conseq.v17")
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("obsolete GS conseq.v17 remained readable")
