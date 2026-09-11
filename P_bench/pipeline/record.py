"""FrameRecord assembly + jsonl I/O (append-style; numpy -> python)."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from typing import Iterator, List

import numpy as np

from pipeline import (
    actions as action_geometry, config, consensus as consensus_fields,
    capability_contracts, dataset_contracts, gs_semantic,
    outcome as outcome_fields,
)
from pipeline.floor_plane import FloorPlaneEstimate, FloorPlaneFitResult


def stored_floor_plane(rec: dict):
    """Rebuild a record's canonical plane, or say why it cannot be rebuilt.
    Returns ``(plane, reason)`` with exactly one set. The single place anything
    turns a persisted record back into a floor: the validator reads it to check
    the published height, and the viewer reads it to rebuild a frame. Neither
    may re-fit -- that is the per-image estimate returning through the back door
    -- and neither may fall back to a level floor at the origin, which would
    answer floor-relative questions against a surface nobody measured.
    """
    calibration = rec.get("floor_calibration")
    if not isinstance(calibration, dict):
        return None, "floor calibration is missing"
    reasons = calibration.get("rejection_reasons")
    if not isinstance(reasons, list):
        return None, "floor calibration has no rejection_reasons list"
    if not all(isinstance(reason, str) for reason in reasons):
        return None, ("floor calibration rejection_reasons must all be "
                      f"strings: {reasons!r}")
    if reasons:
        return None, f"floor calibration was rejected: {sorted(reasons)}"
    estimate = calibration.get("estimate")
    if estimate is None:
        return None, "floor calibration carries no plane"
    # A persisted record is untrusted input, so every shape is checked here
    # rather than left to raise from inside a JSON accessor: a list, a string
    # or a number would otherwise surface as AttributeError instead of a
    # validation reason a caller can report.
    if not isinstance(estimate, dict):
        return None, f"floor calibration plane must be an object: {estimate!r}"
    try:
        return FloorPlaneEstimate.from_json(estimate), None
    except (ValueError, TypeError) as error:
        return None, f"floor calibration plane is invalid: {error}"


def require_floor_plane(rec: dict) -> FloorPlaneEstimate:
    """:func:`stored_floor_plane`, raising instead of reporting."""
    plane, reason = stored_floor_plane(rec)
    if plane is None:
        raise ValueError(
            f"{rec.get('frame_id')}: {reason}; cannot rebuild this frame")
    return plane


def authenticated_camera_height_above_visible_floor_m(rec: dict) -> float:
    """Rederive the legacy plane-relative oracle scalar (not public QA height).

    A record scalar is never authoritative on its own.  The only accepted
    value is ``n . (0, nominal_camera_offset_m, 0) + d`` from the record's
    accepted :class:`FloorPlaneEstimate`; the stored scalar is checked only as
    a second serialization of that identity.  JSON booleans, non-numbers and
    non-finite values are rejected before arithmetic.
    """
    if not isinstance(rec, dict):
        raise ValueError("camera-height record must be an object")
    sensor = rec.get("sensor")
    if not isinstance(sensor, dict):
        raise ValueError("record sensor is missing")
    nominal = sensor.get("nominal_camera_offset_m")
    if (isinstance(nominal, bool) or
            not isinstance(nominal, (int, float)) or
            not math.isfinite(float(nominal))):
        raise ValueError(
            "sensor nominal_camera_offset_m must be a finite JSON number")
    stored = rec.get("camera_height_above_visible_floor_m")
    if (isinstance(stored, bool) or
            not isinstance(stored, (int, float)) or
            not math.isfinite(float(stored))):
        raise ValueError(
            "camera_height_above_visible_floor_m must be a finite JSON number")
    plane = require_floor_plane(rec)
    derived = float(plane.height_above((0.0, float(nominal), 0.0)))
    if abs(float(stored) - derived) > \
            config.CAMERA_HEIGHT_CALIBRATION_TOLERANCE_M:
        raise ValueError(
            "camera_height_above_visible_floor_m does not match the "
            "record floor calibration")
    return derived

SCHEMA_VERSION = "conseq.v11"
V18_SCHEMA_VERSION = "conseq.v18"
ORACLE_CONTRACT_VERSION = "ground-disc-visible-v8"
BASE_ROLLOUT_KEY_VERSION = "base-rollout.v1"
R2R_V16_COLLECTION_CONTRACT_VERSION = "r2r-visible-space-abcd.v16"
B1K_V5_COLLECTION_CONTRACT_VERSION = "b1k-visible-space-abc1.v5"
B1K_V16_COLLECTION_CONTRACT_VERSION = "b1k-visible-space-abc1.v6"
B1K_OFFICIAL_RENDER_COLLECTION_CONTRACT_VERSION = \
    "b1k-visible-space-abc1.v7"
B1K_RENDERER_INSTANCE_PROTOCOL = "b1k-observation.v5"
# v5/v6 records keep the custom-renderer digest; v7 is official OG rendering.
B1K_LEGACY_OBSERVATION_PROFILE_SHA256 = \
    "b02ba45708cf607eda82ee9d01671805c79670f46d7ca342661e16df5b881088"
B1K_OBSERVATION_PROFILE_SHA256 = \
    "ae50ab14d51c0ccbc0eba398e06ecf79b9b9e5f612f684026e319bf483faee3a"
B1K_V5_C1_RENDER_MODE = "temporary-counterfactual-bank-batch.v1"
B1K_C1_RENDER_MODE = "persistent-counterfactual-bank-batch.v2"
B1K_C1_RENDER_MODE_BY_CONTRACT = {
    B1K_V5_COLLECTION_CONTRACT_VERSION: B1K_V5_C1_RENDER_MODE,
    B1K_V16_COLLECTION_CONTRACT_VERSION: B1K_C1_RENDER_MODE,
    B1K_OFFICIAL_RENDER_COLLECTION_CONTRACT_VERSION: B1K_C1_RENDER_MODE,
}
B1K_OBSERVATION_PROFILE_SHA256_BY_CONTRACT = {
    B1K_V5_COLLECTION_CONTRACT_VERSION:
        B1K_LEGACY_OBSERVATION_PROFILE_SHA256,
    B1K_V16_COLLECTION_CONTRACT_VERSION:
        B1K_LEGACY_OBSERVATION_PROFILE_SHA256,
    B1K_OFFICIAL_RENDER_COLLECTION_CONTRACT_VERSION:
        B1K_OBSERVATION_PROFILE_SHA256,
}
B1K_INSTANCE_TRIANGLES_SCHEMA = "b1k-runtime-instance-triangles.v1"


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        json_value(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_atom_sha256(value: dict) -> str:
    """Public name for canonical hashes used by persisted v16 atoms."""
    return _canonical_sha256(value)


def _authority_surface_capability(source_dataset: str) -> dict:
    value = capability_contracts.snapshot_fields(source_dataset)
    return {**value, "sha256": _canonical_sha256(value)}


def _type_exact_json_equal(left, right) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if (len(left) != len(right) or set(left) != set(right) or
                not all(isinstance(key, str) for key in left)):
            return False
        return all(
            _type_exact_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _type_exact_json_equal(first, second)
            for first, second in zip(left, right))
    return left == right


def _require_v18_gs_source_provenance(
        source: dict, *, scene_id: str) -> str:
    """Authenticate the exact four-asset GS source used by v18."""
    dataset_contracts.validate_source_asset_provenance(source)
    source_dataset = source.get("source_dataset")
    source_scene_id = source.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("v18 record scene_id must be a nonempty string")
    if not isinstance(source_scene_id, str) or not source_scene_id:
        raise ValueError("v18 source scene_id must be a nonempty string")
    if source_scene_id != scene_id:
        raise ValueError("v18 source scene_id does not match record")
    if source.get("official_split") not in \
            dataset_contracts.OFFICIAL_SOURCE_SPLITS:
        raise ValueError("v18 source split is unsupported")
    if source_dataset != "gs":
        raise ValueError("conseq.v18 is reserved for GS collection")
    manifest_sha256 = source.get("source_manifest_sha256")
    if (not isinstance(manifest_sha256, str) or
            not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)):
        raise ValueError("v18 source manifest sha256 is invalid")
    return source_dataset


def action_program_sha256(action_dicts) -> str:
    """The single record-atom digest of one action program.

    Published as ``terminal_rgb_asset.binding.action_sha256`` and used as the
    join key between C1 families and the QA choice descriptors, so every caller
    must reach it through this function rather than re-spelling the wrapper.

    Takes already-canonical action dicts rather than parsing them: the value is
    frozen in published records, and re-parsing could renormalise a stored float
    and silently move an identifier that golden pins.

    The bare-list digest that becomes the candidate tag is a different value and
    lives in ``pipeline.actions.canonical_actions_sha256``.
    """
    return _canonical_sha256({"actions": list(action_dicts)})


def b1k_scene_authority_sha256(source: dict) -> str:
    """Validate B1K provenance and return its canonical runtime authority."""
    binding = authority_binding(source)
    if binding.source_dataset != "b1k":
        raise ValueError("B1K source assets are invalid")
    return binding.source_sha256


def authority_binding(source: dict) -> dataset_contracts.AuthorityBinding:
    """Resolve one typed A3 identity authority from strict provenance."""
    if not isinstance(source, dict):
        raise ValueError("source provenance must be an object")
    for key in ("source_manifest_sha256", "source_assets_sha256"):
        value = source.get(key)
        if (not isinstance(value, str) or
                not re.fullmatch(r"[0-9a-f]{64}", value)):
            raise ValueError(f"invalid {key}")
    return dataset_contracts.resolve_authority_binding(source)



def r2r_v16_collection_contract(
        source_provenance: dict, collection_mode: str, *,
        record_schema_version: str = SCHEMA_VERSION) -> dict:
    """Build the hash-bound record route for strict R2R main oracles.

    v11 is the active R2R route.
    """
    source = source_provenance or {}
    mode = str(collection_mode)
    source_contract = dataset_contracts.dataset_source_contract("r2r")
    if mode != "main":
        raise ValueError("R2R v16 collection mode must be main")
    source_split = source.get("official_split")
    if (source.get("source_dataset") != "r2r" or
            source_split not in dataset_contracts.OFFICIAL_SOURCE_SPLITS or
            source.get("semantic_format") !=
            source_contract.semantic_format):
        raise ValueError(
            "R2R v16 collection requires supported MP3D source provenance")
    schema = str(record_schema_version)
    if schema != SCHEMA_VERSION:
        raise ValueError("R2R collection record schema is unsupported")
    value = {
        "version": R2R_V16_COLLECTION_CONTRACT_VERSION,
        "record_schema_version": schema,
        "oracle_contract_version": ORACLE_CONTRACT_VERSION,
        "collection_mode": mode,
        "source_dataset": "r2r",
        "official_split": source_split,
        "semantic_format": source_contract.semantic_format,
        "source_manifest_sha256": source.get("source_manifest_sha256"),
        "source_assets_sha256": source.get("source_assets_sha256"),
    }
    return {**value, "sha256": _canonical_sha256(value)}


def b1k_v16_collection_contract(
        source_provenance: dict, collection_mode: str, *,
        record_schema_version: str = SCHEMA_VERSION,
        contract_version: str | None = None) -> dict:
    """Build the hash-bound v11 route for B1K main ABC1 collection."""
    if not isinstance(source_provenance, dict):
        raise ValueError("source provenance must be an object")
    source = source_provenance or {}
    mode = str(collection_mode)
    source_contract = dataset_contracts.dataset_source_contract("b1k")
    if mode != "main":
        raise ValueError("B1K collection mode must be main")
    if (source.get("source_dataset") != "b1k" or
            source.get("official_split") != "train" or
            source.get("split_authority") != "project_defined" or
            source.get("semantic_format") !=
            source_contract.semantic_format):
        raise ValueError(
            "B1K collection requires project-defined train OmniGibson "
            "source provenance")
    schema = str(record_schema_version)
    if schema != SCHEMA_VERSION:
        raise ValueError("B1K collection record schema must be conseq.v11")
    version = (
        B1K_OFFICIAL_RENDER_COLLECTION_CONTRACT_VERSION
        if contract_version is None else str(contract_version))
    try:
        c1_render_mode = B1K_C1_RENDER_MODE_BY_CONTRACT[version]
        observation_profile_sha256 = \
            B1K_OBSERVATION_PROFILE_SHA256_BY_CONTRACT[version]
    except KeyError as error:
        raise ValueError("B1K collection contract version is unsupported") \
            from error
    scene_authority_sha256 = b1k_scene_authority_sha256(source)
    value = {
        "version": version,
        "record_schema_version": schema,
        "oracle_contract_version": ORACLE_CONTRACT_VERSION,
        "collection_mode": mode,
        "source_dataset": "b1k",
        "official_split": "train",
        "split_authority": "project_defined",
        "semantic_format": source_contract.semantic_format,
        "source_manifest_sha256": source.get("source_manifest_sha256"),
        "source_assets_sha256": source.get("source_assets_sha256"),
        "scene_authority_sha256": scene_authority_sha256,
        "renderer_instance_protocol": B1K_RENDERER_INSTANCE_PROTOCOL,
        "observation_profile_sha256": observation_profile_sha256,
        "c1_render_mode": c1_render_mode,
    }
    return {**value, "sha256": _canonical_sha256(value)}


def collection_contract(
        source_provenance: dict, collection_mode: str, *,
        record_schema_version: str = SCHEMA_VERSION,
        contract_version: str | None = None) -> dict:
    """Dispatch a strict collection contract from trusted dataset identity."""
    if not isinstance(source_provenance, dict):
        raise ValueError("source provenance must be an object")
    source = source_provenance or {}
    dataset = str(source.get("source_dataset") or "")
    source_contract = dataset_contracts.dataset_source_contract(dataset)
    if not source_contract.main_collection_enabled:
        raise ValueError(
            f"unsupported strict collection source dataset {dataset!r}")
    if dataset == "r2r":
        return r2r_v16_collection_contract(
            source, collection_mode,
            record_schema_version=record_schema_version)
    if dataset == "b1k":
        return b1k_v16_collection_contract(
            source, collection_mode,
            record_schema_version=record_schema_version,
            contract_version=contract_version)
    raise ValueError(
        f"unsupported strict collection source dataset {dataset!r}")


def _base_rollout_payload(*, scene_id, position, yaw_rad, sensor,
                          outcome: dict) -> dict:
    return {
        "version": BASE_ROLLOUT_KEY_VERSION,
        "scene_id": str(scene_id),
        "pose": {
            "position": [float(value) for value in position],
            "yaw_rad": float(yaw_rad),
        },
        "sensor_profile": {
            "nominal_camera_offset_m":
                float(sensor["nominal_camera_offset_m"]),
            "hfov_deg": float(sensor["hfov_deg"]),
            "vfov_deg": float(sensor["vfov_deg"]),
        },
        "body": outcome.get("body") or {},
        "actions": outcome.get("actions") or [],
    }


def base_rollout_key(frame, outcome: dict) -> str:
    """Hash the target-independent physical rollout identity.
    The key names exactly the variables fixed by a base future state.  Target
    identity and any derived target measurement are intentionally absent so
    target-free A1/A2/A3/C1 cannot be multiplied by target count.
    """
    payload = _base_rollout_payload(
        scene_id=frame.scene_id,
        position=frame.position,
        yaw_rad=frame.yaw_rad,
        sensor={
            "nominal_camera_offset_m":
                frame.sensor.nominal_camera_offset_m,
            "hfov_deg": frame.sensor.hfov_deg,
            "vfov_deg": frame.sensor.vfov_deg,
        },
        outcome=outcome,
    )
    return _canonical_sha256(payload)


def stored_base_rollout_key(rec: dict, outcome: dict) -> str:
    """Recompute a base key from untrusted persisted record fields."""
    pose = rec.get("pose") or {}
    return _canonical_sha256(_base_rollout_payload(
        scene_id=rec.get("scene_id"),
        position=pose.get("position") or [],
        yaw_rad=pose.get("yaw_rad"),
        sensor=rec.get("sensor") or {},
        outcome=outcome,
    ))


def _exact_contact_instance_identity(
        frame, outcome: dict, collection_contract: dict | None, *,
        authority_binding: dataset_contracts.AuthorityBinding | None):
    """Return collection-time full-face identity, or ``None`` to withhold A3."""
    gs_identity = (
        isinstance(authority_binding, dataset_contracts.AuthorityBinding) and
        authority_binding.source_dataset == "gs")
    version = (collection_contract or {}).get("version")
    if ((not gs_identity and
         version != R2R_V16_COLLECTION_CONTRACT_VERSION and
         version not in B1K_C1_RENDER_MODE_BY_CONTRACT) or
            (outcome.get("physical") or {}).get("collision") is not True):
        return None
    consensus = outcome.get("oracle_consensus") or {}
    full_id = consensus.get("full_contact_instance_id")
    depth_id = consensus.get("depth_contact_instance_id")
    if (consensus.get("accepted") is not True or
            consensus.get("contact_instance_witness_required") is not True or
            not isinstance(full_id, int) or isinstance(full_id, bool) or
            full_id <= 0 or full_id != depth_id):
        return None
    visible = {
        int(value["instance_id"]): value
        for value in frame.objects
        if value.get("instance_id") is not None
    }
    witness = visible.get(int(full_id))
    certificate = outcome.get("shared_oracle_stability") or {}
    rows = certificate.get("rows") or []
    summary = certificate.get("summary") or {}
    exact = (rows[0].get("exact_contact_identity")
             if len(rows) == len(
                 consensus_fields.R2R_A_STABILITY_PERTURBATIONS) else None)
    if (witness is None or
            summary.get("contact_instance_stable") is not True or
            summary.get("contact_instance_id") != full_id or
            consensus_fields.a_stability_certificate_mismatches(
                outcome.get("actions") or [], certificate, outcome,
                authority_binding=authority_binding) or
            not consensus_fields.a3_exact_contact_identity_valid(
                exact, full_instance_id=full_id,
                depth_instance_id=depth_id,
                authority_binding=authority_binding)):
        return None
    if exact.get("category") != witness.get("category"):
        return None
    return dict(exact)


def _sampling_context(frame) -> dict:
    quality = frame.quality or {}
    clearance = float(quality.get("dist_to_obstacle_m", float("inf")))
    floor_ratio = float(quality.get("visible_floor_ratio", 0.0))
    object_count = sum(
        not obj.get("is_structural", False) for obj in frame.objects)
    return {
        "clearance": (
            "tight" if clearance < 0.5 else
            "near" if clearance < 1.5 else "open"),
        "visible_floor": (
            "low" if floor_ratio < 0.15 else
            "medium" if floor_ratio < 0.35 else "high"),
        "object_density": (
            "sparse" if object_count <= 1 else
            "moderate" if object_count <= 3 else "dense"),
    }


def json_value(o):
    """Convert arrays/scalars to strict JSON; unavailable numbers become null."""
    if isinstance(o, dict):
        return {k: json_value(v) for k, v in o.items()
                if not isinstance(k, str) or not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [json_value(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        value = float(o)
        return value if math.isfinite(value) else None
    if isinstance(o, np.ndarray):
        return json_value(o.tolist())
    return o


def _build_record(frame, outcomes: List[dict], *,
                  record_schema_version: str,
                  image_path: str, floor_calibration, depth_path=None,
                  intervention=None, selection=None,
                  source_provenance=None, collection_contract=None) -> dict:
    """Assemble one private frame record.
    ``floor_calibration`` is the pose's ``FloorPlaneFitResult``. It is required
    and stored in full: the plane decides target eligibility, the obstacle band
    and the published height, so a record that cannot be re-derived from its own
    stored calibration cannot be audited.
    """
    if not isinstance(floor_calibration, FloorPlaneFitResult):
        raise TypeError("build_record needs the pose's FloorPlaneFitResult")
    if floor_calibration.estimate != frame.floor_plane:
        raise ValueError(
            "the stored calibration is not the plane this frame was built with")
    objs = [dict(value) for value in frame.objects]
    v18_capability = None
    gs_collision_binding = None
    gs_scene_capability = None
    source_dataset = None
    if record_schema_version == V18_SCHEMA_VERSION:
        source_dataset = _require_v18_gs_source_provenance(
            source_provenance, scene_id=frame.scene_id)
        if collection_contract is not None:
            raise ValueError(
                "v18 collection contracts are not defined; do not reuse a "
                "legacy contract")
        v18_capability = _authority_surface_capability(source_dataset)
        gs_collision_binding = \
            dataset_contracts.resolve_gs_collision_binding(
                source_provenance)
        gs_scene_capability = gs_semantic.scene_capability_atom(
            source_provenance,
            getattr(
                getattr(frame, "semantic_index", None),
                "alignment_certificate", None),
        )
    n_nonstruct = sum(0 if o["is_structural"] else 1 for o in objs)
    identity_authority_binding = None
    if ((collection_contract or {}).get("version") ==
            R2R_V16_COLLECTION_CONTRACT_VERSION):
        identity_authority_binding = authority_binding(source_provenance)
    elif ((collection_contract or {}).get("version") in
          B1K_C1_RENDER_MODE_BY_CONTRACT):
        identity_authority_binding = authority_binding(source_provenance)
    elif (record_schema_version == V18_SCHEMA_VERSION and
          source_dataset == "gs"):
        identity_authority_binding = authority_binding(source_provenance)
    stored_outcomes = []
    for outcome in outcomes:
        if gs_collision_binding is not None:
            physical = outcome.get("physical") or {}
            if (physical.get("authority") != "gs_collision_mesh" or
                    physical.get("geometry_authority_sha256") !=
                    gs_collision_binding.collision_authority_sha256):
                raise ValueError(
                    "GS outcome collision authority is not bound")
        stored = dict(outcome)
        stored["execution"] = dict(outcome.get("execution") or {})
        stored["execution"]["execution_regime"] = (
            outcome_fields.derive_execution_regime(outcome))
        base_key = base_rollout_key(frame, stored)
        stored["base_rollout_key"] = base_key
        contact_identity = _exact_contact_instance_identity(
            frame, stored, collection_contract,
            authority_binding=identity_authority_binding)
        if contact_identity is not None:
            stored["contact_instance_identity"] = contact_identity
        stored_outcomes.append(stored)
    rec = {
        "schema_version": record_schema_version,
        "oracle_contract_version": ORACLE_CONTRACT_VERSION,
        "substrate": {
            "base_rollout_key_version": BASE_ROLLOUT_KEY_VERSION,
        },
        "frame_id": frame.frame_id,
        "scene_id": frame.scene_id,
        "scene_glb": frame.scene_glb,
        "pose": {"position": list(frame.position), "yaw_rad": frame.yaw_rad},
        "sensor": frame.sensor.to_dict(),
        # Preserve the plane used by the certified depth oracle. Compaction
        # measures public camera-to-ground height independently from source geometry.
        "floor_calibration": floor_calibration.to_json(),
        "camera_height_above_visible_floor_m":
            frame.camera_height_above_visible_floor_m,
        "image_path": image_path,
        "depth_path": depth_path,
        "quality": frame.quality,
        "n_objects": len(objs),
        "n_nonstructural": n_nonstruct,
        "category_inventory": frame.category_inventory,
        "objects": objs,
        "outcomes": stored_outcomes,
        "sampling_context": _sampling_context(frame),
    }
    if intervention is not None:
        rec["intervention"] = intervention
    if selection is not None:
        rec["selection"] = selection
    if source_provenance is not None:
        rec["source"] = dict(source_provenance)
    if collection_contract is not None:
        rec["collection_contract"] = dict(collection_contract)
    if v18_capability is not None:
        rec["authority_surface_capability"] = v18_capability
    if gs_collision_binding is not None:
        rec["gs_collision_authority"] = \
            dataset_contracts.gs_collision_binding_atom(gs_collision_binding)
        expected_plane = {
            "normal_local": [0.0, 1.0, 0.0],
            "offset_m": 0.0,
        }
        if not _type_exact_json_equal(
                rec["floor_calibration"].get("estimate"), expected_plane):
            raise ValueError(
                "GS floor calibration must use the source-bound level "
                "floor reference")
        rec["gs_floor_reference_authority"] = \
            dataset_contracts.gs_collision_floor_reference_atom(
                gs_collision_binding,
                ground_y_m=float(frame.position[1]))
        rec["gs_scene_capability"] = gs_scene_capability
    return json_value(rec)


def build_record(frame, outcomes: List[dict], *,
                 image_path: str, floor_calibration, depth_path=None,
                 intervention=None, selection=None,
                 source_provenance=None, collection_contract=None) -> dict:
    """Build the active v11 record."""
    return _build_record(
        frame, outcomes, record_schema_version=SCHEMA_VERSION,
        image_path=image_path, floor_calibration=floor_calibration,
        depth_path=depth_path, intervention=intervention, selection=selection,
        source_provenance=source_provenance,
        collection_contract=collection_contract)


def build_record_v18(frame, outcomes: List[dict], *,
                     image_path: str, floor_calibration, depth_path=None,
                     intervention=None, selection=None,
                     source_provenance=None, collection_contract=None) -> dict:
    """Build the GS-only v18 record backed by official collision geometry."""
    return _build_record(
        frame, outcomes, record_schema_version=V18_SCHEMA_VERSION,
        image_path=image_path, floor_calibration=floor_calibration,
        depth_path=depth_path, intervention=intervention, selection=selection,
        source_provenance=source_provenance,
        collection_contract=collection_contract)


def decode_records(payload: bytes | str) -> list[dict]:
    """Decode one immutable JSONL snapshot under the current contract."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        source = value.get("schema_version", "conseq.v1")
        if source != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported record schema {source}; recollect as "
                f"{SCHEMA_VERSION}")
        contract = value.get("oracle_contract_version")
        if contract != ORACLE_CONTRACT_VERSION:
            raise ValueError(
                "unsupported oracle contract "
                f"{contract!r}; recollect as {ORACLE_CONTRACT_VERSION}")
        records.append(value)
    return records


def validate_record_v18(value: dict) -> list[str]:
    """Validate the GS-only official-collision record envelope."""
    if not isinstance(value, dict):
        return ["v18 record must be an object"]
    errors = []
    if value.get("schema_version") != V18_SCHEMA_VERSION:
        errors.append(
            f"unsupported record schema {value.get('schema_version')}; "
            f"recollect as {V18_SCHEMA_VERSION}")
    if value.get("oracle_contract_version") != ORACLE_CONTRACT_VERSION:
        errors.append(
            "unsupported oracle contract "
            f"{value.get('oracle_contract_version')!r}; recollect as "
            f"{ORACLE_CONTRACT_VERSION}")
    binding = None
    try:
        _require_v18_gs_source_provenance(
            value.get("source"), scene_id=value.get("scene_id"))
        binding = dataset_contracts.resolve_gs_collision_binding(
            value["source"])
    except (AttributeError, KeyError, StopIteration, TypeError,
            ValueError) as error:
        errors.append(f"v18 GS source provenance is invalid: {error}")
    try:
        capability_contracts.validate_snapshot(
            value.get("authority_surface_capability"), "gs")
    except (TypeError, ValueError):
        errors.append(
            "authority surface capability differs from its frozen schema")
    if binding is not None:
        expected_atom = dataset_contracts.gs_collision_binding_atom(binding)
        if not _type_exact_json_equal(
                value.get("gs_collision_authority"), expected_atom):
            errors.append(
                "GS collision authority differs from exact recomputation")
        try:
            gs_semantic.validate_scene_capability_atom(
                value.get("gs_scene_capability"), value.get("source"))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"GS scene capability is invalid: {error}")
        try:
            ground_y = value["pose"]["position"][1]
            expected_floor = \
                dataset_contracts.gs_collision_floor_reference_atom(
                    binding, ground_y_m=ground_y)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            errors.append(f"GS floor reference authority is invalid: {error}")
        else:
            if not _type_exact_json_equal(
                    value.get("gs_floor_reference_authority"),
                    expected_floor):
                errors.append(
                    "GS floor reference authority differs from exact "
                    "recomputation")
            if not _type_exact_json_equal(
                    (value.get("floor_calibration") or {}).get("estimate"),
                    expected_floor["plane_local"]):
                errors.append(
                    "GS floor calibration differs from official collision "
                    "floor reference")
    for index, outcome in enumerate(value.get("outcomes") or []):
        physical = (
            outcome.get("physical") if isinstance(outcome, dict) else None)
        if not isinstance(physical, dict) or binding is None:
            errors.append(
                f"GS collision physical authority is missing at outcome "
                f"{index}")
        elif (physical.get("authority") != "gs_collision_mesh" or
              physical.get("geometry_authority_sha256") !=
              binding.collision_authority_sha256):
            errors.append(
                f"GS collision physical authority differs at outcome {index}")
    if "collection_contract" in value:
        errors.append("v18 records must not carry a legacy collection contract")
    if "gs_geometry_authority" in value:
        errors.append("v18 records must not carry Gaussian physical authority")
    objects = value.get("objects")
    if not isinstance(objects, list):
        errors.append("v18 objects must be a list")
    else:
        seen = set()
        for index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                errors.append(f"v18 object {index} must be an object")
                continue
            instance_id = obj.get("instance_id")
            if (not isinstance(instance_id, int) or
                    isinstance(instance_id, bool) or instance_id <= 0 or
                    instance_id in seen):
                errors.append(
                    f"v18 object {index} instance id is invalid or duplicated")
            else:
                seen.add(instance_id)
    return errors


def decode_v18_records(payload: bytes | str) -> list[dict]:
    """Decode and locally validate GS official-collision JSONL snapshots."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        errors = validate_record_v18(value)
        if errors:
            raise ValueError("invalid conseq.v18 record: " + "; ".join(errors))
        records.append(value)
    return records


def decode_records_for_schema(
        payload: bytes | str, schema_version: str) -> list[dict]:
    """Decode under one exact authenticated schema; never guess from rows."""
    decoders = {
        SCHEMA_VERSION: decode_records,
        V18_SCHEMA_VERSION: decode_v18_records,
    }
    try:
        decoder = decoders[schema_version]
    except KeyError as error:
        raise ValueError(
            f"candidate source schema is unsupported: {schema_version!r}") \
            from error
    return decoder(payload)


def read_records(path: str) -> Iterator[dict]:
    with open(path, "rb") as handle:
        yield from decode_records(handle.read())
