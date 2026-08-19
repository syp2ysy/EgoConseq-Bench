"""Strict agreement gate between full geometry and visible RGB-D rollouts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
from pathlib import Path
import re

from pipeline import actions as action_geometry
from pipeline import config
from pipeline import dataset_contracts
from pipeline import gs_semantic
from pipeline import semantic as semantic_fields


_R2R_A_STABILITY_ASSET_SHA256 = (
    "4a75c1e5d15c35969063f9162dd3281e124020d1ce87d1858389ffda8f62ef76"
)


def _load_a_stability_asset() -> dict:
    path = Path(__file__).with_name("assets") / "r2r_a_stability.v1.json"
    value = json.loads(path.read_text(encoding="ascii"))
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii")
    if hashlib.sha256(encoded).hexdigest() != \
            _R2R_A_STABILITY_ASSET_SHA256:
        raise RuntimeError("R2R A stability asset hash is invalid")
    if (value.get("schema") != "r2r-a-stability-asset.v1" or
            value.get("certificate_version") != "r2r-a-stability.v1"):
        raise RuntimeError("R2R A stability asset contract is invalid")
    return value


_R2R_A_STABILITY_ASSET = _load_a_stability_asset()
R2R_A_STABILITY_PERTURBATIONS = tuple(
    dict(value) for value in _R2R_A_STABILITY_ASSET["perturbations"])
R2R_A_STABILITY_VERSION = _R2R_A_STABILITY_ASSET["certificate_version"]
_A3_EXACT_CONTACT_IDENTITY_KEYS = frozenset({
    "authority", "schema", "confirmed", "reason", "instance_id", "category",
    "streaming_instance_faces_sha256", "contact_face_distance_m",
    "runner_up_face_distance_m", "face_distance_margin_m",
    "global_query_protocol", "semantic_ply_sha256",
    "global_universe_sha256", "global_winner_instance_id",
    "global_runner_up_instance_id",
})
_B1K_A3_EXACT_CONTACT_IDENTITY_KEYS = frozenset({
    "authority", "schema", "confirmed", "reason", "instance_id", "category",
    "runtime_instance_triangles_sha256", "contact_triangle_distance_m",
    "runner_up_triangle_distance_m", "triangle_distance_margin_m",
    "global_query_protocol", "scene_authority_sha256",
    "global_universe_sha256", "global_winner_instance_id",
    "global_runner_up_instance_id",
})
_GS_VISIBLE_A3_EXACT_CONTACT_IDENTITY_KEYS = frozenset({
    "authority", "schema", "confirmed", "reason", "instance_id", "category",
    "visible_points_sha256", "contact_visible_distance_m",
    "runner_up_visible_distance_m", "visible_distance_margin_m",
    "global_query_protocol", "geometry_authority_sha256",
    "semantic_source_sha256", "alignment_certificate_sha256",
    "global_universe_sha256", "global_winner_instance_id",
    "global_runner_up_instance_id",
})


def _canonical_sha256(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def arc_action_index(actions, arc):
    """The 1-based primitive index a contact arc lands in, and its margin.

    ``boundary`` is the distance to the *nearer* end of that Forward leg, which
    is what ``A2_ACTION_BOUNDARY_MARGIN_M`` gates: a contact sitting close to
    either end could move to a neighbouring public action index while still
    satisfying the frozen arc-agreement tolerance.

    Public because the exact-action-family scan needs the same margin rule when
    it decides whether a proxy contact index is stable enough to pair on.
    Duplicating the formula there is how the two would drift apart.
    """
    if arc is None:
        return None, None
    index, local = action_geometry.contact_action_index(actions, float(arc))
    if index is None:
        return None, None
    action = actions[index]
    if not isinstance(action, action_geometry.Forward):
        return None, None
    boundary = min(float(local), float(action.m) - float(local))
    return int(index) + 1, float(boundary)


_arc_action_index = arc_action_index


def a3_exact_contact_identity_valid(
        identity, *, full_instance_id, depth_instance_id,
        authority_binding: dataset_contracts.AuthorityBinding | None = None
        ) -> bool:
    """Validate one complete-face A3 identity proof without coercion."""
    if not isinstance(identity, dict):
        return False
    if not isinstance(authority_binding, dataset_contracts.AuthorityBinding):
        return False
    instance_id = _positive_integral_instance_id(identity.get("instance_id"))
    full_id = _positive_integral_instance_id(full_instance_id)
    depth_id = _positive_integral_instance_id(depth_instance_id)
    winner_id = _positive_integral_instance_id(
        identity.get("global_winner_instance_id"))
    category = identity.get("category")
    universe_sha256 = identity.get("global_universe_sha256")
    common_valid = (
            identity.get("confirmed") is True and
            identity.get("reason") == "confirmed" and
            instance_id is not None and
            instance_id == full_id == depth_id == winner_id and
            isinstance(category, str) and bool(category.strip()) and
            isinstance(universe_sha256, str) and
            re.fullmatch(r"[0-9a-f]{64}", universe_sha256))
    schema = identity.get("schema")
    if schema == "mp3d-contact-face-identity.v1":
        if (authority_binding.source_dataset != "r2r" or
                authority_binding.identity_schema != schema or
                authority_binding.authoritative_source_role != "semantic"):
            return False
        geometry_hash = identity.get("streaming_instance_faces_sha256")
        source_sha256 = identity.get("semantic_ply_sha256")
        if not (
                common_valid and
                set(identity) == _A3_EXACT_CONTACT_IDENTITY_KEYS and
                identity.get("authority") == "mp3d_full_face_universe" and
                identity.get("global_query_protocol") ==
                "mp3d-complete-face-instance-universe.v1" and
                isinstance(geometry_hash, str) and
                re.fullmatch(r"[0-9a-f]{64}", geometry_hash) and
                isinstance(source_sha256, str) and
                re.fullmatch(r"[0-9a-f]{64}", source_sha256)):
            return False
        if source_sha256 != authority_binding.source_sha256:
            return False
        if universe_sha256 != semantic_fields.complete_face_universe_sha256(
                global_query_protocol=identity["global_query_protocol"],
                semantic_ply_sha256=source_sha256):
            return False
        distance_key = "contact_face_distance_m"
        runner_key = "runner_up_face_distance_m"
        margin_key = "face_distance_margin_m"
    elif schema == "b1k-contact-triangle-identity.v1":
        if (authority_binding.source_dataset != "b1k" or
                authority_binding.identity_schema != schema or
                authority_binding.authoritative_source_role !=
                "scene_authority"):
            return False
        geometry_hash = identity.get("runtime_instance_triangles_sha256")
        source_sha256 = identity.get("scene_authority_sha256")
        if not (
                common_valid and
                set(identity) == _B1K_A3_EXACT_CONTACT_IDENTITY_KEYS and
                identity.get("authority") ==
                "b1k_runtime_triangle_universe" and
                identity.get("global_query_protocol") ==
                "b1k-runtime-instance-triangle-universe.v1" and
                isinstance(geometry_hash, str) and
                re.fullmatch(r"[0-9a-f]{64}", geometry_hash) and
                isinstance(source_sha256, str) and
                re.fullmatch(r"[0-9a-f]{64}", source_sha256)):
            return False
        if source_sha256 != authority_binding.source_sha256:
            return False
        expected_universe = _canonical_sha256({
            "schema": "b1k-complete-triangle-universe-proof.v1",
            "global_query_protocol": identity["global_query_protocol"],
            "scene_authority_sha256": source_sha256,
        })
        if universe_sha256 != expected_universe:
            return False
        distance_key = "contact_triangle_distance_m"
        runner_key = "runner_up_triangle_distance_m"
        margin_key = "triangle_distance_margin_m"
    elif schema == gs_semantic.VISIBLE_CONTACT_IDENTITY_SCHEMA:
        if (authority_binding.source_dataset != "gs" or
                authority_binding.identity_schema != schema or
                authority_binding.authoritative_source_role !=
                "source_bundle"):
            return False
        geometry_hash = identity.get("visible_points_sha256")
        source_sha256 = identity.get("geometry_authority_sha256")
        semantic_sha256 = identity.get("semantic_source_sha256")
        alignment_sha256 = identity.get("alignment_certificate_sha256")
        if not (
                common_valid and
                set(identity) == _GS_VISIBLE_A3_EXACT_CONTACT_IDENTITY_KEYS and
                identity.get("authority") ==
                "gs_initial_visible_depth_instance" and
                identity.get("global_query_protocol") ==
                gs_semantic.VISIBLE_INSTANCE_PROTOCOL and
                all(isinstance(value, str) and
                    re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in (
                        geometry_hash, source_sha256, semantic_sha256,
                        alignment_sha256))):
            return False
        if source_sha256 != authority_binding.source_sha256:
            return False
        if semantic_sha256 != authority_binding.semantic_source_sha256:
            return False
        if universe_sha256 != \
                gs_semantic.complete_visible_instance_universe_sha256(
                    geometry_authority_sha256=source_sha256,
                    semantic_source_sha256=semantic_sha256,
                    alignment_certificate_sha256=alignment_sha256):
            return False
        distance_key = "contact_visible_distance_m"
        runner_key = "runner_up_visible_distance_m"
        margin_key = "visible_distance_margin_m"
    else:
        return False
    try:
        contact_distance = float(identity[distance_key])
    except (KeyError, TypeError, ValueError):
        return False
    if (not math.isfinite(contact_distance) or contact_distance < 0.0 or
            contact_distance >
            config.A3_CONTACT_FACE_MAX_DISTANCE_M + 1e-9):
        return False
    runner_id = identity.get("global_runner_up_instance_id")
    runner_distance = identity.get(runner_key)
    margin = identity.get(margin_key)
    normalized_runner_id = _positive_integral_instance_id(runner_id)
    if normalized_runner_id is None or normalized_runner_id == instance_id:
        return False
    try:
        runner_distance = float(runner_distance)
        margin = float(margin)
    except (TypeError, ValueError):
        return False
    expected_margin = runner_distance - contact_distance
    return bool(
        math.isfinite(runner_distance) and runner_distance >= contact_distance and
        math.isfinite(margin) and
        abs(margin - expected_margin) <= 1e-9 and
        margin > config.A3_CONTACT_FACE_TIE_MARGIN_M + 1e-9)


def build_a_stability_certificate(
        actions, rows, *,
        authority_binding: dataset_contracts.AuthorityBinding | None = None,
        require_contact_instance_witness: bool = True,
        ) -> dict:
    """Canonicalize raw SE(2) rerollouts and derive one shared A certificate."""
    action_values = list(actions)
    parsed_actions = (
        action_values
        if all(isinstance(value, (
            action_geometry.Forward, action_geometry.Turn))
               for value in action_values)
        else action_geometry.parse_actions(action_values)
    )
    raw_rows = list(rows)
    expected = list(R2R_A_STABILITY_PERTURBATIONS)
    observed = [
        {
            "id": row.get("perturbation_id"),
            **dict(row.get("transform") or {}),
        }
        for row in raw_rows
    ]
    if observed != expected:
        raise ValueError("A stability rows do not match the frozen perturbation set")
    canonical_rows = []
    for row in raw_rows:
        physical = copy.deepcopy(row.get("physical") or {})
        depth = copy.deepcopy(row.get("depth_physical") or {})
        exact_identity = copy.deepcopy(row.get("exact_contact_identity"))
        coverage = float(row.get("corridor_coverage"))
        rebuilt = oracle_consensus(
            physical, depth, coverage,
            require_contact_instance_witness=
                bool(require_contact_instance_witness))
        full_index, full_boundary = _arc_action_index(
            parsed_actions, physical.get("first_contact_arc_m"))
        depth_index, depth_boundary = _arc_action_index(
            parsed_actions, depth.get("first_contact_arc_m"))
        canonical_rows.append({
            "perturbation_id": str(row["perturbation_id"]),
            "transform": {
                "x_m": float(row["transform"]["x_m"]),
                "z_m": float(row["transform"]["z_m"]),
                "yaw_deg": float(row["transform"]["yaw_deg"]),
            },
            "physical": physical,
            "depth_physical": depth,
            "corridor_coverage": coverage,
            "consensus": rebuilt,
            "full_original_action_index": full_index,
            "depth_original_action_index": depth_index,
            "full_action_boundary_margin_m": full_boundary,
            "depth_action_boundary_margin_m": depth_boundary,
            "exact_contact_identity": exact_identity,
        })
    accepted = all(
        row["consensus"].get("accepted") is True for row in canonical_rows)
    labels = [bool(row["physical"].get("collision"))
              for row in canonical_rows]
    collision_stable = accepted and len(set(labels)) == 1
    collision = labels[0] if collision_stable else None
    original_action_index = None
    action_index_stable = None
    contact_instance_id = None
    contact_instance_stable = None
    if collision is True:
        action_pairs = [
            (row["full_original_action_index"],
             row["depth_original_action_index"])
            for row in canonical_rows
        ]
        boundaries_ok = all(
            margin is not None and
            margin >= config.A2_ACTION_BOUNDARY_MARGIN_M - 1e-9
            for row in canonical_rows
            for margin in (
                row["full_action_boundary_margin_m"],
                row["depth_action_boundary_margin_m"])
        )
        action_index_stable = bool(
            boundaries_ok and all(
                full is not None and full == depth
                for full, depth in action_pairs) and
            len({full for full, _depth in action_pairs}) == 1)
        if action_index_stable:
            original_action_index = int(action_pairs[0][0])
        instances = []
        for row in canonical_rows:
            exact = row.get("exact_contact_identity") or {}
            full_id = row["consensus"].get("full_contact_instance_id")
            depth_id = row["consensus"].get("depth_contact_instance_id")
            valid = a3_exact_contact_identity_valid(
                exact, full_instance_id=full_id,
                depth_instance_id=depth_id,
                authority_binding=authority_binding)
            schema = exact.get("schema")
            is_b1k = schema == "b1k-contact-triangle-identity.v1"
            is_gs_visible = (
                schema == gs_semantic.VISIBLE_CONTACT_IDENTITY_SCHEMA)
            instances.append((
                full_id, depth_id,
                exact.get("instance_id") if valid else None,
                exact.get("category") if valid else None,
                exact.get(
                    "runtime_instance_triangles_sha256" if is_b1k else
                    "semantic_source_sha256" if is_gs_visible else
                    "streaming_instance_faces_sha256")
                if valid else None,
                exact.get(
                    "scene_authority_sha256" if is_b1k else
                    "geometry_authority_sha256" if is_gs_visible else
                    "semantic_ply_sha256") if valid else None,
                exact.get("global_universe_sha256") if valid else None,
            ))
        contact_instance_stable = bool(
            all(full is not None and full == depth == exact
                for full, depth, exact, *_proof in instances) and
            len({tuple(value[2:]) for value in instances}) == 1)
        if contact_instance_stable:
            contact_instance_id = int(instances[0][2])
    value = {
        "version": R2R_A_STABILITY_VERSION,
        "perturbation_asset_sha256": _R2R_A_STABILITY_ASSET_SHA256,
        "rows": canonical_rows,
        "summary": {
            "collision": collision,
            "collision_label_stable": bool(collision_stable),
            "original_action_index": original_action_index,
            "original_action_index_stable": action_index_stable,
            "contact_instance_id": contact_instance_id,
            "contact_instance_stable": contact_instance_stable,
        },
    }
    if not require_contact_instance_witness:
        # This explicit marker prevents a geometry-only certificate from
        # being reinterpreted as the stronger A3-bearing form.
        # The default/legacy form omits it so existing record bytes stay fixed.
        value["contact_instance_witness_required"] = False
    return {**value, "sha256": _canonical_sha256(value)}


def a_stability_certificate_mismatches(
        actions, stored: dict, nominal_outcome: dict, *,
        authority_binding: dataset_contracts.AuthorityBinding | None = None,
        require_contact_instance_witness: bool | None = None,
        ) -> list[str]:
    """Rebuild a persisted certificate and bind its nominal row to outcome."""
    expected_witness = (
        stored.get("contact_instance_witness_required", True)
        if require_contact_instance_witness is None else
        require_contact_instance_witness)
    if not isinstance(expected_witness, bool):
        return ["contact_instance_witness_required"]
    try:
        rebuilt = build_a_stability_certificate(
            actions, (stored or {}).get("rows") or [],
            authority_binding=authority_binding,
            require_contact_instance_witness=expected_witness)
    except (KeyError, TypeError, ValueError) as error:
        return [f"cannot rebuild: {error}"]
    mismatches = []
    keys = [
        "version", "perturbation_asset_sha256", "rows", "summary", "sha256",
    ]
    if ("contact_instance_witness_required" in stored or
            "contact_instance_witness_required" in rebuilt or
            require_contact_instance_witness is not None):
        keys.append("contact_instance_witness_required")
    for key in keys:
        if stored.get(key) != rebuilt.get(key):
            mismatches.append(key)
    nominal = rebuilt["rows"][0]
    try:
        expected_nominal = build_a_stability_certificate(actions, [{
            "perturbation_id": value["id"],
            "transform": {
                "x_m": value["x_m"], "z_m": value["z_m"],
                "yaw_deg": value["yaw_deg"],
            },
            "physical": (
                nominal_outcome.get("physical") if index == 0 else
                rebuilt["rows"][index]["physical"]),
            "depth_physical": (
                nominal_outcome.get("depth_physical") if index == 0 else
                rebuilt["rows"][index]["depth_physical"]),
            "corridor_coverage": (
                (nominal_outcome.get("evidence") or {}).get(
                    "physical", {}).get("coverage") if index == 0 else
                rebuilt["rows"][index]["corridor_coverage"]),
            "exact_contact_identity": (
                rebuilt["rows"][index].get("exact_contact_identity")),
        } for index, value in enumerate(R2R_A_STABILITY_PERTURBATIONS)],
            authority_binding=authority_binding,
            require_contact_instance_witness=expected_witness)
    except (KeyError, TypeError, ValueError):
        mismatches.append("nominal_outcome")
    else:
        if nominal != expected_nominal["rows"][0]:
            mismatches.append("nominal_outcome")
    return mismatches


def rebuild_outcome_consensus(
        outcome: dict, *,
        require_contact_instance_witness: bool = False) -> dict:
    """Recompute consensus from stored oracle inputs, never stored verdicts."""
    evidence = (outcome.get("evidence", {}).get("physical", {}) or {})
    coverage = evidence.get("coverage")
    if coverage is None:
        raise ValueError("physical evidence coverage is missing")
    return oracle_consensus(
        outcome.get("physical", {}) or {},
        outcome.get("depth_physical", {}) or {},
        float(coverage),
        require_contact_instance_witness=require_contact_instance_witness,
    )


def stored_consensus_mismatches(stored: dict, rebuilt: dict, *,
                                require_complete: bool) -> list[str]:
    """List stored verdict fields that disagree with an authoritative rebuild."""
    stored = stored or {}
    mismatches = []
    for key, expected in rebuilt.items():
        if key not in stored:
            if require_complete:
                mismatches.append(key)
            continue
        observed = stored[key]
        if isinstance(expected, float):
            try:
                matches = abs(float(observed) - expected) <= 1e-9
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected
        if not matches:
            mismatches.append(key)
    return mismatches


def _positive_integral_instance_id(value):
    """Normalize a semantic instance ID, rejecting coercive lookalikes."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    normalized = int(value)
    return normalized if normalized > 0 else None


def oracle_consensus(full_physical: dict, depth_physical: dict, coverage: float,
                     *, contact_tolerance_m: float = config.ORACLE_CONTACT_TOL_M,
                     coverage_min: float = config.EVIDENCE_COVERAGE_MIN,
                     require_contact_instance_witness: bool = False) -> dict:
    full_collision = full_physical.get("collision")
    depth_collision = depth_physical.get("collision")
    full_arc = full_physical.get("first_contact_arc_m")
    depth_arc = depth_physical.get("first_contact_arc_m")
    result = {
        "accepted": False,
        "verdict": "reject",
        "reason": None,
        "full_authority": full_physical.get("authority", "unavailable"),
        "depth_authority": depth_physical.get("authority", "depth"),
        "full_collision": full_collision,
        "depth_collision": depth_collision,
        "full_contact_arc_m": full_arc,
        "depth_contact_arc_m": depth_arc,
        "contact_arc_difference_m": None,
        "contact_tolerance_m": float(contact_tolerance_m),
        "corridor_coverage": float(coverage),
        "coverage_min": float(coverage_min),
    }
    if require_contact_instance_witness:
        result["contact_instance_witness_required"] = True
    if full_collision is None or result["full_authority"] == "unavailable":
        result["reason"] = "full_geometry_unavailable"
        return result
    if coverage < coverage_min:
        result["reason"] = "insufficient_depth_coverage"
        return result
    if bool(full_collision) != bool(depth_collision):
        result["reason"] = "collision_state_mismatch"
        return result
    if not full_collision:
        result.update(accepted=True, verdict="agree_safe", reason="accepted")
        return result
    if full_arc is None or depth_arc is None:
        result["reason"] = "missing_contact_arc"
        return result
    difference = abs(float(full_arc) - float(depth_arc))
    result["contact_arc_difference_m"] = difference
    if difference > contact_tolerance_m + 1e-9:
        result["reason"] = "contact_arc_mismatch"
        return result
    if require_contact_instance_witness:
        full_attribution = (
            (full_physical.get("contact") or {}).get(
                "full_geometry_attribution") or {})
        depth_attribution = (
            (depth_physical.get("contact") or {}).get(
                "depth_mask_attribution") or {})
        full_instance = _positive_integral_instance_id(
            full_attribution.get("instance_id"))
        depth_instance = _positive_integral_instance_id(
            depth_attribution.get("instance_id"))
        result.update({
            "full_contact_instance_id": full_instance,
            "depth_contact_instance_id": depth_instance,
        })
        if (full_attribution.get("unattributed") is not False or
                depth_attribution.get("unattributed") is not False or
                full_instance is None or depth_instance is None):
            result["reason"] = "missing_contact_instance_witness"
            return result
        if full_instance != depth_instance:
            result["reason"] = "contact_instance_mismatch"
            return result
    result.update(accepted=True, verdict="agree_collision", reason="accepted")
    return result
