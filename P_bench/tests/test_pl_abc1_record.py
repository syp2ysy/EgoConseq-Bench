from __future__ import annotations

from pipeline import abc1_record


def _outcome(group: str, actions: list[dict], *, collision: bool) -> dict:
    contact_index = next(
        (index for index, action in enumerate(actions)
         if action["type"] == "forward"), None) if collision else None
    contact = ({"instance_id": 7, "category": "chair"}
               if collision else None)
    return {
        "outcome_id": f"b020-{group}",
        "action_group_id": group,
        "actions": actions,
        "body": {"radius_m": 0.2},
        "physical": {
            "collision": collision,
            "contact_action_index": contact_index,
            "contact_action_local_arc_m": 0.5 if collision else None,
            "contact": contact,
            "minimum_clearance_m": 0.0 if collision else 0.42,
        },
        "depth_physical": {
            "collision": collision,
            "contact_action_index": contact_index,
            "contact_action_local_arc_m": 0.5 if collision else None,
        },
        "oracle_consensus": {"accepted": True},
        "shared_oracle_stability": {
            "version": "nominal-oracle.v1",
            "summary": {
                "evaluation": "nominal",
                "collision": collision,
                "original_action_index": (
                    None if contact_index is None else contact_index + 1),
                "contact_instance_id": 7 if collision else None,
            },
            "rows": ([{
                "consensus": {
                    "full_contact_instance_id": 7,
                    "depth_contact_instance_id": 7,
                },
                "physical": {"contact": {
                    "full_geometry_attribution": {"category": "chair"},
                }},
                "depth_physical": {"contact": {
                    "depth_mask_attribution": {"category": "chair"},
                }},
            }] if collision else []),
        },
        "execution": {
            "completed": not collision,
            "realized_pose": {"x": 0.0, "z": 1.0, "heading_deg": 15.0},
        },
    }


def _legacy_record() -> dict:
    forward = _outcome(
        "forward", [{"type": "forward", "m": 1.0}], collision=False)
    turn = _outcome("turn", [
        {"type": "turn", "deg": 15.0},
        {"type": "forward", "m": 1.0},
        {"type": "turn", "deg": -30.0},
        {"type": "forward", "m": 2.0},
    ], collision=True)
    return {
        "schema_version": "conseq.v11",
        "frame_id": "duplicate-frame-id",
        "scene_id": "scene-a",
        "pose": {"position": [1.0, 0.0, 2.0], "yaw_rad": 0.25},
        "sensor": {
            "nominal_camera_offset_m": 1.0,
            "hfov_deg": 79.0,
            "vfov_deg": 63.45,
            "resolution": [640, 480],
        },
        "camera_height_above_visible_floor_m": 1.02,
        "floor_calibration": {
            "estimate": {"normal_local": [0.0, 1.0, 0.0], "offset_m": 0.02},
        },
        "image_path": "img/frame.png",
        "objects": [
            {"instance_id": 7, "category": "chair",
             "centroid_px": [100.0, 100.0], "bbox_xyxy_px": [1, 2, 3, 4]},
            {"instance_id": 8, "category": "table",
             "centroid_px": [200.0, 100.0], "bbox_xyxy_px": [5, 2, 9, 4]},
        ],
        "surface_point_target": {
            "instance_id": 7,
            "category": "chair",
            "surface_anchor": {
                "initial_robot_xyz_m": [0.1, 0.2, 2.0],
                "pixel_xy_px": [320, 240],
            },
        },
        "selection": {
            "action_group_ids": ["forward", "turn"],
            "required_radii_m": [0.2],
            "materialized_action_bank": ["deliberately-heavy"],
            "proposal_provenance": {
                "forward": {
                    "protocol": "depth-conditioned-action-bank-v5",
                    "variant": "natural_dynamic",
                },
                "turn": {
                    "protocol": "depth-conditioned-action-bank-v5",
                    "variant": "a2_rank_collision",
                },
            },
        },
        "source": {"source_dataset": "r2r"},
        "collection_contract": {"version": "legacy"},
        "quality": {"unused": True},
        "outcomes": [forward, turn],
    }


def test_legacy_record_compacts_to_the_direct_abc1_schema():
    compact = abc1_record.from_legacy(
        _legacy_record(), dataset="r2r",
        source_records_sha256="a" * 64, byte_offset=123)

    assert compact["schema_version"] == abc1_record.SCHEMA_VERSION
    assert compact["record_uid"] == abc1_record.historical_record_uid(
        "r2r", "a" * 64, 123)
    assert compact["pose"] == _legacy_record()["pose"]
    assert compact["sensor"]["nominal_camera_offset_m"] == 1.0
    assert compact["body_radii_m"] == [0.2]
    assert {case["starts_with"] for case in compact["cases"]} == {
        "forward", "turn"}
    assert compact["cases"][1]["first_collision_action_index_1based"] == 2
    assert compact["cases"][1]["contact"] == {
        "instance_id": 7, "category": "chair"}
    assert compact["surface_point_target"] == {
        "instance_id": 7,
        "category": "chair",
        "points": [{"point_id": "p1", "pixel_xy_px": [320, 240],
                    "world_xyz_m": [0.6020833236620184, 0.2,
                                    0.037434760653258126]}],
    }
    forbidden = {
        "source", "collection_contract", "selection", "quality",
        "objects", "outcomes", "floor_calibration", "depth_path",
    }
    assert forbidden.isdisjoint(compact)


def test_historical_uid_does_not_merge_duplicate_frame_ids():
    first = abc1_record.from_legacy(
        _legacy_record(), dataset="r2r",
        source_records_sha256="a" * 64, byte_offset=100)
    second = abc1_record.from_legacy(
        _legacy_record(), dataset="r2r",
        source_records_sha256="a" * 64, byte_offset=200)

    assert first["frame_id"] == second["frame_id"]
    assert first["record_uid"] != second["record_uid"]


def test_new_uid_is_bound_to_shard_pose_and_sensor():
    first = abc1_record.new_record_uid(
        dataset="gs", shard_id="shard-1", scene_id="scene",
        pose_index=3, sensor_tag="100-f0790", pose={
            "position": [1.0, 0.0, 2.0], "yaw_rad": 0.5})
    second = abc1_record.new_record_uid(
        dataset="gs", shard_id="shard-1", scene_id="scene",
        pose_index=4, sensor_tag="100-f0790", pose={
            "position": [1.0, 0.0, 2.0], "yaw_rad": 0.5})

    assert first != second
    assert first.startswith("gs-")


def test_fresh_collection_is_compacted_with_a_fixed_bank_capacity():
    record = _legacy_record()
    record["intervention"] = {"group_id": "scene-pose"}

    compact = abc1_record.from_collected(
        record, dataset="r2r", shard_id="shard-1",
        pose_index=3, sensor_tag="h100-f0790")

    assert compact["collection_group_id"] == "scene-pose"
    assert compact["action_group_capacity"] == 2
    forward = next(case for case in compact["cases"]
                   if case["group_id"] == "forward")
    assert forward["task_outputs"]["A4"]["checkpoint_seed"] == \
        f"{compact['record_uid']}:{forward['case_id']}"


def test_compaction_preserves_raw_provenance_and_case_level_outputs():
    compact = abc1_record.from_legacy(
        _legacy_record(), dataset="r2r",
        source_records_sha256="a" * 64, byte_offset=123)
    forward = next(case for case in compact["cases"]
                   if case["group_id"] == "forward")
    turn = next(case for case in compact["cases"]
                if case["group_id"] == "turn")

    assert (forward["proposal_protocol"], forward["proposal_variant"]) == (
        "depth-conditioned-action-bank-v5", "natural_dynamic")
    assert forward["task_outputs"]["A1"] == {"answer": "no_collision"}
    assert set(forward["task_outputs"]) == {"A1", "A4", "B1", "B2"}
    assert set(turn["task_outputs"]) == {"A2", "A3"}
    assert turn["task_outputs"]["A2"] == {
        "collision_action_index_1based": 2,
        "forward_ordinal_1based": 1,
        "distance_rank": "shortest",
        "minimum_action_boundary_margin_m": 0.5,
    }
    assert turn["task_outputs"]["A3"]["category"] == "chair"
    assert "C1" not in forward["task_outputs"]
    assert "A1" not in turn["task_outputs"]


def test_auxiliary_action_source_does_not_create_an_a1_example():
    record = _legacy_record()
    record["selection"]["proposal_provenance"]["forward"]["variant"] = \
        "action_refresh"

    compact = abc1_record.from_legacy(
        record, dataset="r2r", source_records_sha256="a" * 64,
        byte_offset=123)
    forward = next(case for case in compact["cases"]
                   if case["group_id"] == "forward")

    assert "A1" not in forward["task_outputs"]


def test_nonnominal_small_distance_change_does_not_create_b1():
    record = _legacy_record()
    outcome = record["outcomes"][0]
    outcome["execution"]["realized_pose"] = {
        "x": 0.0, "z": 0.0, "heading_deg": 0.0}
    outcome["shared_oracle_stability"] = {
        "version": "perturbation.v1",
        "summary": {"collision": False, "collision_label_stable": True},
    }

    compact = abc1_record.from_legacy(
        record, dataset="r2r", source_records_sha256="a" * 64,
        byte_offset=123)
    forward = next(case for case in compact["cases"]
                   if case["group_id"] == "forward")

    assert "B1" not in forward["task_outputs"]
    assert "B2" in forward["task_outputs"]


def test_a3_requires_two_visible_contact_categories():
    record = _legacy_record()
    record["objects"] = record["objects"][:1]

    compact = abc1_record.from_legacy(
        record, dataset="r2r", source_records_sha256="a" * 64,
        byte_offset=123)
    turn = next(case for case in compact["cases"]
                if case["group_id"] == "turn")

    assert "A2" in turn["task_outputs"]
    assert "A3" not in turn["task_outputs"]


def test_a2_requires_nonnominal_collision_to_clear_action_boundaries():
    record = _legacy_record()
    outcome = record["outcomes"][1]
    outcome["physical"]["contact_action_local_arc_m"] = 0.1
    outcome["depth_physical"]["contact_action_local_arc_m"] = 0.1
    outcome["shared_oracle_stability"]["version"] = "perturbation.v1"
    outcome["shared_oracle_stability"]["summary"].update({
        "collision_label_stable": True,
        "original_action_index_stable": True,
        "contact_instance_stable": True,
    })

    compact = abc1_record.from_legacy(
        record, dataset="r2r", source_records_sha256="a" * 64,
        byte_offset=123)
    turn = next(case for case in compact["cases"]
                if case["group_id"] == "turn")

    assert "A2" not in turn["task_outputs"]


def test_gs_task_capability_excludes_a3_and_b1():
    assert abc1_record.supported_tasks("gs") == (
        "A1", "A2", "A4", "B2", "C1")


def test_delta_install_keeps_fixed_geometry_and_fixed_capacity():
    base = abc1_record.from_legacy(
        _legacy_record(), dataset="r2r",
        source_records_sha256="a" * 64, byte_offset=123)
    fingerprint = abc1_record.physical_fingerprint(base)
    capacity = base["action_group_capacity"]
    added = _outcome("new", [
        {"type": "turn", "deg": -15.0},
        {"type": "forward", "m": 2.0},
    ], collision=False)

    updated = abc1_record.install_delta(
        base, outcomes=[added], provenance={"new": {}})

    assert updated is not None
    assert abc1_record.physical_fingerprint(updated) == fingerprint
    assert updated["action_group_capacity"] == capacity
    assert len({case["group_id"] for case in updated["cases"]}) <= capacity
    assert {case["starts_with"] for case in updated["cases"]} == {
        "forward", "turn"}


def test_c1_delta_is_installed_atomically_or_not_at_all():
    base = abc1_record.from_legacy(
        _legacy_record(), dataset="r2r",
        source_records_sha256="a" * 64, byte_offset=123)
    family = []
    provenance = {}
    for index in range(4):
        group = f"c1-{index}"
        value = _outcome(
            group, [{"type": "forward", "m": 1.0 + index * 0.5}],
            collision=False)
        value["terminal_rgb_asset"] = {
            "path": f"terminal/{index}.png",
            "png_sha256": f"png-{index}",
            "pixel_sha256": f"pixel-{index}",
        }
        family.append(value)
        provenance[group] = ({
            "base_action_sha256": abc1_record.action_program_sha256(
                family[0]["actions"]),
        } if index else {})

    assert abc1_record.install_delta(
        base, outcomes=family, provenance=provenance, mode="c1") is None

    base["action_group_capacity"] = 4
    updated = abc1_record.install_delta(
        base, outcomes=family, provenance=provenance, mode="c1")

    assert updated is not None
    assert {case["group_id"] for case in updated["cases"]} == {
        "c1-0", "c1-1", "c1-2", "c1-3"}
    assert len(updated["c1_families"]) == 1


def test_c1_family_rejects_duplicate_terminal_pixels():
    base = abc1_record.from_legacy(
        _legacy_record(), dataset="r2r",
        source_records_sha256="a" * 64, byte_offset=123)
    base["action_group_capacity"] = 4
    family = []
    provenance = {}
    for index in range(4):
        group = f"c1-{index}"
        value = _outcome(
            group, [{"type": "forward", "m": 1.0 + index * 0.5}],
            collision=False)
        value["terminal_rgb_asset"] = {
            "path": f"terminal/{index}.png",
            "png_sha256": f"png-{index}",
            "pixel_sha256": "pixel-2" if index == 3 else f"pixel-{index}",
        }
        family.append(value)
        provenance[group] = ({
            "base_action_sha256": abc1_record.action_program_sha256(
                family[0]["actions"]),
        } if index else {})

    updated = abc1_record.install_delta(
        base, outcomes=family, provenance=provenance, mode="c1")

    assert updated is not None
    assert updated["c1_families"] == []


def test_delta_retention_keeps_an_existing_c1_family_as_one_bundle():
    base = abc1_record.from_legacy(
        _legacy_record(), dataset="r2r",
        source_records_sha256="a" * 64, byte_offset=123)
    base["action_group_capacity"] = 5
    family = []
    provenance = {}
    for index in range(4):
        group = f"c1-{index}"
        value = _outcome(
            group, [{"type": "forward", "m": 1.0 + index * 0.5}],
            collision=False)
        value["terminal_rgb_asset"] = {
            "path": f"terminal/{index}.png",
            "png_sha256": f"png-{index}",
            "pixel_sha256": f"pixel-{index}",
        }
        family.append(value)
        provenance[group] = ({
            "base_action_sha256": abc1_record.action_program_sha256(
                family[0]["actions"]),
        } if index else {})
    base = abc1_record.install_delta(
        base, outcomes=family, provenance=provenance, mode="c1")
    assert base is not None and len(base["c1_families"]) == 1
    base["cases"].append({
        **base["cases"][0],
        "case_id": "extra", "group_id": "000-extra",
        "starts_with": "turn", "collision": True,
    })

    added = _outcome(
        "new", [{"type": "forward", "m": 2.5}], collision=False)
    updated = abc1_record.install_delta(
        base, outcomes=[added], provenance={"new": {
            "protocol": "depth-conditioned-action-bank-v5",
            "variant": "action_refresh",
        }})

    assert updated is not None
    assert len(updated["c1_families"]) == 1
    assert set(updated["c1_families"][0]["member_case_ids"]) <= {
        case["case_id"] for case in updated["cases"]}
