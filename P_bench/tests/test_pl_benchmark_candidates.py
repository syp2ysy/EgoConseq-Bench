from pipeline import abc1_record, abc1_task_outputs, benchmark_candidates
from pipeline.actions import Forward, Turn, actions_to_dicts
from post_QA.seen_build import spec



def test_candidate_metadata_records_start_type_and_a2_cell():
    outcome = {
        "outcome_id": "out", "actions": [
            {"type": "turn", "deg": 15.0},
            {"type": "forward", "m": 1.0},
            {"type": "turn", "deg": -15.0},
            {"type": "forward", "m": 2.0},
        ],
        "body": {"radius_m": 0.2},
        "a2_design": {"cell": {
            "forward_ordinal_1based": 2,
            "distance_rank": "longest",
        }},
    }
    record = {
        "frame_id": "frame", "scene_id": "scene",
        "pose": {"position": [0.0, 0.0, 0.0]},
        "sensor": {"nominal_camera_offset_m": 1.0, "hfov_deg": 79.0},
        "image_path": "rgb/frame.png",
    }

    row = benchmark_candidates.describe(
        dataset="r2r", source_path="records.jsonl", byte_offset=12,
        record_sha256="a" * 64, record=record, outcome=outcome,
        task_id="A2", item_id="item", answer_bucket="o2-longest")

    assert row is not None
    assert row.starts_with == "turn"
    assert row.action_length == 4
    assert row.a2_ordinal == 2
    assert row.a2_rank == "longest"


def test_projection_rejects_a_legacy_45_degree_initial_turn():
    outcome = {
        "outcome_id": "out", "actions": [
            {"type": "turn", "deg": 45.0},
            {"type": "forward", "m": 1.0},
        ],
        "body": {"radius_m": 0.2},
    }
    record = {
        "frame_id": "frame", "scene_id": "scene",
        "pose": {"position": [0.0, 0.0, 0.0]},
        "sensor": {"nominal_camera_offset_m": 1.0, "hfov_deg": 79.0},
        "image_path": "rgb/frame.png",
    }

    assert benchmark_candidates.describe(
        dataset="r2r", source_path="records.jsonl", byte_offset=0,
        record_sha256="a" * 64, record=record, outcome=outcome,
        task_id="A1", item_id="item", answer_bucket="no_collision") is None


def test_a2_cell_is_derived_when_the_legacy_cache_is_absent():
    actions = (Forward(3.0), Turn(15.0), Forward(1.0))
    outcome = {
        "actions": actions_to_dicts(actions),
        "physical": {
            "collision": True, "contact_action_index": 2,
            "contact_action_local_arc_m": 0.5,
        },
        "depth_physical": {
            "collision": True, "contact_action_index": 2,
            "contact_action_local_arc_m": 0.5,
        },
        "oracle_consensus": {"accepted": True},
        "execution": {"completed": False},
        "shared_oracle_stability": {"summary": {
            "collision": True, "collision_label_stable": True,
            "original_action_index": 3,
            "original_action_index_stable": True,
        }},
    }

    output = abc1_task_outputs.build_task_outputs(
        {}, outcome, record_uid="r2r-source", proposal_protocol=None,
        proposal_variant=None, allowed_tasks=("A2",))["A2"]
    assert output["forward_ordinal_1based"] == 2
    assert output["distance_rank"] == "shortest"


def test_compact_record_projects_all_abc1_tasks_without_certificates():
    collision = {
        "case_id": "collision", "group_id": "collision",
        "actions": [
            {"type": "forward", "m": 3.0},
            {"type": "turn", "deg": 15.0},
            {"type": "forward", "m": 1.0},
        ],
        "starts_with": "forward", "body_radius_m": 0.2,
        "collision": True, "completed": False,
        "first_collision_action_index_1based": 3,
        "endpoint_pose": {"x": 0.0, "z": 3.5, "heading_deg": 15.0},
        "minimum_clearance_m": 0.0,
        "contact": {"instance_id": 7, "category": "chair"},
        "task_outputs": {
            "A1": {"answer": "collision"},
            "A2": {
                "collision_action_index_1based": 3,
                "forward_ordinal_1based": 2,
                "distance_rank": "shortest",
                "minimum_action_boundary_margin_m": 0.5,
            },
            "A3": {
                "instance_id": 7, "category": "chair",
                "choices": [{"id": "chair", "text": "chair"},
                            {"id": "table", "text": "table"}],
            },
        },
    }
    safe_cases = [{
        "case_id": f"safe-{index}", "group_id": f"safe-{index}",
        "actions": [{"type": "forward", "m": 1.0 + index * 0.5}],
        "starts_with": "forward", "body_radius_m": 0.2,
        "collision": False, "completed": True,
        "first_collision_action_index_1based": None,
        "endpoint_pose": {
            "x": float(index), "z": 1.0, "heading_deg": 0.0},
        "minimum_clearance_m": 0.4,
        "terminal_rgb_path": f"terminal/{index}.png",
        "task_outputs": {
            "A1": {"answer": "no_collision"},
            "A4": {"answer": "front", "checkpoint_seed": "seed",
                   "checkpoint": {"pose": {
                       "x": 0.0, "z": 0.5, "heading_deg": 0.0}}},
            "B1": {"initial_distance_m": 4.0,
                   "endpoint_distance_m": 3.0,
                   "distance_change_m": 1.0},
            "B2": {"answer": "front", "horizontal_direction": "front",
                   "vertical_direction": "level"},
        },
    } for index in range(4)]
    record = {
        "schema_version": "abc1.record.v2",
        "record_uid": "r2r-record",
        "dataset": "r2r", "scene_id": "scene", "frame_id": "frame",
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {
            "nominal_camera_offset_m": 1.0,
            "camera_height_above_floor_m": 1.0,
            "hfov_deg": 79.0, "vfov_deg": 63.45,
            "resolution": [640, 480],
        },
        "body_radii_m": [0.2], "image_path": "img/frame.png",
        "action_group_capacity": 8,
        "visible_entities": [{"instance_id": 7, "category": "chair"}],
        "surface_point_target": {
            "instance_id": 7, "category": "chair",
            "point_initial_robot_xyz_m": [2.0, 1.0, 4.0],
            "pixel_xy_px": [320, 240],
        },
        "cases": [collision, *safe_cases],
        "c1_families": [{
            "query_case_id": "safe-0",
            "member_case_ids": [f"safe-{index}" for index in range(4)],
        }],
    }

    rows = benchmark_candidates.project_record(
        dataset="r2r", source_path="records.jsonl", byte_offset=0,
        record_sha256="b" * 64, record=record,
        allowed_tasks=("A1", "A2", "A3", "A4", "B1", "B2", "C1"))

    assert {row.task_id for row in rows} == {
        "A1", "A2", "A3", "A4", "B1", "B2", "C1"}
    assert all(row.record_id == "r2r-record" for row in rows)
    a2_row = next(row for row in rows if row.task_id == "A2")
    assert (a2_row.action_length, a2_row.starts_with,
            a2_row.a2_ordinal, a2_row.a2_rank) == (
                3, "forward", 2, "shortest")


def test_compact_projection_does_not_recreate_a_missing_task_output():
    record = {
        "schema_version": "abc1.record.v2",
        "record_uid": "r2r-record", "dataset": "r2r",
        "scene_id": "scene", "frame_id": "frame",
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {"nominal_camera_offset_m": 1.0, "hfov_deg": 79.0},
        "image_path": "img/frame.png",
        "surface_point_target": {
            "instance_id": 7, "category": "chair",
            "point_initial_robot_xyz_m": [0.0, 1.0, 4.0],
            "pixel_xy_px": [320, 240],
        },
        "cases": [{
            "case_id": "safe", "group_id": "safe",
            "actions": [{"type": "forward", "m": 1.0}],
            "starts_with": "forward", "body_radius_m": 0.2,
            "collision": False, "completed": True,
            "endpoint_pose": {"x": 0.0, "z": 1.0, "heading_deg": 0.0},
            "task_outputs": {"B2": {
                "answer": "front", "horizontal_direction": "front",
                "vertical_direction": "level",
            }},
        }],
        "c1_families": [],
    }

    rows = benchmark_candidates.project_record(
        dataset="r2r", source_path="records.jsonl", byte_offset=0,
        record_sha256="b" * 64, record=record,
        allowed_tasks=("B1", "B2"))

    assert [row.task_id for row in rows] == ["B2"]
