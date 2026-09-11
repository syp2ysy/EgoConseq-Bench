import json

from pipeline import abc1_record
from post_QA.seen_build import inventory


def _record(scene, frame, *, schema="conseq.v11", starts_with="forward"):
    first = (
        {"type": "turn", "deg": 15.0}
        if starts_with == "turn" else
        {"type": "forward", "m": 1.0}
    )
    group = f"{frame}-group"
    return {
        "schema_version": schema,
        "scene_id": scene,
        "frame_id": frame,
        "pose": {"position": [1.0, 2.0, 3.0], "yaw_rad": 0.5},
        "sensor": {
            "nominal_camera_offset_m": 1.0,
            "hfov_deg": 79.0,
            "vfov_deg": 63.45,
        },
        "camera_height_above_visible_floor_m": 1.02,
        "selection": {
            "required_radii_m": [0.2],
            "action_group_ids": [group],
        },
        "outcomes": [{
            "action_group_id": group,
            "actions": [first],
            "body": {"shape": "disc", "radius_m": 0.2},
        }],
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True)
    path.write_text("".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8")


def test_repair_plan_does_not_schedule_tasks_without_shortfalls(tmp_path):
    records = tmp_path / "gs" / "records.jsonl"
    _write_jsonl(records, [_record("s", "f")])
    plan = tmp_path / "plan.jsonl"
    inventory.write_plan({"gs": records}, plan, repair_report={"slots": []})
    header, rows = inventory.read_plan(plan)
    assert header["repair_only"] is True
    assert not any(row["collect_a2"] or row["collect_c1"] or row["force_turn"]
                   for row in rows)


def test_repair_plan_retains_original_scene_split_and_manifest(tmp_path):
    records = tmp_path / "gs" / "records.jsonl"
    _write_jsonl(records, [_record("s", "f")])
    records.with_name("run_meta.json").write_text(json.dumps({
        "source_split": "val",
        "params": {"gs_source_manifest": "/assets/gs/splits/val.json"}}))
    plan = tmp_path / "plan.jsonl"
    inventory.write_plan({"gs": records}, plan, split="test/unseen",
                         repair_report={"slots": []})
    header, rows = inventory.read_plan(plan)
    assert header["split"] == "test/unseen"
    assert rows[0]["scene_source"] == {
        "split": "val", "manifest": "/assets/gs/splits/val.json"}


def test_a2_repair_skips_records_with_rejected_initial_images():
    rows = [{"dataset": "gs", "scene_id": "s", "record_uid": uid,
             "source_record_sha256": uid, "collect_c1": False}
            for uid in ("bad-image", "good-image")]
    inventory._assign_a2_attempts(rows, 2, {
        "eligible_record_uids": ["good-image"], "slots": [{
            "slot": ["gs", "A2", 4, "turn", 2, "shortest"], "shortfall": 1}]})
    assert not rows[0]["collect_a2"]
    assert rows[1]["collect_a2"]


def test_inventory_groups_records_by_scene_and_preserves_fixed_inputs(tmp_path):
    records = tmp_path / "b1k" / "records.jsonl"
    _write_jsonl(records, [
        _record("scene-z", "z0"),
        _record("scene-a", "a0", schema="conseq.v18", starts_with="turn"),
        _record("scene-z", "z1"),
    ])
    plan_path = tmp_path / "plan.jsonl"

    summary = inventory.write_plan({"b1k": records}, plan_path)
    header, rows = inventory.read_plan(plan_path)

    assert summary == {"records": 3, "scenes": 2, "datasets": {"b1k": 3}}
    assert header["schema"] == inventory.PLAN_SCHEMA
    assert [(row["scene_id"], row["frame_id"]) for row in rows] == [
        ("scene-a", "a0"), ("scene-z", "z0"), ("scene-z", "z1")]
    z_workers = {
        row["worker"] for row in rows if row["scene_id"] == "scene-z"}
    assert len(z_workers) == 1
    assert all(row["worker"] in {"b1k-0", "b1k-1"} for row in rows)
    assert rows[0]["pose"] == {
        "position": [1.0, 2.0, 3.0], "yaw_rad": 0.5}
    assert rows[0]["body_radius_m"] == 0.2
    assert rows[0]["camera_height_m"] == 1.0
    assert rows[0]["hfov_deg"] == 79.0
    assert rows[0]["has_turn_first"] is True
    assert rows[0]["has_noncompliant_initial_turn"] is False
    assert rows[0]["action_group_count"] == 1
    assert rows[0]["action_group_capacity"] == 2


def test_inventory_byte_offsets_rebind_to_the_original_record(tmp_path):
    records = tmp_path / "r2r" / "records.jsonl"
    expected = [_record("s", "f0"), _record("s", "f1")]
    _write_jsonl(records, expected)
    plan_path = tmp_path / "plan.jsonl"

    inventory.write_plan({"r2r": records}, plan_path)
    _header, rows = inventory.read_plan(plan_path)

    with records.open("rb") as source:
        rebound = []
        for row in rows:
            source.seek(row["byte_offset"])
            rebound.append(json.loads(source.readline()))
    assert rebound == expected


def test_inventory_uses_offsets_not_frame_ids_as_record_identity(tmp_path):
    records = tmp_path / "r2r" / "records.jsonl"
    first = _record("s", "same")
    second = _record("s", "same")
    second["pose"]["position"][0] = 9.0
    _write_jsonl(records, [first, second])

    inventory.write_plan({"r2r": records}, tmp_path / "plan.jsonl")
    header, rows = inventory.read_plan(tmp_path / "plan.jsonl")

    assert header["schema"] == inventory.PLAN_SCHEMA
    assert len({row["record_uid"] for row in rows}) == 2
    assert len({row["plan_row_id"] for row in rows}) == 2
    assert all(row["plan_id"] == header["plan_id"] for row in rows)
    assert rows[0]["source_record_sha256"] != rows[1]["source_record_sha256"]


def test_inventory_spreads_collection_over_radii_already_in_each_record(tmp_path):
    records = tmp_path / "r2r" / "records.jsonl"
    source = [_record("s", f"f{index}") for index in range(20)]
    for record in source:
        record["selection"]["required_radii_m"] = [0.15, 0.2, 0.25]
    _write_jsonl(records, source)

    inventory.write_plan({"r2r": records}, tmp_path / "plan.jsonl")
    _header, rows = inventory.read_plan(tmp_path / "plan.jsonl")

    assert {row["body_radius_m"] for row in rows} == {0.15, 0.2, 0.25}


def test_inventory_recovers_a_compact_radius_from_its_cases(tmp_path):
    records = tmp_path / "r2r" / "records.jsonl"
    record = {
        "schema_version": abc1_record.SCHEMA_VERSION,
        "record_uid": "r2r-record", "dataset": "r2r",
        "scene_id": "scene", "frame_id": "frame",
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {"nominal_camera_offset_m": 1.0,
                   "hfov_deg": 79.0, "vfov_deg": 63.45},
        "body_radii_m": [], "action_group_capacity": 2,
        "cases": [{
            "case_id": "case", "group_id": "group",
            "body_radius_m": 0.2, "starts_with": "forward",
            "collision": False, "completed": True,
        }],
    }
    _write_jsonl(records, [record])

    inventory.write_plan({"r2r": records}, tmp_path / "plan.jsonl")
    _header, rows = inventory.read_plan(tmp_path / "plan.jsonl")

    assert rows[0]["body_radius_m"] == 0.2


def test_inventory_never_splits_one_scene_across_b1k_workers(tmp_path):
    records = tmp_path / "b1k" / "records.jsonl"
    _write_jsonl(records, [
        _record("shared", f"f{index}") for index in range(5)
    ] + [_record("other", "o0")])

    inventory.write_plan({"b1k": records}, tmp_path / "plan.jsonl")
    _header, rows = inventory.read_plan(tmp_path / "plan.jsonl")

    workers = {row["worker"] for row in rows if row["scene_id"] == "shared"}
    assert len(workers) == 1


def test_inventory_does_not_reuse_a_legacy_45_degree_start(tmp_path):
    records = tmp_path / "r2r" / "records.jsonl"
    value = _record("scene", "frame", starts_with="turn")
    value["outcomes"][0]["actions"][0]["deg"] = 45.0
    _write_jsonl(records, [value])

    inventory.write_plan({"r2r": records}, tmp_path / "plan.jsonl")
    _header, rows = inventory.read_plan(tmp_path / "plan.jsonl")

    assert rows[0]["has_turn_first"] is False
    assert rows[0]["has_noncompliant_initial_turn"] is True


def test_sources_from_manifest_uses_declared_paths_not_path_shape(tmp_path):
    custom = tmp_path / "snapshots" / "records-b1k.jsonl"
    _write_jsonl(custom, [_record("scene", "frame")])
    (tmp_path / "manifest.json").write_text(json.dumps({
        "datasets": [{
            "dataset": "b1k",
            "records_path": "snapshots/records-b1k.jsonl",
        }],
    }))

    assert inventory.sources_from_manifest(tmp_path) == [
        ("b1k", custom.resolve())]


def test_plan_marks_a_bounded_scene_spread_for_c1_turn_collection(tmp_path):
    records = tmp_path / "r2r" / "records.jsonl"
    rows = []
    for index in range(4):
        record = _record(f"scene-{index % 2}", f"f{index}")
        groups = [f"f{index}-g{group}" for group in range(6)]
        record["selection"]["action_group_ids"] = groups
        record["outcomes"] = [{
            "action_group_id": group,
            "actions": [{"type": "forward", "m": 1.0}],
            "body": {"shape": "disc", "radius_m": 0.2},
            "physical": {"collision": False},
            "execution": {"completed": True},
        } for group in groups]
        rows.append(record)
    _write_jsonl(records, rows)

    inventory.write_plan(
        {"r2r": records}, tmp_path / "plan.jsonl",
        c1_attempts_per_dataset=2)
    _header, planned = inventory.read_plan(tmp_path / "plan.jsonl")
    marked = [row for row in planned if row["collect_c1"]]

    assert len(marked) == 2
    assert {row["scene_id"] for row in marked} == {"scene-0", "scene-1"}
    assert {row["c1_length"] for row in marked} <= {2, 4, 6}


def test_plan_assigns_both_a2_starts_but_never_turn_first_l3(tmp_path):
    records = tmp_path / "b1k" / "records.jsonl"
    _write_jsonl(records, [
        _record(f"scene-{index % 2}", f"f{index}") for index in range(12)
    ])

    inventory.write_plan(
        {"b1k": records}, tmp_path / "plan.jsonl",
        c1_attempts_per_dataset=0, a2_attempt_factor=1)
    _header, planned = inventory.read_plan(tmp_path / "plan.jsonl")
    marked = [row for row in planned if row["collect_a2"]]

    assert marked
    assert {row["a2_starts_with"] for row in marked} == {"forward", "turn"}
    assert all(row["a2_starts_with"] == "forward"
               for row in marked if row["a2_length"] == 3)
    assert all(
        row["a2_cell"]["collision_action_index_1based"] % 2 ==
        (0 if row["a2_starts_with"] == "turn" else 1)
        for row in marked)


def _compact_record(dataset, uid, scene="scene", frame="frame"):
    return {
        "schema_version": abc1_record.SCHEMA_VERSION,
        "record_uid": uid, "dataset": dataset,
        "scene_id": scene, "frame_id": frame,
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {"nominal_camera_offset_m": 1.0,
                   "camera_height_above_floor_m": 1.0,
                   "hfov_deg": 79.0, "vfov_deg": 63.45,
                   "resolution": [640, 480]},
        "floor_plane": {"normal_local": [0.0, 1.0, 0.0],
                        "offset_m": 0.0},
        "body_radii_m": [0.15, 0.2, 0.25],
        "image_path": f"img/{frame}.png", "action_group_capacity": 1,
        "visible_entities": [], "cases": [], "c1_families": [],
    }


def test_surface_plan_schedules_every_record_without_quota_fields(tmp_path):
    records = tmp_path / "gs" / "records.jsonl"
    _write_jsonl(records, [
        _compact_record("gs", f"gs-{index}", scene=f"s{index % 2}",
                        frame=f"f{index}")
        for index in range(6)
    ])
    plan = tmp_path / "plan.jsonl"

    summary = inventory.write_surface_plan({"gs": records}, plan, seed=23)
    header, rows = inventory.read_plan(plan)

    assert summary["records"] == 6
    assert header["mode"] == "surfaces"
    assert len(rows) == 6
    assert {row["worker"] for row in rows} == {"gs"}
    assert {row["body_radius_m"] for row in rows} != {0.15}
    assert not any(
        key in row for row in rows
        for key in ("collect_a2", "collect_c1", "force_turn", "a2_cell"))


def test_surface_plan_refuses_to_replace_a_different_immutable_plan(tmp_path):
    records = tmp_path / "gs" / "records.jsonl"
    _write_jsonl(records, [_compact_record("gs", "gs-one")])
    plan = tmp_path / "plan.jsonl"
    inventory.write_surface_plan({"gs": records}, plan)
    records.write_text(
        json.dumps(_compact_record("gs", "gs-two")) + "\n",
        encoding="utf-8")

    try:
        inventory.write_surface_plan({"gs": records}, plan)
    except FileExistsError:
        pass
    else:
        raise AssertionError("a different source must not replace the plan")


def test_surface_plan_recovers_b1k_profile_by_historical_uid(tmp_path):
    legacy_root = tmp_path / "legacy"
    legacy_records = legacy_root / "b1k" / "records.jsonl"
    original = _record("scene", "frame")
    original["collection_contract"] = {
        "version": "b1k-visible-space-abc1.v5",
        "observation_profile_sha256": "b02ba457",
    }
    _write_jsonl(legacy_records, [original])
    digest = inventory._source_digest(legacy_records)
    uid = abc1_record.historical_record_uid("b1k", digest, 0)
    current = tmp_path / "current" / "b1k" / "records.jsonl"
    _write_jsonl(current, [_compact_record("b1k", uid)])

    inventory.write_surface_plan(
        {"b1k": current}, tmp_path / "plan.jsonl",
        legacy_root=legacy_root)
    _header, rows = inventory.read_plan(tmp_path / "plan.jsonl")

    assert rows[0]["observation_profile"] == {
        "collection_contract_version": "b1k-visible-space-abc1.v5",
        "sha256": "b02ba457",
    }
