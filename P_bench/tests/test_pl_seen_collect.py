import hashlib
import json
from types import SimpleNamespace

from pipeline import a2, action_proposal
from post_QA.seen_build import collect, inventory


def test_compact_refine_loads_the_original_gs_validation_scene(monkeypatch):
    import sys
    from pipeline import scene_pool

    def discover(root, manifest, *, requested, source_split="train"):
        return [SimpleNamespace(
            scene_id=requested[0], source_dataset="gs", scene_path="scene.ply",
            official_split=source_split,
            provenance=lambda: {"official_split": source_split})]

    monkeypatch.setattr(scene_pool, "discover_gs_scenes", discover)
    monkeypatch.setattr(scene_pool, "discover_gs_train_scenes", discover)
    monkeypatch.setitem(sys.modules, "pipeline.gs_sim", SimpleNamespace(
        GsSimSession=lambda scene, **kwargs: scene))
    record = {"schema_version": collect.abc1_record.SCHEMA_VERSION,
              "scene_id": "held-out", "sensor": {
                  "nominal_camera_offset_m": 1.0, "camera_height_above_floor_m": 1.1,
                  "hfov_deg": 79.0, "vfov_deg": 63.45},
              "floor_plane": {"normal_local": [0.0, 1.0, 0.0], "offset_m": 0.0}}
    scene = collect.make_scene_opener()("gs", "held-out", [({
        "scene_source": {"split": "val", "manifest": "/gs/val.json"}}, record)])
    assert record["camera_height_above_visible_floor_m"] == 1.0
    assert scene.official_split == "val"
    assert record["source"]["official_split"] == "val"


def _record(frame):
    return {
        "schema_version": "conseq.v11",
        "scene_id": frame.split("-")[0],
        "frame_id": frame,
    }


def _source(path, frames):
    offsets = {}
    with path.open("wb") as stream:
        for frame in frames:
            offsets[frame] = stream.tell()
            stream.write(json.dumps(_record(frame)).encode() + b"\n")
    return offsets


def _plan(path, source, offsets):
    rows = []
    for frame, offset in offsets.items():
        scene = frame.split("-")[0]
        rows.append({
            "kind": "record", "dataset": "gs", "scene_id": scene,
            "frame_id": frame, "record_uid": f"gs:{frame}", "worker": "gs",
            "source_path": str(source), "byte_offset": offset,
            "source_record_sha256": hashlib.sha256(
                json.dumps(_record(frame)).encode()).hexdigest(),
            "plan_id": "test-plan", "plan_row_id": f"row-{len(rows):06d}",
        })
    rows.sort(key=lambda row: (row["scene_id"], row["byte_offset"]))
    values = [{
        "kind": "header", "schema": inventory.PLAN_SCHEMA,
        "plan_id": "test-plan", "repair_only": False,
    }, *rows]
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


def test_worker_opens_each_scene_once_and_finishes_it_before_the_next(tmp_path):
    source = tmp_path / "records.jsonl"
    offsets = _source(source, ["z-0", "a-0", "z-1", "a-1"])
    plan = tmp_path / "plan.jsonl"
    _plan(plan, source, offsets)
    events = []

    class Session:
        def __init__(self, scene):
            self.scene = scene

        def close(self):
            events.append(("close", self.scene))

    def open_scene(dataset, scene, records):
        events.append(("open", scene, [record["frame_id"] for _, record in records]))
        return Session(scene)

    def process(session, row, record):
        events.append(("record", session.scene, record["frame_id"]))
        return {"kept": True}

    summary = collect.run_worker(
        plan, "gs", tmp_path / "gs.jsonl",
        open_scene=open_scene, process_record=process)

    assert summary == {"worker": "gs", "planned": 4, "completed": 4}
    assert events == [
        ("open", "a", ["a-0", "a-1"]),
        ("record", "a", "a-0"),
        ("record", "a", "a-1"),
        ("close", "a"),
        ("open", "z", ["z-0", "z-1"]),
        ("record", "z", "z-0"),
        ("record", "z", "z-1"),
        ("close", "z"),
    ]


def test_worker_keeps_duplicate_frame_ids_bound_to_their_own_pose(tmp_path):
    source = tmp_path / "records.jsonl"
    values = [_record("s-same"), _record("s-same")]
    for index, value in enumerate(values, 1):
        value.update({
            "pose": {"position": [float(index), 0.0, 0.0], "yaw_rad": 0.0},
            "sensor": {
                "nominal_camera_offset_m": 1.0,
                "hfov_deg": 79.0, "vfov_deg": 63.45,
            },
            "camera_height_above_visible_floor_m": 1.0,
            "selection": {"required_radii_m": [0.2]},
            "outcomes": [],
        })
    source.write_text("".join(json.dumps(value) + "\n" for value in values))
    plan = tmp_path / "plan.jsonl"
    inventory.write_plan({"gs": source}, plan)
    seen = []

    class Session:
        def close(self):
            pass

    collect.run_worker(
        plan, "gs", tmp_path / "gs.jsonl",
        open_scene=lambda _dataset, _scene, _records: Session(),
        process_record=lambda _session, row, record:
            seen.append((row["record_uid"], record["pose"]["position"][0])) or
            {"kept": True})

    assert [position for _uid, position in seen] == [1.0, 2.0]
    results = [json.loads(line) for line in
               (tmp_path / "gs.jsonl").read_text().splitlines()]
    header, _rows = inventory.read_plan(plan)
    assert len({row["record_uid"] for row in results}) == 2
    assert {row["plan_id"] for row in results} == {header["plan_id"]}
    assert len({row["plan_row_id"] for row in results}) == 2


def test_worker_resumes_without_creating_failure_or_checkpoint_files(tmp_path):
    source = tmp_path / "records.jsonl"
    offsets = _source(source, ["s-0", "s-1"])
    plan = tmp_path / "plan.jsonl"
    _plan(plan, source, offsets)
    output = tmp_path / "gs.jsonl"
    _header, rows = inventory.read_plan(plan)
    output.write_text(json.dumps({
        "schema": collect.RESULT_SCHEMA, "plan_id": "test-plan",
        "plan_row_id": rows[0]["plan_row_id"],
        "record_uid": rows[0]["record_uid"],
        "source_record_sha256": rows[0]["source_record_sha256"],
        "status": "complete"}) + "\n")
    processed = []

    class Session:
        def close(self):
            pass

    collect.run_worker(
        plan, "gs", output,
        open_scene=lambda _dataset, _scene, _records: Session(),
        process_record=lambda _session, row, _record:
            processed.append(row["record_uid"]) or {"kept": True})

    assert processed == ["gs:s-1"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "gs.jsonl", "plan.jsonl", "records.jsonl"]


def test_record_processor_reuses_an_existing_turn_group_without_rendering():
    record = {
        "outcomes": [{
            "action_group_id": "turn", "actions": [
                {"type": "turn", "deg": 15.0},
                {"type": "forward", "m": 1.0},
            ],
        }],
    }
    processor = collect.make_record_processor(
        seed=1, max_programs=10, max_full_attempts=1)

    assert processor(object(), {"record_id": "gs:f"}, record) == {
        "mode": "reused"}


def test_untargeted_repair_never_opens_a_scene(tmp_path, monkeypatch):
    plan = tmp_path / "plan.jsonl"
    source = tmp_path / "records.jsonl"
    offsets = _source(source, ["s-0"])
    _plan(plan, source, offsets)
    header, rows = inventory.read_plan(plan)
    header.update(schema=inventory.PLAN_SCHEMA, repair_only=True)
    plan.write_text("".join(json.dumps(row) + "\n" for row in [header, *rows]))

    def unexpected(*args):
        raise AssertionError("untargeted repair must not load or render records")

    monkeypatch.setattr(collect, "_load_records", lambda rows: [] if not rows else unexpected())
    result = collect.run_worker(plan, "gs", tmp_path / "gs.jsonl",
                                open_scene=unexpected, process_record=unexpected)
    assert result["completed"] == 1
    assert json.loads((tmp_path / "gs.jsonl").read_text())["mode"] == "reused"


def test_record_processor_prioritizes_planned_c1_family(monkeypatch, tmp_path):
    record = {"outcomes": []}
    monkeypatch.setattr(collect, "_build_frame", lambda _session, _record: "frame")
    monkeypatch.setattr(collect, "_collect_c1_family", lambda *args, **kwargs: {
        "outcomes": [{"action_group_id": "query"}],
        "provenance": {"query": {}}, "staged_assets": [],
    })
    processor = collect.make_record_processor(
        seed=1, max_programs=10, max_full_attempts=1,
        stage_root=tmp_path)

    result = processor(object(), {
        "record_id": "r2r:f", "dataset": "r2r",
        "body_radius_m": 0.2,
        "collect_c1": True, "c1_length": 4,
    }, record)

    assert result["mode"] == "c1"
    assert result["outcomes"][0]["action_group_id"] == "query"


def test_record_processor_uses_planned_a2_cell(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "_build_frame", lambda _session, _record: "frame")
    seen = {}

    def collect_a2(*_args, **kwargs):
        seen.update(kwargs["row"]["a2_cell"])
        seen["starts_with"] = kwargs["row"]["a2_starts_with"]
        return {"outcomes": [], "provenance": {}}

    monkeypatch.setattr(collect, "_collect_a2_group", collect_a2)
    processor = collect.make_record_processor(
        seed=1, max_programs=10, max_full_attempts=1,
        stage_root=tmp_path)
    result = processor(object(), {
        "record_id": "b1k:f", "dataset": "b1k", "collect_c1": False,
        "body_radius_m": 0.2,
        "collect_a2": True, "a2_length": 4,
        "a2_starts_with": "turn",
        "a2_cell": {
            "forward_ordinal_1based": 1, "distance_rank": "shortest",
            "collision_action_index_1based": 2,
        },
    }, {"outcomes": []})

    assert result["mode"] == "a2"
    assert seen["collision_action_index_1based"] == 2
    assert seen["starts_with"] == "turn"


def test_a2_collection_keeps_a_certified_case_from_another_cell(monkeypatch):
    class Sim:
        def recompute_navmesh(self, _radius, *, height):
            pass

        def nav(self, _position, _yaw):
            return object()

    class Proxy:
        pass

    monkeypatch.setattr(action_proposal, "FrameDepthProxy",
                        lambda _frame, _radius: Proxy())
    monkeypatch.setattr(
        a2, "collision_proposals",
        lambda *_args, **_kwargs: iter([SimpleNamespace(
            collision_actions=(object(),))]))
    monkeypatch.setattr(
        collect.fixed_pose_actions, "proxy_verdict",
        lambda *_args, **_kwargs: {"collision": True})
    monkeypatch.setattr(
        collect.fixed_pose_actions, "certify_program",
        lambda *_args, **_kwargs: {
            "outcomes": [{
                "action_group_id": "actual",
                "a2_design": {"cell": {
                    "forward_ordinal_1based": 2,
                    "distance_rank": "longest",
                }},
            }],
            "provenance": {"actual": {
                "protocol": "depth-conditioned-action-bank-v5",
                "variant": "natural_dynamic",
            }},
        })
    row = {
        "body_radius_m": 0.2, "a2_length": 4,
        "a2_starts_with": "forward",
        "a2_cell": {"forward_ordinal_1based": 1,
                    "distance_rank": "shortest"},
    }
    record = {
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {"hfov_deg": 79.0},
    }

    result = collect._collect_a2_group(
        Sim(), object(), record, row=row, seed=1,
        max_programs=1, max_full_attempts=1)

    assert result is not None
    assert result["outcomes"][0]["a2_design"]["cell"] == {
        "forward_ordinal_1based": 2, "distance_rank": "longest"}
    assert result["provenance"]["actual"]["variant"] == "a2_rank_collision"


def test_record_processor_prunes_a_legacy_initial_turn_if_replacement_fails(
        monkeypatch):
    record = {
        "selection": {"action_group_ids": ["legacy"]},
        "outcomes": [{
            "action_group_id": "legacy",
            "actions": [{"type": "turn", "deg": 45.0},
                        {"type": "forward", "m": 1.0}],
        }],
    }
    monkeypatch.setattr(collect, "_build_frame", lambda *_args: "frame")
    monkeypatch.setattr(
        collect.fixed_pose_actions, "collect_turn_group",
        lambda *_args, **_kwargs: None)
    processor = collect.make_record_processor(
        seed=1, max_programs=10, max_full_attempts=1)

    result = processor(object(), {
        "record_id": "r2r:f", "body_radius_m": 0.2,
        "collect_c1": False, "collect_a2": False,
    }, record)

    assert result == {"mode": "pruned", "outcomes": [], "provenance": {}}


def test_worker_does_not_resume_from_a_result_bound_to_changed_source(tmp_path):
    source = tmp_path / "records.jsonl"
    offsets = _source(source, ["s-0"])
    plan = tmp_path / "plan.jsonl"
    _plan(plan, source, offsets)
    _header, rows = inventory.read_plan(plan)
    output = tmp_path / "gs.jsonl"
    output.write_text(json.dumps({
        "schema": collect.RESULT_SCHEMA, "plan_id": "test-plan",
        "plan_row_id": rows[0]["plan_row_id"],
        "record_uid": rows[0]["record_uid"],
        "source_record_sha256": "0" * 64, "status": "complete",
    }) + "\n")
    processed = []

    class Session:
        def close(self):
            pass

    collect.run_worker(
        plan, "gs", output,
        open_scene=lambda *_args: Session(),
        process_record=lambda _session, row, _record:
            processed.append(row["record_uid"]) or {"mode": "reused"})

    assert processed == ["gs:s-0"]


def test_surface_processor_collects_at_one_existing_uid_selected_radius(
        monkeypatch):
    from pipeline import surface_points

    normalized = {
        "schema_version": "abc1.record.v3", "record_uid": "gs-record",
        "dataset": "gs", "body_radii_m": [0.15, 0.2, 0.25], "cases": [],
    }
    target = {"instance_id": 7, "category": "chair", "points": [{
        "point_id": "p1", "pixel_xy_px": [2, 3],
        "world_xyz_m": [1.0, 2.0, 3.0],
    }]}
    calls = []
    monkeypatch.setattr(collect.abc1_record, "normalize", lambda _record: normalized)
    monkeypatch.setattr(collect, "_build_frame",
                        lambda _session, _record: calls.append("frame") or "frame")
    monkeypatch.setattr(surface_points, "select_target",
                        lambda *_args, **_kwargs: target)

    def collect_safe(*_args, **kwargs):
        calls.append(("safe", kwargs["radius_m"]))
        return {"outcomes": [{"outcome_id": "new"}], "provenance": {}}

    monkeypatch.setattr(collect.fixed_pose_actions, "collect_safe_group", collect_safe)
    monkeypatch.setattr(
        collect.abc1_record, "compact_case",
        lambda *_args, **_kwargs: {
            "case_id": "new", "group_id": "new", "starts_with": "turn",
            "collision": False, "completed": True,
        })
    processor = collect.make_surface_processor(
        seed=20260906, max_programs=500, max_full_attempts=4)

    result = processor(object(), {
        "record_uid": "gs-record", "dataset": "gs", "body_radius_m": 0.2,
    }, {"schema_version": "abc1.record.v2"})

    assert calls[0] == "frame"
    assert calls[1][0] == "safe"
    assert calls[1][1] in normalized["body_radii_m"]
    assert result == {
        "mode": "surfaces", "surface_status": "added_safe",
        "surface_point_target": target,
        "cases": [{
            "case_id": "new", "group_id": "new", "starts_with": "turn",
            "collision": False, "completed": True,
        }],
    }


def test_surface_processor_reuses_safe_case_and_does_not_collect(monkeypatch):
    from pipeline import surface_points

    normalized = {
        "schema_version": "abc1.record.v3", "record_uid": "r2r-record",
        "dataset": "r2r", "body_radii_m": [0.2],
        "cases": [{"collision": False, "completed": True}],
    }
    monkeypatch.setattr(collect.abc1_record, "normalize", lambda _record: normalized)
    monkeypatch.setattr(collect, "_build_frame", lambda *_args: "frame")
    monkeypatch.setattr(surface_points, "select_target", lambda *_args, **_kwargs: {
        "instance_id": 1, "category": "wall", "points": []})
    monkeypatch.setattr(
        collect.fixed_pose_actions, "collect_safe_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing safe case must be reused")))

    result = collect.make_surface_processor(
        seed=7, max_programs=8, max_full_attempts=2)(
            object(), {"record_uid": "r2r-record", "dataset": "r2r"}, {})

    assert result["surface_status"] == "existing_safe"
    assert result["cases"] == []


def test_collect_dispatches_surface_plan_to_surface_processor(
        tmp_path, monkeypatch):
    plan = tmp_path / "plan.jsonl"
    plan.write_text(json.dumps({
        "kind": "header", "schema": inventory.PLAN_SCHEMA,
        "plan_id": "surface-plan", "mode": "surfaces",
    }) + "\n")
    marker = object()
    monkeypatch.setattr(
        collect, "make_surface_processor", lambda **_kwargs: marker)
    monkeypatch.setattr(
        collect, "make_record_processor",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("surface plan used the action repair processor")))

    def run(_plan, worker, output, **kwargs):
        assert kwargs["process_record"] is marker
        return {"worker": worker, "output": str(output)}

    monkeypatch.setattr(collect, "run_worker", run)
    output = tmp_path / "gs.jsonl"

    result = collect.collect_records(plan, "gs", output)

    assert result == {"worker": "gs", "output": str(output)}
