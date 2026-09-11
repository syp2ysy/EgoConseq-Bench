from itertools import islice

from pipeline import abc1_record, action_proposal, fixed_pose_actions, validate
from pipeline import actions as action_geometry


def _record(groups):
    outcomes = []
    bank = []
    provenance = {}
    for index, (group, actions) in enumerate(groups):
        outcomes.append({
            "outcome_id": f"b020-{group}",
            "action_group_id": group,
            "action_group_label": "safe",
            "seq_len": len(actions),
            "actions": actions,
            "body": {"shape": "disc", "radius_m": 0.2},
        })
        bank.append({
            "tag": group, "length": len(actions), "actions": actions,
            "variant": "old", "template_id": f"old-{index}",
        })
        provenance[group] = {"variant": "old"}
    return {
        "selection": {
            "action_group_ids": [group for group, _ in groups],
            "action_group_labels": {
                group: "safe" for group, _ in groups},
            "materialized_action_bank": bank,
            "proposal_provenance": provenance,
        },
        "outcomes": outcomes,
    }


def test_turn_first_programs_use_the_shared_initial_turn_vocabulary():
    programs = list(islice(
        fixed_pose_actions.candidate_programs(
            length=6, seed=9, limit=100), 100))

    assert len(programs) == 100
    assert {program[0].deg for program in programs} <= {-30, -15, 15, 30}
    assert all(fixed_pose_actions.starts_with_turn(program) for program in programs)
    assert any(abs(program[2].deg) == 45 for program in programs)


def test_legacy_45_degree_start_is_not_a_compliant_turn_first_group():
    record = _record([("old", [
        {"type": "turn", "deg": 45.0},
        {"type": "forward", "m": 1.0},
    ])])

    assert fixed_pose_actions.record_has_turn_first(record) is False


def test_compact_record_uses_the_central_schema_check():
    record = {
        "schema_version": abc1_record.SCHEMA_VERSION,
        "cases": [{
            "starts_with": "turn",
            "actions": [{"type": "turn", "deg": 15.0},
                        {"type": "forward", "m": 1.0}],
        }],
    }

    assert fixed_pose_actions.record_has_turn_first(record) is True


def test_install_one_turn_group_keeps_every_old_group_and_adds_only_one():
    record = _record([
        ("old-a", [{"type": "forward", "m": 1.0}]),
        ("old-b", [{"type": "forward", "m": 2.0}]),
    ])
    actions = [{"type": "turn", "deg": 15.0},
               {"type": "forward", "m": 1.0}]
    group = action_proposal.candidate_tag(
        fixed_pose_actions.parse_actions(actions))
    outcome = {
        "outcome_id": f"b020-{group}", "action_group_id": group,
        "action_group_label": "safe", "seq_len": 2,
        "actions": actions, "body": {"shape": "disc", "radius_m": 0.2},
    }

    updated = fixed_pose_actions.install_groups(
        record, [outcome], {group: {"variant": "fixed_pose_turn"}})

    assert updated is not None
    assert updated["selection"]["action_group_ids"] == [
        "old-a", "old-b", group]
    assert len(updated["selection"]["action_group_ids"]) == 3
    assert fixed_pose_actions.record_has_turn_first(updated)
    assert record["selection"]["action_group_ids"] == ["old-a", "old-b"]


def test_install_groups_removes_a_legacy_45_degree_start():
    record = _record([("legacy-45", [
        {"type": "turn", "deg": 45.0},
        {"type": "forward", "m": 1.0},
    ])])
    actions = [{"type": "turn", "deg": 15.0},
               {"type": "forward", "m": 1.0}]
    group = action_proposal.candidate_tag(
        fixed_pose_actions.parse_actions(actions))
    outcome = {
        "outcome_id": f"b020-{group}", "action_group_id": group,
        "action_group_label": "safe", "seq_len": 2,
        "actions": actions, "body": {"shape": "disc", "radius_m": 0.2},
    }

    updated = fixed_pose_actions.install_groups(
        record, [outcome], {group: {"variant": "fixed_pose_turn"}})

    assert updated["selection"]["action_group_ids"] == [group]
    assert {row["action_group_id"] for row in updated["outcomes"]} == {group}


def test_install_groups_refuses_to_evict_protected_benchmark_groups():
    record = _record([
        ("old-a", [{"type": "forward", "m": 1.0}]),
    ])
    new = []
    provenance = {}
    for index in range(3):
        actions = [
            {"type": "turn", "deg": 15.0},
            {"type": "forward", "m": 0.5 + index * 0.5},
        ]
        group = action_proposal.candidate_tag(
            fixed_pose_actions.parse_actions(actions))
        new.append({
            "outcome_id": f"b020-{group}", "action_group_id": group,
            "action_group_label": "safe", "seq_len": 2,
            "actions": actions, "body": {"shape": "disc", "radius_m": 0.2},
        })
        provenance[group] = {"variant": "fixed_pose_turn"}

    assert fixed_pose_actions.install_groups(
        record, new, provenance, protected_group_ids={"old-a"}) is None


def test_install_groups_drops_pair_metadata_for_evicted_groups():
    record = _record([
        ("old-a", [{"type": "forward", "m": 1.0}]),
        ("old-b", [{"type": "forward", "m": 2.0}]),
    ])
    record["selection"]["matched_action_units"] = [{
        "safe_group_id": "old-a", "collision_group_id": "old-b"}]
    new = []
    provenance = {}
    for index in range(3):
        values = [{"type": "turn", "deg": 15.0},
                  {"type": "forward", "m": 0.5 + index * 0.5}]
        group = action_proposal.candidate_tag(
            fixed_pose_actions.parse_actions(values))
        new.append({
            "outcome_id": f"b020-{group}", "action_group_id": group,
            "action_group_label": "safe", "seq_len": 2,
            "actions": values, "body": {"radius_m": 0.2},
        })
        provenance[group] = {"variant": "fixed_pose_turn"}

    updated = fixed_pose_actions.install_groups(record, new, provenance)

    assert updated["selection"]["matched_action_units"] == []


class _SafeProxy:
    def rollout(self, actions):
        return {
            "collision": False,
            "first_contact_arc_m": None,
            "contact_action_index": None,
        }

    def coverage(self, actions, max_arc_m=None):
        return 1.0


def test_proxy_shortlist_is_bounded_and_contains_only_turn_first_paths():
    rows = fixed_pose_actions.proxy_shortlist(
        _SafeProxy(), half_fov_deg=39.5, seed=3,
        max_programs=40, shortlist=5)

    assert len(rows) == 5
    assert all(fixed_pose_actions.starts_with_turn(actions)
               for actions, _verdict in rows)
    assert all(verdict["collision"] is False for _actions, verdict in rows)


def test_safe_proxy_rejects_an_out_of_view_path_before_depth_rollout():
    class Proxy(_SafeProxy):
        def __init__(self):
            self.rollout_calls = 0

        def rollout(self, actions):
            self.rollout_calls += 1
            return super().rollout(actions)

    proxy = Proxy()
    sequence = action_geometry.parse_actions([
        {"type": "turn", "deg": 30.0},
        {"type": "forward", "m": 3.0},
        {"type": "turn", "deg": 45.0},
        {"type": "forward", "m": 3.0},
    ])

    assert fixed_pose_actions.proxy_verdict(
        proxy, sequence, half_fov_deg=39.5,
        desired_collision=False) is None
    assert proxy.rollout_calls == 0


def test_collect_turn_group_shortlists_only_its_full_oracle_budget(monkeypatch):
    seen = {}

    def shortlist(*_args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(fixed_pose_actions, "proxy_shortlist", shortlist)

    class Sim:
        def recompute_navmesh(self, _radius, height):
            pass

        def nav(self, _position, _yaw):
            return object()

    record = _record([("old", [{"type": "forward", "m": 1.0}])])
    record.update({
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {"hfov_deg": 79.0},
    })

    fixed_pose_actions.collect_turn_group(
        Sim(), object(), record, seed=1, max_programs=50,
        max_full_attempts=3, radius_m=0.2, proxy=_SafeProxy())

    assert seen["shortlist"] == 3


def test_safe_collection_searches_both_starts_without_changing_geometry(monkeypatch):
    seen = {}

    def shortlist(*_args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(fixed_pose_actions, "proxy_shortlist", shortlist)

    class Sim:
        def recompute_navmesh(self, radius, height):
            seen["radius"] = radius

        def nav(self, position, yaw):
            seen["pose"] = (position, yaw)

    record = {"pose": {"position": [1, 0, 2], "yaw_rad": 0.25},
              "sensor": {"hfov_deg": 79}}
    assert fixed_pose_actions.collect_safe_group(
        Sim(), object(), record, seed=7, radius_m=0.2,
        proxy=_SafeProxy()) is None
    assert seen["desired_collision"] is False
    assert set(seen["starts"]) == {"forward", "turn"}
    assert seen["max_programs"] == 500
    assert seen["shortlist"] == 4
    assert seen["radius"] == 0.2
    assert seen["pose"] == ([1, 0, 2], 0.25)


def test_collect_turn_group_uses_the_record_pose_and_radius_once():
    record = _record([
        ("old", [{"type": "forward", "m": 1.0}]),
    ])
    record.update({
        "scene_id": "scene", "frame_id": "frame",
        "pose": {"position": [1.0, 0.0, 2.0], "yaw_rad": 0.25},
        "sensor": {
            "nominal_camera_offset_m": 1.0,
            "hfov_deg": 79.0, "vfov_deg": 63.45,
        },
    })
    record["selection"]["required_radii_m"] = [0.2]

    class Sim:
        def __init__(self):
            self.calls = []

        def recompute_navmesh(self, radius, height):
            self.calls.append(("radius", radius, height))

        def nav(self, position, yaw):
            self.calls.append(("nav", list(position), yaw))
            return "nav"

    def judge(_frame, body, actions, *, nav,
              require_contact_instance_witness):
        distance = action_geometry.total_forward_m(actions)
        return {
            "body": body.to_dict(),
            "actions": action_geometry.actions_to_dicts(actions),
            "physical": {
                "collision": False, "first_contact_arc_m": None,
                "authority": "test", "contact": None,
            },
            "depth_physical": {
                "collision": False, "first_contact_arc_m": None,
                "contact": None,
            },
            "oracle_consensus": {"accepted": True},
            "evidence": {"physical": {"coverage": 1.0}},
            "execution": {
                "completed": True, "stop_reason": "completed",
                "nominal_forward_m": distance,
                "executed_forward_m": distance,
                "realized_pose": {"x": 0.0, "z": distance,
                                  "heading_deg": 15.0},
            },
            "checkpoints": [],
            "provenance": {},
        }

    sim = Sim()
    delta = fixed_pose_actions.collect_turn_group(
        sim, object(), record, seed=4, max_programs=20,
        max_full_attempts=2, radius_m=0.2,
        proxy=_SafeProxy(), judge_fn=judge)

    assert delta is not None
    assert len(delta["outcomes"]) == 1
    assert fixed_pose_actions.starts_with_turn(delta["outcomes"][0]["actions"])
    provenance = next(iter(delta["provenance"].values()))
    assert provenance["variant"] == "action_refresh"
    assert sim.calls[0][0:2] == ("radius", 0.2)
    assert sim.calls[1] == ("nav", [1.0, 0.0, 2.0], 0.25)


def test_collision_turn_group_carries_an_a2_rank_certificate():
    record = _record([("old", [{"type": "forward", "m": 1.0}])])
    record.update({
        "scene_id": "scene", "frame_id": "frame",
        "pose": {"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "sensor": {
            "nominal_camera_offset_m": 1.0,
            "hfov_deg": 110.0, "vfov_deg": 93.93,
        },
    })
    record["selection"]["required_radii_m"] = [0.2]
    record["source"] = {"source_dataset": "r2r"}
    judge_call = {}

    class CollisionProxy:
        def rollout(self, values):
            distances = [action.m for action in values
                         if isinstance(action, action_geometry.Forward)]
            if len(distances) != len(set(distances)):
                return {"collision": None}
            return {
                "collision": True, "first_contact_arc_m": 0.25,
                "contact_action_index": 1,
                "contact_action_local_arc_m": 0.25,
            }

        def coverage(self, values, max_arc_m=None):
            return 1.0

    class Sim:
        def recompute_navmesh(self, radius, height):
            pass

        def nav(self, position, yaw):
            return object()

    def judge(_frame, body, values, **kwargs):
        judge_call.update(kwargs)
        actions_dict = action_geometry.actions_to_dicts(values)
        contact = {
            "collision": True, "first_contact_arc_m": 0.25,
            "contact_action_index": 1,
            "contact_action_local_arc_m": 0.25,
            "authority": "test", "collision_source": "geometry",
            "contact": None,
        }
        return {
            "body": body.to_dict(), "actions": actions_dict,
            "physical": dict(contact), "depth_physical": dict(contact),
            "oracle_consensus": {"accepted": True},
            "evidence": {"physical": {"coverage": 1.0}},
            "execution": {
                "completed": False, "stop_reason": "collision",
                "nominal_forward_m": action_geometry.total_forward_m(values),
                "executed_forward_m": 0.25, "stop_arc_m": 0.25,
                "realized_pose": {"x": 0.0, "z": 0.25,
                                  "heading_deg": 15.0},
            },
            "checkpoints": [], "provenance": {},
        }

    delta = fixed_pose_actions.collect_turn_group(
        Sim(), object(), record, seed=7, max_programs=100,
        max_full_attempts=4, radius_m=0.2,
        proxy=CollisionProxy(), judge_fn=judge,
        desired_collision=True, lengths=(4,))

    assert delta is not None
    design = delta["outcomes"][0]["a2_design"]
    assert design["collision_action_index_1based"] == 2
    assert design["cell"]["forward_ordinal_1based"] == 1
    assert judge_call["require_contact_instance_witness"] is True
    outcome = delta["outcomes"][0]
    assert validate._a_stability_validation_errors(
        outcome, fixed_pose_actions.parse_actions(outcome["actions"]),
        "[test]") == []
