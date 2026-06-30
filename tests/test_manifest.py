"""
Tests for egoconseq.manifest: Case schema, round-trip I/O, model_payload isolation.
"""
from egoconseq.manifest import Case, write_jsonl, read_jsonl


def test_roundtrip(tmp_path):
    c = Case(
        case_id="x1",
        operation_id="O5",
        readout_tag="pair_flip",
        scene_id="s",
        pose=[0, 0, 0, 0],
        body={"radius_m": 0.25},
        action={"type": "forward", "horizon_m": 1.0, "horizon_reference": "metric_fixed"},
        d_safe_visible_m=1.4,
        d_safe_navmesh_m=1.5,
        oracle_agreement="agree",
        gates={"visible_sweep_ratio": 0.82},
        answer={"answer_type": "pair_flip", "label": "small_only"},
        tags={"geometry_tag": "narrow-gap"},
        group_id="g1",
    )
    p = tmp_path / "m.jsonl"
    write_jsonl([c], p)
    got = read_jsonl(p)
    assert got[0].operation_id == "O5"
    assert got[0].answer["label"] == "small_only"


def test_model_payload_keys_only():
    """model_payload() must expose ONLY image_path and question — nothing else."""
    c = Case(
        case_id="x2",
        image_path="/data/imgs/x2.jpg",
        question="Will the robot collide?",
        pose=[1.0, 2.0, 0.5, 0.0],
        d_safe_navmesh_m=0.8,
    )
    payload = c.model_payload()
    assert set(payload.keys()) == {"image_path", "question"}, (
        f"model_payload must expose exactly {{image_path, question}}, got {set(payload.keys())}"
    )
    assert payload["image_path"] == "/data/imgs/x2.jpg"
    assert payload["question"] == "Will the robot collide?"


def test_defaults_allow_partial_construction():
    """All §8.2 fields must have defaults so partial construction works."""
    c = Case(case_id="minimal")
    # spot-check a handful of optional fields
    assert c.episode_id is None
    assert c.image_path is None
    assert c.question is None
    assert c.oracle_agreement is None
    assert c.contact_point_3d is None
    assert c.requires_hidden_geometry is None
    assert c.fov_supported is None
    assert c.group_id is None
    # grouped dict fields default to empty dict
    assert c.body == {}
    assert c.action == {}
    assert c.gates == {}
    assert c.answer == {}
    assert c.tags == {}
    assert c.sensor_profile == {}


def test_roundtrip_preserves_nested():
    """Nested dicts and lists survive write_jsonl → read_jsonl without mutation."""
    c = Case(
        case_id="nest1",
        pose=[0.1, 0.2, 0.3, 1.57],
        sensor_profile={"camera_height": 1.5, "hfov": 79, "resolution": [640, 480]},
        body={"radius_m": 0.18, "width_m": 0.36, "height_m": 1.5},
        action={"type": "turn", "angle_deg": 30},
        answer={"answer_type": "pair_flip", "label": "large_only"},
        contact_point_3d=[1.1, 2.2, 3.3],
        contact_pixel=[320, 240],
        d_safe_visible_body_widths=2.5,
        margin_body_widths=0.5,
    )
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "nested.jsonl"
        write_jsonl([c], p)
        got = read_jsonl(p)[0]
    assert got.pose == [0.1, 0.2, 0.3, 1.57]
    assert got.sensor_profile["resolution"] == [640, 480]
    assert got.body["width_m"] == 0.36
    assert got.action["angle_deg"] == 30
    assert got.contact_point_3d == [1.1, 2.2, 3.3]
    assert got.contact_pixel == [320, 240]
    assert got.d_safe_visible_body_widths == 2.5
    assert got.margin_body_widths == 0.5


def test_to_dict_from_dict_identity():
    """to_dict / from_dict round-trip must be lossless (without file I/O)."""
    c = Case(
        case_id="id_test",
        operation_id="O1",
        readout_tag="single",
        oracle_agreement="disagree",
        gates={"valid_depth_ratio": 0.9, "depth_hole_ratio": 0.05},
        tags={"difficulty": "hard", "body_radius": "wide"},
    )
    d = c.to_dict()
    c2 = Case.from_dict(d)
    assert c2.case_id == c.case_id
    assert c2.gates["valid_depth_ratio"] == 0.9
    assert c2.tags["difficulty"] == "hard"
    assert c2.to_dict() == d
