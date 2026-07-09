"""Validator: clean record passes; each mutation triggers the right V-check."""

import copy

from pipeline import record, consequence, validate
from pipeline.body import Cylinder
from pipeline.actions import Forward, Turn
from tests._synthetic import make_frame


def _clean_record():
    fr = make_frame()
    ocs = []
    for i, acts in enumerate([[Forward(1.0)], [Turn(90)], [Turn(90), Forward(1.0)]]):
        oc = consequence.judge(fr, Cylinder(0.25), acts, nav=None)
        oc["outcome_id"] = f"o{i}"
        ocs.append(oc)
    return record.build_record(fr, ocs, image_path="img/x.png")


def _viol_codes(rec):
    return " ".join(validate.validate_record(rec))


def test_clean_record_passes():
    assert validate.validate_record(_clean_record()) == []


def test_v4_pose_end_full_mutation():
    rec = _clean_record()
    rec["outcomes"][0]["pose_end_full"]["x"] += 0.5
    assert "V4" in _viol_codes(rec)


def test_v3_collided_contact_inconsistency():
    rec = _clean_record()
    # pick a non-collided outcome (pure turn) and flip collided -> inconsistent
    oc = next(o for o in rec["outcomes"] if not o["collided"])
    oc["collided"] = True   # arc/contact still None
    assert "V3" in _viol_codes(rec)


def test_v2_pure_turn_distance_changed():
    rec = _clean_record()
    # outcome o1 is the pure turn [Turn(90)]
    turn = next(o for o in rec["outcomes"] if o["total_forward_m"] == 0)
    turn["object_relations"][0]["delta_full"]["d_centroid_m"] = 0.9
    assert "V2" in _viol_codes(rec)


def test_v1_triangle_inequality():
    rec = _clean_record()
    rec["outcomes"][0]["object_relations"][0]["full"]["dist_centroid_m"] += 5.0
    assert "V1" in _viol_codes(rec)


def test_v6_in_fov_bearing():
    rec = _clean_record()
    r = rec["outcomes"][0]["object_relations"][0]["full"]
    r["in_fov"] = True; r["bearing_deg"] = 120.0
    assert "V6" in _viol_codes(rec)


def test_v8_end_heading_offset():
    rec = _clean_record()
    rec["outcomes"][0]["view_exit"]["end_heading_offset_deg"] += 20.0
    assert "V8" in _viol_codes(rec)


def test_v5_object_bearing():
    rec = _clean_record()
    rec["objects"][0]["bearing_deg"] += 30.0
    assert "V5" in _viol_codes(rec)
