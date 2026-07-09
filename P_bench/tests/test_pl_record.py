"""Record round-trip + numpy->python conversion."""

import numpy as np

from pipeline import record, consequence
from pipeline.body import Cylinder
from pipeline.actions import Forward, Turn
from tests._synthetic import make_frame


def _jsonable_recursive(o):
    """Assert no numpy scalars survive serialization."""
    if isinstance(o, dict):
        return all(_jsonable_recursive(v) for v in o.values())
    if isinstance(o, list):
        return all(_jsonable_recursive(v) for v in o)
    return not isinstance(o, (np.generic, np.ndarray))


def test_build_record_roundtrip(tmp_path):
    fr = make_frame()
    ocs = []
    for i, acts in enumerate([[Forward(1.0)], [Turn(90)]]):
        oc = consequence.judge(fr, Cylinder(0.25), acts, nav=None)
        oc["outcome_id"] = f"o{i}"
        ocs.append(oc)
    rec = record.build_record(fr, ocs, image_path="img/x.png")

    assert _jsonable_recursive(rec)                       # numpy fully converted
    assert "_points_xz" not in rec["objects"][0]          # transient stripped
    assert rec["objects"][0]["dist_geodesic_m"] is None   # key present

    path = tmp_path / "r.jsonl"
    record.append_record(str(path), rec)
    record.append_record(str(path), rec)                  # append-style
    loaded = list(record.read_records(str(path)))
    assert len(loaded) == 2
    assert loaded[0]["frame_id"] == fr.frame_id
    assert loaded[0]["n_objects"] == len(rec["objects"])
    assert loaded[0]["outcomes"][0]["outcome_id"] == "o0"
