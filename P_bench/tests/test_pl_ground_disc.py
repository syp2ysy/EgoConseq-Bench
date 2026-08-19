"""Ground-disc contract tests independent of Habitat."""

import json

import pytest

from pipeline import record
from pipeline.geometry import Disc


def test_disc_public_contract_contains_radius_only():
    body = Disc(radius_m=0.25)

    assert body.to_dict() == {"shape": "disc", "radius_m": 0.25}
    assert not hasattr(body, "height_m")
    with pytest.raises(ValueError):
        Disc(radius_m=0.0)


def test_current_records_require_ground_disc_oracle_contract(tmp_path):
    current = {
        "schema_version": record.SCHEMA_VERSION,
        "frame_id": "f",
        "objects": [],
        "outcomes": [],
    }
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(current) + "\n")

    with pytest.raises(ValueError, match="oracle contract"):
        list(record.read_records(path))
