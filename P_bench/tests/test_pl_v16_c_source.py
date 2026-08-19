"""Terminal PNG record, source-snapshot, and collection wiring tests."""

import copy
import dataclasses
import json

import numpy as np
import pytest

from pipeline import (
    collection_runtime, config, future_view_selection, record,
    source_manifest, validate,
)
from tests.test_pl_v16_c_assets import _materialize


def _record_with_terminal(tmp_path):
    atom, base, outcome, _terminal, rec = _materialize(tmp_path)
    outcome["terminal_rgb_asset"] = atom
    outcome["base_rollout_key"] = atom["binding"]["base_rollout_key"]
    rec.update({
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "scene_id": base.scene_id,
        "frame_id": base.frame_id,
        "pose": {
            "position": base.position.tolist(),
            "yaw_rad": base.yaw_rad,
        },
        "sensor": base.sensor.to_dict(),
        "outcomes": [outcome],
    })
    return rec, outcome, atom


def test_terminal_asset_validator_rederives_record_fields_and_png(tmp_path):
    """Catches a self-hashed atom being trusted without its record or bytes."""
    rec, outcome, atom = _record_with_terminal(tmp_path)

    snapshot = future_view_selection.validate_terminal_rgb_asset(
        rec, outcome, asset_root=tmp_path)

    assert snapshot.png_sha256 == atom["png_sha256"]
    assert snapshot.pixel_sha256 == atom["pixel_sha256"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("binding", "action_sha256"), "0" * 64, "binding"),
        (("terminal_pose", "local", "z"), 3.0, "terminal pose"),
        (("sensor", "hfov_deg"), 110.0, "sensor"),
        (("source", "source_assets_sha256"), "0" * 64, "source"),
        (("renderer", "protocol"), "other", "renderer"),
    ],
)
def test_terminal_asset_validator_rejects_coordinated_atom_tamper(
        tmp_path, path, value, message):
    """Catches recomputing only the atom SHA after changing trusted bindings."""
    rec, outcome, _atom = _record_with_terminal(tmp_path)
    cursor = outcome["terminal_rgb_asset"]
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    atom = outcome["terminal_rgb_asset"]
    atom["sha256"] = record.canonical_atom_sha256({
        key: item for key, item in atom.items() if key != "sha256"
    })

    with pytest.raises(ValueError, match=message):
        future_view_selection.validate_terminal_rgb_asset(
            rec, outcome, asset_root=tmp_path)


def test_terminal_asset_validator_rejects_changed_or_symlinked_png(tmp_path):
    """Catches record-valid metadata authorizing replaced terminal bytes."""
    rec, outcome, atom = _record_with_terminal(tmp_path)
    path = tmp_path / atom["path"]
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="payload identity"):
        future_view_selection.validate_terminal_rgb_asset(
            rec, outcome, asset_root=tmp_path)

    path.unlink()
    outside = tmp_path.parent / f"{tmp_path.name}-terminal.png"
    outside.write_bytes(original)
    path.symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="canonical|escapes|symlink"):
            future_view_selection.validate_terminal_rgb_asset(
                rec, outcome, asset_root=tmp_path)
    finally:
        outside.unlink(missing_ok=True)


def test_record_validator_routes_terminal_atom_through_asset_bytes(tmp_path):
    """Catches the production record validator omitting terminal-byte checks."""
    rec, outcome, atom = _record_with_terminal(tmp_path)
    clean = validate.validate_record_local(
        rec, context=source_manifest.LEGACY_RECORD_VALIDATION_CONTEXT,
        asset_root=tmp_path)
    assert not any("terminal RGB" in error for error in clean)
    (tmp_path / atom["path"]).write_bytes(b"tampered")

    errors = validate.validate_record_local(
        rec, context=source_manifest.LEGACY_RECORD_VALIDATION_CONTEXT,
        asset_root=tmp_path)

    assert any("terminal RGB payload identity" in error for error in errors)


def test_collection_attaches_only_existing_cached_clear_terminal(tmp_path):
    """Catches a missing cache triggering a rerender or a collision frame asset."""
    atom, base, clear, terminal, rec = _materialize(tmp_path / "seed")
    clear.pop("terminal_rgb_asset", None)
    collision = copy.deepcopy(clear)
    collision["outcome_id"] = "collision"
    collision["physical"]["collision"] = True
    collision["execution"].update(completed=False, stop_reason="collision")
    missing = copy.deepcopy(clear)
    missing["outcome_id"] = "missing"
    missing["checkpoints"][-1]["pose"]["x"] = 0.5
    missing["execution"]["realized_pose"]["x"] = 0.5
    key = future_view_selection.terminal_render_cache_key(clear)
    contract = record.r2r_v16_collection_contract(rec["source"], "main")

    result = collection_runtime.attach_terminal_rgb_assets(
        tmp_path / "shard", base, [clear, collision, missing],
        {key: terminal}, source=rec["source"],
        collection_contract=contract)

    assert result == {"materialized": 1, "withheld": 1}
    assert clear["terminal_rgb_asset"]["schema"] == \
        "terminal-rgb-asset.v1"
    assert "terminal_rgb_asset" not in collision
    assert missing["terminal_rgb_asset_withhold"] == \
        "terminal_cache_miss"
    assert missing["terminal_rgb_asset_withhold_authority"] == \
        "collection_runtime_attested"


def test_natural_collection_renders_only_missing_clear_terminal(tmp_path):
    """The lazy RGB renderer runs after clear certification, never collision."""
    _atom, base, clear, terminal, rec = _materialize(tmp_path / "seed")
    clear.pop("terminal_rgb_asset", None)
    collision = copy.deepcopy(clear)
    collision["outcome_id"] = "collision"
    collision["physical"]["collision"] = True
    collision["execution"].update(completed=False, stop_reason="collision")
    cache = {}
    calls = []

    def render(pose):
        calls.append(pose)
        cache[future_view_selection.terminal_render_cache_key(clear)] = terminal

    result = collection_runtime.attach_terminal_rgb_assets(
        tmp_path / "shard", base, [clear, collision], cache,
        source=rec["source"],
        collection_contract=record.r2r_v16_collection_contract(
            rec["source"], "main"),
        terminal_renderer=render,
    )

    expected = clear["checkpoints"][-1]["pose"]
    assert calls == [tuple(expected[field] for field in (
        "x", "z", "heading_deg"))]
    assert result == {"materialized": 1, "withheld": 0}
    assert "terminal_rgb_asset" in clear
    assert "terminal_rgb_asset" not in collision


def test_terminal_publication_depends_only_on_the_current_outcome(tmp_path):
    _atom, base, clear, terminal, rec = _materialize(tmp_path / "seed")
    clear.pop("terminal_rgb_asset", None)
    key = future_view_selection.terminal_render_cache_key(clear)
    contract = record.r2r_v16_collection_contract(rec["source"], "main")

    result = collection_runtime.attach_terminal_rgb_assets(
        tmp_path / "shard", base, [clear], {key: terminal},
        source=rec["source"], collection_contract=contract)

    assert result == {"materialized": 1, "withheld": 0}


def test_terminal_checkpoint_accepts_only_ulp_scale_endpoint_drift(tmp_path):
    """Catches exact float equality rejecting the same analytic endpoint."""
    _atom, _base, clear, _terminal, _rec = _materialize(tmp_path)
    endpoint = clear["execution"]["realized_pose"]
    checkpoint = clear["checkpoints"][-1]["pose"]
    checkpoint["x"] = float(endpoint["x"]) + 5e-13
    checkpoint["z"] = float(endpoint["z"]) - 5e-13
    checkpoint["heading_deg"] = float(endpoint["heading_deg"]) + 5e-13

    key = future_view_selection.terminal_render_cache_key(clear)

    assert key == tuple(round(float(endpoint[field]), 6) for field in (
        "x", "z", "heading_deg"))

    checkpoint["x"] = float(endpoint["x"]) + 2e-9
    with pytest.raises(ValueError, match="disagrees with realized endpoint"):
        future_view_selection.terminal_render_cache_key(clear)


def test_terminal_checkpoint_uses_dimensioned_position_and_heading_tolerances(
        tmp_path, monkeypatch):
    _atom, _base, clear, _terminal, _rec = _materialize(tmp_path)
    endpoint = clear["execution"]["realized_pose"]
    checkpoint = clear["checkpoints"][-1]["pose"]
    monkeypatch.setattr(config, "C1_TERMINAL_POSITION_TOL_M", 1e-8)
    monkeypatch.setattr(config, "C1_TERMINAL_HEADING_TOL_DEG", 1e-10)
    checkpoint["x"] = float(endpoint["x"]) + 2e-9

    future_view_selection.terminal_render_cache_key(clear)

    checkpoint["heading_deg"] = float(endpoint["heading_deg"]) + 2e-9
    with pytest.raises(
            future_view_selection.TerminalRGBAssetError) as captured:
        future_view_selection.terminal_render_cache_key(clear)
    assert captured.value.reason == "terminal_pose_disagreement"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda outcome: outcome.__setitem__("checkpoints", []),
         "terminal_checkpoint_missing"),
        (lambda outcome: outcome["checkpoints"][-1].__setitem__(
            "requested_progress", "bad"),
         "terminal_checkpoint_incomplete"),
        (lambda outcome: outcome["checkpoints"][-1]["pose"].__setitem__(
            "x", "bad"),
         "terminal_checkpoint_pose_invalid"),
        (lambda outcome: outcome["checkpoints"][-1]["pose"].__setitem__(
            "x", float(outcome["execution"]["realized_pose"]["x"]) + 2e-9),
         "terminal_pose_disagreement"),
    ],
)
def test_checkpoint_failures_carry_typed_record_recomputable_reasons(
        tmp_path, mutation, reason):
    _atom, _base, clear, _terminal, _rec = _materialize(tmp_path)
    mutation(clear)

    with pytest.raises(
            future_view_selection.TerminalRGBAssetError) as captured:
        future_view_selection.terminal_render_cache_key(clear)

    assert captured.value.reason == reason
    assert captured.value.authority == "record_recomputable"


def test_collection_does_not_turn_unknown_value_error_into_withhold(
        tmp_path, monkeypatch):
    _atom, base, clear, _terminal, rec = _materialize(tmp_path / "seed")
    clear.pop("terminal_rgb_asset", None)
    contract = record.r2r_v16_collection_contract(rec["source"], "main")

    def fail_unknown(*_args, **_kwargs):
        raise ValueError("implementation regression")

    monkeypatch.setattr(
        future_view_selection, "materialize_terminal_rgb_asset", fail_unknown)

    with pytest.raises(ValueError, match="implementation regression"):
        collection_runtime.attach_terminal_rgb_assets(
            tmp_path / "shard", base, [clear], {}, source=rec["source"],
            collection_contract=contract)


@pytest.mark.parametrize(
    ("mutation", "reason", "authority"),
    [
        ("source", "terminal_provenance_invalid", "record_recomputable"),
        ("cache", "terminal_provenance_invalid",
         "collection_runtime_attested"),
        ("encoding", "terminal_encoding_invalid",
         "collection_runtime_attested"),
        ("publication", "terminal_publication_invalid",
         "collection_runtime_attested"),
    ],
)
def test_materialization_failures_are_typed_at_the_failing_operation(
        tmp_path, monkeypatch, mutation, reason, authority):
    _atom, base, clear, terminal, rec = _materialize(tmp_path / "seed")
    clear.pop("terminal_rgb_asset", None)
    source = copy.deepcopy(rec["source"])
    cached = terminal
    if mutation == "source":
        source["scene_id"] = "another-scene"
    elif mutation == "cache":
        cached = dataclasses.replace(
            terminal, position=np.asarray(terminal.position) + [1.0, 0.0, 0.0])
    elif mutation == "encoding":
        cached = dataclasses.replace(
            terminal, rgb=np.asarray(terminal.rgb, dtype=np.float32))
    else:
        monkeypatch.setattr(
            future_view_selection, "_publish_terminal_png",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("publication regression")))
    contract = record.r2r_v16_collection_contract(source, "main")
    key = future_view_selection.terminal_render_cache_key(clear)

    with pytest.raises(
            future_view_selection.TerminalRGBAssetError) as captured:
        future_view_selection.materialize_terminal_rgb_asset(
            tmp_path / "shard", base_frame=base, outcome=clear,
            render_cache={key: cached}, source=source,
            collection_contract=contract)

    assert captured.value.reason == reason
    assert captured.value.authority == authority


@pytest.mark.parametrize(
    ("reason", "authority"),
    [
        ("terminal_checkpoint_missing", "record_recomputable"),
        ("terminal_checkpoint_incomplete", "record_recomputable"),
        ("terminal_checkpoint_pose_invalid", "record_recomputable"),
        ("terminal_pose_disagreement", "record_recomputable"),
        ("terminal_cache_miss", "collection_runtime_attested"),
        ("terminal_provenance_invalid", "record_recomputable"),
        ("terminal_provenance_invalid", "collection_runtime_attested"),
        ("terminal_encoding_invalid", "collection_runtime_attested"),
        ("terminal_publication_invalid", "collection_runtime_attested"),
    ],
)
def test_typed_withhold_accepts_only_the_frozen_legal_matrix(
        reason, authority):
    outcome = {
        "terminal_rgb_asset_withhold": reason,
        "terminal_rgb_asset_withhold_authority": authority,
    }

    errors = validate._terminal_asset_withhold_errors(
        "[f:o]", outcome, require_typed=True)

    assert not [error for error in errors if "reason/authority" in error]


@pytest.mark.parametrize(
    ("reason", "authority"),
    [
        ("terminal_cache_miss", "record_recomputable"),
        ("terminal_checkpoint_missing", "collection_runtime_attested"),
        ("terminal_encoding_invalid", "record_recomputable"),
        ("terminal_publication_invalid", "record_recomputable"),
    ],
)
def test_typed_withhold_rejects_illegal_reason_authority_pairs(
        reason, authority):
    outcome = {
        "terminal_rgb_asset_withhold": reason,
        "terminal_rgb_asset_withhold_authority": authority,
    }

    errors = validate._terminal_asset_withhold_errors(
        "[f:o]", outcome, require_typed=True)

    assert any("reason/authority" in error for error in errors)


def test_legacy_policy_allows_old_withhold_but_suffix_policy_rejects_it():
    outcome = {
        "terminal_rgb_asset_withhold": "terminal_rgb_asset_invalid",
    }

    assert validate._terminal_asset_withhold_errors(
        "[f:o]", outcome, require_typed=False) == []
    assert any(
        "authority" in error
        for error in validate._terminal_asset_withhold_errors(
            "[f:o]", outcome, require_typed=True))


def test_terminal_asset_and_withhold_are_mutually_exclusive():
    outcome = {
        "terminal_rgb_asset": {"schema": "terminal-rgb-asset.v1"},
        "terminal_rgb_asset_withhold": "terminal_cache_miss",
        "terminal_rgb_asset_withhold_authority":
            "collection_runtime_attested",
    }

    errors = validate._terminal_asset_withhold_errors(
        "[f:o]", outcome, require_typed=True)

    assert any("mutually exclusive" in error for error in errors)


def test_record_recomputable_provenance_withhold_must_match_record(tmp_path):
    rec, outcome, _atom = _record_with_terminal(tmp_path)
    outcome.pop("terminal_rgb_asset")
    outcome.update({
        "terminal_rgb_asset_withhold": "terminal_provenance_invalid",
        "terminal_rgb_asset_withhold_authority": "record_recomputable",
    })

    clean_errors = validate._terminal_asset_withhold_errors(
        "[f:o]", outcome, require_typed=True, rec=rec)
    assert any("not reproduced" in error for error in clean_errors)

    rec["collection_contract"] = {"invalid": True}
    invalid_errors = validate._terminal_asset_withhold_errors(
        "[f:o]", outcome, require_typed=True, rec=rec)
    assert not any("not reproduced" in error for error in invalid_errors)
