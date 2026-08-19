"""Trust-boundary regressions for records and counterfactual C1 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline import (
    candidate_preview,
    gate_authority,
    record,
    scene_pool,
    source_manifest,
    validate,
)
from scripts import check_abc_golden
from tests._synthetic import clean_record
from tests.test_pl_c1_counterfactual import _counterfactual_projection
from tests.test_pl_v16_b_validation import _trusted_record


ROOT = Path(__file__).resolve().parents[1]
CHECK_RECORDS = ROOT / "scripts" / "check_records.py"
BUILD_PREVIEW = ROOT / "scripts" / "build_v16_candidate_preview.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_run_meta(path: Path) -> None:
    path.write_text(json.dumps({
        "record_schema_version": record.SCHEMA_VERSION,
    }, sort_keys=True))


def _refresh_private_manifest(artifact: Path) -> None:
    path = artifact / "private" / "manifest.json"
    manifest = json.loads(path.read_text())
    for name in manifest["files"]:
        manifest["files"][name] = _sha256(artifact / "private" / name)
    path.write_text(json.dumps(manifest, sort_keys=True))


def _committed_authority_manifest(
        tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    records = source_root / "records.jsonl"
    if not records.exists():
        _write_jsonl(records, [clean_record()])
    run_meta = source_root / "run_meta.json"
    _write_run_meta(run_meta)
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps({
        "schema": "egoconseq.golden-manifest.v1",
        "name": "counterfactual-c1-source-test",
        "inputs": [{
            "shard": "source",
            "records.jsonl": _sha256(records),
            "run_meta.json": _sha256(run_meta),
        }],
    }, sort_keys=True))
    return manifest, records, run_meta


def _counterfactual_artifact(tmp_path: Path):
    projection = _counterfactual_projection(tmp_path)
    records = tmp_path / "records.jsonl"
    run_meta = tmp_path / "run_meta.json"
    _write_run_meta(run_meta)
    authority = gate_authority.resolve_preview_source_authority_from_inputs(
        records_paths=[records],
        expected_run_meta_sha256=[None],
        authority_id="test:counterfactual-c1-source")
    artifact = tmp_path / "candidate_qa"
    candidate_preview.write_preview_artifact(
        projection, artifact, source_records_path=records,
        expected_source_authority=authority)
    return artifact, authority


def test_local_validation_never_loads_r2r_source_authority(
        tmp_path, monkeypatch):
    rec, context, _spec = _trusted_record(tmp_path, catalog_root=tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("local validation loaded source authority")

    monkeypatch.setattr(
        scene_pool, "_trusted_r2r_scene_for_context", forbidden)
    validate.validate_record_local(rec, context=context, asset_root=tmp_path)


def test_source_bound_validation_fails_when_source_is_unavailable(tmp_path):
    rec, context, _spec = _trusted_record(tmp_path, catalog_root=tmp_path)
    context = source_manifest.RecordValidationContext(
        route=context.route,
        expected_schema_version=context.expected_schema_version,
        expected_oracle_contract_version=
            context.expected_oracle_contract_version,
        expected_collection_contracts=context.expected_collection_contracts,
        authority_sha256=context.authority_sha256,
        r2r_train_episodes=str(tmp_path / "missing-episodes.json.gz"),
        mp3d_root=str(tmp_path / "missing-mp3d"),
        expected_setting_sampling_policy=
            context.expected_setting_sampling_policy,
        expected_action_sampling_policy=
            context.expected_action_sampling_policy,
    )

    errors = validate.validate_record_source_bound(
        rec, context=context, asset_root=tmp_path)
    assert any("trusted R2R source unavailable" in error for error in errors)


def test_check_records_local_level_needs_no_source_authority(tmp_path):
    records = tmp_path / "records.jsonl"
    _write_jsonl(records, [clean_record()])
    result = subprocess.run(
        [sys.executable, str(CHECK_RECORDS), str(records),
         "--validation-level", "local"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 violations" in result.stdout


def test_check_records_requires_an_explicit_validation_level(tmp_path):
    records = tmp_path / "records.jsonl"
    _write_jsonl(records, [clean_record()])
    result = subprocess.run(
        [sys.executable, str(CHECK_RECORDS), str(records)],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "--validation-level" in result.stderr


@pytest.mark.parametrize("option", [
    "--future-view-gate",
    "--future-view-gate-sha256",
    "--future-view-gate-authority-id",
])
def test_preview_builder_rejects_retired_c1_gate_options(tmp_path, option):
    result = subprocess.run(
        [sys.executable, str(BUILD_PREVIEW),
         "--records", str(tmp_path / "records.jsonl"),
         "--run-meta-sha256", "1" * 64,
         "--output", str(tmp_path / "out"),
         "--report", str(tmp_path / "report"), option, "retired"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_committed_manifest_requires_an_explicit_root(tmp_path):
    manifest, _records, _run_meta = _committed_authority_manifest(tmp_path)
    with pytest.raises(ValueError, match="explicit trusted root"):
        gate_authority.resolve_preview_source_authority_from_manifest(manifest)


def test_committed_manifest_rejects_a_final_symlink(tmp_path):
    manifest, _records, _run_meta = _committed_authority_manifest(tmp_path)
    link = tmp_path / "manifest-link.json"
    link.symlink_to(manifest)
    with pytest.raises(ValueError, match="manifest cannot be opened"):
        gate_authority.resolve_preview_source_authority_from_manifest(
            link, root=tmp_path)


def test_committed_manifest_resolves_repo_relative_sources(tmp_path):
    manifest, records, run_meta = _committed_authority_manifest(tmp_path)
    authority = gate_authority.resolve_preview_source_authority_from_manifest(
        manifest, root=tmp_path)
    assert authority.authority_kind == \
        gate_authority.COMMITTED_MANIFEST_AUTHORITY_KIND
    assert authority.sources == ({
        "path": str(records.absolute()),
        "records_sha256": _sha256(records),
        "run_meta_sha256": _sha256(run_meta),
    },)


def test_committed_authority_binding_is_checkout_relative(tmp_path):
    bindings = []
    for name in ("checkout-a", "checkout-b"):
        root = tmp_path / name
        root.mkdir()
        manifest, _records, _run_meta = _committed_authority_manifest(root)
        authority = \
            gate_authority.resolve_preview_source_authority_from_manifest(
                manifest, root=root)
        bindings.append(authority.binding())
    assert bindings[0] == bindings[1]
    assert str(tmp_path) not in json.dumps(bindings[0], sort_keys=True)
    assert bindings[0]["sources"][0]["locator"] == {
        "kind": "authority_root_relative",
        "path": "source/records.jsonl",
    }


def _builder_args(tmp_path: Path, manifest: Path, records: Path):
    from scripts.build_v16_candidate_preview import build_arg_parser
    return build_arg_parser().parse_args([
        "--records", str(records.relative_to(tmp_path)),
        "--run-meta-sha256", _sha256(records.with_name("run_meta.json")),
        "--source-authority-manifest", str(manifest),
        "--output", str(tmp_path / "out"),
        "--report", str(tmp_path / "report"),
    ])


def test_formal_builder_resolves_committed_manifest_authority(tmp_path):
    from scripts import build_v16_candidate_preview
    manifest, records, _run_meta = _committed_authority_manifest(tmp_path)
    authority = build_v16_candidate_preview.resolve_build_authority(
        _builder_args(tmp_path, manifest, records), root=tmp_path)
    assert authority.authority_kind == \
        gate_authority.COMMITTED_MANIFEST_AUTHORITY_KIND


def test_formal_builder_rejects_manifest_source_conflict(tmp_path):
    from scripts import build_v16_candidate_preview
    manifest, records, _run_meta = _committed_authority_manifest(tmp_path)
    args = _builder_args(tmp_path, manifest, records)
    args.run_meta_sha256 = ["0" * 64]
    with pytest.raises(ValueError, match="conflict with committed manifest"):
        build_v16_candidate_preview.resolve_build_authority(
            args, root=tmp_path)


def test_source_authority_identity_is_persisted_in_source_map(tmp_path):
    artifact, authority = _counterfactual_artifact(tmp_path)
    source_map = json.loads(
        (artifact / "private" / "source_map.json").read_text())
    assert source_map["source_authority"] == authority.binding()
    assert "future_view_gate" not in source_map["source_authority"]
    candidate_preview.validate_preview_artifact(
        artifact, expected_source_authority=authority)


def test_standalone_preview_consumers_require_external_source_authority(
        tmp_path):
    artifact, _authority = _counterfactual_artifact(tmp_path)
    with pytest.raises(ValueError, match="external source authority required"):
        candidate_preview.validate_preview_artifact(artifact)
    with pytest.raises(ValueError, match="external source authority required"):
        candidate_preview.evaluate_preview_artifact(
            artifact, gt_as_pred=True)
    with pytest.raises(ValueError, match="external source authority required"):
        candidate_preview.render_preview_html(
            artifact / "benchmark.json", tmp_path / "report")


def test_public_renderer_rejects_a_validated_dict_bypass(tmp_path):
    artifact, _authority = _counterfactual_artifact(tmp_path)
    forged = json.loads((artifact / "benchmark.json").read_text())
    with pytest.raises(TypeError, match="validated_benchmark"):
        candidate_preview.render_preview_html(
            artifact / "benchmark.json", tmp_path / "report",
            validated_benchmark=forged)


def test_build_routes_only_its_validated_value_to_private_renderer():
    assert hasattr(candidate_preview, "_render_preview_html_validated")
    names = set(candidate_preview.build_main_preview.__code__.co_names)
    assert "_render_preview_html_validated" in names
    assert "render_preview_html" not in names


@pytest.mark.parametrize("substitution", ["records", "run_meta"])
def test_preview_rejects_coordinated_source_substitution(
        tmp_path, substitution):
    artifact, authority = _counterfactual_artifact(tmp_path)
    source_map_path = artifact / "private" / "source_map.json"
    source_map = json.loads(source_map_path.read_text())
    binding = source_map["source_authority"]
    path = tmp_path / (
        "records.jsonl" if substitution == "records" else "run_meta.json")
    path.write_bytes(path.read_bytes() + b"\n")
    key = ("records_sha256" if substitution == "records" else
           "run_meta_sha256")
    binding["sources"][0][key] = _sha256(path)
    source_map_path.write_text(json.dumps(source_map, sort_keys=True))
    _refresh_private_manifest(artifact)
    with pytest.raises(ValueError, match="source authority binding"):
        candidate_preview.validate_preview_artifact(
            artifact, expected_source_authority=authority)


@pytest.mark.parametrize("forgery", [
    "member_action_sha", "option_permutation", "family_identity",
])
def test_preview_rejects_a_self_hashed_counterfactual_forgery(
        tmp_path, forgery):
    artifact, authority = _counterfactual_artifact(tmp_path)
    answers_path = artifact / "private" / "answers.jsonl"
    answers = [json.loads(line)
               for line in answers_path.read_text().splitlines()]
    answer = next(row for row in answers
                  if row["task_id"] == "C1_future_view_selection")
    certificate = answer["selection_certificate"]
    if forgery == "member_action_sha":
        certificate["choices"][0]["action_sha256"] = "0" * 64
    elif forgery == "option_permutation":
        certificate["permutation_outcome_ids"][:2] = list(reversed(
            certificate["permutation_outcome_ids"][:2]))
    else:
        certificate["family_id"] = "forged-family"
    certificate["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in certificate.items() if key != "sha256"
    })
    answer["oracle_ref"]["future_view_selection_sha256"] = \
        certificate["sha256"]
    _write_jsonl(answers_path, answers)
    _refresh_private_manifest(artifact)
    benchmark_path = artifact / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    case = next(row for row in benchmark["cases"]
                if row["public"]["task_id"] ==
                "C1_future_view_selection")
    case["oracle_ref"]["future_view_selection_sha256"] = \
        certificate["sha256"]
    benchmark_path.write_text(json.dumps(benchmark, sort_keys=True))
    with pytest.raises(
            ValueError, match="counterfactual selection certificate changed"):
        candidate_preview.validate_preview_artifact(
            artifact, expected_source_authority=authority)


def test_active_golden_rejects_retired_c1_gate_contract():
    manifest = {
        "required_gate": {"path": "retired.json", "sha256": "0" * 64},
        "build_argv": [
            "python", "scripts/build_v16_candidate_preview.py",
            "--future-view-gate", "retired.json",
        ],
    }
    assert check_abc_golden._check_no_retired_c1_gate(manifest) == [
        "active golden manifest contains retired C1 gate",
        ("active golden build argv contains retired C1 gate option(s): "
         "--future-view-gate"),
    ]


def test_active_golden_accepts_counterfactual_only_build_argv():
    assert check_abc_golden._check_no_retired_c1_gate({
        "build_argv": [
            "python", "scripts/build_v16_candidate_preview.py",
            "--records", "source/records.jsonl",
        ],
    }) == []
