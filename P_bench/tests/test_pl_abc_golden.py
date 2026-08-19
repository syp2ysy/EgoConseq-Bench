import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import check_abc_golden


ROOT = Path(__file__).resolve().parents[1]


def test_golden_digest_gate_rejects_unregistered_rebuilt_file(tmp_path):
    expected = tmp_path / "candidate_qa" / "benchmark.json"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"registered")
    extra = tmp_path / "candidate_qa" / "private" / \
        "record_contexts.jsonl"
    extra.parent.mkdir()
    extra.write_bytes(b"new private table\n")
    manifest = {"outputs_sha256": {
        "candidate_qa/benchmark.json": hashlib.sha256(
            b"registered").hexdigest(),
    }}

    failures = check_abc_golden._check_digests(manifest, tmp_path)

    assert failures == [
        "unexpected rebuilt file: "
        "candidate_qa/private/record_contexts.jsonl"]


def test_golden_gate_checks_every_declared_collection_input(tmp_path):
    shard = tmp_path / "shard"
    shard.mkdir()
    values = {
        "records.jsonl": b"record\n",
        "run_meta.json": b"meta\n",
        "collection_funnel.json": b"funnel\n",
    }
    for name, payload in values.items():
        (shard / name).write_bytes(payload)
    action = tmp_path / "actions.json"
    action.write_bytes(b"actions\n")
    manifest = {
        "inputs": [{
            "shard": "shard",
            **{
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in values.items()
            },
        }],
        "collection_inputs_sha256": {
            "actions.json": hashlib.sha256(b"actions\n").hexdigest(),
        },
    }

    assert check_abc_golden._check_input_digests(
        manifest, root=tmp_path) == []

    (shard / "records.jsonl").write_bytes(b"changed\n")
    action.unlink()

    assert check_abc_golden._check_input_digests(
        manifest, root=tmp_path) == [
        "input digest changed: shard/records.jsonl",
        "missing collection input: actions.json",
    ]


def test_golden_gate_rejects_inaccurate_superseded_diff_accounting(tmp_path):
    superseded_path = tmp_path / "docs" / "golden" / "superseded.json"
    superseded_path.parent.mkdir(parents=True)
    superseded_bytes = json.dumps({
        "outputs_sha256": {
            "candidate_qa/benchmark.json": "old-benchmark",
            "candidate_qa/report.json": "same-report",
            "candidate_qa/public/images/example.png": "same-image",
        },
    }).encode()
    superseded_path.write_bytes(superseded_bytes)
    manifest = {
        "outputs_sha256": {
            "candidate_qa/benchmark.json": "new-benchmark",
            "candidate_qa/report.json": "same-report",
            "candidate_qa/public/images/example.png": "same-image",
        },
        "semantic_diff_from_superseded": {
            "superseded_manifest": "docs/golden/superseded.json",
            "superseded_manifest_sha256": hashlib.sha256(
                superseded_bytes).hexdigest(),
            "changed_non_image_outputs": [
                "candidate_qa/benchmark.json",
                "candidate_qa/report.json",
            ],
            "image_outputs_changed": False,
            "unchanged_output_count": 1,
        },
    }

    assert check_abc_golden._check_semantic_diff(
        manifest, root=tmp_path) == [
        "declared changed non-image outputs do not match superseded diff: "
        "expected ['candidate_qa/benchmark.json'], declared "
        "['candidate_qa/benchmark.json', 'candidate_qa/report.json']",
        "declared unchanged output count does not match superseded diff: "
        "expected 2, declared 1",
    ]


def test_active_golden_diff_matches_archived_superseded_manifest():
    manifest_path = check_abc_golden.DEFAULT_MANIFEST
    manifest = json.loads(manifest_path.read_text())

    assert check_abc_golden._check_semantic_diff(manifest) == []


def test_active_golden_covers_all_three_datasets_and_documents_r2r_a1_gap():
    manifest = json.loads(check_abc_golden.DEFAULT_MANIFEST.read_text())

    assert manifest["datasets"] == ["r2r", "gs", "b1k"]
    assert {entry["dataset"] for entry in manifest["inputs"]} == {
        "r2r", "gs", "b1k"}
    r2r = next(
        entry for entry in manifest["inputs"] if entry["dataset"] == "r2r")
    assert r2r["expected_zero_tasks"] == ["A1_collision"]


def test_golden_replay_requires_official_six_task_macro(monkeypatch, tmp_path):
    monkeypatch.setattr(
        check_abc_golden, "_run",
        lambda _argv: SimpleNamespace(
            returncode=0, stderr="", stdout=json.dumps({
                "overall": 1.0, "six_task_macro": 0.5})))

    assert check_abc_golden._check_gt_replay(
        tmp_path, source_authority_manifest=tmp_path / "authority.json") == [
            "gt-as-pred six-task macro is 0.5, not 1.0"]
