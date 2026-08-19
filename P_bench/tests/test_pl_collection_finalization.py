"""The collector's finalization atom is the one downstream trust boundary."""

import json

import pytest

from pipeline import collection_closeout


def _write(path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_v2_finalization_authenticates_the_finished_shard_once(tmp_path):
    records = tmp_path / "records.jsonl"
    run_meta = tmp_path / "run_meta.json"
    funnel = tmp_path / "collection_funnel.json"
    _write(records, {"record": 1})
    _write(run_meta, {"run_contract_sha256": "a" * 64})
    _write(funnel, {"status": "failed"})

    marker = collection_closeout.begin_finalization(
        tmp_path, run_contract_sha256="a" * 64)
    collection_closeout.seal_finalization(
        tmp_path,
        marker,
        records_path=records,
        run_meta_path=run_meta,
        funnel_path=funnel,
        status="partial",
        source_validation="passed",
        record_count=1,
    )

    value = collection_closeout.load_finalization(
        tmp_path,
        expected_run_contract_sha256="a" * 64,
        allowed_statuses={"partial"},
    )
    assert value["schema"] == "egoconseq.collection-finalization.v2"
    assert value["source_validation"] == "passed"
    assert value["record_count"] == 1

    records.write_text(records.read_text() + "{}\n")
    with pytest.raises(ValueError, match="records digest"):
        collection_closeout.load_finalization(tmp_path)


def test_v1_finalization_is_not_a_resume_adapter(tmp_path):
    _write(tmp_path / "collection_finalization.json", {
        "schema": "egoconseq.collection-finalization.v1",
        "status": "sealed",
    })
    with pytest.raises(ValueError, match="schema"):
        collection_closeout.load_finalization(tmp_path)
