import hashlib
import json

from post_QA import templates


def test_freeze_binds_both_splits_and_refreshes_report_without_changing_qa(tmp_path):
    from post_QA.seen_build.freeze import freeze

    originals = {}
    for split in ("seen", "unseen"):
        root = tmp_path / "benchmark" / split
        metadata = tmp_path / "metadata" / split
        (root / "images").mkdir(parents=True)
        metadata.mkdir(parents=True)
        (root / "images/input.png").write_bytes(b"saved-image")
        question = templates.QUESTION_TEMPLATES["B3"][0]["question"].format(
            checkpoint_action=3, checkpoint_fraction=50, target="the marked point")
        qa = [{"id": split, "task_id": "B3", "dataset": "r2r",
               "images": ["images/input.png"], "messages": [
                   {"role": "system", "content": templates.SYSTEM_PROMPT},
                   {"role": "user", "content": "Question: " + question + "\n\nAnswer in meters."},
                   {"role": "assistant", "content": "2.00 m"}]}]
        (root / "QA.json").write_text(json.dumps(qa))
        (metadata / "record_index.json").write_text(json.dumps({
            "items": [{"record_uid": split, "dataset": "r2r", "item_id": split}]}))
        (metadata / "report.json").write_text(json.dumps({"prompts": {"templates": {"A1_01": 1}}}))
        originals[split] = (root / "QA.json").read_bytes()

    result = freeze(tmp_path)
    assert result["total"] == 2
    assert {r["record_uid"] for r in result["items"]} == {"seen", "unseen"}
    for split, original in originals.items():
        root = tmp_path / "benchmark" / split
        assert (root / "QA.json").read_bytes() == original
        info = result["splits"][split]
        assert info["qa_sha256"] == hashlib.sha256(original).hexdigest()
        report = json.loads((tmp_path / "metadata" / split / "report.json").read_text())
        assert report["prompts"]["templates"] == {"B3_01": 1}
    assert json.loads((tmp_path / "metadata/frozen.json").read_text()) == result
