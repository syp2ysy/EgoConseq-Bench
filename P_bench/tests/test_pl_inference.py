"""Inference transport and resume integrity, without loading model weights."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_inference_common", ROOT / "data/benchmark/inference/common.py")
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def test_api_runs_concurrently_preserving_question_order_and_raw_answer(monkeypatch):
    barrier = threading.Barrier(2)
    model = "/cache/snapshots/revision1"

    def open_request(request, timeout):
        if request.full_url.endswith("/models"):
            data = {"data": [{"id": model, "root": model}]}
        else:
            body = json.loads(request.data)
            barrier.wait(timeout=3)  # A serial implementation cannot pass.
            data = {"choices": [{"message": {"content": body["messages"][0]["content"],
                                               "reasoning_content": "kept separately"},
                                  "finish_reason": "length"}]}
        return io.BytesIO(json.dumps(data).encode())

    monkeypatch.setattr(common, "urlopen", open_request)
    monkeypatch.setattr(common, "api_messages", lambda item, root: item["messages"])
    args = SimpleNamespace(model=model, base_url="http://localhost/v1", max_new_tokens=4096,
                           chat_template_kwargs={}, api_workers=2)
    predict, metadata = common.api_predictor(args)
    batch = [("seen", Path("."), {"messages": [{"role": "user", "content": text}]})
             for text in ("first long answer\n1.25 m", "second answer")]
    result = predict(batch)
    assert [r["answer"] for r in result] == [r[2]["messages"][0]["content"] for r in batch]
    assert all(r["finish_reason"] == "length" and r["reasoning"] == "kept separately" for r in result)
    assert metadata["model_revision"] == "revision1"


def test_api_rejects_wrong_served_model(monkeypatch):
    monkeypatch.setattr(common, "urlopen", lambda *a, **kw: io.BytesIO(b'{"data":[{"id":"wrong"}]}'))
    with pytest.raises(ValueError, match="does not serve"):
        common.api_predictor(SimpleNamespace(model="requested", base_url="http://localhost/v1"))


def test_saved_inputs_exclude_gt_and_preserve_c1_image_order(tmp_path):
    item = {"images": [f"{i}.png" for i in range(5)], "messages": [
        {"role": "system", "content": "saved system"},
        {"role": "user", "content": "initial <image> A <image> B <image> C <image> D <image>"},
        {"role": "assistant", "content": "C"}]}
    messages = common.build_messages(item, tmp_path)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert [b["image"] for b in messages[1]["content"] if b["type"] == "image"] == [
        str(tmp_path / f"{i}.png") for i in range(5)]


def setup_run(tmp_path, monkeypatch, factory):
    item = {"id": "one", "task_id": "A1", "dataset": "gs", "images": [], "messages": []}
    rows = [("seen", tmp_path, item), ("unseen", tmp_path, item)]
    monkeypatch.setattr(common, "load_questions", lambda *a: (rows, {"seen": "sha", "unseen": "sha"}))
    output = tmp_path / "response.json"
    monkeypatch.setattr(sys, "argv", [str(Path(common.__file__)), "--output", str(output), "--batch-size", "2"])
    return output, lambda: common.run(factory, model="model", backend="transformers")


@pytest.mark.parametrize("corruption", ["duplicate", "foreign", "wrong_task"])
def test_resume_rejects_corrupted_prediction_identity(tmp_path, monkeypatch, corruption):
    output, run = setup_run(tmp_path, monkeypatch, lambda args: (
        lambda batch: [{"answer": "Collision", "finish_reason": "stop"} for _ in batch], {}))
    run()
    saved = json.loads(output.read_text())
    if corruption == "duplicate":
        saved["predictions"].append(saved["predictions"][0])
    elif corruption == "foreign":
        saved["predictions"][0]["id"] = "foreign"
    else:
        saved["predictions"][0]["task_id"] = "A3"
    output.write_text(json.dumps(saved))
    with pytest.raises(SystemExit) as error:
        run()
    assert error.value.code == 2


def test_short_batch_cannot_silently_drop_questions(tmp_path, monkeypatch):
    output, run = setup_run(tmp_path, monkeypatch, lambda args: (
        lambda batch: [{"answer": "Collision", "finish_reason": "stop"}], {}))
    with pytest.raises(ValueError, match="different number"):
        run()
    assert not output.exists()


def test_resume_rejects_changed_model_revision(tmp_path, monkeypatch):
    revision = ["old"]
    output, run = setup_run(tmp_path, monkeypatch, lambda args: (
        lambda batch: [{"answer": "Collision", "finish_reason": "stop"} for _ in batch],
        {"model_revision": revision[0]}))
    run()
    saved = json.loads(output.read_text())
    saved["predictions"].pop()
    output.write_text(json.dumps(saved))
    revision[0] = "new"
    with pytest.raises(SystemExit) as error:
        run()
    assert error.value.code == 2
