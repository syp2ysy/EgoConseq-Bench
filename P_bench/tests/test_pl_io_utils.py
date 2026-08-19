"""Shared non-durable atomic JSON/JSONL I/O primitives."""

import json
import os
import threading

import pytest

from pipeline import io_utils
from pipeline.io_utils import (
    atomic_write_json,
    atomic_write_text,
    linear_quantile,
    read_jsonl,
    sha256_file,
)


def _leftovers(directory, keep):
    return [entry.name for entry in directory.iterdir() if entry != keep]


def test_atomic_write_text_lands_world_readable_by_default(tmp_path):
    # mkstemp yields a 0600 file; os.replace would carry that mode onto a
    # public manifest unless it is reset, so the default must land 0644.
    path = tmp_path / "manifest.json"
    atomic_write_text(path, "payload\n")
    assert path.read_text() == "payload\n"
    assert (path.stat().st_mode & 0o777) == 0o644
    assert _leftovers(tmp_path, path) == []


def test_atomic_write_text_honours_custom_mode(tmp_path):
    path = tmp_path / "private.json"
    atomic_write_text(path, "secret\n", mode=0o600)
    assert (path.stat().st_mode & 0o777) == 0o600


def test_atomic_write_json_defaults_are_pretty_sorted_with_newline(tmp_path):
    path = tmp_path / "out.json"
    atomic_write_json(path, {"b": 1, "a": 2})
    text = path.read_text()
    assert text.endswith("\n")
    assert list(json.loads(text)) == ["a", "b"]  # sort_keys=True
    assert (path.stat().st_mode & 0o777) == 0o644


def test_atomic_write_json_rejects_nan_when_disallowed(tmp_path):
    path = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        atomic_write_json(path, {"x": float("nan")}, allow_nan=False)
    assert not path.exists()
    assert _leftovers(tmp_path, path) == []


def test_atomic_write_text_cleans_tmp_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "out.json"

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(path, "payload\n")
    assert not path.exists()
    assert _leftovers(tmp_path, path) == []


def test_atomic_write_binary_durable_fsyncs_file_and_parent(
        tmp_path, monkeypatch):
    path = tmp_path / "asset.bin"
    fsync_calls = []
    monkeypatch.setattr(
        os, "fsync", lambda descriptor: fsync_calls.append(descriptor))

    io_utils.atomic_write_binary(
        path, lambda stream: stream.write(b"payload"), durable=True)

    assert path.read_bytes() == b"payload"
    assert len(fsync_calls) == 2
    assert _leftovers(tmp_path, path) == []


def test_atomic_write_binary_cleans_tmp_when_callback_fails(tmp_path):
    path = tmp_path / "asset.bin"

    def fail_after_writing(stream):
        stream.write(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        io_utils.atomic_write_binary(path, fail_after_writing, durable=True)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_concurrent_writes_are_atomic_and_leave_no_tmp(tmp_path):
    path = tmp_path / "shared.json"
    contents = [f"writer-{index}\n" for index in range(8)]
    barrier = threading.Barrier(len(contents))

    def write(text):
        barrier.wait()
        atomic_write_text(path, text)

    threads = [threading.Thread(target=write, args=(t,)) for t in contents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Exactly one writer's full content survives; the rename never tears a file
    # nor leaves a colliding temp behind.
    assert path.read_text() in contents
    assert _leftovers(tmp_path, path) == []


def test_read_jsonl_missing_ok_returns_empty(tmp_path):
    assert read_jsonl(tmp_path / "absent.jsonl", missing_ok=True) == []


def test_read_jsonl_missing_raises_by_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "absent.jsonl")


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n  \n{"b": 2}\n')
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_require_dict_reports_path_and_lineno(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n[1, 2, 3]\n')
    with pytest.raises(ValueError) as excinfo:
        read_jsonl(path, require_dict=True)
    assert f"{path}:2" in str(excinfo.value)


def test_read_jsonl_allows_non_dict_lines_without_require_dict(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('[1, 2]\n"scalar"\n')
    assert read_jsonl(path) == [[1, 2], "scalar"]


def test_sha256_file_and_linear_quantile_are_shared_deterministic_helpers(
        tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad")
    assert linear_quantile([1.0, 2.0, 4.0], 0.25) == 1.5
