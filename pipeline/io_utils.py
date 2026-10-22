"""Atomic JSON/JSONL I/O shared by the offline benchmark scripts.

These primitives default to a crash-safe *rename* (unique temp file +
``os.replace``) without ``fsync`` durability; pass ``durable=True`` where the
file must survive power loss (the collection funnel and ``run_meta.json`` —
their counters must never lag the fsync-backed ``records.jsonl`` written by
the collection hot path's own ``scripts/collect._atomic_write``). Everything
else that writes a plan, manifest, or report used to carry a near-identical
private copy of this logic; consolidating them here removes that duplication.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable


def sha256_file(path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path) -> None:
    """Durably commit directory-entry changes below *path*."""
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_binary(
        path, writer: Callable[[BinaryIO], object], *, mode: int,
        durable: bool) -> None:
    """Run *writer* against a private file and atomically publish the result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "w+b") as stream:
            writer(stream)
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        if durable:
            fsync_directory(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_binary(
        path, writer: Callable[[BinaryIO], object], *, mode: int = 0o644,
        durable: bool = False) -> None:
    """Atomically publish binary output produced by ``writer(stream)``.

    ``durable=True`` fsyncs both the completed file and its parent directory.
    """
    _atomic_write_binary(path, writer, mode=mode, durable=durable)


def atomic_write_text(
        path, text: str, *, mode: int = 0o644,
        durable: bool = False) -> None:
    """Atomically replace *path* with *text* (UTF-8) at permission *mode*.
    ``tempfile.mkstemp`` creates a private ``0600`` file whose permissions
    ``os.replace`` would carry over to the destination, so the mode is reset
    explicitly before the rename -- otherwise a public manifest would land
    unreadable to downstream readers.  A concurrent writer of the same *path*
    gets its own uniquely named temp file, and any failure removes it.
    ``durable=True`` additionally fsyncs the temp file before the rename and
    the parent directory after it, so the replacement survives power loss.
    """
    encoded = str(text).encode("utf-8")
    _atomic_write_binary(
        path, lambda stream: stream.write(encoded),
        mode=mode, durable=durable)


def atomic_write_json(
        path, value, *, allow_nan: bool = True, sort_keys: bool = True,
        indent: int = 2, mode: int = 0o644, durable: bool = False) -> None:
    """Atomically write *value* as pretty JSON with a trailing newline."""
    text = json.dumps(
        value, indent=indent, sort_keys=sort_keys, allow_nan=allow_nan) + "\n"
    atomic_write_text(path, text, mode=mode, durable=durable)


