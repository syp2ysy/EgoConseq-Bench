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

import errno
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable


def git_revision_state(
        root=None, *, missing_ok: bool = False) -> tuple[str | None, bool | None]:
    """Return the repository commit and dirty state for *root*.
    ``missing_ok`` is reserved for provenance attached to exploratory outputs;
    formal collection callers keep the default and fail when Git is
    unavailable.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        if missing_ok:
            return None, None
        raise
    return commit, bool(status.strip())


def sha256_file(path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy_file(source, destination) -> bool:
    """Stage *source* at *destination*, preferring a zero-copy hard link.

    Returns ``True`` for a hard link and ``False`` for the cross-filesystem or
    unsupported-link copy fallback.  Both paths preserve the exact bytes; the
    caller remains responsible for any content-digest check it requires.
    """
    source = Path(source)
    destination = Path(destination)
    try:
        os.link(source, destination)
        return True
    except OSError as error:
        if error.errno not in {
                errno.EXDEV, errno.EPERM, errno.EACCES,
                getattr(errno, "ENOTSUP", errno.EPERM)}:
            raise
    shutil.copyfile(source, destination)
    return False


def linear_quantile(values, fraction: float) -> float:
    """Linearly interpolated quantile over finite numeric values."""
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("quantile fraction must lie in [0, 1]")
    ordered = sorted(
        float(value) for value in values
        if value is not None and math.isfinite(float(value)))
    if not ordered:
        raise ValueError("quantile requires at least one finite value")
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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


def read_jsonl(
        path, *, missing_ok: bool = False, require_dict: bool = False) -> list:
    """Read a JSONL file into a list, skipping blank lines.
    ``missing_ok`` returns ``[]`` for an absent file instead of raising;
    ``require_dict`` rejects any non-object line with a ``path:lineno`` error.
    """
    path = Path(path)
    if missing_ok and not path.is_file():
        return []
    values = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if require_dict and not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object")
            values.append(value)
    return values
