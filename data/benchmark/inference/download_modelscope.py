#!/usr/bin/env python3
"""Download a pinned public ModelScope checkpoint with verified, resumable files."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(base_url, revision, destination, item):
    relative = PurePosixPath(item["Path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe repository path: {relative}")
    path = destination / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size, expected_hash = item["Size"], item["Sha256"]
    if path.is_file() and path.stat().st_size == expected_size and sha256(path) == expected_hash:
        print(f"CACHED {relative}", flush=True)
        return
    partial = path.with_name(path.name + ".part")
    url = base_url + "/repo?" + urlencode({"Revision": revision, "FilePath": str(relative)})
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                partial.unlink()
                offset = 0
            if offset < expected_size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with urlopen(Request(url, headers=headers), timeout=120) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if offset and not content_range.startswith(f"bytes {offset}-"):
                        if content_range or response.status != 200:
                            raise ValueError(f"Unexpected range response: {content_range}")
                        offset = 0  # A server without Range support sent the full file.
                    with partial.open("ab" if offset else "wb") as stream:
                        for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                            stream.write(chunk)
            if partial.stat().st_size != expected_size:
                raise ValueError(f"Incomplete {relative}: {partial.stat().st_size}/{expected_size}")
            if sha256(partial) != expected_hash:
                partial.unlink()
                raise ValueError(f"SHA256 mismatch: {relative}")
            partial.replace(path)
            print(f"VERIFIED {relative}: {expected_size} bytes, SHA256 {expected_hash}", flush=True)
            return
        except Exception as error:
            if attempt == 5:
                raise
            print(f"RETRY {relative} ({attempt}/5): {error}", flush=True)
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--expected-weight-sha256", action="append", default=[],
                        metavar="FILENAME=SHA256")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    base_url = "https://modelscope.cn/api/v1/models/" + args.repo

    def update(status, **fields):
        state = json.loads(args.state.read_text()) if args.state.exists() else {}
        record = state.setdefault(args.alias, {})
        record.pop("error", None)
        record.update(status=status, repo=args.repo, path=str(args.destination),
                      revision=args.revision, source="modelscope",
                      source_url="https://modelscope.cn/models/" + args.repo,
                      updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        args.state.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.state)

    update("downloading")
    try:
        url = base_url + "/repo/files?" + urlencode({"Revision": args.revision, "Recursive": "true"})
        with urlopen(url, timeout=30) as response:
            listing = json.load(response)
        if not listing.get("Success"):
            raise ValueError(f"ModelScope listing failed: {listing.get('Message')}")
        files = [item for item in listing["Data"]["Files"] if item["Type"] == "blob"]
        indexed = {item["Path"]: item for item in files}
        weights = [item for item in files if item["Path"].endswith(".safetensors")]
        if "config.json" not in indexed or not weights:
            raise ValueError("Checkpoint must contain config.json and safetensors weights")
        for expected in args.expected_weight_sha256:
            filename, digest = expected.split("=", 1)
            if indexed.get(filename, {}).get("Sha256") != digest:
                raise ValueError(f"Pinned weight hash does not match repository: {filename}")
        args.destination.mkdir(parents=True, exist_ok=True)
        files.sort(key=lambda item: item["Size"])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed(pool.submit(
                    download_file, base_url, args.revision, args.destination, item) for item in files):
                future.result()
        if not (args.destination / "config.json").is_file():
            raise ValueError("Downloaded config.json is missing")
        total = sum(item["Size"] for item in weights)
        update("complete", weight_files=len(weights), weight_bytes=total,
               weight_sha256={item["Path"]: item["Sha256"] for item in weights})
        print(f"READY {args.alias}: {len(weights)} verified weights, {total / 2**30:.2f} GiB", flush=True)
    except Exception as error:
        update("blocked", error=str(error))
        raise


if __name__ == "__main__":
    main()
