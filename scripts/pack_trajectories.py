"""Pack complete RGB trajectories into indexed, uncompressed ZIP archives."""

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

ANNOTATIONS = {"R2R": "annotations_v1-3.json", "RxR": "annotations.json"}


def _verify_archive(destination, expected):
    with ZipFile(destination) as archive:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != set(expected):
            raise ValueError(f"Frame list mismatch: {destination}")
        for name, digest in expected.items():
            if archive.getinfo(name).compress_type != ZIP_STORED:
                raise ValueError(f"Expected ZIP_STORED: {destination}: {name}")
            if hashlib.sha256(archive.read(name)).digest() != digest:
                raise ValueError(f"Frame bytes mismatch: {destination}: {name}")


def pack_trajectory(rgb_dir, destination):
    """Verify every frame; publish atomically, or validate an existing archive."""
    rgb_dir, destination = Path(rgb_dir), Path(destination)
    paths = sorted(rgb_dir.iterdir())
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(f"Expected a nonempty RGB directory containing only files: {rgb_dir}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reused = destination.exists()
    expected = {}
    source_bytes = 0
    temporary = None
    try:
        if not reused:
            fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
            os.close(fd)
        archive = ZipFile(temporary, "w", compression=ZIP_STORED) if temporary else None
        try:
            for path in paths:
                data = path.read_bytes()
                name = f"rgb/{path.name}"
                expected[name] = hashlib.sha256(data).digest()
                source_bytes += len(data)
                if archive is not None:
                    archive.writestr(name, data)
        finally:
            if archive is not None:
                archive.close()
        _verify_archive(temporary or destination, expected)
        if temporary:
            os.replace(temporary, destination)
            temporary = None
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
    return {"frames": len(paths), "source_bytes": source_bytes,
            "archive_bytes": destination.stat().st_size, "reused": reused}


def _atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=list(ANNOTATIONS), default=list(ANNOTATIONS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.source_root.resolve() == args.output_root.resolve():
        parser.error("--output-root must differ from --source-root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {"source_root": str(args.source_root.resolve()), "output_root": str(args.output_root.resolve()),
              "storage": "ZIP_STORED", "verification": "SHA-256 of every original and archived frame", "datasets": {}}
    for dataset in args.datasets:
        source = args.source_root / dataset
        output = args.output_root / dataset
        annotation_bytes = (source / ANNOTATIONS[dataset]).read_bytes()
        episodes = json.loads(annotation_bytes)
        videos = set()
        for episode in episodes:
            video = episode["video"]
            if video.startswith("images/"):
                video = video[len("images/") :]
            if not video or Path(video).is_absolute() or ".." in Path(video).parts:
                raise ValueError(f"Invalid trajectory path: {video}")
            videos.add(video)
        jobs = [(source / "images" / video / "rgb", output / "images" / (video + ".zip"))
                for video in sorted(videos)]
        output.mkdir(parents=True, exist_ok=True)
        annotation_target = output / ANNOTATIONS[dataset]
        if annotation_target.exists() and annotation_target.read_bytes() != annotation_bytes:
            raise ValueError(f"Annotation mismatch: {annotation_target}")
        totals = {"trajectories": len(jobs), "frames": 0, "source_bytes": 0, "archive_bytes": 0, "reused": 0,
                  "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest()}
        print(f"{dataset}: packing and verifying {len(jobs)} trajectories", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(pack_trajectory, rgb, destination) for rgb, destination in jobs]
            for count, future in enumerate(futures, 1):
                result = future.result()
                for key in ("frames", "source_bytes", "archive_bytes", "reused"):
                    totals[key] += result[key]
                if count % 250 == 0 or count == len(jobs):
                    print(f"{dataset}: {count}/{len(jobs)}, {totals['frames']:,} frames verified, "
                          f"elapsed {time.monotonic() - start:.1f}s", flush=True)
        _atomic_write(annotation_target, annotation_bytes)
        report["datasets"][dataset] = totals
        report["elapsed_seconds"] = time.monotonic() - start
        _atomic_write(args.output_root / "packing_report.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
