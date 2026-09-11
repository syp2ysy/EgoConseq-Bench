#!/usr/bin/env python3
"""Serve the saved benchmark with compressed HTML and in-memory list thumbnails."""

import argparse
from functools import lru_cache, partial
import gzip
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image


@lru_cache(maxsize=128)
def encoded_asset(path, file_version, variant):
    """Cache by modification time and size, including coarse-timestamp filesystems."""
    if variant == "gzip":
        return gzip.compress(path.read_bytes(), compresslevel=4, mtime=0)
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((480, 360), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85)
        return output.getvalue()


class BenchmarkHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            path = path / "index.html"
        query = parse_qs(urlsplit(self.path).query)
        variant = None
        if path.suffix == ".html" and "gzip" in self.headers.get("Accept-Encoding", ""):
            variant = "gzip"
        elif path.suffix == ".png" and query.get("thumbnail") == ["1"]:
            variant = "thumbnail"
        if variant is None or not path.is_file():
            return super().send_head()
        stat = path.stat()
        data = encoded_asset(path, (stat.st_mtime_ns, stat.st_size), variant)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8" if variant == "gzip" else "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        if variant == "gzip":
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Cache-Control", "no-cache")
        else:
            self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        return io.BytesIO(data)

    def log_request(self, code="-", size="-"):
        if code not in {200, 304}:
            super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/benchmark")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    handler = partial(BenchmarkHandler, directory=str(args.directory.resolve()))
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(f"Benchmark: http://localhost:{args.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
