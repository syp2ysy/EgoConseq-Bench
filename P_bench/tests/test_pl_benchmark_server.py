"""Browser transport must remain independent of the benchmark's saved inputs."""

import gzip
import io

from PIL import Image


def test_compressed_html_preserves_content_and_refreshes_after_rebuild(tmp_path):
    from scripts.serve_benchmark import encoded_asset

    path = tmp_path / "index.html"
    path.write_bytes(b"<html>B3</html>" * 100)
    compressed = encoded_asset(path, (path.stat().st_mtime_ns, path.stat().st_size), "gzip")
    assert gzip.decompress(compressed) == path.read_bytes()
    assert len(compressed) < 100
    path.write_bytes(b"<html>Updated B3</html>")
    assert gzip.decompress(encoded_asset(path, (path.stat().st_mtime_ns, path.stat().st_size), "gzip")) == b"<html>Updated B3</html>"


def test_thumbnail_preserves_source_image_and_aspect_ratio(tmp_path):
    from scripts.serve_benchmark import encoded_asset

    path = tmp_path / "original.png"
    Image.new("RGB", (1280, 960), "red").save(path)
    original = path.read_bytes()
    thumbnail = encoded_asset(path, (path.stat().st_mtime_ns, path.stat().st_size), "thumbnail")
    with Image.open(io.BytesIO(thumbnail)) as image:
        assert image.size == (480, 360)
        assert image.format == "JPEG"
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
