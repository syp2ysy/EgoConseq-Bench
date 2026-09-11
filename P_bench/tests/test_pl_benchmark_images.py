import numpy as np
from PIL import Image
from post_QA.seen_build.images import quality


def test_image_quality_rejects_black_holes_and_blank_views():
    pixels = np.random.default_rng(3).integers(30, 240, (480, 640, 3), dtype=np.uint8)
    assert quality(Image.fromarray(pixels))['accepted']
    pixels[:, :160] = 0
    assert not quality(Image.fromarray(pixels))['accepted']
    assert not quality(Image.new('RGB', (640, 480), (120, 120, 120)))['accepted']


def test_dark_fraction_counts_near_black_not_colored_or_normal_shadows():
    from post_QA.seen_build.images import dark_fraction

    rgb = np.full((10, 10, 3), 8, dtype=np.uint8)
    rgb[0] = (180, 0, 0)
    assert dark_fraction(Image.fromarray(rgb)) == .9
    rgb[:5] = 100
    assert dark_fraction(Image.fromarray(rgb)) == .5
