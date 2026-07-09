"""Scene discovery logic (pure; tmp fake dirs, no Habitat)."""

from pipeline.scene_pool import discover_semantic_scenes


def test_discovers_only_scenes_with_semantic_glb(tmp_path):
    a = tmp_path / "00800-AAA"; a.mkdir()
    (a / "AAA.basis.glb").write_text("")
    (a / "AAA.semantic.glb").write_text("")   # has semantic -> included

    b = tmp_path / "00801-BBB"; b.mkdir()
    (b / "BBB.basis.glb").write_text("")       # no semantic -> excluded

    found = discover_semantic_scenes(str(tmp_path))
    assert len(found) == 1
    assert found[0].endswith("00800-AAA/AAA.basis.glb")


def test_empty_dir(tmp_path):
    assert discover_semantic_scenes(str(tmp_path)) == []
