"""Fresh clones derive runtime paths from documented environment variables."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _read_config(environment):
    output = subprocess.check_output([
        sys.executable, "-c",
        "import json; from pipeline import config; "
        "print(json.dumps({"
        "'r2r': config.R2R_TRAIN_EPISODES, "
        "'mp3d': config.MP3D_ROOT, "
        "'gs': config.GS_ROOT, "
        "'gs_manifest': config.GS_TRAIN_MANIFEST}))",
    ], cwd=ROOT, env=environment, text=True)
    return json.loads(output)


def test_shared_data_root_supplies_portable_defaults(tmp_path):
    environment = dict(os.environ)
    for name in (
            "EGOCONSEQ_R2R_TRAIN_EPISODES", "EGOCONSEQ_MP3D_ROOT",
            "EGOCONSEQ_GS_ROOT", "EGOCONSEQ_GS_TRAIN_MANIFEST"):
        environment.pop(name, None)
    data_root = tmp_path / "sources"
    environment["EGOCONSEQ_DATA_ROOT"] = str(data_root)

    values = _read_config(environment)

    assert values == {
        "r2r": str(data_root / "r2r_vlnce_v1-3/train/train.json.gz"),
        "mp3d": str(data_root / "scene_datasets/mp3d"),
        "gs": str(data_root / "gs"),
        "gs_manifest": str(data_root / "gs/splits/train.json"),
    }


def test_specific_source_paths_override_shared_data_root(tmp_path):
    environment = dict(os.environ)
    environment["EGOCONSEQ_DATA_ROOT"] = str(tmp_path / "shared")
    overrides = {
        "EGOCONSEQ_R2R_TRAIN_EPISODES": str(tmp_path / "r2r.json.gz"),
        "EGOCONSEQ_MP3D_ROOT": str(tmp_path / "mp3d"),
        "EGOCONSEQ_GS_ROOT": str(tmp_path / "gs"),
        "EGOCONSEQ_GS_TRAIN_MANIFEST": str(tmp_path / "gs.json"),
    }
    environment.update(overrides)

    values = _read_config(environment)

    assert values == {
        "r2r": overrides["EGOCONSEQ_R2R_TRAIN_EPISODES"],
        "mp3d": overrides["EGOCONSEQ_MP3D_ROOT"],
        "gs": overrides["EGOCONSEQ_GS_ROOT"],
        "gs_manifest": overrides["EGOCONSEQ_GS_TRAIN_MANIFEST"],
    }


def test_active_python_sources_do_not_embed_operator_home_paths():
    offenders = []
    for directory in (ROOT / "pipeline", ROOT / "scripts"):
        for path in directory.rglob("*.py"):
            if "/home/zhangshan" in path.read_text():
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
