"""Scene discovery (pure; no Habitat)."""

from __future__ import annotations

import glob
import os
from typing import List

from pipeline import config


def discover_semantic_scenes(val_dir: str = config.HM3D_VAL_DIR) -> List[str]:
    """Return basis.glb paths for scenes that have a semantic.glb sibling."""
    out = []
    for glb in sorted(glob.glob(os.path.join(val_dir, "*", "*.basis.glb"))):
        if os.path.exists(glb.replace(".basis.glb", ".semantic.glb")):
            out.append(glb)
    return out
