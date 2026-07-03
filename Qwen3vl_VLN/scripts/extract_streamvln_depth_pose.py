#!/usr/bin/env python3
"""Generic StreamVLN-style depth/pose extraction entrypoint.

This is a thin alias around ``extract_r2r_depth_pose.py``. It replays
StreamVLN-style annotations through Habitat and writes depth/pose files next to
the existing RGB trajectory folders. Use it for R2R directly, and for RxR only
when the supplied Habitat ``--data_path`` matches the StreamVLN RxR annotation
ids and RGB trajectories.
"""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_r2r_depth_pose import extract_depth_pose, parse_args  # noqa: E402


if __name__ == "__main__":
    extract_depth_pose(parse_args())
