"""Expand B1K training records in place, with scene-owned GPU workers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.collection_expansion import main


if __name__ == "__main__":
    raise SystemExit(main())
