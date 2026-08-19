"""Collect structured consequence records."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import collection_funnel
from pipeline.collection_cli import build_parser
from pipeline.collection_runtime import run_collection
from pipeline.collection_support import preserve_b1k_cli_status


def main(*, hard_exit=None) -> int:
    parser = build_parser()
    args = parser.parse_args()
    funnel_path = Path(args.out) / "collection_funnel.json"
    try:
        return preserve_b1k_cli_status(
            args, run_collection(args, parser))
    except KeyboardInterrupt:
        collection_funnel.CollectionFunnel.mark_interrupted(
            funnel_path, reason="keyboard_interrupt")
        print(
            "collection interrupted; resumable state was preserved",
            file=sys.stderr, flush=True,
        )
        preserve_b1k_cli_status(args, 130, hard_exit=hard_exit)
        return 130
    except Exception as error:
        collection_funnel.CollectionFunnel.mark_failed(
            funnel_path, reason=type(error).__name__)
        if getattr(args, "backend", "r2r") == "b1k":
            traceback.print_exc()
            preserve_b1k_cli_status(args, 1, hard_exit=hard_exit)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
