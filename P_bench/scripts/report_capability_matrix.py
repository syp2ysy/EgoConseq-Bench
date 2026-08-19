#!/usr/bin/env python3
"""Report static authority-surface capabilities without starting a simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import capability_matrix, scene_pool  # noqa: E402


def _parse_scene_identity(value: str) -> tuple[str, str]:
    dataset, separator, scene_id = str(value).partition(":")
    if not separator or not dataset or not scene_id:
        raise argparse.ArgumentTypeError(
            "scene identity must be DATASET:SCENE_ID")
    return dataset, scene_id


def _require_pair(parser: argparse.ArgumentParser, first, second,
                  *, label: str) -> bool:
    if (first is None) != (second is None):
        parser.error(f"{label} discovery requires both paired arguments")
    return first is not None


def _discover_scene_identities(args, parser: argparse.ArgumentParser
                               ) -> list[tuple[str, str]]:
    specs = []
    if _require_pair(
            parser, args.r2r_train_episodes, args.mp3d_root, label="R2R"):
        specs.extend(scene_pool.discover_r2r_train_scenes(
            args.r2r_train_episodes, args.mp3d_root))
    if _require_pair(parser, args.b1k_root, args.b1k_manifest, label="B1K"):
        specs.extend(scene_pool.discover_b1k_train_scenes(
            args.b1k_root, args.b1k_manifest))
    if _require_pair(parser, args.gs_root, args.gs_manifest, label="GS"):
        specs.extend(scene_pool.discover_gs_train_scenes(
            args.gs_root, args.gs_manifest))
    return capability_matrix.scene_identities_from_specs(specs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", action="append", type=_parse_scene_identity, default=[],
        help="explicit DATASET:SCENE_ID capability row (repeatable)")
    parser.add_argument("--r2r-train-episodes", type=Path)
    parser.add_argument("--mp3d-root", type=Path)
    parser.add_argument("--b1k-root", type=Path)
    parser.add_argument("--b1k-manifest", type=Path)
    parser.add_argument("--gs-root", type=Path)
    parser.add_argument("--gs-manifest", type=Path)
    parser.add_argument(
        "--output", type=Path,
        help="optional JSON output path; stdout is used when omitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        scene_identities = [*args.scene, *_discover_scene_identities(args, parser)]
        report = capability_matrix.build_capability_report(
            scene_identities=scene_identities)
    except (OSError, ValueError, scene_pool.SceneCatalogError) as error:
        parser.error(str(error))
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
