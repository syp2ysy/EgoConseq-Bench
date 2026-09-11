#!/usr/bin/env python3
"""Refine ABC1 records, compile a visual-diverse benchmark, or export SFT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from post_QA.seen_build import collect, inventory  # noqa: E402


DEFAULT_RECORDS_ROOT = ROOT / "data/metadata/train/seen_updates/current"
DEFAULT_WORK_DIR = ROOT / "data/metadata/train/seen_updates/work/actions"
DEFAULT_SURFACE_OUTPUT = ROOT / "data" / "metadata" / "train" / "seen_updates"
DEFAULT_SURFACE_ROOT = DEFAULT_SURFACE_OUTPUT / "current"
DEFAULT_SURFACE_WORK = DEFAULT_SURFACE_OUTPUT / "work" / "surfaces"
WORKERS = ("b1k-0", "b1k-1", "gs", "r2r")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    a3 = commands.add_parser("repair-a3", help="verify B1K contact categories without changing other record fields")
    a3.add_argument("--seen-records", type=Path, default=DEFAULT_RECORDS_ROOT)
    a3.add_argument("--unseen-records", type=Path, default=ROOT / "data/metadata/test/unseen_updates/current")
    a3.add_argument("--benchmark-root", type=Path, default=ROOT / "data/benchmark")
    a3.add_argument("--sft-root", type=Path, default=ROOT / "data/sft/seen_v2")
    a3.add_argument("--work", type=Path, default=ROOT / "data/metadata/a3_repair")
    a3.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    a3.add_argument("--b1k-python", type=Path,
                    default=Path(os.environ.get("EGOCONSEQ_B1K_PYTHON", sys.executable)))
    a3.add_argument("--b1k-data-root", type=Path,
                    default=Path(os.environ.get("B1K_DATA_ROOT", Path(os.environ.get(
                        "EGOCONSEQ_DATA_ROOT", ROOT / "data/sources")) / "behavior-1k-v3.9.1")))
    a3.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    a3.add_argument("--drop-unresolved", action="store_true", help="withdraw audited A3 cases and their QA without replaying scenes")

    parameters = commands.add_parser("repair-parameters", help="synchronize configured relative height in frozen QA/SFT without rerendering")
    parameters.add_argument("--seen-records", type=Path, default=DEFAULT_RECORDS_ROOT)
    parameters.add_argument("--unseen-records", type=Path, default=ROOT / "data/metadata/test/unseen_updates/current")
    parameters.add_argument("--benchmark-root", type=Path, default=ROOT / "data/benchmark")
    parameters.add_argument("--sft-root", type=Path, default=ROOT / "data/sft/seen_v2")

    prompts = commands.add_parser("refresh-prompts", help="update frozen Benchmark/SFT text and HTML, keeping images and GT")
    prompts.add_argument("--benchmark-root", type=Path, default=ROOT / "data/benchmark")
    prompts.add_argument("--sft-root", type=Path, default=ROOT / "data/sft/seen_v2")
    prompts.add_argument("--seed", type=int, default=20260908)
    prompts.add_argument("--keep-templates", action="store_true", help="refresh wording without reassigning question templates")

    clean = commands.add_parser("clean-sft", help="remove extremely dark SFT views and synchronize exported A3 names")
    clean.add_argument("--root", type=Path, default=ROOT / "data/sft/seen_v2")
    clean.add_argument("--benchmark-root", type=Path, default=ROOT / "data/benchmark")
    clean.add_argument("--workers", type=int, default=16)

    plan = commands.add_parser("plan", help="scan records into scene order")
    plan.add_argument(
        "--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    plan.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    plan.add_argument(
        "--supply-report", type=Path, default=None,
        help="target only shortfalls reported by a previous dry-run")
    plan.add_argument("--c1-attempts-per-dataset", type=int, default=1300)
    plan.add_argument("--a2-attempt-factor", type=int, default=2)

    run = commands.add_parser("collect", help="collect one scene-owned shard")
    run.add_argument("--plan", type=Path, default=DEFAULT_WORK_DIR / "plan.jsonl")
    run.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    run.add_argument("--worker", choices=WORKERS, default="gs")
    run.add_argument("--seed", type=int, default=20260904)
    run.add_argument("--max-programs", type=int, default=500)
    run.add_argument("--max-full-attempts", type=int, default=4)
    run.add_argument("--mp3d-root", type=Path, default=collect.DEFAULT_MP3D_ROOT)
    run.add_argument("--gs-root", type=Path, default=collect.DEFAULT_GS_ROOT)
    run.add_argument(
        "--gs-manifest", type=Path, default=collect.DEFAULT_GS_MANIFEST)
    run.add_argument(
        "--b1k-manifest", type=Path, default=collect.DEFAULT_B1K_MANIFEST)

    dry_run = commands.add_parser(
        "dry-run", help="report all record/candidate shortfalls without writing data")
    dry_run.add_argument(
        "--plan", type=Path, default=DEFAULT_WORK_DIR / "plan.jsonl")
    dry_run.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    dry_run.add_argument("--output", type=Path, default=None)
    dry_run.add_argument("--seed", type=int, default=20260904)

    materialize = commands.add_parser(
        "materialize-records",
        help="write compact records from a completed collection plan")
    materialize.add_argument(
        "--plan", type=Path, default=DEFAULT_WORK_DIR / "plan.jsonl")
    materialize.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    materialize.add_argument("--output", type=Path, required=True)

    update = commands.add_parser(
        "update-records", help="resume and repair records in the background; no QA build")
    update.add_argument("--plan", type=Path, default=DEFAULT_WORK_DIR / "plan.jsonl")
    update.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    update.add_argument("--output", type=Path, required=True)
    update.add_argument("--wait-pids", type=int, nargs="*", default=[])
    update.add_argument("--max-rounds", type=int, default=3)
    update.add_argument("--finish-current", action="store_true",
                        help="merge a stopped repair and reuse uncollected source records")
    update.add_argument("--python", type=Path, default=Path(sys.executable))
    update.add_argument("--b1k-python", type=Path, required=True)
    update.add_argument("--b1k-data-root", type=Path, required=True)
    update.add_argument("--benchmark-output", type=Path, default=None,
                        help="after refinement, compile QA and HTML at this explicit destination")
    update.add_argument("--device", default="cuda:0")

    refine = commands.add_parser(
        "refine-surfaces",
        help="refine surface targets for every current compact record")
    refine.add_argument("--root", type=Path, default=DEFAULT_SURFACE_ROOT)
    refine.add_argument("--output", type=Path, default=DEFAULT_SURFACE_OUTPUT)
    refine.add_argument("--work", type=Path, default=DEFAULT_SURFACE_WORK)
    refine.add_argument("--seed", type=int, default=20260906)
    refine.add_argument("--python", type=Path, default=Path(sys.executable))
    refine.add_argument("--b1k-python", type=Path, required=True)
    refine.add_argument("--b1k-data-root", type=Path, required=True)

    compile_command = commands.add_parser(
        "compile", help="compile a records catalog directly into ABC1 QA")
    compile_command.add_argument(
        "--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    compile_command.add_argument("--output", type=Path, required=True)
    compile_command.add_argument("--seed", type=int, default=20260904)
    compile_command.add_argument("--device", default="cuda:0")
    compile_command.add_argument("--similarity-threshold", type=float, default=0.90)

    browser = commands.add_parser(
        "browser", help="join published Seen and Unseen in one HTML; no QA rebuild")
    browser.add_argument("--root", type=Path, default=ROOT / "data/benchmark")
    browser.add_argument("--output", type=Path)

    freeze = commands.add_parser("freeze", help="seal Seen/Unseen QA and their SFT record exclusions")
    freeze.add_argument("--root", type=Path, default=ROOT / "data/benchmark")

    refresh = commands.add_parser(
        "refresh-presentation", help="refresh existing surface markers and prompts, keeping selection and GT")
    refresh.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    refresh.add_argument("--benchmark-root", type=Path, required=True)

    directions = commands.add_parser(
        "rebalance-directions", help="rebalance A4/B2 across 24 directions, keeping other QA")
    directions.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    directions.add_argument("--benchmark-root", type=Path, required=True)
    directions.add_argument("--device", default="cuda:0")

    sft_command = commands.add_parser(
        "compile-sft", help="cover remaining records with balanced, training-only SFT QA")
    sft_command.add_argument(
        "--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    sft_command.add_argument("--output", type=Path, required=True)
    sft_command.add_argument("--exclude-index", type=Path,
                             default=ROOT / "data/benchmark/metadata/frozen.json",
                             help="exclude every QA from the benchmark's records")
    sft_command.add_argument("--seed", type=int, default=20260906)
    sft_command.add_argument("--workers", type=int, default=8)

    sft_view = commands.add_parser("export-sft", help="render a parameter ablation from frozen SFT selection")
    sft_view.add_argument("--root", type=Path, required=True)
    sft_view.add_argument("--hide-params", nargs="*", choices=("height", "radius", "fov"), default=[])

    benchmark_views = commands.add_parser(
        "export-benchmark", help="derive four parameter ablations from saved Seen/Unseen QA")
    benchmark_views.add_argument("--root", type=Path, default=ROOT / "data/benchmark")

    for command in (plan, dry_run, update, compile_command):
        command.add_argument("--spec", type=Path, default=None,
                             help="benchmark quota specification (default: seen 5000)")

    return parser


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "export-benchmark":
        from post_QA.seen_build.prompts import export_benchmark_views

        _print(export_benchmark_views(args.root))
        return 0
    if args.command == "repair-a3":
        from post_QA.seen_build import a3_repair

        if args.job:
            a3_repair.worker(args.job)
        else:
            _print(a3_repair.run(args))
        return 0
    if args.command == "refresh-prompts":
        from post_QA.seen_build.prompts import refresh

        result = refresh(args.benchmark_root, args.sft_root, seed=args.seed,
                         keep_templates=args.keep_templates)
        _print({"version": result["version"],
                "benchmark": {key: value["qa_items"] for key, value in result["benchmark"].items()},
                "sft_qa": result["sft"]["qa_items"]})
        return 0
    if args.command == "clean-sft":
        from post_QA.seen_build.sft_cleanup import clean

        result = clean(args.root, args.benchmark_root, workers=args.workers)
        _print({key: value for key, value in result.items()
                if key not in {"dark_images", "removed_items"}})
        return 0
    if args.command == "repair-parameters":
        from post_QA.seen_build.parameters import repair

        result = repair(args.seen_records, args.unseen_records, args.benchmark_root,
                        args.sft_root)
        _print({"report": str(args.benchmark_root / "parameter_audit.json"),
                "benchmark": {key: value["qa_items"] for key, value in result["benchmark"].items()},
                "sft_qa": result["sft"]["qa_items"]})
        return 0
    if args.command == "freeze":
        from post_QA.seen_build.freeze import freeze

        result = freeze(args.root)
        _print({"path": str(args.root / "metadata/frozen.json"), "total": result["total"],
                "splits": result["splits"]})
        return 0
    if getattr(args, "spec", None) is not None:
        from post_QA.seen_build import spec
        spec.configure(args.spec)
    if args.command == "plan":
        path = args.work_dir / "plan.jsonl"
        repair_report = (None if args.supply_report is None else
                         json.loads(args.supply_report.read_text()))
        summary = inventory.write_plan(
            inventory.sources_from_manifest(args.records_root), path,
            c1_attempts_per_dataset=args.c1_attempts_per_dataset,
            a2_attempt_factor=args.a2_attempt_factor,
            split=json.loads((args.records_root / "manifest.json").read_text()).get("split", "train/seen"),
            repair_report=repair_report,
            progress=lambda dataset, count:
                print(f"[plan] {dataset}: {count} records", flush=True))
        _print({"plan": str(path.resolve()), **summary})
        return 0
    if args.command == "collect":
        output = args.work_dir / f"{args.worker}.jsonl"
        _print(collect.collect_records(
            args.plan, args.worker, output,
            seed=args.seed, max_programs=args.max_programs,
            max_full_attempts=args.max_full_attempts,
            mp3d_root=args.mp3d_root, gs_root=args.gs_root,
            gs_manifest=args.gs_manifest, b1k_manifest=args.b1k_manifest))
        return 0
    if args.command == "dry-run":
        from post_QA.seen_build import supply

        output = args.output or args.work_dir / "supply.json"
        _print(supply.dry_run(
            args.plan, args.work_dir, seed=args.seed, output_path=output,
            progress=lambda dataset, count:
                print(f"[dry-run] {dataset}: {count} records", flush=True)))
        return 0
    if args.command == "materialize-records":
        from post_QA.seen_build import release

        _print(release.materialize_records(
            args.plan, args.work_dir, args.output))
        return 0
    if args.command == "update-records":
        from post_QA.seen_build import records_update

        if args.finish_current:
            _print(records_update.finish_records(args.output))
            return 0
        _print(records_update.update_records(
            args.plan, args.work_dir, args.output,
            python=args.python, b1k_python=args.b1k_python,
            b1k_data_root=args.b1k_data_root, wait_pids=tuple(args.wait_pids),
            max_rounds=args.max_rounds, benchmark_output=args.benchmark_output,
            device=args.device))
        return 0
    if args.command == "refine-surfaces":
        from post_QA.seen_build import records_update

        _print(records_update.refine_surfaces(
            args.root, args.output, args.work,
            seed=args.seed, python=args.python,
            b1k_python=args.b1k_python,
            b1k_data_root=args.b1k_data_root))
        return 0
    if args.command == "compile":
        from post_QA.seen_build import release

        _print(release.compile_records(
            args.records_root, args.output, seed=args.seed,
            device=args.device, similarity_threshold=args.similarity_threshold))
        return 0
    if args.command == "browser":
        from visualization.benchmark_browser import build_combined_benchmark_browser

        paths = build_combined_benchmark_browser(args.root, args.output)
        _print({key: str(path) for key, path in paths.items()})
        return 0
    if args.command == "refresh-presentation":
        from post_QA.seen_build import release

        _print(release.refresh_presentation(args.records_root, args.benchmark_root))
        return 0
    if args.command == "rebalance-directions":
        from post_QA.seen_build import release

        _print(release.rebalance_directions(
            args.records_root, args.benchmark_root, device=args.device))
        return 0
    if args.command == "compile-sft":
        from post_QA.seen_build import sft

        result = sft.compile_records(args.records_root, args.output,
                                     exclude_index=args.exclude_index, seed=args.seed,
                                     workers=args.workers)
        _print({key: result[key] for key in (
            "path", "record_count", "covered_records", "excluded_records",
            "qa_items", "task_totals")})
        return 0
    if args.command == "export-sft":
        from post_QA.seen_build import sft

        _print(sft.export_view(args.root, hide_params=args.hide_params))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
