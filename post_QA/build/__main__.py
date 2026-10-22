"""Build a benchmark, its radius-pair diagnostic set and (when the spec has one) the SFT from a meta root.

    python -m post_QA.build --spec post_QA/specs/seen.json [--datasets r2r gs] [--draft]
    python -m post_QA.build --spec post_QA/specs/unseen.json

The spec gives the meta root ("meta"; only observations of its "split" are read), the name N and the output O.
A release (every dataset of the spec, all checks passed, no hard-quota shortfall) writes
    O/benchmark/N/QA[_<ablation>].json                main benchmark
    O/benchmark/N_radius_pairs/QA.json                radius-pair diagnostic set (full configuration only)
    O/metadata/N/{selection.parquet,report.json,preview/}
    <sft dir>/json/<ablation>.jsonl, <sft dir>/metadata/{selection,excluded}.parquet   (with an SFT)
and, last, O/release.json; then data/qa_draft/N/ and the marked images of the meta root that no release
(data/benchmark*/release.json) references are removed. Anything else is a draft in data/qa_draft/N/. Images are
not copied: item image paths are relative to the meta root (release.json "image_root").
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image, ImageDraw

from pipeline import io_utils
from post_QA import templates as T
from post_QA.build import candidates as C, images, render as R
from post_QA.build.select import Selector

ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in value) if lines
            else json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    path.write_text(text, encoding="utf-8")


def _preview(meta, facts, path):
    """Contact sheet of up to 24 items of one task: first image and answer."""
    sheet = Image.new("RGB", (6 * 214, 4 * 180), "white")
    draw = ImageDraw.Draw(sheet)
    for k, f in enumerate(facts[:24]):
        with Image.open(meta / f["images"][0]) as image:
            sheet.paste(image.convert("RGB").resize((210, 158)), (k % 6 * 214, k // 6 * 180))
        draw.text((k % 6 * 214 + 2, k // 6 * 180 + 160), f"{f['dataset']} {f['answer']}"[:34], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


def _review_pending(meta, obs, quality):
    """Queue for the separate reviewer. Contains no task text, answer or marked image."""
    pending = {}
    for dataset, table in obs.items():
        for oid, row in table[table["image_review"] == "unreviewed"].iterrows():
            pending.setdefault(row.image_sha256, {"image_sha256": row.image_sha256,
                "path": str(meta / row.image), "dataset": dataset, "scene": row.scene,
                "obs_id": oid, "role": "initial", "decision": row.image_review})
    for dataset, path in sorted(quality.considered):
        sha = Path(path).stem
        state = quality.reviews.status(sha)
        if state == "unreviewed":
            pending.setdefault(sha, {"image_sha256": sha, "path": str(meta / path), "dataset": dataset,
                "scene": "", "obs_id": "", "role": "option", "decision": state})
    return pd.DataFrame(pending.values(), columns=["image_sha256", "path", "dataset", "scene", "obs_id", "role", "decision"])


def _write_review_pending(meta, obs, quality, spec_path, report):
    directory = ROOT / "data/analysis/image_review/gaps" / spec_path.stem
    directory.mkdir(parents=True, exist_ok=True)
    pending = _review_pending(meta, obs, quality)
    pending.to_csv(directory / "review_pending.csv", index=False)
    report = {**report, "pending_images": len(pending), "directory": str(directory)}
    _write_json(directory / "report.json", report)
    print(json.dumps(report), flush=True)


def _prune(meta):
    """Remove the marked images of the meta root that no release references (every data/benchmark*/release.json,
    image paths resolved against its image root); nothing is removed if a release cannot be read."""
    used = set()
    for manifest in (ROOT / "data").glob("benchmark*/release.json"):
        try:
            version = json.loads(manifest.read_text())
            for part in version["partitions"].values():
                root = (manifest.parent / part["image_root"]).resolve()
                sft = [manifest.parent / part["sft"]["dir"] / "metadata"] if part["sft"] else []
                for selection in [manifest.parent / part["metadata_dir"]] + sft:
                    used |= {(root / p).resolve() for paths in pd.read_parquet(selection / "selection.parquet",
                                                                                 columns=["images"])["images"] for p in paths}
        except (OSError, KeyError, ValueError) as error:
            print(f"marked images kept: {manifest} unreadable ({error})", flush=True)
            return
    for path in (meta / "marked").rglob("*.png"):
        if path.resolve() not in used:
            path.unlink()


def _benchmark(obs, cands, pairs, spec, dino, quality, *, prepare_reviews=False):
    caps = {}
    while True:  # integer capacities only decrease, so retries stop even when a quota is infeasible
        selector = Selector(obs, cands, pairs, spec, dino, quality, caps, prepare_reviews=prepare_reviews)
        selector.benchmark()
        short = {(s["dataset"], s["task"], int(s["cell"].split("=")[1])): s["picked"]
                 for s in selector.shortfalls if s["cell"].startswith("label=") and "picked" in s}
        tighter = {k: v for k, v in short.items() if k not in caps or v < caps[k]}
        if not tighter:
            return selector
        print(json.dumps({"label_caps": {"/".join(map(str, k)): v for k, v in tighter.items()}}), flush=True)
        caps.update(tighter)


def merge_release(current, name, part):
    """Combine separately built seen/unseen partitions under one benchmark root."""
    partitions = {**current.get("partitions", {}), name: part}
    return {"schema": "egoconseq.qa-release.v1", "status": part["status"],
            "partitions": partitions,
            "sets": {k: {**v, "image_root": p["image_root"]}
                     for p in partitions.values() for k, v in p["sets"].items()},
            "files": {f"{n}/{k}": v for n, p in partitions.items() for k, v in p["files"].items()},
            "metadata": {k: v for p in partitions.values() for k, v in p["metadata"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--draft", action="store_true", help="write data/qa_draft/<name>/ instead of a release")
    parser.add_argument("--prepare-reviews", action="store_true",
                        help="select provisionally to list unaudited C1 options; write only a review queue, never QA")
    parser.add_argument("--sft-total", type=int, help="override the spec's SFT size (drafts only)")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if not spec.get("image_reviews"):
        raise SystemExit("use a new spec with a frozen image_reviews table; v10 specs belong to the archived builder")
    meta, name, sft = (ROOT / spec["meta"]).resolve(), spec["name"], "sft" in spec
    sets = {"benchmark": name, "radius_pairs": f"{name}_radius_pairs"}
    meta_card = C.card(meta)
    datasets = args.datasets or spec["datasets"]
    missing = [d for d in datasets if d not in meta_card["datasets"]]
    if missing:
        raise SystemExit(f"not exported in {meta}: {missing}")
    release = not args.draft and not args.prepare_reviews and not args.sft_total and sorted(datasets) == sorted(spec["datasets"])
    if args.sft_total:
        spec["sft"]["total"] = args.sft_total
    radii = meta_card["gt_contract"]["radii_m"]

    reviews = images.Reviews(ROOT / spec["image_reviews"])
    quality = images.Qualities(meta, reviews)
    obs = {d: C.observations(meta, d, spec, quality) for d in datasets}
    empty = [d for d, o in obs.items() if not o["bench_image"].any()]
    if empty:  # incomplete audits must not trigger an expensive, empty selection or masquerade as bad images
        _write_review_pending(meta, obs, quality, args.spec, {"status": "incomplete", "datasets_without_eligible_images": empty,
            "reason": "no_eligible_initial_images",
            "review_states": {d: reviews.summary(o["image_sha256"]) for d, o in obs.items()}})
        raise SystemExit(f"no eligible initial images for {empty}; see the review coverage report")
    jobs = [(d, task) for d in datasets for task in C.tasks_of(meta_card, d)]
    with ProcessPoolExecutor(min(len(jobs), 12)) as pool:  # every task table is read in parallel
        results = dict(zip(jobs, pool.map(C.candidates, *zip(*[(meta, d, t, spec, radii) for d, t in jobs]))))
    cands = {d: {t: results[d, t][0] for d2, t in jobs if d2 == d} for d in datasets}
    pairs = {d: results[d, "A1"][1] for d in datasets}
    stats = {f"{d}/{t}": results[d, t][2] for d, t in jobs}
    print(json.dumps({"observations": {d: len(o) for d, o in obs.items()}, "candidates": stats}), flush=True)

    dino = images.Dino(spec["benchmark"]["dino"]["model"])
    selector = _benchmark(obs, cands, pairs, spec, dino, quality, prepare_reviews=args.prepare_reviews)
    selector.radius_pairs()
    excluded = selector.sft() if sft else set()
    R.review_shortfalls(selector.shortfalls, obs, cands, quality)
    if args.prepare_reviews:
        # Audit only options provisionally selected, not every unused family traversed by the sampler.
        quality.considered = {(r["dataset"], p) for r in selector.rows if r["task"] == "C1" for p in r["options"]}
        _write_review_pending(meta, obs, quality, args.spec, {"status": "review_plan",
                    "selection_rows": len(selector.rows), "shortfalls": selector.shortfalls,
                    "note": "Provisional selection only; unreviewed C1 images cannot enter a QA release."})
        return
    selection = R.present(pd.DataFrame(selector.rows), radii, spec["seed"])
    out, facts, marked = R.items(meta, selection, obs)
    problems, hashes = R.check(meta, selection, out, facts, marked, obs, spec, selector.cells, excluded,
                               quality, selector.similar)
    print(json.dumps({"problems": len(problems), "first": problems[:20]}), flush=True)
    hard = [s for s in selector.shortfalls if s["set"] != "sft"]
    release = release and not hard and not problems
    checks = {"edge_density": "report_only", "reviews": reviews.identity,
              "review_policy": "all displayed images require keep; missing reviews are coverage gaps",
              "dino": dino.identity,
              "near_view": spec["benchmark"]["dino"]["near_view"], "threshold": spec["benchmark"]["dino"]["threshold"],
              "max_dark_fraction": {k: spec[k]["max_dark_fraction"] for k in ("benchmark", "sft") if k in spec}}
    extra = {"candidates": stats, "shortfalls": selector.shortfalls, "problems": problems[:200],
             "problem_count": len(problems), "image_checks": checks, "benchmark_supply": R.losses(cands, pairs, obs),
             "observation_reviews": {d: o["image_review"].value_counts().to_dict() for d, o in obs.items()},
             "scene_reviews": {d: {s: g["image_review"].value_counts().to_dict() for s, g in o.groupby("scene")}
                               for d, o in obs.items()},
             "option_reviews": reviews.summary(Path(p).stem for _, p in quality.considered),
             "prompt_version": T.PROMPT_VERSION}
    if sft:
        log = pd.DataFrame(selector.log).drop_duplicates(["kind", "dataset", "id"])
        extra["sft_exclusions"] = {f"{d}/{k}/{r}": int(n)
                                   for (d, k, r), n in log.groupby(["dataset", "kind", "reason"]).size().items()}
    report = R.report(facts, obs, spec, extra)

    if release:
        targets = {"benchmark": ROOT / spec["output"], **({"sft": ROOT / spec["sft"]["dir"]} if sft else {})}
    else:
        draft = ROOT / "data" / "qa_draft" / name
        targets = {"benchmark": draft / "benchmark", **({"sft": draft / "sft"} if sft else {})}
    metadata_dir = Path("metadata") / name
    if release and (any((targets["benchmark"] / p).exists() for p in
                        (f"benchmark/{name}", f"benchmark/{name}_radius_pairs", metadata_dir))
                    or sft and targets["sft"].exists()):
        raise SystemExit(f"{name} already exists; use --draft before replacing the current data")
    staging = {k: v.parent / f"_{v.name}_{name}.tmp" for k, v in targets.items()}
    for path in staging.values():
        shutil.rmtree(path, ignore_errors=True)
    files = {}
    for (kind, version), rows in sorted(out.items()):
        file_name = "QA.json" if version == "full" else f"QA_{version}.json"
        if kind == "sft":
            path = staging["sft"] / "json" / f"{version}.jsonl"
            _write_json(path, rows, lines=True)
        else:
            path = staging["benchmark"] / "benchmark" / sets[kind] / file_name
            _write_json(path, rows)
        files[f"{kind}/{version}"] = path
    facts_frame = pd.DataFrame(facts).drop(columns=["gt", "actions"])
    facts_frame["pair_id"] = facts_frame["pair_id"].fillna("").astype(str)
    facts_frame["image_sha256"] = [[hashes[p] for p in f["images"]] for f in facts]
    for key in targets:
        part = facts_frame[facts_frame["set"].isin(["benchmark", "radius_pairs"] if key == "benchmark" else ["sft"])]
        metadata = staging[key] / (metadata_dir if key == "benchmark" else Path("metadata"))
        metadata.mkdir(parents=True, exist_ok=True)
        if reviews.raw is not None:
            (metadata / "image_reviews.csv").write_bytes(reviews.raw)
        if reviews.provenance is not None:
            (metadata / "image_reviews.json").write_bytes(reviews.provenance)
        part.to_parquet(metadata / "selection.parquet", index=False)
        _write_json(metadata / "report.json",
                    {**report, "items": {k: v for k, v in report["items"].items()
                                         if (k.split("/")[0] == "sft") == (key == "sft")}})
    if sft:
        log.to_parquet(staging["sft"] / "metadata" / "excluded.parquet", index=False)
    for task in T.TASK_IDS:
        chosen = sorted((f for f in facts if f["set"] == "benchmark" and f["task"] == task),
                        key=lambda f: hashlib.sha256(f["id"].encode()).digest())
        if chosen:
            _preview(meta, chosen, staging["benchmark"] / metadata_dir / "preview" / f"{task}.jpg")

    for key, target in targets.items():  # publish: move the staged trees, then write release.json last
        if not release:
            shutil.rmtree(target, ignore_errors=True)
            staging[key].rename(target)
            continue
        target.mkdir(parents=True, exist_ok=True)
        if key == "benchmark":
            for relative in (Path("benchmark") / name, Path("benchmark") / f"{name}_radius_pairs", metadata_dir):
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                (staging[key] / relative).rename(target / relative)
            shutil.rmtree(staging[key])
        else:
            for child in staging[key].iterdir():
                child.rename(target / child.name)
            staging[key].rmdir()

    def published(path):
        key = next(k for k in staging if path.is_relative_to(staging[k]))
        return targets[key] / path.relative_to(staging[key])

    code = (sorted((ROOT / "post_QA").rglob("*.py")) + sorted((ROOT / "post_QA").glob("*.json")) + [args.spec.resolve()]
            + [ROOT / "pipeline" / f for f in ("__init__.py", "config.py", "actions.py", "io_utils.py",
                                               "assets/scene_partitions.v1.json")])  # everything post_QA reads
    manifest = {
        "schema": "egoconseq.qa-release.v1", "status": "release" if release else "draft",
        "prompt_version": T.PROMPT_VERSION, "datasets": datasets, "seed": spec["seed"], "domains": T.DOMAINS,
        "image_root": os.path.relpath(meta, targets["benchmark"]),
        "metadata_dir": metadata_dir.as_posix(),
        "sets": {sets[k]: {"dir": f"benchmark/{sets[k]}", "scoring": "pairs" if k == "radius_pairs" else "items",
                           "tasks": sorted({f["task"] for f in facts if f["set"] == k})} for k in sets},
        "sft": {"dir": os.path.relpath(targets["sft"], targets["benchmark"])} if sft else None,
        "files": {k: {"path": published(p).relative_to(ROOT).as_posix(), "sha256": io_utils.sha256_file(published(p))}
                  for k, p in files.items()},
        "metadata": {p.relative_to(ROOT).as_posix(): io_utils.sha256_file(p) for k, t in targets.items()
                     for p in sorted((t / (metadata_dir if k == "benchmark" else Path("metadata"))).iterdir()) if p.is_file()},
        "inputs": {"meta_card_sha256": io_utils.sha256_file(meta / "dataset_card.json"),
                   "meta_files": {p.relative_to(meta).as_posix(): io_utils.sha256_file(p) for d in datasets
                                  for p in sorted(meta.glob(f"*/{d}.parquet")) + sorted(meta.glob(f"tasks/*/{d}.parquet"))},
                   "code_sha256": {p.relative_to(ROOT).as_posix(): io_utils.sha256_file(p) for p in code}},
        "image_checks": checks, "shortfalls": selector.shortfalls,
    }
    manifest_path = targets["benchmark"] / "release.json"
    current = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    io_utils.atomic_write_json(manifest_path, merge_release(current, name, manifest), durable=True)
    if release:
        shutil.rmtree(ROOT / "data" / "qa_draft" / name, ignore_errors=True)
        _prune(meta)
    print(json.dumps({"status": manifest["status"], "items": {k[0]: len(v) for k, v in out.items() if k[1] == "full"},
                      "shortfalls": selector.shortfalls}, default=str))


if __name__ == "__main__":
    main()
