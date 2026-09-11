"""Materialize refined records or compile QA directly from the existing catalog."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from pipeline import abc1_record, candidate_sources, io_utils
from post_QA.seen_build import artifact, catalog, inventory, selection, spec


def load_results(
        work_dir: Path, plan_header: dict,
        plan_rows: list[dict]) -> dict[str, dict]:
    """Return only deltas bound to this exact plan row."""
    plan_id = str(plan_header["plan_id"])
    by_row = {str(row["plan_row_id"]): row for row in plan_rows}
    rows = {}
    for path in sorted(Path(work_dir).glob("*.jsonl")):
        if path.name == "plan.jsonl":
            continue
        for result in io_utils.read_jsonl(path, require_dict=True):
            row_id = result.get("plan_row_id")
            planned = by_row.get(str(row_id))
            if (result.get("schema") == "egoconseq.seen-build-result.v2" and
                    result.get("plan_id") == plan_id and planned is not None and
                    result.get("record_uid") == planned["record_uid"] and
                    result.get("source_record_sha256") ==
                    planned["source_record_sha256"]):
                rows[str(row_id)] = result
    return rows


def _read_source_record(row: dict, streams: dict[str, object]) -> dict:
    path = str(row["source_path"])
    source = streams.get(path)
    if source is None:
        source = Path(path).open("rb")
        streams[path] = source
    source.seek(int(row["byte_offset"]))
    payload = source.readline().rstrip(b"\r\n")
    if hashlib.sha256(payload).hexdigest() != row["source_record_sha256"]:
        raise ValueError(f"source record changed: {row['record_uid']}")
    return json.loads(payload)


def compact_record(row: dict, record: dict, result: dict) -> dict | None:
    compact = abc1_record.from_legacy(
        record, dataset=str(row["dataset"]),
        source_records_sha256="",
        byte_offset=int(row["byte_offset"]),
        record_uid=str(row["record_uid"]))
    compact = abc1_record.normalize(compact)
    if result.get("mode") == "surfaces":
        updated = abc1_record.install_surface_delta(compact, result)
        audit_surface_merge(compact, updated)
        return updated
    if result.get("mode") not in {"added", "a2", "c1", "pruned"}:
        return compact
    return abc1_record.install_delta(
        compact, outcomes=result.get("outcomes") or [],
        provenance=result.get("provenance") or {},
        mode=str(result.get("mode")))


def _protected_case(case: dict) -> dict:
    value = json.loads(json.dumps(case))
    outputs = value.get("task_outputs") or {}
    value["task_outputs"] = {
        task: output for task, output in outputs.items()
        if task not in {"A4", "B1", "B2"}
    }
    return value


def audit_surface_merge(original: dict, updated: dict) -> None:
    """Fail if surface installation changes frozen geometry or old evidence."""
    geometry_keys = (
        "record_uid", "dataset", "scene_id", "frame_id", "pose", "sensor",
        "floor_plane", "body_radii_m", "image_path", "visible_entities",
    )
    if any(original.get(key) != updated.get(key) for key in geometry_keys):
        raise ValueError("surface merge changed protected record geometry")
    old_cases = list(original.get("cases") or [])
    new_cases = list(updated.get("cases") or [])
    if len(new_cases) not in {len(old_cases), len(old_cases) + 1}:
        raise ValueError("surface merge changed protected case count")
    if [_protected_case(case) for case in new_cases[:len(old_cases)]] != [
            _protected_case(case) for case in old_cases]:
        raise ValueError("surface merge changed protected existing cases")
    if updated.get("c1_families") != original.get("c1_families"):
        raise ValueError("surface merge changed protected C1 families")
    if len(new_cases) == len(old_cases) + 1 and not (
            new_cases[-1].get("collision") is False and
            new_cases[-1].get("completed") is True):
        raise ValueError("surface merge appended a non-safe case")


def _record_asset_paths(record: dict) -> set[str]:
    paths = {str(record["image_path"])}
    paths.update(
        str(case["terminal_rgb_path"])
        for case in record.get("cases") or []
        if case.get("terminal_rgb_path"))
    return paths


def _flat_source_catalog(values) -> list[dict]:
    flattened = []

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict) and value:
            flattened.append(value)

    visit(values)
    unique = {}
    for value in flattened:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, value)
    return [unique[key] for key in sorted(unique)]


def _source_references(header: dict, dataset: str,
                       dataset_rows: list[dict]) -> list[dict]:
    declared = {
        str(Path(value["path"]).resolve()): value
        for value in header.get("sources") or []
        if str(value.get("dataset")) == dataset
    }
    references = []
    seen = set()
    for row in dataset_rows:
        path = str(Path(row["source_path"]).resolve())
        if path in seen:
            continue
        seen.add(path)
        source = declared.get(path)
        if source is None:
            raise ValueError(f"plan source is undeclared: {path}")
        references.append({
            "dataset": dataset, "records_path": path,
            "records_sha256": str(source["records_sha256"]),
        })
    return references


def materialize_records(
        plan_path: Path, work_dir: Path, output_root: Path, *,
        reuse_uncollected: bool = False) -> dict:
    """Write scene-contiguous records directly from source plus worker deltas."""
    header, rows = inventory.read_plan(plan_path)
    results = load_results(work_dir, header, rows)
    missing = [row["plan_row_id"] for row in rows
               if row["plan_row_id"] not in results]
    if missing and not reuse_uncollected:
        raise ValueError(f"collection is incomplete: {len(missing)} records")
    for row_id in missing:
        results[str(row_id)] = {"status": "uncollected", "mode": "reused"}
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    manifest_rows = []
    modes = Counter()
    for dataset in spec.DATASETS:
        dataset_rows = by_dataset.get(dataset)
        if not dataset_rows:
            continue
        target_dir = output_root / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        staged_assets = {
            str(asset["path"]): Path(asset["source"])
            for result in results.values()
            if str(result.get("dataset")) == dataset
            for asset in result.get("staged_assets") or []
        }
        target_path = target_dir / "records.jsonl"
        digest = hashlib.sha256()
        referenced_assets = set()
        asset_sources = {}
        rejected_capacity = 0
        scene_ranges = []
        active_scene = None
        active_start = 0
        active_count = 0
        source_streams = {}
        try:
            with target_path.open("wb") as destination:
                for row in dataset_rows:
                    scene = str(row["scene_id"])
                    if scene != active_scene:
                        if active_scene is not None:
                            scene_ranges.append({
                                "scene_id": active_scene,
                                "start_byte": active_start,
                                "end_byte": destination.tell(),
                                "record_count": active_count,
                            })
                        active_scene = scene
                        active_start = destination.tell()
                        active_count = 0
                    record = _read_source_record(row, source_streams)
                    result = results[str(row["plan_row_id"])]
                    mode = result.get("mode")
                    compact = compact_record(row, record, result)
                    if compact is None:
                        rejected_capacity += 1
                        compact = compact_record(
                            row, record, {"mode": "reused"})
                        mode = "capacity_rejected"
                    payload = json.dumps(
                        compact, separators=(",", ":"),
                        allow_nan=False).encode("utf-8") + b"\n"
                    destination.write(payload)
                    digest.update(payload)
                    referenced_assets.update(_record_asset_paths(compact))
                    source_dir = Path(row["source_path"]).parent
                    for relative in _record_asset_paths(compact):
                        asset_sources.setdefault(
                            relative,
                            staged_assets.get(relative, source_dir / relative))
                    active_count += 1
                    modes[str(mode or result["status"])] += 1
                if active_scene is not None:
                    scene_ranges.append({
                        "scene_id": active_scene,
                        "start_byte": active_start,
                        "end_byte": destination.tell(),
                        "record_count": active_count,
                    })
        finally:
            for source in source_streams.values():
                source.close()
        for relative in sorted(referenced_assets):
            source = asset_sources[relative]
            target = target_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            io_utils.link_or_copy_file(source, target)
        source_metas = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted({
                            Path(row["source_path"]).with_name("run_meta.json")
                            for row in dataset_rows})]
        source_meta = source_metas[0]
        source_catalog = _flat_source_catalog([
            value.get("source_catalog") for value in source_metas])
        if not source_catalog:
            source_catalog = _source_references(
                header, dataset, dataset_rows)
        run_meta = {
            "schema": "egoconseq.abc1-records-dataset.v1",
            "dataset": dataset,
            "split": header.get("split", "train/seen"),
            "scene_source": dataset_rows[0].get("scene_source", {}),
            "supported_tasks": list(abc1_record.supported_tasks(dataset)),
            "record_schema_version": abc1_record.SCHEMA_VERSION,
            "surface_relation_schema": "surface-point-relation.v4",
            "action_protocol": "alternating-forward-turn.v1",
            "oracle_protocol": str(source_meta.get(
                "oracle_contract_version") or "ground-disc-visible-v8"),
            "initial_turns_deg": sorted(abc1_record.INITIAL_TURNS_DEG),
            "source_catalog": source_catalog,
            "code_revision": source_meta.get("code_revision"),
            "record_count": len(dataset_rows),
            "records_sha256": digest.hexdigest(),
            "scene_byte_ranges": scene_ranges,
            "capacity_rejected": rejected_capacity,
        }
        run_meta_path = target_dir / "run_meta.json"
        io_utils.atomic_write_json(run_meta_path, run_meta, allow_nan=False)
        manifest_rows.append({
            "dataset": dataset,
            "record_count": len(dataset_rows),
            "scene_count": len(scene_ranges),
            "records_path": f"{dataset}/records.jsonl",
            "records_sha256": digest.hexdigest(),
            "run_meta_sha256": io_utils.sha256_file(run_meta_path),
        })
    manifest = {
        "schema": "egoconseq.abc1-records.v1",
        "split": header.get("split", "train/seen"),
        "record_count": len(rows),
        "datasets": manifest_rows,
    }
    io_utils.atomic_write_json(
        output_root / "manifest.json", manifest, allow_nan=False)
    surface_results = [result for result in results.values()
                       if result.get("mode") == "surfaces"]
    surface_statuses = Counter(
        str(result.get("surface_status") or "unknown")
        for result in surface_results)
    starts = Counter(
        str(case.get("starts_with") or "unknown")
        for result in surface_results for case in result.get("cases") or [])
    surface_summary = {
        "record_coverage": len(surface_results),
        "target_records": sum(
            result.get("surface_point_target") is not None
            for result in surface_results),
        "target_points": sum(len(
            (result.get("surface_point_target") or {}).get("points") or [])
            for result in surface_results),
        "added_safe": surface_statuses["added_safe"],
        "existing_safe": surface_statuses["existing_safe"],
        "no_target": surface_statuses["no_target"],
        "no_safe": surface_statuses["no_safe"],
        "starts": dict(sorted(starts.items())),
    }
    return {
        "record_count": len(rows),
        "uncollected_reused": len(missing),
        "datasets": {row["dataset"]: row["record_count"]
                     for row in manifest_rows},
        "modes": dict(sorted(modes.items())),
        **({"surface": surface_summary} if surface_results else {}),
    }


def _validate_artifact(items: list[dict], root: Path) -> None:
    expected = sum(sum(tasks.values())
                   for tasks in spec.DATASET_TASK_TOTALS.values())
    if len(items) != expected or len({item["id"] for item in items}) != expected:
        raise ValueError("benchmark item count differs from the spec")
    counts = Counter((item["dataset"], item["task_id"]) for item in items)
    wanted = Counter({
        (dataset, task): count
        for dataset, tasks in spec.DATASET_TASK_TOTALS.items()
        for task, count in tasks.items() if count
    })
    if counts != wanted:
        raise ValueError("benchmark dataset/task totals differ")
    referenced = set()
    for item in items:
        image_count = 5 if item["task_id"] == "C1" else 1
        question = next(message["content"] for message in item["messages"]
                        if message["role"] == "user")
        if len(item["images"]) != image_count or \
                question.count("<image>") != image_count:
            raise ValueError(f"image slots differ: {item['id']}")
        referenced.update(item["images"])
    stored = {
        path.relative_to(root).as_posix()
        for path in (root / "images").rglob("*.png")
    }
    if referenced != stored:
        raise ValueError("benchmark images contain missing files or orphans")


def materialize_benchmark(
        selected, records_root: Path, output_dir: Path,
        private_dir: Path) -> dict:
    """Compile public QA and its private source index without legacy quotas."""
    source_by_path = catalog.by_path(records_root)
    rows = list(selected)
    output_dir = Path(output_dir)
    private_dir = Path(private_dir)
    with tempfile.TemporaryDirectory(
            prefix=".marked-images-", dir=output_dir.parent) as temporary:
        build_root = Path(temporary)
        materialized = []
        streams = {}
        try:
            for candidate in sorted(rows, key=lambda value: (
                    spec.DATASETS.index(value.dataset), value.byte_offset,
                    value.item_id)):
                source = source_by_path[
                    str(Path(candidate.source_path).resolve())]
                stream = streams.get(source.records_path)
                if stream is None:
                    stream = source.records_path.open("rb")
                    streams[source.records_path] = stream
                stream.seek(candidate.byte_offset)
                record = json.loads(stream.readline())
                if candidate_sources.canonical_sha256(record) != \
                        candidate.record_sha256:
                    raise ValueError(
                        f"selected record changed: {candidate.item_id}")
                materialized.append(artifact.materialize_item(
                    candidate, source, record, output_dir, build_root))
        finally:
            for stream in streams.values():
                stream.close()
    task_order = {task: index for index, task in enumerate(spec.TASKS)}
    materialized.sort(key=lambda value: (
        spec.DATASETS.index(value[0]["dataset"]),
        task_order[value[0]["task_id"]], value[0]["id"]))
    items = [item for item, _entry in materialized]
    entries = [entry for _item, entry in materialized]
    io_utils.atomic_write_json(output_dir / "qa.json", items, allow_nan=False)
    _validate_artifact(items, output_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    index = artifact.write_index(
        entries, output_dir / "qa.json", private_dir / "record_index.json")
    return {"qa_items": len(items), "record_index_items": index["record_count"]}


def compile_records(
        records_root: Path, output_root: Path, *, seed: int,
        device="cuda:0", similarity_threshold=0.90) -> dict:
    """Compile any compact shard catalog directly into ABC1 QA and HTML."""
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        from post_QA.seen_build import visual_selection
        candidates = selection.enumerate_candidates(
            records_root, seed=seed, pool_factor=20, preserve_lengths=True,
            progress=lambda dataset, count:
                print(f"[scan] {dataset}: {count} records", flush=True))
        selected, report = visual_selection.select(
            candidates, seed=seed, device=device, threshold=similarity_threshold)
        report["benchmark_id"] = spec._SPEC["benchmark_id"]
        report["source_split"] = json.loads(
            (Path(records_root) / "manifest.json").read_text()).get("split")
        print("[compile] materializing selected QA and images", flush=True)
        result = materialize_benchmark(
            selected, records_root, stage / "benchmark", stage / "private")
        from visualization.benchmark_browser import build_benchmark_browser

        build_benchmark_browser(
            stage / "benchmark", stage / "private" / "record_index.json",
            stage, selection_report=report)
        io_utils.atomic_write_json(
            stage / "report.json", report, allow_nan=False)
        stage.rename(output_root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "path": str(output_root.resolve()),
        "candidate_pool": len(candidates),
        **result,
        "selection": report,
    }


def rebalance_directions(records_root: Path, benchmark_root: Path, *, device="cuda:0") -> dict:
    """Replace only A4/B2 items and atomically publish QA, index and HTML together."""
    from pipeline import benchmark_candidates
    from post_QA.seen_build import direction_balance, images
    from visualization.benchmark_browser import build_benchmark_browser

    root = Path(benchmark_root).resolve()
    items = json.loads((root / "benchmark/qa.json").read_text())
    entries = json.loads((root / "private/record_index.json").read_text())["items"]
    report = json.loads((root / "report.json").read_text())
    seed = report["seed"]
    by_id = {row["id"]: row for row in items}
    sources = {(s.dataset, s.index_path): s for s in catalog.load(records_root)}
    selected = []
    with ExitStack() as opened:
        streams = {key: opened.enter_context(s.records_path.open("rb"))
                   for key, s in sources.items()}
        for entry in entries:
            key = entry["dataset"], entry["source"]["records_path"]
            source, stream = sources[key], streams[key]
            stream.seek(entry["source"]["byte_offset"])
            record = json.loads(stream.readline())
            if (record["record_uid"] != entry["record_uid"] or
                    candidate_sources.canonical_sha256(record) != entry["record_sha256"]):
                raise ValueError(f"selected record changed: {entry['item_id']}")
            candidates = benchmark_candidates.project_record(
                dataset=entry["dataset"], source_path=str(source.records_path),
                byte_offset=entry["source"]["byte_offset"], record_sha256=entry["record_sha256"],
                record=record, allowed_tasks=(entry["task_id"],))
            row = next(c for c in candidates if c.item_id == entry["item_id"])
            if row.task_id == "C1":
                row = replace(row, c1_label=by_id[row.item_id]["messages"][-1]["content"])
            selected.append(row)
    print("[directions] existing selection loaded; scanning A4/B2 alternatives", flush=True)
    candidates = selection.enumerate_candidates(
        records_root, seed=seed, pool_factor=20, preserve_lengths=True,
        task_ids=direction_balance.TASKS,
        progress=lambda dataset, count: print(f"[scan] {dataset}: {count} records", flush=True))
    bank = images.ImageBank(device)
    updated = direction_balance.refine_images(
        selected, candidates, bank, seed=seed,
        threshold=report["visual_selection"]["cosine_threshold"])
    changed = [row for row in updated if row.item_id not in by_id]
    changed = selection._present(changed, seed)
    paths = [direction_balance.image_path(row) for row in updated]
    report.update(selection.summarize(updated))
    report["direction_balance"] = direction_balance.summary(updated)
    report["answer_buckets"] = dict(sorted(Counter(
        f"{r.dataset}/{r.task_id}/{r.answer_bucket}" for r in updated).items()))
    report["scenes"] = dict(sorted(Counter(f"{r.dataset}/{r.scene_id}" for r in updated).items()))
    report["action_signatures"] = {task: len({r.action_signature for r in updated if r.task_id == task})
                                   for task in spec.TASKS}
    report["physical_parameters"] = {name: dict(sorted(Counter(str(getattr(r, name)) for r in updated).items()))
                                      for name in ("body_radius_m", "camera_height_m", "hfov_deg")}
    report["image_quality"] = {r.item_id: bank.metrics[path] for r, path in zip(updated, paths)}
    report["visual_selection"].update(bank.nearest_pairs(paths, [r.item_id for r in updated]))
    report["visual_selection"]["direction_update"] = {
        "encoded_initial_images": len(bank.features), **dict(bank.timings)}
    report["presentation"] = {"version": "directions-24-v1", "replaced_items": len(changed)}

    work = Path(tempfile.mkdtemp(prefix=f".{root.name}-directions-", dir=root.parent))
    stage, backup = work / "new", work / "old"
    try:
        shutil.copytree(root, stage, copy_function=io_utils.link_or_copy_file)
        entry_by_id = {entry["item_id"]: entry for entry in entries}
        source_by_path = {str(s.records_path): s for s in sources.values()}
        with tempfile.TemporaryDirectory(prefix="marked-", dir=work) as temporary, ExitStack() as opened:
            streams = {path: opened.enter_context(s.records_path.open("rb"))
                       for path, s in source_by_path.items()}
            for row in changed:
                stream = streams[row.source_path]
                stream.seek(row.byte_offset)
                record = json.loads(stream.readline())
                if candidate_sources.canonical_sha256(record) != row.record_sha256:
                    raise ValueError(f"selected record changed: {row.item_id}")
                item, entry = artifact.materialize_item(
                    row, source_by_path[row.source_path], record, stage / "benchmark", Path(temporary))
                by_id[row.item_id], entry_by_id[row.item_id] = item, entry
        kept = {row.item_id for row in updated}
        for item in items:
            if item["id"] not in kept:
                for path in item["images"]:
                    (stage / "benchmark" / path).unlink()
        items = [by_id[row.item_id] for row in updated]
        entries = [entry_by_id[row.item_id] for row in updated]
        qa_path = stage / "benchmark/qa.json"
        io_utils.atomic_write_json(qa_path, items, allow_nan=False)
        _validate_artifact(items, stage / "benchmark")
        artifact.write_index(entries, qa_path, stage / "private/record_index.json")
        io_utils.atomic_write_json(stage / "report.json", report, allow_nan=False)
        (stage / "index.html").unlink()
        build_benchmark_browser(stage / "benchmark", stage / "private/record_index.json",
                                stage, selection_report=report)
        root.rename(backup)
        try:
            stage.rename(root)
        except BaseException:
            backup.rename(root)
            raise
    finally:
        if root.exists():
            shutil.rmtree(work)
    return {"path": str(root), "replaced_items": len(changed),
            "direction_balance": report["direction_balance"]}


def refresh_presentation(records_root: Path, benchmark_root: Path) -> dict:
    """Redraw surface questions from their frozen index; never select or change GT."""
    from pipeline import viz
    from post_QA import templates
    from visualization.benchmark_browser import build_benchmark_browser

    root = Path(benchmark_root).resolve()
    items = json.loads((root / "benchmark/qa.json").read_text())
    index = json.loads((root / "private/record_index.json").read_text())
    by_id = {row["item_id"]: row for row in index["items"]}
    sources = {(source.dataset, source.index_path): source
               for source in catalog.load(records_root)}
    work = Path(tempfile.mkdtemp(prefix=f".{root.name}-refresh-", dir=root.parent))
    stage, backup = work / "new", work / "old"
    refreshed = 0
    try:
        shutil.copytree(root, stage, copy_function=io_utils.link_or_copy_file)
        with ExitStack() as opened:
            streams = {}
            for item in items:
                task = item["task_id"]
                if task not in {"A4", "B1", "B2", "B3"}:
                    continue
                entry = by_id[item["id"]]
                source = sources[entry["dataset"], entry["source"]["records_path"]]
                if source.records_path not in streams:
                    streams[source.records_path] = opened.enter_context(source.records_path.open("rb"))
                stream = streams[source.records_path]
                stream.seek(entry["source"]["byte_offset"])
                record = json.loads(stream.readline())
                if (record["record_uid"] != entry["record_uid"] or
                        candidate_sources.canonical_sha256(record) != entry["record_sha256"]):
                    raise ValueError(f"selected record changed: {item['id']}")
                case = artifact._case(record, entry["usage"]["outcome_id"])
                image = candidate_sources.resolve_record_asset(
                    source.records_path.parent, record["image_path"])
                raw = viz.authenticate_raw_rgb_image(
                    image, expected_sha256=entry["initial_image_sha256"],
                    expected_resolution=record["sensor"]["resolution"])
                target = entry["usage"]["target_point"]
                marked = viz.materialize_target_point_image(
                    raw, target["surface_anchor"], stage / "benchmark" / item["images"][0])
                model = artifact._compact_model(record, case, image, raw.sha256)
                model["target"] = "the marked point"
                if task in {"A4", "B3"}:
                    checkpoint = entry["usage"]["checkpoint"]
                    model["checkpoint"] = {
                        "action_index": checkpoint["action_index_1based"],
                        "fraction": checkpoint["fraction"],
                    }
                choices = templates.QUESTION_TEMPLATES[task]
                template = artifact._template(task, entry["usage"].get("template_id", choices[0]["id"]))
                entry["usage"]["template_id"] = template["id"]
                next(message for message in item["messages"] if message["role"] == "user")["content"] = (
                    templates.render_question(task, artifact._inputs(model), template))
                target["marker"] = marked["marker"]
                target["marked_image_sha256"] = marked["sha256"]
                refreshed += 1
                if refreshed % 250 == 0:
                    print(f"[refresh] redrawn={refreshed}", flush=True)
        qa_path = stage / "benchmark/qa.json"
        io_utils.atomic_write_json(qa_path, items, allow_nan=False)
        artifact.write_index(index["items"], qa_path, stage / "private/record_index.json")
        report = json.loads((stage / "report.json").read_text())
        report["presentation"] = {"version": "surface-dot-v2", "refreshed_items": refreshed}
        io_utils.atomic_write_json(stage / "report.json", report, allow_nan=False)
        # The copied HTML is hardlinked to the live page; replace, never truncate it.
        (stage / "index.html").unlink()
        build_benchmark_browser(stage / "benchmark", stage / "private/record_index.json",
                                stage, selection_report=report)
        root.rename(backup)
        stage.rename(root)
    except BaseException:
        # A failed/interrupted switch must not delete the only live benchmark.
        if backup.exists() and not root.exists():
            backup.rename(root)
        shutil.rmtree(work)
        raise
    shutil.rmtree(work)
    return {"path": str(root), "qa_items": len(items), "refreshed_items": refreshed}
