#!/usr/bin/env python3
"""Fail closed unless the ABC golden artifact still rebuilds exactly.

Reads the golden manifest, rebuilds the candidate artifact from the registered
input shards into a throwaway directory, and exits non-zero if any frozen
guarantee breaks: per-file digests, superseded-transition accounting, task
coverage, record validation, or the scorer-integrity replay. Every check runs
even after an earlier one fails, so one invocation reports the whole picture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import gate_authority  # noqa: E402


DEFAULT_MANIFEST = (
    ROOT / "docs" / "golden" /
    "2026-08-14-r2r-gs-b1k-abc-golden.json")
OUTPUT_PLACEHOLDER = "<OUT>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True, check=False)


def _rebuild(manifest: dict, out_dir: Path) -> tuple[list[str], dict]:
    """Replay the manifest's recorded argv into *out_dir*."""
    argv = [sys.executable if index == 0 else
            token.replace(OUTPUT_PLACEHOLDER, str(out_dir))
            for index, token in enumerate(manifest["build_argv"])]
    completed = _run(argv)
    if completed.returncode != 0:
        return ([f"rebuild failed (exit {completed.returncode}): "
                 f"{completed.stderr.strip()[-400:]}"], {})
    try:
        return ([], json.loads(completed.stdout))
    except json.JSONDecodeError:
        return ([f"rebuild produced no JSON summary: "
                 f"{completed.stdout.strip()[-400:]}"], {})


def _check_digests(manifest: dict, out_dir: Path) -> list[str]:
    failures = []
    expected_paths = set(manifest["outputs_sha256"])
    rebuilt_paths = {
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*") if path.is_file()
    }
    for relative in sorted(rebuilt_paths - expected_paths):
        failures.append(f"unexpected rebuilt file: {relative}")
    for relative, expected in sorted(manifest["outputs_sha256"].items()):
        produced = out_dir / relative
        if not produced.is_file():
            failures.append(f"missing rebuilt file: {relative}")
            continue
        actual = _sha256(produced)
        if actual != expected:
            failures.append(
                f"digest changed: {relative}\n"
                f"    expected {expected}\n"
                f"    rebuilt  {actual}")
    return failures


def _check_input_digests(manifest: dict, *, root: Path = ROOT) -> list[str]:
    """Verify every frozen record, funnel, and collection authority input."""
    failures = []
    for entry in manifest.get("inputs", []):
        shard = Path(entry["shard"])
        for name in (
                "records.jsonl", "run_meta.json", "collection_funnel.json"):
            relative = shard / name
            path = root / relative
            if not path.is_file():
                failures.append(f"missing input: {relative.as_posix()}")
            elif _sha256(path) != entry[name]:
                failures.append(
                    f"input digest changed: {relative.as_posix()}")
    for relative_name, expected in sorted(
            manifest.get("collection_inputs_sha256", {}).items()):
        path = root / relative_name
        if not path.is_file():
            failures.append(f"missing collection input: {relative_name}")
        elif _sha256(path) != expected:
            failures.append(
                f"collection input digest changed: {relative_name}")
    return failures


def _check_no_retired_c1_gate(manifest: dict) -> list[str]:
    """Keep the active Golden replay on the sole counterfactual C1 route."""
    failures = []
    if "required_gate" in manifest:
        failures.append("active golden manifest contains retired C1 gate")
    argv = manifest.get("build_argv")
    if not isinstance(argv, list):
        return failures + ["golden manifest build_argv is invalid"]
    retired_options = {
        "--future-view-gate",
        "--future-view-gate-sha256",
        "--future-view-gate-authority-id",
    }
    present = sorted({str(token) for token in argv} & retired_options)
    if present:
        failures.append(
            "active golden build argv contains retired C1 gate option(s): "
            + ", ".join(present))
    return failures


def _check_semantic_diff(manifest: dict, *, root: Path = ROOT) -> list[str]:
    """Recompute the declared transition from its archived predecessor."""
    transition = manifest.get("semantic_diff_from_superseded")
    if not isinstance(transition, dict):
        return ["golden manifest semantic superseded diff is missing"]

    relative_name = transition.get("superseded_manifest")
    if not isinstance(relative_name, str) or not relative_name:
        return ["superseded golden manifest path is invalid"]
    relative_path = Path(relative_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return ["superseded golden manifest must be a contained relative path"]
    superseded_path = gate_authority.lexical_absolute_path(
        relative_path, root=root)

    expected_sha256 = transition.get("superseded_manifest_sha256")
    if (not isinstance(expected_sha256, str) or
            len(expected_sha256) != 64 or
            any(character not in "0123456789abcdef"
                for character in expected_sha256)):
        return ["superseded golden manifest SHA is invalid"]
    if not superseded_path.is_file():
        return [f"superseded golden manifest is missing: {relative_name}"]
    if _sha256(superseded_path) != expected_sha256:
        return [f"superseded golden manifest digest changed: {relative_name}"]
    try:
        superseded = gate_authority.load_authority_manifest(superseded_path)
    except (TypeError, ValueError) as error:
        return [f"superseded golden manifest is invalid: {error}"]

    current_outputs = manifest.get("outputs_sha256")
    superseded_outputs = superseded.get("outputs_sha256")
    if not isinstance(current_outputs, dict):
        return ["golden manifest outputs_sha256 is invalid"]
    if not isinstance(superseded_outputs, dict):
        return ["superseded golden manifest outputs_sha256 is invalid"]

    all_outputs = set(current_outputs) | set(superseded_outputs)
    changed_outputs = {
        name for name in all_outputs
        if current_outputs.get(name) != superseded_outputs.get(name)
    }
    image_prefix = "candidate_qa/public/images/"
    changed_images = {
        name for name in changed_outputs if name.startswith(image_prefix)
    }
    changed_non_images = sorted(changed_outputs - changed_images)
    unchanged_count = len(all_outputs) - len(changed_outputs)

    failures = []
    declared_changed = transition.get("changed_non_image_outputs")
    if (not isinstance(declared_changed, list) or
            any(not isinstance(name, str) for name in declared_changed) or
            len(set(declared_changed)) != len(declared_changed)):
        failures.append(
            "declared changed non-image outputs are invalid")
    elif sorted(declared_changed) != changed_non_images:
        failures.append(
            "declared changed non-image outputs do not match superseded "
            f"diff: expected {changed_non_images!r}, declared "
            f"{sorted(declared_changed)!r}")

    declared_images_changed = transition.get("image_outputs_changed")
    actual_images_changed = bool(changed_images)
    if (not isinstance(declared_images_changed, bool) or
            declared_images_changed != actual_images_changed):
        failures.append(
            "declared image change flag does not match superseded diff: "
            f"expected {actual_images_changed!r}, declared "
            f"{declared_images_changed!r}")

    declared_unchanged = transition.get("unchanged_output_count")
    if (not isinstance(declared_unchanged, int) or
            isinstance(declared_unchanged, bool) or
            declared_unchanged != unchanged_count):
        failures.append(
            "declared unchanged output count does not match superseded diff: "
            f"expected {unchanged_count}, declared {declared_unchanged!r}")
    return failures


def _check_coverage(manifest: dict, summary: dict) -> list[str]:
    expected = manifest["coverage"]
    actual = summary.get("coverage")
    if actual != expected:
        return [f"coverage changed: expected {expected}, rebuilt {actual}"]
    return []


def _check_records(manifest: dict) -> list[str]:
    failures = []
    for entry in manifest["inputs"]:
        shard = ROOT / entry["shard"]
        completed = _run([
            sys.executable, str(ROOT / "scripts" / "check_records.py"),
            str(shard / "records.jsonl"),
            "--validation-level", "source",
            "--run-meta", str(shard / "run_meta.json"),
            "--expected-run-meta-sha256", entry["run_meta.json"]])
        report = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0 or "0 violations" not in report:
            failures.append(
                f"record validation failed: {entry['shard']}\n"
                f"    {report[-400:]}")
    return failures


def _check_gt_replay(
        out_dir: Path, *, source_authority_manifest: Path,
        source_authority_root: Path = ROOT) -> list[str]:
    completed = _run([
        sys.executable, str(ROOT / "scripts" / "eval_benchmark.py"),
        "--benchmark", str(out_dir / "candidate_qa"), "--gt-as-pred",
        "--source-authority-manifest", str(source_authority_manifest),
        "--source-authority-root", str(source_authority_root)])
    if completed.returncode != 0:
        return [f"gt-as-pred replay failed to run: "
                f"{completed.stderr.strip()[-400:]}"]
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return [f"gt-as-pred replay produced no JSON: "
                f"{completed.stdout.strip()[-400:]}"]
    if result.get("overall") != 1.0:
        return [f"gt-as-pred overall is {result.get('overall')!r}, not 1.0"]
    if result.get("six_task_macro") != 1.0:
        return ["gt-as-pred six-task macro is "
                f"{result.get('six_task_macro')!r}, not 1.0"]
    return []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    manifest = gate_authority.load_authority_manifest(args.manifest)
    if manifest.get("status") == "superseded":
        # Say so here rather than letting the rebuild fail somewhere inside
        # check_records, where the message is about a schema version and gives
        # no hint that the manifest itself was retired.
        print(f"{args.manifest} is a superseded provenance record, not a "
              f"runnable gate.", file=sys.stderr)
        print(f"  superseded by: {manifest.get('superseded_by')}",
              file=sys.stderr)
        print(f"  reason: {manifest.get('superseded_reason')}", file=sys.stderr)
        return 2
    failures: list[str] = _check_input_digests(manifest)
    failures.extend(_check_no_retired_c1_gate(manifest))
    failures.extend(_check_semantic_diff(manifest))

    with tempfile.TemporaryDirectory(prefix="abc-golden-") as scratch:
        out_dir = Path(scratch) / "rebuild"
        rebuild_failures, summary = _rebuild(manifest, out_dir)
        failures.extend(rebuild_failures)
        if not rebuild_failures:
            failures.extend(_check_digests(manifest, out_dir))
            failures.extend(_check_coverage(manifest, summary))
            failures.extend(_check_gt_replay(
                out_dir, source_authority_manifest=args.manifest))
        failures.extend(_check_records(manifest))

    if failures:
        print(f"ABC golden check FAILED ({len(failures)} problem(s)):",
              file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    transition = manifest["semantic_diff_from_superseded"]
    image_status = (
        "images changed" if transition["image_outputs_changed"] else
        "images unchanged")
    print(
        "ABC golden check OK: active rebuild matches "
        f"{len(manifest['outputs_sha256'])}-file manifest; archived "
        f"transition {len(transition['changed_non_image_outputs'])} "
        "non-image changed / "
        f"{transition['unchanged_output_count']} unchanged, {image_status}; "
        f"coverage {manifest['coverage']}, {len(manifest['inputs'])} shards "
        "clean, gt-as-pred overall/six-task-macro 1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
