"""Validate records at an explicit local or source-bound trust level."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import source_manifest, validate


_LOWER_HEX = frozenset("0123456789abcdef")


def _sha256_argument(value):
    if len(value) != 64 or any(char not in _LOWER_HEX for char in value):
        raise argparse.ArgumentTypeError(
            "SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--max-print", type=int, default=40)
    ap.add_argument(
        "--validation-level", required=True, choices=("local", "source"))
    ap.add_argument("--run-meta")
    ap.add_argument(
        "--expected-run-meta-sha256", type=_sha256_argument)
    args = ap.parse_args()

    if args.validation_level == "local" and (
            args.run_meta or args.expected_run_meta_sha256):
        ap.error("local validation cannot consume source authority")
    if args.validation_level == "source" and (
            not args.run_meta or not args.expected_run_meta_sha256):
        ap.error(
            "source validation requires --run-meta and "
            "--expected-run-meta-sha256")
    context = None
    try:
        if args.validation_level == "source":
            metadata, digest = source_manifest.load_object_identity(
                Path(args.run_meta), label="record validation run metadata")
            if digest != args.expected_run_meta_sha256:
                raise ValueError(
                    "run metadata hash does not match expected digest")
            backend = (metadata.get("params") or {}).get("backend") or "r2r"
            context_builder = {
                "b1k": source_manifest.b1k_v16_context_from_run_meta,
                "gs": source_manifest.gs_v18_context_from_run_meta,
                "r2r": source_manifest.r2r_v16_context_from_run_meta,
            }.get(backend)
            if context_builder is None:
                raise ValueError(
                    f"run metadata backend is unsupported: {backend!r}")
            context = context_builder(
                metadata, authority_sha256=args.expected_run_meta_sha256)
    except (OSError, TypeError, ValueError) as error:
        ap.error(str(error))

    if args.validation_level == "local":
        total, viol = validate.validate_file_local(args.records)
    else:
        total, viol = validate.validate_file_source_bound(
            args.records, context=context)
    print(f"checked {total} records; {len(viol)} violations")
    for v in viol[:args.max_print]:
        print("  ", v)
    if len(viol) > args.max_print:
        print(f"  ... and {len(viol) - args.max_print} more")
    sys.exit(1 if viol else 0)


if __name__ == "__main__":
    main()
