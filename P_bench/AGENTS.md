# Repository Guidelines

## Project Structure

`pipeline/` is the only benchmark implementation. Habitat runtime dependencies
stay in `sim.py`; geometry, records, QA construction, validation, and evaluation
remain deterministic and NumPy-oriented. CLI entry points live in `scripts/`.
Tests are `tests/test_pl_*.py`; shared synthetic fixtures belong in
`tests/_synthetic.py`.

R2R records live under
`data/candidate_pool/<run>/records/r2r/main-r<round>/`. Compiled QA lives under
`data/candidate_pool/<run>/candidate_qa/`; the static case browser lives under
`data/candidate_pool/<run>/candidate_qa_report/`. Authoritative runs are
registered in `docs/runs.json`.

## Commands

Use the Habitat Python 3.9 environment:

```bash
PY="${EGOCONSEQ_HABITAT_PYTHON:-python}"
RUN=data/candidate_pool/<r2r-abc-run>
BENCH="$RUN/candidate_qa"
AUTH=/path/to/source-authority.json

$PY -m pytest -q
$PY -m pytest tests/test_pl_v16_candidate_preview.py -v
$PY scripts/check_abc_golden.py
$PY scripts/eval_benchmark.py --benchmark "$BENCH" \
  --source-authority-manifest "$AUTH" --gt-as-pred

# check_records needs an explicit validation authority; it refuses to guess.
SHARD="$RUN/records/r2r/main-r00"
$PY scripts/check_records.py "$SHARD/records.jsonl" \
  --run-meta "$SHARD/run_meta.json" \
  --expected-run-meta-sha256 "$(sha256sum "$SHARD/run_meta.json" | cut -d' ' -f1)"
```

`scripts/check_abc_golden.py` is the standard per-commit gate: it rebuilds the
golden artifact from the registered shards and exits non-zero if any output
digest, the task coverage, record validation, or the `--gt-as-pred` replay
changed. Run it before every commit that touches the pipeline.

`--gt-as-pred` only checks scorer integrity. Candidate artifacts always report
`headline_eligible=false`.

## Active Benchmark Contract

The benchmark tests consequences inside the initial visible space. Public input
is one initial RGB plus body radius, camera optical-center height, HFOV/VFOV, a
canonical action sequence, and a target when B requires one. Depth, semantics,
navmesh, endpoint pose, execution regime, and GT certificates are private.

The active taxonomy has exactly six tasks:

- A1: collision;
- A2: collision action index;
- A3: first-contact visible object/surface category;
- B1: endpoint distance to an initial-visible target;
- B2: endpoint direction of that target;
- C1: true future-view image selection.

Collision and safety require full/depth rollout consensus plus swept-corridor
coverage from the initial frame. D is deferred and has no active chain mode or
compatibility path. The legacy Q1--Q10 formal artifact stack is removed.

## Threat Model

This repository generates research ground truth in a single-user, local,
offline environment. Source datasets are read-only inputs the operator
downloaded; generated artifacts can be recomputed at any time; there is no
multi-tenancy, no concurrent writer, and no untrusted caller inside the
process.

Defend against exactly these:

- GT drifting away from the asset it was derived from;
- a corrupted, truncated, or substituted source file;
- the wrong source being bound to a record;
- canonicalization drift between two implementations of the same digest;
- behavioral regression in an oracle or validator.

Do not defend against a malicious concurrent process, a hostile filesystem,
or code executing inside this interpreter. Those threats are out of scope, so
`O_TMPFILE`/`linkat` publication protocols, repeated inode/stat re-reads,
`/proc/self/fd` re-resolution, and sealed containers that stop in-process
mutation are not warranted here. A single `sha256` assertion at the trust
boundary, plus `O_NOFOLLOW` on authoritative inputs, covers everything in
scope. Review comments proposing stronger defenses must first show the threat
falls inside this model.

## Coding Style

Use four spaces, `snake_case`, `CapWords`, and `UPPER_SNAKE_CASE`. Centralize
numeric gates in `pipeline/config.py`; never duplicate margins. Add type hints
to public APIs and document coordinate frames, units, and array shapes. Keep
changes PEP 8-compatible.

## Testing

Add focused deterministic tests before behavioral changes. Prefer synthetic
fixtures over Habitat for geometry and QA logic. Run targeted tests during
development and the full suite before handoff. Current records use `conseq.v11`
with oracle contract `ground-disc-visible-v8`; candidate QA uses
`egoconseq.qa.v16-candidate-preview`. Do not add obsolete-schema adapters.

## Commits

Use narrow Conventional Commit-style subjects such as `feat:`, `fix:`, and
`docs:`. Preserve unrelated user changes. Do not commit credentials or
unintended generated artifacts.
