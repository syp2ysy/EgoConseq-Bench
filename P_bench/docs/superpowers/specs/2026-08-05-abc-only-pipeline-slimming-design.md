# ABC-only Pipeline Slimming Completion Design

Date: 2026-08-05

Status: approved in the repository slimming review and the follow-up request
to continue implementation.

## Goal

Reduce the active pipeline to the six frozen ABC tasks without changing their
semantics:

- `A1_collision`
- `A2_collision_step_grounding`
- `A3_contact_object`
- `B1_endpoint_distance`
- `B2_endpoint_direction`
- `C1_future_view_selection`

D remains deferred.  Old Q-family data, continuation masks, generic goal and
projection data, review-only assets, and probe compatibility are not part of
the active record contract.

## Constraints

- R2R `main` is the authoritative collection path.
- Public inputs and the six task definitions remain unchanged.
- Full/depth rollout consensus, swept-corridor coverage, A3 visible-instance
  evidence, exact B target geometry, and C terminal PNG authority remain.
- Record schema changes from `conseq.v9` to `conseq.v10` with no v9 adapter.
- Candidate artifacts remain non-headline candidate previews.
- Authoritative inputs use one SHA-256 check at the trust boundary and
  `O_NOFOLLOW`; hostile concurrent mutation is out of scope.
- The existing tag `abc-only-checkpoint-2026-08-05` preserves the last D
  continuation-oracle implementation before its removal.

## Stage 1: Candidate artifact normalization

The public QA schema remains `egoconseq.qa.v16-candidate-preview` because no
public item, answer semantics, or evaluator contract changes.  The container
format becomes `egoconseq.candidate-preview.v2`, and source atoms become
`egoconseq.qa-source-atom.v17-preview`.

Each source atom stores the full outcome and a `record_sha256` reference.  A
new private `record_contexts.jsonl` stores the outcome-free record context
once per referenced record.  Its digest keys must be exactly the set used by
the emitted atoms: rejected records and records excluded by task caps must not
appear.  Rows and atoms are canonically ordered.  Missing, duplicate,
reordered, conflicting, or source-mismatched contexts fail validation.

The Golden gate compares the complete rebuilt file inventory, not only files
listed in the manifest.  An intentional artifact-version transition updates
the Golden digests without recollecting records; task coverage and
`--gt-as-pred` must remain unchanged.

## Stage 2: Threat-model and formal-source convergence

The deleted formal-source builder is not restored.  `source_manifest.py`
keeps only source/run authority and validation contexts used by active
candidate compilation and `check_records`.  Dead formal schema names and
comments are removed.

Authoritative readers retain:

- canonical containment checks;
- final-component `O_NOFOLLOW`;
- one regular-file check;
- one digest comparison against the authenticated expected value.

They remove repeated before/after inode snapshots, same-inode mutation
defences, symlink-target cache invalidation, and tests that require a hostile
concurrent writer.  Ordinary atomic temp-file replacement remains appropriate
for generated outputs.

## Stage 3: `conseq.v10`

The v10 producer stops computing and storing these v9-only branches:

- `terminal_options` and `start_terminal_options`;
- terminal/start safe masks in `future_state`;
- `object_consequences` and goal/geodesic evidence;
- `target_reference_sets`, `target_projection_keys`, and
  `target_projections`;
- `review_evidence` and legacy future-review WebP assets;
- Q4/sensor-consistency probes and their validators.

Active consumers read the realized endpoint directly from `execution`.
The active record retains collision execution, full/depth physical evidence,
oracle consensus and stability, A3 category evidence, the source-pinned B
target plus exact endpoint relation, and C future-view/terminal-PNG evidence.

The collector must not perform continuation rollouts after the realized
endpoint.  The validator requires the exact v10 shape and rejects missing or
extra retired branches without accepting v9.

## Recollection and Golden transition

Fresh R2R `main` shards are collected under v10 from the registered golden
scenes/settings.  Their `run_meta.json` files must contain the active
single-setting sampling policy and only R2R-relevant collection authority.
The new authoritative run is registered in `docs/runs.json`.

The v10 Golden freezes:

- every input record/run-meta/funnel digest;
- the complete candidate output inventory and digests;
- coverage for all six tasks;
- clean record validation;
- `--gt-as-pred` overall `1.0`.

No v9 compatibility code is retained after the new Golden passes.

## Verification

Each stage has focused deterministic tests before implementation changes.
Final acceptance requires:

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest -q
$PY -m compileall -q pipeline scripts
$PY scripts/check_abc_golden.py
```

Static scans must find no live Q-family, D continuation, formal-source builder,
or out-of-scope concurrent-mutation implementation in `pipeline/`, `scripts/`,
or active tests.
