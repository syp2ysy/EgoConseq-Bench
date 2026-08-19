# Multi-Dataset Case Browser Design

## Goal

Publish one static, locally served review page containing every available R2R
and BEHAVIOR-1K pilot QA case. Reviewers must be able to switch datasets and
then inspect A1, A2, A3, B1, B2, or C1 without opening another page.

## Scope and truthfulness

- R2R contains the existing 2,021 review cases: 1,779 A/B cases and 242 C1
  calibration-only cases.
- BEHAVIOR-1K is explicitly labelled **BEHAVIOR-1K Pilot**. Its two canonical
  source-validated previews contain 22 cases: A1=11, A2=1, A3=1, B1=5,
  B2=4, C1=0.
- The BEHAVIOR-1K C1 count remains zero and is displayed as unavailable; the
  browser never upgrades pilot data to formal or headline-eligible status.
- This is a read-only review projection. It does not change records, Task-GT,
  candidate QA, answers, or scoring.

## Chosen architecture

Use the existing static case-browser artifact as the interchange format. A
new merger reads one already generated case-browser report per dataset,
validates its embedded `egoconseq.case-browser.v1` payload and every referenced
image digest, relabels the source dimension as the dataset dimension, and
writes one merged report.

This avoids weakening candidate validation and avoids recompiling the older
R2R candidate artifacts. BEHAVIOR-1K candidate artifacts are validated by the
committed B1K adapter before their intermediate browser report is produced.

The merged report stages every referenced image under its own content-addressed
`_assets/<sha256>.png` directory. It attempts a hard link first and falls back
to a byte copy across filesystems. This makes a single HTTP document root
sufficient even though BEHAVIOR-1K source assets live outside the repository.

## Interaction design

The visual direction is a compact scientific review console, preserving the
existing dark header, task-color system, evidence cards, drawer, and image
lightbox. Its signature control is a prominent dataset switch rail above the
task tabs:

- All datasets;
- R2R, with live case count;
- BEHAVIOR-1K Pilot, with live case count and pilot status.

The existing source select becomes a dataset select in merged mode. Dataset
buttons drive that select, so all existing filtering, pagination, live task
counts, search, GT visibility, and URL state remain deterministic. Each case
retains its original artifact source inside technical metadata for audit.

When BEHAVIOR-1K Pilot is active, the C1 task tab remains visible with count
zero. This is intentionally different from hiding the task.

## Data flow

1. Produce the existing R2R browser report.
2. Validate the canonical Ihlen and grocery-cafe candidate artifacts with the
   B1K adapter and produce one intermediate B1K browser report.
3. Invoke the multi-dataset merger with stable dataset identifiers and labels.
4. Validate payload structure, unique case IDs, coverage, and every image
   SHA-256 while staging assets.
5. Write `index.html` plus `_assets/` in the existing combined report directory.
6. Serve the repository root on port 8789.

## Failure handling

- Unknown browser schema, malformed dataset identifier, duplicate dataset ID,
  duplicate case ID, missing image, or image digest mismatch aborts the build.
- Existing output assets are reused only when their digest matches their
  content-addressed name.
- BEHAVIOR-1K C1=0 is a valid visible state, not a build failure.
- No source artifact is modified.

## Verification

- Focused tests cover dataset merging, source provenance preservation, zero C1
  coverage, duplicate rejection, asset staging, corruption rejection, CLI
  parsing, and dataset controls in HTML.
- The existing case-browser tests must remain green.
- Full `pytest -q`, `scripts/check_abc_golden.py`, and `git diff --check` run
  before handoff.
- A local HTTP and headless-browser smoke checks the combined counts, dataset
  controls, task navigation, and successful image loading.

