# Multi-Dataset Case Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and serve one static browser containing R2R and BEHAVIOR-1K pilot QA with dataset and A1--C1 switching.

**Architecture:** Keep candidate artifacts immutable. Merge validated browser payloads in a new focused module, stage content-addressed images into the merged report, and reuse the existing review UI with an opt-in dataset switch rail so ordinary candidate reports keep their contract.

**Tech Stack:** Python 3.9, deterministic JSON/HTML, vanilla JavaScript, CSS, pytest, local HTTP server.

## Global Constraints

- `pipeline/` remains the only benchmark implementation.
- The merged browser is candidate-only and always `headline_eligible=false`.
- BEHAVIOR-1K remains labelled Pilot with C1=0.
- A/B remain open-text displays; only C1 renders four image choices.
- No candidate record, Task-GT, answer, or scorer changes.
- All external images are digest-checked and staged under the report root.
- Existing Golden candidate output must remain byte-identical.

---

### Task 1: Dataset payload merger and asset bundler

**Files:**
- Create: `pipeline/dataset_case_browser.py`
- Test: `tests/test_pl_dataset_case_browser.py`

**Interfaces:**
- Produces: `DatasetReport(dataset_id, label, report_path)` and `merge_dataset_reports(reports, output_root) -> Path`.
- Consumes: embedded `egoconseq.case-browser.v1` payloads and image paths relative to each input report.

- [ ] Write failing tests for two-dataset coverage, preserved source provenance, C1=0, duplicate IDs, and content-addressed image staging.
- [ ] Run the focused test and verify failures are caused by the missing module/API.
- [ ] Implement strict HTML payload extraction, validation, merge, and hard-link/copy staging.
- [ ] Run the focused tests and keep them green.

### Task 2: Opt-in dataset controls in the existing renderer

**Files:**
- Modify: `pipeline/case_browser.py`
- Test: `tests/test_pl_dataset_case_browser.py`

**Interfaces:**
- Produces: `render_case_browser_payload(payload, report_root, *, dataset_mode=False) -> Path`.
- Default `render_case_browser()` output must remain byte-identical.

- [ ] Write a failing HTML test requiring dataset buttons, a dataset-labelled select, live counts, and the unchanged A1--C1 tabs.
- [ ] Verify RED.
- [ ] Extract the current document renderer without changing default bytes; add dataset-only CSS/HTML/JavaScript extensions.
- [ ] Verify the new test and all existing case-browser tests pass.

### Task 3: Reproducible merger CLI

**Files:**
- Create: `scripts/build_dataset_case_browser.py`
- Test: `tests/test_pl_dataset_case_browser.py`

**Interfaces:**
- Consumes repeatable `--dataset DATASET_ID:LABEL=INDEX_HTML` and `--output`.
- Calls `merge_dataset_reports` and prints coverage plus dataset counts.

- [ ] Write failing parser and duplicate-dataset tests.
- [ ] Verify RED.
- [ ] Implement the CLI with strict slug parsing and deterministic ordering.
- [ ] Verify GREEN.

### Task 4: Produce and serve the combined artifact

**Files:**
- Generate: `data/candidate_pool/abc_c1_all_cases_review_20260809/candidate_qa_report/index.html`
- Generate: `data/candidate_pool/abc_c1_all_cases_review_20260809/candidate_qa_report/_assets/`

**Interfaces:**
- R2R input: the existing 2,021-case browser.
- B1K input: a B1K-adapter-validated intermediate browser containing canonical Ihlen and grocery-cafe previews.

- [ ] Produce the B1K intermediate report using the committed adapter validation path.
- [ ] Run the merger into a fresh temporary output and verify expected coverage R2R=2,021 and B1K=22.
- [ ] Publish the verified merged files into the existing report directory.
- [ ] Start the repository HTTP server on port 8789.

### Task 5: Verification

**Files:**
- Modify only the Golden manifest if and only if the ordinary single-source browser bytes intentionally changed; the implementation target is no change.

- [ ] Run `pytest tests/test_pl_dataset_case_browser.py tests/test_pl_case_browser.py -q`.
- [ ] Run full `pytest -q`.
- [ ] Run `scripts/check_abc_golden.py` and require 19 byte-identical files.
- [ ] Run `git diff --check`.
- [ ] Fetch the combined URL and run a headless-browser smoke for dataset controls, counts, task tabs, and image loads.

