# Benchmark Design Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the review-first benchmark home screen with a gallery-first
L/Q/variant explorer that supports large-scale browsing, comparison, and
persistent image/item shortlists without changing the benchmark artifact.

**Architecture:** Add a deterministic, simulator-free `ExplorerIndex` that
normalizes public item metadata, selects diverse representatives, filters, and
paginates. Both HTTP viewers expose the same index and use the existing
append-only review store for a separate authenticated curation event stream.
The dependency-free viewer keeps Human Review intact as a secondary mode and
adds a dense responsive Design Explorer as the default.

**Tech Stack:** Python 3.9 standard library, existing `ThreadingHTTPServer`,
vanilla HTML/CSS/JavaScript, pytest, headless Chrome.

---

## File Structure

- Create `pipeline/explorer.py`: deterministic catalog normalization,
  representative selection, filtering, pagination, detail records, and curation
  coverage summaries.
- Create `tests/test_pl_explorer.py`: pure unit tests for the index and query
  semantics.
- Modify `scripts/serve_benchmark.py`: curation persistence and explorer HTTP
  routes.
- Modify `scripts/serve_viz.py`: expose the same explorer and curation routes
  when it hosts `benchmark_viewer.html`.
- Modify `scripts/benchmark_viewer.html`: default Design Explorer, gallery,
  detail drawer, comparison, shortlist editor, and retained Human Review mode.
- Modify `tests/test_pl_serve_benchmark.py`: authenticated route and persistence
  tests.
- Modify `tests/test_pl_benchmark_ui.py`: static viewer contract tests.
- Modify `tests/test_pl_viz.py`: shared-viewer endpoint contract.

### Task 1: Deterministic Explorer Index

**Files:**
- Create: `pipeline/explorer.py`
- Create: `tests/test_pl_explorer.py`

- [ ] **Step 1: Write failing normalization and summary tests**

Create synthetic Q1/Q4/Q9 items spanning levels, variants, backends, images,
groups, and scenes. Assert that:

```python
index = ExplorerIndex(items, answers)
summary = index.summary(include_private=False)
assert summary["totals"] == {
    "items": 6, "images": 4, "groups": 3, "scenes": 3}
assert summary["levels"]["L2"]["families"]["Q4"]["variants"][
    "radius_counterfactual"]["items"] == 2
assert "evidence_counts" not in summary
```

Also assert every registered `pipeline.benchmark.QUESTION_SPECS` entry has a
design contract with `construct`, `public_inputs`, `physical_gt`,
`evidence_head`, `answer_shape`, `eligibility`, and `anti_shortcut`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
$PY -m pytest tests/test_pl_explorer.py -q
```

Expected: collection fails because `pipeline.explorer` does not exist.

- [ ] **Step 3: Implement normalization, catalog summary, and design contracts**

Expose the exact Python 3.9-compatible call forms exercised by the tests:

```python
index = ExplorerIndex(items, answers)
public_summary = index.summary(include_private=False)
private_summary = index.summary(
    include_private=True, curations=curation_records)
```

Normalize level from the registered family taxonomy, retain the artifact order,
compute unique image/group/scene counts, and reject duplicate or missing item
IDs. Keep private answer/evidence fields out unless `include_private=True`.

- [ ] **Step 4: Run the summary tests and verify GREEN**

Run the same targeted command. Expected: all summary tests pass.

- [ ] **Step 5: Write failing filter, pagination, and representative tests**

Cover combined filters for level/family/variant/backend/format/scene, numeric
ranges, evidence/GT in private mode, and selection/tag state. Assert:

```python
page = index.query(
    {"family": "Q1", "backend": "hm3d", "page": "1", "page_size": "24"},
    include_private=False)
assert page["page"] == 1
assert page["page_size"] == 24
assert page["total"] == 2
assert all("answer" not in case for case in page["cases"])
```

Build more than five cases for one variant and assert representative mode returns
exactly five, deterministically, while covering multiple backends/scenes when
available.

- [ ] **Step 6: Verify RED, then implement query and representative selection**

The representative selector greedily maximizes unseen values for backend,
scene, evidence, GT label, body radius, and sensor profile; ties use a stable
item-ID hash. Implement one-based pagination with page sizes restricted to
`24`, `32`, and `48`, plus these exact query and detail call forms:

```python
page = index.query(
    params, include_private=False, curations=curation_records)
detail = index.detail(
    "item-id", include_private=True, curations=curation_records)
assert detail is None or isinstance(detail, dict)
```

- [ ] **Step 7: Run all explorer tests and commit**

```bash
$PY -m pytest tests/test_pl_explorer.py -q
git add pipeline/explorer.py tests/test_pl_explorer.py
git commit -m "feat: index benchmark cases for design exploration"
```

### Task 2: Append-Only Curation Store

**Files:**
- Modify: `scripts/serve_benchmark.py`
- Test: `tests/test_pl_serve_benchmark.py`

- [ ] **Step 1: Write failing store tests**

Add tests that persist image and item events, reduce to latest state per
`(scope, target_id, reviewer)`, preserve history, and reject:

```python
{"scope": "image", "target_id": "", "selected": True}
{"scope": "case", "target_id": "i", "selected": True}
{"scope": "item", "target_id": "i", "selected": 1}
{"scope": "item", "target_id": "i", "selected": True, "tags": "good"}
```

Assert tags are deduplicated in order and timestamps are generated server-side.

- [ ] **Step 2: Run the tests and verify RED**

```bash
$PY -m pytest tests/test_pl_serve_benchmark.py -q
```

Expected: `ReviewStore` has no curation methods.

- [ ] **Step 3: Implement curation persistence**

Add `curations.jsonl` and expose these exact call forms:

```python
stored = store.append_curation(curation_event)
history = store.curation_events()
latest = store.latest_curations(reviewer="alice", selected=True)
```

Enforce exact booleans, `scope in {"image", "item"}`, nonempty bounded IDs and
reviewers, at most 20 string tags of at most 64 characters, and a note of at
most 4,000 characters.

- [ ] **Step 4: Run the tests and commit**

```bash
$PY -m pytest tests/test_pl_serve_benchmark.py -q
git add scripts/serve_benchmark.py tests/test_pl_serve_benchmark.py
git commit -m "feat: persist benchmark image and item curations"
```

### Task 3: Explorer And Curation HTTP Routes

**Files:**
- Modify: `scripts/serve_benchmark.py`
- Modify: `scripts/serve_viz.py`
- Test: `tests/test_pl_serve_benchmark.py`
- Test: `tests/test_pl_viz.py`

- [ ] **Step 1: Write failing public explorer route tests**

Assert `/api/explorer/summary`, `/api/explorer/cases`, and
`/api/explorer/case?id=i` work without review authentication but contain neither
`structured_answer` nor `evidence_status`. Assert invalid page sizes and unknown
IDs return 400/404 with stable JSON errors.

- [ ] **Step 2: Write failing private curation route tests**

Assert locked/public requests to `/api/curations` and
`/api/curations/export` fail, while an authenticated session can:

```python
POST /api/curations
GET  /api/curations?reviewer=alice
GET  /api/curations/export?reviewer=alice
```

The POST must reject target IDs that are not present in the loaded artifact.
The export response must be `application/x-ndjson`, selected-only by default,
and include `Content-Disposition`.

- [ ] **Step 3: Run route tests and verify RED**

```bash
$PY -m pytest tests/test_pl_serve_benchmark.py tests/test_pl_viz.py -q
```

Expected: new routes return 404.

- [ ] **Step 4: Implement shared route behavior**

Instantiate `ExplorerIndex` once in each handler factory. Public routes pass
`include_private=False`; authenticated requests pass `True`. Return private
curation state only after cookie authentication. `GET /api/curations` returns:

```json
{"records": [], "coverage": {"total": 0, "by_scope": {}, "by_family": {}}}
```

Update both server implementations so `/explorer` does not depend on which CLI
is hosting the shared page.

- [ ] **Step 5: Run route tests and commit**

```bash
$PY -m pytest tests/test_pl_serve_benchmark.py tests/test_pl_viz.py -q
git add scripts/serve_benchmark.py scripts/serve_viz.py \
  tests/test_pl_serve_benchmark.py tests/test_pl_viz.py
git commit -m "feat: expose paginated benchmark explorer routes"
```

### Task 4: Gallery-First Viewer

**Files:**
- Modify: `scripts/benchmark_viewer.html`
- Modify: `tests/test_pl_benchmark_ui.py`

- [ ] **Step 1: Write failing static UI contract tests**

Require:

- mode tabs with Design Explorer selected by default;
- persistent taxonomy and variant controls;
- contract band;
- representative/all/selected/compare controls;
- 24/32/48 page-size control and pagination;
- image-level and item-level shortlist buttons;
- detail drawer and comparison region;
- `/api/explorer/*` and `/api/curations` calls;
- the existing Human Review controls and answer flow.

- [ ] **Step 2: Run UI tests and verify RED**

```bash
$PY -m pytest tests/test_pl_benchmark_ui.py -q
```

Expected: new Explorer markers are absent.

- [ ] **Step 3: Implement the responsive Design Explorer shell**

Add top-level mode tabs, keep the existing review shell hidden initially, and
build:

```html
<main id="designExplorer">
  <aside id="taxonomyRail"></aside>
  <section>
    <header id="designContract"></header>
    <form id="galleryFilters"></form>
    <div id="caseGallery"></div>
    <nav id="galleryPagination"></nav>
  </section>
</main>
<aside id="caseDrawer" role="dialog" aria-modal="true"></aside>
<section id="compareWorkspace" hidden></section>
```

Use a four/two/one-column 4:3 gallery at desktop/tablet/mobile widths. Cards
show question design and metrics without nested card containers.

- [ ] **Step 4: Implement gallery state and queries**

Load capabilities, manifest, summary, and the first representative page. Filter
changes reset the page and refetch `/api/explorer/cases`. Render loading, empty,
error, and retry states. Lazy-load the existing `/api/items` Human Review data
only when that mode is opened.

- [ ] **Step 5: Implement detail, compare, and curation flows**

The drawer fetches `/api/explorer/case`. Compare accepts two to four cases and
aligns image/prompt/actions/body/sensor/GT/evidence. Authenticated image/item
shortlist controls POST curation events and retain note/tag editor state on
failure. Add latest-state JSONL export.

- [ ] **Step 6: Run UI and server tests, then commit**

```bash
$PY -m pytest tests/test_pl_benchmark_ui.py \
  tests/test_pl_serve_benchmark.py tests/test_pl_viz.py -q
git add scripts/benchmark_viewer.html tests/test_pl_benchmark_ui.py
git commit -m "feat: make case gallery the benchmark explorer home"
```

### Task 5: Real Artifact And Responsive Validation

**Files:**
- Modify only if validation finds a defect:
  `scripts/benchmark_viewer.html`, `pipeline/explorer.py`,
  `scripts/serve_benchmark.py`
- Test corresponding changed modules.

- [ ] **Step 1: Run targeted and full automated tests**

```bash
$PY -m pytest tests/test_pl_explorer.py tests/test_pl_benchmark_ui.py \
  tests/test_pl_serve_benchmark.py tests/test_pl_viz.py -q
$PY -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Start the real enriched combined artifact**

Run `scripts/serve_benchmark.py` in review mode against:

```text
data/candidate_pool/preview_v6_20260725_d9e37486a696/
candidate_qa_combined
```

Use its explicit private directory and the existing local review-token
mechanism. Confirm manifest count is 7,254.

- [ ] **Step 3: Validate API behavior on the real artifact**

Check representative counts, all-case pagination, private-field isolation,
authenticated curation persistence, selected-only filtering, and JSONL export.

- [ ] **Step 4: Capture and inspect responsive screenshots**

Use headless Chrome or Playwright at:

- 1440×1000
- 900×1000
- 390×844

Inspect Design Explorer, drawer, compare, Human Review, empty state, and selected
state. Confirm nonblank images, no overlap, no clipped controls, stable 4:3 card
geometry, and readable longest labels.

- [ ] **Step 5: Re-run affected tests after visual fixes**

Any discovered defect first receives a failing regression test, followed by the
minimal fix and the targeted/full verification commands.

- [ ] **Step 6: Commit validation fixes and report the live URL**

```bash
git add scripts/benchmark_viewer.html pipeline/explorer.py \
  scripts/serve_benchmark.py scripts/serve_viz.py \
  tests/test_pl_explorer.py tests/test_pl_benchmark_ui.py \
  tests/test_pl_serve_benchmark.py tests/test_pl_viz.py
git commit -m "fix: polish benchmark explorer responsive states"
```

Report the local URL, changed files, exact test count, and any validation that
could not be run.
