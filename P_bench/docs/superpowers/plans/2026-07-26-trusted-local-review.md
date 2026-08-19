# Trusted Local Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a loopback-only benchmark explorer expose private GT and accept curation writes without showing or requiring the Review access login.

**Architecture:** Add an explicit `--trusted-local-review` CLI mode beside the existing authenticated `--review-mode`. The handler treats trusted-local requests as already authenticated, while argument validation prevents binding that mode to a non-loopback address. Existing public and token-authenticated modes retain their current behavior.

**Tech Stack:** Python 3.9, `http.server`, argparse, pytest.

---

### Task 1: Freeze The Trusted-Local Access Contract

**Files:**
- Modify: `tests/test_pl_serve_benchmark.py`

- [ ] **Step 1: Write failing handler tests**

Add a test that creates a handler with:

```python
handler = make_handler(
    tmp_path, _artifact(tmp_path), ReviewStore(tmp_path / "reviews"),
    review_mode=True, trusted_local_review=True)
```

Without a cookie, assert:

```python
GET /api/capabilities
== {"mode": "review", "authenticated": True}
GET /api/explorer/case?id=i
contains "private"
POST /api/curations
returns 200
```

- [ ] **Step 2: Write failing CLI tests**

Assert `parse_args` accepts:

```bash
--benchmark artifact
--trusted-local-review
--private-dir artifact/private
```

without `EGOCONSEQ_REVIEW_TOKEN`, and rejects the same mode when `--host
0.0.0.0` or when `--private-dir` is absent.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest tests/test_pl_serve_benchmark.py -q
```

Expected: failures because `trusted_local_review` and
`--trusted-local-review` do not yet exist.

### Task 2: Implement Trusted-Local Review

**Files:**
- Modify: `scripts/serve_benchmark.py`

- [ ] **Step 1: Extend the handler**

Add `trusted_local_review=False` to `make_handler`. Define unlocked review
access as:

```python
def _authenticated(self):
    return trusted_local_review or auth.authenticated(
        self.headers.get("Cookie", ""))
```

Return review capabilities using that same predicate so the UI never renders
the login control in trusted-local mode.

- [ ] **Step 2: Extend CLI validation**

Add:

```python
--trusted-local-review
```

The mode requires `--private-dir`, requires a loopback host, and does not require
`EGOCONSEQ_REVIEW_TOKEN`. It is mutually exclusive with `--review-mode`.

- [ ] **Step 3: Wire artifact loading and the handler**

Use `review_enabled = args.review_mode or args.trusted_local_review` for private
artifact loading and review-store setup. Pass both `review_mode=review_enabled`
and `trusted_local_review=args.trusted_local_review` to `make_handler`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
$PY -m pytest tests/test_pl_serve_benchmark.py -q
$PY -m pytest -q
```

Expected: all tests pass.

### Task 3: Switch And Verify The Live Explorer

**Files:**
- No source changes.

- [ ] **Step 1: Restart the server**

Run without a token:

```bash
$PY scripts/serve_benchmark.py \
  --host 127.0.0.1 --port 8768 \
  --benchmark data/candidate_pool/preview_v6_20260725_d9e37486a696/candidate_qa_combined \
  --trusted-local-review \
  --private-dir data/candidate_pool/preview_v6_20260725_d9e37486a696/candidate_qa_combined/private \
  --reviews data/candidate_pool/preview_v6_20260725_d9e37486a696/reviews
```

- [ ] **Step 2: Verify live access**

Without cookies or authorization headers, assert:

```bash
curl -s http://127.0.0.1:8768/api/capabilities
curl -s 'http://127.0.0.1:8768/api/explorer/case?id=<known-id>'
```

Expected: authenticated review capability and private GT in the case response.
Post and then read back one curation using an existing case ID to verify writes;
do not alter artifact data.

- [ ] **Step 3: Commit**

```bash
git add scripts/serve_benchmark.py tests/test_pl_serve_benchmark.py \
  docs/superpowers/plans/2026-07-26-trusted-local-review.md
git commit -m "feat: add trusted local benchmark review"
```
