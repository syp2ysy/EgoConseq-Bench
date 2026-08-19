# Pipeline Layout

Active R2R/B1K/GS ABC1 data flow:

```text
actions.py / action_sampling.py
  -> collection_runtime.py
  -> consequence.py + rollout.py + consensus.py
  -> semantic.py + objects.py + record.py
  -> benchmark_tasks.py + benchmark_builders.py (active v11 QA)
  -> validate.py (local integrity / source-bound replay)
  -> candidate_preview.py + gate_authority.py
```

- `action_proposal.py` builds each pose's v3 action bank from three explicit
  arms: boundary-paired actions for A2/A3/B, RGB-D-budgeted dynamic-natural
  actions for the main A1 population, and five rotating frozen controls for an
  exact cross-pose A1 slice. Paired and C1-neighbour actions cannot publish A1.
- `actions.py` owns the frozen 0.5 m / 15 degree action vocabulary.
- `sim.py`, `frame.py`, `perception.py`, `rollout.py` own Habitat and geometry.
- `consensus.py` owns the full/depth visible-space decision.
- `semantic.py` is the only MP3D PLY/house and exact instance geometry authority.
- `gs_collision.py` consumes the preprocessed official SAGE-3D collision USD
  as GS's physical rollout authority. Gaussian ellipsoids are diagnostic only
  and can neither publish nor veto a consequence.
- `gs_semantic.py` applies Habitat-GS's official fixed InteriorGS coordinate
  transform, then assigns only initial-visible RGB-D points to unique label
  boxes. Those observed points provide A3 category evidence and GS's explicit
  visible-anchor B protocol; bbox faces never masquerade as object surfaces.
- `record.py` persists hash-bound A/B/C atoms.
- `benchmark.py` contains the six active task IDs.
- `benchmark_tasks.py` decides eligibility and canonical answers.
- `benchmark_builders.py` formats Closed Exact questions and choices.
- `candidate_preview.py` compiles, validates, scores, and renders the candidate artifact.
- `a1_common_support.py` keeps legacy v2 common support read-only and applies
  the v3 split: natural A1 is balanced within `(length, 1 m forward bin)`;
  controls are balanced only when action, body, height, and FOV match exactly.
- `validate_record_local` / `validate_file_local` never load dataset sources;
  `validate_record_source_bound` / `validate_file_source_bound` authenticate the
  registered R2R/MP3D sources and rederive source facts.
- Candidate compilation verifies the registered records/run-metadata hashes
  once and performs local record validation. It never reopens raw dataset
  geometry or reruns a simulator.
- `gate_authority.py` resolves records and run metadata from an external
  source authority. `private/source_map.json` stores checkout-relative
  locators and hashes, never an absolute workspace path.
- Candidate-only C1 uses deterministic counterfactual neighbours of one clear
  query program. The true option is the query endpoint; three independently
  certified neighbour programs provide distinct terminal images at the same
  sensor profile. New terminal observations are allowed. This route never
  claims headline eligibility. R2R, B1K, and GS share this sole active C1
  candidate construction.
- The old directed matched/atomic implementation, its CLI, audits, gates, and
  tests are removed.

D/chain and the legacy Q1–Q10 artifact stack are intentionally absent. Do not recreate compatibility facades.

R2R and B1K use `conseq.v11`; GS uses `conseq.v18` because its official
collision binding and visible-anchor semantic atoms are source-distinct. The
active candidate compiler remains `egoconseq.qa.v16-candidate-preview`. Its
active three-dataset regression fixture is frozen by
`docs/golden/2026-08-14-r2r-gs-b1k-abc-golden.json`. Historical manifests in
`docs/golden/` are provenance only; `docs/golden/README.md` identifies the one
active per-commit gate.
C1 formal scientific readiness is still pending a separately preregistered
human pilot and held-out confirmation; counterfactual outputs are candidate
data only.
