# EgoConseq-Bench Design Explorer

**Status:** Approved interaction design, ready for implementation planning.

## Purpose

The benchmark viewer must make the question design legible before it asks anyone
to score answers. Its default view is a dense catalog of real cases organized by
reasoning level, question family, and variant. Human response collection remains
available as a secondary mode.

The primary users are benchmark authors selecting representative images,
auditing question construction, and comparing cases across backends and sensor
profiles. The explorer must handle the current 7,254-item artifact without
forcing the user through one case at a time.

## Design Direction

- **Tone:** quiet, technical, and evidence-first.
- **Layout:** persistent taxonomy rail, compact filter band, high-density case
  gallery, and an on-demand detail drawer.
- **Typography:** the existing system sans-serif for scanning; monospace only
  for metric values, action primitives, identifiers, and structured GT.
- **Surfaces:** white and neutral gray working surfaces with restrained blue,
  green, amber, and red semantic accents.
- **Motion:** limited to drawer transitions, filter feedback, and selection
  state; no decorative animation.
- **Signature differentiator:** every gallery card carries a compact
  level/family header and a two-part physics/evidence status strip, making the
  benchmark taxonomy and its two-layer label contract visible while scanning.

The interface remains an operational research tool. It does not use a landing
page, oversized hero text, decorative sections, or cards nested inside cards.

## Information Architecture

The top-level interface has two modes:

1. **Design Explorer** is the default. It supports taxonomy inspection,
   large-scale case browsing, comparison, and curation.
2. **Human Review** preserves the existing answer, confidence, reveal, oracle,
   and accept/flag/reject workflow.

The Design Explorer has four stable regions:

1. **Taxonomy rail:** `L0` through `L4`, their Q families, and variants. Each
   row reports item, unique-image, group, scene, and backend counts.
2. **Design contract band:** describes the selected family/variant's construct,
   public inputs, physical GT, evidence head, answer shape, eligibility rule,
   and anti-shortcut expectation.
3. **Filter and view toolbar:** controls gallery population, sorting, view mode,
   and page size.
4. **Case gallery:** displays many real cases simultaneously. Selecting a card
   opens a detail drawer without discarding gallery position.

## Gallery Behavior

The gallery supports four views:

- **Representative:** the default. It shows up to five deterministic,
  diversity-oriented cases per selected variant. Representatives should vary
  backend, scene, evidence state, answer label, body radius, and sensor profile
  when the available population permits it.
- **All cases:** server-paginated browsing with 32 cases per page by default and
  options for 24, 32, or 48.
- **Selected:** only image-level or item-level shortlist entries.
- **Compare:** two to four explicitly checked cases in aligned columns.

Desktop uses four gallery columns when space permits, tablet uses two, and mobile
uses one. Cards use a stable 4:3 image region so metadata never shifts the grid.
The user can continue through all matching pages; the representative limit never
caps the available catalog.

Each case card shows:

- RGB observation;
- level, family, variant, backend, and format;
- scene and intervention-group identifiers in compact form;
- body radius when applicable, effective camera height, HFOV, and VFOV;
- action summary and primitive count;
- question text and compact option summary;
- authenticated image/item selection state;
- authenticated evidence status and GT label;
- image-level and item-level shortlist controls.

Question text is clamped in the grid and available in full in the drawer.
Metadata chips wrap without resizing the image region.

## Filters And Sorting

The server query accepts:

- level, family, and variant;
- backend and scene;
- closed/open format;
- evidence status when authenticated;
- body-radius range;
- effective camera-height range;
- HFOV and VFOV;
- action primitive count and total forward-distance range;
- GT answer label when authenticated;
- image/item selection state;
- curation tag.

Sort modes are:

- representative diversity;
- artifact order;
- scene then image;
- action length;
- body radius;
- recently curated.

Unsupported or private-only filters are disabled with a direct explanation.
Changing a filter resets pagination and clears only transient compare selection,
not the persisted shortlist.

## Case Detail And Comparison

Clicking a card opens a right-side drawer containing:

- full-resolution public image;
- complete prompt, actions, options, and public model input;
- authenticated structured answer and evidence details;
- physical/evidence/publication eligibility summary;
- sibling cases sharing the same image;
- the selected family/variant design contract;
- image-level and item-level curation editors.

The compare view aligns image, prompt, actions, body/sensor inputs, GT, evidence,
and curation notes for two to four cases. Differences are highlighted by field,
but unchanged values remain visible so causal siblings can be audited.

## Persistent Curation

Curation is independent of human answers, reviews, and the immutable benchmark
artifact.

Two selection scopes are supported:

- **Image selection** identifies an RGB observation suitable for inspection,
  publication, or later question curation.
- **Item selection** identifies a specific Q/variant case whose construction is
  suitable.

Each curation event contains:

```json
{
  "scope": "image",
  "target_id": "public/image/path.jpg",
  "selected": true,
  "tags": ["good_visual", "clear_geometry", "paper_example"],
  "note": "Action path and obstacle boundary are easy to inspect.",
  "reviewer": "name",
  "updated_at": "server-generated UTC timestamp"
}
```

For item selections, `scope` is `item` and `target_id` is the public item ID.
Events are append-only. The server presents the latest event for each
`(scope, target_id, reviewer)` key, preserving an audit trail while supporting
deselection.

The Selected view defaults to the reviewer name currently entered in the
topbar. An authenticated user may switch to an all-curators view for coverage
inspection; saving always writes the explicit current reviewer.

Suggested tags include:

- `good_visual`
- `clear_geometry`
- `good_body_counterfactual`
- `good_action_consequence`
- `paper_example`
- `ambiguous`
- `bad_crop`
- `render_quality`
- `shortcut_risk`
- `incorrect_gt`

Tags remain extensible strings so curation does not require schema changes.
The server validates field types, maximum lengths, and allowed scope.

The explorer exports the current latest-state selections as JSONL. The export
includes both selected and explicitly deselected events only when audit-history
export is requested; the default export contains selected latest-state records.
It also exposes a compact coverage summary by level, family, variant, backend,
scope, and tag.

## Server Architecture

The existing `serve_benchmark.py` process remains the only service. The loaded
artifact is indexed once at startup by a focused `ExplorerIndex`:

- normalized level/family/variant fields;
- image-to-items and scene-to-items maps;
- precomputed public filter values;
- authenticated answer/evidence lookups;
- deterministic representative ranking.

New read endpoints:

- `GET /api/explorer/summary`
- `GET /api/explorer/cases`
- `GET /api/explorer/case?id=...`
- `GET /api/curations`
- `GET /api/curations/export`

New write endpoint:

- `POST /api/curations`

`/api/explorer/cases` uses one-based page numbers and returns a bounded page plus
`page`, `page_size`, `total`, and `total_pages`. It never returns private answer
fields without an authenticated review session. Existing `/api/items` remains
for backward compatibility with the Human Review mode during this change.

The curation store is a separate append-only `curations.jsonl` beside the
existing response/review files. Writes use the existing server-side locked
append pattern. Curation reads and writes require review authentication; public
mode can browse the public gallery but cannot see or modify shortlist state.

All declared artifact files continue to pass through
`pipeline.validate.assert_artifact_contract`. No legacy schema adapter is added.

## Frontend Boundaries

The viewer remains dependency-free HTML, CSS, and JavaScript. To keep the page
understandable, behavior is divided into explicit modules within the script:

- API and session state;
- taxonomy/summary rendering;
- filter and pagination state;
- gallery/card rendering;
- detail and comparison rendering;
- curation state and export;
- existing Human Review behavior.

The existing item-review workflow is retained rather than reimplemented. Its
state is mounted only when the Human Review mode is active.

## Error And Empty States

- Artifact or contract failure: replace the workspace with the server error and
  expose no partial catalog.
- Empty filter result: show the active filters and a single reset command.
- Page fetch failure: keep the existing gallery visible and provide retry.
- Missing image: preserve card dimensions and mark the asset unavailable.
- Authentication expiry: keep public browsing active, hide private fields, and
  request unlock before saving curation.
- Curation write failure: retain the editor contents and show the server error.
- Export with no selections: return a valid empty JSONL file and zeroed summary.

## Accessibility And Responsiveness

- Taxonomy, mode, and view controls are keyboard-operable buttons or tabs.
- Gallery cards do not rely on color alone for status.
- Drawer focus is moved on open and restored on close.
- Compare selection exposes count and disabled reasons.
- Images have case-specific alt text.
- All controls retain visible focus styles and at least 40-pixel touch targets on
  mobile.
- Text wraps safely at 320-pixel viewport width without horizontal page scroll.

## Verification

### Python/API tests

- summary counts across L/Q/variant/image/scene/backend;
- deterministic representative selection and five-per-variant cap;
- bounded pagination and combined filters;
- private evidence and GT never appear without authentication;
- valid image and item curation events persist;
- invalid scope, value type, or oversized note is rejected;
- latest-state reduction and deselection;
- selected-only and tag filters;
- JSONL export and coverage summary;
- existing response/review endpoints remain unchanged.

### Viewer contract tests

- Design Explorer is the default and Human Review remains reachable;
- taxonomy and contract band exist;
- representative/all/selected/compare views exist;
- gallery includes image-level and item-level controls;
- pagination and filter controls call the explorer endpoints;
- authenticated curation calls are separate from review calls;
- accessible dialog/drawer labels and responsive CSS are present.

### Browser validation

Run the real enriched artifact and verify with Playwright or headless Chrome:

- desktop at 1440×1000;
- tablet at 900×1000;
- mobile at 390×844;
- representative, all-cases, selected, compare, drawer, empty, locked, and
  authenticated states;
- no overlap, clipped text, blank image grid, or unintended layout shift;
- selection persists after reload and JSONL export downloads correctly.

## Non-Goals

- Curation never edits items, answers, manifests, or release gates.
- The explorer does not select benchmark samples automatically.
- This change does not add model scoring, human aggregate analytics, or new QA
  generation behavior.
- No frontend framework, database, or external service is introduced.
