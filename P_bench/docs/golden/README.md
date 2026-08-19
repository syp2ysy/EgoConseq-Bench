# ABC Golden manifests

The only active per-commit gate is
`2026-08-14-r2r-gs-b1k-abc-golden.json`. It authenticates one compact shard
from each supported dataset and covers all six ABC1 tasks.

All other JSON manifests in this directory are historical provenance. Some
older filenames predate the `.superseded.json` naming convention; their
`superseded_by` metadata, rather than the filename, is authoritative. They are
retained to make protocol transitions auditable and must not be used as active
collection or publication inputs.

The active R2R shard is intentionally small and contributes no A1 items. A1 is
covered by the GS and B1K shards; the Golden checks aggregate six-task coverage
across all three authenticated sources.
