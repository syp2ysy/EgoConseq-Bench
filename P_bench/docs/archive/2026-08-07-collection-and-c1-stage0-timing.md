# Collection and C1 Stage-0 timing evidence

Status: archived execution evidence. These measurements are diagnostic, not
benchmark gates. Raw profiles and generated collector outputs were removed in
the 2026-08-09 repository freeze; the recovery snapshot remains at
`/tmp/pbench_backup_0409.tgz` on the originating machine.

## Collector performance

The controlled single-record run on `PuKPg4mmafe` took 49.62 s with the
inherited 32-thread OpenBLAS pool and 40.88 s with numerical libraries pinned
to one thread. The record bytes were identical
(`d5b384fe2c86989f09e77caabe390910fabdd00ab491c199b4ff4c209e7c6d56`).
Four independent pinned workers completed in 42.58--44.49 s; their outputs
were mutually byte-identical
(`fd59bb478b940210aead373d0674e422d138d98eecfcf3c1d0df7ae8054ddf07`).
This corresponds to about 3.68x throughput over one pinned worker and 4.46x
over the original oversubscribed single worker.

The instrumented one-record cProfile run took 45.47 s. Its largest cumulative
components were A3 complete-face attribution (17.52 s, 38.5%), full-frame
semantic assignment (14.55 s, 32.0%), and simulator initialization (7.88 s,
17.3%). Exact AABB pruning and pose-batched contact queries preserved the four
v12 golden-shard `records.jsonl` digests. On the interrupted natural pilot,
1,113 exact requests at 485 unique points across 63 records and 11 scenes had
zero field-level mismatches; measured face-work reduction was 12.78x minimum,
55.19x median, and 160.24x maximum.

Frozen operating guidance from the measurement was four disjoint scene
shards, one output directory per shard, numerical thread pools pinned to one,
eight poses per scene, and eight semantic query workers. This changes only
execution; the seven perturbations and exact oracle remain unchanged.

## Directed C1 Stage-0 grid

The pose-pool gate covered 24 scenes, 23 complete scenes, and 498 accepted
initial views in 612.9 s. Nine runs crossed match-cell preset
`{loose, medium, tight}` with proposal budget `{24, 48, 96}`:

| Preset | Budget 24 | Budget 48 | Budget 96 |
| --- | ---: | ---: | ---: |
| loose | 463.4 s / 0 records | 496.6 s / 0 records | 586.0 s / 5 records |
| medium | 464.4 s / 0 records | 501.4 s / 0 records | 574.8 s / 0 records |
| tight | 462.8 s / 0 records | 500.1 s / 0 records | 602.1 s / 0 records |

Every emitted record validated with zero violations. Only `loose/budget-96`
produced formal output: five records and two capped formally safe families.
This sparse pilot is why directed C1 scientific readiness remains pending;
the structural implementation is retained for regression and capacity work,
but these measurements do not authorize a publication gate.
