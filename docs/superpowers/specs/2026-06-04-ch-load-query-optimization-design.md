# CH Load + Query Optimization — Design

**Date:** 2026-06-04
**Branch:** tc/ch-streaming-load
**Status:** Approved

## Goal

Evidence-driven optimization of two tracks, profiled at 100M points (scales
linearly so far), final verification at 3B:

1. **Load track** — single-pass load (`LoadFromCHSinglePass`): at 3B the
   morton+rowNumber INSERT is 58 min (83%) and the rollup batch populate is
   10.7 min (15%) of a 69.6 min load.
2. **Query track** — z14 city-viewport query: 3984 ms / 993 MB alloc per op
   at 3B (z2 = 3.4 ms, z8 = 17.5 ms are fine).

Success bar: profile first, fix what the profiles say, keep every win that
does not break the 3B memory ceiling. No fixed numeric targets. Final 3B
re-verify run at the end.

## Instrumentation harness (Phase 0)

Built once, reusable:

- **Query tagging:** heavy statements get `log_comment` values
  (`sp-insert`, `sp-leafbounds`, `rollup-z{N}`, `q-leafpoints`) so they are
  identifiable in CH system tables. Gated by env `CLUSTOPHER_CH_PROFILE=1`;
  the production path is unchanged when unset.
- **Per-query profiling settings** (same gate):
  - `log_processors_profiles = 1` — per-operator elapsed/rows in
    `system.processors_profile_log`; splits the INSERT pipeline into
    read → sort → spill-merge → rowNumber → MergeTree sink.
  - `query_profiler_real_time_period_ns = 10ms`,
    `query_profiler_cpu_time_period_ns = 10ms` — sampled stacks into
    `system.trace_log`.
- **Analysis queries** saved as `.sql` files (run post-bench):
  - `system.query_log` — wall, `memory_usage`, `read_rows/bytes`,
    ProfileEvents (`ExternalSortWritePart`, `ExternalSortMerge`, disk I/O)
    per tagged query.
  - `system.processors_profile_log` — per-operator `elapsed_us` / rows.
    The key table: distinguishes sort vs spill vs single-threaded
    `rowNumberInAllBlocks` merge vs MergeTree sink cost.
  - `system.trace_log` — top stacks aggregated.
- **Go side (query track):** pprof CPU + heap around the z14 bench loop;
  CH side of `fetchLeafPoints` via the same tagging.
- **Per-zoom rollup timing:** `populateRollupsBatch` currently logs all 9
  zooms as one lump; add per-zoom timing.

## Baseline capture @ 100M (Phase 1)

One instrumented run: single-pass load + z2/z8/z14 query bench. Output: an
evidence report ranking time sinks, saved to `benchmark_results/`.

## Fix loop (Phase 2)

Top finding → change → re-run 100M → keep or revert. One change at a time.

Candidate hypotheses going in (evidence reorders or kills them):

| # | Hypothesis | Track |
|---|------------|-------|
| H1 | Rollup cascade z10→z2 kills 8 of 9 full scans. `tile(z-1) = intDiv(tile(z), 2)` is exact because `floor(floor(v)/2) == floor(v/2)`; all rollup columns are additive. | Load |
| H2 | INSERT dominated by external-sort spill merge passes + single-threaded `rowNumberInAllBlocks` final merge. | Load |
| H3 | Pre-sorted staging (morton key materialized into ORDER BY) eliminates the runtime sort. Schema migration — only if profiling proves sort dominates AND settings cannot fix it. | Load |
| H4 | Spill/thread settings suboptimal (1 GB external-sort threshold is very aggressive). | Load |
| H5 | z14 time split between CH fetch volume (Map columns?) and Go-side clustering; 993 MB alloc/op says Go allocations matter. | Query |

**Hard constraint:** every kept change must respect the 3B anti-OOM math
(80 GB query cap, spill bounds, 94 GB host). No change that only works when
100M fits in RAM.

## Final verify (Phase 3)

3B run with the kept set. Update README scale + latency tables. Existing
tests (`go test ./cluster`, golden/validation) stay green throughout.

## Non-goals

- No staging-schema migration unless H3 is proven necessary (see above).
- No distributed CH.
- No query-semantics changes — results must stay byte-identical.
