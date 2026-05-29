# Clustopher

[![Go Tests](https://github.com/thomas-cabral/clustopher/actions/workflows/go-test.yml/badge.svg)](https://github.com/thomas-cabral/clustopher/actions/workflows/go-test.yml)

Clustopher is a spatial point-clustering engine for datasets in the tens of millions of points, built to stay interactive (sub-100ms `GetClusters`) at every zoom level on a single box.

It splits the problem in two. ClickHouse is the canonical store: raw points live in a `MergeTree` partitioned by cluster, and a stack of materialized views maintains a per-zoom rollup table for zooms 2–16. In Go, a bounds-only "skeleton" KD-tree (`{Bounds, IDMin, IDMax, Count}` per leaf — 28 bytes) sits in memory as a spatial index over those points, ~6.5 MB resident for 15M points. At query time the zoom level decides the path: low/mid zoom runs a single SQL aggregation against the rollup table; high zoom walks the skeleton, classifies leaves as fully-inside vs straddling the cluster radius, batches `id BETWEEN` fetches into ClickHouse for the partial leaves, and runs the Supercluster-style radius clustering only on that small leftover set.

The motivation was a pure in-memory predecessor (KD-tree + grid clustering + zstd snapshots) that held every point and its metadata in Go heap. At 15M points it pinned ~9 GB of heap, took 30+ seconds to answer a low-zoom continental query, and couldn't be reloaded without deserializing a multi-GB snapshot. Pushing the points into ClickHouse and keeping only leaf bounds in Go collapses both the memory footprint and the cold-load story while letting low-zoom queries become a single rollup-table scan instead of a tree traversal.

## Key Features

### Spatial Clustering
- Bounds-only skeleton KD-tree in Go for fast spatial pruning at high zoom levels
- ClickHouse materialized-view rollups for low-zoom aggregation
- Dynamic routing between rollup and skeleton paths based on zoom level (`ZSplit`, default 11)
- Verified at **3 billion** points on a single box (94 GB RAM, single-node CH 24.8) with sub-20 ms low/mid-zoom queries via a CH-side single-pass load path

### Metrics & Metadata
- Support for arbitrary numeric metrics on points
- Automatic metric rollup during clustering (sum, average, min, max)
- Metadata preservation and aggregation in clusters
- Real-time statistics and summaries for visible data

### Storage & Performance
- Raw points stored in ClickHouse (`clustopher.points` MergeTree table)
- Per-zoom rollup aggregations via ClickHouse materialized views (zoom 2–16)
- In-memory skeleton tree (bounds-only leaf index) for spatial pruning at high zoom
- SQL aggregation at low zoom; skeleton + CH point fetch at high zoom

### Interactive Visualization
- Real-time map visualization using Mapbox GL
- Dynamic cluster updates based on viewport
- Interactive cluster exploration with zoom-to-cluster
- Detailed popup information for clusters and points
- Statistics panel showing current view metrics

## Architecture

### Backend (Go)
- `cluster` package: Core clustering implementation
  - Bounds-only skeleton KD-tree construction (`SkeletonTree`)
  - ClickHouse client, ingest, and query logic
  - Migrations embedded and applied at startup
  - Metric aggregation and rollup
- `runner` package: gRPC service wrapping `Supercluster`
  - LRU skeleton cache (configurable max in-memory clusters)
  - Cluster creation, listing, and querying over gRPC
- `cmd/runners`: gRPC server entry point (requires `CLICKHOUSE_DSN`)
- `cmd/api`: HTTP REST proxy to the gRPC runner (no direct CH access)

### Frontend (Svelte)
- Interactive map visualization
- Cluster management interface
- Real-time statistics display
- Responsive layout and controls

## Getting Started

### Prerequisites
- Go 1.21+
- Node.js 18+
- Docker (for ClickHouse)
- Mapbox API key

### Installation

```bash
# Start ClickHouse
docker compose up -d clickhouse

# Backend — gRPC runner (runs migrations on startup)
CLICKHOUSE_DSN=clickhouse://default:@localhost:9000/clustopher \
  go run ./cmd/runners

# Backend — HTTP API (in a second terminal)
go run ./cmd/api

# Frontend
cd frontend
npm install
npm run dev
```

### Configuration

Environment variables:
- `CLICKHOUSE_DSN`: ClickHouse connection string, e.g. `clickhouse://default:@localhost:9000/clustopher`
- `VITE_MAPBOX_TOKEN`: Your Mapbox API key

Runner flags (`cmd/runners`):
- `--ch-dsn`: ClickHouse DSN (overrides `CLICKHOUSE_DSN`)
- `--port`: gRPC listen port (default `50051`)
- `--max-clusters`: maximum skeleton trees to keep in memory (default `5`)

API flags (`cmd/api`):
- `--runners-addr`: address of the gRPC runner (default `localhost:50051`)
- `--port`: HTTP listen port (default `8000`)

### Running Tests
```bash
CLICKHOUSE_DSN=clickhouse://default:@localhost:9000/clustopher_test go test ./... -count=1
```

## Usage

### Creating a New Cluster
1. Use the cluster management interface to specify point count
2. System generates random test points within global bounds
3. Points are projected, sorted into leaf order, and persisted to ClickHouse; a skeleton tree is built in memory

### Loading Existing Clusters
1. View available clusters in the management interface (listed from CH)
2. Load a cluster — skeleton tree is rebuilt from CH point data
3. Interact with the map to explore data

### Exploring Data
- Pan and zoom to view different clustering levels
- Click clusters to zoom in and explore
- Hover over clusters/points to view detailed metrics
- View real-time statistics for the current viewport

## Technical Details

### Performance Characteristics

Measured on AMD Ryzen 9 5900X (24 threads), single-node ClickHouse 24.8 in Docker, random points uniformly distributed across the CONUS bounding box. Viewports are zoom-appropriate (continental at low zoom, state at mid zoom, city at high zoom):

Three load paths exist, in increasing scale ceiling:

- `LoadFromCHStaging` (KD): exact KD-tree median-partition sort in Go. Tightest leaf bounds, best high-zoom latency, but heap grows linearly → tops out ~300 M on this box.
- `LoadFromCHStreaming` (Morton-stream): pushes the spatial sort into ClickHouse via a Morton-encoded `ORDER BY`, streams sorted rows back in `NodeSize` chunks, writes an `(external→internal)` id map, then `INSERT … SELECT … JOIN staging ⨝ id_map`. Go heap bounded by the skeleton tree alone (≈ 1.6 GB at 1 B). Verified to 1 B; the 3B×3B populate JOIN exhausts CH-server RAM at 3 B.
- `LoadFromCHSinglePass` (CH-single, `CLUSTOPHER_SCALE_SINGLEPASS=1`): assigns internal ids entirely in ClickHouse with `rowNumberInAllBlocks()` over the Morton `ORDER BY`, writing the canonical `points` table in one `INSERT … SELECT` — **no id-map table and no JOIN**. Skeleton leaf bounds are read back with a single `GROUP BY intDiv(id-1, NodeSize)` (web-mercator monotonicity makes corner projection exact). This is what unlocked 3 B.

#### Scale (load + storage)

| Points | CH stage gen | Load (skeleton + canonical points) | Rollup OPTIMIZE FINAL | Resident heap after load | Skeleton leaves |
|--------|-------------:|-----------------------------------:|----------------------:|-------------------------:|----------------:|
|   5 M  |   1.8 s |    17.7 s |   5.3 s |   10 MB |  78 K |
|  15 M  |   5.3 s |    33.9 s |   5.9 s |   26 MB | 234 K |
|  50 M  |  16.8 s |    90.7 s |   6.1 s |   74 MB | 781 K |
| 100 M  |  37.4 s |   168.4 s |   6.0 s |  174 MB | 1.5 M |
| 200 M  |  74.5 s |   315.9 s |   6.3 s |  336 MB | 3.1 M |
| 300 M  | 106.6 s |   475.4 s |   6.2 s |  418 MB | 4.7 M |
| 500 M  | 180.1 s |   793.1 s |   6.4 s |  812 MB | 7.8 M |
| **1 B** | **393.1 s** | **1 732.9 s** | **6.4 s** | **1 580 MB** | **15.6 M** |
| **3 B** † | **1 333.5 s** | **4 175.1 s** | **6.1 s** | **4 708 MB** | **46.9 M** |

† 3 B uses `LoadFromCHSinglePass` (the Morton-stream JOIN exhausts CH RAM at this size). Load breakdown: 58.0 min INSERT (Morton sort + `rowNumberInAllBlocks` + write), 46 s leaf-bounds read, 9 s skeleton build, 10.7 min deferred rollup batch. The INSERT ran with tight CH caps (1 GB per-thread external-sort threshold, 10 threads, 80 GB hard cap) to keep the wide metrics+metadata payload spilling to disk rather than blowing the 94 GB host — looser caps OOM, this completes.

Resident heap is what the skeleton tree pins after `Load…` returns + GC. Raw points live in ClickHouse; Go holds only the bounds-only leaf index. Rollup `OPTIMIZE FINAL` collapses per-insert parts into a single sorted run per zoom partition — needed once after bulk load so the rollup path scans contiguous data.

Rollup MVs are only created for zooms `[MinRollupZoom..MaxRollupZoom]` (currently 2–10), because the routing cutoff `ZSplit=11` means zooms 11+ are answered by the in-memory skeleton tree, never by the rollup path. Dropping the z11–z16 MVs cuts both incremental Load wall-clock (fewer MVs to fan to per insert block) and `OPTIMIZE FINAL` from minutes to single-digit seconds at every dataset size measured.

##### Deferred MV populate (`CLUSTOPHER_DEFERRED_ROLLUPS=1`, default on)

Each MV defined `FROM clustopher.points` fires per insert block, so 1 B point insertions used to drive ~9 B partial MV row writes (one per zoom level per source row) plus their downstream SummingMergeTree merges. Loading instead detaches all rollup MVs for the duration of `populatePointsFromStaging`, then runs one pre-aggregated `INSERT … SELECT … GROUP BY tile_x, tile_y` per zoom from the now-populated `points` table. Each batch writes at most 4ᴺ rollup rows (only ~1.5 M total across z2–z10 at 1 B points) so the SummingMergeTree starts in already-merged shape.

Wall-clock effect at 1 B points: Load drops from 46 min (v3, incremental MVs) to **29 min** (deferred); total cold-load wall drops 49 min → **36 min**. The win scales with N — at 5 M the per-insert MV cost was tiny so deferred is slightly slower; at 100 M+ it's a clean 30–40 % cut.

#### Query latency (`GetClustersCH`, 5-iter warm avg)

| Points | z2 / CONUS 60°×24° | z8 / state 5°×5° | z14 / city 0.5°×0.5° | z14 clusters returned | z14 alloc/op |
|--------|-------------------:|-----------------:|---------------------:|----------------------:|-------------:|
|   5 M  |  **3.7 ms** | **11.8 ms** |   **13.4 ms** |   2 112 |   4.39 MB |
|  15 M  |  **3.5 ms** | **11.1 ms** |   **19.8 ms** |   4 792 |  10.09 MB |
|  50 M  |  **4.9 ms** | **15.1 ms** |   **46.2 ms** |  12 529 |  26.87 MB |
| 100 M  |  **4.7 ms** | **11.8 ms** |   **75.4 ms** |  21 539 |  47.00 MB |
| 200 M  |  **4.2 ms** | **12.1 ms** |  **155.2 ms** |  36 636 |  83.62 MB |
| 300 M  |  **4.0 ms** | **13.0 ms** |  **269.2 ms** |  47 373 | 115.90 MB |
| 500 M  |  **5.3 ms** | **13.0 ms** |  **508.8 ms** |  59 064 | 163.22 MB |
| **1 B** |  **4.3 ms** | **14.2 ms** | **1 000.3 ms** |  68 986 | 262.08 MB |
| **3 B** |  **3.4 ms** | **17.5 ms** | **3 984.2 ms** |  58 727 | 993.57 MB |

z2 and z8 (`zoom < ZSplit`, default 11) hit the per-zoom rollup materialized views in ClickHouse: latency is flat across the entire 600× size range because the rollup-row count is bounded by the zoom-tile grid, not the underlying point count.

z14 (`zoom >= ZSplit`) walks the in-memory skeleton tree, fetches the matching leaf-id ranges from ClickHouse, and runs the Supercluster-style radius clustering on the returned points. Latency scales with the *density of points inside the viewport* — at 1 B CONUS points the 0.5° × 0.5° viewport contains ≈ 170 K points (most of the wall time is `fetchLeafPoints` + `clusterPoints` on the ~120 K points pulled back into Go).

#### Why streaming unlocked 500 M → 1 B

The KD-sort load path tops out around 300 M on a 94 GB box because it materializes one `[]chSpatialRow` (12 B/row) and one `[]KDPoint` (16 B/row) in Go and then runs an in-place median-partition sort. At 500 M that's ~14 GB of resident Go heap on top of the CH-server working set and OS page cache — the combination thrashes swap and OOMs.

`LoadFromCHStreaming` moves the spatial sort to ClickHouse via `ORDER BY mortonEncode(lng_u32, lat_u32)` (a Z-order space-filling curve) and consumes rows one block at a time. Go accumulates `NodeSize` points per chunk → emits a `SkeletonLeaf` → frees the chunk. Steady-state Go heap = the skeleton tree only (~1.5 GB at 1 B points). The `INSERT … SELECT … JOIN staging ⨝ id_map` is unchanged but the `ORDER BY internal_id` was dropped: MergeTree already sorts blocks by its table `ORDER BY` on insert, so re-sorting 1 B rows in the JOIN output was costing ~70 GB of CH-server RAM with no observable benefit.

Tradeoff: Morton leaves are slightly looser than KD leaves (about 15 % more points returned for the same z14 viewport at the same N), so high-zoom latency is mildly worse than the KD path would be at the same size. For datasets large enough to exhaust the KD path, that's the right trade.

#### Why single-pass unlocked 1 B → 3 B

The Morton-stream path still does an `INSERT … SELECT … staging ⨝ id_map` to attach the Go-assigned internal id back onto the full payload. At 3 B that join's output is ~450 GB of `(x, y, metrics, metadata)` and the merge-join working set + MergeTree part formation push CH-server RAM past the host ceiling.

`LoadFromCHSinglePass` removes the id-map table and the join entirely. Internal ids are assigned inside ClickHouse: `SELECT toUInt32(rowNumberInAllBlocks() + 1) AS id, … FROM staging_points ORDER BY mortonEncode(…)`. `rowNumberInAllBlocks()` is evaluated *after* the global `ORDER BY`, so ids 1..N land in exact Morton order in a single streaming `INSERT` — no second table, no join. The skeleton is then rebuilt from the canonical table with one aggregate scan:

```sql
SELECT intDiv(id-1, NodeSize) AS leaf,
       min(x), max(x), min(y), max(y), min(id), max(id), count()
FROM points WHERE cluster_id = ? GROUP BY leaf ORDER BY leaf
```

Web-mercator projection is monotonic in each axis within the supported latitude band, so projecting the `(min_lng, max_lat)` and `(max_lng, min_lat)` corners of each leaf yields exact pixel-space bounds — no need to stream all 3 B rows through Go. At 3 B the leaf-bounds read is 46 s and the skeleton rebuild 9 s.

Cost: `rowNumberInAllBlocks()` forces a single-threaded final merge in CH, so the big `INSERT` can't fully parallelize. At 100 M (fits in RAM, no spill) the single-pass load is 131 s vs streaming's 168 s — 22 % faster. At 3 B the tight anti-OOM caps force heavy spill and the INSERT stretches to 58 min, but it *completes* where streaming OOMs. Net: single-pass is both faster at mid-scale and the only path that reaches 3 B on this hardware.

#### Comparison to the pre-ClickHouse baseline (15 M points)

| Zoom | Old (full bbox, in-memory KD-tree) | New (zoom-appropriate viewport, CH-backed) | Speedup |
|------|-----------------------------------:|-------------------------------------------:|--------:|
| 2    | 30.35 s | 3.5 ms |  ~8 700× |
| 8    | 18.19 s | 11.1 ms |  ~1 600× |
| 14   | 67.42 s | 19.8 ms | ~3 400× |

Columns are not strictly apples-to-apples — the old benchmarks always queried the full CONUS bbox, unrealistic at z14 where a real viewport is one neighborhood. With zoom-appropriate viewports the new system stays interactive across the entire zoom range, and now scales to ~200× the point count of the old in-memory system (3 B vs ~15 M before the heap blew up).

Raw results: `benchmark_results/scale_deferred_full.txt`, `benchmark_results/scale_deferred_1B.txt`, `benchmark_results/scale_singlepass_3B.txt`, `benchmark_results/baseline_15M.txt`.

> **Note on `CH stage gen` column**: jumped vs the prior README (e.g. 12.8 s → 37.4 s at 100 M) because `generateDenseStagingPoints` now emits richer test metadata (4 categorical + 3 numeric fields per point via separate `cityHash64(number, salt)` draws). That's unrelated to deferred-rollup work — it's bench-rig cost, not production cost.

### Limitations
- Single-node ClickHouse only; no distributed CH support
- No point-level updates: reloading a cluster drops its CH partition and reinserts all points
- Metric aggregations are sum/avg only (no arbitrary aggregation functions)
- Skeleton tree is rebuilt from CH on first access after process restart or LRU eviction

## License

Apache License 2.0. See [LICENSE](LICENSE).
