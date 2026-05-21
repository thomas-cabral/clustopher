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
- Verified at 300 M points with sub-20 ms low/mid-zoom queries and 137 ms at z14 city viewport on a single box

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

#### Scale (load + storage)

| Points | CH stage gen | Load (skeleton + canonical points) | Rollup OPTIMIZE FINAL | Resident heap after load | Skeleton leaves |
|--------|-------------:|-----------------------------------:|----------------------:|-------------------------:|----------------:|
|   5 M  |   0.7 s |    9.2 s |   17.1 s |   18 MB |  78 K |
|  15 M  |   1.8 s |   28.6 s |   43.8 s |   33 MB | 234 K |
|  50 M  |   6.3 s |  103.1 s |  117.6 s |   83 MB | 781 K |
| 100 M  |  12.2 s |  216.4 s |  212.5 s |  176 MB | 1.5 M |
| 200 M  |  26.2 s |  536.2 s |  444.2 s |  334 MB | 3.1 M |
| 300 M  |  45.3 s |  833.7 s |  569.0 s |  444 MB | 4.7 M |

Resident heap is what the skeleton tree pins after `LoadFromCHStaging` + GC. Raw points live in ClickHouse; Go holds only the bounds-only leaf index. Rollup `OPTIMIZE FINAL` collapses per-insert parts into a single sorted run per zoom partition — needed once after bulk load so the rollup path scans contiguous data.

#### Query latency (`GetClustersCH`, 5-iter warm avg)

| Points | z2 / CONUS 60°×24° | z8 / state 5°×5° | z14 / city 0.5°×0.5° | z14 clusters returned | z14 alloc/op |
|--------|-------------------:|-----------------:|---------------------:|----------------------:|-------------:|
|   5 M  |  **2.9 ms** | **10.5 ms** |   **3.3 ms** |   1 600 |  0.48 MB |
|  15 M  |  **5.5 ms** | **12.9 ms** |   **6.6 ms** |   3 772 |  1.33 MB |
|  50 M  |  **3.7 ms** | **10.4 ms** |  **13.3 ms** |  11 395 |  4.20 MB |
| 100 M  |  **3.6 ms** | **10.2 ms** |  **29.0 ms** |  20 795 |  8.80 MB |
| 200 M  |  **3.5 ms** | **14.9 ms** |  **69.1 ms** |  34 408 | 15.07 MB |
| 300 M  |  **3.7 ms** | **16.5 ms** | **136.8 ms** |  44 683 | 20.46 MB |

z2 and z8 (`zoom < ZSplit`, default 11) hit the per-zoom rollup materialized views in ClickHouse: latency is flat across the entire size range because a rollup-row count is bounded by the zoom-tile grid, not the underlying point count.

z14 (`zoom >= ZSplit`) walks the in-memory skeleton tree, fetches the matching leaf-id ranges from ClickHouse, and runs the Supercluster-style radius clustering on the returned points. Latency scales with the *density of points inside the viewport* — at 300 M CONUS points the 0.5° × 0.5° viewport contains ≈ 50 K points, and the system returns 44 683 clusters in 137 ms (95 % of which is `fetchLeafPoints` + `clusterPoints`).

#### Limits encountered

300 M is the realistic ceiling on this hardware (94 GB RAM) with the current Load path. During `LoadFromCHStaging` Go materializes one `[]chSpatialRow` (12 B/row) and one `[]KDPoint` (16 B/row), and the `INSERT … SELECT … JOIN staging ⨝ id_map` runs server-side. Even with the in-place skeleton sort (introduced for this scale push) and `join_algorithm='full_sorting_merge'` (which bounds CH-server JOIN RAM), 500 M points combined Go heap (~14 GB) + ClickHouse working set + OS page cache for the ~50 GB staging partition exhaust RAM. Pushing past 300 M needs a streaming Load path: project rows on read, drop the spatial slice, keep external IDs as a packed `[]uint32`. That's the next optimization target.

#### Comparison to the pre-ClickHouse baseline (15 M points)

| Zoom | Old (full bbox, in-memory KD-tree) | New (zoom-appropriate viewport, CH-backed) | Speedup |
|------|-----------------------------------:|-------------------------------------------:|--------:|
| 2    | 30.35 s | 5.5 ms |  ~5 500× |
| 8    | 18.19 s | 12.9 ms |  ~1 400× |
| 14   | 67.42 s | 6.6 ms | ~10 200× |

Columns are not strictly apples-to-apples — the old benchmarks always queried the full CONUS bbox, unrealistic at z14 where a real viewport is one neighborhood. With zoom-appropriate viewports the new system stays interactive across the entire zoom range, and now scales to 20× the point count of the old in-memory system (300 M vs ~15 M before the heap blew up).

Raw results: `benchmark_results/scale_full_rerun.txt`, `benchmark_results/baseline_15M.txt`.

### Limitations
- Single-node ClickHouse only; no distributed CH support
- No point-level updates: reloading a cluster drops its CH partition and reinserts all points
- Metric aggregations are sum/avg only (no arbitrary aggregation functions)
- Skeleton tree is rebuilt from CH on first access after process restart or LRU eviction
