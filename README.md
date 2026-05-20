# Clustopher

[![Go Tests](https://github.com/thomas-cabral/clustopher/actions/workflows/go-test.yml/badge.svg)](https://github.com/thomas-cabral/clustopher/actions/workflows/go-test.yml)

Clustopher is a high-performance spatial clustering system designed to handle large-scale point datasets (30M+ points) with associated metrics and metadata. It implements a KD-tree based clustering approach similar to Mapbox's Supercluster, with ClickHouse as the canonical data store for points and per-zoom aggregations.

## Key Features

### Spatial Clustering
- Bounds-only skeleton KD-tree in Go for fast spatial pruning at high zoom levels
- ClickHouse materialized-view rollups for low-zoom aggregation
- Dynamic routing between rollup and skeleton paths based on zoom level (`ZSplit`, default 11)
- Support for 30M+ points while maintaining interactive query performance

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

TBD — updated post-15M-point benchmark; see `benchmark_results/ch_15M.txt` once available.

### Limitations
- Single-node ClickHouse only; no distributed CH support
- No point-level updates: reloading a cluster drops its CH partition and reinserts all points
- Metric aggregations are sum/avg only (no arbitrary aggregation functions)
- Skeleton tree is rebuilt from CH on first access after process restart or LRU eviction
