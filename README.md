# Clustopher

# Clustopher

[![Go Tests](https://github.com/thomas-cabral/clustopher/actions/workflows/go-test.yml/badge.svg)](https://github.com/thomas-cabral/clustopher/actions/workflows/go-test.yml)

Clustopher is a high-performance spatial clustering system designed to handle large-scale point datasets (30M+ points) with associated metrics and metadata. It implements a KD-tree based clustering approach similar to Mapbox's Supercluster, but with additional features for metric aggregation and metadata preservation.

## Key Features

### Spatial Clustering
- Custom KD-tree implementation optimized for spatial data
- Dynamic clustering based on zoom levels and viewport bounds
- Efficient point aggregation for smooth map visualization
- Support for 30M+ points while maintaining interactive performance

### Metrics & Metadata
- Support for arbitrary numeric metrics on points
- Automatic metric rollup during clustering (sum, average, min, max)
- Metadata preservation and aggregation in clusters
- Real-time statistics and summaries for visible data

### Storage & Performance
- Compressed storage of pre-computed trees using zstd
- Fast loading of saved clusters for instant (based on cluster size) visualization
- Efficient memory management for large datasets

### Interactive Visualization
- Real-time map visualization using Mapbox GL
- Dynamic cluster updates based on viewport
- Interactive cluster exploration with zoom-to-cluster
- Detailed popup information for clusters and points
- Statistics panel showing current view metrics

## Architecture

### Backend (Go)
- `cluster` package: Core clustering implementation
  - KD-tree construction and querying
  - Metric aggregation and rollup
  - Compressed storage and loading
- REST API Test endpoints for:
  - Cluster creation and management
  - Viewport-based querying
  - Metadata and statistics

### Frontend (Svelte)
- Interactive map visualization
- Cluster management interface
- Real-time statistics display
- Responsive layout and controls

## Usage

### Creating a New Cluster
1. Use the cluster management interface to specify point count
2. System generates random test points within Continental US bounds
3. Points are processed and stored in a compressed format

### Loading Existing Clusters
1. View available clusters in the management interface
2. Load a cluster for visualization
3. Interact with the map to explore data

### Exploring Data
- Pan and zoom to view different clustering levels
- Click clusters to zoom in and explore
- Hover over clusters/points to view detailed metrics
- View real-time statistics for the current viewport

## Future Enhancements

### Planned Features
- Distributed architecture with worker nodes
- gRPC communication between components
- Multiple concurrent cluster support
- Custom point generation and import
- Advanced metric aggregation options

### Scaling Considerations
Currently limited to single-server deployment. Future versions will implement:
- Worker pool for distributed processing
- Load balancing across cluster nodes
- Shared storage for cluster data
- Horizontal scaling capabilities

## Technical Details

### Performance Characteristics
- Memory Usage: ~2GB for 1M points with metrics
- Load Time: ~30s for 1M points (initial clustering)
- Query Time: <50ms for typical viewport operations
- Storage: ~100MB per 1M points (compressed)

### Limitations
- Single cluster loaded at a time
- In-memory processing only
- Limited to pre-generated test data
- Single-server deployment

## Getting Started

### Prerequisites
- Go 1.21+
- Node.js 18+
- Mapbox API key

### Installation

```bash
Backend
cd clustopher
go mod download
go run main.go

Frontend
cd frontend
npm install
npm run dev
```


### Configuration
Set the following environment variables:
- `VITE_MAPBOX_TOKEN`: Your Mapbox API key

### Running Tests
```bash
go test
```
