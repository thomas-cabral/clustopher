package cluster

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"

	"runtime/debug"
	"time"

	"github.com/google/uuid"
	"github.com/klauspost/compress/zstd"
)

// Add Pool to KDTree struct
type KDNode struct {
	PointIdx int32  // 4 bytes - index into points array
	Left     int32  // 4 bytes - index into nodes array
	Right    int32  // 4 bytes - index into nodes array
	Axis     uint8  // 1 byte  - 0 or 1 is sufficient
	MinChild uint32 // 4 bytes
	MaxChild uint32 // 4 bytes
}

type KDTree struct {
	Nodes    []KDNode  // All nodes in a single slice
	Points   []KDPoint // All points in a single slice
	NodeSize int
	Bounds   KDBounds
	Pool     *MetricsPool // Reference to shared metrics pool
}

type KDPoint struct {
	X, Y      float32                // 8 bytes
	ID        uint32                 // 4 bytes
	NumPoints uint32                 // 4 bytes
	MetricIdx uint32                 // 4 bytes - index into metrics pool
	Metadata  map[string]interface{} // Keep metadata for clustering
}

type Point struct {
	ID       uint32
	X, Y     float32
	Metrics  map[string]float32
	Metadata map[string]interface{}
}

type MetricsPool struct {
	Metrics []map[string]float32
	Lookup  map[string]int // For deduplication
	mu      sync.RWMutex   // Protect concurrent access
}

type SharedPools struct {
	MetricsPool  []map[string]float32
	MetricsKeys  map[string]uint32 // For deduplication
	StringPool   map[string]uint32 // For deduplicating strings
	StringValues []string          // Actual string storage
}

// 3. Optimize ClusterNode to use shared pools
type ClusterNode struct {
	ID       uint32
	X, Y     float32
	Count    uint32
	Children []uint32
	Metrics  ClusterMetrics
	Metadata map[string]json.RawMessage // Use json.RawMessage to handle different types
}

// Supercluster implements the clustering algorithm
type Supercluster struct {
	Tree    *KDTree // Single KD-tree for all zoom levels
	Points  []Point // Original input points
	Options SuperclusterOptions
}

type SuperclusterOptions struct {
	MinZoom   int
	MaxZoom   int
	MinPoints int
	Radius    float64
	NodeSize  int
	Extent    int
	Log       bool
}

// GeoJSON types
type Feature struct {
	Type       string                 `json:"type"`
	Geometry   Geometry               `json:"geometry"`
	Properties map[string]interface{} `json:"properties"`
}

type FeatureCollection struct {
	Type     string    `json:"type"`
	Features []Feature `json:"features"`
}

type Geometry struct {
	Type        string    `json:"type"`
	Coordinates []float64 `json:"coordinates"`
}

// ToGeoJSON converts clusters to GeoJSON format
func (sc *Supercluster) ToGeoJSON(bounds KDBounds, zoom int) (*FeatureCollection, error) {
	// Get clusters for the given bounds and zoom level
	clusters := sc.GetClusters(bounds, zoom)

	// Convert clusters to GeoJSON features
	features := make([]Feature, len(clusters))
	for i, cluster := range clusters {
		// Create properties map
		properties := make(map[string]interface{})
		properties["cluster"] = cluster.Count > 1
		properties["cluster_id"] = cluster.ID
		properties["point_count"] = cluster.Count

		// Add metadata if it exists
		if cluster.Metadata != nil {
			for k, v := range cluster.Metadata {
				properties[k] = v
			}
		}

		// Create feature
		features[i] = Feature{
			Type: "Feature",
			Geometry: Geometry{
				Type:        "Point",
				Coordinates: []float64{float64(cluster.X), float64(cluster.Y)},
			},
			Properties: properties,
		}
	}

	return &FeatureCollection{
		Type:     "FeatureCollection",
		Features: features,
	}, nil
}

// NewSupercluster creates a new clustering instance with the specified options.
// It validates and sets default values for the options if not provided.
func NewSupercluster(options SuperclusterOptions) *Supercluster {
	// Set default values if not provided
	if options.MinZoom < 0 {
		options.MinZoom = 0
	}
	if options.MaxZoom <= 0 {
		options.MaxZoom = 16
	}
	if options.NodeSize <= 0 {
		options.NodeSize = 64
	}
	if options.Extent <= 0 {
		options.Extent = 512
	}
	if options.Radius <= 0 {
		options.Radius = 40
	}
	if options.MinPoints <= 0 {
		options.MinPoints = 3
	}

	// Validate zoom levels
	if options.MinZoom > options.MaxZoom {
		options.MinZoom = options.MaxZoom
	}
	if options.MaxZoom > 16 {
		options.MaxZoom = 16
	}

	return &Supercluster{
		Tree:    nil, // Will be initialized when Load() is called
		Points:  nil, // Will be initialized when Load() is called
		Options: options,
	}
}

// Modified NewKDTree that actually uses the pools
func NewKDTree(points []KDPoint, nodeSize int, metricsPool *MetricsPool) *KDTree {
	maxNodes := len(points) * 2 // Worst case for a binary tree
	tree := &KDTree{
		Nodes:    make([]KDNode, 0, maxNodes),
		Points:   make([]KDPoint, len(points)),
		NodeSize: nodeSize,
		Pool:     metricsPool,
	}

	// Copy points to avoid modifying input
	copy(tree.Points, points)

	// Calculate bounds
	bounds := KDBounds{
		MinX: float32(math.Inf(1)),
		MinY: float32(math.Inf(1)),
		MaxX: float32(math.Inf(-1)),
		MaxY: float32(math.Inf(-1)),
	}

	for _, p := range points {
		bounds.Extend(p.X, p.Y)
	}
	tree.Bounds = bounds

	// Build tree if we have points
	if len(points) > 0 {
		tree.buildNodes(0, len(points)-1, 0)
	}

	return tree
}

// Modified buildNode to use the node pool
func (t *KDTree) buildNodes(start, end, depth int) int32 {
	if start > end {
		return -1
	}

	nodeIdx := int32(len(t.Nodes))
	t.Nodes = append(t.Nodes, KDNode{})
	node := &t.Nodes[nodeIdx]

	if end-start <= t.NodeSize {
		node.PointIdx = int32(start)
		node.Left = -1
		node.Right = -1
		setMinMaxChild(node, t.Points[start:end+1])
		return nodeIdx
	}

	axis := depth % 2
	median := (start + end) / 2

	sortPointsRange(t.Points[start:end+1], axis)

	node.PointIdx = int32(median)
	node.Axis = uint8(axis)

	node.Left = t.buildNodes(start, median-1, depth+1)
	node.Right = t.buildNodes(median+1, end, depth+1)

	setMinMaxChild(node, t.Points[start:end+1])
	return nodeIdx
}

func setMinMaxChild(node *KDNode, points []KDPoint) {
	node.MinChild = points[0].ID
	node.MaxChild = points[0].ID
	for _, p := range points[1:] {
		if p.ID < node.MinChild {
			node.MinChild = p.ID
		}
		if p.ID > node.MaxChild {
			node.MaxChild = p.ID
		}
	}
}

func sortPointsRange(points []KDPoint, axis int) {
	if axis == 0 {
		sort.Slice(points, func(i, j int) bool {
			return points[i].X < points[j].X
		})
	} else {
		sort.Slice(points, func(i, j int) bool {
			return points[i].Y < points[j].Y
		})
	}
}

func metricsKey(metrics map[string]float32) string {
	keys := make([]string, 0, len(metrics))
	for k := range metrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "%s:%.6f;", k, metrics[k])
	}
	return b.String()
}

func NewMetricsPool() *MetricsPool {
	return &MetricsPool{
		Metrics: make([]map[string]float32, 0),
		Lookup:  make(map[string]int),
	}
}

// CleanupCluster releases memory held by the cluster
func (sc *Supercluster) CleanupCluster() {
	if sc == nil {
		return
	}

	// Clear tree structures
	if sc.Tree != nil {
		// Clear nodes array
		sc.Tree.Nodes = nil

		// Clear points array
		sc.Tree.Points = nil

		// Clear metrics pool
		if sc.Tree.Pool != nil {
			sc.Tree.Pool.Metrics = nil
			sc.Tree.Pool.Lookup = nil
			sc.Tree.Pool = nil
		}

		sc.Tree = nil
	}

	// Clear original points
	sc.Points = nil

	// Force immediate garbage collection
	runtime.GC()
	debug.FreeOSMemory()
}

// 6. Implement batch processing for large datasets
func (sc *Supercluster) LoadBatched(points []Point, batchSize int) {
	totalPoints := len(points)
	metricsPool := NewMetricsPool()

	// Process in batches
	for i := 0; i < totalPoints; i += batchSize {
		end := i + batchSize
		if end > totalPoints {
			end = totalPoints
		}

		batch := points[i:end]
		sc.processBatch(batch, metricsPool)

		// Force GC between batches if memory pressure is high
		if i > 0 && i%(batchSize*10) == 0 {
			runtime.GC()
		}
	}
}

// processBatch handles a batch of points during loading
func (sc *Supercluster) processBatch(batch []Point, metricsPool *MetricsPool) {
	kdPoints := make([]KDPoint, len(batch))

	for i, p := range batch {
		// Get or add metrics to pool
		metricsIdx := metricsPool.Add(p.Metrics)

		kdPoints[i] = KDPoint{
			X:         p.X,
			Y:         p.Y,
			ID:        p.ID,
			NumPoints: 1,
			MetricIdx: metricsIdx,
			Metadata:  p.Metadata,
		}
	}

	// If this is the first batch, create the tree
	if sc.Tree == nil {
		sc.Tree = NewKDTree(kdPoints, sc.Options.NodeSize, metricsPool)
	} else {
		// Otherwise, add points to existing tree
		for _, point := range kdPoints {
			sc.Tree.Insert(point)
		}
	}

	// Append to original points slice
	sc.Points = append(sc.Points, batch...)
}

// Add helper method to MetricsPool for clarity
func (mp *MetricsPool) GetPointMetrics(point KDPoint) map[string]float32 {
	if mp == nil {
		return nil
	}
	return mp.Get(point.MetricIdx)
}

// Add inserts metrics into the pool and returns the index
func (mp *MetricsPool) Add(metrics map[string]float32) uint32 {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	key := metricsKey(metrics)
	if idx, exists := mp.Lookup[key]; exists {
		return uint32(idx)
	}

	idx := len(mp.Metrics)
	metricsCopy := make(map[string]float32, len(metrics))
	for k, v := range metrics {
		metricsCopy[k] = v
	}

	mp.Metrics = append(mp.Metrics, metricsCopy)
	mp.Lookup[key] = idx

	return uint32(idx)
}

// Get retrieves metrics by index
func (mp *MetricsPool) Get(idx uint32) map[string]float32 {
	mp.mu.RLock()
	defer mp.mu.RUnlock()

	if int(idx) >= len(mp.Metrics) {
		return nil
	}
	return mp.Metrics[idx]
}

// Insert adds a new point to an existing KDTree
func (t *KDTree) Insert(point KDPoint) {
	// If tree is empty, create first node
	if len(t.Nodes) == 0 {
		t.Nodes = append(t.Nodes, KDNode{
			PointIdx: 0,
			Left:     -1,
			Right:    -1,
			Axis:     0,
			MinChild: point.ID,
			MaxChild: point.ID,
		})
		t.Points = append(t.Points, point)
		return
	}

	// Update bounds
	t.Bounds.Extend(point.X, point.Y)

	// Add point to points slice and get its index
	pointIdx := int32(len(t.Points))
	t.Points = append(t.Points, point)

	// Insert into tree structure
	t.insertNode(0, pointIdx, 0) // Start at root (index 0)
}

// insertNode recursively finds the right place for a point
func (t *KDTree) insertNode(nodeIdx int32, pointIdx int32, depth int) {
	if nodeIdx < 0 || int(nodeIdx) >= len(t.Nodes) {
		return
	}

	node := &t.Nodes[nodeIdx]
	newPoint := t.Points[pointIdx]

	// Update min/max child IDs
	if newPoint.ID < node.MinChild {
		node.MinChild = newPoint.ID
	}
	if newPoint.ID > node.MaxChild {
		node.MaxChild = newPoint.ID
	}

	// Choose axis based on depth
	axis := depth % 2

	var compareVal, nodeVal float32
	if axis == 0 {
		compareVal = newPoint.X
		nodeVal = t.Points[node.PointIdx].X
	} else {
		compareVal = newPoint.Y
		nodeVal = t.Points[node.PointIdx].Y
	}

	// Insert into appropriate subtree
	if compareVal < nodeVal {
		if node.Left == -1 {
			// Create new node
			newNodeIdx := int32(len(t.Nodes))
			t.Nodes = append(t.Nodes, KDNode{
				PointIdx: pointIdx,
				Left:     -1,
				Right:    -1,
				Axis:     uint8((axis + 1) % 2),
				MinChild: newPoint.ID,
				MaxChild: newPoint.ID,
			})
			node.Left = newNodeIdx
		} else {
			t.insertNode(node.Left, pointIdx, depth+1)
		}
	} else {
		if node.Right == -1 {
			// Create new node
			newNodeIdx := int32(len(t.Nodes))
			t.Nodes = append(t.Nodes, KDNode{
				PointIdx: pointIdx,
				Left:     -1,
				Right:    -1,
				Axis:     uint8((axis + 1) % 2),
				MinChild: newPoint.ID,
				MaxChild: newPoint.ID,
			})
			node.Right = newNodeIdx
		} else {
			t.insertNode(node.Right, pointIdx, depth+1)
		}
	}
}

// Load initializes the cluster index with points
func (sc *Supercluster) Load(points []Point) {
	fmt.Printf("Loading %d points\n", len(points))

	metricsPool := NewMetricsPool()
	kdPoints := make([]KDPoint, len(points))

	for i, p := range points {
		metricIdx := metricsPool.Add(p.Metrics)
		kdPoints[i] = KDPoint{
			X:         p.X,
			Y:         p.Y,
			ID:        p.ID,
			NumPoints: 1,
			MetricIdx: metricIdx,
			Metadata:  p.Metadata,
		}
	}

	sc.Points = points
	sc.Tree = NewKDTree(kdPoints, sc.Options.NodeSize, metricsPool)
}

// GetClusters returns clusters for the given bounds and zoom level
func (sc *Supercluster) GetClusters(bounds KDBounds, zoom int) []ClusterNode {
	fmt.Printf("Getting clusters for zoom level %d\n", zoom)
	fmt.Printf("Bounds: MinX: %f, MinY: %f, MaxX: %f, MaxY: %f\n",
		bounds.MinX, bounds.MinY, bounds.MaxX, bounds.MaxY)
	fmt.Printf("Total points in tree: %d\n", len(sc.Tree.Points))

	// Project bounds to tile space for current zoom level
	minP := sc.projectFast(bounds.MinX, bounds.MaxY, zoom)
	maxP := sc.projectFast(bounds.MaxX, bounds.MinY, zoom)

	fmt.Printf("Projected bounds: Min(%f,%f) Max(%f,%f)\n",
		minP[0], minP[1], maxP[0], maxP[1])

	// Get all points in the bounds
	var points []KDPoint
	for _, p := range sc.Tree.Points {
		proj := sc.projectFast(p.X, p.Y, zoom)

		// Check if point is within bounds
		if proj[0] >= minP[0] && proj[0] <= maxP[0] &&
			proj[1] >= minP[1] && proj[1] <= maxP[1] {
			points = append(points, p)
		}
	}

	// Calculate clustering radius
	zoomFactor := math.Pow(2, float64(sc.Options.MaxZoom-zoom))
	radius := float32(sc.Options.Radius * zoomFactor / float64(sc.Options.Extent))

	// Project and cluster points
	projectedPoints := sc.projectPoints(points, zoom, sc.Tree.Pool)
	clusters := sc.clusterPoints(projectedPoints, radius)

	// Convert back to lng/lat
	for i := range clusters {
		unproj := sc.unprojectFast(clusters[i].X, clusters[i].Y, zoom)
		clusters[i].X = unproj[0]
		clusters[i].Y = unproj[1]
	}

	return clusters
}

func (sc *Supercluster) projectPoints(points []KDPoint, zoom int, metricsPool *MetricsPool) []KDPoint {
	projectedPoints := make([]KDPoint, 0, len(points))

	for _, p := range points {
		proj := sc.projectFast(p.X, p.Y, zoom)

		// Create projected point keeping the same MetricIdx
		projectedPoints = append(projectedPoints, KDPoint{
			X:         proj[0],
			Y:         proj[1],
			ID:        p.ID,
			NumPoints: p.NumPoints,
			MetricIdx: p.MetricIdx, // Keep the same metrics index
			Metadata:  p.Metadata,  // Keep metadata if needed for clustering
		})
	}

	return projectedPoints
}

func (sc *Supercluster) clusterPoints(points []KDPoint, radius float32) []ClusterNode {
	fmt.Printf("Clustering %d points with radius %f\n", len(points), radius)

	var clusters []ClusterNode
	processed := make(map[uint32]bool)

	for _, p := range points {
		if processed[p.ID] {
			continue
		}

		// Find nearby points
		var nearby []KDPoint
		for _, other := range points {
			if other.ID == p.ID {
				continue
			}

			dx := other.X - p.X
			dy := other.Y - p.Y
			if dx*dx+dy*dy <= radius*radius {
				nearby = append(nearby, other)
			}
		}

		// If we have enough points, create a cluster
		if len(nearby) >= sc.Options.MinPoints {
			cluster := createCluster(append(nearby, p), sc.Tree.Pool)
			clusters = append(clusters, cluster)

			// Mark points as processed
			for _, np := range nearby {
				processed[np.ID] = true
			}
			processed[p.ID] = true
		} else {
			// Add as individual point
			pointMetrics := sc.Tree.Pool.Get(p.MetricIdx)
			clusters = append(clusters, ClusterNode{
				ID:      p.ID,
				X:       p.X,
				Y:       p.Y,
				Count:   1,
				Metrics: ClusterMetrics{Values: pointMetrics},
			})
			processed[p.ID] = true
		}
	}

	fmt.Printf("Created %d clusters from %d points\n", len(clusters), len(points))
	return clusters
}

func createCluster(points []KDPoint, metricsPool *MetricsPool) ClusterNode {
	var sumX, sumY float64
	metrics := make(map[string]float64)
	var totalPoints uint32

	// Create a map to aggregate metadata
	metadata := make(map[string]interface{})
	metadataCounts := make(map[string]int)

	// Accumulate values
	for _, p := range points {
		weight := float64(p.NumPoints)
		sumX += float64(p.X) * weight
		sumY += float64(p.Y) * weight
		totalPoints += p.NumPoints

		// Get metrics from pool and accumulate
		if pointMetrics := metricsPool.Get(p.MetricIdx); pointMetrics != nil {
			for k, v := range pointMetrics {
				metrics[k] += float64(v) * weight
			}
		}

		// Aggregate metadata
		for k, v := range p.Metadata {
			key := fmt.Sprintf("%s:%v", k, v)
			metadataCounts[key]++

			// Store the actual key-value pair
			if metadataCounts[key] == 1 {
				metadata[k] = v
			}
		}
	}

	// Calculate averages
	invTotal := 1.0 / float64(totalPoints)
	cluster := ClusterNode{
		ID:    uuid.New().ID(),
		X:     float32(sumX * invTotal),
		Y:     float32(sumY * invTotal),
		Count: totalPoints,
		Metrics: ClusterMetrics{
			Values: make(map[string]float32),
		},
		Metadata: make(map[string]json.RawMessage),
	}

	// Average metrics
	for k, sum := range metrics {
		cluster.Metrics.Values[k] = float32(sum * invTotal)
	}

	// Add metadata to cluster
	// Only keep metadata values that appear in all points
	for k, v := range metadata {
		if metadataCounts[fmt.Sprintf("%s:%v", k, v)] == len(points) {
			// Convert to json.RawMessage
			jsonBytes, err := json.Marshal(v)
			if err == nil {
				cluster.Metadata[k] = jsonBytes
			}
		}
	}

	return cluster
}

type ClusterMetrics struct {
	Values map[string]float32
}

type KDBounds struct {
	MinX, MinY, MaxX, MaxY float32
}

// Extend expands bounds to include another point
func (b *KDBounds) Extend(x, y float32) {
	b.MinX = float32(math.Min(float64(b.MinX), float64(x)))
	b.MinY = float32(math.Min(float64(b.MinY), float64(y)))
	b.MaxX = float32(math.Max(float64(b.MaxX), float64(x)))
	b.MaxY = float32(math.Max(float64(b.MaxY), float64(y)))
}

// Project functions are needed for GetClusters
// projectFast converts lng/lat to tile coordinates
func (sc *Supercluster) projectFast(lng, lat float32, zoom int) [2]float32 {
	sin := float32(math.Sin(float64(lat) * math.Pi / 180))
	x := (lng + 180) / 360
	y := float32(0.5 - 0.25*math.Log(float64((1+sin)/(1-sin)))/math.Pi)

	scale := float32(math.Pow(2, float64(zoom)))
	return [2]float32{
		x * scale * float32(sc.Options.Extent),
		y * scale * float32(sc.Options.Extent),
	}
}

// unprojectFast converts tile coordinates back to lng/lat
func (sc *Supercluster) unprojectFast(x, y float32, zoom int) [2]float32 {
	scale := float32(math.Pow(2, float64(zoom)))

	// Convert to normalized coordinates
	x = x / (scale * float32(sc.Options.Extent))
	y = y / (scale * float32(sc.Options.Extent))

	// Convert to lng/lat
	lng := x*360 - 180
	lat := float32(math.Atan(math.Sinh(float64(math.Pi*(1-2*y))))) * 180 / math.Pi

	return [2]float32{lng, lat}
}

func (sc *Supercluster) SaveCompressed(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return fmt.Errorf("failed to create file: %v", err)
    }
    defer file.Close()

    bufWriter := bufio.NewWriterSize(file, 1024*1024)
    enc, err := zstd.NewWriter(bufWriter,
        zstd.WithEncoderLevel(zstd.SpeedBestCompression))
    if err != nil {
        return fmt.Errorf("failed to create zstd writer: %v", err)
    }
    defer enc.Close()

    // Write sizes first for allocation
    binary.Write(enc, binary.LittleEndian, uint32(len(sc.Tree.Nodes)))
    binary.Write(enc, binary.LittleEndian, uint32(len(sc.Tree.Points)))
    binary.Write(enc, binary.LittleEndian, uint32(len(sc.Tree.Pool.Metrics)))
    
    // Write Options
    binary.Write(enc, binary.LittleEndian, sc.Options.MinZoom)
    binary.Write(enc, binary.LittleEndian, sc.Options.MaxZoom)
    binary.Write(enc, binary.LittleEndian, sc.Options.MinPoints)
    binary.Write(enc, binary.LittleEndian, float64(sc.Options.Radius))
    binary.Write(enc, binary.LittleEndian, sc.Options.NodeSize)
    binary.Write(enc, binary.LittleEndian, sc.Options.Extent)

    // Write nodes
    for _, node := range sc.Tree.Nodes {
        binary.Write(enc, binary.LittleEndian, node.PointIdx)
        binary.Write(enc, binary.LittleEndian, node.Left)
        binary.Write(enc, binary.LittleEndian, node.Right)
        binary.Write(enc, binary.LittleEndian, node.Axis)
        binary.Write(enc, binary.LittleEndian, node.MinChild)
        binary.Write(enc, binary.LittleEndian, node.MaxChild)
    }

    // Write points
    for _, point := range sc.Tree.Points {
        binary.Write(enc, binary.LittleEndian, point.X)
        binary.Write(enc, binary.LittleEndian, point.Y)
        binary.Write(enc, binary.LittleEndian, point.ID)
        binary.Write(enc, binary.LittleEndian, point.NumPoints)
        binary.Write(enc, binary.LittleEndian, point.MetricIdx)
        
        // Write metadata size
        binary.Write(enc, binary.LittleEndian, uint32(len(point.Metadata)))
        
        // Write each metadata key-value pair
        for k, v := range point.Metadata {
            // Write key
            keyBytes := []byte(k)
            binary.Write(enc, binary.LittleEndian, uint32(len(keyBytes)))
            enc.Write(keyBytes)
            
            // Convert value to JSON bytes
            valueBytes, err := json.Marshal(v)
            if err != nil {
                return fmt.Errorf("failed to marshal metadata value: %v", err)
            }
            
            // Write value
            binary.Write(enc, binary.LittleEndian, uint32(len(valueBytes)))
            enc.Write(valueBytes)
        }
    }

    // Write metrics
    for _, metrics := range sc.Tree.Pool.Metrics {
        binary.Write(enc, binary.LittleEndian, uint32(len(metrics)))
        for k, v := range metrics {
            binary.Write(enc, binary.LittleEndian, uint32(len(k)))
            enc.Write([]byte(k))
            binary.Write(enc, binary.LittleEndian, v)
        }
    }

    if err := enc.Close(); err != nil {
        return fmt.Errorf("failed to close encoder: %v", err)
    }

    if err := bufWriter.Flush(); err != nil {
        return fmt.Errorf("failed to flush buffer: %v", err)
    }

    return nil
}

func LoadCompressedSupercluster(filename string) (*Supercluster, error) {
    start := time.Now()
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open file: %v", err)
    }
    defer file.Close()

    bufReader := bufio.NewReaderSize(file, 1024*1024)
    dec, err := zstd.NewReader(bufReader)
    if err != nil {
        return nil, fmt.Errorf("failed to create zstd reader: %v", err)
    }
    defer dec.Close()

    // Read sizes
    var numNodes, numPoints, numMetrics uint32
    binary.Read(dec, binary.LittleEndian, &numNodes)
    binary.Read(dec, binary.LittleEndian, &numPoints)
    binary.Read(dec, binary.LittleEndian, &numMetrics)

    // Read options
    var options SuperclusterOptions
    binary.Read(dec, binary.LittleEndian, &options.MinZoom)
    binary.Read(dec, binary.LittleEndian, &options.MaxZoom)
    binary.Read(dec, binary.LittleEndian, &options.MinPoints)
    binary.Read(dec, binary.LittleEndian, &options.Radius)
    binary.Read(dec, binary.LittleEndian, &options.NodeSize)
    binary.Read(dec, binary.LittleEndian, &options.Extent)

    // Create cluster with options
    sc := NewSupercluster(options)

    // Read nodes
    nodes := make([]KDNode, numNodes)
    for i := range nodes {
        binary.Read(dec, binary.LittleEndian, &nodes[i].PointIdx)
        binary.Read(dec, binary.LittleEndian, &nodes[i].Left)
        binary.Read(dec, binary.LittleEndian, &nodes[i].Right)
        binary.Read(dec, binary.LittleEndian, &nodes[i].Axis)
        binary.Read(dec, binary.LittleEndian, &nodes[i].MinChild)
        binary.Read(dec, binary.LittleEndian, &nodes[i].MaxChild)
    }

    fmt.Printf("Nodes read took: %v\n", time.Since(start))
    pointsStart := time.Now()

    // Read points
    points := make([]KDPoint, numPoints)
    for i := range points {
        binary.Read(dec, binary.LittleEndian, &points[i].X)
        binary.Read(dec, binary.LittleEndian, &points[i].Y)
        binary.Read(dec, binary.LittleEndian, &points[i].ID)
        binary.Read(dec, binary.LittleEndian, &points[i].NumPoints)
        binary.Read(dec, binary.LittleEndian, &points[i].MetricIdx)
        
        // Read metadata
        var metadataSize uint32
        binary.Read(dec, binary.LittleEndian, &metadataSize)
        
        points[i].Metadata = make(map[string]interface{}, metadataSize)
        
        // Read each metadata key-value pair
        for j := uint32(0); j < metadataSize; j++ {
            // Read key
            var keySize uint32
            binary.Read(dec, binary.LittleEndian, &keySize)
            keyBytes := make([]byte, keySize)
            io.ReadFull(dec, keyBytes)
            
            // Read value
            var valueSize uint32
            binary.Read(dec, binary.LittleEndian, &valueSize)
            valueBytes := make([]byte, valueSize)
            io.ReadFull(dec, valueBytes)
            
            // Unmarshal value
            var value interface{}
            if err := json.Unmarshal(valueBytes, &value); err != nil {
                return nil, fmt.Errorf("failed to unmarshal metadata value: %v", err)
            }
            
            points[i].Metadata[string(keyBytes)] = value
        }
    }

    fmt.Printf("Points read took: %v\n", time.Since(pointsStart))
    metricsStart := time.Now()

    // Read metrics pool
    metricsPool := NewMetricsPool()
    metricsPool.Metrics = make([]map[string]float32, numMetrics)
    
    for i := range metricsPool.Metrics {
        var numPairs uint32
        binary.Read(dec, binary.LittleEndian, &numPairs)
        
        metrics := make(map[string]float32, numPairs)
        for j := uint32(0); j < numPairs; j++ {
            var keyLen uint32
            binary.Read(dec, binary.LittleEndian, &keyLen)
            
            keyBytes := make([]byte, keyLen)
            io.ReadFull(dec, keyBytes)
            
            var value float32
            binary.Read(dec, binary.LittleEndian, &value)
            
            metrics[string(keyBytes)] = value
        }
        metricsPool.Metrics[i] = metrics
    }

    fmt.Printf("Metrics read took: %v\n", time.Since(metricsStart))

    sc.Tree = &KDTree{
        Pool:     metricsPool,
        NodeSize: options.NodeSize,
        Nodes:    nodes,
        Points:   points,
    }

    fmt.Printf("Total load time: %v\n", time.Since(start))
    return sc, nil
}
