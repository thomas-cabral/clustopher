package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
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
	Tree      *KDTree // Single KD-tree for all zoom levels
	Points    []Point // Original input points
	Options   SuperclusterOptions
	zoomScale []float64 // Pre-calculated zoom scales
	latLookup []float32 // Pre-calculated latitude projections
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

// Add object pools for frequently allocated structures
var (
	clusterNodePool = sync.Pool{
		New: func() interface{} {
			return &ClusterNode{
				Metrics: ClusterMetrics{
					Values: make(map[string]float32, 8), // Pre-allocate with typical size
				},
				Metadata: make(map[string]json.RawMessage, 4),
			}
		},
	}

	kdPointPool = sync.Pool{
		New: func() interface{} {
			return &KDPoint{
				Metadata: make(map[string]interface{}, 4),
			}
		},
	}
)

// Pre-allocate buffers for JSON encoding/decoding
var jsonBufferPool = sync.Pool{
	New: func() interface{} {
		return make([]byte, 0, 1024)
	},
}

// Add a point slice pool to reuse allocations
var pointSlicePool = sync.Pool{
	New: func() interface{} {
		s := make([]KDPoint, 0, 1024) // Initial reasonable capacity
		return &s
	},
}

// Add string interning pool
var stringPool = sync.Pool{
	New: func() interface{} {
		return make(map[string]string, 100)
	},
}

// Add a grid cell pool to reuse slice allocations
var gridCellPool = sync.Pool{
	New: func() interface{} {
		s := make([]int, 0, 32) // Typical cell capacity
		return &s
	},
}

// Add lookup table for common latitude values
const (
	latTableSize = 1024
	latTableStep = 180.0 / float32(latTableSize)
)

// Pre-allocate common metadata keys
var commonMetadataKeys = sync.Pool{
	New: func() interface{} {
		return make(map[string]struct{}, 16)
	},
}

// Add buffer pools for loading
var (
	readBufferPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 32*1024) // 32KB chunks
		},
	}

	pointsBufferPool = sync.Pool{
		New: func() interface{} {
			return make([]KDPoint, 0, 1024)
		},
	}
)

// Add a metadata value pool
var metadataValuePool = sync.Pool{
	New: func() interface{} {
		return make(map[interface{}]int, 4)
	},
}

// ToGeoJSON converts clusters to GeoJSON format
func (sc *Supercluster) ToGeoJSON(bounds KDBounds, zoom int) (*FeatureCollection, error) {
	// Get clusters for the given bounds and zoom level
	clusters := sc.GetClusters(bounds, zoom)

	// Convert clusters to GeoJSON features
	features := make([]Feature, len(clusters))
	for i, c := range clusters {
		// Create properties map
		properties := make(map[string]interface{})
		properties["cluster"] = c.Count > 1

		if c.Count > 1 {
			// Cluster properties
			properties["cluster_id"] = c.ID
			properties["point_count"] = c.Count
		} else {
			// Individual point properties
			properties["id"] = c.ID

			// Add metadata if it exists
			if c.Metadata != nil {
				for k, v := range c.Metadata {
					// Unmarshal the JSON raw message
					var freqMap map[string]float64
					if err := json.Unmarshal(v, &freqMap); err == nil {
						// If there's exactly one value with 100% frequency, use that value
						if len(freqMap) == 1 {
							for value, freq := range freqMap {
								if math.Abs(freq-1.0) < 0.0001 {
									properties[k] = value
									continue
								}
							}
						}
						// Otherwise use the frequency map
						properties[k] = freqMap
					}
				}
			}
		}

		// Add metrics if they exist
		if c.Metrics.Values != nil {
			for k, v := range c.Metrics.Values {
				properties[k] = v
			}
		}

		features[i] = Feature{
			Type: "Feature",
			Geometry: Geometry{
				Type:        "Point",
				Coordinates: []float64{float64(c.X), float64(c.Y)},
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

	sc := &Supercluster{
		Options:   options,
		zoomScale: make([]float64, options.MaxZoom+1),
		latLookup: make([]float32, latTableSize+1),
	}

	// Pre-calculate zoom scales
	for z := 0; z <= options.MaxZoom; z++ {
		sc.zoomScale[z] = math.Pow(2, float64(z))
	}

	for i := 0; i <= latTableSize; i++ {
		lat := -90.0 + float64(i)*float64(latTableStep)
		// Clamp latitude to avoid infinity
		if lat > 85.0511 {
			lat = 85.0511
		} else if lat < -85.0511 {
			lat = -85.0511
		}

		latRad := lat * math.Pi / 180.0
		sin := math.Sin(latRad)
		y := 0.5 - 0.25*math.Log((1+sin)/(1-sin))/math.Pi
		sc.latLookup[i] = float32(y)
	}

	return sc
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
	if sc == nil {
		return nil
	}
	fmt.Printf("Getting clusters for zoom level %d\n", zoom)
	fmt.Printf("Bounds: MinX: %f, MinY: %f, MaxX: %f, MaxY: %f\n",
		bounds.MinX, bounds.MinY, bounds.MaxX, bounds.MaxY)
	fmt.Printf("Total points in tree: %d\n", len(sc.Tree.Points))

	// Estimate number of clusters based on viewport size
	estimatedClusters := sc.estimateClusterCount(bounds, zoom)
	clusters := make([]ClusterNode, 0, estimatedClusters)

	// Get a slice from the pool
	pointsPtr := pointSlicePool.Get().(*[]KDPoint)
	points := (*pointsPtr)[:0] // Reset length but keep capacity
	defer pointSlicePool.Put(pointsPtr)

	// Pre-calculate projection for bounds once
	minP := sc.projectFast(bounds.MinX, bounds.MaxY, zoom)
	maxP := sc.projectFast(bounds.MaxX, bounds.MinY, zoom)

	// Project points in batches to improve cache locality
	const batchSize = 1024
	projectedBatch := make([][2]float32, batchSize)

	for i := 0; i < len(sc.Tree.Points); i += batchSize {
		end := i + batchSize
		if end > len(sc.Tree.Points) {
			end = len(sc.Tree.Points)
		}

		// Project batch
		batch := sc.Tree.Points[i:end]
		for j, p := range batch {
			projectedBatch[j] = sc.projectFast(p.X, p.Y, zoom)
		}

		// Filter points in bounds
		for j, proj := range projectedBatch[:end-i] {
			if proj[0] >= minP[0] && proj[0] <= maxP[0] &&
				proj[1] >= minP[1] && proj[1] <= maxP[1] {
				points = append(points, sc.Tree.Points[i+j])
			}
		}
	}

	// Project and cluster points
	projectedPoints := sc.projectPoints(points, zoom, sc.Tree.Pool)
	clusters = append(clusters, sc.clusterPoints(projectedPoints, float32(sc.Options.Radius))...)

	// Convert back to lng/lat
	for i := range clusters {
		unproj := sc.unprojectFast(clusters[i].X, clusters[i].Y, zoom)
		clusters[i].X = unproj[0]
		clusters[i].Y = unproj[1]
	}

	return clusters
}

func (sc *Supercluster) estimateClusterCount(bounds KDBounds, zoom int) int {
	// Simple estimation based on viewport size and zoom level
	viewportArea := float64((bounds.MaxX - bounds.MinX) * (bounds.MaxY - bounds.MinY))
	totalArea := float64((sc.Tree.Bounds.MaxX - sc.Tree.Bounds.MinX) *
		(sc.Tree.Bounds.MaxY - sc.Tree.Bounds.MinY))

	// Avoid division by zero
	if totalArea <= 0 {
		return 1
	}

	ratio := viewportArea / totalArea
	// Clamp ratio to reasonable bounds
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}

	totalPoints := len(sc.Tree.Points)

	// At high zoom levels, more individual points
	var estimate float64
	if zoom >= sc.Options.MaxZoom-2 {
		estimate = float64(totalPoints) * ratio * 0.75
	} else {
		// At low zoom levels, more clustering
		estimate = float64(totalPoints) * ratio * 0.25
	}

	// Ensure we return a reasonable capacity
	if estimate <= 0 {
		return 1
	}
	if estimate > float64(totalPoints) {
		return totalPoints
	}

	return int(estimate)
}

func (sc *Supercluster) clusterPoints(points []KDPoint, radius float32) []ClusterNode {
	fmt.Printf("Clustering %d points with radius %f\n", len(points), radius)

	var clusters []ClusterNode
	processed := make(map[uint32]bool)

	// Use a smaller grid cell size to ensure we don't miss nearby points
	cellSize := radius / 2 // Half the radius for better accuracy

	// Pre-allocate grid with more conservative size estimate
	estimatedCells := int(float64(len(points)) * 0.25) // Increase from 0.1 to 0.25
	grid := make(map[[2]int]*[]int, estimatedCells)
	defer func() {
		for _, indices := range grid {
			*indices = (*indices)[:0]
			gridCellPool.Put(indices)
		}
	}()

	// Insert points into grid using smaller cell size
	for i, p := range points {
		cell := [2]int{
			int(p.X / cellSize),
			int(p.Y / cellSize),
		}

		indices, ok := grid[cell]
		if !ok {
			slicePtr := gridCellPool.Get().(*[]int)
			*slicePtr = (*slicePtr)[:0]
			indices = slicePtr
			grid[cell] = indices
		}
		*indices = append(*indices, i)
	}

	for i, p := range points {
		if processed[p.ID] {
			continue
		}

		// Find nearby points
		var nearby []KDPoint
		nearby = append(nearby, p) // Include current point

		// Only look at points that could be within radius based on X coordinate
		for j := i + 1; j < len(points); j++ {
			other := points[j]
			if other.X-p.X > radius {
				break // Points are sorted by X, so we can stop looking
			}
			if processed[other.ID] {
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
			cluster := createCluster(nearby, sc.Tree.Pool)
			clusters = append(clusters, cluster)

			// Mark points as processed
			for _, np := range nearby {
				processed[np.ID] = true
			}
		} else if !processed[p.ID] {
			// Add as individual point
			clusters = append(clusters, ClusterNode{
				ID:       p.ID,
				X:        p.X,
				Y:        p.Y,
				Count:    1,
				Metrics:  ClusterMetrics{Values: sc.Tree.Pool.Get(p.MetricIdx)},
				Metadata: convertMetadataToJSON(p.Metadata),
			})
			processed[p.ID] = true
		}
	}

	fmt.Printf("Created %d clusters from %d points\n", len(clusters), len(points))
	return clusters
}

// Modify metadata conversion to use string interning
func convertMetadataToJSON(metadata map[string]interface{}) map[string]json.RawMessage {
	if metadata == nil {
		return nil
	}

	// Get string pool
	strPool := stringPool.Get().(map[string]string)
	defer func() {
		// Clear and return to pool
		for k := range strPool {
			delete(strPool, k)
		}
		stringPool.Put(strPool)
	}()

	result := make(map[string]json.RawMessage, len(metadata))
	buf := jsonBufferPool.Get().([]byte)
	defer jsonBufferPool.Put(buf)

	for k, v := range metadata {
		// Intern string key
		if interned, ok := strPool[k]; ok {
			k = interned
		} else {
			strPool[k] = k
		}

		frequencies := map[string]float64{
			fmt.Sprintf("%v", v): 1.0,
		}

		// Reuse buffer
		buf = buf[:0]
		buf, err := json.Marshal(frequencies)
		if err == nil {
			// Copy buffer since it will be reused
			jsonCopy := make([]byte, len(buf))
			copy(jsonCopy, buf)
			result[k] = jsonCopy
		}
	}
	return result
}

// Modify createCluster to use object pool
func createCluster(points []KDPoint, metricsPool *MetricsPool) ClusterNode {
	cluster := clusterNodePool.Get().(*ClusterNode)
	// Reset maps instead of reallocating
	for k := range cluster.Metrics.Values {
		delete(cluster.Metrics.Values, k)
	}
	for k := range cluster.Metadata {
		delete(cluster.Metadata, k)
	}

	var sumX, sumY float64
	metrics := make(map[string]float64)
	var totalPoints uint32
	uniquePoints := make(map[uint32]bool)

	// Get pre-allocated maps
	metadataKeys := commonMetadataKeys.Get().(map[string]struct{})
	defer func() {
		for k := range metadataKeys {
			delete(metadataKeys, k)
		}
		commonMetadataKeys.Put(metadataKeys)
	}()

	// First collect all unique metadata keys
	for _, p := range points {
		if p.Metadata != nil {
			for k := range p.Metadata {
				metadataKeys[k] = struct{}{}
			}
		}
	}

	// Get a pre-allocated map for metadata values
	valueMap := metadataValuePool.Get().(map[interface{}]int)
	defer func() {
		for k := range valueMap {
			delete(valueMap, k)
		}
		metadataValuePool.Put(valueMap)
	}()

	// Calculate frequencies one key at a time to reuse the map
	for key := range metadataKeys {
		for k := range valueMap {
			delete(valueMap, k)
		}

		total := 0
		for _, p := range points {
			if v, ok := p.Metadata[key]; ok {
				count := int(p.NumPoints)
				valueMap[v] += count
				total += count
			}
		}

		// Convert to frequencies
		frequencies := make(map[string]float64, len(valueMap))
		for value, count := range valueMap {
			frequencies[fmt.Sprintf("%v", value)] = float64(count) / float64(total)
		}

		if jsonBytes, err := json.Marshal(frequencies); err == nil {
			cluster.Metadata[key] = jsonBytes
		}
	}

	// First pass: count frequencies of each metadata value
	for _, p := range points {
		if !uniquePoints[p.ID] {
			uniquePoints[p.ID] = true
			totalPoints += p.NumPoints

			weight := float64(p.NumPoints)
			sumX += float64(p.X) * weight
			sumY += float64(p.Y) * weight

			// Get metrics from pool and accumulate
			if pointMetrics := metricsPool.Get(p.MetricIdx); pointMetrics != nil {
				for k, v := range pointMetrics {
					if p.NumPoints > 1 {
						// For clusters, v is already the total sum
						metrics[k] += float64(v)
					} else {
						// For individual points, add the raw value
						metrics[k] += float64(v)
					}
				}
			}

			// Count frequencies of metadata values
			if p.Metadata != nil {
				for _, v := range p.Metadata {
					if _, exists := valueMap[v]; !exists {
						valueMap[v] = 0
					}
					valueMap[v] += int(p.NumPoints)
				}
			}
		}
	}

	// Calculate position averages
	invTotal := 1.0 / float64(totalPoints)
	cluster.X = float32(sumX * invTotal)
	cluster.Y = float32(sumY * invTotal)
	cluster.Count = totalPoints

	// Store summed metrics
	for k, sum := range metrics {
		cluster.Metrics.Values[k] = float32(sum)
	}

	// Don't put back in pool if it will be used
	// clusterNodePool.Put(cluster)
	return *cluster
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

// projectFast converts lng/lat to tile coordinates
func (sc *Supercluster) projectFast(lng, lat float32, zoom int) [2]float32 {
	if zoom < 0 || zoom >= len(sc.zoomScale) {
		zoom = 0
	}

	scale := float32(sc.zoomScale[zoom])
	extent := float32(sc.Options.Extent)
	scaleExtent := scale * extent

	// Clamp latitude to avoid infinity in Mercator projection
	if lat > 85.0511 {
		lat = 85.0511
	} else if lat < -85.0511 {
		lat = -85.0511
	}

	// Use lookup table with interpolation for better accuracy
	latIdx := int((lat + 90.0) / latTableStep)
	if latIdx < 0 {
		latIdx = 0
	} else if latIdx >= latTableSize {
		latIdx = latTableSize - 1
	}

	// Linear interpolation between lookup table values
	latFrac := (lat+90.0)/latTableStep - float32(latIdx)
	y1 := sc.latLookup[latIdx]
	y2 := sc.latLookup[latIdx+1]
	y := (y1 + (y2-y1)*latFrac) * scaleExtent

	x := (lng + 180.0) * (scaleExtent / 360.0)
	return [2]float32{x, y}
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

// projectPoints converts points from lng/lat to tile coordinates
func (sc *Supercluster) projectPoints(points []KDPoint, zoom int, metricsPool *MetricsPool) []KDPoint {
	projectedPoints := make([]KDPoint, len(points))

	for i, p := range points {
		proj := sc.projectFast(p.X, p.Y, zoom)

		// Create projected point keeping all other properties
		projectedPoints[i] = KDPoint{
			X:         proj[0],
			Y:         proj[1],
			ID:        p.ID,
			NumPoints: p.NumPoints,
			MetricIdx: p.MetricIdx,
			Metadata:  p.Metadata,
		}
	}

	return projectedPoints
}
