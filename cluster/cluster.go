package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"sort"

	"sync"
	"time"
)

// Global string interning pool to deduplicate strings
var globalStringPool = &StringPool{
	strings: make(map[string]string, 10000),
}

// StringPool provides global string interning
type StringPool struct {
	strings map[string]string
	mu      sync.RWMutex
}

// Intern returns a single instance of a string to reduce memory usage
func (p *StringPool) Intern(s string) string {
	// First check without a write lock
	p.mu.RLock()
	interned, ok := p.strings[s]
	p.mu.RUnlock()

	if ok {
		return interned
	}

	// Need to add it - acquire write lock
	p.mu.Lock()
	defer p.mu.Unlock()

	// Check again to handle race conditions
	if interned, ok = p.strings[s]; ok {
		return interned
	}

	// Store and return the string itself (which becomes the interned version)
	p.strings[s] = s
	return s
}

// MetadataKey represents a string value in a compact way
type MetadataKey uint32

// MetadataValue represents a value that can be one of several types
type MetadataValue struct {
	// Only one of these fields is used depending on the type
	StringVal string
	NumVal    float64
	BoolVal   bool
	Type      byte // 0=string, 1=number, 2=bool
}

// MetadataStore provides efficient storage of metadata
type MetadataStore struct {
	// Maps from key string to key ID
	keyToID map[string]MetadataKey
	// Maps from key ID to original string
	idToKey []string
	// Maps from point ID to a metadata record
	pointMeta map[uint32][]MetadataEntry
	// Pool for string interning
	stringPool *StringPool

	nextKeyID MetadataKey
	mu        sync.RWMutex
}

// MetadataEntry is a compact key-value pair
type MetadataEntry struct {
	Key   MetadataKey
	Value MetadataValue
}

// NewMetadataStore creates a new metadata store
func NewMetadataStore() *MetadataStore {
	return &MetadataStore{
		keyToID:    make(map[string]MetadataKey),
		idToKey:    make([]string, 0, 100),
		pointMeta:  make(map[uint32][]MetadataEntry),
		stringPool: globalStringPool,
	}
}

// AddMetadata adds metadata for a point
func (ms *MetadataStore) AddMetadata(pointID uint32, metadata map[string]interface{}) {
	if len(metadata) == 0 {
		return
	}

	ms.mu.Lock()
	defer ms.mu.Unlock()

	entries := make([]MetadataEntry, 0, len(metadata))

	for k, v := range metadata {
		// Intern the key string to save memory
		k = ms.stringPool.Intern(k)

		// Get or create key ID
		keyID, ok := ms.keyToID[k]
		if !ok {
			keyID = ms.nextKeyID
			ms.nextKeyID++
			ms.keyToID[k] = keyID
			ms.idToKey = append(ms.idToKey, k)
		}

		// Convert and store the value based on its type
		var metaValue MetadataValue

		switch val := v.(type) {
		case string:
			metaValue.StringVal = ms.stringPool.Intern(val)
			metaValue.Type = 0
		case float64:
			metaValue.NumVal = val
			metaValue.Type = 1
		case float32:
			metaValue.NumVal = float64(val)
			metaValue.Type = 1
		case int:
			metaValue.NumVal = float64(val)
			metaValue.Type = 1
		case int32:
			metaValue.NumVal = float64(val)
			metaValue.Type = 1
		case int64:
			metaValue.NumVal = float64(val)
			metaValue.Type = 1
		case bool:
			metaValue.BoolVal = val
			metaValue.Type = 2
		default:
			// Skip unsupported types
			continue
		}

		entries = append(entries, MetadataEntry{
			Key:   keyID,
			Value: metaValue,
		})
	}

	if len(entries) > 0 {
		ms.pointMeta[pointID] = entries
	}
}

// GetMetadata retrieves metadata for a point
func (ms *MetadataStore) GetMetadata(pointID uint32) map[string]interface{} {
	ms.mu.RLock()
	entries, ok := ms.pointMeta[pointID]
	ms.mu.RUnlock()

	if !ok || len(entries) == 0 {
		return nil
	}

	result := make(map[string]interface{}, len(entries))

	for _, entry := range entries {
		if int(entry.Key) >= len(ms.idToKey) {
			continue
		}

		key := ms.idToKey[entry.Key]

		switch entry.Value.Type {
		case 0: // string
			result[key] = entry.Value.StringVal
		case 1: // number
			result[key] = entry.Value.NumVal
		case 2: // bool
			result[key] = entry.Value.BoolVal
		}
	}

	return result
}

// GetMetadataAsJSON returns metadata in JSON format
func (ms *MetadataStore) GetMetadataAsJSON(pointID uint32) map[string]json.RawMessage {
	meta := ms.GetMetadata(pointID)
	if meta == nil {
		return nil
	}

	result := make(map[string]json.RawMessage, len(meta))

	for k, v := range meta {
		if jsonBytes, err := json.Marshal(v); err == nil {
			result[k] = jsonBytes
		}
	}

	return result
}

// CalculateFrequencies calculates frequency distributions for cluster metadata
func (ms *MetadataStore) CalculateFrequencies(pointIDs []uint32) map[string]json.RawMessage {
	if len(pointIDs) == 0 {
		return nil
	}

	ms.mu.RLock()
	defer ms.mu.RUnlock()

	// Count unique keys across all points
	keySet := make(map[MetadataKey]bool)
	for _, id := range pointIDs {
		entries, ok := ms.pointMeta[id]
		if !ok {
			continue
		}

		for _, entry := range entries {
			keySet[entry.Key] = true
		}
	}

	if len(keySet) == 0 {
		return nil
	}

	result := make(map[string]json.RawMessage, len(keySet))

	// For each key, calculate value frequencies
	for keyID := range keySet {
		if int(keyID) >= len(ms.idToKey) {
			continue
		}

		key := ms.idToKey[keyID]

		// Count value frequencies
		valueFreq := make(map[string]int)
		totalPoints := 0

		for _, id := range pointIDs {
			entries, ok := ms.pointMeta[id]
			if !ok {
				continue
			}

			for _, entry := range entries {
				if entry.Key != keyID {
					continue
				}

				// Convert value to string representation
				var valueStr string
				switch entry.Value.Type {
				case 0: // string
					valueStr = entry.Value.StringVal
				case 1: // number
					valueStr = fmt.Sprintf("%g", entry.Value.NumVal)
				case 2: // bool
					valueStr = fmt.Sprintf("%t", entry.Value.BoolVal)
				}

				valueFreq[valueStr]++
				totalPoints++
			}
		}

		// Convert to frequency percentages
		freqMap := make(map[string]float64, len(valueFreq))
		for val, count := range valueFreq {
			freqMap[val] = float64(count) / float64(totalPoints)
		}

		// Marshal to JSON
		if jsonBytes, err := json.Marshal(freqMap); err == nil {
			result[key] = jsonBytes
		}
	}

	return result
}

// Memory-efficient metrics storage
type MetricsStore struct {
	// Maps metric key to column index
	keyToColumn map[string]int
	// Original metric keys
	keys []string
	// Each column stores values for a specific metric
	columns [][]float32
	// Maps point ID to row index
	pointToRow map[uint32]int
	stringPool *StringPool
	mu         sync.RWMutex
}

// NewMetricsStore creates a new metrics store
func NewMetricsStore() *MetricsStore {
	return &MetricsStore{
		keyToColumn: make(map[string]int),
		keys:        make([]string, 0, 16),
		columns:     make([][]float32, 0, 16),
		pointToRow:  make(map[uint32]int),
		stringPool:  globalStringPool,
	}
}

// AddMetrics adds metrics for a point
func (ms *MetricsStore) AddMetrics(pointID uint32, metrics map[string]float32) int {
	if len(metrics) == 0 {
		return -1
	}

	ms.mu.Lock()
	defer ms.mu.Unlock()

	// Get or create row for this point
	rowIdx, ok := ms.pointToRow[pointID]
	if !ok {
		rowIdx = len(ms.pointToRow)
		ms.pointToRow[pointID] = rowIdx
	}

	// Add metrics to columns
	for k, v := range metrics {
		// Intern the key to save memory
		k = ms.stringPool.Intern(k)

		// Get or create column
		colIdx, ok := ms.keyToColumn[k]
		if !ok {
			colIdx = len(ms.keys)
			ms.keys = append(ms.keys, k)
			ms.columns = append(ms.columns, make([]float32, rowIdx+1))
			ms.keyToColumn[k] = colIdx
		}

		// Ensure column has enough rows
		if rowIdx >= len(ms.columns[colIdx]) {
			// Extend column with zeroes
			newSize := rowIdx + 1
			if newSize < len(ms.columns[colIdx])*2 {
				newSize = len(ms.columns[colIdx]) * 2
			}

			newColumn := make([]float32, newSize)
			copy(newColumn, ms.columns[colIdx])
			ms.columns[colIdx] = newColumn
		}

		// Store metric value
		ms.columns[colIdx][rowIdx] = v
	}

	return rowIdx
}

// GetMetrics retrieves metrics for a point
func (ms *MetricsStore) GetMetrics(pointID uint32) map[string]float32 {
	ms.mu.RLock()
	rowIdx, ok := ms.pointToRow[pointID]
	ms.mu.RUnlock()

	if !ok {
		return nil
	}

	ms.mu.RLock()
	defer ms.mu.RUnlock()

	result := make(map[string]float32, len(ms.keys))

	for colIdx, key := range ms.keys {
		if rowIdx < len(ms.columns[colIdx]) {
			result[key] = ms.columns[colIdx][rowIdx]
		}
	}

	return result
}

// Memory-optimized KDPoint structure
type KDPoint struct {
	X, Y      float32 // 8 bytes
	ID        uint32  // 4 bytes
	NumPoints uint32  // 4 bytes - used for clusters
}

// Point represents an input point with metadata
type Point struct {
	ID       uint32
	X, Y     float32
	Metrics  map[string]float32
	Metadata map[string]interface{}
}

type KDNode struct {
	PointIdx int32   // 4 bytes - index into points array
	Left     int32   // 4 bytes - index into nodes array
	Right    int32   // 4 bytes - index into nodes array
	Axis     uint8   // 1 byte  - 0 or 1 is sufficient
	_        [3]byte // Padding for alignment
	MinChild uint32  // 4 bytes
	MaxChild uint32  // 4 bytes
	Bounds   KDBounds
}

// KDBounds defines a bounding box
type KDBounds struct {
	MinX, MinY, MaxX, MaxY float32
}

// Extend expands bounds to include a point
func (b *KDBounds) Extend(x, y float32) {
	b.MinX = float32(math.Min(float64(b.MinX), float64(x)))
	b.MinY = float32(math.Min(float64(b.MinY), float64(y)))
	b.MaxX = float32(math.Max(float64(b.MaxX), float64(x)))
	b.MaxY = float32(math.Max(float64(b.MaxY), float64(y)))
}

// intersectsBounds checks if this bounds intersects another
func (b KDBounds) intersectsBounds(other KDBounds) bool {
	return b.MaxX >= other.MinX && b.MinX <= other.MaxX &&
		b.MaxY >= other.MinY && b.MinY <= other.MaxY
}

// KDTree is a k-d tree spatial index
type KDTree struct {
	Nodes    []KDNode  // All nodes in a single slice
	Points   []KDPoint // All points in a single slice
	NodeSize int
	Bounds   KDBounds
}

// ClusterNode represents a cluster of points
type ClusterNode struct {
	ID       uint32
	X, Y     float32
	Count    uint32
	Children []uint32
	Metrics  map[string]float32
	Metadata map[string]json.RawMessage
}

// Supercluster implements the clustering algorithm
type Supercluster struct {
	Tree          *KDTree // K-d tree for spatial queries
	Points        []Point // Original input points
	Options       SuperclusterOptions
	zoomScale     []float64      // Pre-calculated zoom scales
	latLookup     []float32      // Pre-calculated latitude projections
	metadataStore *MetadataStore // Efficient metadata storage
	metricsStore  *MetricsStore  // Efficient metrics storage
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

// Object pools to reduce memory allocations
var (
	// Stack pool for tree traversal
	stackPool = sync.Pool{
		New: func() interface{} {
			s := make([]int32, 0, 64)
			return &s
		},
	}

	// Buffer pools
	pointSlicePool = sync.Pool{
		New: func() interface{} {
			s := make([]KDPoint, 0, 1024)
			return &s
		},
	}
)

// Constants
const (
	latTableSize = 1024
	latTableStep = 180.0 / float32(latTableSize)
)

// NewSupercluster creates a new clustering instance
func NewSupercluster(options SuperclusterOptions) *Supercluster {
	// Set default values
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
		Options:       options,
		zoomScale:     make([]float64, options.MaxZoom+1),
		latLookup:     make([]float32, latTableSize+1),
		metadataStore: NewMetadataStore(),
		metricsStore:  NewMetricsStore(),
	}

	// Pre-calculate zoom scales
	for z := 0; z <= options.MaxZoom; z++ {
		sc.zoomScale[z] = math.Pow(2, float64(z))
	}

	// Fill lookup table for latitude projections
	for i := 0; i <= latTableSize; i++ {
		lat := -90.0 + float64(i)*float64(latTableStep)
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

// Load initializes the cluster index with points
func (sc *Supercluster) Load(points []Point) {
	fmt.Printf("Loading %d points\n", len(points))

	// For large datasets, process in batches
	if len(points) > 1000000 {
		sc.loadBatched(points, 1000000)
		return
	}

	// Convert points to KDPoints
	kdPoints := make([]KDPoint, len(points))

	// Process metrics and metadata
	for i, p := range points {
		// Add metadata to store
		sc.metadataStore.AddMetadata(p.ID, p.Metadata)

		// Add metrics to store
		sc.metricsStore.AddMetrics(p.ID, p.Metrics)

		// Create KDPoint (minimal data needed for spatial operations)
		kdPoints[i] = KDPoint{
			X:         p.X,
			Y:         p.Y,
			ID:        p.ID,
			NumPoints: 1,
		}
	}

	// Build KD-tree
	sc.Tree = sc.buildKDTree(kdPoints)
	sc.Points = points
}

// loadBatched processes points in batches to reduce memory usage
func (sc *Supercluster) loadBatched(points []Point, batchSize int) {
	// Determine number of batches
	numPoints := len(points)
	numBatches := (numPoints + batchSize - 1) / batchSize

	fmt.Printf("Processing in %d batches of size %d\n", numBatches, batchSize)

	// First collect all points with minimal data to build tree
	allKdPoints := make([]KDPoint, numPoints)

	for i, p := range points {
		allKdPoints[i] = KDPoint{
			X:         p.X,
			Y:         p.Y,
			ID:        p.ID,
			NumPoints: 1,
		}
	}

	// Build tree with all points
	sc.Tree = sc.buildKDTree(allKdPoints)

	// Now process metadata and metrics in batches
	for i := 0; i < numPoints; i += batchSize {
		end := i + batchSize
		if end > numPoints {
			end = numPoints
		}

		fmt.Printf("Processing batch %d/%d (points %d-%d)\n",
			i/batchSize+1, numBatches, i, end-1)

		// Process batch
		for j := i; j < end; j++ {
			p := points[j]
			sc.metadataStore.AddMetadata(p.ID, p.Metadata)
			sc.metricsStore.AddMetrics(p.ID, p.Metrics)
		}

		// Force GC between batches
		if i > 0 && i%(batchSize*5) == 0 {
			runtime.GC()
			debug.FreeOSMemory()
		}
	}

	sc.Points = points
}

// buildKDTree constructs a KD-tree from points
func (sc *Supercluster) buildKDTree(points []KDPoint) *KDTree {
	maxNodes := len(points) * 2 // Worst case for a binary tree
	tree := &KDTree{
		Nodes:    make([]KDNode, 0, maxNodes),
		Points:   make([]KDPoint, len(points)),
		NodeSize: sc.Options.NodeSize,
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

// buildNodes constructs the KD-tree recursively
func (t *KDTree) buildNodes(start, end, depth int) int32 {
	if start > end {
		return -1
	}

	nodeIdx := int32(len(t.Nodes))
	t.Nodes = append(t.Nodes, KDNode{})
	node := &t.Nodes[nodeIdx]

	// Calculate node bounds
	nodeBounds := KDBounds{
		MinX: float32(math.Inf(1)),
		MinY: float32(math.Inf(1)),
		MaxX: float32(math.Inf(-1)),
		MaxY: float32(math.Inf(-1)),
	}

	for i := start; i <= end; i++ {
		nodeBounds.Extend(t.Points[i].X, t.Points[i].Y)
	}

	node.Bounds = nodeBounds

	// Leaf node
	if end-start <= t.NodeSize {
		node.PointIdx = int32(start)
		node.Left = -1
		node.Right = -1
		setMinMaxChild(node, t.Points[start:end+1])
		return nodeIdx
	}

	// Split node
	axis := depth % 2
	median := (start + end) / 2

	// Use quickselect for large datasets
	if end-start > 1000 {
		quickselectKDPoints(t.Points[start:end+1], median-start, axis)
	} else {
		sortPointsRange(t.Points[start:end+1], axis)
	}

	node.PointIdx = int32(median)
	node.Axis = uint8(axis)

	// Recursively build left and right subtrees
	node.Left = t.buildNodes(start, median-1, depth+1)
	node.Right = t.buildNodes(median+1, end, depth+1)

	setMinMaxChild(node, t.Points[start:end+1])
	return nodeIdx
}

// Helper functions for KD-tree construction
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

// quickselectKDPoints finds the k-th smallest element by axis
func quickselectKDPoints(points []KDPoint, k int, axis int) {
	// Base cases
	if len(points) <= 1 {
		return
	}

	// Choose pivot and partition
	pivotIdx := partition(points, axis)

	if k == pivotIdx {
		return
	} else if k < pivotIdx {
		quickselectKDPoints(points[:pivotIdx], k, axis)
	} else {
		quickselectKDPoints(points[pivotIdx+1:], k-pivotIdx-1, axis)
	}
}

// partition divides array around pivot
func partition(points []KDPoint, axis int) int {
	// Choose pivot (median of 3)
	n := len(points)
	mid := n / 2

	// Sort first, middle, and last elements
	if getAxisValue(points[0], axis) > getAxisValue(points[mid], axis) {
		points[0], points[mid] = points[mid], points[0]
	}
	if getAxisValue(points[0], axis) > getAxisValue(points[n-1], axis) {
		points[0], points[n-1] = points[n-1], points[0]
	}
	if getAxisValue(points[mid], axis) > getAxisValue(points[n-1], axis) {
		points[mid], points[n-1] = points[n-1], points[mid]
	}

	// Use middle element as pivot
	pivot := getAxisValue(points[mid], axis)

	// Move pivot to end
	points[mid], points[n-1] = points[n-1], points[mid]

	// Partition
	i := 0
	for j := 0; j < n-1; j++ {
		if getAxisValue(points[j], axis) <= pivot {
			points[i], points[j] = points[j], points[i]
			i++
		}
	}

	// Move pivot to its final place
	points[i], points[n-1] = points[n-1], points[i]

	return i
}

// getAxisValue returns X or Y value based on axis
func getAxisValue(p KDPoint, axis int) float32 {
	if axis == 0 {
		return p.X
	}
	return p.Y
}

// CleanupCluster releases memory
func (sc *Supercluster) CleanupCluster() {
	if sc == nil {
		return
	}

	// Clear tree
	if sc.Tree != nil {
		sc.Tree.Nodes = nil
		sc.Tree.Points = nil
		sc.Tree = nil
	}

	// Clear all points
	sc.Points = nil

	// Clear storage
	sc.metadataStore = NewMetadataStore()
	sc.metricsStore = NewMetricsStore()

	// Force GC
	runtime.GC()
	debug.FreeOSMemory()
}

// GetClusters returns clusters for the given bounds and zoom level
func (sc *Supercluster) GetClusters(bounds KDBounds, zoom int) []ClusterNode {
	if sc == nil {
		return nil
	}

	startTime := time.Now()

	if sc.Options.Log {
		fmt.Printf("Getting clusters for zoom level %d\n", zoom)
		fmt.Printf("Bounds: MinX: %f, MinY: %f, MaxX: %f, MaxY: %f\n",
			bounds.MinX, bounds.MinY, bounds.MaxX, bounds.MaxY)
	}

	if sc.Tree == nil || len(sc.Tree.Points) == 0 {
		if sc.Options.Log {
			fmt.Printf("No points in tree\n")
		}
		return nil
	}

	if sc.Options.Log {
		fmt.Printf("Total points in tree: %d\n", len(sc.Tree.Points))
	}

	// Get a slice from the pool for points
	pointsPtr := pointSlicePool.Get().(*[]KDPoint)
	points := (*pointsPtr)[:0] // Reset length but keep capacity
	defer pointSlicePool.Put(pointsPtr)

	// Pre-calculate projection for bounds
	minP := sc.projectFast(bounds.MinX, bounds.MaxY, zoom)
	maxP := sc.projectFast(bounds.MaxX, bounds.MinY, zoom)

	// Create search bounds
	searchBounds := KDBounds{
		MinX: minP[0],
		MinY: minP[1],
		MaxX: maxP[0],
		MaxY: maxP[1],
	}

	// Find points in viewport
	filterTime := time.Now()
	points = sc.findPointsInViewport(searchBounds, zoom, points)
	if sc.Options.Log {
		fmt.Printf("Found %d points in %.2fms\n",
			len(points), float64(time.Since(filterTime).Milliseconds()))
	}

	if len(points) == 0 {
		return nil
	}

	// Estimate clusters and create result array
	estimatedClusters := min(len(points), max(10, len(points)/10))
	clusters := make([]ClusterNode, 0, estimatedClusters)

	// Cluster points
	clusterTime := time.Now()
	if len(points) > 50000 {
		clusters = append(clusters, sc.clusterPointsWithGrid(points, float32(sc.Options.Radius))...)
	} else {
		clusters = append(clusters, sc.clusterPoints(points, float32(sc.Options.Radius))...)
	}
	if sc.Options.Log {
		fmt.Printf("Clustering created %d clusters in %.2fms\n",
			len(clusters), float64(time.Since(clusterTime).Milliseconds()))
	}

	// Convert back to lng/lat
	unprojTime := time.Now()
	sc.unprojectClusters(clusters, zoom)
	if sc.Options.Log {
		fmt.Printf("Unprojection completed in %.2fms\n",
			float64(time.Since(unprojTime).Milliseconds()))
		fmt.Printf("Total processing time: %.2fms\n",
			float64(time.Since(startTime).Milliseconds()))
	}

	return clusters
}

// findPointsInViewport finds points within the viewport
func (sc *Supercluster) findPointsInViewport(bounds KDBounds, zoom int, result []KDPoint) []KDPoint {
	// For large datasets, we use different strategies based on zoom level
	totalPoints := len(sc.Tree.Points)

	// At high zoom levels (zoomed in), use spatial index
	if zoom >= sc.Options.MaxZoom-2 {
		return sc.findPointsWithSpatialIndex(bounds, zoom, result)
	}

	// At medium zoom levels, use sampling to choose best approach
	if totalPoints > 100000 {
		// Sample points to estimate viewport density
		sampleSize := 1000
		step := totalPoints / sampleSize
		inViewport := 0

		for i := 0; i < totalPoints; i += step {
			p := sc.Tree.Points[i]
			proj := sc.projectFast(p.X, p.Y, zoom)

			if proj[0] >= bounds.MinX && proj[0] <= bounds.MaxX &&
				proj[1] >= bounds.MinY && proj[1] <= bounds.MaxY {
				inViewport++
			}
		}

		// Estimate percentage of points in viewport
		viewportRatio := float64(inViewport) / float64(sampleSize)

		if viewportRatio < 0.01 {
			// Very few points in viewport - use spatial index
			return sc.findPointsWithSpatialIndex(bounds, zoom, result)
		} else if viewportRatio > 0.4 {
			// Many points in viewport - use parallel scan
			return sc.findPointsWithParallelScan(bounds, zoom, result)
		}
	}

	// Default: use sequential scan for medium-sized datasets
	return sc.findPointsWithSequentialScan(bounds, zoom, result)
}

// findPointsWithSpatialIndex uses KD-tree traversal to find points
func (sc *Supercluster) findPointsWithSpatialIndex(bounds KDBounds, zoom int, result []KDPoint) []KDPoint {
	if len(sc.Tree.Nodes) == 0 {
		return result
	}

	// Get stack from pool for tree traversal
	stackPtr := stackPool.Get().(*[]int32)
	stack := (*stackPtr)[:0] // Reset length but keep capacity
	defer stackPool.Put(stackPtr)

	// Start with root node
	stack = append(stack, 0)

	for len(stack) > 0 {
		// Pop node from stack
		nodeIdx := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if nodeIdx < 0 || int(nodeIdx) >= len(sc.Tree.Nodes) {
			continue
		}

		node := &sc.Tree.Nodes[nodeIdx]

		// Skip if node is completely outside bounds
		nodeBounds := sc.projectBounds(node.Bounds, zoom)
		if !bounds.intersectsBounds(nodeBounds) {
			continue
		}

		// If this is a leaf node, check all points
		if node.Left == -1 && node.Right == -1 {
			// Get range of points in this node
			startIdx := int(node.PointIdx)
			endIdx := startIdx + sc.Options.NodeSize
			if endIdx > len(sc.Tree.Points) {
				endIdx = len(sc.Tree.Points)
			}

			// Check each point
			for i := startIdx; i < endIdx; i++ {
				p := sc.Tree.Points[i]
				proj := sc.projectFast(p.X, p.Y, zoom)

				if proj[0] >= bounds.MinX && proj[0] <= bounds.MaxX &&
					proj[1] >= bounds.MinY && proj[1] <= bounds.MaxY {
					// Add projected point to result
					result = append(result, KDPoint{
						X:         proj[0],
						Y:         proj[1],
						ID:        p.ID,
						NumPoints: p.NumPoints,
					})
				}
			}
			continue
		}

		// For non-leaf nodes, add children to stack
		if node.Right != -1 {
			stack = append(stack, node.Right)
		}
		if node.Left != -1 {
			stack = append(stack, node.Left)
		}
	}

	return result
}

// findPointsWithSequentialScan scans all points sequentially
func (sc *Supercluster) findPointsWithSequentialScan(bounds KDBounds, zoom int, result []KDPoint) []KDPoint {
	for i := range sc.Tree.Points {
		p := sc.Tree.Points[i]
		proj := sc.projectFast(p.X, p.Y, zoom)

		if proj[0] >= bounds.MinX && proj[0] <= bounds.MaxX &&
			proj[1] >= bounds.MinY && proj[1] <= bounds.MaxY {
			// Add projected point to result
			result = append(result, KDPoint{
				X:         proj[0],
				Y:         proj[1],
				ID:        p.ID,
				NumPoints: p.NumPoints,
			})
		}
	}

	return result
}

// findPointsWithParallelScan scans points in parallel
func (sc *Supercluster) findPointsWithParallelScan(bounds KDBounds, zoom int, result []KDPoint) []KDPoint {
	numPoints := len(sc.Tree.Points)
	numCPU := runtime.NumCPU()
	pointsPerCPU := (numPoints + numCPU - 1) / numCPU

	// Create channel for results
	resultChan := make(chan []KDPoint, numCPU)

	// Launch workers
	var wg sync.WaitGroup
	for i := 0; i < numCPU; i++ {
		wg.Add(1)

		// Calculate range for this worker
		start := i * pointsPerCPU
		end := start + pointsPerCPU
		if end > numPoints {
			end = numPoints
		}

		go func(start, end int) {
			defer wg.Done()

			// Allocate local result
			localResult := make([]KDPoint, 0, (end-start)/4)

			// Process points
			for i := start; i < end; i++ {
				p := sc.Tree.Points[i]
				proj := sc.projectFast(p.X, p.Y, zoom)

				if proj[0] >= bounds.MinX && proj[0] <= bounds.MaxX &&
					proj[1] >= bounds.MinY && proj[1] <= bounds.MaxY {
					// Add projected point
					localResult = append(localResult, KDPoint{
						X:         proj[0],
						Y:         proj[1],
						ID:        p.ID,
						NumPoints: p.NumPoints,
					})
				}
			}

			resultChan <- localResult
		}(start, end)
	}

	// Wait for workers and close channel
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	for localResult := range resultChan {
		result = append(result, localResult...)
	}

	return result
}

// projectBounds projects a bounding box
func (sc *Supercluster) projectBounds(bounds KDBounds, zoom int) KDBounds {
	minP := sc.projectFast(bounds.MinX, bounds.MaxY, zoom)
	maxP := sc.projectFast(bounds.MaxX, bounds.MinY, zoom)

	return KDBounds{
		MinX: minP[0],
		MinY: minP[1],
		MaxX: maxP[0],
		MaxY: maxP[1],
	}
}

// projectFast converts lng/lat to tile coordinates
func (sc *Supercluster) projectFast(lng, lat float32, zoom int) [2]float32 {
	// Ensure zoom is valid
	if zoom < 0 || zoom >= len(sc.zoomScale) {
		zoom = 0
	}

	// Get the zoom scale
	scale := float64(sc.zoomScale[zoom])
	extent := float64(sc.Options.Extent)

	// Clamp latitude
	if lat > 85.0511 {
		lat = 85.0511
	} else if lat < -85.0511 {
		lat = -85.0511
	}

	// Convert lat/lng to radians
	latRad := float64(lat) * math.Pi / 180.0

	// Mercator projection formula
	sin := math.Sin(latRad)
	y := float32(0.5 - 0.25*math.Log((1.0+sin)/(1.0-sin))/math.Pi)

	// Scale by zoom level and extent
	x := float32((float64(lng) + 180.0) / 360.0 * scale * extent)
	y = float32(float64(y) * scale * extent)

	return [2]float32{x, y}
}

// unprojectFast converts tile coordinates back to lng/lat
func (sc *Supercluster) unprojectFast(x, y float32, zoom int) [2]float32 {
	// Ensure zoom is valid
	if zoom < 0 || zoom >= len(sc.zoomScale) {
		zoom = 0
	}

	// Get the zoom scale
	scale := float64(sc.zoomScale[zoom])
	extent := float64(sc.Options.Extent)

	// Normalize coordinates (0-1)
	x = float32(float64(x) / (scale * extent))
	y = float32(float64(y) / (scale * extent))

	// Convert to lng/lat
	lng := float32(float64(x)*360.0 - 180.0)

	// Reverse the mercator projection
	y2 := (1.0 - float64(y)*2.0) * math.Pi
	latRad := math.Atan(math.Exp(y2))*2.0 - math.Pi/2.0
	lat := float32(latRad * 180.0 / math.Pi)

	return [2]float32{lng, lat}
}

// unprojectClusters converts all clusters back to geographic coordinates
func (sc *Supercluster) unprojectClusters(clusters []ClusterNode, zoom int) {
	// For small sets, don't use parallelism
	if len(clusters) < 100 {
		for i := range clusters {
			unproj := sc.unprojectFast(clusters[i].X, clusters[i].Y, zoom)
			clusters[i].X = unproj[0]
			clusters[i].Y = unproj[1]
		}
		return
	}

	// Use parallelism for larger sets
	numCPU := runtime.NumCPU()
	clusterPerCPU := (len(clusters) + numCPU - 1) / numCPU

	var wg sync.WaitGroup
	wg.Add(numCPU)

	for i := 0; i < numCPU; i++ {
		start := i * clusterPerCPU
		end := start + clusterPerCPU
		if end > len(clusters) {
			end = len(clusters)
		}

		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				unproj := sc.unprojectFast(clusters[i].X, clusters[i].Y, zoom)
				clusters[i].X = unproj[0]
				clusters[i].Y = unproj[1]
			}
		}(start, end)
	}

	wg.Wait()
}

// Original clustering algorithm with memory optimizations
func (sc *Supercluster) clusterPoints(points []KDPoint, radius float32) []ClusterNode {
	if len(points) == 0 {
		return nil
	}

	numPoints := len(points)
	if sc.Options.Log {
		fmt.Printf("Clustering %d points with radius %f\n", numPoints, radius)
	}

	var clusters []ClusterNode
	processed := make(map[uint32]bool, numPoints)

	// Sort points by X coordinate for more efficient nearby point finding
	sort.Slice(points, func(i, j int) bool {
		return points[i].X < points[j].X
	})

	// Temp buffer for nearby points
	nearby := make([]KDPoint, 0, 32)

	// Process each point
	for i, p := range points {
		if processed[p.ID] {
			continue
		}

		// Reset nearby points buffer, keeping capacity
		nearby = nearby[:0]
		nearby = append(nearby, p) // Start with current point

		// Find nearby points
		radiusSquared := radius * radius

		for j := i + 1; j < len(points); j++ {
			other := points[j]

			// Early exit if X distance is greater than radius
			if other.X-p.X > radius {
				break // No need to check further points due to sorting
			}

			if processed[other.ID] {
				continue
			}

			// Check distance
			dx := other.X - p.X
			dy := other.Y - p.Y
			distSq := dx*dx + dy*dy

			if distSq <= radiusSquared {
				nearby = append(nearby, other)
			}
		}

		// Create cluster if enough points
		if len(nearby) >= sc.Options.MinPoints {
			cluster := sc.createCluster(nearby)
			clusters = append(clusters, cluster)

			// Mark points as processed
			for _, np := range nearby {
				processed[np.ID] = true
			}
		} else if !processed[p.ID] {
			// Add as individual point
			clusters = append(clusters, sc.createSinglePointCluster(p))
			processed[p.ID] = true
		}
	}

	if sc.Options.Log {
		fmt.Printf("Created %d clusters from %d points\n", len(clusters), numPoints)
	}
	return clusters
}

// Grid-based clustering for larger datasets
func (sc *Supercluster) clusterPointsWithGrid(points []KDPoint, radius float32) []ClusterNode {
	if len(points) == 0 {
		return nil
	}

	numPoints := len(points)
	if sc.Options.Log {
		fmt.Printf("Clustering %d points with grid (radius %f)\n", numPoints, radius)
	}

	// Pre-allocate result
	clusters := make([]ClusterNode, 0, numPoints/5)
	processed := make(map[uint32]bool, numPoints)

	// Calculate grid bounds
	minX, minY := float32(math.MaxFloat32), float32(math.MaxFloat32)
	maxX, maxY := float32(-math.MaxFloat32), float32(-math.MaxFloat32)

	// Find bounds
	for _, p := range points {
		if p.X < minX {
			minX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}

	// Set cell size slightly smaller than radius for good coverage
	cellSize := radius * 0.75

	// Determine grid dimensions with small padding
	// gridWidth := int((maxX - minX) / cellSize) + 3
	// gridHeight := int((maxY - minY) / cellSize) + 3

	// Create grid
	grid := make(map[[2]int][]int)

	// Insert points into grid
	for i, p := range points {
		cellX := int((p.X-minX)/cellSize) + 1
		cellY := int((p.Y-minY)/cellSize) + 1

		cell := [2]int{cellX, cellY}
		grid[cell] = append(grid[cell], i)
	}

	// Reusable buffer for collecting nearby points
	nearby := make([]KDPoint, 0, 64)

	// Process points
	for _, p := range points {
		if processed[p.ID] {
			continue
		}

		// Get grid cell for this point
		cellX := int((p.X-minX)/cellSize) + 1
		cellY := int((p.Y-minY)/cellSize) + 1

		// Reset nearby points buffer
		nearby = nearby[:0]
		nearby = append(nearby, p)

		// Check 3x3 surrounding cells
		radiusSquared := radius * radius

		for dy := -1; dy <= 1; dy++ {
			ny := cellY + dy

			for dx := -1; dx <= 1; dx++ {
				nx := cellX + dx
				cell := [2]int{nx, ny}

				// Check points in this cell
				if cellIndices, ok := grid[cell]; ok {
					for _, pointIdx := range cellIndices {
						other := points[pointIdx]

						if other.ID == p.ID || processed[other.ID] {
							continue
						}

						// Fast distance check
						dx := other.X - p.X
						dy := other.Y - p.Y
						distSq := dx*dx + dy*dy

						if distSq <= radiusSquared {
							nearby = append(nearby, other)
						}
					}
				}
			}
		}

		// Create cluster if enough points
		if len(nearby) >= sc.Options.MinPoints {
			cluster := sc.createCluster(nearby)
			clusters = append(clusters, cluster)

			// Mark points as processed
			for _, np := range nearby {
				processed[np.ID] = true
			}
		} else if !processed[p.ID] {
			// Add as individual point
			clusters = append(clusters, sc.createSinglePointCluster(p))
			processed[p.ID] = true
		}
	}

	if sc.Options.Log {
		fmt.Printf("Created %d clusters from %d points using grid\n", len(clusters), numPoints)
	}
	return clusters
}

// createCluster creates a cluster from points
func (sc *Supercluster) createCluster(points []KDPoint) ClusterNode {
	// Handle empty input properly
	if len(points) == 0 {
		return ClusterNode{
			Count:    0,
			Metrics:  make(map[string]float32),
			Metadata: make(map[string]json.RawMessage),
		}
	}

	var sumX, sumY float64
	var totalPoints uint32
	uniquePoints := make(map[uint32]bool)

	// Collect all point IDs for metadata aggregation
	pointIDs := make([]uint32, 0, len(points))

	// First pass - calculate weighted center
	for _, p := range points {
		if !uniquePoints[p.ID] {
			uniquePoints[p.ID] = true
			weight := float64(p.NumPoints)
			sumX += float64(p.X) * weight
			sumY += float64(p.Y) * weight
			totalPoints += p.NumPoints
			pointIDs = append(pointIDs, p.ID)
		}
	}

	// Create cluster node
	cluster := ClusterNode{
		ID:       points[0].ID, // Now this is safe because we've checked len(points) > 0
		X:        float32(sumX / float64(totalPoints)),
		Y:        float32(sumY / float64(totalPoints)),
		Count:    totalPoints,
		Metrics:  make(map[string]float32),
		Metadata: make(map[string]json.RawMessage),
	}

	// Add metrics from all points
	if len(pointIDs) > 0 {
		metrics := sc.aggregateMetrics(pointIDs)
		if metrics != nil {
			cluster.Metrics = metrics
		}

		// Calculate metadata frequencies
		metadata := sc.metadataStore.CalculateFrequencies(pointIDs)
		if metadata != nil {
			cluster.Metadata = metadata
		}
	}

	return cluster
}

// createSinglePointCluster creates a cluster for a single point
func (sc *Supercluster) createSinglePointCluster(p KDPoint) ClusterNode {
	cluster := ClusterNode{
		ID:    p.ID,
		X:     p.X,
		Y:     p.Y,
		Count: 1,
	}

	// Get metrics for this point
	cluster.Metrics = sc.metricsStore.GetMetrics(p.ID)

	// Get metadata for this point
	cluster.Metadata = sc.metadataStore.GetMetadataAsJSON(p.ID)

	return cluster
}

// aggregateMetrics sums metrics for all points in a cluster
func (sc *Supercluster) aggregateMetrics(pointIDs []uint32) map[string]float32 {
	if len(pointIDs) == 0 {
		return nil
	}

	// If just one point, return its metrics directly
	if len(pointIDs) == 1 {
		return sc.metricsStore.GetMetrics(pointIDs[0])
	}

	// Sum metrics across all points
	result := make(map[string]float32)

	for _, id := range pointIDs {
		metrics := sc.metricsStore.GetMetrics(id)
		if metrics == nil {
			continue
		}

		for k, v := range metrics {
			result[k] += v
		}
	}

	return result
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
		}

		// Add metrics
		if c.Metrics != nil {
			for k, v := range c.Metrics {
				properties[k] = v
			}
		}

		// Add metadata
		if c.Metadata != nil {
			for k, v := range c.Metadata {
				var value interface{}
				if err := json.Unmarshal(v, &value); err == nil {
					properties[k] = value
				}
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

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
