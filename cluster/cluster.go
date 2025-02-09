package cluster

import (
	"bufio"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sort"
	"time"

	"runtime/debug"

	"github.com/google/uuid"
	"github.com/klauspost/compress/zstd"
)

// KDTree implements a KD-tree for spatial indexing
type KDTree struct {
	Root     *KDNode
	Points   []KDPoint // Store all points for easy access
	NodeSize int
	Bounds   KDBounds
}

// KDNode represents a node in the KD-tree
type KDNode struct {
	Point    KDPoint
	Left     *KDNode
	Right    *KDNode
	Axis     int    // 0 for X, 1 for Y
	MinChild uint32 // Minimum point ID in subtree
	MaxChild uint32 // Maximum point ID in subtree
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

// NewKDTree creates a new KD-tree with optimized construction
func NewKDTree(points []KDPoint, nodeSize int) *KDTree {
	fmt.Printf("Creating KD-tree with %d points\n", len(points))

	if len(points) == 0 {
		return &KDTree{
			Points:   []KDPoint{},
			NodeSize: nodeSize,
		}
	}

	// Make a copy of the points slice to avoid modifying the original
	pointsCopy := make([]KDPoint, len(points))
	copy(pointsCopy, points)

	bounds := KDBounds{
		MinX: float32(math.Inf(1)),
		MinY: float32(math.Inf(1)),
		MaxX: float32(math.Inf(-1)),
		MaxY: float32(math.Inf(-1)),
	}

	// Calculate bounds
	for _, p := range pointsCopy {
		bounds.Extend(p.X, p.Y)
	}

	// Create tree
	tree := &KDTree{
		Points:   pointsCopy,
		NodeSize: nodeSize,
		Bounds:   bounds,
	}

	// Build tree recursively
	tree.Root = tree.buildNode(pointsCopy, 0)

	fmt.Printf("Created KD-tree with %d points and bounds: MinX=%f, MinY=%f, MaxX=%f, MaxY=%f\n",
		len(tree.Points), bounds.MinX, bounds.MinY, bounds.MaxX, bounds.MaxY)

	return tree
}

// buildNode constructs a KD-tree node recursively
func (t *KDTree) buildNode(points []KDPoint, depth int) *KDNode {
	n := len(points)
	if n == 0 {
		return nil
	}

	// Create leaf node if we're at nodeSize or below
	if n <= t.NodeSize {
		node := &KDNode{
			Point:    points[0],
			MinChild: points[0].ID,
			MaxChild: points[0].ID,
		}
		// Update min/max IDs
		for _, p := range points[1:] {
			if p.ID < node.MinChild {
				node.MinChild = p.ID
			}
			if p.ID > node.MaxChild {
				node.MaxChild = p.ID
			}
		}
		return node
	}

	// Choose axis based on depth
	axis := depth % 2

	// Sort points by the chosen axis
	if axis == 0 {
		sort.Slice(points, func(i, j int) bool {
			return points[i].X < points[j].X
		})
	} else {
		sort.Slice(points, func(i, j int) bool {
			return points[i].Y < points[j].Y
		})
	}

	// Find median
	median := n / 2

	// Create node
	node := &KDNode{
		Point: points[median],
		Axis:  axis,
	}

	// Recursively build left and right subtrees
	node.Left = t.buildNode(points[:median], depth+1)
	node.Right = t.buildNode(points[median+1:], depth+1)

	// Update min/max child IDs
	node.MinChild = node.Point.ID
	node.MaxChild = node.Point.ID

	if node.Left != nil {
		if node.Left.MinChild < node.MinChild {
			node.MinChild = node.Left.MinChild
		}
		if node.Left.MaxChild > node.MaxChild {
			node.MaxChild = node.Left.MaxChild
		}
	}
	if node.Right != nil {
		if node.Right.MinChild < node.MinChild {
			node.MinChild = node.Right.MinChild
		}
		if node.Right.MaxChild > node.MaxChild {
			node.MaxChild = node.Right.MaxChild
		}
	}

	return node
}

// Load initializes the cluster index with points
func (sc *Supercluster) Load(points []Point) {
	fmt.Printf("Loading %d points\n", len(points))

	kdPoints := make([]KDPoint, len(points))
	for i, p := range points {
		kdPoints[i] = KDPoint{
			X:         p.X,
			Y:         p.Y,
			ID:        p.ID,
			NumPoints: 1,
			Metrics:   p.Metrics,
		}
	}

	sc.Points = points
	sc.Tree = NewKDTree(kdPoints, sc.Options.NodeSize)
	fmt.Printf("Created KD-tree with %d points\n", len(sc.Tree.Points))
}

// rangeSearch finds all points within radius of center
func (t *KDTree) rangeSearch(center KDPoint, radius float32, zoom int, results *[]KDPoint) {
	if t.Root == nil {
		return
	}
	t.rangeSearchNode(t.Root, center, radius, zoom, results)
}

func (t *KDTree) rangeSearchNode(node *KDNode, center KDPoint, radius float32, zoom int, results *[]KDPoint) {
	if node == nil {
		return
	}

	// Calculate distance in projected space
	dx := node.Point.X - center.X
	dy := node.Point.Y - center.Y
	dist2 := dx*dx + dy*dy     // squared distance
	radius2 := radius * radius // squared radius

	// If point is within radius, add it
	if dist2 <= radius2 {
		*results = append(*results, node.Point)
	}

	// Check which child nodes need to be searched
	var splitDist float32
	if node.Axis == 0 {
		splitDist = center.X - node.Point.X
	} else {
		splitDist = center.Y - node.Point.Y
	}

	// Search child nodes if they could contain points within the radius
	split2 := splitDist * splitDist
	if split2 <= radius2 { // if the splitting plane is within radius
		// Search both children
		t.rangeSearchNode(node.Left, center, radius, zoom, results)
		t.rangeSearchNode(node.Right, center, radius, zoom, results)
	} else {
		// Only search the side of the split that the center is on
		if splitDist <= 0 {
			t.rangeSearchNode(node.Left, center, radius, zoom, results)
		} else {
			t.rangeSearchNode(node.Right, center, radius, zoom, results)
		}
	}
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
		// Project point to current zoom level
		proj := sc.projectFast(p.X, p.Y, zoom)

		// Check if point is within bounds
		if proj[0] >= minP[0] && proj[0] <= maxP[0] &&
			proj[1] >= minP[1] && proj[1] <= maxP[1] {
			points = append(points, p)
		}
	}

	fmt.Printf("Found %d points within bounds\n", len(points))

	// Calculate clustering radius for this zoom level
	zoomFactor := math.Pow(2, float64(sc.Options.MaxZoom-zoom))
	radius := float32(sc.Options.Radius * zoomFactor / float64(sc.Options.Extent))

	fmt.Printf("Clustering radius: %f\n", radius)

	// Cluster the points using projected coordinates
	var projectedPoints []KDPoint
	for _, p := range points {
		proj := sc.projectFast(p.X, p.Y, zoom)
		projectedPoints = append(projectedPoints, KDPoint{
			X:         proj[0],
			Y:         proj[1],
			ID:        p.ID,
			NumPoints: p.NumPoints,
			Metrics:   p.Metrics,
		})
	}

	// Cluster the projected points
	clusters := sc.clusterPoints(projectedPoints, radius)
	fmt.Printf("Created %d clusters\n", len(clusters))

	// Convert cluster coordinates back to lng/lat
	for i := range clusters {
		unproj := sc.unprojectFast(clusters[i].X, clusters[i].Y, zoom)
		clusters[i].X = unproj[0]
		clusters[i].Y = unproj[1]
	}

	return clusters
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

		fmt.Printf("Point %d has %d nearby points\n", p.ID, len(nearby))

		// If we have enough points, create a cluster
		if len(nearby) >= sc.Options.MinPoints {
			cluster := createCluster(append(nearby, p))
			clusters = append(clusters, cluster)

			// Mark points as processed
			for _, np := range nearby {
				processed[np.ID] = true
			}
			processed[p.ID] = true
		} else {
			// Add as individual point
			clusters = append(clusters, ClusterNode{
				ID:      p.ID,
				X:       p.X,
				Y:       p.Y,
				Count:   1,
				Metrics: ClusterMetrics{Values: p.Metrics},
			})
			processed[p.ID] = true
		}
	}

	fmt.Printf("Created %d clusters from %d points\n", len(clusters), len(points))
	return clusters
}

func createCluster(points []KDPoint) ClusterNode {
	var sumX, sumY float64
	metrics := make(map[string]float64)
	var totalPoints uint32

	// Accumulate values
	for _, p := range points {
		weight := float64(p.NumPoints)
		sumX += float64(p.X) * weight
		sumY += float64(p.Y) * weight
		totalPoints += p.NumPoints

		for k, v := range p.Metrics {
			metrics[k] += float64(v) * weight
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
	}

	// Average metrics
	for k, sum := range metrics {
		cluster.Metrics.Values[k] = float32(sum * invTotal)
	}

	return cluster
}

// Add necessary existing types and helper functions
type Point struct {
	ID       uint32
	X, Y     float32
	Metrics  map[string]float32
	Metadata map[string]interface{}
}

// Modify KDPoint to be more memory efficient
type KDPoint struct {
	X, Y      float32            // 8 bytes
	ID        uint32             // 4 bytes
	ParentID  uint32             // 4 bytes
	NumPoints uint32             // 4 bytes
	Metrics   map[string]float32 // Change back to map for compatibility
}

type ClusterNode struct {
	ID       uint32
	X, Y     float32
	Count    uint32
	Children []uint32
	Metrics  ClusterMetrics
	Metadata map[string]json.RawMessage
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

//SaveCompressed saves the KDTree to a zstd compressed file
func (t *KDTree) SaveCompressed(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Create zstd writer
	enc, err := zstd.NewWriter(file)
	if err != nil {
		return fmt.Errorf("failed to create zstd writer: %v", err)
	}
	defer enc.Close()

	// Create gob encoder
	gobEnc := gob.NewEncoder(enc)

	serialTree := struct {
		Points   []KDPoint
		NodeSize int
		Bounds   KDBounds
		Root     *KDNode
	}{
		Points:   t.Points,
		NodeSize: t.NodeSize,
		Bounds:   t.Bounds,
		Root:     t.Root,
	}

	if err := gobEnc.Encode(serialTree); err != nil {
		return fmt.Errorf("failed to encode tree: %v", err)
	}

	return nil
}

// LoadCompressed loads a KDTree from a zstd compressed file
func LoadCompressedKDTree(filename string) (*KDTree, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Create zstd reader
	dec, err := zstd.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create zstd reader: %v", err)
	}
	defer dec.Close()

	// Create gob decoder
	gobDec := gob.NewDecoder(dec)

	var serialTree struct {
		Points   []KDPoint
		NodeSize int
		Bounds   KDBounds
		Root     *KDNode
	}

	if err := gobDec.Decode(&serialTree); err != nil {
		return nil, fmt.Errorf("failed to decode tree: %v", err)
	}

	return &KDTree{
		Points:   serialTree.Points,
		NodeSize: serialTree.NodeSize,
		Bounds:   serialTree.Bounds,
		Root:     serialTree.Root,
	}, nil
}

// SaveCompressed saves the Supercluster to a zstd compressed file
// func (sc *Supercluster) SaveCompressed(filename string) error {
// 	fmt.Printf("Attempting to save cluster to %s\n", filename)
// 	// Create the file
// 	file, err := os.Create(filename)
// 	if err != nil {
// 		return fmt.Errorf("failed to create file: %v", err)
// 	}
// 	defer file.Close()

// 	// Create zstd writer
// 	enc, err := zstd.NewWriter(file)
// 	if err != nil {
// 		return fmt.Errorf("failed to create zstd writer: %v", err)
// 	}
// 	defer enc.Close()

// 	// Create encoder
// 	gobEnc := gob.NewEncoder(enc)

// 	// Create a serializable version of the supercluster
// 	serialCluster := struct {
// 		Tree    *KDTree
// 		Points  []Point
// 		Options SuperclusterOptions
// 	}{
// 		Tree:    sc.Tree,
// 		Points:  sc.Points,
// 		Options: sc.Options,
// 	}

// 	fmt.Printf("Serializing cluster with %d points\n", len(serialCluster.Points))

// 	// Encode the cluster
// 	if err := gobEnc.Encode(serialCluster); err != nil {
// 		return fmt.Errorf("failed to encode cluster: %v", err)
// 	}

// 	// Verify file was written
// 	if info, err := os.Stat(filename); err == nil {
// 		fmt.Printf("Successfully wrote cluster file: %s (size: %d bytes)\n", filename, info.Size())
// 	} else {
// 		fmt.Printf("Error verifying saved file: %v\n", err)
// 	}

// 	return nil
// }

// LoadCompressedSupercluster loads a Supercluster from a zstd compressed file
func LoadCompressedSupercluster(filename string) (*Supercluster, error) {
	fmt.Printf("Starting to load cluster from %s\n", filename)

	// Open the file
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Create buffered reader
	bufReader := bufio.NewReaderSize(file, 1024*1024) // 1MB buffer

	// Create zstd reader with options for memory usage
	dec, err := zstd.NewReader(bufReader,
		zstd.WithDecoderLowmem(true),
		zstd.WithDecoderMaxMemory(1024*1024*1024)) // 1GB max memory
	if err != nil {
		return nil, fmt.Errorf("failed to create zstd reader: %v", err)
	}
	defer dec.Close()

	// Create decoder
	gobDec := gob.NewDecoder(dec)

	// Create a serializable version of the supercluster
	var serialCluster struct {
		Tree    *KDTree
		Points  []Point
		Options SuperclusterOptions
	}

	fmt.Printf("Starting to decode cluster...\n")
	memBefore := runtime.MemStats{}
	runtime.ReadMemStats(&memBefore)

	// Decode the cluster
	if err := gobDec.Decode(&serialCluster); err != nil {
		return nil, fmt.Errorf("failed to decode cluster: %v", err)
	}

	memAfter := runtime.MemStats{}
	runtime.ReadMemStats(&memAfter)

	fmt.Printf("Memory used for loading: %d MB\n",
		(memAfter.Alloc-memBefore.Alloc)/1024/1024)
	fmt.Printf("Loaded cluster with %d points\n", len(serialCluster.Points))

	// Create and return the supercluster
	cluster := &Supercluster{
		Tree:    serialCluster.Tree,
		Points:  serialCluster.Points,
		Options: serialCluster.Options,
	}

	// Periodically force GC during loading
	defer runtime.GC()

	// Use a timer to periodically clean up
	cleanup := time.NewTicker(5 * time.Second)
	defer cleanup.Stop()

	go func() {
		for range cleanup.C {
			runtime.GC()
			debug.FreeOSMemory()
		}
	}()

	return cluster, nil
}

// Add this function to help with memory management
func (sc *Supercluster) CleanupCluster() {
	if sc != nil {
		sc.Tree = nil
		sc.Points = nil
		// Force garbage collection after clearing large data structures
		runtime.GC()
	}
}

func (sc *Supercluster) GetMemoryStats() string {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return fmt.Sprintf(
		"Memory Stats:\n"+
			"Allocated: %v MB\n"+
			"Total Allocated: %v MB\n"+
			"System Memory: %v MB\n"+
			"Number of GC: %v\n",
		m.Alloc/1024/1024,
		m.TotalAlloc/1024/1024,
		m.Sys/1024/1024,
		m.NumGC,
	)
}

// Add a new function to load points in batches
func LoadPointsBatch(reader io.Reader, batchSize int) ([]Point, error) {
	points := make([]Point, 0, batchSize)
	decoder := gob.NewDecoder(reader)

	for i := 0; i < batchSize; i++ {
		var point Point
		err := decoder.Decode(&point)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		points = append(points, point)
	}

	return points, nil
}

// Add new batch loading function
func LoadCompressedSuperclusterBatch(filename string, batchSize int) (*Supercluster, error) {
	fmt.Printf("Starting to load cluster from %s in batches\n", filename)

	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	bufReader := bufio.NewReaderSize(file, 1024*1024)
	dec, err := zstd.NewReader(bufReader,
		zstd.WithDecoderLowmem(true),
		zstd.WithDecoderMaxMemory(1024*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("failed to create zstd reader: %v", err)
	}
	defer dec.Close()

	gobDec := gob.NewDecoder(dec)

	// First read the header/options
	var options SuperclusterOptions
	if err := gobDec.Decode(&options); err != nil {
		return nil, fmt.Errorf("failed to decode options: %v", err)
	}

	cluster := NewSupercluster(options)

	// Read points in batches
	var points []Point
	for {
		batch := make([]Point, 0, batchSize)
		for i := 0; i < batchSize; i++ {
			var point Point
			if err := gobDec.Decode(&point); err != nil {
				if err == io.EOF {
					break
				}
				return nil, fmt.Errorf("failed to decode point: %v", err)
			}
			batch = append(batch, point)
		}

		if len(batch) == 0 {
			break
		}

		points = append(points, batch...)

		// Force GC after each batch
		runtime.GC()
		debug.FreeOSMemory()
	}

	// Load points into cluster
	cluster.Load(points)

	return cluster, nil
}

// Modify SaveCompressed to support batch loading format
func (sc *Supercluster) SaveCompressed(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	enc, err := zstd.NewWriter(file)
	if err != nil {
		return fmt.Errorf("failed to create zstd writer: %v", err)
	}
	defer enc.Close()

	gobEnc := gob.NewEncoder(enc)

	// First write options
	if err := gobEnc.Encode(sc.Options); err != nil {
		return fmt.Errorf("failed to encode options: %v", err)
	}

	// Then write points
	for _, point := range sc.Points {
		if err := gobEnc.Encode(point); err != nil {
			return fmt.Errorf("failed to encode point: %v", err)
		}
	}

	return nil
}
