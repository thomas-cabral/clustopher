package cluster

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"

	"github.com/google/uuid"
)

const (
	minZoom = 0
	maxZoom = 16
)

// KDBounds represents a bounding box
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

// KDPoint represents a point in KD-tree
type KDPoint struct {
	X, Y      float32 // Changed from float64 to float32
	ID        uint32
	ParentID  uint32
	NumPoints uint32
	Metrics   map[string]float32 // Changed from float64 to float32
}

// KDTree implements a KD-tree for spatial indexing
type KDTree struct {
	Points   []KDPoint
	NodeSize int
	Bounds   KDBounds
}

// NewKDTree creates a new KD-tree
func NewKDTree(points []KDPoint, nodeSize int) *KDTree {
	bounds := KDBounds{
		MinX: float32(math.Inf(1)), // Maximum possible value
		MinY: float32(math.Inf(1)),
		MaxX: float32(math.Inf(-1)), // Minimum possible value
		MaxY: float32(math.Inf(-1)),
	}

	for _, p := range points {
		bounds.Extend(p.X, p.Y)
	}

	return &KDTree{
		Points:   points,
		NodeSize: nodeSize,
		Bounds:   bounds,
	}
}

// Supercluster implements the clustering algorithm
type Supercluster struct {
	Trees    map[int]*KDTree       // KD-trees for each zoom level
	Points   []Point               // Original points
	Clusters map[int][]ClusterNode // Clusters by zoom level
	Options  SuperclusterOptions
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

// Point represents a single location with associated data
type Point struct {
	ID       uint32
	X, Y     float32                // Changed from float64 to float32
	Metrics  map[string]float32     // Changed from float64 to float32
	Metadata map[string]interface{} // Consider making this optional or more specific
}

// ClusterNode represents a node in the cluster hierarchy
type ClusterNode struct {
	ID       uint32                     // Unique identifier for the cluster
	X        float32                    // Longitude of cluster center
	Y        float32                    // Latitude of cluster center
	Count    uint32                     // Number of points in this cluster
	Children []uint32                   // IDs of child clusters
	Metrics  ClusterMetrics             // Calculated metrics for this cluster
	Metadata map[string]json.RawMessage // Flexible metadata storage
}

// ClusterMetrics holds calculated metrics for a cluster
type ClusterMetrics struct {
	Values map[string]float32 // Changed from float64 to float32
}

// GeoJSON structures
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

// MarshalJSON implements custom JSON marshaling for ClusterNode
func (n *ClusterNode) MarshalJSON() ([]byte, error) {
	type Alias ClusterNode
	return json.Marshal(&struct {
		*Alias
		Properties map[string]interface{} `json:"properties"`
	}{
		Alias: (*Alias)(n),
		Properties: map[string]interface{}{
			"point_count": n.Count,
			"metrics":     n.Metrics.Values,
		},
	})
}

// UnmarshalJSON implements custom JSON unmarshaling for ClusterNode
func (n *ClusterNode) UnmarshalJSON(data []byte) error {
	type Alias ClusterNode
	aux := &struct {
		*Alias
		Properties map[string]interface{} `json:"properties"`
	}{
		Alias: (*Alias)(n),
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	// Extract properties
	if props := aux.Properties; props != nil {
		if count, ok := props["point_count"].(float32); ok {
			n.Count = uint32(count)
		}
		if metrics, ok := props["metrics"].(map[string]interface{}); ok {
			for k, v := range metrics {
				if val, ok := v.(float32); ok {
					if n.Metrics.Values == nil {
						n.Metrics.Values = make(map[string]float32)
					}
					n.Metrics.Values[k] = val
				}
			}
		}
	}

	return nil
}

// NewPoint creates a new point with initialized maps
func NewPoint(id uint32, x, y float32) *Point {
	return &Point{
		ID:       id,
		X:        x,
		Y:        y,
		Metrics:  make(map[string]float32),
		Metadata: make(map[string]interface{}),
	}
}

// NewClusterNode creates a new cluster node with initialized maps
func NewClusterNode(id uint32, x, y float32) *ClusterNode {
	return &ClusterNode{
		ID:       id,
		X:        x,
		Y:        y,
		Children: make([]uint32, 0),
		Metrics: ClusterMetrics{
			Values: make(map[string]float32),
		},
		Metadata: make(map[string]json.RawMessage),
	}
}

// AddChild adds a child cluster to this node
func (n *ClusterNode) AddChild(childID uint32) {
	n.Children = append(n.Children, childID)
}

// SetMetric sets a metric value for the cluster
func (n *ClusterNode) SetMetric(name string, value float32) {
	if n.Metrics.Values == nil {
		n.Metrics.Values = make(map[string]float32)
	}
	n.Metrics.Values[name] = value
}

// GetMetric gets a metric value from the cluster
func (n *ClusterNode) GetMetric(name string) (float32, bool) {
	if n.Metrics.Values == nil {
		return 0, false
	}
	val, ok := n.Metrics.Values[name]
	return val, ok
}

// SetMetadata sets metadata for the cluster
func (n *ClusterNode) SetMetadata(key string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if n.Metadata == nil {
		n.Metadata = make(map[string]json.RawMessage)
	}
	n.Metadata[key] = data
	return nil
}

// GetMetadata gets and unmarshals metadata from the cluster
func (n *ClusterNode) GetMetadata(key string, value interface{}) error {
	if n.Metadata == nil {
		return fmt.Errorf("no metadata found for key: %s", key)
	}
	data, ok := n.Metadata[key]
	if !ok {
		return fmt.Errorf("key not found: %s", key)
	}
	return json.Unmarshal(data, value)
}

// NewSupercluster creates a new clustering instance
func NewSupercluster(options SuperclusterOptions) *Supercluster {
	if options.MinZoom == 0 {
		options.MinZoom = 0
	}
	if options.MaxZoom == 0 {
		options.MaxZoom = 16
	}
	if options.NodeSize == 0 {
		options.NodeSize = 64
	}
	if options.Extent == 0 {
		options.Extent = 512
	}
	if options.Radius == 0 {
		options.Radius = 40
	}

	return &Supercluster{
		Trees:    make(map[int]*KDTree),
		Clusters: make(map[int][]ClusterNode),
		Options:  options,
	}
}

// Load initializes the cluster index with points
func (sc *Supercluster) Load(points []Point) {
	fmt.Printf("Initial points: %d\n", len(points))

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

	totalPoints := uint32(len(points))
	for z := sc.Options.MaxZoom; z >= sc.Options.MinZoom; z-- {
		if z < sc.Options.MaxZoom {
			kdPoints = sc.clusterPoints(kdPoints, z)

			// Verify point count
			var count uint32
			for _, p := range kdPoints {
				count += p.NumPoints
			}
			fmt.Printf("Zoom level %d - Clusters: %d, Total points: %d\n",
				z, len(kdPoints), count)

			if count != totalPoints {
				fmt.Printf("WARNING: Lost points at zoom %d (expected %d, got %d)\n",
					z, totalPoints, count)
			}
		}
		sc.Trees[z] = NewKDTree(kdPoints, sc.Options.NodeSize)
	}
}

// clusterPoints clusters points for a given zoom level
func (sc *Supercluster) clusterPoints(points []KDPoint, zoom int) []KDPoint {
    // Calculate radius with higher precision
    radius := sc.Options.Radius / float64(sc.Options.Extent) * math.Pow(2, float64(sc.Options.MaxZoom-zoom))
    
    clusters := make([]KDPoint, 0)
    processed := make(map[uint32]struct{})

    // Track points before clustering
    var totalPointsBefore uint32
    pointsBefore := make(map[uint32]struct{})
    for _, p := range points {
        totalPointsBefore += p.NumPoints
        pointsBefore[p.ID] = struct{}{}
    }

    // Group points by cell with boundary tracking
    cells := make(map[uint64][]KDPoint)
    pointsInCells := make(map[uint32]uint64) // Track which cell each point goes into
    
    for _, p := range points {
        projected := sc.projectFast(p.X, p.Y, zoom)
        
        // Use higher precision for cell calculations
        projX := float64(projected[0]) / radius
        projY := float64(projected[1]) / radius
        
        // Check for points near cell boundaries
        fracX := math.Abs(projX - math.Floor(projX))
        fracY := math.Abs(projY - math.Floor(projY))
        
        if (fracX < 1e-10 || math.Abs(fracX-1) < 1e-10) ||
           (fracY < 1e-10 || math.Abs(fracY-1) < 1e-10) {
            
        }
        
        // More precise rounding for cell assignment
        cellX := int32(math.Floor(projX + 0.5))
        cellY := int32(math.Floor(projY + 0.5))
        
        cellKey := (uint64(cellX) << 32) | uint64(uint32(cellY))
        cells[cellKey] = append(cells[cellKey], p)
        pointsInCells[p.ID] = cellKey
    }

    // Process cells
    cellClusters := make(map[uint64]*KDPoint)
    for cellKey, cellPoints := range cells {
        if len(cellPoints) < sc.Options.MinPoints {
            continue
        }

        cluster := processCell(cellPoints, processed, sc.Options.MinPoints)
        if cluster != nil {
            cellClusters[cellKey] = cluster
            clusters = append(clusters, *cluster)
        }
    }

    // Track unclustered points
    var preservedPoints uint32
    unclusteredIDs := make(map[uint32]struct{})
    
    for _, p := range points {
        if _, exists := processed[p.ID]; !exists {
            preservedPoints += p.NumPoints
            clusters = append(clusters, p)
            processed[p.ID] = struct{}{}
            unclusteredIDs[p.ID] = struct{}{}
        }
    }

    // Check for lost points
    for id := range pointsBefore {
        if _, inProcessed := processed[id]; !inProcessed {
            if cellKey, inCell := pointsInCells[id]; inCell {
                if sc.Options.Log {
                    fmt.Printf("Zoom %d - Lost point ID %d was assigned to cell %d but not processed\n",
                        zoom, id, cellKey)
                }
            } else {
                if sc.Options.Log {
                    fmt.Printf("Zoom %d - Lost point ID %d was never assigned to a cell\n",
                        zoom, id)
                }
            }
        }
    }

    // Final count verification
    var totalPointsAfter uint32
    for _, c := range clusters {
        totalPointsAfter += c.NumPoints
    }
    
    if totalPointsAfter != totalPointsBefore {
        fmt.Printf("Zoom %d - Point count mismatch - Before: %d, After: %d (diff: %d)\n",
            zoom, totalPointsBefore, totalPointsAfter, totalPointsBefore - totalPointsAfter)
        fmt.Printf("Zoom %d - Points distribution - Clustered: %d, Preserved: %d\n",
            zoom, totalPointsAfter-preservedPoints, preservedPoints)
    }

    return clusters
}

func processCell(points []KDPoint, processed map[uint32]struct{}, minClusterPoints int) *KDPoint {
    // First collect all unprocessed points
    unprocessedPoints := make([]KDPoint, 0, len(points))
    var totalUnprocessedCount uint32
    
    for _, p := range points {
        if _, exists := processed[p.ID]; !exists {
            unprocessedPoints = append(unprocessedPoints, p)
            totalUnprocessedCount += p.NumPoints
        }
    }

    // If not enough points to cluster, return nil
    if len(unprocessedPoints) < minClusterPoints {
        return nil
    }

    // Create cluster
    cluster := &KDPoint{
        Metrics:   make(map[string]float32),
        NumPoints: totalUnprocessedCount,
    }

    // Use double precision for accumulation
    var sumX, sumY float64
    metricSums := make(map[string]float64)

    // First pass: accumulate sums using double precision
    for _, p := range unprocessedPoints {
        pointWeight := float64(p.NumPoints)
        sumX += float64(p.X) * pointWeight
        sumY += float64(p.Y) * pointWeight
        
        for k, v := range p.Metrics {
            metricSums[k] += float64(v) * pointWeight
        }
    }

    // Second pass: compute final values with proper normalization
    if totalUnprocessedCount > 0 {
        invTotal := 1.0 / float64(totalUnprocessedCount)
        cluster.X = float32(sumX * invTotal)
        cluster.Y = float32(sumY * invTotal)
        
        for k, sum := range metricSums {
            cluster.Metrics[k] = float32(sum * invTotal)
        }
    }

    // Mark points as processed
    for _, p := range unprocessedPoints {
        processed[p.ID] = struct{}{}
    }

    cluster.ID = uuid.New().ID()
    return cluster
}

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

// unproject converts tile coordinates to lng/lat
func (sc *Supercluster) unproject(x, y float64, zoom int) [2]float64 {
	size := float64(sc.Options.Extent) * math.Pow(2, float64(zoom))
	lng := x*360/size - 180
	lat := math.Atan(math.Sinh(math.Pi*(1-2*y/size))) * 180 / math.Pi
	return [2]float64{lng, lat}
}

// project converts longitude/latitude to tile coordinates
func (sc *Supercluster) project(lng, lat float64, zoom int) [2]float64 {
	sin := math.Sin(lat * math.Pi / 180)
	x := (lng + 180) / 360
	y := 0.5 - 0.25*math.Log((1+sin)/(1-sin))/math.Pi

	// Scale to the zoom level
	scale := math.Pow(2, float64(zoom))
	return [2]float64{
		x * scale * float64(sc.Options.Extent),
		y * scale * float64(sc.Options.Extent),
	}
}

// GetClusters returns clusters within specified bounds and zoom level
func (sc *Supercluster) GetClusters(bounds KDBounds, zoom int) []ClusterNode {
	tree := sc.Trees[zoom]
	if tree == nil {
		fmt.Printf("Debug: No tree found for zoom level %d\n", zoom)
		return nil
	}

	// Project bounds to tile space
	minP := sc.projectFast(bounds.MinX, bounds.MaxY, zoom) // Note: Y is inverted in tile space
	maxP := sc.projectFast(bounds.MaxX, bounds.MinY, zoom)

	fmt.Printf("Debug: Projected bounds at zoom %d: min(%f,%f) max(%f,%f)\n",
		zoom, minP[0], minP[1], maxP[0], maxP[1])

	// Debugging: Print some sample points
	if len(tree.Points) > 0 {
		p := tree.Points[0]
		proj := sc.projectFast(p.X, p.Y, zoom)
		fmt.Printf("Debug: Sample point at zoom %d: orig(%f,%f) proj(%f,%f)\n",
			zoom, p.X, p.Y, proj[0], proj[1])
	}

	clusters := make([]ClusterNode, 0)

	// Process each point
	for _, p := range tree.Points {
		proj := sc.projectFast(p.X, p.Y, zoom)

		// Check if the point is within the projected bounds
		if proj[0] >= minP[0] && proj[0] <= maxP[0] &&
			proj[1] >= minP[1] && proj[1] <= maxP[1] {

			cluster := ClusterNode{
				ID:      p.ID,
				X:       p.X,
				Y:       p.Y,
				Count:   p.NumPoints,
				Metrics: ClusterMetrics{Values: p.Metrics},
			}
			clusters = append(clusters, cluster)
		}
	}

	fmt.Printf("Debug: Found %d clusters within bounds at zoom %d\n",
		len(clusters), zoom)

	return clusters
}

// Convert clusters to GeoJSON
func (sc *Supercluster) ToGeoJSON(bounds KDBounds, zoom int) (*FeatureCollection, error) {
	clusters := sc.GetClusters(bounds, zoom)

	features := make([]Feature, len(clusters))
	for i, cluster := range clusters {
		// Create properties map
		properties := make(map[string]interface{})
		properties["cluster"] = cluster.Count > 1
		properties["point_count"] = cluster.Count

		// Add metrics if they exist
		if cluster.Metrics.Values != nil {
			properties["metrics"] = cluster.Metrics.Values
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

func generateClusterID(x, y float32, zoom int) uint32 {
	// Create a unique ID based on position and zoom
	h := fnv.New32()
	fmt.Fprintf(h, "%.6f:%.6f:%d", x, y, zoom)
	return h.Sum32()
}

// queryTree recursively searches KD-tree within bounds
func (sc *Supercluster) queryTree(tree *KDTree, minX, minY, maxX, maxY float32, callback func(KDPoint)) {
	// Project the points at the current zoom level
	zoom := sc.Options.MaxZoom // Use appropriate zoom level
	for _, p := range tree.Points {
		// Project the point coordinates
		projected := sc.projectFast(p.X, p.Y, zoom)

		// Check if the projected point is within bounds
		if projected[0] >= minX && projected[0] <= maxX &&
			projected[1] >= minY && projected[1] <= maxY {
			callback(p)
		}
	}
}

// Example usage:
func Example() {
	options := SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 3,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	}

	cluster := NewSupercluster(options)

	// Generate some test points
	points := []Point{
		{X: -122.4194, Y: 37.7749, ID: 1, Metrics: map[string]float32{"value": 100}},
		{X: -122.4195, Y: 37.7748, ID: 2, Metrics: map[string]float32{"value": 200}},
		// ... more points
	}

	// Load points into cluster index
	cluster.Load(points)

	// Get clusters for specific bounds and zoom level
	bounds := KDBounds{
		MinX: -122.5,
		MinY: 37.7,
		MaxX: -122.3,
		MaxY: 37.8,
	}

	clusters := cluster.GetClusters(bounds, 12)
	for _, c := range clusters {
		fmt.Printf("Cluster at (%f, %f) with %d points\n", c.X, c.Y, c.Count)
	}
}
