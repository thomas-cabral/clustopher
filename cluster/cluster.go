package cluster

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
)

const (
	minZoom = 0
	maxZoom = 16
)

// KDBounds represents a bounding box
type KDBounds struct {
	MinX, MinY, MaxX, MaxY float64
}

// Extend expands bounds to include another point
func (b *KDBounds) Extend(x, y float64) {
	b.MinX = math.Min(b.MinX, x)
	b.MinY = math.Min(b.MinY, y)
	b.MaxX = math.Max(b.MaxX, x)
	b.MaxY = math.Max(b.MaxY, y)
}

// KDPoint represents a point in KD-tree
type KDPoint struct {
	X, Y      float64
	ID        uint32
	ParentID  uint32
	NumPoints uint32
	Metrics   map[string]float64
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
		MinX: math.Inf(1),
		MinY: math.Inf(1),
		MaxX: math.Inf(-1),
		MaxY: math.Inf(-1),
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
	ID       uint32                 // Unique identifier
	X        float64                // Longitude
	Y        float64                // Latitude
	Metrics  map[string]float64     // Numerical metrics
	Metadata map[string]interface{} // Arbitrary metadata
}

// ClusterMetrics holds calculated metrics for a cluster
type ClusterMetrics struct {
	Values map[string]float64 // Map of metric name to value
}

// ClusterNode represents a node in the cluster hierarchy
type ClusterNode struct {
	ID       uint32                     // Unique identifier for the cluster
	X        float64                    // Longitude of cluster center
	Y        float64                    // Latitude of cluster center
	Count    uint32                     // Number of points in this cluster
	Children []uint32                   // IDs of child clusters
	Metrics  ClusterMetrics             // Calculated metrics for this cluster
	Metadata map[string]json.RawMessage // Flexible metadata storage
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
		if count, ok := props["point_count"].(float64); ok {
			n.Count = uint32(count)
		}
		if metrics, ok := props["metrics"].(map[string]interface{}); ok {
			for k, v := range metrics {
				if val, ok := v.(float64); ok {
					if n.Metrics.Values == nil {
						n.Metrics.Values = make(map[string]float64)
					}
					n.Metrics.Values[k] = val
				}
			}
		}
	}

	return nil
}

// NewPoint creates a new point with initialized maps
func NewPoint(id uint32, x, y float64) *Point {
	return &Point{
		ID:       id,
		X:        x,
		Y:        y,
		Metrics:  make(map[string]float64),
		Metadata: make(map[string]interface{}),
	}
}

// NewClusterNode creates a new cluster node with initialized maps
func NewClusterNode(id uint32, x, y float64) *ClusterNode {
	return &ClusterNode{
		ID:       id,
		X:        x,
		Y:        y,
		Children: make([]uint32, 0),
		Metrics: ClusterMetrics{
			Values: make(map[string]float64),
		},
		Metadata: make(map[string]json.RawMessage),
	}
}

// AddChild adds a child cluster to this node
func (n *ClusterNode) AddChild(childID uint32) {
	n.Children = append(n.Children, childID)
}

// SetMetric sets a metric value for the cluster
func (n *ClusterNode) SetMetric(name string, value float64) {
	if n.Metrics.Values == nil {
		n.Metrics.Values = make(map[string]float64)
	}
	n.Metrics.Values[name] = value
}

// GetMetric gets a metric value from the cluster
func (n *ClusterNode) GetMetric(name string) (float64, bool) {
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
	// Convert points to KDPoints
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

	// Create KD-trees for each zoom level
	for z := sc.Options.MaxZoom; z >= sc.Options.MinZoom; z-- {
		// Generate clusters for previous zoom level
		if z < sc.Options.MaxZoom {
			kdPoints = sc.clusterPoints(kdPoints, z)
		}

		// Create KD-tree for current zoom level
		sc.Trees[z] = NewKDTree(kdPoints, sc.Options.NodeSize)
	}
}

// clusterPoints clusters points for a given zoom level
func (sc *Supercluster) clusterPoints(points []KDPoint, zoom int) []KDPoint {
	// Calculate radius in world coordinates
	r := sc.Options.Radius / float64(sc.Options.Extent) * math.Pow(2, float64(sc.Options.MaxZoom-zoom))

	clusters := make([]KDPoint, 0)
	processed := make(map[uint32]bool)

	// Group points by cell
	cells := make(map[string][]KDPoint)
	for _, p := range points {
		projected := sc.project(p.X, p.Y, zoom)
		cellKey := fmt.Sprintf("%d:%d", int(projected[0]/r), int(projected[1]/r))
		cells[cellKey] = append(cells[cellKey], p)
	}

	// Process each cell
	for _, cellPoints := range cells {
		if len(cellPoints) < sc.Options.MinPoints {
			// Add individual points if below threshold
			clusters = append(clusters, cellPoints...)
			continue
		}

		// Create a cluster from points in this cell
		cluster := KDPoint{
			X:         0,
			Y:         0,
			NumPoints: 0,
			Metrics:   make(map[string]float64),
		}

		// Calculate weighted centroid and sum metrics
		for _, p := range cellPoints {
			if processed[p.ID] {
				continue
			}

			cluster.X += p.X * float64(p.NumPoints)
			cluster.Y += p.Y * float64(p.NumPoints)
			cluster.NumPoints += p.NumPoints

			// Sum metrics
			for k, v := range p.Metrics {
				cluster.Metrics[k] += v * float64(p.NumPoints)
			}

			processed[p.ID] = true
		}

		// Finalize cluster
		if cluster.NumPoints >= uint32(sc.Options.MinPoints) {
			cluster.X /= float64(cluster.NumPoints)
			cluster.Y /= float64(cluster.NumPoints)

			// Average metrics
			for k := range cluster.Metrics {
				cluster.Metrics[k] /= float64(cluster.NumPoints)
			}

			// Generate unique ID for cluster
			cluster.ID = uint32(len(clusters) + 1)

			clusters = append(clusters, cluster)
		}
	}

	return clusters
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
	minP := sc.project(bounds.MinX, bounds.MaxY, zoom) // Note: Y is inverted in tile space
	maxP := sc.project(bounds.MaxX, bounds.MinY, zoom)

	fmt.Printf("Debug: Projected bounds at zoom %d: min(%f,%f) max(%f,%f)\n",
		zoom, minP[0], minP[1], maxP[0], maxP[1])

	// Debugging: Print some sample points
	if len(tree.Points) > 0 {
		p := tree.Points[0]
		proj := sc.project(p.X, p.Y, zoom)
		fmt.Printf("Debug: Sample point at zoom %d: orig(%f,%f) proj(%f,%f)\n",
			zoom, p.X, p.Y, proj[0], proj[1])
	}

	clusters := make([]ClusterNode, 0)

	// Process each point
	for _, p := range tree.Points {
		proj := sc.project(p.X, p.Y, zoom)

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
				Coordinates: []float64{cluster.X, cluster.Y},
			},
			Properties: properties,
		}
	}

	return &FeatureCollection{
		Type:     "FeatureCollection",
		Features: features,
	}, nil
}

func generateClusterID(x, y float64, zoom int) uint32 {
	// Create a unique ID based on position and zoom
	h := fnv.New32()
	fmt.Fprintf(h, "%.6f:%.6f:%d", x, y, zoom)
	return h.Sum32()
}

// queryTree recursively searches KD-tree within bounds
func (sc *Supercluster) queryTree(tree *KDTree, minX, minY, maxX, maxY float64, callback func(KDPoint)) {
	// Project the points at the current zoom level
	zoom := sc.Options.MaxZoom // Use appropriate zoom level
	for _, p := range tree.Points {
		// Project the point coordinates
		projected := sc.project(p.X, p.Y, zoom)

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
		{X: -122.4194, Y: 37.7749, ID: 1, Metrics: map[string]float64{"value": 100}},
		{X: -122.4195, Y: 37.7748, ID: 2, Metrics: map[string]float64{"value": 200}},
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
