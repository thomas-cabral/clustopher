package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
)

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
	Skeleton  *SkeletonTree // Bounds-only leaf index
	Options   SuperclusterOptions
	zoomScale []float64 // Pre-calculated zoom scales
	latLookup []float32 // Pre-calculated latitude projections

	// CH integration (optional; if nil, Load behaves as in-mem only).
	ch        *CHClient
	clusterID string
}

// SetCHClient injects a CH client. If nil, Load behaves like Phase 1 (in-mem only).
func (sc *Supercluster) SetCHClient(c *CHClient) { sc.ch = c }

// SetClusterID sets the cluster_id used in CH partition values. Required when
// SetCHClient is non-nil.
func (sc *Supercluster) SetClusterID(id string) { sc.clusterID = id }

type SuperclusterOptions struct {
	MinZoom   int
	MaxZoom   int
	MinPoints int
	Radius    float64
	NodeSize  int
	Extent    int
	Log       bool

	// ZSplit selects the routing cutoff in GetClusters: queries with zoom
	// < ZSplit use the CH rollup path; queries with zoom >= ZSplit use the
	// skeleton-tree path. Default 11 (set by NewSupercluster when 0).
	ZSplit int
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
	if options.ZSplit == 0 {
		options.ZSplit = 11
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
func (sc *Supercluster) Load(points []Point) error {
	fmt.Printf("Loading %d points\n", len(points))

	// For large datasets, process in batches
	if len(points) > 1000000 {
		return sc.loadBatched(points, 1000000)
	}

	return sc.buildSkeletonAndPersist(points)
}

// loadBatched handles large point sets (kept for the Load routing path).
func (sc *Supercluster) loadBatched(points []Point, batchSize int) error {
	fmt.Printf("Loading %d points\n", len(points))
	return sc.buildSkeletonAndPersist(points)
}

// buildSkeletonAndPersist projects points to MaxZoom, sorts into leaf order,
// builds the skeleton with sequential internal ids, and (if a CH client is
// configured) inserts the points to CH in leaf-order id order.
func (sc *Supercluster) buildSkeletonAndPersist(points []Point) error {
	projected := make([]KDPoint, len(points))
	for i, p := range points {
		proj := sc.projectFast(p.X, p.Y, sc.Options.MaxZoom)
		projected[i] = KDPoint{ID: p.ID, X: proj[0], Y: proj[1], NumPoints: 1}
	}
	sorted := SortPointsIntoLeafOrder(projected, sc.Options.NodeSize)
	tree, remap := BuildSkeletonWithRemap(sorted, sc.Options.NodeSize)
	sc.Skeleton = tree

	if sc.ch != nil {
		if sc.clusterID == "" {
			return fmt.Errorf("CH client set but clusterID empty; call SetClusterID first")
		}
		if err := sc.insertToCH(context.Background(), sorted, remap, points); err != nil {
			return fmt.Errorf("insert to CH: %w", err)
		}
	}
	return nil
}

// insertToCH inserts sorted points (with internal ids already assigned) into
// ClickHouse. remap[i] is the external id for sorted[i]. original is used to
// look up metrics/metadata by external id.
func (sc *Supercluster) insertToCH(ctx context.Context, sorted []KDPoint, remap []uint32, original []Point) error {
	// Build a lookup from external_id -> original Point (for metrics/metadata).
	byExt := make(map[uint32]*Point, len(original))
	for i := range original {
		byExt[original[i].ID] = &original[i]
	}

	const batchSize = 100_000
	rows := make([]CHPointRow, 0, batchSize)
	flush := func() error {
		if len(rows) == 0 {
			return nil
		}
		if err := sc.ch.InsertPoints(ctx, rows); err != nil {
			return err
		}
		rows = rows[:0]
		return nil
	}

	for i, p := range sorted {
		ext := remap[i]
		orig := byExt[ext]
		metaStr := map[string]string{}
		if orig != nil {
			for k, v := range orig.Metadata {
				metaStr[k] = fmt.Sprintf("%v", v)
			}
		}
		// unproject from MaxZoom pixel space back to lng/lat for storage.
		ll := sc.unprojectFast(p.X, p.Y, sc.Options.MaxZoom)
		var metrics map[string]float32
		if orig != nil {
			metrics = orig.Metrics
		}
		rows = append(rows, CHPointRow{
			ClusterID:  sc.clusterID,
			ID:         p.ID, // internal id (already remapped)
			ExternalID: ext,
			X:          ll[0],
			Y:          ll[1],
			Metrics:    metrics,
			Metadata:   metaStr,
		})
		if len(rows) >= batchSize {
			if err := flush(); err != nil {
				return err
			}
		}
	}
	return flush()
}


// CleanupCluster releases memory
func (sc *Supercluster) CleanupCluster() {
	if sc == nil {
		return
	}

	// Force GC
	runtime.GC()
	debug.FreeOSMemory()
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

	return cluster
}

// createSinglePointCluster creates a cluster for a single point
func (sc *Supercluster) createSinglePointCluster(p KDPoint) ClusterNode {
	return ClusterNode{
		ID:    p.ID,
		X:     p.X,
		Y:     p.Y,
		Count: 1,
	}
}

// ToGeoJSON converts clusters to GeoJSON format
func (sc *Supercluster) ToGeoJSON(bounds KDBounds, zoom int) (*FeatureCollection, error) {
	// Get clusters for the given bounds and zoom level
	clusters, err := sc.GetClustersCH(context.Background(), bounds, zoom)
	if err != nil {
		return nil, err
	}

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

// Open rebuilds the skeleton tree from CH for an existing cluster. Points
// are streamed in (cluster_id, id) order — that order is already leaf-order
// because Load wrote them that way — so the build is a single linear pass.
func (sc *Supercluster) Open(ctx context.Context) error {
	if sc.ch == nil {
		return fmt.Errorf("Open requires CH client; call SetCHClient first")
	}
	if sc.clusterID == "" {
		return fmt.Errorf("Open requires clusterID")
	}

	rows, err := sc.ch.Conn().Query(ctx,
		"SELECT id, x, y FROM clustopher.points WHERE cluster_id = ? ORDER BY id", sc.clusterID)
	if err != nil {
		return fmt.Errorf("query points: %w", err)
	}
	defer rows.Close()

	var projected []KDPoint
	for rows.Next() {
		var id uint32
		var x, y float32
		if err := rows.Scan(&id, &x, &y); err != nil {
			return fmt.Errorf("scan: %w", err)
		}
		proj := sc.projectFast(x, y, sc.Options.MaxZoom)
		projected = append(projected, KDPoint{ID: id, X: proj[0], Y: proj[1], NumPoints: 1})
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("rows iter: %w", err)
	}

	// Points are already in leaf order (we wrote them that way). BuildSkeleton
	// accepts pre-sorted input.
	sc.Skeleton = BuildSkeleton(projected, sc.Options.NodeSize)
	return nil
}
