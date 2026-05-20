package cluster

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

func newTestSupercluster(t *testing.T, clusterID string, opts SuperclusterOptions) *Supercluster {
	t.Helper()
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}
	ctx := context.Background()
	c, err := NewCHClient(ctx, CHConfig{DSN: dsn})
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	if err := RunMigrations(ctx, c.Conn(), "migrations"); err != nil {
		t.Fatalf("migrations: %v", err)
	}
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
	}
	sc := NewSupercluster(opts)
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)
	t.Cleanup(func() {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
		for z := 2; z <= 16; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
		}
		c.Close()
	})
	return sc
}

func TestClusterMetricsRollup(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	})

	// Metrics aggregation now happens in CH; createCluster only handles geometry/count.
	kdPoints := []KDPoint{
		{X: 0, Y: 0, ID: 1, NumPoints: 1},
		{X: 0.1, Y: 0.1, ID: 2, NumPoints: 1},
		{X: 0.2, Y: 0.2, ID: 3, NumPoints: 1},
	}

	cluster := sc.createCluster(kdPoints)

	if cluster.Count != 3 {
		t.Errorf("Expected count 3, got %d", cluster.Count)
	}

	// Super-cluster geometry
	superKdPoints := []KDPoint{
		{X: 0.1, Y: 0.1, ID: 4, NumPoints: 3},
		{X: 1, Y: 1, ID: 5, NumPoints: 2},
	}

	superCluster := sc.createCluster(superKdPoints)

	expectedTotalPoints := uint32(5)
	if superCluster.Count != expectedTotalPoints {
		t.Errorf("Expected total points to be %d, got %d", expectedTotalPoints, superCluster.Count)
	}
}

func TestEmptyCluster(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{})
	points := []KDPoint{}

	// Test creating cluster with no points
	cluster := sc.createCluster(points)

	if cluster.Count != 0 {
		t.Errorf("Expected empty cluster count to be 0, got %d", cluster.Count)
	}
	if len(cluster.Metrics) != 0 {
		t.Errorf("Expected empty cluster to have no metrics, got %d metrics", len(cluster.Metrics))
	}
}

func TestSinglePointCluster(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{})

	points := []KDPoint{
		{X: 1.5, Y: 2.5, ID: 1, NumPoints: 1},
	}

	cluster := sc.createCluster(points)

	if cluster.X != 1.5 || cluster.Y != 2.5 {
		t.Errorf("Expected position (1.5,2.5), got (%f,%f)", cluster.X, cluster.Y)
	}

	if cluster.Count != 1 {
		t.Errorf("Expected count 1, got %d", cluster.Count)
	}
}


func TestNestedClusterWeights(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{})

	cluster1Points := []KDPoint{
		{X: 0, Y: 0, ID: 1, NumPoints: 1},
		{X: 0.1, Y: 0.1, ID: 2, NumPoints: 1},
		{X: 0.2, Y: 0.2, ID: 3, NumPoints: 1},
	}

	cluster1 := sc.createCluster(cluster1Points)

	superClusterPoints := []KDPoint{
		{X: cluster1.X, Y: cluster1.Y, ID: 4, NumPoints: cluster1.Count},
		{X: 1.0, Y: 1.0, ID: 5, NumPoints: 1},
		{X: 1.1, Y: 1.1, ID: 6, NumPoints: 1},
	}

	superCluster := sc.createCluster(superClusterPoints)

	expectedTotalPoints := uint32(5) // 3 from cluster1 + 2 individual points
	if superCluster.Count != expectedTotalPoints {
		t.Errorf("Expected total points to be %d, got %d", expectedTotalPoints, superCluster.Count)
	}
}

func TestClusterBoundsCalculation(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{NodeSize: 64})

	// Create KDPoints
	points := []KDPoint{
		{X: -10, Y: 5, ID: 1, NumPoints: 1},
		{X: 10, Y: -5, ID: 2, NumPoints: 1},
		{X: 0, Y: 0, ID: 3, NumPoints: 1},
	}

	// Build KD-tree
	sc.Tree = sc.buildKDTree(points)

	// Test bounds calculation
	if sc.Tree.Bounds.MinX != -10 || sc.Tree.Bounds.MaxX != 10 {
		t.Errorf("Expected X bounds [-10, 10], got [%f, %f]", sc.Tree.Bounds.MinX, sc.Tree.Bounds.MaxX)
	}
	if sc.Tree.Bounds.MinY != -5 || sc.Tree.Bounds.MaxY != 5 {
		t.Errorf("Expected Y bounds [-5, 5], got [%f, %f]", sc.Tree.Bounds.MinY, sc.Tree.Bounds.MaxY)
	}
}




func TestProjectionRoundTrip(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		NodeSize:  64,
		Radius:    40,
		Extent:    512,
		MinPoints: 3,
	})

	// Test some known coordinates
	testCases := []struct {
		lng, lat float32
		zoom     int
	}{
		{0, 0, 0},
		{180, 85, 10},
		{-180, -85, 5},
		{45, 45, 8},
	}

	for _, tc := range testCases {
		projected := sc.projectFast(tc.lng, tc.lat, tc.zoom)
		unprojected := sc.unprojectFast(projected[0], projected[1], tc.zoom)

		// Allow for small floating point differences
		const epsilon = 0.0001
		if math.Abs(float64(tc.lng-unprojected[0])) > epsilon ||
			math.Abs(float64(tc.lat-unprojected[1])) > epsilon {
			t.Errorf("Projection round trip failed for (%f,%f) at zoom %d: got (%f,%f)",
				tc.lng, tc.lat, tc.zoom, unprojected[0], unprojected[1])
		}
	}
}

func TestGetClusters(t *testing.T) {
	opts := SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	}
	sc := newTestSupercluster(t, "TEST_GetClusters", opts)

	// Create test points in Continental US
	testPoints := []Point{
		{ID: 1, X: -100.0, Y: 40.0, Metrics: map[string]float32{"value": 100}},
		{ID: 2, X: -100.1, Y: 40.1, Metrics: map[string]float32{"value": 200}},
		{ID: 3, X: -100.2, Y: 40.2, Metrics: map[string]float32{"value": 300}},
		{ID: 4, X: -105.0, Y: 35.0, Metrics: map[string]float32{"value": 400}}, // Far from others
	}

	// Load points
	if err := sc.Load(testPoints); err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Test getting clusters at different zoom levels
	bounds := KDBounds{
		MinX: -180.0,
		MinY: 20.0,
		MaxX: -60.0,
		MaxY: 50.0,
	}

	// Test at low zoom (should cluster some points).
	// The CH-backed path clusters points 2 and 3 together (they are very close),
	// while point 1 and point 4 remain as singletons — yielding 3 results total.
	clusters := sc.GetClusters(bounds, 5)

	if len(clusters) != 3 {
		t.Errorf("Expected 3 results at zoom 5 (1 cluster + 2 singletons), got %d", len(clusters))
	}

	// Test at high zoom (should not cluster)
	clusters = sc.GetClusters(bounds, 15)

	// Should have all individual points
	if len(clusters) != 4 {
		t.Errorf("Expected 4 points at zoom 15, got %d", len(clusters))
	}

	// Test metrics aggregation in clusters
	for _, c := range clusters {
		if c.Count > 1 {
			// Check if metrics are summed correctly
			if value, ok := c.Metrics["value"]; !ok || value == 0 {
				t.Error("Expected non-zero value metric in cluster")
			}
		}
	}
}

func TestLoad(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	})

	// Test loading empty points
	if err := sc.Load([]Point{}); err != nil {
		t.Fatalf("Load empty: %v", err)
	}
	if sc.Tree == nil {
		t.Error("Expected non-nil tree even with empty points")
	}

	// Test loading points with metrics and metadata
	points := []Point{
		{
			ID:       1,
			X:        -100.0,
			Y:        40.0,
			Metrics:  map[string]float32{"value": 100},
			Metadata: map[string]interface{}{"type": "store"},
		},
		{
			ID:       2,
			X:        -101.0,
			Y:        41.0,
			Metrics:  map[string]float32{"value": 200},
			Metadata: map[string]interface{}{"type": "store"},
		},
	}

	if err := sc.Load(points); err != nil {
		t.Fatalf("Load: %v", err)
	}

	if len(sc.Tree.Points) != len(points) {
		t.Errorf("Expected %d points in tree, got %d", len(points), len(sc.Tree.Points))
	}
}

func TestClusterQueryProfile(t *testing.T) {
	// Skip during normal testing unless explicitly enabled
	if testing.Short() {
		t.Skip("Skipping profile test in short mode")
	}

	filename := "test_data/cluster-300000p-20250226-120950-5899860c.zst" 

	// Parse zoom levels from environment, default to [2,5,10]
	zoomLevels := []int{2, 5, 10}
	if zoomStr := os.Getenv("ZOOM_LEVELS"); zoomStr != "" {
		var levels []int
		for _, s := range strings.Split(zoomStr, ",") {
			z, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil {
				t.Fatalf("Invalid zoom level in ZOOM_LEVELS: %s", s)
			}
			levels = append(levels, z)
		}
		zoomLevels = levels
	}

	// Load the cluster
	fmt.Printf("Loading cluster from %s\n", filename)
	start := time.Now()
	sc, err := LoadCompressedSupercluster(filename)
	if err != nil {
		t.Fatalf("Failed to load cluster: %v", err)
	}
	fmt.Printf("Loaded cluster in %v\n", time.Since(start))

	// Get the total bounds of the data
	bounds := sc.Tree.Bounds
	fmt.Printf("Total data bounds: MinX: %f, MinY: %f, MaxX: %f, MaxY: %f\n",
		bounds.MinX, bounds.MinY, bounds.MaxX, bounds.MaxY)

	// Define viewport sizes (in degrees) for different zoom levels
	viewportSizes := map[int]float32{
		2:  40.0, // Large viewport at low zoom
		5:  20.0, // Medium viewport
		10: 5.0,  // Small viewport at high zoom
	}

	// Define test viewports - we'll test center and corners of the data bounds
	type viewport struct {
		centerX, centerY float32
		description      string
	}

	// Calculate center and corner viewports
	centerX := (bounds.MinX + bounds.MaxX) / 2
	centerY := (bounds.MinY + bounds.MaxY) / 2
	viewports := []viewport{
		{centerX, centerY, "center"},
		{bounds.MinX + 1, bounds.MinY + 1, "bottom-left"},
		{bounds.MaxX - 1, bounds.MaxY - 1, "top-right"},
		{bounds.MinX + 1, bounds.MaxY - 1, "top-left"},
		{bounds.MaxX - 1, bounds.MinY + 1, "bottom-right"},
	}

	// Query each zoom level
	for _, zoom := range zoomLevels {
		fmt.Printf("\n=== Zoom Level %d ===\n", zoom)
		viewSize := viewportSizes[zoom]

		// Test each viewport position
		for _, vp := range viewports {
			// Calculate viewport bounds
			queryBounds := KDBounds{
				MinX: vp.centerX - viewSize/2,
				MaxX: vp.centerX + viewSize/2,
				MinY: vp.centerY - viewSize/2,
				MaxY: vp.centerY + viewSize/2,
			}

			fmt.Printf("\nQuerying %s viewport at zoom %d\n", vp.description, zoom)
			fmt.Printf("Viewport bounds: MinX: %f, MinY: %f, MaxX: %f, MaxY: %f\n",
				queryBounds.MinX, queryBounds.MinY, queryBounds.MaxX, queryBounds.MaxY)

			start := time.Now()
			clusters := sc.GetClusters(queryBounds, zoom)
			duration := time.Since(start)

			fmt.Printf("Found %d clusters in %v\n", len(clusters), duration)

			// Print some stats about the clusters
			var totalPoints uint32
			pointsPerCluster := make(map[uint32]int)
			for _, c := range clusters {
				totalPoints += c.Count
				pointsPerCluster[c.Count]++
			}

			fmt.Printf("Total points in viewport: %d\n", totalPoints)
			fmt.Printf("Cluster size distribution:\n")

			// Get sorted cluster sizes for consistent output
			var sizes []uint32
			for size := range pointsPerCluster {
				sizes = append(sizes, size)
			}
			sort.Slice(sizes, func(i, j int) bool { return sizes[i] < sizes[j] })

			for _, size := range sizes {
				count := pointsPerCluster[size]
				if count > 0 {
					fmt.Printf("  %d points: %d clusters\n", size, count)
				}
			}

			// Print a sample of cluster details
			if len(clusters) > 0 {
				fmt.Printf("\nSample cluster details (up to 3 clusters):\n")
				numSamples := min(3, len(clusters))
				for i := 0; i < numSamples; i++ {
					c := clusters[i]
					fmt.Printf("  Cluster %d: position=(%f,%f), points=%d\n",
						i, c.X, c.Y, c.Count)
					// Print first few metrics if any exist
					if len(c.Metrics) > 0 {
						fmt.Printf("    Metrics: ")
						printed := 0
						for k, v := range c.Metrics {
							if printed < 3 {
								fmt.Printf("%s=%.2f ", k, v)
								printed++
							}
						}
						fmt.Println()
					}
				}
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
