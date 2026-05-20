package cluster

import (
	"context"
	"math"
	"os"
	"strconv"
	"testing"
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

	if sc.Skeleton == nil {
		t.Error("Expected non-nil skeleton after Load")
	}
}


func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
