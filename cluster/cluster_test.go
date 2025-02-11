package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"testing"
)

func TestClusterMetricsRollup(t *testing.T) {
	// Create a test metrics pool
	pool := NewMetricsPool()

	// Create test points with known metrics
	points := []KDPoint{
		{
			X: 0, Y: 0,
			ID:        1,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"sales": 100, "customers": 10}),
		},
		{
			X: 0.1, Y: 0.1,
			ID:        2,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"sales": 200, "customers": 20}),
		},
		{
			X: 0.2, Y: 0.2,
			ID:        3,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"sales": 300, "customers": 30}),
		},
	}

	// Create a cluster from these points
	cluster := createCluster(points, pool)

	// Verify the metrics are summed correctly
	expectedSales := float32(600)    // 100 + 200 + 300
	expectedCustomers := float32(60) // 10 + 20 + 30

	if cluster.Metrics.Values["sales"] != expectedSales {
		t.Errorf("Expected sales to be %f, got %f", expectedSales, cluster.Metrics.Values["sales"])
	}
	if cluster.Metrics.Values["customers"] != expectedCustomers {
		t.Errorf("Expected customers to be %f, got %f", expectedCustomers, cluster.Metrics.Values["customers"])
	}

	// Test nested clusters
	nestedPoints := []KDPoint{
		{
			X: 0, Y: 0,
			ID:        4,
			NumPoints: 3,                                                           // This represents a cluster of 3 points
			MetricIdx: pool.Add(map[string]float32{"sales": 600, "customers": 60}), // The cluster we created above
		},
		{
			X: 1, Y: 1,
			ID:        5,
			NumPoints: 2, // Another cluster
			MetricIdx: pool.Add(map[string]float32{"sales": 400, "customers": 40}),
		},
	}

	// Create a super-cluster containing nested clusters
	superCluster := createCluster(nestedPoints, pool)

	// Verify the metrics are correctly weighted and summed
	expectedSuperSales := float32(1000)    // 600 + 400
	expectedSuperCustomers := float32(100) // 60 + 40

	if superCluster.Metrics.Values["sales"] != expectedSuperSales {
		t.Errorf("Expected super-cluster sales to be %f, got %f", expectedSuperSales, superCluster.Metrics.Values["sales"])
	}
	if superCluster.Metrics.Values["customers"] != expectedSuperCustomers {
		t.Errorf("Expected super-cluster customers to be %f, got %f", expectedSuperCustomers, superCluster.Metrics.Values["customers"])
	}

	// Verify the total number of points
	expectedTotalPoints := uint32(5) // 3 + 2
	if superCluster.Count != expectedTotalPoints {
		t.Errorf("Expected total points to be %d, got %d", expectedTotalPoints, superCluster.Count)
	}
}

func TestEmptyCluster(t *testing.T) {
	pool := NewMetricsPool()
	points := []KDPoint{}

	// Test creating cluster with no points
	cluster := createCluster(points, pool)

	if cluster.Count != 0 {
		t.Errorf("Expected empty cluster count to be 0, got %d", cluster.Count)
	}
	if len(cluster.Metrics.Values) != 0 {
		t.Errorf("Expected empty cluster to have no metrics, got %d metrics", len(cluster.Metrics.Values))
	}
}

func TestSinglePointCluster(t *testing.T) {
	pool := NewMetricsPool()
	points := []KDPoint{
		{
			X: 1.5, Y: 2.5,
			ID:        1,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 100}),
			Metadata: map[string]interface{}{
				"type": "store",
				"name": "Store A",
			},
		},
	}

	cluster := createCluster(points, pool)

	// Test position
	if cluster.X != 1.5 || cluster.Y != 2.5 {
		t.Errorf("Expected position (1.5,2.5), got (%f,%f)", cluster.X, cluster.Y)
	}

	// Test count
	if cluster.Count != 1 {
		t.Errorf("Expected count 1, got %d", cluster.Count)
	}

	// Test metrics
	if cluster.Metrics.Values["value"] != 100 {
		t.Errorf("Expected value 100, got %f", cluster.Metrics.Values["value"])
	}

	// Test metadata frequency
	if raw, ok := cluster.Metadata["type"]; !ok {
		t.Error("Expected 'type' metadata to be preserved")
	} else {
		var freqMap map[string]float64
		if err := json.Unmarshal(raw, &freqMap); err != nil {
			t.Errorf("Failed to unmarshal type metadata: %v", err)
		}
		if freq, ok := freqMap["store"]; !ok || freq != 1.0 {
			t.Errorf("Expected frequency 1.0 for 'store', got %f", freq)
		}
	}
}

func TestClusterWithMixedMetadata(t *testing.T) {
	pool := NewMetricsPool()
	points := []KDPoint{
		{
			X: 0, Y: 0,
			ID:        1,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 100}),
			Metadata: map[string]interface{}{
				"type": "store",
				"city": "New York",
			},
		},
		{
			X: 0.1, Y: 0.1,
			ID:        2,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 200}),
			Metadata: map[string]interface{}{
				"type": "store",
				"city": "Boston",
			},
		},
	}

	cluster := createCluster(points, pool)

	// Test metadata frequencies
	if raw, ok := cluster.Metadata["type"]; !ok {
		t.Error("Expected 'type' metadata to be preserved")
	} else {
		var freqMap map[string]float64
		if err := json.Unmarshal(raw, &freqMap); err != nil {
			t.Errorf("Failed to unmarshal type metadata: %v", err)
		}
		if freq, ok := freqMap["store"]; !ok || freq != 1.0 {
			t.Errorf("Expected frequency 1.0 for 'store', got %f", freq)
		}
	}

	// Test city frequencies
	if raw, ok := cluster.Metadata["city"]; !ok {
		t.Error("Expected 'city' metadata to be preserved")
	} else {
		var freqMap map[string]float64
		if err := json.Unmarshal(raw, &freqMap); err != nil {
			t.Errorf("Failed to unmarshal city metadata: %v", err)
		}
		if freq, ok := freqMap["New York"]; !ok || freq != 0.5 {
			t.Errorf("Expected frequency 0.5 for 'New York', got %f", freq)
		}
		if freq, ok := freqMap["Boston"]; !ok || freq != 0.5 {
			t.Errorf("Expected frequency 0.5 for 'Boston', got %f", freq)
		}
	}
}

func TestNestedClusterWeights(t *testing.T) {
	pool := NewMetricsPool()

	// Create a cluster of 3 points
	cluster1Points := []KDPoint{
		{
			X: 0, Y: 0,
			ID:        1,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 100}),
		},
		{
			X: 0.1, Y: 0.1,
			ID:        2,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 200}),
		},
		{
			X: 0.2, Y: 0.2,
			ID:        3,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 300}),
		},
	}

	cluster1 := createCluster(cluster1Points, pool)
	fmt.Println(cluster1.Metrics.Values)

	// Create another cluster using the first cluster and two more points
	superClusterPoints := []KDPoint{
		{
			X: cluster1.X, Y: cluster1.Y,
			ID:        4,
			NumPoints: cluster1.Count,
			MetricIdx: pool.Add(cluster1.Metrics.Values),
		},
		{
			X: 1.0, Y: 1.0,
			ID:        5,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 400}),
		},
		{
			X: 1.1, Y: 1.1,
			ID:        6,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 500}),
		},
	}

	superCluster := createCluster(superClusterPoints, pool)

	// Test total points
	expectedTotalPoints := uint32(5) // 3 from cluster1 + 2 individual points
	if superCluster.Count != expectedTotalPoints {
		t.Errorf("Expected total points to be %d, got %d", expectedTotalPoints, superCluster.Count)
	}

	// Test weighted sum of values
	expectedValue := float32(1500) // (100+200+300) + 400 + 500
	if superCluster.Metrics.Values["value"] != expectedValue {
		t.Errorf("Expected super-cluster value to be %f, got %f", expectedValue, superCluster.Metrics.Values["value"])
	}
}

func TestClusterBoundsCalculation(t *testing.T) {
	pool := NewMetricsPool()
	points := []KDPoint{
		{X: -10, Y: 5, ID: 1, NumPoints: 1, MetricIdx: pool.Add(map[string]float32{"value": 100})},
		{X: 10, Y: -5, ID: 2, NumPoints: 1, MetricIdx: pool.Add(map[string]float32{"value": 200})},
		{X: 0, Y: 0, ID: 3, NumPoints: 1, MetricIdx: pool.Add(map[string]float32{"value": 300})},
	}

	tree := NewKDTree(points, 64, pool)

	// Test bounds calculation
	if tree.Bounds.MinX != -10 || tree.Bounds.MaxX != 10 {
		t.Errorf("Expected X bounds [-10, 10], got [%f, %f]", tree.Bounds.MinX, tree.Bounds.MaxX)
	}
	if tree.Bounds.MinY != -5 || tree.Bounds.MaxY != 5 {
		t.Errorf("Expected Y bounds [-5, 5], got [%f, %f]", tree.Bounds.MinY, tree.Bounds.MaxY)
	}
}

func TestMetricsPoolDeduplication(t *testing.T) {
	pool := NewMetricsPool()

	// Add same metrics multiple times
	metrics1 := map[string]float32{"value": 100, "count": 1}
	metrics2 := map[string]float32{"value": 100, "count": 1}

	idx1 := pool.Add(metrics1)
	idx2 := pool.Add(metrics2)

	// Should get same index for identical metrics
	if idx1 != idx2 {
		t.Errorf("Expected same index for identical metrics, got %d and %d", idx1, idx2)
	}

	// Length of pool should be 1
	if len(pool.Metrics) != 1 {
		t.Errorf("Expected metrics pool length 1, got %d", len(pool.Metrics))
	}
}

func TestClusterWithNilMetadata(t *testing.T) {
	pool := NewMetricsPool()
	points := []KDPoint{
		{
			X: 0, Y: 0,
			ID:        1,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 100}),
			Metadata:  nil,
		},
		{
			X: 0.1, Y: 0.1,
			ID:        2,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 200}),
			Metadata: map[string]interface{}{
				"type": "store",
			},
		},
	}

	cluster := createCluster(points, pool)

	// Should handle nil metadata gracefully
	if cluster.Metadata == nil {
		t.Error("Expected non-nil metadata map in cluster")
	}
}

func TestMetricsPoolThreadSafety(t *testing.T) {
	pool := NewMetricsPool()
	const numGoroutines = 10
	const numMetricsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(n int) {
			defer wg.Done()
			for j := 0; j < numMetricsPerGoroutine; j++ {
				metrics := map[string]float32{
					"value": float32(n*numMetricsPerGoroutine + j),
				}
				pool.Add(metrics)
			}
		}(i)
	}

	wg.Wait()

	// Verify no data races occurred by checking if we can still add and get metrics
	testIdx := pool.Add(map[string]float32{"test": 1.0})
	if pool.Get(testIdx) == nil {
		t.Error("Failed to get metrics after concurrent operations")
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
	// Create a test supercluster
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	})

	// Create test points in Continental US
	testPoints := []Point{
		{ID: 1, X: -100.0, Y: 40.0, Metrics: map[string]float32{"value": 100}},
		{ID: 2, X: -100.1, Y: 40.1, Metrics: map[string]float32{"value": 200}},
		{ID: 3, X: -100.2, Y: 40.2, Metrics: map[string]float32{"value": 300}},
		{ID: 4, X: -105.0, Y: 35.0, Metrics: map[string]float32{"value": 400}}, // Far from others
	}

	// Load points
	sc.Load(testPoints)

	// Test getting clusters at different zoom levels
	bounds := KDBounds{
		MinX: -180.0,
		MinY: 20.0,
		MaxX: -60.0,
		MaxY: 50.0,
	}

	// Test at low zoom (should cluster)
	clusters := sc.GetClusters(bounds, 5)
	if len(clusters) != 2 { // Should have 2 clusters: 3 close points + 1 far point
		t.Errorf("Expected 2 clusters at zoom 5, got %d", len(clusters))
	}

	// Test at high zoom (should not cluster)
	clusters = sc.GetClusters(bounds, 15)
	if len(clusters) != 4 { // Should have all individual points
		t.Errorf("Expected 4 points at zoom 15, got %d", len(clusters))
	}

	// Test metrics aggregation in clusters
	for _, c := range clusters {
		if c.Count > 1 {
			// Check if metrics are summed correctly
			if value, ok := c.Metrics.Values["value"]; !ok || value == 0 {
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
	sc.Load([]Point{})
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

	sc.Load(points)

	if len(sc.Tree.Points) != len(points) {
		t.Errorf("Expected %d points in tree, got %d", len(points), len(sc.Tree.Points))
	}

	// Verify metrics were added to pool
	if sc.Tree.Pool == nil || len(sc.Tree.Pool.Metrics) == 0 {
		t.Error("Expected metrics to be added to pool")
	}
}

func TestProjectPoints(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	})

	pool := NewMetricsPool()
	points := []KDPoint{
		{
			X:         -100.0, // longitude
			Y:         40.0,   // latitude
			ID:        1,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 100}),
		},
		{
			X:         -120.0,
			Y:         35.0,
			ID:        2,
			NumPoints: 1,
			MetricIdx: pool.Add(map[string]float32{"value": 200}),
		},
	}

	// Test projection at different zoom levels
	zooms := []int{0, 8, 16}
	for _, zoom := range zooms {
		projected := sc.projectPoints(points, zoom, pool)

		if len(projected) != len(points) {
			t.Errorf("Expected %d projected points at zoom %d, got %d",
				len(points), zoom, len(projected))
		}

		// Verify projection and unprojection round trip
		for i, p := range projected {
			unproj := sc.unprojectFast(p.X, p.Y, zoom)

			// Allow for small floating point differences
			if math.Abs(float64(unproj[0]-points[i].X)) > 0.0001 ||
				math.Abs(float64(unproj[1]-points[i].Y)) > 0.0001 {
				t.Errorf("Projection round trip failed at zoom %d for point %d", zoom, i)
			}

			// Verify metrics and metadata were preserved
			if p.MetricIdx != points[i].MetricIdx {
				t.Errorf("Expected MetricIdx %d, got %d", points[i].MetricIdx, p.MetricIdx)
			}
		}
	}
}

func TestClusterPoints(t *testing.T) {
	// Create test points first
	pool := NewMetricsPool()
	points := []KDPoint{
		// Cluster 1: 3 points close together
		{X: 0, Y: 0, ID: 1, NumPoints: 1, MetricIdx: pool.Add(map[string]float32{"value": 100})},
		{X: 10, Y: 10, ID: 2, NumPoints: 1, MetricIdx: pool.Add(map[string]float32{"value": 200})},
		{X: 20, Y: 20, ID: 3, NumPoints: 1, MetricIdx: pool.Add(map[string]float32{"value": 300})},

		// Isolated point
		{X: 1000, Y: 1000, ID: 4, NumPoints: 1, MetricIdx: pool.Add(map[string]float32{"value": 400})},
	}

	// Create and initialize the supercluster with the points
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    100, // Large enough to cluster nearby points
		Extent:    512,
		NodeSize:  64,
	})

	// Initialize the tree with the points
	sc.Tree = NewKDTree(points, sc.Options.NodeSize, pool)

	clusters := sc.clusterPoints(points, 50) // Radius that should cluster the first 3 points

	// Should have 2 clusters: one with 3 points and one isolated point
	if len(clusters) != 2 {
		t.Errorf("Expected 2 clusters, got %d", len(clusters))
	}

	// Verify cluster properties
	for _, c := range clusters {
		if c.Count == 3 {
			// Check metrics aggregation
			if value, ok := c.Metrics.Values["value"]; !ok || value != 600 { // 100 + 200 + 300
				t.Errorf("Expected cluster value of 600, got %f", value)
			}
		} else if c.Count == 1 {
			// Check isolated point
			if value, ok := c.Metrics.Values["value"]; !ok || value != 400 {
				t.Errorf("Expected point value of 400, got %f", value)
			}
		} else {
			t.Errorf("Unexpected cluster count: %d", c.Count)
		}
	}
}

