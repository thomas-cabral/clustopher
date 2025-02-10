package cluster

import (
	"encoding/json"
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

	// Test metadata preservation
	if raw, ok := cluster.Metadata["type"]; !ok {
		t.Error("Expected 'type' metadata to be preserved")
	} else {
		var typeStr string
		if err := json.Unmarshal(raw, &typeStr); err != nil {
			t.Errorf("Failed to unmarshal type metadata: %v", err)
		}
		if typeStr != "store" {
			t.Errorf("Expected type 'store', got '%s'", typeStr)
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

	// Test that only common metadata is preserved
	if raw, ok := cluster.Metadata["type"]; !ok {
		t.Error("Expected common 'type' metadata to be preserved")
	} else {
		var typeStr string
		if err := json.Unmarshal(raw, &typeStr); err != nil {
			t.Errorf("Failed to unmarshal type metadata: %v", err)
		}
		if typeStr != "store" {
			t.Errorf("Expected type 'store', got '%s'", typeStr)
		}
	}

	// Test that different metadata is not preserved
	if _, ok := cluster.Metadata["city"]; ok {
		t.Error("Expected 'city' metadata to be dropped due to different values")
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
		t.Errorf("Expected total value to be %f, got %f", expectedValue, superCluster.Metrics.Values["value"])
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
