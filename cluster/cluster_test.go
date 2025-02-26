package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestClusterMetricsRollup(t *testing.T) {
	// Create a new supercluster with our metrics and metadata stores
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	})

	// Create test points with known metrics
	points := []Point{
		{
			ID:      1,
			X:       0,
			Y:       0,
			Metrics: map[string]float32{"sales": 100, "customers": 10},
		},
		{
			ID:      2,
			X:       0.1,
			Y:       0.1,
			Metrics: map[string]float32{"sales": 200, "customers": 20},
		},
		{
			ID:      3,
			X:       0.2,
			Y:       0.2,
			Metrics: map[string]float32{"sales": 300, "customers": 30},
		},
	}

	// Add points to metrics store
	for _, p := range points {
		sc.metricsStore.AddMetrics(p.ID, p.Metrics)
	}

	// Create KDPoints (with projected coordinates for clustering)
	kdPoints := []KDPoint{
		{X: 0, Y: 0, ID: 1, NumPoints: 1},
		{X: 0.1, Y: 0.1, ID: 2, NumPoints: 1},
		{X: 0.2, Y: 0.2, ID: 3, NumPoints: 1},
	}

	// Create a cluster from these points
	cluster := sc.createCluster(kdPoints)

	// Verify the metrics are summed correctly
	expectedSales := float32(600)    // 100 + 200 + 300
	expectedCustomers := float32(60) // 10 + 20 + 30

	if cluster.Metrics["sales"] != expectedSales {
		t.Errorf("Expected sales to be %f, got %f", expectedSales, cluster.Metrics["sales"])
	}
	if cluster.Metrics["customers"] != expectedCustomers {
		t.Errorf("Expected customers to be %f, got %f", expectedCustomers, cluster.Metrics["customers"])
	}

	// Test nested clusters by creating a "super cluster"
	// First, add the cluster we just created as a point in the metrics store
	sc.metricsStore.AddMetrics(4, cluster.Metrics) // ID 4 for first cluster

	// Add another cluster with different metrics
	sc.metricsStore.AddMetrics(5, map[string]float32{"sales": 400, "customers": 40})

	// Create KDPoints for the super cluster
	superKdPoints := []KDPoint{
		{X: 0.1, Y: 0.1, ID: 4, NumPoints: 3}, // Cluster with 3 points
		{X: 1, Y: 1, ID: 5, NumPoints: 2},     // Cluster with 2 points
	}

	superCluster := sc.createCluster(superKdPoints)

	// Verify the metrics are correctly aggregated
	expectedSuperSales := float32(1000)    // 600 + 400
	expectedSuperCustomers := float32(100) // 60 + 40

	if superCluster.Metrics["sales"] != expectedSuperSales {
		t.Errorf("Expected super-cluster sales to be %f, got %f", expectedSuperSales, superCluster.Metrics["sales"])
	}
	if superCluster.Metrics["customers"] != expectedSuperCustomers {
		t.Errorf("Expected super-cluster customers to be %f, got %f", expectedSuperCustomers, superCluster.Metrics["customers"])
	}

	// Verify the total number of points
	expectedTotalPoints := uint32(5) // 3 + 2
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

	// Add metrics and metadata for point ID 1
	sc.metricsStore.AddMetrics(1, map[string]float32{"value": 100})
	sc.metadataStore.AddMetadata(1, map[string]interface{}{
		"type": "store",
		"name": "Store A",
	})

	// Create a KDPoint (without the metadata and metrics - they're in the stores)
	points := []KDPoint{
		{X: 1.5, Y: 2.5, ID: 1, NumPoints: 1},
	}

	cluster := sc.createCluster(points)

	// Test position
	if cluster.X != 1.5 || cluster.Y != 2.5 {
		t.Errorf("Expected position (1.5,2.5), got (%f,%f)", cluster.X, cluster.Y)
	}

	// Test count
	if cluster.Count != 1 {
		t.Errorf("Expected count 1, got %d", cluster.Count)
	}

	// Test metrics
	if cluster.Metrics["value"] != 100 {
		t.Errorf("Expected value 100, got %f", cluster.Metrics["value"])
	}

	// Test metadata - first, get the JSON
	metadataJSON := sc.metadataStore.CalculateFrequencies([]uint32{1})

	// Then check if it contains the expected values
	if raw, ok := metadataJSON["type"]; !ok {
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
	sc := NewSupercluster(SuperclusterOptions{})

	// Add metadata for two points
	sc.metadataStore.AddMetadata(1, map[string]interface{}{
		"type": "store",
		"city": "New York",
	})
	sc.metadataStore.AddMetadata(2, map[string]interface{}{
		"type": "store",
		"city": "Boston",
	})

	// Add metrics
	sc.metricsStore.AddMetrics(1, map[string]float32{"value": 100})
	sc.metricsStore.AddMetrics(2, map[string]float32{"value": 200})

	// // Create KDPoints
	// points := []KDPoint{
	// 	{X: 0, Y: 0, ID: 1, NumPoints: 1},
	// 	{X: 0.1, Y: 0.1, ID: 2, NumPoints: 1},
	// }

	// Generate metadata frequencies for these points
	metadataJSON := sc.metadataStore.CalculateFrequencies([]uint32{1, 2})

	// Test metadata frequencies
	if raw, ok := metadataJSON["type"]; !ok {
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
	if raw, ok := metadataJSON["city"]; !ok {
		t.Error("Expected 'city' metadata to be preserved")
	} else {
		var freqMap map[string]float64
		if err := json.Unmarshal(raw, &freqMap); err != nil {
			t.Errorf("Failed to unmarshal city metadata: %v", err)
		}
		if freq, ok := freqMap["New York"]; !ok || math.Abs(freq-0.5) > 0.001 {
			t.Errorf("Expected frequency 0.5 for 'New York', got %f", freq)
		}
		if freq, ok := freqMap["Boston"]; !ok || math.Abs(freq-0.5) > 0.001 {
			t.Errorf("Expected frequency 0.5 for 'Boston', got %f", freq)
		}
	}
}

func TestNestedClusterWeights(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{})

	// Add metrics for individual points
	sc.metricsStore.AddMetrics(1, map[string]float32{"value": 100})
	sc.metricsStore.AddMetrics(2, map[string]float32{"value": 200})
	sc.metricsStore.AddMetrics(3, map[string]float32{"value": 300})

	// Create first-level cluster points
	cluster1Points := []KDPoint{
		{X: 0, Y: 0, ID: 1, NumPoints: 1},
		{X: 0.1, Y: 0.1, ID: 2, NumPoints: 1},
		{X: 0.2, Y: 0.2, ID: 3, NumPoints: 1},
	}

	// Create the first-level cluster
	cluster1 := sc.createCluster(cluster1Points)

	// Add the cluster1 metrics to store with ID 4
	sc.metricsStore.AddMetrics(4, cluster1.Metrics)

	// Add more individual points
	sc.metricsStore.AddMetrics(5, map[string]float32{"value": 400})
	sc.metricsStore.AddMetrics(6, map[string]float32{"value": 500})

	// Create super-cluster points
	superClusterPoints := []KDPoint{
		{X: cluster1.X, Y: cluster1.Y, ID: 4, NumPoints: cluster1.Count},
		{X: 1.0, Y: 1.0, ID: 5, NumPoints: 1},
		{X: 1.1, Y: 1.1, ID: 6, NumPoints: 1},
	}

	// Create the super-cluster
	superCluster := sc.createCluster(superClusterPoints)

	// Test total points
	expectedTotalPoints := uint32(5) // 3 from cluster1 + 2 individual points
	if superCluster.Count != expectedTotalPoints {
		t.Errorf("Expected total points to be %d, got %d", expectedTotalPoints, superCluster.Count)
	}

	// Test weighted sum of values
	expectedValue := float32(1500) // (100+200+300) + 400 + 500
	if superCluster.Metrics["value"] != expectedValue {
		t.Errorf("Expected super-cluster value to be %f, got %f", expectedValue, superCluster.Metrics["value"])
	}
}

func TestClusterBoundsCalculation(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{NodeSize: 64})

	// Add metrics for points
	sc.metricsStore.AddMetrics(1, map[string]float32{"value": 100})
	sc.metricsStore.AddMetrics(2, map[string]float32{"value": 200})
	sc.metricsStore.AddMetrics(3, map[string]float32{"value": 300})

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

func TestMetricsStoreDeduplication(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{})

	// Add same metrics multiple times
	metrics1 := map[string]float32{"value": 100, "count": 1}
	metrics2 := map[string]float32{"value": 100, "count": 1} // Same as metrics1
	metrics3 := map[string]float32{"value": 200, "count": 2} // Different metrics

	// Add metrics for multiple points
	sc.metricsStore.AddMetrics(1, metrics1)
	sc.metricsStore.AddMetrics(2, metrics2)
	sc.metricsStore.AddMetrics(3, metrics3)

	// Verify that identical metrics are correctly stored and retrieved
	metrics1Retrieved := sc.metricsStore.GetMetrics(1)
	metrics2Retrieved := sc.metricsStore.GetMetrics(2)
	metrics3Retrieved := sc.metricsStore.GetMetrics(3)

	// Test that identical metrics (1 and 2) have the same values
	if !metricsEqual(metrics1Retrieved, metrics2Retrieved) {
		t.Error("Expected identical metrics to have same values")
	}
	// Test that different metrics (1 and 3) have different values
	if metricsEqual(metrics1Retrieved, metrics3Retrieved) {
		t.Error("Expected different metrics to have different values")
	}

	// Verify the actual values are still correct
	if metrics1Retrieved["value"] != 100 || metrics1Retrieved["count"] != 1 {
		t.Errorf("Expected metrics1 value=100 count=1, got value=%f count=%f",
			metrics1Retrieved["value"], metrics1Retrieved["count"])
	}

	if metrics2Retrieved["value"] != 100 || metrics2Retrieved["count"] != 1 {
		t.Errorf("Expected metrics2 value=100 count=1, got value=%f count=%f",
			metrics2Retrieved["value"], metrics2Retrieved["count"])
	}

	if metrics3Retrieved["value"] != 200 || metrics3Retrieved["count"] != 2 {
		t.Errorf("Expected metrics3 value=200 count=2, got value=%f count=%f",
			metrics3Retrieved["value"], metrics3Retrieved["count"])
	}
}

// Helper function to compare metrics maps
func metricsEqual(m1, m2 map[string]float32) bool {
	if len(m1) != len(m2) {
		return false
	}
	for k, v1 := range m1 {
		if v2, ok := m2[k]; !ok || v1 != v2 {
			return false
		}
	}
	return true
}

func TestClusterWithNilMetadata(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{})

	// Add metrics for points (but no metadata for first point)
	sc.metricsStore.AddMetrics(1, map[string]float32{"value": 100})
	sc.metricsStore.AddMetrics(2, map[string]float32{"value": 200})

	// Only add metadata for the second point
	sc.metadataStore.AddMetadata(2, map[string]interface{}{
		"type": "store",
	})

	// Create KDPoints
	// points := []KDPoint{
	// 	{X: 0, Y: 0, ID: 1, NumPoints: 1},
	// 	{X: 0.1, Y: 0.1, ID: 2, NumPoints: 1},
	// }

	// Create cluster
	// cluster := sc.createCluster(points)

	// Get metadata for the cluster
	metadataJSON := sc.metadataStore.CalculateFrequencies([]uint32{1, 2})

	// Should have type metadata from the second point
	if len(metadataJSON) == 0 {
		t.Error("Expected non-empty metadata map in cluster")
	}

	// Specifically check for "type" metadata
	if raw, ok := metadataJSON["type"]; !ok {
		t.Error("Expected 'type' metadata to be preserved even with one point having nil metadata")
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

func TestMetricsStoreThreadSafety(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{})
	const numGoroutines = 10
	const numMetricsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(n int) {
			defer wg.Done()
			for j := 0; j < numMetricsPerGoroutine; j++ {
				// Use a unique point ID for each metrics set
				pointID := uint32(n*numMetricsPerGoroutine + j)
				metrics := map[string]float32{
					"value": float32(pointID),
				}
				sc.metricsStore.AddMetrics(pointID, metrics)
			}
		}(i)
	}

	wg.Wait()

	// Verify we can still retrieve metrics
	for i := 0; i < numGoroutines; i++ {
		for j := 0; j < numMetricsPerGoroutine; j++ {
			pointID := uint32(i*numMetricsPerGoroutine + j)
			metrics := sc.metricsStore.GetMetrics(pointID)
			if metrics == nil || metrics["value"] != float32(pointID) {
				t.Errorf("Failed to get correct metrics after concurrent operations for point %d", pointID)
				break
			}
		}
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

	// Should have 2 clusters: one for the 3 close points and one for the far point
	if len(clusters) != 2 {
		t.Errorf("Expected 2 clusters at zoom 5, got %d", len(clusters))
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

	// Check that metrics were added to the store
	metrics1 := sc.metricsStore.GetMetrics(1)
	if metrics1 == nil || metrics1["value"] != 100 {
		t.Error("Expected metrics to be added to metrics store")
	}

	// Check that metadata was added to the store
	metadata := sc.metadataStore.GetMetadata(1)
	if metadata == nil || metadata["type"] != "store" {
		t.Error("Expected metadata to be added to metadata store")
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
