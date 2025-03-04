package cluster

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

// generateRandomPoints creates n random points within a geographic bounding box
func generateRandomPoints(n int, minLng, maxLng, minLat, maxLat float32) []Point {
	points := make([]Point, n)
	// Use deterministic seed for reproducibility
	source := rand.NewSource(42)
	r := rand.New(source)

	for i := 0; i < n; i++ {
		points[i] = Point{
			ID: uint32(i + 1),
			X:  minLng + r.Float32()*(maxLng-minLng),
			Y:  minLat + r.Float32()*(maxLat-minLat),
			Metrics: map[string]float32{
				"value": r.Float32() * 100,
			},
			Metadata: map[string]interface{}{
				"type": "test",
			},
		}
	}
	return points
}

// generateKDPoints creates KDPoints from regular points at a specific zoom level
func generateKDPoints(sc *Supercluster, points []Point, zoom int) []KDPoint {
	kdPoints := make([]KDPoint, len(points))
	for i, p := range points {
		projected := sc.projectFast(p.X, p.Y, zoom)
		kdPoints[i] = KDPoint{
			ID:        p.ID,
			X:         projected[0],
			Y:         projected[1],
			NumPoints: 1,
		}
	}
	return kdPoints
}

// benchmarkClustering runs clustering benchmarks with different point counts and zoom levels
func benchmarkClustering(b *testing.B, numPoints int, zoom int) {
	// Create the supercluster with default options
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 3,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
		Log:       false,
	})

	// Generate random points in the US region
	points := generateRandomPoints(numPoints, -125.0, -65.0, 25.0, 49.0)

	// Convert to KDPoints at the specified zoom level
	kdPoints := generateKDPoints(sc, points, zoom)

	// Track memory usage before and after
	var memStatsBefore, memStatsAfter runtime.MemStats
	runtime.ReadMemStats(&memStatsBefore)

	// Reset timer before the actual benchmark
	b.ResetTimer()

	// Run benchmarks
	for i := 0; i < b.N; i++ {
		// Choose the most appropriate clustering method based on data characteristics
		// This matches the logic we want to use in the real application
		if numPoints > 50000 ||
			(numPoints > 10000 && zoom < sc.Options.MaxZoom/2) ||
			zoom < sc.Options.MaxZoom/4 {
			// For very large datasets or low zoom levels, use grid-based clustering
			sc.clusterPointsWithGrid(kdPoints, float32(sc.Options.Radius), zoom)
		} else if numPoints > 5000 && zoom > sc.Options.MaxZoom/2 {
			// For medium to large datasets at higher zoom levels, use KDTree-based clustering
			sc.clusterPointsWithKDTree(kdPoints, float32(sc.Options.Radius), zoom)
		} else {
			// For small datasets, use traditional clustering
			sc.clusterPoints(kdPoints, float32(sc.Options.Radius))
		}
	}

	b.StopTimer()

	// Measure memory after benchmark
	runtime.ReadMemStats(&memStatsAfter)
	allocMB := float64(memStatsAfter.TotalAlloc-memStatsBefore.TotalAlloc) / 1024 / 1024

	// Report additional metrics
	b.ReportMetric(allocMB, "MB/op")
}

// Benchmark with different point sizes and zoom levels
func BenchmarkClusteringSmall_LowZoom(b *testing.B) {
	benchmarkClustering(b, 1000, 2)
}

func BenchmarkClusteringSmall_MidZoom(b *testing.B) {
	benchmarkClustering(b, 1000, 8)
}

func BenchmarkClusteringSmall_HighZoom(b *testing.B) {
	benchmarkClustering(b, 1000, 14)
}

func BenchmarkClusteringMedium_LowZoom(b *testing.B) {
	benchmarkClustering(b, 10000, 2)
}

func BenchmarkClusteringMedium_MidZoom(b *testing.B) {
	benchmarkClustering(b, 10000, 8)
}

func BenchmarkClusteringMedium_HighZoom(b *testing.B) {
	benchmarkClustering(b, 10000, 14)
}

func BenchmarkClusteringLarge_LowZoom(b *testing.B) {
	benchmarkClustering(b, 100000, 2)
}

func BenchmarkClusteringLarge_MidZoom(b *testing.B) {
	benchmarkClustering(b, 100000, 8)
}

func BenchmarkClusteringLarge_HighZoom(b *testing.B) {
	benchmarkClustering(b, 100000, 14)
}

// TestProfileClustering profiles clustering by comparing methods
func TestProfileClustering(t *testing.T) {
	// Skip during normal testing unless explicitly enabled
	if testing.Short() {
		t.Skip("Skipping profile test in short mode")
	}

	pointCounts := []int{1000, 10000, 100000}
	zoomLevels := []int{2, 8, 14}

	fmt.Println("Starting clustering profiling...")
	fmt.Println("=================================")

	for _, numPoints := range pointCounts {
		for _, zoom := range zoomLevels {
			// Create supercluster
			sc := NewSupercluster(SuperclusterOptions{
				MinZoom:   0,
				MaxZoom:   16,
				MinPoints: 3,
				Radius:    40,
				Extent:    512,
				NodeSize:  64,
				Log:       false,
			})

			// Generate points
			fmt.Printf("Testing %d points at zoom level %d\n", numPoints, zoom)
			points := generateRandomPoints(numPoints, -125.0, -65.0, 25.0, 49.0)
			kdPoints := generateKDPoints(sc, points, zoom)

			// Measure original clustering (old logic)
			var originalDuration time.Duration
			{
				start := time.Now()
				if len(kdPoints) > 50000 {
					sc.clusterPointsWithGrid(kdPoints, float32(sc.Options.Radius))
				} else {
					sc.clusterPoints(kdPoints, float32(sc.Options.Radius))
				}
				originalDuration = time.Since(start)
			}

			// Measure grid-based clustering
			var gridDuration time.Duration
			{
				start := time.Now()
				if len(kdPoints) > 50000 ||
					(len(kdPoints) > 10000 && zoom < sc.Options.MaxZoom/2) ||
					zoom < sc.Options.MaxZoom/4 {
					sc.clusterPointsWithGrid(kdPoints, float32(sc.Options.Radius), zoom)
				} else {
					sc.clusterPoints(kdPoints, float32(sc.Options.Radius))
				}
				gridDuration = time.Since(start)
			}

			// Measure KDTree-based clustering
			var kdTreeDuration time.Duration
			{
				start := time.Now()
				if numPoints > 5000 && zoom > sc.Options.MaxZoom/3 {
					sc.clusterPointsWithKDTree(kdPoints, float32(sc.Options.Radius), zoom)
				} else if len(kdPoints) > 50000 ||
					(len(kdPoints) > 10000 && zoom < sc.Options.MaxZoom/2) ||
					zoom < sc.Options.MaxZoom/4 {
					sc.clusterPointsWithGrid(kdPoints, float32(sc.Options.Radius), zoom)
				} else {
					sc.clusterPoints(kdPoints, float32(sc.Options.Radius))
				}
				kdTreeDuration = time.Since(start)
			}

			// Calculate improvements
			gridImprovement := float64(originalDuration-gridDuration) / float64(originalDuration) * 100
			kdTreeImprovement := float64(originalDuration-kdTreeDuration) / float64(originalDuration) * 100

			fmt.Printf("  Original: %v\n", originalDuration)
			fmt.Printf("  Grid:     %v (%.2f%%)\n", gridDuration, gridImprovement)
			fmt.Printf("  KDTree:   %v (%.2f%%)\n", kdTreeDuration, kdTreeImprovement)
			fmt.Println()
		}
	}
}
