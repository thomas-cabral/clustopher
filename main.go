package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
	"web/clustopher/cluster"

	"github.com/gin-gonic/gin"
)

// Example usage
func ExportClustersToFile(filename string) error {
	// Setup your cluster as before
	bounds := cluster.KDBounds{
		MinX: -125.0, // Roughly West Coast of US
		MinY: 25.0,   // Roughly Southern US border
		MaxX: -67.0,  // Roughly East Coast of US
		MaxY: 49.0,   // Roughly Northern US border
	}

	points := generateTestPoints(3000000, bounds)
	options := cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    100,
		Extent:    512,
		NodeSize:  64,
	}

	cluster := cluster.NewSupercluster(options)
	cluster.Load(points)

	// Convert to GeoJSON
	geojson, err := cluster.ToGeoJSON(bounds, 10) // Zoom level 10 for example
	if err != nil {
		return fmt.Errorf("failed to convert to GeoJSON: %v", err)
	}

	// Write to file
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(geojson); err != nil {
		return fmt.Errorf("failed to encode GeoJSON: %v", err)
	}

	return nil
}

type ClusterServer struct {
	cluster *cluster.Supercluster
}

func NewClusterServer(numPoints int) *ClusterServer {
	// Use the existing point generation code
	bounds := cluster.KDBounds{
		MinX: -125.0, // Roughly West Coast of US
		MinY: 25.0,   // Roughly Southern US border
		MaxX: -67.0,  // Roughly East Coast of US
		MaxY: 49.0,   // Roughly Northern US border
	}

	fmt.Printf("Generating %d points in the Continental US...\n", numPoints)
	points := generateTestPoints(numPoints, bounds)

	options := cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    100,
		Extent:    512,
		NodeSize:  64,
		Log:       true,
	}

	fmt.Printf("Initializing supercluster with options: %+v\n", options)
	cluster := cluster.NewSupercluster(options)

	// Memory profiling before loading points
	var memStatsBefore runtime.MemStats
	runtime.ReadMemStats(&memStatsBefore)

	fmt.Println("Loading points into cluster...")
	loadStart := time.Now()
	cluster.Load(points)
	loadDuration := time.Since(loadStart)
	fmt.Printf("Points loaded in %v\n", loadDuration)

	// Memory profiling after loading points
	var memStatsAfter runtime.MemStats
	runtime.ReadMemStats(&memStatsAfter)

	// Calculate memory usage
	memAllocated := memStatsAfter.Alloc - memStatsBefore.Alloc
	memSys := memStatsAfter.Sys - memStatsBefore.Sys
	numGC := memStatsAfter.NumGC - memStatsBefore.NumGC

	fmt.Printf("Memory Usage Stats:\n")
	fmt.Printf("  Allocated Memory: %v bytes\n", memAllocated)
	fmt.Printf("  System Memory: %v bytes\n", memSys)
	fmt.Printf("  Number of GC: %v\n", numGC)

	return &ClusterServer{
		cluster: cluster,
	}
}

func main() {
	// Allow configuring number of points via environment variable or default to 10000
	numPoints := 10000000

	fmt.Println("Starting NewClusterServer...")
	start := time.Now()
	server := NewClusterServer(numPoints)
	duration := time.Since(start)
	fmt.Printf("NewClusterServer initialized in %v\n", duration)

	r := gin.Default()

	// Enable CORS
	r.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Origin, Content-Type")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// Get clusters based on zoom and bounds
	r.GET("/api/clusters", func(c *gin.Context) {
		zoom, err := strconv.Atoi(c.Query("zoom"))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid zoom parameter"})
			return
		}

		north, err := strconv.ParseFloat(c.Query("north"), 64)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid north parameter"})
			return
		}

		south, err := strconv.ParseFloat(c.Query("south"), 64)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid south parameter"})
			return
		}

		east, err := strconv.ParseFloat(c.Query("east"), 64)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid east parameter"})
			return
		}

		west, err := strconv.ParseFloat(c.Query("west"), 64)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid west parameter"})
			return
		}

		bounds := cluster.KDBounds{
			MinX: float32(west),
			MinY: float32(south),
			MaxX: float32(east),
			MaxY: float32(north),
		}

		fmt.Printf("Getting clusters for zoom %d with bounds: %+v\n", zoom, bounds)

		clusters := server.cluster.GetClusters(bounds, zoom)

		// Convert to GeoJSON
		features := make([]map[string]interface{}, len(clusters))
		for i, cluster := range clusters {
			properties := map[string]interface{}{
				"cluster":     cluster.Count > 1,
				"point_count": cluster.Count,
			}

			if cluster.Metrics.Values != nil {
				properties["metrics"] = cluster.Metrics.Values
			}

			features[i] = map[string]interface{}{
				"type": "Feature",
				"geometry": map[string]interface{}{
					"type":        "Point",
					"coordinates": []float64{float64(cluster.X), float64(cluster.Y)},
				},
				"properties": properties,
			}
		}

		geojson := map[string]interface{}{
			"type":     "FeatureCollection",
			"features": features,
		}

		c.JSON(http.StatusOK, geojson)
	})

	// Get statistics about the current clustering
// 	r.GET("/api/stats", func(c *gin.Context) {
// 		stats := map[string]interface{}{
// 			"total_points": len(server.cluster.Points),
// 			"options":      server.cluster.Options,
// 			"zoom_levels":  map[int]int{},
// 		}

// 		// Count points at each zoom level
// 		for zoom, tree := range server.cluster.Tree {
// 			if tree != nil {
// 				stats["zoom_levels"].(map[int]int)[zoom] = len(tree.Points)
// 			}
// 		}

// 		// Get current memory stats
// 		var memStats runtime.MemStats
// 		runtime.ReadMemStats(&memStats)

// 		stats["memory"] = map[string]interface{}{
// 			"Alloc":      memStats.Alloc,
// 			"TotalAlloc": memStats.TotalAlloc,
// 			"Sys":        memStats.Sys,
// 			"NumGC":      memStats.NumGC,
// 		}

// 		c.JSON(http.StatusOK, stats)
// 	})

	fmt.Println("Starting server on :8080...")
	r.Run(":8000")
}

// func min(a, b int) int {
// 	if a < b {
// 		return a
// 	}
// 	return b
// }

// generateTestPoints creates n random points within specified bounds
func generateTestPoints(n int, bounds cluster.KDBounds) []cluster.Point {
	rand.Seed(time.Now().UnixNano())
	points := make([]cluster.Point, n)

	for i := 0; i < n; i++ {
		x := bounds.MinX + rand.Float32()*(bounds.MaxX-bounds.MinX)
		y := bounds.MinY + rand.Float32()*(bounds.MaxY-bounds.MinY)

		points[i] = cluster.Point{
			ID: uint32(i + 1),
			X:  x,
			Y:  y,
			Metrics: map[string]float32{
				"value": rand.Float32() * 100,
				"size":  rand.Float32() * 50,
			},
			Metadata: map[string]interface{}{
				"timestamp": time.Now().Add(-time.Duration(rand.Intn(7*24)) * time.Hour),
				"category":  []string{"A", "B", "C"}[rand.Intn(3)],
			},
		}
	}

	return points
}

func TestSupercluster(t *testing.T) {
	// Test configuration
	bounds := cluster.KDBounds{
		MinX: -125.0, // Roughly West Coast of US
		MinY: 25.0,   // Roughly Southern US border
		MaxX: -67.0,  // Roughly East Coast of US
		MaxY: 49.0,   // Roughly Northern US border
	}

	numPoints := 1000
	points := generateTestPoints(numPoints, bounds)

	// Initialize cluster with options
	options := cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 3,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
		Log:       true,
	}

	cluster := cluster.NewSupercluster(options)
	cluster.Load(points)

	// Test different zoom levels
	testZoomLevels := []int{0, 5, 10, 15}
	for _, zoom := range testZoomLevels {
		t.Run("TestZoomLevel"+string(rune(zoom+'0')), func(t *testing.T) {
			clusters := cluster.GetClusters(bounds, zoom)

			// Basic validation
			if len(clusters) == 0 {
				t.Errorf("Expected clusters at zoom level %d, got none", zoom)
			}

			// Validate cluster properties
			for _, c := range clusters {
				if c.Count == 0 {
					t.Errorf("Cluster has zero points at zoom %d", zoom)
				}

				if c.X < bounds.MinX || c.X > bounds.MaxX || c.Y < bounds.MinY || c.Y > bounds.MaxY {
					t.Errorf("Cluster at zoom %d is outside bounds: (%f, %f)", zoom, c.X, c.Y)
				}

				// Check metrics
				for metricName, value := range c.Metrics.Values {
					if value < 0 || value > 100 {
						t.Errorf("Invalid metric value for %s: %f", metricName, value)
					}
				}
			}

			// Verify clustering behavior
			if zoom == 0 {
				// At lowest zoom, expect fewer clusters
				if len(clusters) > numPoints/options.MinPoints {
					t.Errorf("Too many clusters at zoom 0: %d", len(clusters))
				}
			} else if zoom == options.MaxZoom {
				// At max zoom, expect most points to be unclustered
				if len(clusters) < numPoints/2 {
					t.Errorf("Too few clusters at max zoom: %d", len(clusters))
				}
			}
		})
	}
}

// Benchmark clustering performance
func BenchmarkSupercluster(b *testing.B) {
	bounds := cluster.KDBounds{
		MinX: -180,
		MinY: -85,
		MaxX: 180,
		MaxY: 85,
	}

	points := generateTestPoints(10000, bounds)
	options := cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 3,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		cluster := cluster.NewSupercluster(options)
		cluster.Load(points)

		// Test querying at different zoom levels
		// testBounds := cluster.KDBounds{
		// 	MinX: -122.5,
		// 	MinY: 37.7,
		// 	MaxX: -122.3,
		// 	MaxY: 37.9,
		// }

		// for zoom := 0; zoom <= 16; zoom += 4 {
		// 	cluster.GetClusters(testBounds, zoom)
		// }
	}
}

// Example usage function
func ExampleSupercluster() {
	// Generate test points
	bounds := cluster.KDBounds{
		MinX: -122.5,
		MinY: 37.7,
		MaxX: -122.3,
		MaxY: 37.9,
	}

	points := generateTestPoints(1000, bounds)

	// Initialize and use cluster
	options := cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 3,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	}

	cluster := cluster.NewSupercluster(options)
	cluster.Load(points)

	// Get clusters at zoom level 10
	clusters := cluster.GetClusters(bounds, 10)

	for _, c := range clusters {
		// Process clusters
		_ = c
	}
}
