package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"web/clustopher/cluster"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func init() {
	// Register types for gob encoding/decoding
	gob.Register(time.Time{})
	gob.Register([]string{})
}

const CLUSTER_SAVE_DIR = "data/clusters"

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

func formatFileSize(size int64) string {
	const unit = 1024
	if size < unit {
		return fmt.Sprintf("%d B", size)
	}
	div, exp := int64(unit), 0
	for n := size / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(size)/float64(div), "KMGTPE"[exp])
}

func generateClusterFilename(size int) string {
	timestamp := time.Now().Format("20060102-150405")
	id := uuid.New().String()[:8] // Use first 8 chars of UUID for brevity
	return filepath.Join(CLUSTER_SAVE_DIR, fmt.Sprintf("cluster-%dp-%s-%s.zst", size, timestamp, id))
}

func findLatestClusterFile() (string, error) {
	files, err := os.ReadDir(CLUSTER_SAVE_DIR)
	if err != nil {
		return "", err
	}

	var latest string
	var latestTime time.Time

	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".zst" {
			info, err := file.Info()
			if err != nil {
				continue
			}
			if latest == "" || info.ModTime().After(latestTime) {
				latest = filepath.Join(CLUSTER_SAVE_DIR, file.Name())
				latestTime = info.ModTime()
			}
		}
	}

	if latest == "" {
		return "", fmt.Errorf("no cluster files found")
	}
	return latest, nil
}

func NewClusterServer(numPoints int) *ClusterServer {
	fmt.Printf("\n=== Starting NewClusterServer with %d points ===\n", numPoints)
	// Create clusters directory if it doesn't exist
	if err := os.MkdirAll(CLUSTER_SAVE_DIR, 0755); err != nil {
		fmt.Printf("Failed to create clusters directory: %v\n", err)
	}

	// If we're creating a new cluster, don't try to load existing
	if numPoints > 0 {
		fmt.Printf("Generating new cluster with %d points...\n", numPoints)
		// Use the existing point generation code
		bounds := cluster.KDBounds{
			MinX: -125.0,
			MinY: 25.0,
			MaxX: -67.0,
			MaxY: 49.0,
		}

		fmt.Printf("Generating points in the Continental US...\n")
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

		fmt.Printf("Creating new supercluster...\n")
		supercluster := cluster.NewSupercluster(options)
		supercluster.Load(points)

		// Save the cluster after loading with new filename format
		savePath := generateClusterFilename(numPoints)
		fmt.Printf("Saving new cluster to %s...\n", savePath)
		if err := supercluster.SaveCompressed(savePath); err != nil {
			fmt.Printf("ERROR: Failed to save cluster: %v\n", err)
		} else {
			fmt.Printf("Successfully saved new cluster\n")
		}

		fmt.Printf("=== Finished creating new cluster ===\n")
		return &ClusterServer{
			cluster: supercluster,
		}
	}

	// Try to load existing cluster
	if latestFile, err := findLatestClusterFile(); err == nil {
		loadStart := time.Now()
		if cluster, err := cluster.LoadCompressedSupercluster(latestFile); err == nil {
			loadDuration := time.Since(loadStart)
			fileInfo, _ := os.Stat(latestFile)
			fmt.Printf("Loaded existing cluster from %s (file size: %s) in %v\n",
				latestFile, formatFileSize(fileInfo.Size()), loadDuration)
			return &ClusterServer{
				cluster: cluster,
			}
		} else {
			fmt.Printf("Could not load existing cluster (%v)\n", err)
		}
	} else {
		fmt.Printf("No existing cluster found in %s\n", CLUSTER_SAVE_DIR)
	}

	return nil // Return nil if no cluster was loaded
}

// Add this new type for cluster info
type ClusterInfo struct {
	ID        string    `json:"id"`
	NumPoints int       `json:"numPoints"`
	Timestamp time.Time `json:"timestamp"`
	FileSize  int64     `json:"fileSize"`
}

// Add these new handler functions
func (s *ClusterServer) listClusters() ([]ClusterInfo, error) {
	absPath, err := filepath.Abs(CLUSTER_SAVE_DIR)
	fmt.Printf("\n=== Listing clusters ===\n")
	fmt.Printf("Looking for clusters in absolute path: %s\n", absPath)

	// List all files in directory
	if files, err := os.ReadDir(CLUSTER_SAVE_DIR); err == nil {
		fmt.Printf("All files in directory:\n")
		for _, f := range files {
			fmt.Printf("  - %s\n", f.Name())
		}
	}

	files, err := os.ReadDir(CLUSTER_SAVE_DIR)
	if err != nil {
		fmt.Printf("Error reading directory %s: %v\n", CLUSTER_SAVE_DIR, err)
		return nil, err
	}

	fmt.Printf("Found %d files in %s\n", len(files), CLUSTER_SAVE_DIR)
	clusters := make([]ClusterInfo, 0)
	for _, file := range files {
		fmt.Printf("Processing file: %s\n", file.Name())
		if !file.IsDir() && filepath.Ext(file.Name()) == ".zst" {
			info, err := file.Info()
			if err != nil {
				fmt.Printf("Error getting file info for %s: %v\n", file.Name(), err)
				continue
			}

			// Parse filename to get cluster info
			// Format: cluster-{numPoints}p-{timestamp}-{id}.zst
			name := strings.TrimSuffix(file.Name(), ".zst")
			parts := strings.Split(name, "-")
			fmt.Printf("Filename parts: %v\n", parts)
			if len(parts) != 5 {
				fmt.Printf("Invalid filename format for %s\n", name)
				continue
			}

			numPoints, err := strconv.Atoi(strings.TrimSuffix(parts[1], "p"))
			if err != nil {
				fmt.Printf("Error parsing numPoints from %s: %v\n", parts[1], err)
				continue
			}

			timestamp, err := time.Parse("20060102-150405", parts[2]+"-"+parts[3])
			if err != nil {
				fmt.Printf("Error parsing timestamp from %s: %v\n", parts[2], err)
				continue
			}

			fmt.Printf("Adding cluster: ID=%s, Points=%d, Time=%v, Size=%d\n",
				parts[4], numPoints, timestamp, info.Size())
			clusters = append(clusters, ClusterInfo{
				ID:        parts[4],
				NumPoints: numPoints,
				Timestamp: timestamp,
				FileSize:  info.Size(),
			})
		}
	}

	fmt.Printf("Returning %d clusters\n", len(clusters))
	// Sort by timestamp descending
	sort.Slice(clusters, func(i, j int) bool {
		return clusters[i].Timestamp.After(clusters[j].Timestamp)
	})

	return clusters, nil
}

func (s *ClusterServer) loadClusterById(id string) error {
	files, err := os.ReadDir(CLUSTER_SAVE_DIR)
	if err != nil {
		return err
	}

	// Find the file with matching ID
	var clusterFile string
	for _, file := range files {
		if strings.Contains(file.Name(), id) {
			clusterFile = filepath.Join(CLUSTER_SAVE_DIR, file.Name())
			break
		}
	}

	if clusterFile == "" {
		return fmt.Errorf("cluster with ID %s not found", id)
	}

	// Load the cluster
	loadedCluster, err := cluster.LoadCompressedSupercluster(clusterFile)
	if err != nil {
		return fmt.Errorf("failed to load cluster: %v", err)
	}

	s.cluster = loadedCluster
	return nil
}

func main() {
	// Ensure cluster directory exists with correct permissions
	absPath, _ := filepath.Abs(CLUSTER_SAVE_DIR)
	fmt.Printf("Ensuring cluster directory exists: %s\n", absPath)
	if err := os.MkdirAll(CLUSTER_SAVE_DIR, 0755); err != nil {
		fmt.Printf("Error creating cluster directory: %v\n", err)
	}
	if info, err := os.Stat(CLUSTER_SAVE_DIR); err == nil {
		fmt.Printf("Cluster directory exists with permissions: %v\n", info.Mode())
	} else {
		fmt.Printf("Error checking cluster directory: %v\n", err)
	}

	fmt.Println("Starting NewClusterServer...")
	start := time.Now()
	server := NewClusterServer(0) // Pass 0 to only try loading existing cluster
	duration := time.Since(start)
	fmt.Printf("NewClusterServer initialized in %v\n", duration)

	if server == nil {
		// No existing cluster found, create empty server
		server = &ClusterServer{}
	}

	r := gin.Default()

	// Enable CORS
	r.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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

	// List available clusters
	r.GET("/api/clusters/list", func(c *gin.Context) {
		clusters, err := server.listClusters()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, clusters)
	})

	// Create new cluster
	r.POST("/api/clusters", func(c *gin.Context) {
		var req struct {
			NumPoints int `json:"numPoints"`
		}
		fmt.Printf("\n=== Received request to create new cluster ===\n")
		if err := c.BindJSON(&req); err != nil {
			fmt.Printf("ERROR: Failed to parse request: %v\n", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
			return
		}

		fmt.Printf("Creating new cluster with %d points\n", req.NumPoints)
		oldServer := server // Keep reference to old server
		newServer := NewClusterServer(req.NumPoints)
		if newServer == nil {
			fmt.Printf("ERROR: Failed to create new cluster server\n")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create cluster"})
			return
		}

		// Update the global server variable
		server = newServer

		// Clean up old server if needed
		if oldServer != nil && oldServer.cluster != nil {
			fmt.Printf("Cleaned up old server\n")
		}

		// Verify the new server
		if server.cluster == nil {
			fmt.Printf("ERROR: New server has nil cluster\n")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to initialize cluster"})
			return
		}

		fmt.Printf("New cluster created successfully\n")
		c.JSON(http.StatusOK, gin.H{"message": "New cluster created"})
	})

	// Add this to your main() function where the other routes are
	r.POST("/api/clusters/load/:id", func(c *gin.Context) {
		id := c.Param("id")
		if err := server.loadClusterById(id); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"message": "Cluster loaded successfully"})
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

	// Create a channel to listen for interrupt signals
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)

	// Start server in a goroutine
	go func() {
		fmt.Println("Starting server on :8000...")
		if err := r.Run(":8000"); err != nil {
			fmt.Printf("Server error: %v\n", err)
		}
	}()

	// Wait for interrupt signal
	<-quit
	fmt.Println("\nShutting down server...")

	// Save cluster before shutting down with new filename format
	savePath := generateClusterFilename(len(server.cluster.Points))
	fmt.Printf("Saving cluster to %s...\n", savePath)
	saveStart := time.Now()
	if err := server.cluster.SaveCompressed(savePath); err != nil {
		fmt.Printf("Failed to save cluster on shutdown: %v\n", err)
	} else {
		saveDuration := time.Since(saveStart)
		if fileInfo, err := os.Stat(savePath); err == nil {
			fmt.Printf("Cluster saved successfully in %v (file size: %s)\n",
				saveDuration, formatFileSize(fileInfo.Size()))
		} else {
			fmt.Println("Cluster saved successfully")
		}
	}

	fmt.Println("Server stopped")
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
