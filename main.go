package main

import (
	"encoding/gob"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
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

type ClusterServer struct {
	cluster *cluster.Supercluster
}

func (s *ClusterServer) Cleanup() {
	if s.cluster != nil {
		s.cluster.CleanupCluster()
		s.cluster = nil
	}
	runtime.GC()
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

func NewClusterServer(numPoints int) *ClusterServer {
	fmt.Printf("\n=== Starting NewClusterServer with %d points ===\n", numPoints)

	// Create clusters directory if it doesn't exist
	if err := os.MkdirAll(CLUSTER_SAVE_DIR, 0755); err != nil {
		fmt.Printf("Failed to create clusters directory: %v\n", err)
	}

	// Only generate new cluster if numPoints > 0
	if numPoints > 0 {
		fmt.Printf("Generating new cluster with %d points...\n", numPoints)
		bounds := cluster.KDBounds{
			MinX: -125.0,
			MinY: 25.0,
			MaxX: -67.0,
			MaxY: 49.0,
		}

		fmt.Printf("Generating points in the Continental US...\n")
		points := cluster.GenerateTestPoints(numPoints, bounds)

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

		loadStart := time.Now() // Start timer for Load
		supercluster.Load(points)
		loadDuration := time.Since(loadStart) // End timer for Load
		fmt.Printf("Points loaded in %v\n", loadDuration)

		// Save the cluster after loading with new filename format
		savePath := generateClusterFilename(numPoints)
		fmt.Printf("Saving new cluster to %s...\n", savePath)
		saveStart := time.Now() // Start timer for Save
		if err := supercluster.SaveCompressed(savePath); err != nil {
			fmt.Printf("ERROR: Failed to save cluster: %v\n", err)
		} else {
			saveDuration := time.Since(saveStart) // End timer for Save
			if fileInfo, err := os.Stat(savePath); err == nil {
				fmt.Printf("Successfully saved new cluster in %v (file size: %s)\n",
					saveDuration, formatFileSize(fileInfo.Size()))
			} else {
				fmt.Printf("Successfully saved new cluster in %v\n", saveDuration)
			}
		}

		fmt.Printf("=== Finished creating new cluster ===\n")
		return &ClusterServer{
			cluster: supercluster,
		}
	}

	// Return empty server if numPoints is 0
	return &ClusterServer{}
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
	if err != nil {
		return nil, err
	}
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

func (s *ClusterServer) loadClusterById(id string) (*ClusterInfo, error) {
	files, err := os.ReadDir(CLUSTER_SAVE_DIR)
	if err != nil {
		return nil, err
	}

	var clusterFile string
	var loadedClusterInfo *ClusterInfo // To store cluster info

	for _, file := range files {
		if strings.Contains(file.Name(), id) {
			clusterFile = filepath.Join(CLUSTER_SAVE_DIR, file.Name())
			// Parse filename to get cluster info to return
			name := strings.TrimSuffix(file.Name(), ".zst")
			parts := strings.Split(name, "-")
			if len(parts) == 5 {
				numPoints, _ := strconv.Atoi(strings.TrimSuffix(parts[1], "p"))
				timestamp, _ := time.Parse("20060102-150405", parts[2]+"-"+parts[3])
				fileInfo, _ := os.Stat(clusterFile) // Get file size
				loadedClusterInfo = &ClusterInfo{
					ID:        parts[4],
					NumPoints: numPoints,
					Timestamp: timestamp,
					FileSize:  fileInfo.Size(),
				}
			}
			break
		}
	}

	if clusterFile == "" {
		return nil, fmt.Errorf("cluster with ID %s not found", id)
	}

	loadStart := time.Now() // Start timer for LoadCompressedSupercluster
	loadedCluster, err := cluster.LoadCompressedSupercluster(clusterFile)
	loadDuration := time.Since(loadStart) // End timer for LoadCompressedSupercluster
	fmt.Printf("Cluster loaded from file in %v\n", loadDuration)

	if err != nil {
		return nil, fmt.Errorf("failed to load cluster: %v", err)
	}

	if loadedCluster == nil {
		return nil, fmt.Errorf("loaded cluster is nil")
	}

	s.cluster = loadedCluster
	return loadedClusterInfo, nil // Return ClusterInfo
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

	// Initialize with empty server instead of loading last cluster
	server := &ClusterServer{}
	fmt.Println("Started with empty cluster server - waiting for cluster to be loaded...")

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
			fmt.Printf("Processing cluster/point %d: Count=%d, Has Metrics=%v\n",
				i, cluster.Count, cluster.Metrics.Values != nil)

			properties := map[string]interface{}{
				"cluster":     cluster.Count > 1,
				"point_count": cluster.Count,
				"id":          cluster.ID,
			}

			// Add metrics to properties if they exist
			if cluster.Metrics.Values != nil && len(cluster.Metrics.Values) > 0 {
				fmt.Printf("Adding metrics for cluster/point %d: %+v\n", i, cluster.Metrics.Values)
				properties["metrics"] = cluster.Metrics.Values
			} else {
				fmt.Printf("No metrics found for cluster/point %d\n", i)
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

		// Debug: Print the first feature's properties
		if len(features) > 0 {
			fmt.Printf("First feature properties: %+v\n", features[0]["properties"])
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
		fmt.Printf("Received request to load cluster with ID: %s\n", id)

		// Clean up existing cluster before loading new one
		if server.cluster != nil {
			server.Cleanup()
		}

		clusterInfo, err := server.loadClusterById(id) // Capture ClusterInfo
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"message": "Cluster loaded successfully", "clusterInfo": clusterInfo}) // Return ClusterInfo in response
	})

	// Add this new handler after your existing /api/clusters endpoint
	r.GET("/api/clusters/metadata", func(c *gin.Context) {
		// Parse query parameters
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

		// Get clusters for the current view
		clusters := server.cluster.GetClusters(bounds, zoom)

		// Calculate metadata summaries
		summary := cluster.CalculateMetadataSummary(clusters)

		c.JSON(http.StatusOK, summary)
	})

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

