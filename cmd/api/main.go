package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"web/clustopher/proto"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
)

type Server struct {
	clusterClient    proto.ClusterServiceClient
	defaultClusterID string // Store the ID of the most recently created/loaded cluster
}

func NewServer(clusterClient proto.ClusterServiceClient) *Server {
	return &Server{
		clusterClient: clusterClient,
	}
}

func getBoundsFromQuery(c *gin.Context) (*proto.Bounds, error) {
	north, err := strconv.ParseFloat(c.Query("north"), 64)
	if err != nil {
		return nil, fmt.Errorf("invalid north parameter")
	}

	south, err := strconv.ParseFloat(c.Query("south"), 64)
	if err != nil {
		return nil, fmt.Errorf("invalid south parameter")
	}

	east, err := strconv.ParseFloat(c.Query("east"), 64)
	if err != nil {
		return nil, fmt.Errorf("invalid east parameter")
	}

	west, err := strconv.ParseFloat(c.Query("west"), 64)
	if err != nil {
		return nil, fmt.Errorf("invalid west parameter")
	}

	return &proto.Bounds{
		MinX: float32(west),
		MinY: float32(south),
		MaxX: float32(east),
		MaxY: float32(north),
	}, nil
}

func main() {
	// Connect to cluster runner
	conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
	if err != nil {
		fmt.Printf("Failed to connect to cluster runner: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	clusterClient := proto.NewClusterServiceClient(conn)
	server := NewServer(clusterClient)

	// Get list of clusters and set default if any exist
	if resp, err := clusterClient.ListClusters(context.Background(), &proto.ListClustersRequest{}); err == nil && len(resp.Clusters) > 0 {
		server.defaultClusterID = resp.Clusters[0].Id // Use most recent cluster as default
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

	type ClusterInfo struct {
		Id string `json:"id"`
		NumPoints int `json:"numPoints"`
		Timestamp string `json:"timestamp"`
		FileSize int64 `json:"fileSize"`
	}

	// List available clusters
	r.GET("/api/clusters/list", func(c *gin.Context) {
		resp, err := server.clusterClient.ListClusters(context.Background(), &proto.ListClustersRequest{})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		clusters := make([]ClusterInfo, len(resp.Clusters))
		for i, cluster := range resp.Clusters {
			clusters[i] = ClusterInfo{
				Id:        cluster.Id,
				NumPoints: int(cluster.NumPoints),
				Timestamp: cluster.Timestamp,
				FileSize:  cluster.FileSize,
			}
		}

		c.JSON(http.StatusOK, clusters)
	})

	// Handle both routes with and without cluster ID
	r.GET("/api/clusters", func(c *gin.Context) {
		if server.defaultClusterID == "" {
			c.JSON(http.StatusNotFound, gin.H{"error": "No clusters available"})
			return
		}
		handleGetClusters(c, server, server.defaultClusterID)
	})

	r.GET("/api/clusters/:id", func(c *gin.Context) {
		handleGetClusters(c, server, c.Param("id"))
	})

	// Handle both metadata routes
	r.GET("/api/clusters/metadata", func(c *gin.Context) {
		if server.defaultClusterID == "" {
			c.JSON(http.StatusNotFound, gin.H{"error": "No clusters available"})
			return
		}
		handleGetMetadata(c, server, server.defaultClusterID)
	})

	r.GET("/api/clusters/:id/metadata", func(c *gin.Context) {
		handleGetMetadata(c, server, c.Param("id"))
	})

	// Create new cluster
	r.POST("/api/clusters", func(c *gin.Context) {
		var req struct {
			NumPoints int `json:"numPoints"`
		}
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
			return
		}

		resp, err := server.clusterClient.CreateCluster(context.Background(), &proto.CreateClusterRequest{
			NumPoints: int32(req.NumPoints),
		})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Update default cluster ID
		server.defaultClusterID = resp.Cluster.Id

		c.JSON(http.StatusOK, resp.Cluster)
	})

	// Load cluster
	r.POST("/api/clusters/:id/load", func(c *gin.Context) {
		id := c.Param("id")
		resp, err := server.clusterClient.LoadCluster(context.Background(), &proto.LoadClusterRequest{
			ClusterId: id,
		})
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Update default cluster ID
		server.defaultClusterID = id

		c.JSON(http.StatusOK, gin.H{
			"message":     "Cluster loaded successfully",
			"clusterInfo": resp.Cluster,
		})
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
}

func handleGetClusters(c *gin.Context, server *Server, clusterID string) {
	zoom, err := strconv.Atoi(c.Query("zoom"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid zoom parameter"})
		return
	}

	bounds, err := getBoundsFromQuery(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	resp, err := server.clusterClient.GetClusters(context.Background(), &proto.GetClustersRequest{
		ClusterId: clusterID,
		Zoom:      int32(zoom),
		Bounds:    bounds,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Convert to GeoJSON
	features := make([]map[string]interface{}, len(resp.Features))
	for i, f := range resp.Features {
		properties := map[string]interface{}{
			"cluster":     f.IsCluster,
			"point_count": f.Count,
			"id":          f.Id,
		}
		if f.Metrics != nil {
			properties["metrics"] = f.Metrics
		}

		features[i] = map[string]interface{}{
			"type": "Feature",
			"geometry": map[string]interface{}{
				"type":        "Point",
				"coordinates": []float64{float64(f.X), float64(f.Y)},
			},
			"properties": properties,
		}
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"type":     "FeatureCollection",
		"features": features,
	})
}

func handleGetMetadata(c *gin.Context, server *Server, clusterID string) {
	zoom, err := strconv.Atoi(c.Query("zoom"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid zoom parameter"})
		return
	}

	bounds, err := getBoundsFromQuery(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	resp, err := server.clusterClient.GetMetadata(context.Background(), &proto.GetMetadataRequest{
		ClusterId: clusterID,
		Zoom:      int32(zoom),
		Bounds:    bounds,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Convert the response to match frontend expectations
	metricsSummary := make(map[string]map[string]float64)
	for metric, stats := range resp.MetricsSummary {
		metricsSummary[metric] = map[string]float64{
			"min":     stats.Min,
			"max":     stats.Max,
			"average": stats.Average,
		}
	}

	metadataSummary := make(map[string]interface{})
	for key, value := range resp.MetadataSummary {
		if value.Distribution != nil {
			metadataSummary[key] = map[string]interface{}{
				"distribution": map[string]interface{}{
					"values": value.Distribution.Values,
				},
			}
		} else if value.TimeRange != nil {
			metadataSummary[key] = map[string]interface{}{
				"earliest": value.TimeRange.Earliest,
				"latest":   value.TimeRange.Latest,
			}
		} else if value.Range != nil {
			metadataSummary[key] = map[string]interface{}{
				"min":     value.Range.Min,
				"max":     value.Range.Max,
				"average": value.Range.Average,
			}
		} else if value.SingleValue != "" {
			metadataSummary[key] = value.SingleValue
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"totalPoints":     resp.TotalPoints,
		"numClusters":     resp.NumClusters,
		"numSinglePoints": resp.NumSinglePoints,
		"metricsSummary":  metricsSummary,
		"metadataSummary": metadataSummary,
	})
}
