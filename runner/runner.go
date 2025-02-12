package runner

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"web/clustopher/cluster"
	pb "web/clustopher/proto"

	"github.com/google/uuid"
)

type ClusterRunner struct {
	pb.UnimplementedClusterServiceServer
	clusters     map[string]*cluster.Supercluster
	clusterLock  sync.RWMutex
	lastAccessed map[string]time.Time
	maxClusters  int
}

// ... (previous functions remain the same until CreateCluster) ...

func (r *ClusterRunner) CreateCluster(ctx context.Context, req *pb.CreateClusterRequest) (*pb.CreateClusterResponse, error) {
	fmt.Printf("Creating new cluster with %d points\n", req.NumPoints)

	// Generate test points
	bounds := cluster.KDBounds{
		MinX: -180.0,
		MinY: -90.0,
		MaxX: 180.0,
		MaxY: 90.0,
	}

	points := cluster.GenerateTestPoints(int(req.NumPoints), bounds)

	// Create supercluster with default options
	options := cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    100,
		Extent:    512,
		NodeSize:  64,
		Log:       true,
	}

	supercluster := cluster.NewSupercluster(options)
	supercluster.Load(points)

	// Generate filename with timestamp and UUID
	savePath := generateClusterFilename(int(req.NumPoints))
	fmt.Printf("Saving new cluster to %s...\n", savePath)

	// Save the cluster
	if err := supercluster.SaveCompressed(savePath); err != nil {
		return nil, fmt.Errorf("failed to save cluster: %v", err)
	}

	// Extract ID from filename
	// Format: cluster-{numPoints}p-{timestamp}-{id}.zst
	parts := strings.Split(filepath.Base(savePath), "-")
	if len(parts) != 5 {
		return nil, fmt.Errorf("invalid filename format")
	}
	id := strings.TrimSuffix(parts[4], ".zst")

	// Add to loaded clusters
	r.clusterLock.Lock()
	r.clusters[id] = supercluster
	r.lastAccessed[id] = time.Now()
	r.clusterLock.Unlock()

	// Get file info for response
	fileInfo, err := os.Stat(savePath)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %v", err)
	}

	return &pb.CreateClusterResponse{
		Cluster: &pb.ClusterInfo{
			Id:        id,
			NumPoints: req.NumPoints,
			Timestamp: time.Now().Format(time.RFC3339),
			FileSize:  fileInfo.Size(),
		},
	}, nil
}

func generateClusterFilename(numPoints int) string {
	timestamp := time.Now().Format("20060102-150405")
	id := uuid.New().String()[:8] // Use first 8 chars of UUID for brevity
	return filepath.Join("data/clusters", fmt.Sprintf("cluster-%dp-%s-%s.zst", numPoints, timestamp, id))
}

func NewClusterRunner(maxClusters int) *ClusterRunner {
	runner := &ClusterRunner{
		clusters:     make(map[string]*cluster.Supercluster),
		lastAccessed: make(map[string]time.Time),
		maxClusters:  maxClusters,
	}

	// Start cleanup goroutine
	go runner.cleanupInactiveClusters()

	return runner
}

func (r *ClusterRunner) cleanupInactiveClusters() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		r.clusterLock.Lock()
		now := time.Now()

		// Find clusters inactive for more than 30 minutes
		var toRemove []string
		for id, lastAccess := range r.lastAccessed {
			if now.Sub(lastAccess) > 30*time.Minute {
				toRemove = append(toRemove, id)
			}
		}

		// Remove inactive clusters
		for _, id := range toRemove {
			if cluster, exists := r.clusters[id]; exists {
				cluster.CleanupCluster()
				delete(r.clusters, id)
				delete(r.lastAccessed, id)
			}
		}

		r.clusterLock.Unlock()
	}
}

func findClusterFile(id string) (string, error) {
	files, err := os.ReadDir("data/clusters")
	if err != nil {
		return "", fmt.Errorf("failed to read clusters directory: %v", err)
	}

	for _, file := range files {
		if strings.Contains(file.Name(), id) && strings.HasSuffix(file.Name(), ".zst") {
			return filepath.Join("data/clusters", file.Name()), nil
		}
	}

	return "", fmt.Errorf("no cluster file found with id %s", id)
}

func (r *ClusterRunner) loadClusterIfNeeded(id string) error {
	r.clusterLock.Lock()
	defer r.clusterLock.Unlock()

	// Update access time if cluster is already loaded
	if _, exists := r.clusters[id]; exists {
		r.lastAccessed[id] = time.Now()
		return nil
	}

	// Check if we need to remove least recently used cluster
	if len(r.clusters) >= r.maxClusters {
		var oldestID string
		var oldestTime time.Time
		first := true

		for id, accessTime := range r.lastAccessed {
			if first || accessTime.Before(oldestTime) {
				oldestID = id
				oldestTime = accessTime
				first = false
			}
		}

		if oldestID != "" {
			r.clusters[oldestID].CleanupCluster()
			delete(r.clusters, oldestID)
			delete(r.lastAccessed, oldestID)
		}
	}

	// Find the cluster file
	clusterFile, err := findClusterFile(id)
	if err != nil {
		return fmt.Errorf("failed to find cluster file: %v", err)
	}

	// Load the requested cluster
	supercluster, err := cluster.LoadCompressedSupercluster(clusterFile)
	if err != nil {
		return fmt.Errorf("failed to load cluster %s: %v", id, err)
	}

	r.clusters[id] = supercluster
	r.lastAccessed[id] = time.Now()
	return nil
}

// Rest of the gRPC service implementations remain the same
func (r *ClusterRunner) ListClusters(ctx context.Context, req *pb.ListClustersRequest) (*pb.ListClustersResponse, error) {
	clusters, err := cluster.ListSavedClusters()
	if err != nil {
		return nil, err
	}

	pbClusters := make([]*pb.ClusterInfo, len(clusters))
	for i, c := range clusters {
		pbClusters[i] = &pb.ClusterInfo{
			Id:        c.ID,
			NumPoints: int32(c.NumPoints),
			Timestamp: c.Timestamp.Format(time.RFC3339),
			FileSize:  c.FileSize,
		}
	}

	return &pb.ListClustersResponse{Clusters: pbClusters}, nil
}

func (r *ClusterRunner) LoadCluster(ctx context.Context, req *pb.LoadClusterRequest) (*pb.LoadClusterResponse, error) {
	if err := r.loadClusterIfNeeded(req.ClusterId); err != nil {
		return nil, err
	}

	info, err := cluster.GetClusterInfo(req.ClusterId)
	if err != nil {
		return nil, err
	}

	return &pb.LoadClusterResponse{
		Cluster: &pb.ClusterInfo{
			Id:        info.ID,
			NumPoints: int32(info.NumPoints),
			Timestamp: info.Timestamp.Format(time.RFC3339),
			FileSize:  info.FileSize,
		},
	}, nil
}

func (r *ClusterRunner) GetClusters(ctx context.Context, req *pb.GetClustersRequest) (*pb.GetClustersResponse, error) {
	if err := r.loadClusterIfNeeded(req.ClusterId); err != nil {
		return nil, err
	}

	r.clusterLock.RLock()
	sc := r.clusters[req.ClusterId]
	r.clusterLock.RUnlock()

	bounds := cluster.KDBounds{
		MinX: req.Bounds.MinX,
		MinY: req.Bounds.MinY,
		MaxX: req.Bounds.MaxX,
		MaxY: req.Bounds.MaxY,
	}

	clusters := sc.GetClusters(bounds, int(req.Zoom))

	features := make([]*pb.ClusterFeature, len(clusters))
	for i, c := range clusters {
		features[i] = &pb.ClusterFeature{
			X:         c.X,
			Y:         c.Y,
			Count:     c.Count,
			Id:        c.ID,
			IsCluster: c.Count > 1,
			Metrics:   c.Metrics.Values,
		}
	}

	return &pb.GetClustersResponse{Features: features}, nil
}

func (r *ClusterRunner) GetMetadata(ctx context.Context, req *pb.GetMetadataRequest) (*pb.GetMetadataResponse, error) {
	if err := r.loadClusterIfNeeded(req.ClusterId); err != nil {
		return nil, err
	}

	r.clusterLock.RLock()
	sc := r.clusters[req.ClusterId]
	r.clusterLock.RUnlock()

	bounds := cluster.KDBounds{
		MinX: req.Bounds.MinX,
		MinY: req.Bounds.MinY,
		MaxX: req.Bounds.MaxX,
		MaxY: req.Bounds.MaxY,
	}

	clusters := sc.GetClusters(bounds, int(req.Zoom))
	summary := cluster.CalculateMetadataSummary(clusters)

	// Convert metricsSummary
	metricsSummary := make(map[string]*pb.MetricStats)
	for metric, stats := range summary.MetricsSummary {
		metricsSummary[metric] = &pb.MetricStats{
			Min:     float64(stats.Min),
			Max:     float64(stats.Max),
			Average: float64(stats.Average),
		}
	}

	// Convert metadataSummary
	metadataSummary := make(map[string]*pb.MetadataValue)
	for key, value := range summary.MetadataSummary {
		switch v := value.(type) {
		case map[string]float64: // For category and region distributions
			// Convert percentages to match frontend expectations
			metadataSummary[key] = &pb.MetadataValue{
				Distribution: &pb.Distribution{
					Values: v,
				},
			}
		case struct {
			Earliest time.Time
			Latest   time.Time
		}:
			metadataSummary[key] = &pb.MetadataValue{
				TimeRange: &pb.TimeRange{
					Earliest: v.Earliest.Format(time.RFC3339),
					Latest:   v.Latest.Format(time.RFC3339),
				},
			}
		case struct {
			Min     float64
			Max     float64
			Average float64
		}:
			metadataSummary[key] = &pb.MetadataValue{
				Range: &pb.Range{
					Min:     v.Min,
					Max:     v.Max,
					Average: v.Average,
				},
			}
		case string:
			metadataSummary[key] = &pb.MetadataValue{
				SingleValue: v,
			}
		default:
			// Skip unknown types
			continue
		}
	}

	return &pb.GetMetadataResponse{
		TotalPoints:     int32(summary.TotalPoints),
		NumClusters:     int32(summary.NumClusters),
		NumSinglePoints: int32(summary.NumSinglePoints),
		MetricsSummary:  metricsSummary,
		MetadataSummary: metadataSummary,
	}, nil
}
