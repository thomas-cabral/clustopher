package runner

import (
	"context"
	"fmt"
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
	ch           *cluster.CHClient
}

func NewClusterRunner(maxClusters int, ch *cluster.CHClient) *ClusterRunner {
	runner := &ClusterRunner{
		clusters:     make(map[string]*cluster.Supercluster),
		lastAccessed: make(map[string]time.Time),
		maxClusters:  maxClusters,
		ch:           ch,
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

		// Evict inactive clusters (skeleton released; CH remains canonical)
		for _, id := range toRemove {
			if sc, exists := r.clusters[id]; exists {
				sc.CleanupCluster()
				delete(r.clusters, id)
				delete(r.lastAccessed, id)
			}
		}

		r.clusterLock.Unlock()
	}
}

// loadClusterIfNeeded ensures the cluster skeleton is in memory, opening it
// from CH if it was evicted or never loaded in this process. Returns an error
// if the cluster_id doesn't exist in CH.
func (r *ClusterRunner) loadClusterIfNeeded(ctx context.Context, id string) error {
	r.clusterLock.Lock()
	defer r.clusterLock.Unlock()

	// Update access time if skeleton is already in memory.
	if _, exists := r.clusters[id]; exists {
		r.lastAccessed[id] = time.Now()
		return nil
	}

	// Evict LRU entry if we're at capacity.
	if len(r.clusters) >= r.maxClusters {
		var oldestID string
		var oldestTime time.Time
		first := true

		for cid, accessTime := range r.lastAccessed {
			if first || accessTime.Before(oldestTime) {
				oldestID = cid
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

	// Rebuild skeleton from CH.
	sc := cluster.NewSupercluster(cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    100,
		Extent:    512,
		NodeSize:  64,
	})
	sc.SetCHClient(r.ch)
	sc.SetClusterID(id)

	if err := sc.Open(ctx); err != nil {
		return fmt.Errorf("open cluster %s from CH: %w", id, err)
	}

	r.clusters[id] = sc
	r.lastAccessed[id] = time.Now()
	return nil
}

func (r *ClusterRunner) CreateCluster(ctx context.Context, req *pb.CreateClusterRequest) (*pb.CreateClusterResponse, error) {
	fmt.Printf("Creating new cluster with %d points\n", req.NumPoints)

	bounds := cluster.KDBounds{
		MinX: -180.0,
		MinY: -90.0,
		MaxX: 180.0,
		MaxY: 90.0,
	}

	points := cluster.GenerateTestPoints(int(req.NumPoints), bounds)

	options := cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    100,
		Extent:    512,
		NodeSize:  64,
		Log:       true,
	}

	id := uuid.New().String()[:8]

	sc := cluster.NewSupercluster(options)
	sc.SetCHClient(r.ch)
	sc.SetClusterID(id)

	if err := sc.Load(points); err != nil {
		return nil, fmt.Errorf("failed to load points: %v", err)
	}

	r.clusterLock.Lock()
	r.clusters[id] = sc
	r.lastAccessed[id] = time.Now()
	r.clusterLock.Unlock()

	return &pb.CreateClusterResponse{
		Cluster: &pb.ClusterInfo{
			Id:        id,
			NumPoints: req.NumPoints,
			Timestamp: time.Now().Format(time.RFC3339),
			FileSize:  0,
		},
	}, nil
}

func (r *ClusterRunner) ListClusters(ctx context.Context, req *pb.ListClustersRequest) (*pb.ListClustersResponse, error) {
	rows, err := r.ch.Conn().Query(ctx,
		"SELECT cluster_id, count() AS n FROM clustopher.points GROUP BY cluster_id ORDER BY cluster_id")
	if err != nil {
		return nil, fmt.Errorf("list clusters: %w", err)
	}
	defer rows.Close()

	var pbClusters []*pb.ClusterInfo
	for rows.Next() {
		var clusterID string
		var n uint64
		if err := rows.Scan(&clusterID, &n); err != nil {
			return nil, fmt.Errorf("scan row: %w", err)
		}
		pbClusters = append(pbClusters, &pb.ClusterInfo{
			Id:        clusterID,
			NumPoints: int32(n),
			Timestamp: "",
			FileSize:  0,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	return &pb.ListClustersResponse{Clusters: pbClusters}, nil
}

func (r *ClusterRunner) LoadCluster(ctx context.Context, req *pb.LoadClusterRequest) (*pb.LoadClusterResponse, error) {
	if err := r.loadClusterIfNeeded(ctx, req.ClusterId); err != nil {
		return nil, err
	}

	// Return basic info; NumPoints comes from CH.
	row := r.ch.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.points WHERE cluster_id = ?", req.ClusterId)
	var n uint64
	if err := row.Scan(&n); err != nil {
		return nil, fmt.Errorf("count points: %w", err)
	}

	return &pb.LoadClusterResponse{
		Cluster: &pb.ClusterInfo{
			Id:        req.ClusterId,
			NumPoints: int32(n),
			Timestamp: "",
			FileSize:  0,
		},
	}, nil
}

func (r *ClusterRunner) GetClusters(ctx context.Context, req *pb.GetClustersRequest) (*pb.GetClustersResponse, error) {
	if err := r.loadClusterIfNeeded(ctx, req.ClusterId); err != nil {
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
			Metrics:   c.Metrics,
		}
	}

	return &pb.GetClustersResponse{Features: features}, nil
}

func (r *ClusterRunner) GetMetadata(ctx context.Context, req *pb.GetMetadataRequest) (*pb.GetMetadataResponse, error) {
	if err := r.loadClusterIfNeeded(ctx, req.ClusterId); err != nil {
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
		case cluster.TimestampRange:
			metadataSummary[key] = &pb.MetadataValue{
				TimeRange: &pb.TimeRange{
					Earliest: v.Earliest.Format(time.RFC3339),
					Latest:   v.Latest.Format(time.RFC3339),
				},
			}
		case cluster.MetadataRange:
			metadataSummary[key] = &pb.MetadataValue{
				Range: &pb.Range{
					Min:     v.Min,
					Max:     v.Max,
					Average: v.Average,
				},
			}
		case map[string]float64:
			metadataSummary[key] = &pb.MetadataValue{
				Distribution: &pb.Distribution{
					Values: v,
				},
			}
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
