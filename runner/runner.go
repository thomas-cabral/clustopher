package runner

import (
	"context"
	"fmt"
	"runtime"
	"runtime/debug"
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
// from CH if it was evicted or never loaded in this process. Returns the
// *Supercluster so callers never need to re-look-up the pointer (avoiding
// a TOCTOU race where an LRU eviction could occur between load and use).
func (r *ClusterRunner) loadClusterIfNeeded(ctx context.Context, id string) (*cluster.Supercluster, error) {
	// Fast path: already in memory.
	r.clusterLock.Lock()
	if sc, exists := r.clusters[id]; exists {
		r.lastAccessed[id] = time.Now()
		r.clusterLock.Unlock()
		return sc, nil
	}
	r.clusterLock.Unlock()

	// Slow path: build skeleton from CH outside the lock so we don't hold it
	// during a potentially long network call.
	sc := cluster.NewSupercluster(cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 2,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
	})
	sc.SetCHClient(r.ch)
	sc.SetClusterID(id)

	if err := sc.Open(ctx); err != nil {
		return nil, fmt.Errorf("open cluster %s from CH: %w", id, err)
	}

	// Re-acquire write lock to insert. Double-check in case a concurrent
	// request already loaded the same cluster while we were in Open().
	r.clusterLock.Lock()
	defer r.clusterLock.Unlock()

	if existing, exists := r.clusters[id]; exists {
		// Another goroutine won the race; discard our copy and use theirs.
		sc.CleanupCluster()
		r.lastAccessed[id] = time.Now()
		return existing, nil
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

	r.clusters[id] = sc
	r.lastAccessed[id] = time.Now()
	return sc, nil
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
		// Must match the /40 scaling baked into migrations/002_rollup_template.sql
		// — queryRollup uses Options.Radius to derive the tile range, and any
		// other value will pick wrong tiles from the MV.
		Radius:   40,
		Extent:   512,
		NodeSize: 64,
		Log:      true,
	}

	id := uuid.New().String()[:8]

	sc := cluster.NewSupercluster(options)
	sc.SetCHClient(r.ch)
	sc.SetClusterID(id)

	if err := sc.Load(points); err != nil {
		return nil, fmt.Errorf("failed to load points: %v", err)
	}

	// Release the 15M-point source slice + force the runtime to hand pages
	// back to the OS. Without this RSS stays pinned at the Load high-water
	// mark for the lifetime of the runner — Skeleton itself is tiny.
	points = nil
	runtime.GC()
	debug.FreeOSMemory()

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
	if _, err := r.loadClusterIfNeeded(ctx, req.ClusterId); err != nil {
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
	sc, err := r.loadClusterIfNeeded(ctx, req.ClusterId)
	if err != nil {
		return nil, err
	}

	bounds := cluster.KDBounds{
		MinX: req.Bounds.MinX,
		MinY: req.Bounds.MinY,
		MaxX: req.Bounds.MaxX,
		MaxY: req.Bounds.MaxY,
	}

	clusters, err := sc.GetClustersCH(ctx, bounds, int(req.Zoom))
	if err != nil {
		return nil, fmt.Errorf("get clusters: %w", err)
	}

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
	sc, err := r.loadClusterIfNeeded(ctx, req.ClusterId)
	if err != nil {
		return nil, err
	}

	bounds := cluster.KDBounds{
		MinX: req.Bounds.MinX,
		MinY: req.Bounds.MinY,
		MaxX: req.Bounds.MaxX,
		MaxY: req.Bounds.MaxY,
	}

	clusters, err := sc.GetClustersCH(ctx, bounds, int(req.Zoom))
	if err != nil {
		return nil, fmt.Errorf("get clusters: %w", err)
	}
	summary := cluster.CalculateMetadataSummary(clusters)

	// summary.TotalPoints from CalculateMetadataSummary only counts what's
	// visible in the current viewport. The UI sidebar wants the cluster's
	// true total, so override with a direct SELECT count() against the
	// canonical points table.
	var totalPoints uint64
	if err := r.ch.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.points WHERE cluster_id = ?",
		req.ClusterId).Scan(&totalPoints); err != nil {
		return nil, fmt.Errorf("count cluster points: %w", err)
	}

	// Backfill metadata distribution from CH. The rollup-MV path used at low
	// zoom doesn't carry per-point metadata, so cluster nodes returned by
	// GetClustersCH have empty Metadata maps when zoom < ZSplit. Probe the
	// canonical points table directly within the viewport so the UI sidebar
	// always shows metadata totals regardless of zoom.
	metadataDist, err := r.queryMetadataDistribution(ctx, req.ClusterId, bounds)
	if err != nil {
		// Non-fatal — fall back to whatever CalculateMetadataSummary produced.
		fmt.Printf("metadata distribution probe failed: %v\n", err)
	}

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

	// Overlay CH-side metadata distribution on top of any in-memory summary.
	// The CH probe is authoritative because it scans every point in the
	// viewport (no rollup MV truncation, no skeleton path attribution gaps).
	for key, valueCounts := range metadataDist {
		metadataSummary[key] = &pb.MetadataValue{
			Distribution: &pb.Distribution{Values: valueCounts},
		}
	}

	return &pb.GetMetadataResponse{
		TotalPoints:     int32(totalPoints),
		NumClusters:     int32(summary.NumClusters),
		NumSinglePoints: int32(summary.NumSinglePoints),
		MetricsSummary:  metricsSummary,
		MetadataSummary: metadataSummary,
	}, nil
}

// queryMetadataDistribution returns a per-key map of metadata-value counts
// across all points inside the viewport for the given cluster. It unrolls the
// Map(String, String) metadata column via arrayJoin so categorical fields
// (e.g. {"type":"test"}) end up as {"type": {"test": N}} in the response.
func (r *ClusterRunner) queryMetadataDistribution(ctx context.Context, clusterID string, bounds cluster.KDBounds) (map[string]map[string]float64, error) {
	rows, err := r.ch.Conn().Query(ctx, `
        SELECT kv.1 AS k, kv.2 AS v, count() AS cnt
        FROM (
            SELECT arrayJoin(
                arrayMap((mk, mv) -> (mk, mv), mapKeys(metadata), mapValues(metadata))
            ) AS kv
            FROM clustopher.points
            WHERE cluster_id = ?
              AND x BETWEEN ? AND ?
              AND y BETWEEN ? AND ?
        )
        GROUP BY k, v
        ORDER BY k, cnt DESC
    `, clusterID, bounds.MinX, bounds.MaxX, bounds.MinY, bounds.MaxY)
	if err != nil {
		return nil, fmt.Errorf("metadata dist query: %w", err)
	}
	defer rows.Close()

	out := make(map[string]map[string]float64)
	var (
		k, v string
		cnt  uint64
	)
	for rows.Next() {
		if err := rows.Scan(&k, &v, &cnt); err != nil {
			return nil, fmt.Errorf("scan metadata dist row: %w", err)
		}
		inner, ok := out[k]
		if !ok {
			inner = make(map[string]float64)
			out[k] = inner
		}
		inner[v] = float64(cnt)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("metadata dist rows: %w", err)
	}
	return out, nil
}
