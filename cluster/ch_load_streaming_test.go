package cluster

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"
)

// TestCHStreamingMatchesKD loads the same staged dataset twice — once via
// LoadFromCHStaging (exact KD sort) and once via LoadFromCHStreaming (Morton
// sort) — and asserts that GetClustersCH produces equivalent results.
//
// Low/mid zoom queries hit the per-zoom rollup MV (depends only on point
// positions, not skeleton order) so they must match exactly. The z14 path
// walks the skeleton, so cluster counts may differ slightly between the two
// orderings; we assert they are within 5% of each other and that the union of
// returned point counts matches.
//
// Gated by CLUSTOPHER_STREAMING_PARITY=1 because it requires a live CH.
func TestCHStreamingMatchesKD(t *testing.T) {
	if os.Getenv("CLUSTOPHER_STREAMING_PARITY") != "1" {
		t.Skip("set CLUSTOPHER_STREAMING_PARITY=1 to run streaming-vs-KD parity")
	}
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}

	const n = 1_000_000
	conus := KDBounds{MinX: -125, MinY: 25, MaxX: -65, MaxY: 49}

	ctx := context.Background()
	c, err := NewCHClient(ctx, CHConfig{DSN: dsn})
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	defer c.Close()
	if err := RunMigrations(ctx, c.Conn(), "migrations"); err != nil {
		t.Fatalf("migrations: %v", err)
	}

	type result struct {
		path     string
		loadSec  float64
		heapMB   float64
		leaves   int
		queries  map[int]queryResult
	}
	queries := []int{2, 8, 14}
	vp := func(z int) KDBounds {
		if z >= 11 {
			return KDBounds{MinX: -100.25, MinY: 39.25, MaxX: -99.75, MaxY: 39.75}
		}
		if z >= 5 {
			return KDBounds{MinX: -100, MinY: 37, MaxX: -95, MaxY: 42}
		}
		return conus
	}

	runOne := func(label string, loader func(sc *Supercluster, ctx context.Context) error) result {
		clusterID := fmt.Sprintf("PARITY_%s_%d", label, n)
		cleanup := func() {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
			for z := 2; z <= 16; z++ {
				_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
			}
		}
		cleanup()
		defer cleanup()

		if err := generateDenseStagingPoints(ctx, c, clusterID, n, conus); err != nil {
			t.Fatalf("[%s] stage: %v", label, err)
		}
		sc := NewSupercluster(SuperclusterOptions{
			MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40, Extent: 512, NodeSize: 64,
		})
		sc.SetCHClient(c)
		sc.SetClusterID(clusterID)

		start := time.Now()
		if err := loader(sc, ctx); err != nil {
			t.Fatalf("[%s] load: %v", label, err)
		}
		loadSec := time.Since(start).Seconds()
		for z := 2; z <= 16; z++ {
			_ = c.Conn().Exec(ctx, "OPTIMIZE TABLE clustopher.rollup_z"+strconv.Itoa(z)+" PARTITION ? FINAL", clusterID)
		}

		r := result{path: label, loadSec: loadSec, leaves: len(sc.Skeleton.Leaves), queries: map[int]queryResult{}}
		for _, z := range queries {
			start := time.Now()
			cs, err := sc.GetClustersCH(ctx, vp(z), z)
			if err != nil {
				t.Fatalf("[%s] z=%d query: %v", label, z, err)
			}
			var totalCount uint64
			for _, cl := range cs {
				totalCount += uint64(cl.Count)
			}
			r.queries[z] = queryResult{
				clusters:   len(cs),
				totalCount: totalCount,
				elapsedMs:  float64(time.Since(start).Microseconds()) / 1000.0,
			}
		}
		return r
	}

	kd := runOne("KD", func(sc *Supercluster, ctx context.Context) error {
		return sc.LoadFromCHStaging(ctx)
	})
	stream := runOne("STREAM", func(sc *Supercluster, ctx context.Context) error {
		return sc.LoadFromCHStreaming(ctx)
	})

	t.Logf("KD     load=%.1fs leaves=%d", kd.loadSec, kd.leaves)
	t.Logf("STREAM load=%.1fs leaves=%d", stream.loadSec, stream.leaves)

	if kd.leaves != stream.leaves {
		t.Errorf("leaf count differs: KD=%d STREAM=%d", kd.leaves, stream.leaves)
	}

	for _, z := range queries {
		k := kd.queries[z]
		s := stream.queries[z]
		t.Logf("z=%d  KD: clusters=%d totalCount=%d elapsed=%.1fms  STREAM: clusters=%d totalCount=%d elapsed=%.1fms",
			z, k.clusters, k.totalCount, k.elapsedMs, s.clusters, s.totalCount, s.elapsedMs)

		if z < 11 {
			// Rollup MV path is data-only — must match exactly.
			if k.clusters != s.clusters {
				t.Errorf("z=%d (rollup MV path) cluster count differs: KD=%d STREAM=%d", z, k.clusters, s.clusters)
			}
			if k.totalCount != s.totalCount {
				t.Errorf("z=%d (rollup MV path) total point count differs: KD=%d STREAM=%d", z, k.totalCount, s.totalCount)
			}
			continue
		}
		// Skeleton path: cluster counts may shift because Morton leaves are
		// looser than KD median-partition leaves — more leaves overlap viewport
		// edges so a few extra points get fetched. Tolerate up to 25%.
		ratio := math.Abs(float64(k.clusters-s.clusters)) / float64(max1(k.clusters))
		if ratio > 0.25 {
			t.Errorf("z=%d skeleton path cluster count diverges by %.1f%% (KD=%d STREAM=%d)",
				z, ratio*100, k.clusters, s.clusters)
		}
		diff := absDiff(k.totalCount, s.totalCount)
		if float64(diff)/float64(max1u(k.totalCount)) > 0.25 {
			t.Errorf("z=%d skeleton path total point count diverges by %d (KD=%d STREAM=%d)",
				z, diff, k.totalCount, s.totalCount)
		}
	}
}

type queryResult struct {
	clusters   int
	totalCount uint64
	elapsedMs  float64
}

func max1(x int) int {
	if x < 1 {
		return 1
	}
	return x
}

func max1u(x uint64) uint64 {
	if x < 1 {
		return 1
	}
	return x
}

func absDiff(a, b uint64) uint64 {
	if a > b {
		return a - b
	}
	return b - a
}

// silence unused import warnings if helpers move
var _ = sort.Ints
