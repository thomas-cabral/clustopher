package cluster

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"
)

// TestZ14QueryProfile drives GetClustersCH at z14 against a persistent 100M
// cluster for Go-side CPU/heap profiling. The cluster is staged + loaded on
// the first run and reused afterwards (partition QPROF_<N> is intentionally
// NOT dropped), so a second run profiles queries, not the load.
//
//	CLUSTOPHER_QPROF=1 CLICKHOUSE_DSN=... go test ./cluster -run TestZ14QueryProfile -v \
//	  -cpuprofile=../benchmark_results/z14_cpu.pprof -memprofile=../benchmark_results/z14_mem.pprof
func TestZ14QueryProfile(t *testing.T) {
	if os.Getenv("CLUSTOPHER_QPROF") != "1" {
		t.Skip("set CLUSTOPHER_QPROF=1")
	}
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}

	n := 100_000_000
	if v := os.Getenv("CLUSTOPHER_QPROF_N"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil || parsed < 1 {
			t.Fatalf("invalid CLUSTOPHER_QPROF_N=%q", v)
		}
		n = parsed
	}
	iters := 20
	if v := os.Getenv("CLUSTOPHER_QPROF_ITERS"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil || parsed < 1 {
			t.Fatalf("invalid CLUSTOPHER_QPROF_ITERS=%q", v)
		}
		iters = parsed
	}

	conus := KDBounds{MinX: -125, MinY: 25, MaxX: -65, MaxY: 49}
	city := KDBounds{MinX: -100.25, MinY: 39.25, MaxX: -99.75, MaxY: 39.75}

	ctx := context.Background()
	c, err := NewCHClient(ctx, CHConfig{DSN: dsn})
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	defer c.Close()
	if err := RunMigrations(ctx, c.Conn(), "migrations"); err != nil {
		t.Fatalf("migrations: %v", err)
	}

	cid := "QPROF_" + strconv.Itoa(n)
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40, Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(cid)

	var have uint64
	if err := c.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.points WHERE cluster_id = ?", cid).Scan(&have); err != nil {
		t.Fatalf("count existing: %v", err)
	}
	if have == uint64(n) {
		t.Logf("reusing existing partition %s", cid)
		start := time.Now()
		leaves, err := sc.readLeafBoundsFromPoints(ctx)
		if err != nil {
			t.Fatalf("leaf bounds: %v", err)
		}
		sc.Skeleton = BuildSkeletonFromLeaves(leaves)
		t.Logf("skeleton rebuilt in %s (%d leaves)", time.Since(start).Round(time.Millisecond), len(leaves))
	} else {
		t.Logf("staging %d points (existing=%d)", n, have)
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", cid)
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", cid)
		if err := generateDenseStagingPoints(ctx, c, cid, n, conus); err != nil {
			t.Fatalf("stage: %v", err)
		}
		start := time.Now()
		if err := sc.LoadFromCHSinglePass(ctx); err != nil {
			t.Fatalf("load: %v", err)
		}
		t.Logf("loaded in %s", time.Since(start).Round(time.Second))
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", cid)
	}

	// Warmup, then timed loop — this is the pprof target.
	for i := 0; i < 2; i++ {
		if _, err := sc.GetClustersCH(ctx, city, 14); err != nil {
			t.Fatalf("warmup query: %v", err)
		}
	}
	start := time.Now()
	var clusters []ClusterNode
	for i := 0; i < iters; i++ {
		clusters, err = sc.GetClustersCH(ctx, city, 14)
		if err != nil {
			t.Fatalf("query %d: %v", i, err)
		}
	}
	elapsed := time.Since(start)
	t.Logf("z14 city viewport: %d iters, avg %s, clusters=%d",
		iters, (elapsed / time.Duration(iters)).Round(time.Microsecond), len(clusters))
}
