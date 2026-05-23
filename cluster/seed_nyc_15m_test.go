package cluster

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

// TestSeedNYC15M seeds a 15M-point cluster confined to NYC bounds using the
// Morton streaming load path. Gated by SEED_NYC_15M=1.
//
//	SEED_NYC_15M=1 CLUSTOPHER_CLUSTER_ID=NYC_15M \
//	  CLICKHOUSE_DSN=clickhouse://default:@127.0.0.1:19000/clustopher \
//	  go test ./cluster -run TestSeedNYC15M -v -timeout 1h
func TestSeedNYC15M(t *testing.T) {
	if os.Getenv("SEED_NYC_15M") != "1" {
		t.Skip("set SEED_NYC_15M=1 to seed a 15M NYC cluster")
	}
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}

	clusterID := os.Getenv("CLUSTOPHER_CLUSTER_ID")
	if clusterID == "" {
		clusterID = "NYC_15M"
	}

	ctx := context.Background()
	c, err := NewCHClient(ctx, CHConfig{DSN: dsn})
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	defer c.Close()
	if err := RunMigrations(ctx, c.Conn(), "migrations"); err != nil {
		t.Fatalf("migrations: %v", err)
	}

	// NYC bounding box.
	nyc := KDBounds{MinX: -74.2591, MinY: 40.4774, MaxX: -73.7004, MaxY: 40.9176}

	// Clean any prior partition with this id.
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		_ = c.Conn().Exec(ctx, fmt.Sprintf("ALTER TABLE clustopher.rollup_z%d DROP PARTITION ?", z), clusterID)
	}

	const n = 15_000_000
	start := time.Now()
	if err := generateDenseStagingPoints(ctx, c, clusterID, n, nyc); err != nil {
		t.Fatalf("stage: %v", err)
	}
	t.Logf("staged %d NYC points in %.1fs", n, time.Since(start).Seconds())

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40, Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)

	start = time.Now()
	if err := sc.LoadFromCHStreaming(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Logf("loaded NYC cluster id=%s in %.1fs leaves=%d", clusterID, time.Since(start).Seconds(), len(sc.Skeleton.Leaves))

	start = time.Now()
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		_ = c.Conn().Exec(ctx, fmt.Sprintf("OPTIMIZE TABLE clustopher.rollup_z%d PARTITION ? FINAL", z), clusterID)
	}
	t.Logf("optimized rollups in %.1fs", time.Since(start).Seconds())

	// Drop staging — canonical points are now populated.
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)

	t.Logf("DONE. Cluster '%s' ready: 15M points across NYC (%v).", clusterID, nyc)
}
