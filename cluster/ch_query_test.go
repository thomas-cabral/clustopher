package cluster

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func setupQueryFixture(t *testing.T, clusterID string, n int) *Supercluster {
	t.Helper()
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}
	ctx := context.Background()
	c, err := NewCHClient(ctx, CHConfig{DSN: dsn})
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	if err := RunMigrations(ctx, c.Conn(), "migrations"); err != nil {
		t.Fatalf("migrations: %v", err)
	}
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
	}
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
		Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)
	pts := generateRandomPoints(n, -125, -65, 25, 49)
	if err := sc.Load(pts); err != nil {
		t.Fatalf("Load: %v", err)
	}
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "OPTIMIZE TABLE clustopher.rollup_z"+strconv.Itoa(z)+" PARTITION ? FINAL", clusterID)
	}
	return sc
}

// Silence unused imports for now; ch_query_test.go uses them in later tasks.
var _ = json.Unmarshal
var _ = filepath.Join

func TestQueryTree_HighZoomTotalsMatch(t *testing.T) {
	sc := setupQueryFixture(t, "TEST_TR", 10000)
	defer sc.ch.Close()

	bounds := KDBounds{MinX: -100, MinY: 35, MaxX: -95, MaxY: 40}
	clusters, err := sc.queryTree(context.Background(), bounds, 14)
	if err != nil {
		t.Fatalf("queryTree: %v", err)
	}

	var total uint32
	for _, c := range clusters {
		total += c.Count
	}
	if total == 0 {
		t.Fatal("no points returned")
	}
	if total > 10000 {
		t.Fatalf("total %d > 10000", total)
	}
}

func TestQueryRollup_TotalsMatchInput(t *testing.T) {
	sc := setupQueryFixture(t, "TEST_RU", 10000)
	defer sc.ch.Close()

	bounds := KDBounds{MinX: -125, MinY: 25, MaxX: -65, MaxY: 49}
	clusters, err := sc.queryRollup(context.Background(), bounds, 8)
	if err != nil {
		t.Fatalf("queryRollup: %v", err)
	}
	var total uint32
	for _, c := range clusters {
		total += c.Count
	}
	if total != 10000 {
		t.Fatalf("rollup total count = %d, want 10000", total)
	}
}
