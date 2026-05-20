package cluster

import (
	"context"
	"os"
	"strconv"
	"testing"
)

func TestInsertPoints_WritesAllRows(t *testing.T) {
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
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
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION 'TEST_INSERT'")

	rows := []CHPointRow{
		{ClusterID: "TEST_INSERT", ID: 1, ExternalID: 100, X: -100, Y: 40, Metrics: map[string]float32{"v": 1}, Metadata: map[string]string{"k": "a"}},
		{ClusterID: "TEST_INSERT", ID: 2, ExternalID: 101, X: -101, Y: 41, Metrics: map[string]float32{"v": 2}, Metadata: map[string]string{"k": "b"}},
	}
	if err := c.InsertPoints(ctx, rows); err != nil {
		t.Fatalf("InsertPoints: %v", err)
	}

	var n uint64
	if err := c.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.points WHERE cluster_id = 'TEST_INSERT'").Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 2 {
		t.Fatalf("count = %d, want 2", n)
	}

	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION 'TEST_INSERT'")
}

func TestSuperclusterLoad_WritesToCH(t *testing.T) {
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
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
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION 'TEST_LOAD'")

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
		Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID("TEST_LOAD")
	pts := generateRandomPoints(2000, -125, -65, 25, 49)
	if err := sc.Load(pts); err != nil {
		t.Fatalf("Load: %v", err)
	}

	var n uint64
	if err := c.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.points WHERE cluster_id='TEST_LOAD'").Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 2000 {
		t.Fatalf("count = %d, want 2000", n)
	}
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION 'TEST_LOAD'")
}

func TestInsertPoints_TriggersRollupMVs(t *testing.T) {
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
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
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION 'TEST_MV'")
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION 'TEST_MV'")
	}

	rows := []CHPointRow{
		{ClusterID: "TEST_MV", ID: 1, X: -100, Y: 40, Metrics: map[string]float32{"v": 1}},
		{ClusterID: "TEST_MV", ID: 2, X: -100.001, Y: 40.001, Metrics: map[string]float32{"v": 3}},
	}
	if err := c.InsertPoints(ctx, rows); err != nil {
		t.Fatalf("insert: %v", err)
	}

	if err := c.Conn().Exec(ctx, "OPTIMIZE TABLE clustopher.rollup_z8 FINAL"); err != nil {
		t.Fatalf("optimize: %v", err)
	}

	var cnt uint64
	if err := c.Conn().QueryRow(ctx,
		"SELECT sum(cnt) FROM clustopher.rollup_z8 WHERE cluster_id = 'TEST_MV'").Scan(&cnt); err != nil {
		t.Fatalf("scan: %v", err)
	}
	if cnt != 2 {
		t.Fatalf("rollup_z8 cnt sum = %d, want 2", cnt)
	}

	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION 'TEST_MV'")
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION 'TEST_MV'")
	}
}
