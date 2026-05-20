package cluster

import (
	"context"
	"os"
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
