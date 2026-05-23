package cluster

import (
	"context"
	"os"
	"strconv"
	"testing"

	"github.com/ClickHouse/clickhouse-go/v2"
	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
)

func chTestConn(t *testing.T) driver.Conn {
	t.Helper()
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}
	opts, err := clickhouse.ParseDSN(dsn)
	if err != nil {
		t.Fatalf("parse dsn: %v", err)
	}
	conn, err := clickhouse.Open(opts)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := conn.Ping(context.Background()); err != nil {
		t.Fatalf("ping: %v", err)
	}
	return conn
}

func TestRunMigrations_CreatesTables(t *testing.T) {
	conn := chTestConn(t)
	defer conn.Close()

	if err := RunMigrations(context.Background(), conn, "migrations"); err != nil {
		t.Fatalf("RunMigrations: %v", err)
	}

	for _, table := range []string{"points", "staging_points", "point_id_map", "point_id_map_load"} {
		var n uint64
		row := conn.QueryRow(context.Background(),
			"SELECT count() FROM system.tables WHERE database='clustopher' AND name=?", table)
		if err := row.Scan(&n); err != nil {
			t.Fatalf("scan %s: %v", table, err)
		}
		if n != 1 {
			t.Fatalf("expected %s table to exist, got count=%d", table, n)
		}
	}

	for z := 2; z <= 16; z++ {
		var c uint64
		q := "SELECT count() FROM system.tables WHERE database='clustopher' AND name=?"
		row := conn.QueryRow(context.Background(), q, "rollup_z"+strconv.Itoa(z))
		if err := row.Scan(&c); err != nil {
			t.Fatalf("scan rollup_z%d: %v", z, err)
		}
		if c != 1 {
			t.Fatalf("expected rollup_z%d to exist", z)
		}
	}

	for z := 2; z <= 16; z++ {
		var c uint64
		q := "SELECT count() FROM system.tables WHERE database='clustopher' AND name=?"
		if err := conn.QueryRow(context.Background(), q, "mv_rollup_z"+strconv.Itoa(z)).Scan(&c); err != nil {
			t.Fatalf("scan mv_rollup_z%d: %v", z, err)
		}
		if c != 1 {
			t.Fatalf("expected mv_rollup_z%d to exist", z)
		}
	}
}
