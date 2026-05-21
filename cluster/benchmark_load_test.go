package cluster

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
)

// BenchmarkLoadCH_15M times only the Load phase (project + sort + skeleton +
// insertToCH). Skips OPTIMIZE/query setup so the profile focuses on creation.
//
//	go test ./cluster -run=^$ -bench=BenchmarkLoadCH_15M -benchtime=1x \
//	    -cpuprofile=load_cpu.pprof -memprofile=load_mem.pprof
func BenchmarkLoadCH_15M(b *testing.B) {
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		b.Skip("CLICKHOUSE_DSN not set")
	}
	ctx := context.Background()
	c, err := NewCHClient(ctx, CHConfig{DSN: dsn})
	if err != nil {
		b.Fatalf("client: %v", err)
	}
	defer c.Close()
	if err := RunMigrations(ctx, c.Conn(), "migrations"); err != nil {
		b.Fatalf("migrations: %v", err)
	}

	const n = 15_000_000
	pts := generateRandomPoints(n, -125, -65, 25, 49)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clusterID := fmt.Sprintf("PROF_%d_%d", n, i)
		// Clean any prior partitions for this id.
		b.StopTimer()
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
		b.StartTimer()

		if err := sc.Load(pts); err != nil {
			b.Fatalf("Load: %v", err)
		}

		b.StopTimer()
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
		for z := 2; z <= 16; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
		}
		b.StartTimer()
	}
}
