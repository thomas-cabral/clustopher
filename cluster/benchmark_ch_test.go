package cluster

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"testing"
)

func benchmarkClusteringCHHuge(b *testing.B, numPoints int, zoom int) {
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
	clusterID := fmt.Sprintf("BENCH_%d_%d", numPoints, zoom)
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

	pts := generateRandomPoints(numPoints, -125, -65, 25, 49)
	if err := sc.Load(pts); err != nil {
		b.Fatalf("Load: %v", err)
	}
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "OPTIMIZE TABLE clustopher.rollup_z"+strconv.Itoa(z)+" PARTITION ? FINAL", clusterID)
	}

	bounds := KDBounds{MinX: -125, MinY: 25, MaxX: -65, MaxY: 49}

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := sc.GetClustersCH(ctx, bounds, zoom)
		if err != nil {
			b.Fatalf("GetClustersCH: %v", err)
		}
	}
	b.StopTimer()

	runtime.ReadMemStats(&after)
	b.ReportMetric(float64(after.TotalAlloc-before.TotalAlloc)/1024/1024, "MB/op")

	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
	}
}

func BenchmarkClusteringCHHuge_LowZoom(b *testing.B)  { benchmarkClusteringCHHuge(b, 15_000_000, 2) }
func BenchmarkClusteringCHHuge_MidZoom(b *testing.B)  { benchmarkClusteringCHHuge(b, 15_000_000, 8) }
func BenchmarkClusteringCHHuge_HighZoom(b *testing.B) { benchmarkClusteringCHHuge(b, 15_000_000, 14) }
