package cluster

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
)

// viewportForZoom returns a realistic viewport for the given zoom level.
// Low zooms = continental/CONUS view; high zooms = city-scale.
func viewportForZoom(zoom int) KDBounds {
	switch {
	case zoom <= 4:
		// Full CONUS (matches dataset extent).
		return KDBounds{MinX: -125, MinY: 25, MaxX: -65, MaxY: 49}
	case zoom <= 10:
		// ~5° × 5° region (state-scale).
		return KDBounds{MinX: -100, MinY: 37, MaxX: -95, MaxY: 42}
	default:
		// ~0.05° × 0.05° (neighborhood / city block).
		return KDBounds{MinX: -100.025, MinY: 39.475, MaxX: -99.975, MaxY: 39.525}
	}
}

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

	bounds := viewportForZoom(zoom)

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

func TestPerfClusteringCHDenseNYC_15M(t *testing.T) {
	if os.Getenv("CLUSTOPHER_PERF_TESTS") != "1" {
		t.Skip("set CLUSTOPHER_PERF_TESTS=1 to run the 15M dense NYC perf test")
	}
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

	const n = 15_000_000
	clusterID := "PERF_DENSE_NYC_15M"
	nycBounds := KDBounds{MinX: -74.2591, MinY: 40.4774, MaxX: -73.7004, MaxY: 40.9176}

	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
	}
	defer func() {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
		for z := 2; z <= 16; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
		}
	}()

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
		Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)

	start := time.Now()
	pts := generateRandomPoints(n, nycBounds.MinX, nycBounds.MaxX, nycBounds.MinY, nycBounds.MaxY)
	t.Logf("generated %d NYC points in %s", n, time.Since(start))

	start = time.Now()
	if err := sc.Load(pts); err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Logf("loaded %d NYC points into ClickHouse in %s", n, time.Since(start))
	pts = nil
	runtime.GC()

	start = time.Now()
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "OPTIMIZE TABLE clustopher.rollup_z"+strconv.Itoa(z)+" PARTITION ? FINAL", clusterID)
	}
	t.Logf("optimized rollup partitions in %s", time.Since(start))

	for _, zoom := range []int{8, 10, 11, 12, 14} {
		start = time.Now()
		clusters, err := sc.GetClustersCH(ctx, nycBounds, zoom)
		elapsed := time.Since(start)
		if err != nil {
			t.Fatalf("GetClustersCH zoom=%d: %v", zoom, err)
		}
		var total uint64
		for _, c := range clusters {
			total += uint64(c.Count)
		}
		t.Logf("zoom=%d viewport=NYC clusters=%d total_count=%d elapsed=%s", zoom, len(clusters), total, elapsed)
	}
}

// BenchmarkClusteringCHDenseNYC_15M is intended for Go pprof runs against a
// dense single-city dataset.
//
//	CLICKHOUSE_DSN=clickhouse://default:@127.0.0.1:19000/clustopher \
//	  go test ./cluster -run '^$' -bench '^BenchmarkClusteringCHDenseNYC_15M$' \
//	  -benchtime=1x -count=1 -timeout=30m -benchmem \
//	  -cpuprofile=nyc_cpu.pprof -memprofile=nyc_mem.pprof
//
// Inspect with:
//
//	go tool pprof -http=:0 ./cluster.test nyc_cpu.pprof
//	go tool pprof -http=:0 ./cluster.test nyc_mem.pprof
func BenchmarkClusteringCHDenseNYC_15M(b *testing.B) {
	b.ReportAllocs()

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
	clusterID := "BENCH_DENSE_NYC_15M"
	nycBounds := KDBounds{MinX: -74.2591, MinY: 40.4774, MaxX: -73.7004, MaxY: 40.9176}

	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
	}
	defer func() {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
		for z := 2; z <= 16; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
		}
	}()

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
		Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)

	pts := generateRandomPoints(n, nycBounds.MinX, nycBounds.MaxX, nycBounds.MinY, nycBounds.MaxY)
	loadStart := time.Now()
	if err := sc.Load(pts); err != nil {
		b.Fatalf("Load: %v", err)
	}
	b.Logf("loaded %d NYC points in %s", n, time.Since(loadStart))
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "OPTIMIZE TABLE clustopher.rollup_z"+strconv.Itoa(z)+" PARTITION ? FINAL", clusterID)
	}

	for _, zoom := range []int{8, 10, 11, 12, 14} {
		b.Run(fmt.Sprintf("Zoom%d_FullNYCViewport", zoom), func(b *testing.B) {
			var before, after runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)

			var clusters []ClusterNode
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clusters, err = sc.GetClustersCH(ctx, nycBounds, zoom)
				if err != nil {
					b.Fatalf("GetClustersCH: %v", err)
				}
			}
			b.StopTimer()

			runtime.ReadMemStats(&after)
			b.ReportMetric(float64(after.TotalAlloc-before.TotalAlloc)/1024/1024, "MB/op")
			b.ReportMetric(float64(len(clusters)), "clusters/op")
		})
	}
}

// BenchmarkClusteringCHFirstDenseNYC_15M loads generated dense NYC data into
// ClickHouse staging first, then builds the exact KD leaf order from compact
// spatial rows streamed out of CH. This profiles the proposed CH-first ingest
// path without Go materializing []Point maps/interfaces for the source data.
//
//	CLICKHOUSE_DSN=clickhouse://default:@127.0.0.1:19000/clustopher \
//	  go test ./cluster -run '^$' -bench '^BenchmarkClusteringCHFirstDenseNYC_15M$' \
//	  -benchtime=1x -count=1 -timeout=30m -benchmem \
//	  -cpuprofile=benchmark_results/nyc_chfirst_cpu.pprof \
//	  -memprofile=benchmark_results/nyc_chfirst_mem.pprof -v
func BenchmarkClusteringCHFirstDenseNYC_15M(b *testing.B) {
	benchmarkClusteringCHFirstStaged(b, "BENCH_CHFIRST_DENSE_NYC_15M", 15_000_000, KDBounds{
		MinX: -74.2591, MinY: 40.4774, MaxX: -73.7004, MaxY: 40.9176,
	})
}

func BenchmarkClusteringCHFirstWesternHemisphere_15M(b *testing.B) {
	benchmarkClusteringCHFirstStaged(b, "BENCH_CHFIRST_WESTERN_HEMISPHERE_15M", 15_000_000, KDBounds{
		MinX: -180, MinY: -60, MaxX: 0, MaxY: 85,
	})
}

func benchmarkClusteringCHFirstStaged(b *testing.B, clusterID string, n int, bounds KDBounds) {
	b.ReportAllocs()

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

	cleanup := func() {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
		for z := 2; z <= 16; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
		}
	}
	cleanup()
	defer cleanup()

	stageStart := time.Now()
	if err := generateDenseStagingPoints(ctx, c, clusterID, n, bounds); err != nil {
		b.Fatalf("generate staging: %v", err)
	}
	b.Logf("generated %d staged points in CH in %s", n, time.Since(stageStart))

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
		Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)

	loadStart := time.Now()
	if err := sc.LoadFromCHStaging(ctx); err != nil {
		b.Fatalf("LoadFromCHStaging: %v", err)
	}
	b.Logf("loaded %d points from CH staging in %s", n, time.Since(loadStart))
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "OPTIMIZE TABLE clustopher.rollup_z"+strconv.Itoa(z)+" PARTITION ? FINAL", clusterID)
	}

	for _, zoom := range []int{8, 10, 11, 12, 14} {
		b.Run(fmt.Sprintf("Zoom%d_FullNYCViewport", zoom), func(b *testing.B) {
			var before, after runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)

			var clusters []ClusterNode
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clusters, err = sc.GetClustersCH(ctx, bounds, zoom)
				if err != nil {
					b.Fatalf("GetClustersCH: %v", err)
				}
			}
			b.StopTimer()

			runtime.ReadMemStats(&after)
			b.ReportMetric(float64(after.TotalAlloc-before.TotalAlloc)/1024/1024, "MB/op")
			b.ReportMetric(float64(len(clusters)), "clusters/op")
		})
	}
}

func generateDenseStagingPoints(ctx context.Context, c *CHClient, clusterID string, n int, bounds KDBounds) error {
	ctx = clickhouse.Context(ctx, clickhouse.WithSettings(insertSettings))
	return c.Conn().Exec(ctx, `
        INSERT INTO clustopher.staging_points (cluster_id, external_id, x, y, metrics, metadata)
        SELECT
            ? AS cluster_id,
            toUInt32(number + 1) AS external_id,
            toFloat32(? + randCanonical() * (? - ?)) AS x,
            toFloat32(? + randCanonical() * (? - ?)) AS y,
            map('value', toFloat32(randCanonical() * 100)) AS metrics,
            map('type', 'test') AS metadata
        FROM numbers(?)
    `, clusterID, bounds.MinX, bounds.MaxX, bounds.MinX, bounds.MinY, bounds.MaxY, bounds.MinY, uint64(n))
}
