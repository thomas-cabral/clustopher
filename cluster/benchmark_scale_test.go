package cluster

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestScalePerf runs the CH-first staging path at escalating dataset sizes and
// dumps a markdown table with load time, per-zoom GetClustersCH timing, and
// resident skeleton size. Partitions are dropped between sizes so CH is reset
// to a clean state each round.
//
// Gated by CLUSTOPHER_SCALE_TESTS=1.
//
// Sizes (in millions, comma-separated) override the default with
// CLUSTOPHER_SCALE_SIZES, e.g. "50,100,200,500".
//
//	CLUSTOPHER_SCALE_TESTS=1 \
//	CLUSTOPHER_SCALE_SIZES=50,100,200 \
//	CLICKHOUSE_DSN=clickhouse://default:@127.0.0.1:19000/clustopher \
//	  go test ./cluster -run TestScalePerf -v -timeout 4h
func TestScalePerf(t *testing.T) {
	if os.Getenv("CLUSTOPHER_SCALE_TESTS") != "1" {
		t.Skip("set CLUSTOPHER_SCALE_TESTS=1 to run scale perf test")
	}
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}

	sizes := parseScaleSizes(os.Getenv("CLUSTOPHER_SCALE_SIZES"))
	if len(sizes) == 0 {
		sizes = []int{15_000_000, 50_000_000, 100_000_000, 200_000_000}
	}

	zooms := []int{2, 8, 14}
	conus := KDBounds{MinX: -125, MinY: 25, MaxX: -65, MaxY: 49}

	// Override the default viewportForZoom for the high-zoom case: 0.05° × 0.05°
	// over a uniform CONUS sample is too sparse for the skeleton path to do real
	// work at higher N. Use a 0.5° × 0.5° city-scale window in the dense centre
	// of CONUS so the skeleton walk has actual leaves to chew on.
	scaleViewport := func(zoom int) KDBounds {
		if zoom >= 11 {
			return KDBounds{MinX: -100.25, MinY: 39.25, MaxX: -99.75, MaxY: 39.75}
		}
		return viewportForZoom(zoom)
	}
	scaleViewportTag := func(zoom int) string {
		if zoom >= 11 {
			return "city (0.5°×0.5°)"
		}
		return viewportTag(zoom)
	}

	type zoomResult struct {
		zoom        int
		elapsedMs   float64
		mbPerOp     float64
		clusters    int
		viewportTag string
	}
	type sizeResult struct {
		n                  int
		stageSec           float64
		loadSec            float64
		optimizeSec        float64
		skeletonResidentMB float64
		leafCount          int
		zoom               []zoomResult
		err                string
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

	var results []sizeResult

	for _, n := range sizes {
		clusterID := fmt.Sprintf("SCALE_%d", n)
		t.Logf("=== scale size %s ===", humanCount(n))

		cleanup := func() {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
			for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
				_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
			}
		}
		cleanup()

		res := sizeResult{n: n}

		stageStart := time.Now()
		if err := generateDenseStagingPoints(ctx, c, clusterID, n, conus); err != nil {
			res.err = fmt.Sprintf("stage: %v", err)
			results = append(results, res)
			cleanup()
			t.Logf("FAIL %s: %s", humanCount(n), res.err)
			continue
		}
		res.stageSec = time.Since(stageStart).Seconds()
		t.Logf("staged %s points in CH in %.1fs", humanCount(n), res.stageSec)

		sc := NewSupercluster(SuperclusterOptions{
			MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
			Extent: 512, NodeSize: 64,
		})
		sc.SetCHClient(c)
		sc.SetClusterID(clusterID)

		loadStart := time.Now()
		streaming := os.Getenv("CLUSTOPHER_SCALE_STREAMING") == "1"
		var loadErr error
		if streaming {
			loadErr = sc.LoadFromCHStreaming(ctx)
		} else {
			loadErr = sc.LoadFromCHStaging(ctx)
		}
		if loadErr != nil {
			res.err = fmt.Sprintf("load: %v", loadErr)
			results = append(results, res)
			cleanup()
			t.Logf("FAIL %s: %s", humanCount(n), res.err)
			continue
		}
		res.loadSec = time.Since(loadStart).Seconds()
		loadPath := "KD"
		if streaming {
			loadPath = "Morton-stream"
		}
		t.Logf("loaded %s points (%s) in %.1fs", humanCount(n), loadPath, res.loadSec)

		// Free intermediates and force GC so the resident-after-load measurement
		// reflects what the skeleton actually pins.
		runtime.GC()
		debug.FreeOSMemory()

		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		res.skeletonResidentMB = float64(ms.HeapAlloc) / (1024 * 1024)
		if sc.Skeleton != nil {
			res.leafCount = len(sc.Skeleton.Leaves)
		}

		// Free staging now that the canonical points table is populated.
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)

		optStart := time.Now()
		for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
			_ = c.Conn().Exec(ctx, "OPTIMIZE TABLE clustopher.rollup_z"+strconv.Itoa(z)+" PARTITION ? FINAL", clusterID)
		}
		res.optimizeSec = time.Since(optStart).Seconds()
		t.Logf("optimized rollup partitions in %.1fs", res.optimizeSec)

		for _, z := range zooms {
			bounds := scaleViewport(z)
			t.Logf("z=%d query bounds=%+v skeleton_leaves=%d", z, bounds, res.leafCount)
			// Warmup query, ignored.
			warm, err := sc.GetClustersCH(ctx, bounds, z)
			if err != nil {
				t.Logf("warmup zoom=%d err: %v", z, err)
			} else {
				t.Logf("warmup zoom=%d clusters=%d", z, len(warm))
			}

			var before, after runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)

			const iters = 5
			start := time.Now()
			var clusters []ClusterNode
			for i := 0; i < iters; i++ {
				clusters, err = sc.GetClustersCH(ctx, bounds, z)
				if err != nil {
					t.Fatalf("GetClustersCH z=%d: %v", z, err)
				}
			}
			elapsed := time.Since(start) / iters
			runtime.ReadMemStats(&after)

			zr := zoomResult{
				zoom:        z,
				elapsedMs:   float64(elapsed.Microseconds()) / 1000.0,
				mbPerOp:     float64(after.TotalAlloc-before.TotalAlloc) / float64(iters) / (1024 * 1024),
				clusters:    len(clusters),
				viewportTag: scaleViewportTag(z),
			}
			res.zoom = append(res.zoom, zr)
			t.Logf("zoom=%d viewport=%s clusters=%d avg=%s alloc=%.2fMB/op",
				z, zr.viewportTag, zr.clusters, elapsed, zr.mbPerOp)
		}

		results = append(results, res)
		cleanup()
		runtime.GC()
		debug.FreeOSMemory()
	}

	// Render markdown report.
	var b strings.Builder
	fmt.Fprintf(&b, "# Clustopher scale results\n\n")
	fmt.Fprintf(&b, "Bench host: %s, GOMAXPROCS=%d, CONUS bounds (-125,25,-65,49).\n\n",
		runtime.GOOS+"/"+runtime.GOARCH, runtime.GOMAXPROCS(0))
	fmt.Fprintf(&b, "## Load + storage\n\n")
	fmt.Fprintf(&b, "| Points | CH stage gen | Load (skeleton build + CH point write) | Rollup OPTIMIZE | Resident heap after load | Skeleton leaves |\n")
	fmt.Fprintf(&b, "|--------|--------------|----------------------------------------|------------------|--------------------------|------------------|\n")
	for _, r := range results {
		if r.err != "" {
			fmt.Fprintf(&b, "| %s | — | — | — | — | FAIL: %s |\n", humanCount(r.n), r.err)
			continue
		}
		fmt.Fprintf(&b, "| %s | %.1fs | %.1fs | %.1fs | %.1f MB | %s |\n",
			humanCount(r.n), r.stageSec, r.loadSec, r.optimizeSec,
			r.skeletonResidentMB, humanCount(r.leafCount))
	}
	fmt.Fprintf(&b, "\n## Query latency (GetClustersCH, 5-iter avg)\n\n")
	fmt.Fprintf(&b, "| Points | Zoom | Viewport | Clusters returned | Avg latency | Alloc/op |\n")
	fmt.Fprintf(&b, "|--------|------|----------|-------------------|-------------|----------|\n")
	for _, r := range results {
		for _, zr := range r.zoom {
			fmt.Fprintf(&b, "| %s | %d | %s | %d | **%.1f ms** | %.2f MB |\n",
				humanCount(r.n), zr.zoom, zr.viewportTag, zr.clusters, zr.elapsedMs, zr.mbPerOp)
		}
	}

	report := b.String()
	if err := os.MkdirAll("../benchmark_results", 0o755); err == nil {
		out := fmt.Sprintf("../benchmark_results/scale_%s.md", time.Now().Format("20060102_150405"))
		if werr := os.WriteFile(out, []byte(report), 0o644); werr == nil {
			t.Logf("wrote report to %s", out)
		}
	}
	t.Log("\n" + report)
}

func parseScaleSizes(s string) []int {
	if s == "" {
		return nil
	}
	out := []int{}
	for _, tok := range strings.Split(s, ",") {
		tok = strings.TrimSpace(strings.ToLower(tok))
		if tok == "" {
			continue
		}
		mult := 1
		switch {
		case strings.HasSuffix(tok, "b"):
			mult = 1_000_000_000
			tok = strings.TrimSuffix(tok, "b")
		case strings.HasSuffix(tok, "m"):
			mult = 1_000_000
			tok = strings.TrimSuffix(tok, "m")
		case strings.HasSuffix(tok, "k"):
			mult = 1_000
			tok = strings.TrimSuffix(tok, "k")
		}
		v, err := strconv.Atoi(tok)
		if err != nil {
			continue
		}
		out = append(out, v*mult)
	}
	return out
}

func humanCount(n int) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(n)/1_000_000_000)
	case n >= 1_000_000:
		return fmt.Sprintf("%dM", n/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%dK", n/1_000)
	default:
		return strconv.Itoa(n)
	}
}

func viewportTag(zoom int) string {
	switch {
	case zoom <= 4:
		return "CONUS (60°×24°)"
	case zoom <= 10:
		return "state (5°×5°)"
	default:
		return "neighborhood (0.05°×0.05°)"
	}
}
