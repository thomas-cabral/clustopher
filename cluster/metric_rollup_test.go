package cluster

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
)

// metricSpec describes how a single metric key is generated per-point and
// what its dataset-wide sum should be, given n points.
//
// CHExpr is a ClickHouse expression evaluated in the SELECT used to build the
// staging table. It can reference `number` (the row's 0-based index) and must
// produce a Float32. ExpectedSum returns the exact ground-truth sum across
// all n points in float64.
type metricSpec struct {
	CHExpr      string
	ExpectedSum func(n int) float64
}

// TestMetricRollupAllZooms stages a deterministic dataset where each metric
// key has a known per-point formula and a closed-form total. The test then:
//
//  1. probes clustopher.staging_points to sanity-check the staged dataset,
//  2. loads via LoadFromCHStreaming so the rollup MVs fire on insert,
//  3. probes each rollup_z{N} table directly to confirm SummingMergeTree is
//     actually summing the Map columns (catches the SimpleAggregateFunction
//     bug in the rollup schema),
//  4. for each zoom 2..16, queries GetClustersCH with a world viewport so all
//     points must be returned, and asserts Σ Count == n and the per-metric
//     reconstructed Σ Count*mean matches the expected sum within eps. This
//     catches both the rollup-merge bug (low/mid zoom) and the
//     partial-leaf-path metric-drop bug (high zoom).
//
// Gated by CLUSTOPHER_METRIC_ROLLUP=1; needs a live ClickHouse. Env:
//
//	CLICKHOUSE_DSN              standard CH DSN (required)
//	CLUSTOPHER_METRIC_ROLLUP    must be "1" to enable
//	CLUSTOPHER_METRIC_ROLLUP_N  optional point count override (default 50000)
//	                            must be a multiple of seqMod (100) so the
//	                            varying-metric expected sum stays exact.
func TestMetricRollupAllZooms(t *testing.T) {
	if os.Getenv("CLUSTOPHER_METRIC_ROLLUP") != "1" {
		t.Skip("set CLUSTOPHER_METRIC_ROLLUP=1 to run metric rollup verification")
	}
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}

	n := 50_000
	if v := os.Getenv("CLUSTOPHER_METRIC_ROLLUP_N"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil || parsed < 1 {
			t.Fatalf("invalid CLUSTOPHER_METRIC_ROLLUP_N=%q", v)
		}
		n = parsed
	}

	const seqMod = 100
	if n%seqMod != 0 {
		t.Fatalf("n=%d must be a multiple of seqMod=%d", n, seqMod)
	}

	// Generation box well away from Mercator poles.
	genBBox := KDBounds{MinX: -100, MinY: 35, MaxX: -90, MaxY: 45}
	// World viewport stays clear of ±90 (latToY -> ±∞) but covers every
	// generated point at every zoom.
	worldVP := KDBounds{MinX: -180, MinY: -85, MaxX: 180, MaxY: 85}

	specs := map[string]metricSpec{
		// Constant per-point sanity metric.
		"value_const_1": {
			CHExpr:      "toFloat32(1)",
			ExpectedSum: func(n int) float64 { return float64(n) },
		},
		// Distinct constant — value differs from value_const_1 so a
		// "always returns 1" implementation can't sneak through.
		"value_const_7": {
			CHExpr:      "toFloat32(7)",
			ExpectedSum: func(n int) float64 { return float64(n) * 7 },
		},
		// Varying metric: (number mod 100) + 1, so each integer 1..100
		// appears exactly n/seqMod times. Catches the SummingMergeTree
		// Map-merge bug, which only manifests when per-point values
		// differ (mean of a single sample row ≠ mean of all rows).
		"value_seq_mod": {
			CHExpr:      "toFloat32((number % 100) + 1)",
			ExpectedSum: func(n int) float64 { return float64(n) / float64(seqMod) * float64(seqMod*(seqMod+1)/2) },
		},
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

	clusterID := fmt.Sprintf("METRIC_ROLLUP_%d", n)
	cleanup := func() {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
		for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
		}
	}
	cleanup()
	defer cleanup()

	if err := generateMetricPoints(ctx, c, clusterID, n, genBBox, specs); err != nil {
		t.Fatalf("stage: %v", err)
	}

	keys := make([]string, 0, len(specs))
	for k := range specs {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	expectedSums := make(map[string]float64, len(specs))
	for _, k := range keys {
		expectedSums[k] = specs[k].ExpectedSum(n)
	}

	// Sanity-check the staging row before loading.
	if err := assertCHSums(ctx, c, "staging_points sanity",
		"SELECT count(), sumMap(metrics) FROM clustopher.staging_points WHERE cluster_id = ?",
		clusterID, uint64(n), expectedSums, 0, t); err != nil {
		t.Fatalf("staging sanity: %v", err)
	}

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40, Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)

	start := time.Now()
	if err := sc.LoadFromCHStreaming(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Logf("loaded n=%d in %.2fs leaves=%d", n, time.Since(start).Seconds(), len(sc.Skeleton.Leaves))

	// Force rollup MV merges so per-tile rows are fully merged. Only zooms
	// [MinRollupZoom..MaxRollupZoom] have MV tables; higher zooms answer via
	// the skeleton path and have no rollup to verify directly.
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		_ = c.Conn().Exec(ctx, "OPTIMIZE TABLE clustopher.rollup_z"+strconv.Itoa(z)+" PARTITION ? FINAL", clusterID)
	}

	// 1) Verify the canonical points table received the right data.
	if err := assertCHSums(ctx, c, "clustopher.points",
		"SELECT count(), sumMap(metrics) FROM clustopher.points WHERE cluster_id = ?",
		clusterID, uint64(n), expectedSums, 0, t); err != nil {
		t.Errorf("points sanity: %v", err)
	}

	// 2) Direct rollup-table probe: catches SummingMergeTree map-merge bug
	// independent of the Go query path. Limited to the zooms that actually
	// have rollup MV tables; high zooms (>= ZSplit) are validated via path 3.
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		label := fmt.Sprintf("rollup_z%02d", z)
		q := fmt.Sprintf(
			"SELECT sum(cnt), sumMap(metric_sums) FROM clustopher.rollup_z%d WHERE cluster_id = ?", z)
		if err := assertCHSums(ctx, c, label, q, clusterID, uint64(n), expectedSums, 1e-9, t); err != nil {
			t.Errorf("%s: %v", label, err)
		}
		// Also verify metric_cnts sums to n per key — SummingMergeTree
		// must sum it too.
		row := c.Conn().QueryRow(ctx,
			"SELECT sumMap(metric_cnts) FROM clustopher.rollup_z"+strconv.Itoa(z)+" WHERE cluster_id = ?", clusterID)
		var cntsMap map[string]uint64
		if err := row.Scan(&cntsMap); err != nil {
			t.Errorf("%s metric_cnts scan: %v", label, err)
			continue
		}
		for _, k := range keys {
			if cntsMap[k] != uint64(n) {
				t.Errorf("%s metric_cnts[%s] = %d, want %d", label, k, cntsMap[k], n)
			}
		}
	}

	// 3) Go-level: GetClustersCH per zoom with world viewport. Σ Count must
	// equal n; Σ Count*mean must equal expected sum within eps.
	const relEps = 1e-4

	var anyFail bool
	for z := 2; z <= 16; z++ {
		qStart := time.Now()
		clusters, err := sc.GetClustersCH(ctx, worldVP, z)
		if err != nil {
			t.Errorf("z=%d GetClustersCH: %v", z, err)
			anyFail = true
			continue
		}

		var totalCount uint64
		observedSums := make(map[string]float64, len(specs))
		for _, cl := range clusters {
			totalCount += uint64(cl.Count)
			for _, k := range keys {
				observedSums[k] += float64(cl.Count) * float64(cl.Metrics[k])
			}
		}

		var sumStr strings.Builder
		for i, k := range keys {
			if i > 0 {
				sumStr.WriteString(" ")
			}
			fmt.Fprintf(&sumStr, "%s=%.1f/%.1f", k, observedSums[k], expectedSums[k])
		}
		path := "rollup"
		if z >= sc.Options.ZSplit {
			path = "skel"
		}
		t.Logf("z=%02d path=%-6s clusters=%6d count=%d/%d %s elapsed=%.1fms",
			z, path, len(clusters), totalCount, uint64(n), sumStr.String(),
			float64(time.Since(qStart).Microseconds())/1000.0)

		if totalCount != uint64(n) {
			t.Errorf("z=%d count mismatch: observed=%d expected=%d", z, totalCount, n)
			anyFail = true
		}
		for _, k := range keys {
			exp := expectedSums[k]
			obs := observedSums[k]
			denom := math.Abs(exp)
			if denom == 0 {
				denom = 1
			}
			rel := math.Abs(obs-exp) / denom
			if rel > relEps {
				t.Errorf("z=%d metric %s sum mismatch: observed=%.6f expected=%.6f rel=%.2e",
					z, k, obs, exp, rel)
				anyFail = true
			}
		}
	}
	if anyFail {
		t.Fatal("metric rollup verification failed")
	}
}

// assertCHSums runs a single query returning (count, sumMap(metrics)) for the
// supplied clusterID and checks the row against expectedCount + expectedSums.
// relEps > 0 enables float-relative comparison; 0 forces exact uint match.
func assertCHSums(
	ctx context.Context,
	c *CHClient,
	label string,
	query string,
	clusterID string,
	expectedCount uint64,
	expectedSums map[string]float64,
	relEps float64,
	t *testing.T,
) error {
	row := c.Conn().QueryRow(ctx, query, clusterID)
	var (
		cnt  uint64
		sums map[string]float64
	)
	if err := row.Scan(&cnt, &sums); err != nil {
		return fmt.Errorf("%s scan: %w", label, err)
	}
	t.Logf("%s cnt=%d sums=%v", label, cnt, sums)
	if cnt != expectedCount {
		return fmt.Errorf("%s cnt=%d, want %d", label, cnt, expectedCount)
	}
	keys := make([]string, 0, len(expectedSums))
	for k := range expectedSums {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		exp := expectedSums[k]
		got := sums[k]
		if relEps == 0 {
			if got != exp {
				return fmt.Errorf("%s sum[%s]=%g, want %g", label, k, got, exp)
			}
			continue
		}
		denom := math.Abs(exp)
		if denom == 0 {
			denom = 1
		}
		if math.Abs(got-exp)/denom > relEps {
			return fmt.Errorf("%s sum[%s]=%g, want %g (rel=%.2e)",
				label, k, got, exp, math.Abs(got-exp)/denom)
		}
	}
	return nil
}

// generateMetricPoints stages n points with deterministic cityHash64-driven
// coordinates inside bbox. Each metric in specs is generated per-point via its
// CHExpr (a ClickHouse expression over `number`), so callers can encode both
// constant-per-point metrics and varying ones — and the closed-form
// ExpectedSum lets the test compute ground truth in-Go.
func generateMetricPoints(ctx context.Context, c *CHClient, clusterID string, n int, bbox KDBounds, specs map[string]metricSpec) error {
	if n < 1 {
		return fmt.Errorf("generateMetricPoints requires n>=1, got %d", n)
	}
	if len(specs) == 0 {
		return fmt.Errorf("generateMetricPoints requires at least one metric")
	}
	ctx = clickhouse.Context(ctx, clickhouse.WithSettings(insertSettings))

	keys := make([]string, 0, len(specs))
	for k := range specs {
		if strings.ContainsAny(k, "'\\") {
			return fmt.Errorf("metric key %q contains disallowed character", k)
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys)*2)
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("'%s'", k))
		parts = append(parts, specs[k].CHExpr)
	}
	metricsExpr := "map(" + strings.Join(parts, ", ") + ")"

	query := fmt.Sprintf(`
        INSERT INTO clustopher.staging_points (cluster_id, external_id, x, y, metrics, metadata)
        SELECT
            ? AS cluster_id,
            toUInt32(number + 1) AS external_id,
            toFloat32(? + (cityHash64(number, 1) / 18446744073709551615.0) * (? - ?)) AS x,
            toFloat32(? + (cityHash64(number, 2) / 18446744073709551615.0) * (? - ?)) AS y,
            %s AS metrics,
            map('type', 'metric_rollup_test') AS metadata
        FROM numbers(?)
    `, metricsExpr)

	return c.Conn().Exec(ctx, query,
		clusterID,
		bbox.MinX, bbox.MaxX, bbox.MinX,
		bbox.MinY, bbox.MaxY, bbox.MinY,
		uint64(n),
	)
}