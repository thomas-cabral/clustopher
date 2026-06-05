package cluster

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
)

// TestRollupCascadeEquivalence verifies that the cascade rollup populate
// (z10 from points, z9..z2 each derived from the zoom above) produces the
// same per-tile aggregates as the direct populate (every zoom from points).
//
// Integer columns (cnt, metric_cnts, tile coords) must match exactly; float
// sums are compared with a relative tolerance because the two strategies sum
// in different orders.
func TestRollupCascadeEquivalence(t *testing.T) {
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}

	const n = 200_000
	genBBox := KDBounds{MinX: -124, MinY: 26, MaxX: -66, MaxY: 48}
	specs := map[string]metricSpec{
		"value_const_1": {CHExpr: "toFloat32(1)"},
		"value_seq_mod": {CHExpr: "toFloat32((number % 100) + 1)"},
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

	clusterID := "ROLLUP_CASCADE_EQ"
	snapTable := func(z int) string { return "clustopher.tmp_cascade_snap_z" + strconv.Itoa(z) }
	cleanup := func() {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
		for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
			_ = c.Conn().Exec(ctx, "DROP TABLE IF EXISTS "+snapTable(z))
		}
	}
	cleanup()
	defer cleanup()

	if err := generateMetricPoints(ctx, c, clusterID, n, genBBox, specs); err != nil {
		t.Fatalf("stage: %v", err)
	}

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40, Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)
	t.Setenv("CLUSTOPHER_DEFERRED_ROLLUPS", "0") // load without any rollup populate
	if err := sc.detachRollupMVs(ctx); err != nil {
		t.Fatalf("detach MVs: %v", err)
	}
	defer func() { _ = sc.attachRollupMVs(context.Background()) }()
	if err := sc.LoadFromCHSinglePass(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}

	// Reference: direct populate, snapshot fully-aggregated tiles per zoom.
	if err := sc.populateRollupsDirect(ctx); err != nil {
		t.Fatalf("direct populate: %v", err)
	}
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		q := fmt.Sprintf(`
            CREATE TABLE %s ENGINE = Memory AS
            SELECT tile_x, tile_y, sum(cnt) AS cnt, sum(sum_x) AS sum_x, sum(sum_y) AS sum_y,
                   sumMap(metric_sums) AS metric_sums, sumMap(metric_cnts) AS metric_cnts
            FROM clustopher.rollup_z%d
            WHERE cluster_id = ?
            GROUP BY tile_x, tile_y
        `, snapTable(z), z)
		if err := c.Conn().Exec(ctx, q, clusterID); err != nil {
			t.Fatalf("snapshot z%d: %v", z, err)
		}
	}

	// Candidate: cascade populate into freshly dropped partitions.
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		if err := c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID); err != nil {
			t.Fatalf("drop rollup_z%d: %v", z, err)
		}
	}
	if err := sc.populateRollupsCascade(ctx); err != nil {
		t.Fatalf("cascade populate: %v", err)
	}

	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		var snapRows, cascadeRows, mismatches uint64
		if err := c.Conn().QueryRow(ctx,
			"SELECT count() FROM "+snapTable(z)).Scan(&snapRows); err != nil {
			t.Fatalf("count snapshot z%d: %v", z, err)
		}
		if err := c.Conn().QueryRow(ctx, fmt.Sprintf(
			"SELECT uniqExact(tile_x, tile_y) FROM clustopher.rollup_z%d WHERE cluster_id = ?", z),
			clusterID).Scan(&cascadeRows); err != nil {
			t.Fatalf("count cascade z%d: %v", z, err)
		}
		if snapRows == 0 {
			t.Fatalf("z%d: snapshot empty — reference populate produced no tiles", z)
		}
		if snapRows != cascadeRows {
			t.Errorf("z%d: tile count mismatch direct=%d cascade=%d", z, snapRows, cascadeRows)
		}

		q := fmt.Sprintf(`
            SELECT count() FROM (
                SELECT * FROM %s AS s
                FULL OUTER JOIN (
                    SELECT tile_x, tile_y, sum(cnt) AS ccnt, sum(sum_x) AS csum_x, sum(sum_y) AS csum_y,
                           sumMap(metric_sums) AS cmetric_sums, sumMap(metric_cnts) AS cmetric_cnts
                    FROM clustopher.rollup_z%d
                    WHERE cluster_id = ?
                    GROUP BY tile_x, tile_y
                ) AS c USING (tile_x, tile_y)
                WHERE s.cnt != c.ccnt
                   OR abs(s.sum_x - c.csum_x) > greatest(abs(s.sum_x) * 1e-9, 1e-6)
                   OR abs(s.sum_y - c.csum_y) > greatest(abs(s.sum_y) * 1e-9, 1e-6)
                   OR s.metric_cnts != c.cmetric_cnts
                   OR arraySort(mapKeys(s.metric_sums)) != arraySort(mapKeys(c.cmetric_sums))
                   OR arrayExists(k -> abs(s.metric_sums[k] - c.cmetric_sums[k]) > greatest(abs(s.metric_sums[k]) * 1e-9, 1e-6), mapKeys(s.metric_sums))
            )
        `, snapTable(z), z)
		if err := c.Conn().QueryRow(ctx, q, clusterID).Scan(&mismatches); err != nil {
			t.Fatalf("compare z%d: %v", z, err)
		}
		if mismatches != 0 {
			t.Errorf("z%d: %d mismatched tiles between direct and cascade populate", z, mismatches)
		}
	}
}
