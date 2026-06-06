package cluster

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
)

// TestSinglePassPartitionedEquivalence verifies the partitioned single-pass
// INSERT produces a canonical points table equivalent to the global INSERT:
// ids are exactly 1..N, the morton key is non-decreasing in id order, and the
// external_id population is preserved.
func TestSinglePassPartitionedEquivalence(t *testing.T) {
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}

	const n = 500_000
	genBBox := KDBounds{MinX: -124, MinY: 26, MaxX: -66, MaxY: 48}
	specs := map[string]metricSpec{
		"value_const_1": {CHExpr: "toFloat32(1)"},
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

	clusterID := "SP_PART_EQ"
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

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40, Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)

	t.Setenv("CLUSTOPHER_SINGLEPASS_PARTITIONS", "4")
	t.Setenv("CLUSTOPHER_SINGLEPASS_CONCURRENCY", "2")
	if err := sc.LoadFromCHSinglePass(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}

	// ids are exactly 1..N with no gaps or duplicates.
	var cnt, idMin, idMax, idUniq uint64
	if err := c.Conn().QueryRow(ctx, `
        SELECT count(), toUInt64(min(id)), toUInt64(max(id)), uniqExact(id)
        FROM clustopher.points WHERE cluster_id = ?
    `, clusterID).Scan(&cnt, &idMin, &idMax, &idUniq); err != nil {
		t.Fatalf("id stats: %v", err)
	}
	if cnt != n || idMin != 1 || idMax != n || idUniq != n {
		t.Errorf("id space broken: cnt=%d min=%d max=%d uniq=%d want n=%d", cnt, idMin, idMax, idUniq, n)
	}

	// Morton key non-decreasing in id order (the global sort invariant).
	var violations uint64
	mq := fmt.Sprintf(`
        SELECT countIf(m < mPrev AND rn > 1) FROM (
            SELECT
                %s AS m,
                lagInFrame(%s) OVER w AS mPrev,
                row_number() OVER w AS rn
            FROM clustopher.points
            WHERE cluster_id = ?
            WINDOW w AS (ORDER BY id ASC ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)
        )
    `, mortonExprSQL, mortonExprSQL)
	if err := c.Conn().QueryRow(ctx, mq, clusterID).Scan(&violations); err != nil {
		t.Fatalf("morton monotonicity query: %v", err)
	}
	if violations != 0 {
		t.Errorf("morton order violated at %d positions", violations)
	}

	// external_id population preserved exactly (ids unique per cluster).
	var stagingUniq, pointsUniq uint64
	var stagingSum, pointsSum uint64
	if err := c.Conn().QueryRow(ctx, `
        SELECT uniqExact(external_id), sum(toUInt64(external_id))
        FROM clustopher.staging_points WHERE cluster_id = ?
    `, clusterID).Scan(&stagingUniq, &stagingSum); err != nil {
		t.Fatalf("staging external ids: %v", err)
	}
	if err := c.Conn().QueryRow(ctx, `
        SELECT uniqExact(external_id), sum(toUInt64(external_id))
        FROM clustopher.points WHERE cluster_id = ?
    `, clusterID).Scan(&pointsUniq, &pointsSum); err != nil {
		t.Fatalf("points external ids: %v", err)
	}
	if stagingUniq != pointsUniq || stagingSum != pointsSum {
		t.Errorf("external_id population changed: staging uniq=%d sum=%d, points uniq=%d sum=%d",
			stagingUniq, stagingSum, pointsUniq, pointsSum)
	}

	// Skeleton sanity: leaf count matches ceil(n / NodeSize).
	wantLeaves := (n + 63) / 64
	if got := len(sc.Skeleton.Leaves); got != wantLeaves {
		t.Errorf("skeleton leaves=%d want %d", got, wantLeaves)
	}
}
