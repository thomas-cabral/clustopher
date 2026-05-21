package cluster

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"
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

func TestSuperclusterOpen_RebuildsSkeleton(t *testing.T) {
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
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION 'TEST_OPEN'")

	sc1 := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
		Extent: 512, NodeSize: 64,
	})
	sc1.SetCHClient(c)
	sc1.SetClusterID("TEST_OPEN")
	pts := generateRandomPoints(5000, -125, -65, 25, 49)
	if err := sc1.Load(pts); err != nil {
		t.Fatalf("Load: %v", err)
	}
	leafCount1 := len(sc1.Skeleton.Leaves)

	sc2 := NewSupercluster(sc1.Options)
	sc2.SetCHClient(c)
	sc2.SetClusterID("TEST_OPEN")
	if err := sc2.Open(ctx); err != nil {
		t.Fatalf("Open: %v", err)
	}

	if sc2.Skeleton == nil {
		t.Fatal("Skeleton nil after Open")
	}
	if len(sc2.Skeleton.Leaves) != leafCount1 {
		t.Fatalf("leaves: %d vs %d", len(sc2.Skeleton.Leaves), leafCount1)
	}
	var sum uint32
	for _, l := range sc2.Skeleton.Leaves {
		sum += l.Count
	}
	if sum != 5000 {
		t.Fatalf("skeleton total = %d", sum)
	}

	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION 'TEST_OPEN'")
}

func TestSuperclusterLoadFromCHStaging_WritesCanonicalPoints(t *testing.T) {
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

	clusterID := "TEST_STAGE_LOAD_" + strconv.FormatInt(time.Now().UnixNano(), 10)
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
	}
	defer func() {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
		for z := 2; z <= 16; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
		}
	}()

	rows := []CHStagingPointRow{
		{ClusterID: clusterID, ExternalID: 100, X: -74.0, Y: 40.7, Metrics: map[string]float32{"v": 1}, Metadata: map[string]string{"kind": "a"}},
		{ClusterID: clusterID, ExternalID: 101, X: -74.1, Y: 40.8, Metrics: map[string]float32{"v": 2}, Metadata: map[string]string{"kind": "b"}},
		{ClusterID: clusterID, ExternalID: 102, X: -73.9, Y: 40.6, Metrics: map[string]float32{"v": 3}, Metadata: map[string]string{"kind": "c"}},
	}
	if err := c.InsertStagingPoints(ctx, rows); err != nil {
		t.Fatalf("InsertStagingPoints: %v", err)
	}

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
		Extent: 512, NodeSize: 2,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)
	if err := sc.LoadFromCHStaging(ctx); err != nil {
		t.Fatalf("LoadFromCHStaging: %v", err)
	}
	if sc.Skeleton == nil || len(sc.Skeleton.Leaves) == 0 {
		t.Fatal("missing skeleton")
	}

	var n uint64
	if err := c.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.points WHERE cluster_id = ?", clusterID).Scan(&n); err != nil {
		t.Fatalf("count points: %v", err)
	}
	if n != uint64(len(rows)) {
		t.Fatalf("points count = %d, want %d", n, len(rows))
	}

	var maps uint64
	if err := c.Conn().QueryRow(ctx, `
        SELECT count()
        FROM clustopher.points
        WHERE cluster_id = ? AND metrics['v'] > 0 AND metadata['kind'] != ''
    `, clusterID).Scan(&maps); err != nil {
		t.Fatalf("count maps: %v", err)
	}
	if maps != uint64(len(rows)) {
		t.Fatalf("preserved map rows = %d, want %d", maps, len(rows))
	}

	var maxID uint32
	if err := c.Conn().QueryRow(ctx,
		"SELECT max(id) FROM clustopher.points WHERE cluster_id = ?", clusterID).Scan(&maxID); err != nil {
		t.Fatalf("max id: %v", err)
	}
	if maxID != uint32(len(rows)) {
		t.Fatalf("max internal id = %d, want %d", maxID, len(rows))
	}
}

func TestLoadFromCHStaging_CanonicalMatchesStagingMetrics(t *testing.T) {
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

	clusterID := "TEST_STAGE_SOURCE_TRUTH_" + strconv.FormatInt(time.Now().UnixNano(), 10)
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
	_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
	for z := 2; z <= 16; z++ {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
	}
	defer func() {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", clusterID)
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", clusterID)
		for z := 2; z <= 16; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", clusterID)
		}
	}()

	rows := []CHStagingPointRow{
		{ClusterID: clusterID, ExternalID: 10, X: -74.00, Y: 40.70, Metrics: map[string]float32{"v": 1, "w": 10}, Metadata: map[string]string{"kind": "a"}},
		{ClusterID: clusterID, ExternalID: 20, X: -74.01, Y: 40.71, Metrics: map[string]float32{"v": 2, "w": 20}, Metadata: map[string]string{"kind": "b"}},
		{ClusterID: clusterID, ExternalID: 30, X: -74.02, Y: 40.72, Metrics: map[string]float32{"v": 3, "w": 30}, Metadata: map[string]string{"kind": "a"}},
		{ClusterID: clusterID, ExternalID: 40, X: -74.03, Y: 40.73, Metrics: map[string]float32{"v": 4, "w": 40}, Metadata: map[string]string{"kind": "b"}},
	}
	if err := c.InsertStagingPoints(ctx, rows); err != nil {
		t.Fatalf("InsertStagingPoints: %v", err)
	}

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 2, Radius: 40,
		Extent: 512, NodeSize: 2,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(clusterID)
	if err := sc.LoadFromCHStaging(ctx); err != nil {
		t.Fatalf("LoadFromCHStaging: %v", err)
	}

	var stagingN, pointsN uint64
	if err := c.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.staging_points WHERE cluster_id = ?", clusterID).Scan(&stagingN); err != nil {
		t.Fatalf("count staging: %v", err)
	}
	if err := c.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.points WHERE cluster_id = ?", clusterID).Scan(&pointsN); err != nil {
		t.Fatalf("count points: %v", err)
	}
	if pointsN != stagingN {
		t.Fatalf("canonical count = %d, staging count = %d", pointsN, stagingN)
	}

	var stagingV, pointsV float64
	if err := c.Conn().QueryRow(ctx,
		"SELECT sum(metrics['v']) FROM clustopher.staging_points WHERE cluster_id = ?", clusterID).Scan(&stagingV); err != nil {
		t.Fatalf("sum staging metric: %v", err)
	}
	if err := c.Conn().QueryRow(ctx,
		"SELECT sum(metrics['v']) FROM clustopher.points WHERE cluster_id = ?", clusterID).Scan(&pointsV); err != nil {
		t.Fatalf("sum points metric: %v", err)
	}
	if pointsV != stagingV {
		t.Fatalf("canonical metric sum = %f, staging metric sum = %f", pointsV, stagingV)
	}

	bounds := KDBounds{MinX: -74.1, MinY: 40.6, MaxX: -73.9, MaxY: 40.8}
	leafIdxs := make([]int32, len(sc.Skeleton.Leaves))
	for i := range leafIdxs {
		leafIdxs[i] = int32(i)
	}
	treeClusters, skipped, err := sc.aggregateLeaves(ctx, leafIdxs)
	if err != nil {
		t.Fatalf("aggregateLeaves: %v", err)
	}
	if len(skipped) != 0 {
		t.Fatalf("aggregateLeaves skipped %d leaves", len(skipped))
	}
	var treeTotal uint64
	var treeV float64
	for _, c := range treeClusters {
		treeTotal += uint64(c.Count)
		treeV += float64(c.Metrics["v"]) * float64(c.Count)
	}
	if treeTotal != stagingN {
		t.Fatalf("aggregate leaf total = %d, staging count = %d", treeTotal, stagingN)
	}
	if treeV != stagingV {
		t.Fatalf("aggregate leaf weighted v sum = %f, staging v sum = %f", treeV, stagingV)
	}

	clusters, err := sc.queryRollup(ctx, bounds, 8)
	if err != nil {
		t.Fatalf("queryRollup: %v", err)
	}
	var rollupTotal uint64
	for _, c := range clusters {
		rollupTotal += uint64(c.Count)
	}
	if rollupTotal != stagingN {
		t.Fatalf("rollup total = %d, staging count = %d", rollupTotal, stagingN)
	}
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
