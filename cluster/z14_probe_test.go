package cluster

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"
)

func TestZ14Probe(t *testing.T) {
	if os.Getenv("Z14_PROBE") != "1" {
		t.Skip("set Z14_PROBE=1")
	}
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("no DSN")
	}

	nStr := os.Getenv("PROBE_N")
	if nStr == "" {
		nStr = "30000000"
	}
	n, _ := strconv.Atoi(nStr)

	ctx := context.Background()
	c, err := NewCHClient(ctx, CHConfig{DSN: dsn})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if err := RunMigrations(ctx, c.Conn(), "migrations"); err != nil {
		t.Fatal(err)
	}

	cid := fmt.Sprintf("Z14_PROBE_%d", n)
	conus := KDBounds{MinX: -125, MinY: 25, MaxX: -65, MaxY: 49}
	city := KDBounds{MinX: -100.25, MinY: 39.25, MaxX: -99.75, MaxY: 39.75}

	cleanup := func() {
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.staging_points DROP PARTITION ?", cid)
		_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", cid)
		for z := 2; z <= 16; z++ {
			_ = c.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", cid)
		}
	}
	cleanup()

	t.Logf("staging gen N=%d", n)
	start := time.Now()
	if err := generateDenseStagingPoints(ctx, c, cid, n, conus); err != nil {
		t.Fatal(err)
	}
	t.Logf("staged in %s", time.Since(start))

	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40, Extent: 512, NodeSize: 64,
	})
	sc.SetCHClient(c)
	sc.SetClusterID(cid)

	start = time.Now()
	if err := sc.LoadFromCHStaging(ctx); err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded in %s, leaves=%d", time.Since(start), len(sc.Skeleton.Leaves))

	// Probe 1: project city viewport to skeleton space.
	tl := sc.projectFast(city.MinX, city.MaxY, 16)
	br := sc.projectFast(city.MaxX, city.MinY, 16)
	vp := KDBounds{MinX: tl[0], MinY: tl[1], MaxX: br[0], MaxY: br[1]}
	t.Logf("city vp lnglat: %+v", city)
	t.Logf("city vp projected: %+v", vp)

	// Probe 2: count overlapping leaves.
	idxs := sc.Skeleton.RangeLeaves(vp)
	t.Logf("overlapping leaves: %d", len(idxs))
	if len(idxs) > 0 {
		t.Logf("first 3 leaves:")
		for i, idx := range idxs {
			if i >= 3 {
				break
			}
			lf := sc.Skeleton.Leaves[idx]
			t.Logf("  leaf[%d] count=%d bounds=%+v idmin=%d idmax=%d", idx, lf.Count, lf.Bounds, lf.IDMin, lf.IDMax)
		}
	}

	// Probe 3: CH-side: points inside city viewport.
	var chCount uint64
	if err := c.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.points WHERE cluster_id = ? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?",
		cid, city.MinX, city.MaxX, city.MinY, city.MaxY).Scan(&chCount); err != nil {
		t.Fatal(err)
	}
	t.Logf("CH points in city viewport: %d", chCount)

	// Probe 4: actual GetClustersCH at z=14.
	clusters, err := sc.GetClustersCH(ctx, city, 14)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("GetClustersCH z=14 clusters=%d", len(clusters))
	for i, cl := range clusters {
		if i >= 3 {
			break
		}
		t.Logf("  cl[%d] count=%d x=%f y=%f", i, cl.Count, cl.X, cl.Y)
	}

	// Probe 5: root node + skeleton overall bounds.
	if n := len(sc.Skeleton.Nodes); n > 0 {
		root := sc.Skeleton.Nodes[n-1]
		t.Logf("root node bounds: %+v leafIdx=%d L=%d R=%d", root.Bounds, root.LeafIdx, root.Left, root.Right)
		t.Logf("total nodes=%d total leaves=%d", n, len(sc.Skeleton.Leaves))
	}
}
