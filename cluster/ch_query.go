package cluster

import (
	"context"
	"fmt"
	"math"
)

// queryRollup queries the per-zoom rollup table for the viewport.
func (sc *Supercluster) queryRollup(ctx context.Context, viewport KDBounds, zoom int) ([]ClusterNode, error) {
	if sc.ch == nil {
		return nil, fmt.Errorf("queryRollup requires CH client")
	}
	if zoom < 2 || zoom > 16 {
		return nil, fmt.Errorf("zoom %d out of rollup range [2,16]", zoom)
	}

	// Convert viewport (lng/lat) → tile-coord range at this zoom.
	// tile_x = ((lng + 180) / 360) * 2^zoom * 512 / radius
	radius := float32(sc.Options.Radius)
	scale := float32(math.Pow(2, float64(zoom))) * 512 / radius

	minTX := uint32(((viewport.MinX + 180) / 360) * scale)
	maxTX := uint32(((viewport.MaxX + 180) / 360) * scale)

	sinLat := func(lat float32) float32 {
		return float32(math.Sin(float64(lat) * math.Pi / 180))
	}
	latToY := func(lat float32) float32 {
		s := sinLat(lat)
		return float32((0.5 - math.Log(float64((1+s)/(1-s)))/(4*math.Pi)) * float64(scale))
	}
	minTY := uint32(latToY(viewport.MaxY))
	maxTY := uint32(latToY(viewport.MinY))
	if minTY > maxTY {
		minTY, maxTY = maxTY, minTY
	}

	table := fmt.Sprintf("clustopher.rollup_z%d", zoom)
	q := fmt.Sprintf(`
        SELECT tile_x, tile_y, cnt, sum_x, sum_y, metric_sums, metric_cnts
        FROM %s
        WHERE cluster_id = ?
          AND tile_x BETWEEN ? AND ?
          AND tile_y BETWEEN ? AND ?
    `, table)

	rows, err := sc.ch.Conn().Query(ctx, q, sc.clusterID, minTX, maxTX, minTY, maxTY)
	if err != nil {
		return nil, fmt.Errorf("query rollup: %w", err)
	}
	defer rows.Close()

	out := make([]ClusterNode, 0, 256)
	var (
		tileX, tileY uint32
		cnt          uint64
		sumX, sumY   float64
		metricSums   map[string]float64
		metricCnts   map[string]uint64
	)
	for rows.Next() {
		if err := rows.Scan(&tileX, &tileY, &cnt, &sumX, &sumY, &metricSums, &metricCnts); err != nil {
			return nil, fmt.Errorf("scan rollup row: %w", err)
		}
		cx := float32(sumX / float64(cnt))
		cy := float32(sumY / float64(cnt))
		metrics := make(map[string]float32, len(metricSums))
		for k, s := range metricSums {
			n := metricCnts[k]
			if n > 0 {
				metrics[k] = float32(s / float64(n))
			}
		}
		out = append(out, ClusterNode{
			ID:      synthClusterID(zoom, tileX, tileY),
			X:       cx,
			Y:       cy,
			Count:   uint32(cnt),
			Metrics: metrics,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// synthClusterID derives a stable cluster id from (zoom, tile_x, tile_y) using
// a simple bit-packing. Mirrors how Mapbox supercluster encodes synthetic ids.
func synthClusterID(zoom int, tileX, tileY uint32) uint32 {
	return (tileX << 16) ^ (tileY << 5) ^ uint32(zoom)
}

// queryTree handles zoom >= ZSplit. Walks the skeleton, classifies leaves,
// fetches their contents from CH, and emits clusters.
func (sc *Supercluster) queryTree(ctx context.Context, viewport KDBounds, zoom int) ([]ClusterNode, error) {
	if sc.Skeleton == nil {
		return nil, fmt.Errorf("queryTree requires Skeleton (call Open or Load first)")
	}

	// Convert viewport (lng/lat) to MaxZoom pixel space; skeleton bounds are there.
	topLeft := sc.projectFast(viewport.MinX, viewport.MaxY, sc.Options.MaxZoom)
	botRight := sc.projectFast(viewport.MaxX, viewport.MinY, sc.Options.MaxZoom)
	vp := KDBounds{
		MinX: topLeft[0],
		MinY: topLeft[1],
		MaxX: botRight[0],
		MaxY: botRight[1],
	}

	leafIdxs := sc.Skeleton.RangeLeaves(vp)
	inside := make([]int32, 0, len(leafIdxs))
	partial := make([]int32, 0, len(leafIdxs))
	for _, i := range leafIdxs {
		b := sc.Skeleton.Leaves[i].Bounds
		if b.MinX >= vp.MinX && b.MaxX <= vp.MaxX && b.MinY >= vp.MinY && b.MaxY <= vp.MaxY {
			inside = append(inside, i)
		} else {
			partial = append(partial, i)
		}
	}

	out := make([]ClusterNode, 0, len(leafIdxs))

	if len(inside) > 0 {
		agg, err := sc.aggregateLeaves(ctx, inside)
		if err != nil {
			return nil, err
		}
		out = append(out, agg...)
	}

	if len(partial) > 0 {
		pts, err := sc.fetchLeafPoints(ctx, partial, zoom)
		if err != nil {
			return nil, err
		}
		if len(pts) > 0 {
			clusters := sc.clusterPoints(pts, float32(sc.Options.Radius))
			sc.unprojectClusters(clusters, zoom)
			out = append(out, clusters...)
		}
	}

	return out, nil
}

// aggregateLeaves emits one ClusterNode per inside leaf by aggregating across
// each leaf's id range in CH. One round-trip via UNION ALL of per-leaf SELECTs.
func (sc *Supercluster) aggregateLeaves(ctx context.Context, leaves []int32) ([]ClusterNode, error) {
	if len(leaves) == 0 {
		return nil, nil
	}
	parts := make([]string, 0, len(leaves))
	args := []interface{}{}
	for _, li := range leaves {
		leaf := sc.Skeleton.Leaves[li]
		parts = append(parts, `
            SELECT toInt32(?) AS leaf_idx, count() AS cnt, avg(x) AS cx, avg(y) AS cy,
                   sumMap(metrics) AS msum,
                   sumMap(mapFromArrays(mapKeys(metrics), arrayMap(v -> toUInt64(1), mapValues(metrics)))) AS mcnt
            FROM clustopher.points
            WHERE cluster_id = ? AND id BETWEEN ? AND ?
        `)
		args = append(args, li, sc.clusterID, leaf.IDMin, leaf.IDMax)
	}
	q := joinUnion(parts)
	rows, err := sc.ch.Conn().Query(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("aggregate leaves: %w", err)
	}
	defer rows.Close()

	out := make([]ClusterNode, 0, len(leaves))
	var (
		leafIdx int32
		cnt     uint64
		cx, cy  float64
		msum    map[string]float64
		mcnt    map[string]uint64
	)
	for rows.Next() {
		if err := rows.Scan(&leafIdx, &cnt, &cx, &cy, &msum, &mcnt); err != nil {
			return nil, fmt.Errorf("scan agg row: %w", err)
		}
		if cnt < uint64(sc.Options.MinPoints) {
			continue
		}
		metrics := make(map[string]float32, len(msum))
		for k, s := range msum {
			if n := mcnt[k]; n > 0 {
				metrics[k] = float32(s / float64(n))
			}
		}
		out = append(out, ClusterNode{
			ID:      uint32(leafIdx),
			X:       float32(cx),
			Y:       float32(cy),
			Count:   uint32(cnt),
			Metrics: metrics,
		})
	}
	return out, rows.Err()
}

func joinUnion(parts []string) string {
	s := parts[0]
	for _, p := range parts[1:] {
		s += "\nUNION ALL\n" + p
	}
	return s
}

// fetchLeafPoints fetches all points belonging to the given leaves and returns
// them projected to the target zoom (so existing clusterPoints can run).
func (sc *Supercluster) fetchLeafPoints(ctx context.Context, leaves []int32, zoom int) ([]KDPoint, error) {
	if len(leaves) == 0 {
		return nil, nil
	}
	parts := make([]string, 0, len(leaves))
	args := []interface{}{}
	for _, li := range leaves {
		leaf := sc.Skeleton.Leaves[li]
		parts = append(parts, `(cluster_id = ? AND id BETWEEN ? AND ?)`)
		args = append(args, sc.clusterID, leaf.IDMin, leaf.IDMax)
	}
	q := `SELECT id, x, y FROM clustopher.points WHERE ` + joinOr(parts)

	rows, err := sc.ch.Conn().Query(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("fetch leaf points: %w", err)
	}
	defer rows.Close()

	out := make([]KDPoint, 0, len(leaves)*sc.Options.NodeSize)
	var id uint32
	var x, y float32
	for rows.Next() {
		if err := rows.Scan(&id, &x, &y); err != nil {
			return nil, err
		}
		proj := sc.projectFast(x, y, zoom)
		out = append(out, KDPoint{ID: id, X: proj[0], Y: proj[1], NumPoints: 1})
	}
	return out, rows.Err()
}

func joinOr(parts []string) string {
	s := parts[0]
	for _, p := range parts[1:] {
		s += " OR " + p
	}
	return s
}
