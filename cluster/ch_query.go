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
