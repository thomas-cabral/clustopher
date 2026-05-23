package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

const maxLeafIDRangesPerQuery = 128

type leafIDRange struct {
	min uint32
	max uint32
}

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
        SELECT
          tile_x,
          tile_y,
          sum(cnt) AS cnt,
          sum(sum_x) AS sum_x,
          sum(sum_y) AS sum_y,
          sumMap(metric_sums) AS metric_sums,
          sumMap(metric_cnts) AS metric_cnts
        FROM %s
        WHERE cluster_id = ?
          AND tile_x BETWEEN ? AND ?
          AND tile_y BETWEEN ? AND ?
        GROUP BY tile_x, tile_y
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
		if cnt == 0 {
			continue
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

	// A leaf can be safely aggregated as a single cluster only when its
	// physical extent at the QUERY zoom is no larger than the cluster radius.
	// Otherwise its constituent points would cluster radius-grained under the
	// legacy clusterPoints behavior. Leaf bounds live in MaxZoom pixel space;
	// scale them to query zoom and compare diagonal to radius.
	radius := float32(sc.Options.Radius)
	zoomScale := float32(math.Pow(2, float64(zoom-sc.Options.MaxZoom))) // <= 1 when zoom <= MaxZoom
	for _, i := range leafIdxs {
		leaf := sc.Skeleton.Leaves[i]
		b := leaf.Bounds
		fullyInside := b.MinX >= vp.MinX && b.MaxX <= vp.MaxX && b.MinY >= vp.MinY && b.MaxY <= vp.MaxY
		if !fullyInside {
			partial = append(partial, i)
			continue
		}
		dx := (b.MaxX - b.MinX) * zoomScale
		dy := (b.MaxY - b.MinY) * zoomScale
		// Use squared diagonal vs squared (2*radius) to compare without sqrt.
		// 2*radius is the maximum allowable diameter for a single cluster.
		if dx*dx+dy*dy > (2*radius)*(2*radius) {
			partial = append(partial, i)
			continue
		}
		inside = append(inside, i)
	}

	out := make([]ClusterNode, 0, len(leafIdxs))

	if len(inside) > 0 {
		agg, skipped, err := sc.aggregateLeaves(ctx, inside)
		if err != nil {
			return nil, err
		}
		out = append(out, agg...)
		// Inside leaves with cnt < MinPoints fall through to the partial path so
		// their points are emitted individually (or aggregated with overlapping
		// partial-leaf neighbors).
		partial = append(partial, skipped...)
	}

	if len(partial) > 0 {
		pts, attrs, err := sc.fetchLeafPoints(ctx, partial, zoom)
		if err != nil {
			return nil, err
		}
		if len(pts) > 0 {
			clusters := sc.clusterPoints(pts, float32(sc.Options.Radius))
			attachAttrsFromMembers(clusters, attrs)
			sc.unprojectClusters(clusters, zoom)
			out = append(out, clusters...)
		}
	}

	// Second-pass radius merge: aggregateLeaves emits one ClusterNode per
	// skeleton leaf without applying the cluster radius across leaves, so at
	// dense data the inside-leaf path can return tens of thousands of micro-
	// clusters that should be merged into a few hundred radius-blobs. Run the
	// same radius merge on the combined `out` so the result respects the
	// configured Options.Radius regardless of which sub-path produced each
	// cluster.
	if len(out) > 1 {
		out = sc.mergeClustersByRadius(out, zoom, float32(sc.Options.Radius))
	}

	return out, nil
}

// mergeClustersByRadius coalesces ClusterNodes whose centers fall within
// `radius` pixels at the given query zoom. Inputs and outputs are in lng/lat.
// Counts and metric averages are aggregated weighted by each input cluster's
// existing Count; metadata is carried over from the first contributor in the
// group (we don't have enough per-cluster info to majority-vote across maps
// at this stage — `attachAttrsFromMembers` already did that for skeleton-path
// clusters that have raw member ids).
func (sc *Supercluster) mergeClustersByRadius(clusters []ClusterNode, zoom int, radius float32) []ClusterNode {
	if len(clusters) <= 1 {
		return clusters
	}
	type pt struct {
		idx  int
		x, y float32
	}
	pts := make([]pt, len(clusters))
	for i, c := range clusters {
		proj := sc.projectFast(c.X, c.Y, zoom)
		pts[i] = pt{idx: i, x: proj[0], y: proj[1]}
	}
	sort.Slice(pts, func(i, j int) bool { return pts[i].x < pts[j].x })

	processed := make([]bool, len(clusters))
	out := make([]ClusterNode, 0, len(clusters))
	r2 := radius * radius

	for i := range pts {
		srcIdx := pts[i].idx
		if processed[srcIdx] {
			continue
		}
		group := []int{srcIdx}
		for j := i + 1; j < len(pts); j++ {
			if pts[j].x-pts[i].x > radius {
				break
			}
			if processed[pts[j].idx] {
				continue
			}
			dx := pts[j].x - pts[i].x
			dy := pts[j].y - pts[i].y
			if dx*dx+dy*dy <= r2 {
				group = append(group, pts[j].idx)
			}
		}
		if len(group) == 1 {
			processed[srcIdx] = true
			out = append(out, clusters[srcIdx])
			continue
		}
		var sumX, sumY float64
		var totalCount uint32
		metricSum := make(map[string]float64)
		merged := ClusterNode{
			Metrics:  make(map[string]float32),
			Metadata: make(map[string]json.RawMessage),
		}
		metaSeeded := false
		for _, gi := range group {
			c := clusters[gi]
			w := float64(c.Count)
			sumX += float64(c.X) * w
			sumY += float64(c.Y) * w
			totalCount += c.Count
			for k, v := range c.Metrics {
				metricSum[k] += float64(v) * w
			}
			if !metaSeeded && len(c.Metadata) > 0 {
				for k, v := range c.Metadata {
					merged.Metadata[k] = v
				}
				metaSeeded = true
			}
			processed[gi] = true
		}
		merged.ID = clusters[group[0]].ID
		merged.X = float32(sumX / float64(totalCount))
		merged.Y = float32(sumY / float64(totalCount))
		merged.Count = totalCount
		for k, s := range metricSum {
			merged.Metrics[k] = float32(s / float64(totalCount))
		}
		out = append(out, merged)
	}
	return out
}

// pointAttrs carries per-point metrics + metadata fetched from CH alongside
// the geometry. Used by the partial-leaf path so the in-Go clusterPoints stage
// can attach aggregated metric/metadata back onto each emitted ClusterNode.
type pointAttrs struct {
	Metrics  map[string]float32
	Metadata map[string]string
}

// attachAttrsFromMembers fills each cluster's Metrics/Metadata by aggregating
// the per-member attributes recorded in attrs. Metrics are averaged so the
// per-cluster value matches the convention used by the rollup and
// aggregateLeaves paths (sum/cnt); metadata picks the most common value per
// key across members. Clusters whose Children list is empty (defensive — all
// clusters built by clusterPoints carry their member IDs) are left untouched.
func attachAttrsFromMembers(clusters []ClusterNode, attrs map[uint32]pointAttrs) {
	if len(attrs) == 0 {
		return
	}
	for i := range clusters {
		members := clusters[i].Children
		if len(members) == 0 {
			continue
		}
		sums := make(map[string]float64)
		counts := make(map[string]uint64)
		metaFreq := make(map[string]map[string]int)
		for _, id := range members {
			a, ok := attrs[id]
			if !ok {
				continue
			}
			for k, v := range a.Metrics {
				sums[k] += float64(v)
				counts[k]++
			}
			for k, v := range a.Metadata {
				inner, ok := metaFreq[k]
				if !ok {
					inner = make(map[string]int)
					metaFreq[k] = inner
				}
				inner[v]++
			}
		}
		if len(sums) > 0 {
			metrics := make(map[string]float32, len(sums))
			for k, s := range sums {
				if n := counts[k]; n > 0 {
					metrics[k] = float32(s / float64(n))
				}
			}
			clusters[i].Metrics = metrics
		}
		if len(metaFreq) > 0 {
			meta := make(map[string]json.RawMessage, len(metaFreq))
			for k, freq := range metaFreq {
				var topVal string
				var topCnt int
				for v, c := range freq {
					if c > topCnt {
						topVal = v
						topCnt = c
					}
				}
				if b, err := json.Marshal(topVal); err == nil {
					meta[k] = b
				}
			}
			clusters[i].Metadata = meta
		}
	}
}

// aggregateLeaves emits one ClusterNode per inside leaf by aggregating across
// each leaf's id range in CH. Leaves are compressed into contiguous internal-id
// ranges so dense full-viewport queries do not serialize giant leaf arrays into
// ClickHouse query text. Leaves whose point count falls below MinPoints are
// returned in skipped so the caller can route them through the partial-leaf path
// (fetchLeafPoints + clusterPoints) to avoid silently dropping their points.
func (sc *Supercluster) aggregateLeaves(ctx context.Context, leaves []int32) (clusters []ClusterNode, skipped []int32, err error) {
	if len(leaves) == 0 {
		return nil, nil, nil
	}

	ranges := sc.leafIDRanges(leaves)
	if len(ranges) == 0 {
		return nil, nil, nil
	}
	nodeSize := uint64(sc.Options.NodeSize)

	out := make([]ClusterNode, 0, len(leaves))
	var skip []int32
	for start := 0; start < len(ranges); start += maxLeafIDRangesPerQuery {
		end := start + maxLeafIDRangesPerQuery
		if end > len(ranges) {
			end = len(ranges)
		}
		where, rangeArgs := leafRangePredicate(ranges[start:end])
		q := fmt.Sprintf(`
        SELECT
            toInt32(intDiv(id - 1, %d)) AS leaf_idx,
            count() AS cnt,
            avg(x) AS cx,
            avg(y) AS cy,
            sumMap(metrics) AS msum,
            sumMap(mapFromArrays(mapKeys(metrics), arrayMap(v -> toUInt64(1), mapValues(metrics)))) AS mcnt
        FROM clustopher.points
        WHERE cluster_id = ?
          AND (%s)
        GROUP BY leaf_idx
    `, nodeSize, where)

		args := make([]any, 0, 1+len(rangeArgs))
		args = append(args, sc.clusterID)
		args = append(args, rangeArgs...)

		rows, err := sc.ch.Conn().Query(ctx, q, args...)
		if err != nil {
			return nil, nil, fmt.Errorf("aggregate leaves: %w", err)
		}
		var (
			leafIdx int32
			cnt     uint64
			cx, cy  float64
			msum    map[string]float64
			mcnt    map[string]uint64
		)
		for rows.Next() {
			if err := rows.Scan(&leafIdx, &cnt, &cx, &cy, &msum, &mcnt); err != nil {
				rows.Close()
				return nil, nil, fmt.Errorf("scan agg row: %w", err)
			}
			if cnt < uint64(sc.Options.MinPoints) {
				skip = append(skip, leafIdx)
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
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, nil, err
		}
		rows.Close()
	}
	return out, skip, nil
}

// fetchLeafPoints fetches all points belonging to the given leaves and returns
// them projected to the target zoom (so existing clusterPoints can run).
// Leaves are compressed into contiguous internal-id ranges to avoid serializing
// large leaf arrays into ClickHouse query text.
func (sc *Supercluster) fetchLeafPoints(ctx context.Context, leaves []int32, zoom int) ([]KDPoint, map[uint32]pointAttrs, error) {
	if len(leaves) == 0 {
		return nil, nil, nil
	}

	ranges := sc.leafIDRanges(leaves)
	if len(ranges) == 0 {
		return nil, nil, nil
	}

	out := make([]KDPoint, 0, len(leaves)*sc.Options.NodeSize)
	attrs := make(map[uint32]pointAttrs, len(leaves)*sc.Options.NodeSize)
	for start := 0; start < len(ranges); start += maxLeafIDRangesPerQuery {
		end := start + maxLeafIDRangesPerQuery
		if end > len(ranges) {
			end = len(ranges)
		}
		where, rangeArgs := leafRangePredicate(ranges[start:end])
		q := fmt.Sprintf(`
        SELECT id, x, y, metrics, metadata FROM clustopher.points
        WHERE cluster_id = ?
          AND (%s)
    `, where)

		args := make([]any, 0, 1+len(rangeArgs))
		args = append(args, sc.clusterID)
		args = append(args, rangeArgs...)

		rows, err := sc.ch.Conn().Query(ctx, q, args...)
		if err != nil {
			return nil, nil, fmt.Errorf("fetch leaf points: %w", err)
		}
		var (
			id       uint32
			x, y     float32
			metrics  map[string]float32
			metadata map[string]string
		)
		for rows.Next() {
			if err := rows.Scan(&id, &x, &y, &metrics, &metadata); err != nil {
				rows.Close()
				return nil, nil, err
			}
			proj := sc.projectFast(x, y, zoom)
			out = append(out, KDPoint{ID: id, X: proj[0], Y: proj[1], NumPoints: 1})
			if len(metrics) > 0 || len(metadata) > 0 {
				attrs[id] = pointAttrs{Metrics: metrics, Metadata: metadata}
			}
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, nil, err
		}
		rows.Close()
	}
	return out, attrs, nil
}

func (sc *Supercluster) leafIDRanges(leaves []int32) []leafIDRange {
	if len(leaves) == 0 || sc.Skeleton == nil {
		return nil
	}
	sorted := make([]int32, 0, len(leaves))
	for _, leafIdx := range leaves {
		if leafIdx >= 0 && int(leafIdx) < len(sc.Skeleton.Leaves) {
			sorted = append(sorted, leafIdx)
		}
	}
	if len(sorted) == 0 {
		return nil
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	ranges := make([]leafIDRange, 0, len(sorted))
	for _, leafIdx := range sorted {
		leaf := sc.Skeleton.Leaves[leafIdx]
		next := leafIDRange{min: leaf.IDMin, max: leaf.IDMax}
		if len(ranges) == 0 {
			ranges = append(ranges, next)
			continue
		}
		last := &ranges[len(ranges)-1]
		if next.min <= last.max || (last.max < ^uint32(0) && next.min == last.max+1) {
			if next.max > last.max {
				last.max = next.max
			}
			continue
		}
		ranges = append(ranges, next)
	}
	return ranges
}

func leafRangePredicate(ranges []leafIDRange) (string, []any) {
	parts := make([]string, len(ranges))
	args := make([]any, 0, len(ranges)*2)
	for i, r := range ranges {
		parts[i] = "id BETWEEN ? AND ?"
		args = append(args, r.min, r.max)
	}
	return strings.Join(parts, " OR "), args
}

// GetClustersCH is the CH-backed equivalent of GetClusters. Routes by zoom:
// zoom < ZSplit → queryRollup; zoom >= ZSplit → queryTree.
func (sc *Supercluster) GetClustersCH(ctx context.Context, bounds KDBounds, zoom int) ([]ClusterNode, error) {
	if zoom < sc.Options.ZSplit {
		return sc.queryRollup(ctx, bounds, zoom)
	}
	return sc.queryTree(ctx, bounds, zoom)
}
