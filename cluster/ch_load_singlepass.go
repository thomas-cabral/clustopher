package cluster

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
)

// LoadFromCHSinglePass writes the canonical points table directly from staging
// using CH-side morton sort + rowNumberInAllBlocks() to assign internal IDs,
// then reads leaf bounds back via GROUP BY for skeleton construction.
//
// Compared with LoadFromCHStreaming this:
//   - Skips the point_id_map_load table + JOIN entirely.
//   - Skips streaming all 3B rows through Go and the parallel id-map writers.
//   - Replaces with one big INSERT...SELECT + a small GROUP BY scan.
//
// Tradeoff: rowNumberInAllBlocks() forces single-threaded final merge in CH,
// so the INSERT is bottlenecked on sort+write throughput. Leaf bounds rely on
// web-mercator monotonicity: projecting the (min_lng, max_lat) and
// (max_lng, min_lat) corners yields exact pixel-space bounds because the
// projection is monotonic in each axis within the supported latitude range.
func (sc *Supercluster) LoadFromCHSinglePass(ctx context.Context) error {
	if sc.ch == nil {
		return fmt.Errorf("LoadFromCHSinglePass requires CH client")
	}
	if sc.clusterID == "" {
		return fmt.Errorf("LoadFromCHSinglePass requires clusterID")
	}

	if err := sc.resetCanonicalCluster(ctx); err != nil {
		return err
	}

	deferred := rollupPopulateDeferred()
	if deferred {
		t := time.Now()
		if err := sc.detachRollupMVs(ctx); err != nil {
			return err
		}
		log.Printf("[ch-single] detached rollup MVs in %s", time.Since(t).Round(time.Second))
		defer func() {
			_ = sc.attachRollupMVs(context.Background())
		}()
	}

	insertSettingsBuilt := clickhouse.Settings{}
	for k, v := range insertSettings {
		insertSettingsBuilt[k] = v
	}
	// At 3B with metrics+metadata Map columns materialized through the sort
	// pipeline, total carry exceeds 200 GB if held in RAM. Need aggressive
	// spilling: lower per-thread sort threshold to 1 GB so spill activates
	// early, drop max_threads to 10 to bound per-thread read/decode buffers,
	// raise max_memory_usage to 80 GB hard cap (host has 94 GB).
	insertSettingsBuilt["max_memory_usage"] = uint64(80 * 1024 * 1024 * 1024)
	insertSettingsBuilt["max_threads"] = uint64(10)
	insertSettingsBuilt["max_bytes_before_external_sort"] = uint64(1 * 1024 * 1024 * 1024)
	insertSettingsBuilt["max_bytes_before_external_group_by"] = uint64(1 * 1024 * 1024 * 1024)

	insertCtx := chQueryCtx(ctx, "sp-insert", insertSettingsBuilt)
	t := time.Now()
	log.Printf("[ch-single] morton+rowNumber INSERT begin")
	if err := sc.ch.Conn().Exec(insertCtx, `
        INSERT INTO clustopher.points (cluster_id, id, external_id, x, y, metrics, metadata)
        SELECT
            cluster_id,
            toUInt32(rowNumberInAllBlocks() + 1) AS id,
            external_id,
            x,
            y,
            metrics,
            metadata
        FROM clustopher.staging_points
        WHERE cluster_id = ?
        ORDER BY mortonEncode(
            toUInt32((x + 180.0) / 360.0 * 4294967295),
            toUInt32((y + 90.0)  / 180.0 * 4294967295)
        )
    `, sc.clusterID); err != nil {
		return fmt.Errorf("single-pass insert: %w", err)
	}
	log.Printf("[ch-single] INSERT done in %s", time.Since(t).Round(time.Second))

	t = time.Now()
	leaves, err := sc.readLeafBoundsFromPoints(ctx)
	if err != nil {
		return err
	}
	log.Printf("[ch-single] read %d leaf bounds in %s", len(leaves), time.Since(t).Round(time.Second))

	t = time.Now()
	sc.Skeleton = BuildSkeletonFromLeaves(leaves)
	log.Printf("[ch-single] skeleton built in %s", time.Since(t).Round(time.Second))

	if deferred {
		t = time.Now()
		if err := sc.populateRollupsBatch(ctx); err != nil {
			return err
		}
		log.Printf("[ch-single] rollup batch populate in %s", time.Since(t).Round(time.Second))
	}
	return nil
}

// readLeafBoundsFromPoints groups the newly inserted points into NodeSize-row
// chunks (by id), aggregates min/max lng+lat and id range per chunk, and
// projects to pixel-space SkeletonLeaf entries.
func (sc *Supercluster) readLeafBoundsFromPoints(ctx context.Context) ([]SkeletonLeaf, error) {
	nodeSize := sc.Options.NodeSize
	if nodeSize < 1 {
		nodeSize = 1
	}
	maxZoom := sc.Options.MaxZoom

	queryCtx := chQueryCtx(ctx, "sp-leafbounds", clickhouse.Settings{
		"max_memory_usage":                   uint64(20 * 1024 * 1024 * 1024),
		"max_threads":                        uint64(20),
		"max_bytes_before_external_group_by": uint64(4 * 1024 * 1024 * 1024),
	})
	rows, err := sc.ch.Conn().Query(queryCtx, `
        SELECT
            intDiv(id - 1, ?) AS leaf,
            min(x) AS min_lng,
            max(x) AS max_lng,
            min(y) AS min_lat,
            max(y) AS max_lat,
            min(id) AS id_min,
            max(id) AS id_max,
            count() AS cnt
        FROM clustopher.points
        WHERE cluster_id = ?
        GROUP BY leaf
        ORDER BY leaf
    `, uint32(nodeSize), sc.clusterID)
	if err != nil {
		return nil, fmt.Errorf("read leaf bounds: %w", err)
	}
	defer rows.Close()

	var leaves []SkeletonLeaf
	var (
		leafID                         int64
		minLng, maxLng, minLat, maxLat float32
		idMin, idMax                   uint32
		cnt                            uint64
	)
	for rows.Next() {
		if err := rows.Scan(&leafID, &minLng, &maxLng, &minLat, &maxLat, &idMin, &idMax, &cnt); err != nil {
			return nil, fmt.Errorf("scan leaf row: %w", err)
		}
		_ = leafID // ordering preserved by ORDER BY leaf in SQL
		topLeft := sc.projectFast(minLng, maxLat, maxZoom)
		botRight := sc.projectFast(maxLng, minLat, maxZoom)
		leaves = append(leaves, SkeletonLeaf{
			Bounds: KDBounds{
				MinX: topLeft[0],
				MinY: topLeft[1],
				MaxX: botRight[0],
				MaxY: botRight[1],
			},
			IDMin: idMin,
			IDMax: idMax,
			Count: uint32(cnt),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("leaf bounds rows: %w", err)
	}
	return leaves, nil
}
