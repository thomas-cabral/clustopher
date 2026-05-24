package cluster

import (
	"context"
	"fmt"
	"os"
	"strconv"

	"github.com/ClickHouse/clickhouse-go/v2"
)

// rollupPopulateMode reports whether rollups should be deferred and batch-
// populated after the canonical points table is fully loaded, instead of
// being maintained incrementally by the rollup materialized views during the
// bulk insert. Enabled by default (CLUSTOPHER_DEFERRED_ROLLUPS != "0").
//
// Deferred populate skips per-insert MV fan-out (currently 9 MVs at
// MaxRollupZoom=10) during the staging->points populate, which is by far the
// largest cost of a bulk load at >100M points, and replaces it with one
// pre-aggregated GROUP BY INSERT per zoom run after the points table is
// finalized. That writes <2M rollup rows total at 1B points instead of
// ~9B partial MV write events.
func rollupPopulateDeferred() bool {
	return os.Getenv("CLUSTOPHER_DEFERRED_ROLLUPS") != "0"
}

// detachRollupMVs removes the materialized-view triggers on clustopher.points
// for the lifetime of a bulk load so inserts don't fan out to all
// MaxRollupZoom-MinRollupZoom+1 rollup tables. The underlying rollup_z* data
// tables are unaffected. Call attachRollupMVs after the bulk insert to
// restore incremental maintenance for future inserts.
//
// Note: DETACH is global — concurrent inserts from other clusters during this
// window will also bypass MV maintenance. Loads are serialized in the runner,
// so this is acceptable for our workload.
func (sc *Supercluster) detachRollupMVs(ctx context.Context) error {
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		stmt := "DETACH TABLE clustopher.mv_rollup_z" + strconv.Itoa(z)
		if err := sc.ch.Conn().Exec(ctx, stmt); err != nil {
			return fmt.Errorf("detach mv_rollup_z%d: %w", z, err)
		}
	}
	return nil
}

// attachRollupMVs restores the MV triggers detached by detachRollupMVs. Safe
// to call multiple times; ATTACH on an already-attached table is a no-op.
func (sc *Supercluster) attachRollupMVs(ctx context.Context) error {
	var firstErr error
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		stmt := "ATTACH TABLE clustopher.mv_rollup_z" + strconv.Itoa(z)
		if err := sc.ch.Conn().Exec(ctx, stmt); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("attach mv_rollup_z%d: %w", z, err)
		}
	}
	return firstErr
}

// populateRollupsBatch runs one pre-aggregated GROUP BY INSERT per rollup
// zoom, populating clustopher.rollup_z{N} for the current cluster from the
// already-populated clustopher.points partition. Mirrors the per-row math
// the MV would have done, but pre-aggregates so the SummingMergeTree gets
// one row per (cluster_id, tile_x, tile_y) instead of one per source point.
//
// At 1B points this writes a total of ~1.5M rollup rows across all 9 zooms
// (tile counts cap at 4^z which is much smaller than the point count for the
// MaxRollupZoom we keep). The MV path would have written ~9B partial rows
// then merged them. Same end state, vastly less write amplification.
func (sc *Supercluster) populateRollupsBatch(ctx context.Context) error {
	radius := DefaultRollupRadius
	settings := clickhouse.Settings{}
	for k, v := range insertSettings {
		settings[k] = v
	}
	insertCtx := clickhouse.Context(ctx, clickhouse.WithSettings(settings))

	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		q := fmt.Sprintf(`
            INSERT INTO clustopher.rollup_z%d (cluster_id, tile_x, tile_y, cnt, sum_x, sum_y, metric_sums, metric_cnts)
            SELECT
                cluster_id,
                toUInt32(((x + 180) / 360) * pow(2, %d) * 512 / %d) AS tile_x,
                toUInt32(((1 - log(tan(y * pi()/180) + 1/cos(y * pi()/180)) / pi()) / 2) * pow(2, %d) * 512 / %d) AS tile_y,
                count() AS cnt,
                sum(x) AS sum_x,
                sum(y) AS sum_y,
                sumMap(mapApply((k, v) -> (k, toFloat64(v)), metrics)) AS metric_sums,
                sumMap(mapApply((k, v) -> (k, toUInt64(1)),  metrics)) AS metric_cnts
            FROM clustopher.points
            WHERE cluster_id = ?
            GROUP BY cluster_id, tile_x, tile_y
        `, z, z, radius, z, radius)
		if err := sc.ch.Conn().Exec(insertCtx, q, sc.clusterID); err != nil {
			return fmt.Errorf("batch populate rollup_z%d: %w", z, err)
		}
	}
	return nil
}
