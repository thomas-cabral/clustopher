package cluster

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"
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

// rollupCascadeEnabled reports whether batch rollup populate should derive
// each zoom from the zoom above it instead of rescanning the full points
// partition per zoom. Enabled by default (CLUSTOPHER_ROLLUP_CASCADE != "0").
func rollupCascadeEnabled() bool {
	return os.Getenv("CLUSTOPHER_ROLLUP_CASCADE") != "0"
}

// populateRollupsBatch populates clustopher.rollup_z{N} for the current
// cluster from the already-populated clustopher.points partition, for all
// zooms in [MinRollupZoom, MaxRollupZoom]. Mirrors the per-row math the MV
// would have done, but pre-aggregates so the SummingMergeTree gets one row
// per (cluster_id, tile_x, tile_y) instead of one per source point.
//
// At 1B points this writes a total of ~1.5M rollup rows across all 9 zooms
// (tile counts cap at 4^z which is much smaller than the point count for the
// MaxRollupZoom we keep). The MV path would have written ~9B partial rows
// then merged them. Same end state, vastly less write amplification.
func (sc *Supercluster) populateRollupsBatch(ctx context.Context) error {
	if rollupCascadeEnabled() {
		return sc.populateRollupsCascade(ctx)
	}
	return sc.populateRollupsDirect(ctx)
}

// rollupFromPointsSQL builds the INSERT that aggregates the points partition
// directly into rollup_z{z}. Shared by the direct (every zoom) and cascade
// (top zoom only) populate strategies.
func rollupFromPointsSQL(z int) string {
	radius := DefaultRollupRadius
	return fmt.Sprintf(`
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
}

// populateRollupsDirect rescans the full points partition once per zoom.
// Kept as the CLUSTOPHER_ROLLUP_CASCADE=0 fallback and as the reference
// implementation the cascade is equivalence-tested against.
func (sc *Supercluster) populateRollupsDirect(ctx context.Context) error {
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		insertCtx := chQueryCtx(ctx, "rollup-z"+strconv.Itoa(z), insertSettings)
		zt := time.Now()
		if err := sc.ch.Conn().Exec(insertCtx, rollupFromPointsSQL(z), sc.clusterID); err != nil {
			return fmt.Errorf("batch populate rollup_z%d: %w", z, err)
		}
		log.Printf("[rollup] z%d populate in %s", z, time.Since(zt).Round(time.Millisecond))
	}
	return nil
}

// populateRollupsCascade scans the points partition once (for MaxRollupZoom),
// then derives each lower zoom from the zoom above it. Exact because
// tile coords at z-1 are intDiv(tile coords at z, 2):
//
//	tile(z) = floor(w * 2^z * 512/radius) and the float expression for z-1
//	is exactly half the one for z (scaling by a power of two is exact in
//	IEEE-754), so floor(floor(2v)/2) == floor(v) applies.
//
// All rollup columns are additive (cnt, sum_x, sum_y, sumMap maps), so
// re-aggregating z's tiles under z-1's coords yields the same result as
// aggregating the source points — up to float summation order, which is
// already unspecified in the direct path (parallel aggregation).
//
// The GROUP BY also makes reading the SummingMergeTree source safe while its
// parts are unmerged: partial rows for the same tile sum into one output row.
//
// At 3B points this replaces 9 full scans (~71 s each) with one full scan
// plus 8 scans of <3M rollup rows.
func (sc *Supercluster) populateRollupsCascade(ctx context.Context) error {
	top := MaxRollupZoom
	insertCtx := chQueryCtx(ctx, "rollup-z"+strconv.Itoa(top), insertSettings)
	zt := time.Now()
	if err := sc.ch.Conn().Exec(insertCtx, rollupFromPointsSQL(top), sc.clusterID); err != nil {
		return fmt.Errorf("cascade populate rollup_z%d: %w", top, err)
	}
	log.Printf("[rollup] z%d populate (from points) in %s", top, time.Since(zt).Round(time.Millisecond))

	for z := top - 1; z >= MinRollupZoom; z-- {
		insertCtx := chQueryCtx(ctx, "rollup-z"+strconv.Itoa(z), insertSettings)
		zt := time.Now()
		q := fmt.Sprintf(`
            INSERT INTO clustopher.rollup_z%d (cluster_id, tile_x, tile_y, cnt, sum_x, sum_y, metric_sums, metric_cnts)
            SELECT
                cluster_id,
                intDiv(tile_x, 2) AS tx,
                intDiv(tile_y, 2) AS ty,
                sum(cnt) AS cnt,
                sum(sum_x) AS sum_x,
                sum(sum_y) AS sum_y,
                sumMap(metric_sums) AS metric_sums,
                sumMap(metric_cnts) AS metric_cnts
            FROM clustopher.rollup_z%d
            WHERE cluster_id = ?
            GROUP BY cluster_id, tx, ty
        `, z, z+1)
		if err := sc.ch.Conn().Exec(insertCtx, q, sc.clusterID); err != nil {
			return fmt.Errorf("cascade populate rollup_z%d: %w", z, err)
		}
		log.Printf("[rollup] z%d populate (from z%d) in %s", z, z+1, time.Since(zt).Round(time.Millisecond))
	}
	return nil
}
