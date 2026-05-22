package cluster

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"runtime"
	"strconv"
	"sync"

	"github.com/ClickHouse/clickhouse-go/v2"
)

const (
	chFirstProjectChunkSize = 250_000
	idMapBatchSize          = 500_000
	idMapInsertWorkers      = 4
)

type chSpatialRow struct {
	ExternalID uint32
	X          float32
	Y          float32
}

// LoadFromCHStaging builds the exact same KD leaf order as Load, but reads only
// compact spatial columns from clustopher.staging_points. It writes the
// external->internal id map back to ClickHouse, then lets ClickHouse populate
// clustopher.points from staging so metrics/metadata never enter Go's hot
// clustering path.
func (sc *Supercluster) LoadFromCHStaging(ctx context.Context) error {
	if sc.ch == nil {
		return fmt.Errorf("LoadFromCHStaging requires CH client")
	}
	if sc.clusterID == "" {
		return fmt.Errorf("LoadFromCHStaging requires clusterID")
	}

	spatial, err := sc.readStagingSpatial(ctx)
	if err != nil {
		return err
	}

	projected := sc.projectStagingSpatial(spatial)

	sorted := SortPointsIntoLeafOrderParallelInPlace(projected, sc.Options.NodeSize)
	projected = nil
	tree, remap := BuildSkeletonWithRemap(sorted, sc.Options.NodeSize)
	sc.Skeleton = tree

	loadID, err := newLoadID()
	if err != nil {
		return err
	}
	defer func() {
		_ = sc.dropPointIDMapLoad(context.Background(), loadID)
	}()

	if err := sc.resetCanonicalCluster(ctx); err != nil {
		return err
	}
	if err := sc.writePointIDMap(ctx, loadID, sorted, remap, spatial); err != nil {
		return err
	}
	if err := sc.populatePointsFromStaging(ctx, loadID); err != nil {
		return err
	}
	return nil
}

func (sc *Supercluster) projectStagingSpatial(spatial []chSpatialRow) []KDPoint {
	projected := make([]KDPoint, len(spatial))
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || len(spatial) < chFirstProjectChunkSize {
		for i, p := range spatial {
			proj := sc.projectFast(p.X, p.Y, sc.Options.MaxZoom)
			projected[i] = KDPoint{ID: uint32(i), X: proj[0], Y: proj[1], NumPoints: 1}
		}
		return projected
	}

	chunks := (len(spatial) + chFirstProjectChunkSize - 1) / chFirstProjectChunkSize
	if workers > chunks {
		workers = chunks
	}

	var wg sync.WaitGroup
	jobs := make(chan int, chunks)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for start := range jobs {
				end := start + chFirstProjectChunkSize
				if end > len(spatial) {
					end = len(spatial)
				}
				for i := start; i < end; i++ {
					p := spatial[i]
					proj := sc.projectFast(p.X, p.Y, sc.Options.MaxZoom)
					projected[i] = KDPoint{ID: uint32(i), X: proj[0], Y: proj[1], NumPoints: 1}
				}
			}
		}()
	}
	for start := 0; start < len(spatial); start += chFirstProjectChunkSize {
		jobs <- start
	}
	close(jobs)
	wg.Wait()
	return projected
}

func (sc *Supercluster) readStagingSpatial(ctx context.Context) ([]chSpatialRow, error) {
	var n uint64
	if err := sc.ch.Conn().QueryRow(ctx,
		"SELECT count() FROM clustopher.staging_points WHERE cluster_id = ?", sc.clusterID).Scan(&n); err != nil {
		return nil, fmt.Errorf("count staging spatial: %w", err)
	}
	if n > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("staging spatial count %d exceeds max int", n)
	}

	rows, err := sc.ch.Conn().Query(ctx, `
        SELECT external_id, x, y
        FROM clustopher.staging_points
        WHERE cluster_id = ?
        ORDER BY external_id
    `, sc.clusterID)
	if err != nil {
		return nil, fmt.Errorf("query staging spatial: %w", err)
	}
	defer rows.Close()

	out := make([]chSpatialRow, 0, int(n))
	for rows.Next() {
		var row chSpatialRow
		if err := rows.Scan(&row.ExternalID, &row.X, &row.Y); err != nil {
			return nil, fmt.Errorf("scan staging spatial: %w", err)
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("staging spatial rows: %w", err)
	}
	return out, nil
}

func (sc *Supercluster) resetCanonicalCluster(ctx context.Context) error {
	if err := sc.ch.Conn().Exec(ctx, "ALTER TABLE clustopher.points DROP PARTITION ?", sc.clusterID); err != nil {
		return fmt.Errorf("drop points partition: %w", err)
	}
	for z := MinRollupZoom; z <= MaxRollupZoom; z++ {
		if err := sc.ch.Conn().Exec(ctx, "ALTER TABLE clustopher.rollup_z"+strconv.Itoa(z)+" DROP PARTITION ?", sc.clusterID); err != nil {
			return fmt.Errorf("drop rollup_z%d partition: %w", z, err)
		}
	}
	return nil
}

func (sc *Supercluster) dropPointIDMapLoad(ctx context.Context, loadID uint64) error {
	if err := sc.ch.Conn().Exec(ctx, "ALTER TABLE clustopher.point_id_map_load DROP PARTITION ?", loadID); err != nil {
		return fmt.Errorf("drop id map load partition: %w", err)
	}
	return nil
}

func (sc *Supercluster) writePointIDMap(ctx context.Context, loadID uint64, sorted []KDPoint, remap []uint32, spatial []chSpatialRow) error {
	chunks := (len(sorted) + idMapBatchSize - 1) / idMapBatchSize
	workers := idMapInsertWorkers
	if workers > chunks {
		workers = chunks
	}
	if workers < 1 {
		return nil
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	jobs := make(chan int, chunks)
	errCh := make(chan error, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			ch := sc.ch
			if worker > 0 {
				clone, err := sc.ch.Clone(ctx)
				if err != nil {
					errCh <- err
					cancel()
					return
				}
				defer clone.Close()
				ch = clone
			}
			for start := range jobs {
				if err := sc.writePointIDMapBatch(ctx, ch, loadID, sorted, remap, spatial, start); err != nil {
					errCh <- err
					cancel()
					return
				}
			}
		}(worker)
	}

sendJobs:
	for start := 0; start < len(sorted); start += idMapBatchSize {
		select {
		case jobs <- start:
		case <-ctx.Done():
			break sendJobs
		}
	}
	close(jobs)
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	if err := ctx.Err(); err != nil && err != context.Canceled {
		return err
	}
	return nil
}

func (sc *Supercluster) writePointIDMapBatch(ctx context.Context, ch *CHClient, loadID uint64, sorted []KDPoint, remap []uint32, spatial []chSpatialRow, start int) error {
	ctx = clickhouse.Context(ctx, clickhouse.WithSettings(insertSettings))
	end := start + idMapBatchSize
	if end > len(sorted) {
		end = len(sorted)
	}
	batch, err := ch.Conn().PrepareBatch(ctx,
		"INSERT INTO clustopher.point_id_map_load (load_id, external_id, internal_id)")
	if err != nil {
		return fmt.Errorf("prepare id map batch: %w", err)
	}
	for i := start; i < end; i++ {
		origIdx := remap[i]
		if int(origIdx) >= len(spatial) {
			return fmt.Errorf("remap index %d out of range %d", origIdx, len(spatial))
		}
		if err := batch.Append(loadID, spatial[origIdx].ExternalID, sorted[i].ID); err != nil {
			return fmt.Errorf("append id map row %d: %w", i, err)
		}
	}
	if err := batch.Send(); err != nil {
		return fmt.Errorf("send id map batch: %w", err)
	}
	return nil
}

func (sc *Supercluster) populatePointsFromStaging(ctx context.Context, loadID uint64) error {
	// Merge insertSettings with merge-join settings so the staging x id_map join
	// streams through sort-merge instead of building a full hash table on one
	// side. At >100M rows a hash join needs 10-20 GB of CH server RAM; merge
	// join keeps the working set bounded to a few hundred MB.
	settings := clickhouse.Settings{}
	for k, v := range insertSettings {
		settings[k] = v
	}
	settings["join_algorithm"] = "full_sorting_merge"
	settings["max_bytes_before_external_sort"] = uint64(8 * 1024 * 1024 * 1024)
	settings["max_bytes_before_external_group_by"] = uint64(8 * 1024 * 1024 * 1024)

	ctx = clickhouse.Context(ctx, clickhouse.WithSettings(settings))
	// No ORDER BY internal_id in the SELECT: MergeTree sorts blocks by the
	// table's ORDER BY (cluster_id, id) on insert anyway. Forcing a full
	// re-sort of the join output costs an extra N-row sort buffer on the CH
	// server (~70 GB peak at 500M, blows past system RAM at 1B+). Without it
	// CH inserts in join-stream order and merges into the final sorted parts
	// in the background.
	if err := sc.ch.Conn().Exec(ctx, `
        INSERT INTO clustopher.points (cluster_id, id, external_id, x, y, metrics, metadata)
        SELECT
            s.cluster_id,
            m.internal_id AS id,
            s.external_id,
            s.x,
            s.y,
            s.metrics,
            s.metadata
        FROM clustopher.staging_points AS s
        INNER JOIN clustopher.point_id_map_load AS m
            ON s.external_id = m.external_id
        WHERE s.cluster_id = ?
          AND m.load_id = ?
    `, sc.clusterID, loadID); err != nil {
		return fmt.Errorf("populate points from staging: %w", err)
	}
	return nil
}

func newLoadID() (uint64, error) {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return 0, fmt.Errorf("generate load id: %w", err)
	}
	id := binary.LittleEndian.Uint64(buf[:])
	if id == 0 {
		id = 1
	}
	return id, nil
}
