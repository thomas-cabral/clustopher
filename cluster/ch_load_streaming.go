package cluster

import (
	"context"
	"fmt"
	"sync"

	"github.com/ClickHouse/clickhouse-go/v2"
)

// chFirstStreamingIDMapBatch is the number of (external,internal) pairs queued
// per ID-map insert batch. Smaller than idMapBatchSize because we are also
// driving the reader thread concurrently, so we want batches to flush at a
// steady rate rather than buffer for very long.
const chFirstStreamingIDMapBatch = 100_000

// LoadFromCHStreaming is a memory-bounded alternative to LoadFromCHStaging
// that pushes the spatial sort into ClickHouse via a Morton-encoded ORDER BY,
// streams the sorted rows back, builds skeleton leaves on the fly in
// NodeSize-row chunks, and concurrently writes the (external,internal) id map
// to ClickHouse.
//
// Compared with LoadFromCHStaging this skips materializing the full
// []chSpatialRow and []KDPoint slices in Go, dropping peak Go heap from
// ~14 GB at 300M to a few hundred MB even at 1B+ points. Tradeoff: Morton
// ordering is a 2D space-filling-curve approximation of the exact KD-tree
// median partition that LoadFromCHStaging produces, so leaf bounds may
// overlap slightly more and high-zoom RangeLeaves queries can visit a few
// extra leaves.
func (sc *Supercluster) LoadFromCHStreaming(ctx context.Context) error {
	if sc.ch == nil {
		return fmt.Errorf("LoadFromCHStreaming requires CH client")
	}
	if sc.clusterID == "" {
		return fmt.Errorf("LoadFromCHStreaming requires clusterID")
	}

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

	nodeSize := sc.Options.NodeSize
	if nodeSize < 1 {
		nodeSize = 1
	}

	// ID-map writer fan-out. Each worker holds its own CH connection so the
	// inserts run in parallel with the streaming reader.
	idMapCh := make(chan idMapEntry, chFirstStreamingIDMapBatch*idMapInsertWorkers)
	idMapErrCh := make(chan error, idMapInsertWorkers)
	idMapCtx, idMapCancel := context.WithCancel(ctx)
	defer idMapCancel()

	var idMapWg sync.WaitGroup
	for w := 0; w < idMapInsertWorkers; w++ {
		idMapWg.Add(1)
		go func(worker int) {
			defer idMapWg.Done()
			ch := sc.ch
			if worker > 0 {
				clone, err := sc.ch.Clone(idMapCtx)
				if err != nil {
					idMapErrCh <- fmt.Errorf("clone ch worker %d: %w", worker, err)
					idMapCancel()
					return
				}
				defer clone.Close()
				ch = clone
			}
			if err := sc.streamingIDMapWorker(idMapCtx, ch, loadID, idMapCh); err != nil {
				idMapErrCh <- err
				idMapCancel()
			}
		}(w)
	}

	// Query: stream rows sorted by Morton index of normalized lng/lat. Bounds
	// for normalization are fixed [-180,180] / [-90,90] so we don't need a
	// pre-pass — Morton spatial locality holds on any subset of the world.
	rows, err := sc.ch.Conn().Query(ctx, `
        SELECT external_id, x, y
        FROM clustopher.staging_points
        WHERE cluster_id = ?
        ORDER BY mortonEncode(
            toUInt32((x + 180.0) / 360.0 * 4294967295),
            toUInt32((y + 90.0)  / 180.0 * 4294967295)
        )
    `, sc.clusterID)
	if err != nil {
		idMapCancel()
		idMapWg.Wait()
		return fmt.Errorf("stream staging by morton: %w", err)
	}

	leaves := make([]SkeletonLeaf, 0, 1024)
	chunk := make([]projectedRow, 0, nodeSize)
	var internalID uint32 = 1

	flushLeaf := func() {
		if len(chunk) == 0 {
			return
		}
		b := KDBounds{
			MinX: chunk[0].px, MaxX: chunk[0].px,
			MinY: chunk[0].py, MaxY: chunk[0].py,
		}
		for _, p := range chunk[1:] {
			if p.px < b.MinX {
				b.MinX = p.px
			}
			if p.px > b.MaxX {
				b.MaxX = p.px
			}
			if p.py < b.MinY {
				b.MinY = p.py
			}
			if p.py > b.MaxY {
				b.MaxY = p.py
			}
		}
		leaves = append(leaves, SkeletonLeaf{
			Bounds: b,
			IDMin:  chunk[0].internalID,
			IDMax:  chunk[len(chunk)-1].internalID,
			Count:  uint32(len(chunk)),
		})
		chunk = chunk[:0]
	}

	maxZoom := sc.Options.MaxZoom
	var (
		extID uint32
		x, y  float32
	)
	for rows.Next() {
		if err := rows.Scan(&extID, &x, &y); err != nil {
			rows.Close()
			idMapCancel()
			idMapWg.Wait()
			return fmt.Errorf("scan streaming row: %w", err)
		}
		proj := sc.projectFast(x, y, maxZoom)
		chunk = append(chunk, projectedRow{
			internalID: internalID,
			px:         proj[0],
			py:         proj[1],
		})

		select {
		case idMapCh <- idMapEntry{externalID: extID, internalID: internalID}:
		case <-idMapCtx.Done():
			rows.Close()
			idMapWg.Wait()
			if err := idMapCtx.Err(); err != nil && err != context.Canceled {
				return err
			}
			// Drain worker error channel, if any.
			close(idMapErrCh)
			for werr := range idMapErrCh {
				if werr != nil {
					return werr
				}
			}
			return fmt.Errorf("id-map writers exited before stream completed")
		}

		internalID++
		if len(chunk) == nodeSize {
			flushLeaf()
		}
	}
	rows.Close()
	flushLeaf()

	close(idMapCh)
	idMapWg.Wait()
	close(idMapErrCh)
	for werr := range idMapErrCh {
		if werr != nil {
			return werr
		}
	}

	if err := rows.Err(); err != nil {
		return fmt.Errorf("streaming rows: %w", err)
	}

	sc.Skeleton = BuildSkeletonFromLeaves(leaves)

	if err := sc.populatePointsFromStaging(ctx, loadID); err != nil {
		return err
	}
	return nil
}

type projectedRow struct {
	internalID uint32
	px         float32
	py         float32
}

type idMapEntry struct {
	externalID uint32
	internalID uint32
}

// streamingIDMapWorker drains the entry channel, batching inserts to
// clustopher.point_id_map_load. Exits cleanly when the channel closes.
func (sc *Supercluster) streamingIDMapWorker(ctx context.Context, ch *CHClient, loadID uint64, in <-chan idMapEntry) error {
	insertCtx := clickhouse.Context(ctx, clickhouse.WithSettings(insertSettings))
	flush := func(buf []idMapEntry) error {
		if len(buf) == 0 {
			return nil
		}
		batch, err := ch.Conn().PrepareBatch(insertCtx,
			"INSERT INTO clustopher.point_id_map_load (load_id, external_id, internal_id)")
		if err != nil {
			return fmt.Errorf("prepare id map batch: %w", err)
		}
		for _, e := range buf {
			if err := batch.Append(loadID, e.externalID, e.internalID); err != nil {
				return fmt.Errorf("append id map row: %w", err)
			}
		}
		if err := batch.Send(); err != nil {
			return fmt.Errorf("send id map batch: %w", err)
		}
		return nil
	}

	buf := make([]idMapEntry, 0, chFirstStreamingIDMapBatch)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case e, ok := <-in:
			if !ok {
				return flush(buf)
			}
			buf = append(buf, e)
			if len(buf) >= chFirstStreamingIDMapBatch {
				if err := flush(buf); err != nil {
					return err
				}
				buf = buf[:0]
			}
		}
	}
}

