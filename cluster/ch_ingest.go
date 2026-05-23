package cluster

import (
	"context"
	"fmt"

	"github.com/ClickHouse/clickhouse-go/v2"
)

// insertSettings tunes the CH server for bulk ingest into clustopher.points.
// parallel_view_processing fans the 16 rollup MVs out in parallel rather than
// sequentially per insert block. The larger block sizes coalesce small inserts
// server-side so SummingMergeTree creates fewer parts to merge.
var insertSettings = clickhouse.Settings{
	"parallel_view_processing":   uint64(1),
	"max_insert_block_size":      uint64(1_048_576),
	"min_insert_block_size_rows": uint64(1_048_576),
}

// CHPointRow is the row-shape used to insert into clustopher.points.
type CHPointRow struct {
	ClusterID  string
	ID         uint32
	ExternalID uint32
	X          float32
	Y          float32
	Metrics    map[string]float32
	Metadata   map[string]string
}

type CHStagingPointRow struct {
	ClusterID  string
	ExternalID uint32
	X          float32
	Y          float32
	Metrics    map[string]float32
	Metadata   map[string]string
}

type CHPointIDMapRow struct {
	LoadID     uint64
	ExternalID uint32
	InternalID uint32
}

// InsertPoints batches rows into clustopher.points using the native protocol.
// Rows should already be in (cluster_id, id) primary-key order to minimize
// MergeTree merges.
func (c *CHClient) InsertPoints(ctx context.Context, rows []CHPointRow) error {
	ctx = clickhouse.Context(ctx, clickhouse.WithSettings(insertSettings))
	batch, err := c.conn.PrepareBatch(ctx,
		"INSERT INTO clustopher.points (cluster_id, id, external_id, x, y, metrics, metadata)")
	if err != nil {
		return fmt.Errorf("prepare batch: %w", err)
	}
	for i := range rows {
		r := &rows[i]
		if err := batch.Append(r.ClusterID, r.ID, r.ExternalID, r.X, r.Y, r.Metrics, r.Metadata); err != nil {
			return fmt.Errorf("append row %d: %w", i, err)
		}
	}
	if err := batch.Send(); err != nil {
		return fmt.Errorf("send batch: %w", err)
	}
	return nil
}

func (c *CHClient) InsertStagingPoints(ctx context.Context, rows []CHStagingPointRow) error {
	ctx = clickhouse.Context(ctx, clickhouse.WithSettings(insertSettings))
	batch, err := c.conn.PrepareBatch(ctx,
		"INSERT INTO clustopher.staging_points (cluster_id, external_id, x, y, metrics, metadata)")
	if err != nil {
		return fmt.Errorf("prepare staging batch: %w", err)
	}
	for i := range rows {
		r := &rows[i]
		if err := batch.Append(r.ClusterID, r.ExternalID, r.X, r.Y, r.Metrics, r.Metadata); err != nil {
			return fmt.Errorf("append staging row %d: %w", i, err)
		}
	}
	if err := batch.Send(); err != nil {
		return fmt.Errorf("send staging batch: %w", err)
	}
	return nil
}

func (c *CHClient) InsertPointIDMap(ctx context.Context, rows []CHPointIDMapRow) error {
	ctx = clickhouse.Context(ctx, clickhouse.WithSettings(insertSettings))
	batch, err := c.conn.PrepareBatch(ctx,
		"INSERT INTO clustopher.point_id_map_load (load_id, external_id, internal_id)")
	if err != nil {
		return fmt.Errorf("prepare id map batch: %w", err)
	}
	for i := range rows {
		r := &rows[i]
		if err := batch.Append(r.LoadID, r.ExternalID, r.InternalID); err != nil {
			return fmt.Errorf("append id map row %d: %w", i, err)
		}
	}
	if err := batch.Send(); err != nil {
		return fmt.Errorf("send id map batch: %w", err)
	}
	return nil
}
