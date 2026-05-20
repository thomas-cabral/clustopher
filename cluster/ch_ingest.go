package cluster

import (
	"context"
	"fmt"
)

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

// InsertPoints batches rows into clustopher.points using the native protocol.
// Rows should already be in (cluster_id, id) primary-key order to minimize
// MergeTree merges.
func (c *CHClient) InsertPoints(ctx context.Context, rows []CHPointRow) error {
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
