package cluster

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
)

// mortonExprSQL is the Z-order key used for the global spatial sort. Shared
// by the single INSERT path and the partitioned path so both produce the
// identical ordering.
const mortonExprSQL = `mortonEncode(
            toUInt32((x + 180.0) / 360.0 * 4294967295),
            toUInt32((y + 90.0)  / 180.0 * 4294967295)
        )`

// singlePassPartitions reports how many morton-range buckets the single-pass
// INSERT should be split into (CLUSTOPHER_SINGLEPASS_PARTITIONS). 1 (default)
// keeps the original single global INSERT.
//
// Why split: rowNumberInAllBlocks() over a global ORDER BY serializes the
// final k-way merge, the squash and the MergeTree sink onto one thread —
// profiling at 100M attributes ~60-80s of a 103s INSERT to that tail. With
// k disjoint morton ranges, ids are assigned per bucket from precomputed
// offsets and the buckets run as independent sort→merge→sink pipelines.
func singlePassPartitions() int {
	n, err := strconv.Atoi(os.Getenv("CLUSTOPHER_SINGLEPASS_PARTITIONS"))
	if err != nil || n < 2 {
		return 1
	}
	if n > 16 {
		n = 16
	}
	return n
}

// singlePassConcurrency reports how many bucket INSERTs may run at once
// (CLUSTOPHER_SINGLEPASS_CONCURRENCY, default 2). Per-pipeline memory and
// thread caps are divided by this value so the aggregate stays inside the
// same budget as the single-INSERT path.
func singlePassConcurrency() int {
	n, err := strconv.Atoi(os.Getenv("CLUSTOPHER_SINGLEPASS_CONCURRENCY"))
	if err != nil || n < 1 {
		return 2
	}
	if n > 8 {
		n = 8
	}
	return n
}

// singlePassInsertPartitioned splits the staging partition into k disjoint
// morton-key ranges and runs one INSERT...SELECT per range, with internal ids
// offset by the exact row counts of the preceding ranges. Concatenated bucket
// order equals the global morton order, so the resulting (id → point)
// assignment is equivalent to the single global INSERT up to tie order of
// identical morton keys (already unspecified there).
func (sc *Supercluster) singlePassInsertPartitioned(ctx context.Context, k int, baseSettings clickhouse.Settings) error {
	// Approximate range boundaries: k-quantiles of the morton key. Precision
	// only affects bucket balance — ids stay exact because counts below are
	// computed against the same boundaries actually used by the INSERTs.
	probs := make([]string, 0, k-1)
	for i := 1; i < k; i++ {
		probs = append(probs, strconv.FormatFloat(float64(i)/float64(k), 'g', -1, 64))
	}
	var qs []float64
	qb := fmt.Sprintf(`SELECT quantilesTDigest(%s)(%s) FROM clustopher.staging_points WHERE cluster_id = ?`,
		strings.Join(probs, ", "), mortonExprSQL)
	if err := sc.ch.Conn().QueryRow(chQueryCtx(ctx, "sp-part-bounds", nil), qb, sc.clusterID).Scan(&qs); err != nil {
		return fmt.Errorf("morton quantiles: %w", err)
	}
	bounds := make([]uint64, 0, len(qs))
	for _, q := range qs {
		var b uint64
		switch {
		case q <= 0 || math.IsNaN(q):
			b = 0
		case q >= math.MaxUint64:
			b = math.MaxUint64
		default:
			b = uint64(q)
		}
		bounds = append(bounds, b)
	}
	sort.Slice(bounds, func(i, j int) bool { return bounds[i] < bounds[j] })
	// Collapse duplicate boundaries (degenerate key distributions) — empty
	// buckets are harmless but pointless.
	uniq := bounds[:0]
	for _, b := range bounds {
		if len(uniq) == 0 || uniq[len(uniq)-1] != b {
			uniq = append(uniq, b)
		}
	}
	bounds = uniq
	if len(bounds) == 0 {
		return fmt.Errorf("morton quantiles collapsed to zero boundaries (k=%d)", k)
	}

	// Bucket predicates over the morton key. Rows equal to a boundary always
	// land in the bucket starting at it, so ties never straddle buckets.
	conds := make([]string, 0, len(bounds)+1)
	for i := 0; i <= len(bounds); i++ {
		switch {
		case i == 0:
			conds = append(conds, fmt.Sprintf("%s < %d", mortonExprSQL, bounds[0]))
		case i == len(bounds):
			conds = append(conds, fmt.Sprintf("%s >= %d", mortonExprSQL, bounds[i-1]))
		default:
			conds = append(conds, fmt.Sprintf("%s >= %d AND %s < %d", mortonExprSQL, bounds[i-1], mortonExprSQL, bounds[i]))
		}
	}

	// Exact per-bucket counts → id offsets.
	countExprs := make([]string, len(conds))
	for i, c := range conds {
		countExprs[i] = "countIf(" + c + ")"
	}
	counts := make([]uint64, len(conds))
	dest := make([]any, len(conds))
	for i := range counts {
		dest[i] = &counts[i]
	}
	qc := "SELECT " + strings.Join(countExprs, ", ") + " FROM clustopher.staging_points WHERE cluster_id = ?"
	if err := sc.ch.Conn().QueryRow(chQueryCtx(ctx, "sp-part-counts", nil), qc, sc.clusterID).Scan(dest...); err != nil {
		return fmt.Errorf("bucket counts: %w", err)
	}
	offsets := make([]uint64, len(counts))
	var total uint64
	for i, c := range counts {
		offsets[i] = total
		total += c
	}
	if total > math.MaxUint32 {
		return fmt.Errorf("total rows %d exceeds uint32 id space", total)
	}

	// Per-pipeline caps: divide the single-INSERT budget across concurrent
	// pipelines so the aggregate matches the proven envelope.
	j := singlePassConcurrency()
	if j > len(conds) {
		j = len(conds)
	}
	pipeSettings := clickhouse.Settings{}
	for key, v := range baseSettings {
		pipeSettings[key] = v
	}
	for _, key := range []string{"max_memory_usage", "max_bytes_before_external_sort", "max_bytes_before_external_group_by"} {
		if v, ok := pipeSettings[key].(uint64); ok {
			pipeSettings[key] = v / uint64(j)
		}
	}
	if v, ok := pipeSettings["max_threads"].(uint64); ok {
		nt := v / uint64(j)
		if nt < 4 {
			nt = 4
		}
		pipeSettings["max_threads"] = nt
	}

	log.Printf("[ch-single] partitioned insert: %d buckets, %d concurrent, counts=%v", len(conds), j, counts)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	jobs := make(chan int, len(conds))
	errCh := make(chan error, j)
	var wg sync.WaitGroup
	for w := 0; w < j; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			ch := sc.ch
			if w > 0 {
				clone, err := sc.ch.Clone(ctx)
				if err != nil {
					errCh <- err
					cancel()
					return
				}
				defer clone.Close()
				ch = clone
			}
			for i := range jobs {
				if counts[i] == 0 {
					continue
				}
				bt := time.Now()
				q := fmt.Sprintf(`
        INSERT INTO clustopher.points (cluster_id, id, external_id, x, y, metrics, metadata)
        SELECT
            cluster_id,
            toUInt32(rowNumberInAllBlocks() + %d) AS id,
            external_id,
            x,
            y,
            metrics,
            metadata
        FROM clustopher.staging_points
        WHERE cluster_id = ? AND (%s)
        ORDER BY %s
    `, offsets[i]+1, conds[i], mortonExprSQL)
				insertCtx := chQueryCtx(ctx, "sp-insert-p"+strconv.Itoa(i), pipeSettings)
				if err := ch.Conn().Exec(insertCtx, q, sc.clusterID); err != nil {
					errCh <- fmt.Errorf("bucket %d insert: %w", i, err)
					cancel()
					return
				}
				log.Printf("[ch-single] bucket %d/%d: %d rows in %s", i+1, len(conds), counts[i], time.Since(bt).Round(time.Second))
			}
		}(w)
	}
sendJobs:
	for i := range conds {
		select {
		case jobs <- i:
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
	return ctx.Err()
}
