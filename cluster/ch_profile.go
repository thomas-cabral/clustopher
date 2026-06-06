package cluster

import (
	"context"
	"os"

	"github.com/ClickHouse/clickhouse-go/v2"
)

// chProfileEnabled reports whether CH-side query profiling is requested via
// CLUSTOPHER_CH_PROFILE=1. When enabled, heavy queries are tagged with a
// log_comment and run with the sampling profiler on, so system.query_log,
// system.processors_profile_log and system.trace_log carry rows attributable
// to each pipeline stage. Off by default; the production path is unchanged.
func chProfileEnabled() bool {
	return os.Getenv("CLUSTOPHER_CH_PROFILE") == "1"
}

// chQueryCtx returns a context carrying the given CH settings, augmented with
// profiling settings + a log_comment tag when CLUSTOPHER_CH_PROFILE=1. The
// settings map is copied, so callers may pass shared maps (e.g.
// insertSettings) without aliasing.
func chQueryCtx(ctx context.Context, tag string, settings clickhouse.Settings) context.Context {
	merged := clickhouse.Settings{}
	for k, v := range settings {
		merged[k] = v
	}
	if chProfileEnabled() {
		merged["log_comment"] = tag
		merged["log_processors_profiles"] = uint64(1)
		merged["query_profiler_real_time_period_ns"] = uint64(10_000_000)
		merged["query_profiler_cpu_time_period_ns"] = uint64(10_000_000)
	}
	if len(merged) == 0 {
		return ctx
	}
	return clickhouse.Context(ctx, clickhouse.WithSettings(merged))
}
