-- Per-tag summary of profiled queries (log_comment set by chQueryCtx).
-- Params: {lookback_min:UInt32}
SELECT
    log_comment AS tag,
    count() AS queries,
    round(sum(query_duration_ms) / 1000, 1) AS total_s,
    formatReadableSize(max(memory_usage)) AS peak_mem,
    sum(read_rows) AS read_rows,
    formatReadableSize(sum(read_bytes)) AS read_bytes,
    sum(written_rows) AS written_rows,
    formatReadableSize(sum(written_bytes)) AS written_bytes,
    sum(ProfileEvents['ExternalSortWritePart']) AS ext_sort_parts,
    sum(ProfileEvents['ExternalSortMerge']) AS ext_sort_merges,
    formatReadableSize(sum(ProfileEvents['ExternalProcessingUncompressedBytesTotal'])) AS ext_spill_bytes,
    formatReadableSize(sum(ProfileEvents['OSReadBytes'])) AS os_read,
    formatReadableSize(sum(ProfileEvents['OSWriteBytes'])) AS os_write,
    round(sum(ProfileEvents['OSCPUVirtualTimeMicroseconds']) / 1e6, 1) AS cpu_s
FROM system.query_log
WHERE type = 'QueryFinish'
  AND log_comment != ''
  AND event_time > now() - INTERVAL {lookback_min:UInt32} MINUTE
GROUP BY tag
ORDER BY total_s DESC
FORMAT PrettyCompactMonoBlock
