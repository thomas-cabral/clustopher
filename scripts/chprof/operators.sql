-- Per-operator breakdown for one tagged query (most recent run wins).
-- Params: {tag:String}, {lookback_min:UInt32}
-- elapsed = actual work; input_wait/output_wait = stalled on upstream/downstream.
SELECT
    name AS processor,
    count() AS instances,
    round(sum(elapsed_us) / 1e6, 1) AS work_s,
    round(max(elapsed_us) / 1e6, 1) AS work_s_max,
    round(sum(input_wait_elapsed_us) / 1e6, 1) AS in_wait_s,
    round(sum(output_wait_elapsed_us) / 1e6, 1) AS out_wait_s,
    sum(input_rows) AS in_rows,
    sum(output_rows) AS out_rows,
    formatReadableSize(sum(input_bytes)) AS in_bytes
FROM system.processors_profile_log
WHERE query_id = (
    SELECT query_id FROM system.query_log
    WHERE type = 'QueryFinish'
      AND log_comment = {tag:String}
      AND event_time > now() - INTERVAL {lookback_min:UInt32} MINUTE
    ORDER BY event_time DESC
    LIMIT 1
)
GROUP BY processor
ORDER BY work_s DESC
LIMIT 25
FORMAT PrettyCompactMonoBlock
