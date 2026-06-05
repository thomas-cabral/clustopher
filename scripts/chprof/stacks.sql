-- Top sampled CPU frames for one tagged query (most recent run).
-- Params: {tag:String}, {lookback_min:UInt32}
-- Needs allow_introspection_functions=1 (set by report.sh).
SELECT
    count() AS samples,
    demangle(addressToSymbol(trace[1])) AS top_frame,
    demangle(addressToSymbol(trace[2])) AS caller
FROM system.trace_log
WHERE trace_type = 'CPU'
  AND query_id = (
    SELECT query_id FROM system.query_log
    WHERE type = 'QueryFinish'
      AND log_comment = {tag:String}
      AND event_time > now() - INTERVAL {lookback_min:UInt32} MINUTE
    ORDER BY event_time DESC
    LIMIT 1
)
GROUP BY top_frame, caller
ORDER BY samples DESC
LIMIT 20
FORMAT PrettyCompactMonoBlock
