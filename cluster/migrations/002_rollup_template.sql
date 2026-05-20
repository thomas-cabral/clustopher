-- TEMPLATE: rollup_z{N}. Runner expands {N} for N in 2..16.
-- Radius=40 is encoded in the MV definition; recreating MVs is required if Radius changes.

CREATE TABLE IF NOT EXISTS clustopher.rollup_z{N} (
    cluster_id  String,
    tile_x      UInt32,
    tile_y      UInt32,
    cnt         UInt64,
    sum_x       Float64,
    sum_y       Float64,
    metric_sums Map(String, Float64),
    metric_cnts Map(String, UInt64)
) ENGINE = SummingMergeTree
PARTITION BY cluster_id
ORDER BY (cluster_id, tile_x, tile_y);

CREATE MATERIALIZED VIEW IF NOT EXISTS clustopher.mv_rollup_z{N}
TO clustopher.rollup_z{N} AS
SELECT
    cluster_id,
    toUInt32(((x + 180) / 360) * pow(2, {N}) * 512 / 40) AS tile_x,
    toUInt32(((1 - log(tan(y * pi()/180) + 1/cos(y * pi()/180)) / pi()) / 2) * pow(2, {N}) * 512 / 40) AS tile_y,
    1                                                     AS cnt,
    x                                                     AS sum_x,
    y                                                     AS sum_y,
    mapApply((k, v) -> (k, toFloat64(v)), metrics)        AS metric_sums,
    mapApply((k, v) -> (k, toUInt64(1)),  metrics)        AS metric_cnts
FROM clustopher.points;
