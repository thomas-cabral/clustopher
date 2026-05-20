CREATE DATABASE IF NOT EXISTS clustopher;

CREATE TABLE IF NOT EXISTS clustopher.points (
    cluster_id   String,
    id           UInt32,
    external_id  UInt32,
    x            Float32,
    y            Float32,
    metrics      Map(String, Float32),
    metadata     Map(String, String)
) ENGINE = MergeTree
PARTITION BY cluster_id
ORDER BY (cluster_id, id)
SETTINGS index_granularity = 8192;
