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

CREATE TABLE IF NOT EXISTS clustopher.staging_points (
    cluster_id   String,
    external_id  UInt32,
    x            Float32,
    y            Float32,
    metrics      Map(String, Float32),
    metadata     Map(String, String)
) ENGINE = MergeTree
PARTITION BY cluster_id
ORDER BY (cluster_id, external_id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS clustopher.point_id_map (
    cluster_id  String,
    external_id UInt32,
    internal_id UInt32
) ENGINE = MergeTree
PARTITION BY cluster_id
ORDER BY (cluster_id, external_id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS clustopher.point_id_map_load (
    load_id     UInt64,
    external_id UInt32,
    internal_id UInt32
) ENGINE = MergeTree
PARTITION BY load_id
ORDER BY (load_id, external_id)
SETTINGS index_granularity = 8192;
