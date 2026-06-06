#!/usr/bin/env bash
# Dump CH-side profiling report for tagged queries (CLUSTOPHER_CH_PROFILE=1 runs).
#
# Usage:
#   report.sh [lookback_minutes]            # summary of all tags
#   report.sh [lookback_minutes] <tag> ...  # + operator/stack breakdown per tag
set -euo pipefail

CH="${CLICKHOUSE_HTTP:-http://127.0.0.1:18123}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOOKBACK="${1:-240}"
shift || true

curl -fsS "$CH/" --data-binary "SYSTEM FLUSH LOGS" > /dev/null

echo "=== query summary (last ${LOOKBACK}m) ==="
curl -fsS "$CH/?param_lookback_min=$LOOKBACK" --data-binary @"$DIR/query_summary.sql"

for tag in "$@"; do
    echo
    echo "=== operators: $tag ==="
    curl -fsS "$CH/?param_lookback_min=$LOOKBACK&param_tag=$tag" \
        --data-binary @"$DIR/operators.sql"
    echo
    echo "=== top stacks: $tag ==="
    curl -fsS "$CH/?param_lookback_min=$LOOKBACK&param_tag=$tag&allow_introspection_functions=1" \
        --data-binary @"$DIR/stacks.sql"
done
