package runner

import (
	"encoding/json"
	"web/clustopher/cluster"

	pb "web/clustopher/proto"
)

// convertMetricsSummary converts cluster.MetricStats to proto.MetricStats
func convertMetricsSummary(metrics map[string]cluster.MetricStats) map[string]*pb.MetricStats {
	result := make(map[string]*pb.MetricStats)
	for k, v := range metrics {
		result[k] = &pb.MetricStats{
			Min:     float64(v.Min),
			Max:     float64(v.Max),
			Average: float64(v.Average),
		}
	}
	return result
}

// convertMetadataSummary converts metadata summary to string map
func convertMetadataSummary(metadata map[string]interface{}) map[string]string {
	result := make(map[string]string)
	for k, v := range metadata {
		// Convert each value to JSON string
		if jsonBytes, err := json.Marshal(v); err == nil {
			result[k] = string(jsonBytes)
		}
	}
	return result
}