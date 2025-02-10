package cluster

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"time"
)

type MetadataSummary struct {
	TotalPoints     int                    `json:"totalPoints"`
	NumClusters     int                    `json:"numClusters"`
	NumSinglePoints int                    `json:"numSinglePoints"`
	MetricsSummary  map[string]MetricStats `json:"metricsSummary"`
	MetadataSummary map[string]interface{} `json:"metadataSummary"`
}

type MetricStats struct {
	Min     float32 `json:"min"`
	Max     float32 `json:"max"`
	Sum     float32 `json:"sum"`
	Average float32 `json:"average"`
}

func CalculateMetadataSummary(clusters []ClusterNode) MetadataSummary {
	summary := MetadataSummary{
		MetricsSummary:  make(map[string]MetricStats),
		MetadataSummary: make(map[string]interface{}),
	}

	if len(clusters) == 0 {
		return summary
	}

	// Initialize metrics tracking
	metricsMap := make(map[string]struct {
		min   float32
		max   float32
		sum   float32
		count int
	})

	// Track metadata frequencies with type-specific handling
	metadataFreq := make(map[string]map[string]int)
	timestampStats := struct {
		min   time.Time
		max   time.Time
		count int
	}{
		min: time.Now(),
		max: time.Time{},
	}

	// Process each cluster
	for _, c := range clusters {
		if c.Count > 1 {
			summary.NumClusters++
		} else {
			summary.NumSinglePoints++
		}
		summary.TotalPoints += int(c.Count)

		// Process metrics
		for metricName, value := range c.Metrics.Values {
			stats, exists := metricsMap[metricName]
			if !exists {
				stats.min = value
				stats.max = value
			} else {
				if value < stats.min {
					stats.min = value
				}
				if value > stats.max {
					stats.max = value
				}
			}
			stats.sum += value
			stats.count++
			metricsMap[metricName] = stats
		}

		// Process metadata with type-specific handling
		for key, rawValue := range c.Metadata {
			if _, exists := metadataFreq[key]; !exists {
				metadataFreq[key] = make(map[string]int)
			}

			// Try to unmarshal the value based on expected types
			switch key {
			case "timestamp":
				var timestamp time.Time
				if err := json.Unmarshal(rawValue, &timestamp); err == nil {
					if timestamp.Before(timestampStats.min) {
						timestampStats.min = timestamp
					}
					if timestamp.After(timestampStats.max) {
						timestampStats.max = timestamp
					}
					timestampStats.count++
				}
			case "category":
				var category string
				if err := json.Unmarshal(rawValue, &category); err == nil {
					metadataFreq[key][category]++
				}
			default:
				// For other metadata types, store as string
				var strValue string
				if err := json.Unmarshal(rawValue, &strValue); err == nil {
					metadataFreq[key][strValue]++
				}
			}
		}
	}

	// Calculate final metrics statistics
	for metricName, stats := range metricsMap {
		summary.MetricsSummary[metricName] = MetricStats{
			Min:     stats.min,
			Max:     stats.max,
			Sum:     stats.sum,
			Average: stats.sum / float32(stats.count),
		}
	}

	// Add timestamp range to metadata summary
	if timestampStats.count > 0 {
		summary.MetadataSummary["timeRange"] = map[string]string{
			"start": timestampStats.min.Format(time.RFC3339),
			"end":   timestampStats.max.Format(time.RFC3339),
		}
	}

	// Find most common categories and other metadata values
	for key, freqMap := range metadataFreq {
		if key == "category" {
			// For categories, include distribution
			distribution := make(map[string]float64)
			total := 0
			for _, count := range freqMap {
				total += count
			}
			for value, count := range freqMap {
				distribution[value] = float64(count) / float64(total) * 100
			}
			summary.MetadataSummary[key] = distribution
		} else {
			// For other metadata, just include most common value
			var mostCommon string
			var maxCount int
			for value, count := range freqMap {
				if count > maxCount {
					maxCount = count
					mostCommon = value
				}
			}
			summary.MetadataSummary[key] = mostCommon
		}
	}

	return summary
}

func GenerateTestPoints(n int, bounds KDBounds) []Point {
	rand.Seed(time.Now().UnixNano())
	points := make([]Point, n)
	randomMetricName := fmt.Sprintf("metric_%d", rand.Intn(1000))

	for i := 0; i < n; i++ {
		x := bounds.MinX + rand.Float32()*(bounds.MaxX-bounds.MinX)
		y := bounds.MinY + rand.Float32()*(bounds.MaxY-bounds.MinY)

		points[i] = Point{
			ID: uint32(i + 1),
			X:  x,
			Y:  y,
			Metrics: map[string]float32{
				"value":          rand.Float32() * 100,
				"size":           rand.Float32() * 50,
				"sales":          rand.Float32() * 1000,
				"customers":      float32(rand.Intn(100)),
				randomMetricName: rand.Float32() * 200,
			},
			Metadata: map[string]interface{}{
				"timestamp": time.Now().Add(-time.Duration(rand.Intn(7*24)) * time.Hour),
				"category":  []string{"A", "B", "C"}[rand.Intn(3)],
			},
		}
	}

	return points
}
