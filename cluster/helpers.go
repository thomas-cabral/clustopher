package cluster

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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

type MetadataRange struct {
	Min     float64 `json:"min"`
	Max     float64 `json:"max"`
	Average float64 `json:"average"`
}

type TimestampRange struct {
	Earliest time.Time `json:"earliest"`
	Latest   time.Time `json:"latest"`
}

type SavedClusterInfo struct {
	ID        string
	NumPoints int
	Timestamp time.Time
	FileSize  int64
}

// ListSavedClusters returns information about all saved clusters
func ListSavedClusters() ([]SavedClusterInfo, error) {
	files, err := os.ReadDir("data/clusters")
	if err != nil {
		return nil, fmt.Errorf("failed to read clusters directory: %v", err)
	}

	clusters := make([]SavedClusterInfo, 0)
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".zst" {
			info, err := file.Info()
			if err != nil {
				continue
			}

			// Parse filename to get cluster info
			// Format: cluster-{numPoints}p-{timestamp}-{id}.zst
			name := strings.TrimSuffix(file.Name(), ".zst")
			parts := strings.Split(name, "-")
			if len(parts) != 5 {
				continue
			}

			numPoints, err := strconv.Atoi(strings.TrimSuffix(parts[1], "p"))
			if err != nil {
				continue
			}

			timestamp, err := time.Parse("20060102-150405", parts[2]+"-"+parts[3])
			if err != nil {
				continue
			}

			clusters = append(clusters, SavedClusterInfo{
				ID:        parts[4],
				NumPoints: numPoints,
				Timestamp: timestamp,
				FileSize:  info.Size(),
			})
		}
	}

	return clusters, nil
}

// GetClusterInfo returns information about a specific cluster
func GetClusterInfo(clusterID string) (*SavedClusterInfo, error) {
	files, err := os.ReadDir("data/clusters")
	if err != nil {
		return nil, fmt.Errorf("failed to read clusters directory: %v", err)
	}

	for _, file := range files {
		if strings.Contains(file.Name(), clusterID) {
			info, err := file.Info()
			if err != nil {
				return nil, err
			}

			// Parse filename to get cluster info
			name := strings.TrimSuffix(file.Name(), ".zst")
			parts := strings.Split(name, "-")
			if len(parts) != 5 {
				continue
			}

			numPoints, err := strconv.Atoi(strings.TrimSuffix(parts[1], "p"))
			if err != nil {
				continue
			}

			timestamp, err := time.Parse("20060102-150405", parts[2]+"-"+parts[3])
			if err != nil {
				continue
			}

			return &SavedClusterInfo{
				ID:        parts[4],
				NumPoints: numPoints,
				Timestamp: timestamp,
				FileSize:  info.Size(),
			}, nil
		}
	}

	return nil, fmt.Errorf("cluster %s not found", clusterID)
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

	// Track metadata values and their frequencies
	metadataValues := make(map[string]map[string]int)
	numericMetadata := make(map[string]struct {
		min   float64
		max   float64
		sum   float64
		count int
	})
	timestampRanges := make(map[string]TimestampRange)

	// Process each cluster or point
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

		// Process metadata
		for key, rawValue := range c.Metadata {
			if _, exists := metadataValues[key]; !exists {
				metadataValues[key] = make(map[string]int)
			}

			var frequencies map[string]float64
			if err := json.Unmarshal(rawValue, &frequencies); err == nil {
				for value, freq := range frequencies {
					count := int(freq * float64(c.Count))

					if key == "timestamp" {
						if ts, err := time.Parse(time.RFC3339, value); err == nil {
							timeRange, exists := timestampRanges[key]
							if !exists {
								timeRange = TimestampRange{
									Earliest: ts,
									Latest:   ts,
								}
							} else {
								if ts.Before(timeRange.Earliest) {
									timeRange.Earliest = ts
								}
								if ts.After(timeRange.Latest) {
									timeRange.Latest = ts
								}
							}
							timestampRanges[key] = timeRange
						}
					} else if numValue, err := strconv.ParseFloat(value, 64); err == nil {
						// Track numeric metadata
						stats, exists := numericMetadata[key]
						if !exists {
							stats.min = numValue
							stats.max = numValue
						} else {
							if numValue < stats.min {
								stats.min = numValue
							}
							if numValue > stats.max {
								stats.max = numValue
							}
						}
						stats.sum += numValue * float64(count)
						stats.count += count
						numericMetadata[key] = stats
					} else {
						// Track categorical metadata
						metadataValues[key][value] += count
					}
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

	// Add metadata summaries
	for key, timeRange := range timestampRanges {
		summary.MetadataSummary[key] = timeRange
	}

	for key, stats := range numericMetadata {
		summary.MetadataSummary[key] = MetadataRange{
			Min:     stats.min,
			Max:     stats.max,
			Average: stats.sum / float64(stats.count),
		}
	}

	for key, freqMap := range metadataValues {
		if _, isNumeric := numericMetadata[key]; !isNumeric {
			if _, isTime := timestampRanges[key]; !isTime {
				// Only process distributions for non-numeric, non-timestamp metadata
				distribution := make(map[string]float64)
				total := 0
				for _, count := range freqMap {
					total += count
				}
				for value, count := range freqMap {
					distribution[value] = float64(count) / float64(total) * 100
				}
				summary.MetadataSummary[key] = distribution
			}
		}
	}

	return summary
}

func GenerateTestPoints(n int, bounds KDBounds) []Point {
	rand.Seed(time.Now().UnixNano())
	points := make([]Point, n)
	randomMetricName := fmt.Sprintf("metric_%d", rand.Intn(1000))

	// Define more diverse categories for global data
	categories := []string{"Urban", "Rural", "Coastal", "Mountain", "Desert", "Forest", "Island"}
	regions := []string{"Americas", "Europe", "Asia", "Africa", "Oceania"}

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
				"timestamp": time.Now().Add(-time.Duration(rand.Intn(365)) * time.Hour * 24), // Expanded to full year
				"category":  categories[rand.Intn(len(categories))],
				"region":    regions[rand.Intn(len(regions))],
				"elevation": rand.Float32() * 5000, // Added elevation in meters
			},
		}
	}

	return points
}
