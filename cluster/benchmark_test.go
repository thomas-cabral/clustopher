package cluster

import (
	"math/rand"
)

// generateRandomPoints creates n random points within a geographic bounding box
func generateRandomPoints(n int, minLng, maxLng, minLat, maxLat float32) []Point {
	points := make([]Point, n)
	// Use deterministic seed for reproducibility
	source := rand.NewSource(42)
	r := rand.New(source)

	for i := 0; i < n; i++ {
		points[i] = Point{
			ID: uint32(i + 1),
			X:  minLng + r.Float32()*(maxLng-minLng),
			Y:  minLat + r.Float32()*(maxLat-minLat),
			Metrics: map[string]float32{
				"value": r.Float32() * 100,
			},
			Metadata: map[string]interface{}{
				"type": "test",
			},
		}
	}
	return points
}
