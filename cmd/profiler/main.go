package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	"web/clustopher/cluster"
)

var (
	cpuprofile  = flag.String("cpuprofile", "", "write cpu profile to file")
	memprofile  = flag.String("memprofile", "", "write memory profile to file")
	heapprofile = flag.String("heapprofile", "", "write heap profile to file")
	numPoints   = flag.Int("points", 100000, "number of points to generate")
	zoomLevel   = flag.Int("zoom", 8, "zoom level to profile")
	testall     = flag.Bool("testall", false, "test all configurations")
)

// generateRandomPoints creates n random points within a geographic bounding box
func generateRandomPoints(n int, minLng, maxLng, minLat, maxLat float32) []cluster.Point {
	points := make([]cluster.Point, n)
	// Use deterministic seed for reproducibility
	source := rand.NewSource(42)
	r := rand.New(source)

	for i := 0; i < n; i++ {
		points[i] = cluster.Point{
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

// generateKDPoints creates KDPoints from regular points at a specific zoom level
func generateKDPoints(sc *cluster.Supercluster, points []cluster.Point, zoom int) []cluster.KDPoint {
	kdPoints := make([]cluster.KDPoint, len(points))
	for i, p := range points {
		// Don't call unexported projectFast method
		// Instead, create a simple projected point directly
		// This is a simplified projection for profiling purposes
		zoomScale := math.Pow(2, float64(zoom))
		lng := p.X
		lat := p.Y
		x := float32(lng * float32(zoomScale))
		y := float32(math.Log(math.Tan(float64(lat)*math.Pi/180+math.Pi/4)) / math.Pi * 256 * float64(zoomScale))

		kdPoints[i] = cluster.KDPoint{
			ID:        p.ID,
			X:         x,
			Y:         y,
			NumPoints: 1,
		}
	}
	return kdPoints
}

func runSingleProfile(numPoints, zoomLevel int) {
	fmt.Printf("Profiling with %d points at zoom level %d\n", numPoints, zoomLevel)

	// Create the supercluster with default options
	sc := cluster.NewSupercluster(cluster.SuperclusterOptions{
		MinZoom:   0,
		MaxZoom:   16,
		MinPoints: 3,
		Radius:    40,
		Extent:    512,
		NodeSize:  64,
		Log:       false,
	})

	// Generate random points in the US region
	points := generateRandomPoints(numPoints, -125.0, -65.0, 25.0, 49.0)

	// Convert to KDPoints at the specified zoom level
	kdPoints := generateKDPoints(sc, points, zoomLevel)

	// Measure memory before clustering
	var memStatsBefore, memStatsAfter runtime.MemStats
	runtime.ReadMemStats(&memStatsBefore)

	// Time the clustering
	start := time.Now()

	// Run the new optimized selection logic
	if numPoints > 50000 ||
		zoomLevel < sc.Options.MaxZoom/4 {
		fmt.Println("Using grid-based clustering")
		sc.ClusterPoints(kdPoints, zoomLevel)
	} else {
		fmt.Println("Using traditional clustering")
		sc.ClusterPoints(kdPoints, zoomLevel)
	}

	duration := time.Since(start)

	// Measure memory after clustering
	runtime.ReadMemStats(&memStatsAfter)

	// Calculate memory usage
	allocMB := float64(memStatsAfter.TotalAlloc-memStatsBefore.TotalAlloc) / 1024 / 1024

	fmt.Printf("Clustering completed in %v\n", duration)
	fmt.Printf("Memory allocated: %.2f MB\n", allocMB)
	fmt.Printf("Memory usage: %.2f MB\n", float64(memStatsAfter.Alloc)/1024/1024)
}

func runProfileBattery() {
	pointCounts := []int{1000, 10000, 50000, 100000}
	zoomLevels := []int{2, 5, 8, 12, 15}

	fmt.Println("Running comprehensive profile battery...")
	fmt.Println("=======================================")

	// Table header
	fmt.Printf("%-10s | %-10s | %-12s | %-15s | %-10s | %-10s\n",
		"Points", "Zoom", "Method", "Duration", "Memory (MB)", "GC Runs")
	fmt.Printf("%s\n", "------------------------------------------------------------------------")

	for _, points := range pointCounts {
		for _, zoom := range zoomLevels {
			// Create the supercluster
			sc := cluster.NewSupercluster(cluster.SuperclusterOptions{
				MinZoom:   0,
				MaxZoom:   16,
				MinPoints: 3,
				Radius:    40,
				Extent:    512,
				NodeSize:  64,
				Log:       false,
			})

			// Generate points
			testPoints := generateRandomPoints(points, -125.0, -65.0, 25.0, 49.0)
			kdPoints := generateKDPoints(sc, testPoints, zoom)

			// Collect GC stats before
			var memStatsBefore, memStatsAfter runtime.MemStats
			runtime.ReadMemStats(&memStatsBefore)

			// Run traditional or grid-based clustering based on new optimization rules
			useGrid := points > 50000 ||
				zoom < sc.Options.MaxZoom/4

			method := "Traditional"
			if useGrid {
				method = "Grid"
			}

			// Time the execution
			start := time.Now()
			if useGrid {
				sc.ClusterPoints(kdPoints, zoom)
			} else {
				sc.ClusterPoints(kdPoints, zoom)
			}
			duration := time.Since(start)

			// Collect stats after
			runtime.ReadMemStats(&memStatsAfter)
			memMB := float64(memStatsAfter.TotalAlloc-memStatsBefore.TotalAlloc) / 1024 / 1024
			gcRuns := memStatsAfter.NumGC - memStatsBefore.NumGC

			// Print result row
			fmt.Printf("%-10d | %-10d | %-12s | %-15s | %-10.2f | %-10d\n",
				points, zoom, method, duration, memMB, gcRuns)
		}

		// Add separator between point counts
		fmt.Printf("%s\n", "------------------------------------------------------------------------")
	}
}

func main() {
	flag.Parse()

	// Set up CPU profiling if requested
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Could not create CPU profile: %v\n", err)
			return
		}
		defer f.Close()

		fmt.Println("Starting CPU profiling...")
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "Could not start CPU profile: %v\n", err)
			return
		}
		defer pprof.StopCPUProfile()
	}

	// Run tests
	if *testall {
		runProfileBattery()
	} else {
		runSingleProfile(*numPoints, *zoomLevel)
	}

	// Write memory profile if requested
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Could not create memory profile: %v\n", err)
			return
		}
		defer f.Close()
		runtime.GC() // Get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "Could not write memory profile: %v\n", err)
		}
	}

	// Write heap profile if requested
	if *heapprofile != "" {
		f, err := os.Create(*heapprofile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Could not create heap profile: %v\n", err)
			return
		}
		defer f.Close()

		memProfile := pprof.Lookup("heap")
		if memProfile == nil {
			fmt.Fprintf(os.Stderr, "Could not find heap profile\n")
			return
		}

		if err := memProfile.WriteTo(f, 0); err != nil {
			fmt.Fprintf(os.Stderr, "Could not write heap profile: %v\n", err)
		}
	}
}
