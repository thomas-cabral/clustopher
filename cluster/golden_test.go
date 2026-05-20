package cluster

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

var updateGolden = flag.Bool("update-golden", false, "regenerate golden fixtures")

// goldenMatrix defines (numPoints, zoom) combinations to snapshot.
var goldenMatrix = []struct {
	n    int
	zoom int
}{
	{1000, 2}, {1000, 8}, {1000, 14},
	{10000, 2}, {10000, 8}, {10000, 14},
	{100000, 2}, {100000, 8}, {100000, 14},
}

// goldenBounds is the full CONUS bbox; viewport queries cover everything.
var goldenBounds = KDBounds{MinX: -125, MinY: 25, MaxX: -65, MaxY: 49}

func TestGoldenSnapshots(t *testing.T) {
	if !*updateGolden {
		if _, err := os.Stat("testdata/golden"); os.IsNotExist(err) {
			t.Skip("no golden fixtures present; run with -update-golden to create")
		}
		return
	}

	if err := os.MkdirAll("testdata/golden", 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	for _, c := range goldenMatrix {
		sc := NewSupercluster(SuperclusterOptions{
			MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
			Extent: 512, NodeSize: 64, Log: false,
		})
		points := generateRandomPoints(c.n, -125.0, -65.0, 25.0, 49.0)
		if err := sc.Load(points); err != nil {
			t.Fatalf("Load: %v", err)
		}
		clusters := sc.GetClusters(goldenBounds, c.zoom)

		path := filepath.Join("testdata/golden", fmt.Sprintf("n%d_z%d.json", c.n, c.zoom))
		f, err := os.Create(path)
		if err != nil {
			t.Fatalf("create: %v", err)
		}
		enc := json.NewEncoder(f)
		enc.SetIndent("", "  ")
		if err := enc.Encode(clusters); err != nil {
			t.Fatalf("encode: %v", err)
		}
		f.Close()
	}
}
