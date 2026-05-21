package cluster

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestSkeletonLeaf_Size(t *testing.T) {
	var l SkeletonLeaf
	if got := unsafe.Sizeof(l); got != 28 {
		t.Fatalf("SkeletonLeaf size = %d, want 28", got)
	}
}

func TestBuildSkeleton_SmallContiguous(t *testing.T) {
	// 4 points already in leaf order, NodeSize=2 → 2 leaves.
	pts := []KDPoint{
		{ID: 1, X: 0, Y: 0, NumPoints: 1},
		{ID: 2, X: 1, Y: 0, NumPoints: 1},
		{ID: 3, X: 10, Y: 10, NumPoints: 1},
		{ID: 4, X: 11, Y: 10, NumPoints: 1},
	}
	tree := BuildSkeleton(pts, 2)

	if len(tree.Leaves) != 2 {
		t.Fatalf("leaves = %d, want 2", len(tree.Leaves))
	}
	if tree.Leaves[0].IDMin != 1 || tree.Leaves[0].IDMax != 2 || tree.Leaves[0].Count != 2 {
		t.Fatalf("leaf0 = %+v", tree.Leaves[0])
	}
	if tree.Leaves[0].Bounds.MinX != 0 || tree.Leaves[0].Bounds.MaxX != 1 {
		t.Fatalf("leaf0 bounds = %+v", tree.Leaves[0].Bounds)
	}
	if tree.Leaves[1].IDMin != 3 || tree.Leaves[1].IDMax != 4 || tree.Leaves[1].Count != 2 {
		t.Fatalf("leaf1 = %+v", tree.Leaves[1])
	}
}

func TestBuildSkeleton_CountSumsToTotal(t *testing.T) {
	pts := make([]KDPoint, 1000)
	for i := range pts {
		pts[i] = KDPoint{ID: uint32(i + 1), X: float32(i), Y: 0, NumPoints: 1}
	}
	tree := BuildSkeleton(pts, 64)
	var total uint32
	for _, l := range tree.Leaves {
		total += l.Count
	}
	if total != 1000 {
		t.Fatalf("count sum = %d, want 1000", total)
	}
}

func TestSortPointsIntoLeafOrder_DeterministicLeaves(t *testing.T) {
	pts := []KDPoint{
		{ID: 1, X: 0, Y: 0}, {ID: 2, X: 10, Y: 10},
		{ID: 3, X: 1, Y: 1}, {ID: 4, X: 11, Y: 11},
		{ID: 5, X: 0, Y: 10}, {ID: 6, X: 10, Y: 0},
		{ID: 7, X: 1, Y: 11}, {ID: 8, X: 11, Y: 1},
	}
	sorted := SortPointsIntoLeafOrder(pts, 2)
	if len(sorted) != 8 {
		t.Fatalf("len = %d", len(sorted))
	}
	full := boundsOver(pts)
	fullArea := (full.MaxX - full.MinX) * (full.MaxY - full.MinY)
	for i := 0; i < 8; i += 2 {
		leafBounds := boundsOver(sorted[i : i+2])
		area := (leafBounds.MaxX - leafBounds.MinX) * (leafBounds.MaxY - leafBounds.MinY)
		if area > 0.5*fullArea {
			t.Errorf("leaf %d covers >50%% of full bbox (area=%f, full=%f)", i/2, area, fullArea)
		}
	}
}

func TestSortPointsIntoLeafOrderParallel_MatchesSequential(t *testing.T) {
	prev := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(prev)

	pts := make([]KDPoint, 4096)
	for i := range pts {
		pts[i] = KDPoint{
			ID: uint32(i + 1),
			X:  float32((i*7919)%10007) / 10007,
			Y:  float32((i*1543)%10009) / 10009,
		}
	}

	want := SortPointsIntoLeafOrder(pts, 16)
	got := sortPointsIntoLeafOrderParallel(pts, 16, 64)
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("point %d id = %d, want %d", i, got[i].ID, want[i].ID)
		}
	}
}

func TestRangeLeaves_ReturnsIntersectingOnly(t *testing.T) {
	pts := []KDPoint{
		{ID: 1, X: 0, Y: 0}, {ID: 2, X: 1, Y: 1},
		{ID: 3, X: 100, Y: 100}, {ID: 4, X: 101, Y: 101},
	}
	sorted := SortPointsIntoLeafOrder(pts, 2)
	tree := BuildSkeleton(sorted, 2)

	leaves := tree.RangeLeaves(KDBounds{MinX: -1, MinY: -1, MaxX: 2, MaxY: 2})
	if len(leaves) != 1 {
		t.Fatalf("got %d leaves, want 1", len(leaves))
	}
	if tree.Leaves[leaves[0]].Count != 2 {
		t.Fatalf("wrong leaf returned")
	}

	all := tree.RangeLeaves(KDBounds{MinX: -1, MinY: -1, MaxX: 200, MaxY: 200})
	if len(all) != 2 {
		t.Fatalf("got %d leaves, want 2", len(all))
	}
}

func TestLoad_PopulatesSkeleton(t *testing.T) {
	sc := NewSupercluster(SuperclusterOptions{
		MinZoom: 0, MaxZoom: 16, MinPoints: 3, Radius: 40,
		Extent: 512, NodeSize: 64,
	})
	pts := generateRandomPoints(5000, -125, -65, 25, 49)
	if err := sc.Load(pts); err != nil {
		t.Fatal(err)
	}

	if sc.Skeleton == nil {
		t.Fatal("Skeleton not populated")
	}
	if len(sc.Skeleton.Leaves) == 0 {
		t.Fatal("Skeleton has no leaves")
	}
	var total uint32
	for _, l := range sc.Skeleton.Leaves {
		total += l.Count
	}
	if total != 5000 {
		t.Fatalf("skeleton point count = %d, want 5000", total)
	}
}

func TestBuildSkeletonWithRemap_AssignsSequentialIDs(t *testing.T) {
	pts := []KDPoint{
		{ID: 100, X: 0, Y: 0}, {ID: 200, X: 1, Y: 1},
		{ID: 300, X: 100, Y: 100}, {ID: 400, X: 101, Y: 101},
	}
	sorted := SortPointsIntoLeafOrder(pts, 2)
	tree, remap := BuildSkeletonWithRemap(sorted, 2)

	if len(remap) != 4 {
		t.Fatalf("remap len = %d", len(remap))
	}
	if tree.Leaves[0].IDMin != 1 || tree.Leaves[0].IDMax != 2 {
		t.Fatalf("leaf0 ids = %d..%d", tree.Leaves[0].IDMin, tree.Leaves[0].IDMax)
	}
	if tree.Leaves[1].IDMin != 3 || tree.Leaves[1].IDMax != 4 {
		t.Fatalf("leaf1 ids = %d..%d", tree.Leaves[1].IDMin, tree.Leaves[1].IDMax)
	}
}
