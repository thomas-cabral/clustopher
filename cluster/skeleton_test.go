package cluster

import (
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
