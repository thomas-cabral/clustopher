package cluster

import (
	"runtime"
	"sync"
)

const parallelSortThreshold = 1_000_000

// SkeletonLeaf is a tree leaf that retains only a bounding box, contiguous
// internal-id range, and point count. The actual point data lives in
// ClickHouse (or, during Phase 1, in the existing in-memory point store).
type SkeletonLeaf struct {
	Bounds KDBounds // 16 bytes
	IDMin  uint32   //  4 bytes
	IDMax  uint32   //  4 bytes
	Count  uint32   //  4 bytes
}

// SkeletonNode is a KD-tree node. LeafIdx >= 0 indicates a leaf node (index
// into SkeletonTree.Leaves); LeafIdx == -1 indicates an internal node.
type SkeletonNode struct {
	LeafIdx int32   // 4 bytes
	Left    int32   // 4 bytes
	Right   int32   // 4 bytes
	Axis    uint8   // 1 byte
	_       [3]byte // padding
	Bounds  KDBounds
}

type SkeletonTree struct {
	Nodes  []SkeletonNode
	Leaves []SkeletonLeaf
}

// BuildSkeleton takes a slice of points already sorted into KD-tree leaf
// order and emits a SkeletonTree. Points are chunked into leaves of size
// nodeSize (last leaf may be smaller). Internal nodes are emitted bottom-up
// by recursively pairing children.
//
// Sorting input into leaf order is the caller's responsibility — see
// SortPointsIntoLeafOrder in Task 1.3.
func BuildSkeleton(points []KDPoint, nodeSize int) *SkeletonTree {
	if nodeSize < 1 {
		nodeSize = 1
	}
	tree := &SkeletonTree{}
	if len(points) == 0 {
		return tree
	}

	// 1. Emit leaves.
	for i := 0; i < len(points); i += nodeSize {
		end := i + nodeSize
		if end > len(points) {
			end = len(points)
		}
		chunk := points[i:end]
		leaf := SkeletonLeaf{
			IDMin: chunk[0].ID,
			IDMax: chunk[len(chunk)-1].ID,
			Count: uint32(len(chunk)),
		}
		leaf.Bounds = boundsOver(chunk)
		tree.Leaves = append(tree.Leaves, leaf)
	}

	// 2. Build internal nodes bottom-up. Each leaf becomes a SkeletonNode with
	//    LeafIdx set; we then pair them into a balanced binary tree.
	type lvl struct{ idx int32 }
	current := make([]lvl, len(tree.Leaves))
	for i, lf := range tree.Leaves {
		n := SkeletonNode{LeafIdx: int32(i), Left: -1, Right: -1, Axis: 0, Bounds: lf.Bounds}
		tree.Nodes = append(tree.Nodes, n)
		current[i] = lvl{idx: int32(len(tree.Nodes) - 1)}
	}

	axis := uint8(0)
	for len(current) > 1 {
		next := make([]lvl, 0, (len(current)+1)/2)
		for i := 0; i < len(current); i += 2 {
			if i+1 == len(current) {
				next = append(next, current[i])
				continue
			}
			l := current[i]
			r := current[i+1]
			n := SkeletonNode{
				LeafIdx: -1,
				Left:    l.idx,
				Right:   r.idx,
				Axis:    axis,
				Bounds:  unionBounds(tree.Nodes[l.idx].Bounds, tree.Nodes[r.idx].Bounds),
			}
			tree.Nodes = append(tree.Nodes, n)
			next = append(next, lvl{idx: int32(len(tree.Nodes) - 1)})
		}
		axis ^= 1
		current = next
	}
	return tree
}

func boundsOver(pts []KDPoint) KDBounds {
	b := KDBounds{
		MinX: pts[0].X, MaxX: pts[0].X,
		MinY: pts[0].Y, MaxY: pts[0].Y,
	}
	for _, p := range pts[1:] {
		if p.X < b.MinX {
			b.MinX = p.X
		}
		if p.X > b.MaxX {
			b.MaxX = p.X
		}
		if p.Y < b.MinY {
			b.MinY = p.Y
		}
		if p.Y > b.MaxY {
			b.MaxY = p.Y
		}
	}
	return b
}

func unionBounds(a, b KDBounds) KDBounds {
	out := a
	if b.MinX < out.MinX {
		out.MinX = b.MinX
	}
	if b.MaxX > out.MaxX {
		out.MaxX = b.MaxX
	}
	if b.MinY < out.MinY {
		out.MinY = b.MinY
	}
	if b.MaxY > out.MaxY {
		out.MaxY = b.MaxY
	}
	return out
}

// SortPointsIntoLeafOrder reorders a slice of points such that consecutive
// chunks of nodeSize are valid KD-tree leaves (i.e. each chunk is spatially
// localized). Sort is in-place on a copy and returns the result. Uses the
// same median-partition strategy as buildKDTree.
func SortPointsIntoLeafOrder(points []KDPoint, nodeSize int) []KDPoint {
	if nodeSize < 1 {
		nodeSize = 1
	}
	out := make([]KDPoint, len(points))
	copy(out, points)
	sortRecursive(out, 0, len(out)-1, 0, nodeSize)
	return out
}

func SortPointsIntoLeafOrderParallel(points []KDPoint, nodeSize int) []KDPoint {
	return sortPointsIntoLeafOrderParallel(points, nodeSize, parallelSortThreshold, true)
}

// SortPointsIntoLeafOrderParallelInPlace mutates the input slice in place and
// returns it. Skips the full-slice copy that SortPointsIntoLeafOrderParallel
// makes for safety, halving peak heap on multi-hundred-million-point loads.
func SortPointsIntoLeafOrderParallelInPlace(points []KDPoint, nodeSize int) []KDPoint {
	return sortPointsIntoLeafOrderParallel(points, nodeSize, parallelSortThreshold, false)
}

func sortPointsIntoLeafOrderParallel(points []KDPoint, nodeSize, threshold int, copyInput bool) []KDPoint {
	if nodeSize < 1 {
		nodeSize = 1
	}
	if threshold < 1 {
		threshold = 1
	}
	var out []KDPoint
	if copyInput {
		out = make([]KDPoint, len(points))
		copy(out, points)
	} else {
		out = points
	}
	if len(out) == 0 {
		return out
	}

	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || len(out) < threshold {
		sortRecursive(out, 0, len(out)-1, 0, nodeSize)
		return out
	}

	tokens := make(chan struct{}, workers-1)
	var wg sync.WaitGroup
	var sortParallel func(lo, hi, axis int)

	run := func(lo, hi, axis int) {
		if hi-lo+1 < threshold {
			sortParallel(lo, hi, axis)
			return
		}
		select {
		case tokens <- struct{}{}:
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer func() { <-tokens }()
				sortParallel(lo, hi, axis)
			}()
		default:
			sortParallel(lo, hi, axis)
		}
	}

	sortParallel = func(lo, hi, axis int) {
		if hi-lo+1 <= nodeSize {
			return
		}
		mid := (lo + hi) / 2
		if axis == 0 {
			quickselectX(out, lo, hi, mid)
		} else {
			quickselectY(out, lo, hi, mid)
		}
		nextAxis := axis ^ 1
		run(lo, mid-1, nextAxis)
		run(mid+1, hi, nextAxis)
	}

	sortParallel(0, len(out)-1, 0)
	wg.Wait()
	return out
}

func sortRecursive(pts []KDPoint, lo, hi, axis, nodeSize int) {
	if hi-lo+1 <= nodeSize {
		return
	}
	mid := (lo + hi) / 2
	if axis == 0 {
		quickselectX(pts, lo, hi, mid)
	} else {
		quickselectY(pts, lo, hi, mid)
	}
	nextAxis := axis ^ 1
	sortRecursive(pts, lo, mid-1, nextAxis, nodeSize)
	sortRecursive(pts, mid+1, hi, nextAxis, nodeSize)
}

func quickselectX(pts []KDPoint, lo, hi, k int) {
	for lo < hi {
		pivot := pts[(lo+hi)/2].X
		i, j := lo, hi
		for i <= j {
			for pts[i].X < pivot {
				i++
			}
			for pts[j].X > pivot {
				j--
			}
			if i <= j {
				pts[i], pts[j] = pts[j], pts[i]
				i++
				j--
			}
		}
		if k <= j {
			hi = j
		} else if k >= i {
			lo = i
		} else {
			return
		}
	}
}

func quickselectY(pts []KDPoint, lo, hi, k int) {
	for lo < hi {
		pivot := pts[(lo+hi)/2].Y
		i, j := lo, hi
		for i <= j {
			for pts[i].Y < pivot {
				i++
			}
			for pts[j].Y > pivot {
				j--
			}
			if i <= j {
				pts[i], pts[j] = pts[j], pts[i]
				i++
				j--
			}
		}
		if k <= j {
			hi = j
		} else if k >= i {
			lo = i
		} else {
			return
		}
	}
}

// BuildSkeletonWithRemap is like BuildSkeleton but additionally reassigns
// point ids to sequential leaf-order ids (1..N) and returns a remap slice
// where remap[i] is the original (external) id of the i-th internal point.
//
// The input slice is mutated: ID fields are overwritten with internal ids.
func BuildSkeletonWithRemap(points []KDPoint, nodeSize int) (*SkeletonTree, []uint32) {
	remap := make([]uint32, len(points))
	for i := range points {
		remap[i] = points[i].ID
		points[i].ID = uint32(i + 1) // 1-indexed
	}
	return BuildSkeleton(points, nodeSize), remap
}

// RangeLeaves returns leaf indexes whose bounds intersect the given viewport.
// Walks the tree depth-first from the root (last node, since BuildSkeleton
// emits the root last).
func (t *SkeletonTree) RangeLeaves(viewport KDBounds) []int32 {
	if len(t.Nodes) == 0 {
		return nil
	}
	out := make([]int32, 0, 32)
	var walk func(nodeIdx int32)
	walk = func(nodeIdx int32) {
		n := &t.Nodes[nodeIdx]
		if !intersects(n.Bounds, viewport) {
			return
		}
		if n.LeafIdx >= 0 {
			out = append(out, n.LeafIdx)
			return
		}
		if n.Left >= 0 {
			walk(n.Left)
		}
		if n.Right >= 0 {
			walk(n.Right)
		}
	}
	walk(int32(len(t.Nodes) - 1))
	return out
}

func intersects(a, b KDBounds) bool {
	return a.MinX <= b.MaxX && a.MaxX >= b.MinX &&
		a.MinY <= b.MaxY && a.MaxY >= b.MinY
}
