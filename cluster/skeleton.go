package cluster

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
