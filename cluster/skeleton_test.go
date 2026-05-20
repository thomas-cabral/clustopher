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
