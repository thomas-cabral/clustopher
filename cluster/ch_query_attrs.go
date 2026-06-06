package cluster

import "encoding/json"

// attrArena is a compact, interned store for per-point metrics/metadata
// fetched by fetchLeafPoints. Profiling the z14 city viewport at 100M showed
// the previous map[uint32]pointAttrs representation cost ~30% of query CPU
// and ~37 MB of allocations per query inside attachAttrsFromMembers: one
// metrics map + one metadata map per point on the way in, and three scratch
// maps per emitted cluster on the way out.
//
// The arena interns key and value strings once, stores per-point attributes
// as contiguous {keyID, val} entries, and lets the aggregation loop run over
// dense slices with no per-cluster map allocations.
type attrArena struct {
	metricKeys []string
	metricIdx  map[string]uint16

	metaKeys    []string
	metaIdx     map[string]uint16
	metaVals    [][]string            // per meta key: valID -> value
	metaValIdx  []map[string]uint32   // per meta key: value -> valID
	metaValJSON [][]json.RawMessage   // per meta key: valID -> memoized JSON

	metricKV []metricKV
	metaKV   []metaKV
	spans    map[uint32]attrSpan // point id -> entry ranges
}

type metricKV struct {
	key uint16
	val float32
}

type metaKV struct {
	key uint16
	val uint32
}

type attrSpan struct {
	mOff, dOff uint32
	mLen, dLen uint16
}

func newAttrArena() *attrArena {
	return &attrArena{
		metricIdx: make(map[string]uint16),
		metaIdx:   make(map[string]uint16),
		spans:     make(map[uint32]attrSpan),
	}
}

func (a *attrArena) metricKey(k string) uint16 {
	if id, ok := a.metricIdx[k]; ok {
		return id
	}
	id := uint16(len(a.metricKeys))
	a.metricKeys = append(a.metricKeys, k)
	a.metricIdx[k] = id
	return id
}

func (a *attrArena) metaKey(k string) uint16 {
	if id, ok := a.metaIdx[k]; ok {
		return id
	}
	id := uint16(len(a.metaKeys))
	a.metaKeys = append(a.metaKeys, k)
	a.metaIdx[k] = id
	a.metaVals = append(a.metaVals, nil)
	a.metaValIdx = append(a.metaValIdx, make(map[string]uint32))
	a.metaValJSON = append(a.metaValJSON, nil)
	return id
}

func (a *attrArena) metaVal(key uint16, v string) uint32 {
	if id, ok := a.metaValIdx[key][v]; ok {
		return id
	}
	id := uint32(len(a.metaVals[key]))
	a.metaVals[key] = append(a.metaVals[key], v)
	a.metaValIdx[key][v] = id
	a.metaValJSON[key] = append(a.metaValJSON[key], nil)
	return id
}

// valJSON returns the JSON encoding of a meta value, memoized — metadata
// values repeat across thousands of clusters per query.
func (a *attrArena) valJSON(key uint16, val uint32) json.RawMessage {
	if a.metaValJSON[key][val] == nil {
		if b, err := json.Marshal(a.metaVals[key][val]); err == nil {
			a.metaValJSON[key][val] = b
		}
	}
	return a.metaValJSON[key][val]
}

// add records one point's attributes. The input maps are copied into the
// arena; callers may reuse or discard them.
func (a *attrArena) add(id uint32, metrics map[string]float32, metadata map[string]string) {
	if len(metrics) == 0 && len(metadata) == 0 {
		return
	}
	sp := attrSpan{mOff: uint32(len(a.metricKV)), dOff: uint32(len(a.metaKV))}
	for k, v := range metrics {
		a.metricKV = append(a.metricKV, metricKV{key: a.metricKey(k), val: v})
	}
	for k, v := range metadata {
		ki := a.metaKey(k)
		a.metaKV = append(a.metaKV, metaKV{key: ki, val: a.metaVal(ki, v)})
	}
	sp.mLen = uint16(uint32(len(a.metricKV)) - sp.mOff)
	sp.dLen = uint16(uint32(len(a.metaKV)) - sp.dOff)
	a.spans[id] = sp
}

// attachAttrsFromMembers fills each cluster's Metrics/Metadata by aggregating
// the per-member attributes recorded in the arena. Metrics are averaged so the
// per-cluster value matches the convention used by the rollup and
// aggregateLeaves paths (sum/cnt); metadata picks the most common value per
// key across members (ties resolved by first value to reach the top count).
// Clusters whose Children list is empty are left untouched.
//
// Scratch state is allocated once per call and reset between clusters via
// touched-entry lists, so per-cluster work allocates only the two small output
// maps actually attached to the ClusterNode.
func attachAttrsFromMembers(clusters []ClusterNode, attrs *attrArena) {
	if attrs == nil || len(attrs.spans) == 0 {
		return
	}
	nm := len(attrs.metricKeys)
	nd := len(attrs.metaKeys)

	sums := make([]float64, nm)
	counts := make([]uint64, nm)
	touchedM := make([]uint16, 0, nm)

	freq := make([][]int32, nd)
	for k := range freq {
		freq[k] = make([]int32, len(attrs.metaVals[k]))
	}
	touchedD := make([]metaKV, 0, 64)
	top := make([]uint32, nd)
	topCnt := make([]int32, nd)
	keyTouched := make([]bool, nd)
	touchedKeys := make([]uint16, 0, nd)

	for i := range clusters {
		members := clusters[i].Children
		if len(members) == 0 {
			continue
		}
		for _, k := range touchedM {
			sums[k] = 0
			counts[k] = 0
		}
		touchedM = touchedM[:0]
		for _, kv := range touchedD {
			freq[kv.key][kv.val] = 0
		}
		touchedD = touchedD[:0]
		for _, k := range touchedKeys {
			keyTouched[k] = false
			topCnt[k] = 0
		}
		touchedKeys = touchedKeys[:0]

		for _, id := range members {
			sp, ok := attrs.spans[id]
			if !ok {
				continue
			}
			for _, kv := range attrs.metricKV[sp.mOff : sp.mOff+uint32(sp.mLen)] {
				if counts[kv.key] == 0 {
					touchedM = append(touchedM, kv.key)
				}
				sums[kv.key] += float64(kv.val)
				counts[kv.key]++
			}
			for _, kv := range attrs.metaKV[sp.dOff : sp.dOff+uint32(sp.dLen)] {
				if freq[kv.key][kv.val] == 0 {
					touchedD = append(touchedD, kv)
				}
				if !keyTouched[kv.key] {
					keyTouched[kv.key] = true
					touchedKeys = append(touchedKeys, kv.key)
				}
				f := freq[kv.key][kv.val] + 1
				freq[kv.key][kv.val] = f
				if f > topCnt[kv.key] {
					topCnt[kv.key] = f
					top[kv.key] = kv.val
				}
			}
		}

		if len(touchedM) > 0 {
			metrics := make(map[string]float32, len(touchedM))
			for _, k := range touchedM {
				metrics[attrs.metricKeys[k]] = float32(sums[k] / float64(counts[k]))
			}
			clusters[i].Metrics = metrics
		}
		if len(touchedKeys) > 0 {
			meta := make(map[string]json.RawMessage, len(touchedKeys))
			for _, k := range touchedKeys {
				meta[attrs.metaKeys[k]] = attrs.valJSON(k, top[k])
			}
			clusters[i].Metadata = meta
		}
	}
}
