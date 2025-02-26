package cluster

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
)

// SaveCompressed saves the cluster state to a compressed file
func (sc *Supercluster) SaveCompressed(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Create progress reporter
	progress := &ProgressReporter{
		totalPoints: len(sc.Tree.Points),
		logEnabled:  sc.Options.Log,
	}
	progress.Start("Saving to compressed file")
	defer progress.Finish()

	// Use large buffer for better performance
	bufWriter := bufio.NewWriterSize(file, 8*1024*1024)
	enc, err := zstd.NewWriter(bufWriter,
		zstd.WithEncoderLevel(zstd.SpeedBestCompression))
	if err != nil {
		return fmt.Errorf("failed to create zstd writer: %v", err)
	}
	defer enc.Close()

	// Write file format version
	binary.Write(enc, binary.LittleEndian, uint32(1)) // Version 1

	// Write header and sizes
	binary.Write(enc, binary.LittleEndian, uint32(len(sc.Tree.Nodes)))
	binary.Write(enc, binary.LittleEndian, uint32(len(sc.Tree.Points)))

	// Write Options
	binary.Write(enc, binary.LittleEndian, int32(sc.Options.MinZoom))
	binary.Write(enc, binary.LittleEndian, int32(sc.Options.MaxZoom))
	binary.Write(enc, binary.LittleEndian, int32(sc.Options.MinPoints))
	binary.Write(enc, binary.LittleEndian, float64(sc.Options.Radius))
	binary.Write(enc, binary.LittleEndian, int32(sc.Options.NodeSize))
	binary.Write(enc, binary.LittleEndian, int32(sc.Options.Extent))
	binary.Write(enc, binary.LittleEndian, sc.Options.Log)

	// Write nodes
	progress.SetStage("Saving tree nodes")
	for i, node := range sc.Tree.Nodes {
		binary.Write(enc, binary.LittleEndian, node.PointIdx)
		binary.Write(enc, binary.LittleEndian, node.Left)
		binary.Write(enc, binary.LittleEndian, node.Right)
		binary.Write(enc, binary.LittleEndian, node.Axis)
		binary.Write(enc, binary.LittleEndian, node.MinChild)
		binary.Write(enc, binary.LittleEndian, node.MaxChild)

		// Write node bounds
		binary.Write(enc, binary.LittleEndian, node.Bounds.MinX)
		binary.Write(enc, binary.LittleEndian, node.Bounds.MinY)
		binary.Write(enc, binary.LittleEndian, node.Bounds.MaxX)
		binary.Write(enc, binary.LittleEndian, node.Bounds.MaxY)

		if i > 0 && i%10000 == 0 {
			progress.Update(i, len(sc.Tree.Nodes))
		}
	}

	// Write points in batches to keep memory usage low
	progress.SetStage("Saving point data")
	batchSize := 100000
	numPoints := len(sc.Tree.Points)
	numBatches := (numPoints + batchSize - 1) / batchSize

	for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
		start := batchIdx * batchSize
		end := (batchIdx + 1) * batchSize
		if end > numPoints {
			end = numPoints
		}

		for i := start; i < end; i++ {
			point := sc.Tree.Points[i]
			binary.Write(enc, binary.LittleEndian, point.X)
			binary.Write(enc, binary.LittleEndian, point.Y)
			binary.Write(enc, binary.LittleEndian, point.ID)
			binary.Write(enc, binary.LittleEndian, point.NumPoints)
		}

		progress.Update(end, numPoints)
	}

	// Serialize metadata store
	progress.SetStage("Saving metadata")
	sc.metadataStore.mu.RLock()
	// Write key mapping
	binary.Write(enc, binary.LittleEndian, uint32(len(sc.metadataStore.idToKey)))
	for _, key := range sc.metadataStore.idToKey {
		keyBytes := []byte(key)
		binary.Write(enc, binary.LittleEndian, uint32(len(keyBytes)))
		enc.Write(keyBytes)
	}

	// Write point metadata entries
	binary.Write(enc, binary.LittleEndian, uint32(len(sc.metadataStore.pointMeta)))

	// Process metadata in batches
	metaCount := 0
	for pointID, entries := range sc.metadataStore.pointMeta {
		binary.Write(enc, binary.LittleEndian, pointID)
		binary.Write(enc, binary.LittleEndian, uint32(len(entries)))

		for _, entry := range entries {
			binary.Write(enc, binary.LittleEndian, entry.Key)
			binary.Write(enc, binary.LittleEndian, entry.Value.Type)

			switch entry.Value.Type {
			case 0: // string
				strBytes := []byte(entry.Value.StringVal)
				binary.Write(enc, binary.LittleEndian, uint32(len(strBytes)))
				enc.Write(strBytes)
			case 1: // number
				binary.Write(enc, binary.LittleEndian, entry.Value.NumVal)
			case 2: // bool
				binary.Write(enc, binary.LittleEndian, entry.Value.BoolVal)
			}
		}

		metaCount++
		if metaCount%10000 == 0 {
			progress.Update(metaCount, len(sc.metadataStore.pointMeta))
		}
	}
	sc.metadataStore.mu.RUnlock()

	// Serialize metrics store
	progress.SetStage("Saving metrics")
	sc.metricsStore.mu.RLock()
	// Write keys
	binary.Write(enc, binary.LittleEndian, uint32(len(sc.metricsStore.keys)))
	for _, key := range sc.metricsStore.keys {
		keyBytes := []byte(key)
		binary.Write(enc, binary.LittleEndian, uint32(len(keyBytes)))
		enc.Write(keyBytes)
	}

	// Write values by column
	for i, col := range sc.metricsStore.columns {
		binary.Write(enc, binary.LittleEndian, uint32(len(col)))

		// Write column in chunks to avoid large memory allocations
		chunkSize := 50000
		for offset := 0; offset < len(col); offset += chunkSize {
			end := offset + chunkSize
			if end > len(col) {
				end = len(col)
			}

			for _, val := range col[offset:end] {
				binary.Write(enc, binary.LittleEndian, val)
			}
		}

		progress.Update(i+1, len(sc.metricsStore.columns))
	}

	// Write point mapping
	binary.Write(enc, binary.LittleEndian, uint32(len(sc.metricsStore.pointToRow)))

	mappingCount := 0
	for pointID, rowIdx := range sc.metricsStore.pointToRow {
		binary.Write(enc, binary.LittleEndian, pointID)
		binary.Write(enc, binary.LittleEndian, int32(rowIdx))

		mappingCount++
		if mappingCount%50000 == 0 {
			progress.Update(mappingCount, len(sc.metricsStore.pointToRow))
		}
	}
	sc.metricsStore.mu.RUnlock()

	// Close and flush
	progress.SetStage("Finalizing file")
	if err := enc.Close(); err != nil {
		return fmt.Errorf("failed to close encoder: %v", err)
	}

	if err := bufWriter.Flush(); err != nil {
		return fmt.Errorf("failed to flush buffer: %v", err)
	}

	return nil
}

// LoadCompressedSupercluster loads a cluster from a compressed file
func LoadCompressedSupercluster(filename string) (*Supercluster, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Create progress reporter
	progress := &ProgressReporter{
		logEnabled: true, // Always enable logging for loading
	}
	progress.Start("Loading from compressed file")
	defer progress.Finish()

	dec, err := zstd.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create zstd reader: %v", err)
	}
	defer dec.Close()

	// Read version
	var version uint32
	binary.Read(dec, binary.LittleEndian, &version)
	if version != 1 {
		return nil, fmt.Errorf("unsupported file format version: %d", version)
	}

	// Read sizes
	var numNodes, numPoints uint32
	binary.Read(dec, binary.LittleEndian, &numNodes)
	binary.Read(dec, binary.LittleEndian, &numPoints)

	progress.totalPoints = int(numPoints)

	// Read options
	var options SuperclusterOptions
	binary.Read(dec, binary.LittleEndian, &options.MinZoom)
	binary.Read(dec, binary.LittleEndian, &options.MaxZoom)
	binary.Read(dec, binary.LittleEndian, &options.MinPoints)
	binary.Read(dec, binary.LittleEndian, &options.Radius)
	binary.Read(dec, binary.LittleEndian, &options.NodeSize)
	binary.Read(dec, binary.LittleEndian, &options.Extent)
	binary.Read(dec, binary.LittleEndian, &options.Log)

	// Create cluster with options
	sc := NewSupercluster(options)

	// Pre-allocate slices with exact sizes
	progress.SetStage("Loading tree nodes")
	nodes := make([]KDNode, numNodes)

	// Read nodes
	for i := range nodes {
		binary.Read(dec, binary.LittleEndian, &nodes[i].PointIdx)
		binary.Read(dec, binary.LittleEndian, &nodes[i].Left)
		binary.Read(dec, binary.LittleEndian, &nodes[i].Right)
		binary.Read(dec, binary.LittleEndian, &nodes[i].Axis)
		binary.Read(dec, binary.LittleEndian, &nodes[i].MinChild)
		binary.Read(dec, binary.LittleEndian, &nodes[i].MaxChild)

		// Read node bounds
		binary.Read(dec, binary.LittleEndian, &nodes[i].Bounds.MinX)
		binary.Read(dec, binary.LittleEndian, &nodes[i].Bounds.MinY)
		binary.Read(dec, binary.LittleEndian, &nodes[i].Bounds.MaxX)
		binary.Read(dec, binary.LittleEndian, &nodes[i].Bounds.MaxY)

		if i > 0 && i%10000 == 0 {
			progress.Update(i, int(numNodes))
		}
	}

	// Read points in batches to reduce memory pressure
	progress.SetStage("Loading points")
	batchSize := 500000
	numBatches := (int(numPoints) + batchSize - 1) / batchSize
	points := make([]KDPoint, numPoints)

	for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
		start := batchIdx * batchSize
		end := (batchIdx + 1) * batchSize
		if end > int(numPoints) {
			end = int(numPoints)
		}

		for i := start; i < end; i++ {
			binary.Read(dec, binary.LittleEndian, &points[i].X)
			binary.Read(dec, binary.LittleEndian, &points[i].Y)
			binary.Read(dec, binary.LittleEndian, &points[i].ID)
			binary.Read(dec, binary.LittleEndian, &points[i].NumPoints)
		}

		progress.Update(end, int(numPoints))

		// Force GC between batches to keep memory usage low
		if batchIdx < numBatches-1 {
			runtime.GC()
		}
	}

	// Create reusable buffer for reading
	buf := make([]byte, 32*1024)

	// Load metadata store
	progress.SetStage("Loading metadata keys")
	var numKeys uint32
	binary.Read(dec, binary.LittleEndian, &numKeys)

	// Read keys
	for i := uint32(0); i < numKeys; i++ {
		var keyLen uint32
		binary.Read(dec, binary.LittleEndian, &keyLen)

		if int(keyLen) > len(buf) {
			buf = make([]byte, keyLen)
		}

		io.ReadFull(dec, buf[:keyLen])
		key := sc.metadataStore.stringPool.Intern(string(buf[:keyLen]))
		sc.metadataStore.keyToID[key] = MetadataKey(i)
		sc.metadataStore.idToKey = append(sc.metadataStore.idToKey, key)

		if i > 0 && i%10000 == 0 {
			progress.Update(int(i), int(numKeys))
		}
	}
	sc.metadataStore.nextKeyID = MetadataKey(numKeys)

	// Read point metadata
	progress.SetStage("Loading point metadata")
	var numPointMetadata uint32
	binary.Read(dec, binary.LittleEndian, &numPointMetadata)

	for i := uint32(0); i < numPointMetadata; i++ {
		var pointID uint32
		binary.Read(dec, binary.LittleEndian, &pointID)

		var numEntries uint32
		binary.Read(dec, binary.LittleEndian, &numEntries)

		entries := make([]MetadataEntry, numEntries)

		for j := uint32(0); j < numEntries; j++ {
			binary.Read(dec, binary.LittleEndian, &entries[j].Key)
			binary.Read(dec, binary.LittleEndian, &entries[j].Value.Type)

			switch entries[j].Value.Type {
			case 0: // string
				var strLen uint32
				binary.Read(dec, binary.LittleEndian, &strLen)

				if int(strLen) > len(buf) {
					buf = make([]byte, strLen)
				}

				io.ReadFull(dec, buf[:strLen])
				entries[j].Value.StringVal = sc.metadataStore.stringPool.Intern(string(buf[:strLen]))

			case 1: // number
				binary.Read(dec, binary.LittleEndian, &entries[j].Value.NumVal)

			case 2: // bool
				binary.Read(dec, binary.LittleEndian, &entries[j].Value.BoolVal)
			}
		}

		sc.metadataStore.pointMeta[pointID] = entries

		if i > 0 && i%10000 == 0 {
			progress.Update(int(i), int(numPointMetadata))
		}
	}

	// Periodic GC to keep memory usage stable
	runtime.GC()

	// Load metrics store
	progress.SetStage("Loading metrics keys")
	var numMetricKeys uint32
	binary.Read(dec, binary.LittleEndian, &numMetricKeys)

	// Read metric keys
	for i := uint32(0); i < numMetricKeys; i++ {
		var keyLen uint32
		binary.Read(dec, binary.LittleEndian, &keyLen)

		if int(keyLen) > len(buf) {
			buf = make([]byte, keyLen)
		}

		io.ReadFull(dec, buf[:keyLen])
		key := sc.metricsStore.stringPool.Intern(string(buf[:keyLen]))

		sc.metricsStore.keys = append(sc.metricsStore.keys, key)
		sc.metricsStore.keyToColumn[key] = int(i)

		if i > 0 && i%1000 == 0 {
			progress.Update(int(i), int(numMetricKeys))
		}
	}

	// Read value columns
	progress.SetStage("Loading metrics values")
	sc.metricsStore.columns = make([][]float32, numMetricKeys)
	for i := uint32(0); i < numMetricKeys; i++ {
		var colSize uint32
		binary.Read(dec, binary.LittleEndian, &colSize)

		column := make([]float32, colSize)

		// Read column in chunks to reduce memory pressure
		chunkSize := 50000
		for offset := uint32(0); offset < colSize; offset += uint32(chunkSize) {
			end := offset + uint32(chunkSize)
			if end > colSize {
				end = colSize
			}

			for j := offset; j < end; j++ {
				binary.Read(dec, binary.LittleEndian, &column[j])
			}
		}

		sc.metricsStore.columns[i] = column
		progress.Update(int(i+1), int(numMetricKeys))
	}

	// Read point to row mapping
	progress.SetStage("Loading metrics mapping")
	var numPointMetrics uint32
	binary.Read(dec, binary.LittleEndian, &numPointMetrics)

	for i := uint32(0); i < numPointMetrics; i++ {
		var pointID uint32
		var rowIdx int32
		binary.Read(dec, binary.LittleEndian, &pointID)
		binary.Read(dec, binary.LittleEndian, &rowIdx)

		sc.metricsStore.pointToRow[pointID] = int(rowIdx)

		if i > 0 && i%50000 == 0 {
			progress.Update(int(i), int(numPointMetrics))
		}
	}

	progress.SetStage("Building tree")
	// Build tree
	sc.Tree = &KDTree{
		Nodes:    nodes,
		Points:   points,
		NodeSize: options.NodeSize,
	}

	// Calculate tree bounds if needed
	if len(points) > 0 {
		bounds := KDBounds{
			MinX: points[0].X,
			MinY: points[0].Y,
			MaxX: points[0].X,
			MaxY: points[0].Y,
		}

		// Calculate bounds in batches
		batchSize := 100000
		for i := 0; i < len(points); i += batchSize {
			end := i + batchSize
			if end > len(points) {
				end = len(points)
			}

			for j := i; j < end; j++ {
				bounds.Extend(points[j].X, points[j].Y)
			}

			progress.Update(end, len(points))
		}

		sc.Tree.Bounds = bounds
	}

	// Create original points slice (only IDs and coordinates)
	sc.Points = make([]Point, len(points))
	for i, p := range points {
		sc.Points[i] = Point{
			ID: p.ID,
			X:  p.X,
			Y:  p.Y,
		}
	}

	return sc, nil
}

// ProgressReporter provides simple progress reporting for long operations
type ProgressReporter struct {
	totalPoints int
	stage       string
	startTime   int64
	lastUpdate  int64
	logEnabled  bool
	mu          sync.Mutex
}

// Start begins tracking progress for an operation
func (p *ProgressReporter) Start(stage string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.stage = stage
	p.startTime = currentTimeMillis()
	p.lastUpdate = p.startTime

	if p.logEnabled {
		fmt.Printf("[%s] Starting...\n", p.stage)
	}
}

// SetStage updates the current operation stage
func (p *ProgressReporter) SetStage(stage string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.stage = stage
	now := currentTimeMillis()
	p.lastUpdate = now

	if p.logEnabled {
		fmt.Printf("[%s] Starting...\n", p.stage)
	}
}

// Update reports progress on the current operation
func (p *ProgressReporter) Update(current, total int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	now := currentTimeMillis()

	// Only update if sufficient time has passed (250ms)
	if now-p.lastUpdate < 250 {
		return
	}

	p.lastUpdate = now

	if p.logEnabled {
		percentage := int((float64(current) / float64(total)) * 100)
		fmt.Printf("[%s] %d/%d (%d%%)\n", p.stage, current, total, percentage)
	}
}

// Finish completes progress tracking
func (p *ProgressReporter) Finish() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.logEnabled {
		elapsed := (currentTimeMillis() - p.startTime) / 1000
		fmt.Printf("[%s] Completed in %d seconds\n", p.stage, elapsed)
	}
}

// currentTimeMillis returns the current time in milliseconds
func currentTimeMillis() int64 {
	return timeNow().UnixNano() / int64(1000000)
}

// timeNow is a function variable that returns the current time
// It can be replaced in tests
var timeNow = func() time.Time {
	return time.Now()
}
