package cluster

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"unsafe"

	"github.com/edsrzf/mmap-go"
	"github.com/klauspost/compress/zstd"
)


// MMapWriter handles writing to memory-mapped files
type MMapWriter struct {
    data mmap.MMap
    offset int
}

func NewMMapWriter(data mmap.MMap) *MMapWriter {
    return &MMapWriter{
        data: data,
        offset: 0,
    }
}

func (w *MMapWriter) WriteUint32(v uint32) {
    binary.LittleEndian.PutUint32(w.data[w.offset:], v)
    w.offset += 4
}

func (w *MMapWriter) WriteFloat64(v float64) {
    binary.LittleEndian.PutUint64(w.data[w.offset:], math.Float64bits(v))
    w.offset += 8
}

func (w *MMapWriter) WriteFloat32(v float32) {
    binary.LittleEndian.PutUint32(w.data[w.offset:], math.Float32bits(v))
    w.offset += 4
}

func (w *MMapWriter) WriteBytes(b []byte) {
    copy(w.data[w.offset:], b)
    w.offset += len(b)
}

// MMapReader handles reading from memory-mapped files
type MMapReader struct {
    data mmap.MMap
    offset int
}

func NewMMapReader(data mmap.MMap) *MMapReader {
    return &MMapReader{
        data: data,
        offset: 0,
    }
}

func (r *MMapReader) ReadUint32() uint32 {
    v := binary.LittleEndian.Uint32(r.data[r.offset:])
    r.offset += 4
    return v
}

func (r *MMapReader) ReadFloat64() float64 {
    v := binary.LittleEndian.Uint64(r.data[r.offset:])
    r.offset += 8
    return math.Float64frombits(v)
}

func (r *MMapReader) ReadFloat32() float32 {
    v := binary.LittleEndian.Uint32(r.data[r.offset:])
    r.offset += 4
    return math.Float32frombits(v)
}

func (r *MMapReader) ReadBytes(n int) []byte {
    b := make([]byte, n)
    copy(b, r.data[r.offset:r.offset+n])
    r.offset += n
    return b
}

// calculateSize calculates total size needed for memory mapping
func (sc *Supercluster) calculateSize() int64 {
    size := int64(0)
    
    // Header sizes (3 uint32s)
    size += 12
    
    // Options
    size += int64(unsafe.Sizeof(sc.Options))
    
    // Nodes
    nodeSize := int64(unsafe.Sizeof(KDNode{}))
    size += nodeSize * int64(len(sc.Tree.Nodes))
    
    // Points
    for _, point := range sc.Tree.Points {
        size += 28 // Fixed fields (X, Y, ID, NumPoints, MetricIdx)
        
        // Metadata size
        size += 4
        for k, v := range point.Metadata {
            size += 4 + int64(len(k))
            valueBytes, _ := json.Marshal(v)
            size += 4 + int64(len(valueBytes))
        }
    }
    
    // Metrics
    for _, metrics := range sc.Tree.Pool.Metrics {
        size += 4 // size of map
        for k, _ := range metrics {
            size += 4 + int64(len(k)) + 4 // key length + key + float32
        }
    }
    
    return size
}

func (sc *Supercluster) SaveMMap(filename string) error {
    // Calculate required size
    size := sc.calculateSize()
    
    // Create and truncate file
    file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0644)
    if err != nil {
        return fmt.Errorf("failed to create file: %v", err)
    }
    defer file.Close()
    
    if err := file.Truncate(size); err != nil {
        return fmt.Errorf("failed to truncate file: %v", err)
    }
    
    // Memory map the file
    mmapData, err := mmap.Map(file, mmap.RDWR, 0)
    if err != nil {
        return fmt.Errorf("failed to mmap file: %v", err)
    }
    defer mmapData.Unmap()
    
    writer := NewMMapWriter(mmapData)
    
    // Write sizes
    writer.WriteUint32(uint32(len(sc.Tree.Nodes)))
    writer.WriteUint32(uint32(len(sc.Tree.Points)))
    writer.WriteUint32(uint32(len(sc.Tree.Pool.Metrics)))
    
    // Write Options
    writer.WriteUint32(uint32(sc.Options.MinZoom))
    writer.WriteUint32(uint32(sc.Options.MaxZoom))
    writer.WriteUint32(uint32(sc.Options.MinPoints))
    writer.WriteFloat64(float64(sc.Options.Radius))
    writer.WriteUint32(uint32(sc.Options.NodeSize))
    writer.WriteUint32(uint32(sc.Options.Extent))
    
    // Write nodes
    for _, node := range sc.Tree.Nodes {
        writer.WriteUint32(uint32(node.PointIdx))
        writer.WriteUint32(uint32(node.Left))
        writer.WriteUint32(uint32(node.Right))
        writer.WriteUint32(uint32(node.Axis))
        writer.WriteUint32(uint32(node.MinChild))
        writer.WriteUint32(uint32(node.MaxChild))
    }
    
    // Write points
    for _, point := range sc.Tree.Points {
        writer.WriteFloat64(float64(point.X))
        writer.WriteFloat64(float64(point.Y))
        writer.WriteUint32(uint32(point.ID))
        writer.WriteUint32(uint32(point.NumPoints))
        writer.WriteUint32(uint32(point.MetricIdx))
        
        // Write metadata
        writer.WriteUint32(uint32(len(point.Metadata)))
        for k, v := range point.Metadata {
            // Write key
            writer.WriteUint32(uint32(len(k)))
            writer.WriteBytes([]byte(k))
            
            // Write value
            valueBytes, _ := json.Marshal(v)
            writer.WriteUint32(uint32(len(valueBytes)))
            writer.WriteBytes(valueBytes)
        }
    }
    
    // Write metrics
    for _, metrics := range sc.Tree.Pool.Metrics {
        writer.WriteUint32(uint32(len(metrics)))
        for k, v := range metrics {
            writer.WriteUint32(uint32(len(k)))
            writer.WriteBytes([]byte(k))
            writer.WriteFloat32(v)
        }
    }
    
    return mmapData.Flush()
}

func LoadMMapSupercluster(filename string) (*Supercluster, error) {
    file, err := os.OpenFile(filename, os.O_RDWR, 0644)
    if err != nil {
        return nil, fmt.Errorf("failed to open file: %v", err)
    }
    defer file.Close()
    
    // Memory map the file
    mmapData, err := mmap.Map(file, mmap.RDWR, 0)
    if err != nil {
        return nil, fmt.Errorf("failed to mmap file: %v", err)
    }
    defer mmapData.Unmap()
    
    reader := NewMMapReader(mmapData)
    
    // Read sizes
    numNodes := reader.ReadUint32()
    numPoints := reader.ReadUint32()
    numMetrics := reader.ReadUint32()
    
    // Read options
    options := SuperclusterOptions{
        MinZoom:   int(reader.ReadUint32()),
        MaxZoom:   int(reader.ReadUint32()),
        MinPoints: int(reader.ReadUint32()),
        Radius:    float64(reader.ReadFloat64()),
        NodeSize:  int(reader.ReadUint32()),
        Extent:    int(reader.ReadUint32()),
    }
    
    // Create cluster with options
    sc := NewSupercluster(options)
    
    // Read nodes
    nodes := make([]KDNode, numNodes)
    for i := range nodes {
        nodes[i] = KDNode{
            PointIdx: int32(reader.ReadUint32()),
            Left:     int32(reader.ReadUint32()),
            Right:    int32(reader.ReadUint32()),
            Axis:     uint8(reader.ReadUint32()),
            MinChild: uint32(reader.ReadUint32()),
            MaxChild: uint32(reader.ReadUint32()),
        }
    }
    
    // Read points
    points := make([]KDPoint, numPoints)
    for i := range points {
        points[i].X = float32(reader.ReadFloat64())
        points[i].Y = float32(reader.ReadFloat64())
        points[i].ID = uint32(reader.ReadUint32())
        points[i].NumPoints = uint32(reader.ReadUint32())
        points[i].MetricIdx = uint32(reader.ReadUint32())
        
        // Read metadata
        metadataSize := reader.ReadUint32()
        points[i].Metadata = make(map[string]interface{}, metadataSize)
        
        for j := uint32(0); j < metadataSize; j++ {
            // Read key
            keySize := reader.ReadUint32()
            key := string(reader.ReadBytes(int(keySize)))
            
            // Read value
            valueSize := reader.ReadUint32()
            valueBytes := reader.ReadBytes(int(valueSize))
            
            var value interface{}
            if err := json.Unmarshal(valueBytes, &value); err != nil {
                return nil, fmt.Errorf("failed to unmarshal metadata value: %v", err)
            }
            
            points[i].Metadata[key] = value
        }
    }
    
    // Read metrics pool
    metricsPool := NewMetricsPool()
    metricsPool.Metrics = make([]map[string]float32, numMetrics)
    
    for i := range metricsPool.Metrics {
        numPairs := reader.ReadUint32()
        metrics := make(map[string]float32, numPairs)
        
        for j := uint32(0); j < numPairs; j++ {
            keySize := reader.ReadUint32()
            key := string(reader.ReadBytes(int(keySize)))
            value := reader.ReadFloat32()
            metrics[key] = value
        }
        
        metricsPool.Metrics[i] = metrics
    }
    
    sc.Tree = &KDTree{
        Pool:     metricsPool,
        NodeSize: options.NodeSize,
        Nodes:    nodes,
        Points:   points,
    }
    
    return sc, nil
}

func (sc *Supercluster) SaveCompressedMMap(filename string) error {
    // First save to temporary mmap file
    tempFile := filename + ".tmp"
    if err := sc.SaveMMap(tempFile); err != nil {
        return fmt.Errorf("failed to save mmap: %v", err)
    }
    defer os.Remove(tempFile)

    // Now compress the mmap file
    src, err := os.Open(tempFile)
    if err != nil {
        return fmt.Errorf("failed to open temp file: %v", err)
    }
    defer src.Close()

    dst, err := os.Create(filename)
    if err != nil {
        return fmt.Errorf("failed to create compressed file: %v", err)
    }
    defer dst.Close()

    enc, err := zstd.NewWriter(dst,
        zstd.WithEncoderLevel(zstd.SpeedBestCompression))
    if err != nil {
        return fmt.Errorf("failed to create zstd writer: %v", err)
    }
    defer enc.Close()

    _, err = io.Copy(enc, src)
    if err != nil {
        return fmt.Errorf("failed to compress data: %v", err)
    }

    return nil
}

func LoadCompressedMMap(filename string) (*Supercluster, error) {
    // Create temporary file for decompressed data
    tempFile := filename + ".tmp"
    dst, err := os.Create(tempFile)
    if err != nil {
        return nil, fmt.Errorf("failed to create temp file: %v", err)
    }
    defer os.Remove(tempFile)
    defer dst.Close()

    // Open compressed file
    src, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open compressed file: %v", err)
    }
    defer src.Close()

    // Create decompressor
    dec, err := zstd.NewReader(src)
    if err != nil {
        return nil, fmt.Errorf("failed to create zstd reader: %v", err)
    }
    defer dec.Close()

    // Decompress to temp file
    if _, err := io.Copy(dst, dec); err != nil {
        return nil, fmt.Errorf("failed to decompress data: %v", err)
    }

    // Sync to ensure all data is written
    if err := dst.Sync(); err != nil {
        return nil, fmt.Errorf("failed to sync temp file: %v", err)
    }

    // Now load using mmap
    return LoadMMapSupercluster(tempFile)
}