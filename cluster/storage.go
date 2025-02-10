package cluster

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/klauspost/compress/zstd"
)

func (sc *Supercluster) SaveCompressed(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return fmt.Errorf("failed to create file: %v", err)
    }
    defer file.Close()

    bufWriter := bufio.NewWriterSize(file, 1024*1024)
    enc, err := zstd.NewWriter(bufWriter,
        zstd.WithEncoderLevel(zstd.SpeedBestCompression))
    if err != nil {
        return fmt.Errorf("failed to create zstd writer: %v", err)
    }
    defer enc.Close()

    // Write sizes first for allocation
    binary.Write(enc, binary.LittleEndian, uint32(len(sc.Tree.Nodes)))
    binary.Write(enc, binary.LittleEndian, uint32(len(sc.Tree.Points)))
    binary.Write(enc, binary.LittleEndian, uint32(len(sc.Tree.Pool.Metrics)))
    
    // Write Options
    binary.Write(enc, binary.LittleEndian, sc.Options.MinZoom)
    binary.Write(enc, binary.LittleEndian, sc.Options.MaxZoom)
    binary.Write(enc, binary.LittleEndian, sc.Options.MinPoints)
    binary.Write(enc, binary.LittleEndian, float64(sc.Options.Radius))
    binary.Write(enc, binary.LittleEndian, sc.Options.NodeSize)
    binary.Write(enc, binary.LittleEndian, sc.Options.Extent)

    // Write nodes
    for _, node := range sc.Tree.Nodes {
        binary.Write(enc, binary.LittleEndian, node.PointIdx)
        binary.Write(enc, binary.LittleEndian, node.Left)
        binary.Write(enc, binary.LittleEndian, node.Right)
        binary.Write(enc, binary.LittleEndian, node.Axis)
        binary.Write(enc, binary.LittleEndian, node.MinChild)
        binary.Write(enc, binary.LittleEndian, node.MaxChild)
    }

    // Write points
    for _, point := range sc.Tree.Points {
        binary.Write(enc, binary.LittleEndian, point.X)
        binary.Write(enc, binary.LittleEndian, point.Y)
        binary.Write(enc, binary.LittleEndian, point.ID)
        binary.Write(enc, binary.LittleEndian, point.NumPoints)
        binary.Write(enc, binary.LittleEndian, point.MetricIdx)
        
        // Write metadata size
        binary.Write(enc, binary.LittleEndian, uint32(len(point.Metadata)))
        
        // Write each metadata key-value pair
        for k, v := range point.Metadata {
            // Write key
            keyBytes := []byte(k)
            binary.Write(enc, binary.LittleEndian, uint32(len(keyBytes)))
            enc.Write(keyBytes)
            
            // Convert value to JSON bytes
            valueBytes, err := json.Marshal(v)
            if err != nil {
                return fmt.Errorf("failed to marshal metadata value: %v", err)
            }
            
            // Write value
            binary.Write(enc, binary.LittleEndian, uint32(len(valueBytes)))
            enc.Write(valueBytes)
        }
    }

    // Write metrics
    for _, metrics := range sc.Tree.Pool.Metrics {
        binary.Write(enc, binary.LittleEndian, uint32(len(metrics)))
        for k, v := range metrics {
            binary.Write(enc, binary.LittleEndian, uint32(len(k)))
            enc.Write([]byte(k))
            binary.Write(enc, binary.LittleEndian, v)
        }
    }

    if err := enc.Close(); err != nil {
        return fmt.Errorf("failed to close encoder: %v", err)
    }

    if err := bufWriter.Flush(); err != nil {
        return fmt.Errorf("failed to flush buffer: %v", err)
    }

    return nil
}

func LoadCompressedSupercluster(filename string) (*Supercluster, error) {
    start := time.Now()
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open file: %v", err)
    }
    defer file.Close()

    bufReader := bufio.NewReaderSize(file, 1024*1024)
    dec, err := zstd.NewReader(bufReader)
    if err != nil {
        return nil, fmt.Errorf("failed to create zstd reader: %v", err)
    }
    defer dec.Close()

    // Read sizes
    var numNodes, numPoints, numMetrics uint32
    binary.Read(dec, binary.LittleEndian, &numNodes)
    binary.Read(dec, binary.LittleEndian, &numPoints)
    binary.Read(dec, binary.LittleEndian, &numMetrics)

    // Read options
    var options SuperclusterOptions
    binary.Read(dec, binary.LittleEndian, &options.MinZoom)
    binary.Read(dec, binary.LittleEndian, &options.MaxZoom)
    binary.Read(dec, binary.LittleEndian, &options.MinPoints)
    binary.Read(dec, binary.LittleEndian, &options.Radius)
    binary.Read(dec, binary.LittleEndian, &options.NodeSize)
    binary.Read(dec, binary.LittleEndian, &options.Extent)

    // Create cluster with options
    sc := NewSupercluster(options)

    // Read nodes
    nodes := make([]KDNode, numNodes)
    for i := range nodes {
        binary.Read(dec, binary.LittleEndian, &nodes[i].PointIdx)
        binary.Read(dec, binary.LittleEndian, &nodes[i].Left)
        binary.Read(dec, binary.LittleEndian, &nodes[i].Right)
        binary.Read(dec, binary.LittleEndian, &nodes[i].Axis)
        binary.Read(dec, binary.LittleEndian, &nodes[i].MinChild)
        binary.Read(dec, binary.LittleEndian, &nodes[i].MaxChild)
    }

    fmt.Printf("Nodes read took: %v\n", time.Since(start))
    pointsStart := time.Now()

    // Read points
    points := make([]KDPoint, numPoints)
    for i := range points {
        binary.Read(dec, binary.LittleEndian, &points[i].X)
        binary.Read(dec, binary.LittleEndian, &points[i].Y)
        binary.Read(dec, binary.LittleEndian, &points[i].ID)
        binary.Read(dec, binary.LittleEndian, &points[i].NumPoints)
        binary.Read(dec, binary.LittleEndian, &points[i].MetricIdx)
        
        // Read metadata
        var metadataSize uint32
        binary.Read(dec, binary.LittleEndian, &metadataSize)
        
        points[i].Metadata = make(map[string]interface{}, metadataSize)
        
        // Read each metadata key-value pair
        for j := uint32(0); j < metadataSize; j++ {
            // Read key
            var keySize uint32
            binary.Read(dec, binary.LittleEndian, &keySize)
            keyBytes := make([]byte, keySize)
            io.ReadFull(dec, keyBytes)
            
            // Read value
            var valueSize uint32
            binary.Read(dec, binary.LittleEndian, &valueSize)
            valueBytes := make([]byte, valueSize)
            io.ReadFull(dec, valueBytes)
            
            // Unmarshal value
            var value interface{}
            if err := json.Unmarshal(valueBytes, &value); err != nil {
                return nil, fmt.Errorf("failed to unmarshal metadata value: %v", err)
            }
            
            points[i].Metadata[string(keyBytes)] = value
        }
    }

    fmt.Printf("Points read took: %v\n", time.Since(pointsStart))
    metricsStart := time.Now()

    // Read metrics pool
    metricsPool := NewMetricsPool()
    metricsPool.Metrics = make([]map[string]float32, numMetrics)
    
    for i := range metricsPool.Metrics {
        var numPairs uint32
        binary.Read(dec, binary.LittleEndian, &numPairs)
        
        metrics := make(map[string]float32, numPairs)
        for j := uint32(0); j < numPairs; j++ {
            var keyLen uint32
            binary.Read(dec, binary.LittleEndian, &keyLen)
            
            keyBytes := make([]byte, keyLen)
            io.ReadFull(dec, keyBytes)
            
            var value float32
            binary.Read(dec, binary.LittleEndian, &value)
            
            metrics[string(keyBytes)] = value
        }
        metricsPool.Metrics[i] = metrics
    }

    fmt.Printf("Metrics read took: %v\n", time.Since(metricsStart))

    sc.Tree = &KDTree{
        Pool:     metricsPool,
        NodeSize: options.NodeSize,
        Nodes:    nodes,
        Points:   points,
    }

    fmt.Printf("Total load time: %v\n", time.Since(start))
    return sc, nil
}
