package cluster

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"

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
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	dec, err := zstd.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create zstd reader: %v", err)
	}
	defer dec.Close()

	// Read sizes first
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

	// Pre-allocate slices with exact sizes
	nodes := make([]KDNode, numNodes)
	points := make([]KDPoint, 0, numPoints)
	metricsPool := &MetricsPool{
		Metrics: make([]map[string]float32, 0, numMetrics),
		Lookup:  make(map[string]int),
	}

	// Read nodes in chunks
	buf := readBufferPool.Get().([]byte)
	defer readBufferPool.Put(buf)

	for i := range nodes {
		binary.Read(dec, binary.LittleEndian, &nodes[i].PointIdx)
		binary.Read(dec, binary.LittleEndian, &nodes[i].Left)
		binary.Read(dec, binary.LittleEndian, &nodes[i].Right)
		binary.Read(dec, binary.LittleEndian, &nodes[i].Axis)
		binary.Read(dec, binary.LittleEndian, &nodes[i].MinChild)
		binary.Read(dec, binary.LittleEndian, &nodes[i].MaxChild)
	}

	// Read points in chunks
	for i := uint32(0); i < numPoints; i++ {
		var point KDPoint
		binary.Read(dec, binary.LittleEndian, &point.X)
		binary.Read(dec, binary.LittleEndian, &point.Y)
		binary.Read(dec, binary.LittleEndian, &point.ID)
		binary.Read(dec, binary.LittleEndian, &point.NumPoints)
		binary.Read(dec, binary.LittleEndian, &point.MetricIdx)

		// Read metadata size
		var metadataSize uint32
		binary.Read(dec, binary.LittleEndian, &metadataSize)

		if metadataSize > 0 {
			point.Metadata = make(map[string]interface{}, metadataSize)

			// Read metadata key-value pairs
			for j := uint32(0); j < metadataSize; j++ {
				var keySize, valueSize uint32
				binary.Read(dec, binary.LittleEndian, &keySize)

				keyBuf := buf[:keySize]
				io.ReadFull(dec, keyBuf)
				key := string(keyBuf)

				binary.Read(dec, binary.LittleEndian, &valueSize)
				valueBuf := buf[keySize : keySize+valueSize]
				io.ReadFull(dec, valueBuf)

				var value interface{}
				json.Unmarshal(valueBuf, &value)
				point.Metadata[key] = value
			}
		}

		points = append(points, point)
	}

	// Read metrics pool
	metricsPool.Metrics = make([]map[string]float32, numMetrics)
	for i := range metricsPool.Metrics {
		var numPairs uint32
		binary.Read(dec, binary.LittleEndian, &numPairs)

		metrics := make(map[string]float32, numPairs)
		for j := uint32(0); j < numPairs; j++ {
			var keyLen uint32
			binary.Read(dec, binary.LittleEndian, &keyLen)

			keyBuf := buf[:keyLen]
			io.ReadFull(dec, keyBuf)
			key := string(keyBuf)

			var value float32
			binary.Read(dec, binary.LittleEndian, &value)

			metrics[key] = value
		}
		metricsPool.Metrics[i] = metrics
	}

	// Build tree
	sc.Tree = &KDTree{
		Nodes:    nodes,
		Points:   points,
		NodeSize: options.NodeSize,
		Pool:     metricsPool,
	}

	return sc, nil
}
