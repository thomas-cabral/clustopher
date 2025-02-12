package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"
	"web/clustopher/proto"
	"web/clustopher/runner"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

func main() {
	// Parse command line flags
	port := flag.Int("port", 50051, "The gRPC server port")
	maxClusters := flag.Int("max-clusters", 5, "Maximum number of clusters to keep in memory")
	flag.Parse()

	// Create listener
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		fmt.Printf("Failed to listen: %v\n", err)
		os.Exit(1)
	}

	// Create gRPC server
	s := grpc.NewServer()
	clusterRunner := runner.NewClusterRunner(*maxClusters)
	proto.RegisterClusterServiceServer(s, clusterRunner)

	// Enable reflection for debugging
	reflection.Register(s)

	// Handle shutdown gracefully
	go func() {
		quit := make(chan os.Signal, 1)
		signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
		<-quit
		fmt.Println("\nShutting down gRPC server...")
		s.GracefulStop()
	}()

	// Start server
	fmt.Printf("Starting gRPC server on port %d...\n", *port)
	if err := s.Serve(lis); err != nil {
		fmt.Printf("Failed to serve: %v\n", err)
		os.Exit(1)
	}
}