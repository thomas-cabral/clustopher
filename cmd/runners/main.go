package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"web/clustopher/cluster"
	"web/clustopher/proto"
	"web/clustopher/runner"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

func main() {
	port := flag.Int("port", 50051, "gRPC server port")
	maxClusters := flag.Int("max-clusters", 5, "maximum number of cluster skeletons to keep in memory")
	chDSN := flag.String("ch-dsn", os.Getenv("CLICKHOUSE_DSN"), "ClickHouse DSN (e.g. clickhouse://default:@localhost:9000/clustopher)")
	flag.Parse()

	if *chDSN == "" {
		log.Fatal("--ch-dsn or CLICKHOUSE_DSN required")
	}

	ctx := context.Background()

	ch, err := cluster.NewCHClient(ctx, cluster.CHConfig{DSN: *chDSN})
	if err != nil {
		log.Fatalf("ch client: %v", err)
	}
	defer ch.Close()

	if err := cluster.RunMigrations(ctx, ch.Conn(), "migrations"); err != nil {
		log.Fatalf("migrations: %v", err)
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("listen: %v", err)
	}

	// Raise default 4 MB message cap — at high zoom skeleton-path responses
	// can return tens of thousands of clusters with Children member-id slices,
	// well past 4 MB.
	const maxMsg = 256 * 1024 * 1024
	s := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxMsg),
		grpc.MaxSendMsgSize(maxMsg),
	)
	clusterRunner := runner.NewClusterRunner(*maxClusters, ch)
	proto.RegisterClusterServiceServer(s, clusterRunner)
	reflection.Register(s)

	go func() {
		quit := make(chan os.Signal, 1)
		signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
		<-quit
		fmt.Println("\nShutting down gRPC server...")
		s.GracefulStop()
	}()

	fmt.Printf("Starting gRPC server on port %d...\n", *port)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
