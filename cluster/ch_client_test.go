package cluster

import (
	"context"
	"os"
	"testing"
)

func TestNewCHClient_PingsServer(t *testing.T) {
	dsn := os.Getenv("CLICKHOUSE_DSN")
	if dsn == "" {
		t.Skip("CLICKHOUSE_DSN not set")
	}
	c, err := NewCHClient(context.Background(), CHConfig{DSN: dsn})
	if err != nil {
		t.Fatalf("NewCHClient: %v", err)
	}
	defer c.Close()
	if err := c.Ping(context.Background()); err != nil {
		t.Fatalf("Ping: %v", err)
	}
}
