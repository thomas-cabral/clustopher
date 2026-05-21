package cluster

import (
	"context"
	"fmt"

	"github.com/ClickHouse/clickhouse-go/v2"
	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
)

type CHConfig struct {
	DSN string // clickhouse://user:pass@host:port/database
}

type CHClient struct {
	conn driver.Conn
}

func NewCHClient(ctx context.Context, cfg CHConfig) (*CHClient, error) {
	opts, err := clickhouse.ParseDSN(cfg.DSN)
	if err != nil {
		return nil, fmt.Errorf("parse dsn: %w", err)
	}
	conn, err := clickhouse.Open(opts)
	if err != nil {
		return nil, fmt.Errorf("open clickhouse: %w", err)
	}
	if err := conn.Ping(ctx); err != nil {
		conn.Close()
		return nil, fmt.Errorf("ping clickhouse: %w", err)
	}
	return &CHClient{conn: conn}, nil
}

func (c *CHClient) Conn() driver.Conn             { return c.conn }
func (c *CHClient) Ping(ctx context.Context) error { return c.conn.Ping(ctx) }
func (c *CHClient) Close() error                 { return c.conn.Close() }
