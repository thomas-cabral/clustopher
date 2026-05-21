package cluster

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"sort"
	"strings"

	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
)

//go:embed migrations/*.sql
var migrationsFS embed.FS

// RunMigrations executes every .sql file in the embedded migrations dir in
// lexical order. Files containing "rollup" are treated as templates: the
// literal token "{N}" is expanded for zoom levels 2..16.
//
// Statements are split on ";" and executed individually. Empty statements are
// skipped. CREATE TABLE IF NOT EXISTS / CREATE MATERIALIZED VIEW IF NOT EXISTS
// keep this idempotent.
func RunMigrations(ctx context.Context, conn driver.Conn, dir string) error {
	entries, err := fs.ReadDir(migrationsFS, dir)
	if err != nil {
		return fmt.Errorf("read migrations dir: %w", err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".sql") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)

	for _, name := range names {
		body, err := fs.ReadFile(migrationsFS, dir+"/"+name)
		if err != nil {
			return fmt.Errorf("read %s: %w", name, err)
		}
		scripts := []string{string(body)}
		if strings.Contains(name, "rollup") {
			scripts = expandPerZoom(string(body))
		}
		for _, script := range scripts {
			for _, stmt := range splitStatements(script) {
				if strings.TrimSpace(stmt) == "" {
					continue
				}
				if err := conn.Exec(ctx, stmt); err != nil {
					return fmt.Errorf("exec (%s): %w\nSTMT: %s", name, err, stmt)
				}
			}
		}
	}
	return nil
}

func expandPerZoom(template string) []string {
	radius := fmt.Sprintf("%d", DefaultRollupRadius)
	out := make([]string, 0, 15)
	for z := 2; z <= 16; z++ {
		s := strings.ReplaceAll(template, "{N}", fmt.Sprintf("%d", z))
		s = strings.ReplaceAll(s, "{RADIUS}", radius)
		out = append(out, s)
	}
	return out
}

// splitStatements splits a SQL script on ';' after stripping '-- ' line comments.
//
// LIMITATION: this is not a full SQL tokenizer. It does NOT handle string
// literals or block comments containing ';'. Keep migration files free of
// such constructs (current usage: DDL only, no DML, no string literals).
func splitStatements(script string) []string {
	// Strip comment lines before splitting, to avoid semicolons inside comments
	// being treated as statement delimiters.
	var nonCommentLines []string
	for _, line := range strings.Split(script, "\n") {
		if !strings.HasPrefix(strings.TrimSpace(line), "--") {
			nonCommentLines = append(nonCommentLines, line)
		}
	}
	stripped := strings.Join(nonCommentLines, "\n")

	parts := strings.Split(stripped, ";")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}
