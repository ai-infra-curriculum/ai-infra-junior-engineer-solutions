# Exercise 05: Database Optimization & Indexing - Implementation Guide

## Overview

This guide demonstrates how to profile, optimize, and monitor database performance for a production ML Model Registry. You'll learn to identify query bottlenecks using EXPLAIN ANALYZE, design strategic indexes, implement maintenance procedures, and establish monitoring dashboards.

**Scenario**: Your registry is experiencing performance degradation after 3 months of growth. Dashboard queries take 5-10 seconds, training run ingestion times out, and CPU usage spikes to 90%. You must optimize the database to meet SLAs.

**Technologies Used:**
- PostgreSQL 14+ (pg_stat_statements, auto_explain)
- EXPLAIN ANALYZE for query profiling
- Index types: B-tree, GIN, BRIN, Partial, Covering
- Maintenance: VACUUM, ANALYZE, REINDEX
- Monitoring: Prometheus, Grafana, pg_stat_monitor

---

## Part 1: Database Performance Profiling

### Step 1.1: Enable Query Statistics Extension

```sql
-- Connect to database
\c ml_registry

-- Enable pg_stat_statements for query tracking
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Verify installation
SELECT * FROM pg_stat_statements LIMIT 5;

-- Reset statistics to start fresh
SELECT pg_stat_statements_reset();
```

**What pg_stat_statements tracks:**
- Total execution time per query
- Number of calls
- Rows returned
- Buffer hits/misses
- I/O wait times

### Step 1.2: Configure PostgreSQL for Performance Monitoring

Create `docker-compose-optimized.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    container_name: ml-registry-postgres
    environment:
      POSTGRES_DB: ml_registry
      POSTGRES_USER: ml_user
      POSTGRES_PASSWORD: ml_password
      # Performance monitoring settings
      POSTGRES_SHARED_PRELOAD_LIBRARIES: pg_stat_statements
      POSTGRES_PG_STAT_STATEMENTS_TRACK: all
      POSTGRES_PG_STAT_STATEMENTS_MAX: 10000
      POSTGRES_LOG_MIN_DURATION_STATEMENT: 1000  # Log queries > 1s
      POSTGRES_LOG_LINE_PREFIX: '%t [%p]: user=%u,db=%d '
      POSTGRES_AUTO_EXPLAIN_LOG_MIN_DURATION: 500  # Explain queries > 500ms
      POSTGRES_AUTO_EXPLAIN_LOG_ANALYZE: 'on'
      POSTGRES_AUTO_EXPLAIN_LOG_BUFFERS: 'on'
      POSTGRES_DEFAULT_STATISTICS_TARGET: 100
      # Performance tuning
      POSTGRES_SHARED_BUFFERS: 256MB
      POSTGRES_EFFECTIVE_CACHE_SIZE: 1GB
      POSTGRES_WORK_MEM: 16MB
      POSTGRES_MAINTENANCE_WORK_MEM: 128MB
      POSTGRES_RANDOM_PAGE_COST: 1.1  # SSD optimization
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    command:
      - postgres
      - -c
      - shared_preload_libraries=pg_stat_statements
      - -c
      - pg_stat_statements.track=all
      - -c
      - log_min_duration_statement=1000

volumes:
  postgres_data:
```

**Start optimized database:**

```bash
docker-compose -f docker-compose-optimized.yml up -d

# Verify settings
docker exec -it ml-registry-postgres psql -U ml_user -d ml_registry -c "SHOW shared_preload_libraries;"
docker exec -it ml-registry-postgres psql -U ml_user -d ml_registry -c "SHOW log_min_duration_statement;"
```

### Step 1.3: Identify Slow Queries with pg_stat_statements

```sql
-- Find top 10 slowest queries by total time
SELECT
    query,
    calls,
    total_exec_time / 1000 AS total_seconds,
    mean_exec_time / 1000 AS mean_seconds,
    max_exec_time / 1000 AS max_seconds,
    stddev_exec_time / 1000 AS stddev_seconds,
    rows / calls AS avg_rows
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY total_exec_time DESC
LIMIT 10;
```

**Example Output (Before Optimization):**

```
                                  query                                  | calls | total_seconds | mean_seconds | max_seconds | avg_rows
-----------------------------------------------------------------------+-------+---------------+--------------+-------------+---------
 SELECT m.model_name, mv.semver, d.status FROM deployments d...       |  1250 |         8942.5 |         7.15 |       12.34 |      50
 SELECT COUNT(*) FROM training_runs WHERE status = $1...              |  3420 |         6234.8 |         1.82 |        5.67 |       1
 SELECT m.model_name, COUNT(mv.version_id) FROM models m...           |   890 |         5123.4 |         5.76 |        8.92 |      25
```

**Key Metrics to Monitor:**
- **total_exec_time**: Total time spent in query (target for reduction)
- **mean_exec_time**: Average time per execution (identify consistently slow queries)
- **max_exec_time**: Worst-case performance (outliers)
- **calls**: Frequency (optimize high-frequency queries first)

### Step 1.4: Analyze Query Plans with EXPLAIN ANALYZE

```sql
-- Example: Slow deployment query
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT
    m.model_name,
    mv.semver,
    d.status,
    d.deployed_at,
    e.environment_name
FROM deployments d
INNER JOIN model_versions mv ON d.version_id = mv.version_id
INNER JOIN models m ON mv.model_id = m.model_id
INNER JOIN environments e ON d.environment_id = e.environment_id
WHERE d.deployed_at > NOW() - INTERVAL '30 days'
  AND d.status = 'healthy'
ORDER BY d.deployed_at DESC
LIMIT 50;
```

**Example Output (Before Optimization):**

```
Limit  (cost=12345.67..12346.79 rows=50 width=120) (actual time=7234.567..7235.123 rows=50 loops=1)
  Buffers: shared hit=89234 read=12456
  ->  Sort  (cost=12345.67..12567.89 rows=8942 width=120) (actual time=7234.556..7234.789 rows=50 loops=1)
        Sort Key: d.deployed_at DESC
        Sort Method: top-N heapsort  Memory: 45kB
        Buffers: shared hit=89234 read=12456
        ->  Hash Join  (cost=456.78..11234.56 rows=8942 width=120) (actual time=234.567..6789.123 rows=8942 loops=1)
              Hash Cond: (d.environment_id = e.environment_id)
              Buffers: shared hit=89234 read=12456
              ->  Hash Join  (cost=234.56..10987.65 rows=8942 width=100) (actual time=123.456..6234.567 rows=8942 loops=1)
                    Hash Cond: (mv.model_id = m.model_id)
                    Buffers: shared hit=87234 read=12000
                    ->  Hash Join  (cost=123.45..10456.78 rows=8942 width=80) (actual time=98.765..5678.901 rows=8942 loops=1)
                          Hash Cond: (d.version_id = mv.version_id)
                          Buffers: shared hit=85234 read=11000
                          ->  Seq Scan on deployments d  (cost=0.00..9876.54 rows=8942 width=60) (actual time=12.345..5123.456 rows=8942 loops=1)
                                Filter: ((deployed_at > (now() - '30 days'::interval)) AND (status = 'healthy'::text))
                                Rows Removed by Filter: 45678
                                Buffers: shared hit=75234 read=10000
                          ->  Hash  (cost=98.76..98.76 rows=1234 width=40) (actual time=45.678..45.679 rows=1234 loops=1)
                                Buckets: 2048  Batches: 1  Memory Usage: 85kB
                                Buffers: shared hit=10000
                                ->  Seq Scan on model_versions mv  (cost=0.00..98.76 rows=1234 width=40) (actual time=0.123..34.567 rows=1234 loops=1)
Planning Time: 2.345 ms
Execution Time: 7235.456 ms
```

**Red Flags (Problems to Fix):**
1. **Seq Scan on deployments** (cost=0.00..9876.54) - Full table scan instead of index
2. **Buffers: shared read=12456** - High disk I/O (not in cache)
3. **Rows Removed by Filter: 45678** - Scanning unnecessary rows
4. **Execution Time: 7235.456 ms** - Way above 1-second SLA

---

## Part 2: Index Strategy Design

### Step 2.1: Identify Missing Indexes

**Analysis from EXPLAIN output:**
- `deployments.deployed_at` - used in WHERE and ORDER BY
- `deployments.status` - used in WHERE clause
- `deployments.version_id` - used in JOIN
- `model_versions.model_id` - used in JOIN

**Current Indexes (from Exercise 02):**

```sql
-- Check existing indexes
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
```

### Step 2.2: Create Strategic Indexes

```sql
-- ============================================================================
-- Deployment Indexes
-- ============================================================================

-- Index 1: Composite index for deployed_at + status (most selective first)
CREATE INDEX idx_deployments_deployed_at_status
ON deployments (deployed_at DESC, status)
WHERE status IN ('healthy', 'deploying', 'degraded');

-- Index 2: Covering index for deployment queries (include all needed columns)
CREATE INDEX idx_deployments_covering
ON deployments (deployed_at DESC, status)
INCLUDE (version_id, environment_id, deployment_name, replicas);

-- Index 3: Partial index for active deployments only
CREATE INDEX idx_deployments_active
ON deployments (deployed_at DESC)
WHERE status IN ('healthy', 'degraded');

-- ============================================================================
-- Training Run Indexes
-- ============================================================================

-- Index 4: Composite index for training run queries
CREATE INDEX idx_training_runs_version_started
ON training_runs (version_id, started_at DESC);

-- Index 5: Partial index for successful runs
CREATE INDEX idx_training_runs_succeeded
ON training_runs (version_id, started_at DESC)
WHERE status = 'succeeded';

-- Index 6: GIN index for JSONB hyperparameter searches
CREATE INDEX idx_training_runs_hyperparameters
ON training_runs USING GIN (hyperparameters);

-- ============================================================================
-- Model and Version Indexes
-- ============================================================================

-- Index 7: Composite index for model lookups
CREATE INDEX idx_models_active_team
ON models (is_active, team_id)
WHERE is_active = TRUE;

-- Index 8: Covering index for version queries
CREATE INDEX idx_model_versions_covering
ON model_versions (model_id, created_at DESC)
INCLUDE (semver, status, framework, artifact_uri);

-- ============================================================================
-- Metric Indexes
-- ============================================================================

-- Index 9: Composite index for metric lookups
CREATE INDEX idx_model_metrics_version_name
ON model_metrics (version_id, metric_name, dataset_split);

-- Index 10: BRIN index for time-series data (efficient for large tables)
CREATE INDEX idx_model_metrics_recorded_brin
ON model_metrics USING BRIN (recorded_at)
WITH (pages_per_range = 128);
```

**Index Types Explained:**

1. **B-tree (default)**: Best for equality and range queries
   - Use case: `WHERE status = 'healthy'`, `ORDER BY deployed_at`

2. **GIN (Generalized Inverted Index)**: Best for JSONB and array searches
   - Use case: `WHERE hyperparameters @> '{"lr": 0.001}'`

3. **BRIN (Block Range Index)**: Best for large, naturally ordered tables
   - Use case: Time-series data (`recorded_at`, `created_at`)
   - Space-efficient: 1000x smaller than B-tree

4. **Partial Index**: Index only rows matching a condition
   - Use case: `WHERE status = 'healthy'` (only 10% of rows)
   - Reduces index size and maintenance overhead

5. **Covering Index (INCLUDE)**: Include non-key columns in index
   - Use case: Avoid table lookups for frequently accessed columns
   - Enables "index-only scans"

### Step 2.3: Verify Index Usage

```sql
-- Re-run the slow query with EXPLAIN ANALYZE
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    m.model_name,
    mv.semver,
    d.status,
    d.deployed_at,
    e.environment_name
FROM deployments d
INNER JOIN model_versions mv ON d.version_id = mv.version_id
INNER JOIN models m ON mv.model_id = m.model_id
INNER JOIN environments e ON d.environment_id = e.environment_id
WHERE d.deployed_at > NOW() - INTERVAL '30 days'
  AND d.status = 'healthy'
ORDER BY d.deployed_at DESC
LIMIT 50;
```

**Example Output (After Indexing):**

```
Limit  (cost=123.45..124.67 rows=50 width=120) (actual time=12.345..12.567 rows=50 loops=1)
  Buffers: shared hit=234
  ->  Nested Loop  (cost=0.42..2134.56 rows=8942 width=120) (actual time=0.123..10.234 rows=50 loops=1)
        Buffers: shared hit=234
        ->  Nested Loop  (cost=0.42..1234.56 rows=8942 width=100) (actual time=0.089..8.123 rows=50 loops=1)
              Buffers: shared hit=184
              ->  Nested Loop  (cost=0.42..567.89 rows=8942 width=80) (actual time=0.056..5.678 rows=50 loops=1)
                    Buffers: shared hit=134
                    ->  Index Scan using idx_deployments_deployed_at_status on deployments d
                          (cost=0.42..234.56 rows=8942 width=60) (actual time=0.034..3.456 rows=50 loops=1)
                          Index Cond: ((deployed_at > (now() - '30 days'::interval)) AND (status = 'healthy'::text))
                          Buffers: shared hit=84
                    ->  Index Scan using model_versions_pkey on model_versions mv
                          (cost=0.42..0.45 rows=1 width=40) (actual time=0.012..0.013 rows=1 loops=50)
                          Index Cond: (version_id = d.version_id)
                          Buffers: shared hit=50
              ->  Index Scan using models_pkey on models m
                    (cost=0.42..0.45 rows=1 width=40) (actual time=0.011..0.012 rows=1 loops=50)
                    Index Cond: (model_id = mv.model_id)
                    Buffers: shared hit=50
        ->  Index Scan using environments_pkey on environments e
              (cost=0.42..0.45 rows=1 width=40) (actual time=0.010..0.011 rows=1 loops=50)
              Index Cond: (environment_id = d.environment_id)
              Buffers: shared hit=50
Planning Time: 1.234 ms
Execution Time: 12.789 ms
```

**Improvements:**
- ✅ **Seq Scan** → **Index Scan** (using idx_deployments_deployed_at_status)
- ✅ **Buffers: shared read=12456** → **shared hit=234** (everything in cache)
- ✅ **Execution Time: 7235.456 ms** → **12.789 ms** (566x faster!)

### Step 2.4: Monitor Index Usage and Bloat

```sql
-- Check index usage statistics
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

**Example Output:**

```
 schemaname |    tablename    |           indexname           | idx_scan | idx_tup_read | index_size
------------+-----------------+-------------------------------+----------+--------------+------------
 public     | deployments     | idx_deployments_deployed_at   |   12456  |      456789  | 128 MB
 public     | models          | models_pkey                   |    8942  |       8942   | 64 MB
 public     | training_runs   | idx_training_runs_version     |    6234  |       98765  | 256 MB
```

**Identify Unused Indexes:**

```sql
-- Find indexes that are never used
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND idx_scan = 0
  AND indexname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexrelid) DESC;
```

**Drop Unused Indexes:**

```sql
-- Example: Drop an unused index
DROP INDEX IF EXISTS idx_models_unused_column;
```

---

## Part 3: Query Optimization Techniques

### Step 3.1: Optimize Join Order and Selectivity

**Problem**: Joining large tables before filtering wastes resources.

**Solution**: Apply filters early, join smaller result sets.

**Before (Inefficient):**

```sql
-- Bad: Joins all tables, then filters
SELECT
    m.model_name,
    COUNT(tr.run_id) AS run_count
FROM models m
INNER JOIN model_versions mv ON m.model_id = mv.model_id
INNER JOIN training_runs tr ON mv.version_id = tr.version_id
WHERE tr.status = 'succeeded'
  AND tr.started_at > NOW() - INTERVAL '7 days'
GROUP BY m.model_id, m.model_name;
```

**After (Optimized):**

```sql
-- Good: Filters training_runs first, then joins
SELECT
    m.model_name,
    COUNT(tr.run_id) AS run_count
FROM training_runs tr
INNER JOIN model_versions mv ON tr.version_id = mv.version_id
INNER JOIN models m ON mv.model_id = m.model_id
WHERE tr.status = 'succeeded'
  AND tr.started_at > NOW() - INTERVAL '7 days'
GROUP BY m.model_id, m.model_name;
```

**Verify with EXPLAIN:**

```sql
EXPLAIN ANALYZE <query>;
-- Look for: "Rows Removed by Filter" should be minimal
```

### Step 3.2: Use CTEs for Complex Queries

**Problem**: Repeated subqueries execute multiple times.

**Solution**: Use Common Table Expressions (CTEs) to compute once.

**Before (Inefficient):**

```sql
SELECT
    model_name,
    (SELECT COUNT(*) FROM model_versions WHERE model_id = m.model_id) AS versions,
    (SELECT COUNT(*) FROM training_runs tr
     JOIN model_versions mv ON tr.version_id = mv.version_id
     WHERE mv.model_id = m.model_id) AS runs
FROM models m;
```

**After (Optimized):**

```sql
WITH version_counts AS (
    SELECT model_id, COUNT(*) AS version_count
    FROM model_versions
    GROUP BY model_id
),
run_counts AS (
    SELECT mv.model_id, COUNT(tr.run_id) AS run_count
    FROM training_runs tr
    JOIN model_versions mv ON tr.version_id = mv.version_id
    GROUP BY mv.model_id
)
SELECT
    m.model_name,
    COALESCE(vc.version_count, 0) AS versions,
    COALESCE(rc.run_count, 0) AS runs
FROM models m
LEFT JOIN version_counts vc ON m.model_id = vc.model_id
LEFT JOIN run_counts rc ON m.model_id = rc.model_id;
```

### Step 3.3: Batch Queries in Application Code

**Problem**: N+1 query problem - loading related objects one at a time.

**Solution**: Use eager loading with SQLAlchemy `joinedload` or `selectinload`.

**Before (N+1 Queries):**

```python
# Bad: Issues 1 query for models + N queries for versions
models = session.query(Model).all()
for model in models:
    print(f"{model.model_name}: {len(model.versions)} versions")  # Triggers lazy load
```

**After (Eager Loading):**

```python
from sqlalchemy.orm import selectinload

# Good: Issues 2 queries total (1 for models, 1 for all versions)
models = session.query(Model).options(selectinload(Model.versions)).all()
for model in models:
    print(f"{model.model_name}: {len(model.versions)} versions")  # No additional query
```

### Step 3.4: Aggregate Early in Subqueries

**Problem**: Aggregating after joining large tables.

**Solution**: Aggregate first, then join smaller result sets.

**Before (Inefficient):**

```sql
SELECT
    m.model_name,
    e.environment_name,
    COUNT(d.deployment_id) AS deployment_count
FROM models m
CROSS JOIN environments e
LEFT JOIN deployments d ON m.model_id = (
    SELECT mv.model_id FROM model_versions mv WHERE mv.version_id = d.version_id
) AND d.environment_id = e.environment_id
GROUP BY m.model_id, m.model_name, e.environment_id, e.environment_name;
```

**After (Optimized):**

```sql
WITH deployment_counts AS (
    SELECT
        mv.model_id,
        d.environment_id,
        COUNT(d.deployment_id) AS deployment_count
    FROM deployments d
    JOIN model_versions mv ON d.version_id = mv.version_id
    GROUP BY mv.model_id, d.environment_id
)
SELECT
    m.model_name,
    e.environment_name,
    COALESCE(dc.deployment_count, 0) AS deployment_count
FROM models m
CROSS JOIN environments e
LEFT JOIN deployment_counts dc ON m.model_id = dc.model_id
    AND e.environment_id = dc.environment_id;
```

---

## Part 4: Database Maintenance Procedures

### Step 4.1: Vacuum and Analyze

**Why**: PostgreSQL uses MVCC (Multi-Version Concurrency Control), which leaves dead tuples after UPDATE/DELETE. VACUUM reclaims space.

```sql
-- ============================================================================
-- Manual Vacuum
-- ============================================================================

-- Standard VACUUM (non-blocking, reclaims space)
VACUUM VERBOSE training_runs;

-- VACUUM FULL (blocking, rewrites table - use during maintenance window)
VACUUM FULL VERBOSE training_runs;

-- ANALYZE updates statistics for query planner
ANALYZE training_runs;

-- VACUUM ANALYZE (both operations)
VACUUM ANALYZE training_runs;

-- ============================================================================
-- Check for Bloat
-- ============================================================================

SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) -
                   pg_relation_size(schemaname||'.'||tablename)) AS index_size,
    n_dead_tup,
    n_live_tup,
    ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) AS dead_tuple_pct
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_dead_tup DESC;
```

**Example Output:**

```
 tablename     | total_size | table_size | index_size | n_dead_tup | n_live_tup | dead_tuple_pct
---------------+------------+------------+------------+------------+------------+---------------
 training_runs | 512 MB     | 384 MB     | 128 MB     |    245678  |    1234567 | 16.60
 deployments   | 256 MB     | 192 MB     | 64 MB      |     89456  |     567890 | 13.61
```

**Schedule Autovacuum (postgresql.conf):**

```conf
# Autovacuum settings
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 1min
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05
```

### Step 4.2: Reindex for Index Bloat

```sql
-- Check index bloat
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexrelid) DESC;

-- Reindex a specific index (rebuilds from scratch)
REINDEX INDEX CONCURRENTLY idx_deployments_deployed_at_status;

-- Reindex entire table (use CONCURRENTLY to avoid locking)
REINDEX TABLE CONCURRENTLY deployments;

-- Reindex entire database (maintenance window required)
REINDEX DATABASE ml_registry;
```

### Step 4.3: Update Table Statistics

```sql
-- Update statistics for accurate query planning
ANALYZE VERBOSE models;
ANALYZE VERBOSE model_versions;
ANALYZE VERBOSE training_runs;
ANALYZE VERBOSE deployments;

-- Increase statistics target for columns with many distinct values
ALTER TABLE training_runs ALTER COLUMN hyperparameters SET STATISTICS 500;
ANALYZE training_runs;

-- Check statistics staleness
SELECT
    schemaname,
    tablename,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze,
    n_mod_since_analyze
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_mod_since_analyze DESC;
```

---

## Part 5: Performance Monitoring and Alerting

### Step 5.1: Create Monitoring Queries

Create `scripts/monitor_performance.sql`:

```sql
-- ============================================================================
-- Query Performance Dashboard
-- ============================================================================

-- Top 10 Slowest Queries
SELECT
    LEFT(query, 80) AS query_snippet,
    calls,
    ROUND(total_exec_time::numeric / 1000, 2) AS total_sec,
    ROUND(mean_exec_time::numeric, 2) AS mean_ms,
    ROUND(max_exec_time::numeric, 2) AS max_ms
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Cache Hit Ratio (should be > 99%)
SELECT
    SUM(heap_blks_read) AS heap_read,
    SUM(heap_blks_hit) AS heap_hit,
    ROUND(100.0 * SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit) + SUM(heap_blks_read), 0), 2) AS cache_hit_ratio
FROM pg_statio_user_tables;

-- Index Hit Ratio (should be > 95%)
SELECT
    SUM(idx_blks_read) AS idx_read,
    SUM(idx_blks_hit) AS idx_hit,
    ROUND(100.0 * SUM(idx_blks_hit) / NULLIF(SUM(idx_blks_hit) + SUM(idx_blks_read), 0), 2) AS index_hit_ratio
FROM pg_statio_user_indexes;

-- Connection Statistics
SELECT
    state,
    COUNT(*) AS connection_count
FROM pg_stat_activity
WHERE datname = 'ml_registry'
GROUP BY state;

-- Long-Running Queries (> 5 seconds)
SELECT
    pid,
    now() - pg_stat_activity.query_start AS duration,
    usename,
    query
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - pg_stat_activity.query_start > INTERVAL '5 seconds'
ORDER BY duration DESC;

-- Lock Monitoring
SELECT
    pg_stat_activity.pid,
    pg_class.relname,
    pg_locks.mode,
    pg_locks.granted,
    pg_stat_activity.query
FROM pg_locks
JOIN pg_class ON pg_locks.relation = pg_class.oid
JOIN pg_stat_activity ON pg_locks.pid = pg_stat_activity.pid
WHERE NOT pg_locks.granted
ORDER BY pg_stat_activity.query_start;
```

### Step 5.2: Create Performance Monitoring Script

Create `scripts/performance_report.py`:

```python
"""Generate database performance report."""

import psycopg2
from datetime import datetime
from tabulate import tabulate

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="ml_registry",
    user="ml_user",
    password="ml_password"
)

def get_slow_queries():
    """Get top 10 slowest queries."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                LEFT(query, 100) AS query_snippet,
                calls,
                ROUND(total_exec_time::numeric / 1000, 2) AS total_sec,
                ROUND(mean_exec_time::numeric, 2) AS mean_ms
            FROM pg_stat_statements
            WHERE query NOT LIKE '%pg_stat_statements%'
            ORDER BY mean_exec_time DESC
            LIMIT 10;
        """)
        return cur.fetchall()

def get_cache_hit_ratio():
    """Calculate cache hit ratio."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ROUND(100.0 * SUM(heap_blks_hit) /
                      NULLIF(SUM(heap_blks_hit) + SUM(heap_blks_read), 0), 2)
            FROM pg_statio_user_tables;
        """)
        return cur.fetchone()[0]

def get_index_usage():
    """Get index usage statistics."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                tablename,
                indexname,
                idx_scan,
                pg_size_pretty(pg_relation_size(indexrelid)) AS size
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            ORDER BY idx_scan DESC
            LIMIT 10;
        """)
        return cur.fetchall()

def generate_report():
    """Generate and print performance report."""
    print("=" * 80)
    print(f"Database Performance Report - {datetime.now()}")
    print("=" * 80)

    print("\n[1] Cache Hit Ratio")
    print(f"    {get_cache_hit_ratio()}% (target: > 99%)")

    print("\n[2] Slowest Queries")
    slow_queries = get_slow_queries()
    print(tabulate(slow_queries,
                   headers=["Query", "Calls", "Total (s)", "Mean (ms)"],
                   tablefmt="grid"))

    print("\n[3] Most Used Indexes")
    indexes = get_index_usage()
    print(tabulate(indexes,
                   headers=["Table", "Index", "Scans", "Size"],
                   tablefmt="grid"))

if __name__ == "__main__":
    generate_report()
    conn.close()
```

**Run the report:**

```bash
pip install psycopg2-binary tabulate
python scripts/performance_report.py
```

### Step 5.3: Set Up Prometheus Exporter (Optional)

```bash
# Install postgres_exporter
docker run -d \
  --name postgres_exporter \
  --network host \
  -e DATA_SOURCE_NAME="postgresql://ml_user:ml_password@localhost:5432/ml_registry?sslmode=disable" \
  prometheuscommunity/postgres-exporter

# Verify metrics endpoint
curl http://localhost:9187/metrics | grep pg_stat
```

**Key Metrics to Monitor:**
- `pg_stat_database_tup_returned` - Rows read
- `pg_stat_database_blks_hit` - Cache hits
- `pg_stat_database_xact_commit` - Committed transactions
- `pg_stat_activity_count` - Active connections
- `pg_stat_statements_mean_exec_time_seconds` - Query latency

---

## Summary

**Completed**: Exercise 05 - Database Optimization & Indexing

**What You Learned:**
- Profiling database performance with pg_stat_statements and EXPLAIN ANALYZE
- Identifying query bottlenecks through execution plans
- Designing strategic indexes (B-tree, GIN, BRIN, Partial, Covering)
- Optimizing query patterns (join order, CTEs, eager loading)
- Implementing maintenance procedures (VACUUM, ANALYZE, REINDEX)
- Monitoring performance metrics and alerting

**Performance Improvements Achieved:**
- ✅ Query latency: 7235ms → 12ms (566x faster)
- ✅ Cache hit ratio: 85% → 99.2%
- ✅ Disk I/O: 12,456 blocks → 234 blocks (98% reduction)
- ✅ Dashboard load time: 8 seconds → 0.5 seconds

**Best Practices:**
1. Always use EXPLAIN ANALYZE before creating indexes
2. Create indexes for columns in WHERE, JOIN, and ORDER BY clauses
3. Use partial indexes for queries filtering on specific values
4. Monitor index usage and drop unused indexes
5. Schedule regular VACUUM ANALYZE operations
6. Set up automated performance monitoring and alerting
7. Test optimizations in staging before production deployment

**Next Exercise**: Exercise 06 - Transactions & Concurrency