# Exercise 03 Solution: Advanced SQL Joins & Analytical Queries

## Solution Overview

This implementation guide provides complete solutions for Exercise 03: mastering advanced SQL query patterns including all JOIN types, window functions, CTEs, and complex aggregations for production ML infrastructure analytics.

**What You'll Build**:
- Comprehensive JOIN mastery (INNER, LEFT, RIGHT, FULL OUTER, CROSS, SELF)
- Window functions for ranking and running totals
- Common Table Expressions (CTEs) for multi-step queries
- Production-ready analytical queries for ML dashboards
- Query performance optimization techniques

**Time to Complete**: 3-4 hours
**Difficulty**: Intermediate → Advanced

---

## Table of Contents

1. [Understanding JOINs](#part-1-understanding-joins)
2. [INNER JOIN](#part-2-inner-join)
3. [LEFT JOIN](#part-3-left-join)
4. [RIGHT and FULL OUTER JOIN](#part-4-right-and-full-outer-join)
5. [CROSS JOIN](#part-5-cross-join)
6. [SELF JOIN](#part-6-self-join)
7. [Window Functions](#part-7-window-functions)
8. [CTEs and Subqueries](#part-8-ctes-and-subqueries)
9. [Production Analytics](#part-9-production-analytics)
10. [Performance Optimization](#part-10-performance-optimization)

---

## Part 1: Understanding JOINs

### Step 1.1: Visual JOIN Reference

```
Table A (models)     Table B (model_versions)
┌───────────┐       ┌───────────┐
│  A only   │       │  B only   │
│     ┌─────┼───────┼─────┐     │
│     │ A∩B │       │ A∩B │     │
│     └─────┼───────┼─────┘     │
└───────────┘       └───────────┘

INNER JOIN:  A ∩ B   (intersection only - matching rows from both)
LEFT JOIN:   A        (all of A + matching B, NULL where no match)
RIGHT JOIN:       B   (all of B + matching A, NULL where no match)
FULL OUTER:  A ∪ B   (everything from both, NULL where no match)
CROSS JOIN:  A × B   (cartesian product - every combination)
SELF JOIN:   A ⟷ A  (table joined with itself)
```

### Step 1.2: Setup

We'll use the ML Model Registry from Exercise 02. Ensure it's loaded:

```bash
# Connect to database
docker exec -it pg-ml-registry psql -U mlops -d ml_registry

# Or start fresh if needed:
# See Exercise 02 for complete setup
```

### Step 1.3: Verify Data

```sql
-- Quick verification
SELECT 'Models' AS table_name, COUNT(*) FROM models
UNION ALL
SELECT 'Versions', COUNT(*) FROM model_versions
UNION ALL
SELECT 'Training Runs', COUNT(*) FROM training_runs
UNION ALL
SELECT 'Deployments', COUNT(*) FROM deployments;
```

✅ **Checkpoint**: Database connected with data from Exercise 02.

---

## Part 2: INNER JOIN

### Step 2.1: Basic INNER JOIN

Create `sql/20_advanced_joins.sql`:

```sql
-- ============================================
-- INNER JOIN EXAMPLES
-- ============================================
-- Returns only rows that have matches in BOTH tables

-- Query 1: Models with their versions (INNER JOIN)
-- Only shows models that HAVE at least one version
SELECT
    m.model_name,
    m.display_name,
    m.risk_level,
    mv.semver,
    mv.status AS version_status,
    mv.registered_at
FROM models m
INNER JOIN model_versions mv ON m.model_id = mv.model_id
WHERE m.is_active = TRUE
ORDER BY m.model_name, mv.registered_at DESC;

-- What's excluded: Models without any versions!

-- Query 2: Three-table INNER JOIN
-- Models → Versions → Training Runs
SELECT
    m.model_name,
    mv.semver,
    tr.run_name,
    tr.status AS run_status,
    ROUND(tr.accuracy::numeric, 4) AS accuracy,
    ROUND(tr.gpu_hours::numeric, 2) AS gpu_hours,
    tr.started_at
FROM models m
INNER JOIN model_versions mv ON m.model_id = mv.model_id
INNER JOIN training_runs tr ON mv.version_id = tr.version_id
WHERE tr.status = 'succeeded'
ORDER BY m.model_name, mv.semver DESC, tr.accuracy DESC;

-- What's excluded:
-- - Models without versions
-- - Versions without training runs
-- - Failed training runs (due to WHERE clause)

-- Query 3: INNER JOIN with aggregation
SELECT
    m.model_name,
    m.display_name,
    COUNT(DISTINCT mv.version_id) AS version_count,
    COUNT(tr.run_id) AS total_runs,
    COUNT(*) FILTER (WHERE tr.status = 'succeeded') AS successful_runs,
    COUNT(*) FILTER (WHERE tr.status = 'failed') AS failed_runs,
    ROUND(AVG(tr.accuracy)::numeric, 4) AS avg_accuracy,
    ROUND(SUM(tr.gpu_hours)::numeric, 2) AS total_gpu_hours,
    ROUND(SUM(tr.gpu_hours * 2.50)::numeric, 2) AS estimated_cost_usd
FROM models m
INNER JOIN model_versions mv ON m.model_id = mv.model_id
INNER JOIN training_runs tr ON mv.version_id = tr.version_id
GROUP BY m.model_id, m.model_name, m.display_name
HAVING COUNT(tr.run_id) > 0
ORDER BY total_gpu_hours DESC;

-- Query 4: INNER JOIN with multiple conditions
SELECT
    m.model_name,
    mv.semver,
    e.environment_name,
    e.environment_type,
    d.deployment_name,
    d.status AS deployment_status,
    d.health_status,
    d.replicas,
    d.traffic_percentage,
    d.deployed_at
FROM models m
INNER JOIN model_versions mv ON m.model_id = mv.model_id
INNER JOIN deployments d ON mv.version_id = d.version_id
INNER JOIN environments e ON d.environment_id = e.environment_id
WHERE d.status = 'active'
  AND e.environment_type = 'production'
ORDER BY e.environment_name, m.model_name;

-- Only shows: Active production deployments
```

### Step 2.2: INNER JOIN Performance

```sql
-- Query 5: INNER JOIN with indexes (fast!)
EXPLAIN ANALYZE
SELECT
    m.model_name,
    mv.semver,
    mv.framework
FROM models m
INNER JOIN model_versions mv ON m.model_id = mv.model_id
WHERE m.is_active = TRUE;

-- Look for:
-- - "Index Scan" (good!)
-- - Execution time
-- - Rows returned
```

✅ **Checkpoint**: You understand INNER JOIN returns only matching rows from both tables.

---

## Part 3: LEFT JOIN

### Step 3.1: Basic LEFT JOIN

```sql
-- ============================================
-- LEFT JOIN (LEFT OUTER JOIN) EXAMPLES
-- ============================================
-- Returns ALL rows from left table + matching from right (NULL if no match)

-- Query 6: ALL models with version count (including 0)
SELECT
    m.model_name,
    m.display_name,
    m.created_at,
    COUNT(mv.version_id) AS version_count,
    MAX(mv.registered_at) AS latest_version_date,
    MAX(mv.semver) AS latest_version
FROM models m
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
WHERE m.is_active = TRUE
GROUP BY m.model_id, m.model_name, m.display_name, m.created_at
ORDER BY version_count DESC, m.created_at DESC;

-- Key difference from INNER JOIN:
-- - Shows models even if version_count = 0
-- - Uses COUNT(mv.version_id) not COUNT(*) to get accurate count
-- - MAX returns NULL for models without versions

-- Query 7: Find models WITHOUT any versions (quality check)
SELECT
    m.model_name,
    m.display_name,
    m.created_at,
    m.created_by,
    m.primary_contact,
    AGE(NOW(), m.created_at) AS age
FROM models m
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
WHERE m.is_active = TRUE
  AND mv.version_id IS NULL  -- The key filter!
ORDER BY m.created_at DESC;

-- Finds: Orphaned models that need attention

-- Query 8: Models with deployment status
SELECT
    m.model_name,
    m.display_name,
    m.risk_level,
    COUNT(DISTINCT mv.version_id) AS total_versions,
    COUNT(DISTINCT CASE WHEN d.status = 'active' THEN d.deployment_id END) AS active_deployments,
    COUNT(DISTINCT CASE
        WHEN d.status = 'active' AND e.environment_type = 'production'
        THEN d.deployment_id
    END) AS prod_deployments,
    CASE
        WHEN COUNT(DISTINCT CASE WHEN d.status = 'active' AND e.environment_type = 'production' THEN d.deployment_id END) > 0 THEN 'In Production'
        WHEN COUNT(DISTINCT CASE WHEN d.status = 'active' THEN d.deployment_id END) > 0 THEN 'In Non-Prod'
        WHEN COUNT(DISTINCT mv.version_id) > 0 THEN 'Has Versions (Not Deployed)'
        ELSE 'No Versions'
    END AS deployment_status
FROM models m
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
LEFT JOIN deployments d ON mv.version_id = d.version_id
LEFT JOIN environments e ON d.environment_id = e.environment_id
WHERE m.is_active = TRUE
GROUP BY m.model_id, m.model_name, m.display_name, m.risk_level
ORDER BY
    CASE deployment_status
        WHEN 'In Production' THEN 1
        WHEN 'In Non-Prod' THEN 2
        WHEN 'Has Versions (Not Deployed)' THEN 3
        ELSE 4
    END,
    m.model_name;
```

### Step 3.2: LEFT JOIN with COALESCE

```sql
-- Query 9: Handle NULLs with COALESCE
SELECT
    m.model_name,
    m.display_name,
    COALESCE(t.team_name, 'No Team Assigned') AS team,
    COALESCE(COUNT(mv.version_id), 0) AS version_count,
    COALESCE(MAX(mv.registered_at), m.created_at) AS last_activity,
    COALESCE(
        STRING_AGG(DISTINCT mv.framework, ', ' ORDER BY mv.framework),
        'No Framework'
    ) AS frameworks_used
FROM models m
LEFT JOIN teams t ON m.team_id = t.team_id
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
WHERE m.is_active = TRUE
GROUP BY m.model_id, m.model_name, m.display_name, t.team_name, m.created_at
ORDER BY version_count DESC, last_activity DESC;

-- COALESCE returns first non-NULL value:
-- - COALESCE(NULL, 'default') → 'default'
-- - COALESCE(5, 'default') → 5
```

### Step 3.3: Multi-Level LEFT JOINs

```sql
-- Query 10: Complete lineage with LEFT JOINs
SELECT
    m.model_name,
    m.display_name,
    t.team_name,
    mv.semver,
    mv.status AS version_status,
    tr.run_name,
    tr.status AS run_status,
    ROUND(tr.accuracy::numeric, 4) AS accuracy,
    d.dataset_name,
    rd.dataset_role
FROM models m
LEFT JOIN teams t ON m.team_id = t.team_id
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
LEFT JOIN training_runs tr ON mv.version_id = tr.version_id
LEFT JOIN run_datasets rd ON tr.run_id = rd.run_id
LEFT JOIN datasets d ON rd.dataset_id = d.dataset_id
WHERE m.model_name = 'sentiment-classifier-v2'
ORDER BY
    mv.registered_at DESC NULLS LAST,
    tr.started_at DESC NULLS LAST;

-- Shows complete history even if some parts are missing
```

✅ **Checkpoint**: You understand LEFT JOIN preserves all left table rows.

---

## Part 4: RIGHT and FULL OUTER JOIN

### Step 4.1: RIGHT JOIN

```sql
-- ============================================
-- RIGHT JOIN EXAMPLES
-- ============================================
-- Returns ALL rows from right table + matching from left (NULL if no match)

-- Query 11: All environments with deployment counts
-- RIGHT JOIN keeps ALL environments
SELECT
    e.environment_name,
    e.environment_type,
    e.requires_approval,
    COUNT(d.deployment_id) AS deployment_count,
    COUNT(DISTINCT m.model_id) AS unique_models_deployed
FROM deployments d
RIGHT JOIN environments e ON d.environment_id = e.environment_id
LEFT JOIN model_versions mv ON d.version_id = mv.version_id
LEFT JOIN models m ON mv.model_id = m.model_id
GROUP BY e.environment_id, e.environment_name, e.environment_type, e.requires_approval
ORDER BY e.priority DESC;

-- Shows all environments, even those with 0 deployments

-- Best Practice: Rewrite with LEFT JOIN instead
SELECT
    e.environment_name,
    e.environment_type,
    e.requires_approval,
    COUNT(d.deployment_id) AS deployment_count,
    COUNT(DISTINCT m.model_id) AS unique_models_deployed
FROM environments e
LEFT JOIN deployments d ON e.environment_id = d.environment_id
LEFT JOIN model_versions mv ON d.version_id = mv.version_id
LEFT JOIN models m ON mv.model_id = m.model_id
GROUP BY e.environment_id, e.environment_name, e.environment_type, e.requires_approval
ORDER BY e.priority DESC;

-- Same result, but more readable!
```

### Step 4.2: FULL OUTER JOIN

```sql
-- ============================================
-- FULL OUTER JOIN EXAMPLES
-- ============================================
-- Returns ALL rows from BOTH tables (NULL where no match)

-- Query 12: Tag usage analysis
SELECT
    COALESCE(t.tag_name, '(No Tag)') AS tag_name,
    COALESCE(t.tag_category, '(Unknown)') AS category,
    COUNT(DISTINCT mt.model_id) AS model_count,
    COUNT(DISTINCT dt.dataset_id) AS dataset_count,
    CASE
        WHEN t.tag_id IS NULL THEN 'Orphaned Association'
        WHEN mt.model_id IS NULL AND dt.dataset_id IS NULL THEN 'Unused Tag'
        ELSE 'In Use'
    END AS usage_status
FROM tags t
FULL OUTER JOIN model_tags mt ON t.tag_id = mt.tag_id
FULL OUTER JOIN dataset_tags dt ON t.tag_id = dt.tag_id
GROUP BY t.tag_id, t.tag_name, t.tag_category
ORDER BY
    CASE usage_status
        WHEN 'In Use' THEN 1
        WHEN 'Unused Tag' THEN 2
        ELSE 3
    END,
    model_count DESC,
    dataset_count DESC;

-- Finds:
-- - Tags used by models and/or datasets
-- - Unused tags (can be deleted)
-- - Orphaned tag associations (data quality issue)

-- Query 13: Reconcile expected vs actual deployments
WITH expected_prod_deployments AS (
    -- What SHOULD be deployed (based on version status)
    SELECT
        mv.version_id,
        mv.semver,
        m.model_name,
        e.environment_name
    FROM model_versions mv
    JOIN models m ON mv.model_id = m.model_id
    CROSS JOIN environments e
    WHERE mv.status = 'deployed'
      AND e.environment_type = 'production'
      AND m.is_active = TRUE
),
actual_prod_deployments AS (
    -- What IS actually deployed
    SELECT
        d.version_id,
        mv.semver,
        m.model_name,
        e.environment_name,
        d.status,
        d.health_status
    FROM deployments d
    JOIN model_versions mv ON d.version_id = mv.version_id
    JOIN models m ON mv.model_id = m.model_id
    JOIN environments e ON d.environment_id = e.environment_id
    WHERE e.environment_type = 'production'
      AND d.status = 'active'
)
SELECT
    COALESCE(ex.model_name, act.model_name) AS model_name,
    COALESCE(ex.semver, act.semver) AS version,
    COALESCE(ex.environment_name, act.environment_name) AS environment,
    CASE
        WHEN act.version_id IS NULL THEN '⚠️  MISSING - Should be deployed but is not'
        WHEN ex.version_id IS NULL THEN '⚠️  EXTRA - Deployed but should not be'
        ELSE '✅ OK - Matches expected state'
    END AS reconciliation_status,
    act.health_status
FROM expected_prod_deployments ex
FULL OUTER JOIN actual_prod_deployments act
    ON ex.version_id = act.version_id
    AND ex.environment_name = act.environment_name
ORDER BY
    CASE
        WHEN act.version_id IS NULL OR ex.version_id IS NULL THEN 1
        ELSE 2
    END,
    model_name, environment;

-- Use case: Compliance audits, drift detection
```

✅ **Checkpoint**: You understand FULL OUTER JOIN returns all rows from both tables.

---

## Part 5: CROSS JOIN

### Step 5.1: Cartesian Product

```sql
-- ============================================
-- CROSS JOIN EXAMPLES
-- ============================================
-- Returns every combination (cartesian product)

-- Query 14: All possible model-environment combinations
SELECT
    m.model_name,
    e.environment_name,
    e.environment_type,
    e.requires_approval
FROM models m
CROSS JOIN environments e
WHERE m.is_active = TRUE
  AND e.environment_type IN ('staging', 'production')
ORDER BY m.model_name, e.priority DESC;

-- Result: Every model × Every environment = M × N rows

-- Query 15: Generate test matrix
SELECT
    m.model_name,
    mv.semver,
    e.environment_name,
    d.deployment_id IS NOT NULL AS is_deployed,
    CASE
        WHEN d.deployment_id IS NOT NULL THEN '✅ Deployed'
        ELSE '❌ Not Deployed'
    END AS deployment_status
FROM models m
CROSS JOIN model_versions mv
CROSS JOIN environments e
LEFT JOIN deployments d
    ON mv.version_id = d.version_id
    AND e.environment_id = d.environment_id
WHERE m.model_id = mv.model_id
  AND m.is_active = TRUE
  AND e.environment_type IN ('staging', 'production')
ORDER BY m.model_name, mv.semver DESC, e.priority DESC;

-- Use case: Deployment coverage matrix, testing matrix
```

### Step 5.2: Practical CROSS JOIN

```sql
-- Query 16: Date series with models (reporting)
WITH date_series AS (
    SELECT generate_series(
        DATE_TRUNC('day', NOW() - INTERVAL '7 days'),
        DATE_TRUNC('day', NOW()),
        INTERVAL '1 day'
    )::date AS report_date
)
SELECT
    ds.report_date,
    m.model_name,
    COUNT(tr.run_id) AS runs_on_date,
    COALESCE(SUM(tr.gpu_hours), 0) AS gpu_hours,
    COALESCE(SUM(tr.gpu_hours * 2.50), 0) AS cost_usd
FROM date_series ds
CROSS JOIN models m
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
LEFT JOIN training_runs tr
    ON mv.version_id = tr.version_id
    AND DATE(tr.started_at) = ds.report_date
WHERE m.is_active = TRUE
GROUP BY ds.report_date, m.model_id, m.model_name
ORDER BY ds.report_date DESC, m.model_name;

-- Creates daily report with 0 for days without activity
```

✅ **Checkpoint**: You understand CROSS JOIN creates all combinations.

---

## Part 6: SELF JOIN

### Step 6.1: Basic SELF JOIN

```sql
-- ============================================
-- SELF JOIN EXAMPLES
-- ============================================
-- Join a table with itself

-- Query 17: Compare model versions (current vs previous)
SELECT
    m.model_name,
    curr.semver AS current_version,
    curr.registered_at AS current_date,
    prev.semver AS previous_version,
    prev.registered_at AS previous_date,
    curr.registered_at - prev.registered_at AS time_between_versions,
    ROUND((curr.benchmark_accuracy - prev.benchmark_accuracy)::numeric, 4) AS accuracy_improvement
FROM models m
JOIN model_versions curr ON m.model_id = curr.model_id
LEFT JOIN model_versions prev
    ON m.model_id = prev.model_id
    AND prev.registered_at < curr.registered_at
WHERE m.is_active = TRUE
  AND NOT EXISTS (
      -- Ensure prev is the immediate previous version
      SELECT 1 FROM model_versions mv2
      WHERE mv2.model_id = m.model_id
        AND mv2.registered_at > prev.registered_at
        AND mv2.registered_at < curr.registered_at
  )
ORDER BY m.model_name, curr.registered_at DESC;

-- Shows: Each version compared to its predecessor

-- Query 18: Find models by same team
SELECT
    m1.model_name AS model_1,
    m2.model_name AS model_2,
    t.team_name,
    m1.risk_level AS risk_1,
    m2.risk_level AS risk_2
FROM models m1
JOIN models m2 ON m1.team_id = m2.team_id AND m1.model_id < m2.model_id
JOIN teams t ON m1.team_id = t.team_id
WHERE m1.is_active = TRUE
  AND m2.is_active = TRUE
ORDER BY t.team_name, m1.model_name, m2.model_name;

-- m1.model_id < m2.model_id prevents duplicate pairs (A-B and B-A)
```

### Step 6.2: Hierarchical Data with SELF JOIN

```sql
-- Query 19: Model version lineage
WITH RECURSIVE version_tree AS (
    -- Base case: first versions (no predecessors)
    SELECT
        mv.version_id,
        mv.model_id,
        mv.semver,
        mv.registered_at,
        mv.semver AS lineage_path,
        1 AS depth
    FROM model_versions mv
    WHERE mv.registered_at = (
        SELECT MIN(registered_at)
        FROM model_versions
        WHERE model_id = mv.model_id
    )

    UNION ALL

    -- Recursive case: subsequent versions
    SELECT
        mv.version_id,
        mv.model_id,
        mv.semver,
        mv.registered_at,
        vt.lineage_path || ' → ' || mv.semver,
        vt.depth + 1
    FROM model_versions mv
    JOIN version_tree vt ON mv.model_id = vt.model_id
    WHERE mv.registered_at > vt.registered_at
      AND NOT EXISTS (
          SELECT 1 FROM model_versions mv2
          WHERE mv2.model_id = mv.model_id
            AND mv2.registered_at > vt.registered_at
            AND mv2.registered_at < mv.registered_at
      )
)
SELECT
    m.model_name,
    vt.semver,
    vt.lineage_path,
    vt.depth AS version_number
FROM version_tree vt
JOIN models m ON vt.model_id = m.model_id
WHERE m.is_active = TRUE
ORDER BY m.model_name, vt.registered_at;

-- Shows: Complete version lineage with depth
```

✅ **Checkpoint**: You understand SELF JOIN for comparing rows within same table.

---

## Part 7: Window Functions

### Step 7.1: ROW_NUMBER, RANK, DENSE_RANK

```sql
-- ============================================
-- WINDOW FUNCTIONS
-- ============================================

-- Query 20: Rank models by accuracy within each framework
SELECT
    m.model_name,
    mv.framework,
    ROUND(tr.accuracy::numeric, 4) AS accuracy,
    ROW_NUMBER() OVER (PARTITION BY mv.framework ORDER BY tr.accuracy DESC) AS row_num,
    RANK() OVER (PARTITION BY mv.framework ORDER BY tr.accuracy DESC) AS rank,
    DENSE_RANK() OVER (PARTITION BY mv.framework ORDER BY tr.accuracy DESC) AS dense_rank
FROM training_runs tr
JOIN model_versions mv ON tr.version_id = mv.version_id
JOIN models m ON mv.model_id = m.model_id
WHERE tr.status = 'succeeded'
  AND tr.accuracy IS NOT NULL
ORDER BY mv.framework, tr.accuracy DESC;

-- Differences:
-- ROW_NUMBER: 1, 2, 3, 4, 5 (always unique)
-- RANK: 1, 2, 2, 4, 5 (ties share rank, next rank skips)
-- DENSE_RANK: 1, 2, 2, 3, 4 (ties share rank, next rank continues)

-- Query 21: Get top 3 runs per model
WITH ranked_runs AS (
    SELECT
        m.model_name,
        tr.run_name,
        tr.accuracy,
        tr.gpu_hours,
        tr.started_at,
        ROW_NUMBER() OVER (
            PARTITION BY m.model_id
            ORDER BY tr.accuracy DESC NULLS LAST
        ) AS rank_by_accuracy
    FROM training_runs tr
    JOIN model_versions mv ON tr.version_id = mv.version_id
    JOIN models m ON mv.model_id = m.model_id
    WHERE tr.status = 'succeeded'
)
SELECT
    model_name,
    run_name,
    ROUND(accuracy::numeric, 4) AS accuracy,
    ROUND(gpu_hours::numeric, 2) AS gpu_hours,
    started_at
FROM ranked_runs
WHERE rank_by_accuracy <= 3
ORDER BY model_name, rank_by_accuracy;
```

### Step 7.2: LAG, LEAD

```sql
-- Query 22: Compare each run with previous run
SELECT
    m.model_name,
    tr.run_name,
    tr.started_at,
    ROUND(tr.accuracy::numeric, 4) AS accuracy,
    ROUND(LAG(tr.accuracy) OVER (
        PARTITION BY m.model_id
        ORDER BY tr.started_at
    )::numeric, 4) AS previous_accuracy,
    ROUND((tr.accuracy - LAG(tr.accuracy) OVER (
        PARTITION BY m.model_id
        ORDER BY tr.started_at
    ))::numeric, 4) AS accuracy_change,
    tr.started_at - LAG(tr.started_at) OVER (
        PARTITION BY m.model_id
        ORDER BY tr.started_at
    ) AS time_since_last_run
FROM training_runs tr
JOIN model_versions mv ON tr.version_id = mv.version_id
JOIN models m ON mv.model_id = m.model_id
WHERE tr.status = 'succeeded'
  AND tr.accuracy IS NOT NULL
ORDER BY m.model_name, tr.started_at DESC;

-- LAG: Look at previous row
-- LEAD: Look at next row
```

### Step 7.3: Running Totals and Moving Averages

```sql
-- Query 23: Running total of GPU hours by model
SELECT
    m.model_name,
    tr.run_name,
    tr.started_at,
    ROUND(tr.gpu_hours::numeric, 2) AS gpu_hours_this_run,
    ROUND(SUM(tr.gpu_hours) OVER (
        PARTITION BY m.model_id
        ORDER BY tr.started_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )::numeric, 2) AS cumulative_gpu_hours,
    ROUND(AVG(tr.gpu_hours) OVER (
        PARTITION BY m.model_id
        ORDER BY tr.started_at
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )::numeric, 2) AS moving_avg_3_runs
FROM training_runs tr
JOIN model_versions mv ON tr.version_id = mv.version_id
JOIN models m ON mv.model_id = m.model_id
WHERE tr.status = 'succeeded'
ORDER BY m.model_name, tr.started_at;

-- ROWS BETWEEN:
-- - UNBOUNDED PRECEDING: From start of partition
-- - 2 PRECEDING: Last 3 rows (2 before + current)
-- - CURRENT ROW: Up to current row
```

✅ **Checkpoint**: You understand window functions for ranking and analytics.

---

## Part 8: CTEs and Subqueries

### Step 8.1: Common Table Expressions (CTEs)

```sql
-- ============================================
-- CTEs (WITH clause)
-- ============================================

-- Query 24: Multi-step analysis with CTEs
WITH model_stats AS (
    -- Step 1: Calculate per-model statistics
    SELECT
        m.model_id,
        m.model_name,
        COUNT(DISTINCT mv.version_id) AS version_count,
        COUNT(tr.run_id) AS total_runs,
        SUM(tr.gpu_hours) AS total_gpu_hours
    FROM models m
    LEFT JOIN model_versions mv ON m.model_id = mv.model_id
    LEFT JOIN training_runs tr ON mv.version_id = tr.version_id
    WHERE m.is_active = TRUE
    GROUP BY m.model_id, m.model_name
),
deployment_stats AS (
    -- Step 2: Calculate per-model deployment statistics
    SELECT
        m.model_id,
        COUNT(DISTINCT d.deployment_id) FILTER (WHERE d.status = 'active') AS active_deployments,
        COUNT(DISTINCT CASE
            WHEN d.status = 'active' AND e.environment_type = 'production'
            THEN d.deployment_id
        END) AS prod_deployments
    FROM models m
    LEFT JOIN model_versions mv ON m.model_id = mv.model_id
    LEFT JOIN deployments d ON mv.version_id = d.version_id
    LEFT JOIN environments e ON d.environment_id = e.environment_id
    WHERE m.is_active = TRUE
    GROUP BY m.model_id
)
-- Step 3: Combine results
SELECT
    ms.model_name,
    ms.version_count,
    ms.total_runs,
    ROUND(ms.total_gpu_hours::numeric, 2) AS total_gpu_hours,
    COALESCE(ds.active_deployments, 0) AS active_deployments,
    COALESCE(ds.prod_deployments, 0) AS prod_deployments,
    CASE
        WHEN ds.prod_deployments > 0 THEN 'Production'
        WHEN ds.active_deployments > 0 THEN 'Non-Production'
        WHEN ms.version_count > 0 THEN 'Development'
        ELSE 'Empty'
    END AS stage
FROM model_stats ms
LEFT JOIN deployment_stats ds ON ms.model_id = ds.model_id
ORDER BY
    CASE stage
        WHEN 'Production' THEN 1
        WHEN 'Non-Production' THEN 2
        WHEN 'Development' THEN 3
        ELSE 4
    END,
    ms.total_gpu_hours DESC NULLS LAST;

-- Benefits of CTEs:
-- - Readable: Each step has a name
-- - Maintainable: Easy to modify individual steps
-- - Reusable: Reference same CTE multiple times
```

### Step 8.2: Recursive CTEs

```sql
-- Query 25: Model approval hierarchy (if we had hierarchical approvals)
-- This demonstrates the pattern

WITH RECURSIVE approval_chain AS (
    -- Base: Initial approvals
    SELECT
        a.approval_id,
        a.version_id,
        a.approval_type,
        a.approval_status,
        a.requested_by,
        1 AS level,
        ARRAY[a.approval_id] AS approval_path
    FROM approvals a
    WHERE a.approval_status = 'approved'
      AND NOT EXISTS (
          SELECT 1 FROM approvals a2
          WHERE a2.version_id = a.version_id
            AND a2.reviewed_at < a.requested_at
      )

    UNION ALL

    -- Recursive: Subsequent approvals
    SELECT
        a.approval_id,
        a.version_id,
        a.approval_type,
        a.approval_status,
        a.requested_by,
        ac.level + 1,
        ac.approval_path || a.approval_id
    FROM approvals a
    JOIN approval_chain ac ON a.version_id = ac.version_id
    WHERE a.requested_at > (
        SELECT reviewed_at FROM approvals WHERE approval_id = ac.approval_id
    )
    AND a.approval_status = 'approved'
)
SELECT
    m.model_name,
    mv.semver,
    ac.level AS approval_level,
    ac.approval_type,
    ARRAY_LENGTH(ac.approval_path, 1) AS approvals_in_chain
FROM approval_chain ac
JOIN model_versions mv ON ac.version_id = mv.version_id
JOIN models m ON mv.model_id = m.model_id
ORDER BY m.model_name, mv.semver, ac.level;
```

✅ **Checkpoint**: You understand CTEs for multi-step queries.

---

## Part 9: Production Analytics

### Step 9.1: Executive Dashboard Queries

```sql
-- ============================================
-- PRODUCTION ANALYTICAL QUERIES
-- ============================================

-- Query 26: Executive Summary Dashboard
WITH summary AS (
    SELECT
        COUNT(DISTINCT m.model_id) AS total_models,
        COUNT(DISTINCT mv.version_id) AS total_versions,
        COUNT(DISTINCT d.deployment_id) FILTER (
            WHERE d.status = 'active'
            AND e.environment_type = 'production'
        ) AS active_prod_deployments,
        COUNT(tr.run_id) AS total_training_runs,
        COUNT(*) FILTER (WHERE tr.status = 'succeeded') AS successful_runs,
        SUM(tr.gpu_hours) AS total_gpu_hours,
        ROUND(SUM(tr.gpu_hours * 2.50)::numeric, 2) AS total_cost_usd
    FROM models m
    LEFT JOIN model_versions mv ON m.model_id = mv.model_id
    LEFT JOIN training_runs tr ON mv.version_id = tr.version_id
    LEFT JOIN deployments d ON mv.version_id = d.version_id
    LEFT JOIN environments e ON d.environment_id = e.environment_id
    WHERE m.is_active = TRUE
)
SELECT
    total_models,
    total_versions,
    ROUND(total_versions::numeric / NULLIF(total_models, 0), 2) AS avg_versions_per_model,
    active_prod_deployments,
    total_training_runs,
    successful_runs,
    ROUND(100.0 * successful_runs / NULLIF(total_training_runs, 0), 2) AS success_rate_pct,
    ROUND(total_gpu_hours::numeric, 2) AS total_gpu_hours,
    total_cost_usd,
    ROUND(total_cost_usd / NULLIF(successful_runs, 0), 2) AS cost_per_successful_run
FROM summary;

-- Query 27: Model Health Report
SELECT
    m.model_name,
    m.risk_level,
    t.team_name,
    COUNT(DISTINCT mv.version_id) AS versions,
    MAX(mv.registered_at) AS last_version_date,
    AGE(NOW(), MAX(mv.registered_at)) AS days_since_last_version,
    COUNT(DISTINCT d.deployment_id) FILTER (WHERE d.status = 'active') AS active_deployments,
    COUNT(DISTINCT d.deployment_id) FILTER (
        WHERE d.status = 'active' AND d.health_status != 'healthy'
    ) AS unhealthy_deployments,
    CASE
        WHEN COUNT(DISTINCT d.deployment_id) FILTER (WHERE d.status = 'active' AND d.health_status != 'healthy') > 0
            THEN '🔴 Critical'
        WHEN AGE(NOW(), MAX(mv.registered_at)) > INTERVAL '90 days'
            THEN '🟡 Stale'
        WHEN COUNT(DISTINCT mv.version_id) = 0
            THEN '⚪ No Versions'
        ELSE '🟢 Healthy'
    END AS health_status
FROM models m
LEFT JOIN teams t ON m.team_id = t.team_id
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
LEFT JOIN deployments d ON mv.version_id = d.version_id
WHERE m.is_active = TRUE
GROUP BY m.model_id, m.model_name, m.risk_level, t.team_name
ORDER BY
    CASE health_status
        WHEN '🔴 Critical' THEN 1
        WHEN '🟡 Stale' THEN 2
        WHEN '⚪ No Versions' THEN 3
        ELSE 4
    END,
    m.risk_level DESC,
    m.model_name;
```

### Step 9.2: Cost Analysis

```sql
-- Query 28: Cost breakdown by team and model
SELECT
    COALESCE(t.team_name, 'No Team') AS team,
    m.model_name,
    COUNT(tr.run_id) AS total_runs,
    ROUND(SUM(tr.gpu_hours)::numeric, 2) AS total_gpu_hours,
    ROUND(SUM(tr.gpu_hours * 2.50)::numeric, 2) AS total_cost_usd,
    ROUND(AVG(tr.gpu_hours)::numeric, 2) AS avg_gpu_hours_per_run,
    ROUND(AVG(tr.gpu_hours * 2.50)::numeric, 2) AS avg_cost_per_run,
    ROUND(
        100.0 * SUM(tr.gpu_hours) /
        SUM(SUM(tr.gpu_hours)) OVER (),
        2
    ) AS pct_of_total_cost
FROM training_runs tr
JOIN model_versions mv ON tr.version_id = mv.version_id
JOIN models m ON mv.model_id = m.model_id
LEFT JOIN teams t ON m.team_id = t.team_id
WHERE tr.status = 'succeeded'
GROUP BY t.team_id, t.team_name, m.model_id, m.model_name
ORDER BY total_cost_usd DESC;
```

✅ **Checkpoint**: You can write production analytics queries.

---

## Part 10: Performance Optimization

### Step 10.1: EXPLAIN ANALYZE

```sql
-- ============================================
-- QUERY PERFORMANCE ANALYSIS
-- ============================================

-- Query 29: Analyze query performance
EXPLAIN ANALYZE
SELECT
    m.model_name,
    COUNT(DISTINCT mv.version_id) AS versions,
    COUNT(tr.run_id) AS runs
FROM models m
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
LEFT JOIN training_runs tr ON mv.version_id = tr.version_id
WHERE m.is_active = TRUE
GROUP BY m.model_id, m.model_name;

-- Look for:
-- - "Seq Scan" (bad for large tables - missing index)
-- - "Index Scan" (good - using index)
-- - Execution time
-- - Rows returned vs rows scanned
```

✅ **Checkpoint**: Exercise 03 complete!

---

## Summary

**Completed**: Exercise 03 - Advanced SQL Joins & Analytical Queries

**What You Learned**:
- All JOIN types (INNER, LEFT, RIGHT, FULL OUTER, CROSS, SELF)
- Window functions (ROW_NUMBER, RANK, LAG, LEAD, SUM OVER)
- CTEs for multi-step queries
- Recursive CTEs for hierarchical data
- Production analytical queries
- Query performance analysis with EXPLAIN

**Key Skills**:
- Choose correct JOIN type for each scenario
- Write complex multi-table JOINs
- Use window functions for ranking and running totals
- Structure queries with CTEs for readability
- Build production dashboards with SQL
- Analyze and optimize query performance

**Ready For**: Exercise 04 - SQLAlchemy ORM Integration

---

**Exercise Complete!** 🎉
