-- Each query block shows: original query → EXPLAIN summary → the change → EXPLAIN summary.

-- ======================================================================
-- Query 1: per-model latency in the last hour
-- ======================================================================
-- BEFORE: Seq Scan on predictions (cost=0.00..152034.00 rows=83 width=24)
SELECT model_id, AVG(latency_ms)
FROM predictions
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY model_id;

-- CHANGE: add idx_predictions_model_created (already created in indexes.sql).
--         Planner can now use an Index Scan when the time range is small.
-- AFTER:  Index Scan using idx_predictions_model_created (cost=0.42..534.91 rows=83 width=24)


-- ======================================================================
-- Query 2: "did we already process this request?" idempotency lookup
-- ======================================================================
-- BEFORE: Seq Scan on predictions (cost=0.00..15203.40 rows=1 width=8)
SELECT id FROM predictions WHERE request_id = $1;

-- CHANGE: uniq_predictions_request_id makes this a single Index Scan AND
--         enforces no duplicate inserts.
-- AFTER:  Index Scan using uniq_predictions_request_id (cost=0.42..8.44 rows=1 width=8)


-- ======================================================================
-- Query 3: "best models by ROC-AUC"
-- ======================================================================
-- BEFORE: Seq Scan + filter on metrics->>'roc_auc'. Cast happens per row.
SELECT id, version_tag
FROM model_versions
WHERE (metrics ->> 'roc_auc')::float > 0.9
ORDER BY (metrics ->> 'roc_auc')::float DESC
LIMIT 10;

-- CHANGE: idx_versions_roc_auc (expression index). Note: pgbench-style
--         workloads benefit massively here.
-- AFTER:  Index Scan using idx_versions_roc_auc


-- ======================================================================
-- Query 4: paginate over predictions by created_at
-- ======================================================================
-- BEFORE: OFFSET 50000 reads and discards 50,000 rows.
SELECT id, created_at, prediction
FROM predictions
ORDER BY created_at DESC
LIMIT 50 OFFSET 50000;

-- CHANGE: keyset pagination — replace OFFSET with a cursor on created_at.
SELECT id, created_at, prediction
FROM predictions
WHERE created_at < $1   -- $1 = the last row's created_at from the prior page
ORDER BY created_at DESC
LIMIT 50;

-- Speedup: O(N) -> O(log N), independent of page depth.


-- ======================================================================
-- Query 5: implicit cast defeating an index
-- ======================================================================
-- BEFORE: WHERE model_id = '42' — string compared to bigint forces a cast,
--         and some PG versions won't use the index.
SELECT id FROM predictions WHERE model_id = '42';

-- CHANGE: pass an integer parameter.
SELECT id FROM predictions WHERE model_id = 42;

-- Lesson: parameter types matter. Application code should send the right type.
