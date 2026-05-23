-- Catalogue of solution queries for exercise-01-sql-basics-crud.
-- Each section corresponds to a numbered task in the learning exercise.

-- ---------------------------------------------------------------------------
-- 1. CREATE: insert a new model row, return the generated id.
-- ---------------------------------------------------------------------------
INSERT INTO models (name, version, framework, artifact_uri, is_active)
VALUES ('fraud-detector', 'v3.2.1', 'pytorch', 's3://models/fraud/v3.2.1', TRUE)
RETURNING id;

-- ---------------------------------------------------------------------------
-- 2. READ: filter predictions for a model in a window, paginate by keyset.
-- ---------------------------------------------------------------------------
SELECT p.id, p.request_id, p.prediction, p.latency_ms, p.created_at
FROM predictions p
JOIN models m ON m.id = p.model_id
WHERE m.name    = 'fraud-detector'
  AND m.version = 'v3.2.1'
  AND p.created_at < $1               -- last-seen timestamp (cursor)
ORDER BY p.created_at DESC
LIMIT 50;

-- ---------------------------------------------------------------------------
-- 3. UPDATE: deactivate every active version of a model before activating
--    a new one. Wrap in a transaction so a partial update can't leave us
--    with zero active versions.
-- ---------------------------------------------------------------------------
BEGIN;
UPDATE models
SET    is_active = FALSE
WHERE  name = 'fraud-detector'
  AND  is_active = TRUE;

UPDATE models
SET    is_active = TRUE
WHERE  name    = 'fraud-detector'
  AND  version = 'v3.2.1';
COMMIT;

-- ---------------------------------------------------------------------------
-- 4. DELETE: prune predictions older than 90 days.
--    DELETE returns the number of affected rows via RETURNING (used by tests).
-- ---------------------------------------------------------------------------
DELETE FROM predictions
WHERE created_at < NOW() - INTERVAL '90 days'
RETURNING id;

-- ---------------------------------------------------------------------------
-- 5. UPSERT: idempotent registration of a model version.
-- ---------------------------------------------------------------------------
INSERT INTO models (name, version, framework, artifact_uri)
VALUES ($1, $2, $3, $4)
ON CONFLICT (name, version)
DO UPDATE SET artifact_uri = EXCLUDED.artifact_uri,
              framework    = EXCLUDED.framework
RETURNING id;

-- ---------------------------------------------------------------------------
-- 6. AGGREGATE: per-model summary over the last 24 hours.
-- ---------------------------------------------------------------------------
SELECT
    m.name,
    m.version,
    COUNT(*)                              AS predictions_count,
    AVG(p.latency_ms)::numeric(10,2)      AS avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY p.latency_ms) AS p95_latency_ms,
    MIN(p.created_at)                     AS first_pred,
    MAX(p.created_at)                     AS last_pred
FROM predictions p
JOIN models m ON m.id = p.model_id
WHERE p.created_at > NOW() - INTERVAL '24 hours'
GROUP BY m.name, m.version
HAVING COUNT(*) > 0
ORDER BY predictions_count DESC;
