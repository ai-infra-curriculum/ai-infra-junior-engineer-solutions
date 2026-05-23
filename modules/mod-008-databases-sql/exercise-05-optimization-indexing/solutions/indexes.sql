-- Indexes added to the ML registry workload, justified inline.

-- 1. Hot read: "predictions for model X in the last 24h" -> compound index with
--    equality on model_id, range on created_at. Order matters (ESR rule).
CREATE INDEX IF NOT EXISTS idx_predictions_model_created
    ON predictions (model_id, created_at DESC);

-- 2. Lookup by request_id (idempotency checks). Unique to prevent duplicates.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_predictions_request_id
    ON predictions (request_id);

-- 3. JSONB metrics: GIN for ad-hoc containment queries like
--    `WHERE metrics @> '{"roc_auc": 0.9}'`. Note: GIN is slower to update
--    than B-tree; acceptable because metrics are written once per training run.
CREATE INDEX IF NOT EXISTS idx_versions_metrics_gin
    ON model_versions USING GIN (metrics);

-- 4. Expression index for the most common scalar metric query.
--    Trades disk for >100x speedup on `WHERE (metrics->>'roc_auc')::float > 0.9`.
CREATE INDEX IF NOT EXISTS idx_versions_roc_auc
    ON model_versions (((metrics ->> 'roc_auc')::float));

-- 5. Partial index: only deployments that are currently active. Tiny index,
--    huge speedup for the "show me what's running" dashboard.
CREATE INDEX IF NOT EXISTS idx_deployments_active
    ON deployments (environment, model_version_id)
    WHERE status = 'active' AND retired_at IS NULL;

-- 6. BRIN for time-series prune jobs. predictions table grows monotonically
--    by created_at, so BRIN is a small, low-maintenance fit.
CREATE INDEX IF NOT EXISTS idx_predictions_created_at_brin
    ON predictions USING BRIN (created_at);

-- After adding indexes, refresh planner stats:
ANALYZE predictions;
ANALYZE model_versions;
ANALYZE deployments;
