-- Solutions for advanced-SQL exercise.
-- Schema reused from exercise-02 (ML model registry).

-- ----------------------------------------------------------------------
-- INNER JOIN: every prediction with its model name and experiment.
-- ----------------------------------------------------------------------
SELECT p.id, p.prediction, m.name AS model_name, e.name AS experiment
FROM predictions p
JOIN model_versions mv ON mv.id = p.model_id
JOIN models m          ON m.id = mv.model_id
JOIN experiments e     ON e.id = m.experiment_id;

-- ----------------------------------------------------------------------
-- LEFT JOIN: every model_version with prediction count (may be zero).
-- ----------------------------------------------------------------------
SELECT
    mv.id,
    mv.version_tag,
    COALESCE(stats.prediction_count, 0) AS prediction_count
FROM model_versions mv
LEFT JOIN (
    SELECT model_id, COUNT(*) AS prediction_count
    FROM predictions
    GROUP BY model_id
) stats ON stats.model_id = mv.id;

-- ----------------------------------------------------------------------
-- Anti-join: model_versions that have NEVER served a prediction.
-- ----------------------------------------------------------------------
SELECT mv.id, mv.version_tag
FROM model_versions mv
WHERE NOT EXISTS (
    SELECT 1 FROM predictions p WHERE p.model_id = mv.id
);

-- ----------------------------------------------------------------------
-- Semi-join: model_versions WITH at least one prediction (deduped via EXISTS).
-- ----------------------------------------------------------------------
SELECT mv.id, mv.version_tag
FROM model_versions mv
WHERE EXISTS (
    SELECT 1 FROM predictions p WHERE p.model_id = mv.id
);

-- ----------------------------------------------------------------------
-- CTE: split a complex aggregation into named steps.
-- ----------------------------------------------------------------------
WITH recent AS (
    SELECT model_id, prediction, latency_ms
    FROM predictions
    WHERE created_at > NOW() - INTERVAL '24 hours'
),
agg AS (
    SELECT
        model_id,
        COUNT(*)                   AS n,
        AVG(latency_ms)            AS avg_latency,
        STDDEV_SAMP(latency_ms)    AS sd_latency
    FROM recent
    GROUP BY model_id
)
SELECT mv.version_tag, agg.n, agg.avg_latency, agg.sd_latency
FROM agg
JOIN model_versions mv ON mv.id = agg.model_id
ORDER BY agg.n DESC;

-- ----------------------------------------------------------------------
-- Recursive CTE: walk the lineage tree from a given model_version up.
-- ----------------------------------------------------------------------
WITH RECURSIVE ancestry AS (
    SELECT child_id, parent_id, 1 AS depth
    FROM model_version_lineage
    WHERE child_id = $1
    UNION ALL
    SELECT a.child_id, mvl.parent_id, a.depth + 1
    FROM ancestry a
    JOIN model_version_lineage mvl ON mvl.child_id = a.parent_id
)
SELECT depth, parent_id FROM ancestry ORDER BY depth;

-- ----------------------------------------------------------------------
-- Window function: rank model_versions per model by ROC-AUC.
-- ----------------------------------------------------------------------
SELECT
    m.name,
    mv.version_tag,
    (mv.metrics ->> 'roc_auc')::float AS roc_auc,
    RANK() OVER (
        PARTITION BY mv.model_id
        ORDER BY (mv.metrics ->> 'roc_auc')::float DESC NULLS LAST
    ) AS rank_within_model
FROM model_versions mv
JOIN models m ON m.id = mv.model_id
WHERE mv.metrics ? 'roc_auc';

-- ----------------------------------------------------------------------
-- LAG/LEAD: each prediction with the latency of the previous one for that model.
-- ----------------------------------------------------------------------
SELECT
    p.model_id,
    p.created_at,
    p.latency_ms,
    LAG(p.latency_ms) OVER (PARTITION BY p.model_id ORDER BY p.created_at) AS prev_latency
FROM predictions p;

-- ----------------------------------------------------------------------
-- LATERAL: top-3 latest predictions per model in one query.
-- ----------------------------------------------------------------------
SELECT m.name, latest.created_at, latest.prediction
FROM models m
JOIN model_versions mv ON mv.model_id = m.id
JOIN LATERAL (
    SELECT created_at, prediction
    FROM predictions p
    WHERE p.model_id = mv.id
    ORDER BY p.created_at DESC
    LIMIT 3
) AS latest ON TRUE;
