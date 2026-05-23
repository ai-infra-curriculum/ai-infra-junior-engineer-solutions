-- Common queries against the ML registry.

-- Latest version per model.
SELECT DISTINCT ON (mv.model_id)
    m.name        AS model_name,
    mv.version_tag,
    mv.trained_at,
    mv.metrics ->> 'roc_auc' AS roc_auc
FROM model_versions mv
JOIN models m ON m.id = mv.model_id
WHERE mv.archived_at IS NULL
ORDER BY mv.model_id, mv.trained_at DESC;

-- Active deployment per model + environment.
SELECT m.name, d.environment, mv.version_tag, d.endpoint_url, d.deployed_at
FROM deployments d
JOIN model_versions mv ON mv.id = d.model_version_id
JOIN models m          ON m.id = mv.model_id
WHERE d.status = 'active' AND d.retired_at IS NULL
ORDER BY m.name, d.environment;

-- Lineage: ancestors of a specific version (CTE walk).
WITH RECURSIVE ancestors AS (
    SELECT parent_id, child_id, 1 AS depth
    FROM model_version_lineage
    WHERE child_id = $1
    UNION ALL
    SELECT mvl.parent_id, a.child_id, a.depth + 1
    FROM model_version_lineage mvl
    JOIN ancestors a ON mvl.child_id = a.parent_id
)
SELECT a.depth, mv.version_tag, mv.trained_at
FROM ancestors a
JOIN model_versions mv ON mv.id = a.parent_id
ORDER BY a.depth;

-- Top-3 model_versions in last 30 days by roc_auc, per experiment.
SELECT *
FROM (
    SELECT
        e.name AS experiment,
        m.name AS model_name,
        mv.version_tag,
        (mv.metrics ->> 'roc_auc')::float AS roc_auc,
        ROW_NUMBER() OVER (
            PARTITION BY e.id
            ORDER BY (mv.metrics ->> 'roc_auc')::float DESC NULLS LAST
        ) AS rk
    FROM model_versions mv
    JOIN models m       ON m.id = mv.model_id
    JOIN experiments e  ON e.id = m.experiment_id
    WHERE mv.trained_at > NOW() - INTERVAL '30 days'
      AND mv.metrics ? 'roc_auc'
) ranked
WHERE rk <= 3
ORDER BY experiment, rk;
