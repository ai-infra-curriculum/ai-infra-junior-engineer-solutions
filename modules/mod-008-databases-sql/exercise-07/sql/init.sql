-- ML Platform PostgreSQL Schema
-- Optimized for structured data requiring JOINs, transactions, and analytics

-- ============================================================================
-- DATASETS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    total_rows BIGINT,
    total_features INTEGER,
    storage_path TEXT,
    format VARCHAR(50) DEFAULT 'parquet',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    tags TEXT[]
);

CREATE INDEX idx_datasets_name ON datasets(name);
CREATE INDEX idx_datasets_created_by ON datasets(created_by);
CREATE INDEX idx_datasets_created_at ON datasets(created_at DESC);

COMMENT ON TABLE datasets IS 'Structured dataset metadata for ML training';
COMMENT ON COLUMN datasets.tags IS 'Array of tags for dataset categorization';

-- ============================================================================
-- TRAINING_RUNS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    model_name VARCHAR(255) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),

    -- Performance metrics
    accuracy FLOAT CHECK (accuracy BETWEEN 0 AND 1),
    precision_score FLOAT CHECK (precision_score BETWEEN 0 AND 1),
    recall FLOAT CHECK (recall_score BETWEEN 0 AND 1),
    f1_score FLOAT CHECK (f1_score BETWEEN 0 AND 1),
    loss FLOAT,
    auc_roc FLOAT CHECK (auc_roc BETWEEN 0 AND 1),

    -- Training metadata
    training_time_seconds INTEGER,
    total_epochs INTEGER,
    batch_size INTEGER,
    learning_rate FLOAT,

    -- Timestamps
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,

    -- Computed columns
    training_hours FLOAT GENERATED ALWAYS AS (training_time_seconds / 3600.0) STORED,
    accuracy_per_hour FLOAT
);

CREATE INDEX idx_training_runs_status ON training_runs(status);
CREATE INDEX idx_training_runs_model ON training_runs(model_name);
CREATE INDEX idx_training_runs_framework ON training_runs(framework);
CREATE INDEX idx_training_runs_dataset ON training_runs(dataset_id);
CREATE INDEX idx_training_runs_accuracy ON training_runs(accuracy DESC) WHERE status = 'completed';
CREATE INDEX idx_training_runs_started_at ON training_runs(started_at DESC);

COMMENT ON TABLE training_runs IS 'Structured training run metadata for analytics and reporting';
COMMENT ON COLUMN training_runs.accuracy_per_hour IS 'Efficiency metric: accuracy achieved per hour of training';

-- ============================================================================
-- MODEL_ARTIFACTS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_artifacts (
    id SERIAL PRIMARY KEY,
    training_run_id INTEGER REFERENCES training_runs(id) ON DELETE CASCADE,
    artifact_type VARCHAR(50) NOT NULL CHECK (artifact_type IN ('model_weights', 'checkpoint', 'onnx', 'torchscript', 'tensorrt')),
    storage_uri TEXT NOT NULL,
    size_bytes BIGINT,
    checksum_sha256 VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Computed
    size_mb FLOAT GENERATED ALWAYS AS (size_bytes / 1048576.0) STORED
);

CREATE INDEX idx_artifacts_training_run ON model_artifacts(training_run_id);
CREATE INDEX idx_artifacts_type ON model_artifacts(artifact_type);

COMMENT ON TABLE model_artifacts IS 'Storage locations and metadata for model artifacts';

-- ============================================================================
-- DEPLOYMENTS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS deployments (
    id SERIAL PRIMARY KEY,
    training_run_id INTEGER REFERENCES training_runs(id) ON DELETE SET NULL,
    deployment_name VARCHAR(255) UNIQUE NOT NULL,
    environment VARCHAR(50) NOT NULL CHECK (environment IN ('development', 'staging', 'production')),
    endpoint_url TEXT,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'deploying', 'active', 'inactive', 'failed')),

    -- Resource allocation
    cpu_cores FLOAT,
    memory_gb FLOAT,
    gpu_count INTEGER DEFAULT 0,
    replicas INTEGER DEFAULT 1,

    -- Timestamps
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_health_check TIMESTAMP,
    deactivated_at TIMESTAMP
);

CREATE INDEX idx_deployments_status ON deployments(status);
CREATE INDEX idx_deployments_environment ON deployments(environment);
CREATE INDEX idx_deployments_training_run ON deployments(training_run_id);

COMMENT ON TABLE deployments IS 'Model deployment metadata and resource allocation';

-- ============================================================================
-- KEY-VALUE STORE (for benchmarking against Redis)
-- ============================================================================
CREATE TABLE IF NOT EXISTS kv_store (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX idx_kv_expires ON kv_store(expires_at) WHERE expires_at IS NOT NULL;

COMMENT ON TABLE kv_store IS 'Key-value store for performance benchmarking against Redis';

-- ============================================================================
-- SAMPLE DATA
-- ============================================================================

-- Insert sample datasets
INSERT INTO datasets (name, description, total_rows, total_features, storage_path, created_by, tags) VALUES
    ('fraud-train-v1', 'Credit card fraud detection dataset', 284807, 30, 's3://ml-data/fraud/train_v1.parquet', 'alice', ARRAY['finance', 'classification', 'imbalanced']),
    ('fraud-train-v2', 'Credit card fraud detection dataset (updated)', 320000, 32, 's3://ml-data/fraud/train_v2.parquet', 'alice', ARRAY['finance', 'classification', 'imbalanced']),
    ('recommendation-train-v2', 'User-item interaction dataset', 10000000, 50, 's3://ml-data/recsys/train_v2.parquet', 'bob', ARRAY['recommendation', 'collaborative-filtering']),
    ('sentiment-tweets', 'Twitter sentiment analysis dataset', 1600000, 15, 's3://ml-data/nlp/sentiment_tweets.parquet', 'charlie', ARRAY['nlp', 'sentiment', 'classification']),
    ('churn-customers', 'Customer churn prediction dataset', 50000, 25, 's3://ml-data/churn/customers.parquet', 'bob', ARRAY['finance', 'classification', 'churn'])
ON CONFLICT (name) DO NOTHING;

-- Insert sample training runs
INSERT INTO training_runs
    (dataset_id, model_name, framework, status, accuracy, precision_score, recall, f1_score, loss, training_time_seconds, total_epochs, batch_size, learning_rate, completed_at)
VALUES
    (1, 'fraud-detector-v1', 'pytorch', 'completed', 0.9845, 0.9712, 0.9834, 0.9772, 0.042, 3600, 50, 32, 0.001, NOW() - INTERVAL '2 days'),
    (1, 'fraud-detector-v2', 'tensorflow', 'completed', 0.9821, 0.9698, 0.9802, 0.9749, 0.051, 4200, 60, 64, 0.0005, NOW() - INTERVAL '1 day'),
    (1, 'fraud-detector-xgb', 'xgboost', 'completed', 0.9789, 0.9645, 0.9778, 0.9711, 0.067, 1800, 100, 128, 0.1, NOW() - INTERVAL '3 days'),
    (2, 'recommender-model', 'pytorch', 'running', NULL, NULL, NULL, NULL, NULL, NULL, 30, 256, 0.001, NULL),
    (3, 'sentiment-bert', 'huggingface', 'completed', 0.9234, 0.9187, 0.9201, 0.9194, 0.124, 7200, 3, 16, 0.00002, NOW() - INTERVAL '5 hours'),
    (5, 'churn-predictor-v1', 'xgboost', 'completed', 0.8934, 0.8756, 0.8812, 0.8784, 0.234, 2100, 100, 64, 0.3, NOW() - INTERVAL '12 hours')
ON CONFLICT DO NOTHING;

-- Update computed accuracy_per_hour
UPDATE training_runs
SET accuracy_per_hour = ROUND((accuracy / (training_time_seconds / 3600.0))::numeric, 4)
WHERE status = 'completed' AND training_time_seconds > 0 AND accuracy IS NOT NULL;

-- Insert sample model artifacts
INSERT INTO model_artifacts (training_run_id, artifact_type, storage_uri, size_bytes, checksum_sha256) VALUES
    (1, 'model_weights', 's3://ml-models/fraud-detector-v1/model.pt', 245600000, 'abc123def456'),
    (1, 'onnx', 's3://ml-models/fraud-detector-v1/model.onnx', 123400000, 'def789ghi012'),
    (2, 'model_weights', 's3://ml-models/fraud-detector-v2/model.h5', 198700000, 'ghi345jkl678'),
    (3, 'model_weights', 's3://ml-models/fraud-detector-xgb/model.xgb', 45600000, 'jkl901mno234'),
    (5, 'model_weights', 's3://ml-models/sentiment-bert/pytorch_model.bin', 438000000, 'mno567pqr890'),
    (6, 'model_weights', 's3://ml-models/churn-predictor-v1/model.xgb', 38900000, 'pqr123stu456')
ON CONFLICT DO NOTHING;

-- Insert sample deployments
INSERT INTO deployments
    (training_run_id, deployment_name, environment, endpoint_url, status, cpu_cores, memory_gb, gpu_count, replicas)
VALUES
    (1, 'fraud-detector-prod', 'production', 'https://api.mlplatform.com/fraud-detector', 'active', 4.0, 16.0, 1, 3),
    (2, 'fraud-detector-staging', 'staging', 'https://staging-api.mlplatform.com/fraud-detector', 'active', 2.0, 8.0, 0, 2),
    (5, 'sentiment-bert-prod', 'production', 'https://api.mlplatform.com/sentiment', 'active', 8.0, 32.0, 2, 4),
    (6, 'churn-predictor-dev', 'development', 'http://localhost:8000/churn', 'inactive', 1.0, 4.0, 0, 1)
ON CONFLICT (deployment_name) DO NOTHING;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Complete model information with dataset details
CREATE OR REPLACE VIEW model_performance AS
SELECT
    tr.id AS training_run_id,
    tr.model_name,
    tr.framework,
    tr.status,
    d.name AS dataset_name,
    d.total_rows AS dataset_size,
    tr.accuracy,
    tr.f1_score,
    tr.training_time_seconds,
    tr.training_hours,
    tr.accuracy_per_hour,
    tr.started_at,
    tr.completed_at
FROM training_runs tr
JOIN datasets d ON tr.dataset_id = d.id
WHERE tr.status = 'completed';

COMMENT ON VIEW model_performance IS 'Consolidated view of model performance metrics with dataset information';

-- View: Deployment status with model performance
CREATE OR REPLACE VIEW deployment_summary AS
SELECT
    dep.id AS deployment_id,
    dep.deployment_name,
    dep.environment,
    dep.status AS deployment_status,
    dep.replicas,
    dep.cpu_cores,
    dep.memory_gb,
    dep.gpu_count,
    tr.model_name,
    tr.framework,
    tr.accuracy,
    tr.f1_score,
    dep.deployed_at,
    dep.last_health_check
FROM deployments dep
LEFT JOIN training_runs tr ON dep.training_run_id = tr.id;

COMMENT ON VIEW deployment_summary IS 'Overview of all deployments with model performance metrics';

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Get top N models by accuracy
CREATE OR REPLACE FUNCTION get_top_models(n INTEGER DEFAULT 5)
RETURNS TABLE (
    model_name VARCHAR,
    framework VARCHAR,
    accuracy FLOAT,
    dataset_name VARCHAR,
    training_hours FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        tr.model_name,
        tr.framework,
        tr.accuracy,
        d.name AS dataset_name,
        tr.training_hours
    FROM training_runs tr
    JOIN datasets d ON tr.dataset_id = d.id
    WHERE tr.status = 'completed' AND tr.accuracy IS NOT NULL
    ORDER BY tr.accuracy DESC
    LIMIT n;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_top_models IS 'Returns top N models by accuracy';

-- Function: Calculate average metrics by framework
CREATE OR REPLACE FUNCTION framework_performance_stats()
RETURNS TABLE (
    framework VARCHAR,
    total_runs BIGINT,
    avg_accuracy NUMERIC,
    avg_training_time NUMERIC,
    best_accuracy FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        tr.framework,
        COUNT(*)::BIGINT AS total_runs,
        ROUND(AVG(tr.accuracy)::NUMERIC, 4) AS avg_accuracy,
        ROUND(AVG(tr.training_time_seconds)::NUMERIC, 2) AS avg_training_time,
        MAX(tr.accuracy) AS best_accuracy
    FROM training_runs tr
    WHERE tr.status = 'completed'
    GROUP BY tr.framework
    ORDER BY avg_accuracy DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION framework_performance_stats IS 'Aggregate performance statistics grouped by framework';

-- ============================================================================
-- GRANTS (if needed for specific users)
-- ============================================================================
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mluser;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mluser;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE 'ML Platform PostgreSQL schema initialized successfully';
    RAISE NOTICE '  - 5 tables created (datasets, training_runs, model_artifacts, deployments, kv_store)';
    RAISE NOTICE '  - 2 views created (model_performance, deployment_summary)';
    RAISE NOTICE '  - 2 functions created (get_top_models, framework_performance_stats)';
    RAISE NOTICE '  - Sample data inserted for testing';
END $$;
