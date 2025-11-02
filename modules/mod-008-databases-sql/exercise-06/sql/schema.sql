-- ML Model Registry Database Schema
-- PostgreSQL 15+

-- Drop tables if they exist (for clean re-creation)
DROP TABLE IF EXISTS experiments CASCADE;
DROP TABLE IF EXISTS model_metadata CASCADE;
DROP TABLE IF EXISTS model_versions CASCADE;
DROP TABLE IF EXISTS models CASCADE;

-- Models table: Top-level model definitions
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    CONSTRAINT unique_model_name UNIQUE(name)
);

-- Model versions table: Versioned instances of models
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    framework VARCHAR(50),  -- pytorch, tensorflow, scikit-learn, etc.
    artifact_uri TEXT,      -- s3://bucket/path/to/model.pt
    stage VARCHAR(20) DEFAULT 'staging',  -- staging, production, archived
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_model_version UNIQUE(model_id, version),
    CONSTRAINT valid_stage CHECK (stage IN ('staging', 'production', 'archived'))
);

-- Experiments table: Training metrics and results
CREATE TABLE experiments (
    id SERIAL PRIMARY KEY,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,  -- accuracy, precision, loss, etc.
    metric_value FLOAT NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model metadata table: Additional metadata with optimistic locking
CREATE TABLE model_metadata (
    model_version_id INTEGER PRIMARY KEY REFERENCES model_versions(id) ON DELETE CASCADE,
    tags JSONB,                         -- {"env": "prod", "team": "ml"}
    parameters JSONB,                   -- {"learning_rate": 0.001, "epochs": 100}
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version_lock INTEGER DEFAULT 0,     -- For optimistic locking
    CONSTRAINT version_lock_non_negative CHECK (version_lock >= 0)
);

-- Indexes for performance
CREATE INDEX idx_model_versions_model_id ON model_versions(model_id);
CREATE INDEX idx_model_versions_stage ON model_versions(stage);
CREATE INDEX idx_experiments_model_version_id ON experiments(model_version_id);
CREATE INDEX idx_experiments_metric_name ON experiments(metric_name);
CREATE INDEX idx_model_metadata_version_lock ON model_metadata(version_lock);

-- Sample data for testing
INSERT INTO models (name, description, created_by) VALUES
    ('fraud-detector', 'Credit card fraud detection model', 'alice'),
    ('recommender-v2', 'Product recommendation engine', 'bob'),
    ('churn-predictor', 'Customer churn prediction model', 'charlie');

INSERT INTO model_versions (model_id, version, framework, artifact_uri, stage) VALUES
    (1, 1, 'pytorch', 's3://ml-models/fraud-detector/v1.pt', 'archived'),
    (1, 2, 'pytorch', 's3://ml-models/fraud-detector/v2.pt', 'staging'),
    (1, 3, 'pytorch', 's3://ml-models/fraud-detector/v3.pt', 'production'),
    (2, 1, 'tensorflow', 's3://ml-models/recommender-v2/v1.h5', 'production'),
    (3, 1, 'scikit-learn', 's3://ml-models/churn-predictor/v1.pkl', 'staging');

INSERT INTO experiments (model_version_id, metric_name, metric_value) VALUES
    (1, 'accuracy', 0.92),
    (1, 'precision', 0.89),
    (1, 'recall', 0.91),
    (2, 'accuracy', 0.94),
    (2, 'precision', 0.91),
    (2, 'recall', 0.93),
    (3, 'accuracy', 0.96),
    (3, 'precision', 0.94),
    (3, 'recall', 0.95),
    (4, 'accuracy', 0.88),
    (4, 'auc', 0.92),
    (5, 'accuracy', 0.85),
    (5, 'f1_score', 0.83);

INSERT INTO model_metadata (model_version_id, tags, parameters) VALUES
    (1, '{"env": "dev", "experiment": "baseline"}', '{"learning_rate": 0.001, "epochs": 50}'),
    (2, '{"env": "staging", "experiment": "improved"}', '{"learning_rate": 0.0001, "epochs": 100}'),
    (3, '{"env": "production", "approved_by": "ml-team"}', '{"learning_rate": 0.0001, "epochs": 150, "batch_size": 32}'),
    (4, '{"env": "production", "model_type": "collaborative_filtering"}', '{"embedding_dim": 128, "num_factors": 64}'),
    (5, '{"env": "staging", "data_version": "2023-10"}', '{"max_depth": 10, "n_estimators": 100}');

-- Verification queries
SELECT 'Models:' as info, COUNT(*) as count FROM models;
SELECT 'Model Versions:' as info, COUNT(*) as count FROM model_versions;
SELECT 'Experiments:' as info, COUNT(*) as count FROM experiments;
SELECT 'Model Metadata:' as info, COUNT(*) as count FROM model_metadata;

-- Show sample data
SELECT m.name, mv.version, mv.stage, mv.framework
FROM models m
JOIN model_versions mv ON m.id = mv.model_id
ORDER BY m.name, mv.version;
