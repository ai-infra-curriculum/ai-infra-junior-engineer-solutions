-- Schema for exercise-01-sql-basics-crud.
-- Two tables: models (the deployed ML models) and predictions (every inference call).
-- This intentionally mirrors what a real model-serving service would persist.

CREATE TABLE IF NOT EXISTS models (
    id            BIGSERIAL PRIMARY KEY,
    name          TEXT        NOT NULL,
    version       TEXT        NOT NULL,
    framework     TEXT        NOT NULL CHECK (framework IN ('pytorch', 'tensorflow', 'sklearn', 'onnx')),
    artifact_uri  TEXT        NOT NULL,
    is_active     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS predictions (
    id             BIGSERIAL PRIMARY KEY,
    model_id       BIGINT      NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    request_id     UUID        NOT NULL,
    features       JSONB       NOT NULL,
    prediction     DOUBLE PRECISION NOT NULL,
    label_received DOUBLE PRECISION,           -- nullable: ground truth arrives later
    latency_ms     DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_model_created
    ON predictions (model_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_request_id
    ON predictions (request_id);
