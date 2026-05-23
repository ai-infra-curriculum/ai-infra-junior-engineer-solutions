-- ML Model Registry Schema (exercise-02 solution).
-- Justifications inline.

-- Optional but useful: native enums for status fields.
DO $$ BEGIN
    CREATE TYPE deployment_environment AS ENUM ('dev', 'staging', 'prod');
    EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE deployment_status AS ENUM ('pending', 'active', 'failed', 'rolled_back');
    EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


-- experiments: a logical grouping of runs (one per ML problem).
CREATE TABLE IF NOT EXISTS experiments (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    description  TEXT,
    owner        TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- models: the conceptual model (one per real-world use case).
-- A model has many versions over time.
CREATE TABLE IF NOT EXISTS models (
    id                      BIGSERIAL PRIMARY KEY,
    experiment_id           BIGINT NOT NULL REFERENCES experiments(id) ON DELETE RESTRICT,
    name                    TEXT NOT NULL UNIQUE,
    -- Denormalized for read-hot dashboards. Always refer to model_versions for truth.
    current_deployment_id   BIGINT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- model_versions: every trained artifact, immutable once created.
CREATE TABLE IF NOT EXISTS model_versions (
    id              BIGSERIAL PRIMARY KEY,
    model_id        BIGINT NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    version_tag     TEXT NOT NULL,
    artifact_uri    TEXT NOT NULL,
    framework       TEXT NOT NULL CHECK (framework IN ('pytorch', 'tensorflow', 'sklearn', 'onnx')),
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics         JSONB NOT NULL DEFAULT '{}'::jsonb,
    trained_by      TEXT NOT NULL,
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    archived_at     TIMESTAMPTZ,
    UNIQUE (model_id, version_tag)
);

-- Hot-path index: per-model history.
CREATE INDEX IF NOT EXISTS idx_versions_model_trained_at
    ON model_versions (model_id, trained_at DESC);

-- Expression index for the most common metric query (roc_auc).
CREATE INDEX IF NOT EXISTS idx_versions_roc_auc
    ON model_versions ((metrics ->> 'roc_auc'));

-- GIN on hyperparameters and metrics for ad-hoc filtering.
CREATE INDEX IF NOT EXISTS idx_versions_hyperparameters
    ON model_versions USING GIN (hyperparameters);
CREATE INDEX IF NOT EXISTS idx_versions_metrics
    ON model_versions USING GIN (metrics);


-- deployments: an instance of a model_version being served somewhere.
CREATE TABLE IF NOT EXISTS deployments (
    id                BIGSERIAL PRIMARY KEY,
    model_version_id  BIGINT NOT NULL REFERENCES model_versions(id) ON DELETE RESTRICT,
    environment       deployment_environment NOT NULL,
    status            deployment_status NOT NULL DEFAULT 'pending',
    endpoint_url      TEXT,
    deployed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retired_at        TIMESTAMPTZ
);

-- Enforce: at most one ACTIVE deployment per (model, environment) at any time.
-- A partial unique index does this cleanly without triggers.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_one_active_per_model_env
    ON deployments (model_version_id, environment)
    WHERE status = 'active' AND retired_at IS NULL;

-- Now wire the denormalized convenience column on models.
ALTER TABLE models
    DROP CONSTRAINT IF EXISTS models_current_deployment_id_fkey,
    ADD  CONSTRAINT models_current_deployment_id_fkey
         FOREIGN KEY (current_deployment_id) REFERENCES deployments(id) ON DELETE SET NULL;


-- model_version_lineage: many-to-many edges between versions.
-- Captures distillation, ensembling, fine-tune lineage, etc.
CREATE TABLE IF NOT EXISTS model_version_lineage (
    child_id   BIGINT NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    parent_id  BIGINT NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    edge_type  TEXT NOT NULL CHECK (edge_type IN ('fine_tune', 'distill', 'ensemble_member', 'retrain')),
    PRIMARY KEY (child_id, parent_id),
    CHECK (child_id <> parent_id)
);

CREATE INDEX IF NOT EXISTS idx_lineage_parent ON model_version_lineage (parent_id);
