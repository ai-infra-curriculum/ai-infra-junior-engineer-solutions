# Exercise 02 Solution: Database Design for ML Model Registry

## Solution Overview

This implementation guide provides a complete solution for Exercise 02: designing and implementing a production-grade, normalized relational database schema for an ML Model Registry. You'll master entity-relationship modeling, normalization principles (1NF, 2NF, 3NF), foreign keys, and referential integrity.

**What You'll Build**:
- Fully normalized ML Model Registry database (3NF)
- 9 core tables with proper relationships
- Foreign keys with CASCADE behavior
- Lookup tables for environments and teams
- Junction tables for many-to-many relationships
- Complete audit trail and versioning system

**Time to Complete**: 3-4 hours
**Difficulty**: Intermediate

---

## Table of Contents

1. [Understanding Normalization](#part-1-understanding-normalization)
2. [Entity Relationships](#part-2-entity-relationships)
3. [Schema Implementation](#part-3-schema-implementation)
4. [Seed Data](#part-4-seed-data)
5. [Querying Related Tables](#part-5-querying-related-tables)
6. [Foreign Key Constraints](#part-6-foreign-key-constraints)
7. [Many-to-Many Relationships](#part-7-many-to-many-relationships)
8. [Production Patterns](#part-8-production-patterns)
9. [Verification & Testing](#verification--testing)

---

## Part 1: Understanding Normalization

### Step 1.1: The Problem - Denormalized Data

Create project directory:

```bash
mkdir -p ~/ml-model-registry
cd ~/ml-model-registry
mkdir -p sql docs scripts
```

**Bad Example** - Everything in one table:

```sql
-- DON'T DO THIS! (Anti-pattern)
CREATE TABLE model_everything (
    id UUID PRIMARY KEY,
    model_name TEXT,
    model_description TEXT,
    model_owner TEXT,
    owner_email TEXT,              -- Redundant
    owner_team TEXT,               -- Redundant
    version TEXT,                  -- Multiple versions = multiple rows
    git_commit TEXT,
    deployment_env TEXT,           -- CSV string: "dev,staging,prod"?
    deployment_dates TEXT,         -- CSV string: "2025-01-01,2025-01-15"?
    training_run_ids TEXT,         -- CSV string: "uuid1,uuid2,uuid3"?
    dataset_names TEXT,            -- CSV string: "dataset1,dataset2"?
    tags TEXT,                     -- CSV string: "nlp,production,bert"?
    approval_status TEXT,
    approved_by TEXT,
    approved_at TIMESTAMP
);
```

**Problems with this approach**:
1. **Update Anomalies**: Change owner email in one row, forget others → inconsistent data
2. **Insertion Anomalies**: Can't add a model without a version
3. **Deletion Anomalies**: Delete last version → lose entire model information
4. **No Referential Integrity**: Can't enforce that deployment_env values are valid
5. **Query Inefficiency**: Can't efficiently find "all models deployed to prod"
6. **Storage Waste**: Duplicate data (model_name, owner, etc.) for every version

### Step 1.2: Normalization Forms

**First Normal Form (1NF)**:
- ✅ Each column contains atomic values (no arrays, no CSV strings)
- ✅ Each row is unique (has primary key)
- ✅ No repeating groups

**Second Normal Form (2NF)**:
- ✅ Meets 1NF
- ✅ No partial dependencies (all non-key columns depend on entire primary key)

**Third Normal Form (3NF)**:
- ✅ Meets 2NF
- ✅ No transitive dependencies (non-key columns don't depend on other non-key columns)

**Example of 3NF violation**:
```sql
-- Bad: owner_team depends on owner_email (transitive dependency)
CREATE TABLE models (
    model_id UUID PRIMARY KEY,
    model_name TEXT,
    owner_email TEXT,
    owner_team TEXT  -- This depends on owner_email, not model_id!
);

-- Good: Separate teams table
CREATE TABLE teams (
    team_id UUID PRIMARY KEY,
    team_name TEXT
);

CREATE TABLE models (
    model_id UUID PRIMARY KEY,
    model_name TEXT,
    team_id UUID REFERENCES teams(team_id)  -- Direct dependency
);
```

### Step 1.3: Identifying Entities

Our normalized schema will have:

**Core Entities** (9 tables):
1. `teams` - Organizational teams (lookup table)
2. `environments` - Deployment targets (lookup table)
3. `models` - ML model catalog
4. `model_versions` - Versioned model artifacts
5. `datasets` - Training data catalog
6. `training_runs` - Individual training executions
7. `deployments` - Where versions are deployed
8. `tags` - Flexible metadata labels
9. `approvals` - Compliance approvals

**Junction Tables** (3 tables):
10. `model_tags` - Many-to-many: models ↔ tags
11. `dataset_tags` - Many-to-many: datasets ↔ tags
12. `run_datasets` - Many-to-many: training_runs ↔ datasets

---

## Part 2: Entity Relationships

### Step 2.1: Relationship Types

**One-to-Many (1:N)**:
- One team → Many models
- One model → Many versions
- One version → Many training runs
- One version → Many deployments
- One environment → Many deployments

**Many-to-Many (M:N)** via junction tables:
- Models ↔ Tags (via `model_tags`)
- Datasets ↔ Tags (via `dataset_tags`)
- Training Runs ↔ Datasets (via `run_datasets`)

### Step 2.2: ER Diagram

```
┌──────────┐
│  teams   │───────┐
└──────────┘       │
                   │ 1:N (team owns models)
                   ▼
            ┌─────────────┐       ┌──────────────┐
            │   models    │◄─────►│  model_tags  │
            └─────────────┘       └──────────────┘
                   │                      │
                   │ 1:N                  │ M:N
                   ▼                      ▼
        ┌───────────────────┐      ┌─────────┐
        │ model_versions    │      │  tags   │
        └───────────────────┘      └─────────┘
                   │                      ▲
         ┌─────────┴─────────┐            │
         │ 1:N               │ 1:N        │ M:N
         ▼                   ▼            │
  ┌─────────────┐    ┌──────────────┐    │
  │training_runs│    │ deployments  │    │
  └─────────────┘    └──────────────┘    │
         │                   │            │
         │ M:N               │ N:1        │
         ▼                   ▼            │
  ┌─────────────┐    ┌──────────────┐    │
  │run_datasets │    │environments  │    │
  └─────────────┘    └──────────────┘    │
         │                                │
         ▼                                │
  ┌─────────────┐                        │
  │  datasets   │◄───────────────────────┘
  └─────────────┘     (dataset_tags)
```

### Step 2.3: Foreign Key Cascading

**CASCADE Strategies**:

| Constraint | Use Case | Example |
|------------|----------|---------|
| `ON DELETE CASCADE` | Child should be deleted with parent | Delete model → delete all versions |
| `ON DELETE SET NULL` | Keep child, remove reference | Delete team → set model.team_id = NULL |
| `ON DELETE RESTRICT` | Prevent deletion if children exist | Can't delete environment if deployments exist |
| `ON UPDATE CASCADE` | Update foreign key if parent changes | Rarely used with UUIDs |

---

## Part 3: Schema Implementation

### Step 3.1: Start PostgreSQL

```bash
# Start PostgreSQL container
docker run --name pg-ml-registry \
  -e POSTGRES_PASSWORD=mlops_secure_pass \
  -e POSTGRES_USER=mlops \
  -e POSTGRES_DB=ml_registry \
  -p 5432:5432 \
  -d postgres:14

# Wait for startup
sleep 5

# Verify
docker exec pg-ml-registry psql -U mlops -d ml_registry -c "SELECT version();"
```

### Step 3.2: Create Complete Schema

Create `sql/10_model_registry_schema.sql`:

```sql
-- ============================================
-- ML Model Registry - Complete Schema
-- ============================================
-- Purpose: Production-grade normalized schema
-- Normalization: 3NF (Third Normal Form)
-- Author: ML Infrastructure Team
-- Date: 2025-11-01
-- Version: 1.0.0
-- ============================================

-- Drop existing objects (development only)
DROP TABLE IF EXISTS model_tags CASCADE;
DROP TABLE IF EXISTS dataset_tags CASCADE;
DROP TABLE IF EXISTS run_datasets CASCADE;
DROP TABLE IF EXISTS approvals CASCADE;
DROP TABLE IF EXISTS deployments CASCADE;
DROP TABLE IF EXISTS training_runs CASCADE;
DROP TABLE IF EXISTS model_versions CASCADE;
DROP TABLE IF EXISTS models CASCADE;
DROP TABLE IF EXISTS datasets CASCADE;
DROP TABLE IF EXISTS tags CASCADE;
DROP TABLE IF EXISTS environments CASCADE;
DROP TABLE IF EXISTS teams CASCADE;

-- ============================================
-- LOOKUP TABLES (Reference Data)
-- ============================================

-- Teams table
CREATE TABLE teams (
    team_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_name TEXT UNIQUE NOT NULL,
    team_email TEXT NOT NULL,
    department TEXT,
    cost_center TEXT,
    manager_name TEXT,
    slack_channel TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_team_email CHECK (
        team_email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
    ),
    CONSTRAINT valid_team_name CHECK (
        team_name ~ '^[a-z0-9-]+$'  -- Kebab-case
    )
);

CREATE INDEX idx_teams_name ON teams(team_name);
COMMENT ON TABLE teams IS 'Organizational teams owning ML models';

-- Environments table
CREATE TABLE environments (
    environment_id SERIAL PRIMARY KEY,
    environment_name TEXT UNIQUE NOT NULL,
    environment_type TEXT NOT NULL,
    description TEXT,
    cluster_name TEXT,
    region TEXT,
    priority INTEGER DEFAULT 0,
    requires_approval BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_env_type CHECK (
        environment_type IN ('development', 'staging', 'production', 'canary', 'shadow')
    )
);

CREATE INDEX idx_environments_type ON environments(environment_type);
COMMENT ON TABLE environments IS 'Deployment targets for ML models';

-- Insert default environments
INSERT INTO environments (environment_name, environment_type, description, cluster_name, region, priority, requires_approval)
VALUES
    ('dev', 'development', 'Local development', 'dev-cluster', 'us-west-2', 0, FALSE),
    ('qa', 'development', 'QA testing', 'qa-cluster', 'us-west-2', 1, FALSE),
    ('staging', 'staging', 'Pre-production staging', 'staging-cluster', 'us-east-1', 2, FALSE),
    ('staging-eu', 'staging', 'EU staging', 'staging-eu-cluster', 'eu-west-1', 2, FALSE),
    ('canary', 'canary', 'Canary deployment (1% traffic)', 'prod-cluster', 'us-east-1', 3, TRUE),
    ('prod-us', 'production', 'Production - United States', 'prod-us-cluster', 'us-east-1', 4, TRUE),
    ('prod-eu', 'production', 'Production - Europe', 'prod-eu-cluster', 'eu-west-1', 4, TRUE),
    ('prod-apac', 'production', 'Production - Asia Pacific', 'prod-apac-cluster', 'ap-southeast-1', 4, TRUE);

-- Tags table (flexible categorization)
CREATE TABLE tags (
    tag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tag_name TEXT UNIQUE NOT NULL,
    tag_category TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_tag_name CHECK (
        tag_name ~ '^[a-z0-9-_]+$'
    ),
    CONSTRAINT valid_tag_category CHECK (
        tag_category IN ('domain', 'framework', 'use-case', 'team', 'priority', 'status')
    )
);

CREATE INDEX idx_tags_name ON tags(tag_name);
CREATE INDEX idx_tags_category ON tags(tag_category);
COMMENT ON TABLE tags IS 'Flexible tagging system for models and datasets';

-- ============================================
-- CORE ENTITIES
-- ============================================

-- Models table (catalog of ML models)
CREATE TABLE models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    objective TEXT,
    use_case TEXT,

    -- Ownership (FK to teams)
    team_id UUID REFERENCES teams(team_id) ON DELETE SET NULL,
    primary_contact TEXT NOT NULL,
    secondary_contact TEXT,

    -- Business classification
    business_unit TEXT,
    risk_level TEXT DEFAULT 'medium',
    compliance_required BOOLEAN DEFAULT FALSE,

    -- Documentation
    github_repo TEXT,
    documentation_url TEXT,
    model_card_url TEXT,

    -- Lifecycle
    is_active BOOLEAN DEFAULT TRUE,
    archived_at TIMESTAMP WITH TIME ZONE,
    archived_reason TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by TEXT DEFAULT CURRENT_USER,

    -- Constraints
    CONSTRAINT valid_risk_level CHECK (
        risk_level IN ('low', 'medium', 'high', 'critical')
    ),
    CONSTRAINT valid_model_name CHECK (
        model_name ~ '^[a-z0-9-]+$'  -- Kebab-case
    )
);

CREATE INDEX idx_models_team ON models(team_id);
CREATE INDEX idx_models_risk ON models(risk_level);
CREATE INDEX idx_models_name ON models(model_name);
CREATE INDEX idx_models_created ON models(created_at DESC);
CREATE INDEX idx_models_active ON models(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE models IS 'Central catalog of ML models';
COMMENT ON COLUMN models.model_name IS 'Unique identifier (kebab-case, e.g. "bert-sentiment-classifier")';
COMMENT ON COLUMN models.risk_level IS 'Business risk: low, medium, high, critical';
COMMENT ON COLUMN models.compliance_required IS 'Requires approval workflow before production';

-- Model Versions table
CREATE TABLE model_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,

    -- Version information
    semver TEXT NOT NULL,  -- 1.2.3
    version_alias TEXT,    -- "latest", "stable", "v1"
    description TEXT,
    release_notes TEXT,

    -- Artifacts
    git_commit TEXT,
    git_branch TEXT,
    git_tag TEXT,
    artifact_uri TEXT NOT NULL,     -- s3://bucket/path or gs://bucket/path
    artifact_size_mb NUMERIC(10,2),
    checksum_sha256 TEXT,

    -- Framework and dependencies
    framework TEXT NOT NULL,
    framework_version TEXT,
    python_version TEXT,
    cuda_version TEXT,
    dependencies JSONB DEFAULT '{}'::jsonb,

    -- Model format
    model_format TEXT,  -- 'onnx', 'torchscript', 'savedmodel', 'pickle'
    input_schema JSONB,
    output_schema JSONB,

    -- Status and lifecycle
    status TEXT NOT NULL DEFAULT 'registered',
    registered_by TEXT NOT NULL,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    promoted_to_production_at TIMESTAMP WITH TIME ZONE,
    deprecated_at TIMESTAMP WITH TIME ZONE,
    deprecation_reason TEXT,

    -- Performance benchmarks
    benchmark_accuracy NUMERIC(5,4),
    benchmark_latency_p50_ms NUMERIC(8,2),
    benchmark_latency_p99_ms NUMERIC(8,2),
    benchmark_throughput_qps NUMERIC(10,2),
    benchmark_memory_mb NUMERIC(10,2),

    -- Metadata
    tags JSONB DEFAULT '[]'::jsonb,
    custom_metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT unique_model_version UNIQUE (model_id, semver),
    CONSTRAINT valid_status CHECK (
        status IN ('registered', 'validated', 'deployed', 'deprecated', 'archived')
    ),
    CONSTRAINT valid_framework CHECK (
        framework IN ('pytorch', 'tensorflow', 'sklearn', 'xgboost', 'jax',
                     'lightgbm', 'catboost', 'onnx', 'huggingface', 'custom')
    ),
    CONSTRAINT valid_semver CHECK (
        semver ~ '^\d+\.\d+\.\d+(-[a-z0-9.]+)?(\+[a-z0-9.]+)?$'
    ),
    CONSTRAINT valid_checksum CHECK (
        checksum_sha256 IS NULL OR LENGTH(checksum_sha256) = 64
    ),
    CONSTRAINT valid_model_format CHECK (
        model_format IN ('onnx', 'torchscript', 'savedmodel', 'pickle', 'joblib', 'mlflow', 'custom')
    )
);

CREATE INDEX idx_versions_model ON model_versions(model_id);
CREATE INDEX idx_versions_status ON model_versions(status);
CREATE INDEX idx_versions_semver ON model_versions(model_id, semver DESC);
CREATE INDEX idx_versions_registered ON model_versions(registered_at DESC);
CREATE INDEX idx_versions_framework ON model_versions(framework);
CREATE INDEX idx_versions_alias ON model_versions(model_id, version_alias)
    WHERE version_alias IS NOT NULL;

COMMENT ON TABLE model_versions IS 'Versioned ML model artifacts with semantic versioning';
COMMENT ON COLUMN model_versions.semver IS 'Semantic version (e.g., 1.2.3, 2.0.0-beta.1)';
COMMENT ON COLUMN model_versions.artifact_uri IS 'Cloud storage URI (s3://, gs://, abfss://)';

-- Datasets table
CREATE TABLE datasets (
    dataset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    purpose TEXT,

    -- Storage
    storage_location TEXT NOT NULL,
    storage_format TEXT NOT NULL,
    storage_size_gb NUMERIC(12,2),
    partition_keys TEXT[],

    -- Schema
    schema_url TEXT,
    schema_version TEXT,
    data_version TEXT,

    -- Ownership (FK to teams)
    team_id UUID REFERENCES teams(team_id) ON DELETE SET NULL,
    data_steward TEXT NOT NULL,

    -- Data quality
    freshness_sla_hours INTEGER,
    last_updated_at TIMESTAMP WITH TIME ZONE,
    row_count BIGINT,
    column_count INTEGER,
    null_percentage NUMERIC(5,2),

    -- Compliance and governance
    contains_pii BOOLEAN DEFAULT FALSE,
    data_classification TEXT DEFAULT 'internal',
    retention_days INTEGER,
    gdpr_compliant BOOLEAN DEFAULT FALSE,

    -- Lineage
    source_system TEXT,
    transformation_pipeline TEXT,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by TEXT DEFAULT CURRENT_USER,

    -- Constraints
    CONSTRAINT valid_freshness CHECK (freshness_sla_hours IS NULL OR freshness_sla_hours > 0),
    CONSTRAINT valid_storage_format CHECK (
        storage_format IN ('parquet', 'csv', 'json', 'avro', 'tfrecord', 'hdf5', 'arrow', 'delta')
    ),
    CONSTRAINT valid_classification CHECK (
        data_classification IN ('public', 'internal', 'confidential', 'restricted')
    ),
    CONSTRAINT valid_dataset_name CHECK (
        dataset_name ~ '^[a-z0-9-_]+$'
    )
);

CREATE INDEX idx_datasets_name ON datasets(dataset_name);
CREATE INDEX idx_datasets_team ON datasets(team_id);
CREATE INDEX idx_datasets_updated ON datasets(last_updated_at DESC);
CREATE INDEX idx_datasets_classification ON datasets(data_classification);
CREATE INDEX idx_datasets_pii ON datasets(contains_pii) WHERE contains_pii = TRUE;

COMMENT ON TABLE datasets IS 'Catalog of datasets used for ML training and validation';
COMMENT ON COLUMN datasets.contains_pii IS 'Contains Personally Identifiable Information';
COMMENT ON COLUMN datasets.data_classification IS 'Security classification: public, internal, confidential, restricted';

-- Training Runs table
CREATE TABLE training_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,

    -- Run identification
    run_name TEXT,
    experiment_name TEXT NOT NULL,
    run_number INTEGER,

    -- Status
    status TEXT NOT NULL DEFAULT 'queued',

    -- Metrics
    accuracy NUMERIC(5,4),
    precision_score NUMERIC(5,4),
    recall_score NUMERIC(5,4),
    f1_score NUMERIC(5,4),
    loss NUMERIC(10,5),
    auc_roc NUMERIC(5,4),

    -- Custom metrics (JSONB)
    custom_metrics JSONB DEFAULT '{}'::jsonb,

    -- Hyperparameters
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Compute resources
    compute_target TEXT,
    instance_type TEXT,
    gpu_count INTEGER DEFAULT 0,
    gpu_type TEXT,
    gpu_hours NUMERIC(10,2) DEFAULT 0,
    cpu_hours NUMERIC(10,2) DEFAULT 0,
    memory_gb INTEGER,
    estimated_cost_usd NUMERIC(10,2),

    -- Execution
    started_by TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    exit_code INTEGER,

    -- Artifacts and logs
    logs_uri TEXT,
    tensorboard_uri TEXT,
    checkpoint_uri TEXT,
    artifacts_uri TEXT,

    -- Notes
    notes TEXT,
    error_message TEXT,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_status CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'timeout')
    ),
    CONSTRAINT valid_metrics CHECK (
        (accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)) AND
        (precision_score IS NULL OR (precision_score >= 0 AND precision_score <= 1)) AND
        (recall_score IS NULL OR (recall_score >= 0 AND recall_score <= 1)) AND
        (f1_score IS NULL OR (f1_score >= 0 AND f1_score <= 1)) AND
        (auc_roc IS NULL OR (auc_roc >= 0 AND auc_roc <= 1))
    ),
    CONSTRAINT valid_gpu_count CHECK (gpu_count >= 0),
    CONSTRAINT valid_gpu_hours CHECK (gpu_hours >= 0)
);

CREATE INDEX idx_runs_version ON training_runs(version_id);
CREATE INDEX idx_runs_status ON training_runs(status);
CREATE INDEX idx_runs_experiment ON training_runs(experiment_name);
CREATE INDEX idx_runs_created ON training_runs(created_at DESC);
CREATE INDEX idx_runs_accuracy ON training_runs(accuracy DESC NULLS LAST);
CREATE INDEX idx_runs_active ON training_runs(status, created_at DESC)
    WHERE status IN ('queued', 'running');

COMMENT ON TABLE training_runs IS 'Individual ML model training executions';
COMMENT ON COLUMN training_runs.hyperparameters IS 'Training hyperparameters (learning rate, batch size, etc.)';

-- Deployments table
CREATE TABLE deployments (
    deployment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
    environment_id INTEGER NOT NULL REFERENCES environments(environment_id) ON DELETE RESTRICT,

    -- Deployment details
    deployment_name TEXT NOT NULL,
    deployment_type TEXT NOT NULL,
    endpoint_url TEXT,
    replicas INTEGER DEFAULT 1,
    traffic_percentage NUMERIC(5,2) DEFAULT 100.00,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending',
    health_status TEXT DEFAULT 'unknown',

    -- Performance
    avg_latency_ms NUMERIC(8,2),
    p99_latency_ms NUMERIC(8,2),
    requests_per_second NUMERIC(10,2),
    error_rate_percentage NUMERIC(5,2),

    -- Infrastructure
    compute_platform TEXT,
    serving_framework TEXT,
    instance_type TEXT,
    autoscaling_enabled BOOLEAN DEFAULT FALSE,
    min_replicas INTEGER DEFAULT 1,
    max_replicas INTEGER DEFAULT 10,

    -- Lifecycle
    deployed_by TEXT NOT NULL,
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_health_check_at TIMESTAMP WITH TIME ZONE,
    deactivated_at TIMESTAMP WITH TIME ZONE,
    deactivation_reason TEXT,

    -- Metadata
    deployment_config JSONB DEFAULT '{}'::jsonb,
    tags JSONB DEFAULT '[]'::jsonb,

    -- Constraints
    CONSTRAINT valid_deployment_type CHECK (
        deployment_type IN ('rolling', 'blue-green', 'canary', 'shadow', 'a-b-test')
    ),
    CONSTRAINT valid_deployment_status CHECK (
        status IN ('pending', 'deploying', 'active', 'degraded', 'failed', 'deactivated')
    ),
    CONSTRAINT valid_health_status CHECK (
        health_status IN ('healthy', 'degraded', 'unhealthy', 'unknown')
    ),
    CONSTRAINT valid_traffic CHECK (
        traffic_percentage >= 0 AND traffic_percentage <= 100
    ),
    CONSTRAINT valid_replicas CHECK (
        replicas > 0 AND
        min_replicas > 0 AND
        max_replicas >= min_replicas
    ),
    CONSTRAINT unique_active_deployment UNIQUE (version_id, environment_id, deployment_name)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_deployments_version ON deployments(version_id);
CREATE INDEX idx_deployments_environment ON deployments(environment_id);
CREATE INDEX idx_deployments_status ON deployments(status);
CREATE INDEX idx_deployments_health ON deployments(health_status);
CREATE INDEX idx_deployments_active ON deployments(status, deployed_at DESC)
    WHERE status = 'active';

COMMENT ON TABLE deployments IS 'ML model deployments across environments';
COMMENT ON COLUMN deployments.traffic_percentage IS 'Percentage of traffic routed to this deployment (0-100)';

-- Approvals table
CREATE TABLE approvals (
    approval_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
    deployment_id UUID REFERENCES deployments(deployment_id) ON DELETE SET NULL,

    -- Approval details
    approval_type TEXT NOT NULL,
    approval_status TEXT NOT NULL DEFAULT 'pending',
    requested_by TEXT NOT NULL,
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Review
    reviewed_by TEXT,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    review_notes TEXT,
    review_checklist JSONB DEFAULT '{}'::jsonb,

    -- Compliance
    compliance_framework TEXT,
    risk_assessment TEXT,
    mitigation_plan TEXT,

    -- Metadata
    expires_at TIMESTAMP WITH TIME ZONE,

    -- Constraints
    CONSTRAINT valid_approval_type CHECK (
        approval_type IN ('security', 'compliance', 'legal', 'technical', 'business', 'emergency')
    ),
    CONSTRAINT valid_approval_status CHECK (
        approval_status IN ('pending', 'approved', 'rejected', 'expired', 'cancelled')
    ),
    CONSTRAINT approval_review_consistency CHECK (
        (approval_status IN ('approved', 'rejected') AND reviewed_by IS NOT NULL AND reviewed_at IS NOT NULL)
        OR
        (approval_status NOT IN ('approved', 'rejected'))
    )
);

CREATE INDEX idx_approvals_version ON approvals(version_id);
CREATE INDEX idx_approvals_deployment ON approvals(deployment_id);
CREATE INDEX idx_approvals_status ON approvals(approval_status);
CREATE INDEX idx_approvals_type ON approvals(approval_type);
CREATE INDEX idx_approvals_pending ON approvals(approval_status, requested_at)
    WHERE approval_status = 'pending';

COMMENT ON TABLE approvals IS 'Compliance and governance approvals for model deployments';
COMMENT ON COLUMN approvals.compliance_framework IS 'e.g., SOC2, GDPR, HIPAA, ISO27001';

-- ============================================
-- JUNCTION TABLES (Many-to-Many)
-- ============================================

-- Model Tags junction table
CREATE TABLE model_tags (
    model_id UUID NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    tag_id UUID NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    added_by TEXT DEFAULT CURRENT_USER,

    PRIMARY KEY (model_id, tag_id)
);

CREATE INDEX idx_model_tags_model ON model_tags(model_id);
CREATE INDEX idx_model_tags_tag ON model_tags(tag_id);

COMMENT ON TABLE model_tags IS 'Many-to-many: Models can have multiple tags';

-- Dataset Tags junction table
CREATE TABLE dataset_tags (
    dataset_id UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    tag_id UUID NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    added_by TEXT DEFAULT CURRENT_USER,

    PRIMARY KEY (dataset_id, tag_id)
);

CREATE INDEX idx_dataset_tags_dataset ON dataset_tags(dataset_id);
CREATE INDEX idx_dataset_tags_tag ON dataset_tags(tag_id);

COMMENT ON TABLE dataset_tags IS 'Many-to-many: Datasets can have multiple tags';

-- Run Datasets junction table
CREATE TABLE run_datasets (
    run_id UUID NOT NULL REFERENCES training_runs(run_id) ON DELETE CASCADE,
    dataset_id UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE RESTRICT,
    dataset_role TEXT NOT NULL,
    dataset_version TEXT,
    split_name TEXT,
    row_count BIGINT,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (run_id, dataset_id, dataset_role),

    CONSTRAINT valid_dataset_role CHECK (
        dataset_role IN ('train', 'validation', 'test', 'holdout')
    )
);

CREATE INDEX idx_run_datasets_run ON run_datasets(run_id);
CREATE INDEX idx_run_datasets_dataset ON run_datasets(dataset_id);

COMMENT ON TABLE run_datasets IS 'Many-to-many: Training runs can use multiple datasets';
COMMENT ON COLUMN run_datasets.dataset_role IS 'train, validation, test, or holdout';

-- ============================================
-- VIEWS (Denormalized for Queries)
-- ============================================

-- View: Complete model information with latest version
CREATE OR REPLACE VIEW v_models_latest AS
SELECT
    m.model_id,
    m.model_name,
    m.display_name,
    m.description,
    m.team_id,
    t.team_name,
    m.risk_level,
    m.compliance_required,
    mv.version_id AS latest_version_id,
    mv.semver AS latest_version,
    mv.status AS latest_version_status,
    mv.framework,
    mv.registered_at AS latest_version_registered_at,
    m.created_at,
    m.is_active
FROM models m
LEFT JOIN teams t ON m.team_id = t.team_id
LEFT JOIN LATERAL (
    SELECT *
    FROM model_versions
    WHERE model_id = m.model_id
    ORDER BY registered_at DESC
    LIMIT 1
) mv ON TRUE
WHERE m.is_active = TRUE;

COMMENT ON VIEW v_models_latest IS 'Models with their latest version information';

-- View: Active deployments with model and environment info
CREATE OR REPLACE VIEW v_active_deployments AS
SELECT
    d.deployment_id,
    d.deployment_name,
    d.status,
    d.health_status,
    e.environment_name,
    e.environment_type,
    m.model_name,
    mv.semver AS version,
    d.endpoint_url,
    d.replicas,
    d.traffic_percentage,
    d.deployed_at,
    d.deployed_by
FROM deployments d
JOIN environments e ON d.environment_id = e.environment_id
JOIN model_versions mv ON d.version_id = mv.version_id
JOIN models m ON mv.model_id = m.model_id
WHERE d.status = 'active';

COMMENT ON VIEW v_active_deployments IS 'Currently active model deployments';

-- ============================================
-- TRIGGERS (Auto-update timestamps)
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to models table
CREATE TRIGGER update_models_updated_at
    BEFORE UPDATE ON models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- COMPLETION MESSAGE
-- ============================================

DO $$
BEGIN
    RAISE NOTICE '✓ ML Model Registry schema created successfully';
    RAISE NOTICE '✓ Tables: 12 (9 core + 3 junction)';
    RAISE NOTICE '✓ Views: 2';
    RAISE NOTICE '✓ Triggers: 1';
    RAISE NOTICE '✓ Normalization: 3NF';
END $$;
```

### Step 3.3: Load Schema

```bash
# Execute schema
docker exec -i pg-ml-registry psql -U mlops -d ml_registry < sql/10_model_registry_schema.sql
```

**Expected Output**:
```
DROP TABLE
... (multiple DROP statements)
CREATE TABLE
CREATE INDEX
COMMENT
... (multiple CREATE statements)
INSERT 0 8
CREATE VIEW
NOTICE:  ✓ ML Model Registry schema created successfully
NOTICE:  ✓ Tables: 12 (9 core + 3 junction)
NOTICE:  ✓ Views: 2
NOTICE:  ✓ Triggers: 1
NOTICE:  ✓ Normalization: 3NF
```

### Step 3.4: Verify Schema

```sql
-- Connect to database
-- docker exec -it pg-ml-registry psql -U mlops -d ml_registry

-- List all tables
\dt

-- Expected output:
--             List of relations
--  Schema |      Name       | Type  | Owner
-- --------+-----------------+-------+-------
--  public | approvals       | table | mlops
--  public | dataset_tags    | table | mlops
--  public | datasets        | table | mlops
--  public | deployments     | table | mlops
--  public | environments    | table | mlops
--  public | model_tags      | table | mlops
--  public | model_versions  | table | mlops
--  public | models          | table | mlops
--  public | run_datasets    | table | mlops
--  public | tags            | table | mlops
--  public | teams           | table | mlops
--  public | training_runs   | table | mlops

-- Check foreign keys on model_versions
SELECT
    conname AS constraint_name,
    conrelid::regclass AS table_name,
    pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = 'model_versions'::regclass
  AND contype = 'f'
ORDER BY conname;

-- Check indexes
SELECT tablename, indexname
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
```

✅ **Checkpoint**: Complete normalized schema created with 12 tables, 2 views, and 1 trigger.

---

## Part 4: Seed Data

### Step 4.1: Create Seed Data Script

Create `sql/11_seed_model_registry.sql`:

```sql
-- ============================================
-- ML Model Registry - Seed Data
-- ============================================
-- Purpose: Realistic seed data for testing
-- Date: 2025-11-01
-- ============================================

-- ============================================
-- TEAMS
-- ============================================

INSERT INTO teams (team_name, team_email, department, cost_center, manager_name, slack_channel)
VALUES
    ('ml-platform', 'ml-platform@company.com', 'Engineering', 'CC-ML-001', 'Sarah Chen', '#ml-platform'),
    ('nlp-research', 'nlp-research@company.com', 'Research', 'CC-ML-002', 'David Kim', '#nlp-research'),
    ('computer-vision', 'cv-team@company.com', 'Engineering', 'CC-ML-003', 'Maria Garcia', '#computer-vision'),
    ('recommendations', 'recsys@company.com', 'Product', 'CC-ML-004', 'James Wilson', '#recommendations'),
    ('forecasting', 'forecasting@company.com', 'Finance', 'CC-FIN-001', 'Emily Brown', '#forecasting');

-- ============================================
-- TAGS
-- ============================================

INSERT INTO tags (tag_name, tag_category, description)
VALUES
    ('nlp', 'domain', 'Natural Language Processing'),
    ('computer-vision', 'domain', 'Computer Vision and Image Processing'),
    ('time-series', 'domain', 'Time Series Forecasting'),
    ('recommendation', 'domain', 'Recommendation Systems'),

    ('pytorch', 'framework', 'PyTorch Framework'),
    ('tensorflow', 'framework', 'TensorFlow Framework'),
    ('sklearn', 'framework', 'Scikit-learn'),

    ('production', 'status', 'Production-ready'),
    ('experimental', 'status', 'Experimental/Research'),
    ('deprecated', 'status', 'Deprecated - do not use'),

    ('classification', 'use-case', 'Classification task'),
    ('regression', 'use-case', 'Regression task'),
    ('detection', 'use-case', 'Object Detection'),
    ('generation', 'use-case', 'Generative model'),

    ('high-priority', 'priority', 'Critical business impact'),
    ('low-latency', 'priority', 'Real-time inference required');

-- ============================================
-- DATASETS
-- ============================================

INSERT INTO datasets (
    dataset_name, display_name, description, purpose,
    storage_location, storage_format, storage_size_gb,
    team_id, data_steward,
    freshness_sla_hours, last_updated_at, row_count, column_count,
    contains_pii, data_classification, retention_days
)
VALUES
(
    'customer-reviews-v1',
    'Customer Product Reviews v1',
    'E-commerce product reviews with ratings',
    'Sentiment analysis and classification',
    's3://ml-datasets/customer-reviews/v1/',
    'parquet',
    15.8,
    (SELECT team_id FROM teams WHERE team_name = 'nlp-research'),
    'alice@company.com',
    24,
    NOW() - INTERVAL '2 hours',
    1250000,
    12,
    TRUE,
    'confidential',
    730
),
(
    'imagenet-subset-2025',
    'ImageNet Subset 2025',
    '500k images from ImageNet for transfer learning',
    'Image classification pre-training',
    's3://ml-datasets/imagenet-subset/',
    'tfrecord',
    125.5,
    (SELECT team_id FROM teams WHERE team_name = 'computer-vision'),
    'bob@company.com',
    NULL,
    NOW() - INTERVAL '30 days',
    500000,
    5,
    FALSE,
    'internal',
    1825
),
(
    'sales-forecast-daily',
    'Daily Sales Forecast Data',
    'Historical daily sales data across all regions',
    'Revenue forecasting',
    'gs://ml-data/sales-forecast/',
    'parquet',
    2.3,
    (SELECT team_id FROM teams WHERE team_name = 'forecasting'),
    'carol@company.com',
    1,
    NOW() - INTERVAL '30 minutes',
    850000,
    25,
    FALSE,
    'confidential',
    2555
),
(
    'user-interactions-log',
    'User Interaction Logs',
    'Click-through and interaction logs',
    'Recommendation system training',
    's3://ml-datasets/user-interactions/',
    'parquet',
    450.2,
    (SELECT team_id FROM teams WHERE team_name = 'recommendations'),
    'dave@company.com',
    6,
    NOW() - INTERVAL '4 hours',
    125000000,
    18,
    TRUE,
    'restricted',
    90
);

-- ============================================
-- MODELS
-- ============================================

INSERT INTO models (
    model_name, display_name, description, objective, use_case,
    team_id, primary_contact, secondary_contact,
    business_unit, risk_level, compliance_required,
    github_repo, documentation_url
)
VALUES
(
    'sentiment-classifier-v2',
    'Product Review Sentiment Classifier v2',
    'BERT-based sentiment classifier for product reviews',
    'Classify review sentiment as positive/negative/neutral',
    'Customer feedback analysis',
    (SELECT team_id FROM teams WHERE team_name = 'nlp-research'),
    'alice@company.com',
    'alice-backup@company.com',
    'E-commerce',
    'medium',
    FALSE,
    'https://github.com/company/sentiment-classifier',
    'https://docs.company.com/models/sentiment-v2'
),
(
    'object-detector-yolo',
    'Product Image Object Detector',
    'YOLOv8 for detecting products in images',
    'Identify and localize products in user-uploaded images',
    'Visual search and catalog',
    (SELECT team_id FROM teams WHERE team_name = 'computer-vision'),
    'bob@company.com',
    NULL,
    'E-commerce',
    'high',
    TRUE,
    'https://github.com/company/object-detector',
    'https://docs.company.com/models/object-detector'
),
(
    'revenue-forecaster',
    'Revenue Forecast Model',
    'Time series model for revenue prediction',
    'Forecast daily revenue for next 30 days',
    'Financial planning',
    (SELECT team_id FROM teams WHERE team_name = 'forecasting'),
    'carol@company.com',
    'emily-backup@company.com',
    'Finance',
    'critical',
    TRUE,
    'https://github.com/company/revenue-forecaster',
    'https://docs.company.com/models/revenue-forecast'
),
(
    'product-recommender',
    'Personalized Product Recommender',
    'Collaborative filtering for product recommendations',
    'Recommend top 10 products for each user',
    'E-commerce personalization',
    (SELECT team_id FROM teams WHERE team_name = 'recommendations'),
    'dave@company.com',
    NULL,
    'E-commerce',
    'high',
    FALSE,
    'https://github.com/company/recommender',
    'https://docs.company.com/models/recommender'
);

-- ============================================
-- MODEL VERSIONS
-- ============================================

INSERT INTO model_versions (
    model_id, semver, version_alias, description,
    git_commit, git_branch, artifact_uri, artifact_size_mb, checksum_sha256,
    framework, framework_version, python_version, cuda_version,
    model_format, status, registered_by, benchmark_accuracy
)
VALUES
-- Sentiment Classifier v2 versions
(
    (SELECT model_id FROM models WHERE model_name = 'sentiment-classifier-v2'),
    '2.0.0',
    'stable',
    'Major update with improved accuracy',
    'abc123def456',
    'main',
    's3://ml-models/sentiment-classifier-v2/2.0.0/model.onnx',
    420.5,
    'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2',
    'pytorch',
    '2.1.0',
    '3.11',
    '12.1',
    'onnx',
    'deployed',
    'alice@company.com',
    0.9234
),
(
    (SELECT model_id FROM models WHERE model_name = 'sentiment-classifier-v2'),
    '2.1.0',
    'latest',
    'Fine-tuned on recent data',
    'def789ghi012',
    'main',
    's3://ml-models/sentiment-classifier-v2/2.1.0/model.onnx',
    425.8,
    'b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3',
    'pytorch',
    '2.1.0',
    '3.11',
    '12.1',
    'onnx',
    'validated',
    'alice@company.com',
    0.9312
),
-- Object Detector versions
(
    (SELECT model_id FROM models WHERE model_name = 'object-detector-yolo'),
    '1.0.0',
    'stable',
    'Initial production release',
    'ghi345jkl678',
    'release/v1',
    's3://ml-models/object-detector/1.0.0/model.pt',
    88.2,
    'c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h4',
    'pytorch',
    '2.0.1',
    '3.10',
    '11.8',
    'torchscript',
    'deployed',
    'bob@company.com',
    0.8523
),
-- Revenue Forecaster versions
(
    (SELECT model_id FROM models WHERE model_name = 'revenue-forecaster'),
    '3.2.1',
    'latest',
    'Q1 2025 model update',
    'mno901pqr234',
    'main',
    's3://ml-models/revenue-forecaster/3.2.1/model.pkl',
    2.5,
    'd4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h4i5',
    'sklearn',
    '1.3.2',
    '3.11',
    NULL,
    'pickle',
    'deployed',
    'carol@company.com',
    NULL
),
-- Product Recommender versions
(
    (SELECT model_id FROM models WHERE model_name = 'product-recommender'),
    '1.5.0',
    'stable',
    'Improved collaborative filtering',
    'stu567vwx890',
    'main',
    's3://ml-models/product-recommender/1.5.0/model.joblib',
    125.3,
    'e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h4i5j6',
    'sklearn',
    '1.3.2',
    '3.11',
    NULL,
    'joblib',
    'deployed',
    'dave@company.com',
    NULL
);

-- ============================================
-- TRAINING RUNS
-- ============================================

INSERT INTO training_runs (
    version_id, run_name, experiment_name, status,
    accuracy, precision_score, recall_score, f1_score, loss,
    hyperparameters, compute_target, gpu_count, gpu_type, gpu_hours,
    started_by, started_at, completed_at
)
VALUES
(
    (SELECT version_id FROM model_versions WHERE semver = '2.0.0' LIMIT 1),
    'sentiment-baseline-001',
    'bert-sentiment-q1-2025',
    'succeeded',
    0.9234,
    0.9156,
    0.9312,
    0.9233,
    0.2876,
    '{"learning_rate": 0.00005, "batch_size": 32, "epochs": 10}'::jsonb,
    'k8s-gpu-a100',
    4,
    'nvidia-a100',
    24.5,
    'alice@company.com',
    NOW() - INTERVAL '10 days',
    NOW() - INTERVAL '9 days 18 hours'
),
(
    (SELECT version_id FROM model_versions WHERE semver = '2.1.0' LIMIT 1),
    'sentiment-finetuned-002',
    'bert-sentiment-q1-2025',
    'succeeded',
    0.9312,
    0.9245,
    0.9389,
    0.9316,
    0.2543,
    '{"learning_rate": 0.00003, "batch_size": 32, "epochs": 5, "warmup_steps": 500}'::jsonb,
    'k8s-gpu-a100',
    4,
    'nvidia-a100',
    12.3,
    'alice@company.com',
    NOW() - INTERVAL '3 days',
    NOW() - INTERVAL '2 days 20 hours'
),
(
    (SELECT version_id FROM model_versions WHERE semver = '1.0.0' AND model_id = (SELECT model_id FROM models WHERE model_name = 'object-detector-yolo') LIMIT 1),
    'yolo-coco-pretrain-001',
    'object-detection-v1',
    'succeeded',
    0.8523,
    0.8345,
    0.8712,
    0.8525,
    1.2345,
    '{"img_size": 640, "batch_size": 16, "epochs": 100, "iou_threshold": 0.5}'::jsonb,
    'gcp-vertex-ai-v100',
    8,
    'nvidia-v100',
    64.2,
    'bob@company.com',
    NOW() - INTERVAL '15 days',
    NOW() - INTERVAL '13 days 8 hours'
);

-- ============================================
-- DEPLOYMENTS
-- ============================================

INSERT INTO deployments (
    version_id, environment_id, deployment_name, deployment_type,
    endpoint_url, replicas, traffic_percentage, status, health_status,
    compute_platform, serving_framework, instance_type,
    deployed_by
)
VALUES
(
    (SELECT version_id FROM model_versions WHERE semver = '2.0.0' LIMIT 1),
    (SELECT environment_id FROM environments WHERE environment_name = 'prod-us'),
    'sentiment-classifier-prod-us',
    'rolling',
    'https://api.company.com/models/sentiment/v2',
    5,
    100.00,
    'active',
    'healthy',
    'kubernetes',
    'torchserve',
    'n1-standard-4',
    'alice@company.com'
),
(
    (SELECT version_id FROM model_versions WHERE semver = '2.1.0' LIMIT 1),
    (SELECT environment_id FROM environments WHERE environment_name = 'staging'),
    'sentiment-classifier-staging',
    'rolling',
    'https://staging-api.company.com/models/sentiment/v2',
    2,
    100.00,
    'active',
    'healthy',
    'kubernetes',
    'torchserve',
    'n1-standard-2',
    'alice@company.com'
),
(
    (SELECT version_id FROM model_versions WHERE semver = '1.0.0' AND model_id = (SELECT model_id FROM models WHERE model_name = 'object-detector-yolo') LIMIT 1),
    (SELECT environment_id FROM environments WHERE environment_name = 'prod-us'),
    'object-detector-prod',
    'blue-green',
    'https://api.company.com/models/detector/v1',
    10,
    100.00,
    'active',
    'healthy',
    'kubernetes',
    'triton',
    'n1-highmem-8-gpu',
    'bob@company.com'
);

-- ============================================
-- APPROVALS
-- ============================================

INSERT INTO approvals (
    version_id, deployment_id, approval_type, approval_status,
    requested_by, reviewed_by, reviewed_at, review_notes,
    compliance_framework, risk_assessment
)
VALUES
(
    (SELECT version_id FROM model_versions WHERE semver = '1.0.0' AND model_id = (SELECT model_id FROM models WHERE model_name = 'object-detector-yolo') LIMIT 1),
    (SELECT deployment_id FROM deployments WHERE deployment_name = 'object-detector-prod'),
    'compliance',
    'approved',
    'bob@company.com',
    'compliance-team@company.com',
    NOW() - INTERVAL '14 days',
    'Model passed all compliance checks. GPU usage within budget. Performance benchmarks met.',
    'SOC2',
    'Medium risk. Standard ML model deployment.'
),
(
    (SELECT version_id FROM model_versions WHERE semver = '3.2.1' LIMIT 1),
    NULL,
    'technical',
    'approved',
    'carol@company.com',
    'tech-lead@company.com',
    NOW() - INTERVAL '5 days',
    'Revenue forecasting model meets accuracy requirements. Approved for production.',
    NULL,
    'Critical - Financial impact. Requires monitoring.'
);

-- ============================================
-- JUNCTION TABLES
-- ============================================

-- Model Tags
INSERT INTO model_tags (model_id, tag_id)
SELECT
    m.model_id,
    t.tag_id
FROM models m
CROSS JOIN tags t
WHERE
    (m.model_name = 'sentiment-classifier-v2' AND t.tag_name IN ('nlp', 'pytorch', 'production', 'classification')) OR
    (m.model_name = 'object-detector-yolo' AND t.tag_name IN ('computer-vision', 'pytorch', 'production', 'detection', 'high-priority')) OR
    (m.model_name = 'revenue-forecaster' AND t.tag_name IN ('time-series', 'sklearn', 'production', 'regression', 'high-priority')) OR
    (m.model_name = 'product-recommender' AND t.tag_name IN ('recommendation', 'sklearn', 'production'));

-- Dataset Tags
INSERT INTO dataset_tags (dataset_id, tag_id)
SELECT
    d.dataset_id,
    t.tag_id
FROM datasets d
CROSS JOIN tags t
WHERE
    (d.dataset_name = 'customer-reviews-v1' AND t.tag_name IN ('nlp', 'classification')) OR
    (d.dataset_name = 'imagenet-subset-2025' AND t.tag_name IN ('computer-vision', 'classification')) OR
    (d.dataset_name = 'sales-forecast-daily' AND t.tag_name IN ('time-series', 'regression')) OR
    (d.dataset_name = 'user-interactions-log' AND t.tag_name IN ('recommendation'));

-- Run Datasets
INSERT INTO run_datasets (run_id, dataset_id, dataset_role, dataset_version, split_name, row_count)
VALUES
(
    (SELECT run_id FROM training_runs WHERE run_name = 'sentiment-baseline-001'),
    (SELECT dataset_id FROM datasets WHERE dataset_name = 'customer-reviews-v1'),
    'train',
    'v1',
    'train',
    1000000
),
(
    (SELECT run_id FROM training_runs WHERE run_name = 'sentiment-baseline-001'),
    (SELECT dataset_id FROM datasets WHERE dataset_name = 'customer-reviews-v1'),
    'validation',
    'v1',
    'val',
    125000
),
(
    (SELECT run_id FROM training_runs WHERE run_name = 'sentiment-baseline-001'),
    (SELECT dataset_id FROM datasets WHERE dataset_name = 'customer-reviews-v1'),
    'test',
    'v1',
    'test',
    125000
),
(
    (SELECT run_id FROM training_runs WHERE run_name = 'yolo-coco-pretrain-001'),
    (SELECT dataset_id FROM datasets WHERE dataset_name = 'imagenet-subset-2025'),
    'train',
    '2025',
    'train',
    450000
),
(
    (SELECT run_id FROM training_runs WHERE run_name = 'yolo-coco-pretrain-001'),
    (SELECT dataset_id FROM datasets WHERE dataset_name = 'imagenet-subset-2025'),
    'validation',
    '2025',
    'val',
    50000
);

-- ============================================
-- VERIFICATION
-- ============================================

-- Summary
SELECT 'Teams' AS table_name, COUNT(*) AS count FROM teams
UNION ALL
SELECT 'Environments', COUNT(*) FROM environments
UNION ALL
SELECT 'Tags', COUNT(*) FROM tags
UNION ALL
SELECT 'Datasets', COUNT(*) FROM datasets
UNION ALL
SELECT 'Models', COUNT(*) FROM models
UNION ALL
SELECT 'Model Versions', COUNT(*) FROM model_versions
UNION ALL
SELECT 'Training Runs', COUNT(*) FROM training_runs
UNION ALL
SELECT 'Deployments', COUNT(*) FROM deployments
UNION ALL
SELECT 'Approvals', COUNT(*) FROM approvals
UNION ALL
SELECT 'Model Tags', COUNT(*) FROM model_tags
UNION ALL
SELECT 'Dataset Tags', COUNT(*) FROM dataset_tags
UNION ALL
SELECT 'Run Datasets', COUNT(*) FROM run_datasets;
```

### Step 4.2: Load Seed Data

```bash
# Execute seed data
docker exec -i pg-ml-registry psql -U mlops -d ml_registry < sql/11_seed_model_registry.sql
```

**Expected Output**:
```
INSERT 0 5
INSERT 0 16
INSERT 0 4
INSERT 0 4
INSERT 0 5
INSERT 0 3
INSERT 0 3
INSERT 0 2
INSERT 0 15
INSERT 0 4
INSERT 0 5
     table_name     | count
--------------------+-------
 Teams              |     5
 Environments       |     8
 Tags               |    16
 Datasets           |     4
 Models             |     4
 Model Versions     |     5
 Training Runs      |     3
 Deployments        |     3
 Approvals          |     2
 Model Tags         |    15
 Dataset Tags       |     4
 Run Datasets       |     5
(12 rows)
```

✅ **Checkpoint**: All seed data loaded successfully.

---

## Part 5: Querying Related Tables

### Step 5.1: Simple Joins

Create `sql/12_query_examples.sql`:

```sql
-- ============================================
-- ML Model Registry - Query Examples
-- ============================================

-- ====================
-- SIMPLE JOINS
-- ====================

-- Query 1: Models with team information (INNER JOIN)
SELECT
    m.model_name,
    m.display_name,
    m.risk_level,
    t.team_name,
    t.team_email,
    t.department
FROM models m
INNER JOIN teams t ON m.team_id = t.team_id
WHERE m.is_active = TRUE
ORDER BY m.model_name;

-- Query 2: Model versions with model information (INNER JOIN)
SELECT
    m.model_name,
    mv.semver,
    mv.version_alias,
    mv.framework,
    mv.status,
    mv.registered_at
FROM model_versions mv
INNER JOIN models m ON mv.model_id = m.model_id
ORDER BY m.model_name, mv.registered_at DESC;

-- Query 3: Deployments with environment and model info (Multiple JOINs)
SELECT
    m.model_name,
    mv.semver AS version,
    e.environment_name,
    e.environment_type,
    d.deployment_name,
    d.status,
    d.health_status,
    d.endpoint_url,
    d.replicas,
    d.deployed_at
FROM deployments d
INNER JOIN model_versions mv ON d.version_id = mv.version_id
INNER JOIN models m ON mv.model_id = m.model_id
INNER JOIN environments e ON d.environment_id = e.environment_id
WHERE d.status = 'active'
ORDER BY e.priority DESC, m.model_name;

-- ====================
-- LEFT JOINS (Include models without versions)
-- ====================

-- Query 4: All models with version count (LEFT JOIN)
SELECT
    m.model_name,
    m.display_name,
    t.team_name,
    COUNT(mv.version_id) AS version_count,
    MAX(mv.registered_at) AS latest_version_date
FROM models m
LEFT JOIN teams t ON m.team_id = t.team_id
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
WHERE m.is_active = TRUE
GROUP BY m.model_id, m.model_name, m.display_name, t.team_name
ORDER BY version_count DESC;

-- Query 5: Models and their production deployments (LEFT JOIN)
SELECT
    m.model_name,
    mv.semver,
    e.environment_name,
    d.status,
    d.deployed_at
FROM models m
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
LEFT JOIN deployments d ON mv.version_id = d.version_id
LEFT JOIN environments e ON d.environment_id = e.environment_id
    AND e.environment_type = 'production'
WHERE m.is_active = TRUE
ORDER BY m.model_name, mv.registered_at DESC;

-- ====================
-- MANY-TO-MANY QUERIES
-- ====================

-- Query 6: Models with their tags
SELECT
    m.model_name,
    ARRAY_AGG(t.tag_name ORDER BY t.tag_name) AS tags
FROM models m
LEFT JOIN model_tags mt ON m.model_id = mt.model_id
LEFT JOIN tags t ON mt.tag_id = t.tag_id
WHERE m.is_active = TRUE
GROUP BY m.model_id, m.model_name
ORDER BY m.model_name;

-- Query 7: Find models by tag
SELECT
    m.model_name,
    m.display_name,
    m.risk_level,
    ARRAY_AGG(DISTINCT t.tag_name ORDER BY t.tag_name) AS all_tags
FROM models m
INNER JOIN model_tags mt ON m.model_id = mt.model_id
INNER JOIN tags t ON mt.tag_id = t.tag_id
WHERE t.tag_name IN ('nlp', 'production')
GROUP BY m.model_id, m.model_name, m.display_name, m.risk_level
HAVING ARRAY_AGG(DISTINCT t.tag_name) @> ARRAY['nlp', 'production']::TEXT[]
ORDER BY m.model_name;

-- Query 8: Training runs with datasets
SELECT
    m.model_name,
    tr.run_name,
    tr.status,
    ARRAY_AGG(
        d.dataset_name || ' (' || rd.dataset_role || ')'
        ORDER BY rd.dataset_role
    ) AS datasets_used
FROM training_runs tr
INNER JOIN model_versions mv ON tr.version_id = mv.version_id
INNER JOIN models m ON mv.model_id = m.model_id
LEFT JOIN run_datasets rd ON tr.run_id = rd.run_id
LEFT JOIN datasets d ON rd.dataset_id = d.dataset_id
GROUP BY m.model_name, tr.run_id, tr.run_name, tr.status
ORDER BY tr.started_at DESC;

-- ====================
-- COMPLEX AGGREGATIONS
-- ====================

-- Query 9: Model deployment summary
SELECT
    m.model_name,
    COUNT(DISTINCT mv.version_id) AS total_versions,
    COUNT(DISTINCT CASE WHEN d.status = 'active' THEN d.deployment_id END) AS active_deployments,
    COUNT(DISTINCT tr.run_id) AS training_runs,
    MAX(mv.registered_at) AS latest_version_date,
    MAX(d.deployed_at) AS latest_deployment_date
FROM models m
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
LEFT JOIN deployments d ON mv.version_id = d.version_id
LEFT JOIN training_runs tr ON mv.version_id = tr.version_id
WHERE m.is_active = TRUE
GROUP BY m.model_id, m.model_name
ORDER BY active_deployments DESC, m.model_name;

-- Query 10: Environment deployment status
SELECT
    e.environment_name,
    e.environment_type,
    COUNT(DISTINCT d.deployment_id) AS total_deployments,
    COUNT(DISTINCT CASE WHEN d.status = 'active' THEN d.deployment_id END) AS active_deployments,
    COUNT(DISTINCT m.model_id) AS unique_models,
    SUM(d.replicas) FILTER (WHERE d.status = 'active') AS total_replicas
FROM environments e
LEFT JOIN deployments d ON e.environment_id = d.environment_id
LEFT JOIN model_versions mv ON d.version_id = mv.version_id
LEFT JOIN models m ON mv.model_id = m.model_id
GROUP BY e.environment_id, e.environment_name, e.environment_type
ORDER BY e.priority DESC;

-- ====================
-- SUBQUERIES
-- ====================

-- Query 11: Models with latest version details (Correlated subquery)
SELECT
    m.model_name,
    m.display_name,
    (
        SELECT mv.semver
        FROM model_versions mv
        WHERE mv.model_id = m.model_id
        ORDER BY mv.registered_at DESC
        LIMIT 1
    ) AS latest_version,
    (
        SELECT mv.status
        FROM model_versions mv
        WHERE mv.model_id = m.model_id
        ORDER BY mv.registered_at DESC
        LIMIT 1
    ) AS latest_status,
    (
        SELECT COUNT(*)
        FROM deployments d
        JOIN model_versions mv ON d.version_id = mv.version_id
        WHERE mv.model_id = m.model_id
          AND d.status = 'active'
    ) AS active_deployment_count
FROM models m
WHERE m.is_active = TRUE
ORDER BY m.model_name;

-- Query 12: Find models never deployed to production
SELECT
    m.model_name,
    m.display_name,
    m.created_at,
    COUNT(mv.version_id) AS version_count
FROM models m
LEFT JOIN model_versions mv ON m.model_id = mv.model_id
WHERE m.is_active = TRUE
  AND m.model_id NOT IN (
      SELECT DISTINCT mv2.model_id
      FROM model_versions mv2
      JOIN deployments d ON mv2.version_id = d.version_id
      JOIN environments e ON d.environment_id = e.environment_id
      WHERE e.environment_type = 'production'
  )
GROUP BY m.model_id, m.model_name, m.display_name, m.created_at
ORDER BY m.created_at DESC;

-- ====================
-- CTEs FOR READABILITY
-- ====================

-- Query 13: Production deployment dashboard (CTE)
WITH production_envs AS (
    SELECT environment_id, environment_name
    FROM environments
    WHERE environment_type = 'production'
),
active_prod_deployments AS (
    SELECT
        d.version_id,
        d.deployment_id,
        d.environment_id,
        d.health_status,
        d.replicas
    FROM deployments d
    WHERE d.status = 'active'
      AND d.environment_id IN (SELECT environment_id FROM production_envs)
)
SELECT
    m.model_name,
    mv.semver,
    pe.environment_name,
    apd.health_status,
    apd.replicas,
    t.team_name AS owning_team
FROM active_prod_deployments apd
JOIN production_envs pe ON apd.environment_id = pe.environment_id
JOIN model_versions mv ON apd.version_id = mv.version_id
JOIN models m ON mv.model_id = m.model_id
LEFT JOIN teams t ON m.team_id = t.team_id
ORDER BY pe.environment_name, m.model_name;
```

### Step 5.2: Execute Query Examples

```bash
# Run queries
docker exec -i pg-ml-registry psql -U mlops -d ml_registry < sql/12_query_examples.sql > output/query_results.txt
```

✅ **Checkpoint**: Complex queries work across related tables.

---

## Part 6: Foreign Key Constraints

### Step 6.1: Test CASCADE Behavior

Create `sql/13_test_foreign_keys.sql`:

```sql
-- ============================================
-- Foreign Key Cascade Testing
-- ============================================

-- ====================
-- TEST 1: ON DELETE CASCADE
-- ====================

-- Scenario: Delete a model should cascade delete all versions
BEGIN;

-- Create test model
INSERT INTO models (model_name, display_name, description, primary_contact, risk_level)
VALUES (
    'test-cascade-model',
    'Test Cascade Model',
    'Model for testing CASCADE behavior',
    'test@company.com',
    'low'
)
RETURNING model_id;

-- Store model_id (you'll see it in output)
-- For demo, we'll use a subquery

-- Create test version
INSERT INTO model_versions (
    model_id,
    semver,
    artifact_uri,
    framework,
    registered_by
)
VALUES (
    (SELECT model_id FROM models WHERE model_name = 'test-cascade-model'),
    '1.0.0',
    's3://test/model.pt',
    'pytorch',
    'test@company.com'
);

-- Verify version exists
SELECT COUNT(*) AS version_count
FROM model_versions
WHERE model_id = (SELECT model_id FROM models WHERE model_name = 'test-cascade-model');
-- Should show: 1

-- Delete the model (should CASCADE delete the version)
DELETE FROM models WHERE model_name = 'test-cascade-model';

-- Verify version was also deleted
SELECT COUNT(*) AS version_count_after_delete
FROM model_versions
WHERE model_id IN (
    SELECT model_id FROM models WHERE model_name = 'test-cascade-model'
);
-- Should show: 0 (version was cascaded deleted)

ROLLBACK;

-- ====================
-- TEST 2: ON DELETE SET NULL
-- ====================

-- Scenario: Delete a team should SET NULL on model.team_id
BEGIN;

-- Create test team
INSERT INTO teams (team_name, team_email, department)
VALUES (
    'test-team',
    'test-team@company.com',
    'Test'
)
RETURNING team_id;

-- Create model owned by test team
INSERT INTO models (
    model_name,
    display_name,
    description,
    team_id,
    primary_contact,
    risk_level
)
VALUES (
    'test-set-null-model',
    'Test SET NULL Model',
    'Model for testing SET NULL behavior',
    (SELECT team_id FROM teams WHERE team_name = 'test-team'),
    'test@company.com',
    'low'
);

-- Verify model has team_id
SELECT model_name, team_id
FROM models
WHERE model_name = 'test-set-null-model';
-- Should show team_id (not NULL)

-- Delete the team (should SET NULL on model.team_id)
DELETE FROM teams WHERE team_name = 'test-team';

-- Verify model.team_id is now NULL
SELECT model_name, team_id
FROM models
WHERE model_name = 'test-set-null-model';
-- Should show team_id = NULL

ROLLBACK;

-- ====================
-- TEST 3: ON DELETE RESTRICT
-- ====================

-- Scenario: Cannot delete environment if deployments exist
BEGIN;

-- Try to delete 'prod-us' environment (has deployments)
DELETE FROM environments WHERE environment_name = 'prod-us';
-- Should FAIL with: ERROR: update or delete on table "environments" violates foreign key constraint

ROLLBACK;

-- Verify: Must delete deployments first, then environment
BEGIN;

-- Count deployments in prod-us
SELECT COUNT(*) AS deployment_count
FROM deployments
WHERE environment_id = (SELECT environment_id FROM environments WHERE environment_name = 'prod-us');

-- To actually delete, must remove deployments first:
-- DELETE FROM deployments WHERE environment_id = (SELECT environment_id FROM environments WHERE environment_name = 'prod-us');
-- DELETE FROM environments WHERE environment_name = 'prod-us';

ROLLBACK;

-- ====================
-- TEST 4: Referential Integrity
-- ====================

-- Scenario: Cannot insert deployment with invalid version_id
BEGIN;

-- Try to insert deployment with non-existent version_id
INSERT INTO deployments (
    version_id,
    environment_id,
    deployment_name,
    deployment_type,
    deployed_by
)
VALUES (
    '00000000-0000-0000-0000-000000000000'::UUID,  -- Invalid UUID
    (SELECT environment_id FROM environments WHERE environment_name = 'dev'),
    'test-invalid-deployment',
    'rolling',
    'test@company.com'
);
-- Should FAIL with: ERROR: insert or update on table "deployments" violates foreign key constraint

ROLLBACK;
```

### Step 6.2: Run Foreign Key Tests

```bash
# Execute tests
docker exec -i pg-ml-registry psql -U mlops -d ml_registry < sql/13_test_foreign_keys.sql
```

**Expected Behavior**:
- Test 1: CASCADE works (version deleted with model)
- Test 2: SET NULL works (model.team_id becomes NULL)
- Test 3: RESTRICT prevents deletion (environment can't be deleted with deployments)
- Test 4: Referential integrity enforced (can't insert invalid foreign key)

✅ **Checkpoint**: Foreign key constraints working correctly.

---

## Part 7: Many-to-Many Relationships

### Step 7.1: Understand Junction Tables

Junction tables connect two entities in a many-to-many relationship:

```
models ←→ model_tags ←→ tags
  (M)         (junction)      (N)

One model can have many tags
One tag can belong to many models
```

### Step 7.2: Query Many-to-Many

```sql
-- Connect to database
-- docker exec -it pg-ml-registry psql -U mlops -d ml_registry

-- Query 1: List all models with their tags
SELECT
    m.model_name,
    STRING_AGG(t.tag_name, ', ' ORDER BY t.tag_name) AS tags
FROM models m
LEFT JOIN model_tags mt ON m.model_id = mt.model_id
LEFT JOIN tags t ON mt.tag_id = t.tag_id
GROUP BY m.model_id, m.model_name
ORDER BY m.model_name;

-- Query 2: Find all models with 'nlp' tag
SELECT
    m.model_name,
    m.display_name
FROM models m
INNER JOIN model_tags mt ON m.model_id = mt.model_id
INNER JOIN tags t ON mt.tag_id = t.tag_id
WHERE t.tag_name = 'nlp';

-- Query 3: Find models with BOTH 'nlp' AND 'production' tags
SELECT
    m.model_name,
    ARRAY_AGG(t.tag_name ORDER BY t.tag_name) AS tags
FROM models m
INNER JOIN model_tags mt ON m.model_id = mt.model_id
INNER JOIN tags t ON mt.tag_id = t.tag_id
WHERE t.tag_name IN ('nlp', 'production')
GROUP BY m.model_id, m.model_name
HAVING COUNT(DISTINCT t.tag_name) = 2;  -- Must have BOTH tags

-- Query 4: Tag usage statistics
SELECT
    t.tag_name,
    t.tag_category,
    COUNT(DISTINCT mt.model_id) AS model_count,
    COUNT(DISTINCT dt.dataset_id) AS dataset_count,
    COUNT(DISTINCT mt.model_id) + COUNT(DISTINCT dt.dataset_id) AS total_usage
FROM tags t
LEFT JOIN model_tags mt ON t.tag_id = mt.tag_id
LEFT JOIN dataset_tags dt ON t.tag_id = dt.tag_id
GROUP BY t.tag_id, t.tag_name, t.tag_category
ORDER BY total_usage DESC, t.tag_name;
```

### Step 7.3: Add/Remove Tags

```sql
-- Add tag to model
BEGIN;

-- Add 'high-priority' tag to revenue forecaster
INSERT INTO model_tags (model_id, tag_id)
SELECT
    m.model_id,
    t.tag_id
FROM models m
CROSS JOIN tags t
WHERE m.model_name = 'revenue-forecaster'
  AND t.tag_name = 'high-priority'
ON CONFLICT (model_id, tag_id) DO NOTHING;  -- Prevent duplicates

COMMIT;

-- Remove tag from model
BEGIN;

DELETE FROM model_tags
WHERE model_id = (SELECT model_id FROM models WHERE model_name = 'revenue-forecaster')
  AND tag_id = (SELECT tag_id FROM tags WHERE tag_name = 'experimental');

COMMIT;
```

✅ **Checkpoint**: Many-to-many relationships work correctly.

---

## Part 8: Production Patterns

### Step 8.1: Audit Trail Pattern

```sql
-- Add audit columns to tables (already in schema)
-- created_at, created_by, updated_at, updated_by

-- Query: Recent model changes
SELECT
    model_name,
    created_by,
    created_at,
    updated_at,
    AGE(NOW(), updated_at) AS time_since_update
FROM models
ORDER BY updated_at DESC
LIMIT 10;
```

### Step 8.2: Soft Delete Pattern

```sql
-- Soft delete: Mark as archived instead of DELETE
BEGIN;

UPDATE models
SET
    is_active = FALSE,
    archived_at = NOW(),
    archived_reason = 'Model deprecated in favor of v3'
WHERE model_name = 'old-model';

COMMIT;

-- Query only active models
SELECT * FROM models WHERE is_active = TRUE;
```

### Step 8.3: Version Lineage

```sql
-- Query: Model version timeline
SELECT
    m.model_name,
    mv.semver,
    mv.version_alias,
    mv.status,
    mv.registered_at,
    mv.promoted_to_production_at,
    LAG(mv.semver) OVER (PARTITION BY m.model_id ORDER BY mv.registered_at) AS previous_version,
    LEAD(mv.semver) OVER (PARTITION BY m.model_id ORDER BY mv.registered_at) AS next_version
FROM model_versions mv
JOIN models m ON mv.model_id = m.model_id
WHERE m.model_name = 'sentiment-classifier-v2'
ORDER BY mv.registered_at;
```

✅ **Checkpoint**: Production patterns implemented.

---

## Verification & Testing

### Create Verification Script

Create `scripts/verify_schema.sh`:

```bash
#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== ML Model Registry Schema Verification ===${NC}\n"

# Test 1: Table count
echo -e "${YELLOW}Test 1: Verifying table count...${NC}"
TABLE_COUNT=$(docker exec pg-ml-registry psql -U mlops -d ml_registry -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';")
echo "Tables: $TABLE_COUNT (expected: 12)"

# Test 2: Foreign keys
echo -e "\n${YELLOW}Test 2: Verifying foreign keys...${NC}"
FK_COUNT=$(docker exec pg-ml-registry psql -U mlops -d ml_registry -t -c "SELECT COUNT(*) FROM information_schema.table_constraints WHERE constraint_type = 'FOREIGN KEY';")
echo "Foreign Keys: $FK_COUNT"

# Test 3: Indexes
echo -e "\n${YELLOW}Test 3: Verifying indexes...${NC}"
INDEX_COUNT=$(docker exec pg-ml-registry psql -U mlops -d ml_registry -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';")
echo "Indexes: $INDEX_COUNT"

# Test 4: Data integrity
echo -e "\n${YELLOW}Test 4: Data integrity checks...${NC}"
docker exec pg-ml-registry psql -U mlops -d ml_registry -c "
SELECT
    'All model versions have valid model_id' AS check_name,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM model_versions
WHERE model_id NOT IN (SELECT model_id FROM models);
"

echo -e "\n${GREEN}Verification complete!${NC}"
```

```bash
chmod +x scripts/verify_schema.sh
./scripts/verify_schema.sh
```

---

## Summary

**Completed**: Exercise 02 - Database Design for ML Model Registry

**What You Built**:
- Fully normalized schema (3NF) with 12 tables
- Foreign keys with CASCADE, SET NULL, and RESTRICT behavior
- Junction tables for many-to-many relationships
- Comprehensive seed data (4 models, 5 versions, 3 deployments)
- Complex queries with JOINs, CTEs, and subqueries
- Production patterns (audit trail, soft delete, versioning)

**Key Skills Acquired**:
- Entity-relationship modeling
- Normalization (1NF, 2NF, 3NF)
- Foreign key constraint design
- Many-to-many relationship implementation
- Complex JOIN queries
- Database schema documentation

**Ready For**: Exercise 03 - Advanced SQL Joins

---

**Exercise Complete!** 🎉
