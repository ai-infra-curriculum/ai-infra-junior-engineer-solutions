# Exercise 01 Solution: SQL Fundamentals & CRUD Operations

## Solution Overview

This implementation guide provides complete solutions for Exercise 01: building a production-ready ML training metadata database with PostgreSQL. You'll learn to design schemas, perform CRUD operations, and write production-quality SQL queries for ML infrastructure.

**What You'll Build**:
- PostgreSQL database for ML training runs tracking
- Table schema with constraints and indexes
- 20+ realistic training run records
- Comprehensive CRUD operation examples
- Production-ready queries for ML infrastructure

**Time to Complete**: 3-4 hours
**Difficulty**: Beginner

---

## Table of Contents

1. [Part 1: Environment Setup](#part-1-environment-setup)
2. [Part 2: Schema Creation](#part-2-schema-creation)
3. [Part 3: Seed Data](#part-3-seed-data)
4. [Part 4: Read Operations](#part-4-read-operations)
5. [Part 5: Update Operations](#part-5-update-operations)
6. [Part 6: Delete Operations](#part-6-delete-operations)
7. [Part 7: Transactions](#part-7-transactions)
8. [Part 8: Advanced Queries](#part-8-advanced-queries)
9. [Part 9: Challenge Solutions](#part-9-challenge-solutions)
10. [Verification & Testing](#verification--testing)

---

## Part 1: Environment Setup

### Step 1.1: Create Project Directory

```bash
# Create exercise directory
mkdir -p ~/ml-training-registry
cd ~/ml-training-registry

# Create subdirectories
mkdir -p sql screenshots scripts

# Create README
cat > README.md << 'EOF'
# ML Training Registry

PostgreSQL database for tracking ML model training jobs across infrastructure.

## Features
- Training run tracking with status management
- Metrics storage (accuracy, loss, precision, recall, F1)
- Resource utilization tracking (GPU/CPU hours)
- Flexible JSONB parameter storage
- Production-ready constraints and indexes

## Setup
```bash
# Start PostgreSQL
docker run --name pg-ml-training \
  -e POSTGRES_PASSWORD=mlops_secure_pass \
  -e POSTGRES_USER=mlops \
  -e POSTGRES_DB=ml_infra \
  -p 5432:5432 \
  -d postgres:14

# Connect
docker exec -it pg-ml-training psql -U mlops -d ml_infra
```

## Schema
- `training_runs`: Main table tracking all training jobs
  - Primary key: UUID for distributed system compatibility
  - Constraints: Status validation, metric ranges, time ordering
  - Indexes: Status, timestamps, model names, framework, JSONB parameters
EOF

echo "✓ Project directory created"
```

### Step 1.2: Start PostgreSQL Container

```bash
# Start PostgreSQL 14 with proper configuration
docker run --name pg-ml-training \
  -e POSTGRES_PASSWORD=mlops_secure_pass \
  -e POSTGRES_USER=mlops \
  -e POSTGRES_DB=ml_infra \
  -e POSTGRES_INITDB_ARGS="--encoding=UTF8 --lc-collate=en_US.UTF-8 --lc-ctype=en_US.UTF-8" \
  -p 5432:5432 \
  -v $(pwd)/data:/var/lib/postgresql/data \
  -d postgres:14

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to start..."
sleep 5

# Verify container is running
docker ps | grep pg-ml-training

# Test connection
docker exec pg-ml-training psql -U mlops -d ml_infra -c "SELECT version();"
```

**Expected Output**:
```
PostgreSQL 14.x on x86_64-pc-linux-gnu, compiled by gcc...
```

**Troubleshooting**:
```bash
# If port 5432 is in use
docker run --name pg-ml-training ... -p 5433:5432 ...

# Check logs if container fails
docker logs pg-ml-training

# Stop and remove container
docker stop pg-ml-training && docker rm pg-ml-training

# Remove volume (WARNING: deletes all data)
rm -rf data/
```

### Step 1.3: Create Connection Helper Script

```bash
cat > scripts/connect.sh << 'EOF'
#!/bin/bash
# Connection helper script

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Connecting to ML Training Registry...${NC}"
echo -e "${YELLOW}Database: ml_infra${NC}"
echo -e "${YELLOW}User: mlops${NC}"
echo ""

# Connect to PostgreSQL
docker exec -it pg-ml-training psql -U mlops -d ml_infra
EOF

chmod +x scripts/connect.sh

echo "✓ Connection script created: scripts/connect.sh"
```

### Step 1.4: Verify psql Commands

Connect to the database and test essential commands:

```bash
# Connect
./scripts/connect.sh
```

Inside psql:
```sql
-- Enable timing for performance monitoring
\timing on

-- List databases
\l

-- Show current connection info
\conninfo

-- Get help
\?

-- List psql commands
\h

-- Clear screen (or Ctrl+L)
\! clear

-- Toggle expanded display (useful for wide results)
\x

-- Quit
\q
```

✅ **Checkpoint**: PostgreSQL is running and accessible.

---

## Part 2: Schema Creation

### Step 2.1: Create Training Runs Table

Create the file `sql/01_create_training_runs.sql`:

```sql
-- ============================================
-- ML Training Runs Table Schema
-- ============================================
-- Purpose: Track all ML model training jobs
-- Author: ML Infrastructure Team
-- Date: 2025-11-01
-- Version: 1.0.0
-- ============================================

-- Drop existing table (development only)
DROP TABLE IF EXISTS training_runs CASCADE;

-- Create training_runs table with comprehensive constraints
CREATE TABLE training_runs (
    -- Primary key: UUID for distributed systems
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Model identification
    model_name TEXT NOT NULL,
    framework TEXT NOT NULL,
    experiment_name TEXT NOT NULL,

    -- Status tracking with default
    status TEXT NOT NULL DEFAULT 'queued',

    -- Performance metrics (NULL if not applicable)
    accuracy NUMERIC(5,4),        -- 0.9234 (4 decimals)
    loss NUMERIC(8,5),            -- 123.45678
    precision_score NUMERIC(5,4),
    recall_score NUMERIC(5,4),
    f1_score NUMERIC(5,4),

    -- Data and compute configuration
    dataset TEXT NOT NULL,
    compute_target TEXT NOT NULL,
    gpu_hours NUMERIC(8,2) DEFAULT 0,
    cpu_hours NUMERIC(8,2) DEFAULT 0,

    -- Flexible parameter storage (JSONB)
    parameters JSONB DEFAULT '{}'::jsonb,

    -- Timestamps (timezone-aware)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Audit and notes
    notes TEXT,
    created_by TEXT DEFAULT CURRENT_USER,

    -- ============================================
    -- CONSTRAINTS
    -- ============================================

    -- Framework validation
    CONSTRAINT valid_framework CHECK (
        framework IN ('pytorch', 'tensorflow', 'sklearn', 'xgboost', 'jax', 'mxnet')
    ),

    -- Status validation
    CONSTRAINT valid_status CHECK (
        status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'timeout')
    ),

    -- Metric range validation (0-1)
    CONSTRAINT valid_accuracy CHECK (
        accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)
    ),
    CONSTRAINT valid_precision CHECK (
        precision_score IS NULL OR (precision_score >= 0 AND precision_score <= 1)
    ),
    CONSTRAINT valid_recall CHECK (
        recall_score IS NULL OR (recall_score >= 0 AND recall_score <= 1)
    ),
    CONSTRAINT valid_f1 CHECK (
        f1_score IS NULL OR (f1_score >= 0 AND f1_score <= 1)
    ),

    -- Resource validation (non-negative)
    CONSTRAINT valid_gpu_hours CHECK (gpu_hours >= 0),
    CONSTRAINT valid_cpu_hours CHECK (cpu_hours >= 0),

    -- Timestamp consistency for completed runs
    CONSTRAINT completed_timestamp CHECK (
        (status IN ('succeeded', 'failed', 'cancelled', 'timeout') AND completed_at IS NOT NULL)
        OR
        (status IN ('queued', 'running') AND completed_at IS NULL)
    ),

    -- Timestamp consistency for started runs
    CONSTRAINT started_timestamp CHECK (
        (status IN ('running', 'succeeded', 'failed', 'cancelled', 'timeout') AND started_at IS NOT NULL)
        OR
        (status = 'queued' AND started_at IS NULL)
    ),

    -- Logical time ordering
    CONSTRAINT time_order CHECK (
        (started_at IS NULL OR started_at >= created_at) AND
        (completed_at IS NULL OR completed_at >= created_at)
    ),

    -- Prevent duplicate runs on same day
    CONSTRAINT unique_daily_run UNIQUE (
        model_name, experiment_name, dataset, DATE(created_at)
    )
);

-- ============================================
-- INDEXES
-- ============================================

-- Single-column indexes for common filters
CREATE INDEX idx_training_runs_status
    ON training_runs(status);

CREATE INDEX idx_training_runs_created_at
    ON training_runs(created_at DESC);

CREATE INDEX idx_training_runs_model_name
    ON training_runs(model_name);

CREATE INDEX idx_training_runs_framework
    ON training_runs(framework);

CREATE INDEX idx_training_runs_compute_target
    ON training_runs(compute_target);

-- GIN index for JSONB queries
CREATE INDEX idx_training_runs_parameters
    ON training_runs USING GIN (parameters);

-- Partial index for active jobs (queued or running)
CREATE INDEX idx_training_runs_active
    ON training_runs(status, created_at DESC)
    WHERE status IN ('queued', 'running');

-- Composite index for date-range queries by status
CREATE INDEX idx_training_runs_status_date
    ON training_runs(status, created_at DESC);

-- ============================================
-- COMMENTS (Documentation)
-- ============================================

COMMENT ON TABLE training_runs IS
    'Tracks all ML model training jobs across the platform';

COMMENT ON COLUMN training_runs.run_id IS
    'Unique identifier for each training run (UUID)';

COMMENT ON COLUMN training_runs.parameters IS
    'JSONB field storing hyperparameters and configuration';

COMMENT ON COLUMN training_runs.gpu_hours IS
    'Total GPU compute hours consumed';

COMMENT ON COLUMN training_runs.status IS
    'Current status: queued, running, succeeded, failed, cancelled, timeout';

COMMENT ON COLUMN training_runs.accuracy IS
    'Model accuracy (0-1) or other primary metric (e.g., mAP for object detection)';

COMMENT ON CONSTRAINT valid_framework ON training_runs IS
    'Ensures framework is one of the supported ML frameworks';

COMMENT ON CONSTRAINT unique_daily_run ON training_runs IS
    'Prevents duplicate runs of same model+experiment+dataset on same day';
```

### Step 2.2: Load Schema

```bash
# Execute SQL file
docker exec -i pg-ml-training psql -U mlops -d ml_infra < sql/01_create_training_runs.sql

# Or from within psql:
# \i sql/01_create_training_runs.sql
```

**Expected Output**:
```
DROP TABLE
CREATE TABLE
CREATE INDEX
CREATE INDEX
CREATE INDEX
CREATE INDEX
CREATE INDEX
CREATE INDEX
CREATE INDEX
CREATE INDEX
COMMENT
COMMENT
COMMENT
COMMENT
COMMENT
COMMENT
COMMENT
COMMENT
```

### Step 2.3: Verify Schema

```bash
# Connect and verify
./scripts/connect.sh
```

Inside psql:
```sql
-- Describe table structure
\d training_runs

-- List all constraints
SELECT
    conname AS constraint_name,
    contype AS constraint_type,
    pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = 'training_runs'::regclass
ORDER BY contype, conname;

-- List all indexes
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'training_runs'
ORDER BY indexname;

-- Check table size (should be 0 bytes)
SELECT pg_size_pretty(pg_total_relation_size('training_runs')) AS total_size;

-- Count rows (should be 0)
SELECT COUNT(*) FROM training_runs;
```

**Expected Output for Constraints**:
```
      constraint_name      | constraint_type |          definition
---------------------------+-----------------+------------------------------
 valid_accuracy            | c               | CHECK (accuracy >= 0 AND ...)
 valid_framework           | c               | CHECK (framework IN (...))
 valid_status              | c               | CHECK (status IN (...))
 training_runs_pkey        | p               | PRIMARY KEY (run_id)
 unique_daily_run          | u               | UNIQUE (model_name, ...)
```

✅ **Checkpoint**: Table created with all constraints and indexes.

---

## Part 3: Seed Data

### Step 3.1: Create Seed Data Script

Create `sql/02_seed_training_runs.sql`:

```sql
-- ============================================
-- ML Training Runs Seed Data
-- ============================================
-- Purpose: Insert realistic training run data
-- Date: 2025-11-01
-- Records: 20 training runs
-- ============================================

-- Insert 20 comprehensive training runs
INSERT INTO training_runs (
    model_name, framework, experiment_name, status,
    accuracy, loss, precision_score, recall_score, f1_score,
    dataset, compute_target, gpu_hours, cpu_hours,
    parameters, created_at, started_at, completed_at, notes
) VALUES

-- ====================
-- SUCCESSFUL RUNS (12)
-- ====================

-- Run 1: ResNet50 Image Classification
(
    'resnet50-classification',
    'pytorch',
    'exp-baseline-2025-q1',
    'succeeded',
    0.9234,
    0.2876,
    0.9156,
    0.9312,
    0.9233,
    'imagenet-1k',
    'k8s-gpu-a100',
    24.5,
    2.1,
    '{"learning_rate": 0.001, "batch_size": 64, "optimizer": "adam", "epochs": 30, "augmentation": true}',
    NOW() - INTERVAL '5 days',
    NOW() - INTERVAL '5 days' + INTERVAL '10 minutes',
    NOW() - INTERVAL '4 days 1 hour',
    'Baseline model achieved target accuracy. Ready for deployment.'
),

-- Run 2: BERT Sentiment Analysis
(
    'bert-sentiment-analysis',
    'tensorflow',
    'exp-bert-base-v1',
    'succeeded',
    0.8923,
    0.3245,
    0.8856,
    0.8991,
    0.8923,
    'imdb-reviews',
    'aws-sagemaker-p3',
    16.2,
    1.5,
    '{"learning_rate": 0.00005, "batch_size": 32, "max_length": 512, "model_checkpoint": "bert-base-uncased"}',
    NOW() - INTERVAL '7 days',
    NOW() - INTERVAL '7 days' + INTERVAL '5 minutes',
    NOW() - INTERVAL '6 days 8 hours',
    'BERT fine-tuning completed. F1 score meets SLA requirements.'
),

-- Run 3: Recommendation System
(
    'recommendation-collaborative-filter',
    'sklearn',
    'exp-svd-baseline',
    'succeeded',
    NULL,  -- Not applicable for recommendation systems
    0.1234,  -- RMSE
    NULL,
    NULL,
    NULL,
    'movielens-25m',
    'on-premise-cpu',
    0.0,
    45.6,
    '{"algorithm": "SVD", "n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}',
    NOW() - INTERVAL '3 days',
    NOW() - INTERVAL '3 days' + INTERVAL '2 minutes',
    NOW() - INTERVAL '2 days 20 hours',
    'SVD baseline for comparison. RMSE: 0.1234'
),

-- Run 4: Object Detection YOLO
(
    'object-detection-yolo',
    'pytorch',
    'exp-yolo-v8-coco',
    'succeeded',
    0.7845,  -- mAP@0.5
    1.2345,
    0.7623,
    0.8012,
    0.7812,
    'coco-2017',
    'gcp-vertex-ai-v100',
    32.4,
    3.2,
    '{"model": "yolov8l", "img_size": 640, "batch_size": 16, "epochs": 50, "iou_threshold": 0.5}',
    NOW() - INTERVAL '10 days',
    NOW() - INTERVAL '10 days' + INTERVAL '15 minutes',
    NOW() - INTERVAL '8 days 12 hours',
    'YOLOv8 training complete. mAP@0.5: 0.7845. Deployed to production.'
),

-- Run 5: Time Series LSTM
(
    'timeseries-lstm-forecast',
    'tensorflow',
    'exp-lstm-multistep-v2',
    'succeeded',
    NULL,
    0.0456,  -- MAE
    NULL,
    NULL,
    NULL,
    'energy-consumption-hourly',
    'aws-batch-gpu',
    8.7,
    0.9,
    '{"layers": [128, 64, 32], "dropout": 0.2, "sequence_length": 168, "forecast_horizon": 24}',
    NOW() - INTERVAL '2 days',
    NOW() - INTERVAL '2 days' + INTERVAL '3 minutes',
    NOW() - INTERVAL '1 day 14 hours',
    'LSTM forecast model. MAE improved by 12% vs baseline.'
),

-- Run 6: XGBoost Tabular Classification
(
    'tabular-xgboost-classification',
    'xgboost',
    'exp-xgb-fraud-detection',
    'succeeded',
    0.9567,
    0.1234,
    0.9423,
    0.9689,
    0.9554,
    'credit-card-fraud',
    'aws-batch-cpu',
    0.0,
    8.9,
    '{"max_depth": 6, "learning_rate": 0.1, "n_estimators": 200, "objective": "binary:logistic"}',
    NOW() - INTERVAL '1 day',
    NOW() - INTERVAL '1 day' + INTERVAL '2 minutes',
    NOW() - INTERVAL '1 day' + INTERVAL '8 hours 45 minutes',
    'XGBoost fraud detection. Precision: 0.9423, Recall: 0.9689. Deployed to prod.'
),

-- Run 7: Semantic Segmentation U-Net
(
    'semantic-segmentation-unet',
    'pytorch',
    'exp-unet-medical-imaging',
    'succeeded',
    0.8834,  -- Dice coefficient
    0.2145,
    0.8756,
    0.8912,
    0.8833,
    'medical-ct-scans',
    'k8s-gpu-v100',
    18.9,
    1.8,
    '{"architecture": "unet", "encoder": "resnet34", "num_classes": 3, "loss": "dice_ce"}',
    NOW() - INTERVAL '9 days',
    NOW() - INTERVAL '9 days' + INTERVAL '12 minutes',
    NOW() - INTERVAL '8 days 6 hours',
    'U-Net segmentation for medical imaging. Dice: 0.8834. Clinical validation pending.'
),

-- Run 8: Prophet Time Series
(
    'time-series-prophet-forecast',
    'sklearn',
    'exp-prophet-sales-forecast',
    'succeeded',
    NULL,
    0.0892,  -- MAPE
    NULL,
    NULL,
    NULL,
    'retail-sales-monthly',
    'on-premise-cpu',
    0.0,
    2.3,
    '{"seasonality_mode": "multiplicative", "yearly_seasonality": true, "weekly_seasonality": false}',
    NOW() - INTERVAL '5 days',
    NOW() - INTERVAL '5 days' + INTERVAL '1 minute',
    NOW() - INTERVAL '5 days' + INTERVAL '2 hours 15 minutes',
    'Prophet forecast model. MAPE: 8.92%. Used for quarterly planning.'
),

-- Run 9: Question Answering DistilBERT
(
    'bert-question-answering',
    'tensorflow',
    'exp-squad-v2-distilbert',
    'succeeded',
    0.8145,  -- F1 score
    1.0234,
    0.8012,
    0.8289,
    0.8145,
    'squad-v2',
    'gcp-vertex-ai-t4',
    12.4,
    1.2,
    '{"model": "distilbert-base-uncased", "max_seq_length": 384, "doc_stride": 128, "batch_size": 12}',
    NOW() - INTERVAL '11 days',
    NOW() - INTERVAL '11 days' + INTERVAL '8 minutes',
    NOW() - INTERVAL '10 days 14 hours',
    'DistilBERT QA model. F1: 0.8145. 40% faster than BERT-base with minimal accuracy loss.'
),

-- Run 10: Image Super-Resolution ESRGAN
(
    'image-super-resolution-esrgan',
    'pytorch',
    'exp-esrgan-4x-upscale',
    'succeeded',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    'div2k-flickr2k',
    'k8s-gpu-a100',
    28.6,
    2.4,
    '{"scale_factor": 4, "generator_lr": 0.0001, "discriminator_lr": 0.0001, "perceptual_loss_weight": 1.0}',
    NOW() - INTERVAL '6 days',
    NOW() - INTERVAL '6 days' + INTERVAL '18 minutes',
    NOW() - INTERVAL '5 days 4 hours',
    'ESRGAN 4x super-resolution. PSNR: 28.45 dB, SSIM: 0.8234. Visual quality excellent.'
),

-- Run 11: Audio Classification
(
    'audio-classification-vgg',
    'tensorflow',
    'exp-audio-vgg-urban-sounds',
    'succeeded',
    0.9123,
    0.2567,
    0.9045,
    0.9201,
    0.9122,
    'urbansound8k',
    'aws-sagemaker-ml-p2',
    6.8,
    0.7,
    '{"model": "vggish", "sample_rate": 16000, "n_mels": 128, "hop_length": 512}',
    NOW() - INTERVAL '3 days',
    NOW() - INTERVAL '3 days' + INTERVAL '4 minutes',
    NOW() - INTERVAL '2 days 18 hours',
    'Audio classification using VGGish. Accuracy: 0.9123 on UrbanSound8K test set.'
),

-- Run 12: Speech Recognition Whisper (Completed)
(
    'speech-recognition-whisper',
    'pytorch',
    'exp-whisper-large-v3',
    'succeeded',
    0.9456,
    0.1892,
    0.9423,
    0.9489,
    0.9456,
    'librispeech-960h',
    'gcp-vertex-ai-a100',
    18.5,
    1.7,
    '{"model": "whisper-large-v3", "batch_size": 16, "gradient_checkpointing": true}',
    NOW() - INTERVAL '30 minutes',
    NOW() - INTERVAL '25 minutes',
    NOW() - INTERVAL '10 minutes',
    'Training completed successfully. Model deployed to production.'
),

-- ====================
-- FAILED RUNS (3)
-- ====================

-- Run 13: GAN OOM Failure
(
    'gan-image-generation',
    'pytorch',
    'exp-stylegan3-highres',
    'failed',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    'ffhq-512x512',
    'k8s-gpu-a100',
    12.3,
    1.1,
    '{"resolution": 512, "batch_size": 8, "g_lr": 0.0025, "d_lr": 0.0025}',
    NOW() - INTERVAL '1 day',
    NOW() - INTERVAL '1 day' + INTERVAL '8 minutes',
    NOW() - INTERVAL '1 day' + INTERVAL '12 hours 20 minutes',
    'Training failed: OOM error. Discriminator loss exploded at iteration 45000. Need to reduce batch size.'
),

-- Run 14: JAX TPU OOM
(
    'nlp-transformer-translation',
    'jax',
    'exp-t5-en-fr',
    'failed',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    'wmt14-en-fr',
    'gcp-tpu-v3',
    0.0,  -- TPU, not GPU
    156.3,
    '{"model": "t5-base", "max_source_length": 128, "max_target_length": 128, "batch_size": 64}',
    NOW() - INTERVAL '4 days',
    NOW() - INTERVAL '4 days' + INTERVAL '20 minutes',
    NOW() - INTERVAL '3 days 18 hours',
    'JAX/Flax training failed: TPU out of memory. Reduce batch size from 64 to 32.'
),

-- Run 15: Anomaly Detection NaN Loss
(
    'anomaly-detection-autoencoder',
    'tensorflow',
    'exp-vae-network-intrusion',
    'failed',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    'kdd-cup-99',
    'on-premise-gpu',
    3.2,
    0.4,
    '{"latent_dim": 16, "encoder_layers": [128, 64], "decoder_layers": [64, 128]}',
    NOW() - INTERVAL '6 days',
    NOW() - INTERVAL '6 days' + INTERVAL '5 minutes',
    NOW() - INTERVAL '6 days' + INTERVAL '3 hours 12 minutes',
    'Training diverged. Loss became NaN after epoch 12. Investigating data preprocessing.'
),

-- ====================
-- RUNNING JOBS (2)
-- ====================

-- Run 16: LLM Fine-tuning (Running)
(
    'llm-fine-tune-mistral',
    'pytorch',
    'exp-mistral-7b-instruct',
    'running',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    'custom-instruction-dataset',
    'aws-p4d-24xlarge',
    48.5,  -- Expensive!
    4.2,
    '{"model": "mistral-7b-v0.1", "lora_r": 16, "lora_alpha": 32, "batch_size": 4, "gradient_accumulation": 8}',
    NOW() - INTERVAL '6 hours',
    NOW() - INTERVAL '5 hours 55 minutes',
    NULL,
    NULL
),

-- Run 17: Multimodal CLIP (Running)
(
    'multimodal-clip-training',
    'pytorch',
    'exp-clip-vit-b32',
    'running',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    'laion-400m-subset',
    'k8s-gpu-a100-cluster',
    156.2,
    12.3,
    '{"vision_model": "vit-b-32", "text_model": "bert-base", "batch_size": 256, "num_gpus": 8}',
    NOW() - INTERVAL '2 days 4 hours',
    NOW() - INTERVAL '2 days 3 hours 50 minutes',
    NULL,
    NULL
),

-- ====================
-- QUEUED JOBS (1)
-- ====================

-- Run 18: Reinforcement Learning (Queued)
(
    'reinforcement-learning-ppo',
    'pytorch',
    'exp-ppo-atari-breakout',
    'queued',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    'atari-breakout',
    'on-premise-gpu',
    0.0,
    0.0,
    '{"algorithm": "ppo", "num_envs": 16, "num_steps": 128, "num_epochs": 4, "clip_range": 0.2}',
    NOW() - INTERVAL '15 minutes',
    NULL,
    NULL,
    NULL
),

-- ====================
-- CANCELLED JOBS (1)
-- ====================

-- Run 19: Graph Neural Network (Cancelled)
(
    'graph-neural-network',
    'pytorch',
    'exp-gnn-molecular-properties',
    'cancelled',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    'qm9-molecules',
    'aws-batch-cpu',
    0.0,
    12.3,
    '{"model": "gcn", "hidden_dim": 128, "num_layers": 4, "pooling": "mean"}',
    NOW() - INTERVAL '8 days',
    NOW() - INTERVAL '8 days' + INTERVAL '10 minutes',
    NOW() - INTERVAL '8 days' + INTERVAL '2 hours 5 minutes',
    'Cancelled by user. Dataset preprocessing issues discovered. Will retry with corrected data.'
),

-- ====================
-- TIMEOUT (1)
-- ====================

-- Run 20: Neural Architecture Search (Timeout)
(
    'neural-architecture-search',
    'tensorflow',
    'exp-nas-efficient-net',
    'timeout',
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    'cifar-100',
    'gcp-tpu-v4',
    0.0,
    456.7,
    '{"search_space": "efficientnet", "num_trials": 1000, "time_budget_hours": 48}',
    NOW() - INTERVAL '12 days',
    NOW() - INTERVAL '12 days' + INTERVAL '30 minutes',
    NOW() - INTERVAL '10 days',
    'Exceeded 48-hour time budget. Architecture search incomplete. Consider resuming from checkpoint.'
);

-- ============================================
-- VERIFICATION QUERIES
-- ============================================

-- Total count
SELECT COUNT(*) AS total_runs FROM training_runs;

-- Breakdown by status
SELECT status, COUNT(*) AS count
FROM training_runs
GROUP BY status
ORDER BY count DESC;

-- Framework distribution
SELECT framework, COUNT(*) AS count
FROM training_runs
GROUP BY framework
ORDER BY count DESC;

-- Average accuracy by framework (for succeeded runs)
SELECT
    framework,
    COUNT(*) AS total_runs,
    COUNT(accuracy) AS runs_with_accuracy,
    ROUND(AVG(accuracy)::numeric, 4) AS avg_accuracy
FROM training_runs
WHERE status = 'succeeded'
GROUP BY framework
ORDER BY avg_accuracy DESC NULLS LAST;
```

### Step 3.2: Load Seed Data

```bash
# Execute seed data script
docker exec -i pg-ml-training psql -U mlops -d ml_infra < sql/02_seed_training_runs.sql
```

**Expected Output**:
```
INSERT 0 20
 total_runs
------------
         20
(1 row)

  status   | count
-----------+-------
 succeeded |    12
 failed    |     3
 running   |     2
 queued    |     1
 cancelled |     1
 timeout   |     1
(6 rows)

 framework  | count
------------+-------
 pytorch    |    10
 tensorflow |     5
 sklearn    |     2
 xgboost    |     1
 jax        |     1
 mxnet      |     0
(6 rows)

  framework  | total_runs | runs_with_accuracy | avg_accuracy
-------------+------------+--------------------+--------------
 xgboost     |          1 |                  1 |       0.9567
 tensorflow  |          4 |                  4 |       0.8885
 pytorch     |          7 |                  7 |       0.8818
(3 rows)
```

### Step 3.3: Verify Data Integrity

```sql
-- Connect to database
-- ./scripts/connect.sh

-- View first 5 runs
SELECT
    run_id,
    model_name,
    framework,
    status,
    ROUND(accuracy::numeric, 4) AS accuracy,
    ROUND(gpu_hours::numeric, 2) AS gpu_hours
FROM training_runs
ORDER BY created_at DESC
LIMIT 5;

-- Check JSONB parameters
SELECT
    model_name,
    framework,
    status,
    parameters->>'learning_rate' AS learning_rate,
    parameters->>'batch_size' AS batch_size
FROM training_runs
WHERE framework = 'pytorch'
LIMIT 5;

-- Validate constraints are working
-- This should FAIL due to invalid framework:
INSERT INTO training_runs (model_name, framework, experiment_name, dataset, compute_target)
VALUES ('test', 'invalid-framework', 'test', 'test', 'test');
-- Expected: ERROR:  new row for relation "training_runs" violates check constraint "valid_framework"

-- This should FAIL due to invalid accuracy:
INSERT INTO training_runs (model_name, framework, experiment_name, dataset, compute_target, accuracy)
VALUES ('test', 'pytorch', 'test', 'test', 'test', 1.5);
-- Expected: ERROR:  new row for relation "training_runs" violates check constraint "valid_accuracy"
```

✅ **Checkpoint**: 20 training runs inserted with realistic data.

---

## Part 4: Read Operations

### Step 4.1: Create CRUD Operations File

Create `sql/03_crud_operations.sql`:

```sql
-- ============================================
-- CRUD Operations Examples
-- ============================================
-- Purpose: Comprehensive CRUD query examples
-- Date: 2025-11-01
-- ============================================

-- ====================
-- READ (SELECT) QUERIES
-- ====================

-- Query 1: Select all columns (development only!)
SELECT * FROM training_runs LIMIT 3;

-- Query 2: Select specific columns
SELECT
    run_id,
    model_name,
    framework,
    status,
    created_at
FROM training_runs
ORDER BY created_at DESC
LIMIT 5;

-- Query 3: Filter with WHERE
SELECT
    model_name,
    status,
    accuracy
FROM training_runs
WHERE status = 'succeeded'
ORDER BY accuracy DESC NULLS LAST;

-- Query 4: Multiple conditions with AND
SELECT
    model_name,
    framework,
    accuracy,
    gpu_hours
FROM training_runs
WHERE status = 'succeeded'
  AND framework = 'pytorch'
  AND accuracy > 0.9
ORDER BY accuracy DESC;

-- Query 5: Multiple conditions with OR
SELECT
    model_name,
    status,
    notes
FROM training_runs
WHERE status = 'failed'
   OR status = 'timeout'
ORDER BY created_at DESC;

-- Query 6: IN operator
SELECT
    model_name,
    framework,
    status
FROM training_runs
WHERE framework IN ('pytorch', 'tensorflow')
ORDER BY framework, model_name;

-- Query 7: BETWEEN for ranges
SELECT
    model_name,
    gpu_hours,
    created_at
FROM training_runs
WHERE gpu_hours BETWEEN 10 AND 30
ORDER BY gpu_hours DESC;

-- Query 8: LIKE for pattern matching
SELECT
    model_name,
    dataset
FROM training_runs
WHERE model_name LIKE '%bert%'
ORDER BY model_name;

-- Query 9: IS NULL / IS NOT NULL
SELECT
    model_name,
    status,
    accuracy
FROM training_runs
WHERE accuracy IS NULL
ORDER BY model_name;

SELECT
    model_name,
    status,
    accuracy
FROM training_runs
WHERE accuracy IS NOT NULL
ORDER BY accuracy DESC;

-- Query 10: ORDER BY multiple columns
SELECT
    model_name,
    framework,
    accuracy
FROM training_runs
ORDER BY framework ASC, accuracy DESC NULLS LAST
LIMIT 10;

-- ====================
-- AGGREGATE QUERIES
-- ====================

-- Count total runs
SELECT COUNT(*) AS total_runs
FROM training_runs;

-- Count non-null accuracies
SELECT COUNT(accuracy) AS runs_with_accuracy
FROM training_runs;

-- Average accuracy
SELECT ROUND(AVG(accuracy)::numeric, 4) AS avg_accuracy
FROM training_runs
WHERE accuracy IS NOT NULL;

-- Min, Max, Sum
SELECT
    MIN(gpu_hours) AS min_gpu,
    MAX(gpu_hours) AS max_gpu,
    ROUND(SUM(gpu_hours)::numeric, 2) AS total_gpu_hours,
    ROUND(AVG(gpu_hours)::numeric, 2) AS avg_gpu_hours
FROM training_runs;

-- Standard deviation
SELECT
    ROUND(STDDEV(accuracy)::numeric, 4) AS stddev_accuracy,
    ROUND(VARIANCE(accuracy)::numeric, 6) AS variance_accuracy
FROM training_runs
WHERE accuracy IS NOT NULL;

-- ====================
-- GROUP BY QUERIES
-- ====================

-- Runs by status
SELECT
    status,
    COUNT(*) AS count
FROM training_runs
GROUP BY status
ORDER BY count DESC;

-- Runs by framework
SELECT
    framework,
    COUNT(*) AS count
FROM training_runs
GROUP BY framework
ORDER BY count DESC;

-- Average accuracy by framework
SELECT
    framework,
    COUNT(*) AS total_runs,
    COUNT(accuracy) AS runs_with_accuracy,
    ROUND(AVG(accuracy)::numeric, 4) AS avg_accuracy,
    ROUND(MIN(accuracy)::numeric, 4) AS min_accuracy,
    ROUND(MAX(accuracy)::numeric, 4) AS max_accuracy
FROM training_runs
WHERE status = 'succeeded'
GROUP BY framework
ORDER BY avg_accuracy DESC NULLS LAST;

-- GPU usage by compute target
SELECT
    compute_target,
    COUNT(*) AS runs,
    ROUND(SUM(gpu_hours)::numeric, 2) AS total_gpu_hours,
    ROUND(AVG(gpu_hours)::numeric, 2) AS avg_gpu_hours,
    ROUND(MAX(gpu_hours)::numeric, 2) AS max_gpu_hours
FROM training_runs
GROUP BY compute_target
ORDER BY total_gpu_hours DESC;

-- HAVING clause (filter groups)
SELECT
    framework,
    COUNT(*) AS runs
FROM training_runs
GROUP BY framework
HAVING COUNT(*) >= 3
ORDER BY runs DESC;

-- Complex grouping
SELECT
    framework,
    status,
    COUNT(*) AS count,
    ROUND(AVG(gpu_hours)::numeric, 2) AS avg_gpu_hours
FROM training_runs
GROUP BY framework, status
ORDER BY framework, status;

-- ====================
-- JSONB QUERIES
-- ====================

-- Access JSON field
SELECT
    model_name,
    parameters->>'learning_rate' AS learning_rate,
    parameters->>'batch_size' AS batch_size,
    parameters->>'optimizer' AS optimizer
FROM training_runs
WHERE parameters ? 'learning_rate'  -- Check if key exists
LIMIT 10;

-- Filter by JSON value
SELECT
    model_name,
    parameters->>'optimizer' AS optimizer
FROM training_runs
WHERE parameters->>'optimizer' = 'adam';

-- Convert JSON value to number for comparison
SELECT
    model_name,
    (parameters->>'batch_size')::int AS batch_size
FROM training_runs
WHERE (parameters->>'batch_size')::int >= 32
ORDER BY batch_size DESC;

-- Check if JSON contains key
SELECT
    model_name,
    parameters->>'lora_r' AS lora_r
FROM training_runs
WHERE parameters ? 'lora_r';  -- LoRA fine-tuning

-- JSON array operations
SELECT
    model_name,
    parameters->'encoder_layers' AS encoder_layers
FROM training_runs
WHERE parameters ? 'encoder_layers';

-- ====================
-- DATE/TIME QUERIES
-- ====================

-- Runs in last 7 days
SELECT
    model_name,
    status,
    created_at
FROM training_runs
WHERE created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Runs today
SELECT
    model_name,
    status,
    created_at
FROM training_runs
WHERE DATE(created_at) = CURRENT_DATE;

-- Runs this week
SELECT
    model_name,
    status,
    created_at
FROM training_runs
WHERE created_at > DATE_TRUNC('week', NOW())
ORDER BY created_at DESC;

-- Calculate duration (for completed runs)
SELECT
    model_name,
    status,
    started_at,
    completed_at,
    completed_at - started_at AS duration,
    EXTRACT(EPOCH FROM (completed_at - started_at)) / 3600 AS duration_hours
FROM training_runs
WHERE completed_at IS NOT NULL
ORDER BY duration DESC;

-- Extract parts of timestamp
SELECT
    model_name,
    EXTRACT(YEAR FROM created_at) AS year,
    EXTRACT(MONTH FROM created_at) AS month,
    EXTRACT(DAY FROM created_at) AS day,
    TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') AS formatted_date
FROM training_runs
LIMIT 5;

-- Age function
SELECT
    model_name,
    created_at,
    AGE(NOW(), created_at) AS age
FROM training_runs
ORDER BY age ASC
LIMIT 5;
```

### Step 4.2: Execute Read Queries

```bash
# Run all read operations
docker exec -i pg-ml-training psql -U mlops -d ml_infra < sql/03_crud_operations.sql > output/read_results.txt 2>&1

# Or execute interactively
./scripts/connect.sh
# \i sql/03_crud_operations.sql
```

### Step 4.3: Production ML Queries

Add to `sql/03_crud_operations.sql`:

```sql
-- ====================
-- PRACTICAL ML INFRASTRUCTURE QUERIES
-- ====================

-- Query: Active jobs (running or queued)
SELECT
    run_id,
    model_name,
    framework,
    status,
    created_at,
    AGE(NOW(), created_at) AS wait_time,
    compute_target,
    gpu_hours AS current_gpu_hours
FROM training_runs
WHERE status IN ('queued', 'running')
ORDER BY created_at ASC;

-- Query: Recent failures (last 7 days) with notes
SELECT
    model_name,
    framework,
    dataset,
    created_at,
    completed_at,
    completed_at - started_at AS duration,
    notes
FROM training_runs
WHERE status = 'failed'
  AND created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Query: Top 10 most accurate models
SELECT
    model_name,
    framework,
    dataset,
    ROUND(accuracy::numeric, 4) AS accuracy,
    ROUND(precision_score::numeric, 4) AS precision,
    ROUND(recall_score::numeric, 4) AS recall,
    ROUND(f1_score::numeric, 4) AS f1,
    ROUND(gpu_hours::numeric, 2) AS gpu_hours
FROM training_runs
WHERE status = 'succeeded'
  AND accuracy IS NOT NULL
ORDER BY accuracy DESC
LIMIT 10;

-- Query: Resource utilization by compute target
SELECT
    compute_target,
    COUNT(*) AS total_runs,
    COUNT(*) FILTER (WHERE status = 'succeeded') AS successful_runs,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_runs,
    ROUND(SUM(gpu_hours)::numeric, 2) AS total_gpu_hours,
    ROUND(AVG(gpu_hours)::numeric, 2) AS avg_gpu_hours_per_run,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'succeeded') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct
FROM training_runs
GROUP BY compute_target
ORDER BY total_gpu_hours DESC;

-- Query: High GPU usage runs (potential cost issues)
SELECT
    run_id,
    model_name,
    framework,
    compute_target,
    ROUND(gpu_hours::numeric, 2) AS gpu_hours,
    completed_at - started_at AS duration,
    parameters->>'batch_size' AS batch_size,
    ROUND((gpu_hours * 2.50)::numeric, 2) AS estimated_cost_usd,
    notes
FROM training_runs
WHERE gpu_hours > 20
ORDER BY gpu_hours DESC;

-- Query: Models that need review (low accuracy or NULL)
SELECT
    run_id,
    model_name,
    framework,
    status,
    COALESCE(accuracy, 0.0) AS accuracy,
    created_at,
    notes
FROM training_runs
WHERE (accuracy IS NULL OR accuracy < 0.7)
  AND status = 'succeeded'
ORDER BY accuracy ASC NULLS FIRST;

-- Query: Framework performance comparison
SELECT
    framework,
    COUNT(*) AS total_runs,
    COUNT(*) FILTER (WHERE status = 'succeeded') AS successful_runs,
    ROUND(AVG(accuracy) FILTER (WHERE accuracy IS NOT NULL)::numeric, 4) AS avg_accuracy,
    ROUND(AVG(gpu_hours)::numeric, 2) AS avg_gpu_hours,
    ROUND(AVG(EXTRACT(EPOCH FROM (completed_at - started_at)) / 3600)
        FILTER (WHERE completed_at IS NOT NULL)::numeric, 2) AS avg_duration_hours
FROM training_runs
GROUP BY framework
ORDER BY avg_accuracy DESC NULLS LAST;

-- Query: Daily training activity
SELECT
    DATE(created_at) AS date,
    COUNT(*) AS runs_started,
    COUNT(*) FILTER (WHERE status = 'succeeded') AS succeeded,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    ROUND(SUM(gpu_hours)::numeric, 2) AS total_gpu_hours,
    ROUND(SUM(gpu_hours * 2.50)::numeric, 2) AS total_cost_usd
FROM training_runs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Query: Long-running jobs (potential hangs)
SELECT
    run_id,
    model_name,
    framework,
    status,
    started_at,
    AGE(NOW(), started_at) AS running_time,
    compute_target,
    ROUND(gpu_hours::numeric, 2) AS gpu_hours_so_far
FROM training_runs
WHERE status = 'running'
  AND started_at < NOW() - INTERVAL '24 hours'
ORDER BY started_at ASC;

-- Query: Model experiment history
SELECT
    experiment_name,
    COUNT(*) AS total_attempts,
    COUNT(*) FILTER (WHERE status = 'succeeded') AS succeeded,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    ROUND(MAX(accuracy)::numeric, 4) AS best_accuracy,
    ROUND(AVG(accuracy) FILTER (WHERE accuracy IS NOT NULL)::numeric, 4) AS avg_accuracy,
    ROUND(SUM(gpu_hours)::numeric, 2) AS total_gpu_hours
FROM training_runs
WHERE model_name LIKE '%bert%'
GROUP BY experiment_name
ORDER BY best_accuracy DESC NULLS LAST;

-- Query: Cost analysis by framework
SELECT
    framework,
    COUNT(*) AS total_runs,
    ROUND(SUM(gpu_hours)::numeric, 2) AS total_gpu_hours,
    ROUND(SUM(gpu_hours * 2.50)::numeric, 2) AS total_cost_usd,
    ROUND(AVG(gpu_hours * 2.50)::numeric, 2) AS avg_cost_per_run_usd
FROM training_runs
GROUP BY framework
ORDER BY total_cost_usd DESC;
```

✅ **Checkpoint**: You can run complex SELECT queries with filtering, aggregation, and JSONB operations.

---

## Part 5: Update Operations

### Step 5.1: Add Update Examples to CRUD File

Add to `sql/03_crud_operations.sql` (or create `sql/04_update_operations.sql`):

```sql
-- ============================================
-- UPDATE OPERATIONS
-- ============================================

-- ====================
-- BASIC UPDATES
-- ====================

-- Update 1: Start a queued job
BEGIN;

-- Preview what will be updated
SELECT run_id, model_name, status, started_at
FROM training_runs
WHERE status = 'queued'
  AND model_name = 'reinforcement-learning-ppo';

-- Perform update
UPDATE training_runs
SET
    status = 'running',
    started_at = NOW(),
    notes = 'Job picked up by scheduler at ' || NOW()
WHERE status = 'queued'
  AND model_name = 'reinforcement-learning-ppo'
RETURNING run_id, model_name, status, started_at;

COMMIT;

-- Update 2: Complete a successful run
BEGIN;

UPDATE training_runs
SET
    status = 'succeeded',
    completed_at = NOW(),
    accuracy = 0.9456,
    loss = 0.1892,
    precision_score = 0.9423,
    recall_score = 0.9489,
    f1_score = 0.9456,
    gpu_hours = gpu_hours + 18.5,  -- Add final GPU usage
    notes = 'Training completed successfully. Model deployed to production.'
WHERE status = 'running'
  AND model_name = 'llm-fine-tune-mistral'
RETURNING run_id, model_name, status, accuracy, gpu_hours, completed_at;

COMMIT;

-- Update 3: Mark a running job as failed
BEGIN;

UPDATE training_runs
SET
    status = 'failed',
    completed_at = NOW(),
    notes = 'Out of memory error. Killed by scheduler. Try reducing batch size to 128.'
WHERE status = 'running'
  AND model_name = 'multimodal-clip-training'
RETURNING run_id, model_name, status, completed_at, notes;

COMMIT;

-- Update 4: Update JSON parameters (merge)
BEGIN;

-- Add tuning metadata to parameters
UPDATE training_runs
SET
    parameters = parameters || '{"tuned": true, "final_lr": 0.00001, "updated_at": "2025-11-01"}'::jsonb
WHERE model_name = 'bert-sentiment-analysis'
  AND status = 'succeeded'
RETURNING model_name, parameters;

COMMIT;

-- Update 5: Bulk update - tag old runs
BEGIN;

UPDATE training_runs
SET
    created_by = 'legacy_system',
    parameters = parameters || '{"legacy": true}'::jsonb
WHERE created_by = 'mlops'
  AND created_at < NOW() - INTERVAL '10 days'
RETURNING model_name, created_by, created_at;

COMMIT;

-- Update 6: Calculate F1 score from precision/recall
BEGIN;

-- Preview calculation
SELECT
    model_name,
    precision_score,
    recall_score,
    f1_score AS current_f1,
    2.0 * (precision_score * recall_score) / NULLIF((precision_score + recall_score), 0) AS calculated_f1
FROM training_runs
WHERE precision_score IS NOT NULL
  AND recall_score IS NOT NULL
  AND f1_score IS NULL;

-- Update
UPDATE training_runs
SET
    f1_score = 2.0 * (precision_score * recall_score) / (precision_score + recall_score)
WHERE precision_score IS NOT NULL
  AND recall_score IS NOT NULL
  AND f1_score IS NULL
  AND (precision_score + recall_score) > 0  -- Avoid division by zero
RETURNING model_name, precision_score, recall_score, f1_score;

COMMIT;

-- Update 7: Increment GPU hours for running job
BEGIN;

UPDATE training_runs
SET
    gpu_hours = gpu_hours + 2.5,
    parameters = parameters || jsonb_build_object(
        'last_updated', NOW()::text,
        'total_updates', COALESCE((parameters->>'total_updates')::int, 0) + 1
    )
WHERE status = 'running'
  AND run_id = (SELECT run_id FROM training_runs WHERE status = 'running' LIMIT 1)
RETURNING model_name, gpu_hours, parameters->>'total_updates' AS update_count;

COMMIT;

-- ====================
-- CONDITIONAL UPDATES
-- ====================

-- Update 8: Auto-cancel long-queued jobs
BEGIN;

-- Preview
SELECT
    run_id,
    model_name,
    status,
    created_at,
    AGE(NOW(), created_at) AS queue_time
FROM training_runs
WHERE status = 'queued'
  AND created_at < NOW() - INTERVAL '1 hour';

-- Update
UPDATE training_runs
SET
    status = 'cancelled',
    completed_at = NOW(),
    notes = 'Auto-cancelled: queued for over 1 hour without starting. Resource capacity exceeded.'
WHERE status = 'queued'
  AND created_at < NOW() - INTERVAL '1 hour'
RETURNING run_id, model_name, status, completed_at, notes;

COMMIT;

-- Update 9: Fix data quality issues
BEGIN;

-- Normalize framework names (if inconsistent data exists)
UPDATE training_runs
SET framework = LOWER(framework)
WHERE framework != LOWER(framework);

-- Ensure completed runs have completion timestamp
UPDATE training_runs
SET completed_at = started_at + INTERVAL '1 hour'
WHERE status IN ('succeeded', 'failed', 'cancelled', 'timeout')
  AND completed_at IS NULL
  AND started_at IS NOT NULL
RETURNING run_id, model_name, status, started_at, completed_at;

COMMIT;

-- Update 10: Add cost estimates
BEGIN;

-- Add cost field to parameters based on GPU hours
UPDATE training_runs
SET parameters = parameters || jsonb_build_object(
    'estimated_cost_usd', ROUND((gpu_hours * 2.50)::numeric, 2),
    'cost_calculated_at', NOW()::text
)
WHERE gpu_hours > 0
  AND NOT (parameters ? 'estimated_cost_usd')
RETURNING
    model_name,
    gpu_hours,
    parameters->>'estimated_cost_usd' AS cost_usd;

COMMIT;
```

### Step 5.2: Safe Update Practices

Create `sql/05_safe_update_patterns.sql`:

```sql
-- ============================================
-- SAFE UPDATE PATTERNS
-- ============================================

-- Pattern 1: Always use transactions
-- Pattern 2: Preview before update
-- Pattern 3: Use RETURNING to verify
-- Pattern 4: Add WHERE clause validation

-- Example: Safe batch update
BEGIN;

-- Step 1: Count affected rows
SELECT COUNT(*) AS will_update
FROM training_runs
WHERE status = 'failed'
  AND created_at < NOW() - INTERVAL '90 days';

-- Step 2: Preview what will change
SELECT
    run_id,
    model_name,
    status,
    created_at,
    'WILL BE DELETED' AS action
FROM training_runs
WHERE status = 'failed'
  AND created_at < NOW() - INTERVAL '90 days'
LIMIT 5;

-- Step 3: If preview looks good, proceed
-- (In production, you might want to add a manual confirmation step)

-- Step 4: Perform the update (or delete)
UPDATE training_runs
SET
    parameters = parameters || '{"archived": true, "archived_at": "' || NOW() || '"}'::jsonb,
    notes = COALESCE(notes || ' | ', '') || 'Archived: old failed run'
WHERE status = 'failed'
  AND created_at < NOW() - INTERVAL '90 days'
RETURNING run_id, model_name, status, created_at;

-- Step 5: Verify count matches expectation
SELECT COUNT(*) AS updated
FROM training_runs
WHERE parameters @> '{"archived": true}'::jsonb;

-- Step 6: If satisfied, commit. Otherwise rollback.
-- COMMIT;
ROLLBACK;  -- Use this to undo if something went wrong

-- Pattern 2: Update with validation
BEGIN;

-- Update but only if constraints will be met
UPDATE training_runs
SET
    accuracy = 0.95,
    status = 'succeeded',
    completed_at = NOW()
WHERE run_id = 'SOME-UUID'
  AND status = 'running'  -- Must be running
  AND started_at IS NOT NULL  -- Must have started
  AND completed_at IS NULL  -- Must not be already completed
RETURNING *;

COMMIT;

-- Pattern 3: Conditional update with CASE
BEGIN;

UPDATE training_runs
SET gpu_hours = CASE
    WHEN compute_target LIKE '%a100%' THEN gpu_hours * 1.0  -- No change
    WHEN compute_target LIKE '%v100%' THEN gpu_hours * 0.8  -- Adjust for V100
    WHEN compute_target LIKE '%t4%' THEN gpu_hours * 0.5   -- Adjust for T4
    ELSE gpu_hours
END
WHERE gpu_hours > 0
RETURNING model_name, compute_target, gpu_hours;

ROLLBACK;  -- This was just an example

-- Pattern 4: Update with subquery
BEGIN;

-- Update runs to mark them as "above average"
UPDATE training_runs t1
SET parameters = parameters || '{"performance": "above_average"}'::jsonb
WHERE accuracy > (
    SELECT AVG(accuracy)
    FROM training_runs
    WHERE accuracy IS NOT NULL
)
AND accuracy IS NOT NULL
RETURNING model_name, accuracy, parameters->>'performance';

ROLLBACK;
```

✅ **Checkpoint**: You can safely update records with WHERE clauses, transactions, and validation.

---

## Part 6: Delete Operations

### Step 6.1: Add Delete Examples

Create `sql/06_delete_operations.sql`:

```sql
-- ============================================
-- DELETE OPERATIONS
-- ============================================

-- WARNING: DELETE is irreversible (unless in transaction)
-- Always use WHERE clause!
-- Always use transactions for safety!

-- ====================
-- SAFE DELETE PATTERNS
-- ====================

-- Delete 1: Remove old test runs
BEGIN;

-- Count before delete
SELECT COUNT(*) AS before_count
FROM training_runs
WHERE notes ILIKE '%test%'
  AND created_at < NOW() - INTERVAL '120 days';

-- Preview what will be deleted
SELECT
    run_id,
    model_name,
    created_at,
    notes
FROM training_runs
WHERE notes ILIKE '%test%'
  AND created_at < NOW() - INTERVAL '120 days'
LIMIT 5;

-- Perform delete (returns deleted rows)
DELETE FROM training_runs
WHERE notes ILIKE '%test%'
  AND created_at < NOW() - INTERVAL '120 days'
  AND status IN ('failed', 'cancelled')  -- Additional safety
RETURNING run_id, model_name, created_at, status;

-- Verify deletion
SELECT COUNT(*) AS after_count
FROM training_runs;

-- If satisfied, commit. Otherwise rollback.
-- COMMIT;
ROLLBACK;  -- Safe to rollback since this is a demo

-- Delete 2: Remove cancelled runs older than 90 days
BEGIN;

-- Count
SELECT COUNT(*)
FROM training_runs
WHERE status = 'cancelled'
  AND created_at < NOW() - INTERVAL '90 days';

-- Delete with RETURNING
DELETE FROM training_runs
WHERE status = 'cancelled'
  AND created_at < NOW() - INTERVAL '90 days'
RETURNING run_id, model_name, created_at, status;

ROLLBACK;

-- Delete 3: Remove specific run by ID
BEGIN;

-- First, view the run
SELECT *
FROM training_runs
WHERE run_id = 'REPLACE-WITH-ACTUAL-UUID'
LIMIT 1;

-- Delete
DELETE FROM training_runs
WHERE run_id = 'REPLACE-WITH-ACTUAL-UUID'
RETURNING *;

ROLLBACK;

-- Delete 4: Remove failed runs with specific error pattern
BEGIN;

-- Find runs with OOM errors
SELECT
    run_id,
    model_name,
    notes
FROM training_runs
WHERE status = 'failed'
  AND notes ILIKE '%out of memory%'
  OR notes ILIKE '%oom%';

-- Delete them
DELETE FROM training_runs
WHERE status = 'failed'
  AND (notes ILIKE '%out of memory%' OR notes ILIKE '%oom%')
  AND created_at < NOW() - INTERVAL '30 days'
RETURNING run_id, model_name, notes;

ROLLBACK;

-- ====================
-- DATA RETENTION POLICY
-- ====================

-- Delete 5: Implement retention policy
BEGIN;

-- Archive old data first (in production, export to S3/GCS)
-- Then delete

-- Delete runs older than 1 year (except successful ones)
DELETE FROM training_runs
WHERE created_at < NOW() - INTERVAL '1 year'
  AND status != 'succeeded'
RETURNING run_id, model_name, status, created_at;

-- Delete successful runs older than 2 years
DELETE FROM training_runs
WHERE created_at < NOW() - INTERVAL '2 years'
  AND status = 'succeeded'
RETURNING run_id, model_name, status, created_at;

ROLLBACK;

-- ====================
-- DELETE vs TRUNCATE
-- ====================

-- DELETE: Removes specific rows, can use WHERE, slower, can rollback
DELETE FROM training_runs WHERE status = 'failed';

-- TRUNCATE: Removes ALL rows, faster, resets sequences, harder to rollback
-- Use with EXTREME caution!
-- TRUNCATE TABLE training_runs;  -- DON'T RUN THIS!

-- TRUNCATE with CASCADE (drops dependent data in other tables)
-- TRUNCATE TABLE training_runs CASCADE;  -- VERY DANGEROUS!

-- ====================
-- SOFT DELETE PATTERN
-- ====================

-- Instead of DELETE, mark as deleted (soft delete)
-- Add column: deleted_at TIMESTAMP
-- Add column: is_deleted BOOLEAN DEFAULT FALSE

-- Example soft delete (requires schema change first):
/*
ALTER TABLE training_runs
ADD COLUMN deleted_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE;

-- Soft delete
UPDATE training_runs
SET
    is_deleted = TRUE,
    deleted_at = NOW()
WHERE status = 'failed'
  AND created_at < NOW() - INTERVAL '90 days';

-- Query only non-deleted
SELECT * FROM training_runs WHERE is_deleted = FALSE;

-- Hard delete (actually remove soft-deleted records)
DELETE FROM training_runs WHERE is_deleted = TRUE AND deleted_at < NOW() - INTERVAL '1 year';
*/
```

### Step 6.2: Create Cleanup Script

Create `scripts/cleanup_old_runs.sh`:

```bash
#!/bin/bash
# Cleanup script for old training runs

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== ML Training Registry Cleanup ===${NC}"
echo ""

# Count before cleanup
echo -e "${GREEN}Counting records before cleanup...${NC}"
BEFORE_COUNT=$(docker exec pg-ml-training psql -U mlops -d ml_infra -t -c "SELECT COUNT(*) FROM training_runs;")
echo "Total runs: $BEFORE_COUNT"

# Show what will be deleted
echo ""
echo -e "${YELLOW}Records to be deleted:${NC}"
docker exec pg-ml-training psql -U mlops -d ml_infra -c "
SELECT
    status,
    COUNT(*) AS count
FROM training_runs
WHERE created_at < NOW() - INTERVAL '90 days'
  AND status IN ('failed', 'cancelled', 'timeout')
GROUP BY status;
"

# Ask for confirmation
echo ""
read -p "$(echo -e ${RED}Do you want to proceed with deletion? [y/N]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${YELLOW}Cleanup cancelled.${NC}"
    exit 0
fi

# Perform cleanup
echo ""
echo -e "${GREEN}Performing cleanup...${NC}"
docker exec pg-ml-training psql -U mlops -d ml_infra -c "
BEGIN;

DELETE FROM training_runs
WHERE created_at < NOW() - INTERVAL '90 days'
  AND status IN ('failed', 'cancelled', 'timeout');

COMMIT;
"

# Count after cleanup
AFTER_COUNT=$(docker exec pg-ml-training psql -U mlops -d ml_infra -t -c "SELECT COUNT(*) FROM training_runs;")
DELETED=$((BEFORE_COUNT - AFTER_COUNT))

echo ""
echo -e "${GREEN}Cleanup complete!${NC}"
echo "Records deleted: $DELETED"
echo "Records remaining: $AFTER_COUNT"
```

```bash
chmod +x scripts/cleanup_old_runs.sh
```

✅ **Checkpoint**: You understand DELETE operations and safety practices.

---

## Part 7: Transactions

### Step 7.1: Transaction Examples

Create `sql/07_transactions.sql`:

```sql
-- ============================================
-- TRANSACTION EXAMPLES
-- ============================================

-- ====================
-- BASIC TRANSACTIONS
-- ====================

-- Transaction basics
-- BEGIN: Start transaction
-- COMMIT: Make changes permanent
-- ROLLBACK: Undo all changes since BEGIN

-- Example 1: Simple transaction
BEGIN;
    INSERT INTO training_runs (model_name, framework, experiment_name, dataset, compute_target)
    VALUES ('test-model', 'pytorch', 'test-exp', 'test-data', 'test-compute');

    SELECT * FROM training_runs WHERE model_name = 'test-model';
ROLLBACK;  -- Undo the insert

-- Verify it was rolled back
SELECT * FROM training_runs WHERE model_name = 'test-model';  -- Should return 0 rows

-- Example 2: Multi-step transaction
BEGIN;
    -- Step 1: Insert new run
    INSERT INTO training_runs (
        model_name, framework, experiment_name,
        status, dataset, compute_target
    )
    VALUES (
        'transactional-test', 'pytorch', 'tx-test',
        'queued', 'test-dataset', 'test-target'
    )
    RETURNING run_id;

    -- Step 2: Update it to running
    UPDATE training_runs
    SET status = 'running', started_at = NOW()
    WHERE model_name = 'transactional-test';

    -- Step 3: Update it to succeeded
    UPDATE training_runs
    SET
        status = 'succeeded',
        completed_at = NOW(),
        accuracy = 0.95,
        gpu_hours = 10.5
    WHERE model_name = 'transactional-test';

    -- Verify all steps
    SELECT * FROM training_runs WHERE model_name = 'transactional-test';
COMMIT;  -- Make it permanent

-- Clean up
DELETE FROM training_runs WHERE model_name = 'transactional-test';

-- ====================
-- TRANSACTION ISOLATION
-- ====================

-- Check current isolation level
SHOW transaction_isolation;

-- Set isolation level (for this transaction only)
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
    SELECT * FROM training_runs WHERE status = 'queued';
COMMIT;

-- ====================
-- SAVEPOINTS
-- ====================

-- Savepoints allow partial rollback within a transaction
BEGIN;
    -- Create a test run
    INSERT INTO training_runs (
        model_name, framework, experiment_name,
        status, dataset, compute_target
    )
    VALUES (
        'savepoint-test-1', 'pytorch', 'sp-test',
        'queued', 'test', 'test'
    );

    SAVEPOINT after_first_insert;

    -- Create another test run
    INSERT INTO training_runs (
        model_name, framework, experiment_name,
        status, dataset, compute_target
    )
    VALUES (
        'savepoint-test-2', 'tensorflow', 'sp-test',
        'queued', 'test', 'test'
    );

    -- Oops, we only want the first one
    ROLLBACK TO SAVEPOINT after_first_insert;

    -- Verify: only first insert remains
    SELECT model_name FROM training_runs WHERE model_name LIKE 'savepoint-test%';

ROLLBACK;  -- Clean up

-- ====================
-- PRACTICAL TRANSACTION: Job Lifecycle
-- ====================

-- Scenario: Move a job through its lifecycle atomically
BEGIN;

-- Variables (in real app, these would be parameters)
DO $$
DECLARE
    job_run_id UUID;
    job_model TEXT := 'lifecycle-demo';
BEGIN
    -- Step 1: Create queued job
    INSERT INTO training_runs (
        model_name, framework, experiment_name,
        status, dataset, compute_target
    )
    VALUES (
        job_model, 'pytorch', 'lifecycle-test',
        'queued', 'demo-data', 'k8s-gpu'
    )
    RETURNING run_id INTO job_run_id;

    RAISE NOTICE 'Created job with ID: %', job_run_id;

    -- Step 2: Start the job
    PERFORM pg_sleep(1);  -- Simulate delay

    UPDATE training_runs
    SET
        status = 'running',
        started_at = NOW()
    WHERE run_id = job_run_id;

    RAISE NOTICE 'Job started at: %', NOW();

    -- Step 3: Complete the job
    PERFORM pg_sleep(2);  -- Simulate training time

    UPDATE training_runs
    SET
        status = 'succeeded',
        completed_at = NOW(),
        accuracy = 0.92,
        loss = 0.15,
        gpu_hours = 5.5,
        notes = 'Lifecycle test completed successfully'
    WHERE run_id = job_run_id;

    RAISE NOTICE 'Job completed at: %', NOW();

    -- Verify final state
    PERFORM model_name, status, accuracy, gpu_hours
    FROM training_runs
    WHERE run_id = job_run_id;
END $$;

ROLLBACK;  -- Clean up demo data

-- ====================
-- ERROR HANDLING
-- ====================

-- Transaction with error handling
BEGIN;
    -- This will fail due to constraint violation
    INSERT INTO training_runs (
        model_name, framework, experiment_name,
        status, dataset, compute_target, accuracy
    )
    VALUES (
        'error-test', 'invalid-framework',  -- Invalid!
        'error-exp', 'queued', 'test', 'test', 0.95
    );
    -- Transaction will be aborted here
ROLLBACK;

-- Safe pattern with DO block
DO $$
BEGIN
    BEGIN
        INSERT INTO training_runs (
            model_name, framework, experiment_name,
            status, dataset, compute_target
        )
        VALUES (
            'error-safe', 'pytorch', 'safe-exp',
            'queued', 'test', 'test'
        );
    EXCEPTION
        WHEN others THEN
            RAISE NOTICE 'Error occurred: %', SQLERRM;
    END;
END $$;
```

✅ **Checkpoint**: You understand transactions and can use BEGIN/COMMIT/ROLLBACK.

---

## Part 8: Advanced Queries

### Step 8.1: Add Advanced Query Examples

Create `sql/08_advanced_queries.sql`:

```sql
-- ============================================
-- ADVANCED QUERIES
-- ============================================

-- ====================
-- SUBQUERIES
-- ====================

-- Subquery in WHERE clause
SELECT
    model_name,
    accuracy,
    (SELECT AVG(accuracy) FROM training_runs WHERE accuracy IS NOT NULL) AS avg_accuracy
FROM training_runs
WHERE accuracy > (
    SELECT AVG(accuracy)
    FROM training_runs
    WHERE accuracy IS NOT NULL
)
AND accuracy IS NOT NULL
ORDER BY accuracy DESC;

-- Subquery in FROM clause (derived table)
SELECT
    framework,
    ROUND(avg_accuracy::numeric, 4) AS avg_accuracy
FROM (
    SELECT
        framework,
        AVG(accuracy) AS avg_accuracy
    FROM training_runs
    WHERE accuracy IS NOT NULL
    GROUP BY framework
) AS framework_stats
WHERE avg_accuracy > 0.85
ORDER BY avg_accuracy DESC;

-- Correlated subquery
SELECT
    t1.model_name,
    t1.accuracy,
    t1.framework,
    (
        SELECT COUNT(*)
        FROM training_runs t2
        WHERE t2.framework = t1.framework
          AND t2.status = 'succeeded'
    ) AS successful_runs_in_framework
FROM training_runs t1
WHERE t1.status = 'succeeded'
ORDER BY t1.accuracy DESC NULLS LAST;

-- ====================
-- COMMON TABLE EXPRESSIONS (CTEs)
-- ====================

-- Single CTE
WITH successful_runs AS (
    SELECT *
    FROM training_runs
    WHERE status = 'succeeded'
      AND accuracy IS NOT NULL
)
SELECT
    framework,
    COUNT(*) AS runs,
    ROUND(AVG(accuracy)::numeric, 4) AS avg_accuracy
FROM successful_runs
GROUP BY framework
ORDER BY avg_accuracy DESC;

-- Multiple CTEs
WITH
pytorch_runs AS (
    SELECT * FROM training_runs WHERE framework = 'pytorch'
),
tensorflow_runs AS (
    SELECT * FROM training_runs WHERE framework = 'tensorflow'
),
framework_comparison AS (
    SELECT
        'PyTorch' AS framework,
        COUNT(*) AS total_runs,
        COUNT(*) FILTER (WHERE status = 'succeeded') AS successful,
        ROUND(AVG(accuracy)::numeric, 4) AS avg_accuracy
    FROM pytorch_runs
    WHERE accuracy IS NOT NULL

    UNION ALL

    SELECT
        'TensorFlow' AS framework,
        COUNT(*) AS total_runs,
        COUNT(*) FILTER (WHERE status = 'succeeded') AS successful,
        ROUND(AVG(accuracy)::numeric, 4) AS avg_accuracy
    FROM tensorflow_runs
    WHERE accuracy IS NOT NULL
)
SELECT * FROM framework_comparison
ORDER BY avg_accuracy DESC;

-- Recursive CTE (example: generate date series)
WITH RECURSIVE date_series AS (
    SELECT
        (NOW() - INTERVAL '30 days')::date AS date
    UNION ALL
    SELECT
        (date + INTERVAL '1 day')::date
    FROM date_series
    WHERE date < CURRENT_DATE
)
SELECT
    ds.date,
    COUNT(tr.run_id) AS runs_on_date,
    COALESCE(SUM(tr.gpu_hours), 0) AS gpu_hours
FROM date_series ds
LEFT JOIN training_runs tr ON DATE(tr.created_at) = ds.date
GROUP BY ds.date
ORDER BY ds.date DESC;

-- ====================
-- WINDOW FUNCTIONS
-- ====================

-- Rank models by accuracy within each framework
SELECT
    model_name,
    framework,
    ROUND(accuracy::numeric, 4) AS accuracy,
    RANK() OVER (PARTITION BY framework ORDER BY accuracy DESC NULLS LAST) AS rank_in_framework,
    ROUND(AVG(accuracy) OVER (PARTITION BY framework)::numeric, 4) AS framework_avg_accuracy
FROM training_runs
WHERE status = 'succeeded'
  AND accuracy IS NOT NULL
ORDER BY framework, rank_in_framework;

-- Running total of GPU hours
SELECT
    model_name,
    created_at,
    ROUND(gpu_hours::numeric, 2) AS gpu_hours,
    ROUND(SUM(gpu_hours) OVER (ORDER BY created_at)::numeric, 2) AS cumulative_gpu_hours,
    ROUND(AVG(gpu_hours) OVER (ORDER BY created_at ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)::numeric, 2) AS moving_avg_5
FROM training_runs
WHERE gpu_hours > 0
ORDER BY created_at;

-- Row number and lead/lag
SELECT
    ROW_NUMBER() OVER (ORDER BY accuracy DESC NULLS LAST) AS rank,
    model_name,
    ROUND(accuracy::numeric, 4) AS accuracy,
    ROUND(LAG(accuracy) OVER (ORDER BY accuracy DESC NULLS LAST)::numeric, 4) AS previous_accuracy,
    ROUND((accuracy - LAG(accuracy) OVER (ORDER BY accuracy DESC NULLS LAST))::numeric, 4) AS accuracy_diff
FROM training_runs
WHERE status = 'succeeded'
  AND accuracy IS NOT NULL
LIMIT 10;

-- ====================
-- CASE STATEMENTS
-- ====================

-- Categorize models by performance
SELECT
    model_name,
    ROUND(accuracy::numeric, 4) AS accuracy,
    CASE
        WHEN accuracy >= 0.95 THEN 'Excellent'
        WHEN accuracy >= 0.90 THEN 'Good'
        WHEN accuracy >= 0.80 THEN 'Fair'
        WHEN accuracy < 0.80 THEN 'Poor'
        ELSE 'Unknown'
    END AS performance_category,
    CASE
        WHEN gpu_hours = 0 THEN 'CPU Only'
        WHEN gpu_hours < 10 THEN 'Light GPU'
        WHEN gpu_hours < 50 THEN 'Moderate GPU'
        ELSE 'Heavy GPU'
    END AS resource_category,
    CASE
        WHEN gpu_hours > 30 THEN '$$$ High Cost'
        WHEN gpu_hours > 10 THEN '$$ Medium Cost'
        ELSE '$ Low Cost'
    END AS cost_estimate
FROM training_runs
WHERE accuracy IS NOT NULL
ORDER BY accuracy DESC;

-- ====================
-- UNION / UNION ALL
-- ====================

-- Combine recent successes and failures
(
    SELECT
        'SUCCESS' AS category,
        model_name,
        framework,
        accuracy,
        created_at
    FROM training_runs
    WHERE status = 'succeeded'
    ORDER BY created_at DESC
    LIMIT 5
)
UNION ALL
(
    SELECT
        'FAILURE' AS category,
        model_name,
        framework,
        NULL AS accuracy,
        created_at
    FROM training_runs
    WHERE status = 'failed'
    ORDER BY created_at DESC
    LIMIT 5
)
ORDER BY created_at DESC;

-- ====================
-- LATERAL JOINS
-- ====================

-- For each framework, get the top 2 most accurate models
SELECT
    f.framework,
    t.model_name,
    ROUND(t.accuracy::numeric, 4) AS accuracy
FROM (
    SELECT DISTINCT framework FROM training_runs
) f
CROSS JOIN LATERAL (
    SELECT model_name, accuracy
    FROM training_runs
    WHERE framework = f.framework
      AND status = 'succeeded'
      AND accuracy IS NOT NULL
    ORDER BY accuracy DESC
    LIMIT 2
) t
ORDER BY f.framework, t.accuracy DESC;

-- ====================
-- GROUPING SETS / ROLLUP / CUBE
-- ====================

-- Aggregate at multiple levels
SELECT
    framework,
    status,
    COUNT(*) AS runs,
    ROUND(AVG(gpu_hours)::numeric, 2) AS avg_gpu_hours
FROM training_runs
GROUP BY GROUPING SETS (
    (framework, status),
    (framework),
    (status),
    ()
)
ORDER BY framework NULLS LAST, status NULLS LAST;

-- ROLLUP (hierarchical aggregation)
SELECT
    framework,
    status,
    COUNT(*) AS runs
FROM training_runs
GROUP BY ROLLUP (framework, status)
ORDER BY framework NULLS LAST, status NULLS LAST;
```

✅ **Checkpoint**: You can write advanced SQL with CTEs, window functions, and subqueries.

---

## Part 9: Challenge Solutions

### Challenge 1: Data Analysis

```sql
-- ============================================
-- CHALLENGE 1 SOLUTIONS
-- ============================================

-- Question 1: What is the success rate for each framework?
SELECT
    framework,
    COUNT(*) AS total_runs,
    COUNT(*) FILTER (WHERE status = 'succeeded') AS successful_runs,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_runs,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'succeeded') / NULLIF(COUNT(*), 0), 2) AS success_rate_pct
FROM training_runs
GROUP BY framework
ORDER BY success_rate_pct DESC;

-- Question 2: Which compute target has the highest average GPU hours?
SELECT
    compute_target,
    COUNT(*) AS total_runs,
    ROUND(AVG(gpu_hours)::numeric, 2) AS avg_gpu_hours,
    ROUND(MAX(gpu_hours)::numeric, 2) AS max_gpu_hours,
    ROUND(SUM(gpu_hours)::numeric, 2) AS total_gpu_hours
FROM training_runs
WHERE gpu_hours > 0
GROUP BY compute_target
ORDER BY avg_gpu_hours DESC
LIMIT 1;

-- Question 3: Find all models that ran longer than 1 day
SELECT
    model_name,
    framework,
    status,
    started_at,
    completed_at,
    completed_at - started_at AS duration,
    ROUND(EXTRACT(EPOCH FROM (completed_at - started_at)) / 3600, 2) AS duration_hours
FROM training_runs
WHERE completed_at IS NOT NULL
  AND started_at IS NOT NULL
  AND (completed_at - started_at) > INTERVAL '1 day'
ORDER BY duration DESC;

-- Question 4: Calculate the total cost if GPU hours cost $2.50/hour
SELECT
    SUM(gpu_hours) AS total_gpu_hours,
    ROUND((SUM(gpu_hours) * 2.50)::numeric, 2) AS total_cost_usd,
    COUNT(*) AS total_runs,
    ROUND((SUM(gpu_hours) * 2.50 / COUNT(*))::numeric, 2) AS avg_cost_per_run_usd
FROM training_runs;

-- Breakdown by framework:
SELECT
    framework,
    COUNT(*) AS runs,
    ROUND(SUM(gpu_hours)::numeric, 2) AS total_gpu_hours,
    ROUND((SUM(gpu_hours) * 2.50)::numeric, 2) AS total_cost_usd,
    ROUND((AVG(gpu_hours) * 2.50)::numeric, 2) AS avg_cost_per_run_usd
FROM training_runs
GROUP BY framework
ORDER BY total_cost_usd DESC;

-- Question 5: Find experiments with multiple attempts
SELECT
    experiment_name,
    COUNT(*) AS total_attempts,
    COUNT(*) FILTER (WHERE status = 'succeeded') AS succeeded,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    COUNT(*) FILTER (WHERE status IN ('queued', 'running')) AS in_progress,
    ARRAY_AGG(model_name ORDER BY created_at) AS models,
    MIN(created_at) AS first_attempt,
    MAX(created_at) AS last_attempt
FROM training_runs
GROUP BY experiment_name
HAVING COUNT(*) > 1
ORDER BY total_attempts DESC;
```

### Challenge 2: Data Manipulation

```sql
-- ============================================
-- CHALLENGE 2 SOLUTIONS
-- ============================================

-- Question 1: Insert a new training run for your favorite ML model
BEGIN;

INSERT INTO training_runs (
    model_name,
    framework,
    experiment_name,
    status,
    dataset,
    compute_target,
    parameters,
    notes
)
VALUES (
    'stable-diffusion-xl',
    'pytorch',
    'exp-sdxl-finetune-2025',
    'queued',
    'laion-aesthetics-6.5',
    'k8s-gpu-a100',
    '{
        "learning_rate": 0.0001,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "resolution": 1024,
        "use_lora": true,
        "lora_rank": 64
    }'::jsonb,
    'Stable Diffusion XL fine-tuning with LoRA for custom style'
)
RETURNING *;

COMMIT;

-- Question 2: Update it from queued → running → succeeded
BEGIN;

-- Get the run_id
SELECT run_id, model_name, status
FROM training_runs
WHERE model_name = 'stable-diffusion-xl'
  AND experiment_name = 'exp-sdxl-finetune-2025';

-- Update to running
UPDATE training_runs
SET
    status = 'running',
    started_at = NOW(),
    notes = notes || ' | Started training at ' || NOW()
WHERE model_name = 'stable-diffusion-xl'
  AND experiment_name = 'exp-sdxl-finetune-2025'
  AND status = 'queued'
RETURNING run_id, model_name, status, started_at;

-- Simulate some time passing (in production, this would be hours later)
-- Update to succeeded
UPDATE training_runs
SET
    status = 'succeeded',
    completed_at = NOW() + INTERVAL '4 hours',  -- Simulating 4-hour training
    accuracy = NULL,  -- Not applicable for diffusion models
    loss = 0.0245,
    gpu_hours = 16.8,
    cpu_hours = 2.1,
    parameters = parameters || '{"final_loss": 0.0245, "total_steps": 50000}'::jsonb,
    notes = notes || ' | Training completed successfully. Model ready for inference.'
WHERE model_name = 'stable-diffusion-xl'
  AND experiment_name = 'exp-sdxl-finetune-2025'
  AND status = 'running'
RETURNING run_id, model_name, status, completed_at, loss, gpu_hours;

COMMIT;

-- Question 3: Calculate and update F1 score from precision/recall
BEGIN;

-- For runs that have precision and recall but missing F1
UPDATE training_runs
SET f1_score = 2.0 * (precision_score * recall_score) / (precision_score + recall_score)
WHERE precision_score IS NOT NULL
  AND recall_score IS NOT NULL
  AND f1_score IS NULL
  AND (precision_score + recall_score) > 0
RETURNING
    model_name,
    ROUND(precision_score::numeric, 4) AS precision,
    ROUND(recall_score::numeric, 4) AS recall,
    ROUND(f1_score::numeric, 4) AS f1;

COMMIT;

-- Question 4: Export the final result to CSV
\copy (SELECT model_name, framework, experiment_name, status, accuracy, loss, precision_score, recall_score, f1_score, gpu_hours, created_at, completed_at FROM training_runs WHERE model_name = 'stable-diffusion-xl') TO '/tmp/sdxl_training_result.csv' WITH CSV HEADER;
```

### Challenge 3: Cleanup

```sql
-- ============================================
-- CHALLENGE 3 SOLUTIONS
-- ============================================

-- Question 1: Find all runs older than 30 days with status 'queued'
BEGIN;

-- Preview
SELECT
    run_id,
    model_name,
    status,
    created_at,
    AGE(NOW(), created_at) AS age
FROM training_runs
WHERE status = 'queued'
  AND created_at < NOW() - INTERVAL '30 days'
ORDER BY created_at ASC;

-- Count
SELECT COUNT(*) AS stale_queued_runs
FROM training_runs
WHERE status = 'queued'
  AND created_at < NOW() - INTERVAL '30 days';

ROLLBACK;

-- Question 2: Update them to 'cancelled' with appropriate notes
BEGIN;

UPDATE training_runs
SET
    status = 'cancelled',
    completed_at = NOW(),
    notes = COALESCE(notes || ' | ', '') || 'Auto-cancelled: queued for over 30 days without starting. Likely stuck in queue or forgotten.'
WHERE status = 'queued'
  AND created_at < NOW() - INTERVAL '30 days'
RETURNING
    run_id,
    model_name,
    created_at,
    completed_at,
    AGE(completed_at, created_at) AS time_in_queue,
    notes;

-- Verify
SELECT COUNT(*) AS cancelled_count
FROM training_runs
WHERE status = 'cancelled'
  AND notes LIKE '%Auto-cancelled%';

COMMIT;

-- Question 3: Delete any test runs (notes contain 'test')
BEGIN;

-- Preview
SELECT
    run_id,
    model_name,
    status,
    created_at,
    notes
FROM training_runs
WHERE notes ILIKE '%test%'
   OR model_name ILIKE '%test%'
ORDER BY created_at DESC;

-- Count
SELECT COUNT(*) AS test_runs
FROM training_runs
WHERE notes ILIKE '%test%'
   OR model_name ILIKE '%test%';

-- Delete
DELETE FROM training_runs
WHERE notes ILIKE '%test%'
   OR model_name ILIKE '%test%'
RETURNING run_id, model_name, created_at, notes;

-- Verify
SELECT COUNT(*) AS remaining_test_runs
FROM training_runs
WHERE notes ILIKE '%test%'
   OR model_name ILIKE '%test%';

COMMIT;
```

✅ **Checkpoint**: All challenges completed successfully!

---

## Verification & Testing

### Step 1: Run All SQL Scripts

```bash
# Create verification script
cat > scripts/verify_all.sh << 'EOF'
#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Verifying ML Training Registry ===${NC}"

# Test 1: Table exists
echo -e "\n${YELLOW}Test 1: Checking if training_runs table exists...${NC}"
docker exec pg-ml-training psql -U mlops -d ml_infra -c "\dt training_runs"

# Test 2: Row count
echo -e "\n${YELLOW}Test 2: Counting total rows...${NC}"
docker exec pg-ml-training psql -U mlops -d ml_infra -c "SELECT COUNT(*) AS total_rows FROM training_runs;"

# Test 3: Status distribution
echo -e "\n${YELLOW}Test 3: Status distribution...${NC}"
docker exec pg-ml-training psql -U mlops -d ml_infra -c "SELECT status, COUNT(*) AS count FROM training_runs GROUP BY status ORDER BY count DESC;"

# Test 4: Check constraints
echo -e "\n${YELLOW}Test 4: Verifying constraints...${NC}"
docker exec pg-ml-training psql -U mlops -d ml_infra -c "SELECT conname FROM pg_constraint WHERE conrelid = 'training_runs'::regclass ORDER BY conname;"

# Test 5: Check indexes
echo -e "\n${YELLOW}Test 5: Verifying indexes...${NC}"
docker exec pg-ml-training psql -U mlops -d ml_infra -c "SELECT indexname FROM pg_indexes WHERE tablename = 'training_runs' ORDER BY indexname;"

# Test 6: Data integrity
echo -e "\n${YELLOW}Test 6: Data integrity checks...${NC}"
docker exec pg-ml-training psql -U mlops -d ml_infra -c "
SELECT
    'All accuracies in range' AS check_name,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM training_runs
WHERE accuracy IS NOT NULL AND (accuracy < 0 OR accuracy > 1);
"

echo -e "\n${GREEN}Verification complete!${NC}"
EOF

chmod +x scripts/verify_all.sh
./scripts/verify_all.sh
```

### Step 2: Performance Testing

```sql
-- Connect and run performance tests
-- ./scripts/connect.sh

-- Enable timing
\timing on

-- Test 1: Index usage for status filter
EXPLAIN ANALYZE
SELECT * FROM training_runs WHERE status = 'succeeded';

-- Test 2: JSONB query performance
EXPLAIN ANALYZE
SELECT * FROM training_runs WHERE parameters @> '{"optimizer": "adam"}'::jsonb;

-- Test 3: Date range query
EXPLAIN ANALYZE
SELECT * FROM training_runs
WHERE created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Test 4: Complex aggregation
EXPLAIN ANALYZE
SELECT
    framework,
    status,
    COUNT(*),
    AVG(gpu_hours)
FROM training_runs
GROUP BY framework, status;
```

### Step 3: Create Final Report

Create `REFLECTION.md`:

```markdown
# Exercise 01 Reflection: SQL Fundamentals & CRUD Operations

**Student**: [Your Name]
**Date**: 2025-11-01
**Exercise Duration**: ~4 hours

---

## 1. Understanding: Constraints and Their Purpose

### Primary Key (UUID)
- **Purpose**: Uniquely identify each training run across distributed systems
- **Why UUID**: Better than auto-increment for distributed databases, no collisions

### Foreign Keys (Future)
- Not used in this exercise, but would link to `models`, `datasets`, `users` tables in production

### CHECK Constraints
1. **valid_framework**: Ensures only supported frameworks (pytorch, tensorflow, etc.)
2. **valid_status**: Enforces valid state machine (queued → running → succeeded/failed)
3. **valid_accuracy**: Ensures metrics are in range [0, 1]
4. **time_order**: Logical timestamp ordering (started_at >= created_at)
5. **completed_timestamp**: Completed runs must have completion timestamp

### UNIQUE Constraints
- **unique_daily_run**: Prevents duplicate runs of same model+experiment+dataset on same day

---

## 2. Comparison: DELETE vs TRUNCATE

| Aspect | DELETE | TRUNCATE |
|--------|--------|----------|
| **Speed** | Slower (row-by-row) | Faster (drops/recreates table) |
| **WHERE Clause** | Yes, can be selective | No, all rows removed |
| **Rollback** | Can rollback in transaction | Harder to rollback |
| **Triggers** | Fires DELETE triggers | Does not fire triggers |
| **Resets Sequences** | No | Yes |
| **Use Case** | Selective deletion | Complete table wipe |

**When to use DELETE**: Removing specific old records (e.g., failed runs > 90 days)

**When to use TRUNCATE**: Clearing entire table for testing/reset (use with caution!)

---

## 3. Integration: Exposing Data to Downstream Services

### REST API
- **FastAPI/Flask**: Create endpoints to query training runs
- **Endpoints**:
  - `GET /api/v1/runs` - List all runs with filters
  - `GET /api/v1/runs/{run_id}` - Get specific run
  - `POST /api/v1/runs` - Create new run
  - `PATCH /api/v1/runs/{run_id}` - Update run status/metrics
  - `GET /api/v1/runs/stats` - Aggregate statistics

### GraphQL
- More flexible querying for complex relationships
- Allows clients to request exactly what they need

### Dashboards
- **Grafana**: Connect PostgreSQL data source, visualize metrics over time
- **Metabase**: Business intelligence for non-technical users
- **Tableau**: Advanced analytics and reporting

### Message Queue
- Publish events (run started, completed, failed) to Kafka/RabbitMQ
- Allow other services to react to training lifecycle events

---

## 4. Design: Schema Improvements

### 1. Separate Tables (Normalization)
```sql
CREATE TABLE models (
    model_id UUID PRIMARY KEY,
    model_name TEXT UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE datasets (
    dataset_id UUID PRIMARY KEY,
    dataset_name TEXT UNIQUE NOT NULL,
    size_gb NUMERIC(10,2)
);

CREATE TABLE training_runs (
    run_id UUID PRIMARY KEY,
    model_id UUID REFERENCES models(model_id),
    dataset_id UUID REFERENCES datasets(dataset_id),
    ...
);
```

### 2. Add Audit Trail
```sql
ALTER TABLE training_runs ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE training_runs ADD COLUMN updated_by TEXT;

-- Trigger to auto-update updated_at
CREATE TRIGGER update_training_runs_updated_at
    BEFORE UPDATE ON training_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### 3. Add Versioning/History
```sql
CREATE TABLE training_runs_history (
    history_id UUID PRIMARY KEY,
    run_id UUID REFERENCES training_runs(run_id),
    changed_at TIMESTAMPTZ DEFAULT NOW(),
    changed_by TEXT,
    changes JSONB
);
```

### 4. Add Tags/Labels
```sql
ALTER TABLE training_runs ADD COLUMN tags TEXT[];
CREATE INDEX idx_training_runs_tags ON training_runs USING GIN (tags);

-- Query: SELECT * FROM training_runs WHERE 'production' = ANY(tags);
```

---

## 5. Challenges: Most Difficult Parts

### Challenge 1: JSONB Queries
Understanding the difference between `->` and `->>` operators took time:
- `->` returns JSONB
- `->>` returns TEXT
- `@>` for containment checks

### Challenge 2: Timestamp Constraints
Creating constraints that enforce timestamp consistency across multiple columns was complex. Especially the CHECK constraint ensuring completed runs have completion timestamps.

### Challenge 3: Window Functions
Understanding PARTITION BY and ORDER BY in window functions for ranking and running totals.

---

## 6. Production: Additional Features

### 1. Model Registry Integration
- Link to model artifacts in S3/GCS
- Store model file paths, checksums, versions

### 2. Lineage Tracking
- Track parent/child relationships between runs
- Link experiments that build on each other

### 3. Resource Quotas
- Per-user GPU hour limits
- Cost budgets by team/project

### 4. Notifications
- Slack/email alerts on run completion/failure
- SLA breach notifications

### 5. Experiment Comparison
- Side-by-side comparison of hyperparameters and results
- Statistical significance testing

### 6. Auto-Scaling Integration
- Trigger cluster scale-up/down based on queue depth
- Optimize resource allocation

### 7. Security & Compliance
- Row-level security (RLS) for multi-tenant
- Audit logs for all operations
- Data encryption at rest

### 8. Advanced Analytics
- Hyperparameter optimization insights
- Cost optimization recommendations
- Performance trend analysis

---

## Key Takeaways

1. **Constraints are critical**: Database constraints enforce data integrity at the lowest level
2. **Transactions are essential**: Always use transactions for multi-step operations
3. **JSONB is powerful**: Flexible schema within structured database
4. **Indexes matter**: Even with small data, understanding index design is crucial
5. **PostgreSQL is feature-rich**: CTEs, window functions, JSONB make it suitable for ML infrastructure

---

## Next Steps

- Exercise 02: Design full model registry with multiple related tables
- Exercise 03: Master JOINs for complex queries across tables
- Exercise 04: Integrate with Python using SQLAlchemy ORM
- Exercise 05: Optimize queries and design indexes for production scale
```

---

## Final Deliverables Checklist

```bash
# Directory structure
tree ~/ml-training-registry

# Expected structure:
# ml-training-registry/
# ├── README.md
# ├── REFLECTION.md
# ├── sql/
# │   ├── 01_create_training_runs.sql
# │   ├── 02_seed_training_runs.sql
# │   ├── 03_crud_operations.sql
# │   ├── 04_update_operations.sql
# │   ├── 05_safe_update_patterns.sql
# │   ├── 06_delete_operations.sql
# │   ├── 07_transactions.sql
# │   └── 08_advanced_queries.sql
# ├── scripts/
# │   ├── connect.sh
# │   ├── cleanup_old_runs.sh
# │   └── verify_all.sh
# └── screenshots/
#     ├── table_structure.png
#     ├── seed_data_verification.png
#     └── query_results.png
```

### Self-Assessment Checklist

- [x] I can create tables with appropriate data types
- [x] I understand PRIMARY KEY, UNIQUE, and CHECK constraints
- [x] I can write SELECT queries with filtering and sorting
- [x] I can use aggregate functions (COUNT, AVG, SUM, etc.)
- [x] I can use GROUP BY and HAVING
- [x] I can query JSONB data
- [x] I can perform INSERT, UPDATE, DELETE operations
- [x] I understand transactions (BEGIN/COMMIT/ROLLBACK)
- [x] I can write safe UPDATE/DELETE queries with WHERE
- [x] I can use EXPLAIN ANALYZE to check query performance
- [x] I understand the difference between DELETE and TRUNCATE
- [x] I can import/export data using \copy

---

## Summary

**Completed**: Exercise 01 - SQL Fundamentals & CRUD Operations

**What You Built**:
- Production-ready PostgreSQL database for ML training tracking
- Comprehensive schema with 16 constraints and 8 indexes
- 20 realistic training run records across multiple frameworks
- 100+ query examples (CRUD, aggregation, JSONB, window functions)
- Safety patterns for updates and deletes
- Transaction examples with rollback/commit

**Lines of SQL Written**: 2,500+

**Key Skills Acquired**:
- PostgreSQL schema design
- CRUD operations with safety
- Transaction management
- JSONB querying
- Performance optimization basics
- Production SQL best practices

**Ready For**: Exercise 02 - Database Design & Normalization

---

**Exercise Complete!** 🎉

You now have a solid foundation in SQL fundamentals and can build production-ready databases for ML infrastructure.
