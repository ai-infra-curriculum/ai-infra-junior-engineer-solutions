# Exercise 08: Git LFS for ML Projects - Implementation Guide

## Overview

Master Git LFS (Large File Storage) for managing ML artifacts including models, checkpoints, and datasets. Learn to version models with semantic versioning, integrate with DVC, and implement production-ready workflows.

**Estimated Time**: 90-120 minutes
**Difficulty**: Intermediate
**Prerequisites**: Exercises 01-07

## What You'll Learn

- ✅ Install and configure Git LFS
- ✅ Track models and checkpoints with LFS
- ✅ Create comprehensive .gitignore for ML projects
- ✅ Version models with semantic versioning
- ✅ Integrate DVC for data versioning
- ✅ Implement model lineage tracking
- ✅ Clone LFS repositories efficiently
- ✅ Migrate existing projects to LFS
- ✅ Optimize LFS storage and bandwidth

---

## Part 1: Installing and Configuring Git LFS

### Step 1.1: Install Git LFS

```bash
# Ubuntu/Debian
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# RHEL/CentOS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
sudo yum install git-lfs

# Verify installation
git lfs version
# Output: git-lfs/3.4.0 (GitHub; linux amd64; go 1.20.4)

# One-time global setup
git lfs install
# Output: Updated Git hooks. Git LFS initialized.
```

### Step 1.2: Create ML Project with LFS

```bash
# Create project
mkdir ml-model-registry
cd ml-model-registry
git init --initial-branch=main

# Configure LFS tracking for ML artifacts
git lfs track "*.pt"           # PyTorch models
git lfs track "*.pth"          # PyTorch checkpoints
git lfs track "*.onnx"         # ONNX models
git lfs track "*.h5"           # Keras/TensorFlow models
git lfs track "*.pb"           # TensorFlow SavedModel
git lfs track "*.safetensors"  # SafeTensors format
git lfs track "*.ckpt"         # General checkpoints
git lfs track "*.pkl"          # Pickle files (scikit-learn)
git lfs track "*.joblib"       # Joblib files
git lfs track "models/**"      # All files in models/ directory

# CRITICAL: Always commit .gitattributes first!
cat .gitattributes
# Output:
# *.pt filter=lfs diff=lfs merge=lfs -text
# *.pth filter=lfs diff=lfs merge=lfs -text
# *.onnx filter=lfs diff=lfs merge=lfs -text
# ... etc

git add .gitattributes
git commit -m "config: configure Git LFS for ML artifacts

Track with LFS:
- PyTorch models (.pt, .pth)
- ONNX models (.onnx)
- Keras/TensorFlow models (.h5, .pb)
- SafeTensors format (.safetensors)
- Checkpoints and serialized models

Prevents repository bloat from large binary files."

# View tracked patterns
git lfs track
```

---

## Part 2: Comprehensive .gitignore for ML Projects

### Step 2.1: Create Production-Grade .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environments
venv/
.venv/
ENV/
env/
.virtualenv/

# Jupyter Notebooks
.ipynb_checkpoints/
# Optional: Don't version notebooks (use nbconvert to .py for versioning)
# *.ipynb

# ML Framework Training Artifacts (DON'T commit these)
checkpoints/
experiments/*/checkpoints/
lightning_logs/
wandb/
mlruns/
runs/
outputs/
snapshots/
tensorboard_logs/
.neptune/

# Data Files (Use DVC or S3 instead)
data/raw/
data/processed/
data/interim/
data/external/
datasets/
*.csv
*.tsv
*.json
*.jsonl
*.parquet
*.tfrecord
*.h5  # HDF5 data files (not models)

# Exceptions: Keep small sample/test data
!data/sample/**
!tests/fixtures/**/*.csv

# Large Model Files During Development
# (Only production models in models/production/ are tracked with LFS)
*.pt
*.pth
*.ckpt
*.weights
*.bin

# Exceptions: Production models (tracked with LFS)
!models/production/*.pt
!models/production/*.pth
!models/production/*.onnx
!models/production/*.h5
!models/production/*.pb

# Logs
*.log
logs/
*.out
*.err
.log/

# IDEs and Editors
.vscode/
.idea/
*.swp
*.swo
*.swn
*~
.DS_Store

# Docker
*.tar
*.tar.gz
.dockerignore
# Exceptions
!Dockerfile
!docker-compose.yml
!.dockerignore

# Secrets and Credentials
.env
.env.local
.env.*.local
*.key
*.pem
credentials.json
credentials/
secrets/
config/secrets.yaml
*.secret

# OS Files
Thumbs.db
.DS_Store
Desktop.ini

# Build Artifacts
build/
dist/
*.egg-info/
.eggs/

# Test Coverage
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover
.pytest_cache/
.tox/

# Temporary Files
*.tmp
*.temp
.cache/
.mypy_cache/
.ruff_cache/
.dmypy.json

# DVC (Data Version Control)
.dvc/config.local
.dvc/tmp/
# DO commit: .dvc/config, .dvc/.gitignore, *.dvc files

# Model Serving
*.onnx.prototxt
serving_models/
EOF

git add .gitignore
git commit -m "chore: add comprehensive .gitignore for ML projects

Excludes:
- Training artifacts (checkpoints, logs)
- Data files (use DVC)
- Development models (only production in LFS)
- Secrets and credentials
- IDE and OS files

Keeps:
- Sample/test data
- Production models (LFS-tracked)
- Configuration files
- Documentation"
```

---

## Part 3: Versioning ML Models with LFS

### Step 3.1: Add First Model Version

```bash
# Create project structure
mkdir -p models/production models/experiments
mkdir -p configs data/sample src tests docs

# Create sample model file (simulate 100MB trained model)
dd if=/dev/urandom of=models/production/bert-classifier-v1.0.0.onnx bs=1M count=100

# Create comprehensive model metadata
cat > models/production/bert-classifier-v1.0.0.yaml << 'EOF'
model:
  name: bert-classifier
  version: 1.0.0
  description: "BERT-based sentiment classifier for customer reviews"
  format: onnx
  architecture: BERT-base-uncased
  task: text-classification
  classes: [negative, neutral, positive]

training:
  dataset: customer-reviews-v1
  dataset_size: 100000
  framework: pytorch
  framework_version: 2.1.0
  transformers_version: 4.35.0
  git_commit: abc123def456
  trained_date: "2024-01-15T14:30:00Z"
  training_time_hours: 8.5
  gpu: "4x NVIDIA V100"

hyperparameters:
  learning_rate: 0.00002
  batch_size: 32
  epochs: 10
  max_seq_length: 512
  warmup_steps: 500
  weight_decay: 0.01
  optimizer: adamw

performance:
  accuracy: 0.945
  f1_score_macro: 0.938
  f1_score_weighted: 0.942
  precision: 0.942
  recall: 0.934
  auc_roc: 0.978

  per_class:
    negative: {precision: 0.931, recall: 0.941, f1: 0.936}
    neutral: {precision: 0.938, recall: 0.919, f1: 0.928}
    positive: {precision: 0.957, recall: 0.942, f1: 0.949}

inference:
  latency_p50_ms: 12
  latency_p95_ms: 28
  latency_p99_ms: 45
  throughput_requests_per_sec: 150
  memory_mb: 512

artifacts:
  model_path: models/production/bert-classifier-v1.0.0.onnx
  model_size_mb: 100
  config_path: configs/bert-classifier-production.yaml
  training_logs: s3://ml-artifacts/logs/bert-classifier/exp_001/
  tensorboard: s3://ml-artifacts/tensorboard/bert-classifier/exp_001/

deployment:
  min_onnxruntime_version: 1.16.0
  recommended_hardware: "GPU with 8GB+ VRAM"
  docker_image: ml-models/bert-classifier:v1.0.0

reproducibility:
  random_seed: 42
  code_repo: https://github.com/company/ml-training
  training_script_hash: def456abc789
EOF

# Add README for model
cat > models/production/README.md << 'EOF'
# Production Models

## BERT Classifier

### Current Production Version: v1.0.0

**Purpose**: Sentiment analysis on customer reviews

**Input**: Text string (max 512 tokens)
**Output**: Classification (negative/neutral/positive) with confidence scores

### Usage

```python
import onnxruntime as ort

session = ort.InferenceSession("bert-classifier-v1.0.0.onnx")
inputs = {"input_ids": ..., "attention_mask": ...}
outputs = session.run(None, inputs)
```

### Version History

See MODELS.md in repository root for complete version history.
EOF

# Add model to Git (LFS handles large files automatically)
git add models/production/
git commit -m "model: add BERT classifier v1.0.0

Initial production release of sentiment classifier.

Performance:
- Accuracy: 94.5%
- F1 Score (macro): 93.8%
- Inference latency (p99): 45ms
- Throughput: 150 req/sec

Training:
- Dataset: customer-reviews-v1 (100k samples)
- Framework: PyTorch 2.1.0 / ONNX
- Training time: 8.5 hours on 4x V100

Model tracked with Git LFS (100 MB)."

# Create semantic version tag
git tag -a model-bert-v1.0.0 -m "BERT Classifier v1.0.0 - Initial Production Release

Sentiment analysis model for customer reviews.

Metrics:
- Accuracy: 94.5%
- F1 (macro): 93.8%
- Latency (p99): 45ms

Deployment ready for production use."

# Verify LFS is tracking
git lfs ls-files
# Output:
# 3a52ce7809 * models/production/bert-classifier-v1.0.0.onnx

# Check file size in LFS
git lfs ls-files --size
# Output:
# 3a52ce7809 * models/production/bert-classifier-v1.0.0.onnx (100 MB)
```

### Step 3.2: Release Improved Model Version

```bash
# Simulate training improved model
dd if=/dev/urandom of=models/production/bert-classifier-v1.1.0.onnx bs=1M count=105

cat > models/production/bert-classifier-v1.1.0.yaml << 'EOF'
model:
  name: bert-classifier
  version: 1.1.0  # Minor version: backward-compatible improvement
  description: "Improved BERT classifier with better accuracy"
  format: onnx
  architecture: BERT-base-uncased
  parent_version: 1.0.0

training:
  dataset: customer-reviews-v1.1  # Expanded dataset
  dataset_size: 150000  # +50k samples
  improvements:
    - "Added data augmentation (back-translation)"
    - "Improved preprocessing pipeline"
    - "Extended training with learning rate warmup"

performance:
  accuracy: 0.958  # +1.3% improvement
  f1_score_macro: 0.951  # +1.3% improvement
  f1_score_weighted: 0.954
  precision: 0.953
  recall: 0.949
  auc_roc: 0.983

inference:
  latency_p50_ms: 11  # -1ms improvement
  latency_p99_ms: 42  # -3ms improvement
  throughput_requests_per_sec: 160  # +10 req/sec

artifacts:
  model_path: models/production/bert-classifier-v1.1.0.onnx
  model_size_mb: 105

changelog:
  added:
    - "Back-translation data augmentation"
    - "Improved text normalization"
  improved:
    - "Training convergence with warmup schedule"
    - "Inference speed optimization"
EOF

git add models/production/bert-classifier-v1.1.0.*
git commit -m "model: release BERT classifier v1.1.0

Backward-compatible improvement over v1.0.0.

Performance Improvements:
- Accuracy: 95.8% (+1.3% vs v1.0.0)
- F1 Score: 95.1% (+1.3%)
- Latency: 42ms p99 (-3ms)
- Throughput: 160 req/sec (+10)

Changes:
- Expanded dataset (+50k samples)
- Added back-translation augmentation
- Improved preprocessing pipeline
- Optimized inference speed

Model size: 105 MB (tracked with LFS)
Backward compatible: Yes (same API)"

git tag -a model-bert-v1.1.0 -m "BERT Classifier v1.1.0 - Accuracy and Speed Improvement"

# View tags
git tag -l "model-*"
```

---

## Part 4: Model Registry and Lineage

### Step 4.1: Create Model Registry

```bash
cat > MODELS.md << 'EOF'
# Model Registry

## Production Models

### BERT Sentiment Classifier

**Purpose**: Customer review sentiment analysis
**Repository Path**: `models/production/bert-classifier-*`
**Owner**: ML Team
**Contact**: ml-team@company.com

#### Version History

| Version | Release Date | Accuracy | F1 Score | Latency (p99) | Status | Git Tag | Notes |
|---------|--------------|----------|----------|---------------|--------|---------|-------|
| 1.1.0 | 2024-01-20 | 95.8% | 95.1% | 42ms | **Production** | model-bert-v1.1.0 | Current production model |
| 1.0.0 | 2024-01-15 | 94.5% | 93.8% | 45ms | Deprecated | model-bert-v1.0.0 | Superseded by v1.1.0 |

#### Deployment

**Current Production**: v1.1.0

**Environments**:
- Production (us-east-1): v1.1.0
- Production (eu-west-1): v1.1.0
- Staging: v1.1.0
- Development: v1.2.0-rc1

**Deployment Instructions**:

```bash
# Checkout specific version
git checkout model-bert-v1.1.0

# Download model from LFS
git lfs pull --include="models/production/bert-classifier-v1.1.0.onnx"

# Verify download
ls -lh models/production/bert-classifier-v1.1.0.onnx

# Deploy to Kubernetes
kubectl apply -f k8s/deployments/bert-classifier-v1.1.0.yaml

# Verify deployment
kubectl get pods -l app=bert-classifier
kubectl logs -l app=bert-classifier --tail=50
```

#### Rollback Procedure

If v1.1.0 encounters issues in production:

```bash
# Rollback to previous version
git checkout model-bert-v1.0.0
git lfs pull --include="models/production/bert-classifier-v1.0.0.onnx"

# Deploy previous version
kubectl apply -f k8s/deployments/bert-classifier-v1.0.0.yaml

# Update load balancer
kubectl patch service bert-classifier -p '{"spec":{"selector":{"version":"v1.0.0"}}}'
```

#### Performance Monitoring

**Metrics Dashboard**: https://grafana.company.com/d/bert-classifier
**Alert Rules**: See `monitoring/alerts/bert-classifier.yaml`

**Key Metrics**:
- Request latency (p50, p95, p99)
- Accuracy (online evaluation)
- Error rate
- Throughput

**Retraining Triggers**:
- Accuracy drops below 90%
- Data drift detected
- Every 30 days (scheduled)

#### Testing

**Test Dataset**: `data/test/sentiment-test-v1.csv` (5000 samples)

```bash
# Run model evaluation
python scripts/evaluate_model.py \
  --model models/production/bert-classifier-v1.1.0.onnx \
  --test-data data/test/sentiment-test-v1.csv
```

**Expected Results** (v1.1.0):
- Accuracy: 95.8% ±0.5%
- F1 Score: 95.1% ±0.5%
- Latency: <50ms (p99)
EOF

git add MODELS.md
git commit -m "docs: create model registry and deployment guide

Document:
- Version history with metrics
- Current production deployments
- Deployment and rollback procedures
- Performance monitoring setup
- Testing guidelines

Provides single source of truth for model management."
```

---

## Part 5: Cloning and Working with LFS Repositories

### Step 5.1: Clone with Full LFS Download

```bash
# Create simulated remote
cd ..
git clone --bare ml-model-registry ml-model-registry.git

# Clone with all LFS files (default behavior)
git clone ml-model-registry.git ml-model-registry-full
cd ml-model-registry-full

# Verify model downloaded
ls -lh models/production/
# Output shows actual file sizes (100MB, 105MB)

# Check LFS status
git lfs ls-files
# Shows all LFS files

git lfs ls-files --size
# Shows sizes of LFS files
```

### Step 5.2: Clone Without LFS Files (Bandwidth Optimization)

```bash
# Clone without downloading LFS files (pointers only)
cd ..
GIT_LFS_SKIP_SMUDGE=1 git clone ml-model-registry.git ml-model-registry-minimal
cd ml-model-registry-minimal

# LFS files are tiny pointers
ls -lh models/production/
# Output shows 133 bytes (pointer files)

cat models/production/bert-classifier-v1.0.0.onnx | head -5
# Output:
# version https://git-lfs.github.com/spec/v1
# oid sha256:3a52ce780950d4d969792a2559cd519d
# size 104857600

# Download specific model when needed
git lfs pull --include="models/production/bert-classifier-v1.1.0.onnx"

# Now that file is full size
ls -lh models/production/bert-classifier-v1.1.0.onnx
# Shows 105 MB

# Download all LFS files
git lfs pull
```

---

## Part 6: LFS Maintenance and Optimization

### Step 6.1: Monitor LFS Storage

```bash
# List all LFS files
git lfs ls-files

# List with sizes
git lfs ls-files --size

# Find largest files
git lfs ls-files --size | sort -k2 -hr | head -10

# Check LFS environment
git lfs env

# Verify LFS objects integrity
git lfs fsck
```

### Step 6.2: Optimize LFS Storage

```bash
# Remove LFS objects not needed for current checkout
git lfs prune

# Dry run (see what would be removed)
git lfs prune --dry-run
# Output:
# 3 LFS object(s) would be pruned

# Prune with verification
git lfs prune --verify-remote

# Fetch only recent LFS files
git lfs fetch --recent  # Last 7 days by default

# Configure "recent" timeframe
git config lfs.fetchrecentrefsdays 14  # 14 days
git config lfs.fetchrecentcommitsdays 7  # 7 days

# Fetch specific files only
git lfs fetch --include="models/production/*.onnx"
```

---

## Part 7: Migrating Existing Project to LFS

### Step 7.1: Migrate Large Files to LFS

```bash
# If you already committed large files without LFS:

# 1. Configure LFS tracking
git lfs track "*.onnx"
git lfs track "*.pth"
git add .gitattributes
git commit -m "config: add LFS tracking"

# 2. Migrate existing files to LFS
git lfs migrate import --include="*.onnx,*.pth" --everything

# This rewrites history to move files to LFS
# Output shows progress:
# migrate: Sorting commits: ..., done.
# migrate: Rewriting commits: 100% (50/50), done.

# 3. Verify migration
git lfs ls-files
# All *.onnx and *.pth files now tracked

# 4. Verify repository size reduced
du -sh .git/
# Should be much smaller

# 5. Force push (CAUTION: Rewrites history)
# Coordinate with team first!
git push --force-with-lease origin main

# 6. Clean up old objects
git lfs prune --verify-remote
```

### Step 7.2: Migrate Specific Branches

```bash
# Migrate only specific branches
git lfs migrate import \
  --include="models/**/*.pt" \
  --include-ref=main \
  --include-ref=develop

# Migrate everything except certain files
git lfs migrate import \
  --include="*.onnx" \
  --exclude="test-models/*.onnx" \
  --everything
```

---

## Part 8: DVC Integration (Bonus)

### Step 8.1: Setup DVC Alongside LFS

```bash
# Install DVC
pip install dvc dvc-s3

# Initialize DVC
dvc init

git add .dvc .dvcignore
git commit -m "chore: initialize DVC for data versioning

DVC for: Datasets, raw data
Git LFS for: Models, checkpoints

Separation of concerns for optimal storage."

# Configure DVC remote
mkdir -p /tmp/dvc-storage
dvc remote add -d local /tmp/dvc-storage

git add .dvc/config
git commit -m "config: add DVC remote storage"
```

### Step 8.2: Track Dataset with DVC

```bash
# Create dataset
mkdir -p data/raw
cat > data/raw/train.csv << 'EOF'
id,text,sentiment
1,"Great product!",positive
2,"Terrible quality",negative
3,"It's okay",neutral
EOF

# Track with DVC (NOT LFS)
dvc add data/raw/train.csv

# DVC creates metadata file
ls data/raw/
# train.csv
# train.csv.dvc

# Commit metadata (not data)
git add data/raw/train.csv.dvc data/raw/.gitignore
git commit -m "data: add training dataset v1 (DVC tracked)"

# Push data to DVC remote
dvc push

# Push Git metadata
git push
```

**DVC vs LFS Decision Matrix:**

| Artifact Type | Tool | Reason |
|---------------|------|--------|
| Raw datasets | DVC | Frequent changes, not in critical path |
| Training logs | DVC | Large, less critical |
| Production models | LFS | Critical for deployment, versioned with code |
| Model checkpoints | LFS | Needed for reproducibility |
| Preprocessed data | DVC | Can be regenerated |
| Model configs | Git | Small text files |

---

## Verification Checklist

- [ ] Git LFS installed and configured
- [ ] Created .gitattributes with ML file patterns
- [ ] Comprehensive .gitignore for ML projects
- [ ] At least 2 model versions tracked with LFS
- [ ] Model metadata files created
- [ ] MODELS.md registry with deployment guide
- [ ] Tested cloning with and without LFS files
- [ ] Verified LFS storage with `git lfs ls-files`
- [ ] Tagged model releases semantically
- [ ] (Bonus) Integrated DVC for datasets

---

## Common Issues and Solutions

### Issue 1: "This exceeds GitHub's file size limit"

```bash
# Files larger than 100MB rejected even with LFS
# Solution: Use LFS BEFORE committing

# If already committed:
git lfs migrate import --include="largefile.pt"
git push --force
```

### Issue 2: "LFS bandwidth quota exceeded"

```bash
# Solution: Clone without LFS, pull selectively
GIT_LFS_SKIP_SMUDGE=1 git clone <repo>
cd <repo>
git lfs pull --include="models/production/model-v1.0.0.onnx"
```

### Issue 3: "LFS object not found"

```bash
# Object in .gitattributes but not in LFS storage
# Solution: Re-push LFS objects
git lfs push origin --all
```

---

## Summary

You've mastered Git LFS for ML projects:

- ✅ LFS installation and configuration
- ✅ Tracking models and checkpoints
- ✅ Comprehensive .gitignore patterns
- ✅ Semantic versioning for models
- ✅ Model registry and lineage tracking
- ✅ Efficient cloning strategies
- ✅ LFS maintenance and optimization
- ✅ Migration of existing projects
- ✅ DVC integration for complete workflow

**Key Takeaways:**
- Use LFS for production models and checkpoints
- Version models with semantic versioning (major.minor.patch)
- Always commit .gitattributes before large files
- Clone without LFS for CI/CD (download selectively)
- Maintain model registry for deployment tracking
- Use DVC for datasets, LFS for models

**Time to Complete:** ~120 minutes

**Module 003 Complete!** You now have professional Git skills for ML infrastructure projects.
