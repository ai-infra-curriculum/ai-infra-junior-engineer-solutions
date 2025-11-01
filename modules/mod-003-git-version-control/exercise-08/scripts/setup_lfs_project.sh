#!/bin/bash

#######################################################################
# Exercise 08: Git LFS for ML Projects - Setup Script
#######################################################################
# Creates a complete ML model repository demonstrating:
# - Git LFS configuration for ML artifacts
# - Model versioning with semantic versioning
# - Comprehensive .gitignore for ML projects
# - Model registry and lineage tracking
# - DVC integration (simulated)
# - LFS maintenance workflows
#######################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$EXERCISE_DIR/ml-model-repository"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Git LFS for ML Projects Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Clean up existing project
if [ -d "$PROJECT_DIR" ]; then
    rm -rf "$PROJECT_DIR"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

#######################################################################
# Part 1: Initialize Repository and Git LFS
#######################################################################

echo -e "${YELLOW}[1/8] Initializing repository and Git LFS...${NC}"

git init
git branch -m master main
git config user.name "ML Engineer"
git config user.email "ml@example.com"

# Configure comprehensive LFS tracking
cat > .gitattributes << 'EOF'
# Git LFS Configuration for ML Projects

# === PyTorch Models ===
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text

# === TensorFlow Models ===
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.keras filter=lfs diff=lfs merge=lfs -text
saved_model/** filter=lfs diff=lfs merge=lfs -text

# === ONNX Models ===
*.onnx filter=lfs diff=lfs merge=lfs -text

# === Other Model Formats ===
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text

# === Pickle Files ===
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text

# === Model Weights ===
weights/*.* filter=lfs diff=lfs merge=lfs -text

# === Datasets (if storing in Git) ===
*.parquet filter=lfs diff=lfs merge=lfs -text
*.feather filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text

# === Compressed Archives ===
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text

# === Production Models Directory ===
models/production/** filter=lfs diff=lfs merge=lfs -text

# === Embeddings ===
embeddings/*.npy filter=lfs diff=lfs merge=lfs -text
embeddings/*.npz filter=lfs diff=lfs merge=lfs -text
EOF

git add .gitattributes
git commit -m "init: configure Git LFS for ML artifacts

Tracking:
- PyTorch models (.pt, .pth, .ckpt)
- TensorFlow models (.h5, .pb, .keras)
- ONNX models (.onnx)
- Model weights and embeddings
- Dataset files (.parquet, .feather)
- Production models directory"

#######################################################################
# Part 2: Create Comprehensive .gitignore
#######################################################################

echo -e "${YELLOW}[2/8] Creating comprehensive .gitignore...${NC}"

cat > .gitignore << 'EOF'
# ============================================
# ML Project .gitignore
# ============================================

# === Python ===
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
MANIFEST
pip-log.txt
pip-delete-this-directory.txt

# === Virtual Environments ===
.venv/
venv/
ENV/
env/
.env.local
virtualenv/

# === Jupyter Notebooks ===
.ipynb_checkpoints/
*.ipynb  # Optional: version separately
# !notebooks/*.ipynb  # Uncomment to track notebooks

# === ML Framework Artifacts ===
# Training artifacts (don't commit these)
checkpoints/
experiments/*/checkpoints/
lightning_logs/
wandb/
mlruns/
mlflow/
runs/
outputs/
snapshots/
tensorboard/

# === Data ===
# Large datasets (use DVC or external storage)
data/raw/
data/processed/
data/interim/
data/external/
datasets/

# Data files
*.csv
*.json
*.jsonl
*.tsv
*.txt
*.tfrecord
*.tfrecords
*.hdf5

# Exceptions: Keep small sample data
!data/sample/*.csv
!data/sample/*.json
!tests/fixtures/*.csv

# === DVC ===
/dvc.lock
.dvc/cache/
.dvc/tmp/

# === Models ===
# Training checkpoints (use LFS for production only)
*.pt
*.pth
*.ckpt
*.h5
*.pb

# Exceptions: Production models tracked with LFS
!models/production/*.pt
!models/production/*.pth
!models/production/*.onnx
!models/production/*.h5

# === Logs ===
*.log
logs/
*.out
*.err

# === IDEs ===
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject
.settings/

# === OS ===
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# === Docker ===
*.tar
*.tar.gz
# Exception: Keep Dockerfiles
!Dockerfile*
!docker-compose*.yml

# === Secrets ===
.env
.env.*
!.env.example
*.key
*.pem
*.p12
*.pfx
credentials.json
secrets/
config/secrets*.yaml
**/secrets/**

# === Build Artifacts ===
build/
dist/
*.egg-info/
*.whl

# === Cache ===
.cache/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
.tox/

# === Temporary Files ===
*.tmp
*.temp
*.bak
*.swp
*~
.~lock.*

# === ML-Specific ===
# Hugging Face cache
.cache/huggingface/

# Downloaded models
models/downloaded/
models/cache/

# Experiment artifacts
experiments/*/artifacts/
experiments/*/logs/

# Model outputs
predictions/
inference_results/

# Profiling
*.prof
*.lprof
EOF

git add .gitignore
git commit -m "chore: add comprehensive .gitignore for ML projects

Ignoring:
- Training artifacts (checkpoints, logs)
- Large datasets (use DVC)
- Temporary/cache files
- IDE and OS files
- Secrets and credentials

Exceptions for:
- Production models (tracked with LFS)
- Sample data for tests
- Docker configurations"

#######################################################################
# Part 3: Create Project Structure
#######################################################################

echo -e "${YELLOW}[3/8] Creating project structure...${NC}"

mkdir -p {models/production,models/experiments,configs,data/sample,src,tests,scripts,docs,deployment}

cat > README.md << 'EOF'
# ML Model Repository

Production-ready ML model repository with Git LFS and DVC integration.

## Repository Structure

```
ml-model-repository/
├── models/
│   ├── production/     # Production models (Git LFS)
│   └── experiments/    # Experimental models
├── configs/            # Model configurations
├── data/
│   └── sample/         # Sample data for testing
├── src/                # Source code
├── tests/              # Test suite
├── scripts/            # Utility scripts
├── deployment/         # Deployment configs
├── docs/               # Documentation
├── MODELS.md           # Model registry
└── .gitattributes      # Git LFS configuration
```

## Quick Start

### Clone Repository

```bash
# Clone with all LFS files
git clone <repo-url>

# Clone without LFS files (faster)
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>
cd ml-model-repository
git lfs pull --include="models/production/model-name-v1.0.0.onnx"
```

### Deploy Model

```bash
# Checkout specific model version
git checkout model-bert-v1.0.0

# Download model if not already present
git lfs pull --include="models/production/bert-*"

# Deploy
kubectl apply -f deployment/bert-classifier-v1.0.0.yaml
```

## Model Versioning

We use semantic versioning for models:

- **MAJOR**: Breaking changes (different input/output format)
- **MINOR**: Backward-compatible improvements
- **PATCH**: Bug fixes

## Documentation

- [Model Registry](MODELS.md) - All production models
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Development Workflow](docs/DEVELOPMENT.md)
EOF

git add README.md
git commit -m "docs: add comprehensive README"

#######################################################################
# Part 4: Create First Model Version (v1.0.0)
#######################################################################

echo -e "${YELLOW}[4/8] Creating model v1.0.0...${NC}"

# Create model files (simulated)
mkdir -p models/production

# Simulate model file (small for demo)
echo "# Simulated BERT Classifier Model v1.0.0" > models/production/bert-classifier-v1.0.0.onnx
echo "model_architecture: bert-base-uncased" >> models/production/bert-classifier-v1.0.0.onnx
echo "parameters: 110M" >> models/production/bert-classifier-v1.0.0.onnx
echo "file_size: 418MB (simulated)" >> models/production/bert-classifier-v1.0.0.onnx

# Create model metadata
cat > models/production/bert-classifier-v1.0.0.yaml << 'EOF'
model:
  name: bert-classifier
  version: 1.0.0
  format: onnx
  architecture: bert-base-uncased
  parameters: 110000000
  file_size_mb: 418

training:
  dataset:
    name: sentiment-analysis
    version: v1.0
    samples: 50000
  framework: pytorch
  framework_version: 2.1.0
  training_date: "2024-01-15"
  training_time_hours: 12.5
  git_commit: abc123def456
  experiment_id: exp-001

hyperparameters:
  learning_rate: 0.00002
  batch_size: 32
  epochs: 3
  max_seq_length: 512
  warmup_steps: 500
  weight_decay: 0.01

performance:
  metrics:
    accuracy: 0.945
    f1_score: 0.938
    precision: 0.942
    recall: 0.934
  inference:
    latency_p50_ms: 12
    latency_p95_ms: 28
    latency_p99_ms: 45
    throughput_qps: 450

hardware:
  training:
    gpus: 4
    gpu_type: "NVIDIA V100"
    memory_gb: 64
  inference:
    min_memory_gb: 2
    recommended_gpu: "T4 or better"

deployment:
  input:
    format: "text"
    max_length: 512
    example: "This movie was fantastic!"
  output:
    format: "json"
    schema:
      label: "positive|negative|neutral"
      confidence: "float [0-1]"
  compatible_with:
    - "onnxruntime>=1.15.0"
    - "transformers>=4.30.0"

artifacts:
  model_path: models/production/bert-classifier-v1.0.0.onnx
  config_path: configs/bert-classifier-production.yaml
  training_logs: s3://ml-artifacts/logs/exp-001/
  dataset_hash: sha256:a1b2c3d4e5f6...

notes: |
  Initial production release of BERT-based sentiment classifier.
  Trained on 50K labeled movie reviews.
  Suitable for general sentiment analysis tasks.
EOF

# Create production config
cat > configs/bert-classifier-production.yaml << 'EOF'
model:
  path: models/production/bert-classifier-v1.0.0.onnx
  type: onnx

serving:
  port: 8080
  workers: 4
  timeout_seconds: 30
  max_batch_size: 32

preprocessing:
  tokenizer: "bert-base-uncased"
  max_length: 512
  padding: true
  truncation: true

inference:
  device: "cuda"  # or "cpu"
  optimization_level: 3
  inter_op_num_threads: 4
  intra_op_num_threads: 4

monitoring:
  enable_metrics: true
  log_predictions: true
  sample_rate: 0.01
EOF

git add models/production/bert-classifier-v1.0.0.* configs/bert-classifier-production.yaml
git commit -m "model: add BERT classifier v1.0.0

Initial production release:
- Accuracy: 94.5%
- F1 Score: 93.8%
- Latency (p99): 45ms
- Trained on 50K samples

Model Details:
- Architecture: bert-base-uncased (110M params)
- Format: ONNX
- Size: 418MB
- Framework: PyTorch 2.1.0"

git tag -a model-bert-v1.0.0 -m "BERT Sentiment Classifier v1.0.0

Production Release
- Accuracy: 94.5%
- Deployment ready
- Full documentation included"

#######################################################################
# Part 5: Create Improved Model Version (v1.1.0)
#######################################################################

echo -e "${YELLOW}[5/8] Creating improved model v1.1.0...${NC}"

# Create improved model
echo "# Simulated BERT Classifier Model v1.1.0 (IMPROVED)" > models/production/bert-classifier-v1.1.0.onnx
echo "model_architecture: bert-base-uncased" >> models/production/bert-classifier-v1.1.0.onnx
echo "parameters: 110M" >> models/production/bert-classifier-v1.1.0.onnx
echo "file_size: 420MB (simulated)" >> models/production/bert-classifier-v1.1.0.onnx
echo "improvements: better preprocessing + data augmentation" >> models/production/bert-classifier-v1.1.0.onnx

# Create metadata for v1.1.0
cat > models/production/bert-classifier-v1.1.0.yaml << 'EOF'
model:
  name: bert-classifier
  version: 1.1.0  # Minor version: backward-compatible improvement
  format: onnx
  architecture: bert-base-uncased
  parameters: 110000000
  file_size_mb: 420
  parent_version: 1.0.0

training:
  dataset:
    name: sentiment-analysis
    version: v1.1  # Updated dataset
    samples: 75000  # +50% more data
  improvements:
    - "Added data augmentation (back-translation)"
    - "Improved text preprocessing"
    - "Extended training data"
  framework: pytorch
  framework_version: 2.1.2
  training_date: "2024-01-22"
  training_time_hours: 18.0
  git_commit: def456abc789
  experiment_id: exp-005

hyperparameters:
  learning_rate: 0.00002
  batch_size: 32
  epochs: 4  # +1 epoch
  max_seq_length: 512
  warmup_steps: 500
  weight_decay: 0.01

performance:
  metrics:
    accuracy: 0.958  # +1.3% improvement
    f1_score: 0.951  # +1.3% improvement
    precision: 0.954
    recall: 0.948
  inference:
    latency_p50_ms: 11  # -1ms improvement
    latency_p95_ms: 26  # -2ms improvement
    latency_p99_ms: 42  # -3ms improvement
    throughput_qps: 465  # +15 QPS

improvements_over_v1_0_0:
  accuracy: "+1.3%"
  f1_score: "+1.3%"
  latency_p99: "-3ms"
  throughput: "+15 QPS"

deployment:
  backward_compatible: true
  migration_notes: "Drop-in replacement for v1.0.0"
  input:
    format: "text"
    max_length: 512
  output:
    format: "json"
    schema:
      label: "positive|negative|neutral"
      confidence: "float [0-1]"

notes: |
  Minor version update with performance improvements.
  Backward compatible with v1.0.0.
  Recommended upgrade for all deployments.
EOF

git add models/production/bert-classifier-v1.1.0.*
git commit -m "model: release BERT classifier v1.1.0

Performance improvements:
- Accuracy: 95.8% (+1.3%)
- F1 Score: 95.1% (+1.3%)
- Latency: 42ms p99 (-3ms)
- Throughput: 465 QPS (+15)

Changes:
- Updated to dataset v1.1 (+25K samples)
- Added data augmentation
- Improved text preprocessing
- Extended training (+1 epoch)

Backward Compatible: Yes
Recommended Action: Upgrade from v1.0.0"

git tag -a model-bert-v1.1.0 -m "BERT Sentiment Classifier v1.1.0

Improved Performance Release
- +1.3% accuracy improvement
- -3ms latency reduction
- Backward compatible
- Recommended upgrade"

#######################################################################
# Part 6: Create Model Registry
#######################################################################

echo -e "${YELLOW}[6/8] Creating model registry...${NC}"

cat > MODELS.md << 'EOF'
# Model Registry

Production model versions with performance metrics and deployment information.

## Active Models

### BERT Sentiment Classifier

Binary sentiment classification for English text.

#### Version History

| Version | Date | Accuracy | F1 | Latency (p99) | Git Tag | Status | Notes |
|---------|------|----------|-----|---------------|---------|--------|-------|
| **1.1.0** | 2024-01-22 | **95.8%** | **95.1%** | **42ms** | model-bert-v1.1.0 | **✅ Production** | Recommended version |
| 1.0.0 | 2024-01-15 | 94.5% | 93.8% | 45ms | model-bert-v1.0.0 | 🟡 Supported | Legacy support |

#### Quick Start

```bash
# Deploy latest version (v1.1.0)
git checkout model-bert-v1.1.0
git lfs pull --include="models/production/bert-classifier-v1.1.0.onnx"
kubectl apply -f deployment/bert-classifier-v1.1.0.yaml
```

#### Version Details

**v1.1.0** (Current Production)
- **Improvements**: +1.3% accuracy, -3ms latency
- **Dataset**: sentiment-analysis v1.1 (75K samples)
- **Changes**: Data augmentation, improved preprocessing
- **Migration**: Drop-in replacement for v1.0.0
- **Deployment**: All environments

**v1.0.0** (Legacy)
- **Status**: Supported until 2024-03-01
- **Dataset**: sentiment-analysis v1.0 (50K samples)
- **Recommendation**: Migrate to v1.1.0

---

## Deployment Instructions

### Standard Deployment

```bash
# 1. Clone repository
git clone <repo-url>
cd ml-model-repository

# 2. Checkout model version
git checkout model-bert-v1.1.0

# 3. Download model files
git lfs pull --include="models/production/bert-classifier-v1.1.0.onnx"

# 4. Verify model
ls -lh models/production/bert-classifier-v1.1.0.onnx

# 5. Deploy
kubectl apply -f deployment/bert-classifier-v1.1.0.yaml
```

### Fast Clone (Without All LFS Files)

```bash
# Clone without LFS files (saves bandwidth)
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>
cd ml-model-repository

# Checkout version
git checkout model-bert-v1.1.0

# Download only this model
git lfs pull --include="models/production/bert-classifier-v1.1.0.onnx"
```

---

## Rollback Procedures

### Rollback from v1.1.0 to v1.0.0

```bash
# 1. Checkout previous version
git checkout model-bert-v1.0.0

# 2. Download model
git lfs pull --include="models/production/bert-classifier-v1.0.0.onnx"

# 3. Deploy
kubectl apply -f deployment/bert-classifier-v1.0.0.yaml

# 4. Verify rollback
kubectl get pods -l model=bert-classifier
```

---

## Model Performance Tracking

### Accuracy Trends

```
v1.0.0: 94.5% ████████████████████████▌
v1.1.0: 95.8% ██████████████████████████
```

### Latency Trends (p99)

```
v1.0.0: 45ms ████████████████████████▌
v1.1.0: 42ms ███████████████████████
```

---

## Version Selection Guide

| Use Case | Recommended Version | Reason |
|----------|---------------------|---------|
| New deployments | v1.1.0 | Best performance |
| Production updates | v1.1.0 | Backward compatible |
| Legacy systems | v1.0.0 | If upgrade testing pending |
| Development/testing | v1.1.0 | Latest features |

---

## Model Metadata

All models include:

- **Model file**: ONNX format (optimized for inference)
- **Metadata**: YAML with training details, metrics, deployment specs
- **Config**: Production configuration
- **Git tag**: For version pinning
- **Documentation**: In `docs/models/`

---

## Adding New Models

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Training new versions
- Model validation requirements
- Metadata templates
- Release process
EOF

git add MODELS.md
git commit -m "docs: create comprehensive model registry

Includes:
- Version history table
- Performance metrics
- Deployment instructions
- Rollback procedures
- Version selection guide
- Model metadata standards"

#######################################################################
# Part 7: Create DVC Configuration (Simulated)
#######################################################################

echo -e "${YELLOW}[7/8] Setting up DVC integration...${NC}"

mkdir -p .dvc data/sample

cat > .dvc/config << 'EOF'
[core]
    remote = s3storage
    autostage = true

['remote "s3storage"']
    url = s3://ml-models-data/dvc-storage
    region = us-west-2
    # access_key_id = <from AWS credentials>
    # secret_access_key = <from AWS credentials>

['remote "local"']
    url = /tmp/dvc-storage

['remote "gcs"']
    url = gs://ml-models-data/dvc-storage
    projectname = ml-project-12345
EOF

cat > .dvcignore << 'EOF'
# DVC ignore patterns
*.pyc
__pycache__/
.git/
.dvc/cache/
.dvc/tmp/
EOF

# Create sample dataset
cat > data/sample/train.csv << 'EOF'
text,label
"This movie was fantastic!",positive
"Terrible waste of time.",negative
"Absolutely loved it!",positive
"Worst movie ever.",negative
"Great acting and plot.",positive
EOF

# Simulate DVC metadata file
cat > data/sample/train.csv.dvc << 'EOF'
outs:
- md5: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
  size: 245
  path: train.csv
EOF

git add .dvc/ .dvcignore data/sample/train.csv.dvc
git commit -m "chore: configure DVC for dataset versioning

DVC Remotes:
- s3storage (primary): S3 bucket
- local: /tmp/dvc-storage (development)
- gcs: Google Cloud Storage (backup)

Sample dataset metadata added.
Actual data managed by DVC."

#######################################################################
# Part 8: Create Utility Scripts
#######################################################################

echo -e "${YELLOW}[8/8] Creating utility scripts...${NC}"

cat > scripts/lfs_status.sh << 'EOF'
#!/bin/bash
# Check Git LFS status and storage

echo "=== Git LFS Files ==="
git lfs ls-files

echo ""
echo "=== LFS Files with Sizes ==="
git lfs ls-files --size | head -20

echo ""
echo "=== LFS Storage Usage ==="
du -sh .git/lfs 2>/dev/null || echo "No LFS cache yet"

echo ""
echo "=== LFS Environment ==="
git lfs env | grep -E "Endpoint|LocalWorkingDir"
EOF

cat > scripts/deploy_model.sh << 'EOF'
#!/bin/bash
# Deploy specific model version

if [ -z "$1" ]; then
    echo "Usage: ./deploy_model.sh <model-tag>"
    echo "Example: ./deploy_model.sh model-bert-v1.1.0"
    exit 1
fi

MODEL_TAG=$1

echo "Deploying model: $MODEL_TAG"

# Checkout model version
git checkout $MODEL_TAG

# Download model files
echo "Downloading model files..."
git lfs pull --include="models/production/*.onnx"

# Verify model exists
MODEL_FILE=$(ls models/production/*.onnx 2>/dev/null | head -1)
if [ -z "$MODEL_FILE" ]; then
    echo "Error: No model file found"
    exit 1
fi

echo "Model ready: $MODEL_FILE"
echo "Size: $(du -h $MODEL_FILE | cut -f1)"
echo ""
echo "To deploy to Kubernetes:"
echo "  kubectl apply -f deployment/${MODEL_TAG}.yaml"
EOF

cat > scripts/list_models.sh << 'EOF'
#!/bin/bash
# List all model versions

echo "=== Model Versions (Git Tags) ==="
git tag -l "model-*" | sort -V

echo ""
echo "=== Production Models ==="
ls -lh models/production/*.onnx 2>/dev/null || echo "No models downloaded yet"

echo ""
echo "=== Model Metadata ==="
ls -1 models/production/*.yaml
EOF

chmod +x scripts/*.sh

git add scripts/
git commit -m "scripts: add utility scripts for model management

Added:
- lfs_status.sh: Check LFS status and storage
- deploy_model.sh: Deploy specific model version
- list_models.sh: List all model versions"

#######################################################################
# Summary
#######################################################################

echo ""
echo -e "${GREEN}✓ ML model repository setup complete!${NC}"
echo ""
echo "Repository Summary:"
echo "  - Git LFS configured for ML artifacts"
echo "  - 2 model versions: v1.0.0 and v1.1.0"
echo "  - Model registry: MODELS.md"
echo "  - DVC integration configured"
echo "  - Utility scripts for management"
echo ""

echo "Git History:"
git log --oneline --graph -10

echo ""
echo "Git Tags:"
git tag -l "model-*"

echo ""
echo "LFS Files:"
git lfs ls-files

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  cd $PROJECT_DIR"
echo "  cat MODELS.md"
echo "  ./scripts/list_models.sh"
echo "  ./scripts/lfs_status.sh"
