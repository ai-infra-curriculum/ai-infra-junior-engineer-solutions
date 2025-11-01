# Exercise 06: ML Workflows - DVC and Model Versioning

## Overview

This exercise demonstrates version control practices specific to machine learning projects, including dataset versioning, model tracking, experiment management, and reproducibility. You'll learn how to combine Git with specialized tools like DVC (Data Version Control) and Git LFS (Large File Storage) to manage the complete ML lifecycle.

## Learning Objectives

By completing this exercise, you will:

- ✅ Understand why standard Git isn't sufficient for ML projects
- ✅ Configure Git LFS for large model files
- ✅ Set up DVC for dataset versioning
- ✅ Track experiments with configuration files
- ✅ Version models with metadata and tags
- ✅ Implement reproducible ML workflows
- ✅ Manage dependencies for reproducibility
- ✅ Create validation scripts for ML artifacts

## Prerequisites

- Completed Exercise 03 (Branching)
- Basic understanding of ML concepts (models, training, datasets)
- Python knowledge (for validation scripts)
- Understanding of YAML and JSON formats

## Setup

Run the setup script to create a complete ML project:

```bash
cd /home/s0v3r1gn/claude/ai-infrastructure-project/repositories/solutions/ai-infra-junior-engineer-solutions/modules/mod-003-git-version-control/exercise-06
bash scripts/setup_ml_project.sh
```

This creates:
- `ml-classification-project/` - Complete ML project with Git history
- Configured Git LFS for model files
- DVC-style configuration for datasets
- 2 experiment configurations
- Model metadata and validation scripts
- 8 commits demonstrating ML workflow

## Part 1: Understanding ML Version Control Challenges

### Why Standard Git Isn't Enough

**Problems with tracking ML artifacts in Git:**

1. **Large Files:**
   ```bash
   # Model files can be huge
   resnet50.pth        # 98 MB
   bert-base.pt        # 440 MB
   gpt2-large.bin      # 3.2 GB

   # Datasets even larger
   imagenet.tar.gz     # 150 GB
   common-crawl.parquet # 500 GB
   ```

2. **Binary Files:**
   - Models are binary (not text-diffable)
   - Git's diff and merge don't work
   - Every change creates full copy
   - Repository size explodes

3. **Frequent Changes:**
   - Models retrained frequently
   - Each checkpoint is a new file
   - Experiments generate many versions
   - History becomes unwieldy

4. **Collaboration Issues:**
   - Large files slow cloning
   - Pushing/pulling takes forever
   - Storage costs increase
   - Team productivity suffers

### The Solution: Specialized Tools

| Tool | Purpose | What It Handles |
|------|---------|-----------------|
| **Git** | Code versioning | Python code, configs, docs |
| **Git LFS** | Large file storage | Model checkpoints, artifacts |
| **DVC** | Data versioning | Datasets, features, pipelines |
| **MLflow/Wandb** | Experiment tracking | Metrics, parameters, runs |

## Part 2: Git LFS (Large File Storage)

### What is Git LFS?

Git LFS replaces large files with small pointer files in Git, while storing the actual content on a separate server.

**How it works:**
```
Without LFS:
├── model.pth (98 MB stored in Git)
└── .git/
    └── objects/
        └── ab12cd... (full 98 MB file)

With LFS:
├── model.pth (pointer file, ~150 bytes)
└── .git/
    ├── objects/
    │   └── ab12cd... (150 byte pointer)
    └── lfs/
        └── objects/
            └── 1a2b3c... (98 MB file, locally cached)
```

### Configuring Git LFS

Navigate to the ML project:
```bash
cd ml-classification-project
```

**Check Git LFS configuration:**
```bash
cat .gitattributes
```

You'll see:
```
# PyTorch Models
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text

# TensorFlow Models
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text

# ONNX Models
*.onnx filter=lfs diff=lfs merge=lfs -text

# Pickle Files (model artifacts)
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text

# Model Weights
weights/*.bin filter=lfs diff=lfs merge=lfs -text

# Large Data Files (if not using DVC)
*.parquet filter=lfs diff=lfs merge=lfs -text
*.feather filter=lfs diff=lfs merge=lfs -text
```

### Understanding .gitattributes

**Pattern breakdown:**
```bash
*.pth filter=lfs diff=lfs merge=lfs -text
│     │          │        │         └─ Treat as binary
│     │          │        └─ Use LFS for merging
│     │          └─ Use LFS for diffing
│     └─ Use LFS filter
└─ File pattern
```

**Common patterns:**
- `*.pth` - PyTorch model files
- `*.h5` - Keras/TensorFlow models
- `*.onnx` - ONNX format models
- `*.pkl` - Pickled Python objects
- `weights/*.bin` - Binary weight files in specific directory

### Checking LFS Files

**List LFS-tracked files:**
```bash
git lfs ls-files
```

Output:
```
1a2b3c4d * models/resnet50/model_v1.0.0.pth
```

**View LFS pointer content:**
```bash
cat models/resnet50/model_v1.0.0.pth
```

Output (pointer file):
```
version https://git-lfs.github.com/spec/v1
oid sha256:1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z7a8b9c0d1e2f
size 98543210
```

### Working with LFS Files

**Add new model to LFS:**
```bash
# Model files matching .gitattributes are automatically tracked
cp ~/trained_model.pth models/resnet50/model_v1.1.0.pth

git add models/resnet50/model_v1.1.0.pth
git commit -m "model: add ResNet-50 v1.1.0"
```

**Verify it's tracked by LFS:**
```bash
git lfs ls-files
```

**Push LFS files:**
```bash
# Push Git repository and LFS files
git push origin main

# LFS upload happens automatically:
# Uploading LFS objects: 100% (1/1), 98 MB | 10 MB/s
```

**Clone with LFS:**
```bash
# Git LFS must be installed
git lfs install

# Clone normally - LFS files downloaded automatically
git clone <repository-url>

# Or clone without downloading LFS files initially
GIT_LFS_SKIP_SMUDGE=1 git clone <repository-url>

# Download LFS files later
cd <repository>
git lfs pull
```

### LFS Storage Management

**Check LFS storage usage:**
```bash
git lfs ls-files -s
```

**Fetch only recent LFS files:**
```bash
# Fetch LFS files for last 10 commits
git lfs fetch --recent

# Fetch only files referenced by current checkout
git lfs fetch origin main
```

**Prune old LFS files:**
```bash
# Remove old LFS files not referenced by recent commits
git lfs prune

# Dry run to see what would be deleted
git lfs prune --dry-run --verbose
```

## Part 3: DVC (Data Version Control)

### What is DVC?

DVC is like Git LFS but specifically designed for data science:
- Versions datasets and ML pipelines
- Stores data in remote storage (S3, GCS, Azure, SSH, etc.)
- Tracks data dependencies
- Enables reproducible ML pipelines

### DVC Architecture

```
Working Directory:
├── data/
│   └── train.csv          # Actual data file (gitignored)
├── train.csv.dvc          # DVC metadata (tracked in Git)
└── .dvc/
    ├── config             # DVC configuration
    └── cache/             # Local cache (gitignored)

Remote Storage:
└── s3://my-bucket/dvc-storage/
    └── ab/
        └── 12cd34ef...    # Actual data content
```

### Exploring DVC Configuration

**Check DVC config:**
```bash
cd ml-classification-project
cat .dvc/config
```

You'll see:
```ini
[core]
    remote = local
    autostage = true
['remote "local"']
    url = /tmp/dvc-storage
['remote "s3"']
    url = s3://my-ml-project/dvc-storage
    region = us-west-2
```

**Configuration explained:**
- `remote = local` - Default remote storage location
- `autostage = true` - Automatically `git add` .dvc files
- `['remote "local"']` - Local filesystem storage (for development)
- `['remote "s3"']` - S3 bucket storage (for production)

### Understanding .dvc Files

**Check dataset metadata:**
```bash
cat data/raw/train.csv.dvc
```

Content:
```yaml
outs:
- md5: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
  size: 145
  path: train.csv
```

**Metadata fields:**
- `md5` - Hash of file content (for verification)
- `size` - File size in bytes
- `path` - Relative path to actual data file

**Why .dvc files?**
- Small (few lines) - safe to commit to Git
- Contains hash to verify data integrity
- Points to data in DVC remote storage
- Enables data versioning without storing data in Git

### DVC Workflow (Simulated)

In a real project with DVC installed:

**1. Track new dataset:**
```bash
# Add dataset to DVC
dvc add data/raw/validation.csv

# This creates:
# - data/raw/validation.csv.dvc (tracked in Git)
# - Updates .gitignore (ignores validation.csv)

# Commit the .dvc file
git add data/raw/validation.csv.dvc .gitignore
git commit -m "data: add validation dataset"
```

**2. Push data to remote:**
```bash
# Upload data to DVC remote (S3, GCS, etc.)
dvc push

# Uploads:
# data/raw/validation.csv → s3://bucket/dvc-storage/ab/12cd...
```

**3. Pull data on another machine:**
```bash
# Clone Git repository
git clone <repo-url>
cd <repo>

# Pull DVC-tracked data
dvc pull

# Downloads:
# s3://bucket/dvc-storage/ab/12cd... → data/raw/validation.csv
```

**4. Update dataset:**
```bash
# Modify dataset
python scripts/preprocess.py  # Creates new train.csv

# Update DVC tracking
dvc add data/raw/train.csv

# Commit new version
git add data/raw/train.csv.dvc
git commit -m "data: update training dataset v2"

# Push new version
dvc push
```

**5. Checkout old dataset version:**
```bash
# Checkout old commit
git checkout HEAD~1 data/raw/train.csv.dvc

# Download corresponding data
dvc checkout

# Now data/raw/train.csv is the old version
```

### DVC vs Git LFS

| Feature | Git LFS | DVC |
|---------|---------|-----|
| **Purpose** | Large files | Data + ML pipelines |
| **Storage** | GitHub LFS, GitLab LFS | S3, GCS, Azure, SSH, local |
| **Use Case** | Model files, artifacts | Datasets, features |
| **Pipeline Support** | ❌ No | ✅ Yes |
| **Remote Types** | Limited | Many options |
| **ML-Specific** | ❌ General purpose | ✅ ML-focused |
| **Versioning** | File-level | File + pipeline |

**Common practice:**
- Use **Git LFS** for model checkpoints (*.pth, *.h5)
- Use **DVC** for datasets and data pipelines
- Use **Git** for code, configs, documentation

## Part 4: Experiment Tracking

### Why Track Experiments?

ML development involves many experiments:
```
Baseline model (exp-001):
- LR: 0.1, Batch: 256, Epochs: 90
- Accuracy: 84.8%

Higher LR (exp-002):
- LR: 0.3, Batch: 256, Epochs: 90
- Accuracy: 86.2% ✅ Improvement!

Larger batch (exp-003):
- LR: 0.3, Batch: 512, Epochs: 90
- Accuracy: 85.9% ❌ Slightly worse

Different optimizer (exp-004):
- LR: 0.3, Batch: 256, Epochs: 90, Optimizer: Adam
- Accuracy: 87.1% ✅ Best so far!
```

Without tracking:
- ❌ Forget what worked
- ❌ Can't reproduce results
- ❌ Lose valuable insights
- ❌ Duplicate experiments

With tracking:
- ✅ Complete history of experiments
- ✅ Reproducible results
- ✅ Compare different approaches
- ✅ Build on successful experiments

### Experiment Configuration Files

**Check baseline experiment:**
```bash
cat experiments/exp-001-baseline.yaml
```

Content:
```yaml
experiment:
  id: "exp-001"
  name: "baseline-resnet50"
  description: "Baseline ResNet-50 with default hyperparameters"
  created_at: "2024-01-15T10:00:00Z"
  git_commit: ""  # Will be filled at runtime

model:
  architecture: "resnet50"
  pretrained: false
  num_classes: 1000

data:
  dataset: "imagenet"
  version: "2023.1"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  dvc_hash: "a1b2c3d4e5f6g7h8"

training:
  epochs: 90
  batch_size: 256
  learning_rate: 0.1
  optimizer: "sgd"
  momentum: 0.9
  weight_decay: 0.0001
  lr_scheduler: "step"
  lr_decay_epochs: [30, 60, 80]
  lr_decay_rate: 0.1

augmentation:
  random_crop: true
  random_flip: true
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

hardware:
  gpus: 4
  distributed: true
  mixed_precision: true

results:
  train_accuracy: 0.876
  val_accuracy: 0.852
  test_accuracy: 0.848
  train_loss: 0.342
  val_loss: 0.389
  training_time_hours: 48.5
  best_epoch: 87
```

**Check improved experiment:**
```bash
cat experiments/exp-002-higher-lr.yaml
```

Key differences:
```yaml
experiment:
  id: "exp-002"
  name: "resnet50-higher-lr"
  description: "ResNet-50 with higher initial learning rate"
  parent_experiment: "exp-001"  # ← Links to baseline

training:
  learning_rate: 0.3  # ← CHANGED: increased from 0.1

results:
  test_accuracy: 0.862  # ← +1.4% improvement!

improvements:
  vs_baseline: "+1.4% accuracy"
  notes: "Higher LR improved convergence speed"
```

### Experiment Configuration Best Practices

**1. Use descriptive IDs:**
```yaml
# Good
id: "exp-001-baseline"
id: "exp-002-higher-lr"
id: "exp-003-adam-optimizer"

# Bad
id: "exp1"
id: "test"
id: "final"
```

**2. Link related experiments:**
```yaml
experiment:
  id: "exp-003"
  parent_experiment: "exp-002"  # Shows lineage
  description: "Building on exp-002, trying Adam optimizer"
```

**3. Record everything:**
```yaml
# Complete configuration
model: {...}
data: {...}
training: {...}
augmentation: {...}
hardware: {...}
results: {...}

# Not just results!
```

**4. Include data version:**
```yaml
data:
  dataset: "imagenet"
  version: "2023.1"  # Dataset version
  dvc_hash: "a1b2c3d4"  # DVC hash for exact data
```

**5. Track Git commit:**
```yaml
experiment:
  git_commit: "abc123def456"  # Code version used
```

### Running Experiments

**Example training script integration:**
```python
#!/usr/bin/env python3
"""Train model with experiment tracking."""

import yaml
import git
from datetime import datetime

def run_experiment(config_path):
    # Load experiment config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Record Git commit
    repo = git.Repo('.')
    config['experiment']['git_commit'] = repo.head.commit.hexsha
    config['experiment']['started_at'] = datetime.utcnow().isoformat()

    # Train model
    model = create_model(config['model'])
    train_loader = create_dataloader(config['data'], config['training'])

    results = train(model, train_loader, config['training'])

    # Update config with results
    config['results'] = results
    config['experiment']['completed_at'] = datetime.utcnow().isoformat()

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Commit results
    repo.index.add([config_path])
    repo.index.commit(f"results: {config['experiment']['id']}")

    return results

if __name__ == "__main__":
    results = run_experiment("experiments/exp-001-baseline.yaml")
    print(f"Test accuracy: {results['test_accuracy']:.1%}")
```

### Comparing Experiments

**View all experiments:**
```bash
git log --oneline --grep="experiment:"
```

Output:
```
2b3c4d5 experiment: higher learning rate (exp-002)
1a2b3c4 experiment: baseline ResNet-50 (exp-001)
```

**Compare experiment files:**
```bash
git diff 1a2b3c4 2b3c4d5 -- experiments/
```

**Extract experiment results:**
```python
#!/usr/bin/env python3
"""Compare experiment results."""

import yaml
import glob

def compare_experiments():
    experiments = []

    for config_file in sorted(glob.glob("experiments/exp-*.yaml")):
        with open(config_file) as f:
            config = yaml.safe_load(f)

        experiments.append({
            'id': config['experiment']['id'],
            'name': config['experiment']['name'],
            'lr': config['training']['learning_rate'],
            'accuracy': config['results']['test_accuracy'],
        })

    # Print comparison table
    print(f"{'ID':<15} {'Name':<30} {'LR':<10} {'Accuracy':<10}")
    print("-" * 70)
    for exp in experiments:
        print(f"{exp['id']:<15} {exp['name']:<30} {exp['lr']:<10.2f} {exp['accuracy']:<10.1%}")

if __name__ == "__main__":
    compare_experiments()
```

Output:
```
ID              Name                           LR         Accuracy
----------------------------------------------------------------------
exp-001         baseline-resnet50             0.10       84.8%
exp-002         resnet50-higher-lr            0.30       86.2%
```

## Part 5: Model Versioning

### Model Metadata

**Check model metadata:**
```bash
cat models/resnet50/model_v1.0.0.json
```

Content:
```json
{
  "model_name": "resnet50",
  "version": "1.0.0",
  "architecture": "ResNet-50",
  "framework": "pytorch",
  "framework_version": "2.1.0",
  "created_at": "2024-01-15T10:00:00Z",
  "training_config": {
    "experiment_id": "exp-001",
    "dataset": "imagenet",
    "dataset_version": "2023.1",
    "epochs": 90,
    "batch_size": 256,
    "learning_rate": 0.1,
    "optimizer": "sgd"
  },
  "metrics": {
    "train_accuracy": 0.876,
    "val_accuracy": 0.852,
    "test_accuracy": 0.848,
    "train_loss": 0.342,
    "val_loss": 0.389
  },
  "data_version": "imagenet_v2023.1",
  "data_dvc_hash": "a1b2c3d4e5f6g7h8",
  "git_commit": "abc123def456",
  "git_tag": "model-v1.0.0",
  "deployment": {
    "input_shape": [3, 224, 224],
    "output_classes": 1000,
    "preprocessing": "ImageNet normalization",
    "compatible_with": ["pytorch>=2.0.0"]
  },
  "performance": {
    "inference_time_ms": 15.3,
    "model_size_mb": 98.5,
    "parameters_millions": 25.6
  }
}
```

### Metadata Fields Explained

**Essential fields:**
- `model_name` - Model identifier
- `version` - Semantic version (X.Y.Z)
- `architecture` - Model architecture
- `framework` - ML framework used
- `framework_version` - Specific framework version

**Training provenance:**
- `experiment_id` - Links to experiment config
- `dataset` - Dataset name
- `dataset_version` - Dataset version
- `data_dvc_hash` - DVC hash for exact data
- `git_commit` - Code version used
- `git_tag` - Git tag for this model

**Performance metrics:**
- `metrics` - Training/validation/test metrics
- `inference_time_ms` - Prediction latency
- `model_size_mb` - Model file size
- `parameters_millions` - Number of parameters

**Deployment info:**
- `input_shape` - Expected input format
- `output_classes` - Number of output classes
- `preprocessing` - Required preprocessing
- `compatible_with` - Framework compatibility

### Semantic Versioning for Models

**Version format: MAJOR.MINOR.PATCH**

```
v1.0.0 → v1.0.1 → v1.1.0 → v2.0.0
│        │        │        └─ Breaking changes
│        │        └─ New features (backward compatible)
│        └─ Bug fixes
└─ Initial release
```

**When to increment:**

**MAJOR** (breaking changes):
- Different input/output format
- Requires different preprocessing
- Architecture fundamentally changed
- Not compatible with v1.x deployments

**MINOR** (new features):
- Improved accuracy (same interface)
- Additional output classes
- Better performance
- Backward compatible

**PATCH** (bug fixes):
- Fixed preprocessing bug
- Corrected inference issue
- Dependency updates
- No functional changes

**Examples:**
```bash
v1.0.0 - Initial ResNet-50 model (84.8% accuracy)
v1.0.1 - Fixed normalization bug
v1.1.0 - Improved to 86.2% accuracy
v1.1.1 - Updated PyTorch dependency
v2.0.0 - Changed to ResNet-101 (different architecture)
```

### Tagging Models

**Check existing tags:**
```bash
git tag
```

Output:
```
model-v1.0.0
```

**View tag details:**
```bash
git show model-v1.0.0
```

Output:
```
tag model-v1.0.0
Tagger: ML Engineer <ml@example.com>
Date:   Mon Jan 15 10:00:00 2024 +0000

Model Release v1.0.0

ResNet-50 image classifier
- Test accuracy: 84.8%
- Production-ready

Training:
- Experiment: exp-001
- Dataset: ImageNet v2023.1
- 90 epochs, 4x GPU, 48.5 hours

Deployment:
- PyTorch 2.0+
- Input: 224x224 RGB
- Output: 1000 classes

commit abc123def456...
```

**Create new model tag:**
```bash
# Train new model version
python train.py --config experiments/exp-002-higher-lr.yaml

# Save model
# Creates: models/resnet50/model_v1.1.0.pth
#          models/resnet50/model_v1.1.0.json

# Commit model
git add models/resnet50/model_v1.1.0.*
git commit -m "model: add ResNet-50 v1.1.0

Improved model from exp-002:
- Test accuracy: 86.2% (+1.4% vs v1.0.0)
- Same architecture, better hyperparameters"

# Tag the release
git tag -a model-v1.1.0 -m "Model Release v1.1.0

ResNet-50 image classifier (improved)
- Test accuracy: 86.2%
- +1.4% improvement over v1.0.0

Training:
- Experiment: exp-002
- Dataset: ImageNet v2023.1 (same)
- Higher learning rate (0.3 vs 0.1)
- 90 epochs, 4x GPU, 47.2 hours

Deployment:
- PyTorch 2.0+ (same as v1.0.0)
- Backward compatible
- Drop-in replacement for v1.0.0"
```

**Push tag:**
```bash
git push origin model-v1.1.0
```

**Checkout specific model version:**
```bash
# Checkout v1.0.0
git checkout model-v1.0.0

# Model files are now at v1.0.0
ls models/resnet50/
# model_v1.0.0.pth
# model_v1.0.0.json

# Return to latest
git checkout main
```

## Part 6: Reproducibility

### Why Reproducibility Matters

**Scenario 1: Production bug**
```
Production model v1.0.0 has a bug in preprocessing.
Need to fix and retrain with exact same setup.

Without reproducibility:
❌ Can't remember exact hyperparameters
❌ Dataset might have changed
❌ Dependencies have updated
❌ Can't reproduce original model

With reproducibility:
✅ Checkout model-v1.0.0 tag
✅ DVC pulls exact dataset version
✅ requirements.txt has pinned versions
✅ experiment config has all parameters
✅ Can reproduce and fix
```

**Scenario 2: Research paper**
```
Published paper with 86.2% accuracy claim.
Reviewers want to verify results.

Without reproducibility:
❌ "Works on my machine"
❌ Can't share exact setup
❌ Results not verifiable
❌ Paper rejected

With reproducibility:
✅ Share Git repo + DVC remote
✅ Exact code, data, configs
✅ Anyone can reproduce
✅ Paper accepted
```

### Reproducibility Checklist

**✅ Code version:**
```bash
git_commit: "abc123def456"  # In experiment config
git tag model-v1.0.0        # Tag model releases
```

**✅ Data version:**
```bash
data_dvc_hash: "a1b2c3d4"   # DVC hash in experiment config
dvc pull                     # Pull exact dataset version
```

**✅ Dependencies:**
```bash
# requirements.txt with pinned versions
torch==2.1.0  # Not torch>=2.0.0
numpy==1.26.0
```

**✅ Configuration:**
```yaml
# All hyperparameters in experiment config
training:
  learning_rate: 0.1
  batch_size: 256
  optimizer: "sgd"
  # ... everything specified
```

**✅ Random seeds:**
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

**✅ Hardware:**
```yaml
hardware:
  gpus: 4
  gpu_type: "V100"
  distributed: true
```

**✅ Environment:**
```yaml
environment.yaml  # Conda environment spec
```

### Dependency Management

**Check requirements:**
```bash
cat requirements.txt
```

Content:
```
# Core ML Framework
torch==2.1.0           # ← Pinned version
torchvision==0.16.0

# Data Processing
numpy==1.26.0
pandas==2.1.1
pillow==10.1.0
scikit-learn==1.3.2

# Experiment Tracking
mlflow==2.8.0
wandb==0.15.12

# Version Control
dvc==3.30.0
dvc-s3==3.0.0

# Configuration
pyyaml==6.0.1
python-dotenv==1.0.0
hydra-core==1.3.2

# Utilities
tqdm==4.66.1
click==8.1.7

# Testing
pytest==7.4.3
pytest-cov==4.1.0

# Code Quality
black==23.11.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.5.0
```

**Why pin versions?**
```bash
# Bad - not reproducible
torch>=2.0.0  # Could install 2.1.0 or 2.5.0
numpy         # Could install any version

# Good - reproducible
torch==2.1.0  # Exact version
numpy==1.26.0
```

**Check Conda environment:**
```bash
cat environment.yaml
```

Content:
```yaml
name: ml-classification
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch=2.1.0
  - torchvision=0.16.0
  - numpy=1.26.0
  - pandas=2.1.1
  - pillow=10.1.0
  - scikit-learn=1.3.2
  - pip
  - pip:
    - mlflow==2.8.0
    - wandb==0.15.12
    - dvc==3.30.0
    - dvc-s3==3.0.0
    - pyyaml==6.0.1
    - hydra-core==1.3.2
```

**Create environment:**
```bash
# From requirements.txt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# From environment.yaml
conda env create -f environment.yaml
conda activate ml-classification
```

**Freeze current environment:**
```bash
# Pip
pip freeze > requirements-frozen.txt

# Conda
conda env export > environment-frozen.yaml
```

### Reproducible Training Script

**Example:**
```python
#!/usr/bin/env python3
"""Reproducible training script."""

import torch
import numpy as np
import random
import yaml
import git
from datetime import datetime

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def record_provenance(config):
    """Record complete provenance information."""
    repo = git.Repo('.')

    # Git information
    config['provenance'] = {
        'git_commit': repo.head.commit.hexsha,
        'git_branch': repo.active_branch.name,
        'git_remote': repo.remotes.origin.url,
        'git_dirty': repo.is_dirty(),
    }

    # Timestamp
    config['provenance']['started_at'] = datetime.utcnow().isoformat()

    # Python environment
    import sys
    config['provenance']['python_version'] = sys.version

    # PyTorch version
    config['provenance']['pytorch_version'] = torch.__version__
    config['provenance']['cuda_version'] = torch.version.cuda

    # Hardware
    config['provenance']['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config['provenance']['gpu_name'] = torch.cuda.get_device_name(0)
        config['provenance']['gpu_count'] = torch.cuda.device_count()

    return config

def train_with_reproducibility(config_path):
    """Train model with full reproducibility."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Set random seed: {seed}")

    # Record provenance
    config = record_provenance(config)

    # Verify data hash
    with open(f"{config['data']['path']}.dvc") as f:
        dvc_meta = yaml.safe_load(f)
    assert dvc_meta['outs'][0]['md5'] == config['data']['dvc_hash'], \
        "Data hash mismatch! Dataset has changed."

    # Train model
    print("Training with configuration:")
    print(yaml.dump(config, default_flow_style=False))

    model = create_model(config)
    results = train(model, config)

    # Update config with results
    config['results'] = results
    config['provenance']['completed_at'] = datetime.utcnow().isoformat()

    # Save updated config
    output_path = config_path.replace('.yaml', '-results.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Results saved to: {output_path}")
    print(f"Test accuracy: {results['test_accuracy']:.1%}")

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train.py <config.yaml>")
        sys.exit(1)

    results = train_with_reproducibility(sys.argv[1])
```

## Part 7: Validation Scripts

### Model Metadata Validation

**Run metadata validator:**
```bash
cd ml-classification-project
python scripts/check_model_metadata.py models/resnet50/model_v1.0.0.json
```

Expected output:
```
✓ models/resnet50/model_v1.0.0.json is valid
```

**Check validator source:**
```bash
cat scripts/check_model_metadata.py
```

The script validates:
- All required fields present
- Version format (X.Y.Z)
- Data types correct
- References valid (experiment ID, Git commit, etc.)

**Validation logic:**
```python
REQUIRED_FIELDS = [
    "model_name",
    "version",
    "architecture",
    "framework",
    "created_at",
    "training_config",
    "metrics"
]

def validate_metadata(filepath):
    """Validate model metadata JSON."""
    with open(filepath) as f:
        metadata = json.load(f)

    # Check required fields
    missing = [field for field in REQUIRED_FIELDS if field not in metadata]
    if missing:
        print(f"Error: Missing fields: {missing}")
        return False

    # Check version format
    version = metadata.get("version", "")
    if not version or len(version.split(".")) != 3:
        print(f"Error: Invalid version format. Use X.Y.Z")
        return False

    print(f"✓ {filepath} is valid")
    return True
```

### Experiment Validation

**Run experiment validator:**
```bash
python scripts/validate_experiment.py experiments/exp-001-baseline.yaml
```

Expected output:
```
✓ experiments/exp-001-baseline.yaml is valid
```

**Validate all experiments:**
```bash
python scripts/validate_experiment.py experiments/*.yaml
```

Expected output:
```
✓ experiments/exp-001-baseline.yaml is valid
✓ experiments/exp-002-higher-lr.yaml is valid
```

**Validation checks:**
- Required sections present (experiment, model, data, training)
- Experiment ID format (exp-XXX)
- Valid YAML syntax
- Numeric fields in valid ranges
- References exist (parent_experiment, etc.)

### Pre-Commit Hooks

**Create pre-commit hook:**
```bash
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for ML project

echo "Running pre-commit checks..."

# Check Python syntax
if ! python3 -m py_compile scripts/*.py; then
    echo "❌ Python syntax errors found"
    exit 1
fi

# Validate model metadata
for metadata in models/*/*.json; do
    if [ -f "$metadata" ]; then
        if ! python scripts/check_model_metadata.py "$metadata"; then
            echo "❌ Invalid model metadata: $metadata"
            exit 1
        fi
    fi
done

# Validate experiment configs
for config in experiments/*.yaml; do
    if [ -f "$config" ]; then
        if ! python scripts/validate_experiment.py "$config"; then
            echo "❌ Invalid experiment config: $config"
            exit 1
        fi
    fi
done

echo "✓ All pre-commit checks passed"
exit 0
EOF

chmod +x .git/hooks/pre-commit
```

**Test pre-commit hook:**
```bash
# Make a change
echo "test" >> README.md
git add README.md

# Try to commit - hook runs automatically
git commit -m "test commit"
```

Output:
```
Running pre-commit checks...
✓ All pre-commit checks passed
[main abc123d] test commit
 1 file changed, 1 insertion(+)
```

## Part 8: Complete ML Workflow

### End-to-End Example

**1. Start new experiment:**
```bash
# Copy baseline config
cp experiments/exp-001-baseline.yaml experiments/exp-003-adam-optimizer.yaml

# Edit new config
vim experiments/exp-003-adam-optimizer.yaml
# Change: optimizer: "sgd" → "adam"
# Change: id: "exp-001" → "exp-003"
# Add: parent_experiment: "exp-001"
```

**2. Verify data version:**
```bash
# Check current data hash
cat data/raw/train.csv.dvc | grep md5

# Matches experiment config
grep dvc_hash experiments/exp-003-adam-optimizer.yaml
```

**3. Train model:**
```bash
# Run training
python train.py experiments/exp-003-adam-optimizer.yaml

# Generates:
# - Updated experiment config with results
# - Model checkpoint: models/resnet50/model_v1.2.0.pth
# - Model metadata: models/resnet50/model_v1.2.0.json
```

**4. Validate results:**
```bash
# Validate model metadata
python scripts/check_model_metadata.py models/resnet50/model_v1.2.0.json

# Validate experiment config
python scripts/validate_experiment.py experiments/exp-003-adam-optimizer.yaml
```

**5. Commit experiment:**
```bash
git add experiments/exp-003-adam-optimizer.yaml
git commit -m "experiment: Adam optimizer (exp-003)

Results:
- Test accuracy: 87.1% (+2.3% vs baseline)
- Training time: 45.8 hours
- Change: SGD → Adam optimizer

Best model so far!"
```

**6. Commit model:**
```bash
git add models/resnet50/model_v1.2.0.*
git commit -m "model: add ResNet-50 v1.2.0

Model checkpoint from exp-003:
- Architecture: ResNet-50
- Test accuracy: 87.1%
- Framework: PyTorch 2.1.0
- Optimizer: Adam (new)
- Weights tracked with Git LFS"
```

**7. Tag release:**
```bash
git tag -a model-v1.2.0 -m "Model Release v1.2.0

ResNet-50 image classifier with Adam optimizer
- Test accuracy: 87.1%
- +2.3% improvement over baseline
- Best model to date

Training:
- Experiment: exp-003
- Dataset: ImageNet v2023.1
- Adam optimizer (instead of SGD)
- 90 epochs, 4x GPU, 45.8 hours

Deployment:
- PyTorch 2.0+
- Backward compatible with v1.x
- Drop-in replacement"
```

**8. Push everything:**
```bash
# Push Git repository
git push origin main

# Push Git tags
git push origin model-v1.2.0

# Push DVC data (if using real DVC)
# dvc push

# Push LFS files (automatic with git push)
```

**9. Document results:**
```bash
# Update README or results table
cat >> RESULTS.md << EOF
## Model v1.2.0 (2024-01-17)

| Metric | Value |
|--------|-------|
| Test Accuracy | 87.1% |
| Improvement | +2.3% vs baseline |
| Training Time | 45.8 hours |
| Experiment | exp-003 |
| Key Change | Adam optimizer |

### Hyperparameters
- Learning rate: 0.001
- Optimizer: Adam
- Batch size: 256
- Epochs: 90

### Reproducibility
\`\`\`bash
git checkout model-v1.2.0
dvc pull
python train.py experiments/exp-003-adam-optimizer.yaml
\`\`\`
EOF

git add RESULTS.md
git commit -m "docs: add model v1.2.0 results"
git push
```

### Complete Git History

**View the full history:**
```bash
git log --oneline --graph
```

Expected output:
```
* 9a0b1c2 docs: add model v1.2.0 results
* 8f9e0d1 model: add ResNet-50 v1.2.0
* 7e8d9c0 experiment: Adam optimizer (exp-003)
* 6d7c8b9 ci: add validation scripts for ML artifacts
* 5c6b7a8 deps: add dependency specifications
* 4b5a6f7 model: add ResNet-50 v1.0.0
* 3a4f5e6 experiment: higher learning rate (exp-002)
* 2f3e4d5 experiment: baseline ResNet-50 (exp-001)
* 1e2d3c4 data: add initial training and test datasets
* 0d1c2b3 config: configure DVC for data versioning
* abc123d init: initialize ML project with DVC and Git LFS
```

**View model tags:**
```bash
git tag -l "model-*"
```

Output:
```
model-v1.0.0
model-v1.2.0
```

## Part 9: Best Practices

### Project Structure

**Recommended structure:**
```
ml-project/
├── .git/                   # Git repository
├── .dvc/                   # DVC configuration
│   ├── config              # DVC remotes
│   └── .gitignore          # DVC cache ignored
├── .gitattributes          # Git LFS configuration
├── .gitignore              # Git ignore patterns
├── data/                   # Datasets (DVC-tracked)
│   ├── raw/                # Raw data
│   │   ├── train.csv       # Actual data (gitignored)
│   │   └── train.csv.dvc   # DVC metadata (tracked)
│   └── processed/          # Processed data
├── models/                 # Model checkpoints (LFS-tracked)
│   └── resnet50/
│       ├── model_v1.0.0.pth      # Model weights (LFS)
│       └── model_v1.0.0.json     # Model metadata (Git)
├── experiments/            # Experiment configs (Git-tracked)
│   ├── exp-001-baseline.yaml
│   └── exp-002-improved.yaml
├── src/                    # Source code (Git-tracked)
│   ├── models/             # Model definitions
│   ├── data/               # Data loaders
│   └── training/           # Training scripts
├── tests/                  # Unit tests (Git-tracked)
├── scripts/                # Utility scripts (Git-tracked)
│   ├── validate_experiment.py
│   └── check_model_metadata.py
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
├── environment.yaml        # Conda environment
└── README.md               # Documentation
```

### Naming Conventions

**Experiments:**
```bash
exp-001-baseline              # ID + short description
exp-002-higher-lr             # What changed
exp-003-adam-optimizer
exp-004-data-augmentation
```

**Models:**
```bash
model_v1.0.0.pth             # Semantic versioning
model_v1.0.0.json            # Matching metadata
model_v1.1.0.pth             # Minor improvement
model_v2.0.0.pth             # Breaking change
```

**Branches:**
```bash
experiment/exp-005-new-arch   # For experiments
model/v1.3.0                  # For model development
data/update-imagenet-v2024    # For data updates
```

**Tags:**
```bash
model-v1.0.0                  # Model releases
data-v2023.1                  # Dataset versions
experiment-baseline           # Important experiments
```

### Commit Messages for ML

**Experiment commits:**
```bash
git commit -m "experiment: <name> (exp-XXX)

Results:
- Metric 1: X.X%
- Metric 2: Y.Y
- Change: what you changed

Notes: additional context"
```

**Model commits:**
```bash
git commit -m "model: add <name> <version>

Model checkpoint from exp-XXX:
- Architecture: <arch>
- Test accuracy: X.X%
- Framework: <framework>
- Weights tracked with Git LFS"
```

**Data commits:**
```bash
git commit -m "data: <action> <dataset>

- Version: <version>
- Size: X samples
- Format: <format>
- DVC hash: <hash>

Run 'dvc pull' to download."
```

### What to Track Where

| Content | Tool | Location | Example |
|---------|------|----------|---------|
| **Code** | Git | src/, scripts/ | train.py, model.py |
| **Configs** | Git | experiments/, configs/ | exp-001.yaml |
| **Small data** | Git | data/samples/ | sample_image.jpg |
| **Large data** | DVC | data/raw/, data/processed/ | imagenet.tar.gz |
| **Models** | Git LFS | models/ | model_v1.0.0.pth |
| **Metadata** | Git | models/, experiments/ | model_v1.0.0.json |
| **Notebooks** | Git | notebooks/ | analysis.ipynb |
| **Docs** | Git | docs/, README.md | usage.md |
| **Results** | Git | results/ | metrics.csv |
| **Artifacts** | Git LFS | artifacts/ | visualizations.png |

### Collaboration Workflow

**For team members:**

```bash
# 1. Clone repository
git clone <repo-url>
cd <repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull data
dvc pull

# 4. Create experiment branch
git checkout -b experiment/exp-010-my-idea

# 5. Run experiment
cp experiments/exp-001-baseline.yaml experiments/exp-010-my-idea.yaml
# Edit config
python train.py experiments/exp-010-my-idea.yaml

# 6. Commit results
git add experiments/exp-010-my-idea.yaml
git commit -m "experiment: my idea (exp-010)"

# 7. If successful, commit model
git add models/*/model_v*.*
git commit -m "model: add version from exp-010"

# 8. Push and create PR
git push origin experiment/exp-010-my-idea
# Create pull request on GitHub
```

### Documentation Best Practices

**README.md should include:**
- Project overview
- Setup instructions
- How to run experiments
- How to train models
- How to reproduce results
- Model versions and performance
- Dataset information
- Contributing guidelines

**Each experiment should document:**
- Motivation (why this experiment?)
- Hypothesis (what do you expect?)
- Configuration (what changed?)
- Results (what happened?)
- Analysis (why did it work/not work?)
- Next steps (what to try next?)

## Part 10: Troubleshooting

### Common Issues

**Issue 1: Git LFS not installed**
```bash
# Symptom
git clone <repo>
# model.pth is 150 bytes (pointer file)

# Solution
git lfs install
git lfs pull
```

**Issue 2: DVC data not pulled**
```bash
# Symptom
FileNotFoundError: data/raw/train.csv

# Solution
dvc pull
```

**Issue 3: Dependency mismatch**
```bash
# Symptom
RuntimeError: Model was trained with PyTorch 2.1, you have 2.3

# Solution
pip install torch==2.1.0  # Match requirements.txt
```

**Issue 4: Can't reproduce results**
```bash
# Symptom
Test accuracy: 78.3% (expected 84.8%)

# Checklist
1. Correct Git commit? git checkout model-v1.0.0
2. Correct data version? dvc pull
3. Correct dependencies? pip list | grep torch
4. Set random seed? Check train.py
5. Same hardware? Check experiment config
```

**Issue 5: Model file too large for Git**
```bash
# Symptom
remote: error: File model.pth is 150 MB; this exceeds GitHub's file size limit of 100 MB

# Solution
# Should have used Git LFS!
git rm --cached model.pth
echo "*.pth filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
git add .gitattributes
git lfs track "*.pth"
git add model.pth
git commit -m "fix: track model.pth with Git LFS"
```

### Verification Commands

**Check LFS tracking:**
```bash
git lfs ls-files          # List LFS files
git lfs status            # Check LFS status
cat .gitattributes        # Verify LFS patterns
```

**Check DVC setup:**
```bash
dvc status                # Check tracked files
dvc list . --dvc-only     # List DVC-tracked files
cat .dvc/config           # Verify remotes
```

**Verify reproducibility:**
```bash
# Check Git commit
git rev-parse HEAD

# Check data hash
cat data/raw/*.dvc | grep md5

# Check dependencies
pip freeze | grep -E "torch|numpy"

# Verify experiment config
cat experiments/exp-001-baseline.yaml | grep -E "learning_rate|batch_size"
```

## Summary

You've learned how to:

- ✅ **Configure Git LFS** for large model files
- ✅ **Set up DVC** for dataset versioning
- ✅ **Track experiments** with configuration files
- ✅ **Version models** with semantic versioning and tags
- ✅ **Ensure reproducibility** with pinned dependencies and provenance
- ✅ **Validate artifacts** with automated scripts
- ✅ **Implement ML workflows** from experiment to production

### Key Takeaways

1. **Use the right tool for each artifact type:**
   - Git: code, configs, docs
   - Git LFS: model checkpoints
   - DVC: datasets, features

2. **Track everything needed for reproducibility:**
   - Code version (Git commit)
   - Data version (DVC hash)
   - Dependencies (requirements.txt)
   - Configuration (experiment YAML)
   - Random seeds

3. **Version models semantically:**
   - MAJOR: breaking changes
   - MINOR: improvements
   - PATCH: bug fixes

4. **Document thoroughly:**
   - Experiment motivation and results
   - Model metadata and performance
   - Reproducibility instructions

5. **Automate validation:**
   - Pre-commit hooks
   - Validation scripts
   - CI/CD checks

## Next Steps

1. **Practice with real models:**
   - Train actual models
   - Track experiments
   - Version and release models

2. **Explore advanced DVC:**
   - DVC pipelines
   - Remote storage (S3, GCS)
   - Data dependencies

3. **Integrate with MLflow/Wandb:**
   - Automated experiment tracking
   - Interactive dashboards
   - Hyperparameter optimization

4. **Set up CI/CD:**
   - Automated testing
   - Model validation
   - Deployment pipelines

## Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Weights & Biases](https://docs.wandb.ai/)
- [PyTorch Model Serialization](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [ML Reproducibility Best Practices](https://www.tensorflow.org/guide/checkpoint)
