# Exercise 06: Git for ML Workflows - DVC and Model Versioning - Implementation Guide

## Overview

Master ML-specific version control including data versioning with DVC, model tracking with Git LFS, experiment management, and reproducible ML pipelines. Learn professional ML engineering workflows.

**Estimated Time**: 90-120 minutes
**Difficulty**: Intermediate to Advanced
**Prerequisites**: Exercise 05 - Collaboration

## What You'll Learn

- ✅ Data versioning with DVC
- ✅ Model file tracking with Git LFS
- ✅ Experiment configuration management
- ✅ Reproducible ML pipelines
- ✅ Model release tagging
- ✅ Dependency locking
- ✅ ML-specific Git hooks
- ✅ Model and data provenance

---

## Part 1: Understanding ML Version Control

### Step 1.1: The ML Version Control Challenge

**Traditional Git Problems for ML:**
- ❌ Binary files (models, data) blow up repository size
- ❌ No diff for binary files
- ❌ Slow clones with large files
- ❌ GitHub/GitLab file size limits

**ML-Specific Requirements:**
- ✅ Version code, data, AND models together
- ✅ Track experiment configurations
- ✅ Reproducible training runs
- ✅ Model lineage and provenance
- ✅ Efficient storage for large files

**Solution Stack:**
```
Code         → Git
Data         → DVC (Data Version Control)
Models       → Git LFS (Large File Storage)
Experiments  → Git + YAML configs
Pipeline     → DVC pipelines
```

### Step 1.2: Install Required Tools

```bash
# Install DVC
pip install dvc dvc-s3  # For S3 storage
# OR
# pip install dvc dvc-gs   # For Google Cloud Storage
# pip install dvc dvc-azure # For Azure Blob Storage

# Install Git LFS
# macOS
brew install git-lfs

# Ubuntu/Debian
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Initialize Git LFS (one-time setup)
git lfs install

# Verify installations
dvc version
# Output: DVC version: 3.30.0

git lfs version
# Output: git-lfs/3.4.0

# Create new ML project
mkdir ml-image-classifier
cd ml-image-classifier
git init --initial-branch=main
```

---

## Part 2: Data Versioning with DVC

### Step 2.1: Initialize DVC

```bash
# Initialize DVC in repository
dvc init

# This creates:
ls -la .dvc/
# .dvc/.gitignore
# .dvc/config
# .dvc/plots/
# .dvc/tmp/

# Also creates .dvcignore (like .gitignore for DVC)
cat .dvcignore
# Output:
# # Add patterns of files dvc should ignore
# *.pyc
# .DS_Store

# Commit DVC initialization
git status
git add .dvc .dvcignore
git commit -m "init: initialize DVC for data version control

Setup DVC for tracking datasets and ML artifacts:
- Data files stored separately from Git
- Lightweight metadata in Git
- Full data in DVC remote storage"
```

### Step 2.2: Configure DVC Remote Storage

```bash
# Option 1: Local storage (for this exercise)
mkdir -p /tmp/dvc-storage-ml-classifier
dvc remote add -d local /tmp/dvc-storage-ml-classifier

# Option 2: AWS S3 (production)
# dvc remote add -d myremote s3://my-bucket/dvc-storage
# dvc remote modify myremote region us-west-2

# Option 3: Google Cloud Storage
# dvc remote add -d myremote gs://my-bucket/dvc-storage

# Option 4: Azure Blob Storage
# dvc remote add -d myremote azure://mycontainer/path

# View configuration
cat .dvc/config
# Output:
# [core]
#     remote = local
# ['remote "local"']
#     url = /tmp/dvc-storage-ml-classifier

# Commit remote configuration
git add .dvc/config
git commit -m "config: add DVC remote storage

Configure local DVC remote for development.
Production deployment will use cloud storage (S3/GCS/Azure)."
```

### Step 2.3: Track Dataset with DVC

```bash
# Create dataset structure
mkdir -p data/raw data/processed

# Create sample training dataset
cat > data/raw/train_images.csv << 'EOF'
image_id,image_path,label,split
img_001,images/cat_001.jpg,cat,train
img_002,images/dog_001.jpg,dog,train
img_003,images/cat_002.jpg,cat,train
img_004,images/dog_002.jpg,dog,train
img_005,images_cat_003.jpg,cat,train
img_006,images/dog_003.jpg,dog,train
img_007,images/cat_004.jpg,cat,train
img_008,images/dog_004.jpg,dog,train
EOF

# Create test dataset
cat > data/raw/test_images.csv << 'EOF'
image_id,image_path,label,split
img_101,images/cat_101.jpg,cat,test
img_102,images/dog_101.jpg,dog,test
img_103,images/cat_102.jpg,cat,test
img_104,images/dog_102.jpg,dog,test
EOF

# Track datasets with DVC
dvc add data/raw/train_images.csv
dvc add data/raw/test_images.csv

# DVC creates .dvc files (metadata)
ls data/raw/
# Output:
# test_images.csv
# test_images.csv.dvc
# train_images.csv
# train_images.csv.dvc

# View DVC metadata file
cat data/raw/train_images.csv.dvc
# Output:
# outs:
# - md5: 3a52ce780950d4d969792a2559cd519d
#   size: 285
#   hash: md5
#   path: train_images.csv

# DVC automatically adds data files to .gitignore
cat data/raw/.gitignore
# Output:
# /train_images.csv
# /test_images.csv

# Commit DVC metadata (NOT the data files!)
git add data/raw/*.dvc data/raw/.gitignore
git commit -m "data: add initial training and test datasets

Datasets:
- train_images.csv: 8 samples (4 cats, 4 dogs)
- test_images.csv: 4 samples (2 cats, 2 dogs)

Data versioned with DVC, tracked via metadata files.
Actual data stored in DVC remote."

# Push data to DVC remote storage
dvc push

# Output:
# Collecting stages from the workspace
# 2 files pushed
```

### Step 2.4: Update Dataset Version

```bash
# Simulate data collection - add more samples
cat >> data/raw/train_images.csv << 'EOF'
img_009,images/cat_005.jpg,cat,train
img_010,images/dog_005.jpg,dog,train
img_011,images/cat_006.jpg,cat,train
img_012,images/dog_006.jpg,dog,train
EOF

# Update DVC tracking
dvc add data/raw/train_images.csv

# Check what changed
git diff data/raw/train_images.csv.dvc
# Output shows MD5 hash changed:
# - md5: 3a52ce780950d4d969792a2559cd519d
# + md5: 8f14e45fceea167a5a36dedd4bea2543

# Commit dataset update
git add data/raw/train_images.csv.dvc
git commit -m "data: expand training set to 12 samples

Added 4 more training examples:
- 2 additional cat images
- 2 additional dog images

Total training samples: 12 (6 cats, 6 dogs)
Improves class balance and model generalization."

# Push new version to DVC storage
dvc push
```

### Step 2.5: Retrieve Old Dataset Versions

**Time travel your data!**

```bash
# View dataset history
git log --oneline -- data/raw/train_images.csv.dvc
# Output:
# abc1234 data: expand training set to 12 samples
# def5678 data: add initial training and test datasets

# Check out old version
git checkout def5678 data/raw/train_images.csv.dvc

# Pull that version's data from DVC
dvc checkout data/raw/train_images.csv

# View file (should be original 8 samples)
wc -l data/raw/train_images.csv
# Output: 9 data/raw/train_images.csv (8 data + 1 header)

# Return to latest
git checkout main data/raw/train_images.csv.dvc
dvc checkout

# Verify back to 12 samples
wc -l data/raw/train_images.csv
# Output: 13 data/raw/train_images.csv (12 data + 1 header)
```

---

## Part 3: Model Tracking with Git LFS

### Step 3.1: Configure Git LFS

```bash
# Track model file patterns with LFS
git lfs track "models/**/*.pth"      # PyTorch
git lfs track "models/**/*.h5"       # Keras/TensorFlow
git lfs track "models/**/*.pkl"      # Scikit-learn
git lfs track "models/**/*.onnx"     # ONNX
git lfs track "models/**/*.pb"       # TensorFlow SavedModel
git lfs track "models/**/*.joblib"   # Joblib serialized

# View LFS configuration
cat .gitattributes
# Output:
# models/**/*.pth filter=lfs diff=lfs merge=lfs -text
# models/**/*.h5 filter=lfs diff=lfs merge=lfs -text
# models/**/*.pkl filter=lfs diff=lfs merge=lfs -text
# models/**/*.onnx filter=lfs diff=lfs merge=lfs -text
# models/**/*.pb filter=lfs diff=lfs merge=lfs -text
# models/**/*.joblib filter=lfs diff=lfs merge=lfs -text

# Commit LFS configuration
git add .gitattributes
git commit -m "config: track model files with Git LFS

Configure LFS for model artifacts:
- PyTorch models (.pth)
- Keras/TensorFlow models (.h5, .pb)
- Scikit-learn models (.pkl, .joblib)
- ONNX models (.onnx)

Large binary files stored efficiently with LFS."
```

### Step 3.2: Version Model Checkpoints

```bash
# Create model directory
mkdir -p models/resnet50_classifier

# Simulate trained model file
cat > models/resnet50_classifier/model_v1.0.0.pth << 'EOF'
# Simulated PyTorch model checkpoint
# In real scenario, this would be actual model weights (can be 100s of MB)
MODEL_CHECKPOINT_V1.0.0
ARCHITECTURE: RESNET50
CLASSES: [cat, dog]
ACCURACY: 0.892
EOF

# Create model metadata (JSON, not tracked by LFS)
cat > models/resnet50_classifier/model_v1.0.0.json << 'EOF'
{
  "model_name": "resnet50_classifier",
  "version": "1.0.0",
  "architecture": "ResNet-50",
  "framework": "pytorch",
  "framework_version": "2.1.0",
  "created_at": "2025-01-15T14:30:00Z",
  "git_commit": "abc123def456",

  "training_config": {
    "dataset": "dogs_vs_cats",
    "dataset_version": "v2_2025-01-15",
    "data_dvc_commit": "def5678",
    "num_classes": 2,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "lr_scheduler": "cosine",
    "image_size": [224, 224],
    "augmentation": ["random_flip", "random_crop", "color_jitter"]
  },

  "metrics": {
    "train_accuracy": 0.952,
    "train_loss": 0.134,
    "val_accuracy": 0.908,
    "val_loss": 0.256,
    "test_accuracy": 0.892,
    "test_loss": 0.298,
    "f1_score": 0.886,
    "precision": 0.891,
    "recall": 0.882
  },

  "hardware": {
    "gpu": "NVIDIA V100",
    "training_time_hours": 2.3,
    "gpu_memory_mb": 8192
  },

  "model_size": {
    "parameters": 25557032,
    "file_size_mb": 97.5
  }
}
EOF

# Add model files (LFS handles .pth automatically)
git add models/
git commit -m "model: add ResNet-50 classifier v1.0.0

Initial model release:
- Architecture: ResNet-50 pretrained on ImageNet
- Task: Binary classification (cats vs dogs)
- Test accuracy: 89.2%
- F1 score: 0.886

Training details:
- Dataset: dogs_vs_cats v2 (12 training samples)
- 50 epochs, batch size 32
- Adam optimizer, LR 0.001
- Training time: 2.3 hours on V100

Model weights (97.5 MB) tracked with Git LFS.
Metadata tracked in Git for searchability."

# If remote configured, push model
# git push origin main
# Git LFS automatically uploads large files to LFS storage
```

### Step 3.3: Tag Model Releases

```bash
# Create annotated tag for model release
git tag -a model-v1.0.0 -m "Model Release v1.0.0: ResNet-50 Dogs vs Cats Classifier

## Model Details
- Architecture: ResNet-50 (pretrained ImageNet)
- Task: Binary classification
- Classes: cat, dog

## Performance
- Test accuracy: 89.2%
- F1 score: 0.886
- Inference time: ~15ms per image (V100)

## Training
- Dataset: dogs_vs_cats v2
- Data commit: def5678
- Training time: 2.3 hours
- Optimizer: Adam (LR 0.001)

## Deployment
- Compatible with PyTorch 2.0+
- Input: 224x224 RGB images
- Output: 2-class probabilities
- Model size: 97.5 MB

## Reproducibility
git checkout model-v1.0.0
git lfs pull
dvc checkout

pip install -r requirements.lock
python inference.py --model models/resnet50_classifier/model_v1.0.0.pth"

# View tag
git show model-v1.0.0 --stat

# List all model tags
git tag -l "model-*"
# Output:
# model-v1.0.0
```

---

## Part 4: Experiment Management

### Step 4.1: Version Experiment Configurations

```bash
# Create experiments directory
mkdir -p experiments

# Experiment 1: Baseline
cat > experiments/exp-001-baseline-resnet50.yaml << 'EOF'
experiment:
  id: "exp-001"
  name: "baseline-resnet50"
  description: "Baseline ResNet-50 with standard hyperparameters"
  created_at: "2025-01-15T10:00:00Z"
  created_by: "ml-engineer@company.com"
  parent_experiment: null

model:
  architecture: "resnet50"
  pretrained: true
  pretrained_source: "imagenet"
  num_classes: 2

data:
  dataset: "dogs_vs_cats"
  version: "v2_2025-01-15"
  dvc_commit: "def5678"
  train_samples: 12
  test_samples: 4
  train_split: 0.75
  val_split: 0.25
  class_balance: {cat: 6, dog: 6}

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  weight_decay: 0.0001
  lr_scheduler: "cosine"
  early_stopping_patience: 10

augmentation:
  random_flip: true
  random_crop: true
  color_jitter: true
  jitter_brightness: 0.2
  jitter_contrast: 0.2
  normalize: true

hardware:
  device: "cuda"
  gpu_type: "V100"
  num_gpus: 1

results:
  train_accuracy: 0.952
  val_accuracy: 0.908
  test_accuracy: 0.892
  train_loss: 0.134
  val_loss: 0.256
  test_loss: 0.298
  f1_score: 0.886
  training_time_hours: 2.3
  converged_epoch: 42
EOF

git add experiments/exp-001-baseline-resnet50.yaml
git commit -m "experiment: baseline ResNet-50 classifier

Experiment ID: exp-001
Test accuracy: 89.2%
Training time: 2.3 hours

Baseline model for comparison with future experiments."

# Experiment 2: Higher learning rate
cat > experiments/exp-002-higher-lr.yaml << 'EOF'
experiment:
  id: "exp-002"
  name: "resnet50-higher-lr"
  description: "ResNet-50 with increased learning rate"
  created_at: "2025-01-16T09:00:00Z"
  created_by: "ml-engineer@company.com"
  parent_experiment: "exp-001"

model:
  architecture: "resnet50"
  pretrained: true
  pretrained_source: "imagenet"
  num_classes: 2

data:
  dataset: "dogs_vs_cats"
  version: "v2_2025-01-15"
  dvc_commit: "def5678"
  train_samples: 12
  test_samples: 4
  train_split: 0.75
  val_split: 0.25

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.003  # CHANGED: 0.001 → 0.003
  optimizer: "adam"
  weight_decay: 0.0001
  lr_scheduler: "cosine"
  early_stopping_patience: 10

augmentation:
  random_flip: true
  random_crop: true
  color_jitter: true
  jitter_brightness: 0.2
  jitter_contrast: 0.2
  normalize: true

hardware:
  device: "cuda"
  gpu_type: "V100"
  num_gpus: 1

results:
  train_accuracy: 0.971
  val_accuracy: 0.924
  test_accuracy: 0.908
  train_loss: 0.098
  val_loss: 0.221
  test_loss: 0.267
  f1_score: 0.902
  training_time_hours: 2.1
  converged_epoch: 38

comparison_to_baseline:
  accuracy_improvement: +0.016  # +1.6%
  training_time_improvement: -0.2h
  conclusion: "Higher LR improves accuracy and converges faster"
EOF

git add experiments/exp-002-higher-lr.yaml
git commit -m "experiment: higher learning rate improves performance

Experiment ID: exp-002
Test accuracy: 90.8% (+1.6% vs baseline)
Training time: 2.1 hours (-0.2h vs baseline)

Change: Learning rate 0.001 → 0.003
Result: Faster convergence, better accuracy"
```

### Step 4.2: Branch Per Experiment

```bash
# Create experiment branch
git switch -c experiment/advanced-augmentation

# Experiment 3: Advanced data augmentation
cat > experiments/exp-003-advanced-augmentation.yaml << 'EOF'
experiment:
  id: "exp-003"
  name: "resnet50-advanced-augmentation"
  description: "ResNet-50 with advanced augmentation strategies"
  created_at: "2025-01-17T11:00:00Z"
  created_by: "ml-engineer@company.com"
  parent_experiment: "exp-002"

model:
  architecture: "resnet50"
  pretrained: true
  pretrained_source: "imagenet"
  num_classes: 2

data:
  dataset: "dogs_vs_cats"
  version: "v2_2025-01-15"
  dvc_commit: "def5678"
  train_samples: 12
  test_samples: 4

training:
  epochs: 60  # Increased for augmentation
  batch_size: 32
  learning_rate: 0.003
  optimizer: "adam"
  weight_decay: 0.0001
  lr_scheduler: "cosine"

augmentation:
  # Standard augmentations
  random_flip: true
  random_crop: true
  color_jitter: true
  jitter_brightness: 0.2
  jitter_contrast: 0.2
  normalize: true
  # Advanced augmentations (NEW)
  random_rotation: true
  rotation_degrees: 15
  random_affine: true
  cutout: true
  cutout_size: 16
  mixup: true
  mixup_alpha: 0.2

results:
  train_accuracy: 0.989
  val_accuracy: 0.941
  test_accuracy: 0.924
  f1_score: 0.919
  training_time_hours: 2.8

comparison_to_exp_002:
  accuracy_improvement: +0.016  # +1.6%
  conclusion: "Advanced augmentation reduces overfitting, improves generalization"
EOF

git add experiments/exp-003-advanced-augmentation.yaml
git commit -m "experiment: advanced augmentation boosts accuracy

Experiment ID: exp-003
Test accuracy: 92.4% (+1.6% vs exp-002)

Added augmentations:
- Random rotation (±15°)
- Random affine transformations
- Cutout regularization
- Mixup (alpha=0.2)

Trade-off: +0.7h training time, +2.4% accuracy"

# If experiment successful, merge to main
git switch main
git merge experiment/advanced-augmentation --no-ff -m "Merge experiment: advanced augmentation

Experiment exp-003 shows significant improvement.
Integrating advanced augmentation as new baseline."

# Tag successful experiment
git tag -a exp-003-success -m "Successful Experiment: Advanced Augmentation

Best model so far: 92.4% test accuracy"
```

---

## Part 5: Reproducibility

### Step 5.1: Lock Dependencies

```bash
# Create comprehensive requirements
cat > requirements.txt << 'EOF'
# Core ML Framework
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# Data Processing
numpy==1.26.2
pandas==2.1.4
pillow==10.1.0
opencv-python==4.8.1.78

# ML Tools
scikit-learn==1.3.2
scipy==1.11.4

# Experiment Tracking
mlflow==2.9.2
wandb==0.16.1

# Version Control
dvc[s3]==3.36.1
gitpython==3.1.40

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# Utilities
pyyaml==6.0.1
python-dotenv==1.0.0
tqdm==4.66.1
click==8.1.7

# Code Quality
black==23.12.1
flake8==6.1.0
mypy==1.7.1
pytest==7.4.3

# Pre-commit Hooks
pre-commit==3.6.0
EOF

# Install and lock exact versions
pip install -r requirements.txt
pip freeze > requirements.lock

git add requirements.txt requirements.lock
git commit -m "deps: lock dependency versions for reproducibility

requirements.txt: High-level dependencies
requirements.lock: Exact versions (pip freeze output)

Ensures identical environments across:
- Development machines
- CI/CD pipelines
- Production deployments
- Experiment reproduction"

# Create conda environment spec
cat > environment.yaml << 'EOF'
name: ml-image-classifier
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11.7
  - pytorch=2.1.0
  - torchvision=0.16.0
  - numpy=1.26.2
  - pandas=2.1.4
  - pillow=10.1.0
  - scikit-learn=1.3.2
  - matplotlib=3.8.2
  - pyyaml=6.0.1
  - pip=23.3.2
  - pip:
    - mlflow==2.9.2
    - wandb==0.16.1
    - dvc[s3]==3.36.1
    - pre-commit==3.6.0
EOF

git add environment.yaml
git commit -m "deps: add conda environment specification

Enables conda users to create identical environment:
conda env create -f environment.yaml"
```

### Step 5.2: Document Reproducibility

```bash
# Create comprehensive reproduction guide
mkdir -p docs
cat > docs/REPRODUCING_EXPERIMENTS.md << 'EOF'
# Reproducing Experiments

## Prerequisites

### 1. Environment Setup
```bash
# Option A: pip
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.lock

# Option B: conda
conda env create -f environment.yaml
conda activate ml-image-classifier
```

### 2. Data and Model Setup
```bash
# Clone repository
git clone <repo-url>
cd ml-image-classifier

# Pull DVC data
dvc pull

# Pull Git LFS files
git lfs pull
```

## Reproducing Specific Experiments

### Experiment 001: Baseline
```bash
# Checkout experiment commit
git checkout <exp-001-commit-hash>

# Get exact data version
dvc checkout

# Install exact dependencies
pip install -r requirements.lock

# Run training
python train.py --config experiments/exp-001-baseline-resnet50.yaml

# Expected results:
# - Test accuracy: ~89.2%
# - Training time: ~2.3 hours (V100 GPU)
# - Model saved to: models/resnet50_classifier/exp-001/
```

### Experiment 002: Higher Learning Rate
```bash
git checkout <exp-002-commit-hash>
dvc checkout
python train.py --config experiments/exp-002-higher-lr.yaml

# Expected: Test accuracy ~90.8%
```

### Experiment 003: Advanced Augmentation
```bash
git checkout exp-003-success  # Using tag
dvc checkout
python train.py --config experiments/exp-003-advanced-augmentation.yaml

# Expected: Test accuracy ~92.4%
```

## Reproducing Model Releases

### Model v1.0.0
```bash
# Checkout model tag
git checkout model-v1.0.0

# Pull model weights
git lfs pull

# Get training data version
dvc checkout

# Run inference
python inference.py \
  --model models/resnet50_classifier/model_v1.0.0.pth \
  --image test_image.jpg
```

## Troubleshooting

### "DVC file not found"
```bash
# Ensure DVC is configured
dvc remote list
dvc pull -v  # Verbose output
```

### "Git LFS file pointer instead of actual file"
```bash
git lfs pull
# Or for specific file:
git lfs pull --include="models/**/*.pth"
```

### "Different results from paper"
Possible causes:
1. Random seed not set
2. Different hardware (GPU vs CPU)
3. Different CUDA/cuDNN versions
4. Different dataset version

### "Missing dependencies"
```bash
# Use locked requirements
pip install -r requirements.lock

# Not requirements.txt (may get newer versions)
```

## Verifying Reproduction

### Checksum Verification
```bash
# Verify model file
sha256sum models/resnet50_classifier/model_v1.0.0.pth
# Should match: abc123...

# Verify data file
dvc status data/
# Should show: Data and pipelines are up to date
```

### Metric Verification
Expected metrics (±0.5% due to randomness):
- Experiment 001: 89.2% test accuracy
- Experiment 002: 90.8% test accuracy
- Experiment 003: 92.4% test accuracy

### Environment Verification
```bash
python --version  # Should be 3.11.7
torch.__version__  # Should be 2.1.0
git log -1 --oneline  # Verify commit
dvc version  # Should be 3.36.1
```
EOF

git add docs/REPRODUCING_EXPERIMENTS.md
git commit -m "docs: add comprehensive experiment reproduction guide

Document how to reproduce:
- All experiments (001-003)
- Model releases
- Training environments

Includes:
- Step-by-step instructions
- Expected results
- Troubleshooting tips
- Verification procedures"
```

---

## Part 6: ML-Specific Git Hooks

### Step 6.1: Install and Configure Pre-Commit

```bash
# Install pre-commit framework
pip install pre-commit

# Create pre-commit configuration
cat > .pre-commit-config.yaml << 'EOF'
repos:
  # Standard code quality checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']  # Prevent committing large files to Git
      - id: check-merge-conflict
      - id: detect-private-key
      - id: mixed-line-ending

  # Python code formatting
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11
        args: ['--line-length=88']

  # Python linting
  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']

  # Import sorting
  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ['--profile=black']

  # ML-specific custom hooks
  - repo: local
    hooks:
      # Validate model metadata
      - id: validate-model-metadata
        name: Validate model metadata JSON files
        entry: python scripts/hooks/validate_model_metadata.py
        language: python
        files: 'models/.*/.*\.json$'
        pass_filenames: true

      # Validate experiment configs
      - id: validate-experiment-config
        name: Validate experiment YAML configs
        entry: python scripts/hooks/validate_experiment_config.py
        language: python
        files: 'experiments/.*\.yaml$'
        pass_filenames: true

      # Check DVC files
      - id: check-dvc-files
        name: Verify DVC files are committed
        entry: python scripts/hooks/check_dvc_files.py
        language: python
        files: '\.dvc$'
        pass_filenames: true

      # Prevent model weights in Git
      - id: prevent-model-in-git
        name: Prevent model weights in Git (use LFS)
        entry: python scripts/hooks/prevent_model_in_git.py
        language: python
        files: '\.(pth|h5|pkl|onnx|pb)$'
        pass_filenames: true
EOF

git add .pre-commit-config.yaml
git commit -m "ci: add pre-commit hooks configuration"

# Create hook scripts
mkdir -p scripts/hooks

# Hook 1: Validate model metadata
cat > scripts/hooks/validate_model_metadata.py << 'EOF'
#!/usr/bin/env python3
"""Validate model metadata JSON files."""
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = [
    "model_name", "version", "architecture", "framework",
    "created_at", "training_config", "metrics"
]

def validate_metadata(filepath):
    """Validate model metadata structure."""
    try:
        with open(filepath) as f:
            metadata = json.load(f)

        # Check required fields
        missing = [f for f in REQUIRED_FIELDS if f not in metadata]
        if missing:
            print(f"❌ {filepath}: Missing required fields: {missing}")
            return False

        # Validate version format (X.Y.Z)
        version = metadata.get("version", "")
        if not version or len(version.split(".")) != 3:
            print(f"❌ {filepath}: Invalid version format. Use X.Y.Z")
            return False

        # Check metrics exist
        metrics = metadata.get("metrics", {})
        if not metrics.get("test_accuracy"):
            print(f"❌ {filepath}: Missing test_accuracy in metrics")
            return False

        print(f"✅ {filepath}: Valid")
        return True

    except json.JSONDecodeError as e:
        print(f"❌ {filepath}: Invalid JSON - {e}")
        return False

if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        sys.exit(0)

    all_valid = all(validate_metadata(f) for f in files)
    sys.exit(0 if all_valid else 1)
EOF

# Hook 2: Validate experiment configs
cat > scripts/hooks/validate_experiment_config.py << 'EOF'
#!/usr/bin/env python3
"""Validate experiment configuration YAML files."""
import yaml
import sys

REQUIRED_SECTIONS = ["experiment", "model", "data", "training", "results"]

def validate_experiment(filepath):
    """Validate experiment YAML structure."""
    try:
        with open(filepath) as f:
            config = yaml.safe_load(f)

        # Check required sections
        missing = [s for s in REQUIRED_SECTIONS if s not in config]
        if missing:
            print(f"❌ {filepath}: Missing sections: {missing}")
            return False

        # Check experiment ID format
        exp = config.get("experiment", {})
        exp_id = exp.get("id")
        if not exp_id or not exp_id.startswith("exp-"):
            print(f"❌ {filepath}: Invalid experiment ID. Use exp-XXX format")
            return False

        # Check results exist
        results = config.get("results", {})
        if not results.get("test_accuracy"):
            print(f"❌ {filepath}: Missing test_accuracy in results")
            return False

        print(f"✅ {filepath}: Valid")
        return True

    except yaml.YAMLError as e:
        print(f"❌ {filepath}: Invalid YAML - {e}")
        return False

if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        sys.exit(0)

    all_valid = all(validate_experiment(f) for f in files)
    sys.exit(0 if all_valid else 1)
EOF

# Make hooks executable
chmod +x scripts/hooks/*.py

git add scripts/hooks/
git commit -m "ci: add ML-specific validation hooks

Custom hooks:
- validate_model_metadata.py: Check model JSON structure
- validate_experiment_config.py: Check experiment YAML format

Ensures metadata quality before commits."

# Install pre-commit hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

---

## Verification Checklist

After completing all ML workflow exercises:

- [ ] DVC initialized and configured
- [ ] Dataset tracked with DVC
- [ ] Can retrieve old dataset versions
- [ ] Git LFS configured for models
- [ ] Model checkpoint tracked with LFS
- [ ] Model release tagged (model-v1.0.0)
- [ ] Experiments documented in YAML
- [ ] Dependencies locked (requirements.lock)
- [ ] Reproduction guide written
- [ ] Pre-commit hooks installed and working
- [ ] Can reproduce experiments from commits

---

## Summary

You've mastered ML-specific Git workflows:

- ✅ Data versioning with DVC
- ✅ Model tracking with Git LFS
- ✅ Experiment configuration management
- ✅ Reproducible training pipelines
- ✅ Model release tagging
- ✅ Dependency locking
- ✅ ML-specific Git hooks
- ✅ Complete model and data provenance

**Key Takeaways:**
- Version code, data, and models together
- DVC for large datasets, LFS for models
- Experiment configs in Git enable reproducibility
- Lock dependencies for identical environments
- Pre-commit hooks prevent mistakes

**Time to Complete:** ~120 minutes

**Next Exercise:** Exercise 07 - Advanced Git Techniques
