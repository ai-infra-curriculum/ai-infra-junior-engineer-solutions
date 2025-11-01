#!/bin/bash

#######################################################################
# Exercise 06: ML Workflows - Setup Script
#######################################################################
# Creates a complete ML project demonstrating:
# - DVC-style data versioning (simulated)
# - Git LFS configuration
# - Experiment tracking
# - Reproducibility practices
#######################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$EXERCISE_DIR/ml-classification-project"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setting up ML Project${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Clean up existing project
if [ -d "$PROJECT_DIR" ]; then
    rm -rf "$PROJECT_DIR"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
git init
git config user.name "ML Engineer"
git config user.email "ml@example.com"

echo -e "${YELLOW}[1/8] Creating project structure...${NC}"

mkdir -p {data/raw,data/processed,models/resnet50,experiments,scripts,configs,.dvc}

#######################################################################
# Git LFS Configuration
#######################################################################

echo -e "${YELLOW}[2/8] Configuring Git LFS...${NC}"

cat > .gitattributes << 'EOF'
# Git LFS Configuration for ML Project

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
EOF

#######################################################################
# DVC-Style Configuration (Simulated)
#######################################################################

echo -e "${YELLOW}[3/8] Setting up data versioning...${NC}"

cat > .dvc/config << 'EOF'
[core]
    remote = local
    autostage = true
['remote "local"']
    url = /tmp/dvc-storage
['remote "s3"']
    url = s3://my-ml-project/dvc-storage
    region = us-west-2
EOF

cat > .dvcignore << 'EOF'
# DVC ignore patterns
*.pyc
__pycache__/
.git/
.dvc/cache/
EOF

# Create sample dataset
cat > data/raw/train.csv << 'EOF'
id,feature1,feature2,feature3,label
1,0.5,0.3,0.8,0
2,0.8,0.6,0.9,1
3,0.2,0.4,0.3,0
4,0.9,0.7,0.85,1
5,0.3,0.2,0.4,0
EOF

cat > data/raw/test.csv << 'EOF'
id,feature1,feature2,feature3,label
101,0.6,0.4,0.7,1
102,0.7,0.5,0.8,1
103,0.3,0.3,0.4,0
EOF

# Create DVC-style metadata files
cat > data/raw/train.csv.dvc << 'EOF'
outs:
- md5: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
  size: 145
  path: train.csv
EOF

cat > data/raw/test.csv.dvc << 'EOF'
outs:
- md5: q1w2e3r4t5y6u7i8o9p0a1s2d3f4g5h6
  size: 98
  path: test.csv
EOF

#######################################################################
# Experiment Configurations
#######################################################################

echo -e "${YELLOW}[4/8] Creating experiment configurations...${NC}"

cat > experiments/exp-001-baseline.yaml << 'EOF'
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
EOF

cat > experiments/exp-002-higher-lr.yaml << 'EOF'
experiment:
  id: "exp-002"
  name: "resnet50-higher-lr"
  description: "ResNet-50 with higher initial learning rate"
  parent_experiment: "exp-001"
  created_at: "2024-01-16T09:00:00Z"
  git_commit: ""

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
  learning_rate: 0.3  # CHANGED: increased from 0.1
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
  train_accuracy: 0.891
  val_accuracy: 0.867
  test_accuracy: 0.862
  train_loss: 0.298
  val_loss: 0.356
  training_time_hours: 47.2
  best_epoch: 84

improvements:
  vs_baseline: "+1.4% accuracy"
  notes: "Higher LR improved convergence speed"
EOF

#######################################################################
# Model Metadata
#######################################################################

echo -e "${YELLOW}[5/8] Creating model metadata...${NC}"

cat > models/resnet50/model_v1.0.0.json << 'EOF'
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
EOF

# Simulate model file (small dummy file)
echo "model_weights_v1.0.0" > models/resnet50/model_v1.0.0.pth

#######################################################################
# Dependencies
#######################################################################

echo -e "${YELLOW}[6/8] Creating dependency files...${NC}"

cat > requirements.txt << 'EOF'
# Core ML Framework
torch==2.1.0
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
EOF

cat > environment.yaml << 'EOF'
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
EOF

#######################################################################
# Validation Scripts
#######################################################################

echo -e "${YELLOW}[7/8] Creating validation scripts...${NC}"

cat > scripts/check_model_metadata.py << 'EOF'
#!/usr/bin/env python3
"""Validate model metadata files."""

import json
import sys
from pathlib import Path

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

    missing = [field for field in REQUIRED_FIELDS if field not in metadata]

    if missing:
        print(f"Error in {filepath}: Missing fields: {missing}")
        return False

    # Check version format
    version = metadata.get("version", "")
    if not version or len(version.split(".")) != 3:
        print(f"Error in {filepath}: Invalid version format. Use X.Y.Z")
        return False

    print(f"✓ {filepath} is valid")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_model_metadata.py <metadata_file>")
        sys.exit(1)

    files = sys.argv[1:]
    all_valid = all(validate_metadata(f) for f in files)
    sys.exit(0 if all_valid else 1)
EOF

cat > scripts/validate_experiment.py << 'EOF'
#!/usr/bin/env python3
"""Validate experiment configuration files."""

import yaml
import sys

REQUIRED_SECTIONS = ["experiment", "model", "data", "training"]

def validate_experiment(filepath):
    """Validate experiment YAML."""
    with open(filepath) as f:
        config = yaml.safe_load(f)

    missing = [s for s in REQUIRED_SECTIONS if s not in config]

    if missing:
        print(f"Error in {filepath}: Missing sections: {missing}")
        return False

    # Check experiment ID format
    exp_id = config.get("experiment", {}).get("id")
    if not exp_id or not exp_id.startswith("exp-"):
        print(f"Error in {filepath}: Invalid experiment ID format. Use exp-XXX")
        return False

    print(f"✓ {filepath} is valid")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_experiment.py <experiment_file>")
        sys.exit(1)

    files = sys.argv[1:]
    all_valid = all(validate_experiment(f) for f in files)
    sys.exit(0 if all_valid else 1)
EOF

chmod +x scripts/*.py

#######################################################################
# Git Commits
#######################################################################

echo -e "${YELLOW}[8/8] Creating Git history...${NC}"

# Create .gitignore first
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
.pytest_cache/

# Data (tracked by DVC)
/data/raw/train.csv
/data/raw/test.csv
/data/processed/

# Models (tracked by Git LFS)
# .pth files are in .gitattributes

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# ML
mlruns/
wandb/
.hydra/

# DVC
/dvc.lock
EOF

# Initial commit
git add .gitignore .dvcignore .dvc/ .gitattributes
git commit -m "init: initialize ML project with DVC and Git LFS

Configure version control for ML workflows:
- Git LFS for model files (.pth, .h5, .onnx, .pkl)
- DVC for dataset versioning
- Ignore patterns for Python and ML artifacts"

# Add DVC configuration
git add .dvc/ .dvcignore
git commit -m "config: configure DVC for data versioning

DVC remotes:
- local: /tmp/dvc-storage (for local development)
- s3: s3://my-ml-project/dvc-storage (for production)"

# Add datasets
git add data/raw/*.dvc
git commit -m "data: add initial training and test datasets

Datasets tracked with DVC:
- train.csv: 5 samples
- test.csv: 3 samples

Run 'dvc pull' to download actual data files."

# Add experiments
git add experiments/exp-001-baseline.yaml
git commit -m "experiment: baseline ResNet-50 (exp-001)

Results:
- Test accuracy: 84.8%
- Training time: 48.5 hours on 4x GPU
- Configuration: Default hyperparameters"

git add experiments/exp-002-higher-lr.yaml
git commit -m "experiment: higher learning rate (exp-002)

Results:
- Test accuracy: 86.2% (+1.4% vs baseline)
- Training time: 47.2 hours
- Change: LR 0.1 → 0.3

Improved convergence and final accuracy."

# Add model
git add models/resnet50/
git commit -m "model: add ResNet-50 v1.0.0

Model checkpoint from exp-001:
- Architecture: ResNet-50
- Test accuracy: 84.8%
- Framework: PyTorch 2.1.0
- Weights tracked with Git LFS"

# Tag model
git tag -a model-v1.0.0 -m "Model Release v1.0.0

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
- Output: 1000 classes"

# Add dependencies
git add requirements.txt environment.yaml
git commit -m "deps: add dependency specifications

Added:
- requirements.txt: Pip dependencies
- environment.yaml: Conda environment

Pinned versions for reproducibility."

# Add validation scripts
git add scripts/
git commit -m "ci: add validation scripts for ML artifacts

Scripts:
- check_model_metadata.py: Validate model JSON
- validate_experiment.py: Validate experiment YAML

Ensures consistency and completeness of ML artifacts."

echo -e "${GREEN}✓ ML project setup complete!${NC}"
echo ""
echo "Project structure:"
tree -L 2 -I '__pycache__|*.pyc' . || ls -la

echo ""
echo "Git history:"
git log --oneline --graph

echo ""
echo "Git tags:"
git tag

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  cd $PROJECT_DIR"
echo "  git log --oneline"
echo "  cat experiments/exp-001-baseline.yaml"
echo "  python scripts/validate_experiment.py experiments/*.yaml"
