# Implementation Guide - Exercise 01: Environment Setup and Management

This guide provides step-by-step instructions for implementing this exercise from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Part 1: Project Structure Setup](#part-1-project-structure-setup)
3. [Part 2: Virtual Environment Management](#part-2-virtual-environment-management)
4. [Part 3: Dependency Management](#part-3-dependency-management)
5. [Part 4: Environment Variables](#part-4-environment-variables)
6. [Part 5: Automation Scripts](#part-5-automation-scripts)
7. [Part 6: Documentation](#part-6-documentation)
8. [Verification](#verification)
9. [Extension Challenges](#extension-challenges)

---

## Prerequisites

**Required:**
- Python 3.11 or higher installed
- Terminal/command line access
- Text editor or IDE
- Git (optional but recommended)

**Check Python version:**
```bash
python3 --version  # Should show 3.11 or higher
```

**If Python 3.11+ not installed:**
```bash
# Ubuntu/Debian:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv

# macOS:
brew install python@3.11

# Windows:
# Download from python.org
```

---

## Part 1: Project Structure Setup

**Time Estimate:** 15 minutes

### Step 1.1: Create Project Directory

```bash
# Create main project directory
mkdir sentiment-classifier
cd sentiment-classifier

# Create subdirectories
mkdir -p src/utils tests data models configs docs

# Verify structure
tree -L 2  # or: ls -R
```

**Expected output:**
```
sentiment-classifier/
├── configs/
├── data/
├── docs/
├── models/
├── src/
│   └── utils/
└── tests/
```

### Step 1.2: Create .gitignore

```bash
# Create .gitignore file
cat > .gitignore <<'EOF'
# Python Environment
venv/
env/
ENV/
.venv/
.Python
*.pyc
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/
.eggs/

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# ML artifacts
data/*
!data/.gitkeep
models/*
!models/.gitkeep
*.pkl
*.pth
*.h5
*.onnx

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Logs
*.log
logs/

# OS
Thumbs.db
EOF
```

### Step 1.3: Create .gitkeep Files

```bash
# Create .gitkeep to track empty directories
touch data/.gitkeep
touch models/.gitkeep

# Verify
ls -la data/ models/
```

---

## Part 2: Virtual Environment Management

**Time Estimate:** 20 minutes

### Step 2.1: Create Virtual Environment

```bash
# Create venv
python3.11 -m venv venv

# Verify creation
ls -la venv/
```

**Expected directories:**
- `venv/bin/` - Executables (python, pip, activate)
- `venv/lib/` - Python packages
- `venv/include/` - C headers
- `venv/pyvenv.cfg` - Configuration

### Step 2.2: Activate Virtual Environment

```bash
# Activate (Linux/macOS)
source venv/bin/activate

# OR activate (Windows)
venv\Scripts\activate

# Verify activation
which python  # Should show venv/bin/python
python --version  # Should show 3.11+
```

**Your prompt should now show `(venv)` prefix.**

### Step 2.3: Upgrade pip

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Verify
pip --version
```

### Step 2.4: Test Virtual Environment Isolation

```bash
# Install a package in venv
pip install requests

# Verify it's in venv
pip show requests
# Location: .../venv/lib/python3.11/site-packages

# Deactivate and check
deactivate
python -c "import requests"  # Should fail (ModuleNotFoundError)

# Reactivate
source venv/bin/activate
python -c "import requests"  # Should work
```

**✓ Checkpoint:** Virtual environment created, activated, and isolated.

---

## Part 3: Dependency Management

**Time Estimate:** 30 minutes

### Step 3.1: Create requirements.txt

```bash
# Create production requirements file
cat > requirements.txt <<'EOF'
# Production Dependencies
# All versions pinned for reproducibility

# Deep Learning
torch==2.1.0
transformers==4.35.0

# Data Processing
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.2

# Configuration & Environment
python-dotenv==1.0.0
pyyaml==6.0.1

# Utilities
tqdm==4.66.1
EOF
```

### Step 3.2: Create requirements-dev.txt

```bash
# Create development requirements file
cat > requirements-dev.txt <<'EOF'
# Development Dependencies
# Includes all production dependencies plus development tools

-r requirements.txt

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0

# Code Quality
black==23.11.0
mypy==1.7.1
flake8==6.1.0
isort==5.12.0

# Type Stubs
types-PyYAML==6.0.12.12
EOF
```

### Step 3.3: Install Dependencies

```bash
# Install development dependencies (includes production)
pip install -r requirements-dev.txt

# This may take 5-10 minutes for PyTorch
```

**While installing, read about each package:**
- **torch**: PyTorch deep learning framework
- **transformers**: HuggingFace transformers library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **python-dotenv**: Load environment variables from .env
- **pyyaml**: YAML file parsing
- **pytest**: Testing framework
- **black**: Code formatter
- **mypy**: Type checker

### Step 3.4: Verify Installation

```bash
# Check installed packages
pip list

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

### Step 3.5: Freeze Dependencies

```bash
# Generate complete dependency list (including transitive)
pip freeze > requirements-frozen.txt

# View the file
wc -l requirements-frozen.txt  # Should have 50+ packages
```

**✓ Checkpoint:** All dependencies installed and verified.

---

## Part 4: Environment Variables

**Time Estimate:** 25 minutes

### Step 4.1: Create .env.example

```bash
# Create environment variable template
cat > .env.example <<'EOF'
# Sentiment Classifier Configuration
# Copy this file to .env and configure with your settings
# NEVER commit .env to version control!

# Model Configuration
MODEL_NAME=distilbert-base-uncased
MAX_LENGTH=128

# Data Paths
DATA_PATH=./data/sentiment_data.csv
MODEL_OUTPUT_PATH=./models

# Training Parameters
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=3
WARMUP_STEPS=500

# Reproducibility
RANDOM_SEED=42

# Device Configuration
# Options: cuda, cpu, mps (for Apple Silicon)
DEVICE=cuda

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/training.log

# MLflow Tracking (Optional)
# MLFLOW_TRACKING_URI=http://localhost:5000
# MLFLOW_EXPERIMENT_NAME=sentiment-classifier

# HuggingFace Hub (Optional)
# HF_TOKEN=your_huggingface_token_here
EOF
```

### Step 4.2: Create .env from Template

```bash
# Copy template to .env
cp .env.example .env

# Edit with your settings (optional for now)
# nano .env
```

### Step 4.3: Test Environment Variable Loading

```bash
# Create test script
cat > test_env.py <<'EOF'
from dotenv import load_dotenv
import os

load_dotenv()

print("Environment Variables:")
print(f"MODEL_NAME: {os.getenv('MODEL_NAME')}")
print(f"BATCH_SIZE: {os.getenv('BATCH_SIZE')}")
print(f"RANDOM_SEED: {os.getenv('RANDOM_SEED')}")
EOF

# Run test
python test_env.py

# Clean up
rm test_env.py
```

**Expected output:**
```
Environment Variables:
MODEL_NAME: distilbert-base-uncased
BATCH_SIZE: 32
RANDOM_SEED: 42
```

**✓ Checkpoint:** Environment variables configured and loading correctly.

---

## Part 5: Automation Scripts

**Time Estimate:** 30 minutes

### Step 5.1: Create setup.sh

See the complete `sentiment-classifier/setup.sh` file in the solution.

**Key features to include:**
1. Python version check
2. Virtual environment creation
3. pip upgrade
4. Dependency installation
5. .env file setup
6. Directory creation
7. User feedback

```bash
# Make executable
chmod +x setup.sh

# Test it (from parent directory)
cd ..
rm -rf sentiment-classifier/venv  # Clean for testing
cd sentiment-classifier
./setup.sh
```

### Step 5.2: Create Source Code Files

**Create package structure:**
```bash
# src/__init__.py
touch src/__init__.py
touch src/utils/__init__.py

# Main scripts (see solution for complete code)
touch src/train.py
touch src/evaluate.py

# Utility modules
touch src/utils/data_loader.py
touch src/utils/metrics.py
```

### Step 5.3: Create Test Files

```bash
# Test package
touch tests/__init__.py
touch tests/test_data_loader.py
touch tests/test_metrics.py
```

### Step 5.4: Create Configuration Files

```bash
# Training configuration
cat > configs/training_config.yaml <<'EOF'
# Training Configuration
model_name: distilbert-base-uncased
max_length: 128
data_path: ./data/sentiment_data.csv
test_size: 0.2
batch_size: 32
learning_rate: 2.0e-5
num_epochs: 3
warmup_steps: 500
weight_decay: 0.01
random_seed: 42
output_dir: ./models
EOF

# Model configuration
cat > configs/model_config.yaml <<'EOF'
# Model Architecture Configuration
model_type: distilbert
pretrained_model: distilbert-base-uncased
num_labels: 2
dropout_rate: 0.1
max_sequence_length: 128
padding_strategy: max_length
truncation: true
EOF
```

**✓ Checkpoint:** Project structure complete with automation.

---

## Part 6: Documentation

**Time Estimate:** 20 minutes

### Step 6.1: Create Project README

```bash
# Create README.md (see solution for complete version)
cat > README.md <<'EOF'
# Sentiment Classifier - ML Training Project

A production-ready sentiment analysis project.

## Quick Start

```bash
./setup.sh
source venv/bin/activate
python src/train.py
```

## Features

- Virtual environment isolation
- Pinned dependencies
- Environment configuration
- Automated testing
- Type hints

## Training

```bash
python src/train.py --config configs/training_config.yaml
```

## Testing

```bash
pytest tests/ -v
```
EOF
```

### Step 6.2: Create Documentation Files

```bash
# Setup notes
touch docs/SETUP_NOTES.md

# Answers to exercise questions
touch docs/ANSWERS.md
```

See solution files for complete content.

---

## Verification

### Step 7.1: Create Verification Script

Create `scripts/verify_setup.py` (see solution for complete code).

```bash
# Make executable
chmod +x scripts/verify_setup.py

# Run verification
python scripts/verify_setup.py
```

**Expected output:**
```
================================
Setup Verification
================================

✓ Python Version: PASSED
✓ Virtual Environment: PASSED
✓ Required Packages: PASSED
✓ Project Structure: PASSED
✓ Environment Variables: PASSED
✓ Directory Permissions: PASSED

All checks passed! (6/6)
Setup is complete and ready for training.
```

### Step 7.2: Test Code Quality

```bash
# Run linting
black src/ tests/ --check
mypy src/

# Run tests
pytest tests/ -v --cov=src

# Check coverage
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Step 7.3: Test Training (Optional)

```bash
# Create dummy data
cat > data/sentiment_data.csv <<'EOF'
text,label
"This is great!",1
"Terrible experience.",0
"Absolutely wonderful!",1
"Very disappointed.",0
"Excellent service!",1
EOF

# Run training (will take a few minutes)
python src/train.py --config configs/training_config.yaml
```

---

## Extension Challenges

### Challenge 1: Docker Integration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/train.py"]
```

```bash
# Build and run
docker build -t sentiment-classifier .
docker run --rm sentiment-classifier
```

### Challenge 2: Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

```bash
pip install pre-commit
pre-commit install
```

### Challenge 3: GitHub Actions CI/CD

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    - name: Run tests
      run: pytest tests/ -v --cov=src
```

### Challenge 4: pip-tools Integration

```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in (high-level dependencies)
cat > requirements.in <<'EOF'
torch>=2.0
transformers>=4.30
pandas>=2.0
EOF

# Compile to requirements.txt
pip-compile requirements.in

# Install
pip-sync requirements.txt
```

### Challenge 5: Poetry Migration

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Initialize
poetry init

# Add dependencies
poetry add torch transformers pandas
poetry add --group dev pytest black mypy

# Install
poetry install

# Run commands
poetry run python src/train.py
poetry run pytest
```

---

## Common Issues and Solutions

### Issue: Permission denied on setup.sh
```bash
chmod +x setup.sh
```

### Issue: Python version too old
```bash
# Install Python 3.11
sudo apt install python3.11 python3.11-venv
python3.11 -m venv venv
```

### Issue: pip SSL certificate error
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue: Virtual environment not activating
```bash
# Use 'source' not 'sh'
source venv/bin/activate  # Correct

# Not:
sh venv/bin/activate     # Wrong
```

---

## Summary Checklist

- [ ] Project structure created
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (requirements.txt, requirements-dev.txt)
- [ ] Environment variables configured (.env from .env.example)
- [ ] setup.sh script created and tested
- [ ] Source code files created (src/)
- [ ] Test files created (tests/)
- [ ] Configuration files created (configs/)
- [ ] Documentation written (README.md, docs/)
- [ ] Verification script passed
- [ ] Code quality checks passed
- [ ] Ready for development!

---

**Estimated Total Time:** 2-3 hours

**Next Steps:**
- Proceed to Exercise 02 (Data Structures)
- Apply this pattern to your own projects
- Explore extension challenges

---

**Last Updated:** 2025-10-30
**Version:** 1.0.0
