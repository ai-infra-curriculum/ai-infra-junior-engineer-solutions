#!/bin/bash
#
# create_ml_project.sh - Create standardized ML project structure
#
# Usage: ./create_ml_project.sh PROJECT_NAME
#
# This script creates a complete ML project directory structure with:
# - Organized subdirectories for data, models, source code
# - Initial README and .gitignore files
# - Proper permissions and structure
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

VERSION="1.0.0"
SCRIPT_NAME=$(basename "$0")

# =============================================================================
# Helper Functions
# =============================================================================

# Print usage information
usage() {
    cat << EOF
Usage: $SCRIPT_NAME PROJECT_NAME

Create a standardized ML project directory structure.

Arguments:
    PROJECT_NAME    Name of the project to create

Options:
    -h, --help     Show this help message
    -v, --version  Show version information

Example:
    $SCRIPT_NAME my-ml-classifier
    $SCRIPT_NAME image-segmentation-model

EOF
    exit 0
}

# Print version information
version() {
    echo "$SCRIPT_NAME version $VERSION"
    exit 0
}

# Print error message and exit
error() {
    echo "Error: $1" >&2
    exit 1
}

# Print success message
success() {
    echo "✓ $1"
}

# Validate project name
validate_project_name() {
    local name="$1"

    # Check if empty
    if [[ -z "$name" ]]; then
        error "Project name cannot be empty"
    fi

    # Check if contains only valid characters (alphanumeric, hyphens, underscores)
    if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        error "Project name can only contain letters, numbers, hyphens, and underscores"
    fi

    # Check if already exists
    if [[ -e "$name" ]]; then
        error "Directory '$name' already exists"
    fi
}

# =============================================================================
# Main Script
# =============================================================================

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    echo "Error: Missing project name"
    echo "Try '$SCRIPT_NAME --help' for more information."
    exit 1
fi

case "$1" in
    -h|--help)
        usage
        ;;
    -v|--version)
        version
        ;;
    *)
        PROJECT_NAME="$1"
        ;;
esac

# Validate project name
validate_project_name "$PROJECT_NAME"

# Create project structure
echo "Creating ML project: $PROJECT_NAME"
echo ""

# Create main project directory
mkdir "$PROJECT_NAME"
success "Created project directory: $PROJECT_NAME"

# Create data directories
mkdir -p "$PROJECT_NAME/data/raw"
mkdir -p "$PROJECT_NAME/data/processed"
mkdir -p "$PROJECT_NAME/data/external"
success "Created data directories (raw, processed, external)"

# Create model directories
mkdir -p "$PROJECT_NAME/models/checkpoints"
mkdir -p "$PROJECT_NAME/models/production"
success "Created model directories (checkpoints, production)"

# Create notebook directories
mkdir -p "$PROJECT_NAME/notebooks/exploratory"
mkdir -p "$PROJECT_NAME/notebooks/reports"
success "Created notebook directories (exploratory, reports)"

# Create source code directories
mkdir -p "$PROJECT_NAME/src/preprocessing"
mkdir -p "$PROJECT_NAME/src/training"
mkdir -p "$PROJECT_NAME/src/evaluation"
mkdir -p "$PROJECT_NAME/src/utils"
success "Created source code directories (preprocessing, training, evaluation, utils)"

# Create other directories
mkdir -p "$PROJECT_NAME/tests"
mkdir -p "$PROJECT_NAME/configs"
mkdir -p "$PROJECT_NAME/scripts"
mkdir -p "$PROJECT_NAME/docs"
mkdir -p "$PROJECT_NAME/logs"
success "Created additional directories (tests, configs, scripts, docs, logs)"

# Create README.md
cat > "$PROJECT_NAME/README.md" << 'EOF'
# PROJECT_NAME_PLACEHOLDER

## Overview

Brief description of the ML project.

## Project Structure

```
PROJECT_NAME_PLACEHOLDER/
├── data/                   # Data files
│   ├── raw/               # Original, immutable data
│   ├── processed/         # Cleaned, transformed data
│   └── external/          # External datasets and references
├── models/                # Trained models
│   ├── checkpoints/       # Training checkpoints
│   └── production/        # Production-ready models
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/       # Exploratory data analysis
│   └── reports/           # Final analysis and reports
├── src/                   # Source code
│   ├── preprocessing/     # Data preprocessing modules
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation and metrics
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── configs/               # Configuration files
├── scripts/               # Utility scripts
├── docs/                  # Documentation
├── logs/                  # Log files
├── README.md              # This file
└── .gitignore            # Git ignore patterns
```

## Setup

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

```bash
# Place raw data in data/raw/
# Run preprocessing
python src/preprocessing/preprocess.py
```

### Training

```bash
# Train model
python src/training/train.py --config configs/train_config.yaml
```

### Evaluation

```bash
# Evaluate model
python src/evaluation/evaluate.py --model models/production/model.pkl
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint code
flake8 src/
```

## Project Organization

- Keep raw data immutable (never modify data/raw/)
- Process data and save to data/processed/
- Save model checkpoints during training
- Move best models to models/production/
- Document experiments in notebooks/
- Write reusable code in src/
- Add tests for all modules

## License

[Your License Here]

## Contact

[Your Contact Information]
EOF

# Replace placeholder with actual project name
sed -i "s/PROJECT_NAME_PLACEHOLDER/$PROJECT_NAME/g" "$PROJECT_NAME/README.md"
success "Created README.md"

# Create .gitignore
cat > "$PROJECT_NAME/.gitignore" << 'EOF'
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

# Virtual environments
venv/
env/
ENV/
.venv

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data files (too large for git)
data/raw/*
data/processed/*
data/external/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep

# Models (too large for git)
models/checkpoints/*
models/production/*
!models/checkpoints/.gitkeep
!models/production/.gitkeep

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Environment variables
.env
.env.local
EOF
success "Created .gitignore"

# Create .gitkeep files to track empty directories
touch "$PROJECT_NAME/data/raw/.gitkeep"
touch "$PROJECT_NAME/data/processed/.gitkeep"
touch "$PROJECT_NAME/data/external/.gitkeep"
touch "$PROJECT_NAME/models/checkpoints/.gitkeep"
touch "$PROJECT_NAME/models/production/.gitkeep"
touch "$PROJECT_NAME/logs/.gitkeep"
success "Created .gitkeep files for empty directories"

# Create initial Python package files
touch "$PROJECT_NAME/src/__init__.py"
touch "$PROJECT_NAME/src/preprocessing/__init__.py"
touch "$PROJECT_NAME/src/training/__init__.py"
touch "$PROJECT_NAME/src/evaluation/__init__.py"
touch "$PROJECT_NAME/src/utils/__init__.py"
success "Created Python package __init__.py files"

# Create sample requirements.txt
cat > "$PROJECT_NAME/requirements.txt" << 'EOF'
# Core ML libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Deep learning (uncomment as needed)
# torch>=1.10.0
# tensorflow>=2.8.0

# Data visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.0.0

# Development
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950

# Utilities
pyyaml>=6.0
python-dotenv>=0.19.0
tqdm>=4.62.0
EOF
success "Created requirements.txt"

# Create sample config file
cat > "$PROJECT_NAME/configs/train_config.yaml" << 'EOF'
# Training configuration

model:
  type: random_forest
  parameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42

data:
  train_path: data/processed/train.csv
  val_path: data/processed/val.csv
  test_path: data/processed/test.csv

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  early_stopping_patience: 5

logging:
  log_dir: logs/
  checkpoint_dir: models/checkpoints/
  save_frequency: 1
EOF
success "Created sample train_config.yaml"

# Print summary
echo ""
echo "=========================================="
echo "Project created successfully!"
echo "=========================================="
echo ""
echo "Project: $PROJECT_NAME"
echo "Location: $(pwd)/$PROJECT_NAME"
echo ""
echo "Directory structure:"
cd "$PROJECT_NAME"
find . -type d | head -20 | sed 's/^/  /'
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_NAME"
echo "  2. Create a virtual environment: python -m venv venv"
echo "  3. Activate it: source venv/bin/activate"
echo "  4. Install dependencies: pip install -r requirements.txt"
echo "  5. Start coding!"
echo ""
