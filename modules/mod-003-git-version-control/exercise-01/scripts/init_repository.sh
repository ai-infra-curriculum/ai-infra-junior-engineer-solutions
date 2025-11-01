#!/bin/bash
###############################################################################
# Git Repository Initialization Script
###############################################################################
#
# Purpose: Initialize a Git repository with proper structure and commits
#          demonstrating Git best practices for ML projects
#
# This script creates:
# - Initial repository structure
# - Multiple atomic commits showing best practices
# - Proper .gitignore and .gitattributes
# - Well-formed commit messages
#
# Usage: ./init_repository.sh
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
REPO_DIR="../example-repo"
GIT_USER_NAME="${GIT_USER_NAME:-ML Engineer}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-ml.engineer@example.com}"

###############################################################################
# Helper Functions
###############################################################################

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo ""
}

###############################################################################
# Main Script
###############################################################################

main() {
    log_step "Git Repository Initialization for ML Project"

    # Check if repo directory exists
    if [ ! -d "$REPO_DIR" ]; then
        log_error "Repository directory not found: $REPO_DIR"
        exit 1
    fi

    cd "$REPO_DIR"

    # Check if already a git repository
    if [ -d ".git" ]; then
        log_info "Git repository already exists. Removing it..."
        rm -rf .git
    fi

    ###########################################################################
    # Step 1: Initialize Repository
    ###########################################################################

    log_step "Step 1: Initialize Git Repository"

    git init
    log_success "Initialized empty Git repository"

    # Configure Git user
    git config user.name "$GIT_USER_NAME"
    git config user.email "$GIT_USER_EMAIL"
    log_info "Configured Git user: $GIT_USER_NAME <$GIT_USER_EMAIL>"

    # Configure Git settings
    git config core.autocrlf input
    git config core.ignorecase false
    log_info "Configured Git settings"

    ###########################################################################
    # Step 2: Initial Commit - Project Structure
    ###########################################################################

    log_step "Step 2: First Commit - Project Structure"

    # Add .gitignore and .gitattributes first
    git add .gitignore .gitattributes

    git commit -m "Initial commit: Add .gitignore and .gitattributes

- Comprehensive .gitignore for Python/ML projects
- Exclude model files, datasets, and temporary files
- .gitattributes for proper file handling
- Configure line endings and binary files

These files establish Git configuration before adding code."

    log_success "Committed .gitignore and .gitattributes"

    ###########################################################################
    # Step 3: Add Project Documentation
    ###########################################################################

    log_step "Step 3: Add Project Documentation"

    git add README.md .env.example

    git commit -m "Add project documentation and configuration template

- README.md with project overview and usage instructions
- .env.example as environment configuration template
- Installation and deployment instructions
- API endpoint documentation

Documentation helps onboard new developers quickly."

    log_success "Committed documentation"

    ###########################################################################
    # Step 4: Add Dependencies
    ###########################################################################

    log_step "Step 4: Add Dependencies"

    git add requirements.txt

    git commit -m "Add Python dependencies

- FastAPI for REST API
- PyTorch and torchvision for ML models
- PIL for image processing
- PyYAML for configuration
- Development dependencies (pytest, black, flake8)

All dependencies pinned to specific versions for reproducibility."

    log_success "Committed requirements.txt"

    ###########################################################################
    # Step 5: Add Configuration Files
    ###########################################################################

    log_step "Step 5: Add Configuration"

    git add configs/

    git commit -m "Add application configuration files

- configs/default.yaml: Base configuration for all environments
- configs/production.yaml: Production-specific settings

YAML-based configuration allows easy environment management:
- Model settings (architecture, device, batch size)
- API settings (CORS, rate limiting, timeouts)
- Logging configuration (level, format, rotation)
- Security settings (auth, HTTPS)
- Resource limits (memory, GPU, workers)"

    log_success "Committed configuration files"

    ###########################################################################
    # Step 6: Add Logging Utilities
    ###########################################################################

    log_step "Step 6: Add Logging Module"

    git add src/utils/ src/__init__.py

    git commit -m "Add structured logging module

Implements comprehensive logging utilities:
- JSON and text log formatting
- Context propagation for request tracking
- Performance logging with timing
- Log rotation support
- Integration with monitoring systems

Structured logging enables:
- Easy log parsing and aggregation
- Request tracing across services
- Performance monitoring
- Error tracking and debugging"

    log_success "Committed logging utilities"

    ###########################################################################
    # Step 7: Add Image Preprocessing
    ###########################################################################

    log_step "Step 7: Add Image Preprocessing"

    git add src/preprocessing/

    git commit -m "Add image preprocessing module

Implements image preprocessing pipeline:
- Resizing and normalization
- ImageNet statistics (mean, std)
- Data augmentation (optional)
- Tensor conversion
- Batch processing support

Features:
- Configurable target size
- PIL and numpy array support
- Denormalization for visualization
- Advanced augmentation pipeline (rotation, flip, color jitter)"

    log_success "Committed preprocessing module"

    ###########################################################################
    # Step 8: Add Model Classifier
    ###########################################################################

    log_step "Step 8: Add Image Classification Model"

    git add src/models/

    git commit -m "Add image classification model wrapper

Implements PyTorch model wrapper for inference:
- Support for multiple architectures (ResNet, MobileNet, EfficientNet)
- Automatic device selection (CPU/GPU)
- Pre-trained model loading
- Batch inference support
- Top-k predictions with confidence scores

Model features:
- Clean inference interface
- ImageNet class labels
- Model parameter counting
- Memory estimation
- Error handling and logging"

    log_success "Committed classifier model"

    ###########################################################################
    # Step 9: Add FastAPI Application
    ###########################################################################

    log_step "Step 9: Add FastAPI REST API"

    git add src/api/

    git commit -m "Add FastAPI inference API

Implements production-ready REST API:
- Single and batch image prediction endpoints
- Health check endpoint
- Model information endpoint
- OpenAPI documentation (Swagger/ReDoc)

API features:
- CORS middleware
- File upload handling
- Input validation
- Error handling with proper HTTP codes
- Structured response models
- Request/response logging
- Async support for high performance

Endpoints:
- GET  /        : API information
- GET  /health  : Health check
- POST /predict : Single image prediction
- POST /predict/batch : Batch prediction
- GET  /classes : Available classes
- GET  /model/info : Model metadata"

    log_success "Committed FastAPI application"

    ###########################################################################
    # Step 10: Add Directory Structure
    ###########################################################################

    log_step "Step 10: Add Project Directory Structure"

    # Create .gitkeep files for empty directories
    touch data/raw/.gitkeep
    touch data/processed/.gitkeep
    touch tests/.gitkeep
    touch scripts/.gitkeep
    touch docs/.gitkeep

    git add data/ tests/ scripts/ docs/

    git commit -m "Add project directory structure

Create standard ML project directories:
- data/raw/       : Raw, unprocessed datasets
- data/processed/ : Processed, ready-to-use data
- tests/          : Unit and integration tests
- scripts/        : Utility scripts
- docs/           : Additional documentation

Empty directories tracked with .gitkeep files.
Actual data files excluded via .gitignore."

    log_success "Committed directory structure"

    ###########################################################################
    # Summary
    ###########################################################################

    log_step "Repository Initialization Complete"

    echo "Repository Statistics:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Total Commits: $(git rev-list --count HEAD)"
    echo "Files Tracked: $(git ls-files | wc -l)"
    echo ""

    echo "Recent Commits:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    git log --oneline --graph --all -10
    echo ""

    echo "Repository Structure:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    git ls-tree -r --name-only HEAD | head -20
    echo ""

    log_success "Git repository initialized with 10 atomic commits"

    echo ""
    echo "Next Steps:"
    echo "  1. Review commit history: git log"
    echo "  2. Check repository status: git status"
    echo "  3. View a specific commit: git show <commit-hash>"
    echo "  4. Create a branch: git checkout -b feature/new-feature"
    echo ""
}

###############################################################################
# Execute Main Function
###############################################################################

main "$@"
