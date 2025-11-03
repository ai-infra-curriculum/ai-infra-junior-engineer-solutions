#!/bin/bash
#
# setup_ml_permissions.sh - Set up ML project with proper permissions
#
# Usage: ./setup_ml_permissions.sh PROJECT_NAME
#
# This script creates an ML project structure with security-appropriate permissions:
# - Public areas (755): Documentation, logs
# - Collaborative areas (775): Notebooks, experiments, team data
# - Controlled areas (755/644): Production models, configurations
# - Private areas (700/600): Secrets, credentials
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

VERSION="1.0.0"
SCRIPT_NAME=$(basename "$0")

# Colors for output
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'  # No Color
else
    GREEN=''
    BLUE=''
    YELLOW=''
    RED=''
    NC=''
fi

# =============================================================================
# Helper Functions
# =============================================================================

usage() {
    cat << EOF
Usage: $SCRIPT_NAME PROJECT_NAME [OPTIONS]

Create an ML project with security-appropriate file permissions.

Arguments:
    PROJECT_NAME    Name of the ML project to create

Options:
    -h, --help      Show this help message
    -v, --version   Show version information
    -c, --collaborative  Use collaborative umask (0002) for all files
    -s, --secure    Use secure umask (0077) for sensitive files

Examples:
    $SCRIPT_NAME my-ml-project
    $SCRIPT_NAME team-project --collaborative
    $SCRIPT_NAME secure-project --secure

EOF
    exit 0
}

version() {
    echo "$SCRIPT_NAME version $VERSION"
    exit 0
}

error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# =============================================================================
# Parse Arguments
# =============================================================================

COLLABORATIVE=false
SECURE=false

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -v|--version)
            version
            ;;
        -c|--collaborative)
            COLLABORATIVE=true
            shift
            ;;
        -s|--secure)
            SECURE=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    set -- "${POSITIONAL_ARGS[@]}"
fi

# Validate arguments
if [[ $# -eq 0 ]]; then
    echo "Error: Missing project name"
    echo "Try '$SCRIPT_NAME --help' for more information."
    exit 1
fi

PROJECT_NAME="$1"

# Validate project name
if [[ -e "$PROJECT_NAME" ]]; then
    error "Directory '$PROJECT_NAME' already exists"
fi

if [[ ! "$PROJECT_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    error "Project name can only contain letters, numbers, hyphens, and underscores"
fi

# =============================================================================
# Main Script
# =============================================================================

echo ""
echo "=========================================="
echo " ML Project Permission Setup"
echo "=========================================="
echo ""
info "Creating project: $PROJECT_NAME"

if [[ "$COLLABORATIVE" == "true" ]]; then
    warning "Using collaborative mode (group writable)"
fi

if [[ "$SECURE" == "true" ]]; then
    warning "Using secure mode (restrictive permissions)"
fi

echo ""

# Create main project directory
mkdir -p "$PROJECT_NAME"
success "Created project directory"

# Create directory structure
info "Creating directory structure..."

# Data directories
mkdir -p "$PROJECT_NAME/datasets/raw"
mkdir -p "$PROJECT_NAME/datasets/processed"
mkdir -p "$PROJECT_NAME/datasets/external"

# Model directories
mkdir -p "$PROJECT_NAME/models/checkpoints"
mkdir -p "$PROJECT_NAME/models/production"

# Notebook directories
mkdir -p "$PROJECT_NAME/notebooks/exploratory"
mkdir -p "$PROJECT_NAME/notebooks/reports"

# Source code directories
mkdir -p "$PROJECT_NAME/src/preprocessing"
mkdir -p "$PROJECT_NAME/src/training"
mkdir -p "$PROJECT_NAME/src/evaluation"
mkdir -p "$PROJECT_NAME/src/utils"

# Configuration directories
mkdir -p "$PROJECT_NAME/configs/training"
mkdir -p "$PROJECT_NAME/configs/deployment"
mkdir -p "$PROJECT_NAME/configs/secrets"

# Other directories
mkdir -p "$PROJECT_NAME/scripts"
mkdir -p "$PROJECT_NAME/tests"
mkdir -p "$PROJECT_NAME/docs"
mkdir -p "$PROJECT_NAME/logs"
mkdir -p "$PROJECT_NAME/shared"

success "Created directory structure"

# =============================================================================
# Set Directory Permissions
# =============================================================================

info "Setting directory permissions..."

# Immutable data (755 - owner writes, all read)
chmod 755 "$PROJECT_NAME/datasets/raw"

# Collaborative data (775 - group writes)
chmod 775 "$PROJECT_NAME/datasets/processed"
chmod 775 "$PROJECT_NAME/datasets/external"

# Model directories
chmod 775 "$PROJECT_NAME/models/checkpoints"     # Team collaboration
chmod 755 "$PROJECT_NAME/models/production"      # Controlled

# Collaborative areas (775)
chmod 775 "$PROJECT_NAME/notebooks"
chmod 775 "$PROJECT_NAME/notebooks/exploratory"
chmod 775 "$PROJECT_NAME/notebooks/reports"
chmod 775 "$PROJECT_NAME/shared"

# Source code (755)
chmod 755 "$PROJECT_NAME/src"
chmod 755 "$PROJECT_NAME/src"/*

# Scripts (755 - executable)
chmod 755 "$PROJECT_NAME/scripts"

# Tests (755)
chmod 755 "$PROJECT_NAME/tests"

# Configs (755 general, 700 secrets)
chmod 755 "$PROJECT_NAME/configs"
chmod 755 "$PROJECT_NAME/configs/training"
chmod 755 "$PROJECT_NAME/configs/deployment"
chmod 700 "$PROJECT_NAME/configs/secrets"  # PRIVATE

# Documentation and logs (755)
chmod 755 "$PROJECT_NAME/docs"
chmod 755 "$PROJECT_NAME/logs"

success "Set directory permissions"

# =============================================================================
# Create Sample Files
# =============================================================================

info "Creating sample files..."

# README
cat > "$PROJECT_NAME/README.md" << EOF
# $PROJECT_NAME

## Overview

ML project with security-appropriate file permissions.

## Directory Structure

- **datasets/**: Dataset storage
  - raw/ (755): Immutable source data
  - processed/ (775): Team-processed data
  - external/ (755): External datasets
- **models/**: Model storage
  - checkpoints/ (775): Training checkpoints (collaborative)
  - production/ (755): Production models (controlled)
- **notebooks/ (775)**: Jupyter notebooks (team editable)
- **src/ (755)**: Source code
- **configs/**: Configuration files
  - secrets/ (700): **PRIVATE** - API keys, credentials
- **scripts/ (755)**: Utility scripts (executable)
- **logs/ (755)**: Application logs
- **shared/ (775)**: Team shared files

## Permission Policy

### File Permissions
- Source code: 644 (rw-r--r--)
- Scripts: 755 (rwxr-xr-x)
- Data files: 664 (rw-rw-r--) for collaborative, 644 for read-only
- Production models: 444 (r--r--r--) - read-only
- Secrets: 600 (rw-------) - owner only

### Directory Permissions
- Public: 755 (rwxr-xr-x)
- Collaborative: 775 (rwxrwxr-x)
- Private: 700 (rwx------)

## Security Notes

- **Never commit secrets to Git**
- Secrets directory (configs/secrets/) is private (700)
- Production models should be read-only (444)
- Use appropriate umask: 0002 for collaboration, 0077 for sensitive work

## Setup

1. Set umask for collaborative work:
   \`\`\`bash
   umask 0002
   \`\`\`

2. Create files with proper permissions (they'll inherit from umask)

3. For secrets, use restrictive permissions:
   \`\`\`bash
   chmod 600 configs/secrets/*
   \`\`\`

## Maintenance

Run permission audit regularly:
\`\`\`bash
../scripts/audit_permissions.sh .
\`\`\`

Fix permissions if needed:
\`\`\`bash
../scripts/fix_permissions.sh .
\`\`\`
EOF

chmod 644 "$PROJECT_NAME/README.md"

# .gitignore
cat > "$PROJECT_NAME/.gitignore" << 'EOF'
# Datasets (too large for Git)
datasets/raw/*
datasets/processed/*
datasets/external/*
!datasets/*/.gitkeep

# Models (too large for Git)
models/checkpoints/*
models/production/*
!models/*/.gitkeep

# Secrets (NEVER commit)
configs/secrets/*
!configs/secrets/.gitkeep
*.key
*.pem
*secret*
*password*
credentials.*

# Python
__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints/

# Logs
logs/*.log
*.log

# Environment
.env
.env.local
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
.DS_Store
EOF

chmod 644 "$PROJECT_NAME/.gitignore"

# Sample configuration
cat > "$PROJECT_NAME/configs/training/default.yaml" << 'EOF'
# Training configuration
model:
  architecture: resnet50
  pretrained: true

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: adam

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
EOF

chmod 644 "$PROJECT_NAME/configs/training/default.yaml"

# Sample secret file (with placeholder)
cat > "$PROJECT_NAME/configs/secrets/README.md" << 'EOF'
# Secrets Directory

This directory contains sensitive information (API keys, credentials, etc.).

**Permissions**: 700 (owner only can access)
**File permissions**: 600 (owner only can read/write)

## Usage

Create secret files with restrictive permissions:

\`\`\`bash
umask 0077  # Temporary restrictive umask
cat > api_keys.yaml << 'SECRETS'
aws:
  access_key: YOUR_KEY_HERE
  secret_key: YOUR_SECRET_HERE
openai:
  api_key: YOUR_KEY_HERE
SECRETS

chmod 600 api_keys.yaml
\`\`\`

## Security Checklist

- [ ] Directory has 700 permissions
- [ ] Files have 600 permissions
- [ ] Secrets are in .gitignore
- [ ] No secrets committed to Git
- [ ] Access logged and audited

## Emergency

If secrets are compromised:
1. Rotate all keys immediately
2. Review access logs
3. Audit Git history for leaks
4. Update .gitignore if needed
EOF

chmod 600 "$PROJECT_NAME/configs/secrets/README.md"  # Even README is private

# Sample Python script
cat > "$PROJECT_NAME/scripts/train.py" << 'EOF'
#!/usr/bin/env python3
"""
Sample training script with proper permissions.

This script should have 755 permissions (executable).
"""

import sys

def main():
    print("Training script placeholder")
    print(f"Permissions: {oct(sys.modules[__name__].__file__)}")
    # Training logic here

if __name__ == "__main__":
    main()
EOF

chmod 755 "$PROJECT_NAME/scripts/train.py"

# .gitkeep files for empty directories
touch "$PROJECT_NAME/datasets/raw/.gitkeep"
touch "$PROJECT_NAME/datasets/processed/.gitkeep"
touch "$PROJECT_NAME/models/checkpoints/.gitkeep"
touch "$PROJECT_NAME/models/production/.gitkeep"
touch "$PROJECT_NAME/configs/secrets/.gitkeep"
chmod 600 "$PROJECT_NAME/configs/secrets/.gitkeep"  # Private

success "Created sample files"

# =============================================================================
# Create PERMISSIONS.md Documentation
# =============================================================================

cat > "$PROJECT_NAME/PERMISSIONS.md" << 'EOF'
# Permission Policy

## Directory Permissions Reference

| Directory | Permissions | Meaning | Purpose |
|-----------|-------------|---------|---------|
| datasets/raw | 755 | rwxr-xr-x | Immutable data - owner modifies, all read |
| datasets/processed | 775 | rwxrwxr-x | Team collaborative data processing |
| models/checkpoints | 775 | rwxrwxr-x | Team shares training checkpoints |
| models/production | 755 | rwxr-xr-x | Production models - controlled access |
| notebooks | 775 | rwxrwxr-x | Team collaborative notebooks |
| src | 755 | rwxr-xr-x | Source code - standard access |
| scripts | 755 | rwxr-xr-x | Executable scripts |
| configs | 755 | rwxr-xr-x | Configuration files |
| **configs/secrets** | **700** | **rwx------** | **PRIVATE - Owner only** |
| logs | 755 | rwxr-xr-x | Logs - system writes, team reads |
| shared | 775 | rwxrwxr-x | Team collaboration area |

## File Permission Guidelines

| File Type | Permissions | Example |
|-----------|-------------|---------|
| Source code | 644 | -rw-r--r-- |
| Scripts (*.sh, *.py) | 755 | -rwxr-xr-x |
| Data files (collaborative) | 664 | -rw-rw-r-- |
| Data files (read-only) | 644 | -rw-r--r-- |
| Production models | 444 | -r--r--r-- (read-only) |
| Configuration files | 644 | -rw-r--r-- |
| **Secrets** | **600** | **-rw-------** |
| Logs | 644 | -rw-r--r-- |

## Umask Settings

### Collaborative Work (Recommended for Teams)
\`\`\`bash
umask 0002
# Files: 664 (rw-rw-r--)
# Dirs:  775 (rwxrwxr-x)
\`\`\`

### Secure Work (Sensitive Data)
\`\`\`bash
umask 0077
# Files: 600 (rw-------)
# Dirs:  700 (rwx------)
\`\`\`

### Standard (Default)
\`\`\`bash
umask 0022
# Files: 644 (rw-r--r--)
# Dirs:  755 (rwxr-xr-x)
\`\`\`

## Permission Commands Quick Reference

### Change Permissions
\`\`\`bash
chmod 644 file.txt         # Standard file
chmod 755 script.sh        # Executable
chmod 600 secret.key       # Private file
chmod 775 shared_dir/      # Collaborative directory
\`\`\`

### Change Ownership (requires sudo)
\`\`\`bash
sudo chown user:group file
sudo chgrp group file
\`\`\`

### Check Permissions
\`\`\`bash
ls -l file                 # Long listing
stat -c '%a %n' file       # Numeric permissions
getfacl file               # ACLs (if set)
\`\`\`

## Security Checklist

- [ ] No world-writable files (perm 002)
- [ ] No world-writable directories (perm 002)
- [ ] Secrets have 600 permissions
- [ ] Secret directories have 700 permissions
- [ ] Scripts are executable (755)
- [ ] Production files are read-only (444)
- [ ] Appropriate umask is set
- [ ] .gitignore excludes secrets

## Audit and Fix

Run security audit:
\`\`\`bash
./audit_permissions.sh .
\`\`\`

Automatically fix common issues:
\`\`\`bash
./fix_permissions.sh .
\`\`\`
EOF

chmod 644 "$PROJECT_NAME/PERMISSIONS.md"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo " Project Created Successfully!"
echo "=========================================="
echo ""

echo "Project: $PROJECT_NAME"
echo "Location: $(pwd)/$PROJECT_NAME"
echo ""

info "Directory structure created with secure permissions:"
cd "$PROJECT_NAME"
find . -type d -maxdepth 2 | while read dir; do
    perms=$(stat -c '%a' "$dir" 2>/dev/null)
    printf "  %-40s %s\n" "$dir" "$perms"
done
cd ..

echo ""
warning "Important Security Notes:"
echo "  • configs/secrets/ is PRIVATE (700) - only owner can access"
echo "  • Never commit secrets to Git (check .gitignore)"
echo "  • Use umask 0002 for collaborative work"
echo "  • Run audit regularly: ./audit_permissions.sh $PROJECT_NAME"

echo ""
info "Next steps:"
echo "  1. cd $PROJECT_NAME"
echo "  2. Review PERMISSIONS.md for guidelines"
echo "  3. Set umask: umask 0002 (add to ~/.bashrc for permanence)"
echo "  4. Start developing with proper permissions"
echo ""
