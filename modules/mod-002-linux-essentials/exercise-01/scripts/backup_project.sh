#!/bin/bash
#
# backup_project.sh - Create intelligent backup of ML project
#
# Usage: ./backup_project.sh PROJECT_PATH [OUTPUT_DIR]
#
# This script creates a compressed backup of an ML project while:
# - Excluding large data files (data/, models/)
# - Excluding cache files (__pycache__, .pyc)
# - Creating timestamped archives
# - Preserving directory structure
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

VERSION="1.0.0"
SCRIPT_NAME=$(basename "$0")

# Default exclusions for ML projects
EXCLUDE_PATTERNS=(
    "data/raw/*"
    "data/processed/*"
    "data/external/*"
    "models/checkpoints/*"
    "models/production/*"
    "__pycache__"
    "*.pyc"
    "*.pyo"
    ".ipynb_checkpoints"
    "*.log"
    "*.tmp"
    ".git"
    "venv"
    "env"
    ".venv"
    "node_modules"
)

# =============================================================================
# Helper Functions
# =============================================================================

# Print usage information
usage() {
    cat << EOF
Usage: $SCRIPT_NAME PROJECT_PATH [OUTPUT_DIR]

Create an intelligent backup of an ML project, excluding large data and cache files.

Arguments:
    PROJECT_PATH    Path to the ML project directory to backup
    OUTPUT_DIR      Directory to save backup (default: current directory)

Options:
    -h, --help      Show this help message
    -v, --version   Show version information
    --include-data  Include data files in backup
    --include-models Include model files in backup

Excluded by default:
    - data/raw/*, data/processed/*, data/external/*
    - models/checkpoints/*, models/production/*
    - __pycache__, *.pyc, *.pyo
    - .ipynb_checkpoints
    - *.log, *.tmp
    - .git, venv, env, .venv, node_modules

Example:
    $SCRIPT_NAME ~/projects/ml-classifier
    $SCRIPT_NAME ../my-project ~/backups
    $SCRIPT_NAME my-project --include-data

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

# Format bytes to human-readable
format_size() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "$1 bytes"
}

# =============================================================================
# Parse Arguments
# =============================================================================

INCLUDE_DATA=false
INCLUDE_MODELS=false

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -v|--version)
            version
            ;;
        --include-data)
            INCLUDE_DATA=true
            shift
            ;;
        --include-models)
            INCLUDE_MODELS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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
    echo "Error: Missing project path"
    echo "Try '$SCRIPT_NAME --help' for more information."
    exit 1
fi

PROJECT_PATH="$1"
OUTPUT_DIR="${2:-.}"  # Default to current directory if not specified

# =============================================================================
# Validate Paths
# =============================================================================

# Validate project path
if [[ ! -d "$PROJECT_PATH" ]]; then
    error "Directory '$PROJECT_PATH' does not exist"
fi

# Get absolute paths
PROJECT_PATH=$(cd "$PROJECT_PATH" && pwd)
PROJECT_NAME=$(basename "$PROJECT_PATH")

# Create output directory if it doesn't exist
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    success "Created output directory: $OUTPUT_DIR"
fi

OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

# =============================================================================
# Create Backup
# =============================================================================

# Generate timestamped filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="${PROJECT_NAME}_backup_${TIMESTAMP}.tar.gz"
BACKUP_PATH="$OUTPUT_DIR/$BACKUP_NAME"

echo "Creating backup of: $PROJECT_NAME"
echo "Source: $PROJECT_PATH"
echo "Output: $BACKUP_PATH"
echo ""

# Adjust exclusions based on flags
if [[ "$INCLUDE_DATA" == "true" ]]; then
    echo "Including data files in backup"
    EXCLUDE_PATTERNS=("${EXCLUDE_PATTERNS[@]/data\/*/}")
fi

if [[ "$INCLUDE_MODELS" == "true" ]]; then
    echo "Including model files in backup"
    EXCLUDE_PATTERNS=("${EXCLUDE_PATTERNS[@]/models\/*/}")
fi

# Build tar exclude options
EXCLUDE_OPTS=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    if [[ -n "$pattern" ]]; then
        EXCLUDE_OPTS+=(--exclude="$pattern")
    fi
done

echo "Excluded patterns:"
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    if [[ -n "$pattern" ]]; then
        echo "  - $pattern"
    fi
done
echo ""

# Create backup
echo "Creating compressed archive..."
cd "$(dirname "$PROJECT_PATH")"
tar czf "$BACKUP_PATH" "${EXCLUDE_OPTS[@]}" "$PROJECT_NAME" 2>&1 | grep -v "Removing leading" || true

# Verify backup was created
if [[ ! -f "$BACKUP_PATH" ]]; then
    error "Failed to create backup"
fi

success "Backup created successfully"

# =============================================================================
# Backup Information
# =============================================================================

echo ""
echo "=========================================="
echo "Backup Summary"
echo "=========================================="
echo ""

# Backup file info
backup_size=$(stat -c%s "$BACKUP_PATH" 2>/dev/null || stat -f%z "$BACKUP_PATH" 2>/dev/null)
backup_size_human=$(format_size "$backup_size")

echo "Backup file: $BACKUP_NAME"
echo "Location: $OUTPUT_DIR"
echo "Size: $backup_size_human"

# Original project size
original_size=$(du -sb "$PROJECT_PATH" | cut -f1)
original_size_human=$(format_size "$original_size")

echo "Original size: $original_size_human"

# Compression ratio
if [[ $backup_size -gt 0 ]]; then
    ratio=$(awk "BEGIN {printf \"%.1f\", ($original_size / $backup_size)}")
    echo "Compression ratio: ${ratio}x"
fi

# File count in backup
echo ""
echo "Analyzing backup contents..."
file_count=$(tar -tzf "$BACKUP_PATH" | wc -l)
echo "Files in backup: $file_count"

# Show directory structure
echo ""
echo "Directory structure:"
tar -tzf "$BACKUP_PATH" | head -20 | sed 's/^/  /'
if [[ $file_count -gt 20 ]]; then
    echo "  ... and $((file_count - 20)) more files"
fi

# =============================================================================
# Verification
# =============================================================================

echo ""
echo "Verifying backup integrity..."

# Test archive
if tar -tzf "$BACKUP_PATH" &>/dev/null; then
    success "Backup integrity verified"
else
    error "Backup verification failed - archive may be corrupted"
fi

# =============================================================================
# Restore Instructions
# =============================================================================

echo ""
echo "=========================================="
echo "Restore Instructions"
echo "=========================================="
echo ""
echo "To restore this backup:"
echo "  tar -xzf $BACKUP_NAME"
echo ""
echo "To list contents without extracting:"
echo "  tar -tzf $BACKUP_NAME"
echo ""
echo "To extract to a specific directory:"
echo "  tar -xzf $BACKUP_NAME -C /path/to/directory"
echo ""

# =============================================================================
# Cleanup Suggestions
# =============================================================================

# Check if there are old backups
old_backups=$(find "$OUTPUT_DIR" -name "${PROJECT_NAME}_backup_*.tar.gz" -type f | wc -l)

if [[ $old_backups -gt 3 ]]; then
    echo "=========================================="
    echo "Cleanup Suggestion"
    echo "=========================================="
    echo ""
    echo "Found $old_backups backup(s) for this project."
    echo "Consider cleaning up old backups:"
    echo ""
    echo "List all backups:"
    echo "  ls -lh $OUTPUT_DIR/${PROJECT_NAME}_backup_*.tar.gz"
    echo ""
    echo "Remove backups older than 30 days:"
    echo "  find $OUTPUT_DIR -name '${PROJECT_NAME}_backup_*.tar.gz' -mtime +30 -delete"
    echo ""
fi

echo "Backup completed successfully!"
echo ""
