#!/bin/bash
#
# cleanup.sh - Clean temporary and cache files from ML project
#
# Usage: ./cleanup.sh PROJECT_PATH
#
# This script removes:
# - Python cache files (__pycache__, *.pyc, *.pyo)
# - Jupyter checkpoint files (.ipynb_checkpoints)
# - Temporary files (*.tmp, *~)
# - Log files (*.log)
# - Build artifacts (build/, dist/, *.egg-info)
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

VERSION="1.0.0"
SCRIPT_NAME=$(basename "$0")

# Cleanup targets
CLEANUP_TARGETS=(
    "__pycache__"
    "*.pyc"
    "*.pyo"
    ".ipynb_checkpoints"
    "*.tmp"
    "*~"
    "*.log"
    ".pytest_cache"
    ".mypy_cache"
    ".coverage"
    "htmlcov"
    "build"
    "dist"
    "*.egg-info"
)

# =============================================================================
# Helper Functions
# =============================================================================

# Print usage information
usage() {
    cat << EOF
Usage: $SCRIPT_NAME PROJECT_PATH

Clean temporary and cache files from an ML project.

Arguments:
    PROJECT_PATH    Path to the ML project directory

Options:
    -h, --help      Show this help message
    -v, --version   Show version information
    -y, --yes       Skip confirmation prompt
    --dry-run       Show what would be deleted without actually deleting

Removes:
    - Python cache: __pycache__/, *.pyc, *.pyo
    - Jupyter checkpoints: .ipynb_checkpoints/
    - Temporary files: *.tmp, *~
    - Log files: *.log
    - Test cache: .pytest_cache/, .mypy_cache/, .coverage, htmlcov/
    - Build artifacts: build/, dist/, *.egg-info/

Example:
    $SCRIPT_NAME ~/projects/ml-classifier
    $SCRIPT_NAME ../my-project --dry-run
    $SCRIPT_NAME my-project -y

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

SKIP_CONFIRMATION=false
DRY_RUN=false

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -v|--version)
            version
            ;;
        -y|--yes)
            SKIP_CONFIRMATION=true
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
set -- "${POSITIONAL_ARGS[@]}"

# Validate arguments
if [[ $# -eq 0 ]]; then
    echo "Error: Missing project path"
    echo "Try '$SCRIPT_NAME --help' for more information."
    exit 1
fi

PROJECT_PATH="$1"

# =============================================================================
# Validate Path
# =============================================================================

# Validate project path
if [[ ! -d "$PROJECT_PATH" ]]; then
    error "Directory '$PROJECT_PATH' does not exist"
fi

# Get absolute path
PROJECT_PATH=$(cd "$PROJECT_PATH" && pwd)
PROJECT_NAME=$(basename "$PROJECT_PATH")

# =============================================================================
# Scan for Files to Clean
# =============================================================================

echo "Scanning project: $PROJECT_NAME"
echo "Path: $PROJECT_PATH"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN MODE - No files will be deleted]"
    echo ""
fi

# Arrays to store found items
declare -a files_to_remove
declare -a dirs_to_remove
total_size=0

# Scan for each cleanup target
echo "Scanning for files to clean..."
echo ""

# Python cache directories
while IFS= read -r -d '' dir; do
    dirs_to_remove+=("$dir")
    size=$(du -sb "$dir" 2>/dev/null | cut -f1 || echo 0)
    total_size=$((total_size + size))
done < <(find "$PROJECT_PATH" -type d -name "__pycache__" -print0 2>/dev/null)

# Python compiled files
while IFS= read -r -d '' file; do
    files_to_remove+=("$file")
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    total_size=$((total_size + size))
done < <(find "$PROJECT_PATH" -type f \( -name "*.pyc" -o -name "*.pyo" \) -print0 2>/dev/null)

# Jupyter checkpoints
while IFS= read -r -d '' dir; do
    dirs_to_remove+=("$dir")
    size=$(du -sb "$dir" 2>/dev/null | cut -f1 || echo 0)
    total_size=$((total_size + size))
done < <(find "$PROJECT_PATH" -type d -name ".ipynb_checkpoints" -print0 2>/dev/null)

# Temporary files
while IFS= read -r -d '' file; do
    files_to_remove+=("$file")
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    total_size=$((total_size + size))
done < <(find "$PROJECT_PATH" -type f \( -name "*.tmp" -o -name "*~" \) -print0 2>/dev/null)

# Log files
while IFS= read -r -d '' file; do
    files_to_remove+=("$file")
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    total_size=$((total_size + size))
done < <(find "$PROJECT_PATH" -type f -name "*.log" -print0 2>/dev/null)

# Test cache directories
for cache_dir in ".pytest_cache" ".mypy_cache" "htmlcov"; do
    while IFS= read -r -d '' dir; do
        dirs_to_remove+=("$dir")
        size=$(du -sb "$dir" 2>/dev/null | cut -f1 || echo 0)
        total_size=$((total_size + size))
    done < <(find "$PROJECT_PATH" -type d -name "$cache_dir" -print0 2>/dev/null)
done

# Coverage files
while IFS= read -r -d '' file; do
    files_to_remove+=("$file")
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    total_size=$((total_size + size))
done < <(find "$PROJECT_PATH" -type f -name ".coverage" -print0 2>/dev/null)

# Build directories
for build_dir in "build" "dist"; do
    while IFS= read -r -d '' dir; do
        dirs_to_remove+=("$dir")
        size=$(du -sb "$dir" 2>/dev/null | cut -f1 || echo 0)
        total_size=$((total_size + size))
    done < <(find "$PROJECT_PATH" -type d -name "$build_dir" -print0 2>/dev/null)
done

# Egg info directories
while IFS= read -r -d '' dir; do
    dirs_to_remove+=("$dir")
    size=$(du -sb "$dir" 2>/dev/null | cut -f1 || echo 0)
    total_size=$((total_size + size))
done < <(find "$PROJECT_PATH" -type d -name "*.egg-info" -print0 2>/dev/null)

# =============================================================================
# Display Cleanup Summary
# =============================================================================

echo "=========================================="
echo "Cleanup Summary"
echo "=========================================="
echo ""

file_count=${#files_to_remove[@]}
dir_count=${#dirs_to_remove[@]}
total_count=$((file_count + dir_count))

echo "Files to remove: $file_count"
echo "Directories to remove: $dir_count"
echo "Total items: $total_count"
echo "Disk space to recover: $(format_size $total_size)"
echo ""

if [[ $total_count -eq 0 ]]; then
    echo "No files to clean - project is already clean!"
    exit 0
fi

# Display items by category
if [[ ${#dirs_to_remove[@]} -gt 0 ]]; then
    echo "Directories to remove:"
    for dir in "${dirs_to_remove[@]}"; do
        rel_path=$(echo "$dir" | sed "s|$PROJECT_PATH/||")
        echo "  - $rel_path"
    done | head -20
    if [[ ${#dirs_to_remove[@]} -gt 20 ]]; then
        echo "  ... and $((${#dirs_to_remove[@]} - 20)) more directories"
    fi
    echo ""
fi

if [[ ${#files_to_remove[@]} -gt 0 ]] && [[ ${#files_to_remove[@]} -le 20 ]]; then
    echo "Files to remove:"
    for file in "${files_to_remove[@]}"; do
        rel_path=$(echo "$file" | sed "s|$PROJECT_PATH/||")
        echo "  - $rel_path"
    done
    echo ""
fi

# =============================================================================
# Confirmation
# =============================================================================

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would remove $total_count items ($(format_size $total_size))"
    exit 0
fi

if [[ "$SKIP_CONFIRMATION" == "false" ]]; then
    echo "This will permanently delete $total_count items."
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
    echo ""
fi

# =============================================================================
# Perform Cleanup
# =============================================================================

echo "Cleaning up..."
echo ""

removed_files=0
removed_dirs=0
failed=0

# Remove files
for file in "${files_to_remove[@]}"; do
    if rm -f "$file" 2>/dev/null; then
        ((removed_files++))
    else
        ((failed++))
        echo "Warning: Failed to remove $file" >&2
    fi
done

# Remove directories
for dir in "${dirs_to_remove[@]}"; do
    if rm -rf "$dir" 2>/dev/null; then
        ((removed_dirs++))
    else
        ((failed++))
        echo "Warning: Failed to remove $dir" >&2
    fi
done

# =============================================================================
# Cleanup Results
# =============================================================================

echo "=========================================="
echo "Cleanup Complete"
echo "=========================================="
echo ""

success "Removed $removed_files files"
success "Removed $removed_dirs directories"
success "Recovered $(format_size $total_size) disk space"

if [[ $failed -gt 0 ]]; then
    echo ""
    echo "Warning: Failed to remove $failed items"
fi

echo ""
echo "Project cleaned successfully!"
echo ""
