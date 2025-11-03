#!/bin/bash
#
# project_stats.sh - Generate comprehensive ML project statistics
#
# Usage: ./project_stats.sh PROJECT_PATH
#
# This script analyzes an ML project and provides:
# - File and directory counts
# - Disk usage statistics
# - File type distributions
# - Largest files and directories
# - Recent modifications
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
Usage: $SCRIPT_NAME PROJECT_PATH

Generate comprehensive statistics for an ML project.

Arguments:
    PROJECT_PATH    Path to the ML project directory

Options:
    -h, --help      Show this help message
    -v, --version   Show version information

Example:
    $SCRIPT_NAME ~/projects/ml-classifier
    $SCRIPT_NAME ../my-ml-project

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

# Print section header
section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Format bytes to human-readable
format_size() {
    local size=$1
    if (( size < 1024 )); then
        echo "${size}B"
    elif (( size < 1048576 )); then
        echo "$((size / 1024))K"
    elif (( size < 1073741824 )); then
        echo "$((size / 1048576))M"
    else
        echo "$((size / 1073741824))G"
    fi
}

# =============================================================================
# Main Script
# =============================================================================

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    echo "Error: Missing project path"
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
        PROJECT_PATH="$1"
        ;;
esac

# Validate project path
if [[ ! -d "$PROJECT_PATH" ]]; then
    error "Directory '$PROJECT_PATH' does not exist"
fi

# Get absolute path
PROJECT_PATH=$(cd "$PROJECT_PATH" && pwd)
PROJECT_NAME=$(basename "$PROJECT_PATH")

# =============================================================================
# Generate Statistics
# =============================================================================

section "Project Statistics: $PROJECT_NAME"

# Basic counts
echo "Project Path: $PROJECT_PATH"
echo ""

# Count directories
dir_count=$(find "$PROJECT_PATH" -type d 2>/dev/null | wc -l)
echo "Total Directories: $dir_count"

# Count files
file_count=$(find "$PROJECT_PATH" -type f 2>/dev/null | wc -l)
echo "Total Files: $file_count"

# Total size
if command -v du &> /dev/null; then
    total_size=$(du -sb "$PROJECT_PATH" 2>/dev/null | cut -f1)
    total_size_human=$(du -sh "$PROJECT_PATH" 2>/dev/null | cut -f1)
    echo "Total Size: $total_size_human"
fi

# =============================================================================
# Directory Breakdown
# =============================================================================

section "Directory Breakdown"

# Analyze key directories
for dir in data models src notebooks tests configs scripts docs logs; do
    dir_path="$PROJECT_PATH/$dir"
    if [[ -d "$dir_path" ]]; then
        file_count=$(find "$dir_path" -type f 2>/dev/null | wc -l)
        if command -v du &> /dev/null; then
            dir_size=$(du -sh "$dir_path" 2>/dev/null | cut -f1)
            printf "%-15s %5d files  %10s\n" "$dir/" "$file_count" "$dir_size"
        else
            printf "%-15s %5d files\n" "$dir/" "$file_count"
        fi
    fi
done

# =============================================================================
# File Type Analysis
# =============================================================================

section "File Type Distribution"

# Count by extension
declare -A ext_counts
while IFS= read -r file; do
    ext="${file##*.}"
    # Skip if no extension or file is a directory
    if [[ "$ext" == "$file" ]] || [[ -d "$file" ]]; then
        continue
    fi
    # Convert to lowercase
    ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
    # Initialize if doesn't exist
    if [[ -z "${ext_counts[".$ext"]:-}" ]]; then
        ext_counts[".$ext"]=0
    fi
    ((ext_counts[".$ext"]++))
done < <(find "$PROJECT_PATH" -type f 2>/dev/null)

# Sort and display
if [[ ${#ext_counts[@]} -gt 0 ]]; then
    for ext in "${!ext_counts[@]}"; do
        echo "$ext ${ext_counts[$ext]}"
    done | sort -k2 -rn | head -10 | while read ext count; do
        printf "%-10s %5d files\n" "$ext" "$count"
    done
else
    echo "No files with extensions found"
fi

# =============================================================================
# Python Files Analysis
# =============================================================================

if find "$PROJECT_PATH" -name "*.py" -type f 2>/dev/null | grep -q .; then
    section "Python Files Analysis"

    py_count=$(find "$PROJECT_PATH" -name "*.py" -type f 2>/dev/null | wc -l)
    echo "Total Python files: $py_count"

    # Count lines of Python code
    if command -v wc &> /dev/null; then
        py_lines=$(find "$PROJECT_PATH" -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
        if [[ -n "$py_lines" ]]; then
            echo "Total lines of Python code: $py_lines"
        fi
    fi

    # List Python files by directory
    echo ""
    echo "Python files by directory:"
    find "$PROJECT_PATH" -name "*.py" -type f 2>/dev/null | \
        sed "s|$PROJECT_PATH/||" | \
        xargs -I {} dirname {} | \
        sort | uniq -c | sort -rn | head -10 | \
        while read count dir; do
            printf "  %-30s %3d files\n" "$dir" "$count"
        done
fi

# =============================================================================
# Notebook Analysis
# =============================================================================

if find "$PROJECT_PATH" -name "*.ipynb" -type f 2>/dev/null | grep -q .; then
    section "Jupyter Notebook Analysis"

    nb_count=$(find "$PROJECT_PATH" -name "*.ipynb" -type f 2>/dev/null | wc -l)
    echo "Total notebooks: $nb_count"

    echo ""
    echo "Recent notebooks:"
    find "$PROJECT_PATH" -name "*.ipynb" -type f -printf "%T@ %p\n" 2>/dev/null | \
        sort -rn | head -5 | \
        while read timestamp path; do
            rel_path=$(echo "$path" | sed "s|$PROJECT_PATH/||")
            echo "  - $rel_path"
        done
fi

# =============================================================================
# Largest Files
# =============================================================================

section "Largest Files (Top 10)"

find "$PROJECT_PATH" -type f -exec du -h {} + 2>/dev/null | \
    sort -rh | head -10 | \
    while read size path; do
        rel_path=$(echo "$path" | sed "s|$PROJECT_PATH/||")
        printf "%-10s %s\n" "$size" "$rel_path"
    done

# =============================================================================
# Largest Directories
# =============================================================================

section "Largest Directories (Top 10)"

find "$PROJECT_PATH" -type d -exec du -sh {} + 2>/dev/null | \
    sort -rh | head -10 | \
    while read size path; do
        rel_path=$(echo "$path" | sed "s|$PROJECT_PATH/||")
        # Skip if empty (root directory)
        if [[ -z "$rel_path" ]]; then
            rel_path="."
        fi
        printf "%-10s %s\n" "$size" "$rel_path"
    done

# =============================================================================
# Recent Modifications
# =============================================================================

section "Recently Modified Files (Last 7 Days)"

# Find files modified in last 7 days
recent_files=$(find "$PROJECT_PATH" -type f -mtime -7 2>/dev/null | wc -l)
echo "Files modified in last 7 days: $recent_files"

if [[ $recent_files -gt 0 ]]; then
    echo ""
    echo "Most recent modifications:"
    find "$PROJECT_PATH" -type f -mtime -7 -printf "%T@ %p\n" 2>/dev/null | \
        sort -rn | head -10 | \
        while read timestamp path; do
            rel_path=$(echo "$path" | sed "s|$PROJECT_PATH/||")
            # Convert timestamp to readable date
            date_str=$(date -d "@${timestamp%.*}" "+%Y-%m-%d %H:%M" 2>/dev/null || date -r "${timestamp%.*}" "+%Y-%m-%d %H:%M" 2>/dev/null || echo "")
            if [[ -n "$date_str" ]]; then
                printf "  %-20s %s\n" "$date_str" "$rel_path"
            else
                echo "  - $rel_path"
            fi
        done
fi

# =============================================================================
# Git Information (if available)
# =============================================================================

if [[ -d "$PROJECT_PATH/.git" ]] && command -v git &> /dev/null; then
    section "Git Repository Information"

    cd "$PROJECT_PATH"

    # Current branch
    branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo "Current branch: $branch"

    # Last commit
    last_commit=$(git log -1 --pretty=format:"%h - %s (%cr)" 2>/dev/null || echo "No commits")
    echo "Last commit: $last_commit"

    # Commit count
    commit_count=$(git rev-list --count HEAD 2>/dev/null || echo "0")
    echo "Total commits: $commit_count"

    # Changed files
    changed=$(git status --porcelain 2>/dev/null | wc -l)
    echo "Uncommitted changes: $changed files"
fi

# =============================================================================
# Cache and Temporary Files
# =============================================================================

section "Cache and Temporary Files"

# Python cache
pycache_count=$(find "$PROJECT_PATH" -type d -name "__pycache__" 2>/dev/null | wc -l)
pyc_count=$(find "$PROJECT_PATH" -name "*.pyc" -o -name "*.pyo" 2>/dev/null | wc -l)
echo "Python cache directories (__pycache__): $pycache_count"
echo "Python compiled files (*.pyc/*.pyo): $pyc_count"

# Jupyter checkpoints
checkpoint_count=$(find "$PROJECT_PATH" -type d -name ".ipynb_checkpoints" 2>/dev/null | wc -l)
echo "Jupyter checkpoint directories: $checkpoint_count"

# Log files
log_count=$(find "$PROJECT_PATH" -name "*.log" -type f 2>/dev/null | wc -l)
echo "Log files (*.log): $log_count"

# Temp files
tmp_count=$(find "$PROJECT_PATH" -name "*.tmp" -o -name "*~" -type f 2>/dev/null | wc -l)
echo "Temporary files (*.tmp, *~): $tmp_count"

if command -v du &> /dev/null; then
    cache_size=$(find "$PROJECT_PATH" \( -type d -name "__pycache__" -o -name ".ipynb_checkpoints" \) -exec du -sb {} + 2>/dev/null | awk '{s+=$1} END {print s}')
    if [[ -n "$cache_size" ]] && [[ "$cache_size" -gt 0 ]]; then
        cache_size_human=$(format_size "$cache_size")
        echo "Total cache size: $cache_size_human"
    fi
fi

# =============================================================================
# Summary
# =============================================================================

section "Summary"

echo "Project: $PROJECT_NAME"
echo "Path: $PROJECT_PATH"
echo "Directories: $dir_count"
echo "Files: $file_count"
if [[ -n "${total_size_human:-}" ]]; then
    echo "Size: $total_size_human"
fi
echo ""
echo "Run './cleanup.sh $PROJECT_PATH' to remove cache and temporary files"
echo ""
