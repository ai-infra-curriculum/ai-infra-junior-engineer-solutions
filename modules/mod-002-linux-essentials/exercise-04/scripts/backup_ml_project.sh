#!/bin/bash
#
# backup_ml_project.sh - Backup and Restore for ML Projects
#
# Usage: ./backup_ml_project.sh <command> <project_name> [options]
#
# Commands: backup, restore, list, verify
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_DIR="${SCRIPT_DIR}/../logs"
readonly LOG_FILE="${LOG_DIR}/backup.log"
readonly BACKUP_DIR="${HOME}/ml-backups"
readonly PROJECTS_DIR="${HOME}/ml-projects"
readonly MAX_BACKUPS=5

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' NC=''
fi

# Logging
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_error() { log "ERROR" "$@" >&2; }
log_success() { log "SUCCESS" "$@"; }
log_warning() { log "WARNING" "$@"; }

error_exit() {
    log_error "$1"
    exit 1
}

# Setup
setup() {
    mkdir -p "$LOG_DIR" "$BACKUP_DIR" "$PROJECTS_DIR"
}

# Usage
usage() {
    cat << EOF
Usage: $SCRIPT_NAME <command> <project_name> [options]

Backup and restore ML projects with metadata and versioning.

Commands:
  backup <project>           - Create project backup
  restore <backup_file>      - Restore from backup
  list                       - List all backups
  verify <backup_file>       - Verify backup integrity

Examples:
  $SCRIPT_NAME backup my-project
  $SCRIPT_NAME restore my-project_20240101_120000.tar.gz
  $SCRIPT_NAME list
  $SCRIPT_NAME verify my-project_20240101_120000.tar.gz

Projects Location: $PROJECTS_DIR
Backups Location:  $BACKUP_DIR

Features:
  - Compressed backups (tar.gz)
  - Automatic metadata generation
  - Retention policy (keeps last $MAX_BACKUPS backups)
  - Integrity verification
  - Excludes large files (.git, cache, raw data)

EOF
    exit 0
}

# Create backup metadata
create_metadata() {
    local project_name="$1"
    local project_dir="$2"
    local backup_file="$3"

    local metadata_file="${backup_file}.meta"

    # Gather project statistics
    local file_count=$(find "$project_dir" -type f | wc -l)
    local dir_count=$(find "$project_dir" -type d | wc -l)
    local total_size=$(du -sh "$project_dir" | awk '{print $1}')

    # Create metadata
    cat > "$metadata_file" << EOF
{
  "project": "$project_name",
  "backup_file": "$(basename "$backup_file")",
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')",
  "created_by": "${USER:-unknown}",
  "hostname": "$(hostname)",
  "statistics": {
    "files": $file_count,
    "directories": $dir_count,
    "size": "$total_size"
  },
  "source_directory": "$project_dir"
}
EOF

    log_success "✓ Metadata created"
}

# Backup project
backup_project() {
    local project_name="$1"
    local project_dir="${PROJECTS_DIR}/${project_name}"

    log_info "=== Backup Project: $project_name ==="

    # Verify project exists
    if [ ! -d "$project_dir" ]; then
        error_exit "Project directory not found: $project_dir"
    fi

    # Generate backup filename
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_name="${project_name}_${timestamp}.tar.gz"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    log_info "Project: $project_name"
    log_info "Source:  $project_dir"
    log_info "Backup:  $backup_path"

    # Create backup with compression
    log_info "Creating compressed backup..."

    tar -czf "$backup_path" \
        --exclude='*.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.venv' \
        --exclude='venv' \
        --exclude='node_modules' \
        --exclude='data/raw' \
        --exclude='*.log' \
        --exclude='.cache' \
        -C "$PROJECTS_DIR" \
        "$project_name" 2>&1 | tee -a "$LOG_FILE"

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        error_exit "Backup creation failed"
    fi

    log_success "✓ Backup created"

    # Create metadata
    create_metadata "$project_name" "$project_dir" "$backup_path"

    # Display backup info
    local backup_size=$(du -h "$backup_path" | awk '{print $1}')
    log_info "Backup size: $backup_size"

    # Apply retention policy
    apply_retention_policy "$project_name"

    log_success "=== Backup Complete ==="

    echo ""
    echo -e "${BLUE}Backup Summary:${NC}"
    echo "  Project:     $project_name"
    echo "  Backup file: $backup_name"
    echo "  Size:        $backup_size"
    echo "  Location:    $BACKUP_DIR"
    echo ""
}

# Apply retention policy
apply_retention_policy() {
    local project_name="$1"

    log_info "Applying retention policy (keep last $MAX_BACKUPS backups)"

    # Count backups for this project
    local backup_count=$(find "$BACKUP_DIR" -name "${project_name}_*.tar.gz" | wc -l)

    if [ "$backup_count" -gt "$MAX_BACKUPS" ]; then
        local to_remove=$((backup_count - MAX_BACKUPS))
        log_info "Removing $to_remove old backup(s)"

        # Find and remove oldest backups
        find "$BACKUP_DIR" -name "${project_name}_*.tar.gz" -type f -printf '%T+ %p\n' | \
            sort | head -n "$to_remove" | awk '{print $2}' | \
            while read -r old_backup; do
                log_info "Removing: $(basename "$old_backup")"
                rm -f "$old_backup" "${old_backup}.meta"
            done

        log_success "✓ Retention policy applied"
    else
        log_info "No cleanup needed ($backup_count/$MAX_BACKUPS backups)"
    fi
}

# List backups
list_backups() {
    log_info "=== Available Backups ==="

    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A "$BACKUP_DIR"/*.tar.gz 2>/dev/null)" ]; then
        echo "No backups found in $BACKUP_DIR"
        return 0
    fi

    echo ""
    printf "%-40s %10s %20s\n" "Backup File" "Size" "Date"
    printf "%s\n" "$(printf '%.0s-' {1..75})"

    find "$BACKUP_DIR" -name "*.tar.gz" -type f -printf '%T@ %p\n' | \
        sort -rn | \
        while read -r timestamp backup_file; do
            local filename=$(basename "$backup_file")
            local size=$(du -h "$backup_file" | awk '{print $1}')
            local date=$(date -d "@${timestamp%.*}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r "${timestamp%.*}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
            printf "%-40s %10s %20s\n" "$filename" "$size" "$date"
        done

    echo ""
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    local backup_path="${BACKUP_DIR}/${backup_file}"

    log_info "=== Verifying Backup: $backup_file ==="

    # Check if backup exists
    if [ ! -f "$backup_path" ]; then
        error_exit "Backup file not found: $backup_path"
    fi

    # Verify tar archive
    log_info "Verifying archive integrity..."

    if tar -tzf "$backup_path" > /dev/null 2>&1; then
        log_success "✓ Archive integrity verified"
    else
        error_exit "Archive verification failed"
    fi

    # Display backup contents summary
    log_info "Archive contents:"
    local file_count=$(tar -tzf "$backup_path" | wc -l)
    log_info "  Total files: $file_count"

    # Check metadata
    local metadata_file="${backup_path}.meta"
    if [ -f "$metadata_file" ]; then
        log_success "✓ Metadata found"
        echo ""
        echo "Metadata:"
        cat "$metadata_file" | sed 's/^/  /'
    else
        log_warning "⚠ Metadata file not found"
    fi

    log_success "=== Verification Complete ==="
    echo ""
}

# Restore from backup
restore_backup() {
    local backup_file="$1"
    local backup_path="${BACKUP_DIR}/${backup_file}"

    log_info "=== Restore from Backup: $backup_file ==="

    # Verify backup exists
    if [ ! -f "$backup_path" ]; then
        error_exit "Backup file not found: $backup_path"
    fi

    # Extract project name from backup filename
    local project_name=$(echo "$backup_file" | sed 's/_[0-9]*_[0-9]*.tar.gz//')
    local restore_dir="${PROJECTS_DIR}/${project_name}"

    log_info "Project:  $project_name"
    log_info "Restore to: $restore_dir"

    # Check if project already exists
    if [ -d "$restore_dir" ]; then
        echo -e "${YELLOW}⚠ Project directory already exists: $restore_dir${NC}"
        echo -n "Overwrite? (yes/no): "
        read -r response

        if [ "$response" != "yes" ]; then
            log_info "Restore cancelled"
            exit 0
        fi

        # Backup existing directory
        local existing_backup="${restore_dir}.backup_$(date '+%Y%m%d_%H%M%S')"
        log_info "Moving existing directory to: $existing_backup"
        mv "$restore_dir" "$existing_backup"
    fi

    # Extract backup
    log_info "Extracting backup..."

    tar -xzf "$backup_path" -C "$PROJECTS_DIR" 2>&1 | tee -a "$LOG_FILE"

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        error_exit "Restore failed"
    fi

    log_success "✓ Backup extracted"

    # Verify restoration
    if [ -d "$restore_dir" ]; then
        local restored_files=$(find "$restore_dir" -type f | wc -l)
        log_success "✓ Restored $restored_files files"
    else
        error_exit "Restore verification failed"
    fi

    log_success "=== Restore Complete ==="

    echo ""
    echo -e "${BLUE}Restore Summary:${NC}"
    echo "  Project:   $project_name"
    echo "  Location:  $restore_dir"
    echo "  From:      $backup_file"
    echo ""
}

# Main function
main() {
    setup

    if [ $# -eq 0 ]; then
        usage
    fi

    local command="$1"

    case "$command" in
        backup)
            if [ $# -lt 2 ]; then
                error_exit "Project name required. Usage: $SCRIPT_NAME backup <project_name>"
            fi
            backup_project "$2"
            ;;
        restore)
            if [ $# -lt 2 ]; then
                error_exit "Backup file required. Usage: $SCRIPT_NAME restore <backup_file>"
            fi
            restore_backup "$2"
            ;;
        list)
            list_backups
            ;;
        verify)
            if [ $# -lt 2 ]; then
                error_exit "Backup file required. Usage: $SCRIPT_NAME verify <backup_file>"
            fi
            verify_backup "$2"
            ;;
        -h|--help)
            usage
            ;;
        *)
            error_exit "Unknown command: $command"
            ;;
    esac
}

main "$@"
