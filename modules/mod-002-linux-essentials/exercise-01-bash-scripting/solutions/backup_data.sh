#!/bin/bash
#
# backup_data.sh - Backup and restore ML training data
#
# Description:
#   Automates backup of ML datasets with compression, encryption,
#   and retention policies. Supports local and remote backups.
#
# Usage:
#   ./backup_data.sh [OPTIONS] ACTION [PATH]
#
# Actions:
#   backup      Create a new backup
#   restore     Restore from backup
#   list        List available backups
#   cleanup     Remove old backups based on retention policy
#
# Options:
#   -d, --destination DIR   Backup destination directory
#   -r, --retention DAYS    Retention period in days (default: 30)
#   -c, --compress          Enable compression (default: true)
#   -e, --encrypt           Enable encryption
#   -n, --dry-run          Perform dry-run
#   -v, --verbose          Enable verbose output
#   -h, --help             Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/ml-backup.log"

# Default configuration
BACKUP_DESTINATION="${BACKUP_DESTINATION:-/backup/ml-data}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESS=true
ENCRYPT=false
DRY_RUN=false
VERBOSE=false

# Backup metadata
readonly BACKUP_PREFIX="ml-data"
readonly METADATA_FILE=".backup-metadata.json"

# ===========================
# Logging Functions
# ===========================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        log "DEBUG" "$@"
    fi
}

error_exit() {
    log "ERROR" "$1"
    exit "${2:-1}"
}

# ===========================
# Utility Functions
# ===========================

human_readable_size() {
    local size=$1
    local units=("B" "KB" "MB" "GB" "TB")
    local unit_index=0

    while [[ $size -gt 1024 ]] && [[ $unit_index -lt 4 ]]; do
        size=$((size / 1024))
        ((unit_index++))
    done

    echo "${size}${units[$unit_index]}"
}

calculate_checksum() {
    local file="$1"
    sha256sum "$file" | awk '{print $1}'
}

check_disk_space() {
    local required_space=$1  # in MB
    local destination=$2

    local available_space=$(df -m "$destination" | awk 'NR==2 {print $4}')

    if [[ $available_space -lt $required_space ]]; then
        error_exit "Insufficient disk space. Required: ${required_space}MB, Available: ${available_space}MB" 1
    fi

    log_verbose "Disk space check passed. Available: ${available_space}MB"
}

# ===========================
# Backup Functions
# ===========================

create_backup_name() {
    local source_path="$1"
    local source_name=$(basename "$source_path")
    local timestamp=$(date +%Y%m%d_%H%M%S)

    echo "${BACKUP_PREFIX}_${source_name}_${timestamp}"
}

create_backup() {
    local source_path="$1"

    if [[ ! -e "$source_path" ]]; then
        error_exit "Source path does not exist: $source_path" 1
    fi

    log "INFO" "Starting backup of: $source_path"

    local backup_name=$(create_backup_name "$source_path")
    local backup_dir="$BACKUP_DESTINATION/$backup_name"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would create backup: $backup_dir"
        return 0
    fi

    # Create backup directory
    mkdir -p "$backup_dir"

    # Calculate source size
    local source_size=$(du -sm "$source_path" | cut -f1)
    log "INFO" "Source size: ${source_size}MB"

    # Check disk space (require 2x source size for safety)
    check_disk_space $((source_size * 2)) "$BACKUP_DESTINATION"

    # Start backup
    local backup_file="${backup_dir}/data.tar"
    log "INFO" "Creating backup archive..."

    if [[ -d "$source_path" ]]; then
        tar -cf "$backup_file" -C "$(dirname "$source_path")" "$(basename "$source_path")"
    else
        tar -cf "$backup_file" -C "$(dirname "$source_path")" "$(basename "$source_path")"
    fi

    # Compress if enabled
    if [[ "$COMPRESS" == true ]]; then
        log "INFO" "Compressing backup..."
        gzip "$backup_file"
        backup_file="${backup_file}.gz"
    fi

    # Encrypt if enabled
    if [[ "$ENCRYPT" == true ]]; then
        log "INFO" "Encrypting backup..."
        if ! command -v gpg &> /dev/null; then
            log "WARNING" "GPG not found. Skipping encryption."
        else
            gpg --symmetric --cipher-algo AES256 "$backup_file"
            rm "$backup_file"
            backup_file="${backup_file}.gpg"
        fi
    fi

    # Calculate checksum
    log "INFO" "Calculating checksum..."
    local checksum=$(calculate_checksum "$backup_file")
    echo "$checksum" > "${backup_file}.sha256"

    # Create metadata
    create_backup_metadata "$backup_dir" "$source_path" "$backup_file" "$checksum"

    # Get final backup size
    local backup_size=$(du -sh "$backup_dir" | cut -f1)

    log "SUCCESS" "Backup created successfully!"
    log "INFO" "  Location: $backup_dir"
    log "INFO" "  Size: $backup_size"
    log "INFO" "  Checksum: $checksum"

    return 0
}

create_backup_metadata() {
    local backup_dir="$1"
    local source_path="$2"
    local backup_file="$3"
    local checksum="$4"

    local metadata_path="$backup_dir/$METADATA_FILE"

    cat > "$metadata_path" <<EOF
{
  "backup_name": "$(basename "$backup_dir")",
  "source_path": "$source_path",
  "backup_file": "$(basename "$backup_file")",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "checksum": "$checksum",
  "compressed": $COMPRESS,
  "encrypted": $ENCRYPT,
  "hostname": "$(hostname)",
  "user": "$USER"
}
EOF

    log_verbose "Metadata created: $metadata_path"
}

# ===========================
# Restore Functions
# ===========================

list_backups() {
    log "INFO" "Available backups in: $BACKUP_DESTINATION"
    log "INFO" "=========================================="

    if [[ ! -d "$BACKUP_DESTINATION" ]]; then
        log "WARNING" "Backup destination does not exist"
        return 1
    fi

    local backup_count=0

    for backup_dir in "$BACKUP_DESTINATION"/${BACKUP_PREFIX}_*; do
        if [[ ! -d "$backup_dir" ]]; then
            continue
        fi

        ((backup_count++))

        local metadata_file="$backup_dir/$METADATA_FILE"
        if [[ -f "$metadata_file" ]]; then
            local backup_name=$(basename "$backup_dir")
            local timestamp=$(jq -r '.timestamp' "$metadata_file" 2>/dev/null || echo "unknown")
            local source=$(jq -r '.source_path' "$metadata_file" 2>/dev/null || echo "unknown")
            local size=$(du -sh "$backup_dir" | cut -f1)

            echo "$backup_count. $backup_name"
            echo "   Timestamp: $timestamp"
            echo "   Source: $source"
            echo "   Size: $size"
            echo ""
        else
            echo "$backup_count. $(basename "$backup_dir") (no metadata)"
            echo ""
        fi
    done

    if [[ $backup_count -eq 0 ]]; then
        log "INFO" "No backups found"
    else
        log "INFO" "Total backups: $backup_count"
    fi

    return 0
}

restore_backup() {
    local backup_name="$1"
    local restore_path="${2:-.}"

    local backup_dir="$BACKUP_DESTINATION/$backup_name"

    if [[ ! -d "$backup_dir" ]]; then
        error_exit "Backup not found: $backup_name" 1
    fi

    log "INFO" "Restoring backup: $backup_name"
    log "INFO" "Destination: $restore_path"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would restore backup to: $restore_path"
        return 0
    fi

    # Read metadata
    local metadata_file="$backup_dir/$METADATA_FILE"
    if [[ ! -f "$metadata_file" ]]; then
        log "WARNING" "Metadata file not found"
    fi

    # Find backup file
    local backup_file=""
    for file in "$backup_dir"/data.tar*; do
        if [[ -f "$file" ]]; then
            backup_file="$file"
            break
        fi
    done

    if [[ -z "$backup_file" ]]; then
        error_exit "Backup data file not found in: $backup_dir" 1
    fi

    # Verify checksum
    local checksum_file="${backup_file}.sha256"
    if [[ -f "$checksum_file" ]]; then
        log "INFO" "Verifying checksum..."
        local expected_checksum=$(cat "$checksum_file")
        local actual_checksum=$(calculate_checksum "$backup_file")

        if [[ "$expected_checksum" != "$actual_checksum" ]]; then
            error_exit "Checksum verification failed!" 1
        fi
        log "SUCCESS" "Checksum verified"
    else
        log "WARNING" "No checksum file found"
    fi

    # Decrypt if needed
    if [[ "$backup_file" == *.gpg ]]; then
        log "INFO" "Decrypting backup..."
        if ! gpg --decrypt "$backup_file" > "${backup_file%.gpg}"; then
            error_exit "Decryption failed" 1
        fi
        backup_file="${backup_file%.gpg}"
    fi

    # Decompress if needed
    if [[ "$backup_file" == *.gz ]]; then
        log "INFO" "Decompressing backup..."
        gunzip -k "$backup_file"
        backup_file="${backup_file%.gz}"
    fi

    # Extract backup
    log "INFO" "Extracting backup..."
    mkdir -p "$restore_path"

    if tar -xf "$backup_file" -C "$restore_path"; then
        log "SUCCESS" "Backup restored successfully to: $restore_path"
    else
        error_exit "Failed to extract backup" 1
    fi

    return 0
}

# ===========================
# Cleanup Functions
# ===========================

cleanup_old_backups() {
    log "INFO" "Cleaning up backups older than $RETENTION_DAYS days"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would clean up old backups"
    fi

    local removed_count=0
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%s 2>/dev/null || date -v-${RETENTION_DAYS}d +%s)

    for backup_dir in "$BACKUP_DESTINATION"/${BACKUP_PREFIX}_*; do
        if [[ ! -d "$backup_dir" ]]; then
            continue
        fi

        local backup_time=$(stat -c %Y "$backup_dir" 2>/dev/null || stat -f %m "$backup_dir")

        if [[ $backup_time -lt $cutoff_date ]]; then
            log "INFO" "Removing old backup: $(basename "$backup_dir")"

            if [[ "$DRY_RUN" == false ]]; then
                rm -rf "$backup_dir"
            fi

            ((removed_count++))
        fi
    done

    if [[ $removed_count -eq 0 ]]; then
        log "INFO" "No old backups to remove"
    else
        log "SUCCESS" "Removed $removed_count old backup(s)"
    fi

    return 0
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] ACTION [PATH]

Backup and restore ML training data.

ACTIONS:
    backup PATH         Create a new backup
    restore BACKUP_NAME [PATH]
                       Restore from backup
    list               List available backups
    cleanup            Remove old backups

OPTIONS:
    -d, --destination DIR   Backup destination directory
                           Default: /backup/ml-data
    -r, --retention DAYS    Retention period (default: 30)
    -c, --compress         Enable compression (default: true)
    -e, --encrypt          Enable encryption
    -n, --dry-run         Perform dry-run
    -v, --verbose         Enable verbose output
    -h, --help            Display this help message

EXAMPLES:
    # Create backup
    $SCRIPT_NAME backup /data/training-data

    # Create encrypted backup
    $SCRIPT_NAME --encrypt backup /data/models

    # List backups
    $SCRIPT_NAME list

    # Restore backup
    $SCRIPT_NAME restore ml-data_training-data_20240124_120000

    # Cleanup old backups
    $SCRIPT_NAME --retention 7 cleanup

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -d|--destination)
                BACKUP_DESTINATION="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -c|--compress)
                COMPRESS=true
                shift
                ;;
            -e|--encrypt)
                ENCRYPT=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            backup|restore|list|cleanup)
                ACTION="$1"
                shift
                break
                ;;
            -*)
                error_exit "Unknown option: $1" 1
                ;;
            *)
                error_exit "Invalid argument: $1" 1
                ;;
        esac
    done

    # Get remaining arguments
    ARGS=("$@")
}

# ===========================
# Main Function
# ===========================

main() {
    log "INFO" "=========================================="
    log "INFO" "ML Data Backup Script"
    log "INFO" "=========================================="

    parse_arguments "$@"

    if [[ -z "${ACTION:-}" ]]; then
        error_exit "Action is required (backup, restore, list, cleanup)" 1
    fi

    case "$ACTION" in
        backup)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                error_exit "Backup action requires PATH argument" 1
            fi
            create_backup "${ARGS[0]}"
            ;;
        restore)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                error_exit "Restore action requires BACKUP_NAME argument" 1
            fi
            restore_backup "${ARGS[0]}" "${ARGS[1]:-}"
            ;;
        list)
            list_backups
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        *)
            error_exit "Unknown action: $ACTION" 1
            ;;
    esac

    log "INFO" "=========================================="
    log "SUCCESS" "Operation completed successfully!"
    log "INFO" "=========================================="

    exit 0
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
