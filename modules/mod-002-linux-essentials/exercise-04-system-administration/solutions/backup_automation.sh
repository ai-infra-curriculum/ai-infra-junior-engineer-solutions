#!/bin/bash
#
# backup_automation.sh - Automated backup and recovery for ML infrastructure
#
# Description:
#   Comprehensive backup solution with incremental/full backups, rotation,
#   verification, and restoration capabilities for ML data and configurations.
#
# Usage:
#   ./backup_automation.sh [COMMAND] [OPTIONS]
#
# Commands:
#   backup                   Perform backup
#   restore BACKUP_ID        Restore from backup
#   list                     List available backups
#   verify BACKUP_ID         Verify backup integrity
#   cleanup                  Remove old backups
#   schedule                 Setup automated backups
#
# Options:
#   -t, --type TYPE          Backup type: full, incremental (default: incremental)
#   -d, --dest DIR           Backup destination directory
#   -s, --source DIRS        Source directories (comma-separated)
#   -k, --keep N             Keep N backups (default: 7)
#   -c, --compress           Enable compression
#   -e, --encrypt            Enable encryption
#   -v, --verbose            Verbose output
#   -n, --dry-run            Dry run mode
#   -h, --help               Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/backup-automation.log"

# Default backup configuration
BACKUP_ROOT="/var/backups/ml-infrastructure"
BACKUP_TYPE="incremental"
KEEP_BACKUPS=7
COMPRESSION=true
ENCRYPTION=false
DRY_RUN=false
VERBOSE=false

# Default source directories for ML infrastructure
DEFAULT_SOURCES=(
    "/etc"
    "/home"
    "/opt"
    "/var/lib/docker"
    "/var/log"
)

# Exclude patterns
EXCLUDE_PATTERNS=(
    "*.tmp"
    "*.cache"
    "*.pyc"
    "__pycache__"
    ".git"
    "node_modules"
    "*.log"
)

# ===========================
# Colors
# ===========================

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RESET='\033[0m'
readonly BOLD='\033[1m'

# ===========================
# Logging
# ===========================

log_message() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"

    if [[ "$VERBOSE" == true ]]; then
        echo -e "[$level] $message"
    fi
}

log_info() {
    log_message "INFO" "$@"
}

log_success() {
    log_message "SUCCESS" "$@"
}

log_warning() {
    log_message "WARNING" "$@"
}

log_error() {
    log_message "ERROR" "$@"
}

# ===========================
# Backup Functions
# ===========================

generate_backup_id() {
    echo "backup-$(date +%Y%m%d-%H%M%S)"
}

get_backup_dir() {
    local backup_id="$1"
    echo "$BACKUP_ROOT/$backup_id"
}

create_backup_metadata() {
    local backup_dir="$1"
    local backup_type="$2"
    local sources="$3"

    cat > "$backup_dir/metadata.txt" <<EOF
Backup ID: $(basename "$backup_dir")
Backup Type: $backup_type
Created: $(date '+%Y-%m-%d %H:%M:%S')
Hostname: $(hostname)
Kernel: $(uname -r)
Sources: $sources
Compression: $COMPRESSION
Encryption: $ENCRYPTION
User: ${SUDO_USER:-$USER}
EOF

    log_info "Metadata created for backup: $(basename "$backup_dir")"
}

perform_backup() {
    local sources=("$@")

    if [[ ${#sources[@]} -eq 0 ]]; then
        sources=("${DEFAULT_SOURCES[@]}")
    fi

    echo -e "${BOLD}${CYAN}Starting Backup${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo "Type: $BACKUP_TYPE"
    echo "Destination: $BACKUP_ROOT"
    echo "Sources: ${sources[*]}"
    echo "Compression: $COMPRESSION"
    echo "Encryption: $ENCRYPTION"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}DRY-RUN MODE - No changes will be made${RESET}"
        echo ""
    fi

    log_info "Backup started - Type: $BACKUP_TYPE, Sources: ${sources[*]}"

    # Create backup directory
    local backup_id=$(generate_backup_id)
    local backup_dir=$(get_backup_dir "$backup_id")

    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$backup_dir"
        echo -e "${GREEN}✓ Backup directory created: $backup_dir${RESET}"
    else
        echo "[DRY-RUN] Would create: $backup_dir"
    fi

    # Create metadata
    if [[ "$DRY_RUN" == false ]]; then
        create_backup_metadata "$backup_dir" "$BACKUP_TYPE" "${sources[*]}"
    fi

    # Find last full backup for incremental
    local last_full_backup=""
    if [[ "$BACKUP_TYPE" == "incremental" ]]; then
        last_full_backup=$(find_last_full_backup)

        if [[ -n "$last_full_backup" ]]; then
            echo "Incremental backup based on: $last_full_backup"
            log_info "Incremental backup based on: $last_full_backup"
        else
            echo -e "${YELLOW}No full backup found, performing full backup${RESET}"
            BACKUP_TYPE="full"
            log_warning "No full backup found, switching to full backup"
        fi
    fi

    echo ""
    echo -e "${BOLD}Backing up sources:${RESET}"

    local total_size=0
    local file_count=0

    # Backup each source
    for source in "${sources[@]}"; do
        if [[ ! -e "$source" ]]; then
            echo -e "${YELLOW}  Skipping (not found): $source${RESET}"
            log_warning "Source not found: $source"
            continue
        fi

        echo -e "${BLUE}  Processing: $source${RESET}"

        local source_name=$(echo "$source" | tr '/' '_' | sed 's/^_//')
        local archive_name="${source_name}.tar"

        if [[ "$COMPRESSION" == true ]]; then
            archive_name="${archive_name}.gz"
        fi

        local archive_path="$backup_dir/$archive_name"

        # Build tar command
        local tar_cmd="tar"

        # Add compression flag
        if [[ "$COMPRESSION" == true ]]; then
            tar_cmd="$tar_cmd -czf"
        else
            tar_cmd="$tar_cmd -cf"
        fi

        tar_cmd="$tar_cmd $archive_path"

        # Add exclude patterns
        for pattern in "${EXCLUDE_PATTERNS[@]}"; do
            tar_cmd="$tar_cmd --exclude=$pattern"
        done

        # Add incremental option
        if [[ "$BACKUP_TYPE" == "incremental" ]] && [[ -n "$last_full_backup" ]]; then
            local snapshot_file="$backup_dir/${source_name}.snar"
            local last_snapshot="$last_full_backup/${source_name}.snar"

            if [[ -f "$last_snapshot" ]]; then
                cp "$last_snapshot" "$snapshot_file"
                tar_cmd="$tar_cmd --listed-incremental=$snapshot_file"
            fi
        elif [[ "$BACKUP_TYPE" == "full" ]]; then
            local snapshot_file="$backup_dir/${source_name}.snar"
            tar_cmd="$tar_cmd --listed-incremental=$snapshot_file"
        fi

        tar_cmd="$tar_cmd $source"

        # Execute backup
        if [[ "$DRY_RUN" == false ]]; then
            if eval "$tar_cmd" 2>/dev/null; then
                local size=$(du -h "$archive_path" | cut -f1)
                local files=$(tar -tzf "$archive_path" 2>/dev/null | wc -l)

                echo -e "${GREEN}    ✓ Backed up: $archive_name ($size, $files files)${RESET}"

                total_size=$((total_size + $(stat -f%z "$archive_path" 2>/dev/null || stat -c%s "$archive_path" 2>/dev/null || echo 0)))
                file_count=$((file_count + files))

                log_success "Backed up: $source -> $archive_name ($size)"
            else
                echo -e "${RED}    ✗ Failed to backup: $source${RESET}"
                log_error "Failed to backup: $source"
            fi
        else
            echo "    [DRY-RUN] Would backup: $source -> $archive_name"
        fi
    done

    echo ""

    # Create checksum file
    if [[ "$DRY_RUN" == false ]]; then
        echo -e "${BLUE}Generating checksums...${RESET}"
        cd "$backup_dir"
        find . -type f -name "*.tar*" -exec sha256sum {} \; > checksums.sha256
        echo -e "${GREEN}✓ Checksums generated${RESET}"
        log_info "Checksums generated for backup: $backup_id"
    fi

    # Encryption
    if [[ "$ENCRYPTION" == true ]] && [[ "$DRY_RUN" == false ]]; then
        echo -e "${BLUE}Encrypting backup...${RESET}"
        encrypt_backup "$backup_dir"
    fi

    # Summary
    echo ""
    echo -e "${CYAN}========================================${RESET}"
    echo -e "${GREEN}${BOLD}Backup completed successfully!${RESET}"
    echo "Backup ID: $backup_id"
    echo "Location: $backup_dir"

    if [[ "$DRY_RUN" == false ]]; then
        local backup_size=$(du -sh "$backup_dir" | cut -f1)
        echo "Size: $backup_size"
        echo "Files: $file_count"
    fi

    echo "Type: $BACKUP_TYPE"
    echo ""

    log_success "Backup completed: $backup_id"

    return 0
}

find_last_full_backup() {
    # Find most recent full backup
    local backups=$(list_backups_by_date)

    for backup_dir in $backups; do
        local metadata="$backup_dir/metadata.txt"
        if [[ -f "$metadata" ]]; then
            local type=$(grep "^Backup Type:" "$metadata" | cut -d: -f2 | xargs)
            if [[ "$type" == "full" ]]; then
                echo "$backup_dir"
                return 0
            fi
        fi
    done

    return 1
}

# ===========================
# Restore Functions
# ===========================

restore_backup() {
    local backup_id="$1"
    local restore_dest="${2:-.}"

    local backup_dir=$(get_backup_dir "$backup_id")

    if [[ ! -d "$backup_dir" ]]; then
        echo -e "${RED}Error: Backup not found: $backup_id${RESET}"
        log_error "Backup not found: $backup_id"
        return 1
    fi

    echo -e "${BOLD}${CYAN}Restoring Backup${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo "Backup ID: $backup_id"
    echo "Source: $backup_dir"
    echo "Destination: $restore_dest"
    echo ""

    # Check metadata
    local metadata="$backup_dir/metadata.txt"
    if [[ -f "$metadata" ]]; then
        echo -e "${BOLD}Backup Information:${RESET}"
        cat "$metadata"
        echo ""
    fi

    # Verify backup first
    echo "Verifying backup integrity..."
    if ! verify_backup_integrity "$backup_id"; then
        echo -e "${RED}Error: Backup verification failed${RESET}"
        echo "Backup may be corrupted. Restore aborted."
        return 1
    fi

    echo -e "${GREEN}✓ Backup verification passed${RESET}"
    echo ""

    # Confirm restore
    if [[ "$FORCE" != true ]]; then
        read -p "Are you sure you want to restore? This may overwrite existing files. (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            echo "Restore cancelled"
            return 0
        fi
    fi

    echo ""
    echo -e "${BOLD}Restoring files:${RESET}"

    log_info "Restore started: $backup_id to $restore_dest"

    # Decrypt if encrypted
    local working_dir="$backup_dir"
    if [[ "$ENCRYPTION" == true ]]; then
        echo "Decrypting backup..."
        working_dir=$(decrypt_backup "$backup_dir")
    fi

    # Restore each archive
    local restored_count=0
    for archive in "$working_dir"/*.tar* ; do
        if [[ ! -f "$archive" ]] || [[ "$archive" == *"checksums"* ]]; then
            continue
        fi

        local archive_name=$(basename "$archive")
        echo -e "${BLUE}  Restoring: $archive_name${RESET}"

        if tar -xzf "$archive" -C "$restore_dest" 2>/dev/null || tar -xf "$archive" -C "$restore_dest" 2>/dev/null; then
            echo -e "${GREEN}    ✓ Restored: $archive_name${RESET}"
            ((restored_count++))
            log_success "Restored: $archive_name"
        else
            echo -e "${RED}    ✗ Failed to restore: $archive_name${RESET}"
            log_error "Failed to restore: $archive_name"
        fi
    done

    echo ""
    echo -e "${CYAN}========================================${RESET}"
    echo -e "${GREEN}${BOLD}Restore completed!${RESET}"
    echo "Restored $restored_count archive(s)"
    echo ""

    log_success "Restore completed: $backup_id"

    return 0
}

# ===========================
# Backup Management
# ===========================

list_backups() {
    echo -e "${BOLD}${CYAN}Available Backups${RESET}"
    echo -e "${CYAN}========================================${RESET}"

    if [[ ! -d "$BACKUP_ROOT" ]]; then
        echo "No backups found"
        return 0
    fi

    printf "%-25s %-15s %-10s %-20s\n" "BACKUP ID" "TYPE" "SIZE" "DATE"
    echo "------------------------------------------------------------------------"

    local backups=$(ls -t "$BACKUP_ROOT" 2>/dev/null || true)

    if [[ -z "$backups" ]]; then
        echo "No backups found"
        return 0
    fi

    for backup_name in $backups; do
        local backup_dir="$BACKUP_ROOT/$backup_name"

        if [[ ! -d "$backup_dir" ]]; then
            continue
        fi

        local size=$(du -sh "$backup_dir" 2>/dev/null | cut -f1 || echo "N/A")
        local date=$(stat -c %y "$backup_dir" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1 || stat -f %Sm "$backup_dir" 2>/dev/null || echo "N/A")

        local backup_type="unknown"
        local metadata="$backup_dir/metadata.txt"
        if [[ -f "$metadata" ]]; then
            backup_type=$(grep "^Backup Type:" "$metadata" | cut -d: -f2 | xargs || echo "unknown")
        fi

        printf "%-25s %-15s %-10s %-20s\n" "$backup_name" "$backup_type" "$size" "$date"
    done

    echo ""

    # Summary
    local total_backups=$(ls "$BACKUP_ROOT" 2>/dev/null | wc -l)
    local total_size=$(du -sh "$BACKUP_ROOT" 2>/dev/null | cut -f1 || echo "N/A")

    echo "Total backups: $total_backups"
    echo "Total size: $total_size"
    echo ""
}

list_backups_by_date() {
    if [[ ! -d "$BACKUP_ROOT" ]]; then
        return 0
    fi

    find "$BACKUP_ROOT" -maxdepth 1 -type d -name "backup-*" | sort -r
}

verify_backup_integrity() {
    local backup_id="$1"
    local backup_dir=$(get_backup_dir "$backup_id")

    if [[ ! -d "$backup_dir" ]]; then
        echo -e "${RED}Error: Backup not found: $backup_id${RESET}"
        return 1
    fi

    echo -e "${BOLD}${CYAN}Verifying Backup: $backup_id${RESET}"
    echo ""

    log_info "Verifying backup: $backup_id"

    # Check metadata
    if [[ ! -f "$backup_dir/metadata.txt" ]]; then
        echo -e "${RED}✗ Metadata file missing${RESET}"
        log_error "Metadata file missing for backup: $backup_id"
        return 1
    fi

    echo -e "${GREEN}✓ Metadata file found${RESET}"

    # Verify checksums
    if [[ -f "$backup_dir/checksums.sha256" ]]; then
        echo "Verifying checksums..."
        cd "$backup_dir"

        if sha256sum -c checksums.sha256 &>/dev/null; then
            echo -e "${GREEN}✓ All checksums verified${RESET}"
            log_success "Checksums verified for backup: $backup_id"
        else
            echo -e "${RED}✗ Checksum verification failed${RESET}"
            log_error "Checksum verification failed for backup: $backup_id"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ No checksum file found${RESET}"
    fi

    # Verify archives
    echo "Verifying archives..."
    local archives_ok=true

    for archive in "$backup_dir"/*.tar* ; do
        if [[ ! -f "$archive" ]] || [[ "$archive" == *"checksums"* ]]; then
            continue
        fi

        local archive_name=$(basename "$archive")

        # Test archive integrity
        if tar -tzf "$archive" &>/dev/null || tar -tf "$archive" &>/dev/null; then
            echo -e "${GREEN}  ✓ $archive_name${RESET}"
        else
            echo -e "${RED}  ✗ $archive_name${RESET}"
            archives_ok=false
            log_error "Archive corrupted: $archive_name in backup $backup_id"
        fi
    done

    if [[ "$archives_ok" == true ]]; then
        echo ""
        echo -e "${GREEN}${BOLD}✓ Backup verification passed${RESET}"
        log_success "Backup verification passed: $backup_id"
        return 0
    else
        echo ""
        echo -e "${RED}${BOLD}✗ Backup verification failed${RESET}"
        log_error "Backup verification failed: $backup_id"
        return 1
    fi
}

cleanup_old_backups() {
    echo -e "${BOLD}${CYAN}Cleaning Up Old Backups${RESET}"
    echo "Keep last $KEEP_BACKUPS backups"
    echo ""

    if [[ ! -d "$BACKUP_ROOT" ]]; then
        echo "No backup directory found"
        return 0
    fi

    log_info "Cleanup started - Keep: $KEEP_BACKUPS"

    local backups=($(ls -t "$BACKUP_ROOT" 2>/dev/null || true))
    local backup_count=${#backups[@]}

    if [[ $backup_count -le $KEEP_BACKUPS ]]; then
        echo "No backups to remove (total: $backup_count, keep: $KEEP_BACKUPS)"
        return 0
    fi

    local remove_count=$((backup_count - KEEP_BACKUPS))
    echo "Removing $remove_count old backup(s)..."
    echo ""

    local removed=0
    for ((i=$KEEP_BACKUPS; i<$backup_count; i++)); do
        local backup_name="${backups[$i]}"
        local backup_dir="$BACKUP_ROOT/$backup_name"

        if [[ ! -d "$backup_dir" ]]; then
            continue
        fi

        local size=$(du -sh "$backup_dir" | cut -f1)
        echo -e "${BLUE}Removing: $backup_name ($size)${RESET}"

        if [[ "$DRY_RUN" == false ]]; then
            if rm -rf "$backup_dir"; then
                echo -e "${GREEN}  ✓ Removed${RESET}"
                log_success "Removed old backup: $backup_name"
                ((removed++))
            else
                echo -e "${RED}  ✗ Failed to remove${RESET}"
                log_error "Failed to remove backup: $backup_name"
            fi
        else
            echo "  [DRY-RUN] Would remove"
            ((removed++))
        fi
    done

    echo ""
    echo -e "${GREEN}Cleanup completed: $removed backup(s) removed${RESET}"
    log_success "Cleanup completed: $removed backups removed"
}

# ===========================
# Encryption (Placeholder)
# ===========================

encrypt_backup() {
    local backup_dir="$1"

    echo -e "${YELLOW}Note: Encryption requires GPG setup${RESET}"
    echo "To enable encryption, configure GPG keys"

    # Placeholder for GPG encryption
    # gpg --encrypt --recipient user@example.com "$backup_dir"/*.tar*

    log_info "Encryption skipped - not configured"
}

decrypt_backup() {
    local backup_dir="$1"

    echo -e "${YELLOW}Note: Decryption requires GPG setup${RESET}"

    # Placeholder for GPG decryption
    # gpg --decrypt "$backup_dir"/*.tar*.gpg

    echo "$backup_dir"
}

# ===========================
# Scheduling
# ===========================

setup_scheduled_backups() {
    echo -e "${BOLD}${CYAN}Setting Up Scheduled Backups${RESET}"
    echo ""

    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}Error: Root privileges required for scheduling${RESET}"
        return 1
    fi

    echo "Backup schedule options:"
    echo "1. Daily backup at 2 AM"
    echo "2. Weekly backup on Sunday at 2 AM"
    echo "3. Custom schedule"
    echo "0. Cancel"
    echo ""

    read -p "Select option: " option

    local cron_schedule=""
    case "$option" in
        1)
            cron_schedule="0 2 * * *"
            ;;
        2)
            cron_schedule="0 2 * * 0"
            ;;
        3)
            echo "Enter cron schedule (e.g., '0 2 * * *' for daily at 2 AM):"
            read -p "Schedule: " cron_schedule
            ;;
        0)
            echo "Cancelled"
            return 0
            ;;
        *)
            echo "Invalid option"
            return 1
            ;;
    esac

    # Create cron job
    local cron_cmd="$SCRIPT_DIR/$SCRIPT_NAME backup --type full"
    local cron_entry="$cron_schedule $cron_cmd >> $LOG_FILE 2>&1"

    # Add to crontab
    (crontab -l 2>/dev/null | grep -v "$SCRIPT_NAME"; echo "$cron_entry") | crontab -

    echo ""
    echo -e "${GREEN}✓ Scheduled backup configured${RESET}"
    echo "Schedule: $cron_schedule"
    echo "Command: $cron_cmd"
    echo ""

    # Also add weekly cleanup
    local cleanup_schedule="0 3 * * 0"  # Sunday at 3 AM
    local cleanup_cmd="$SCRIPT_DIR/$SCRIPT_NAME cleanup"
    local cleanup_entry="$cleanup_schedule $cleanup_cmd >> $LOG_FILE 2>&1"

    (crontab -l 2>/dev/null | grep -v "$SCRIPT_NAME cleanup"; echo "$cleanup_entry") | crontab -

    echo -e "${GREEN}✓ Scheduled cleanup configured${RESET}"
    echo "Schedule: $cleanup_schedule (weekly)"
    echo ""

    log_success "Scheduled backups configured: $cron_schedule"
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [COMMAND] [OPTIONS]

Automated backup and recovery for ML infrastructure.

COMMANDS:
    backup                      Perform backup
    restore BACKUP_ID [DEST]    Restore from backup
    list                        List available backups
    verify BACKUP_ID            Verify backup integrity
    cleanup                     Remove old backups
    schedule                    Setup automated backups

OPTIONS:
    -t, --type TYPE            Backup type: full, incremental (default: incremental)
    -d, --dest DIR             Backup destination (default: $BACKUP_ROOT)
    -s, --source DIRS          Source directories (comma-separated)
    -k, --keep N               Keep N backups (default: $KEEP_BACKUPS)
    -c, --compress             Enable compression (default: enabled)
    -e, --encrypt              Enable encryption (requires GPG setup)
    -f, --force                Force operation without confirmation
    -v, --verbose              Verbose output
    -n, --dry-run              Dry run mode
    -h, --help                 Display this help message

DEFAULT SOURCES:
$(printf "    %s\n" "${DEFAULT_SOURCES[@]}")

EXAMPLES:
    # Full backup with defaults
    $SCRIPT_NAME backup --type full

    # Incremental backup
    $SCRIPT_NAME backup

    # Backup specific directories
    $SCRIPT_NAME backup --source /opt,/home --type full

    # List backups
    $SCRIPT_NAME list

    # Verify backup
    $SCRIPT_NAME verify backup-20240124-140000

    # Restore backup
    $SCRIPT_NAME restore backup-20240124-140000

    # Cleanup old backups
    $SCRIPT_NAME cleanup --keep 5

    # Setup scheduled backups
    $SCRIPT_NAME schedule

    # Dry run
    $SCRIPT_NAME backup --dry-run

BACKUP TYPES:
    full         - Full backup of all files
    incremental  - Only files changed since last full backup

NOTES:
    - Most operations require root privileges
    - Run with sudo for full functionality
    - Incremental backups require a previous full backup
    - Encryption requires GPG configuration
    - Backups are stored in: $BACKUP_ROOT

LOGS:
    Log file: $LOG_FILE

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

    local command="$1"
    shift

    case "$command" in
        backup)
            local sources=()

            while [[ $# -gt 0 ]]; do
                case "$1" in
                    -t|--type)
                        BACKUP_TYPE="$2"
                        shift 2
                        ;;
                    -d|--dest)
                        BACKUP_ROOT="$2"
                        shift 2
                        ;;
                    -s|--source)
                        IFS=',' read -ra sources <<< "$2"
                        shift 2
                        ;;
                    -c|--compress)
                        COMPRESSION=true
                        shift
                        ;;
                    -e|--encrypt)
                        ENCRYPTION=true
                        shift
                        ;;
                    -v|--verbose)
                        VERBOSE=true
                        shift
                        ;;
                    -n|--dry-run)
                        DRY_RUN=true
                        shift
                        ;;
                    *)
                        shift
                        ;;
                esac
            done

            perform_backup "${sources[@]}"
            ;;

        restore)
            if [[ $# -lt 1 ]]; then
                echo "Error: Backup ID required"
                usage
                exit 1
            fi

            local backup_id="$1"
            local dest="${2:-.}"
            shift

            while [[ $# -gt 0 ]]; do
                case "$1" in
                    -f|--force)
                        FORCE=true
                        shift
                        ;;
                    -v|--verbose)
                        VERBOSE=true
                        shift
                        ;;
                    *)
                        shift
                        ;;
                esac
            done

            restore_backup "$backup_id" "$dest"
            ;;

        list)
            list_backups
            ;;

        verify)
            if [[ $# -lt 1 ]]; then
                echo "Error: Backup ID required"
                usage
                exit 1
            fi

            verify_backup_integrity "$1"
            ;;

        cleanup)
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    -k|--keep)
                        KEEP_BACKUPS="$2"
                        shift 2
                        ;;
                    -n|--dry-run)
                        DRY_RUN=true
                        shift
                        ;;
                    -v|--verbose)
                        VERBOSE=true
                        shift
                        ;;
                    *)
                        shift
                        ;;
                esac
            done

            cleanup_old_backups
            ;;

        schedule)
            setup_scheduled_backups
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# ===========================
# Main Function
# ===========================

main() {
    # Ensure log directory exists
    sudo mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    sudo touch "$LOG_FILE" 2>/dev/null || touch "$LOG_FILE" 2>/dev/null || true

    # Ensure backup root exists
    sudo mkdir -p "$BACKUP_ROOT" 2>/dev/null || mkdir -p "$BACKUP_ROOT" 2>/dev/null || true

    log_info "Backup automation script started"

    parse_arguments "$@"
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
