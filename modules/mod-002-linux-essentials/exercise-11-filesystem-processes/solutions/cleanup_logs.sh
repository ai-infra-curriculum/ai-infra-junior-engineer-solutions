#!/bin/bash
#
# cleanup_logs.sh - Automated log file cleanup for ML infrastructure
#
# Description:
#   Manage log files for ML workloads including rotation, compression,
#   archival, and cleanup based on retention policies.
#
# Usage:
#   ./cleanup_logs.sh [OPTIONS] [PATH]
#
# Options:
#   -a, --age DAYS          Delete logs older than DAYS (default: 30)
#   -s, --size SIZE         Delete logs larger than SIZE (e.g., 100M, 1G)
#   -c, --compress          Compress old logs
#   -r, --archive PATH      Archive logs to PATH before deletion
#   -p, --pattern PATTERN   File pattern to match (default: *.log)
#   -d, --dry-run          Show what would be done without doing it
#   -v, --verbose          Verbose output
#   -f, --force            Skip confirmation prompts
#   -k, --keep N           Keep N most recent logs
#   -t, --rotate           Rotate logs (backup with timestamp)
#   -h, --help             Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/log-cleanup.log"

# Defaults
LOG_PATH="/var/log"
MAX_AGE_DAYS=30
MAX_SIZE=""
COMPRESS_OLD=false
ARCHIVE_PATH=""
FILE_PATTERN="*.log"
DRY_RUN=false
VERBOSE=false
FORCE=false
KEEP_COUNT=0
ROTATE_LOGS=false

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

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [[ "$VERBOSE" == true ]]; then
        echo -e "[${timestamp}] [${level}] ${message}"
    fi

    echo "[${timestamp}] [${level}] ${message}" >> "$LOG_FILE"
}

log_action() {
    local action="$1"
    local file="$2"
    local result="${3:-}"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[DRY-RUN] $action: $file $result${RESET}"
    else
        echo -e "${GREEN}$action: $file $result${RESET}"
    fi

    log "INFO" "$action: $file $result"
}

# ===========================
# File Operations
# ===========================

get_file_age_days() {
    local file="$1"

    # Get file modification time
    local file_time
    if [[ "$OSTYPE" == "darwin"* ]]; then
        file_time=$(stat -f %m "$file")
    else
        file_time=$(stat -c %Y "$file")
    fi

    local current_time=$(date +%s)
    local age_seconds=$((current_time - file_time))
    local age_days=$((age_seconds / 86400))

    echo "$age_days"
}

get_file_size_bytes() {
    local file="$1"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        stat -f %z "$file"
    else
        stat -c %s "$file"
    fi
}

size_to_bytes() {
    local size=$1

    # Extract number and unit
    local number=$(echo "$size" | sed 's/[^0-9.]//g')
    local unit=$(echo "$size" | sed 's/[0-9.]//g' | tr '[:lower:]' '[:upper:]')

    case "$unit" in
        K|KB)
            echo "$((${number%.*} * 1024))"
            ;;
        M|MB)
            echo "$((${number%.*} * 1024 * 1024))"
            ;;
        G|GB)
            echo "$((${number%.*} * 1024 * 1024 * 1024))"
            ;;
        T|TB)
            echo "$((${number%.*} * 1024 * 1024 * 1024 * 1024))"
            ;;
        *)
            echo "${number%.*}"
            ;;
    esac
}

human_readable_size() {
    local bytes=$1
    local units=("B" "KB" "MB" "GB" "TB")
    local unit_index=0

    while [[ $bytes -gt 1024 ]] && [[ $unit_index -lt 4 ]]; do
        bytes=$((bytes / 1024))
        ((unit_index++))
    done

    echo "${bytes}${units[$unit_index]}"
}

# ===========================
# Log Analysis
# ===========================

analyze_logs() {
    echo -e "${BOLD}${CYAN}Log File Analysis${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    log "INFO" "Starting log analysis in $LOG_PATH"

    # Find all log files
    local all_logs=$(find "$LOG_PATH" -type f -name "$FILE_PATTERN" 2>/dev/null)
    local total_count=$(echo "$all_logs" | grep -c . || echo 0)

    if [[ $total_count -eq 0 ]]; then
        echo "No log files found matching pattern: $FILE_PATTERN"
        return 0
    fi

    echo "Total log files: $total_count"
    echo ""

    # Calculate total size
    local total_size=0
    while IFS= read -r file; do
        local size=$(get_file_size_bytes "$file")
        total_size=$((total_size + size))
    done <<< "$all_logs"

    local total_size_hr=$(human_readable_size "$total_size")
    echo "Total size: $total_size_hr"
    echo ""

    # Count by age
    local old_count=0
    local old_size=0

    while IFS= read -r file; do
        local age=$(get_file_age_days "$file")
        if [[ $age -gt $MAX_AGE_DAYS ]]; then
            ((old_count++))
            local size=$(get_file_size_bytes "$file")
            old_size=$((old_size + size))
        fi
    done <<< "$all_logs"

    if [[ $old_count -gt 0 ]]; then
        local old_size_hr=$(human_readable_size "$old_size")
        echo -e "${YELLOW}Old files (>$MAX_AGE_DAYS days): $old_count files, $old_size_hr${RESET}"
    else
        echo "No old files to clean up"
    fi

    echo ""

    # Largest files
    echo -e "${BOLD}Top 10 Largest Log Files:${RESET}"
    find "$LOG_PATH" -type f -name "$FILE_PATTERN" -exec ls -lh {} \; 2>/dev/null | \
        sort -k5 -hr | \
        head -10 | \
        awk '{printf "  %-10s  %-20s  %s\n", $5, $6" "$7" "$8, $9}'

    echo ""
}

# ===========================
# Compression
# ===========================

compress_file() {
    local file="$1"

    if [[ "$DRY_RUN" == true ]]; then
        log_action "Would compress" "$file"
        return 0
    fi

    log "INFO" "Compressing $file"

    if gzip -9 "$file" 2>/dev/null; then
        log_action "Compressed" "$file" "-> ${file}.gz"
        return 0
    else
        echo -e "${RED}Failed to compress: $file${RESET}"
        log "ERROR" "Failed to compress $file"
        return 1
    fi
}

compress_old_logs() {
    echo -e "${BOLD}Compressing Old Logs${RESET}"
    echo "-------------------------------------"

    local compressed_count=0

    find "$LOG_PATH" -type f -name "$FILE_PATTERN" 2>/dev/null | while read -r file; do
        local age=$(get_file_age_days "$file")

        # Compress files older than 7 days but keep files older than MAX_AGE_DAYS for deletion
        if [[ $age -gt 7 ]] && [[ $age -le $MAX_AGE_DAYS ]]; then
            if compress_file "$file"; then
                ((compressed_count++))
            fi
        fi
    done

    echo ""
    echo "Compressed $compressed_count files"
    echo ""

    log "INFO" "Compressed $compressed_count log files"
}

# ===========================
# Archival
# ===========================

archive_file() {
    local file="$1"
    local archive_dest="$ARCHIVE_PATH/$(basename "$file")"

    if [[ "$DRY_RUN" == true ]]; then
        log_action "Would archive" "$file" "-> $archive_dest"
        return 0
    fi

    # Create archive directory if it doesn't exist
    mkdir -p "$ARCHIVE_PATH"

    log "INFO" "Archiving $file to $archive_dest"

    if cp -p "$file" "$archive_dest"; then
        # Compress archived file
        if gzip -9 "$archive_dest" 2>/dev/null; then
            log_action "Archived and compressed" "$file" "-> ${archive_dest}.gz"
            return 0
        else
            log_action "Archived" "$file" "-> $archive_dest"
            return 0
        fi
    else
        echo -e "${RED}Failed to archive: $file${RESET}"
        log "ERROR" "Failed to archive $file"
        return 1
    fi
}

# ===========================
# Rotation
# ===========================

rotate_log() {
    local file="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local rotated="${file}.${timestamp}"

    if [[ "$DRY_RUN" == true ]]; then
        log_action "Would rotate" "$file" "-> $rotated"
        return 0
    fi

    log "INFO" "Rotating $file"

    if mv "$file" "$rotated"; then
        # Create new empty log file
        touch "$file"

        # Preserve permissions
        if [[ -f "$rotated" ]]; then
            chmod --reference="$rotated" "$file" 2>/dev/null || true
            chown --reference="$rotated" "$file" 2>/dev/null || true
        fi

        log_action "Rotated" "$file" "-> $rotated"

        # Compress rotated file
        if [[ "$COMPRESS_OLD" == true ]]; then
            compress_file "$rotated"
        fi

        return 0
    else
        echo -e "${RED}Failed to rotate: $file${RESET}"
        log "ERROR" "Failed to rotate $file"
        return 1
    fi
}

rotate_logs() {
    echo -e "${BOLD}Rotating Logs${RESET}"
    echo "-------------------------------------"

    local rotated_count=0

    find "$LOG_PATH" -type f -name "$FILE_PATTERN" 2>/dev/null | while read -r file; do
        # Skip empty files
        local size=$(get_file_size_bytes "$file")
        if [[ $size -gt 0 ]]; then
            if rotate_log "$file"; then
                ((rotated_count++))
            fi
        fi
    done

    echo ""
    echo "Rotated $rotated_count log files"
    echo ""

    log "INFO" "Rotated $rotated_count log files"
}

# ===========================
# Cleanup
# ===========================

cleanup_old_logs() {
    echo -e "${BOLD}Cleaning Up Old Logs${RESET}"
    echo "-------------------------------------"

    local deleted_count=0
    local deleted_size=0

    # Find old files
    local old_files=$(find "$LOG_PATH" -type f -name "$FILE_PATTERN" 2>/dev/null | while read -r file; do
        local age=$(get_file_age_days "$file")
        if [[ $age -gt $MAX_AGE_DAYS ]]; then
            echo "$file"
        fi
    done)

    if [[ -z "$old_files" ]]; then
        echo "No old log files to clean up"
        return 0
    fi

    # Count files
    local file_count=$(echo "$old_files" | grep -c . || echo 0)

    # Calculate total size
    local total_size=0
    while IFS= read -r file; do
        local size=$(get_file_size_bytes "$file")
        total_size=$((total_size + size))
    done <<< "$old_files"

    local total_size_hr=$(human_readable_size "$total_size")

    echo "Found $file_count old log files (>$MAX_AGE_DAYS days)"
    echo "Total size: $total_size_hr"
    echo ""

    # Confirm deletion
    if [[ "$FORCE" == false ]] && [[ "$DRY_RUN" == false ]]; then
        echo -e "${YELLOW}Delete these files? (y/N)${RESET}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Cleanup cancelled"
            return 0
        fi
    fi

    # Process files
    while IFS= read -r file; do
        # Archive if requested
        if [[ -n "$ARCHIVE_PATH" ]]; then
            archive_file "$file" || continue
        fi

        # Delete file
        if [[ "$DRY_RUN" == true ]]; then
            log_action "Would delete" "$file"
        else
            if rm -f "$file"; then
                log_action "Deleted" "$file"
                ((deleted_count++))
                deleted_size=$((deleted_size + $(get_file_size_bytes "$file" 2>/dev/null || echo 0)))
            else
                echo -e "${RED}Failed to delete: $file${RESET}"
                log "ERROR" "Failed to delete $file"
            fi
        fi
    done <<< "$old_files"

    echo ""
    local deleted_size_hr=$(human_readable_size "$deleted_size")
    echo -e "${GREEN}Cleanup complete: $deleted_count files, $deleted_size_hr freed${RESET}"
    echo ""

    log "INFO" "Cleaned up $deleted_count files, freed $deleted_size_hr"
}

cleanup_large_logs() {
    if [[ -z "$MAX_SIZE" ]]; then
        return 0
    fi

    echo -e "${BOLD}Cleaning Up Large Logs${RESET}"
    echo "-------------------------------------"

    local max_size_bytes=$(size_to_bytes "$MAX_SIZE")
    local deleted_count=0

    find "$LOG_PATH" -type f -name "$FILE_PATTERN" 2>/dev/null | while read -r file; do
        local size=$(get_file_size_bytes "$file")

        if [[ $size -gt $max_size_bytes ]]; then
            local size_hr=$(human_readable_size "$size")
            echo "Large file: $file ($size_hr)"

            # Archive if requested
            if [[ -n "$ARCHIVE_PATH" ]]; then
                archive_file "$file"
            fi

            # Delete file
            if [[ "$DRY_RUN" == true ]]; then
                log_action "Would delete" "$file" "($size_hr)"
            else
                if [[ "$FORCE" == true ]] || confirm_delete "$file"; then
                    if rm -f "$file"; then
                        log_action "Deleted" "$file" "($size_hr)"
                        ((deleted_count++))
                    fi
                fi
            fi
        fi
    done

    echo ""
    echo "Cleaned up $deleted_count large files"
    echo ""

    log "INFO" "Cleaned up $deleted_count large log files"
}

confirm_delete() {
    local file="$1"

    echo -ne "${YELLOW}Delete $file? (y/N)${RESET} "
    read -r response
    [[ "$response" =~ ^[Yy]$ ]]
}

# ===========================
# Keep Recent Files
# ===========================

keep_recent_logs() {
    if [[ $KEEP_COUNT -eq 0 ]]; then
        return 0
    fi

    echo -e "${BOLD}Keeping $KEEP_COUNT Most Recent Logs${RESET}"
    echo "-------------------------------------"

    # Get files sorted by modification time
    find "$LOG_PATH" -type f -name "$FILE_PATTERN" -printf '%T@ %p\n' 2>/dev/null | \
        sort -rn | \
        tail -n +$((KEEP_COUNT + 1)) | \
        cut -d' ' -f2- | \
        while read -r file; do
            # Archive if requested
            if [[ -n "$ARCHIVE_PATH" ]]; then
                archive_file "$file"
            fi

            # Delete file
            if [[ "$DRY_RUN" == true ]]; then
                log_action "Would delete (keeping recent)" "$file"
            else
                if rm -f "$file"; then
                    log_action "Deleted (keeping recent)" "$file"
                fi
            fi
        done

    echo ""
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] [PATH]

Automated log file cleanup for ML infrastructure.

OPTIONS:
    -a, --age DAYS            Delete logs older than DAYS (default: $MAX_AGE_DAYS)
    -s, --size SIZE           Delete logs larger than SIZE (e.g., 100M, 1G)
    -c, --compress            Compress old logs before cleanup
    -r, --archive PATH        Archive logs to PATH before deletion
    -p, --pattern PATTERN     File pattern to match (default: $FILE_PATTERN)
    -d, --dry-run            Show what would be done without doing it
    -v, --verbose            Verbose output
    -f, --force              Skip confirmation prompts
    -k, --keep N             Keep N most recent logs
    -t, --rotate             Rotate logs (backup with timestamp)
    -h, --help               Display this help message

EXAMPLES:
    # Analyze logs
    $SCRIPT_NAME /var/log/ml

    # Clean up logs older than 30 days
    $SCRIPT_NAME -a 30 /var/log/ml

    # Dry-run cleanup
    $SCRIPT_NAME -d -a 30 /var/log/ml

    # Compress and archive before cleanup
    $SCRIPT_NAME -c -r /backup/logs -a 30 /var/log/ml

    # Rotate logs
    $SCRIPT_NAME -t -c /var/log/ml

    # Keep only 10 most recent logs
    $SCRIPT_NAME -k 10 /var/log/ml

    # Clean up logs larger than 1GB
    $SCRIPT_NAME -s 1G /var/log/ml

    # Comprehensive cleanup
    $SCRIPT_NAME -a 30 -c -r /backup -k 10 -v /var/log/ml

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -a|--age)
                MAX_AGE_DAYS="$2"
                shift 2
                ;;
            -s|--size)
                MAX_SIZE="$2"
                shift 2
                ;;
            -c|--compress)
                COMPRESS_OLD=true
                shift
                ;;
            -r|--archive)
                ARCHIVE_PATH="$2"
                shift 2
                ;;
            -p|--pattern)
                FILE_PATTERN="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -k|--keep)
                KEEP_COUNT="$2"
                shift 2
                ;;
            -t|--rotate)
                ROTATE_LOGS=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                LOG_PATH="$1"
                shift
                ;;
        esac
    done
}

# ===========================
# Main Function
# ===========================

main() {
    parse_arguments "$@"

    # Validate path
    if [[ ! -d "$LOG_PATH" ]]; then
        echo -e "${RED}Error: Directory does not exist: $LOG_PATH${RESET}"
        exit 1
    fi

    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"

    log "INFO" "========================================="
    log "INFO" "Log Cleanup Script Started"
    log "INFO" "Path: $LOG_PATH"
    log "INFO" "Pattern: $FILE_PATTERN"
    log "INFO" "Max Age: $MAX_AGE_DAYS days"
    if [[ -n "$MAX_SIZE" ]]; then
        log "INFO" "Max Size: $MAX_SIZE"
    fi
    log "INFO" "========================================="

    # Show dry-run notice
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}${BOLD}DRY-RUN MODE - No changes will be made${RESET}"
        echo ""
    fi

    # Analyze logs
    analyze_logs

    # Rotate logs if requested
    if [[ "$ROTATE_LOGS" == true ]]; then
        rotate_logs
    fi

    # Compress old logs if requested
    if [[ "$COMPRESS_OLD" == true ]]; then
        compress_old_logs
    fi

    # Keep recent logs
    if [[ $KEEP_COUNT -gt 0 ]]; then
        keep_recent_logs
    fi

    # Clean up old logs
    cleanup_old_logs

    # Clean up large logs
    if [[ -n "$MAX_SIZE" ]]; then
        cleanup_large_logs
    fi

    echo -e "${GREEN}${BOLD}Log cleanup complete!${RESET}"
    log "INFO" "Log cleanup completed successfully"
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
