#!/bin/bash
#
# log_rotation.sh - Log management and rotation for ML infrastructure
#
# Usage: ./log_rotation.sh [COMMAND] [OPTIONS]
#

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/log-rotation.log"
readonly LOG_DIRS=("/var/log" "/opt/ml/logs" "/home")
readonly MAX_LOG_SIZE="100M"
readonly LOG_RETENTION_DAYS=30
readonly ARCHIVE_DIR="/var/log/archives"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'
BOLD='\033[1m'

VERBOSE=false
DRY_RUN=false

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
    [[ "$VERBOSE" == true ]] && echo "$*"
}

rotate_logs() {
    echo -e "${BOLD}${CYAN}Rotating Logs${RESET}"
    echo ""

    local rotated=0

    for log_dir in "${LOG_DIRS[@]}"; do
        if [[ ! -d "$log_dir" ]]; then
            continue
        fi

        echo -e "${BLUE}Checking: $log_dir${RESET}"

        # Find large log files
        while IFS= read -r logfile; do
            [[ -z "$logfile" ]] && continue

            local size=$(du -h "$logfile" | cut -f1)
            local filename=$(basename "$logfile")

            echo -e "  ${YELLOW}Large log found: $filename ($size)${RESET}"

            if [[ "$DRY_RUN" == false ]]; then
                # Rotate log
                local timestamp=$(date +%Y%m%d-%H%M%S)
                local archive_name="${filename}.${timestamp}"

                # Compress and move
                gzip -c "$logfile" > "${ARCHIVE_DIR}/${archive_name}.gz" 2>/dev/null && \
                > "$logfile" && \
                echo -e "    ${GREEN}✓ Rotated: $archive_name.gz${RESET}" || \
                echo -e "    ${RED}✗ Failed to rotate${RESET}"

                ((rotated++))
            else
                echo "    [DRY-RUN] Would rotate: $filename"
                ((rotated++))
            fi
        done < <(find "$log_dir" -maxdepth 3 -type f -name "*.log" -size +${MAX_LOG_SIZE} 2>/dev/null)
    done

    echo ""
    echo -e "${GREEN}Rotation complete: $rotated log(s) rotated${RESET}"
    log_message "Rotated $rotated logs"
}

cleanup_old_logs() {
    echo -e "${BOLD}${CYAN}Cleaning Up Old Logs${RESET}"
    echo ""

    local deleted=0

    # Remove old compressed logs
    echo "Removing logs older than $LOG_RETENTION_DAYS days..."

    for log_dir in "${LOG_DIRS[@]}" "$ARCHIVE_DIR"; do
        if [[ ! -d "$log_dir" ]]; then
            continue
        fi

        local old_logs=$(find "$log_dir" -type f \( -name "*.log.gz" -o -name "*.log.*" \) -mtime +$LOG_RETENTION_DAYS 2>/dev/null | wc -l)

        if [[ $old_logs -gt 0 ]]; then
            echo -e "${BLUE}  $log_dir: $old_logs old log(s)${RESET}"

            if [[ "$DRY_RUN" == false ]]; then
                find "$log_dir" -type f \( -name "*.log.gz" -o -name "*.log.*" \) -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null
                deleted=$((deleted + old_logs))
            else
                echo "    [DRY-RUN] Would delete $old_logs file(s)"
            fi
        fi
    done

    echo ""
    echo -e "${GREEN}Cleanup complete: $deleted log(s) deleted${RESET}"
    log_message "Deleted $deleted old logs"
}

compress_logs() {
    echo -e "${BOLD}${CYAN}Compressing Logs${RESET}"
    echo ""

    local compressed=0

    for log_dir in "${LOG_DIRS[@]}"; do
        if [[ ! -d "$log_dir" ]]; then
            continue
        fi

        # Find uncompressed old logs
        while IFS= read -r logfile; do
            [[ -z "$logfile" ]] && continue

            local filename=$(basename "$logfile")

            if [[ "$DRY_RUN" == false ]]; then
                if gzip "$logfile" 2>/dev/null; then
                    echo -e "  ${GREEN}✓ Compressed: $filename${RESET}"
                    ((compressed++))
                fi
            else
                echo "  [DRY-RUN] Would compress: $filename"
                ((compressed++))
            fi
        done < <(find "$log_dir" -maxdepth 3 -type f -name "*.log" -mtime +7 2>/dev/null)
    done

    echo ""
    echo -e "${GREEN}Compression complete: $compressed log(s) compressed${RESET}"
    log_message "Compressed $compressed logs"
}

analyze_logs() {
    echo -e "${BOLD}${CYAN}Log Analysis Report${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    # Total log size
    local total_size=0
    for log_dir in "${LOG_DIRS[@]}"; do
        if [[ -d "$log_dir" ]]; then
            local dir_size=$(du -sh "$log_dir" 2>/dev/null | cut -f1)
            echo "  $log_dir: $dir_size"
        fi
    done

    echo ""
    echo -e "${BOLD}Top 10 Largest Logs:${RESET}"
    find "${LOG_DIRS[@]}" -type f -name "*.log*" 2>/dev/null | \
        xargs du -h 2>/dev/null | sort -rh | head -10 | \
        awk '{printf "  %8s  %s\n", $1, $2}'

    echo ""
    echo -e "${BOLD}Recent Errors (last 24h):${RESET}"
    journalctl --since "24 hours ago" -p err --no-pager -n 10 2>/dev/null || \
        echo "  No recent errors found"

    echo ""
}

vacuum_journal() {
    echo -e "${BOLD}${CYAN}Vacuuming System Journal${RESET}"
    echo ""

    local before=$(journalctl --disk-usage 2>/dev/null | grep -oP '\d+\.\d+[MG]' || echo "Unknown")
    echo "Journal size before: $before"

    if [[ "$DRY_RUN" == false ]]; then
        sudo journalctl --vacuum-time=30d 2>/dev/null
        sudo journalctl --vacuum-size=500M 2>/dev/null
    else
        echo "[DRY-RUN] Would vacuum journal to 30 days / 500M"
    fi

    local after=$(journalctl --disk-usage 2>/dev/null | grep -oP '\d+\.\d+[MG]' || echo "Unknown")
    echo "Journal size after: $after"

    log_message "Journal vacuumed: $before -> $after"
}

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [COMMAND] [OPTIONS]

Log management and rotation for ML infrastructure.

COMMANDS:
    rotate      Rotate large log files
    cleanup     Remove old log files
    compress    Compress uncompressed logs
    analyze     Analyze log usage
    vacuum      Vacuum systemd journal
    all         Run all tasks

OPTIONS:
    -v, --verbose    Verbose output
    -n, --dry-run    Dry run mode
    -h, --help       Display help

EXAMPLES:
    $SCRIPT_NAME rotate
    $SCRIPT_NAME cleanup
    $SCRIPT_NAME all --verbose

EOF
}

main() {
    mkdir -p "$ARCHIVE_DIR" 2>/dev/null || true
    touch "$LOG_FILE" 2>/dev/null || true

    [[ $# -eq 0 ]] && { usage; exit 1; }

    local command="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose) VERBOSE=true; shift ;;
            -n|--dry-run) DRY_RUN=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) shift ;;
        esac
    done

    case "$command" in
        rotate)  rotate_logs ;;
        cleanup) cleanup_old_logs ;;
        compress) compress_logs ;;
        analyze) analyze_logs ;;
        vacuum)  vacuum_journal ;;
        all)
            rotate_logs
            echo ""
            compress_logs
            echo ""
            cleanup_old_logs
            echo ""
            vacuum_journal
            ;;
        *) echo "Unknown command: $command"; usage; exit 1 ;;
    esac
}

main "$@"
