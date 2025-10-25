#!/bin/bash
#
# disk_manager.sh - Disk management and monitoring for ML infrastructure
#
# Usage: ./disk_manager.sh [COMMAND] [OPTIONS]
#

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/disk-manager.log"
readonly ALERT_THRESHOLD=80
readonly LARGE_FILE_SIZE="100M"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'
BOLD='\033[1m'

VERBOSE=false

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
    [[ "$VERBOSE" == true ]] && echo "$*"
}

check_disk_usage() {
    echo -e "${BOLD}${CYAN}Disk Usage Report${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    echo -e "${BOLD}Filesystem Usage:${RESET}"

    df -h | grep -E '^Filesystem|^/dev/' | while read -r line; do
        if echo "$line" | grep -q "Filesystem"; then
            echo "$line"
        else
            local usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')
            local mount=$(echo "$line" | awk '{print $6}')

            if [[ $usage -ge $ALERT_THRESHOLD ]]; then
                echo -e "${RED}$line${RESET} [CRITICAL]"
                log_message "ALERT: High disk usage on $mount: ${usage}%"
            elif [[ $usage -ge $((ALERT_THRESHOLD - 10)) ]]; then
                echo -e "${YELLOW}$line${RESET} [WARNING]"
                log_message "WARNING: Disk usage on $mount: ${usage}%"
            else
                echo "$line"
            fi
        fi
    done

    echo ""
    echo -e "${BOLD}Inode Usage:${RESET}"
    df -i | grep -E '^Filesystem|^/dev/' | head -5
    echo ""
}

find_large_files() {
    local directory="${1:-.}"
    local top_n="${2:-20}"

    echo -e "${BOLD}${CYAN}Largest Files in $directory${RESET}"
    echo ""

    echo "Searching for files larger than $LARGE_FILE_SIZE..."
    echo ""

    sudo find "$directory" -type f -size +${LARGE_FILE_SIZE} 2>/dev/null | \
        xargs -I {} ls -lh {} 2>/dev/null | \
        sort -k5 -hr | \
        head -"$top_n" | \
        awk '{printf "%8s  %s\n", $5, $9}' || \
        echo "No large files found"

    echo ""
    log_message "Large files search completed in $directory"
}

find_large_directories() {
    local directory="${1:-.}"
    local top_n="${2:-10}"

    echo -e "${BOLD}${CYAN}Largest Directories in $directory${RESET}"
    echo ""

    sudo du -h "$directory" 2>/dev/null | \
        sort -rh | \
        head -"$top_n" | \
        awk '{printf "%8s  %s\n", $1, $2}' || \
        echo "Could not analyze directories"

    echo ""
    log_message "Large directories search completed in $directory"
}

cleanup_temp_files() {
    echo -e "${BOLD}${CYAN}Cleaning Temporary Files${RESET}"
    echo ""

    local cleaned_size=0
    local cleaned_count=0

    # /tmp files older than 7 days
    echo "Cleaning /tmp..."
    local tmp_files=$(find /tmp -type f -mtime +7 2>/dev/null | wc -l)
    if [[ $tmp_files -gt 0 ]]; then
        echo -e "  ${BLUE}Found $tmp_files file(s)${RESET}"
        sudo find /tmp -type f -mtime +7 -delete 2>/dev/null
        cleaned_count=$((cleaned_count + tmp_files))
        echo -e "  ${GREEN}✓ Cleaned /tmp${RESET}"
    else
        echo "  No old files in /tmp"
    fi

    # /var/tmp files older than 30 days
    echo "Cleaning /var/tmp..."
    local vartmp_files=$(find /var/tmp -type f -mtime +30 2>/dev/null | wc -l)
    if [[ $vartmp_files -gt 0 ]]; then
        echo -e "  ${BLUE}Found $vartmp_files file(s)${RESET}"
        sudo find /var/tmp -type f -mtime +30 -delete 2>/dev/null
        cleaned_count=$((cleaned_count + vartmp_files))
        echo -e "  ${GREEN}✓ Cleaned /var/tmp${RESET}"
    else
        echo "  No old files in /var/tmp"
    fi

    # Package manager cache
    echo "Cleaning package cache..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get clean 2>/dev/null
        echo -e "  ${GREEN}✓ Cleaned APT cache${RESET}"
    fi

    echo ""
    echo -e "${GREEN}Cleanup completed: $cleaned_count file(s) removed${RESET}"
    log_message "Cleanup completed: $cleaned_count files removed"
}

analyze_ml_directories() {
    echo -e "${BOLD}${CYAN}ML Data Directory Analysis${RESET}"
    echo ""

    local ml_dirs=(
        "/data"
        "/opt/ml"
        "/var/lib/docker"
        "/home"
    )

    for dir in "${ml_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "  $dir: $size"
        fi
    done

    echo ""

    # Docker specific
    if command -v docker &>/dev/null; then
        echo -e "${BOLD}Docker Storage:${RESET}"
        docker system df 2>/dev/null || echo "  Docker not running"
        echo ""
    fi
}

check_smart_status() {
    echo -e "${BOLD}${CYAN}SMART Disk Health${RESET}"
    echo ""

    if ! command -v smartctl &>/dev/null; then
        echo -e "${YELLOW}⚠ smartctl not installed${RESET}"
        echo "  Install: apt install smartmontools"
        return
    fi

    # Get all disks
    local disks=$(lsblk -d -o NAME,TYPE | awk '$2=="disk" {print "/dev/" $1}')

    for disk in $disks; do
        echo -e "${BOLD}Disk: $disk${RESET}"

        local health=$(sudo smartctl -H "$disk" 2>/dev/null | grep "SMART overall-health" | awk '{print $NF}' || echo "UNKNOWN")

        if [[ "$health" == "PASSED" ]]; then
            echo -e "  ${GREEN}✓ Health: $health${RESET}"
        else
            echo -e "  ${RED}✗ Health: $health${RESET}"
            log_message "ALERT: Disk health issue on $disk: $health"
        fi

        # Temperature
        local temp=$(sudo smartctl -A "$disk" 2>/dev/null | grep "Temperature_Celsius" | awk '{print $10}' || echo "N/A")
        echo "  Temperature: ${temp}°C"

        echo ""
    done

    log_message "SMART health check completed"
}

monitor_io_stats() {
    echo -e "${BOLD}${CYAN}I/O Statistics${RESET}"
    echo ""

    if ! command -v iostat &>/dev/null; then
        echo -e "${YELLOW}⚠ iostat not installed${RESET}"
        echo "  Install: apt install sysstat"
        return
    fi

    echo "Current I/O statistics:"
    iostat -x 1 2 | tail -n +4

    echo ""
}

generate_disk_report() {
    local report_file="/var/log/disk-report-$(date +%Y%m%d-%H%M%S).txt"

    echo "Generating disk report..."

    {
        echo "========================================"
        echo "Disk Management Report"
        echo "========================================"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Hostname: $(hostname)"
        echo ""

        check_disk_usage
        echo ""

        analyze_ml_directories
        echo ""

        find_large_directories "/" 15
        echo ""

        check_smart_status
        echo ""

        echo "========================================"
        echo "End of Report"
        echo "========================================"
    } | tee "$report_file"

    echo ""
    echo -e "${GREEN}✓ Report saved: $report_file${RESET}"
    log_message "Disk report generated: $report_file"
}

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [COMMAND] [OPTIONS]

Disk management and monitoring for ML infrastructure.

COMMANDS:
    check                    Check disk usage
    large-files [DIR]        Find largest files
    large-dirs [DIR]         Find largest directories
    cleanup                  Clean up temporary files
    analyze                  Analyze ML directories
    smart                    Check SMART disk health
    iostat                   Show I/O statistics
    report                   Generate comprehensive report
    monitor                  Continuous monitoring

OPTIONS:
    -t, --threshold N        Alert threshold percentage (default: 80)
    -n, --top N              Number of results (default: 10)
    -v, --verbose            Verbose output
    -h, --help               Display help

EXAMPLES:
    $SCRIPT_NAME check
    $SCRIPT_NAME large-files /home
    $SCRIPT_NAME cleanup
    $SCRIPT_NAME smart
    $SCRIPT_NAME report

LOGS:
    Log file: $LOG_FILE

EOF
}

main() {
    touch "$LOG_FILE" 2>/dev/null || true

    [[ $# -eq 0 ]] && { usage; exit 1; }

    local command="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose) VERBOSE=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) break ;;
        esac
    done

    log_message "Disk manager started - Command: $command"

    case "$command" in
        check)
            check_disk_usage
            ;;
        large-files)
            find_large_files "${1:-.}" "${2:-20}"
            ;;
        large-dirs)
            find_large_directories "${1:-.}" "${2:-10}"
            ;;
        cleanup)
            cleanup_temp_files
            ;;
        analyze)
            analyze_ml_directories
            ;;
        smart)
            check_smart_status
            ;;
        iostat)
            monitor_io_stats
            ;;
        report)
            generate_disk_report
            ;;
        monitor)
            while true; do
                clear
                check_disk_usage
                analyze_ml_directories
                sleep 60
            done
            ;;
        *)
            echo "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
