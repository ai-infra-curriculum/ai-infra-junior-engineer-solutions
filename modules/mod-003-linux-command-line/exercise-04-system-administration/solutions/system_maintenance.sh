#!/bin/bash
#
# system_maintenance.sh - Automated system maintenance for ML infrastructure
#
# Description:
#   Comprehensive system maintenance including updates, cleanup, log rotation,
#   disk space management, and health reporting.
#
# Usage:
#   ./system_maintenance.sh [OPTIONS]
#
# Options:
#   -u, --update          Update system packages
#   -c, --cleanup         Clean up old files and packages
#   -l, --logs            Rotate and clean logs
#   -d, --disk            Analyze disk usage
#   -r, --report FILE     Generate maintenance report
#   -a, --all             Run all maintenance tasks
#   -n, --dry-run         Show what would be done
#   -v, --verbose         Verbose output
#   -h, --help            Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/system-maintenance.log"

# Defaults
UPDATE_PACKAGES=false
CLEANUP_FILES=false
ROTATE_LOGS=false
ANALYZE_DISK=false
REPORT_FILE=""
ALL_TASKS=false
DRY_RUN=false
VERBOSE=false

# Maintenance settings
readonly OLD_FILE_DAYS=30
readonly LOG_MAX_SIZE="100M"
readonly DISK_WARN_THRESHOLD=80
readonly PACKAGE_CACHE_DAYS=7

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

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"

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
# System Updates
# ===========================

update_packages() {
    echo -e "${BOLD}${CYAN}[1/4] Updating System Packages${RESET}"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would update packages"
        return 0
    fi

    # Detect package manager
    if command -v apt-get &> /dev/null; then
        update_apt_packages
    elif command -v yum &> /dev/null; then
        update_yum_packages
    elif command -v dnf &> /dev/null; then
        update_dnf_packages
    else
        log_warning "Package manager not recognized"
        return 1
    fi
}

update_apt_packages() {
    log_info "Updating APT packages..."

    # Update package lists
    echo "Updating package lists..."
    if sudo apt-get update; then
        log_success "Package lists updated"
    else
        log_error "Failed to update package lists"
        return 1
    fi

    # Check for upgradable packages
    local upgradable=$(apt list --upgradable 2>/dev/null | grep -c upgradable || echo 0)
    echo "Upgradable packages: $upgradable"

    if [[ $upgradable -eq 0 ]]; then
        echo -e "${GREEN}System is up to date${RESET}"
        return 0
    fi

    # Upgrade packages
    echo "Upgrading packages..."
    if sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y; then
        log_success "Packages upgraded successfully"
    else
        log_error "Package upgrade failed"
        return 1
    fi

    # Check for security updates
    local security_updates=$(apt list --upgradable 2>/dev/null | grep -ci security || echo 0)
    if [[ $security_updates -gt 0 ]]; then
        log_warning "$security_updates security updates still available"
        echo "Installing security updates..."
        sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
    fi

    echo -e "${GREEN}✓ System packages updated${RESET}"
    echo ""
}

update_yum_packages() {
    log_info "Updating YUM packages..."

    if sudo yum update -y; then
        log_success "YUM packages updated"
        echo -e "${GREEN}✓ System packages updated${RESET}"
    else
        log_error "YUM update failed"
        return 1
    fi

    echo ""
}

update_dnf_packages() {
    log_info "Updating DNF packages..."

    if sudo dnf upgrade -y; then
        log_success "DNF packages updated"
        echo -e "${GREEN}✓ System packages updated${RESET}"
    else
        log_error "DNF upgrade failed"
        return 1
    fi

    echo ""
}

# ===========================
# Cleanup Operations
# ===========================

cleanup_system() {
    echo -e "${BOLD}${CYAN}[2/4] Cleaning Up System${RESET}"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would clean up system"
        return 0
    fi

    cleanup_package_cache
    cleanup_old_kernels
    cleanup_temp_files
    cleanup_old_logs

    echo -e "${GREEN}✓ System cleanup completed${RESET}"
    echo ""
}

cleanup_package_cache() {
    echo "Cleaning package cache..."

    if command -v apt-get &> /dev/null; then
        local cache_size_before=$(du -sh /var/cache/apt/archives 2>/dev/null | cut -f1 || echo "0")

        sudo apt-get clean
        sudo apt-get autoclean
        sudo apt-get autoremove -y

        local cache_size_after=$(du -sh /var/cache/apt/archives 2>/dev/null | cut -f1 || echo "0")

        echo "  Cache cleaned (was: $cache_size_before, now: $cache_size_after)"
        log_info "Package cache cleaned"

    elif command -v yum &> /dev/null; then
        sudo yum clean all
        log_info "YUM cache cleaned"

    elif command -v dnf &> /dev/null; then
        sudo dnf clean all
        log_info "DNF cache cleaned"
    fi
}

cleanup_old_kernels() {
    echo "Checking for old kernels..."

    if command -v apt-get &> /dev/null; then
        local current_kernel=$(uname -r)
        local old_kernels=$(dpkg -l 'linux-image-*' | grep "^ii" | awk '{print $2}' | grep -v "$current_kernel" || true)

        if [[ -n "$old_kernels" ]]; then
            local count=$(echo "$old_kernels" | wc -l)
            echo "  Found $count old kernel(s)"

            # Keep one old kernel for safety
            if [[ $count -gt 1 ]]; then
                echo "  Removing old kernels (keeping one for rollback)..."
                echo "$old_kernels" | head -n -1 | xargs sudo apt-get purge -y 2>/dev/null || true
                log_info "Removed old kernels"
            fi
        else
            echo "  No old kernels to remove"
        fi
    fi
}

cleanup_temp_files() {
    echo "Cleaning temporary files..."

    # Clean /tmp (files older than 7 days)
    local tmp_count=$(find /tmp -type f -mtime +7 2>/dev/null | wc -l)
    if [[ $tmp_count -gt 0 ]]; then
        echo "  Removing $tmp_count old files from /tmp..."
        sudo find /tmp -type f -mtime +7 -delete 2>/dev/null || true
        log_info "Cleaned $tmp_count temporary files"
    else
        echo "  No old temporary files to remove"
    fi

    # Clean user cache directories
    if [[ -d "$HOME/.cache" ]]; then
        local cache_size=$(du -sh "$HOME/.cache" 2>/dev/null | cut -f1 || echo "0")
        echo "  User cache size: $cache_size"
    fi
}

cleanup_old_logs() {
    echo "Cleaning old log files..."

    # Find old log files
    local old_logs=$(find /var/log -type f -name "*.log" -mtime +$OLD_FILE_DAYS 2>/dev/null || true)
    local log_count=$(echo "$old_logs" | grep -c . || echo 0)

    if [[ $log_count -gt 0 ]]; then
        echo "  Found $log_count old log files"
        echo "  Compressing old logs..."

        echo "$old_logs" | while read -r logfile; do
            if [[ -f "$logfile" ]] && [[ ! "$logfile" =~ \.gz$ ]]; then
                sudo gzip "$logfile" 2>/dev/null || true
            fi
        done

        log_info "Compressed $log_count old log files"
    else
        echo "  No old log files to clean"
    fi
}

# ===========================
# Log Management
# ===========================

rotate_logs() {
    echo -e "${BOLD}${CYAN}[3/4] Rotating Logs${RESET}"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would rotate logs"
        return 0
    fi

    # Force logrotate
    if command -v logrotate &> /dev/null; then
        echo "Running logrotate..."
        sudo logrotate -f /etc/logrotate.conf 2>&1 | head -20
        log_info "Logs rotated"
    else
        echo "logrotate not installed"
    fi

    # Vacuum systemd journal
    echo "Cleaning systemd journal..."
    local journal_size_before=$(sudo journalctl --disk-usage | grep -oP '\d+\.\d+[MG]')

    sudo journalctl --vacuum-time=7d
    sudo journalctl --vacuum-size=100M

    local journal_size_after=$(sudo journalctl --disk-usage | grep -oP '\d+\.\d+[MG]')

    echo "  Journal size: $journal_size_before -> $journal_size_after"
    log_info "Systemd journal cleaned"

    echo -e "${GREEN}✓ Log rotation completed${RESET}"
    echo ""
}

# ===========================
# Disk Analysis
# ===========================

analyze_disk_usage() {
    echo -e "${BOLD}${CYAN}[4/4] Analyzing Disk Usage${RESET}"
    echo ""

    # Filesystem usage
    echo -e "${BOLD}Filesystem Usage:${RESET}"
    df -h | grep -E '^Filesystem|^/dev/' | while read -r line; do
        if echo "$line" | grep -q "Filesystem"; then
            echo "$line"
        else
            local usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')
            if [[ $usage -ge $DISK_WARN_THRESHOLD ]]; then
                echo -e "${RED}$line${RESET}"
                log_warning "High disk usage: $line"
            else
                echo "$line"
            fi
        fi
    done
    echo ""

    # Top directories
    echo -e "${BOLD}Top 10 Directories by Size:${RESET}"
    sudo du -h / 2>/dev/null | sort -rh | head -10 || \
        echo "Could not analyze all directories (permission denied)"
    echo ""

    # ML-specific directories
    echo -e "${BOLD}ML Data Directories:${RESET}"
    for dir in /data /opt/ml /var/log /home; do
        if [[ -d "$dir" ]]; then
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "N/A")
            echo "  $dir: $size"
        fi
    done
    echo ""

    # Inode usage
    echo -e "${BOLD}Inode Usage:${RESET}"
    df -i | grep -E '^Filesystem|^/dev/' | head -5
    echo ""

    log_info "Disk usage analyzed"
}

# ===========================
# Report Generation
# ===========================

generate_maintenance_report() {
    local report=$(cat <<EOF
========================================
System Maintenance Report
========================================
Generated: $(date '+%Y-%m-%d %H:%M:%S')
Hostname: $(hostname)
Kernel: $(uname -r)
Uptime: $(uptime -p)

System Information:
-------------------
$(lsb_release -d 2>/dev/null | cut -f2- || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
Memory: $(free -h | awk 'NR==2 {print $3 "/" $2}')
CPU Cores: $(nprep --all)

Disk Usage:
-----------
$(df -h | grep -E '^Filesystem|^/dev/')

Package Status:
---------------
Installed Packages: $(dpkg -l 2>/dev/null | grep "^ii" | wc -l || rpm -qa 2>/dev/null | wc -l || echo "N/A")
Upgradable Packages: $(apt list --upgradable 2>/dev/null | grep -c upgradable || echo "N/A")

Service Status:
---------------
$(systemctl list-units --type=service --state=failed --no-pager --no-legend | wc -l) failed services
$(systemctl list-units --type=service --state=running --no-pager --no-legend | wc -l) running services

Recent Errors (last 24h):
-------------------------
$(sudo journalctl --since "24 hours ago" -p err --no-pager -n 10 2>/dev/null || echo "Could not retrieve journal")

========================================
Maintenance Tasks Completed:
========================================
$(tail -20 "$LOG_FILE" 2>/dev/null || echo "No recent logs")

========================================
EOF
)

    echo "$report"

    if [[ -n "$REPORT_FILE" ]]; then
        echo "$report" > "$REPORT_FILE"
        echo ""
        echo "Report saved to: $REPORT_FILE"
        log_info "Maintenance report generated: $REPORT_FILE"
    fi
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Automated system maintenance for ML infrastructure.

OPTIONS:
    -u, --update            Update system packages
    -c, --cleanup           Clean up old files and packages
    -l, --logs              Rotate and clean logs
    -d, --disk              Analyze disk usage
    -r, --report FILE       Generate maintenance report
    -a, --all               Run all maintenance tasks
    -n, --dry-run          Show what would be done
    -v, --verbose          Verbose output
    -h, --help             Display this help message

EXAMPLES:
    # Run all maintenance tasks
    $SCRIPT_NAME --all

    # Update packages only
    $SCRIPT_NAME --update

    # Cleanup and generate report
    $SCRIPT_NAME --cleanup --report maintenance.txt

    # Dry-run (see what would be done)
    $SCRIPT_NAME --all --dry-run

    # Comprehensive maintenance with report
    $SCRIPT_NAME -a -r /var/log/maintenance-\$(date +%Y%m%d).txt

NOTE:
    This script requires root privileges for most operations.
    Run with sudo for full functionality.

SCHEDULED MAINTENANCE:
    Add to crontab for automated maintenance:
    0 2 * * 0 /opt/scripts/system_maintenance.sh --all --report /var/log/maintenance.txt

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -u|--update)
                UPDATE_PACKAGES=true
                shift
                ;;
            -c|--cleanup)
                CLEANUP_FILES=true
                shift
                ;;
            -l|--logs)
                ROTATE_LOGS=true
                shift
                ;;
            -d|--disk)
                ANALYZE_DISK=true
                shift
                ;;
            -r|--report)
                REPORT_FILE="$2"
                shift 2
                ;;
            -a|--all)
                ALL_TASKS=true
                UPDATE_PACKAGES=true
                CLEANUP_FILES=true
                ROTATE_LOGS=true
                ANALYZE_DISK=true
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
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# ===========================
# Main Function
# ===========================

main() {
    parse_arguments "$@"

    # Check if no tasks selected
    if [[ "$UPDATE_PACKAGES" == false ]] && \
       [[ "$CLEANUP_FILES" == false ]] && \
       [[ "$ROTATE_LOGS" == false ]] && \
       [[ "$ANALYZE_DISK" == false ]] && \
       [[ -z "$REPORT_FILE" ]]; then
        echo "No maintenance tasks selected"
        usage
        exit 1
    fi

    echo -e "${BOLD}${CYAN}System Maintenance${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}DRY-RUN MODE - No changes will be made${RESET}"
    fi

    echo ""

    log_info "==================================="
    log_info "System maintenance started"
    log_info "==================================="

    # Run selected tasks
    if [[ "$UPDATE_PACKAGES" == true ]]; then
        update_packages
    fi

    if [[ "$CLEANUP_FILES" == true ]]; then
        cleanup_system
    fi

    if [[ "$ROTATE_LOGS" == true ]]; then
        rotate_logs
    fi

    if [[ "$ANALYZE_DISK" == true ]]; then
        analyze_disk_usage
    fi

    # Generate report
    if [[ -n "$REPORT_FILE" ]] || [[ "$ALL_TASKS" == true ]]; then
        generate_maintenance_report
    fi

    echo -e "${CYAN}========================================${RESET}"
    echo -e "${GREEN}${BOLD}Maintenance completed successfully!${RESET}"
    echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"

    log_success "System maintenance completed"
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
