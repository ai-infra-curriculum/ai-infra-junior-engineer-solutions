#!/bin/bash
###############################################################################
# ML Artifacts Cleanup Script - Automated Cleanup of ML Resources
###############################################################################
#
# Purpose: Clean up old experiments, checkpoints, Docker images, and caches
#
# Usage: ./cleanup_ml_artifacts.sh [OPTIONS]
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-/opt/ml/experiments}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/opt/ml/checkpoints}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-/opt/ml/tensorboard}"
LOGS_DIR="${LOGS_DIR:-/opt/ml/logs}"

EXPERIMENTS_RETENTION="${EXPERIMENTS_RETENTION:-30}"  # days
CHECKPOINTS_RETENTION="${CHECKPOINTS_RETENTION:-7}"  # days
LOGS_RETENTION="${LOGS_RETENTION:-14}"  # days

LOG_FILE="${LOG_FILE:-/var/log/ml-cleanup.log}"
DRY_RUN="${DRY_RUN:-false}"
AGGRESSIVE="${AGGRESSIVE:-false}"
QUIET="${QUIET:-false}"

# Statistics
TOTAL_FREED=0

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [ "$QUIET" != "true" ]; then
        case "$level" in
            INFO)
                echo -e "${BLUE}[INFO]${NC} $message"
                ;;
            SUCCESS)
                echo -e "${GREEN}[SUCCESS]${NC} $message"
                ;;
            WARNING)
                echo -e "${YELLOW}[WARNING]${NC} $message"
                ;;
            ERROR)
                echo -e "${RED}[ERROR]${NC} $message"
                ;;
        esac
    fi

    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Help function
show_help() {
    cat << EOF
ML Artifacts Cleanup Script

Usage: $0 [OPTIONS]

Options:
    --experiments-dir PATH     Experiments directory (default: /opt/ml/experiments)
    --checkpoints-dir PATH     Checkpoints directory (default: /opt/ml/checkpoints)
    --experiments-retention N  Keep experiments for N days (default: 30)
    --checkpoints-retention N  Keep checkpoints for N days (default: 7)
    --logs-retention N         Keep logs for N days (default: 14)
    --log-file PATH            Log file location (default: /var/log/ml-cleanup.log)
    --aggressive               More aggressive cleanup (Docker, pip, apt)
    --dry-run                  Show what would be deleted without deleting
    --quiet                    Suppress console output
    -h, --help                 Show this help message

Environment Variables:
    EXPERIMENTS_DIR, CHECKPOINTS_DIR, EXPERIMENTS_RETENTION, CHECKPOINTS_RETENTION

Examples:
    # Basic cleanup
    $0

    # Dry run to preview
    $0 --dry-run

    # Aggressive cleanup with custom retention
    $0 --aggressive --checkpoints-retention 3

    # Keep experiments longer
    $0 --experiments-retention 60

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiments-dir)
            EXPERIMENTS_DIR="$2"
            shift 2
            ;;
        --checkpoints-dir)
            CHECKPOINTS_DIR="$2"
            shift 2
            ;;
        --experiments-retention)
            EXPERIMENTS_RETENTION="$2"
            shift 2
            ;;
        --checkpoints-retention)
            CHECKPOINTS_RETENTION="$2"
            shift 2
            ;;
        --logs-retention)
            LOGS_RETENTION="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --aggressive)
            AGGRESSIVE="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --quiet)
            QUIET="true"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log ERROR "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to get directory size
get_size() {
    local path="$1"
    if [ -e "$path" ]; then
        du -sh "$path" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

# Function to calculate freed space
calculate_freed() {
    local before="$1"
    local after="$2"

    # Convert to bytes for calculation
    local before_bytes=$(du -sb "$before" 2>/dev/null | cut -f1 || echo 0)
    local after_bytes=$(du -sb "$after" 2>/dev/null | cut -f1 || echo 0)

    local freed=$((before_bytes - after_bytes))
    TOTAL_FREED=$((TOTAL_FREED + freed))

    # Convert to human readable
    if [ $freed -gt 0 ]; then
        echo $(numfmt --to=iec-i --suffix=B $freed)
    else
        echo "0B"
    fi
}

# Start
log INFO "╔═══════════════════════════════════════════════════════════╗"
log INFO "║  ML Artifacts Cleanup System                              ║"
log INFO "╚═══════════════════════════════════════════════════════════╝"
log INFO ""

if [ "$DRY_RUN" = "true" ]; then
    log WARNING "DRY RUN MODE - No changes will be made"
    log INFO ""
fi

log INFO "Configuration:"
log INFO "  Experiments retention: $EXPERIMENTS_RETENTION days"
log INFO "  Checkpoints retention: $CHECKPOINTS_RETENTION days"
log INFO "  Logs retention: $LOGS_RETENTION days"
log INFO "  Aggressive mode: $AGGRESSIVE"
log INFO ""

# Show disk usage before
log INFO "Disk usage before cleanup:"
df -h / | tail -1
log INFO ""

# 1. Clean old experiments
log INFO "1. Cleaning Old Experiments (older than $EXPERIMENTS_RETENTION days)..."

if [ -d "$EXPERIMENTS_DIR" ]; then
    before_size=$(get_size "$EXPERIMENTS_DIR")
    old_experiments=$(find "$EXPERIMENTS_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +$EXPERIMENTS_RETENTION 2>/dev/null || true)

    if [ -n "$old_experiments" ]; then
        count=$(echo "$old_experiments" | wc -l)
        log INFO "  Found $count old experiment directory(ies)"

        if [ "$DRY_RUN" != "true" ]; then
            while IFS= read -r exp_dir; do
                exp_name=$(basename "$exp_dir")
                exp_size=$(get_size "$exp_dir")
                log INFO "    Removing: $exp_name ($exp_size)"
                rm -rf "$exp_dir"
            done <<< "$old_experiments"

            after_size=$(get_size "$EXPERIMENTS_DIR")
            freed=$(calculate_freed "$EXPERIMENTS_DIR" "$EXPERIMENTS_DIR")
            log SUCCESS "  Cleaned up old experiments (freed: $freed)"
        else
            while IFS= read -r exp_dir; do
                exp_name=$(basename "$exp_dir")
                exp_size=$(get_size "$exp_dir")
                log INFO "    Would remove: $exp_name ($exp_size)"
            done <<< "$old_experiments"
        fi
    else
        log SUCCESS "  No old experiments to remove"
    fi
else
    log WARNING "  Experiments directory not found: $EXPERIMENTS_DIR"
fi
log INFO ""

# 2. Clean old checkpoints
log INFO "2. Cleaning Old Checkpoints (older than $CHECKPOINTS_RETENTION days)..."

if [ -d "$CHECKPOINTS_DIR" ]; then
    old_checkpoints=$(find "$CHECKPOINTS_DIR" -type f \( -name "*.ckpt" -o -name "*.pth" -o -name "*.h5" -o -name "*.pb" \) -mtime +$CHECKPOINTS_RETENTION 2>/dev/null || true)

    if [ -n "$old_checkpoints" ]; then
        count=$(echo "$old_checkpoints" | wc -l)
        log INFO "  Found $count old checkpoint file(s)"

        if [ "$DRY_RUN" != "true" ]; then
            total_size=0
            while IFS= read -r ckpt_file; do
                file_name=$(basename "$ckpt_file")
                file_size=$(du -h "$ckpt_file" | cut -f1)
                log INFO "    Removing: $file_name ($file_size)"
                rm -f "$ckpt_file"
            done <<< "$old_checkpoints"

            log SUCCESS "  Cleaned up old checkpoints"
        else
            while IFS= read -r ckpt_file; do
                file_name=$(basename "$ckpt_file")
                file_size=$(du -h "$ckpt_file" | cut -f1)
                log INFO "    Would remove: $file_name ($file_size)"
            done <<< "$old_checkpoints"
        fi
    else
        log SUCCESS "  No old checkpoints to remove"
    fi
else
    log WARNING "  Checkpoints directory not found: $CHECKPOINTS_DIR"
fi
log INFO ""

# 3. Clean old TensorBoard logs
log INFO "3. Cleaning Old TensorBoard Logs (older than $LOGS_RETENTION days)..."

if [ -d "$TENSORBOARD_DIR" ]; then
    old_tb_logs=$(find "$TENSORBOARD_DIR" -type f -name "events.out.tfevents.*" -mtime +$LOGS_RETENTION 2>/dev/null || true)

    if [ -n "$old_tb_logs" ]; then
        count=$(echo "$old_tb_logs" | wc -l)
        log INFO "  Found $count old TensorBoard log file(s)"

        if [ "$DRY_RUN" != "true" ]; then
            while IFS= read -r log_file; do
                rm -f "$log_file"
            done <<< "$old_tb_logs"

            log SUCCESS "  Cleaned up old TensorBoard logs"
        else
            log INFO "  Would remove $count TensorBoard log files"
        fi
    else
        log SUCCESS "  No old TensorBoard logs to remove"
    fi
else
    log INFO "  TensorBoard directory not found (skipping)"
fi
log INFO ""

# 4. Clean old application logs
log INFO "4. Cleaning Old Application Logs (older than $LOGS_RETENTION days)..."

if [ -d "$LOGS_DIR" ]; then
    old_logs=$(find "$LOGS_DIR" -type f \( -name "*.log" -o -name "*.log.*" \) -mtime +$LOGS_RETENTION 2>/dev/null || true)

    if [ -n "$old_logs" ]; then
        count=$(echo "$old_logs" | wc -l)
        log INFO "  Found $count old log file(s)"

        if [ "$DRY_RUN" != "true" ]; then
            while IFS= read -r log_file; do
                rm -f "$log_file"
            done <<< "$old_logs"

            log SUCCESS "  Cleaned up old logs"
        else
            log INFO "  Would remove $count log files"
        fi
    else
        log SUCCESS "  No old logs to remove"
    fi
else
    log INFO "  Logs directory not found (skipping)"
fi
log INFO ""

# Aggressive cleanup
if [ "$AGGRESSIVE" = "true" ]; then
    log WARNING "Running aggressive cleanup..."
    log INFO ""

    # 5. Clean Docker images
    if command -v docker &> /dev/null && docker ps &>/dev/null 2>&1; then
        log INFO "5. Cleaning Docker Resources..."

        if [ "$DRY_RUN" != "true" ]; then
            log INFO "  Removing unused Docker images..."
            docker image prune -a --filter "until=168h" -f 2>&1 | tee -a "$LOG_FILE" | grep "^Total reclaimed space:" | while read line; do
                log SUCCESS "  Docker: $line"
            done

            log INFO "  Removing unused Docker volumes..."
            docker volume prune -f 2>&1 | tee -a "$LOG_FILE" | grep "^Total reclaimed space:" | while read line; do
                log SUCCESS "  Docker volumes: $line"
            done
        else
            log INFO "  Would run: docker image prune -a --filter until=168h"
            log INFO "  Would run: docker volume prune"
        fi
    else
        log INFO "5. Docker not available or not accessible (skipping)"
    fi
    log INFO ""

    # 6. Clean pip cache
    log INFO "6. Cleaning pip Cache..."
    if command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
        pip_cmd=$(command -v pip3 2>/dev/null || command -v pip)

        if [ "$DRY_RUN" != "true" ]; then
            cache_size=$($pip_cmd cache info 2>/dev/null | grep "Cache size:" | awk '{print $3, $4}' || echo "unknown")
            log INFO "  Current cache size: $cache_size"

            $pip_cmd cache purge 2>&1 | tee -a "$LOG_FILE" > /dev/null
            log SUCCESS "  pip cache purged"
        else
            log INFO "  Would run: pip cache purge"
        fi
    else
        log INFO "  pip not found (skipping)"
    fi
    log INFO ""

    # 7. Clean apt cache (Ubuntu/Debian)
    if command -v apt-get &> /dev/null; then
        log INFO "7. Cleaning APT Cache..."

        if [ "$DRY_RUN" != "true" ]; then
            cache_size=$(du -sh /var/cache/apt/archives 2>/dev/null | cut -f1 || echo "unknown")
            log INFO "  Current cache size: $cache_size"

            sudo apt-get clean 2>&1 | tee -a "$LOG_FILE" > /dev/null
            sudo apt-get autoclean 2>&1 | tee -a "$LOG_FILE" > /dev/null
            log SUCCESS "  APT cache cleaned"
        else
            log INFO "  Would run: sudo apt-get clean && sudo apt-get autoclean"
        fi
    fi
    log INFO ""

    # 8. Clean /tmp
    log INFO "8. Cleaning Temporary Files..."
    temp_size=$(du -sh /tmp 2>/dev/null | cut -f1 || echo "unknown")
    log INFO "  /tmp size: $temp_size"

    if [ "$DRY_RUN" != "true" ]; then
        old_temp=$(find /tmp -type f -atime +7 2>/dev/null | wc -l)
        if [ "$old_temp" -gt 0 ]; then
            log INFO "  Removing $old_temp old temp file(s)..."
            sudo find /tmp -type f -atime +7 -delete 2>/dev/null || true
            log SUCCESS "  Old temp files removed"
        else
            log INFO "  No old temp files to remove"
        fi
    else
        old_temp=$(find /tmp -type f -atime +7 2>/dev/null | wc -l)
        log INFO "  Would remove $old_temp old temp files"
    fi
    log INFO ""
fi

# Summary
log INFO "╔═══════════════════════════════════════════════════════════╗"
log INFO "║  Cleanup Summary                                          ║"
log INFO "╚═══════════════════════════════════════════════════════════╝"
log INFO ""

# Show disk usage after
log INFO "Disk usage after cleanup:"
df -h / | tail -1
log INFO ""

if [ "$DRY_RUN" != "true" ]; then
    log SUCCESS "Cleanup completed successfully!"
else
    log INFO "Dry run completed - no changes made"
fi

log INFO ""
log INFO "Recommendations:"
log INFO "  • Schedule this script to run weekly (cron or systemd timer)"
log INFO "  • Adjust retention periods based on your needs"
log INFO "  • Monitor disk usage: df -h /"
log INFO "  • Review logs: tail -f $LOG_FILE"
log INFO ""

exit 0
