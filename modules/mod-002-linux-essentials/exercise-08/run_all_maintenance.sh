#!/bin/bash
###############################################################################
# Master Automation Script - Run All ML Infrastructure Maintenance Tasks
###############################################################################
#
# Purpose: Orchestrate all maintenance tasks in correct order
#
# Usage: ./run_all_maintenance.sh [OPTIONS]
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
RUN_HEALTH_CHECK="${RUN_HEALTH_CHECK:-true}"
RUN_GPU_MONITOR="${RUN_GPU_MONITOR:-true}"
RUN_BACKUP="${RUN_BACKUP:-true}"
RUN_CLEANUP="${RUN_CLEANUP:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
LOG_FILE="${LOG_FILE:-/var/log/ml-maintenance.log}"

# Help function
show_help() {
    cat << EOF
Master ML Infrastructure Maintenance Script

Usage: $0 [OPTIONS]

Options:
    --skip-health          Skip health check
    --skip-gpu             Skip GPU monitoring
    --skip-backup          Skip backups
    --skip-cleanup         Skip cleanup
    --continue-on-error    Continue even if a task fails
    --log-file PATH        Log file location
    -h, --help             Show this help message

Examples:
    # Run all tasks
    $0

    # Skip backup, run everything else
    $0 --skip-backup

    # Continue even if something fails
    $0 --continue-on-error

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-health)
            RUN_HEALTH_CHECK="false"
            shift
            ;;
        --skip-gpu)
            RUN_GPU_MONITOR="false"
            shift
            ;;
        --skip-backup)
            RUN_BACKUP="false"
            shift
            ;;
        --skip-cleanup)
            RUN_CLEANUP="false"
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR="true"
            shift
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging
log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" >> "$LOG_FILE"
    echo "$message"
}

# Header
echo -e "${BOLD}${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   ML Infrastructure Automated Maintenance Suite           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
log "Maintenance suite started"

START_TIME=$(date +%s)
TASKS_RUN=0
TASKS_SUCCEEDED=0
TASKS_FAILED=0

# Function to run a task
run_task() {
    local task_name="$1"
    local script_path="$2"
    shift 2
    local args=("$@")

    echo -e "${BOLD}${BLUE}━━━ $task_name ━━━${NC}"
    echo ""

    TASKS_RUN=$((TASKS_RUN + 1))

    if [ ! -f "$script_path" ]; then
        echo -e "${RED}✗ Script not found: $script_path${NC}"
        log "ERROR: $task_name - Script not found"
        TASKS_FAILED=$((TASKS_FAILED + 1))

        if [ "$CONTINUE_ON_ERROR" != "true" ]; then
            exit 1
        fi
        return 1
    fi

    if bash "$script_path" "${args[@]}"; then
        echo -e "${GREEN}✓ $task_name completed successfully${NC}"
        log "SUCCESS: $task_name completed"
        TASKS_SUCCEEDED=$((TASKS_SUCCEEDED + 1))
        echo ""
        return 0
    else
        echo -e "${RED}✗ $task_name failed${NC}"
        log "ERROR: $task_name failed"
        TASKS_FAILED=$((TASKS_FAILED + 1))
        echo ""

        if [ "$CONTINUE_ON_ERROR" != "true" ]; then
            log "Maintenance suite aborted due to error"
            exit 1
        fi
        return 1
    fi
}

# 1. Health Check (run first to baseline system state)
if [ "$RUN_HEALTH_CHECK" = "true" ]; then
    run_task "System Health Check" \
        "${SCRIPT_DIR}/health-check/health_check.sh"
fi

# 2. GPU Monitoring (check GPU health)
if [ "$RUN_GPU_MONITOR" = "true" ]; then
    if command -v nvidia-smi &> /dev/null; then
        run_task "GPU Health Monitoring" \
            "${SCRIPT_DIR}/monitoring/monitor_gpus.sh"
    else
        echo -e "${YELLOW}! Skipping GPU monitoring (nvidia-smi not found)${NC}"
        echo ""
    fi
fi

# 3. Backups (before cleanup, in case we need to restore)
if [ "$RUN_BACKUP" = "true" ]; then
    run_task "Model Backups" \
        "${SCRIPT_DIR}/backups/backup_models.sh"
fi

# 4. Cleanup (after backups are safe)
if [ "$RUN_CLEANUP" = "true" ]; then
    run_task "Artifact Cleanup" \
        "${SCRIPT_DIR}/cleanup/cleanup_ml_artifacts.sh"
fi

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "${BOLD}${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                      Summary                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "Tasks run:      $TASKS_RUN"
echo -e "${GREEN}Tasks succeeded: $TASKS_SUCCEEDED${NC}"
if [ "$TASKS_FAILED" -gt 0 ]; then
    echo -e "${RED}Tasks failed:    $TASKS_FAILED${NC}"
fi
echo "Duration:       ${DURATION}s"
echo ""

if [ "$TASKS_FAILED" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ All maintenance tasks completed successfully!${NC}"
    log "Maintenance suite completed successfully"
    exit 0
else
    echo -e "${YELLOW}! Some tasks failed. Review logs for details.${NC}"
    log "Maintenance suite completed with errors"
    exit 1
fi
