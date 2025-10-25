#!/bin/bash
#
# health_check.sh - Monitor ML infrastructure health
#
# Description:
#   Comprehensive health check for ML services including API servers,
#   databases, cache, GPU availability, and system resources.
#
# Usage:
#   ./health_check.sh [OPTIONS]
#
# Options:
#   -s, --services SERVICES  Comma-separated list of services to check
#   -i, --interval SECONDS   Check interval for continuous monitoring
#   -t, --timeout SECONDS    Timeout for each check (default: 5)
#   -r, --restart            Auto-restart failed services
#   -a, --alert              Send alerts on failures
#   -v, --verbose            Enable verbose output
#   -h, --help               Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/ml-health-check.log"

# Default configuration
SERVICES="all"
CHECK_INTERVAL=""
TIMEOUT=5
AUTO_RESTART=false
SEND_ALERTS=false
VERBOSE=false

# Service definitions
declare -A SERVICE_PORTS=(
    ["api"]="8000"
    ["database"]="5432"
    ["redis"]="6379"
    ["grafana"]="3000"
)

declare -A SERVICE_COMMANDS=(
    ["api"]="systemctl"
    ["database"]="pg_isready"
    ["redis"]="redis-cli"
)

# Health thresholds
readonly CPU_THRESHOLD=80
readonly MEMORY_THRESHOLD=85
readonly DISK_THRESHOLD=90

# ===========================
# Colors
# ===========================

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly RESET='\033[0m'

# ===========================
# Logging
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

print_status() {
    local service="$1"
    local status="$2"
    local message="${3:-}"

    case "$status" in
        ok)
            echo -e "${GREEN}✓${RESET} $service: ${GREEN}OK${RESET} $message"
            ;;
        warning)
            echo -e "${YELLOW}!${RESET} $service: ${YELLOW}WARNING${RESET} $message"
            ;;
        error)
            echo -e "${RED}✗${RESET} $service: ${RED}ERROR${RESET} $message"
            ;;
    esac
}

# ===========================
# Service Checks
# ===========================

check_port() {
    local host="${1:-localhost}"
    local port="$2"
    local timeout="${3:-$TIMEOUT}"

    log_verbose "Checking port $port on $host"

    if timeout "$timeout" bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

check_api_service() {
    log_verbose "Checking API service..."

    local api_url="http://localhost:${SERVICE_PORTS[api]}/health"

    if curl -sf --max-time "$TIMEOUT" "$api_url" > /dev/null 2>&1; then
        print_status "API Service" "ok"
        return 0
    else
        print_status "API Service" "error" "Health endpoint unreachable"
        return 1
    fi
}

check_database() {
    log_verbose "Checking database..."

    if command -v pg_isready &> /dev/null; then
        if pg_isready -h localhost -p "${SERVICE_PORTS[database]}" -t "$TIMEOUT" > /dev/null 2>&1; then
            print_status "PostgreSQL" "ok"
            return 0
        else
            print_status "PostgreSQL" "error" "Database not accepting connections"
            return 1
        fi
    elif check_port "localhost" "${SERVICE_PORTS[database]}" "$TIMEOUT"; then
        print_status "PostgreSQL" "ok" "(port check only)"
        return 0
    else
        print_status "PostgreSQL" "error" "Port not responding"
        return 1
    fi
}

check_redis() {
    log_verbose "Checking Redis..."

    if command -v redis-cli &> /dev/null; then
        if redis-cli -p "${SERVICE_PORTS[redis]}" ping 2>/dev/null | grep -q PONG; then
            print_status "Redis" "ok"
            return 0
        else
            print_status "Redis" "error" "Redis not responding to PING"
            return 1
        fi
    elif check_port "localhost" "${SERVICE_PORTS[redis]}" "$TIMEOUT"; then
        print_status "Redis" "ok" "(port check only)"
        return 0
    else
        print_status "Redis" "error" "Port not responding"
        return 1
    fi
}

check_gpu() {
    log_verbose "Checking GPU..."

    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi > /dev/null 2>&1; then
            local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
            local gpu_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/NR}')

            print_status "GPU" "ok" "($gpu_count GPUs, ${gpu_utilization}% avg utilization)"
            return 0
        else
            print_status "GPU" "error" "nvidia-smi command failed"
            return 1
        fi
    else
        print_status "GPU" "warning" "nvidia-smi not found (no GPU or drivers not installed)"
        return 2
    fi
}

# ===========================
# System Resource Checks
# ===========================

check_cpu() {
    log_verbose "Checking CPU usage..."

    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    cpu_usage=${cpu_usage%.*}  # Remove decimal

    if [[ $cpu_usage -ge $CPU_THRESHOLD ]]; then
        print_status "CPU" "warning" "${cpu_usage}% usage (threshold: ${CPU_THRESHOLD}%)"
        return 1
    else
        print_status "CPU" "ok" "${cpu_usage}% usage"
        return 0
    fi
}

check_memory() {
    log_verbose "Checking memory usage..."

    local memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')

    if [[ $memory_usage -ge $MEMORY_THRESHOLD ]]; then
        print_status "Memory" "warning" "${memory_usage}% usage (threshold: ${MEMORY_THRESHOLD}%)"
        return 1
    else
        print_status "Memory" "ok" "${memory_usage}% usage"
        return 0
    fi
}

check_disk() {
    log_verbose "Checking disk usage..."

    local disk_usage=$(df -h / | awk 'NR==2 {print $(NF-1)}' | sed 's/%//')

    if [[ $disk_usage -ge $DISK_THRESHOLD ]]; then
        print_status "Disk" "warning" "${disk_usage}% usage (threshold: ${DISK_THRESHOLD}%)"
        return 1
    else
        print_status "Disk" "ok" "${disk_usage}% usage"
        return 0
    fi
}

# ===========================
# Service Management
# ===========================

restart_service() {
    local service="$1"

    log "INFO" "Attempting to restart service: $service"

    case "$service" in
        api)
            if systemctl is-active --quiet ml-api; then
                sudo systemctl restart ml-api
                log "INFO" "Restarted ml-api service"
            else
                log "WARNING" "ml-api service not managed by systemd"
            fi
            ;;
        database)
            if systemctl is-active --quiet postgresql; then
                sudo systemctl restart postgresql
                log "INFO" "Restarted postgresql service"
            fi
            ;;
        redis)
            if systemctl is-active --quiet redis; then
                sudo systemctl restart redis
                log "INFO" "Restarted redis service"
            fi
            ;;
        *)
            log "WARNING" "Unknown service for restart: $service"
            ;;
    esac
}

# ===========================
# Alert Functions
# ===========================

send_alert() {
    local subject="$1"
    local message="$2"

    log "ALERT" "$subject"

    # Log to syslog
    logger -t "ml-health-check" "$subject: $message"

    # Send email if configured
    if command -v mail &> /dev/null && [[ -n "${ALERT_EMAIL:-}" ]]; then
        echo "$message" | mail -s "[ML Health Check] $subject" "$ALERT_EMAIL"
    fi

    # Send to Slack if configured
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local payload=$(cat <<EOF
{
  "text": "*ML Health Check Alert*",
  "attachments": [{
    "color": "danger",
    "title": "$subject",
    "text": "$message",
    "footer": "ML Health Check",
    "ts": $(date +%s)
  }]
}
EOF
)
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "$payload" \
            2>/dev/null || true
    fi
}

# ===========================
# Main Health Check
# ===========================

run_health_check() {
    echo ""
    echo "=========================================="
    echo "ML Infrastructure Health Check"
    echo "=========================================="
    echo "Timestamp: $(date)"
    echo ""

    local failed_checks=()
    local warning_checks=()

    # Service checks
    echo "Services:"
    echo "----------"

    if [[ "$SERVICES" == "all" ]] || [[ "$SERVICES" == *"api"* ]]; then
        check_api_service || failed_checks+=("api")
    fi

    if [[ "$SERVICES" == "all" ]] || [[ "$SERVICES" == *"database"* ]]; then
        check_database || failed_checks+=("database")
    fi

    if [[ "$SERVICES" == "all" ]] || [[ "$SERVICES" == *"redis"* ]]; then
        check_redis || failed_checks+=("redis")
    fi

    if [[ "$SERVICES" == "all" ]] || [[ "$SERVICES" == *"gpu"* ]]; then
        check_gpu || true  # Don't fail if no GPU
    fi

    # System resource checks
    echo ""
    echo "System Resources:"
    echo "----------"

    check_cpu || warning_checks+=("cpu")
    check_memory || warning_checks+=("memory")
    check_disk || warning_checks+=("disk")

    # Summary
    echo ""
    echo "=========================================="

    if [[ ${#failed_checks[@]} -eq 0 ]] && [[ ${#warning_checks[@]} -eq 0 ]]; then
        echo -e "${GREEN}✓ All checks passed${RESET}"
        log "SUCCESS" "All health checks passed"
    else
        if [[ ${#failed_checks[@]} -gt 0 ]]; then
            echo -e "${RED}✗ Failed checks: ${failed_checks[*]}${RESET}"
            log "ERROR" "Failed checks: ${failed_checks[*]}"

            # Auto-restart if enabled
            if [[ "$AUTO_RESTART" == true ]]; then
                for service in "${failed_checks[@]}"; do
                    restart_service "$service"
                done
            fi

            # Send alerts if enabled
            if [[ "$SEND_ALERTS" == true ]]; then
                send_alert \
                    "Health Check Failures Detected" \
                    "The following services failed health checks: ${failed_checks[*]}"
            fi
        fi

        if [[ ${#warning_checks[@]} -gt 0 ]]; then
            echo -e "${YELLOW}! Warnings: ${warning_checks[*]}${RESET}"
            log "WARNING" "Warnings: ${warning_checks[*]}"
        fi
    fi

    echo "=========================================="
    echo ""

    # Return non-zero if any checks failed
    [[ ${#failed_checks[@]} -eq 0 ]]
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Monitor ML infrastructure health.

OPTIONS:
    -s, --services SERVICES  Comma-separated list of services
                            (api, database, redis, gpu, all)
                            Default: all
    -i, --interval SECONDS   Check interval for continuous monitoring
    -t, --timeout SECONDS    Timeout for each check (default: 5)
    -r, --restart           Auto-restart failed services
    -a, --alert             Send alerts on failures
    -v, --verbose           Enable verbose output
    -h, --help              Display this help message

EXAMPLES:
    # Run single health check
    $SCRIPT_NAME

    # Continuous monitoring every 60 seconds
    $SCRIPT_NAME --interval 60

    # Check specific services with auto-restart
    $SCRIPT_NAME --services api,database --restart

    # Enable alerts
    $SCRIPT_NAME --alert --interval 300

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -s|--services)
                SERVICES="$2"
                shift 2
                ;;
            -i|--interval)
                CHECK_INTERVAL="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -r|--restart)
                AUTO_RESTART=true
                shift
                ;;
            -a|--alert)
                SEND_ALERTS=true
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

    if [[ -n "$CHECK_INTERVAL" ]]; then
        log "INFO" "Starting continuous health monitoring (interval: ${CHECK_INTERVAL}s)"
        while true; do
            run_health_check
            sleep "$CHECK_INTERVAL"
        done
    else
        run_health_check
    fi

    exit 0
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
