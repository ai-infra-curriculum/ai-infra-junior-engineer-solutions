#!/bin/bash
#
# monitor_system.sh - System Monitoring and Alerting for ML Infrastructure
#
# Usage: ./monitor_system.sh <command> [interval]
#
# Commands: monitor, check, report
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_DIR="${SCRIPT_DIR}/../logs"
readonly LOG_FILE="${LOG_DIR}/monitoring.log"
readonly METRICS_FILE="${LOG_DIR}/metrics.csv"

# Thresholds
readonly CPU_THRESHOLD=80
readonly MEMORY_THRESHOLD=85
readonly DISK_THRESHOLD=90
readonly GPU_MEMORY_THRESHOLD=90

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
log_warning() { log "WARNING" "$@"; }
log_alert() { log "ALERT" "$@"; }

# Setup
setup() {
    mkdir -p "$LOG_DIR"

    # Initialize metrics file if not exists
    if [ ! -f "$METRICS_FILE" ]; then
        echo "timestamp,cpu_usage,memory_usage,disk_usage,gpu_usage,gpu_memory,training_jobs" > "$METRICS_FILE"
    fi
}

# Usage
usage() {
    cat << EOF
Usage: $SCRIPT_NAME <command> [interval]

System monitoring and alerting for ML infrastructure.

Commands:
  monitor [interval]  - Continuous monitoring (default interval: 5s)
  check               - One-time system check
  report              - Generate monitoring report

Examples:
  $SCRIPT_NAME monitor       # Monitor with 5s interval
  $SCRIPT_NAME monitor 10    # Monitor with 10s interval
  $SCRIPT_NAME check         # Single check
  $SCRIPT_NAME report        # Generate report

Thresholds:
  CPU:        ${CPU_THRESHOLD}%
  Memory:     ${MEMORY_THRESHOLD}%
  Disk:       ${DISK_THRESHOLD}%
  GPU Memory: ${GPU_MEMORY_THRESHOLD}%

EOF
    exit 0
}

# Get CPU usage
get_cpu_usage() {
    # Use top command for CPU usage
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}'
}

# Get memory usage
get_memory_usage() {
    free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}'
}

# Get disk usage
get_disk_usage() {
    df -h / | tail -1 | awk '{print $5}' | sed 's/%//'
}

# Get GPU usage (if available)
get_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1
    else
        echo "N/A"
    fi
}

# Get GPU memory usage (if available)
get_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk -F', ' '{printf "%.1f", $1/$2 * 100}'
    else
        echo "N/A"
    fi
}

# Count training jobs
count_training_jobs() {
    # Count Python processes (likely training jobs)
    pgrep -f "python.*train" | wc -l
}

# Check if metric exceeds threshold
check_threshold() {
    local value="$1"
    local threshold="$2"
    local metric_name="$3"

    # Handle N/A values
    if [ "$value" = "N/A" ]; then
        return 0
    fi

    # Convert to integer for comparison
    local value_int=$(printf "%.0f" "$value")

    if [ "$value_int" -gt "$threshold" ]; then
        log_alert "$metric_name: ${value}% (threshold: ${threshold}%)"
        return 1
    fi

    return 0
}

# Send alert (placeholder for actual alerting)
send_alert() {
    local message="$1"

    log_alert "$message"

    # In production, integrate with:
    # - Email notifications
    # - Slack/Teams webhooks
    # - PagerDuty
    # - Custom monitoring systems

    echo -e "${RED}ALERT:${NC} $message"
}

# Collect metrics
collect_metrics() {
    local cpu=$(get_cpu_usage)
    local memory=$(get_memory_usage)
    local disk=$(get_disk_usage)
    local gpu=$(get_gpu_usage)
    local gpu_memory=$(get_gpu_memory)
    local training_jobs=$(count_training_jobs)

    # Log to CSV
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp,$cpu,$memory,$disk,$gpu,$gpu_memory,$training_jobs" >> "$METRICS_FILE"

    # Return metrics as associative array (simulated with echo)
    echo "$cpu|$memory|$disk|$gpu|$gpu_memory|$training_jobs"
}

# Display metrics dashboard
display_dashboard() {
    local metrics="$1"

    IFS='|' read -r cpu memory disk gpu gpu_memory training_jobs <<< "$metrics"

    clear

    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     ML Infrastructure Monitoring          ║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════╣${NC}"

    # CPU
    printf "${BLUE}║${NC} CPU Usage:    "
    if (( $(echo "$cpu > $CPU_THRESHOLD" | bc -l) )); then
        printf "${RED}%.1f%%${NC} ⚠️  " "$cpu"
    else
        printf "${GREEN}%.1f%%${NC}    " "$cpu"
    fi
    printf "${BLUE}(threshold: %d%%)${NC}\n" "$CPU_THRESHOLD"

    # Memory
    printf "${BLUE}║${NC} Memory Usage: "
    if (( $(echo "$memory > $MEMORY_THRESHOLD" | bc -l) )); then
        printf "${RED}%.1f%%${NC} ⚠️  " "$memory"
    else
        printf "${GREEN}%.1f%%${NC}    " "$memory"
    fi
    printf "${BLUE}(threshold: %d%%)${NC}\n" "$MEMORY_THRESHOLD"

    # Disk
    printf "${BLUE}║${NC} Disk Usage:   "
    if [ "$disk" -gt "$DISK_THRESHOLD" ]; then
        printf "${RED}%d%%${NC} ⚠️  " "$disk"
    else
        printf "${GREEN}%d%%${NC}    " "$disk"
    fi
    printf "${BLUE}(threshold: %d%%)${NC}\n" "$DISK_THRESHOLD"

    # GPU
    if [ "$gpu" != "N/A" ]; then
        printf "${BLUE}║${NC} GPU Usage:    "
        printf "${GREEN}%s%%${NC}\n" "$gpu"

        printf "${BLUE}║${NC} GPU Memory:   "
        if (( $(echo "$gpu_memory > $GPU_MEMORY_THRESHOLD" | bc -l) )); then
            printf "${RED}%.1f%%${NC} ⚠️  " "$gpu_memory"
        else
            printf "${GREEN}%.1f%%${NC}    " "$gpu_memory"
        fi
        printf "${BLUE}(threshold: %d%%)${NC}\n" "$GPU_MEMORY_THRESHOLD"
    else
        echo -e "${BLUE}║${NC} GPU:          ${YELLOW}Not available${NC}"
    fi

    # Training jobs
    echo -e "${BLUE}║${NC} Training Jobs: ${GREEN}${training_jobs}${NC}"

    echo -e "${BLUE}╠════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC} Updated: $(date '+%Y-%m-%d %H:%M:%S')          ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
}

# Check system once
check_system() {
    log_info "=== System Check ==="

    local metrics=$(collect_metrics)
    IFS='|' read -r cpu memory disk gpu gpu_memory training_jobs <<< "$metrics"

    echo ""
    echo -e "${BLUE}System Status:${NC}"
    echo "  CPU Usage:      ${cpu}%"
    echo "  Memory Usage:   ${memory}%"
    echo "  Disk Usage:     ${disk}%"

    if [ "$gpu" != "N/A" ]; then
        echo "  GPU Usage:      ${gpu}%"
        echo "  GPU Memory:     ${gpu_memory}%"
    else
        echo "  GPU:            Not available"
    fi

    echo "  Training Jobs:  $training_jobs"
    echo ""

    # Check thresholds
    local alerts=0

    check_threshold "$cpu" "$CPU_THRESHOLD" "CPU" || ((alerts++))
    check_threshold "$memory" "$MEMORY_THRESHOLD" "Memory" || ((alerts++))
    check_threshold "$disk" "$DISK_THRESHOLD" "Disk" || ((alerts++))
    check_threshold "$gpu_memory" "$GPU_MEMORY_THRESHOLD" "GPU Memory" || ((alerts++))

    if [ "$alerts" -eq 0 ]; then
        echo -e "${GREEN}✓ All metrics within thresholds${NC}"
    else
        echo -e "${RED}✗ $alerts metric(s) exceeded threshold${NC}"
    fi

    echo ""
}

# Monitor continuously
monitor_system() {
    local interval="${1:-5}"

    log_info "Starting continuous monitoring (interval: ${interval}s)"

    while true; do
        local metrics=$(collect_metrics)
        display_dashboard "$metrics"

        # Check thresholds and alert
        IFS='|' read -r cpu memory disk gpu gpu_memory training_jobs <<< "$metrics"

        if ! check_threshold "$cpu" "$CPU_THRESHOLD" "CPU" || \
           ! check_threshold "$memory" "$MEMORY_THRESHOLD" "Memory" || \
           ! check_threshold "$disk" "$DISK_THRESHOLD" "Disk" || \
           ! check_threshold "$gpu_memory" "$GPU_MEMORY_THRESHOLD" "GPU Memory"; then
            # Alert triggered
            :
        fi

        sleep "$interval"
    done
}

# Generate monitoring report
generate_report() {
    log_info "Generating monitoring report"

    if [ ! -f "$METRICS_FILE" ] || [ $(wc -l < "$METRICS_FILE") -lt 2 ]; then
        echo "No metrics data available. Run monitoring first."
        exit 1
    fi

    local report_file="${LOG_DIR}/monitoring_report_$(date '+%Y%m%d_%H%M%S').txt"

    {
        echo "=== System Monitoring Report ==="
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        echo "Statistics (Last 100 records):"
        echo ""

        # CPU stats
        echo "CPU Usage:"
        tail -100 "$METRICS_FILE" | awk -F',' 'NR>1 {sum+=$2; if($2>max||NR==2)max=$2; if($2<min||NR==2)min=$2; count++} END {printf "  Average: %.1f%%\n  Maximum: %.1f%%\n  Minimum: %.1f%%\n", sum/count, max, min}'

        # Memory stats
        echo ""
        echo "Memory Usage:"
        tail -100 "$METRICS_FILE" | awk -F',' 'NR>1 {sum+=$3; if($3>max||NR==2)max=$3; if($3<min||NR==2)min=$3; count++} END {printf "  Average: %.1f%%\n  Maximum: %.1f%%\n  Minimum: %.1f%%\n", sum/count, max, min}'

        # Disk stats
        echo ""
        echo "Disk Usage:"
        tail -100 "$METRICS_FILE" | awk -F',' 'NR>1 {sum+=$4; if($4>max||NR==2)max=$4; if($4<min||NR==2)min=$4; count++} END {printf "  Average: %.1f%%\n  Maximum: %.1f%%\n  Minimum: %.1f%%\n", sum/count, max, min}'

        echo ""
        echo "Recent Metrics (Last 10):"
        echo ""
        tail -11 "$METRICS_FILE" | column -t -s','

    } > "$report_file"

    cat "$report_file"

    echo ""
    echo "Report saved to: $report_file"
}

# Main function
main() {
    setup

    if [ $# -eq 0 ]; then
        usage
    fi

    local command="$1"

    case "$command" in
        monitor)
            local interval="${2:-5}"
            monitor_system "$interval"
            ;;
        check)
            check_system
            ;;
        report)
            generate_report
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            ;;
    esac
}

main "$@"
