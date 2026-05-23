#!/bin/bash
#
# system_monitor.sh - Comprehensive system monitoring for ML infrastructure
#
# Description:
#   Monitor CPU, memory, disk usage, processes, and services. Generate alerts
#   and reports for system health monitoring.
#
# Usage:
#   ./system_monitor.sh [OPTIONS]
#
# Options:
#   -c, --check           Run system health checks
#   -m, --monitor         Continuous monitoring mode
#   -r, --report FILE     Generate monitoring report
#   -a, --alert           Enable alerting
#   -t, --threshold N     Set alert threshold (default: 80)
#   -i, --interval N      Monitoring interval in seconds (default: 60)
#   -v, --verbose         Verbose output
#   -h, --help            Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/system-monitor.log"
readonly REPORT_DIR="/var/log/monitoring-reports"

# Defaults
CHECK_MODE=false
MONITOR_MODE=false
REPORT_FILE=""
ENABLE_ALERTS=false
CPU_THRESHOLD=80
MEMORY_THRESHOLD=80
DISK_THRESHOLD=80
MONITOR_INTERVAL=60
VERBOSE=false

# ML Infrastructure specific
readonly ML_PROCESSES=(
    "python"
    "jupyter"
    "mlflow"
    "tensorboard"
    "docker"
)

# ===========================
# Colors
# ===========================

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
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

log_warning() {
    log_message "WARNING" "$@"
}

log_error() {
    log_message "ERROR" "$@"
}

log_alert() {
    log_message "ALERT" "$@"
}

# ===========================
# System Metrics
# ===========================

get_cpu_usage() {
    # Get CPU usage percentage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')

    # Alternative method using mpstat if available
    if command -v mpstat &> /dev/null; then
        cpu_usage=$(mpstat 1 1 | awk '/Average/ {print 100 - $NF}')
    fi

    echo "${cpu_usage%.*}"
}

get_memory_usage() {
    # Get memory usage percentage
    local mem_total=$(free | grep Mem | awk '{print $2}')
    local mem_used=$(free | grep Mem | awk '{print $3}')
    local mem_percent=$((mem_used * 100 / mem_total))

    echo "$mem_percent"
}

get_disk_usage() {
    # Get root filesystem usage
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    echo "$disk_usage"
}

get_load_average() {
    # Get 1, 5, 15 minute load averages
    uptime | awk -F'load average:' '{print $2}' | xargs
}

get_process_count() {
    ps aux | wc -l
}

get_running_ml_processes() {
    local count=0
    for process in "${ML_PROCESSES[@]}"; do
        local proc_count=$(pgrep -x "$process" | wc -l)
        count=$((count + proc_count))
    done
    echo "$count"
}

# ===========================
# GPU Monitoring
# ===========================

check_gpu_available() {
    if command -v nvidia-smi &> /dev/null; then
        return 0
    fi
    return 1
}

get_gpu_usage() {
    if ! check_gpu_available; then
        echo "N/A"
        return
    fi

    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1
}

get_gpu_memory() {
    if ! check_gpu_available; then
        echo "N/A"
        return
    fi

    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    local percent=$((used * 100 / total))
    echo "$percent"
}

get_gpu_temperature() {
    if ! check_gpu_available; then
        echo "N/A"
        return
    fi

    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1
}

get_gpu_info() {
    if ! check_gpu_available; then
        echo "No NVIDIA GPU detected"
        return
    fi

    echo -e "${BOLD}GPU Information:${RESET}"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | \
    while IFS=, read -r index name driver memory; do
        echo "  GPU $index: $name"
        echo "    Driver: $driver"
        echo "    Memory: $memory"
    done
}

# ===========================
# Service Monitoring
# ===========================

check_service_status() {
    local service="$1"

    if ! systemctl list-unit-files "$service.service" &> /dev/null; then
        echo "NOT_INSTALLED"
        return
    fi

    if systemctl is-active --quiet "$service"; then
        echo "RUNNING"
    elif systemctl is-failed --quiet "$service"; then
        echo "FAILED"
    else
        echo "STOPPED"
    fi
}

monitor_critical_services() {
    local services=(
        "docker"
        "ssh"
        "cron"
    )

    echo -e "${BOLD}Critical Services:${RESET}"

    local failed_services=()

    for service in "${services[@]}"; do
        local status=$(check_service_status "$service")

        case "$status" in
            RUNNING)
                echo -e "  ${GREEN}✓${RESET} $service: $status"
                ;;
            FAILED)
                echo -e "  ${RED}✗${RESET} $service: $status"
                failed_services+=("$service")
                log_alert "Service $service is FAILED"
                ;;
            STOPPED)
                echo -e "  ${YELLOW}○${RESET} $service: $status"
                log_warning "Service $service is STOPPED"
                ;;
            NOT_INSTALLED)
                echo -e "  ${YELLOW}−${RESET} $service: $status"
                ;;
        esac
    done

    if [[ ${#failed_services[@]} -gt 0 ]]; then
        if [[ "$ENABLE_ALERTS" == true ]]; then
            send_alert "Critical services failed: ${failed_services[*]}"
        fi
    fi
}

# ===========================
# Process Monitoring
# ===========================

monitor_top_processes() {
    echo -e "${BOLD}Top Processes by CPU:${RESET}"
    ps aux --sort=-%cpu | head -6 | tail -5 | \
    awk '{printf "  %-10s %5s%% %5s%% %s\n", $1, $3, $4, $11}'

    echo ""
    echo -e "${BOLD}Top Processes by Memory:${RESET}"
    ps aux --sort=-%mem | head -6 | tail -5 | \
    awk '{printf "  %-10s %5s%% %5s%% %s\n", $1, $3, $4, $11}'
}

monitor_ml_processes() {
    echo -e "${BOLD}ML Processes:${RESET}"

    local found=false
    for process in "${ML_PROCESSES[@]}"; do
        local count=$(pgrep -c "$process" 2>/dev/null || echo 0)
        if [[ $count -gt 0 ]]; then
            echo "  $process: $count running"
            found=true
        fi
    done

    if [[ "$found" == false ]]; then
        echo "  No ML processes detected"
    fi
}

check_zombie_processes() {
    local zombie_count=$(ps aux | awk '$8=="Z"' | wc -l)

    if [[ $zombie_count -gt 0 ]]; then
        echo -e "${YELLOW}Warning: $zombie_count zombie processes detected${RESET}"
        log_warning "Zombie processes detected: $zombie_count"

        if [[ "$ENABLE_ALERTS" == true ]]; then
            send_alert "Zombie processes detected: $zombie_count"
        fi
    fi
}

# ===========================
# Disk Monitoring
# ===========================

monitor_disk_space() {
    echo -e "${BOLD}Disk Usage:${RESET}"

    df -h | grep -E '^Filesystem|^/dev/' | while read -r line; do
        if echo "$line" | grep -q "Filesystem"; then
            echo "$line"
        else
            local usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')

            if [[ $usage -ge $DISK_THRESHOLD ]]; then
                echo -e "${RED}$line${RESET}"
                local mount=$(echo "$line" | awk '{print $6}')
                log_alert "Disk usage critical on $mount: ${usage}%"

                if [[ "$ENABLE_ALERTS" == true ]]; then
                    send_alert "Disk usage critical on $mount: ${usage}%"
                fi
            elif [[ $usage -ge $((DISK_THRESHOLD - 10)) ]]; then
                echo -e "${YELLOW}$line${RESET}"
                local mount=$(echo "$line" | awk '{print $6}')
                log_warning "Disk usage high on $mount: ${usage}%"
            else
                echo "$line"
            fi
        fi
    done
}

find_large_files() {
    echo -e "${BOLD}Largest Files (Top 10):${RESET}"

    # Find in common directories, limit search depth
    sudo find /var/log /tmp /home -type f -size +100M 2>/dev/null | \
    head -10 | \
    xargs -I {} ls -lh {} 2>/dev/null | \
    awk '{printf "  %5s %s\n", $5, $9}' || echo "  No large files found"
}

# ===========================
# Network Monitoring
# ===========================

monitor_network() {
    echo -e "${BOLD}Network Status:${RESET}"

    # Active connections
    local tcp_connections=$(ss -tan | grep ESTAB | wc -l)
    echo "  Established TCP connections: $tcp_connections"

    # Listening ports
    local listening_ports=$(ss -tln | grep LISTEN | wc -l)
    echo "  Listening ports: $listening_ports"

    # Network interfaces
    echo ""
    echo -e "${BOLD}Network Interfaces:${RESET}"
    ip -brief addr | while read -r line; do
        echo "  $line"
    done
}

check_port_availability() {
    local important_ports=(
        "22:SSH"
        "80:HTTP"
        "443:HTTPS"
        "8888:Jupyter"
        "5000:MLflow"
        "6006:TensorBoard"
    )

    echo ""
    echo -e "${BOLD}Important Ports:${RESET}"

    for port_info in "${important_ports[@]}"; do
        IFS=: read -r port name <<< "$port_info"
        if ss -tln | grep -q ":$port "; then
            echo -e "  ${GREEN}✓${RESET} Port $port ($name): Listening"
        else
            echo -e "  ${YELLOW}○${RESET} Port $port ($name): Not listening"
        fi
    done
}

# ===========================
# System Health Check
# ===========================

perform_health_check() {
    echo -e "${BOLD}${CYAN}System Health Check${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Hostname: $(hostname)"
    echo ""

    # System metrics
    echo -e "${BOLD}System Metrics:${RESET}"

    local cpu_usage=$(get_cpu_usage)
    local mem_usage=$(get_memory_usage)
    local disk_usage=$(get_disk_usage)
    local load_avg=$(get_load_average)

    # CPU
    if [[ $cpu_usage -ge $CPU_THRESHOLD ]]; then
        echo -e "  CPU Usage:    ${RED}${cpu_usage}%${RESET} [CRITICAL]"
        log_alert "CPU usage critical: ${cpu_usage}%"
        [[ "$ENABLE_ALERTS" == true ]] && send_alert "CPU usage critical: ${cpu_usage}%"
    elif [[ $cpu_usage -ge $((CPU_THRESHOLD - 10)) ]]; then
        echo -e "  CPU Usage:    ${YELLOW}${cpu_usage}%${RESET} [WARNING]"
        log_warning "CPU usage high: ${cpu_usage}%"
    else
        echo -e "  CPU Usage:    ${GREEN}${cpu_usage}%${RESET} [OK]"
    fi

    # Memory
    if [[ $mem_usage -ge $MEMORY_THRESHOLD ]]; then
        echo -e "  Memory Usage: ${RED}${mem_usage}%${RESET} [CRITICAL]"
        log_alert "Memory usage critical: ${mem_usage}%"
        [[ "$ENABLE_ALERTS" == true ]] && send_alert "Memory usage critical: ${mem_usage}%"
    elif [[ $mem_usage -ge $((MEMORY_THRESHOLD - 10)) ]]; then
        echo -e "  Memory Usage: ${YELLOW}${mem_usage}%${RESET} [WARNING]"
        log_warning "Memory usage high: ${mem_usage}%"
    else
        echo -e "  Memory Usage: ${GREEN}${mem_usage}%${RESET} [OK]"
    fi

    # Disk
    if [[ $disk_usage -ge $DISK_THRESHOLD ]]; then
        echo -e "  Disk Usage:   ${RED}${disk_usage}%${RESET} [CRITICAL]"
        log_alert "Disk usage critical: ${disk_usage}%"
        [[ "$ENABLE_ALERTS" == true ]] && send_alert "Disk usage critical: ${disk_usage}%"
    elif [[ $disk_usage -ge $((DISK_THRESHOLD - 10)) ]]; then
        echo -e "  Disk Usage:   ${YELLOW}${disk_usage}%${RESET} [WARNING]"
        log_warning "Disk usage high: ${disk_usage}%"
    else
        echo -e "  Disk Usage:   ${GREEN}${disk_usage}%${RESET} [OK]"
    fi

    echo "  Load Average: $load_avg"
    echo "  Processes:    $(get_process_count)"
    echo "  Uptime:       $(uptime -p)"
    echo ""

    # GPU metrics
    if check_gpu_available; then
        local gpu_usage=$(get_gpu_usage)
        local gpu_memory=$(get_gpu_memory)
        local gpu_temp=$(get_gpu_temperature)

        echo -e "${BOLD}GPU Metrics:${RESET}"
        echo "  GPU Usage:    ${gpu_usage}%"
        echo "  GPU Memory:   ${gpu_memory}%"
        echo "  Temperature:  ${gpu_temp}°C"
        echo ""
    fi

    # Services
    monitor_critical_services
    echo ""

    # Processes
    monitor_ml_processes
    echo ""

    check_zombie_processes
    echo ""

    # Disk
    monitor_disk_space
    echo ""

    # Network
    monitor_network
    check_port_availability
    echo ""

    log_info "Health check completed"
}

# ===========================
# Continuous Monitoring
# ===========================

continuous_monitor() {
    echo -e "${BOLD}${CYAN}Continuous System Monitoring${RESET}"
    echo "Monitoring interval: ${MONITOR_INTERVAL}s"
    echo "Alert threshold: CPU=${CPU_THRESHOLD}%, MEM=${MEMORY_THRESHOLD}%, DISK=${DISK_THRESHOLD}%"
    echo "Press Ctrl+C to stop"
    echo ""

    trap 'echo -e "\n${YELLOW}Monitoring stopped${RESET}"; exit 0' INT

    while true; do
        clear
        perform_health_check

        sleep "$MONITOR_INTERVAL"
    done
}

# ===========================
# Report Generation
# ===========================

generate_monitoring_report() {
    mkdir -p "$REPORT_DIR"

    local report_file="${REPORT_FILE:-$REPORT_DIR/monitor-$(date +%Y%m%d-%H%M%S).txt}"

    local report=$(cat <<EOF
========================================
System Monitoring Report
========================================
Generated: $(date '+%Y-%m-%d %H:%M:%S')
Hostname: $(hostname)
Kernel: $(uname -r)
Uptime: $(uptime -p)

System Information:
-------------------
Distribution: $(lsb_release -d 2>/dev/null | cut -f2- || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
CPU Cores: $(nproc)
Memory Total: $(free -h | awk 'NR==2 {print $2}')
Memory Used: $(free -h | awk 'NR==2 {print $3}')
Memory Available: $(free -h | awk 'NR==2 {print $7}')

Current Metrics:
----------------
CPU Usage: $(get_cpu_usage)%
Memory Usage: $(get_memory_usage)%
Disk Usage: $(get_disk_usage)%
Load Average: $(get_load_average)
Process Count: $(get_process_count)
ML Processes: $(get_running_ml_processes)

GPU Information:
----------------
$(get_gpu_info)
GPU Usage: $(get_gpu_usage)%
GPU Memory: $(get_gpu_memory)%
GPU Temperature: $(get_gpu_temperature)°C

Disk Usage:
-----------
$(df -h | grep -E '^Filesystem|^/dev/')

Top Processes (CPU):
--------------------
$(ps aux --sort=-%cpu | head -11)

Top Processes (Memory):
-----------------------
$(ps aux --sort=-%mem | head -11)

Network Status:
---------------
$(ss -tan | grep ESTAB | wc -l) established TCP connections
$(ss -tln | grep LISTEN | wc -l) listening ports

Network Interfaces:
-------------------
$(ip -brief addr)

Recent Alerts (last 24h):
--------------------------
$(grep ALERT "$LOG_FILE" 2>/dev/null | tail -20 || echo "No recent alerts")

Recent Warnings (last 24h):
---------------------------
$(grep WARNING "$LOG_FILE" 2>/dev/null | tail -20 || echo "No recent warnings")

========================================
End of Report
========================================
EOF
)

    echo "$report" | tee "$report_file"

    log_info "Monitoring report generated: $report_file"

    return 0
}

# ===========================
# Alerting
# ===========================

send_alert() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Log alert
    log_alert "$message"

    # Send to syslog
    logger -t "system-monitor" -p user.alert "$message"

    # Write to alert file
    echo "[$timestamp] $message" >> "/var/log/system-alerts.log"

    # Integration points for external alerting systems
    # Uncomment and configure as needed:

    # Email alert
    # echo "$message" | mail -s "System Alert: $(hostname)" admin@example.com

    # Slack webhook
    # curl -X POST -H 'Content-type: application/json' \
    #   --data "{\"text\":\"System Alert: $message\"}" \
    #   "$SLACK_WEBHOOK_URL"

    # PagerDuty
    # curl -X POST https://events.pagerduty.com/v2/enqueue \
    #   -H 'Content-Type: application/json' \
    #   -d "{\"routing_key\":\"$PAGERDUTY_KEY\",\"event_action\":\"trigger\",\"payload\":{\"summary\":\"$message\",\"severity\":\"critical\",\"source\":\"$(hostname)\"}}"
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Comprehensive system monitoring for ML infrastructure.

OPTIONS:
    -c, --check             Run system health check
    -m, --monitor           Continuous monitoring mode
    -r, --report [FILE]     Generate monitoring report
    -a, --alert             Enable alerting
    -t, --threshold N       Set alert threshold (default: 80)
    -i, --interval N        Monitoring interval in seconds (default: 60)
    -v, --verbose           Verbose output
    -h, --help              Display this help message

EXAMPLES:
    # Run health check
    $SCRIPT_NAME --check

    # Continuous monitoring
    $SCRIPT_NAME --monitor

    # Monitor with alerts
    $SCRIPT_NAME --monitor --alert

    # Generate report
    $SCRIPT_NAME --report

    # Custom thresholds
    $SCRIPT_NAME --check --threshold 90

    # Monitor with custom interval
    $SCRIPT_NAME --monitor --interval 30

THRESHOLDS:
    Default alert thresholds:
    - CPU: ${CPU_THRESHOLD}%
    - Memory: ${MEMORY_THRESHOLD}%
    - Disk: ${DISK_THRESHOLD}%

LOGS:
    System log: $LOG_FILE
    Alert log: /var/log/system-alerts.log
    Reports: $REPORT_DIR

NOTE:
    Some operations require root privileges.
    Run with sudo for full functionality.

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
            -c|--check)
                CHECK_MODE=true
                shift
                ;;
            -m|--monitor)
                MONITOR_MODE=true
                shift
                ;;
            -r|--report)
                if [[ -n "${2:-}" ]] && [[ ! "$2" =~ ^- ]]; then
                    REPORT_FILE="$2"
                    shift 2
                else
                    REPORT_FILE="$REPORT_DIR/monitor-$(date +%Y%m%d-%H%M%S).txt"
                    shift
                fi
                ;;
            -a|--alert)
                ENABLE_ALERTS=true
                shift
                ;;
            -t|--threshold)
                CPU_THRESHOLD="$2"
                MEMORY_THRESHOLD="$2"
                DISK_THRESHOLD="$2"
                shift 2
                ;;
            -i|--interval)
                MONITOR_INTERVAL="$2"
                shift 2
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

    # Ensure log directory exists
    sudo mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    sudo touch "$LOG_FILE" 2>/dev/null || touch "$LOG_FILE" 2>/dev/null || true

    # Ensure report directory exists
    mkdir -p "$REPORT_DIR" 2>/dev/null || true

    log_info "System monitoring started"

    if [[ "$CHECK_MODE" == true ]]; then
        perform_health_check
    fi

    if [[ "$MONITOR_MODE" == true ]]; then
        continuous_monitor
    fi

    if [[ -n "$REPORT_FILE" ]]; then
        generate_monitoring_report
    fi
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
