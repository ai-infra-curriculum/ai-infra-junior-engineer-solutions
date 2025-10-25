#!/bin/bash
#
# monitor_resources.sh - Real-time ML infrastructure resource monitoring
#
# Description:
#   Comprehensive system resource monitoring for ML workloads including
#   CPU, memory, disk, GPU, and process tracking with alert thresholds.
#
# Usage:
#   ./monitor_resources.sh [OPTIONS]
#
# Options:
#   -i, --interval SECONDS   Update interval (default: 2)
#   -t, --top N             Show top N processes (default: 10)
#   -g, --gpu-only          Monitor GPUs only
#   -a, --alerts            Enable alert notifications
#   -l, --log FILE          Log to file
#   -o, --once              Run once (no continuous monitoring)
#   -h, --help              Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_DIR="/var/log/ml-monitoring"

# Default settings
UPDATE_INTERVAL=2
TOP_PROCESSES=10
GPU_ONLY=false
ENABLE_ALERTS=false
LOG_FILE=""
RUN_ONCE=false

# Alert thresholds
readonly CPU_THRESHOLD=80
readonly MEM_THRESHOLD=85
readonly DISK_THRESHOLD=90
readonly GPU_MEM_THRESHOLD=90

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
# Utility Functions
# ===========================

log_message() {
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [[ -n "$LOG_FILE" ]]; then
        echo "[$timestamp] $message" >> "$LOG_FILE"
    fi
}

send_alert() {
    local level="$1"
    local message="$2"

    if [[ "$ENABLE_ALERTS" == true ]]; then
        # Log to syslog
        logger -t "ml-monitor" -p "user.$level" "$message"

        # Could integrate with other alert systems here
        # e.g., email, Slack, PagerDuty
    fi

    log_message "ALERT [$level] $message"
}

get_color_for_value() {
    local value=$1
    local warning_threshold=$2
    local critical_threshold=$3

    if [[ $value -ge $critical_threshold ]]; then
        echo "$RED"
    elif [[ $value -ge $warning_threshold ]]; then
        echo "$YELLOW"
    else
        echo "$GREEN"
    fi
}

# ===========================
# CPU Monitoring
# ===========================

get_cpu_usage() {
    # Get CPU usage percentage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

    # Handle comma vs dot as decimal separator
    cpu_usage=$(echo "$cpu_usage" | tr ',' '.')

    # Remove decimal part for integer comparison
    local cpu_int=${cpu_usage%.*}

    echo "$cpu_usage" "$cpu_int"
}

monitor_cpu() {
    local cpu_usage cpu_int
    read cpu_usage cpu_int <<< "$(get_cpu_usage)"

    local color=$(get_color_for_value "$cpu_int" 70 "$CPU_THRESHOLD")

    echo -e "${BOLD}CPU Usage:${RESET} ${color}${cpu_usage}%${RESET}"

    # Alert if threshold exceeded
    if [[ $cpu_int -ge $CPU_THRESHOLD ]]; then
        send_alert "warning" "High CPU usage: ${cpu_usage}%"
    fi

    # Per-core usage
    if command -v mpstat &> /dev/null; then
        echo -e "\n${BOLD}Per-Core Usage:${RESET}"
        mpstat -P ALL 1 1 | awk 'NR>3 && $2 ~ /[0-9]+/ {printf "  CPU %2s: %5.1f%%\n", $2, 100-$NF}'
    fi

    # Load averages
    local load_avg=$(cat /proc/loadavg | awk '{print $1, $2, $3}')
    local cpu_count=$(grep -c processor /proc/cpuinfo)
    echo -e "\n${BOLD}Load Average:${RESET} $load_avg (${cpu_count} CPUs)"

    log_message "CPU: ${cpu_usage}%, Load: $load_avg"
}

# ===========================
# Memory Monitoring
# ===========================

get_memory_usage() {
    # Get memory usage percentage
    free | awk 'NR==2 {printf "%.0f %.0f %.0f %.0f", $3/$2*100, $2/1024/1024, $3/1024/1024, $4/1024/1024}'
}

monitor_memory() {
    local mem_pct mem_total mem_used mem_free
    read mem_pct mem_total mem_used mem_free <<< "$(get_memory_usage)"

    local color=$(get_color_for_value "$mem_pct" 75 "$MEM_THRESHOLD")

    echo -e "${BOLD}Memory Usage:${RESET} ${color}${mem_pct}%${RESET}"
    echo -e "  Total: ${mem_total} GB | Used: ${mem_used} GB | Free: ${mem_free} GB"

    # Swap usage
    local swap_info=$(free -h | awk 'NR==3 {print $2, $3, $4}')
    if [[ "$swap_info" != "0B 0B 0B" ]]; then
        echo -e "  ${BOLD}Swap:${RESET} $swap_info"
    fi

    # Alert if threshold exceeded
    if [[ $mem_pct -ge $MEM_THRESHOLD ]]; then
        send_alert "warning" "High memory usage: ${mem_pct}%"
    fi

    log_message "Memory: ${mem_pct}%, Used: ${mem_used}GB"
}

# ===========================
# Disk Monitoring
# ===========================

monitor_disk() {
    echo -e "${BOLD}Disk Usage:${RESET}"

    # Get all mounted filesystems
    df -h | awk 'NR>1 && $1 ~ /^\/dev\// {print $0}' | while read -r line; do
        local usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')
        local color=$(get_color_for_value "$usage" 80 "$DISK_THRESHOLD")

        echo -e "  ${color}$(echo "$line" | awk '{printf "%-20s %5s / %5s (%5s) %s\n", $1, $3, $2, $5, $6}')${RESET}"

        # Alert if threshold exceeded
        local mount_point=$(echo "$line" | awk '{print $6}')
        if [[ $usage -ge $DISK_THRESHOLD ]]; then
            send_alert "warning" "High disk usage on $mount_point: ${usage}%"
        fi
    done

    # I/O statistics if available
    if command -v iostat &> /dev/null; then
        echo -e "\n${BOLD}I/O Statistics:${RESET}"
        iostat -x 1 2 | awk 'NR>6 && NF>0 && $1 !~ /^(Device|$)/ {printf "  %-10s  r/s: %6.1f  w/s: %6.1f  util: %5.1f%%\n", $1, $4, $5, $NF}' | tail -n +2
    fi

    log_message "Disk usage logged"
}

# ===========================
# GPU Monitoring
# ===========================

monitor_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        if [[ "$GPU_ONLY" == false ]]; then
            echo -e "${YELLOW}GPU monitoring not available (nvidia-smi not found)${RESET}"
        fi
        return 1
    fi

    echo -e "${BOLD}GPU Information:${RESET}"
    echo ""

    # Get GPU information
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit \
        --format=csv,noheader,nounits | while IFS=',' read -r idx name temp gpu_util mem_util mem_used mem_total power_draw power_limit; do

        # Trim whitespace
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs)
        temp=$(echo "$temp" | xargs)
        gpu_util=$(echo "$gpu_util" | xargs)
        mem_util=$(echo "$mem_util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        power_draw=$(echo "$power_draw" | xargs)
        power_limit=$(echo "$power_limit" | xargs)

        # Calculate memory percentage
        local mem_pct=0
        if [[ $mem_total -gt 0 ]]; then
            mem_pct=$((mem_used * 100 / mem_total))
        fi

        # Colors based on utilization
        local gpu_color=$(get_color_for_value "$gpu_util" 80 95)
        local mem_color=$(get_color_for_value "$mem_pct" 80 "$GPU_MEM_THRESHOLD")
        local temp_color=$(get_color_for_value "$temp" 75 85)

        echo -e "${BOLD}GPU $idx:${RESET} $name"
        echo -e "  ${gpu_color}GPU Util: ${gpu_util}%${RESET} | ${mem_color}Mem: ${mem_used}MB / ${mem_total}MB (${mem_pct}%)${RESET}"
        echo -e "  ${temp_color}Temp: ${temp}°C${RESET} | Power: ${power_draw}W / ${power_limit}W"

        # Alert on high GPU memory
        if [[ $mem_pct -ge $GPU_MEM_THRESHOLD ]]; then
            send_alert "warning" "High GPU $idx memory usage: ${mem_pct}%"
        fi

        echo ""
    done

    # GPU processes
    echo -e "${BOLD}GPU Processes:${RESET}"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | while IFS=',' read -r pid process mem; do
        echo -e "  PID $(echo $pid | xargs): $(echo $process | xargs) ($(echo $mem | xargs))"
    done || echo "  No GPU processes running"

    log_message "GPU status logged"
}

# ===========================
# Process Monitoring
# ===========================

monitor_processes() {
    echo -e "${BOLD}Top $TOP_PROCESSES Processes (by CPU):${RESET}"

    ps aux --sort=-%cpu | head -n $((TOP_PROCESSES + 1)) | awk '
        NR==1 {printf "  %-8s %5s %5s %8s %8s %s\n", "USER", "PID", "%CPU", "%MEM", "VSZ", "COMMAND"}
        NR>1 {printf "  %-8s %5s %5.1f %5.1f %8s %s\n", $1, $2, $3, $4, $5, $11}
    '

    echo ""
    echo -e "${BOLD}Top $TOP_PROCESSES Processes (by Memory):${RESET}"

    ps aux --sort=-%mem | head -n $((TOP_PROCESSES + 1)) | awk '
        NR==1 {printf "  %-8s %5s %5s %8s %8s %s\n", "USER", "PID", "%CPU", "%MEM", "RSS", "COMMAND"}
        NR>1 {printf "  %-8s %5s %5.1f %5.1f %8s %s\n", $1, $2, $3, $4, $6, $11}
    '

    # ML-specific processes
    echo ""
    echo -e "${BOLD}ML Training Processes:${RESET}"

    local ml_procs=$(ps aux | grep -E "(python.*train|pytorch|tensorflow|jupyter)" | grep -v grep)
    if [[ -n "$ml_procs" ]]; then
        echo "$ml_procs" | awk '{printf "  PID %5s: %5.1f%% CPU, %5.1f%% MEM - %s\n", $2, $3, $4, substr($0, index($0,$11))}'
    else
        echo "  No ML training processes detected"
    fi

    log_message "Process snapshot logged"
}

# ===========================
# Network Monitoring
# ===========================

monitor_network() {
    echo -e "${BOLD}Network Interfaces:${RESET}"

    # Get interface statistics
    if command -v ip &> /dev/null; then
        ip -s link | awk '
            /^[0-9]+:/ {
                iface=$2
                gsub(/:/, "", iface)
            }
            /RX:/ {
                getline
                rx=$1
            }
            /TX:/ {
                getline
                tx=$1
                if (iface != "lo")
                    printf "  %-10s  RX: %10s bytes  TX: %10s bytes\n", iface, rx, tx
            }
        '
    fi

    # Active connections
    local connections=$(ss -tun | grep -c ESTAB 2>/dev/null || echo 0)
    echo -e "\n${BOLD}Active Connections:${RESET} $connections"

    # Listening ports
    echo -e "\n${BOLD}Listening Services:${RESET}"
    ss -tuln | awk 'NR>1 && $1 ~ /LISTEN/ {printf "  %-6s  %-20s\n", $1, $5}' | sort -u | head -n 10
}

# ===========================
# System Information
# ===========================

show_system_info() {
    echo -e "${BOLD}${CYAN}System Information${RESET}"
    echo -e "${CYAN}========================================${RESET}"

    # Hostname and uptime
    echo -e "${BOLD}Hostname:${RESET} $(hostname)"
    echo -e "${BOLD}Uptime:${RESET} $(uptime -p 2>/dev/null || uptime | awk '{print $3, $4}')"

    # Kernel and OS
    echo -e "${BOLD}Kernel:${RESET} $(uname -r)"
    if [[ -f /etc/os-release ]]; then
        echo -e "${BOLD}OS:${RESET} $(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)"
    fi

    # CPU info
    local cpu_model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    local cpu_count=$(grep -c processor /proc/cpuinfo)
    echo -e "${BOLD}CPU:${RESET} $cpu_model (${cpu_count} cores)"

    # Total memory
    local total_mem=$(free -h | awk 'NR==2 {print $2}')
    echo -e "${BOLD}Memory:${RESET} $total_mem"

    echo ""
}

# ===========================
# Main Monitoring Loop
# ===========================

monitor_all() {
    if [[ "$RUN_ONCE" == false ]]; then
        clear
    fi

    show_system_info

    if [[ "$GPU_ONLY" == false ]]; then
        echo -e "${CYAN}========================================${RESET}"
        monitor_cpu
        echo ""

        echo -e "${CYAN}========================================${RESET}"
        monitor_memory
        echo ""

        echo -e "${CYAN}========================================${RESET}"
        monitor_disk
        echo ""
    fi

    echo -e "${CYAN}========================================${RESET}"
    monitor_gpu || true
    echo ""

    if [[ "$GPU_ONLY" == false ]]; then
        echo -e "${CYAN}========================================${RESET}"
        monitor_processes
        echo ""

        echo -e "${CYAN}========================================${RESET}"
        monitor_network
        echo ""
    fi

    echo -e "${CYAN}========================================${RESET}"
    echo -e "${BOLD}Last Update:${RESET} $(date '+%Y-%m-%d %H:%M:%S')"

    if [[ "$RUN_ONCE" == false ]]; then
        echo -e "\n${YELLOW}Press Ctrl+C to exit${RESET}"
    fi
}

continuous_monitor() {
    while true; do
        monitor_all
        sleep "$UPDATE_INTERVAL"
    done
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Real-time ML infrastructure resource monitoring.

OPTIONS:
    -i, --interval SECONDS   Update interval (default: $UPDATE_INTERVAL)
    -t, --top N             Show top N processes (default: $TOP_PROCESSES)
    -g, --gpu-only          Monitor GPUs only
    -a, --alerts            Enable alert notifications
    -l, --log FILE          Log to file
    -o, --once              Run once (no continuous monitoring)
    -h, --help              Display this help message

EXAMPLES:
    # Default monitoring
    $SCRIPT_NAME

    # Update every 5 seconds
    $SCRIPT_NAME --interval 5

    # GPU monitoring only
    $SCRIPT_NAME --gpu-only

    # With alerts and logging
    $SCRIPT_NAME --alerts --log /var/log/ml-monitor.log

    # Single snapshot
    $SCRIPT_NAME --once

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -i|--interval)
                UPDATE_INTERVAL="$2"
                shift 2
                ;;
            -t|--top)
                TOP_PROCESSES="$2"
                shift 2
                ;;
            -g|--gpu-only)
                GPU_ONLY=true
                shift
                ;;
            -a|--alerts)
                ENABLE_ALERTS=true
                shift
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            -o|--once)
                RUN_ONCE=true
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

    # Create log file if specified
    if [[ -n "$LOG_FILE" ]]; then
        mkdir -p "$(dirname "$LOG_FILE")"
        touch "$LOG_FILE"
        log_message "Monitoring started"
    fi

    if [[ "$RUN_ONCE" == true ]]; then
        monitor_all
    else
        # Handle Ctrl+C gracefully
        trap 'echo -e "\n${YELLOW}Monitoring stopped${RESET}"; exit 0' INT

        continuous_monitor
    fi
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
