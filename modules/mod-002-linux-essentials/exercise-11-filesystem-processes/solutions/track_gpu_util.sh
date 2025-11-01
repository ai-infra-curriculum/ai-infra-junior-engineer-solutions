#!/bin/bash
#
# track_gpu_util.sh - GPU utilization tracking and logging
#
# Description:
#   Monitor and track GPU utilization for ML workloads including
#   utilization metrics, memory usage, temperature, and process tracking.
#   Supports logging, alerts, and historical analysis.
#
# Usage:
#   ./track_gpu_util.sh [OPTIONS]
#
# Options:
#   -i, --interval SECONDS   Update interval (default: 5)
#   -d, --duration SECONDS   Total monitoring duration (0 = infinite)
#   -l, --log FILE          Log metrics to file
#   -a, --alert             Enable alert notifications
#   -g, --gpu ID            Monitor specific GPU (default: all)
#   -t, --threshold PCT     Utilization threshold for alerts (default: 90)
#   -s, --statistics        Show statistics summary
#   -p, --processes         Track GPU processes
#   -c, --csv               Output in CSV format
#   -h, --help              Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_DIR="/var/log/gpu-tracking"

# Defaults
UPDATE_INTERVAL=5
DURATION=0
LOG_FILE=""
ENABLE_ALERTS=false
GPU_ID=""
UTIL_THRESHOLD=90
SHOW_STATS=false
TRACK_PROCESSES=false
CSV_OUTPUT=false

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

# Arrays for statistics
declare -a GPU_UTIL_HISTORY
declare -a GPU_MEM_HISTORY
declare -a GPU_TEMP_HISTORY
declare -a GPU_POWER_HISTORY

# ===========================
# GPU Detection
# ===========================

check_nvidia_smi() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not found${RESET}"
        echo "NVIDIA drivers or CUDA toolkit may not be installed."
        exit 1
    fi

    if ! nvidia-smi > /dev/null 2>&1; then
        echo -e "${RED}Error: nvidia-smi failed to run${RESET}"
        echo "GPU drivers may not be properly configured."
        exit 1
    fi
}

get_gpu_count() {
    nvidia-smi --query-gpu=count --format=csv,noheader | head -1
}

# ===========================
# GPU Metrics Collection
# ===========================

get_gpu_metrics() {
    local gpu_id="${1:-}"

    local query="index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit"

    if [[ -n "$gpu_id" ]]; then
        nvidia-smi --id="$gpu_id" --query-gpu="$query" --format=csv,noheader,nounits
    else
        nvidia-smi --query-gpu="$query" --format=csv,noheader,nounits
    fi
}

collect_gpu_stats() {
    local metrics="$1"

    # Parse metrics
    local gpu_util=$(echo "$metrics" | awk -F',' '{print $4}' | xargs)
    local mem_util=$(echo "$metrics" | awk -F',' '{print $5}' | xargs)
    local temp=$(echo "$metrics" | awk -F',' '{print $3}' | xargs)
    local power=$(echo "$metrics" | awk -F',' '{print $8}' | xargs)

    # Store in history
    GPU_UTIL_HISTORY+=("$gpu_util")
    GPU_MEM_HISTORY+=("$mem_util")
    GPU_TEMP_HISTORY+=("$temp")
    GPU_POWER_HISTORY+=("$power")
}

# ===========================
# Statistics Calculation
# ===========================

calculate_stats() {
    local -n array=$1

    if [[ ${#array[@]} -eq 0 ]]; then
        echo "0 0 0 0"
        return
    fi

    # Calculate min, max, avg
    local sum=0
    local min=${array[0]}
    local max=${array[0]}

    for value in "${array[@]}"; do
        sum=$((sum + value))

        if [[ $value -lt $min ]]; then
            min=$value
        fi

        if [[ $value -gt $max ]]; then
            max=$value
        fi
    done

    local count=${#array[@]}
    local avg=$((sum / count))

    # Calculate median
    local sorted=($(printf '%s\n' "${array[@]}" | sort -n))
    local median=${sorted[$((count / 2))]}

    echo "$min $max $avg $median"
}

# ===========================
# Display Functions
# ===========================

display_gpu_status() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [[ "$CSV_OUTPUT" == false ]]; then
        clear
        echo -e "${BOLD}${CYAN}GPU Utilization Tracker${RESET}"
        echo -e "${CYAN}========================================${RESET}"
        echo -e "${BOLD}Time:${RESET} $timestamp"
        echo ""
    fi

    get_gpu_metrics "$GPU_ID" | while IFS=',' read -r idx name temp gpu_util mem_util mem_used mem_total power_draw power_limit; do
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

        if [[ "$CSV_OUTPUT" == true ]]; then
            echo "$timestamp,$idx,$name,$temp,$gpu_util,$mem_util,$mem_used,$mem_total,$mem_pct,$power_draw,$power_limit"
        else
            # Color coding
            local gpu_color="$GREEN"
            if [[ $gpu_util -ge $UTIL_THRESHOLD ]]; then
                gpu_color="$RED"
            elif [[ $gpu_util -ge $((UTIL_THRESHOLD - 20)) ]]; then
                gpu_color="$YELLOW"
            fi

            local temp_color="$GREEN"
            if [[ $temp -ge 85 ]]; then
                temp_color="$RED"
            elif [[ $temp -ge 75 ]]; then
                temp_color="$YELLOW"
            fi

            local mem_color="$GREEN"
            if [[ $mem_pct -ge 90 ]]; then
                mem_color="$RED"
            elif [[ $mem_pct -ge 75 ]]; then
                mem_color="$YELLOW"
            fi

            echo -e "${BOLD}GPU $idx:${RESET} $name"
            echo -e "  ${gpu_color}GPU Utilization: ${gpu_util}%${RESET}"
            echo -e "  ${mem_color}Memory: ${mem_used} MB / ${mem_total} MB (${mem_pct}%)${RESET}"
            echo -e "  ${mem_color}Memory Utilization: ${mem_util}%${RESET}"
            echo -e "  ${temp_color}Temperature: ${temp}°C${RESET}"
            echo -e "  Power: ${power_draw}W / ${power_limit}W"
            echo ""
        fi

        # Collect statistics
        collect_gpu_stats "$idx,$name,$temp,$gpu_util,$mem_util,$mem_used,$mem_total,$power_draw,$power_limit"

        # Check for alerts
        if [[ "$ENABLE_ALERTS" == true ]]; then
            if [[ $gpu_util -ge $UTIL_THRESHOLD ]]; then
                send_alert "High GPU $idx utilization: ${gpu_util}%"
            fi

            if [[ $temp -ge 85 ]]; then
                send_alert "High GPU $idx temperature: ${temp}°C"
            fi

            if [[ $mem_pct -ge 95 ]]; then
                send_alert "High GPU $idx memory usage: ${mem_pct}%"
            fi
        fi

        # Log to file
        if [[ -n "$LOG_FILE" ]]; then
            echo "$timestamp,GPU$idx,$gpu_util,$mem_util,$mem_pct,$temp,$power_draw" >> "$LOG_FILE"
        fi
    done

    if [[ "$TRACK_PROCESSES" == true ]] && [[ "$CSV_OUTPUT" == false ]]; then
        display_gpu_processes
    fi
}

display_gpu_processes() {
    echo -e "${BOLD}GPU Processes:${RESET}"
    echo "-------------------------------------"

    local processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)

    if [[ -n "$processes" ]]; then
        printf "%-8s %-30s %12s\n" "PID" "Process" "GPU Memory"
        echo "-------------------------------------"

        echo "$processes" | while IFS=',' read -r pid process mem; do
            pid=$(echo "$pid" | xargs)
            process=$(echo "$process" | xargs)
            mem=$(echo "$mem" | xargs)

            printf "%-8s %-30s %12s\n" "$pid" "$process" "$mem"
        done
    else
        echo "No GPU processes running"
    fi

    echo ""
}

# ===========================
# Alerting
# ===========================

send_alert() {
    local message="$1"

    # Log to syslog
    logger -t "gpu-tracker" "$message"

    # Print to stderr
    echo -e "${RED}[ALERT] $message${RESET}" >&2

    # Could integrate with other alert systems
    # e.g., email, Slack, PagerDuty
}

# ===========================
# Statistics Summary
# ===========================

show_statistics() {
    echo -e "${BOLD}${CYAN}Statistics Summary${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    local samples=${#GPU_UTIL_HISTORY[@]}
    echo "Total samples: $samples"
    echo ""

    if [[ $samples -eq 0 ]]; then
        echo "No data collected yet"
        return
    fi

    # GPU Utilization stats
    local util_stats=($(calculate_stats GPU_UTIL_HISTORY))
    echo -e "${BOLD}GPU Utilization (%)${RESET}"
    echo "  Min: ${util_stats[0]}"
    echo "  Max: ${util_stats[1]}"
    echo "  Avg: ${util_stats[2]}"
    echo "  Median: ${util_stats[3]}"
    echo ""

    # Memory Utilization stats
    local mem_stats=($(calculate_stats GPU_MEM_HISTORY))
    echo -e "${BOLD}Memory Utilization (%)${RESET}"
    echo "  Min: ${mem_stats[0]}"
    echo "  Max: ${mem_stats[1]}"
    echo "  Avg: ${mem_stats[2]}"
    echo "  Median: ${mem_stats[3]}"
    echo ""

    # Temperature stats
    local temp_stats=($(calculate_stats GPU_TEMP_HISTORY))
    echo -e "${BOLD}Temperature (°C)${RESET}"
    echo "  Min: ${temp_stats[0]}"
    echo "  Max: ${temp_stats[1]}"
    echo "  Avg: ${temp_stats[2]}"
    echo "  Median: ${temp_stats[3]}"
    echo ""

    # Power stats
    local power_stats=($(calculate_stats GPU_POWER_HISTORY))
    echo -e "${BOLD}Power Draw (W)${RESET}"
    echo "  Min: ${power_stats[0]}"
    echo "  Max: ${power_stats[1]}"
    echo "  Avg: ${power_stats[2]}"
    echo "  Median: ${power_stats[3]}"
    echo ""
}

# ===========================
# Monitoring Loop
# ===========================

start_monitoring() {
    local start_time=$(date +%s)
    local iteration=0

    if [[ "$CSV_OUTPUT" == true ]]; then
        echo "Timestamp,GPU,Name,Temp,GPU_Util,Mem_Util,Mem_Used,Mem_Total,Mem_Pct,Power_Draw,Power_Limit"
    fi

    while true; do
        display_gpu_status

        ((iteration++))

        # Check duration
        if [[ $DURATION -gt 0 ]]; then
            local current_time=$(date +%s)
            local elapsed=$((current_time - start_time))

            if [[ $elapsed -ge $DURATION ]]; then
                echo -e "${YELLOW}Monitoring duration reached${RESET}"
                break
            fi
        fi

        if [[ "$CSV_OUTPUT" == false ]]; then
            echo -e "${CYAN}========================================${RESET}"
            echo "Iteration: $iteration | Interval: ${UPDATE_INTERVAL}s"

            if [[ $DURATION -gt 0 ]]; then
                local remaining=$((DURATION - ($(date +%s) - start_time)))
                echo "Time remaining: ${remaining}s"
            fi

            echo -e "\n${YELLOW}Press Ctrl+C to stop${RESET}"
        fi

        sleep "$UPDATE_INTERVAL"
    done
}

# ===========================
# Cleanup
# ===========================

cleanup() {
    if [[ "$SHOW_STATS" == true ]]; then
        echo ""
        show_statistics
    fi

    echo ""
    echo -e "${GREEN}GPU tracking stopped${RESET}"

    if [[ -n "$LOG_FILE" ]]; then
        echo "Log saved to: $LOG_FILE"
    fi
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

GPU utilization tracking and logging.

OPTIONS:
    -i, --interval SECONDS     Update interval (default: $UPDATE_INTERVAL)
    -d, --duration SECONDS     Total duration (0 = infinite, default: $DURATION)
    -l, --log FILE            Log metrics to file
    -a, --alert               Enable alert notifications
    -g, --gpu ID              Monitor specific GPU (default: all)
    -t, --threshold PCT       Utilization threshold for alerts (default: $UTIL_THRESHOLD)
    -s, --statistics          Show statistics summary on exit
    -p, --processes           Track GPU processes
    -c, --csv                 Output in CSV format
    -h, --help                Display this help message

EXAMPLES:
    # Basic monitoring
    $SCRIPT_NAME

    # Monitor every 2 seconds for 60 seconds
    $SCRIPT_NAME -i 2 -d 60

    # Log to file with statistics
    $SCRIPT_NAME -l gpu-metrics.log -s

    # Monitor specific GPU with alerts
    $SCRIPT_NAME -g 0 -a -t 85

    # CSV output with process tracking
    $SCRIPT_NAME -c -p -l gpu-data.csv

    # Alert on high utilization
    $SCRIPT_NAME --alert --threshold 90 --interval 10

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
            -d|--duration)
                DURATION="$2"
                shift 2
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            -a|--alert)
                ENABLE_ALERTS=true
                shift
                ;;
            -g|--gpu)
                GPU_ID="$2"
                shift 2
                ;;
            -t|--threshold)
                UTIL_THRESHOLD="$2"
                shift 2
                ;;
            -s|--statistics)
                SHOW_STATS=true
                shift
                ;;
            -p|--processes)
                TRACK_PROCESSES=true
                shift
                ;;
            -c|--csv)
                CSV_OUTPUT=true
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

    # Check for nvidia-smi
    check_nvidia_smi

    # Create log directory if needed
    if [[ -n "$LOG_FILE" ]]; then
        mkdir -p "$(dirname "$LOG_FILE")"

        # Write CSV header if CSV output
        if [[ "$CSV_OUTPUT" == true ]] && [[ ! -f "$LOG_FILE" ]]; then
            echo "Timestamp,GPU,GPU_Util,Mem_Util,Mem_Pct,Temp,Power" > "$LOG_FILE"
        fi
    fi

    # Set up cleanup trap
    trap cleanup EXIT INT TERM

    # Start monitoring
    start_monitoring
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
