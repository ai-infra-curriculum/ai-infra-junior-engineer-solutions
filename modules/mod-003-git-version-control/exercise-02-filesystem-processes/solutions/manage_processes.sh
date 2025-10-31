#!/bin/bash
#
# manage_processes.sh - ML process management utility
#
# Description:
#   Comprehensive process management for ML workloads including listing,
#   monitoring, killing, and resource control of training processes.
#
# Usage:
#   ./manage_processes.sh [ACTION] [OPTIONS]
#
# Actions:
#   list              List ML processes
#   monitor PID       Monitor specific process
#   kill PID          Kill process gracefully
#   killall PATTERN   Kill all matching processes
#   priority PID VAL  Set process priority
#   affinity PID CPUS Set CPU affinity
#   limits PID        Show process resource limits
#   tree [PID]        Show process tree
#
# Options:
#   -v, --verbose     Verbose output
#   -f, --force       Force kill (SIGKILL)
#   -w, --watch       Continuous monitoring
#   -h, --help        Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/ml-process-manager.log"

# Defaults
VERBOSE=false
FORCE_KILL=false
WATCH_MODE=false

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

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [[ "$VERBOSE" == true ]]; then
        echo -e "[${timestamp}] [${level}] ${message}"
    fi

    echo "[${timestamp}] [${level}] ${message}" >> "$LOG_FILE"
}

# ===========================
# Process Detection
# ===========================

is_ml_process() {
    local cmd="$1"

    # Check if process is ML-related
    if echo "$cmd" | grep -qE "(python.*train|pytorch|tensorflow|jupyter|python.*\.py)"; then
        return 0
    fi
    return 1
}

get_all_processes() {
    ps aux --sort=-%cpu | awk 'NR>1 {print $0}'
}

get_ml_processes() {
    get_all_processes | while read -r line; do
        local cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
        if is_ml_process "$cmd"; then
            echo "$line"
        fi
    done
}

# ===========================
# Process Listing
# ===========================

list_processes() {
    echo -e "${BOLD}${CYAN}ML Training Processes${RESET}"
    echo -e "${CYAN}=====================================================================================================${RESET}"

    printf "%-8s %6s %6s %6s %10s %8s %8s %s\n" \
        "USER" "PID" "%CPU" "%MEM" "VSZ" "RSS" "TIME" "COMMAND"

    echo "-----------------------------------------------------------------------------------------------------"

    local count=0

    get_ml_processes | while read -r line; do
        local user=$(echo "$line" | awk '{print $1}')
        local pid=$(echo "$line" | awk '{print $2}')
        local cpu=$(echo "$line" | awk '{print $3}')
        local mem=$(echo "$line" | awk '{print $4}')
        local vsz=$(echo "$line" | awk '{print $5}')
        local rss=$(echo "$line" | awk '{print $6}')
        local time=$(echo "$line" | awk '{print $10}')
        local cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i}' | cut -c1-60)

        # Color code based on resource usage
        local cpu_int=${cpu%.*}
        local mem_int=${mem%.*}

        local color="$RESET"
        if [[ $cpu_int -gt 80 ]] || [[ $mem_int -gt 80 ]]; then
            color="$RED"
        elif [[ $cpu_int -gt 50 ]] || [[ $mem_int -gt 50 ]]; then
            color="$YELLOW"
        fi

        printf "${color}%-8s %6s %6s %6s %10s %8s %8s %s${RESET}\n" \
            "$user" "$pid" "$cpu" "$mem" "$vsz" "$rss" "$time" "$cmd"

        ((count++))
    done

    echo ""
    echo "Total ML processes: $count"

    log "INFO" "Listed $count ML processes"
}

# ===========================
# Process Monitoring
# ===========================

monitor_process() {
    local pid="$1"

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${RED}Process $pid does not exist${RESET}"
        return 1
    fi

    local monitor_fn() {
        clear
        echo -e "${BOLD}${CYAN}Process Monitor - PID: $pid${RESET}"
        echo -e "${CYAN}========================================${RESET}"
        echo ""

        # Basic process info
        echo -e "${BOLD}Process Information:${RESET}"
        ps -p "$pid" -o user,pid,ppid,pgid,stat,pri,nice,pcpu,pmem,vsz,rss,etime,cmd | \
            awk 'NR==1 {print $0} NR==2 {print $0}'
        echo ""

        # Command line
        echo -e "${BOLD}Command Line:${RESET}"
        cat "/proc/$pid/cmdline" | tr '\0' ' '
        echo -e "\n"

        # Working directory
        echo -e "${BOLD}Working Directory:${RESET}"
        readlink "/proc/$pid/cwd" 2>/dev/null || echo "N/A"
        echo ""

        # Open files count
        local open_files=$(ls -1 "/proc/$pid/fd" 2>/dev/null | wc -l)
        echo -e "${BOLD}Open File Descriptors:${RESET} $open_files"
        echo ""

        # Memory details
        echo -e "${BOLD}Memory Details:${RESET}"
        if [[ -f "/proc/$pid/status" ]]; then
            grep -E "(VmSize|VmRSS|VmData|VmStk|VmExe)" "/proc/$pid/status" | \
                awk '{printf "  %-10s %s\n", $1, $2" "$3}'
        fi
        echo ""

        # CPU affinity
        if command -v taskset &> /dev/null; then
            local affinity=$(taskset -p "$pid" 2>/dev/null | awk -F: '{print $2}' | xargs)
            echo -e "${BOLD}CPU Affinity:${RESET} $affinity"
            echo ""
        fi

        # Threads
        local thread_count=$(ps -T -p "$pid" 2>/dev/null | wc -l)
        ((thread_count--)) # Subtract header
        echo -e "${BOLD}Threads:${RESET} $thread_count"
        echo ""

        # I/O statistics
        if [[ -f "/proc/$pid/io" ]]; then
            echo -e "${BOLD}I/O Statistics:${RESET}"
            cat "/proc/$pid/io" | awk '{printf "  %-20s %s\n", $1, $2}'
            echo ""
        fi

        # GPU usage if available
        if command -v nvidia-smi &> /dev/null; then
            local gpu_info=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$pid,")
            if [[ -n "$gpu_info" ]]; then
                echo -e "${BOLD}GPU Usage:${RESET}"
                echo "  Memory: $(echo $gpu_info | cut -d',' -f2)"
                echo ""
            fi
        fi

        echo -e "${CYAN}========================================${RESET}"
        echo -e "${BOLD}Last Update:${RESET} $(date '+%Y-%m-%d %H:%M:%S')"

        if [[ "$WATCH_MODE" == true ]]; then
            echo -e "\n${YELLOW}Press Ctrl+C to stop monitoring${RESET}"
        fi
    }

    if [[ "$WATCH_MODE" == true ]]; then
        trap 'echo -e "\n${YELLOW}Monitoring stopped${RESET}"; exit 0' INT

        while ps -p "$pid" > /dev/null 2>&1; do
            monitor_fn
            sleep 2
        done

        echo -e "${RED}Process $pid has terminated${RESET}"
    else
        monitor_fn
    fi

    log "INFO" "Monitored process $pid"
}

# ===========================
# Process Control
# ===========================

kill_process() {
    local pid="$1"

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${RED}Process $pid does not exist${RESET}"
        return 1
    fi

    # Show process info
    echo -e "${BOLD}Process to kill:${RESET}"
    ps -p "$pid" -o user,pid,pcpu,pmem,cmd | awk 'NR<=2'
    echo ""

    if [[ "$FORCE_KILL" == true ]]; then
        echo -e "${YELLOW}Force killing process $pid...${RESET}"
        kill -9 "$pid"
        log "WARNING" "Force killed process $pid"
        echo -e "${GREEN}Process $pid killed (SIGKILL)${RESET}"
    else
        echo -e "${YELLOW}Attempting graceful shutdown of process $pid...${RESET}"
        kill -15 "$pid"

        # Wait for process to terminate
        local timeout=10
        local elapsed=0

        while ps -p "$pid" > /dev/null 2>&1; do
            if [[ $elapsed -ge $timeout ]]; then
                echo -e "${YELLOW}Process did not terminate gracefully, force killing...${RESET}"
                kill -9 "$pid"
                log "WARNING" "Force killed process $pid after timeout"
                break
            fi

            sleep 1
            ((elapsed++))
        done

        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${GREEN}Process $pid terminated gracefully${RESET}"
            log "INFO" "Killed process $pid gracefully"
        else
            echo -e "${GREEN}Process $pid killed (SIGKILL)${RESET}"
        fi
    fi
}

killall_pattern() {
    local pattern="$1"

    echo -e "${BOLD}Searching for processes matching: $pattern${RESET}"
    echo ""

    local pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}')

    if [[ -z "$pids" ]]; then
        echo -e "${YELLOW}No processes found matching pattern${RESET}"
        return 0
    fi

    echo -e "${BOLD}Matching processes:${RESET}"
    ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2, $11}'
    echo ""

    local count=$(echo "$pids" | wc -w)
    echo -e "${YELLOW}Found $count processes. Kill all? (y/N)${RESET}"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        for pid in $pids; do
            echo "Killing PID $pid..."
            kill_process "$pid"
        done
        echo -e "${GREEN}All matching processes terminated${RESET}"
        log "INFO" "Killed $count processes matching '$pattern'"
    else
        echo "Cancelled"
    fi
}

# ===========================
# Process Priority
# ===========================

set_priority() {
    local pid="$1"
    local priority="$2"

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${RED}Process $pid does not exist${RESET}"
        return 1
    fi

    # Validate priority range
    if [[ $priority -lt -20 ]] || [[ $priority -gt 19 ]]; then
        echo -e "${RED}Priority must be between -20 (highest) and 19 (lowest)${RESET}"
        return 1
    fi

    echo -e "${YELLOW}Setting priority of process $pid to $priority...${RESET}"

    if renice -n "$priority" -p "$pid" > /dev/null 2>&1; then
        echo -e "${GREEN}Priority set successfully${RESET}"

        # Show new priority
        local new_priority=$(ps -p "$pid" -o ni= | xargs)
        echo "New priority (NI): $new_priority"

        log "INFO" "Set priority of process $pid to $priority"
    else
        echo -e "${RED}Failed to set priority (may need sudo)${RESET}"
        return 1
    fi
}

# ===========================
# CPU Affinity
# ===========================

set_cpu_affinity() {
    local pid="$1"
    local cpus="$2"

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${RED}Process $pid does not exist${RESET}"
        return 1
    fi

    if ! command -v taskset &> /dev/null; then
        echo -e "${RED}taskset command not available${RESET}"
        return 1
    fi

    echo -e "${YELLOW}Setting CPU affinity of process $pid to CPUs: $cpus${RESET}"

    if taskset -cp "$cpus" "$pid" > /dev/null 2>&1; then
        echo -e "${GREEN}CPU affinity set successfully${RESET}"

        # Show new affinity
        local affinity=$(taskset -p "$pid" | awk -F: '{print $2}' | xargs)
        echo "New affinity: $affinity"

        log "INFO" "Set CPU affinity of process $pid to $cpus"
    else
        echo -e "${RED}Failed to set CPU affinity${RESET}"
        return 1
    fi
}

# ===========================
# Process Limits
# ===========================

show_limits() {
    local pid="$1"

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${RED}Process $pid does not exist${RESET}"
        return 1
    fi

    echo -e "${BOLD}${CYAN}Resource Limits for Process $pid${RESET}"
    echo -e "${CYAN}========================================${RESET}"

    if [[ -f "/proc/$pid/limits" ]]; then
        cat "/proc/$pid/limits"
    else
        echo "Limits not available"
    fi

    log "INFO" "Showed limits for process $pid"
}

# ===========================
# Process Tree
# ===========================

show_process_tree() {
    local pid="${1:-}"

    echo -e "${BOLD}${CYAN}Process Tree${RESET}"
    echo -e "${CYAN}========================================${RESET}"

    if command -v pstree &> /dev/null; then
        if [[ -n "$pid" ]]; then
            pstree -p "$pid"
        else
            # Show ML-related process trees
            get_ml_processes | awk '{print $2}' | while read -r p; do
                echo ""
                echo -e "${BOLD}PID $p:${RESET}"
                pstree -p "$p"
            done
        fi
    else
        echo "pstree command not available"
        echo "Using ps tree view:"
        ps auxf | grep -E "(python|jupyter)" | grep -v grep
    fi
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [ACTION] [OPTIONS]

ML process management utility.

ACTIONS:
    list                    List ML processes
    monitor PID             Monitor specific process
    kill PID                Kill process gracefully
    killall PATTERN         Kill all matching processes
    priority PID VALUE      Set process priority (-20 to 19)
    affinity PID CPUS       Set CPU affinity (e.g., 0,1,2 or 0-3)
    limits PID              Show process resource limits
    tree [PID]              Show process tree

OPTIONS:
    -v, --verbose           Verbose output
    -f, --force             Force kill (SIGKILL)
    -w, --watch             Continuous monitoring
    -h, --help              Display this help message

EXAMPLES:
    # List ML processes
    $SCRIPT_NAME list

    # Monitor process
    $SCRIPT_NAME monitor 12345

    # Monitor with updates
    $SCRIPT_NAME monitor 12345 --watch

    # Kill process
    $SCRIPT_NAME kill 12345

    # Force kill
    $SCRIPT_NAME kill 12345 --force

    # Kill all Python training processes
    $SCRIPT_NAME killall "python.*train"

    # Set process priority
    $SCRIPT_NAME priority 12345 10

    # Set CPU affinity
    $SCRIPT_NAME affinity 12345 0-3

    # Show limits
    $SCRIPT_NAME limits 12345

    # Show process tree
    $SCRIPT_NAME tree

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

    # Parse action
    ACTION="$1"
    shift

    # Parse options and arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--force)
                FORCE_KILL=true
                shift
                ;;
            -w|--watch)
                WATCH_MODE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# ===========================
# Main Function
# ===========================

main() {
    ARGS=()
    parse_arguments "$@"

    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"

    case "$ACTION" in
        list)
            list_processes
            ;;
        monitor)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                echo "Error: PID required"
                usage
                exit 1
            fi
            monitor_process "${ARGS[0]}"
            ;;
        kill)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                echo "Error: PID required"
                usage
                exit 1
            fi
            kill_process "${ARGS[0]}"
            ;;
        killall)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                echo "Error: Pattern required"
                usage
                exit 1
            fi
            killall_pattern "${ARGS[0]}"
            ;;
        priority)
            if [[ ${#ARGS[@]} -lt 2 ]]; then
                echo "Error: PID and priority value required"
                usage
                exit 1
            fi
            set_priority "${ARGS[0]}" "${ARGS[1]}"
            ;;
        affinity)
            if [[ ${#ARGS[@]} -lt 2 ]]; then
                echo "Error: PID and CPU list required"
                usage
                exit 1
            fi
            set_cpu_affinity "${ARGS[0]}" "${ARGS[1]}"
            ;;
        limits)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                echo "Error: PID required"
                usage
                exit 1
            fi
            show_limits "${ARGS[0]}"
            ;;
        tree)
            show_process_tree "${ARGS[0]:-}"
            ;;
        *)
            echo "Unknown action: $ACTION"
            usage
            exit 1
            ;;
    esac
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
