#!/bin/bash
#
# gpu_process_manager.sh - Manage GPU processes
#
# Usage:
#   ./gpu_process_manager.sh list
#   ./gpu_process_manager.sh kill <PID>
#   ./gpu_process_manager.sh monitor
#

set -e
set -u

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

# Check nvidia-smi
check_nvidia_smi() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}nvidia-smi not available${NC}" >&2
        echo "This tool requires an NVIDIA GPU with drivers installed." >&2
        return 1
    fi
    return 0
}

# List GPU processes
list_gpu_processes() {
    if ! check_nvidia_smi; then
        return 1
    fi

    echo -e "${BLUE}=== GPU Processes ===${NC}"
    echo ""

    if ! nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -q .; then
        echo "No GPU processes running"
        return 0
    fi

    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
}

# Kill GPU process
kill_gpu_process() {
    local pid=$1

    if ! check_nvidia_smi; then
        return 1
    fi

    if [ -z "$pid" ]; then
        echo -e "${RED}Error: PID required${NC}" >&2
        echo "Usage: $0 kill <PID>" >&2
        return 1
    fi

    # Verify PID is numeric
    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Error: Invalid PID: $pid${NC}" >&2
        return 1
    fi

    # Check if process is using GPU
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "^$pid$"; then
        echo -e "${RED}Error: PID $pid not found in GPU processes${NC}" >&2
        echo ""
        echo "Current GPU processes:"
        list_gpu_processes
        return 1
    fi

    echo -e "${BLUE}Stopping GPU process $pid...${NC}"

    # Try graceful termination
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}Sending SIGTERM...${NC}"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
    fi

    # Check if still running
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}Process still running, force killing...${NC}"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    # Verify termination
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${RED}Failed to kill process $pid${NC}" >&2
        return 1
    else
        echo -e "${GREEN}✓ Process $pid stopped${NC}"
        return 0
    fi
}

# Monitor GPUs
monitor_gpus() {
    if ! check_nvidia_smi; then
        return 1
    fi

    exec ./monitor_gpu.sh
}

# Show usage
usage() {
    cat << EOF
Usage: $0 {list|kill <PID>|monitor}

Commands:
    list            List all GPU processes
    kill <PID>      Kill GPU process by PID
    monitor         Start continuous GPU monitoring

Examples:
    $0 list
    $0 kill 12345
    $0 monitor
EOF
}

# Main command handler
case "${1:-}" in
    list)
        list_gpu_processes
        ;;
    kill)
        if [ $# -lt 2 ]; then
            echo -e "${RED}Error: PID required${NC}" >&2
            usage
            exit 1
        fi
        kill_gpu_process "$2"
        ;;
    monitor)
        monitor_gpus
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo -e "${RED}Error: Invalid command: ${1:-}${NC}" >&2
        echo ""
        usage
        exit 1
        ;;
esac
