#!/bin/bash
#
# monitor_gpu.sh - Monitor GPU usage for ML training
#
# Requires: NVIDIA GPU with nvidia-smi installed
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

# Check if nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}nvidia-smi not found${NC}" >&2
    echo ""
    echo "GPU monitoring requires an NVIDIA GPU with drivers installed."
    echo ""
    echo "To install NVIDIA drivers:"
    echo "  Ubuntu/Debian: sudo apt install nvidia-driver-XXX"
    echo "  Check available versions: apt search nvidia-driver"
    echo ""
    echo "Or skip GPU monitoring if no NVIDIA GPU is available."
    exit 1
fi

echo -e "${BLUE}=== GPU Monitoring ===${NC}"
echo ""

# Show GPU information
echo -e "${BLUE}GPU Information:${NC}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv

echo ""
echo -e "${BLUE}Current GPU Usage:${NC}"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv

echo ""
echo -e "${BLUE}GPU Processes:${NC}"
if ! nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "No GPU processes running"
else
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi

echo ""
echo -e "${YELLOW}Starting continuous monitoring...${NC}"
echo -e "${YELLOW}Update interval: 2 seconds${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Continuous monitoring
nvidia-smi -l 2
