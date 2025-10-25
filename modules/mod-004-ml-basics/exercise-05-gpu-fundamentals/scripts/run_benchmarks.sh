#!/bin/bash
#
# Run all GPU benchmark scripts
# This script runs GPU detection, performance benchmarks, memory management,
# and model inference comparisons
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}GPU Fundamentals - Benchmark Suite${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if src directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo -e "${RED}Error: src directory not found at $SRC_DIR${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

# 1. GPU Detection
echo -e "\n${YELLOW}============================================================${NC}"
echo -e "${YELLOW}1. GPU Detection and Information${NC}"
echo -e "${YELLOW}============================================================${NC}"
python3 "$SRC_DIR/check_gpu.py" || {
    echo -e "${RED}GPU detection failed${NC}"
    exit 1
}

# 2. CPU vs GPU Performance Benchmark
echo -e "\n${YELLOW}============================================================${NC}"
echo -e "${YELLOW}2. CPU vs GPU Performance Benchmark${NC}"
echo -e "${YELLOW}============================================================${NC}"
python3 "$SRC_DIR/cpu_vs_gpu_benchmark.py" || {
    echo -e "${RED}Performance benchmark failed${NC}"
    exit 1
}

# 3. Memory Management
echo -e "\n${YELLOW}============================================================${NC}"
echo -e "${YELLOW}3. GPU Memory Management${NC}"
echo -e "${YELLOW}============================================================${NC}"
python3 "$SRC_DIR/memory_management.py" || {
    echo -e "${YELLOW}Memory management demo skipped (may require GPU)${NC}"
}

# 4. Model Inference Comparison
echo -e "\n${YELLOW}============================================================${NC}"
echo -e "${YELLOW}4. Model Inference Comparison${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}This may take several minutes as it downloads and runs a model...${NC}"
python3 "$SRC_DIR/model_inference_comparison.py" || {
    echo -e "${YELLOW}Model inference comparison failed (may require transformers)${NC}"
}

# Summary
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Benchmark Suite Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"

# Check if GPU was available
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "\n${GREEN}GPU benchmarks completed successfully.${NC}"
    echo -e "Review the results above to see CPU vs GPU performance differences."
else
    echo -e "\n${YELLOW}Benchmarks completed in CPU-only mode.${NC}"
    echo -e "For GPU benchmarks, run on a system with NVIDIA GPU and CUDA support."
fi

echo -e "\nFor more details, see:"
echo "  - README.md: Overview and usage"
echo "  - STEP_BY_STEP.md: Implementation details and concepts"
