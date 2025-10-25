#!/bin/bash
#
# Setup script for GPU Fundamentals Exercise
# Installs required dependencies and verifies environment
#

set -e  # Exit on error

echo "============================================================"
echo "GPU Fundamentals Exercise - Setup"
echo "============================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check pip
echo -e "\n${YELLOW}Checking pip...${NC}"
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip3 not found. Please install pip.${NC}"
    exit 1
fi
echo "pip3 found: $(pip3 --version)"

# Check for NVIDIA GPU
echo -e "\n${YELLOW}Checking for NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}No NVIDIA GPU detected or nvidia-smi not available.${NC}"
    echo "This is fine - the code will run in CPU-only mode."
    GPU_AVAILABLE=false
fi

# Install dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip3 install -r requirements.txt
else
    echo -e "${RED}requirements.txt not found!${NC}"
    echo "Installing core dependencies manually..."

    if [ "$GPU_AVAILABLE" = true ]; then
        echo "Installing PyTorch with CUDA support..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch (CPU only)..."
        pip3 install torch torchvision
    fi

    echo "Installing additional dependencies..."
    pip3 install transformers psutil pytest
fi

# Verify installations
echo -e "\n${YELLOW}Verifying installations...${NC}"

# Check PyTorch
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo -e "${RED}Failed to import PyTorch${NC}"
    exit 1
}

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check transformers
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')" || {
    echo -e "${YELLOW}Warning: transformers not installed${NC}"
}

# Check psutil
python3 -c "import psutil; print(f'psutil version: {psutil.__version__}')" || {
    echo -e "${YELLOW}Warning: psutil not installed${NC}"
}

# Check pytest
python3 -c "import pytest; print(f'pytest version: {pytest.__version__}')" || {
    echo -e "${YELLOW}Warning: pytest not installed${NC}"
}

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"

echo -e "\nNext steps:"
echo "1. Run GPU detection: python3 src/check_gpu.py"
echo "2. Run benchmarks: ./scripts/run_benchmarks.sh"
echo "3. Run tests: ./scripts/test.sh"

if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "\n${GREEN}GPU is available - you can run all benchmarks!${NC}"
else
    echo -e "\n${YELLOW}Running in CPU-only mode - some features will be limited.${NC}"
fi
