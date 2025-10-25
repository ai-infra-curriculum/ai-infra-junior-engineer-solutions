#!/bin/bash
#
# Test script for GPU Fundamentals Exercise
# Runs pytest test suite with appropriate options
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}GPU Fundamentals - Test Suite${NC}"
echo -e "${BLUE}============================================================${NC}"

cd "$PROJECT_ROOT"

# Check if pytest is installed
if ! python3 -c "import pytest" 2>/dev/null; then
    echo -e "${RED}Error: pytest not installed${NC}"
    echo "Install with: pip3 install pytest"
    exit 1
fi

# Check GPU availability
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}GPU available - running full test suite${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}GPU not available - some tests will be skipped${NC}"
    GPU_AVAILABLE=false
fi

# Run tests
echo -e "\n${YELLOW}Running tests...${NC}"

# Default pytest options
PYTEST_OPTS="-v --tb=short"

# Add coverage if available
if python3 -c "import pytest_cov" 2>/dev/null; then
    PYTEST_OPTS="$PYTEST_OPTS --cov=src --cov-report=term-missing"
fi

# Run pytest
python3 -m pytest tests/ $PYTEST_OPTS

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "${GREEN}============================================================${NC}"

    if [ "$GPU_AVAILABLE" = false ]; then
        echo -e "\n${YELLOW}Note: Some GPU-specific tests were skipped.${NC}"
        echo "Run on a GPU-enabled system for complete test coverage."
    fi
else
    echo -e "\n${RED}============================================================${NC}"
    echo -e "${RED}Some tests failed!${NC}"
    echo -e "${RED}============================================================${NC}"
    exit 1
fi

# Additional test information
echo -e "\nTest coverage:"
echo "  - GPU detection: ✓"
echo "  - Performance benchmarking: ✓"
echo "  - Memory management: ✓"
echo "  - Device-agnostic code: ✓"

if [ "$GPU_AVAILABLE" = true ]; then
    echo "  - GPU-specific tests: ✓"
else
    echo "  - GPU-specific tests: Skipped (no GPU)"
fi

echo -e "\nFor more testing options:"
echo "  pytest tests/ -v              # Verbose output"
echo "  pytest tests/ -k test_name    # Run specific test"
echo "  pytest tests/ --markers       # Show available markers"
echo "  pytest tests/ -x              # Stop on first failure"
