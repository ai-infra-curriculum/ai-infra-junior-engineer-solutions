#!/bin/bash
# Test script for LLM Basics exercise

set -e  # Exit on error

echo "=========================================="
echo "Running LLM Basics Tests"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if virtual environment exists
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run: ./scripts/setup.sh"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Change to project directory
cd "$PROJECT_DIR"

# Run pytest with coverage
echo ""
echo "Running tests with pytest..."
echo ""

pytest tests/ \
    -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html \
    "$@"

echo ""
echo "=========================================="
echo "Tests Complete!"
echo "=========================================="
echo ""
echo "Coverage report generated in: htmlcov/index.html"
echo ""
