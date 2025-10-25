#!/bin/bash
# Setup script for LLM Basics exercise
# This script sets up the Python environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "LLM Basics Exercise - Environment Setup"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"

# Check Python version
echo ""
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at: $VENV_DIR"
else
    echo ""
    echo "Virtual environment already exists at: $VENV_DIR"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
    echo "Dependencies installed successfully!"
else
    echo "Error: requirements.txt not found"
    exit 1
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

echo ""
echo "Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Checking Transformers..."
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo ""
echo "Checking Flask..."
python3 -c "import flask; print(f'Flask version: {flask.__version__}')"

echo ""
echo "Checking pytest..."
python3 -c "import pytest; print(f'pytest version: {pytest.__version__}')"

# Set up environment variables
ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "Creating .env file with default configuration..."
    cat > "$ENV_FILE" << EOF
# LLM API Configuration
MODEL_NAME=gpt2
DEVICE=-1
MAX_LENGTH_LIMIT=200
DEFAULT_MAX_LENGTH=50
DEFAULT_TEMPERATURE=0.7
HOST=0.0.0.0
PORT=5000
DEBUG=false
EOF
    echo ".env file created at: $ENV_FILE"
else
    echo ""
    echo ".env file already exists"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the basic generation script:"
echo "   python src/basic_generation.py"
echo ""
echo "3. Start the API server:"
echo "   ./scripts/run.sh"
echo ""
echo "4. Run tests:"
echo "   ./scripts/test.sh"
echo ""
