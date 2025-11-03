#!/bin/bash

# Sentiment Classifier Setup Script
# Automates virtual environment creation, dependency installation, and configuration

set -e  # Exit on error

echo "================================"
echo "Sentiment Classifier Setup"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 11 ]; then
    echo -e "${RED}✗ Error: Python 3.11+ required. Found: $PYTHON_VERSION${NC}"
    echo ""
    echo "Install Python 3.11:"
    echo "  Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "  macOS: brew install python@3.11"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# Create virtual environment
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment ready${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded to $(pip --version | awk '{print $2}')${NC}"
echo ""

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements-dev.txt" ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt --quiet
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
else
    echo "Installing production dependencies..."
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Production dependencies installed${NC}"
fi
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env file from .env.example..."
        cp .env.example .env
        echo -e "${GREEN}✓ .env file created${NC}"
        echo -e "${YELLOW}⚠ Please edit .env with your configuration${NC}"
    else
        echo -e "${YELLOW}⚠ No .env.example found${NC}"
    fi
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data models
touch data/.gitkeep models/.gitkeep
echo -e "${GREEN}✓ Project directories created${NC}"
echo ""

# Display summary
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Edit .env file with your configuration"
echo "  3. Run training: python src/train.py"
echo "  4. Run tests: pytest tests/ -v"
echo ""
echo "To deactivate: deactivate"
echo ""

# Display installed packages
echo "Installed packages:"
pip list --format=columns | head -15
echo "..."
echo ""

echo -e "${GREEN}✓ Setup successful!${NC}"
