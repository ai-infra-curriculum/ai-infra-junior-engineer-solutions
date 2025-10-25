#!/bin/bash
# Run script for LLM API server

set -e  # Exit on error

echo "=========================================="
echo "Starting LLM API Server"
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

# Load environment variables if .env exists
ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Set defaults if not provided
export MODEL_NAME="${MODEL_NAME:-gpt2}"
export DEVICE="${DEVICE:--1}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-5000}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Device: $DEVICE"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""
echo "Starting server..."
echo "Press Ctrl+C to stop"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Run the API server
python src/llm_api.py
