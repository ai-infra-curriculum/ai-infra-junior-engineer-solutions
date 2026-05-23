#!/bin/bash

# Setup script for Airflow Workflow Monitoring Exercise
# This script prepares the environment for running Airflow

set -e  # Exit on error

echo "=========================================="
echo "Airflow Workflow Monitoring - Setup"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
echo "Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}Docker is installed${NC}"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs plugins dags src tests config

# Create .env file with AIRFLOW_UID
echo ""
echo "Setting up environment variables..."
if [ ! -f .env ]; then
    echo "AIRFLOW_UID=$(id -u)" > .env
    echo -e "${GREEN}Created .env file${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Display environment variables
echo ""
echo "Environment configuration:"
cat .env

# Check if containers are already running
echo ""
echo "Checking for running containers..."
cd docker
if docker-compose ps | grep -q "Up"; then
    echo -e "${YELLOW}Warning: Some containers are already running${NC}"
    read -p "Do you want to stop them and start fresh? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping existing containers..."
        docker-compose down
    fi
fi

# Pull latest images
echo ""
echo "Pulling Docker images..."
docker-compose pull

# Build custom image if Dockerfile exists
if [ -f Dockerfile ]; then
    echo ""
    echo "Building custom Airflow image..."
    docker-compose build
fi

cd ..

# Create __init__.py files if they don't exist
echo ""
echo "Ensuring Python packages are initialized..."
touch dags/__init__.py
touch src/__init__.py
touch tests/__init__.py

echo ""
echo -e "${GREEN}=========================================="
echo "Setup completed successfully!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Run: ./scripts/start.sh"
echo "  2. Wait 1-2 minutes for services to start"
echo "  3. Access Airflow UI: http://localhost:8080"
echo "     Username: airflow"
echo "     Password: airflow"
echo ""
echo "Additional services:"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo ""
