#!/bin/bash

# Start script for Airflow Workflow Monitoring Exercise
# This script starts all services

set -e  # Exit on error

echo "=========================================="
echo "Airflow Workflow Monitoring - Start"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Running setup first..."
    ./scripts/setup.sh
fi

# Start services
echo "Starting all services..."
echo ""
cd docker

# Start in detached mode
docker-compose up -d

echo ""
echo -e "${BLUE}Waiting for services to initialize...${NC}"
echo "This may take 1-2 minutes..."
echo ""

# Wait for database to be ready
echo "Waiting for PostgreSQL..."
sleep 5

# Check if postgres is healthy
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker-compose ps postgres | grep -q "healthy"; then
        echo -e "${GREEN}PostgreSQL is ready${NC}"
        break
    fi
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${YELLOW}PostgreSQL health check timeout, continuing anyway...${NC}"
fi

echo ""

# Wait for Airflow webserver
echo "Waiting for Airflow webserver..."
sleep 10

RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker-compose ps airflow-webserver | grep -q "healthy"; then
        echo -e "${GREEN}Airflow webserver is ready${NC}"
        break
    fi
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo ""
echo ""

# Display service status
echo "Service Status:"
echo "==============="
docker-compose ps

cd ..

echo ""
echo -e "${GREEN}=========================================="
echo "All services started successfully!"
echo "==========================================${NC}"
echo ""
echo "Access the following services:"
echo ""
echo -e "${BLUE}Airflow UI:${NC}"
echo "  URL: http://localhost:8080"
echo "  Username: airflow"
echo "  Password: airflow"
echo ""
echo -e "${BLUE}Prometheus:${NC}"
echo "  URL: http://localhost:9090"
echo ""
echo -e "${BLUE}Grafana:${NC}"
echo "  URL: http://localhost:3000"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose -f docker/docker-compose.yml logs -f"
echo "  - Stop services: docker-compose -f docker/docker-compose.yml down"
echo "  - Trigger DAG: docker-compose -f docker/docker-compose.yml exec airflow-webserver airflow dags trigger ml_pipeline_dag"
echo ""
