#!/bin/bash

# Test script for Airflow Workflow Monitoring Exercise
# This script runs all tests

set -e  # Exit on error

echo "=========================================="
echo "Airflow Workflow Monitoring - Tests"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if services are running
echo "Checking if Airflow is running..."
cd docker
if ! docker-compose ps airflow-webserver | grep -q "Up"; then
    echo -e "${RED}Error: Airflow is not running${NC}"
    echo "Please start services first: ./scripts/start.sh"
    exit 1
fi
cd ..

echo -e "${GREEN}Airflow is running${NC}"
echo ""

# Run DAG integrity tests
echo "=========================================="
echo "Running DAG Integrity Tests"
echo "=========================================="
echo ""

docker-compose -f docker/docker-compose.yml exec -T airflow-webserver \
    pytest /opt/airflow/tests/test_dags.py -v --tb=short

echo ""

# Run task tests
echo "=========================================="
echo "Running Task Tests"
echo "=========================================="
echo ""

docker-compose -f docker/docker-compose.yml exec -T airflow-webserver \
    pytest /opt/airflow/tests/test_tasks.py -v --tb=short

echo ""

# Test DAG imports
echo "=========================================="
echo "Testing DAG Imports"
echo "=========================================="
echo ""

echo "Testing ml_pipeline_dag.py..."
docker-compose -f docker/docker-compose.yml exec -T airflow-webserver \
    python /opt/airflow/dags/ml_pipeline_dag.py

echo -e "${GREEN}ml_pipeline_dag.py: OK${NC}"
echo ""

echo "Testing monitoring_dag.py..."
docker-compose -f docker/docker-compose.yml exec -T airflow-webserver \
    python /opt/airflow/dags/monitoring_dag.py

echo -e "${GREEN}monitoring_dag.py: OK${NC}"
echo ""

# List DAGs
echo "=========================================="
echo "Listing DAGs"
echo "=========================================="
echo ""

docker-compose -f docker/docker-compose.yml exec -T airflow-webserver \
    airflow dags list

echo ""

# Check for DAG import errors
echo "=========================================="
echo "Checking for Import Errors"
echo "=========================================="
echo ""

IMPORT_ERRORS=$(docker-compose -f docker/docker-compose.yml exec -T airflow-webserver \
    airflow dags list-import-errors 2>/dev/null || echo "")

if [ -z "$IMPORT_ERRORS" ]; then
    echo -e "${GREEN}No import errors found${NC}"
else
    echo -e "${YELLOW}Import errors:${NC}"
    echo "$IMPORT_ERRORS"
fi

echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}All Tests Completed${NC}"
echo "=========================================="
echo ""
echo "Test Summary:"
echo "  - DAG integrity tests: PASSED"
echo "  - Task unit tests: PASSED"
echo "  - DAG imports: PASSED"
echo ""
echo "You can now:"
echo "  1. Access Airflow UI: http://localhost:8080"
echo "  2. Trigger DAGs manually"
echo "  3. View metrics in Prometheus: http://localhost:9090"
echo "  4. Create dashboards in Grafana: http://localhost:3000"
echo ""
