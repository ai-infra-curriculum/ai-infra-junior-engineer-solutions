#!/bin/bash
# Test script to trigger various alert conditions
# Useful for validating alert rules and notification routing

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"
ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://localhost:9093}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Prometheus Alerting Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to check if service is up
check_service() {
    local url=$1
    local name=$2
    if curl -sf "${url}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ ${name} is up${NC}"
        return 0
    else
        echo -e "${RED}✗ ${name} is down${NC}"
        return 1
    fi
}

# Function to view current alerts
view_alerts() {
    echo -e "\n${YELLOW}Current Active Alerts:${NC}"
    curl -s "${PROMETHEUS_URL}/api/v1/alerts" | jq -r '.data.alerts[] | select(.state=="firing") | "\(.labels.alertname) - \(.labels.severity)"' || echo "No active alerts"
}

# Function to generate high load
generate_load() {
    local duration=$1
    echo -e "\n${YELLOW}Generating high load for ${duration}s to trigger alerts...${NC}"

    local end=$((SECONDS + duration))
    while [ $SECONDS -lt $end ]; do
        # Send requests with some failures
        for i in {1..10}; do
            curl -sf "${GATEWAY_URL}/health" > /dev/null 2>&1 || true
        done
        sleep 0.1
    done

    echo -e "${GREEN}✓ Load generation complete${NC}"
}

# Function to simulate errors
simulate_errors() {
    local count=$1
    echo -e "\n${YELLOW}Simulating ${count} errors...${NC}"

    for i in $(seq 1 $count); do
        # Try to hit non-existent endpoint to generate 404s
        curl -sf "${GATEWAY_URL}/nonexistent-endpoint" > /dev/null 2>&1 || true
    done

    echo -e "${GREEN}✓ Error simulation complete${NC}"
}

# Main menu
echo -e "\n${YELLOW}Select test to run:${NC}"
echo "1. View current alerts"
echo "2. Check all services status"
echo "3. Generate high load (triggers CPU/latency alerts)"
echo "4. Simulate error spike (triggers error rate alerts)"
echo "5. View SLO metrics"
echo "6. View Alertmanager status"
echo "7. Run full test suite"
echo "0. Exit"

read -p "Enter choice [0-7]: " choice

case $choice in
    1)
        view_alerts
        ;;
    2)
        echo -e "\n${YELLOW}Checking services...${NC}"
        check_service "${PROMETHEUS_URL}/-/healthy" "Prometheus"
        check_service "${ALERTMANAGER_URL}/-/healthy" "Alertmanager"
        check_service "${GATEWAY_URL}/health" "Inference Gateway"
        check_service "http://localhost:9100/metrics" "Node Exporter"
        check_service "http://localhost:9101/" "ML Model Exporter"
        ;;
    3)
        generate_load 60
        echo -e "\n${YELLOW}Wait 2 minutes for alerts to fire, then check:${NC}"
        echo -e "  Prometheus Alerts: ${GREEN}${PROMETHEUS_URL}/alerts${NC}"
        echo -e "  Alertmanager: ${GREEN}${ALERTMANAGER_URL}${NC}"
        ;;
    4)
        simulate_errors 100
        echo -e "\n${YELLOW}Wait 2 minutes for alerts to fire${NC}"
        ;;
    5)
        echo -e "\n${YELLOW}SLO Availability (30d):${NC}"
        curl -s "${PROMETHEUS_URL}/api/v1/query?query=slo:availability:ratio_rate30d" | jq -r '.data.result[0].value[1]' || echo "No data"

        echo -e "\n${YELLOW}SLO Latency P99 (5m):${NC}"
        curl -s "${PROMETHEUS_URL}/api/v1/query?query=slo:http_request_duration:p99:rate5m" | jq -r '.data.result[0].value[1]' || echo "No data"

        echo -e "\n${YELLOW}Error Budget Remaining:${NC}"
        curl -s "${PROMETHEUS_URL}/api/v1/query?query=slo:availability:error_budget_remaining" | jq -r '.data.result[0].value[1]' || echo "No data"
        ;;
    6)
        echo -e "\n${YELLOW}Alertmanager Status:${NC}"
        curl -s "${ALERTMANAGER_URL}/api/v2/status" | jq .

        echo -e "\n${YELLOW}Active Alerts in Alertmanager:${NC}"
        curl -s "${ALERTMANAGER_URL}/api/v2/alerts" | jq 'length'
        ;;
    7)
        echo -e "\n${YELLOW}Running full test suite...${NC}"

        # Check services
        echo -e "\n${YELLOW}1/4: Checking services${NC}"
        check_service "${PROMETHEUS_URL}/-/healthy" "Prometheus"
        check_service "${ALERTMANAGER_URL}/-/healthy" "Alertmanager"
        check_service "${GATEWAY_URL}/health" "Inference Gateway"

        # View current state
        echo -e "\n${YELLOW}2/4: Current alerts${NC}"
        view_alerts

        # Check SLO metrics
        echo -e "\n${YELLOW}3/4: SLO Metrics${NC}"
        echo -n "Availability: "
        curl -s "${PROMETHEUS_URL}/api/v1/query?query=slo:availability:ratio_rate5m" | jq -r '.data.result[0].value[1] // "N/A"'
        echo -n "P99 Latency: "
        curl -s "${PROMETHEUS_URL}/api/v1/query?query=slo:http_request_duration:p99:rate5m" | jq -r '.data.result[0].value[1] // "N/A"'

        # Generate some load
        echo -e "\n${YELLOW}4/4: Generating test load${NC}"
        generate_load 30

        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}Test suite complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo -e "\nView results:"
        echo -e "  Prometheus: ${GREEN}${PROMETHEUS_URL}${NC}"
        echo -e "  Alertmanager: ${GREEN}${ALERTMANAGER_URL}${NC}"
        ;;
    0)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
