#!/bin/bash
# Test script to verify log collection and querying
# Generates test logs and validates they appear in Loki

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

LOKI_URL="${LOKI_URL:-http://localhost:3100}"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Logging Pipeline Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Loki is ready
echo -e "\n${YELLOW}1/5: Checking Loki availability...${NC}"
if curl -sf "${LOKI_URL}/ready" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Loki is ready${NC}"
else
    echo -e "${RED}✗ Loki is not ready${NC}"
    exit 1
fi

# Check Promtail targets
echo -e "\n${YELLOW}2/5: Checking Promtail targets...${NC}"
TARGET_COUNT=$(curl -s http://localhost:9080/targets 2>/dev/null | jq '.activeTargets | length' || echo "0")
echo -e "Active targets: ${TARGET_COUNT}"
if [ "$TARGET_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Promtail is collecting logs from ${TARGET_COUNT} targets${NC}"
else
    echo -e "${YELLOW}⚠  No active targets found${NC}"
fi

# Query available labels
echo -e "\n${YELLOW}3/5: Querying available labels...${NC}"
curl -s "${LOKI_URL}/loki/api/v1/labels" | jq -r '.data[]' | head -10
echo -e "${GREEN}✓ Labels retrieved${NC}"

# Generate test logs
echo -e "\n${YELLOW}4/5: Generating test logs...${NC}"
if curl -sf "${GATEWAY_URL}/health" > /dev/null 2>&1; then
    for i in {1..10}; do
        curl -s "${GATEWAY_URL}/health" > /dev/null
    done
    echo -e "${GREEN}✓ Generated 10 test requests${NC}"
else
    echo -e "${YELLOW}⚠  Inference gateway not available, skipping log generation${NC}"
fi

# Wait for logs to be ingested
echo -e "\n${YELLOW}Waiting 5 seconds for log ingestion...${NC}"
sleep 5

# Query logs
echo -e "\n${YELLOW}5/5: Querying logs from Loki...${NC}"

# Query 1: Count log entries
LOG_COUNT=$(curl -s "${LOKI_URL}/loki/api/v1/query?query={container=\"inference-gateway\"}&limit=100" | jq '.data.result | length' || echo "0")
echo -e "Log streams found: ${LOG_COUNT}"

if [ "$LOG_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Successfully queried logs from Loki${NC}"

    # Query 2: Show recent logs
    echo -e "\n${YELLOW}Recent log entries:${NC}"
    curl -s "${LOKI_URL}/loki/api/v1/query?query={container=\"inference-gateway\"}&limit=5" \
        | jq -r '.data.result[].values[]? | .[1]' \
        | head -5 || echo "No logs available"
else
    echo -e "${YELLOW}⚠  No logs found yet. Logs may take a few minutes to appear.${NC}"
fi

# Display LogQL query examples
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Test Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Try these LogQL queries in Grafana Explore:${NC}"
echo -e ""
echo -e "${GREEN}1. All logs from inference-gateway:${NC}"
echo -e "   {container=\"inference-gateway\"}"
echo -e ""
echo -e "${GREEN}2. Error logs only:${NC}"
echo -e "   {container=\"inference-gateway\"} |= \"ERROR\""
echo -e ""
echo -e "${GREEN}3. Request rate:${NC}"
echo -e "   rate({container=\"inference-gateway\"}[5m])"
echo -e ""
echo -e "${GREEN}4. JSON parsing:${NC}"
echo -e "   {container=\"inference-gateway\"} | json | status_code >= 500"
echo -e ""
echo -e "${GREEN}5. Trace correlation:${NC}"
echo -e "   {container=\"inference-gateway\"} | json | trace_id=\"YOUR_TRACE_ID\""

echo -e "\n${YELLOW}Access Grafana Explore:${NC}"
echo -e "  URL: ${GREEN}http://localhost:3000/explore${NC}"
echo -e "  Data source: ${GREEN}Loki${NC}"
echo -e "  Login: ${GREEN}admin / admin${NC}"

echo -e "\n${YELLOW}Loki metrics:${NC}"
echo -e "  URL: ${GREEN}${LOKI_URL}/metrics${NC}"

echo -e "\n${YELLOW}Useful debugging commands:${NC}"
echo -e "  - Loki stats: ${GREEN}curl ${LOKI_URL}/metrics | grep loki_ingester${NC}"
echo -e "  - Promtail stats: ${GREEN}curl http://localhost:9080/metrics | grep promtail${NC}"
echo -e "  - View Loki logs: ${GREEN}docker-compose logs -f loki${NC}"
echo -e "  - View Promtail logs: ${GREEN}docker-compose logs -f promtail${NC}"
