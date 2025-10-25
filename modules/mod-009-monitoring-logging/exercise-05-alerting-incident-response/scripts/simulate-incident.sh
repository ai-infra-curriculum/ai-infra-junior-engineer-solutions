#!/bin/bash
# Incident Simulation Script
# Purpose: Trigger realistic incidents for runbook practice and testing

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Incident Simulation Suite${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Available Incident Scenarios:${NC}"
echo "1. High Error Rate (5xx errors)"
echo "2. Latency Spike (slow responses)"
echo "3. Traffic Surge (load spike)"
echo "4. Resource Exhaustion (CPU/Memory)"
echo "5. Complete Service Outage"
echo "6. SLO Fast Burn (error budget consumption)"
echo "0. Exit"

read -p "Select scenario [0-6]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}🔥 Simulating: High Error Rate${NC}"
        echo "This will generate 5xx errors to trigger HighErrorRate alert"
        echo "Expected alert: HighErrorRate (>5% error rate)"
        echo "Runbook: runbooks/001-high-error-rate.md"

        echo -e "\n${YELLOW}Generating errors...${NC}"
        for i in {1..100}; do
            # Hit non-existent endpoint to generate 404 (or 500 if configured)
            curl -sf "${GATEWAY_URL}/trigger-error" > /dev/null 2>&1 || true

            # Also send malformed requests
            curl -X POST -H "Content-Type: application/json" \
                -d '{"invalid": "data"}' \
                "${GATEWAY_URL}/predict" > /dev/null 2>&1 || true

            # Small delay
            sleep 0.1
        done

        echo -e "${GREEN}✓ Generated 200 error-inducing requests${NC}"
        echo -e "\n${YELLOW}Check alert status:${NC}"
        echo "  Prometheus: ${PROMETHEUS_URL}/alerts"
        echo "  Dashboard: http://localhost:3000/d/app-performance"
        echo -e "\n${YELLOW}Expected alert in 2-5 minutes${NC}"
        ;;

    2)
        echo -e "\n${YELLOW}⏱  Simulating: Latency Spike${NC}"
        echo "This will send slow requests to trigger latency alerts"
        echo "Expected alert: SLOLatencyP99Violation (P99 > 300ms)"

        echo -e "\n${YELLOW}Generating slow requests...${NC}"
        for i in {1..50}; do
            # Send requests with large images or complex payloads
            # This simulates slow model inference
            timeout 10 curl -X POST \
                -F "file=@/dev/urandom" \
                "${GATEWAY_URL}/predict" > /dev/null 2>&1 || true

            sleep 0.2
        done

        echo -e "${GREEN}✓ Generated 50 slow requests${NC}"
        echo -e "\n${YELLOW}Check metrics:${NC}"
        echo "  Query: slo:http_request_duration:p99:rate5m"
        echo "  Dashboard: http://localhost:3000/d/slo-overview"
        ;;

    3)
        echo -e "\n${YELLOW}📈 Simulating: Traffic Surge${NC}"
        echo "This will generate high request volume"
        echo "Expected: Possible saturation alerts (CPU/Memory)"

        echo -e "\n${YELLOW}Generating traffic burst...${NC}"
        echo "Sending 1000 requests as fast as possible..."

        # Parallel requests using background processes
        for i in {1..1000}; do
            curl -sf "${GATEWAY_URL}/health" > /dev/null 2>&1 &

            # Limit parallelism
            if (( i % 100 == 0 )); then
                wait
                echo "  Sent ${i}/1000..."
            fi
        done
        wait

        echo -e "${GREEN}✓ Generated 1000 requests in burst${NC}"
        echo -e "\n${YELLOW}Check resource usage:${NC}"
        echo "  Query: container:cpu_usage:percent"
        echo "  Query: container:memory_usage:percent"
        ;;

    4)
        echo -e "\n${YELLOW}💾 Simulating: Resource Exhaustion${NC}"
        echo "This will stress CPU/Memory to trigger resource alerts"
        echo "Expected alerts: HighCPUUsage, HighMemoryUsage"

        echo -e "\n${RED}WARNING: This will actually stress your system!${NC}"
        read -p "Continue? [y/N]: " confirm

        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo -e "\n${YELLOW}Generating CPU load...${NC}"

            # CPU stress (5 seconds)
            docker-compose exec -T inference-gateway bash -c \
                "for i in {1..4}; do dd if=/dev/zero of=/dev/null & done; sleep 5; killall dd" \
                2>/dev/null || echo "Stress generated"

            echo -e "${GREEN}✓ CPU stress applied${NC}"
            echo -e "\n${YELLOW}Check alerts in 2-3 minutes${NC}"
        else
            echo "Skipped"
        fi
        ;;

    5)
        echo -e "\n${YELLOW}❌ Simulating: Complete Service Outage${NC}"
        echo "This will stop the inference-gateway service"
        echo "Expected alert: ServiceDown"

        echo -e "\n${RED}WARNING: This will bring down the service!${NC}"
        read -p "Continue? [y/N]: " confirm

        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo -e "\n${YELLOW}Stopping service...${NC}"
            docker-compose stop inference-gateway

            echo -e "${GREEN}✓ Service stopped${NC}"
            echo -e "\n${YELLOW}Expected alert: ServiceDown (within 1 minute)${NC}"
            echo -e "\n${YELLOW}To restore service:${NC}"
            echo "  docker-compose start inference-gateway"
        else
            echo "Skipped"
        fi
        ;;

    6)
        echo -e "\n${YELLOW}🔥 Simulating: SLO Fast Burn${NC}"
        echo "This will generate sustained error rate to burn error budget fast"
        echo "Expected alert: SLOAvailabilityFastBurn"
        echo "Runbook: runbooks/002-slo-burn-rate.md"

        echo -e "\n${YELLOW}Generating sustained error rate...${NC}"
        echo "This will run for 10 minutes to trigger multi-window alert"

        START_TIME=$(date +%s)
        DURATION=600  # 10 minutes

        echo "Started at $(date)"
        echo "Will run until $(date -d @$((START_TIME + DURATION)))"

        while [ $(($(date +%s) - START_TIME)) -lt $DURATION ]; do
            # Generate errors at ~10/second
            for i in {1..10}; do
                curl -sf "${GATEWAY_URL}/trigger-error" > /dev/null 2>&1 || true &
            done
            sleep 1

            # Progress indicator every minute
            ELAPSED=$(($(date +%s) - START_TIME))
            if (( ELAPSED % 60 == 0 )); then
                echo "  ${ELAPSED}/600 seconds elapsed..."
            fi
        done
        wait

        echo -e "${GREEN}✓ Sustained error generation complete${NC}"
        echo -e "\n${YELLOW}Check burn rate:${NC}"
        echo "  Query: slo:availability:burn_rate:1h"
        echo "  Query: slo:availability:burn_rate:6h"
        echo "  Dashboard: http://localhost:3000/d/slo-overview"
        echo -e "\n${YELLOW}Alert should fire when both 1h and 6h burn rate > 14.4${NC}"
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

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Simulation Complete${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Monitor alerts: ${PROMETHEUS_URL}/alerts"
echo "2. Check dashboards: http://localhost:3000"
echo "3. Follow runbook procedures"
echo "4. Practice incident response workflow"
echo "5. Document actions in incident template"

echo -e "\n${YELLOW}View fired alerts:${NC}"
echo "  curl ${PROMETHEUS_URL}/api/v1/alerts | jq '.data.alerts[] | select(.state==\"firing\")'"

echo -e "\n${YELLOW}Cleanup (if needed):${NC}"
echo "  # Restart service: docker-compose restart inference-gateway"
echo "  # Clear traffic: (wait a few minutes)"
echo "  # Silence alert: http://localhost:9093 (Alertmanager)"
