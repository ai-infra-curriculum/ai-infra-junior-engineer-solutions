#!/bin/bash
# Setup script for Prometheus monitoring stack
# Creates directories, validates configuration, and prepares environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Prometheus Stack Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if running from correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found. Please run from project root.${NC}"
    exit 1
fi

# Create data directories
echo -e "\n${YELLOW}Creating data directories...${NC}"
mkdir -p data/prometheus
mkdir -p data/alertmanager
mkdir -p data/pushgateway

# Set correct permissions (Prometheus runs as nobody user, UID 65534)
echo -e "${YELLOW}Setting directory permissions...${NC}"
chmod 777 data/prometheus data/alertmanager data/pushgateway

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠  Please edit .env and add your credentials${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Validate Prometheus configuration
echo -e "\n${YELLOW}Validating Prometheus configuration...${NC}"
if command -v docker &> /dev/null; then
    docker run --rm \
        -v "$(pwd)/config/prometheus:/etc/prometheus" \
        prom/prometheus:v2.48.0 \
        promtool check config /etc/prometheus/prometheus.yml
    echo -e "${GREEN}✓ Prometheus configuration valid${NC}"

    # Check recording rules
    docker run --rm \
        -v "$(pwd)/config/prometheus:/etc/prometheus" \
        prom/prometheus:v2.48.0 \
        promtool check rules /etc/prometheus/recording_rules.yml
    echo -e "${GREEN}✓ Recording rules valid${NC}"

    # Check alerting rules
    docker run --rm \
        -v "$(pwd)/config/prometheus:/etc/prometheus" \
        prom/prometheus:v2.48.0 \
        promtool check rules /etc/prometheus/alerting_rules.yml
    echo -e "${GREEN}✓ Alerting rules valid${NC}"
else
    echo -e "${YELLOW}⚠  Docker not found, skipping configuration validation${NC}"
fi

# Validate Alertmanager configuration
echo -e "\n${YELLOW}Validating Alertmanager configuration...${NC}"
if command -v docker &> /dev/null; then
    docker run --rm \
        -v "$(pwd)/config/alertmanager:/etc/alertmanager" \
        prom/alertmanager:v0.26.0 \
        amtool check-config /etc/alertmanager/alertmanager.yml
    echo -e "${GREEN}✓ Alertmanager configuration valid${NC}"
else
    echo -e "${YELLOW}⚠  Docker not found, skipping Alertmanager validation${NC}"
fi

# Build custom exporter
echo -e "\n${YELLOW}Building ML Model Exporter...${NC}"
if command -v docker &> /dev/null; then
    docker build -t ml-model-exporter:latest ./exporters/ml-model-exporter
    echo -e "${GREEN}✓ ML Model Exporter built${NC}"
else
    echo -e "${YELLOW}⚠  Docker not found, skipping exporter build${NC}"
fi

# Check if inference-gateway image exists
echo -e "\n${YELLOW}Checking for inference-gateway image...${NC}"
if docker images | grep -q "inference-gateway"; then
    echo -e "${GREEN}✓ inference-gateway image found${NC}"
else
    echo -e "${YELLOW}⚠  inference-gateway image not found${NC}"
    echo -e "${YELLOW}   Please build it from Exercise 01 before starting the stack${NC}"
fi

# Display next steps
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Edit .env file and add your credentials (Slack, PagerDuty, SMTP)"
echo -e "2. Build inference-gateway image from Exercise 01 (if not already done)"
echo -e "3. Start the stack: ${GREEN}docker-compose up -d${NC}"
echo -e "4. Check logs: ${GREEN}docker-compose logs -f${NC}"
echo -e "5. Access services:"
echo -e "   - Prometheus: ${GREEN}http://localhost:9090${NC}"
echo -e "   - Alertmanager: ${GREEN}http://localhost:9093${NC}"
echo -e "   - cAdvisor: ${GREEN}http://localhost:8080${NC}"
echo -e "   - Inference Gateway: ${GREEN}http://localhost:8000${NC}"
echo -e "   - ML Model Exporter: ${GREEN}http://localhost:9101${NC}"

echo -e "\n${YELLOW}Useful commands:${NC}"
echo -e "- View all metrics targets: ${GREEN}curl http://localhost:9090/api/v1/targets | jq${NC}"
echo -e "- View all alerts: ${GREEN}curl http://localhost:9090/api/v1/alerts | jq${NC}"
echo -e "- Test Prometheus query: ${GREEN}curl 'http://localhost:9090/api/v1/query?query=up' | jq${NC}"
echo -e "- Reload Prometheus config: ${GREEN}curl -X POST http://localhost:9090/-/reload${NC}"
