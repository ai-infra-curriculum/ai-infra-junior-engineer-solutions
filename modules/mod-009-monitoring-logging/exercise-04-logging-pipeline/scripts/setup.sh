#!/bin/bash
# Setup script for centralized logging pipeline
# Creates directories, validates configuration, prepares environment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Logging Pipeline Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if running from correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found. Please run from project root.${NC}"
    exit 1
fi

# Create data directories
echo -e "\n${YELLOW}Creating data directories...${NC}"
mkdir -p data/{loki,promtail-positions,grafana,prometheus}

# Loki subdirectories
mkdir -p data/loki/{chunks,boltdb-shipper-active,boltdb-shipper-cache,compactor,wal,rules,rules-temp}

# Set correct permissions
echo -e "${YELLOW}Setting directory permissions...${NC}"
chmod 777 data/loki data/loki/*
chmod 777 data/promtail-positions
chmod 777 data/grafana
chmod 777 data/prometheus

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Validate Loki configuration
echo -e "\n${YELLOW}Validating Loki configuration...${NC}"
if command -v docker &> /dev/null; then
    docker run --rm \
        -v "$(pwd)/config/loki:/etc/loki" \
        grafana/loki:2.9.3 \
        -config.file=/etc/loki/loki-config.yaml \
        -verify-config || {
            echo -e "${RED}✗ Loki configuration validation failed${NC}"
            exit 1
        }
    echo -e "${GREEN}✓ Loki configuration valid${NC}"
else
    echo -e "${YELLOW}⚠  Docker not found, skipping validation${NC}"
fi

# Validate Promtail configuration
echo -e "\n${YELLOW}Validating Promtail configuration...${NC}"
if command -v docker &> /dev/null; then
    docker run --rm \
        -v "$(pwd)/config/promtail:/etc/promtail" \
        grafana/promtail:2.9.3 \
        -config.file=/etc/promtail/promtail-config.yaml \
        -dry-run || {
            echo -e "${RED}✗ Promtail configuration validation failed${NC}"
            exit 1
        }
    echo -e "${GREEN}✓ Promtail configuration valid${NC}"
else
    echo -e "${YELLOW}⚠  Docker not found, skipping validation${NC}"
fi

# Check if inference-gateway image exists
echo -e "\n${YELLOW}Checking for inference-gateway image...${NC}"
if docker images | grep -q "inference-gateway"; then
    echo -e "${GREEN}✓ inference-gateway image found${NC}"
else
    echo -e "${YELLOW}⚠  inference-gateway image not found${NC}"
    echo -e "${YELLOW}   Build it from Exercise 01 for full stack functionality${NC}"
fi

# Display next steps
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Start the stack: ${GREEN}docker-compose up -d${NC}"
echo -e "2. Wait 30 seconds for services to initialize"
echo -e "3. Check logs: ${GREEN}docker-compose logs -f loki promtail${NC}"
echo -e "4. Access services:"
echo -e "   - Loki API: ${GREEN}http://localhost:3100/ready${NC}"
echo -e "   - Grafana: ${GREEN}http://localhost:3000${NC}"
echo -e "   - Promtail metrics: ${GREEN}http://localhost:9080/metrics${NC}"

echo -e "\n${YELLOW}Verify log collection:${NC}"
echo -e "1. Check Promtail targets: ${GREEN}curl http://localhost:9080/targets | jq${NC}"
echo -e "2. Query Loki labels: ${GREEN}curl http://localhost:3100/loki/api/v1/labels | jq${NC}"
echo -e "3. Query logs: ${GREEN}curl 'http://localhost:3100/loki/api/v1/query?query={container=\"loki\"}' | jq${NC}"

echo -e "\n${YELLOW}In Grafana (http://localhost:3000):${NC}"
echo -e "1. Login: ${GREEN}admin / admin${NC}"
echo -e "2. Navigate to: ${GREEN}Explore${NC}"
echo -e "3. Select data source: ${GREEN}Loki${NC}"
echo -e "4. Try query: ${GREEN}{container=\"inference-gateway\"}${NC}"

echo -e "\n${YELLOW}Useful commands:${NC}"
echo -e "  - View all containers: ${GREEN}docker-compose ps${NC}"
echo -e "  - View Loki logs: ${GREEN}docker-compose logs -f loki${NC}"
echo -e "  - View Promtail logs: ${GREEN}docker-compose logs -f promtail${NC}"
echo -e "  - Check Loki stats: ${GREEN}curl http://localhost:3100/metrics | grep loki${NC}"
echo -e "  - Test log ingestion: ${GREEN}./scripts/test-logging.sh${NC}"
