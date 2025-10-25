#!/bin/bash
# Setup script for Grafana dashboards
# Creates directories, validates configuration, generates dashboards

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Grafana Dashboard Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if running from correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found. Please run from project root.${NC}"
    exit 1
fi

# Create data directories
echo -e "\n${YELLOW}Creating data directories...${NC}"
mkdir -p data/{grafana,grafana-logs,prometheus,loki}

# Set correct permissions
echo -e "${YELLOW}Setting directory permissions...${NC}"
chmod 777 data/*

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠  Please edit .env and update credentials${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Generate dashboards
echo -e "\n${YELLOW}Generating dashboard JSON files...${NC}"
if command -v python3 &> /dev/null; then
    python3 scripts/generate-dashboards.py
    echo -e "${GREEN}✓ Dashboards generated${NC}"
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo -e "${YELLOW}  Dashboards already exist in config/dashboards/${NC}"
fi

# Validate YAML files
echo -e "\n${YELLOW}Validating configuration files...${NC}"
if command -v yamllint &> /dev/null; then
    yamllint config/grafana/provisioning/ || echo -e "${YELLOW}⚠  YAML validation warnings (non-critical)${NC}"
else
    echo -e "${YELLOW}⚠  yamllint not found, skipping validation${NC}"
fi

# Display next steps
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Edit .env file with your credentials"
echo -e "2. Start the stack: ${GREEN}docker-compose up -d${NC}"
echo -e "3. Wait 30 seconds for Grafana to initialize"
echo -e "4. Access Grafana: ${GREEN}http://localhost:3000${NC}"
echo -e "5. Login with: ${GREEN}admin / admin${NC} (change password on first login)"
echo -e "6. Navigate to Dashboards → Browse"

echo -e "\n${YELLOW}Available dashboards:${NC}"
echo -e "  - ${GREEN}ML Platform/${NC} SLO Overview"
echo -e "  - ${GREEN}ML Platform/${NC} Application Performance"
echo -e "  - ${GREEN}Infrastructure/${NC} Infrastructure Health"

echo -e "\n${YELLOW}Data sources configured:${NC}"
echo -e "  - ${GREEN}Prometheus${NC} (http://prometheus:9090)"
echo -e "  - ${GREEN}Loki${NC} (http://loki:3100)"
echo -e "  - ${GREEN}Jaeger${NC} (http://jaeger:16686)"

echo -e "\n${YELLOW}Useful commands:${NC}"
echo -e "  - View logs: ${GREEN}docker-compose logs -f grafana${NC}"
echo -e "  - Restart Grafana: ${GREEN}docker-compose restart grafana${NC}"
echo -e "  - Check health: ${GREEN}curl http://localhost:3000/api/health | jq${NC}"
