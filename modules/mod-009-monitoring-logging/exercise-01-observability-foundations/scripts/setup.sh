#!/bin/bash
set -e

echo "=========================================="
echo "Inference Gateway - Setup"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ Docker Compose not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose installed${NC}"

# Create .env from template if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo -e "${YELLOW}⚠ Review and customize .env file before starting${NC}"
else
    echo -e "${YELLOW}.env file already exists, skipping${NC}"
fi

# Pull Docker images
echo ""
echo "Pulling Docker images..."
docker-compose pull prometheus jaeger

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review .env configuration"
echo "2. Start services:"
echo "   docker-compose up -d"
echo ""
echo "3. Access UIs:"
echo "   Application:  http://localhost:8000"
echo "   API Docs:     http://localhost:8000/docs"
echo "   Prometheus:   http://localhost:9090"
echo "   Jaeger:       http://localhost:16686"
echo ""
echo "4. Test the service:"
echo "   curl http://localhost:8000/health"
echo "   curl http://localhost:8000/ready"
echo "   ./scripts/load_test.sh"
echo ""
