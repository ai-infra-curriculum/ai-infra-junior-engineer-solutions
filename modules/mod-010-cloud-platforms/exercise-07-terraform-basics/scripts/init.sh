#!/bin/bash
# init.sh - Initialize Terraform working directory
# This script initializes Terraform and downloads required providers

set -e

COLOR_GREEN='\033[0;32m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m'

echo -e "${COLOR_BLUE}========================================"
echo "Terraform Initialization"
echo -e "========================================${COLOR_NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "main.tf" ] && [ ! -f "terraform/main.tf" ]; then
    echo "Error: main.tf not found. Run this script from the project root or terraform directory."
    exit 1
fi

# Navigate to terraform directory if needed
if [ -d "terraform" ] && [ ! -f "main.tf" ]; then
    cd terraform
fi

echo -e "${COLOR_GREEN}[+]${COLOR_NC} Initializing Terraform..."
terraform init -upgrade

echo ""
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Validating configuration..."
terraform validate

echo ""
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Formatting Terraform files..."
terraform fmt -recursive

echo ""
echo -e "${COLOR_GREEN}✓${COLOR_NC} Terraform initialized successfully!"
echo ""
echo "Next steps:"
echo "  - Review configuration: terraform plan"
echo "  - Apply changes: terraform apply"
echo "  - Or use: ./scripts/plan.sh and ./scripts/apply.sh"
echo ""
