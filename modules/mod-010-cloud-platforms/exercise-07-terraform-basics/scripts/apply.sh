#!/bin/bash
# apply.sh - Apply Terraform configuration
# This script applies the Terraform configuration to create/update infrastructure

set -e

COLOR_GREEN='\033[0;32m'
COLOR_BLUE='\033[0;34m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_NC='\033[0m'

echo -e "${COLOR_BLUE}========================================"
echo "Terraform Apply"
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

# Check if Terraform is initialized
if [ ! -d ".terraform" ]; then
    echo -e "${COLOR_YELLOW}[*]${COLOR_NC} Terraform not initialized. Running init..."
    terraform init
    echo ""
fi

# Parse command line arguments
AUTO_APPROVE=false
PLAN_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes|--auto-approve)
            AUTO_APPROVE=true
            shift
            ;;
        -p|--plan)
            PLAN_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-y|--yes] [-p|--plan PLAN_FILE]"
            exit 1
            ;;
    esac
done

# Validate configuration first
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Validating configuration..."
if terraform validate; then
    echo -e "${COLOR_GREEN}✓${COLOR_NC} Configuration is valid"
else
    echo -e "${COLOR_RED}✗${COLOR_NC} Configuration validation failed"
    exit 1
fi
echo ""

# Show plan first if not using a saved plan
if [ -z "$PLAN_FILE" ]; then
    echo -e "${COLOR_GREEN}[+]${COLOR_NC} Showing execution plan..."
    echo ""
    terraform plan
    echo ""

    # Ask for confirmation unless auto-approved
    if [ "$AUTO_APPROVE" = false ]; then
        echo -e "${COLOR_YELLOW}[!]${COLOR_NC} This will create/modify real AWS resources."
        echo -e "${COLOR_YELLOW}[!]${COLOR_NC} Review the plan above carefully."
        echo ""
        read -p "Do you want to apply these changes? (yes/no): " -r
        echo ""

        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            echo "Apply cancelled."
            exit 0
        fi
    fi
fi

# Apply the configuration
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Applying Terraform configuration..."
echo ""

if [ -n "$PLAN_FILE" ]; then
    terraform apply "$PLAN_FILE"
elif [ "$AUTO_APPROVE" = true ]; then
    terraform apply -auto-approve
else
    terraform apply
fi

echo ""
echo -e "${COLOR_GREEN}✓${COLOR_NC} Infrastructure deployed successfully!"
echo ""
echo "To view the outputs:"
echo "  terraform output"
echo ""
echo "To connect to your instance:"
echo "  terraform output -raw ssh_connection_command"
echo ""
echo "To destroy the infrastructure:"
echo "  ./scripts/destroy.sh"
echo ""
