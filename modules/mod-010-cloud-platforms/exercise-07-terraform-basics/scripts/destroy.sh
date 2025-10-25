#!/bin/bash
# destroy.sh - Destroy Terraform-managed infrastructure
# WARNING: This will delete all resources created by Terraform

set -e

COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m'

echo -e "${COLOR_RED}========================================"
echo "Terraform Destroy"
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
    echo -e "${COLOR_RED}[!]${COLOR_NC} Terraform not initialized. Nothing to destroy."
    exit 1
fi

# Parse command line arguments
AUTO_APPROVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes|--auto-approve)
            AUTO_APPROVE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-y|--yes]"
            exit 1
            ;;
    esac
done

# Show current state
echo -e "${COLOR_BLUE}[i]${COLOR_NC} Current infrastructure:"
echo ""
terraform state list || echo "No resources found"
echo ""

# Warning message
echo -e "${COLOR_RED}[!] WARNING [!]${COLOR_NC}"
echo -e "${COLOR_RED}This will destroy ALL resources created by Terraform!${COLOR_NC}"
echo ""
echo "Resources to be destroyed:"
terraform show -no-color | head -n 20
echo ""

# Ask for confirmation unless auto-approved
if [ "$AUTO_APPROVE" = false ]; then
    echo -e "${COLOR_YELLOW}Are you absolutely sure you want to destroy all resources?${COLOR_NC}"
    read -p "Type 'destroy' to confirm: " -r
    echo ""

    if [[ ! $REPLY == "destroy" ]]; then
        echo "Destroy cancelled."
        exit 0
    fi

    # Double confirmation for safety
    echo -e "${COLOR_RED}Last chance!${COLOR_NC} This action cannot be undone."
    read -p "Type 'yes' to proceed: " -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Destroy cancelled."
        exit 0
    fi
fi

# Destroy the infrastructure
echo -e "${COLOR_RED}[+]${COLOR_NC} Destroying infrastructure..."
echo ""

if [ "$AUTO_APPROVE" = true ]; then
    terraform destroy -auto-approve
else
    terraform destroy
fi

echo ""
echo -e "${COLOR_GREEN}✓${COLOR_NC} Infrastructure destroyed successfully!"
echo ""
echo "All AWS resources have been removed."
echo "Terraform state files are preserved for reference."
echo ""
