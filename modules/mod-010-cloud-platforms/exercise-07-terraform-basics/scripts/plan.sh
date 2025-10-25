#!/bin/bash
# plan.sh - Show Terraform execution plan
# This script shows what changes Terraform will make without applying them

set -e

COLOR_GREEN='\033[0;32m'
COLOR_BLUE='\033[0;34m'
COLOR_YELLOW='\033[1;33m'
COLOR_NC='\033[0m'

echo -e "${COLOR_BLUE}========================================"
echo "Terraform Plan"
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
OUTPUT_FILE=""
DETAILED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--out)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -d|--detailed)
            DETAILED=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-o|--out OUTPUT_FILE] [-d|--detailed]"
            exit 1
            ;;
    esac
done

# Run terraform plan
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Generating execution plan..."
echo ""

if [ -n "$OUTPUT_FILE" ]; then
    echo -e "${COLOR_GREEN}[+]${COLOR_NC} Saving plan to: $OUTPUT_FILE"
    terraform plan -out="$OUTPUT_FILE"
else
    if [ "$DETAILED" = true ]; then
        terraform plan -detailed-exitcode || true
    else
        terraform plan
    fi
fi

echo ""
echo -e "${COLOR_GREEN}✓${COLOR_NC} Plan generated successfully!"
echo ""
echo "Review the plan above carefully before applying."
echo ""
echo "To apply this plan:"
if [ -n "$OUTPUT_FILE" ]; then
    echo "  terraform apply $OUTPUT_FILE"
else
    echo "  terraform apply"
    echo "  or: ./scripts/apply.sh"
fi
echo ""
