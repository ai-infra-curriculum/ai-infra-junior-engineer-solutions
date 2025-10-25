#!/bin/bash
# validate.sh - Validate Terraform configuration
# This script validates syntax and format of Terraform files

set -e

COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'
COLOR_YELLOW='\033[1;33m'
COLOR_NC='\033[0m'

echo -e "${COLOR_BLUE}========================================"
echo "Terraform Validation"
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

ERRORS=0

# Check 1: Terraform Format
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Checking Terraform formatting..."
if terraform fmt -check -recursive > /dev/null 2>&1; then
    echo -e "${COLOR_GREEN}✓${COLOR_NC} All files are properly formatted"
else
    echo -e "${COLOR_YELLOW}[!]${COLOR_NC} Some files need formatting. Run: terraform fmt -recursive"
    terraform fmt -check -recursive || true
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 2: Terraform Validation
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Validating Terraform configuration..."

# Initialize if needed
if [ ! -d ".terraform" ]; then
    echo -e "${COLOR_YELLOW}[*]${COLOR_NC} Initializing Terraform..."
    terraform init -backend=false > /dev/null
fi

if terraform validate; then
    echo -e "${COLOR_GREEN}✓${COLOR_NC} Configuration is valid"
else
    echo -e "${COLOR_RED}✗${COLOR_NC} Configuration validation failed"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 3: Variable validation
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Checking for undefined variables..."
if grep -r "var\." *.tf 2>/dev/null | grep -v "variables.tf" > /tmp/used_vars.txt; then
    UNDEFINED=0
    while IFS= read -r line; do
        VAR=$(echo "$line" | grep -o 'var\.[a-zA-Z0-9_]*' | sed 's/var\.//')
        if ! grep -q "variable \"$VAR\"" variables.tf 2>/dev/null; then
            if [ $UNDEFINED -eq 0 ]; then
                echo -e "${COLOR_RED}✗${COLOR_NC} Found undefined variables:"
            fi
            echo "  - $VAR"
            UNDEFINED=$((UNDEFINED + 1))
        fi
    done < /tmp/used_vars.txt

    if [ $UNDEFINED -eq 0 ]; then
        echo -e "${COLOR_GREEN}✓${COLOR_NC} All variables are defined"
    else
        ERRORS=$((ERRORS + 1))
    fi
    rm /tmp/used_vars.txt
else
    echo -e "${COLOR_GREEN}✓${COLOR_NC} No variables to check"
fi
echo ""

# Check 4: Security checks
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Running basic security checks..."
SECURITY_ISSUES=0

# Check for hardcoded secrets
if grep -r -i "password\s*=\s*\"" *.tf 2>/dev/null | grep -v "random_password" > /dev/null; then
    echo -e "${COLOR_RED}✗${COLOR_NC} Found potential hardcoded passwords"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

# Check for open security groups
if grep -r "0.0.0.0/0" *.tf 2>/dev/null | grep -v "#" > /dev/null; then
    echo -e "${COLOR_YELLOW}[!]${COLOR_NC} Found unrestricted access (0.0.0.0/0) - review for production use"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

# Check for disabled encryption
if grep -r "encrypt.*=.*false" *.tf 2>/dev/null | grep -v "#" > /dev/null; then
    echo -e "${COLOR_YELLOW}[!]${COLOR_NC} Found disabled encryption - review for security"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

if [ $SECURITY_ISSUES -eq 0 ]; then
    echo -e "${COLOR_GREEN}✓${COLOR_NC} No obvious security issues found"
else
    echo -e "${COLOR_YELLOW}[!]${COLOR_NC} Found $SECURITY_ISSUES potential security concerns (review recommended)"
fi
echo ""

# Check 5: Best practices
echo -e "${COLOR_GREEN}[+]${COLOR_NC} Checking best practices..."
WARNINGS=0

# Check for tags
if ! grep -q "tags.*=" *.tf 2>/dev/null; then
    echo -e "${COLOR_YELLOW}[!]${COLOR_NC} No tags found - consider adding tags for cost tracking"
    WARNINGS=$((WARNINGS + 1))
fi

# Check for outputs
if [ ! -f "outputs.tf" ]; then
    echo -e "${COLOR_YELLOW}[!]${COLOR_NC} No outputs.tf file found"
    WARNINGS=$((WARNINGS + 1))
fi

# Check for provider version constraints
if ! grep -q "required_version" *.tf 2>/dev/null; then
    echo -e "${COLOR_YELLOW}[!]${COLOR_NC} No Terraform version constraint found"
    WARNINGS=$((WARNINGS + 1))
fi

if [ $WARNINGS -eq 0 ]; then
    echo -e "${COLOR_GREEN}✓${COLOR_NC} Following Terraform best practices"
else
    echo -e "${COLOR_YELLOW}[!]${COLOR_NC} $WARNINGS best practice recommendations"
fi
echo ""

# Summary
echo -e "${COLOR_BLUE}========================================"
echo "Validation Summary"
echo -e "========================================${COLOR_NC}"

if [ $ERRORS -eq 0 ]; then
    echo -e "${COLOR_GREEN}✓ All checks passed!${COLOR_NC}"
    echo ""
    echo "Your Terraform configuration is ready to use."
    exit 0
else
    echo -e "${COLOR_RED}✗ Found $ERRORS error(s)${COLOR_NC}"
    echo ""
    echo "Please fix the errors above before proceeding."
    exit 1
fi
