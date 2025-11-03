#!/bin/bash
#
# audit_permissions.sh - Security audit for file permissions
#
# Usage: ./audit_permissions.sh PROJECT_PATH
#
# Audits for:
# - World-writable files and directories (security risk)
# - Overly permissive files (777)
# - Sensitive files with incorrect permissions
# - SUID/SGID files
# - Files readable by others that should be private
#

set -e
set -u

VERSION="1.0.0"
SCRIPT_NAME=$(basename "$0")

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' NC=''
fi

usage() {
    cat << EOF
Usage: $SCRIPT_NAME PROJECT_PATH

Audit file permissions for security issues.

Arguments:
    PROJECT_PATH    Path to project directory to audit

Options:
    -h, --help      Show this help message
    -v, --version   Show version

Example:
    $SCRIPT_NAME my-ml-project
    $SCRIPT_NAME /path/to/project

EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage
[[ "${1:-}" == "-v" || "${1:-}" == "--version" ]] && echo "$VERSION" && exit 0

PROJECT_ROOT="${1:-.}"

[[ ! -d "$PROJECT_ROOT" ]] && echo "Error: Directory not found: $PROJECT_ROOT" && exit 1

PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)

echo "=========================================="
echo " Permission Security Audit"
echo "=========================================="
echo ""
echo "Project: $(basename "$PROJECT_ROOT")"
echo "Path: $PROJECT_ROOT"
echo "Time: $(date)"
echo ""

ISSUES=0
WARNINGS=0
PASSED=0

# Check world-writable files
echo -e "${BLUE}[1] Checking for world-writable files...${NC}"
WW_FILES=$(find "$PROJECT_ROOT" -type f -perm -002 2>/dev/null | wc -l)
if [[ $WW_FILES -gt 0 ]]; then
    echo -e "${RED}✗ Found $WW_FILES world-writable files (CRITICAL):${NC}"
    find "$PROJECT_ROOT" -type f -perm -002 -ls 2>/dev/null | head -10
    ((ISSUES+=$WW_FILES))
else
    echo -e "${GREEN}✓ No world-writable files${NC}"
    ((PASSED++))
fi
echo ""

# Check world-writable directories
echo -e "${BLUE}[2] Checking for world-writable directories...${NC}"
WW_DIRS=$(find "$PROJECT_ROOT" -type d -perm -002 2>/dev/null | wc -l)
if [[ $WW_DIRS -gt 0 ]]; then
    echo -e "${RED}✗ Found $WW_DIRS world-writable directories (CRITICAL):${NC}"
    find "$PROJECT_ROOT" -type d -perm -002 -ls 2>/dev/null
    ((ISSUES+=$WW_DIRS))
else
    echo -e "${GREEN}✓ No world-writable directories${NC}"
    ((PASSED++))
fi
echo ""

# Check files with 777 permissions
echo -e "${BLUE}[3] Checking for 777 permissions...${NC}"
PERM_777=$(find "$PROJECT_ROOT" -type f -perm 777 2>/dev/null | wc -l)
if [[ $PERM_777 -gt 0 ]]; then
    echo -e "${YELLOW}⚠ Found $PERM_777 files with 777 permissions:${NC}"
    find "$PROJECT_ROOT" -type f -perm 777 -ls 2>/dev/null
    ((WARNINGS+=$PERM_777))
else
    echo -e "${GREEN}✓ No files with 777 permissions${NC}"
    ((PASSED++))
fi
echo ""

# Check sensitive files
echo -e "${BLUE}[4] Checking sensitive files...${NC}"
SENSITIVE=$(find "$PROJECT_ROOT" -type f \( -name "*secret*" -o -name "*password*" -o -name "*key*" -o -name "*.pem" -o -name "credentials.*" \) -perm -004 2>/dev/null | wc -l)
if [[ $SENSITIVE -gt 0 ]]; then
    echo -e "${RED}✗ Found $SENSITIVE sensitive files readable by others:${NC}"
    find "$PROJECT_ROOT" -type f \( -name "*secret*" -o -name "*password*" -o -name "*key*" -o -name "*.pem" -o -name "credentials.*" \) -perm -004 -ls 2>/dev/null
    ((ISSUES+=$SENSITIVE))
else
    echo -e "${GREEN}✓ Sensitive files properly protected${NC}"
    ((PASSED++))
fi
echo ""

# Check secrets directory
echo -e "${BLUE}[5] Checking secrets directory...${NC}"
if [[ -d "$PROJECT_ROOT/configs/secrets" ]]; then
    SECRET_PERM=$(stat -c '%a' "$PROJECT_ROOT/configs/secrets")
    if [[ "$SECRET_PERM" == "700" ]]; then
        echo -e "${GREEN}✓ Secrets directory has correct permissions (700)${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ Secrets directory has incorrect permissions: $SECRET_PERM (should be 700)${NC}"
        ((ISSUES++))
    fi
else
    echo -e "${YELLOW}⚠ No secrets directory found${NC}"
fi
echo ""

# Check SUID/SGID files
echo -e "${BLUE}[6] Checking SUID/SGID files...${NC}"
SUID=$(find "$PROJECT_ROOT" -type f \( -perm -4000 -o -perm -2000 \) 2>/dev/null | wc -l)
if [[ $SUID -gt 0 ]]; then
    echo -e "${YELLOW}⚠ Found $SUID SUID/SGID files (review needed):${NC}"
    find "$PROJECT_ROOT" -type f \( -perm -4000 -o -perm -2000 \) -ls 2>/dev/null
    ((WARNINGS+=$SUID))
else
    echo -e "${GREEN}✓ No SUID/SGID files${NC}"
    ((PASSED++))
fi
echo ""

# Check scripts are executable
echo -e "${BLUE}[7] Checking script permissions...${NC}"
NON_EXEC_SCRIPTS=$(find "$PROJECT_ROOT" -type f -name "*.sh" ! -executable 2>/dev/null | wc -l)
if [[ $NON_EXEC_SCRIPTS -gt 0 ]]; then
    echo -e "${YELLOW}⚠ Found $NON_EXEC_SCRIPTS non-executable scripts:${NC}"
    find "$PROJECT_ROOT" -type f -name "*.sh" ! -executable 2>/dev/null
    ((WARNINGS+=$NON_EXEC_SCRIPTS))
else
    echo -e "${GREEN}✓ All scripts are executable${NC}"
    ((PASSED++))
fi
echo ""

# Summary
echo "=========================================="
echo " Audit Summary"
echo "=========================================="
echo ""

if [[ $ISSUES -eq 0 && $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}✓ PASSED${NC}: All security checks passed"
    echo "  No critical issues found"
    echo "  Total checks passed: $PASSED"
elif [[ $ISSUES -gt 0 ]]; then
    echo -e "${RED}✗ FAILED${NC}: Security issues found"
    echo "  Critical issues: $ISSUES"
    echo "  Warnings: $WARNINGS"
    echo "  Passed checks: $PASSED"
    echo ""
    echo "Run fix script: ./fix_permissions.sh $PROJECT_ROOT"
else
    echo -e "${YELLOW}⚠ WARNING${NC}: Minor issues found"
    echo "  Warnings: $WARNINGS"
    echo "  Passed checks: $PASSED"
fi

echo ""
exit $((ISSUES > 0 ? 1 : 0))
