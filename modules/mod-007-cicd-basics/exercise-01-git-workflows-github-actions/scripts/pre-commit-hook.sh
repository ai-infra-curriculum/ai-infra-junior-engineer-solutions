#!/bin/bash

# pre-commit-hook.sh
# Pre-commit hook for code quality checks

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "Pre-commit Quality Checks"
echo "========================================="
echo ""

# Get list of staged Python files
STAGED_PY_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep "\.py$" || true)

if [ -z "$STAGED_PY_FILES" ]; then
    echo "No Python files to check"
    exit 0
fi

echo "Checking files:"
echo "$STAGED_PY_FILES"
echo ""

# Track failures
FAILED=0

# 1. Check Python syntax
echo "➤ Checking Python syntax..."
for file in $STAGED_PY_FILES; do
    if ! python -m py_compile "$file" 2>/dev/null; then
        echo -e "${RED}✗ Syntax error in $file${NC}"
        FAILED=1
    fi
done

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Python syntax OK${NC}"
fi
echo ""

# 2. Check Black formatting
if command -v black &> /dev/null; then
    echo "➤ Checking Black formatting..."
    if ! black --check --quiet $STAGED_PY_FILES 2>/dev/null; then
        echo -e "${RED}✗ Black formatting issues found${NC}"
        echo ""
        echo "To fix: black $STAGED_PY_FILES"
        FAILED=1
    else
        echo -e "${GREEN}✓ Black formatting OK${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Black not installed (skipping)${NC}"
fi
echo ""

# 3. Check import sorting
if command -v isort &> /dev/null; then
    echo "➤ Checking import sorting..."
    if ! isort --check-only --quiet $STAGED_PY_FILES 2>/dev/null; then
        echo -e "${RED}✗ Import sorting issues found${NC}"
        echo ""
        echo "To fix: isort $STAGED_PY_FILES"
        FAILED=1
    else
        echo -e "${GREEN}✓ Import sorting OK${NC}"
    fi
else
    echo -e "${YELLOW}⚠ isort not installed (skipping)${NC}"
fi
echo ""

# 4. Run Flake8 linting
if command -v flake8 &> /dev/null; then
    echo "➤ Running Flake8 linting..."
    if ! flake8 $STAGED_PY_FILES --max-line-length=100 --ignore=E203,W503 2>/dev/null; then
        echo -e "${RED}✗ Flake8 linting issues found${NC}"
        FAILED=1
    else
        echo -e "${GREEN}✓ Flake8 linting OK${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Flake8 not installed (skipping)${NC}"
fi
echo ""

# 5. Check for large files
echo "➤ Checking for large files..."
LARGE_FILES=""
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
        SIZE_MB=$((SIZE / 1024 / 1024))

        if [ $SIZE_MB -gt 10 ]; then
            LARGE_FILES="${LARGE_FILES}  - $file (${SIZE_MB}MB)\n"
            FAILED=1
        fi
    fi
done

if [ -n "$LARGE_FILES" ]; then
    echo -e "${RED}✗ Large files detected (>10MB):${NC}"
    echo -e "$LARGE_FILES"
    echo "Large files should not be committed to Git"
else
    echo -e "${GREEN}✓ No large files${NC}"
fi
echo ""

# 6. Check for potential secrets
echo "➤ Checking for potential secrets..."
SECRETS_FOUND=0

# Common secret patterns
PATTERNS=(
    "password\s*=\s*['\"][^'\"]+['\"]"
    "api_key\s*=\s*['\"][^'\"]+['\"]"
    "secret\s*=\s*['\"][^'\"]+['\"]"
    "token\s*=\s*['\"][^'\"]+['\"]"
    "aws_access_key_id"
    "aws_secret_access_key"
)

for file in $STAGED_PY_FILES; do
    for pattern in "${PATTERNS[@]}"; do
        if grep -iE "$pattern" "$file" | grep -v "# " | grep -v "example" &>/dev/null; then
            echo -e "${YELLOW}⚠ Potential secret in $file${NC}"
            SECRETS_FOUND=1
        fi
    done
done

if [ $SECRETS_FOUND -eq 1 ]; then
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Commit aborted${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ No secrets detected${NC}"
fi
echo ""

# 7. Check for debugger statements
echo "➤ Checking for debugger statements..."
if git diff --cached | grep -E "(pdb|ipdb|breakpoint\(\))" &>/dev/null; then
    echo -e "${YELLOW}⚠ Debugger statements found${NC}"
    echo "Consider removing debug code before committing"
else
    echo -e "${GREEN}✓ No debugger statements${NC}"
fi
echo ""

# 8. Check for TODO comments
echo "➤ Checking for TODOs..."
TODO_COUNT=$(git diff --cached | grep -c "TODO\|FIXME\|XXX" || true)
if [ $TODO_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $TODO_COUNT TODO/FIXME comments${NC}"
else
    echo -e "${GREEN}✓ No TODO comments added${NC}"
fi
echo ""

# Summary
echo "========================================="
if [ $FAILED -eq 1 ]; then
    echo -e "${RED}✗ Pre-commit checks FAILED${NC}"
    echo "========================================="
    echo ""
    echo "Please fix the issues above before committing."
    echo ""
    echo "To skip these checks (not recommended):"
    echo "  git commit --no-verify"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓ All pre-commit checks PASSED${NC}"
    echo "========================================="
    echo ""
    exit 0
fi
