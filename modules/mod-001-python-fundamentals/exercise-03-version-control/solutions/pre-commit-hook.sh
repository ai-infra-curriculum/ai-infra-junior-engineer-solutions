#!/bin/bash
#
# Pre-commit Hook for AI/ML Projects
#
# This hook validates code quality, formatting, and project-specific rules
# before allowing commits. Install with:
#   cp pre-commit-hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running pre-commit checks...${NC}\n"

# Track if any check fails
FAIL=0

# ============================================================================
# 1. Check for large files (prevent accidentally committing large model files)
# ============================================================================
echo "Checking for large files..."

MAX_FILE_SIZE=10485760  # 10MB in bytes
LARGE_FILES=$(git diff --cached --name-only | while read file; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
        if [ "$size" -gt "$MAX_FILE_SIZE" ]; then
            echo "$file ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo ${size}B))"
        fi
    fi
done)

if [ -n "$LARGE_FILES" ]; then
    echo -e "${RED}✗ Large files detected (>10MB):${NC}"
    echo "$LARGE_FILES"
    echo -e "${YELLOW}Consider using Git LFS or excluding from repository${NC}"
    FAIL=1
else
    echo -e "${GREEN}✓ No large files${NC}"
fi

# ============================================================================
# 2. Check for sensitive data patterns
# ============================================================================
echo -e "\nChecking for sensitive data..."

# Patterns to check for
SENSITIVE_PATTERNS=(
    "password\s*=\s*['\"][^'\"]+['\"]"
    "api_key\s*=\s*['\"][^'\"]+['\"]"
    "secret\s*=\s*['\"][^'\"]+['\"]"
    "token\s*=\s*['\"][^'\"]+['\"]"
    "aws_access_key_id"
    "aws_secret_access_key"
    "private_key"
    "BEGIN RSA PRIVATE KEY"
    "BEGIN PRIVATE KEY"
)

SENSITIVE_FOUND=0
for pattern in "${SENSITIVE_PATTERNS[@]}"; do
    if git diff --cached | grep -iE "$pattern" > /dev/null; then
        echo -e "${RED}✗ Potential sensitive data found: $pattern${NC}"
        SENSITIVE_FOUND=1
    fi
done

if [ $SENSITIVE_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓ No sensitive data detected${NC}"
else
    FAIL=1
fi

# ============================================================================
# 3. Validate Python files
# ============================================================================
echo -e "\nValidating Python files..."

PYTHON_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -n "$PYTHON_FILES" ]; then
    # Check Python syntax
    SYNTAX_ERRORS=0
    for file in $PYTHON_FILES; do
        if [ -f "$file" ]; then
            if ! python3 -m py_compile "$file" 2>/dev/null; then
                echo -e "${RED}✗ Syntax error in $file${NC}"
                SYNTAX_ERRORS=1
            fi
        fi
    done

    if [ $SYNTAX_ERRORS -eq 0 ]; then
        echo -e "${GREEN}✓ Python syntax valid${NC}"
    else
        FAIL=1
    fi

    # Run black formatter check (if available)
    if command -v black &> /dev/null; then
        echo "Checking code formatting with black..."
        if ! echo "$PYTHON_FILES" | xargs black --check --quiet 2>/dev/null; then
            echo -e "${YELLOW}⚠ Code formatting issues found${NC}"
            echo -e "${YELLOW}Run 'black .' to auto-format${NC}"
            # Don't fail on formatting, just warn
        else
            echo -e "${GREEN}✓ Code formatting looks good${NC}"
        fi
    fi

    # Run flake8 linter (if available)
    if command -v flake8 &> /dev/null; then
        echo "Linting with flake8..."
        if ! echo "$PYTHON_FILES" | xargs flake8 --max-line-length=100 --ignore=E203,W503 2>/dev/null; then
            echo -e "${YELLOW}⚠ Linting issues found${NC}"
            # Don't fail on linting warnings, just inform
        else
            echo -e "${GREEN}✓ Linting passed${NC}"
        fi
    fi
else
    echo -e "${GREEN}✓ No Python files to check${NC}"
fi

# ============================================================================
# 4. Check for debug statements and TODOs
# ============================================================================
echo -e "\nChecking for debug statements..."

DEBUG_PATTERNS=(
    "import pdb"
    "pdb.set_trace()"
    "breakpoint()"
    "print\s*\("
    "console\.log\("
    "debugger"
)

DEBUG_FOUND=0
for pattern in "${DEBUG_PATTERNS[@]}"; do
    if git diff --cached | grep -E "$pattern" > /dev/null; then
        echo -e "${YELLOW}⚠ Debug statement found: $pattern${NC}"
        DEBUG_FOUND=1
    fi
done

if [ $DEBUG_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓ No debug statements${NC}"
else
    echo -e "${YELLOW}Consider removing debug statements before committing${NC}"
    # Don't fail, just warn
fi

# Check for TODO/FIXME comments in staged changes
TODO_COUNT=$(git diff --cached | grep -i -E "(TODO|FIXME|XXX|HACK)" | wc -l)
if [ "$TODO_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $TODO_COUNT TODO/FIXME comments in staged changes${NC}"
fi

# ============================================================================
# 5. Validate JSON/YAML files
# ============================================================================
echo -e "\nValidating JSON/YAML files..."

JSON_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.json$' || true)
YAML_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(yml|yaml)$' || true)

CONFIG_ERRORS=0

for file in $JSON_FILES; do
    if [ -f "$file" ]; then
        if ! python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
            echo -e "${RED}✗ Invalid JSON: $file${NC}"
            CONFIG_ERRORS=1
        fi
    fi
done

for file in $YAML_FILES; do
    if [ -f "$file" ]; then
        if command -v python3 &> /dev/null; then
            if ! python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
                echo -e "${RED}✗ Invalid YAML: $file${NC}"
                CONFIG_ERRORS=1
            fi
        fi
    fi
done

if [ $CONFIG_ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ Config files valid${NC}"
else
    FAIL=1
fi

# ============================================================================
# 6. Check for merge conflict markers
# ============================================================================
echo -e "\nChecking for merge conflict markers..."

CONFLICT_MARKERS=$(git diff --cached | grep -E '^[+].*(<{7}|={7}|>{7})' || true)

if [ -n "$CONFLICT_MARKERS" ]; then
    echo -e "${RED}✗ Merge conflict markers found${NC}"
    echo "$CONFLICT_MARKERS"
    FAIL=1
else
    echo -e "${GREEN}✓ No conflict markers${NC}"
fi

# ============================================================================
# 7. Validate commit message format (if set)
# ============================================================================
# Note: Actual commit message validation happens in commit-msg hook
# This is just a reminder
echo -e "\n${YELLOW}Remember to use conventional commit format:${NC}"
echo "  feat(scope): description"
echo "  fix(scope): description"
echo "  docs(scope): description"

# ============================================================================
# Final verdict
# ============================================================================
echo ""
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All pre-commit checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Pre-commit checks failed!${NC}"
    echo -e "${YELLOW}Fix the issues above or use 'git commit --no-verify' to skip checks${NC}"
    exit 1
fi
