#!/bin/bash

# check-code-quality.sh
# Local code quality checks before pushing

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
TARGET_DIR="${1:-examples/}"
REPORT_FILE="code-quality-report.txt"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Code Quality Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Target directory: $TARGET_DIR"
echo "Report file: $REPORT_FILE"
echo ""

# Initialize report
cat > "$REPORT_FILE" << EOF
Code Quality Report
Generated: $(date)
Target: $TARGET_DIR
========================================

EOF

# Track overall result
OVERALL_FAILED=0

# Function to run a check and record results
run_check() {
    local name=$1
    local command=$2

    echo -e "${BLUE}➤ Running $name...${NC}"
    echo "$name" >> "$REPORT_FILE"
    echo "---" >> "$REPORT_FILE"

    if eval "$command" >> "$REPORT_FILE" 2>&1; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo "Status: PASSED" >> "$REPORT_FILE"
    else
        echo -e "${RED}✗ $name failed${NC}"
        echo "Status: FAILED" >> "$REPORT_FILE"
        OVERALL_FAILED=1
    fi

    echo "" >> "$REPORT_FILE"
    echo ""
}

# 1. Black - Code formatting
if command -v black &> /dev/null; then
    run_check "Black formatting" "black --check $TARGET_DIR"
else
    echo -e "${YELLOW}⚠ Black not installed (skipped)${NC}"
fi

# 2. isort - Import sorting
if command -v isort &> /dev/null; then
    run_check "Import sorting (isort)" "isort --check-only $TARGET_DIR"
else
    echo -e "${YELLOW}⚠ isort not installed (skipped)${NC}"
fi

# 3. Flake8 - Linting
if command -v flake8 &> /dev/null; then
    run_check "Flake8 linting" "flake8 $TARGET_DIR --max-line-length=100 --max-complexity=10 --statistics"
else
    echo -e "${YELLOW}⚠ Flake8 not installed (skipped)${NC}"
fi

# 4. Pylint - Code analysis
if command -v pylint &> /dev/null; then
    run_check "Pylint analysis" "pylint $TARGET_DIR --max-line-length=100 --disable=C0111,R0903,W0511"
else
    echo -e "${YELLOW}⚠ Pylint not installed (skipped)${NC}"
fi

# 5. mypy - Type checking
if command -v mypy &> /dev/null; then
    run_check "Type checking (mypy)" "mypy $TARGET_DIR --ignore-missing-imports --show-error-codes"
else
    echo -e "${YELLOW}⚠ mypy not installed (skipped)${NC}"
fi

# 6. Bandit - Security check
if command -v bandit &> /dev/null; then
    run_check "Security analysis (Bandit)" "bandit -r $TARGET_DIR -ll"
else
    echo -e "${YELLOW}⚠ Bandit not installed (skipped)${NC}"
fi

# 7. Safety - Dependency vulnerabilities
if command -v safety &> /dev/null; then
    echo -e "${BLUE}➤ Running Dependency vulnerability check...${NC}"
    if safety check; then
        echo -e "${GREEN}✓ No known vulnerabilities${NC}"
    else
        echo -e "${YELLOW}⚠ Vulnerabilities found${NC}"
        OVERALL_FAILED=1
    fi
    echo ""
else
    echo -e "${YELLOW}⚠ Safety not installed (skipped)${NC}"
fi

# 8. Radon - Code complexity
if command -v radon &> /dev/null; then
    echo -e "${BLUE}➤ Calculating code complexity...${NC}"
    echo "Cyclomatic Complexity:" >> "$REPORT_FILE"
    radon cc $TARGET_DIR -a >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    echo "Maintainability Index:" >> "$REPORT_FILE"
    radon mi $TARGET_DIR >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    echo -e "${GREEN}✓ Complexity metrics calculated${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ Radon not installed (skipped)${NC}"
fi

# 9. Test coverage (if tests exist)
if command -v pytest &> /dev/null && [ -d "$TARGET_DIR" ]; then
    echo -e "${BLUE}➤ Running tests with coverage...${NC}"
    if pytest $TARGET_DIR --cov=$TARGET_DIR --cov-report=term --cov-report=html --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ Tests passed${NC}"
        echo "Tests: PASSED" >> "$REPORT_FILE"
    else
        echo -e "${RED}✗ Tests failed${NC}"
        echo "Tests: FAILED" >> "$REPORT_FILE"
        OVERALL_FAILED=1
    fi
    echo "" >> "$REPORT_FILE"
    echo ""
else
    echo -e "${YELLOW}⚠ pytest not installed or no tests found (skipped)${NC}"
fi

# 10. Dead code detection
if command -v vulture &> /dev/null; then
    echo -e "${BLUE}➤ Detecting dead code...${NC}"
    echo "Dead Code Detection:" >> "$REPORT_FILE"
    vulture $TARGET_DIR --min-confidence 80 >> "$REPORT_FILE" 2>&1 || true
    echo "" >> "$REPORT_FILE"
    echo -e "${GREEN}✓ Dead code check complete${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ Vulture not installed (skipped)${NC}"
fi

# Summary
echo "========================================" >> "$REPORT_FILE"
echo "OVERALL RESULT: $([ $OVERALL_FAILED -eq 0 ] && echo 'PASSED' || echo 'FAILED')" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"

echo -e "${BLUE}========================================${NC}"
if [ $OVERALL_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All quality checks PASSED${NC}"
else
    echo -e "${RED}✗ Some quality checks FAILED${NC}"
fi
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Full report saved to: $REPORT_FILE"
echo ""

# Provide suggestions if checks failed
if [ $OVERALL_FAILED -eq 1 ]; then
    echo "To fix common issues:"
    echo ""
    echo "  # Fix formatting"
    echo "  black $TARGET_DIR"
    echo ""
    echo "  # Fix import sorting"
    echo "  isort $TARGET_DIR"
    echo ""
    echo "  # View detailed Flake8 issues"
    echo "  flake8 $TARGET_DIR --show-source"
    echo ""
    echo "  # Run tests"
    echo "  pytest $TARGET_DIR -v"
    echo ""
fi

# Exit with appropriate code
exit $OVERALL_FAILED
