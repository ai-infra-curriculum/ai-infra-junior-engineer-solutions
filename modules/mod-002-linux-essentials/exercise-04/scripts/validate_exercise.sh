#!/bin/bash
#
# validate_exercise.sh - Validate Exercise 04 completion
#
# Usage: ./validate_exercise.sh
#

set -u

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

echo -e "${BLUE}=== Exercise 04 Validation ===${NC}"
echo ""

PASS=0
FAIL=0
WARN=0

# Check if we're in the right directory
if [ ! -f "deploy_model.sh" ] && [ ! -f "process_data.sh" ]; then
    echo -e "${RED}Error: Must run from scripts/ directory${NC}" >&2
    echo "cd to the scripts/ directory and try again" >&2
    exit 1
fi

# Test 1: Check required scripts exist
echo -e "${BLUE}[1/7] Checking required scripts...${NC}"
SCRIPTS=(
    "deploy_model.sh"
    "process_data.sh"
    "monitor_system.sh"
    "backup_ml_project.sh"
    "script_template.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo -e "  ${GREEN}✓${NC} $script exists"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} $script missing"
        ((FAIL++))
    fi
done
echo ""

# Test 2: Check scripts are executable
echo -e "${BLUE}[2/7] Checking script permissions...${NC}"
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo -e "  ${GREEN}✓${NC} $script is executable"
        ((PASS++))
    elif [ -f "$script" ]; then
        echo -e "  ${YELLOW}⚠${NC} $script not executable (run: chmod +x $script)"
        ((WARN++))
    else
        echo -e "  ${RED}✗${NC} $script not found"
        ((FAIL++))
    fi
done
echo ""

# Test 3: Validate script structure
echo -e "${BLUE}[3/7] Validating script structure...${NC}"

check_script_structure() {
    local script="$1"

    if [ ! -f "$script" ]; then
        return 1
    fi

    local has_shebang=false
    local has_set_opts=false
    local has_usage=false

    # Check for shebang
    if head -1 "$script" | grep -q "^#!/bin/bash"; then
        has_shebang=true
    fi

    # Check for set options
    if grep -q "set -[euo]" "$script"; then
        has_set_opts=true
    fi

    # Check for usage function
    if grep -q "^usage()" "$script"; then
        has_usage=true
    fi

    if [ "$has_shebang" = true ] && [ "$has_set_opts" = true ] && [ "$has_usage" = true ]; then
        echo -e "  ${GREEN}✓${NC} $script has proper structure"
        return 0
    else
        echo -e "  ${YELLOW}⚠${NC} $script missing some best practices"
        [ "$has_shebang" = false ] && echo "      - Missing shebang"
        [ "$has_set_opts" = false ] && echo "      - Missing set options"
        [ "$has_usage" = false ] && echo "      - Missing usage function"
        return 1
    fi
}

for script in "${SCRIPTS[@]}"; do
    if check_script_structure "$script"; then
        ((PASS++))
    else
        ((WARN++))
    fi
done
echo ""

# Test 4: Test deployment script
echo -e "${BLUE}[4/7] Testing deployment script...${NC}"

# Test help
if [ -f "deploy_model.sh" ]; then
    if bash deploy_model.sh --help &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} deploy_model.sh --help works"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} deploy_model.sh --help failed"
        ((FAIL++))
    fi

    # Test with invalid input
    if bash deploy_model.sh invalid_model.xyz staging 2>&1 | grep -q "not found\|Invalid"; then
        echo -e "  ${GREEN}✓${NC} deploy_model.sh validates input"
        ((PASS++))
    else
        echo -e "  ${YELLOW}⚠${NC} deploy_model.sh may not validate input properly"
        ((WARN++))
    fi
fi
echo ""

# Test 5: Test data pipeline script
echo -e "${BLUE}[5/7] Testing data pipeline script...${NC}"

if [ -f "process_data.sh" ]; then
    if bash process_data.sh --help &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} process_data.sh --help works"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} process_data.sh --help failed"
        ((FAIL++))
    fi

    # Test stats command (should work or give appropriate message)
    if bash process_data.sh stats &>/dev/null || bash process_data.sh stats 2>&1 | grep -q "not found\|No data"; then
        echo -e "  ${GREEN}✓${NC} process_data.sh stats command works"
        ((PASS++))
    else
        echo -e "  ${YELLOW}⚠${NC} process_data.sh stats may have issues"
        ((WARN++))
    fi
fi
echo ""

# Test 6: Test monitoring script
echo -e "${BLUE}[6/7] Testing monitoring script...${NC}"

if [ -f "monitor_system.sh" ]; then
    if bash monitor_system.sh --help &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} monitor_system.sh --help works"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} monitor_system.sh --help failed"
        ((FAIL++))
    fi

    # Test check command with timeout
    if timeout 5 bash monitor_system.sh check &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} monitor_system.sh check works"
        ((PASS++))
    else
        echo -e "  ${YELLOW}⚠${NC} monitor_system.sh check may have issues"
        ((WARN++))
    fi
fi
echo ""

# Test 7: Test backup script
echo -e "${BLUE}[7/7] Testing backup script...${NC}"

if [ -f "backup_ml_project.sh" ]; then
    if bash backup_ml_project.sh --help &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} backup_ml_project.sh --help works"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} backup_ml_project.sh --help failed"
        ((FAIL++))
    fi

    # Test list command
    if bash backup_ml_project.sh list &>/dev/null || bash backup_ml_project.sh list 2>&1 | grep -q "No backups"; then
        echo -e "  ${GREEN}✓${NC} backup_ml_project.sh list works"
        ((PASS++))
    else
        echo -e "  ${YELLOW}⚠${NC} backup_ml_project.sh list may have issues"
        ((WARN++))
    fi
fi
echo ""

# Bash scripting knowledge check
echo -e "${BLUE}Knowledge Check Questions:${NC}"
echo ""
echo "Answer these questions to verify understanding:"
echo ""
echo "1. What does 'set -euo pipefail' do?"
echo "   -e: Exit on error"
echo "   -u: Exit on undefined variable"
echo "   -o pipefail: Exit on pipe failure"
echo ""
echo "2. How do you make a variable readonly?"
echo "   readonly VAR=\"value\""
echo ""
echo "3. How do you get the script's directory?"
echo "   SCRIPT_DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)\""
echo ""
echo "4. How do you parse command line arguments?"
echo "   Use a while loop with case statement:"
echo "   while [[ \$# -gt 0 ]]; do"
echo "     case \$1 in"
echo "       -h|--help) usage ;;"
echo "       *) shift ;;"
echo "     esac"
echo "   done"
echo ""
echo "5. How do you log with timestamps?"
echo "   echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] message\" | tee -a logfile"
echo ""
echo "6. How do you implement cleanup on exit?"
echo "   cleanup() { # cleanup code }"
echo "   trap cleanup EXIT"
echo ""

# Summary
echo -e "${BLUE}=== Validation Summary ===${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASS"
echo -e "  ${RED}Failed:${NC}   $FAIL"
echo -e "  ${YELLOW}Warnings:${NC} $WARN"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All critical validations passed!${NC}"
    echo ""
    echo "Exercise 04 is complete. You have successfully:"
    echo "  • Created model deployment automation"
    echo "  • Built data pipeline automation"
    echo "  • Implemented system monitoring"
    echo "  • Created backup and restore system"
    echo "  • Applied bash scripting best practices"
    echo ""

    if [ $WARN -gt 0 ]; then
        echo -e "${YELLOW}Note: Some warnings were found. Review them above.${NC}"
        echo ""
    fi

    exit 0
else
    echo -e "${RED}✗ Some validations failed${NC}"
    echo ""
    echo "Fix the issues above and run validation again."
    echo ""
    exit 1
fi
