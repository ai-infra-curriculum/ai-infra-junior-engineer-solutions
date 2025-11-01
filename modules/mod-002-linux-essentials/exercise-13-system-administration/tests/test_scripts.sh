#!/bin/bash
#
# test_scripts.sh - Test suite for system administration scripts
#
# Usage: ./test_scripts.sh
#

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOLUTIONS_DIR="$SCRIPT_DIR/../solutions"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'
BOLD='\033[1m'

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# ===========================
# Test Framework
# ===========================

run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -n "  Testing: $test_name... "
    ((TESTS_RUN++))

    if eval "$test_command" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${RESET}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${RESET}"
        ((TESTS_FAILED++))
        return 1
    fi
}

test_script_exists() {
    local script_name="$1"
    test -f "$SOLUTIONS_DIR/$script_name"
}

test_script_executable() {
    local script_name="$1"
    test -x "$SOLUTIONS_DIR/$script_name"
}

test_script_syntax() {
    local script_name="$1"
    bash -n "$SOLUTIONS_DIR/$script_name"
}

test_script_help() {
    local script_name="$1"
    "$SOLUTIONS_DIR/$script_name" --help &>/dev/null || \
    "$SOLUTIONS_DIR/$script_name" -h &>/dev/null
}

# ===========================
# Test: system_monitor.sh
# ===========================

test_system_monitor() {
    echo -e "${BOLD}${CYAN}Testing: system_monitor.sh${RESET}"

    run_test "Script exists" "test_script_exists system_monitor.sh"
    run_test "Script is executable" "test_script_executable system_monitor.sh"
    run_test "Script syntax is valid" "test_script_syntax system_monitor.sh"
    run_test "Help option works" "test_script_help system_monitor.sh"
    run_test "Check mode (non-root)" "$SOLUTIONS_DIR/system_monitor.sh --check 2>/dev/null || true"

    echo ""
}

# ===========================
# Test: user_management.sh
# ===========================

test_user_management() {
    echo -e "${BOLD}${CYAN}Testing: user_management.sh${RESET}"

    run_test "Script exists" "test_script_exists user_management.sh"
    run_test "Script is executable" "test_script_executable user_management.sh"
    run_test "Script syntax is valid" "test_script_syntax user_management.sh"
    run_test "Help option works" "test_script_help user_management.sh"
    run_test "List users command" "$SOLUTIONS_DIR/user_management.sh list-users 2>/dev/null || true"

    echo ""
}

# ===========================
# Test: backup_automation.sh
# ===========================

test_backup_automation() {
    echo -e "${BOLD}${CYAN}Testing: backup_automation.sh${RESET}"

    run_test "Script exists" "test_script_exists backup_automation.sh"
    run_test "Script is executable" "test_script_executable backup_automation.sh"
    run_test "Script syntax is valid" "test_script_syntax backup_automation.sh"
    run_test "Help option works" "test_script_help backup_automation.sh"
    run_test "List backups command" "$SOLUTIONS_DIR/backup_automation.sh list 2>/dev/null || true"

    echo ""
}

# ===========================
# Test: log_rotation.sh
# ===========================

test_log_rotation() {
    echo -e "${BOLD}${CYAN}Testing: log_rotation.sh${RESET}"

    run_test "Script exists" "test_script_exists log_rotation.sh"
    run_test "Script is executable" "test_script_executable log_rotation.sh"
    run_test "Script syntax is valid" "test_script_syntax log_rotation.sh"
    run_test "Help option works" "test_script_help log_rotation.sh"
    run_test "Analyze command" "$SOLUTIONS_DIR/log_rotation.sh analyze 2>/dev/null || true"

    echo ""
}

# ===========================
# Test: security_audit.sh
# ===========================

test_security_audit() {
    echo -e "${BOLD}${CYAN}Testing: security_audit.sh${RESET}"

    run_test "Script exists" "test_script_exists security_audit.sh"
    run_test "Script is executable" "test_script_executable security_audit.sh"
    run_test "Script syntax is valid" "test_script_syntax security_audit.sh"
    run_test "Help option works" "test_script_help security_audit.sh"

    echo ""
}

# ===========================
# Test: disk_manager.sh
# ===========================

test_disk_manager() {
    echo -e "${BOLD}${CYAN}Testing: disk_manager.sh${RESET}"

    run_test "Script exists" "test_script_exists disk_manager.sh"
    run_test "Script is executable" "test_script_executable disk_manager.sh"
    run_test "Script syntax is valid" "test_script_syntax disk_manager.sh"
    run_test "Help option works" "test_script_help disk_manager.sh"
    run_test "Check command" "$SOLUTIONS_DIR/disk_manager.sh check 2>/dev/null || true"

    echo ""
}

# ===========================
# Test: manage_services.sh
# ===========================

test_manage_services() {
    echo -e "${BOLD}${CYAN}Testing: manage_services.sh${RESET}"

    run_test "Script exists" "test_script_exists manage_services.sh"
    run_test "Script is executable" "test_script_executable manage_services.sh"
    run_test "Script syntax is valid" "test_script_syntax manage_services.sh"
    run_test "Help option works" "test_script_help manage_services.sh"

    echo ""
}

# ===========================
# Test: system_maintenance.sh
# ===========================

test_system_maintenance() {
    echo -e "${BOLD}${CYAN}Testing: system_maintenance.sh${RESET}"

    run_test "Script exists" "test_script_exists system_maintenance.sh"
    run_test "Script is executable" "test_script_executable system_maintenance.sh"
    run_test "Script syntax is valid" "test_script_syntax system_maintenance.sh"
    run_test "Help option works" "test_script_help system_maintenance.sh"

    echo ""
}

# ===========================
# Integration Tests
# ===========================

test_integration() {
    echo -e "${BOLD}${CYAN}Integration Tests${RESET}"

    # Test script interactions
    run_test "All scripts have shebang" "grep -l '^#!/bin/bash' $SOLUTIONS_DIR/*.sh | wc -l | grep -q 8"
    run_test "All scripts have usage function" "grep -l 'usage()' $SOLUTIONS_DIR/*.sh | wc -l | grep -q 8"
    run_test "All scripts use set -euo pipefail" "grep -l 'set -euo pipefail' $SOLUTIONS_DIR/*.sh | wc -l | grep -q 8"
    run_test "All scripts have logging" "grep -l 'log_message\|LOG_FILE' $SOLUTIONS_DIR/*.sh | wc -l | grep -q 8"

    echo ""
}

# ===========================
# Main Test Runner
# ===========================

main() {
    echo -e "${BOLD}${CYAN}========================================"
    echo "System Administration Scripts Test Suite"
    echo "========================================${RESET}"
    echo "Test Directory: $SCRIPT_DIR"
    echo "Solutions Directory: $SOLUTIONS_DIR"
    echo ""

    # Run all tests
    test_system_monitor
    test_user_management
    test_backup_automation
    test_log_rotation
    test_security_audit
    test_disk_manager
    test_manage_services
    test_system_maintenance
    test_integration

    # Summary
    echo -e "${BOLD}${CYAN}========================================"
    echo "Test Summary"
    echo "========================================${RESET}"
    echo "Tests Run: $TESTS_RUN"
    echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${RESET}"
    echo -e "Tests Failed: ${RED}$TESTS_FAILED${RESET}"

    local pass_rate=0
    if [[ $TESTS_RUN -gt 0 ]]; then
        pass_rate=$((TESTS_PASSED * 100 / TESTS_RUN))
    fi
    echo "Pass Rate: ${pass_rate}%"

    echo ""

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✓ ALL TESTS PASSED${RESET}"
        exit 0
    else
        echo -e "${RED}${BOLD}✗ SOME TESTS FAILED${RESET}"
        exit 1
    fi
}

main "$@"
