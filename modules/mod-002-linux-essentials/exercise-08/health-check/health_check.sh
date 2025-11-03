#!/bin/bash
###############################################################################
# System Health Check Script - Comprehensive ML Infrastructure Health Check
###############################################################################
#
# Purpose: Validate system health for ML infrastructure
#
# Usage: ./health_check.sh [OPTIONS]
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
DISK_WARN_THRESHOLD="${DISK_WARN_THRESHOLD:-80}"
DISK_CRIT_THRESHOLD="${DISK_CRIT_THRESHOLD:-90}"
MEM_WARN_THRESHOLD="${MEM_WARN_THRESHOLD:-85}"
CPU_LOAD_MULTIPLIER="${CPU_LOAD_MULTIPLIER:-1.5}"
GPU_TEMP_WARN="${GPU_TEMP_WARN:-75}"
GPU_TEMP_CRIT="${GPU_TEMP_CRIT:-85}"

SERVICES_TO_CHECK="${SERVICES_TO_CHECK:-docker}"
CHECK_GPU="${CHECK_GPU:-auto}"  # auto, yes, no
CHECK_NETWORK="${CHECK_NETWORK:-yes}"
VERBOSE="${VERBOSE:-false}"
JSON_OUTPUT="${JSON_OUTPUT:-false}"

# Counters
CHECKS_PASSED=0
CHECKS_WARNING=0
CHECKS_FAILED=0
TOTAL_CHECKS=0

# Results for JSON output
declare -a RESULTS

# Functions
check_pass() {
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ "$JSON_OUTPUT" != "true" ]; then
        echo -e "${GREEN}✓${NC} $1"
    fi
    RESULTS+=("$(printf '{"check":"%s","status":"pass","message":"%s"}' "$1" "$1")")
}

check_warn() {
    CHECKS_WARNING=$((CHECKS_WARNING + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ "$JSON_OUTPUT" != "true" ]; then
        echo -e "${YELLOW}!${NC} $1"
    fi
    RESULTS+=("$(printf '{"check":"%s","status":"warning","message":"%s"}' "$(echo "$1" | cut -d: -f1)" "$1")")
}

check_fail() {
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ "$JSON_OUTPUT" != "true" ]; then
        echo -e "${RED}✗${NC} $1"
    fi
    RESULTS+=("$(printf '{"check":"%s","status":"fail","message":"%s"}' "$(echo "$1" | cut -d: -f1)" "$1")")
}

section_header() {
    if [ "$JSON_OUTPUT" != "true" ]; then
        echo ""
        echo -e "${CYAN}${BOLD}━━━ $1 ━━━${NC}"
        echo ""
    fi
}

# Help function
show_help() {
    cat << EOF
System Health Check Script

Usage: $0 [OPTIONS]

Options:
    --disk-warn N          Disk usage warning threshold % (default: 80)
    --disk-crit N          Disk usage critical threshold % (default: 90)
    --mem-warn N           Memory usage warning threshold % (default: 85)
    --services "list"      Space-separated list of services to check
    --check-gpu yes|no     Force GPU checking on/off (default: auto)
    --no-network           Skip network connectivity checks
    --verbose              Show detailed information
    --json                 Output results in JSON format
    -h, --help             Show this help message

Examples:
    # Basic health check
    $0

    # Check specific services
    $0 --services "docker nginx postgresql"

    # Custom thresholds
    $0 --disk-warn 70 --disk-crit 85 --mem-warn 90

    # JSON output for monitoring systems
    $0 --json

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --disk-warn)
            DISK_WARN_THRESHOLD="$2"
            shift 2
            ;;
        --disk-crit)
            DISK_CRIT_THRESHOLD="$2"
            shift 2
            ;;
        --mem-warn)
            MEM_WARN_THRESHOLD="$2"
            shift 2
            ;;
        --services)
            SERVICES_TO_CHECK="$2"
            shift 2
            ;;
        --check-gpu)
            CHECK_GPU="$2"
            shift 2
            ;;
        --no-network)
            CHECK_NETWORK="no"
            shift
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        --json)
            JSON_OUTPUT="true"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Header
if [ "$JSON_OUTPUT" != "true" ]; then
    echo -e "${BOLD}${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║      ML Infrastructure Health Check                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Hostname: $(hostname)"
    echo ""
fi

# 1. Disk Space Check
section_header "1. Disk Space"

DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
DISK_AVAIL=$(df -h / | tail -1 | awk '{print $4}')

if [ "$DISK_USAGE" -lt "$DISK_WARN_THRESHOLD" ]; then
    check_pass "Disk usage: ${DISK_USAGE}% (Available: $DISK_AVAIL)"
elif [ "$DISK_USAGE" -lt "$DISK_CRIT_THRESHOLD" ]; then
    check_warn "Disk usage: ${DISK_USAGE}% (warning threshold: ${DISK_WARN_THRESHOLD}%)"
else
    check_fail "Disk usage: ${DISK_USAGE}% (critical threshold: ${DISK_CRIT_THRESHOLD}%)"
fi

# Check inode usage
INODE_USAGE=$(df -i / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$INODE_USAGE" -lt 80 ]; then
    check_pass "Inode usage: ${INODE_USAGE}%"
elif [ "$INODE_USAGE" -lt 90 ]; then
    check_warn "Inode usage: ${INODE_USAGE}% (high)"
else
    check_fail "Inode usage: ${INODE_USAGE}% (critical)"
fi

# 2. Memory Check
section_header "2. Memory"

MEM_TOTAL=$(free -m | awk 'NR==2 {print $2}')
MEM_USED=$(free -m | awk 'NR==2 {print $3}')
MEM_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100)}')
MEM_AVAIL=$(free -h | awk 'NR==2 {print $7}')

if [ "$MEM_USAGE" -lt "$MEM_WARN_THRESHOLD" ]; then
    check_pass "Memory usage: ${MEM_USAGE}% (Available: $MEM_AVAIL)"
else
    check_warn "Memory usage: ${MEM_USAGE}% (threshold: ${MEM_WARN_THRESHOLD}%)"
fi

# Check swap usage
SWAP_TOTAL=$(free -m | awk 'NR==3 {print $2}')
if [ "$SWAP_TOTAL" -gt 0 ]; then
    SWAP_USED=$(free -m | awk 'NR==3 {print $3}')
    SWAP_USAGE=$(echo "scale=0; $SWAP_USED * 100 / $SWAP_TOTAL" | bc)

    if [ "$SWAP_USAGE" -lt 50 ]; then
        check_pass "Swap usage: ${SWAP_USAGE}%"
    elif [ "$SWAP_USAGE" -lt 80 ]; then
        check_warn "Swap usage: ${SWAP_USAGE}% (high)"
    else
        check_fail "Swap usage: ${SWAP_USAGE}% (excessive swapping)"
    fi
else
    check_warn "No swap configured"
fi

# 3. CPU Load
section_header "3. CPU Load"

CPU_COUNT=$(nproc)
LOAD_1MIN=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
LOAD_5MIN=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $2}' | sed 's/,//')
LOAD_15MIN=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $3}' | sed 's/,//')
LOAD_THRESHOLD=$(echo "$CPU_COUNT * $CPU_LOAD_MULTIPLIER" | bc)

if (( $(echo "$LOAD_1MIN < $LOAD_THRESHOLD" | bc -l) )); then
    check_pass "Load average (1/5/15 min): $LOAD_1MIN / $LOAD_5MIN / $LOAD_15MIN (CPUs: $CPU_COUNT)"
else
    check_warn "Load average: $LOAD_1MIN (high for $CPU_COUNT CPUs, threshold: $LOAD_THRESHOLD)"
fi

# 4. Critical Services
section_header "4. Services"

for service in $SERVICES_TO_CHECK; do
    if systemctl is-active --quiet "$service" 2>/dev/null; then
        check_pass "$service is running"
    elif systemctl list-unit-files | grep -q "^${service}.service"; then
        check_fail "$service is not running"
    else
        check_warn "$service not installed or not a systemd service"
    fi
done

# 5. GPU Status (if available)
section_header "5. GPU Status"

# Auto-detect GPU if set to auto
if [ "$CHECK_GPU" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        CHECK_GPU="yes"
    else
        CHECK_GPU="no"
    fi
fi

if [ "$CHECK_GPU" = "yes" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
        check_pass "Found $GPU_COUNT GPU(s)"

        nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,power.draw,power.limit \
            --format=csv,noheader,nounits | while IFS=',' read -r idx name temp mem_used mem_total power_draw power_limit; do

            temp=$(echo "$temp" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)
            power_draw=$(echo "$power_draw" | xargs)
            power_limit=$(echo "$power_limit" | xargs)

            mem_pct=$(echo "scale=0; $mem_used * 100 / $mem_total" | bc)

            if [ "$temp" -lt "$GPU_TEMP_WARN" ]; then
                check_pass "  GPU $idx ($name): ${temp}°C, Memory: ${mem_pct}%, Power: ${power_draw}W/${power_limit}W"
            elif [ "$temp" -lt "$GPU_TEMP_CRIT" ]; then
                check_warn "  GPU $idx: Temperature ${temp}°C (warm)"
            else
                check_fail "  GPU $idx: Temperature ${temp}°C (critical!)"
            fi
        done
    else
        check_fail "nvidia-smi not working or GPUs not accessible"
    fi
else
    check_warn "GPU check skipped"
fi

# 6. Network Connectivity
if [ "$CHECK_NETWORK" = "yes" ]; then
    section_header "6. Network"

    if ping -c 1 -W 2 8.8.8.8 &> /dev/null; then
        check_pass "Internet connectivity OK (8.8.8.8 reachable)"
    else
        check_fail "No internet connectivity"
    fi

    if command -v dig &> /dev/null; then
        if dig +short google.com @8.8.8.8 | grep -q "."; then
            check_pass "DNS resolution working"
        else
            check_fail "DNS resolution failed"
        fi
    elif command -v nslookup &> /dev/null; then
        if nslookup google.com 8.8.8.8 &> /dev/null; then
            check_pass "DNS resolution working"
        else
            check_fail "DNS resolution failed"
        fi
    fi
fi

# 7. Recent Errors
section_header "7. System Health"

if command -v journalctl &> /dev/null; then
    ERROR_COUNT=$(journalctl --since "1 hour ago" -p err --no-pager 2>/dev/null | grep -c "^" || echo 0)

    if [ "$ERROR_COUNT" -lt 10 ]; then
        check_pass "Recent errors: $ERROR_COUNT (last hour)"
    elif [ "$ERROR_COUNT" -lt 50 ]; then
        check_warn "Recent errors: $ERROR_COUNT (last hour - investigate)"
    else
        check_fail "Recent errors: $ERROR_COUNT (last hour - critical!)"
    fi
fi

# Check for OOM events
OOM_COUNT=$(dmesg -T 2>/dev/null | grep -i "out of memory" | wc -l)
if [ "$OOM_COUNT" -eq 0 ]; then
    check_pass "No OOM killer events"
else
    check_warn "OOM killer events detected: $OOM_COUNT"
fi

# 8. System Uptime
section_header "8. System Info"

UPTIME=$(uptime -p 2>/dev/null || uptime | awk '{print $3, $4}')
check_pass "Uptime: $UPTIME"

# Check for pending reboot
if [ -f /var/run/reboot-required ]; then
    check_warn "System reboot required"
fi

# Summary
if [ "$JSON_OUTPUT" = "true" ]; then
    # Output JSON
    echo "{"
    echo "  \"timestamp\": \"$(date '+%Y-%m-%d %H:%M:%S')\","
    echo "  \"hostname\": \"$(hostname)\","
    echo "  \"checks\": ["
    first=true
    for result in "${RESULTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo ","
        fi
        echo "    $result"
    done
    echo ""
    echo "  ],"
    echo "  \"summary\": {"
    echo "    \"total\": $TOTAL_CHECKS,"
    echo "    \"passed\": $CHECKS_PASSED,"
    echo "    \"warnings\": $CHECKS_WARNING,"
    echo "    \"failed\": $CHECKS_FAILED,"
    echo "    \"health_score\": $(echo "scale=2; ($CHECKS_PASSED / $TOTAL_CHECKS) * 100" | bc)"
    echo "  }"
    echo "}"
else
    # Text summary
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}                        Summary${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  Total checks:  ${BOLD}$TOTAL_CHECKS${NC}"
    echo -e "  ${GREEN}Passed:${NC}        $CHECKS_PASSED"
    echo -e "  ${YELLOW}Warnings:${NC}      $CHECKS_WARNING"
    echo -e "  ${RED}Failed:${NC}        $CHECKS_FAILED"
    echo ""

    HEALTH_SCORE=$(echo "scale=2; ($CHECKS_PASSED / $TOTAL_CHECKS) * 100" | bc)
    echo -e "  Health Score:  ${BOLD}${HEALTH_SCORE}%${NC}"
    echo ""

    if [ "$CHECKS_FAILED" -eq 0 ] && [ "$CHECKS_WARNING" -eq 0 ]; then
        echo -e "${GREEN}${BOLD}✓ All checks passed! System is healthy.${NC}"
    elif [ "$CHECKS_FAILED" -eq 0 ]; then
        echo -e "${YELLOW}! System is operational with warnings.${NC}"
    else
        echo -e "${RED}✗ Critical issues detected! Immediate attention required.${NC}"
    fi

    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
fi

# Exit code based on health
if [ "$CHECKS_FAILED" -gt 0 ]; then
    exit 2  # Critical
elif [ "$CHECKS_WARNING" -gt 0 ]; then
    exit 1  # Warning
else
    exit 0  # OK
fi
