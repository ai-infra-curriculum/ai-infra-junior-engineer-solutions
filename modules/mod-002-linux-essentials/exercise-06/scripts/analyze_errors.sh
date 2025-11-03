#!/bin/bash
#
# analyze_errors.sh - Error Pattern Analysis for ML Systems
#
# Description:
#   Analyzes error patterns in ML system logs, categorizes errors by type,
#   and provides actionable insights for troubleshooting.
#

set -euo pipefail

LOG_DIR="${1:-../sample_logs}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

section() { echo -e "\n${BOLD}${BLUE}=== $* ===${NC}\n"; }
subsection() { echo -e "${CYAN}$*${NC}"; }

section "Error Analysis Report"
echo "Generated: $(date)"
echo "Log Directory: $LOG_DIR"
echo ""

# Count total errors
subsection "Total Errors by File:"
for file in "$LOG_DIR"/*.log; do
    if [ -f "$file" ]; then
        count=$(grep -c "ERROR" "$file" 2>/dev/null || echo 0)
        filename=$(basename "$file")
        printf "  %-30s %3d errors\n" "$filename" "$count"
    fi
done

total_errors=$(grep -rh "ERROR" "$LOG_DIR" 2>/dev/null | wc -l)
echo ""
echo "Total errors across all files: $total_errors"

# Error distribution by type
section "Top 10 Error Types"
grep -rh "ERROR" "$LOG_DIR" 2>/dev/null | \
    sed 's/.*ERROR //' | \
    cut -d':' -f1 | \
    sort | uniq -c | \
    sort -rn | head -10 | \
    awk '{printf "  %3d  %s\n", $1, substr($0, index($0, $2))}'

# Errors by hour
section "Error Timeline (by hour)"
grep -rh "ERROR" "$LOG_DIR" 2>/dev/null | \
    awk '{print $2}' | \
    cut -d':' -f1 | \
    sort | uniq -c | \
    awk '{printf "  %s:00 - %3d errors\n", $2, $1}'

# Critical issues
section "Critical Issues"

subsection "1. CUDA/GPU Errors:"
cuda_count=$(grep -rih "cuda\|gpu" "$LOG_DIR" 2>/dev/null | grep -i "error" | wc -l)
echo "  Count: $cuda_count"
if [ "$cuda_count" -gt 0 ]; then
    echo "  Examples:"
    grep -rih "cuda\|gpu" "$LOG_DIR" 2>/dev/null | grep -i "error" | head -2 | sed 's/^/    /'
    echo ""
    echo -e "  ${YELLOW}Recommendations:${NC}"
    echo "    - Verify NVIDIA drivers are installed: nvidia-smi"
    echo "    - Check CUDA version compatibility with ML frameworks"
    echo "    - Reduce batch size to avoid OOM errors"
    echo "    - Consider upgrading GPU memory"
fi
echo ""

subsection "2. Memory Errors:"
memory_count=$(grep -rih "memory\|oom\|out of memory" "$LOG_DIR" 2>/dev/null | grep -i "error" | wc -l)
echo "  Count: $memory_count"
if [ "$memory_count" -gt 0 ]; then
    echo "  Examples:"
    grep -rih "memory\|oom\|out of memory" "$LOG_DIR" 2>/dev/null | grep -i "error" | head -2 | sed 's/^/    /'
    echo ""
    echo -e "  ${YELLOW}Recommendations:${NC}"
    echo "    - Reduce batch size in training configuration"
    echo "    - Enable gradient checkpointing"
    echo "    - Use mixed precision training (fp16)"
    echo "    - Clear cache periodically: torch.cuda.empty_cache()"
fi
echo ""

subsection "3. Disk/Storage Errors:"
disk_count=$(grep -rih "disk\|space\|i/o" "$LOG_DIR" 2>/dev/null | grep -i "error" | wc -l)
echo "  Count: $disk_count"
if [ "$disk_count" -gt 0 ]; then
    echo "  Examples:"
    grep -rih "disk\|space\|i/o" "$LOG_DIR" 2>/dev/null | grep -i "error" | head -2 | sed 's/^/    /'
    echo ""
    echo -e "  ${YELLOW}Recommendations:${NC}"
    echo "    - Check disk space: df -h"
    echo "    - Clean up old checkpoints and logs"
    echo "    - Set up log rotation"
    echo "    - Move data to larger partition"
fi
echo ""

subsection "4. Network/Connectivity Errors:"
network_count=$(grep -rih "connection\|network\|timeout" "$LOG_DIR" 2>/dev/null | grep -i "error\|failed" | wc -l)
echo "  Count: $network_count"
if [ "$network_count" -gt 0 ]; then
    echo "  Examples:"
    grep -rih "connection\|network\|timeout" "$LOG_DIR" 2>/dev/null | grep -i "error\|failed" | head -2 | sed 's/^/    /'
    echo ""
    echo -e "  ${YELLOW}Recommendations:${NC}"
    echo "    - Check network connectivity: ping hostname"
    echo "    - Verify firewall rules"
    echo "    - Increase timeout values"
    echo "    - Check service status: systemctl status service-name"
fi
echo ""

subsection "5. Authentication/Authorization Errors:"
auth_count=$(grep -rih "auth.*fail\|unauthorized\|permission denied\|invalid.*key" "$LOG_DIR" 2>/dev/null | wc -l)
echo "  Count: $auth_count"
if [ "$auth_count" -gt 0 ]; then
    echo "  Examples:"
    grep -rih "auth.*fail\|unauthorized\|permission denied\|invalid.*key" "$LOG_DIR" 2>/dev/null | head -2 | sed 's/^/    /'
    echo ""
    echo -e "  ${YELLOW}Recommendations:${NC}"
    echo "    - Verify API keys and credentials"
    echo "    - Check file permissions: ls -la"
    echo "    - Rotate expired credentials"
    echo "    - Review access control policies"
fi
echo ""

subsection "6. Data/Input Errors:"
data_count=$(grep -rih "invalid.*input\|shape\|deserialize\|parse" "$LOG_DIR" 2>/dev/null | grep -i "error" | wc -l)
echo "  Count: $data_count"
if [ "$data_count" -gt 0 ]; then
    echo "  Examples:"
    grep -rih "invalid.*input\|shape\|deserialize\|parse" "$LOG_DIR" 2>/dev/null | grep -i "error" | head -2 | sed 's/^/    /'
    echo ""
    echo -e "  ${YELLOW}Recommendations:${NC}"
    echo "    - Validate input data format"
    echo "    - Add input validation and sanitization"
    echo "    - Check data preprocessing pipeline"
    echo "    - Review API contract and schema"
fi
echo ""

subsection "7. Model/Inference Errors:"
model_count=$(grep -rih "model\|inference\|prediction" "$LOG_DIR" 2>/dev/null | grep -i "error\|fail" | wc -l)
echo "  Count: $model_count"
if [ "$model_count" -gt 0 ]; then
    echo "  Examples:"
    grep -rih "model\|inference\|prediction" "$LOG_DIR" 2>/dev/null | grep -i "error\|fail" | head -2 | sed 's/^/    /'
    echo ""
    echo -e "  ${YELLOW}Recommendations:${NC}"
    echo "    - Verify model file exists and is accessible"
    echo "    - Check model version compatibility"
    echo "    - Validate model format (h5, pt, onnx, etc.)"
    echo "    - Review model loading code"
fi
echo ""

# Database errors
subsection "8. Database Errors:"
db_count=$(grep -rih "database\|connection.*refused\|postgres\|mysql" "$LOG_DIR" 2>/dev/null | grep -i "error\|fail" | wc -l)
if [ "$db_count" -gt 0 ]; then
    echo "  Count: $db_count"
    echo "  Examples:"
    grep -rih "database\|connection.*refused\|postgres\|mysql" "$LOG_DIR" 2>/dev/null | grep -i "error\|fail" | head -2 | sed 's/^/    /'
    echo ""
    echo -e "  ${YELLOW}Recommendations:${NC}"
    echo "    - Check database service is running"
    echo "    - Verify connection string and credentials"
    echo "    - Check network connectivity to database"
    echo "    - Review database logs for issues"
    echo ""
fi

# Error severity assessment
section "Error Severity Assessment"

critical_errors=0
high_errors=0
medium_errors=0

# Critical patterns
critical_patterns=("segmentation fault" "core dumped" "no space left" "disk.*full" "fatal")
for pattern in "${critical_patterns[@]}"; do
    count=$(grep -rih "$pattern" "$LOG_DIR" 2>/dev/null | wc -l)
    critical_errors=$((critical_errors + count))
done

# High severity patterns
high_patterns=("cuda.*error" "out of memory" "connection refused" "authentication failed")
for pattern in "${high_patterns[@]}"; do
    count=$(grep -rih "$pattern" "$LOG_DIR" 2>/dev/null | wc -l)
    high_errors=$((high_errors + count))
done

# Medium severity (remaining errors)
medium_errors=$((total_errors - critical_errors - high_errors))

echo "Error Severity Distribution:"
printf "  ${RED}Critical:${NC} %3d errors (immediate action required)\n" "$critical_errors"
printf "  ${YELLOW}High:${NC}     %3d errors (requires attention)\n" "$high_errors"
printf "  ${BLUE}Medium:${NC}   %3d errors (should be monitored)\n" "$medium_errors"
echo ""

# Priority recommendations
section "Priority Action Items"

if [ "$critical_errors" -gt 0 ]; then
    echo -e "${RED}1. CRITICAL - Address immediately:${NC}"
    grep -rih "segmentation fault\|core dumped\|no space left\|disk.*full\|fatal" "$LOG_DIR" 2>/dev/null | \
        head -3 | sed 's/^/   /'
    echo ""
fi

if [ "$cuda_count" -gt 0 ]; then
    echo -e "${YELLOW}2. GPU/CUDA Issues:${NC}"
    echo "   - Run diagnostics: nvidia-smi"
    echo "   - Verify CUDA installation: nvcc --version"
    echo "   - Check available GPU memory"
    echo ""
fi

if [ "$memory_count" -gt 0 ]; then
    echo -e "${YELLOW}3. Memory Management:${NC}"
    echo "   - Review current batch sizes"
    echo "   - Implement gradient checkpointing"
    echo "   - Enable mixed precision training"
    echo ""
fi

if [ "$network_count" -gt 0 ]; then
    echo -e "${YELLOW}4. Network Connectivity:${NC}"
    echo "   - Test network connections"
    echo "   - Review firewall configurations"
    echo "   - Check service endpoints"
    echo ""
fi

# Summary
section "Summary"

if [ "$total_errors" -eq 0 ]; then
    echo -e "${GREEN}✓ No errors found - system appears healthy${NC}"
elif [ "$critical_errors" -eq 0 ] && [ "$total_errors" -lt 10 ]; then
    echo -e "${GREEN}✓ System is generally healthy with minor issues${NC}"
elif [ "$critical_errors" -gt 0 ]; then
    echo -e "${RED}✗ Critical issues detected - immediate action required${NC}"
else
    echo -e "${YELLOW}⚠ Multiple issues detected - investigation recommended${NC}"
fi

echo ""
echo "For detailed analysis, run: ./log_analyzer.sh $LOG_DIR"
echo ""
