#!/bin/bash
###############################################################################
# Scenario 1: Disk Full Error - Investigation Script
###############################################################################
#
# Problem: ML training fails with "OSError: [Errno 28] No space left on device"
# Location: Checkpoints being saved to /var/ml/checkpoints/
#

set -u

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' CYAN='' BOLD='' NC=''
fi

section() { echo -e "\n${BOLD}${BLUE}=== $* ===${NC}\n"; }
subsection() { echo -e "${CYAN}$*${NC}"; }
log_info() { echo -e "  $*"; }

echo -e "${BOLD}${RED}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${RED}║  Scenario 1: Disk Full Error Investigation                ║${NC}"
echo -e "${BOLD}${RED}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Problem:${NC} Training job fails with disk space error"
echo -e "${YELLOW}Location:${NC} /var/ml/checkpoints/"
echo ""

section "Step 1: Check Overall Disk Usage"
echo "Command: df -h"
echo ""
df -h

section "Step 2: Check Inode Usage"
echo "Command: df -i"
echo ""
echo "Note: Sometimes inodes run out before disk space"
echo ""
df -i

section "Step 3: Identify Large Directories"
echo "Command: du -sh /var/* 2>/dev/null | sort -hr | head -10"
echo ""
echo "Top 10 largest directories in /var:"
du -sh /var/* 2>/dev/null | sort -hr | head -10 || echo "Permission denied for some directories"

section "Step 4: Find Large Files (>100MB)"
echo "Command: find /var -type f -size +100M -exec ls -lh {} \\; 2>/dev/null | head -10"
echo ""
find /var -type f -size +100M -exec ls -lh {} \; 2>/dev/null | head -10 || \
    echo "No large files found or permission denied"

section "Step 5: Check ML Checkpoint Directory"
subsection "Directory size:"
du -sh /var/ml/checkpoints/ 2>/dev/null || echo "Directory does not exist or not accessible"

echo ""
subsection "Recent checkpoint files:"
ls -lhtr /var/ml/checkpoints/ 2>/dev/null | tail -10 || echo "Cannot list directory"

section "Step 6: Count Old Checkpoint Files (>7 days)"
old_count=$(find /var/ml/checkpoints -type f -mtime +7 2>/dev/null | wc -l)
echo "Files older than 7 days: $old_count"

if [ "$old_count" -gt 0 ]; then
    echo ""
    subsection "Sample old files:"
    find /var/ml/checkpoints -type f -mtime +7 2>/dev/null | head -5
fi

section "Step 7: Check for Other Space Consumers"
subsection "Temporary files:"
du -sh /tmp 2>/dev/null || echo "/tmp not accessible"
du -sh ~/.cache 2>/dev/null || echo "~/.cache not accessible"

echo ""
subsection "Package caches:"
if command -v apt &>/dev/null; then
    du -sh /var/cache/apt/archives 2>/dev/null || echo "APT cache not accessible"
fi

if command -v yum &>/dev/null; then
    du -sh /var/cache/yum 2>/dev/null || echo "YUM cache not accessible"
fi

echo ""
subsection "Docker (if installed):"
if command -v docker &>/dev/null; then
    docker system df 2>/dev/null || echo "Docker not accessible"
else
    echo "Docker not installed"
fi

section "Step 8: Check Log Files"
echo "Large log files (>100MB):"
find /var/log -type f -size +100M -exec ls -lh {} \; 2>/dev/null || \
    echo "No large log files or permission denied"

section "Analysis Summary"
echo -e "${BOLD}Key Findings:${NC}"
echo ""

# Calculate percentages
root_usage=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$root_usage" -gt 90 ]; then
    echo -e "${RED}✗ CRITICAL: Root filesystem is ${root_usage}% full${NC}"
elif [ "$root_usage" -gt 80 ]; then
    echo -e "${YELLOW}⚠ WARNING: Root filesystem is ${root_usage}% full${NC}"
else
    echo -e "${GREEN}✓ Root filesystem usage is acceptable (${root_usage}%)${NC}"
fi

checkpoint_size=$(du -sh /var/ml/checkpoints 2>/dev/null | awk '{print $1}')
if [ -n "$checkpoint_size" ]; then
    echo -e "  Checkpoint directory size: ${checkpoint_size}"
fi

if [ "$old_count" -gt 0 ]; then
    echo -e "${YELLOW}  Found $old_count old checkpoint files (>7 days)${NC}"
fi

echo ""
echo -e "${BOLD}${BLUE}Next Steps:${NC}"
echo "  1. Review the space usage above"
echo "  2. Identify what's consuming space"
echo "  3. Run the cleanup script: ./cleanup.sh"
echo "  4. Implement prevention measures"
echo ""
