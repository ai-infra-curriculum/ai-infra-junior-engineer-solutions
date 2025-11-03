#!/bin/bash
###############################################################################
# Scenario 4: Out of Memory - Investigation Script
###############################################################################
#
# Problem: Training process killed by OOM killer
# Symptoms: Process terminates unexpectedly, "Killed" message, OOM in dmesg
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
log_error() { echo -e "  ${RED}✗${NC} $*"; }
log_success() { echo -e "  ${GREEN}✓${NC} $*"; }
log_warning() { echo -e "  ${YELLOW}⚠${NC} $*"; }

echo -e "${BOLD}${RED}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${RED}║  Scenario 4: Out of Memory Investigation                  ║${NC}"
echo -e "${BOLD}${RED}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Problem:${NC} Process killed due to insufficient memory"
echo ""

section "Step 1: Current Memory Status"
echo "Command: free -h"
echo ""
free -h

echo ""
subsection "Memory breakdown:"
total_mem=$(free -m | awk 'NR==2 {print $2}')
used_mem=$(free -m | awk 'NR==2 {print $3}')
free_mem=$(free -m | awk 'NR==2 {print $4}')
available_mem=$(free -m | awk 'NR==2 {print $7}')

log_info "Total: ${total_mem}MB"
log_info "Used: ${used_mem}MB"
log_info "Free: ${free_mem}MB"
log_info "Available: ${available_mem}MB"

# Calculate percentage
used_pct=$((used_mem * 100 / total_mem))
if [ $used_pct -gt 90 ]; then
    log_error "Memory usage is critically high: ${used_pct}%"
elif [ $used_pct -gt 80 ]; then
    log_warning "Memory usage is high: ${used_pct}%"
else
    log_success "Memory usage is acceptable: ${used_pct}%"
fi

section "Step 2: Check for OOM Killer Events"
echo "Searching kernel logs for OOM killer messages..."
echo ""

oom_events=$(dmesg -T 2>/dev/null | grep -i "out of memory" | tail -10)

if [ -n "$oom_events" ]; then
    log_error "OOM killer events found!"
    echo ""
    echo "$oom_events"
    echo ""

    log_info "Most recent OOM victim:"
    dmesg -T 2>/dev/null | grep -i "killed process" | tail -1
else
    log_info "No recent OOM killer events in dmesg"
fi

echo ""
subsection "Checking system logs:"
if command -v journalctl &>/dev/null; then
    journal_oom=$(journalctl -k --since "24 hours ago" | grep -i "out of memory" | wc -l)
    if [ "$journal_oom" -gt 0 ]; then
        log_warning "Found $journal_oom OOM events in last 24 hours"
        echo ""
        journalctl -k --since "24 hours ago" | grep -i "out of memory" | tail -5
    else
        log_info "No OOM events in journalctl (last 24h)"
    fi
fi

section "Step 3: Top Memory Consumers"
echo "Current top 10 processes by memory usage:"
echo ""

ps aux --sort=-%mem | head -11

echo ""
subsection "ML/Python processes:"
ps aux | grep -E "python|train" | grep -v grep | head -5 || log_info "No Python processes found"

section "Step 4: Swap Space Status"
echo "Command: swapon --show"
echo ""

if swapon --show | grep -q .; then
    swapon --show
    echo ""

    swap_total=$(free -m | awk 'NR==3 {print $2}')
    swap_used=$(free -m | awk 'NR==3 {print $3}')

    if [ "$swap_total" -gt 0 ]; then
        swap_pct=$((swap_used * 100 / swap_total))
        log_info "Swap usage: ${swap_used}MB / ${swap_total}MB (${swap_pct}%)"

        if [ "$swap_pct" -gt 50 ]; then
            log_warning "Heavy swap usage - system may be thrashing"
        fi
    fi
else
    log_warning "No swap space configured!"
    log_info "Consider adding swap for memory buffer"
fi

section "Step 5: Memory Limits and Cgroups"
subsection "System memory limits:"
echo ""

if [ -f /proc/sys/vm/overcommit_memory ]; then
    overcommit=$(cat /proc/sys/vm/overcommit_memory)
    log_info "Overcommit mode: $overcommit"

    case $overcommit in
        0) log_info "  (Heuristic overcommit - default)" ;;
        1) log_info "  (Always overcommit - dangerous for ML)" ;;
        2) log_info "  (Never overcommit - conservative)" ;;
    esac
fi

echo ""
subsection "If using Docker/containers, check limits:"
log_info "docker stats --no-stream"

if command -v docker &>/dev/null && docker ps &>/dev/null 2>&1; then
    docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null | head -5
else
    log_info "Docker not available or no running containers"
fi

section "Step 6: Memory Fragmentation"
if [ -r /proc/buddyinfo ]; then
    echo "Buddy system info (memory fragmentation):"
    echo ""
    cat /proc/buddyinfo
    echo ""
    log_info "More free blocks in lower orders = more fragmentation"
else
    log_warning "Cannot read /proc/buddyinfo"
fi

section "Step 7: Virtual Memory Statistics"
echo "Command: vmstat 1 5"
echo ""
log_info "Monitoring for 5 seconds..."
vmstat 1 5

echo ""
log_info "Key columns:"
log_info "  si/so = Swap in/out (should be low)"
log_info "  bi/bo = Block in/out (disk I/O)"
log_info "  us/sy = User/system CPU time"
log_info "  wa = I/O wait (high if swapping)"

section "Step 8: Check Memory-Mapped Files"
subsection "Shared memory segments:"
echo ""
ipcs -m

echo ""
subsection "System V IPC usage:"
ipcs -u 2>/dev/null || log_info "No IPC usage stats available"

section "Step 9: Historical Memory Usage"
if command -v sar &>/dev/null; then
    echo "Memory usage trends (last hour):"
    echo ""
    sar -r 1 10 2>/dev/null || log_warning "sar data not available (install sysstat)"
else
    log_info "sar not available (install sysstat package for historical data)"
fi

section "Step 10: Application-Specific Checks"
subsection "Python/ML framework memory:"
echo ""

log_info "If using PyTorch, check:"
log_info "  - torch.cuda.memory_summary()"
log_info "  - Batch size too large"
log_info "  - Model size vs available RAM"
echo ""

log_info "If using TensorFlow, check:"
log_info "  - GPU memory growth settings"
log_info "  - Per-process GPU memory fraction"
echo ""

log_info "Common ML memory issues:"
log_info "  - Loading entire dataset into memory"
log_info "  - Too large batch size"
log_info "  - Memory leaks in training loop"
log_info "  - Not clearing GPU cache between runs"

section "Analysis Summary"
echo -e "${BOLD}Diagnosis:${NC}"
echo ""

# Analyze findings
issues_found=0

if [ -n "$oom_events" ]; then
    log_error "OOM killer has been triggered recently"
    issues_found=$((issues_found + 1))
fi

if [ $used_pct -gt 85 ]; then
    log_error "System memory is critically low"
    issues_found=$((issues_found + 1))
fi

if ! swapon --show | grep -q .; then
    log_warning "No swap space configured"
    issues_found=$((issues_found + 1))
elif [ "$swap_pct" -gt 50 ]; then
    log_warning "Heavy swap usage detected"
    issues_found=$((issues_found + 1))
fi

if [ $issues_found -eq 0 ]; then
    log_success "No immediate memory issues detected"
    echo ""
    log_info "If you experienced OOM, it may have been temporary"
    log_info "Check application logs and review resource requirements"
else
    echo ""
    echo "Recommendations:"

    if [ -n "$oom_events" ]; then
        echo "  1. Reduce application memory usage"
        echo "     - Decrease batch size"
        echo "     - Use data generators instead of loading all data"
        echo "     - Enable gradient checkpointing"
    fi

    if [ $used_pct -gt 85 ]; then
        echo "  2. Increase available memory"
        echo "     - Add more RAM"
        echo "     - Add swap space"
        echo "     - Close unnecessary applications"
    fi

    if ! swapon --show | grep -q .; then
        echo "  3. Add swap space as buffer"
        echo "     - sudo fallocate -l 8G /swapfile"
        echo "     - sudo chmod 600 /swapfile"
        echo "     - sudo mkswap /swapfile"
        echo "     - sudo swapon /swapfile"
    fi
fi

echo ""
echo -e "${BOLD}${BLUE}Next Steps:${NC}"
echo "  1. Review the memory analysis above"
echo "  2. Identify memory-intensive processes"
echo "  3. Reduce application memory footprint"
echo "  4. Run the optimization script: ./optimize.sh"
echo "  5. Consider adding more RAM or swap space"
echo ""
