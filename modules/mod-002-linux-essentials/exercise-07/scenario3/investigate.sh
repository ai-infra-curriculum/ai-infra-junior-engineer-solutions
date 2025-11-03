#!/bin/bash
###############################################################################
# Scenario 3: Hung Process - Investigation Script
###############################################################################
#
# Problem: Training process appears hung and not responding
# Symptoms: High CPU but no progress, or process stuck in uninterruptible sleep
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
echo -e "${BOLD}${RED}║  Scenario 3: Hung Process Investigation                   ║${NC}"
echo -e "${BOLD}${RED}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Problem:${NC} Training process appears hung or unresponsive"
echo ""

# Get PID from argument or search for training processes
TARGET_PID="${1:-}"

if [ -z "$TARGET_PID" ]; then
    section "Finding Training Processes"
    echo "Searching for Python/training processes..."
    echo ""

    ps aux | grep -E "python|train" | grep -v grep | head -10

    echo ""
    log_info "Please run this script with a PID:"
    log_info "  $0 <PID>"
    echo ""
    exit 0
fi

# Verify PID exists
if ! ps -p "$TARGET_PID" > /dev/null 2>&1; then
    log_error "Process $TARGET_PID does not exist"
    exit 1
fi

log_info "Investigating PID: $TARGET_PID"
echo ""

section "Step 1: Basic Process Information"
echo "Command: ps -p $TARGET_PID -f"
echo ""
ps -p "$TARGET_PID" -f

echo ""
subsection "Detailed process info:"
ps -p "$TARGET_PID" -o pid,ppid,user,stat,pcpu,pmem,vsz,rss,etime,cmd

section "Step 2: Process State"
echo "Understanding process state codes:"
log_info "R = Running or runnable (on run queue)"
log_info "S = Interruptible sleep (waiting for event)"
log_info "D = Uninterruptible sleep (usually I/O)"
log_info "Z = Zombie (terminated but not reaped)"
log_info "T = Stopped (by job control or debugging)"
echo ""

state=$(ps -p "$TARGET_PID" -o stat --no-headers)
log_info "Current state: $state"
echo ""

case "$state" in
    R*)
        log_success "Process is running"
        log_info "If it seems hung but is in R state, it may be in infinite loop"
        ;;
    S*)
        log_info "Process is sleeping (interruptible)"
        log_info "This is normal if waiting for I/O or events"
        ;;
    D*)
        log_error "Process is in uninterruptible sleep"
        log_warning "This usually indicates I/O wait or kernel operation"
        log_warning "Cannot be killed with regular signals!"
        ;;
    Z*)
        log_error "Process is a zombie"
        log_info "Parent process needs to reap it"
        ;;
    T*)
        log_warning "Process is stopped"
        log_info "May have been suspended with Ctrl+Z or SIGSTOP"
        ;;
    *)
        log_info "Unknown state: $state"
        ;;
esac

section "Step 3: CPU and Memory Usage"
echo "Resource utilization over time:"
echo ""

log_info "Current snapshot:"
ps -p "$TARGET_PID" -o pid,pcpu,pmem,vsz,rss,etime,cmd

echo ""
log_info "Watching for 10 seconds (sampling every 2s)..."
for i in {1..5}; do
    sleep 2
    cpu=$(ps -p "$TARGET_PID" -o pcpu --no-headers 2>/dev/null | tr -d ' ')
    mem=$(ps -p "$TARGET_PID" -o pmem --no-headers 2>/dev/null | tr -d ' ')
    if [ -n "$cpu" ]; then
        echo "  Sample $i: CPU=${cpu}%, MEM=${mem}%"
    else
        log_error "Process terminated during monitoring"
        exit 0
    fi
done

section "Step 4: What is the Process Doing?"
subsection "Open files and file descriptors:"
echo ""

if command -v lsof &>/dev/null; then
    lsof -p "$TARGET_PID" 2>/dev/null | head -20

    echo ""
    log_info "File descriptor count:"
    fd_count=$(lsof -p "$TARGET_PID" 2>/dev/null | wc -l)
    log_info "  $fd_count open file descriptors"

    if [ "$fd_count" -gt 1000 ]; then
        log_warning "High number of open files!"
    fi
else
    log_warning "lsof not available"
    log_info "Alternative: ls -la /proc/$TARGET_PID/fd/"
    ls -la "/proc/$TARGET_PID/fd/" 2>/dev/null | head -20
fi

section "Step 5: System Calls (What is it waiting for?)"
echo "Tracing system calls for 5 seconds..."
echo ""

if command -v strace &>/dev/null; then
    log_info "Running: timeout 5 strace -p $TARGET_PID"
    log_info "(This will show what the process is doing)"
    echo ""

    timeout 5 strace -p "$TARGET_PID" 2>&1 | head -30 || log_warning "strace failed (may need sudo)"

    echo ""
    log_info "Common patterns:"
    log_info "  futex() = Waiting on locks/synchronization"
    log_info "  read()/write() = I/O operations"
    log_info "  poll()/select() = Waiting for events"
    log_info "  recvfrom()/sendto() = Network I/O"
else
    log_warning "strace not available"
fi

section "Step 6: Stack Trace"
if [ -d "/proc/$TARGET_PID" ]; then
    subsection "Kernel stack trace:"
    echo ""

    if [ -r "/proc/$TARGET_PID/stack" ]; then
        cat "/proc/$TARGET_PID/stack"
    else
        log_warning "Cannot read /proc/$TARGET_PID/stack (may need sudo)"
        log_info "Run: sudo cat /proc/$TARGET_PID/stack"
    fi

    echo ""
    subsection "Thread information:"
    echo ""

    thread_count=$(ls "/proc/$TARGET_PID/task" 2>/dev/null | wc -l)
    log_info "Number of threads: $thread_count"

    if [ "$thread_count" -gt 1 ]; then
        log_info "Thread states:"
        for tid in /proc/$TARGET_PID/task/*; do
            if [ -d "$tid" ]; then
                tid_num=$(basename "$tid")
                state=$(cat "$tid/stat" 2>/dev/null | awk '{print $3}')
                name=$(cat "$tid/comm" 2>/dev/null)
                echo "    TID $tid_num ($name): $state"
            fi
        done
    fi
fi

section "Step 7: Check for Deadlocks"
subsection "Locks held by process:"
echo ""

if [ -r "/proc/$TARGET_PID/locks" ]; then
    locks=$(cat "/proc/$TARGET_PID/locks" 2>/dev/null)
    if [ -n "$locks" ]; then
        echo "$locks"
    else
        log_info "No locks held"
    fi
else
    log_warning "Cannot read lock information"
fi

echo ""
subsection "Process relationships (parent/children):"
echo ""

log_info "Parent process:"
ppid=$(ps -p "$TARGET_PID" -o ppid --no-headers | tr -d ' ')
if [ -n "$ppid" ]; then
    ps -p "$ppid" -o pid,user,cmd --no-headers
fi

echo ""
log_info "Child processes:"
pgrep -P "$TARGET_PID" &>/dev/null
if [ $? -eq 0 ]; then
    ps -p $(pgrep -P "$TARGET_PID" | tr '\n' ',') -o pid,user,stat,cmd --no-headers 2>/dev/null
else
    log_info "No child processes"
fi

section "Step 8: Network Connections"
subsection "Active network connections:"
echo ""

if command -v lsof &>/dev/null; then
    lsof -i -a -p "$TARGET_PID" 2>/dev/null || log_info "No network connections"
elif command -v netstat &>/dev/null; then
    netstat -anp 2>/dev/null | grep "$TARGET_PID" || log_info "No network connections"
else
    log_info "Check: ss -anp | grep $TARGET_PID"
fi

section "Step 9: Recent Log Activity"
subsection "Check if process is logging:"
echo ""

# Try to find log files
log_info "Looking for log files..."
lsof -p "$TARGET_PID" 2>/dev/null | grep -E "\.log|\.out" | head -5

echo ""
log_info "Check system logs for errors:"
log_info "  journalctl -f | grep $TARGET_PID"
log_info "  dmesg | tail -50"

section "Step 10: I/O Statistics"
if [ -r "/proc/$TARGET_PID/io" ]; then
    subsection "I/O statistics:"
    echo ""
    cat "/proc/$TARGET_PID/io"

    echo ""
    log_info "If 'rchar' and 'wchar' are not increasing, process may not be doing I/O"
else
    log_warning "Cannot read I/O statistics"
fi

section "Analysis Summary"
echo -e "${BOLD}Diagnosis:${NC}"
echo ""

state=$(ps -p "$TARGET_PID" -o stat --no-headers 2>/dev/null)
cpu=$(ps -p "$TARGET_PID" -o pcpu --no-headers 2>/dev/null | tr -d ' ' | cut -d'.' -f1)

if [ -z "$state" ]; then
    log_error "Process no longer exists"
elif [[ "$state" == D* ]]; then
    log_error "Process in uninterruptible sleep (D state)"
    echo ""
    echo "Likely causes:"
    echo "  - Waiting for disk I/O"
    echo "  - NFS mount issues"
    echo "  - Kernel bug"
    echo ""
    echo "Actions:"
    echo "  1. Check disk I/O: iostat -x 1"
    echo "  2. Check NFS mounts: mount | grep nfs"
    echo "  3. Check kernel messages: dmesg | tail -50"
    echo "  4. May need to wait or reboot if kernel issue"
elif [[ "$state" == Z* ]]; then
    log_error "Process is a zombie"
    echo ""
    echo "Actions:"
    echo "  1. Check parent process (PID $ppid)"
    echo "  2. Parent may need to be restarted"
    echo "  3. Zombie takes no resources, cosmetic issue only"
elif [[ "$state" == T* ]]; then
    log_warning "Process is stopped"
    echo ""
    echo "Actions:"
    echo "  1. Resume with: kill -CONT $TARGET_PID"
    echo "  2. Or: fg (if in current shell)"
elif [ "$cpu" -gt 90 ]; then
    log_warning "Process using high CPU but may be hung"
    echo ""
    echo "Likely causes:"
    echo "  - Infinite loop"
    echo "  - Busy-wait loop"
    echo "  - Deadlock with spinning"
    echo ""
    echo "Actions:"
    echo "  1. Review strace output above"
    echo "  2. Attach debugger to see where it's looping"
    echo "  3. Check for progress in output/logs"
else
    log_info "Process appears to be waiting normally"
    echo ""
    echo "Verify:"
    echo "  1. Check if process is making progress (logs, output files)"
    echo "  2. Monitor for a longer period"
    echo "  3. Check if it's waiting for external resource (network, GPU)"
fi

echo ""
echo -e "${BOLD}${BLUE}Next Steps:${NC}"
echo "  1. Review the analysis above"
echo "  2. Determine if process is truly hung or just slow"
echo "  3. Try graceful termination: kill -TERM $TARGET_PID"
echo "  4. If needed, force kill: kill -KILL $TARGET_PID"
echo "  5. Run the kill script: ./kill.sh $TARGET_PID"
echo ""
