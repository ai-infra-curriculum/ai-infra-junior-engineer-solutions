#!/bin/bash
###############################################################################
# Scenario 3: Hung Process - Termination Script
###############################################################################
#
# Usage: ./kill.sh <PID> [--force] [--timeout SECONDS]
#
# This script attempts graceful termination first, then escalates if needed.
#

set -u

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ${NC} $*"; }
log_success() { echo -e "${GREEN}✓${NC} $*"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $*"; }
log_error() { echo -e "${RED}✗${NC} $*"; }

# Default values
TARGET_PID=""
FORCE=false
TIMEOUT=10
KILL_CHILDREN=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --timeout|-t)
            TIMEOUT="$2"
            shift 2
            ;;
        --children|-c)
            KILL_CHILDREN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            cat << EOF
Usage: $0 <PID> [OPTIONS]

Gracefully terminate a process with escalation if needed.

Arguments:
  PID                 Process ID to terminate

Options:
  --force, -f         Skip graceful termination, go straight to SIGKILL
  --timeout, -t SEC   Seconds to wait for graceful termination (default: 10)
  --children, -c      Also terminate child processes
  --dry-run           Show what would be done without doing it
  -h, --help          Show this help message

Signal Escalation:
  1. SIGTERM (15)     - Graceful termination (default)
  2. SIGINT (2)       - Interrupt (like Ctrl+C)
  3. SIGQUIT (3)      - Quit with core dump
  4. SIGKILL (9)      - Force kill (cannot be caught)

Examples:
  $0 12345                    # Graceful termination
  $0 12345 --timeout 30       # Wait 30 seconds
  $0 12345 --force            # Immediate force kill
  $0 12345 --children         # Kill process tree

EOF
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            TARGET_PID="$1"
            shift
            ;;
    esac
done

if [ -z "$TARGET_PID" ]; then
    log_error "No PID provided"
    echo "Usage: $0 <PID> [OPTIONS]"
    exit 1
fi

# Validate PID is a number
if ! [[ "$TARGET_PID" =~ ^[0-9]+$ ]]; then
    log_error "Invalid PID: $TARGET_PID"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Process Termination Utility                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No processes will be killed"
    echo ""
fi

# Check if process exists
if ! ps -p "$TARGET_PID" > /dev/null 2>&1; then
    log_error "Process $TARGET_PID does not exist"
    exit 1
fi

# Get process information
log_info "Target process:"
ps -p "$TARGET_PID" -o pid,user,stat,pcpu,pmem,etime,cmd
echo ""

# Check process state
state=$(ps -p "$TARGET_PID" -o stat --no-headers)
if [[ "$state" == D* ]]; then
    log_warning "Process is in uninterruptible sleep (D state)"
    log_warning "Cannot be killed with signals!"
    echo ""
    echo "This process is waiting for kernel operation (usually I/O)."
    echo "Options:"
    echo "  1. Wait for operation to complete"
    echo "  2. Fix underlying issue (disk, NFS mount, etc.)"
    echo "  3. Reboot system (last resort)"
    echo ""
    exit 1
fi

if [[ "$state" == Z* ]]; then
    log_warning "Process is already a zombie (terminated)"
    log_info "The parent process needs to reap it"
    ppid=$(ps -p "$TARGET_PID" -o ppid --no-headers | tr -d ' ')
    echo ""
    log_info "Parent process (PID $ppid):"
    ps -p "$ppid" -o pid,user,cmd --no-headers 2>/dev/null
    echo ""
    echo "Consider restarting the parent process."
    exit 1
fi

if [[ "$state" == T* ]]; then
    log_info "Process is stopped (T state)"
    log_info "Resuming before termination..."

    if [ "$DRY_RUN" = false ]; then
        kill -CONT "$TARGET_PID"
        sleep 1
    else
        log_info "Would run: kill -CONT $TARGET_PID"
    fi
fi

# Find child processes
if [ "$KILL_CHILDREN" = true ]; then
    children=$(pgrep -P "$TARGET_PID" 2>/dev/null | tr '\n' ' ')
    if [ -n "$children" ]; then
        log_info "Child processes found: $children"
        echo ""
        ps -p $children -o pid,user,stat,cmd --no-headers 2>/dev/null
        echo ""
    fi
fi

# Function to check if process is alive
is_alive() {
    ps -p "$1" > /dev/null 2>&1
}

# Function to wait for process to die
wait_for_death() {
    local pid=$1
    local timeout=$2
    local elapsed=0

    while is_alive "$pid" && [ $elapsed -lt $timeout ]; do
        sleep 1
        elapsed=$((elapsed + 1))
        echo -n "."
    done
    echo ""

    ! is_alive "$pid"
}

# Termination sequence
if [ "$FORCE" = true ]; then
    log_warning "Force kill requested"
    echo ""

    log_info "Sending SIGKILL to PID $TARGET_PID..."

    if [ "$DRY_RUN" = false ]; then
        kill -9 "$TARGET_PID" 2>/dev/null

        if [ "$KILL_CHILDREN" = true ] && [ -n "$children" ]; then
            kill -9 $children 2>/dev/null
            log_info "Sent SIGKILL to children: $children"
        fi

        sleep 1

        if ! is_alive "$TARGET_PID"; then
            log_success "Process terminated"
        else
            log_error "Process still alive (may be in D state)"
        fi
    else
        log_info "Would run: kill -9 $TARGET_PID"
    fi
else
    # Graceful termination sequence
    log_info "Attempting graceful termination..."
    echo ""

    # Step 1: SIGTERM (standard termination)
    log_info "Step 1: Sending SIGTERM (signal 15)..."

    if [ "$DRY_RUN" = false ]; then
        kill -TERM "$TARGET_PID" 2>/dev/null

        if [ "$KILL_CHILDREN" = true ] && [ -n "$children" ]; then
            kill -TERM $children 2>/dev/null
            log_info "Sent SIGTERM to children: $children"
        fi

        echo -n "  Waiting up to $TIMEOUT seconds"
        if wait_for_death "$TARGET_PID" "$TIMEOUT"; then
            log_success "Process terminated gracefully"
            exit 0
        fi

        log_warning "Process still alive after SIGTERM"
    else
        log_info "Would run: kill -TERM $TARGET_PID"
    fi

    echo ""

    # Step 2: SIGINT (interrupt)
    log_info "Step 2: Sending SIGINT (signal 2)..."

    if [ "$DRY_RUN" = false ]; then
        kill -INT "$TARGET_PID" 2>/dev/null

        echo -n "  Waiting 5 seconds"
        if wait_for_death "$TARGET_PID" 5; then
            log_success "Process terminated after SIGINT"
            exit 0
        fi

        log_warning "Process still alive after SIGINT"
    else
        log_info "Would run: kill -INT $TARGET_PID"
    fi

    echo ""

    # Step 3: SIGQUIT (quit with core dump)
    log_info "Step 3: Sending SIGQUIT (signal 3)..."
    log_info "  This will generate a core dump"

    if [ "$DRY_RUN" = false ]; then
        kill -QUIT "$TARGET_PID" 2>/dev/null

        echo -n "  Waiting 5 seconds"
        if wait_for_death "$TARGET_PID" 5; then
            log_success "Process terminated after SIGQUIT"

            # Check for core dump
            cwd=$(readlink -f "/proc/$TARGET_PID/cwd" 2>/dev/null)
            if [ -n "$cwd" ] && [ -f "$cwd/core" ]; then
                log_info "Core dump created: $cwd/core"
            fi
            exit 0
        fi

        log_warning "Process still alive after SIGQUIT"
    else
        log_info "Would run: kill -QUIT $TARGET_PID"
    fi

    echo ""

    # Step 4: SIGKILL (force kill)
    log_warning "Step 4: Force termination with SIGKILL (signal 9)..."
    log_warning "  This cannot be caught or ignored"

    if [ "$DRY_RUN" = false ]; then
        kill -9 "$TARGET_PID" 2>/dev/null

        if [ "$KILL_CHILDREN" = true ] && [ -n "$children" ]; then
            kill -9 $children 2>/dev/null
            log_info "Sent SIGKILL to children: $children"
        fi

        sleep 2

        if ! is_alive "$TARGET_PID"; then
            log_success "Process force terminated"
        else
            log_error "Process STILL alive after SIGKILL!"
            log_error "Process may be in uninterruptible sleep (D state)"
            echo ""
            echo "Final state:"
            ps -p "$TARGET_PID" -o pid,stat,wchan,cmd
            echo ""
            echo "This process cannot be killed normally."
            echo "Underlying issue must be resolved or system rebooted."
            exit 1
        fi
    else
        log_info "Would run: kill -9 $TARGET_PID"
    fi
fi

echo ""
log_success "Termination complete!"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Post-Termination Actions                                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Investigate root cause:"
echo "   - Check logs for errors"
echo "   - Review what the process was doing (investigate.sh output)"
echo "   - Identify what caused the hang"
echo ""
echo "2. Prevent recurrence:"
echo "   - Add timeouts to long-running operations"
echo "   - Implement health checks and auto-restart"
echo "   - Monitor for deadlocks in code"
echo ""
echo "3. Clean up resources:"
echo "   - Check for orphaned locks: ipcs -a"
echo "   - Check for shared memory: ipcs -m"
echo "   - Clean up temp files"
echo ""
echo "4. If process was critical:"
echo "   - Restart the service"
echo "   - Check for data corruption"
echo "   - Restore from checkpoint if needed"
echo ""
