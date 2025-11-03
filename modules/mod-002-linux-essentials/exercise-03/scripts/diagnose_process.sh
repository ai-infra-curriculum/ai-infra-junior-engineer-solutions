#!/bin/bash
#
# diagnose_process.sh - Comprehensive process diagnostic
#
# Usage: ./diagnose_process.sh <PID>
#

set -e
set -u

PID="${1:-}"

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

if [ -z "$PID" ]; then
    echo -e "${RED}Error: PID required${NC}" >&2
    echo ""
    echo "Usage: $0 <PID>" >&2
    echo ""
    echo "Example:" >&2
    echo "  $0 12345" >&2
    exit 1
fi

# Verify PID is numeric
if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Invalid PID: $PID${NC}" >&2
    exit 1
fi

# Check if process exists
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${RED}Error: Process $PID not found${NC}" >&2
    exit 1
fi

echo -e "${BLUE}=== Process Diagnostics for PID: $PID ===${NC}"
echo ""

# Basic process information
echo -e "${BLUE}Process Information:${NC}"
ps -p "$PID" -o pid,ppid,user,%cpu,%mem,vsz,rss,stat,start,etime,time,nice,pri,cmd --no-headers | \
    awk '{printf "  PID:        %s\n  PPID:       %s\n  User:       %s\n  CPU:        %s%%\n  Memory:     %s%%\n  VSZ:        %s KB\n  RSS:        %s KB\n  State:      %s\n  Start:      %s\n  Elapsed:    %s\n  CPU Time:   %s\n  Nice:       %s\n  Priority:   %s\n  Command:    ", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13; for(i=14;i<=NF;i++) printf "%s ", $i; print ""}'
echo ""

# Process tree
echo -e "${BLUE}Process Tree:${NC}"
if command -v pstree &> /dev/null; then
    pstree -p "$PID" 2>/dev/null | sed 's/^/  /' || echo "  N/A"
else
    ps -p "$PID" -o pid,cmd | sed 's/^/  /'
fi
echo ""

# Open files (limited)
echo -e "${BLUE}Open Files (first 20):${NC}"
if command -v lsof &> /dev/null; then
    if lsof -p "$PID" 2>/dev/null | head -20 | tail -n +2 | sed 's/^/  /'; then
        :
    else
        echo "  N/A or permission denied"
    fi
else
    echo "  lsof not available"
fi
echo ""

# Network connections
echo -e "${BLUE}Network Connections:${NC}"
if command -v lsof &> /dev/null; then
    if CONNECTIONS=$(lsof -i -a -p "$PID" 2>/dev/null); then
        echo "$CONNECTIONS" | tail -n +2 | sed 's/^/  /' || echo "  None"
    else
        echo "  None or permission denied"
    fi
else
    echo "  lsof not available"
fi
echo ""

# Process limits
echo -e "${BLUE}Resource Limits:${NC}"
if [ -r "/proc/$PID/limits" ]; then
    cat "/proc/$PID/limits" 2>/dev/null | sed 's/^/  /' || echo "  N/A"
else
    echo "  Permission denied"
fi
echo ""

# Current working directory
echo -e "${BLUE}Current Working Directory:${NC}"
if CWD=$(readlink "/proc/$PID/cwd" 2>/dev/null); then
    echo "  $CWD"
else
    echo "  N/A or permission denied"
fi
echo ""

# Command line
echo -e "${BLUE}Command Line:${NC}"
if CMD=$(cat "/proc/$PID/cmdline" 2>/dev/null | tr '\0' ' '); then
    echo "  $CMD"
else
    echo "  N/A or permission denied"
fi
echo ""

# Environment variables (first 10)
echo -e "${BLUE}Environment Variables (first 10):${NC}"
if [ -r "/proc/$PID/environ" ]; then
    cat "/proc/$PID/environ" 2>/dev/null | tr '\0' '\n' | head -10 | sed 's/^/  /' || echo "  N/A"
else
    echo "  Permission denied"
fi
echo ""

# Process state details
echo -e "${BLUE}Process State Details:${NC}"
if [ -r "/proc/$PID/status" ]; then
    grep -E "^(State|Threads|VmPeak|VmSize|VmRSS|VmSwap|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):" "/proc/$PID/status" 2>/dev/null | sed 's/^/  /' || echo "  N/A"
else
    echo "  Permission denied"
fi
echo ""

# Thread information
echo -e "${BLUE}Threads:${NC}"
if [ -d "/proc/$PID/task" ]; then
    NUM_THREADS=$(ls "/proc/$PID/task" 2>/dev/null | wc -l)
    echo "  Total threads: $NUM_THREADS"

    if [ "$NUM_THREADS" -gt 1 ] && [ "$NUM_THREADS" -le 10 ]; then
        echo "  Thread IDs:"
        ls "/proc/$PID/task" 2>/dev/null | sed 's/^/    /'
    fi
else
    echo "  N/A"
fi
echo ""

# Stack trace (requires sudo, may not work)
echo -e "${BLUE}Stack Trace:${NC}"
if [ -r "/proc/$PID/stack" ]; then
    cat "/proc/$PID/stack" 2>/dev/null | sed 's/^/  /' || echo "  Permission denied (requires root)"
else
    echo "  Permission denied (requires root)"
fi
echo ""

# Recommendations
echo -e "${BLUE}=== Diagnostic Summary ===${NC}"
echo ""

# Check state
STATE=$(ps -p "$PID" -o stat --no-headers | tr -d ' ')
case "$STATE" in
    R*)
        echo -e "${GREEN}✓${NC} Process is running normally"
        ;;
    S*)
        echo -e "${BLUE}ℹ${NC} Process is sleeping (waiting for event)"
        echo "  This is normal for most processes"
        ;;
    D*)
        echo -e "${YELLOW}⚠${NC} Process in uninterruptible sleep"
        echo "  Usually waiting for I/O"
        echo "  Check: disk I/O, network, NFS mounts"
        ;;
    T*)
        echo -e "${YELLOW}⚠${NC} Process is stopped"
        echo "  Resume with: kill -CONT $PID"
        ;;
    Z*)
        echo -e "${RED}✗${NC} Process is zombie"
        echo "  Parent process needs to reap it"
        PPID=$(ps -p "$PID" -o ppid --no-headers | tr -d ' ')
        echo "  Parent PID: $PPID"
        echo "  Try: kill -9 $PPID"
        ;;
    *)
        echo -e "${YELLOW}⚠${NC} Unknown state: $STATE"
        ;;
esac

echo ""
