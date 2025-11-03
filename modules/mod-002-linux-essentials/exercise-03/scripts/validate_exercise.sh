#!/bin/bash
#
# validate_exercise.sh - Validate Exercise 03 completion
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

echo -e "${BLUE}=== Exercise 03 Validation ===${NC}"
echo ""

PASS=0
FAIL=0
WARN=0

# Check if we're in the right directory
if [ ! -f "train_model.py" ] || [ ! -f "manage_training.sh" ]; then
    echo -e "${RED}Error: Must run from scripts/ directory${NC}" >&2
    echo "cd to the scripts/ directory and try again" >&2
    exit 1
fi

# Test 1: Check required scripts exist
echo -e "${BLUE}[1/8] Checking required scripts...${NC}"
SCRIPTS=(
    "train_model.py"
    "manage_training.sh"
    "monitor_resources.sh"
    "analyze_resources.py"
    "monitor_gpu.sh"
    "gpu_process_manager.sh"
    "launch_training.sh"
    "diagnose_process.sh"
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
echo -e "${BLUE}[2/8] Checking script permissions...${NC}"
for script in "${SCRIPTS[@]}"; do
    if [ -x "$script" ]; then
        echo -e "  ${GREEN}✓${NC} $script is executable"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} $script not executable"
        ((FAIL++))
    fi
done
echo ""

# Test 3: Test training manager
echo -e "${BLUE}[3/8] Testing training manager...${NC}"
if ./manage_training.sh --help &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} manage_training.sh responds to --help"
    ((PASS++))
else
    echo -e "  ${RED}✗${NC} manage_training.sh --help failed"
    ((FAIL++))
fi
echo ""

# Test 4: Test Python scripts
echo -e "${BLUE}[4/8] Testing Python scripts...${NC}"
if python3 -c "import sys, signal, json, os, time, datetime" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Python dependencies available"
    ((PASS++))
else
    echo -e "  ${RED}✗${NC} Python dependencies missing"
    ((FAIL++))
fi

if python3 train_model.py --help &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} train_model.py responds to --help"
    ((PASS++))
else
    echo -e "  ${RED}✗${NC} train_model.py --help failed"
    ((FAIL++))
fi
echo ""

# Test 5: Test process monitoring
echo -e "${BLUE}[5/8] Testing process monitoring...${NC}"
# Start a test process
sleep 10 &
TEST_PID=$!
sleep 1

if ps -p "$TEST_PID" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Can check if process exists"
    ((PASS++))
else
    echo -e "  ${RED}✗${NC} Process check failed"
    ((FAIL++))
fi

# Kill test process
kill "$TEST_PID" 2>/dev/null
wait "$TEST_PID" 2>/dev/null || true
echo ""

# Test 6: Test signal handling
echo -e "${BLUE}[6/8] Testing signal handling...${NC}"
# Start training for 5 epochs
echo -e "  ${YELLOW}Starting test training (5 epochs)...${NC}"
python3 train_model.py --epochs 5 --checkpoint-interval 2 > /dev/null 2>&1 &
TRAIN_PID=$!
sleep 2

# Send SIGTERM
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    kill -TERM "$TRAIN_PID" 2>/dev/null
    sleep 2

    # Check if it terminated
    if ! ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Process responds to SIGTERM"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} Process did not respond to SIGTERM"
        kill -9 "$TRAIN_PID" 2>/dev/null
        ((FAIL++))
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Test process already terminated"
    ((WARN++))
fi

# Clean up any checkpoint files
rm -f checkpoint_*.json final_model.json 2>/dev/null
echo ""

# Test 7: Check for optional tools
echo -e "${BLUE}[7/8] Checking optional tools...${NC}"
if command -v htop &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} htop available"
else
    echo -e "  ${YELLOW}⚠${NC} htop not available (optional)"
    ((WARN++))
fi

if command -v tmux &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} tmux available"
else
    echo -e "  ${YELLOW}⚠${NC} tmux not available (optional)"
    ((WARN++))
fi

if command -v screen &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} screen available"
else
    echo -e "  ${YELLOW}⚠${NC} screen not available (optional)"
    ((WARN++))
fi

if command -v nvidia-smi &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} nvidia-smi available"
else
    echo -e "  ${YELLOW}⚠${NC} nvidia-smi not available (optional, for GPU monitoring)"
    ((WARN++))
fi
echo ""

# Test 8: Knowledge check
echo -e "${BLUE}[8/8] Knowledge Check Questions:${NC}"
echo ""
echo "Answer these questions to verify understanding:"
echo ""
echo "1. What's the difference between kill -15 and kill -9?"
echo "   kill -15 (SIGTERM): Graceful termination, allows cleanup"
echo "   kill -9 (SIGKILL): Force kill, cannot be caught or ignored"
echo ""
echo "2. What does the STAT column 'D' mean in ps output?"
echo "   D = Uninterruptible sleep (usually waiting for I/O)"
echo ""
echo "3. How do you detach from a screen session?"
echo "   Press: Ctrl+a d"
echo ""
echo "4. How do you detach from a tmux session?"
echo "   Press: Ctrl+b d"
echo ""
echo "5. What command shows real-time process monitoring?"
echo "   top or htop"
echo ""
echo "6. How do you send SIGTERM to a process?"
echo "   kill -TERM <PID> or kill -15 <PID> or kill <PID>"
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
    echo "Exercise 03 is complete. You have successfully:"
    echo "  • Created process management scripts"
    echo "  • Implemented training process control"
    echo "  • Set up resource monitoring"
    echo "  • Configured GPU monitoring (if available)"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some validations failed${NC}"
    echo ""
    echo "Fix the issues above and run validation again."
    echo ""
    exit 1
fi
