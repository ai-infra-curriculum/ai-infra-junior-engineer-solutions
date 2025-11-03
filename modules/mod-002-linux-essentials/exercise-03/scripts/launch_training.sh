#!/bin/bash
#
# launch_training.sh - Launch training in persistent session
#
# Usage: ./launch_training.sh <experiment_name> <training_script> [session_type]
#
# Session types: tmux (default) or screen
#

set -e
set -u

EXPERIMENT_NAME="${1:-}"
TRAINING_SCRIPT="${2:-train_model.py}"
SESSION_TYPE="${3:-tmux}"

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

# Validate experiment name
if [ -z "$EXPERIMENT_NAME" ]; then
    echo -e "${RED}Error: Experiment name required${NC}" >&2
    echo ""
    echo "Usage: $0 <experiment_name> [training_script] [session_type]" >&2
    echo ""
    echo "Examples:" >&2
    echo "  $0 experiment001" >&2
    echo "  $0 experiment001 train.py tmux" >&2
    echo "  $0 experiment001 train.py screen" >&2
    exit 1
fi

# Check if training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo -e "${RED}Error: Training script not found: $TRAINING_SCRIPT${NC}" >&2
    exit 1
fi

# Auto-detect session type availability
if [ "$SESSION_TYPE" = "tmux" ]; then
    if ! command -v tmux &> /dev/null; then
        echo -e "${YELLOW}tmux not installed, using screen${NC}"
        SESSION_TYPE="screen"
    fi
fi

if [ "$SESSION_TYPE" = "screen" ]; then
    if ! command -v screen &> /dev/null; then
        echo -e "${RED}Error: Neither tmux nor screen is installed${NC}" >&2
        echo ""
        echo "Install with:" >&2
        echo "  sudo apt install tmux screen" >&2
        exit 1
    fi
fi

echo -e "${BLUE}=== Launch Training in Persistent Session ===${NC}"
echo ""
echo -e "${BLUE}Experiment:${NC}  $EXPERIMENT_NAME"
echo -e "${BLUE}Script:${NC}      $TRAINING_SCRIPT"
echo -e "${BLUE}Session type:${NC} $SESSION_TYPE"
echo ""

# Get absolute path to script
SCRIPT_PATH="$(cd "$(dirname "$TRAINING_SCRIPT")" && pwd)/$(basename "$TRAINING_SCRIPT")"

if [ "$SESSION_TYPE" = "tmux" ]; then
    # Check if session already exists
    if tmux has-session -t "$EXPERIMENT_NAME" 2>/dev/null; then
        echo -e "${YELLOW}Session '$EXPERIMENT_NAME' already exists${NC}"
        echo ""
        echo "Options:"
        echo "  1. Attach to existing session: tmux attach -t $EXPERIMENT_NAME"
        echo "  2. Kill existing session: tmux kill-session -t $EXPERIMENT_NAME"
        exit 1
    fi

    # Launch with tmux
    echo -e "${GREEN}Creating tmux session: $EXPERIMENT_NAME${NC}"
    tmux new-session -d -s "$EXPERIMENT_NAME"
    tmux send-keys -t "$EXPERIMENT_NAME" "cd $(pwd)" C-m
    tmux send-keys -t "$EXPERIMENT_NAME" "python3 $SCRIPT_PATH" C-m

    echo ""
    echo -e "${GREEN}✓ Training started in tmux session${NC}"
    echo ""
    echo -e "${BLUE}Commands:${NC}"
    echo "  Attach:  tmux attach -t $EXPERIMENT_NAME"
    echo "  Detach:  Ctrl+b d  (from inside session)"
    echo "  List:    tmux ls"
    echo "  Kill:    tmux kill-session -t $EXPERIMENT_NAME"
    echo ""
    echo -e "${BLUE}Tmux Help:${NC}"
    echo "  Split horizontal:  Ctrl+b \""
    echo "  Split vertical:    Ctrl+b %"
    echo "  Navigate panes:    Ctrl+b arrow-keys"
    echo "  New window:        Ctrl+b c"
    echo "  Next window:       Ctrl+b n"

else
    # Check if session already exists
    if screen -ls | grep -q "$EXPERIMENT_NAME"; then
        echo -e "${YELLOW}Session '$EXPERIMENT_NAME' already exists${NC}"
        echo ""
        echo "Options:"
        echo "  1. Attach to existing session: screen -r $EXPERIMENT_NAME"
        echo "  2. Kill existing session: screen -X -S $EXPERIMENT_NAME quit"
        exit 1
    fi

    # Launch with screen
    echo -e "${GREEN}Creating screen session: $EXPERIMENT_NAME${NC}"
    screen -dmS "$EXPERIMENT_NAME" bash -c "cd $(pwd) && python3 $SCRIPT_PATH"

    echo ""
    echo -e "${GREEN}✓ Training started in screen session${NC}"
    echo ""
    echo -e "${BLUE}Commands:${NC}"
    echo "  Attach:  screen -r $EXPERIMENT_NAME"
    echo "  Detach:  Ctrl+a d  (from inside session)"
    echo "  List:    screen -ls"
    echo "  Kill:    screen -X -S $EXPERIMENT_NAME quit"
    echo ""
    echo -e "${BLUE}Screen Help:${NC}"
    echo "  New window:     Ctrl+a c"
    echo "  Next window:    Ctrl+a n"
    echo "  Previous:       Ctrl+a p"
    echo "  List windows:   Ctrl+a \""
fi

echo ""
echo -e "${BLUE}Current sessions:${NC}"
if [ "$SESSION_TYPE" = "tmux" ]; then
    tmux ls 2>/dev/null || echo "No tmux sessions"
else
    screen -ls || echo "No screen sessions"
fi
echo ""
