#!/bin/bash
#
# manage_training.sh - Training process management wrapper
#
# Usage:
#   ./manage_training.sh {start|stop|restart|status|log}
#
# Features:
# - Start training in background
# - Stop training gracefully (SIGTERM) or forcefully (SIGKILL)
# - Check training status
# - View training logs
# - PID file management
#

set -e
set -u

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/training.pid"
LOG_FILE="$LOG_DIR/training.log"
TRAINING_SCRIPT="$SCRIPT_DIR/train_model.py"

# Create log directory
mkdir -p "$LOG_DIR"

# Colors (if terminal supports it)
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'  # No Color
else
    GREEN='' RED='' YELLOW='' BLUE='' NC=''
fi

# Helper functions
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

error() {
    echo -e "${RED}✗ $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if training is running
is_running() {
    if [ ! -f "$PID_FILE" ]; then
        return 1
    fi

    local pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Get training PID
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

# Start training
start_training() {
    echo "Starting ML training process..."
    echo ""

    # Check if already running
    if is_running; then
        local pid=$(get_pid)
        error "Training already running (PID: $pid)"
        echo ""
        info "Use './manage_training.sh status' to check status"
        info "Use './manage_training.sh stop' to stop training"
        return 1
    fi

    # Check if training script exists
    if [ ! -f "$TRAINING_SCRIPT" ]; then
        error "Training script not found: $TRAINING_SCRIPT"
        return 1
    fi

    # Start training in background with nohup
    info "Starting training in background..."
    nohup python3 "$TRAINING_SCRIPT" --epochs 60 --checkpoint-interval 10 \
        > "$LOG_FILE" 2>&1 &

    local pid=$!

    # Save PID
    echo "$pid" > "$PID_FILE"

    # Wait a moment to ensure it started
    sleep 1

    # Verify it's running
    if ps -p "$pid" > /dev/null 2>&1; then
        echo ""
        success "Training started successfully"
        echo ""
        info "PID: $pid"
        info "Log file: $LOG_FILE"
        echo ""
        echo "Commands:"
        echo "  Status:  ./manage_training.sh status"
        echo "  Logs:    ./manage_training.sh log"
        echo "  Stop:    ./manage_training.sh stop"
    else
        error "Training failed to start"
        rm -f "$PID_FILE"
        echo ""
        info "Check log file for errors: $LOG_FILE"
        return 1
    fi
}

# Stop training
stop_training() {
    echo "Stopping ML training process..."
    echo ""

    # Check if PID file exists
    if [ ! -f "$PID_FILE" ]; then
        error "No training process found (PID file missing)"
        return 1
    fi

    local pid=$(get_pid)

    # Check if process is running
    if ! ps -p "$pid" > /dev/null 2>&1; then
        error "Training process not running (stale PID file)"
        rm -f "$PID_FILE"
        return 1
    fi

    info "Sending SIGTERM to PID $pid..."
    echo ""

    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid"

    # Wait for graceful shutdown (up to 10 seconds)
    local count=0
    while [ $count -lt 10 ]; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            success "Training stopped gracefully"
            rm -f "$PID_FILE"
            echo ""
            info "Checkpoints saved in current directory"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done

    echo ""
    echo ""
    warning "Process did not stop gracefully, force killing..."

    # Force kill if still running
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"

    # Verify termination
    sleep 1
    if ps -p "$pid" > /dev/null 2>&1; then
        error "Failed to stop training process"
        return 1
    else
        success "Training stopped (forced)"
        return 0
    fi
}

# Check training status
status_training() {
    echo "Training Process Status"
    echo "=" * 60

    if ! is_running; then
        echo ""
        info "Status: Not running"

        # Check for stale PID file
        if [ -f "$PID_FILE" ]; then
            warning "Stale PID file found, removing..."
            rm -f "$PID_FILE"
        fi

        # Check for log file
        if [ -f "$LOG_FILE" ]; then
            echo ""
            info "Last log entries:"
            tail -5 "$LOG_FILE"
        fi

        return 1
    fi

    local pid=$(get_pid)

    echo ""
    success "Status: Running"
    echo ""
    info "Process Details:"
    echo ""

    # Show process information
    ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers | while read line; do
        echo "  $line"
    done

    echo ""
    info "Working Directory:"
    echo "  $(readlink /proc/$pid/cwd 2>/dev/null || echo "N/A")"

    echo ""
    info "Log File: $LOG_FILE"

    # Show latest log entries
    if [ -f "$LOG_FILE" ]; then
        echo ""
        info "Latest Log Entries:"
        echo ""
        tail -10 "$LOG_FILE" | sed 's/^/  /'
    fi

    echo ""
}

# Tail log file
tail_log() {
    if [ ! -f "$LOG_FILE" ]; then
        error "Log file not found: $LOG_FILE"
        return 1
    fi

    info "Following log file: $LOG_FILE"
    echo "Press Ctrl+C to stop"
    echo ""

    tail -f "$LOG_FILE"
}

# Restart training
restart_training() {
    echo "Restarting training..."
    echo ""

    stop_training
    sleep 2
    start_training
}

# Show usage
usage() {
    cat << EOF
Usage: $0 {start|stop|restart|status|log}

Commands:
    start       Start training in background
    stop        Stop training (gracefully, then force if needed)
    restart     Stop and start training
    status      Show training process status
    log         Follow training log file

Examples:
    $0 start
    $0 status
    $0 log
    $0 stop

Files:
    PID file:   $PID_FILE
    Log file:   $LOG_FILE
    Script:     $TRAINING_SCRIPT
EOF
}

# Main command handler
case "${1:-}" in
    start)
        start_training
        ;;
    stop)
        stop_training
        ;;
    restart)
        restart_training
        ;;
    status)
        status_training
        ;;
    log)
        tail_log
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        error "Invalid command: ${1:-}"
        echo ""
        usage
        exit 1
        ;;
esac
