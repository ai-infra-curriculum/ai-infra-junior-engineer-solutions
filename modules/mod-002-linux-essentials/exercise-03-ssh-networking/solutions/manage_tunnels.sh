#!/bin/bash
#
# manage_tunnels.sh - SSH tunnel manager for ML infrastructure
#
# Description:
#   Manage multiple SSH tunnels for accessing remote ML services
#   (Jupyter, TensorBoard, databases, etc.) with auto-reconnect.
#
# Usage:
#   ./manage_tunnels.sh [ACTION] [TUNNEL_NAME]
#
# Actions:
#   start TUNNEL      Start a tunnel
#   stop TUNNEL       Stop a tunnel
#   restart TUNNEL    Restart a tunnel
#   status [TUNNEL]   Show tunnel status
#   list              List all tunnels
#   logs TUNNEL       Show tunnel logs
#
# Options:
#   -a, --all         Apply action to all tunnels
#   -v, --verbose     Verbose output
#   -h, --help        Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TUNNEL_DIR="$HOME/.ssh/tunnels"
readonly PID_DIR="$TUNNEL_DIR/pids"
readonly LOG_DIR="$TUNNEL_DIR/logs"
readonly CONFIG_FILE="$TUNNEL_DIR/tunnels.conf"

# Defaults
VERBOSE=false
ALL_TUNNELS=false

# ===========================
# Colors
# ===========================

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RESET='\033[0m'
readonly BOLD='\033[1m'

# ===========================
# Initialization
# ===========================

initialize() {
    # Create directories
    mkdir -p "$TUNNEL_DIR" "$PID_DIR" "$LOG_DIR"

    # Create default config if it doesn't exist
    if [[ ! -f "$CONFIG_FILE" ]]; then
        create_default_config
    fi
}

create_default_config() {
    cat > "$CONFIG_FILE" <<'EOF'
# SSH Tunnel Configuration
# Format: tunnel_name|ssh_host|local_port|remote_host|remote_port|description
#
# Examples:
jupyter-gpu1|gpu1.example.com|8888|localhost|8888|Jupyter on GPU1
tensorboard-gpu1|gpu1.example.com|6006|localhost|6006|TensorBoard on GPU1
postgres-dev|dev.example.com|5432|localhost|5432|PostgreSQL on Dev
redis-staging|staging.example.com|6379|localhost|6379|Redis on Staging

# ML Training Servers
# jupyter-train1|ml-train-1|8881|localhost|8888|Training Server 1 Jupyter
# jupyter-train2|ml-train-2|8882|localhost|8888|Training Server 2 Jupyter

# GPU Cluster
# jupyter-gpu2|gpu-node-2|8889|localhost|8888|GPU Node 2 Jupyter
# tensorboard-gpu2|gpu-node-2|6007|localhost|6006|GPU Node 2 TensorBoard

# Databases
# postgres-prod|prod-db.example.com|5433|localhost|5432|Production PostgreSQL
# mongo-dev|dev-db.example.com|27017|localhost|27017|Development MongoDB
EOF

    echo "Created default configuration: $CONFIG_FILE"
    echo "Edit this file to add your tunnels."
}

# ===========================
# Tunnel Management
# ===========================

get_tunnel_config() {
    local tunnel_name="$1"

    grep "^$tunnel_name|" "$CONFIG_FILE" 2>/dev/null || return 1
}

parse_tunnel_config() {
    local config="$1"

    local name=$(echo "$config" | cut -d'|' -f1)
    local ssh_host=$(echo "$config" | cut -d'|' -f2)
    local local_port=$(echo "$config" | cut -d'|' -f3)
    local remote_host=$(echo "$config" | cut -d'|' -f4)
    local remote_port=$(echo "$config" | cut -d'|' -f5)
    local description=$(echo "$config" | cut -d'|' -f6)

    echo "$name $ssh_host $local_port $remote_host $remote_port $description"
}

is_tunnel_running() {
    local tunnel_name="$1"
    local pid_file="$PID_DIR/${tunnel_name}.pid"

    if [[ ! -f "$pid_file" ]]; then
        return 1
    fi

    local pid=$(cat "$pid_file")

    if ps -p "$pid" > /dev/null 2>&1; then
        return 0
    else
        # Stale PID file
        rm -f "$pid_file"
        return 1
    fi
}

start_tunnel() {
    local tunnel_name="$1"

    # Get configuration
    local config=$(get_tunnel_config "$tunnel_name")
    if [[ -z "$config" ]]; then
        echo -e "${RED}Tunnel not found: $tunnel_name${RESET}"
        echo "Available tunnels:"
        list_tunnels
        return 1
    fi

    # Check if already running
    if is_tunnel_running "$tunnel_name"; then
        echo -e "${YELLOW}Tunnel already running: $tunnel_name${RESET}"
        return 0
    fi

    # Parse configuration
    read -r name ssh_host local_port remote_host remote_port description <<< "$(parse_tunnel_config "$config")"

    echo -e "${BLUE}Starting tunnel: $tunnel_name${RESET}"
    echo "  Description: $description"
    echo "  Forward: localhost:$local_port -> $ssh_host -> $remote_host:$remote_port"

    # Check if local port is already in use
    if lsof -i ":$local_port" > /dev/null 2>&1; then
        echo -e "${RED}Error: Local port $local_port already in use${RESET}"
        return 1
    fi

    # Start SSH tunnel
    local log_file="$LOG_DIR/${tunnel_name}.log"
    local pid_file="$PID_DIR/${tunnel_name}.pid"

    # SSH options:
    # -N: No command execution
    # -f: Background
    # -o ServerAliveInterval=60: Keep connection alive
    # -o ExitOnForwardFailure=yes: Exit if port forward fails
    # -L: Local port forward
    ssh -N -f \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=no \
        -L "$local_port:$remote_host:$remote_port" \
        "$ssh_host" \
        >> "$log_file" 2>&1

    # Find SSH process PID
    sleep 1
    local pid=$(pgrep -f "ssh.*-L $local_port:$remote_host:$remote_port.*$ssh_host" | head -1)

    if [[ -n "$pid" ]]; then
        echo "$pid" > "$pid_file"
        echo -e "${GREEN}✓ Tunnel started successfully (PID: $pid)${RESET}"
        echo "  Access at: localhost:$local_port"
        return 0
    else
        echo -e "${RED}Failed to start tunnel${RESET}"
        echo "Check log: $log_file"
        return 1
    fi
}

stop_tunnel() {
    local tunnel_name="$1"

    if ! is_tunnel_running "$tunnel_name"; then
        echo -e "${YELLOW}Tunnel not running: $tunnel_name${RESET}"
        return 0
    fi

    local pid_file="$PID_DIR/${tunnel_name}.pid"
    local pid=$(cat "$pid_file")

    echo -e "${BLUE}Stopping tunnel: $tunnel_name (PID: $pid)${RESET}"

    if kill "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        echo -e "${GREEN}✓ Tunnel stopped${RESET}"
        return 0
    else
        echo -e "${RED}Failed to stop tunnel${RESET}"
        return 1
    fi
}

restart_tunnel() {
    local tunnel_name="$1"

    echo -e "${BLUE}Restarting tunnel: $tunnel_name${RESET}"
    stop_tunnel "$tunnel_name"
    sleep 1
    start_tunnel "$tunnel_name"
}

# ===========================
# Status and Monitoring
# ===========================

show_tunnel_status() {
    local tunnel_name="$1"

    local config=$(get_tunnel_config "$tunnel_name")
    if [[ -z "$config" ]]; then
        echo -e "${RED}Tunnel not found: $tunnel_name${RESET}"
        return 1
    fi

    read -r name ssh_host local_port remote_host remote_port description <<< "$(parse_tunnel_config "$config")"

    echo -e "${BOLD}Tunnel: $tunnel_name${RESET}"
    echo "  Description: $description"
    echo "  Forward: localhost:$local_port -> $ssh_host -> $remote_host:$remote_port"

    if is_tunnel_running "$tunnel_name"; then
        local pid=$(cat "$PID_DIR/${tunnel_name}.pid")
        local uptime=$(ps -o etime= -p "$pid" | xargs)

        echo -e "  Status: ${GREEN}RUNNING${RESET}"
        echo "  PID: $pid"
        echo "  Uptime: $uptime"

        # Test connection
        if nc -z localhost "$local_port" 2>/dev/null; then
            echo -e "  Connection: ${GREEN}OK${RESET}"
        else
            echo -e "  Connection: ${RED}FAILED${RESET}"
        fi
    else
        echo -e "  Status: ${RED}STOPPED${RESET}"
    fi

    echo ""
}

show_all_status() {
    echo -e "${BOLD}${CYAN}SSH Tunnel Status${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    local running_count=0
    local stopped_count=0

    while IFS='|' read -r tunnel_name ssh_host local_port remote_host remote_port description; do
        # Skip comments and empty lines
        [[ "$tunnel_name" =~ ^#.*$ ]] && continue
        [[ -z "$tunnel_name" ]] && continue

        printf "%-20s " "$tunnel_name"

        if is_tunnel_running "$tunnel_name"; then
            local pid=$(cat "$PID_DIR/${tunnel_name}.pid")
            printf "${GREEN}%-10s${RESET} " "RUNNING"
            printf "PID: %-6s " "$pid"
            printf "Port: %s" "$local_port"
            ((running_count++))
        else
            printf "${RED}%-10s${RESET} " "STOPPED"
            printf "%-6s " ""
            printf "Port: %s" "$local_port"
            ((stopped_count++))
        fi

        echo ""
    done < "$CONFIG_FILE"

    echo ""
    echo "Running: $running_count | Stopped: $stopped_count"
}

list_tunnels() {
    echo -e "${BOLD}Available Tunnels:${RESET}"
    echo ""

    while IFS='|' read -r tunnel_name ssh_host local_port remote_host remote_port description; do
        # Skip comments and empty lines
        [[ "$tunnel_name" =~ ^#.*$ ]] && continue
        [[ -z "$tunnel_name" ]] && continue

        printf "  %-20s %s\n" "$tunnel_name" "$description"
    done < "$CONFIG_FILE"

    echo ""
}

show_logs() {
    local tunnel_name="$1"
    local log_file="$LOG_DIR/${tunnel_name}.log"

    if [[ ! -f "$log_file" ]]; then
        echo "No logs found for: $tunnel_name"
        return 1
    fi

    echo -e "${BOLD}Logs for: $tunnel_name${RESET}"
    echo "-------------------------------------"
    tail -50 "$log_file"
}

# ===========================
# Bulk Operations
# ===========================

start_all_tunnels() {
    echo -e "${BOLD}Starting all tunnels...${RESET}"
    echo ""

    local started=0
    local failed=0

    while IFS='|' read -r tunnel_name ssh_host local_port remote_host remote_port description; do
        [[ "$tunnel_name" =~ ^#.*$ ]] && continue
        [[ -z "$tunnel_name" ]] && continue

        if start_tunnel "$tunnel_name"; then
            ((started++))
        else
            ((failed++))
        fi
        echo ""
    done < "$CONFIG_FILE"

    echo "Started: $started | Failed: $failed"
}

stop_all_tunnels() {
    echo -e "${BOLD}Stopping all tunnels...${RESET}"
    echo ""

    local stopped=0

    while IFS='|' read -r tunnel_name ssh_host local_port remote_host remote_port description; do
        [[ "$tunnel_name" =~ ^#.*$ ]] && continue
        [[ -z "$tunnel_name" ]] && continue

        if stop_tunnel "$tunnel_name"; then
            ((stopped++))
        fi
    done < "$CONFIG_FILE"

    echo ""
    echo "Stopped: $stopped tunnels"
}

# ===========================
# Auto-Reconnect Monitor
# ===========================

monitor_tunnels() {
    echo -e "${BOLD}${CYAN}Monitoring SSH Tunnels${RESET}"
    echo "Press Ctrl+C to stop monitoring"
    echo ""

    trap 'echo -e "\n${YELLOW}Monitoring stopped${RESET}"; exit 0' INT

    while true; do
        clear
        show_all_status

        # Check and restart dead tunnels
        while IFS='|' read -r tunnel_name ssh_host local_port remote_host remote_port description; do
            [[ "$tunnel_name" =~ ^#.*$ ]] && continue
            [[ -z "$tunnel_name" ]] && continue

            if ! is_tunnel_running "$tunnel_name"; then
                echo -e "${YELLOW}⚠ Tunnel died: $tunnel_name - Restarting...${RESET}"
                start_tunnel "$tunnel_name"
            fi
        done < "$CONFIG_FILE"

        sleep 10
    done
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [ACTION] [TUNNEL_NAME]

Manage SSH tunnels for ML infrastructure.

ACTIONS:
    start TUNNEL          Start a tunnel
    stop TUNNEL           Stop a tunnel
    restart TUNNEL        Restart a tunnel
    status [TUNNEL]       Show tunnel status
    list                  List all available tunnels
    logs TUNNEL           Show tunnel logs
    monitor               Monitor and auto-restart tunnels

OPTIONS:
    -a, --all             Apply action to all tunnels
    -v, --verbose         Verbose output
    -h, --help            Display this help message

EXAMPLES:
    # List available tunnels
    $SCRIPT_NAME list

    # Start a tunnel
    $SCRIPT_NAME start jupyter-gpu1

    # Stop a tunnel
    $SCRIPT_NAME stop jupyter-gpu1

    # Show status
    $SCRIPT_NAME status

    # Start all tunnels
    $SCRIPT_NAME start --all

    # Monitor tunnels
    $SCRIPT_NAME monitor

    # View logs
    $SCRIPT_NAME logs jupyter-gpu1

CONFIGURATION:
    Edit $CONFIG_FILE to add/modify tunnels

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi

    # Parse action
    ACTION="$1"
    shift

    # Parse options and arguments
    ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -a|--all)
                ALL_TUNNELS=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# ===========================
# Main Function
# ===========================

main() {
    parse_arguments "$@"

    # Initialize
    initialize

    case "$ACTION" in
        start)
            if [[ "$ALL_TUNNELS" == true ]]; then
                start_all_tunnels
            elif [[ ${#ARGS[@]} -eq 0 ]]; then
                echo "Error: Tunnel name required"
                usage
                exit 1
            else
                start_tunnel "${ARGS[0]}"
            fi
            ;;
        stop)
            if [[ "$ALL_TUNNELS" == true ]]; then
                stop_all_tunnels
            elif [[ ${#ARGS[@]} -eq 0 ]]; then
                echo "Error: Tunnel name required"
                usage
                exit 1
            else
                stop_tunnel "${ARGS[0]}"
            fi
            ;;
        restart)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                echo "Error: Tunnel name required"
                usage
                exit 1
            fi
            restart_tunnel "${ARGS[0]}"
            ;;
        status)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                show_all_status
            else
                show_tunnel_status "${ARGS[0]}"
            fi
            ;;
        list)
            list_tunnels
            ;;
        logs)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                echo "Error: Tunnel name required"
                usage
                exit 1
            fi
            show_logs "${ARGS[0]}"
            ;;
        monitor)
            monitor_tunnels
            ;;
        *)
            echo "Unknown action: $ACTION"
            usage
            exit 1
            ;;
    esac
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
