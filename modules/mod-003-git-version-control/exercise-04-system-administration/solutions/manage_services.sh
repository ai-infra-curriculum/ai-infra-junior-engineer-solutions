#!/bin/bash
#
# manage_services.sh - ML infrastructure service manager
#
# Description:
#   Manage multiple ML services with health monitoring, auto-restart,
#   and alerting capabilities.
#
# Usage:
#   ./manage_services.sh [ACTION] [OPTIONS]
#
# Actions:
#   start [SERVICE]      Start service(s)
#   stop [SERVICE]       Stop service(s)
#   restart [SERVICE]    Restart service(s)
#   status [SERVICE]     Show service status
#   enable [SERVICE]     Enable service on boot
#   disable [SERVICE]    Disable service on boot
#   monitor              Monitor and auto-restart services
#   logs SERVICE         View service logs
#
# Options:
#   -a, --all            Apply to all services
#   -f, --follow         Follow logs
#   -n, --lines N        Show N lines of logs
#   -h, --help           Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CONFIG_FILE="$SCRIPT_DIR/services.conf"
readonly LOG_FILE="/var/log/service-manager.log"

# ML Infrastructure Services
readonly ML_SERVICES=(
    "docker"
    "nvidia-persistenced"
    "jupyter"
    "mlflow"
    "postgresql"
    "redis-server"
)

# Defaults
ALL_SERVICES=false
FOLLOW_LOGS=false
LOG_LINES=50

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
# Logging
# ===========================

log_message() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    log_message "INFO" "$@"
}

log_error() {
    log_message "ERROR" "$@"
}

log_warning() {
    log_message "WARNING" "$@"
}

# ===========================
# Service Operations
# ===========================

is_service_installed() {
    local service="$1"

    if systemctl list-unit-files "$service.service" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

is_service_active() {
    local service="$1"

    if systemctl is-active --quiet "$service"; then
        return 0
    else
        return 1
    fi
}

is_service_enabled() {
    local service="$1"

    if systemctl is-enabled --quiet "$service" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

get_service_status() {
    local service="$1"

    if ! is_service_installed "$service"; then
        echo "NOT_INSTALLED"
        return
    fi

    if is_service_active "$service"; then
        echo "ACTIVE"
    else
        if systemctl is-failed --quiet "$service"; then
            echo "FAILED"
        else
            echo "INACTIVE"
        fi
    fi
}

start_service() {
    local service="$1"

    echo -e "${BLUE}Starting service: $service${RESET}"

    if ! is_service_installed "$service"; then
        echo -e "${RED}Service not installed: $service${RESET}"
        log_error "Service not installed: $service"
        return 1
    fi

    if is_service_active "$service"; then
        echo -e "${YELLOW}Service already running: $service${RESET}"
        return 0
    fi

    if sudo systemctl start "$service"; then
        echo -e "${GREEN}✓ Service started: $service${RESET}"
        log_info "Started service: $service"
        return 0
    else
        echo -e "${RED}✗ Failed to start service: $service${RESET}"
        log_error "Failed to start service: $service"
        return 1
    fi
}

stop_service() {
    local service="$1"

    echo -e "${BLUE}Stopping service: $service${RESET}"

    if ! is_service_installed "$service"; then
        echo -e "${RED}Service not installed: $service${RESET}"
        return 1
    fi

    if ! is_service_active "$service"; then
        echo -e "${YELLOW}Service already stopped: $service${RESET}"
        return 0
    fi

    if sudo systemctl stop "$service"; then
        echo -e "${GREEN}✓ Service stopped: $service${RESET}"
        log_info "Stopped service: $service"
        return 0
    else
        echo -e "${RED}✗ Failed to stop service: $service${RESET}"
        log_error "Failed to stop service: $service"
        return 1
    fi
}

restart_service() {
    local service="$1"

    echo -e "${BLUE}Restarting service: $service${RESET}"

    if ! is_service_installed "$service"; then
        echo -e "${RED}Service not installed: $service${RESET}"
        return 1
    fi

    if sudo systemctl restart "$service"; then
        echo -e "${GREEN}✓ Service restarted: $service${RESET}"
        log_info "Restarted service: $service"
        return 0
    else
        echo -e "${RED}✗ Failed to restart service: $service${RESET}"
        log_error "Failed to restart service: $service"
        return 1
    fi
}

enable_service() {
    local service="$1"

    echo -e "${BLUE}Enabling service: $service${RESET}"

    if ! is_service_installed "$service"; then
        echo -e "${RED}Service not installed: $service${RESET}"
        return 1
    fi

    if is_service_enabled "$service"; then
        echo -e "${YELLOW}Service already enabled: $service${RESET}"
        return 0
    fi

    if sudo systemctl enable "$service"; then
        echo -e "${GREEN}✓ Service enabled: $service${RESET}"
        log_info "Enabled service: $service"
        return 0
    else
        echo -e "${RED}✗ Failed to enable service: $service${RESET}"
        log_error "Failed to enable service: $service"
        return 1
    fi
}

disable_service() {
    local service="$1"

    echo -e "${BLUE}Disabling service: $service${RESET}"

    if ! is_service_installed "$service"; then
        echo -e "${RED}Service not installed: $service${RESET}"
        return 1
    fi

    if ! is_service_enabled "$service"; then
        echo -e "${YELLOW}Service already disabled: $service${RESET}"
        return 0
    fi

    if sudo systemctl disable "$service"; then
        echo -e "${GREEN}✓ Service disabled: $service${RESET}"
        log_info "Disabled service: $service"
        return 0
    else
        echo -e "${RED}✗ Failed to disable service: $service${RESET}"
        log_error "Failed to disable service: $service"
        return 1
    fi
}

# ===========================
# Status Display
# ===========================

show_service_status() {
    local service="$1"

    if ! is_service_installed "$service"; then
        printf "%-25s ${YELLOW}%-15s${RESET}\n" "$service" "NOT INSTALLED"
        return
    fi

    local status=$(get_service_status "$service")
    local enabled="DISABLED"

    if is_service_enabled "$service"; then
        enabled="ENABLED"
    fi

    # Get uptime if active
    local uptime=""
    if [[ "$status" == "ACTIVE" ]]; then
        uptime=$(systemctl show "$service" -p ActiveEnterTimestamp --value)
        uptime=$(date -d "$uptime" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "")
    fi

    # Color code status
    local status_color="$RESET"
    case "$status" in
        ACTIVE)
            status_color="$GREEN"
            ;;
        FAILED)
            status_color="$RED"
            ;;
        INACTIVE)
            status_color="$YELLOW"
            ;;
    esac

    printf "%-25s ${status_color}%-15s${RESET} %-15s %s\n" \
        "$service" "$status" "$enabled" "$uptime"
}

show_all_status() {
    echo -e "${BOLD}${CYAN}ML Infrastructure Services${RESET}"
    echo -e "${CYAN}========================================================================${RESET}"
    printf "%-25s %-15s %-15s %s\n" "SERVICE" "STATUS" "BOOT" "STARTED"
    echo "------------------------------------------------------------------------"

    for service in "${ML_SERVICES[@]}"; do
        show_service_status "$service"
    done

    echo ""

    # Summary
    local total=${#ML_SERVICES[@]}
    local active=0
    local failed=0

    for service in "${ML_SERVICES[@]}"; do
        if is_service_installed "$service"; then
            local status=$(get_service_status "$service")
            if [[ "$status" == "ACTIVE" ]]; then
                ((active++))
            elif [[ "$status" == "FAILED" ]]; then
                ((failed++))
            fi
        fi
    done

    echo "Total: $total | Active: $active | Failed: $failed"
    echo ""
}

# ===========================
# Log Viewing
# ===========================

view_service_logs() {
    local service="$1"

    if ! is_service_installed "$service"; then
        echo -e "${RED}Service not installed: $service${RESET}"
        return 1
    fi

    echo -e "${BOLD}Logs for service: $service${RESET}"
    echo "-------------------------------------"

    if [[ "$FOLLOW_LOGS" == true ]]; then
        sudo journalctl -u "$service" -f
    else
        sudo journalctl -u "$service" -n "$LOG_LINES" --no-pager
    fi
}

# ===========================
# Monitoring
# ===========================

monitor_services() {
    echo -e "${BOLD}${CYAN}Service Monitoring Mode${RESET}"
    echo "Monitoring ML infrastructure services..."
    echo "Press Ctrl+C to stop"
    echo ""

    trap 'echo -e "\n${YELLOW}Monitoring stopped${RESET}"; exit 0' INT

    while true; do
        clear
        show_all_status

        # Check for failed services and restart
        for service in "${ML_SERVICES[@]}"; do
            if is_service_installed "$service"; then
                local status=$(get_service_status "$service")

                if [[ "$status" == "FAILED" ]]; then
                    echo -e "${RED}⚠ Service failed: $service - Attempting restart...${RESET}"
                    log_warning "Service failed: $service - Attempting restart"

                    if restart_service "$service"; then
                        send_alert "Service $service was down and has been restarted"
                    else
                        send_alert "Failed to restart service $service"
                    fi
                fi
            fi
        done

        sleep 10
    done
}

send_alert() {
    local message="$1"

    # Log alert
    log_warning "ALERT: $message"

    # Send to syslog
    logger -t "service-manager" -p user.warning "$message"

    # Could integrate with alerting systems here
    # e.g., email, Slack, PagerDuty
}

# ===========================
# Bulk Operations
# ===========================

start_all_services() {
    echo -e "${BOLD}Starting all services...${RESET}"
    echo ""

    local started=0
    local failed=0

    for service in "${ML_SERVICES[@]}"; do
        if is_service_installed "$service"; then
            if start_service "$service"; then
                ((started++))
            else
                ((failed++))
            fi
        fi
        echo ""
    done

    echo "Started: $started | Failed: $failed"
}

stop_all_services() {
    echo -e "${BOLD}Stopping all services...${RESET}"
    echo ""

    local stopped=0

    for service in "${ML_SERVICES[@]}"; do
        if is_service_installed "$service"; then
            if stop_service "$service"; then
                ((stopped++))
            fi
        fi
    done

    echo ""
    echo "Stopped: $stopped services"
}

restart_all_services() {
    echo -e "${BOLD}Restarting all services...${RESET}"
    echo ""

    local restarted=0

    for service in "${ML_SERVICES[@]}"; do
        if is_service_installed "$service"; then
            if restart_service "$service"; then
                ((restarted++))
            fi
        fi
        echo ""
    done

    echo "Restarted: $restarted services"
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [ACTION] [SERVICE] [OPTIONS]

Manage ML infrastructure services.

ACTIONS:
    start [SERVICE]          Start service(s)
    stop [SERVICE]           Stop service(s)
    restart [SERVICE]        Restart service(s)
    status [SERVICE]         Show service status
    enable [SERVICE]         Enable service on boot
    disable [SERVICE]        Disable service on boot
    monitor                  Monitor and auto-restart services
    logs SERVICE             View service logs

OPTIONS:
    -a, --all                Apply to all services
    -f, --follow             Follow logs (for logs action)
    -n, --lines N            Show N lines of logs (default: $LOG_LINES)
    -h, --help               Display this help message

MANAGED SERVICES:
$(printf "    %s\n" "${ML_SERVICES[@]}")

EXAMPLES:
    # Show status of all services
    $SCRIPT_NAME status

    # Start a service
    $SCRIPT_NAME start docker

    # Start all services
    $SCRIPT_NAME start --all

    # View logs
    $SCRIPT_NAME logs jupyter

    # Follow logs
    $SCRIPT_NAME logs mlflow --follow

    # Monitor services
    $SCRIPT_NAME monitor

    # Restart all services
    $SCRIPT_NAME restart --all

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

    # Parse service and options
    SERVICE=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -a|--all)
                ALL_SERVICES=true
                shift
                ;;
            -f|--follow)
                FOLLOW_LOGS=true
                shift
                ;;
            -n|--lines)
                LOG_LINES="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                SERVICE="$1"
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

    # Ensure log file exists
    sudo touch "$LOG_FILE" 2>/dev/null || true

    case "$ACTION" in
        start)
            if [[ "$ALL_SERVICES" == true ]]; then
                start_all_services
            elif [[ -n "$SERVICE" ]]; then
                start_service "$SERVICE"
            else
                echo "Error: Service name required"
                usage
                exit 1
            fi
            ;;
        stop)
            if [[ "$ALL_SERVICES" == true ]]; then
                stop_all_services
            elif [[ -n "$SERVICE" ]]; then
                stop_service "$SERVICE"
            else
                echo "Error: Service name required"
                usage
                exit 1
            fi
            ;;
        restart)
            if [[ "$ALL_SERVICES" == true ]]; then
                restart_all_services
            elif [[ -n "$SERVICE" ]]; then
                restart_service "$SERVICE"
            else
                echo "Error: Service name required"
                usage
                exit 1
            fi
            ;;
        status)
            if [[ -n "$SERVICE" ]]; then
                show_service_status "$SERVICE"
            else
                show_all_status
            fi
            ;;
        enable)
            if [[ -n "$SERVICE" ]]; then
                enable_service "$SERVICE"
            else
                echo "Error: Service name required"
                usage
                exit 1
            fi
            ;;
        disable)
            if [[ -n "$SERVICE" ]]; then
                disable_service "$SERVICE"
            else
                echo "Error: Service name required"
                usage
                exit 1
            fi
            ;;
        monitor)
            monitor_services
            ;;
        logs)
            if [[ -n "$SERVICE" ]]; then
                view_service_logs "$SERVICE"
            else
                echo "Error: Service name required"
                usage
                exit 1
            fi
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
