#!/bin/bash
###############################################################################
# Network Performance Monitoring for ML Infrastructure
###############################################################################
#
# Purpose: Continuously monitor network connectivity and performance
#          for critical ML infrastructure services
#
# Usage: ./network_monitor.sh [OPTIONS]
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
IFACE="${NETWORK_INTERFACE:-eth0}"
LOG_FILE="${LOG_FILE:-/var/log/network_monitor.log}"
INTERVAL="${CHECK_INTERVAL:-60}"
ALERT_THRESHOLD_PACKET_LOSS="${ALERT_THRESHOLD_PACKET_LOSS:-5}"
ALERT_THRESHOLD_LATENCY="${ALERT_THRESHOLD_LATENCY:-100}"
ALERT_EMAIL="${ALERT_EMAIL:-}"
VERBOSE="${VERBOSE:-false}"

# Critical services to monitor
declare -a SERVICES=(
    "ml-api.internal:8080"
    "feature-store.internal:6379"
    "model-db.internal:5432"
    "prometheus.internal:9090"
    "grafana.internal:3000"
)

# Help function
show_help() {
    cat << EOF
Network Performance Monitoring for ML Infrastructure

Usage: $0 [OPTIONS]

Options:
    -i, --interface IFACE       Network interface to monitor (default: eth0)
    -l, --log-file FILE         Log file location (default: /var/log/network_monitor.log)
    -t, --interval SECONDS      Check interval in seconds (default: 60)
    -v, --verbose               Enable verbose output
    -s, --services "svc1:port svc2:port"   Services to monitor
    -h, --help                  Show this help message

Environment Variables:
    NETWORK_INTERFACE           Network interface (default: eth0)
    CHECK_INTERVAL              Check interval (default: 60 seconds)
    ALERT_THRESHOLD_PACKET_LOSS Packet loss % threshold (default: 5)
    ALERT_THRESHOLD_LATENCY     Latency threshold in ms (default: 100)
    ALERT_EMAIL                 Email for alerts

Examples:
    # Basic monitoring
    $0

    # Custom interface and interval
    $0 -i enp0s3 -t 30

    # Verbose monitoring
    $0 -v

    # Custom services
    $0 -s "api.internal:8080 db.internal:5432"

    # Run in background
    nohup $0 -l /var/log/ml-network.log &

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interface)
            IFACE="$2"
            shift 2
            ;;
        -l|--log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        -t|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -s|--services)
            IFS=' ' read -ra SERVICES <<< "$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging function
log() {
    local level="$1"
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"

    if [ "$level" = "ERROR" ] || [ "$level" = "CRITICAL" ]; then
        send_alert "$level: $message"
    fi
}

# Alert function
send_alert() {
    local message="$1"

    # Email alert (if configured)
    if [ -n "$ALERT_EMAIL" ] && command -v mail &> /dev/null; then
        echo "$message" | mail -s "[ML-Network-Alert] $message" "$ALERT_EMAIL"
    fi

    # Webhook alert (if configured)
    if [ -n "${ALERT_WEBHOOK:-}" ]; then
        curl -X POST "$ALERT_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"$message\"}" \
            2>/dev/null || true
    fi

    # Syslog
    logger -t network-monitor -p user.alert "$message"
}

# Check if interface exists
check_interface() {
    if ! ip link show "$IFACE" &> /dev/null; then
        log ERROR "Interface $IFACE does not exist"
        log INFO "Available interfaces: $(ip -o link show | awk -F': ' '{print $2}' | grep -v lo | tr '\n' ' ')"
        exit 1
    fi
}

# Get interface statistics
get_interface_stats() {
    local iface="$1"

    # Use ip -s for statistics
    local stats=$(ip -s link show "$iface")

    # Extract RX and TX stats
    local rx_bytes=$(echo "$stats" | grep -A 1 "RX:" | tail -1 | awk '{print $1}')
    local rx_packets=$(echo "$stats" | grep -A 1 "RX:" | tail -1 | awk '{print $2}')
    local rx_errors=$(echo "$stats" | grep -A 1 "RX:" | tail -1 | awk '{print $3}')
    local rx_dropped=$(echo "$stats" | grep -A 1 "RX:" | tail -1 | awk '{print $4}')

    local tx_bytes=$(echo "$stats" | grep -A 1 "TX:" | tail -1 | awk '{print $1}')
    local tx_packets=$(echo "$stats" | grep -A 1 "TX:" | tail -1 | awk '{print $2}')
    local tx_errors=$(echo "$stats" | grep -A 1 "TX:" | tail -1 | awk '{print $3}')
    local tx_dropped=$(echo "$stats" | grep -A 1 "TX:" | tail -1 | awk '{print $4}')

    # Return as JSON-like string
    echo "RX_BYTES=$rx_bytes RX_PACKETS=$rx_packets RX_ERRORS=$rx_errors RX_DROPPED=$rx_dropped TX_BYTES=$tx_bytes TX_PACKETS=$tx_packets TX_ERRORS=$tx_errors TX_DROPPED=$tx_dropped"
}

# Get connection statistics
get_connection_stats() {
    local established=$(ss -tan state established 2>/dev/null | wc -l)
    local time_wait=$(ss -tan state time-wait 2>/dev/null | wc -l)
    local close_wait=$(ss -tan state close-wait 2>/dev/null | wc -l)
    local listening=$(ss -tln 2>/dev/null | wc -l)

    echo "ESTABLISHED=$established TIME_WAIT=$time_wait CLOSE_WAIT=$close_wait LISTENING=$listening"
}

# Test service connectivity
test_service() {
    local service="$1"
    local host="${service%%:*}"
    local port="${service##*:}"

    # Test connectivity with timeout
    if timeout 2 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Ping test with latency measurement
ping_test() {
    local host="$1"
    local count="${2:-4}"

    # Ping test
    local result=$(ping -c "$count" -W 2 "$host" 2>&1 || true)

    if echo "$result" | grep -q "0% packet loss"; then
        # Extract statistics
        local stats=$(echo "$result" | grep "rtt min/avg/max/mdev")
        local avg_latency=$(echo "$stats" | awk -F'/' '{print $5}')

        echo "STATUS=OK AVG_LATENCY=$avg_latency"
        return 0
    else
        # Packet loss detected
        local packet_loss=$(echo "$result" | grep "packet loss" | awk '{print $6}' | sed 's/%//' || echo "100")

        echo "STATUS=PACKET_LOSS LOSS=$packet_loss"
        return 1
    fi
}

# Monitor bandwidth usage
monitor_bandwidth() {
    local iface="$1"

    # Get current stats
    local stats_before=$(get_interface_stats "$iface")
    eval "$stats_before"
    local rx_before=$RX_BYTES
    local tx_before=$TX_BYTES

    # Wait 1 second
    sleep 1

    # Get stats again
    local stats_after=$(get_interface_stats "$iface")
    eval "$stats_after"
    local rx_after=$RX_BYTES
    local tx_after=$TX_BYTES

    # Calculate bandwidth (bytes per second)
    local rx_bps=$((rx_after - rx_before))
    local tx_bps=$((tx_after - tx_before))

    # Convert to human-readable (use awk for decimal division)
    local rx_mbps=$(awk "BEGIN {printf \"%.2f\", $rx_bps / 1024 / 1024}")
    local tx_mbps=$(awk "BEGIN {printf \"%.2f\", $tx_bps / 1024 / 1024}")

    echo "RX_MBPS=$rx_mbps TX_MBPS=$tx_mbps"
}

# Check DNS resolution
check_dns() {
    local test_domains=("google.com")

    for domain in "${test_domains[@]}"; do
        if ! host "$domain" &> /dev/null; then
            log WARN "DNS resolution failed for $domain"
            return 1
        fi
    done

    return 0
}

# Header
header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   ML Infrastructure Network Monitoring                         ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Interface: $IFACE"
    echo "Log file: $LOG_FILE"
    echo "Check interval: ${INTERVAL}s"
    echo ""
}

# Main monitoring loop
monitor() {
    log INFO "Network monitoring started (Interface: $IFACE, Interval: ${INTERVAL}s)"

    while true; do
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

        if [ "$VERBOSE" = "true" ]; then
            echo -e "${CYAN}=== Check at $timestamp ===${NC}"
        fi

        # 1. Interface statistics
        local stats=$(get_interface_stats "$IFACE")
        eval "$stats"

        if [ "$RX_ERRORS" -gt 0 ] || [ "$TX_ERRORS" -gt 0 ]; then
            log WARN "Interface errors detected: RX=$RX_ERRORS TX=$TX_ERRORS"
        fi

        if [ "$RX_DROPPED" -gt 10 ] || [ "$TX_DROPPED" -gt 10 ]; then
            log WARN "Dropped packets detected: RX=$RX_DROPPED TX=$TX_DROPPED"
        fi

        if [ "$VERBOSE" = "true" ]; then
            echo "Interface stats: RX errors=$RX_ERRORS, TX errors=$TX_ERRORS, RX dropped=$RX_DROPPED, TX dropped=$TX_DROPPED"
        fi

        # 2. Connection statistics
        local conn_stats=$(get_connection_stats)
        eval "$conn_stats"

        if [ "$VERBOSE" = "true" ]; then
            echo "Connections: ESTABLISHED=$ESTABLISHED, TIME_WAIT=$TIME_WAIT, CLOSE_WAIT=$CLOSE_WAIT"
        fi

        log INFO "Connections: EST=$ESTABLISHED, TW=$TIME_WAIT, CW=$CLOSE_WAIT"

        # High TIME_WAIT connections warning
        if [ "$TIME_WAIT" -gt 5000 ]; then
            log WARN "High TIME_WAIT connections: $TIME_WAIT (may indicate connection leaks)"
        fi

        # 3. Bandwidth monitoring
        if [ "$VERBOSE" = "true" ]; then
            echo -n "Bandwidth: "
        fi

        local bandwidth=$(monitor_bandwidth "$IFACE")
        eval "$bandwidth"

        if [ "$VERBOSE" = "true" ]; then
            echo "RX=${RX_MBPS} MB/s, TX=${TX_MBPS} MB/s"
        fi

        log INFO "Bandwidth: RX=${RX_MBPS}MB/s, TX=${TX_MBPS}MB/s"

        # 4. DNS check
        if ! check_dns; then
            log ERROR "DNS resolution issues detected"
        fi

        # 5. Test critical services
        for service in "${SERVICES[@]}"; do
            local host="${service%%:*}"
            local port="${service##*:}"

            if test_service "$service"; then
                if [ "$VERBOSE" = "true" ]; then
                    echo -e "${GREEN}✓${NC} $service reachable"
                fi
                log INFO "Service $service: OK"
            else
                if [ "$VERBOSE" = "true" ]; then
                    echo -e "${RED}✗${NC} $service UNREACHABLE"
                fi
                log ERROR "Service $service: UNREACHABLE"
            fi

            # Ping test for latency
            local ping_result=$(ping_test "$host" 4)
            eval "$ping_result"

            if [ "${STATUS:-}" = "OK" ]; then
                local latency=$(echo "$AVG_LATENCY" | cut -d'.' -f1)

                if [ "$latency" -gt "$ALERT_THRESHOLD_LATENCY" ]; then
                    log WARN "High latency to $host: ${AVG_LATENCY}ms"
                fi

                if [ "$VERBOSE" = "true" ]; then
                    echo "  Latency: ${AVG_LATENCY}ms"
                fi
            elif [ "${STATUS:-}" = "PACKET_LOSS" ]; then
                log WARN "Packet loss to $host: ${LOSS}%"
            fi
        done

        if [ "$VERBOSE" = "true" ]; then
            echo ""
        fi

        # Sleep until next check
        sleep "$INTERVAL"
    done
}

# Signal handlers
cleanup() {
    log INFO "Network monitoring stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main execution
main() {
    # Check prerequisites
    check_interface

    # Show header
    if [ "$VERBOSE" = "true" ]; then
        header
    fi

    # Ensure log directory exists
    mkdir -p "$(dirname "$LOG_FILE")"

    # Start monitoring
    monitor
}

main "$@"
