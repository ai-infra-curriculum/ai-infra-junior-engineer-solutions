#!/bin/bash
#
# check_network_health.sh - Network health checker for ML infrastructure
#
# Description:
#   Comprehensive network health checks including connectivity, latency,
#   bandwidth, port availability, and service status for ML servers.
#
# Usage:
#   ./check_network_health.sh [OPTIONS]
#
# Options:
#   -c, --config FILE     Configuration file (default: ./hosts.conf)
#   -t, --timeout SEC     Connection timeout (default: 5)
#   -r, --report FILE     Generate report to file
#   -a, --alerts          Enable alert notifications
#   -v, --verbose         Verbose output
#   -j, --json            Output in JSON format
#   -h, --help            Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEFAULT_CONFIG="$SCRIPT_DIR/hosts.conf"

# Defaults
CONFIG_FILE="$DEFAULT_CONFIG"
TIMEOUT=5
REPORT_FILE=""
ENABLE_ALERTS=false
VERBOSE=false
JSON_OUTPUT=false

# Test results
declare -A HOST_STATUS
declare -A HOST_LATENCY
declare -A PORT_STATUS

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
    # Create default config if it doesn't exist
    if [[ ! -f "$CONFIG_FILE" ]]; then
        create_default_config
    fi
}

create_default_config() {
    cat > "$DEFAULT_CONFIG" <<'EOF'
# Network Health Check Configuration
# Format: hostname|ip_address|ports|description
#
# Ports can be comma-separated: 22,80,443
# Use "ping" for ICMP-only checks
#
# ML Infrastructure
ml-train-1|192.168.1.101|22,8888,6006|Training Server 1
ml-train-2|192.168.1.102|22,8888,6006|Training Server 2
gpu-node-1|192.168.1.201|22,8888,6006|GPU Node 1
gpu-node-2|192.168.1.202|22,8888,6006|GPU Node 2

# Databases
postgres-prod|192.168.1.50|5432|Production PostgreSQL
redis-cache|192.168.1.51|6379|Redis Cache
mongo-staging|192.168.1.52|27017|MongoDB Staging

# Services
api-server|192.168.1.80|80,443|ML API Server
model-registry|192.168.1.81|5000|Model Registry
mlflow-server|192.168.1.82|5000|MLflow Tracking

# Infrastructure
bastion-host|192.168.1.10|22|Jump Host
monitoring|192.168.1.20|3000,9090|Grafana & Prometheus
EOF

    echo "Created default configuration: $DEFAULT_CONFIG"
}

# ===========================
# Network Tests
# ===========================

test_ping() {
    local host="$1"
    local ip="$2"

    if [[ "$VERBOSE" == true ]]; then
        echo "  Testing ICMP connectivity to $host ($ip)..."
    fi

    local result
    if result=$(ping -c 3 -W "$TIMEOUT" "$ip" 2>&1); then
        # Extract average latency
        local latency=$(echo "$result" | grep -oP 'rtt min/avg/max/mdev = [\d.]+/\K[\d.]+' || echo "0")
        HOST_LATENCY["$host"]=$latency
        HOST_STATUS["$host"]="UP"
        return 0
    else
        HOST_STATUS["$host"]="DOWN"
        return 1
    fi
}

test_tcp_port() {
    local host="$1"
    local ip="$2"
    local port="$3"

    if [[ "$VERBOSE" == true ]]; then
        echo "  Testing port $port on $host ($ip)..."
    fi

    local key="${host}:${port}"

    # Try to connect with timeout
    if timeout "$TIMEOUT" bash -c "cat < /dev/null > /dev/tcp/$ip/$port" 2>/dev/null; then
        PORT_STATUS["$key"]="OPEN"
        return 0
    else
        PORT_STATUS["$key"]="CLOSED"
        return 1
    fi
}

test_http_service() {
    local host="$1"
    local ip="$2"
    local port="$3"

    if [[ "$VERBOSE" == true ]]; then
        echo "  Testing HTTP service on $host:$port..."
    fi

    local url="http://$ip:$port"
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$TIMEOUT" "$url" 2>/dev/null || echo "000")

    if [[ "$http_code" != "000" ]]; then
        return 0
    else
        return 1
    fi
}

measure_bandwidth() {
    local host="$1"
    local ip="$2"

    # Simple bandwidth test using dd over SSH (requires SSH access)
    if ! PORT_STATUS["${host}:22"] == "OPEN" ]]; then
        return 1
    fi

    if [[ "$VERBOSE" == true ]]; then
        echo "  Measuring bandwidth to $host..."
    fi

    # Test download speed
    local speed=$(ssh -o ConnectTimeout="$TIMEOUT" "$ip" "dd if=/dev/zero bs=1M count=10 2>/dev/null" | dd of=/dev/null 2>&1 | grep -oP '\d+ MB/s' || echo "N/A")

    echo "$speed"
}

# ===========================
# Host Testing
# ===========================

test_host() {
    local hostname="$1"
    local ip="$2"
    local ports="$3"
    local description="$4"

    if [[ "$JSON_OUTPUT" == false ]]; then
        echo -e "${BOLD}Testing: $hostname${RESET}"
        echo "  IP: $ip"
        echo "  Description: $description"
    fi

    # Test ICMP connectivity
    if test_ping "$hostname" "$ip"; then
        local latency=${HOST_LATENCY["$hostname"]}
        if [[ "$JSON_OUTPUT" == false ]]; then
            echo -e "  Ping: ${GREEN}OK${RESET} (${latency}ms)"
        fi
    else
        if [[ "$JSON_OUTPUT" == false ]]; then
            echo -e "  Ping: ${RED}FAILED${RESET}"
        fi

        if [[ "$ENABLE_ALERTS" == true ]]; then
            send_alert "Host $hostname ($ip) is not responding to ping"
        fi

        if [[ "$JSON_OUTPUT" == false ]]; then
            echo ""
        fi
        return 1
    fi

    # Test ports
    if [[ "$ports" != "ping" ]]; then
        IFS=',' read -ra PORT_ARRAY <<< "$ports"
        local all_ports_ok=true

        for port in "${PORT_ARRAY[@]}"; do
            port=$(echo "$port" | xargs)  # Trim whitespace

            if test_tcp_port "$hostname" "$ip" "$port"; then
                if [[ "$JSON_OUTPUT" == false ]]; then
                    echo -e "  Port $port: ${GREEN}OPEN${RESET}"
                fi
            else
                if [[ "$JSON_OUTPUT" == false ]]; then
                    echo -e "  Port $port: ${RED}CLOSED${RESET}"
                fi
                all_ports_ok=false

                if [[ "$ENABLE_ALERTS" == true ]]; then
                    send_alert "Port $port on $hostname ($ip) is not accessible"
                fi
            fi
        done
    fi

    if [[ "$JSON_OUTPUT" == false ]]; then
        echo ""
    fi

    return 0
}

# ===========================
# Summary and Reporting
# ===========================

generate_summary() {
    echo -e "${BOLD}${CYAN}Network Health Check Summary${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    local total_hosts=0
    local up_hosts=0
    local down_hosts=0

    for host in "${!HOST_STATUS[@]}"; do
        ((total_hosts++))
        if [[ "${HOST_STATUS[$host]}" == "UP" ]]; then
            ((up_hosts++))
        else
            ((down_hosts++))
        fi
    done

    echo "Total Hosts: $total_hosts"
    echo -e "${GREEN}Up: $up_hosts${RESET}"
    echo -e "${RED}Down: $down_hosts${RESET}"
    echo ""

    # Port summary
    local total_ports=0
    local open_ports=0
    local closed_ports=0

    for port_key in "${!PORT_STATUS[@]}"; do
        ((total_ports++))
        if [[ "${PORT_STATUS[$port_key]}" == "OPEN" ]]; then
            ((open_ports++))
        else
            ((closed_ports++))
        fi
    done

    echo "Total Ports Checked: $total_ports"
    echo -e "${GREEN}Open: $open_ports${RESET}"
    echo -e "${RED}Closed: $closed_ports${RESET}"
    echo ""

    # Latency statistics
    if [[ ${#HOST_LATENCY[@]} -gt 0 ]]; then
        echo -e "${BOLD}Latency Statistics:${RESET}"

        local sum=0
        local count=0
        local min=999999
        local max=0

        for host in "${!HOST_LATENCY[@]}"; do
            local latency=${HOST_LATENCY[$host]}
            sum=$(echo "$sum + $latency" | bc)
            ((count++))

            if (( $(echo "$latency < $min" | bc -l) )); then
                min=$latency
            fi

            if (( $(echo "$latency > $max" | bc -l) )); then
                max=$latency
            fi
        done

        local avg=$(echo "scale=2; $sum / $count" | bc)

        echo "  Min: ${min}ms"
        echo "  Max: ${max}ms"
        echo "  Avg: ${avg}ms"
        echo ""
    fi

    # Failed hosts
    if [[ $down_hosts -gt 0 ]]; then
        echo -e "${RED}${BOLD}Failed Hosts:${RESET}"
        for host in "${!HOST_STATUS[@]}"; do
            if [[ "${HOST_STATUS[$host]}" == "DOWN" ]]; then
                echo -e "  ${RED}✗${RESET} $host"
            fi
        done
        echo ""
    fi

    # Failed ports
    if [[ $closed_ports -gt 0 ]]; then
        echo -e "${RED}${BOLD}Closed Ports:${RESET}"
        for port_key in "${!PORT_STATUS[@]}"; do
            if [[ "${PORT_STATUS[$port_key]}" == "CLOSED" ]]; then
                echo -e "  ${RED}✗${RESET} $port_key"
            fi
        done
        echo ""
    fi
}

generate_json_report() {
    cat <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hosts": [
EOF

    local first_host=true
    while IFS='|' read -r hostname ip ports description; do
        [[ "$hostname" =~ ^#.*$ ]] && continue
        [[ -z "$hostname" ]] && continue

        if [[ "$first_host" == false ]]; then
            echo ","
        fi
        first_host=false

        local status="${HOST_STATUS[$hostname]:-UNKNOWN}"
        local latency="${HOST_LATENCY[$hostname]:-0}"

        cat <<EOF
    {
      "hostname": "$hostname",
      "ip": "$ip",
      "description": "$description",
      "status": "$status",
      "latency_ms": $latency,
      "ports": [
EOF

        if [[ "$ports" != "ping" ]]; then
            IFS=',' read -ra PORT_ARRAY <<< "$ports"
            local first_port=true

            for port in "${PORT_ARRAY[@]}"; do
                port=$(echo "$port" | xargs)
                local port_status="${PORT_STATUS[${hostname}:${port}]:-UNKNOWN}"

                if [[ "$first_port" == false ]]; then
                    echo ","
                fi
                first_port=false

                cat <<EOF
        {
          "port": $port,
          "status": "$port_status"
        }
EOF
            done
        fi

        echo "      ]"
        echo -n "    }"

    done < "$CONFIG_FILE"

    cat <<EOF

  ]
}
EOF
}

# ===========================
# Alerting
# ===========================

send_alert() {
    local message="$1"

    # Log to syslog
    logger -t "network-health" "$message"

    # Print to stderr
    echo -e "${RED}[ALERT] $message${RESET}" >&2
}

# ===========================
# Main Testing Loop
# ===========================

run_health_check() {
    if [[ "$JSON_OUTPUT" == false ]]; then
        echo -e "${BOLD}${CYAN}Network Health Check${RESET}"
        echo -e "${CYAN}========================================${RESET}"
        echo "Config: $CONFIG_FILE"
        echo "Timeout: ${TIMEOUT}s"
        echo ""
    fi

    while IFS='|' read -r hostname ip ports description; do
        # Skip comments and empty lines
        [[ "$hostname" =~ ^#.*$ ]] && continue
        [[ -z "$hostname" ]] && continue

        test_host "$hostname" "$ip" "$ports" "$description"

    done < "$CONFIG_FILE"

    # Generate summary
    if [[ "$JSON_OUTPUT" == true ]]; then
        generate_json_report
    else
        generate_summary
    fi

    # Generate report file
    if [[ -n "$REPORT_FILE" ]]; then
        if [[ "$JSON_OUTPUT" == true ]]; then
            generate_json_report > "$REPORT_FILE"
        else
            {
                run_health_check
            } > "$REPORT_FILE" 2>&1
        fi
        echo "Report saved to: $REPORT_FILE"
    fi
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Network health checker for ML infrastructure.

OPTIONS:
    -c, --config FILE       Configuration file (default: $DEFAULT_CONFIG)
    -t, --timeout SECONDS   Connection timeout (default: $TIMEOUT)
    -r, --report FILE       Generate report to file
    -a, --alerts            Enable alert notifications
    -v, --verbose           Verbose output
    -j, --json              Output in JSON format
    -h, --help              Display this help message

EXAMPLES:
    # Run health check
    $SCRIPT_NAME

    # Use custom config
    $SCRIPT_NAME --config production-hosts.conf

    # Generate report
    $SCRIPT_NAME --report health-report.txt

    # JSON output
    $SCRIPT_NAME --json --report health.json

    # Verbose with alerts
    $SCRIPT_NAME --verbose --alerts

CONFIGURATION:
    Edit $CONFIG_FILE to add/modify hosts

    Format: hostname|ip_address|ports|description
    Ports: Comma-separated (22,80,443) or "ping" for ICMP only

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -r|--report)
                REPORT_FILE="$2"
                shift 2
                ;;
            -a|--alerts)
                ENABLE_ALERTS=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -j|--json)
                JSON_OUTPUT=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
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

    # Validate config file
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${RESET}"
        exit 1
    fi

    # Run health check
    run_health_check
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
