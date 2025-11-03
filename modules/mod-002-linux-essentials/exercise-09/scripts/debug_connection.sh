#!/bin/bash
###############################################################################
# Network Connection Debugging Tool
###############################################################################
#
# Purpose: Systematically debug network connection issues
#
# Usage: ./debug_connection.sh <host> <port>
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <host> <port>"
    echo ""
    echo "Examples:"
    echo "  $0 ml-api.internal 8080"
    echo "  $0 192.168.1.100 5432"
    echo "  $0 google.com 443"
    exit 1
fi

TARGET_HOST="$1"
TARGET_PORT="$2"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Connection Debugging: $TARGET_HOST:$TARGET_PORT${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# 1. DNS Resolution
echo -e "${CYAN}[1] DNS Resolution${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v host &> /dev/null; then
    if host "$TARGET_HOST" > /dev/null 2>&1; then
        IP=$(host "$TARGET_HOST" | awk '/has address/ {print $4}' | head -n1)
        if [ -n "$IP" ]; then
            echo -e "${GREEN}✓${NC} Resolved: $TARGET_HOST → $IP"
        else
            # Might be an IP address
            if [[ "$TARGET_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                IP="$TARGET_HOST"
                echo -e "${GREEN}✓${NC} Using IP address: $IP"
            else
                echo -e "${RED}✗${NC} DNS resolution failed for $TARGET_HOST"
                exit 1
            fi
        fi
    else
        if [[ "$TARGET_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            IP="$TARGET_HOST"
            echo -e "${GREEN}✓${NC} Using IP address: $IP"
        else
            echo -e "${RED}✗${NC} DNS resolution failed for $TARGET_HOST"
            exit 1
        fi
    fi
else
    # host command not available, try getent
    if IP=$(getent hosts "$TARGET_HOST" 2>/dev/null | awk '{print $1}'); then
        echo -e "${GREEN}✓${NC} Resolved: $TARGET_HOST → $IP"
    elif [[ "$TARGET_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        IP="$TARGET_HOST"
        echo -e "${GREEN}✓${NC} Using IP address: $IP"
    else
        echo -e "${RED}✗${NC} DNS resolution failed for $TARGET_HOST"
        exit 1
    fi
fi
echo ""

# 2. Ping Test
echo -e "${CYAN}[2] Ping Test (ICMP)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ping -c 3 -W 2 "$IP" > /dev/null 2>&1; then
    PING_RESULT=$(ping -c 3 -W 2 "$IP" 2>&1 | tail -2)
    echo -e "${GREEN}✓${NC} Host is reachable via ICMP"
    echo "$PING_RESULT"
else
    echo -e "${YELLOW}!${NC} Host not reachable via ICMP (ICMP might be blocked)"
fi
echo ""

# 3. Port Connectivity
echo -e "${CYAN}[3] Port $TARGET_PORT Test${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if timeout 5 bash -c "cat < /dev/null > /dev/tcp/$IP/$TARGET_PORT" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Port $TARGET_PORT is open and accessible"
else
    echo -e "${RED}✗${NC} Port $TARGET_PORT is closed, filtered, or timing out"

    # Additional diagnostics
    echo ""
    echo "Possible reasons:"
    echo "  - Service not running on target"
    echo "  - Firewall blocking the port"
    echo "  - Wrong port number"
    echo "  - Network routing issue"
fi
echo ""

# 4. Traceroute
echo -e "${CYAN}[4] Route to Host${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v traceroute &> /dev/null; then
    traceroute -m 10 -w 2 "$IP" 2>&1 | head -n 10
elif command -v tracepath &> /dev/null; then
    tracepath "$IP" 2>&1 | head -n 10
else
    echo -e "${YELLOW}!${NC} traceroute not available"
fi
echo ""

# 5. Local Firewall Check
echo -e "${CYAN}[5] Local Firewall${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v sudo &> /dev/null; then
    if sudo iptables -L OUTPUT -n 2>/dev/null | grep -q "REJECT\|DROP"; then
        echo -e "${YELLOW}⚠${NC} Outbound firewall rules detected"
        sudo iptables -L OUTPUT -n | grep "$TARGET_PORT" || echo "No specific rules for port $TARGET_PORT"
    else
        echo -e "${GREEN}✓${NC} No restrictive outbound firewall rules"
    fi

    # Check UFW if available
    if command -v ufw &> /dev/null; then
        UFW_STATUS=$(sudo ufw status 2>/dev/null | head -1)
        echo "UFW status: $UFW_STATUS"
    fi
else
    echo -e "${YELLOW}!${NC} Cannot check firewall (no sudo access)"
fi
echo ""

# 6. Local Service Check (if localhost)
if [ "$IP" == "127.0.0.1" ] || [ "$TARGET_HOST" == "localhost" ]; then
    echo -e "${CYAN}[6] Local Service Check${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if ss -tlnp 2>/dev/null | grep -q ":$TARGET_PORT "; then
        echo -e "${GREEN}✓${NC} Service is listening on port $TARGET_PORT"
        ss -tlnp 2>/dev/null | grep ":$TARGET_PORT " || true
    else
        echo -e "${RED}✗${NC} No service listening on port $TARGET_PORT"

        echo ""
        echo "Listening services:"
        ss -tlnp 2>/dev/null | head -10 || netstat -tlnp 2>/dev/null | head -10 || echo "Cannot list services"
    fi
    echo ""
fi

# 7. Network Route Check
echo -e "${CYAN}[7] Routing Table${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v ip &> /dev/null; then
    echo "Route to $IP:"
    ip route get "$IP" 2>/dev/null || echo "Cannot determine route"
else
    route -n 2>/dev/null | grep -E "^0\.0\.0\.0|^default" || echo "Cannot determine route"
fi
echo ""

# 8. MTU Check
echo -e "${CYAN}[8] MTU and Packet Size${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ping -M do -s 1472 -c 1 -W 2 "$IP" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} MTU 1500 is working (standard Ethernet)"
else
    echo -e "${YELLOW}!${NC} MTU 1500 may be too large (packet fragmentation issue)"
    echo "Testing smaller MTU..."

    for mtu in 1400 1200 1000; do
        if ping -M do -s $((mtu-28)) -c 1 -W 2 "$IP" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} MTU $mtu works"
            break
        fi
    done
fi
echo ""

# 9. Port Scan (if nmap available)
if command -v nmap &> /dev/null; then
    echo -e "${CYAN}[9] Port Scan (nmap)${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Scanning common ports on $IP..."
    nmap -p "$TARGET_PORT" "$IP" 2>/dev/null || echo "nmap scan failed"
    echo ""
fi

# 10. Summary and Recommendations
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Summary and Recommendations${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Determine connectivity status
if timeout 5 bash -c "cat < /dev/null > /dev/tcp/$IP/$TARGET_PORT" 2>/dev/null; then
    echo -e "${GREEN}✓ CONNECTION SUCCESSFUL${NC}"
    echo ""
    echo "The connection to $TARGET_HOST:$TARGET_PORT is working."
else
    echo -e "${RED}✗ CONNECTION FAILED${NC}"
    echo ""
    echo "Troubleshooting steps:"

    # DNS issues
    if [ -z "$IP" ]; then
        echo "  1. Fix DNS resolution:"
        echo "     - Check /etc/resolv.conf"
        echo "     - Try using IP address directly"
        echo "     - Check /etc/hosts for local entries"
    fi

    # Firewall issues
    echo "  2. Check firewalls:"
    echo "     - Local firewall: sudo ufw status"
    echo "     - Remote firewall: check target's iptables/ufw"
    echo "     - Cloud security groups (if cloud-hosted)"

    # Service issues
    echo "  3. Verify service is running:"
    echo "     - SSH to target: ssh user@$TARGET_HOST"
    echo "     - Check service: sudo systemctl status <service>"
    echo "     - Check listening ports: sudo ss -tlnp"

    # Network issues
    echo "  4. Check network connectivity:"
    echo "     - Ping test: ping $IP"
    echo "     - Traceroute: traceroute $IP"
    echo "     - MTU issues: check for packet fragmentation"

    # Port issues
    echo "  5. Verify port number:"
    echo "     - Check application configuration"
    echo "     - Check documentation for correct port"
    echo "     - Common ports: SSH(22), HTTP(80), HTTPS(443), PostgreSQL(5432), Redis(6379)"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
