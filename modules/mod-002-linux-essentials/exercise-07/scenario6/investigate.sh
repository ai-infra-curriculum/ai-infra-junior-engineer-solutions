#!/bin/bash
###############################################################################
# Scenario 6: Network Connectivity - Investigation Script
###############################################################################
#
# Problem: Cannot download models, API timeouts, DNS failures
# Common causes: Network config, firewall, DNS, proxy, routing
#

set -u

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' CYAN='' BOLD='' NC=''
fi

section() { echo -e "\n${BOLD}${BLUE}=== $* ===${NC}\n"; }
subsection() { echo -e "${CYAN}$*${NC}"; }
log_info() { echo -e "  $*"; }
log_error() { echo -e "  ${RED}✗${NC} $*"; }
log_success() { echo -e "  ${GREEN}✓${NC} $*"; }
log_warning() { echo -e "  ${YELLOW}⚠${NC} $*"; }

echo -e "${BOLD}${RED}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${RED}║  Scenario 6: Network Connectivity Investigation           ║${NC}"
echo -e "${BOLD}${RED}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Problem:${NC} Cannot connect to external services"
echo ""

# Test target (default: common ML resource)
TEST_HOST="${1:-huggingface.co}"

section "Step 1: Check Network Interfaces"
echo "Command: ip addr show"
echo ""
ip addr show

echo ""
subsection "Active interfaces:"
active_ifaces=$(ip link show | grep "state UP" | awk -F': ' '{print $2}' | grep -v lo)
if [ -n "$active_ifaces" ]; then
    log_success "Active interfaces: $active_ifaces"
else
    log_error "No active network interfaces (except loopback)"
fi

section "Step 2: Check Default Gateway"
echo "Command: ip route show"
echo ""
ip route show

echo ""
subsection "Default gateway:"
gateway=$(ip route | grep default | awk '{print $3}')
if [ -n "$gateway" ]; then
    log_success "Default gateway: $gateway"

    # Ping gateway
    echo ""
    log_info "Testing gateway connectivity..."
    if ping -c 3 -W 2 "$gateway" &>/dev/null; then
        log_success "Gateway is reachable"
    else
        log_error "Cannot reach gateway!"
        log_info "This indicates a local network issue"
    fi
else
    log_error "No default gateway configured!"
fi

section "Step 3: DNS Configuration"
subsection "DNS servers:"
echo ""

if [ -f /etc/resolv.conf ]; then
    cat /etc/resolv.conf | grep -v "^#" | grep -v "^$"
    echo ""

    nameservers=$(grep "^nameserver" /etc/resolv.conf | awk '{print $2}')
    if [ -n "$nameservers" ]; then
        log_success "DNS servers configured"

        # Test each nameserver
        echo ""
        log_info "Testing DNS servers..."
        for ns in $nameservers; do
            if ping -c 1 -W 2 "$ns" &>/dev/null; then
                log_success "  $ns - reachable"
            else
                log_warning "  $ns - not reachable"
            fi
        done
    else
        log_error "No nameservers configured!"
    fi
else
    log_error "/etc/resolv.conf not found"
fi

section "Step 4: DNS Resolution Test"
echo "Testing DNS resolution for: $TEST_HOST"
echo ""

if command -v nslookup &>/dev/null; then
    subsection "Using nslookup:"
    if nslookup "$TEST_HOST" &>/dev/null; then
        nslookup "$TEST_HOST" | grep -A2 "Name:"
        log_success "DNS resolution working"
    else
        log_error "DNS resolution failed"
    fi
else
    log_warning "nslookup not available"
fi

echo ""

if command -v dig &>/dev/null; then
    subsection "Using dig:"
    dig_output=$(dig +short "$TEST_HOST" 2>/dev/null)
    if [ -n "$dig_output" ]; then
        echo "$dig_output"
        log_success "DNS resolution working"
    else
        log_error "dig query failed"
    fi
else
    log_info "dig not available (install: apt install dnsutils)"
fi

echo ""

subsection "Using getent (uses system resolver):"
if getent hosts "$TEST_HOST" &>/dev/null; then
    getent hosts "$TEST_HOST"
    log_success "System resolver working"
else
    log_error "System resolver failed"
fi

section "Step 5: External Connectivity Test"
echo "Testing connectivity to: $TEST_HOST"
echo ""

# Ping test
subsection "ICMP (ping):"
if ping -c 3 -W 5 "$TEST_HOST" &>/dev/null; then
    log_success "Host is reachable via ping"
    ping -c 3 "$TEST_HOST" | tail -2
else
    log_warning "Ping failed (may be blocked by firewall)"
fi

echo ""

# HTTP/HTTPS test
subsection "HTTP/HTTPS connectivity:"
if command -v curl &>/dev/null; then
    # Test HTTPS
    if curl -s -I --connect-timeout 5 "https://$TEST_HOST" &>/dev/null; then
        log_success "HTTPS connection successful"

        # Get headers
        status=$(curl -s -I --connect-timeout 5 "https://$TEST_HOST" | head -1)
        log_info "  $status"
    else
        log_error "HTTPS connection failed"

        # Try to get more details
        error=$(curl -v "https://$TEST_HOST" 2>&1 | grep -i "error\|failed\|timeout" | head -1)
        if [ -n "$error" ]; then
            log_info "  $error"
        fi
    fi

    echo ""

    # Test HTTP (if HTTPS fails)
    if ! curl -s -I --connect-timeout 5 "https://$TEST_HOST" &>/dev/null; then
        log_info "Trying HTTP (port 80)..."
        if curl -s -I --connect-timeout 5 "http://$TEST_HOST" &>/dev/null; then
            log_warning "HTTP works but HTTPS fails (SSL/TLS issue?)"
        else
            log_error "Both HTTP and HTTPS failed"
        fi
    fi
else
    log_warning "curl not available"

    if command -v wget &>/dev/null; then
        if wget --spider --timeout=5 "https://$TEST_HOST" 2>&1 | grep -q "200 OK"; then
            log_success "wget connection successful"
        else
            log_error "wget connection failed"
        fi
    fi
fi

section "Step 6: Check Proxy Settings"
subsection "Environment variables:"
echo ""

proxy_vars=(HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy)
proxy_found=false

for var in "${proxy_vars[@]}"; do
    value="${!var:-}"
    if [ -n "$value" ]; then
        log_info "$var=$value"
        proxy_found=true
    fi
done

if [ "$proxy_found" = false ]; then
    log_info "No proxy environment variables set"
fi

echo ""
subsection "System proxy configuration:"

if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then
    log_info "APT proxy configuration:"
    cat /etc/apt/apt.conf.d/proxy.conf
else
    log_info "No APT proxy configured"
fi

section "Step 7: Firewall Rules"
subsection "iptables rules:"
echo ""

if command -v iptables &>/dev/null; then
    if sudo iptables -L -n | grep -q "Chain"; then
        log_info "Firewall rules exist"
        echo ""
        log_info "Output chain (outgoing traffic):"
        sudo iptables -L OUTPUT -n | head -10

        # Check if there are DROP rules
        if sudo iptables -L OUTPUT -n | grep -q "DROP"; then
            log_warning "Found DROP rules in OUTPUT chain"
        fi
    else
        log_info "No iptables rules configured"
    fi
else
    log_info "iptables not available"
fi

echo ""
subsection "UFW (Uncomplicated Firewall):"

if command -v ufw &>/dev/null; then
    ufw_status=$(sudo ufw status 2>/dev/null)
    echo "$ufw_status"

    if echo "$ufw_status" | grep -q "Status: active"; then
        log_warning "UFW is active - check rules"

        # Check if outgoing is allowed
        if echo "$ufw_status" | grep -q "ALLOW OUT"; then
            log_success "Outgoing connections allowed"
        fi
    else
        log_info "UFW is not active"
    fi
else
    log_info "UFW not installed"
fi

section "Step 8: Port Connectivity Test"
echo "Testing common ML/API ports..."
echo ""

# Common ports for ML services
test_ports=(
    "443:HTTPS"
    "80:HTTP"
    "22:SSH"
    "8080:HTTP-Alt"
)

for port_desc in "${test_ports[@]}"; do
    port=$(echo "$port_desc" | cut -d: -f1)
    desc=$(echo "$port_desc" | cut -d: -f2)

    if command -v nc &>/dev/null; then
        if timeout 2 nc -zv "$TEST_HOST" "$port" 2>&1 | grep -q "succeeded\|open"; then
            log_success "Port $port ($desc) - open"
        else
            log_warning "Port $port ($desc) - closed or filtered"
        fi
    fi
done

if ! command -v nc &>/dev/null; then
    log_info "netcat (nc) not available (install: apt install netcat)"
fi

section "Step 9: Routing Table"
echo "Full routing table:"
echo ""
ip route show table all | head -20

echo ""
subsection "Specific routes:"
ip route get 8.8.8.8 2>/dev/null || log_warning "Cannot determine route to 8.8.8.8"

section "Step 10: Check for VPN/Tunnel Interfaces"
subsection "VPN/tunnel interfaces:"
echo ""

vpn_ifaces=$(ip link show | grep -E "tun|tap|wg|vpn" || true)
if [ -n "$vpn_ifaces" ]; then
    log_warning "VPN/tunnel interfaces detected:"
    echo "$vpn_ifaces"
    log_info "VPN may be affecting connectivity"
else
    log_info "No VPN/tunnel interfaces found"
fi

section "Step 11: SSL/TLS Certificate Issues"
echo "Testing SSL/TLS connectivity..."
echo ""

if command -v openssl &>/dev/null; then
    log_info "Checking SSL certificate for $TEST_HOST..."

    if echo | timeout 5 openssl s_client -connect "$TEST_HOST:443" -servername "$TEST_HOST" 2>&1 | grep -q "Verify return code: 0"; then
        log_success "SSL certificate verification successful"
    else
        verify_code=$(echo | timeout 5 openssl s_client -connect "$TEST_HOST:443" -servername "$TEST_HOST" 2>&1 | grep "Verify return code" | head -1)
        log_warning "SSL verification issue: $verify_code"
    fi
else
    log_info "openssl not available"
fi

section "Step 12: Check System Time"
echo "Checking system time (important for SSL/TLS):"
echo ""

log_info "Current time: $(date)"
log_info "Timezone: $(timedatectl show --property=Timezone --value 2>/dev/null || date +%Z)"

# Check if time is synchronized
if command -v timedatectl &>/dev/null; then
    if timedatectl status | grep -q "synchronized: yes"; then
        log_success "System time is synchronized"
    else
        log_warning "System time may not be synchronized"
        log_info "SSL certificates may fail if time is wrong"
    fi
fi

section "Analysis Summary"
echo -e "${BOLD}Diagnosis:${NC}"
echo ""

issues=0

# Check interface
if [ -z "$active_ifaces" ]; then
    log_error "No active network interfaces"
    issues=$((issues + 1))
else
    log_success "Network interface active"
fi

# Check gateway
if [ -z "$gateway" ]; then
    log_error "No default gateway"
    issues=$((issues + 1))
elif ! ping -c 1 -W 2 "$gateway" &>/dev/null; then
    log_error "Gateway not reachable"
    issues=$((issues + 1))
else
    log_success "Gateway reachable"
fi

# Check DNS
if ! nslookup "$TEST_HOST" &>/dev/null && ! dig +short "$TEST_HOST" &>/dev/null; then
    log_error "DNS resolution failing"
    issues=$((issues + 1))
else
    log_success "DNS resolution working"
fi

# Check external connectivity
if command -v curl &>/dev/null; then
    if curl -s -I --connect-timeout 5 "https://$TEST_HOST" &>/dev/null; then
        log_success "External connectivity working"
    else
        log_error "Cannot reach external host"
        issues=$((issues + 1))
    fi
fi

echo ""

if [ $issues -eq 0 ]; then
    log_success "No major connectivity issues detected"
    echo ""
    log_info "If you're still having problems:"
    log_info "  - Check application-specific proxy settings"
    log_info "  - Verify API keys/authentication"
    log_info "  - Check rate limiting"
else
    echo "Found $issues connectivity issue(s)."
    echo ""
    echo "Common fixes:"

    if [ -z "$gateway" ]; then
        echo "  - Configure default gateway"
    fi

    if ! nslookup "$TEST_HOST" &>/dev/null; then
        echo "  - Fix DNS configuration (/etc/resolv.conf)"
        echo "  - Try using Google DNS: 8.8.8.8"
    fi

    if [ $issues -gt 0 ]; then
        echo "  - Check firewall rules"
        echo "  - Verify proxy settings"
        echo "  - Contact network administrator"
    fi
fi

echo ""
echo -e "${BOLD}${BLUE}Next Steps:${NC}"
echo "  1. Review the analysis above"
echo "  2. Identify the connectivity blocker"
echo "  3. Run the fix script: ./fix.sh"
echo "  4. Test connectivity after fixing"
echo ""
