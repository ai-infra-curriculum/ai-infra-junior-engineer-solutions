#!/bin/bash
#
# audit_security.sh - Security audit for ML infrastructure
#
# Description:
#   Comprehensive security audit including SSH configuration, firewall rules,
#   open ports, weak settings, and security best practices validation.
#
# Usage:
#   ./audit_security.sh [OPTIONS]
#
# Options:
#   -r, --report FILE     Generate report to file
#   -f, --fix             Auto-fix issues (interactive)
#   -s, --severity LEVEL  Minimum severity to report (low, medium, high, critical)
#   -j, --json            Output in JSON format
#   -v, --verbose         Verbose output
#   -h, --help            Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
REPORT_FILE=""
AUTO_FIX=false
MIN_SEVERITY="low"
JSON_OUTPUT=false
VERBOSE=false

# Issue tracking
declare -a ISSUES
declare -a RECOMMENDATIONS

# Severity levels
declare -A SEVERITY_LEVEL=(
    ["critical"]=4
    ["high"]=3
    ["medium"]=2
    ["low"]=1
)

# ===========================
# Colors
# ===========================

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly RESET='\033[0m'
readonly BOLD='\033[1m'

# ===========================
# Issue Reporting
# ===========================

add_issue() {
    local severity="$1"
    local category="$2"
    local title="$3"
    local description="$4"
    local remediation="${5:-}"

    # Check severity filter
    local sev_value=${SEVERITY_LEVEL[$severity]}
    local min_value=${SEVERITY_LEVEL[$MIN_SEVERITY]}

    if [[ $sev_value -lt $min_value ]]; then
        return 0
    fi

    local issue=$(cat <<EOF
SEVERITY: $severity
CATEGORY: $category
TITLE: $title
DESCRIPTION: $description
REMEDIATION: $remediation
---
EOF
)

    ISSUES+=("$issue")
}

add_recommendation() {
    local recommendation="$1"
    RECOMMENDATIONS+=("$recommendation")
}

# ===========================
# SSH Configuration Audit
# ===========================

audit_ssh_config() {
    echo -e "${BOLD}${CYAN}[1/7] Auditing SSH Configuration${RESET}"

    local ssh_config="/etc/ssh/sshd_config"

    if [[ ! -f "$ssh_config" ]]; then
        add_issue "high" "SSH" "SSH config not found" "File $ssh_config does not exist" "Install and configure OpenSSH server"
        return 1
    fi

    # Check PermitRootLogin
    if grep -q "^PermitRootLogin yes" "$ssh_config" 2>/dev/null; then
        add_issue "high" "SSH" "Root login enabled" "SSH allows direct root login" "Set 'PermitRootLogin no' in $ssh_config"
    elif ! grep -q "^PermitRootLogin" "$ssh_config" 2>/dev/null; then
        add_issue "medium" "SSH" "PermitRootLogin not explicitly set" "Default behavior may vary" "Set 'PermitRootLogin no' in $ssh_config"
    else
        echo -e "  ${GREEN}✓${RESET} Root login disabled"
    fi

    # Check PasswordAuthentication
    if grep -q "^PasswordAuthentication yes" "$ssh_config" 2>/dev/null; then
        add_issue "high" "SSH" "Password authentication enabled" "SSH allows password-based login" "Set 'PasswordAuthentication no' and use key-based auth"
    else
        echo -e "  ${GREEN}✓${RESET} Password authentication disabled"
    fi

    # Check SSH protocol
    if grep -q "^Protocol 1" "$ssh_config" 2>/dev/null; then
        add_issue "critical" "SSH" "SSH Protocol 1 enabled" "Insecure SSH protocol version in use" "Remove Protocol directive or set to 2"
    fi

    # Check PermitEmptyPasswords
    if grep -q "^PermitEmptyPasswords yes" "$ssh_config" 2>/dev/null; then
        add_issue "critical" "SSH" "Empty passwords allowed" "SSH allows login without password" "Set 'PermitEmptyPasswords no' in $ssh_config"
    else
        echo -e "  ${GREEN}✓${RESET} Empty passwords not allowed"
    fi

    # Check X11Forwarding
    if grep -q "^X11Forwarding yes" "$ssh_config" 2>/dev/null; then
        add_issue "low" "SSH" "X11 forwarding enabled" "X11 forwarding may not be necessary" "Set 'X11Forwarding no' if not needed"
    fi

    # Check MaxAuthTries
    local max_auth_tries=$(grep "^MaxAuthTries" "$ssh_config" 2>/dev/null | awk '{print $2}')
    if [[ -n "$max_auth_tries" ]] && [[ $max_auth_tries -gt 3 ]]; then
        add_issue "medium" "SSH" "MaxAuthTries too high" "Current value: $max_auth_tries (recommended: 3)" "Set 'MaxAuthTries 3' in $ssh_config"
    fi

    # Check ClientAliveInterval
    if ! grep -q "^ClientAliveInterval" "$ssh_config" 2>/dev/null; then
        add_recommendation "Set ClientAliveInterval to prevent idle connections (e.g., ClientAliveInterval 300)"
    fi

    echo ""
}

# ===========================
# Firewall Audit
# ===========================

audit_firewall() {
    echo -e "${BOLD}${CYAN}[2/7] Auditing Firewall Configuration${RESET}"

    # Check if UFW is installed
    if command -v ufw &> /dev/null; then
        local ufw_status=$(sudo ufw status | head -1)

        if echo "$ufw_status" | grep -q "inactive"; then
            add_issue "high" "Firewall" "Firewall disabled" "UFW firewall is not active" "Enable firewall: sudo ufw enable"
        else
            echo -e "  ${GREEN}✓${RESET} UFW firewall is active"

            # Check default policies
            local default_incoming=$(sudo ufw status verbose | grep "Default:" | grep "incoming" | awk '{print $3}')
            if [[ "$default_incoming" != "deny" ]]; then
                add_issue "high" "Firewall" "Default incoming policy not deny" "Current: $default_incoming" "Set default: sudo ufw default deny incoming"
            else
                echo -e "  ${GREEN}✓${RESET} Default incoming policy: deny"
            fi
        fi
    elif command -v iptables &> /dev/null; then
        local rules_count=$(sudo iptables -L INPUT -n | wc -l)
        if [[ $rules_count -le 2 ]]; then
            add_issue "high" "Firewall" "No iptables rules configured" "System may be exposed" "Configure iptables firewall rules"
        else
            echo -e "  ${GREEN}✓${RESET} iptables rules configured"
        fi
    else
        add_issue "critical" "Firewall" "No firewall installed" "System has no firewall protection" "Install and configure UFW or iptables"
    fi

    echo ""
}

# ===========================
# Open Ports Audit
# ===========================

audit_open_ports() {
    echo -e "${BOLD}${CYAN}[3/7] Auditing Open Ports${RESET}"

    # Get listening ports
    local listening_ports=$(ss -tulpn 2>/dev/null | grep LISTEN || netstat -tulpn 2>/dev/null | grep LISTEN)

    if [[ -z "$listening_ports" ]]; then
        echo "  No listening ports found (or insufficient permissions)"
        echo ""
        return 0
    fi

    # Common dangerous ports
    local dangerous_ports=("23" "21" "69" "135" "139" "445" "3389")

    for port in "${dangerous_ports[@]}"; do
        if echo "$listening_ports" | grep -q ":$port "; then
            add_issue "high" "Ports" "Dangerous port open: $port" "Port $port is listening" "Close port $port or restrict access"
        fi
    done

    # Check for services listening on 0.0.0.0
    local public_services=$(echo "$listening_ports" | grep "0.0.0.0" || true)
    if [[ -n "$public_services" ]]; then
        local count=$(echo "$public_services" | wc -l)
        add_recommendation "Review $count services listening on all interfaces (0.0.0.0)"
    fi

    # List all listening ports
    echo "  Listening ports:"
    echo "$listening_ports" | awk '{print "    " $5}' | sort -u

    echo ""
}

# ===========================
# User and Authentication Audit
# ===========================

audit_users() {
    echo -e "${BOLD}${CYAN}[4/7] Auditing Users and Authentication${RESET}"

    # Check for users with UID 0 (root equivalent)
    local root_users=$(awk -F: '$3 == 0 {print $1}' /etc/passwd)
    local root_count=$(echo "$root_users" | wc -w)

    if [[ $root_count -gt 1 ]]; then
        add_issue "critical" "Users" "Multiple root-equivalent users" "Users with UID 0: $root_users" "Remove UID 0 from non-root users"
    else
        echo -e "  ${GREEN}✓${RESET} Only root has UID 0"
    fi

    # Check for users without passwords
    local users_without_passwords=$(sudo awk -F: '($2 == "" || $2 == "!") {print $1}' /etc/shadow 2>/dev/null | grep -v "^#" || true)
    if [[ -n "$users_without_passwords" ]]; then
        local count=$(echo "$users_without_passwords" | wc -w)
        add_issue "medium" "Users" "$count users without passwords" "Users: $users_without_passwords" "Set passwords or disable accounts"
    else
        echo -e "  ${GREEN}✓${RESET} All users have passwords"
    fi

    # Check password policy
    if [[ -f /etc/login.defs ]]; then
        local pass_max_days=$(grep "^PASS_MAX_DAYS" /etc/login.defs | awk '{print $2}')
        if [[ -n "$pass_max_days" ]] && [[ $pass_max_days -gt 90 ]]; then
            add_issue "low" "Users" "Password max age too long" "Current: $pass_max_days days (recommended: 90)" "Set PASS_MAX_DAYS to 90 in /etc/login.defs"
        fi
    fi

    # Check for inactive users with shell access
    local users_with_shell=$(awk -F: '$7 ~ /(bash|sh|zsh)$/ {print $1}' /etc/passwd)
    echo "  Users with shell access: $(echo "$users_with_shell" | wc -w)"

    echo ""
}

# ===========================
# SSH Keys Audit
# ===========================

audit_ssh_keys() {
    echo -e "${BOLD}${CYAN}[5/7] Auditing SSH Keys${RESET}"

    local users_home=$(find /home -maxdepth 1 -type d 2>/dev/null)

    for user_home in $users_home; do
        local username=$(basename "$user_home")
        local auth_keys="$user_home/.ssh/authorized_keys"

        if [[ -f "$auth_keys" ]]; then
            # Check permissions
            local perms=$(stat -c %a "$auth_keys" 2>/dev/null || stat -f %Mp%Lp "$auth_keys" 2>/dev/null)
            if [[ "$perms" != "600" ]] && [[ "$perms" != "0600" ]]; then
                add_issue "high" "SSH Keys" "Incorrect permissions on authorized_keys" "$auth_keys has permissions $perms" "Set: chmod 600 $auth_keys"
            fi

            # Count keys
            local key_count=$(grep -c "^ssh-" "$auth_keys" 2>/dev/null || echo 0)
            if [[ $key_count -gt 10 ]]; then
                add_recommendation "User $username has $key_count SSH keys - review if all are necessary"
            fi

            # Check for weak key types
            if grep -q "^ssh-rsa.*1024" "$auth_keys" 2>/dev/null; then
                add_issue "medium" "SSH Keys" "Weak RSA key (1024-bit)" "User $username has weak RSA key" "Replace with RSA 4096-bit or Ed25519 key"
            fi
        fi
    done

    # Check root's authorized_keys
    if [[ -f /root/.ssh/authorized_keys ]]; then
        local root_keys=$(grep -c "^ssh-" /root/.ssh/authorized_keys 2>/dev/null || echo 0)
        if [[ $root_keys -gt 0 ]]; then
            add_issue "medium" "SSH Keys" "Root has authorized_keys" "$root_keys keys found for root" "Disable root login and use sudo"
        fi
    fi

    echo "  SSH key audit completed"
    echo ""
}

# ===========================
# System Updates Audit
# ===========================

audit_system_updates() {
    echo -e "${BOLD}${CYAN}[6/7] Auditing System Updates${RESET}"

    # Check for available updates (Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        echo "  Checking for available updates (this may take a moment)..."
        sudo apt-get update > /dev/null 2>&1 || true

        local updates_available=$(apt list --upgradable 2>/dev/null | grep -c "upgradable" || echo 0)
        local security_updates=$(apt list --upgradable 2>/dev/null | grep -ci "security" || echo 0)

        if [[ $security_updates -gt 0 ]]; then
            add_issue "high" "Updates" "$security_updates security updates available" "System has pending security updates" "Run: sudo apt-get upgrade"
            echo -e "  ${RED}⚠${RESET} $security_updates security updates available"
        else
            echo -e "  ${GREEN}✓${RESET} No security updates pending"
        fi

        if [[ $updates_available -gt $security_updates ]]; then
            local regular_updates=$((updates_available - security_updates))
            add_recommendation "$regular_updates regular updates available"
        fi

    # Check for available updates (RHEL/CentOS)
    elif command -v yum &> /dev/null; then
        local updates_available=$(sudo yum check-update 2>/dev/null | grep -c "^[a-zA-Z]" || echo 0)
        if [[ $updates_available -gt 0 ]]; then
            add_issue "medium" "Updates" "$updates_available updates available" "System has pending updates" "Run: sudo yum update"
        else
            echo -e "  ${GREEN}✓${RESET} System is up to date"
        fi
    else
        echo "  Package manager not recognized"
    fi

    # Check kernel version
    local current_kernel=$(uname -r)
    echo "  Current kernel: $current_kernel"

    echo ""
}

# ===========================
# Security Tools Audit
# ===========================

audit_security_tools() {
    echo -e "${BOLD}${CYAN}[7/7] Auditing Security Tools${RESET}"

    # Check for fail2ban
    if command -v fail2ban-client &> /dev/null; then
        if sudo systemctl is-active --quiet fail2ban 2>/dev/null; then
            echo -e "  ${GREEN}✓${RESET} fail2ban is installed and running"
        else
            add_issue "medium" "Security Tools" "fail2ban not running" "fail2ban is installed but not active" "Start: sudo systemctl start fail2ban"
        fi
    else
        add_recommendation "Install fail2ban for brute-force protection"
    fi

    # Check for AppArmor/SELinux
    if command -v aa-status &> /dev/null; then
        if sudo aa-status --enabled 2>/dev/null; then
            echo -e "  ${GREEN}✓${RESET} AppArmor is enabled"
        else
            add_issue "low" "Security Tools" "AppArmor not enabled" "Mandatory Access Control not active" "Enable AppArmor"
        fi
    elif command -v getenforce &> /dev/null; then
        local selinux_status=$(getenforce 2>/dev/null || echo "Unknown")
        if [[ "$selinux_status" == "Enforcing" ]]; then
            echo -e "  ${GREEN}✓${RESET} SELinux is enforcing"
        else
            add_issue "low" "Security Tools" "SELinux not enforcing" "Current status: $selinux_status" "Enable SELinux enforcing mode"
        fi
    else
        add_recommendation "Consider enabling AppArmor or SELinux for Mandatory Access Control"
    fi

    # Check for rootkit detection
    if command -v rkhunter &> /dev/null; then
        echo -e "  ${GREEN}✓${RESET} rkhunter is installed"
    else
        add_recommendation "Install rkhunter for rootkit detection"
    fi

    # Check for intrusion detection
    if command -v aide &> /dev/null; then
        echo -e "  ${GREEN}✓${RESET} AIDE is installed"
    else
        add_recommendation "Install AIDE for file integrity monitoring"
    fi

    echo ""
}

# ===========================
# Report Generation
# ===========================

generate_summary() {
    echo -e "${BOLD}${CYAN}Security Audit Summary${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Count issues by severity
    local critical=0
    local high=0
    local medium=0
    local low=0

    for issue in "${ISSUES[@]}"; do
        if echo "$issue" | grep -q "SEVERITY: critical"; then
            ((critical++))
        elif echo "$issue" | grep -q "SEVERITY: high"; then
            ((high++))
        elif echo "$issue" | grep -q "SEVERITY: medium"; then
            ((medium++))
        elif echo "$issue" | grep -q "SEVERITY: low"; then
            ((low++))
        fi
    done

    local total=$((critical + high + medium + low))

    echo "Total Issues: $total"
    if [[ $critical -gt 0 ]]; then
        echo -e "  ${RED}Critical: $critical${RESET}"
    fi
    if [[ $high -gt 0 ]]; then
        echo -e "  ${RED}High: $high${RESET}"
    fi
    if [[ $medium -gt 0 ]]; then
        echo -e "  ${YELLOW}Medium: $medium${RESET}"
    fi
    if [[ $low -gt 0 ]]; then
        echo -e "  ${BLUE}Low: $low${RESET}"
    fi

    echo ""

    # Display issues
    if [[ ${#ISSUES[@]} -gt 0 ]]; then
        echo -e "${BOLD}Security Issues:${RESET}"
        echo "========================================"
        for issue in "${ISSUES[@]}"; do
            echo "$issue"
        done
    else
        echo -e "${GREEN}No security issues found!${RESET}"
        echo ""
    fi

    # Display recommendations
    if [[ ${#RECOMMENDATIONS[@]} -gt 0 ]]; then
        echo -e "${BOLD}Recommendations:${RESET}"
        echo "========================================"
        for rec in "${RECOMMENDATIONS[@]}"; do
            echo "• $rec"
        done
        echo ""
    fi
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Security audit for ML infrastructure.

OPTIONS:
    -r, --report FILE         Generate report to file
    -f, --fix                 Auto-fix issues (interactive)
    -s, --severity LEVEL      Minimum severity (low, medium, high, critical)
    -j, --json                Output in JSON format
    -v, --verbose             Verbose output
    -h, --help                Display this help message

EXAMPLES:
    # Run security audit
    $SCRIPT_NAME

    # Generate report
    $SCRIPT_NAME --report security-audit.txt

    # Show only high and critical issues
    $SCRIPT_NAME --severity high

    # JSON output
    $SCRIPT_NAME --json --report audit.json

NOTE:
    Some checks require root privileges.
    Run with sudo for complete audit.

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -r|--report)
                REPORT_FILE="$2"
                shift 2
                ;;
            -f|--fix)
                AUTO_FIX=true
                shift
                ;;
            -s|--severity)
                MIN_SEVERITY="$2"
                shift 2
                ;;
            -j|--json)
                JSON_OUTPUT=true
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

    echo -e "${BOLD}${CYAN}ML Infrastructure Security Audit${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    # Run audits
    audit_ssh_config
    audit_firewall
    audit_open_ports
    audit_users
    audit_ssh_keys
    audit_system_updates
    audit_security_tools

    # Generate summary
    generate_summary

    # Save report if requested
    if [[ -n "$REPORT_FILE" ]]; then
        generate_summary > "$REPORT_FILE"
        echo "Report saved to: $REPORT_FILE"
    fi

    # Return exit code based on issues
    if [[ ${#ISSUES[@]} -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
