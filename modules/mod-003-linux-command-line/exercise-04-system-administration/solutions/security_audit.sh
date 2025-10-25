#!/bin/bash
#
# security_audit.sh - Security auditing for ML infrastructure
#
# Usage: ./security_audit.sh [OPTIONS]
#

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/security-audit.log"
readonly REPORT_DIR="/var/log/security-reports"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'
BOLD='\033[1m'

VERBOSE=false
REPORT_FILE=""

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
    [[ "$VERBOSE" == true ]] && echo "$*"
}

check_system_updates() {
    echo -e "${BOLD}System Updates:${RESET}"

    if command -v apt &>/dev/null; then
        local updates=$(apt list --upgradable 2>/dev/null | grep -c upgradable || echo 0)
        if [[ $updates -gt 0 ]]; then
            echo -e "  ${YELLOW}⚠ $updates package(s) available for update${RESET}"
            log_message "WARNING: $updates updates available"
        else
            echo -e "  ${GREEN}✓ System is up to date${RESET}"
        fi
    fi
}

check_user_accounts() {
    echo -e "${BOLD}User Account Security:${RESET}"

    # Check for UID 0 users
    local uid0_users=$(awk -F: '$3 == 0 && $1 != "root" {print $1}' /etc/passwd)
    if [[ -n "$uid0_users" ]]; then
        echo -e "  ${RED}✗ Non-root users with UID 0:${RESET}"
        echo "$uid0_users" | sed 's/^/      /'
        log_message "ALERT: Non-root UID 0 users found: $uid0_users"
    else
        echo -e "  ${GREEN}✓ No unauthorized UID 0 accounts${RESET}"
    fi

    # Check for accounts without passwords
    local no_pass=$(passwd -Sa 2>/dev/null | grep " NP " | wc -l)
    if [[ $no_pass -gt 0 ]]; then
        echo -e "  ${YELLOW}⚠ $no_pass account(s) without password${RESET}"
        log_message "WARNING: $no_pass accounts without password"
    else
        echo -e "  ${GREEN}✓ All accounts have passwords${RESET}"
    fi

    # Check for locked accounts
    local locked=$(passwd -Sa 2>/dev/null | grep " L " | wc -l)
    echo -e "  ${BLUE}ℹ $locked locked account(s)${RESET}"
}

check_ssh_security() {
    echo -e "${BOLD}SSH Security:${RESET}"

    local sshd_config="/etc/ssh/sshd_config"

    if [[ ! -f "$sshd_config" ]]; then
        echo -e "  ${YELLOW}⚠ SSH config not found${RESET}"
        return
    fi

    # Check root login
    if grep -q "^PermitRootLogin yes" "$sshd_config"; then
        echo -e "  ${RED}✗ Root login via SSH is enabled${RESET}"
        log_message "ALERT: Root SSH login enabled"
    else
        echo -e "  ${GREEN}✓ Root login disabled or restricted${RESET}"
    fi

    # Check password authentication
    if grep -q "^PasswordAuthentication yes" "$sshd_config"; then
        echo -e "  ${YELLOW}⚠ Password authentication enabled${RESET}"
        log_message "WARNING: SSH password authentication enabled"
    else
        echo -e "  ${GREEN}✓ Password authentication disabled${RESET}"
    fi

    # Check empty passwords
    if grep -q "^PermitEmptyPasswords yes" "$sshd_config"; then
        echo -e "  ${RED}✗ Empty passwords permitted${RESET}"
        log_message "ALERT: Empty SSH passwords permitted"
    else
        echo -e "  ${GREEN}✓ Empty passwords not permitted${RESET}"
    fi
}

check_firewall() {
    echo -e "${BOLD}Firewall Status:${RESET}"

    # Check UFW
    if command -v ufw &>/dev/null; then
        local ufw_status=$(ufw status | head -1 | awk '{print $2}')
        if [[ "$ufw_status" == "active" ]]; then
            echo -e "  ${GREEN}✓ UFW is active${RESET}"
        else
            echo -e "  ${YELLOW}⚠ UFW is inactive${RESET}"
            log_message "WARNING: UFW firewall inactive"
        fi
    # Check iptables
    elif command -v iptables &>/dev/null; then
        local rules=$(iptables -L | wc -l)
        if [[ $rules -gt 8 ]]; then
            echo -e "  ${GREEN}✓ iptables rules configured${RESET}"
        else
            echo -e "  ${YELLOW}⚠ iptables rules minimal${RESET}"
        fi
    else
        echo -e "  ${RED}✗ No firewall detected${RESET}"
        log_message "ALERT: No firewall detected"
    fi
}

check_file_permissions() {
    echo -e "${BOLD}Critical File Permissions:${RESET}"

    local issues=0

    # Check /etc/passwd
    local passwd_perm=$(stat -c %a /etc/passwd 2>/dev/null || stat -f %A /etc/passwd 2>/dev/null)
    if [[ "$passwd_perm" != "644" ]]; then
        echo -e "  ${YELLOW}⚠ /etc/passwd: $passwd_perm (should be 644)${RESET}"
        ((issues++))
    fi

    # Check /etc/shadow
    local shadow_perm=$(stat -c %a /etc/shadow 2>/dev/null || stat -f %A /etc/shadow 2>/dev/null)
    if [[ "$shadow_perm" != "640" ]] && [[ "$shadow_perm" != "600" ]]; then
        echo -e "  ${RED}✗ /etc/shadow: $shadow_perm (should be 640 or 600)${RESET}"
        ((issues++))
        log_message "ALERT: Incorrect /etc/shadow permissions: $shadow_perm"
    fi

    # Check for world-writable files
    local writable=$(find /etc /home /opt -type f -perm -002 2>/dev/null | head -5 | wc -l)
    if [[ $writable -gt 0 ]]; then
        echo -e "  ${YELLOW}⚠ World-writable files found in critical directories${RESET}"
        ((issues++))
    fi

    if [[ $issues -eq 0 ]]; then
        echo -e "  ${GREEN}✓ No permission issues detected${RESET}"
    fi
}

check_suspicious_processes() {
    echo -e "${BOLD}Suspicious Processes:${RESET}"

    # Check for unusual processes
    local unusual=0

    # Check for processes listening on unexpected ports
    local listeners=$(ss -tlnp 2>/dev/null | grep LISTEN | wc -l)
    echo -e "  ${BLUE}ℹ $listeners listening service(s)${RESET}"

    # Check for processes running as root
    local root_procs=$(ps aux | grep "^root" | wc -l)
    echo -e "  ${BLUE}ℹ $root_procs process(es) running as root${RESET}"

    # Check for high CPU processes
    local high_cpu=$(ps aux --sort=-%cpu | head -2 | tail -1 | awk '{print $11}')
    echo -e "  ${BLUE}ℹ Top CPU process: $high_cpu${RESET}"
}

check_disk_encryption() {
    echo -e "${BOLD}Disk Encryption:${RESET}"

    if command -v lsblk &>/dev/null; then
        local encrypted=$(lsblk -o NAME,TYPE,MOUNTPOINT | grep -c crypt || echo 0)
        if [[ $encrypted -gt 0 ]]; then
            echo -e "  ${GREEN}✓ $encrypted encrypted volume(s) detected${RESET}"
        else
            echo -e "  ${YELLOW}⚠ No encrypted volumes detected${RESET}"
            log_message "WARNING: No disk encryption detected"
        fi
    fi
}

check_failed_logins() {
    echo -e "${BOLD}Failed Login Attempts:${RESET}"

    if [[ -f /var/log/auth.log ]]; then
        local failed=$(grep "Failed password" /var/log/auth.log 2>/dev/null | wc -l)
        if [[ $failed -gt 10 ]]; then
            echo -e "  ${YELLOW}⚠ $failed failed login attempts found${RESET}"
            log_message "WARNING: $failed failed login attempts"

            # Show recent failed attempts
            echo "  Recent failures:"
            grep "Failed password" /var/log/auth.log 2>/dev/null | tail -3 | sed 's/^/    /'
        else
            echo -e "  ${GREEN}✓ Minimal failed login attempts${RESET}"
        fi
    fi
}

check_open_ports() {
    echo -e "${BOLD}Open Ports:${RESET}"

    if command -v ss &>/dev/null; then
        echo "  Listening TCP ports:"
        ss -tln | grep LISTEN | awk '{print $4}' | sed 's/.*://' | sort -n | uniq | sed 's/^/    /'
    fi
}

scan_for_rootkits() {
    echo -e "${BOLD}Rootkit Detection:${RESET}"

    if command -v rkhunter &>/dev/null; then
        echo -e "  ${BLUE}Running rkhunter...${RESET}"
        sudo rkhunter --check --skip-keypress --report-warnings-only 2>/dev/null | head -10 || \
            echo -e "  ${GREEN}✓ No warnings from rkhunter${RESET}"
    elif command -v chkrootkit &>/dev/null; then
        echo -e "  ${BLUE}Running chkrootkit...${RESET}"
        sudo chkrootkit 2>/dev/null | grep -i "infected" || \
            echo -e "  ${GREEN}✓ No infections found${RESET}"
    else
        echo -e "  ${YELLOW}⚠ No rootkit scanner installed${RESET}"
        echo "    Install: apt install rkhunter chkrootkit"
    fi
}

generate_security_report() {
    mkdir -p "$REPORT_DIR"
    local report_file="${REPORT_FILE:-$REPORT_DIR/security-audit-$(date +%Y%m%d-%H%M%S).txt}"

    echo "Generating security audit report..."

    {
        echo "========================================"
        echo "Security Audit Report"
        echo "========================================"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Hostname: $(hostname)"
        echo "Kernel: $(uname -r)"
        echo ""

        echo "System Information:"
        echo "-------------------"
        lsb_release -d 2>/dev/null | cut -f2- || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2
        echo "Uptime: $(uptime -p)"
        echo ""

        echo "User Accounts:"
        echo "--------------"
        echo "Total users: $(awk -F: '$3 >= 1000 && $3 <= 60000' /etc/passwd | wc -l)"
        echo "Sudo users: $(getent group sudo | cut -d: -f4 | tr ',' '\n' | wc -l)"
        echo ""

        echo "Security Checks:"
        echo "----------------"
        check_system_updates
        echo ""
        check_user_accounts
        echo ""
        check_ssh_security
        echo ""
        check_firewall
        echo ""
        check_file_permissions
        echo ""
        check_failed_logins
        echo ""
        check_open_ports
        echo ""

        echo "Recent Security Events:"
        echo "-----------------------"
        journalctl --since "7 days ago" -p warning --no-pager -n 10 2>/dev/null || \
            echo "No recent security events"

        echo ""
        echo "========================================"
        echo "End of Report"
        echo "========================================"
    } | tee "$report_file"

    echo ""
    echo -e "${GREEN}✓ Report saved: $report_file${RESET}"
    log_message "Security audit report generated: $report_file"
}

perform_full_audit() {
    echo -e "${BOLD}${CYAN}Security Audit${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    log_message "Security audit started"

    check_system_updates
    echo ""

    check_user_accounts
    echo ""

    check_ssh_security
    echo ""

    check_firewall
    echo ""

    check_file_permissions
    echo ""

    check_suspicious_processes
    echo ""

    check_disk_encryption
    echo ""

    check_failed_logins
    echo ""

    check_open_ports
    echo ""

    scan_for_rootkits
    echo ""

    echo -e "${CYAN}========================================${RESET}"
    echo -e "${GREEN}${BOLD}Audit completed${RESET}"

    log_message "Security audit completed"
}

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Security auditing for ML infrastructure.

OPTIONS:
    -a, --audit          Perform full security audit
    -r, --report [FILE]  Generate detailed report
    -v, --verbose        Verbose output
    -h, --help           Display help

EXAMPLES:
    $SCRIPT_NAME --audit
    $SCRIPT_NAME --report
    $SCRIPT_NAME --audit --report

CHECKS PERFORMED:
    - System updates
    - User account security
    - SSH configuration
    - Firewall status
    - File permissions
    - Suspicious processes
    - Failed login attempts
    - Open ports
    - Rootkit detection

LOGS:
    Audit log: $LOG_FILE
    Reports: $REPORT_DIR

EOF
}

main() {
    mkdir -p "$REPORT_DIR" 2>/dev/null || true
    touch "$LOG_FILE" 2>/dev/null || true

    [[ $# -eq 0 ]] && { usage; exit 1; }

    local do_audit=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -a|--audit)
                do_audit=true
                shift
                ;;
            -r|--report)
                if [[ -n "${2:-}" ]] && [[ ! "$2" =~ ^- ]]; then
                    REPORT_FILE="$2"
                    shift 2
                else
                    REPORT_FILE=""
                    shift
                fi
                generate_security_report
                exit 0
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

    if [[ "$do_audit" == true ]]; then
        perform_full_audit
    fi
}

main "$@"
