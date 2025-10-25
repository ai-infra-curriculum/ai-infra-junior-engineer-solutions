#!/bin/bash
#
# user_management.sh - User and group management for ML infrastructure
#
# Description:
#   Comprehensive user and group management including account creation,
#   password policies, SSH key management, and user auditing.
#
# Usage:
#   ./user_management.sh [COMMAND] [OPTIONS]
#
# Commands:
#   add-user USER            Add a new user
#   del-user USER            Delete a user
#   mod-user USER            Modify user properties
#   add-group GROUP          Add a new group
#   del-group GROUP          Delete a group
#   add-to-group USER GROUP  Add user to group
#   list-users               List all users
#   list-groups              List all groups
#   audit                    Generate user audit report
#   ssh-key USER             Manage SSH keys
#   set-password USER        Set user password
#   lock USER                Lock user account
#   unlock USER              Unlock user account
#
# Options:
#   --shell SHELL            Set user shell
#   --home DIR               Set home directory
#   --groups GROUPS          Additional groups
#   --expire DATE            Account expiration date
#   --uid UID                Specific UID
#   --gid GID                Specific GID
#   -f, --force              Force operation
#   -v, --verbose            Verbose output
#   -h, --help               Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/user-management.log"
readonly AUDIT_DIR="/var/log/user-audits"

# Defaults
DEFAULT_SHELL="/bin/bash"
ML_GROUPS=("docker" "sudo" "video")
MIN_UID=1000
MAX_UID=60000

FORCE=false
VERBOSE=false

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
    local user="${SUDO_USER:-$USER}"

    echo "[$timestamp] [$level] [$user] $message" | tee -a "$LOG_FILE"

    if [[ "$VERBOSE" == true ]]; then
        echo -e "[$level] $message"
    fi
}

log_info() {
    log_message "INFO" "$@"
}

log_success() {
    log_message "SUCCESS" "$@"
}

log_warning() {
    log_message "WARNING" "$@"
}

log_error() {
    log_message "ERROR" "$@"
}

# ===========================
# Validation Functions
# ===========================

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}Error: This operation requires root privileges${RESET}"
        echo "Please run with sudo"
        exit 1
    fi
}

validate_username() {
    local username="$1"

    # Check username format
    if ! [[ "$username" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]]; then
        echo -e "${RED}Error: Invalid username format${RESET}"
        echo "Username must start with lowercase letter or underscore"
        echo "Can contain lowercase letters, numbers, underscore, hyphen"
        echo "Maximum 32 characters"
        return 1
    fi

    return 0
}

user_exists() {
    local username="$1"
    id "$username" &>/dev/null
}

group_exists() {
    local group="$1"
    getent group "$group" &>/dev/null
}

# ===========================
# User Management
# ===========================

add_user() {
    local username="$1"
    local shell="${2:-$DEFAULT_SHELL}"
    local home_dir="${3:-/home/$username}"
    local groups="${4:-}"
    local uid="${5:-}"

    check_root

    if ! validate_username "$username"; then
        return 1
    fi

    if user_exists "$username"; then
        echo -e "${RED}Error: User '$username' already exists${RESET}"
        log_error "Failed to add user '$username': already exists"
        return 1
    fi

    echo -e "${BLUE}Creating user: $username${RESET}"

    # Build useradd command
    local cmd="useradd"
    cmd="$cmd -m"  # Create home directory
    cmd="$cmd -s $shell"
    cmd="$cmd -d $home_dir"

    if [[ -n "$uid" ]]; then
        cmd="$cmd -u $uid"
    fi

    if [[ -n "$groups" ]]; then
        cmd="$cmd -G $groups"
    fi

    cmd="$cmd $username"

    # Execute user creation
    if $cmd; then
        echo -e "${GREEN}✓ User created: $username${RESET}"
        log_success "User created: $username (home: $home_dir, shell: $shell)"

        # Set initial password
        echo ""
        echo "Setting password for $username:"
        if passwd "$username"; then
            echo -e "${GREEN}✓ Password set${RESET}"
        else
            echo -e "${YELLOW}Warning: Password not set${RESET}"
        fi

        # Create SSH directory
        local ssh_dir="$home_dir/.ssh"
        mkdir -p "$ssh_dir"
        chmod 700 "$ssh_dir"
        chown "$username:$username" "$ssh_dir"
        echo -e "${GREEN}✓ SSH directory created${RESET}"

        # Display user info
        echo ""
        display_user_info "$username"

        return 0
    else
        echo -e "${RED}✗ Failed to create user: $username${RESET}"
        log_error "Failed to create user: $username"
        return 1
    fi
}

delete_user() {
    local username="$1"
    local remove_home="${2:-false}"

    check_root

    if ! user_exists "$username"; then
        echo -e "${RED}Error: User '$username' does not exist${RESET}"
        return 1
    fi

    # Check if user is logged in
    if who | grep -q "^$username "; then
        echo -e "${YELLOW}Warning: User '$username' is currently logged in${RESET}"

        if [[ "$FORCE" != true ]]; then
            echo "Use --force to delete anyway"
            return 1
        fi
    fi

    echo -e "${BLUE}Deleting user: $username${RESET}"

    # Backup user data
    local backup_dir="/var/backups/users"
    mkdir -p "$backup_dir"
    local backup_file="$backup_dir/${username}_$(date +%Y%m%d_%H%M%S).tar.gz"

    local home_dir=$(eval echo ~"$username")
    if [[ -d "$home_dir" ]]; then
        echo "Backing up user data to $backup_file..."
        tar -czf "$backup_file" -C "$(dirname "$home_dir")" "$(basename "$home_dir")" 2>/dev/null || true
        echo -e "${GREEN}✓ User data backed up${RESET}"
    fi

    # Delete user
    local cmd="userdel"
    if [[ "$remove_home" == true ]]; then
        cmd="$cmd -r"  # Remove home directory
    fi
    cmd="$cmd $username"

    if $cmd; then
        echo -e "${GREEN}✓ User deleted: $username${RESET}"
        log_success "User deleted: $username (backup: $backup_file)"
        return 0
    else
        echo -e "${RED}✗ Failed to delete user: $username${RESET}"
        log_error "Failed to delete user: $username"
        return 1
    fi
}

modify_user() {
    local username="$1"
    shift

    check_root

    if ! user_exists "$username"; then
        echo -e "${RED}Error: User '$username' does not exist${RESET}"
        return 1
    fi

    echo -e "${BLUE}Modifying user: $username${RESET}"

    local cmd="usermod"
    local changes=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --shell)
                cmd="$cmd -s $2"
                changes="${changes}shell=$2 "
                shift 2
                ;;
            --home)
                cmd="$cmd -d $2 -m"
                changes="${changes}home=$2 "
                shift 2
                ;;
            --groups)
                cmd="$cmd -aG $2"
                changes="${changes}groups=$2 "
                shift 2
                ;;
            --expire)
                cmd="$cmd -e $2"
                changes="${changes}expire=$2 "
                shift 2
                ;;
            --uid)
                cmd="$cmd -u $2"
                changes="${changes}uid=$2 "
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    cmd="$cmd $username"

    if [[ -n "$changes" ]]; then
        if $cmd; then
            echo -e "${GREEN}✓ User modified: $username${RESET}"
            echo "  Changes: $changes"
            log_success "User modified: $username ($changes)"
            return 0
        else
            echo -e "${RED}✗ Failed to modify user: $username${RESET}"
            log_error "Failed to modify user: $username"
            return 1
        fi
    else
        echo "No changes specified"
        return 1
    fi
}

lock_user() {
    local username="$1"

    check_root

    if ! user_exists "$username"; then
        echo -e "${RED}Error: User '$username' does not exist${RESET}"
        return 1
    fi

    echo -e "${BLUE}Locking user account: $username${RESET}"

    if passwd -l "$username" &>/dev/null; then
        echo -e "${GREEN}✓ User account locked: $username${RESET}"
        log_success "User account locked: $username"
        return 0
    else
        echo -e "${RED}✗ Failed to lock user account: $username${RESET}"
        log_error "Failed to lock user account: $username"
        return 1
    fi
}

unlock_user() {
    local username="$1"

    check_root

    if ! user_exists "$username"; then
        echo -e "${RED}Error: User '$username' does not exist${RESET}"
        return 1
    fi

    echo -e "${BLUE}Unlocking user account: $username${RESET}"

    if passwd -u "$username" &>/dev/null; then
        echo -e "${GREEN}✓ User account unlocked: $username${RESET}"
        log_success "User account unlocked: $username"
        return 0
    else
        echo -e "${RED}✗ Failed to unlock user account: $username${RESET}"
        log_error "Failed to unlock user account: $username"
        return 1
    fi
}

set_user_password() {
    local username="$1"

    check_root

    if ! user_exists "$username"; then
        echo -e "${RED}Error: User '$username' does not exist${RESET}"
        return 1
    fi

    echo -e "${BLUE}Setting password for user: $username${RESET}"

    if passwd "$username"; then
        echo -e "${GREEN}✓ Password set for: $username${RESET}"
        log_success "Password changed for user: $username"
        return 0
    else
        echo -e "${RED}✗ Failed to set password: $username${RESET}"
        log_error "Failed to set password for user: $username"
        return 1
    fi
}

# ===========================
# Group Management
# ===========================

add_group() {
    local group="$1"
    local gid="${2:-}"

    check_root

    if group_exists "$group"; then
        echo -e "${RED}Error: Group '$group' already exists${RESET}"
        return 1
    fi

    echo -e "${BLUE}Creating group: $group${RESET}"

    local cmd="groupadd"
    if [[ -n "$gid" ]]; then
        cmd="$cmd -g $gid"
    fi
    cmd="$cmd $group"

    if $cmd; then
        echo -e "${GREEN}✓ Group created: $group${RESET}"
        log_success "Group created: $group"
        return 0
    else
        echo -e "${RED}✗ Failed to create group: $group${RESET}"
        log_error "Failed to create group: $group"
        return 1
    fi
}

delete_group() {
    local group="$1"

    check_root

    if ! group_exists "$group"; then
        echo -e "${RED}Error: Group '$group' does not exist${RESET}"
        return 1
    fi

    # Check if group has members
    local members=$(getent group "$group" | cut -d: -f4)
    if [[ -n "$members" ]]; then
        echo -e "${YELLOW}Warning: Group '$group' has members: $members${RESET}"

        if [[ "$FORCE" != true ]]; then
            echo "Use --force to delete anyway"
            return 1
        fi
    fi

    echo -e "${BLUE}Deleting group: $group${RESET}"

    if groupdel "$group"; then
        echo -e "${GREEN}✓ Group deleted: $group${RESET}"
        log_success "Group deleted: $group"
        return 0
    else
        echo -e "${RED}✗ Failed to delete group: $group${RESET}"
        log_error "Failed to delete group: $group"
        return 1
    fi
}

add_user_to_group() {
    local username="$1"
    local group="$2"

    check_root

    if ! user_exists "$username"; then
        echo -e "${RED}Error: User '$username' does not exist${RESET}"
        return 1
    fi

    if ! group_exists "$group"; then
        echo -e "${RED}Error: Group '$group' does not exist${RESET}"
        return 1
    fi

    # Check if user already in group
    if id -nG "$username" | grep -qw "$group"; then
        echo -e "${YELLOW}User '$username' is already in group '$group'${RESET}"
        return 0
    fi

    echo -e "${BLUE}Adding user '$username' to group '$group'${RESET}"

    if usermod -aG "$group" "$username"; then
        echo -e "${GREEN}✓ User added to group${RESET}"
        log_success "User '$username' added to group '$group'"
        return 0
    else
        echo -e "${RED}✗ Failed to add user to group${RESET}"
        log_error "Failed to add user '$username' to group '$group'"
        return 1
    fi
}

# ===========================
# SSH Key Management
# ===========================

manage_ssh_key() {
    local username="$1"

    check_root

    if ! user_exists "$username"; then
        echo -e "${RED}Error: User '$username' does not exist${RESET}"
        return 1
    fi

    local home_dir=$(eval echo ~"$username")
    local ssh_dir="$home_dir/.ssh"
    local authorized_keys="$ssh_dir/authorized_keys"

    echo -e "${BOLD}SSH Key Management for: $username${RESET}"
    echo ""

    # Ensure SSH directory exists
    mkdir -p "$ssh_dir"
    chmod 700 "$ssh_dir"
    chown "$username:$username" "$ssh_dir"

    # Check for existing keys
    if [[ -f "$authorized_keys" ]]; then
        echo -e "${GREEN}Authorized keys found:${RESET}"
        local key_count=$(grep -c "^ssh-" "$authorized_keys" 2>/dev/null || echo 0)
        echo "  Keys: $key_count"
        echo ""

        echo "Current keys:"
        cat "$authorized_keys" | grep "^ssh-" | while read -r key; do
            local key_type=$(echo "$key" | awk '{print $1}')
            local key_comment=$(echo "$key" | awk '{print $NF}')
            echo "  - $key_type ($key_comment)"
        done
        echo ""
    else
        echo "No authorized keys found"
        echo ""
    fi

    # Menu
    echo "Options:"
    echo "1. Add SSH key"
    echo "2. Remove SSH key"
    echo "3. View keys"
    echo "4. Generate key pair"
    echo "0. Exit"
    echo ""

    read -p "Select option: " option

    case "$option" in
        1)
            echo ""
            echo "Paste the SSH public key (one line):"
            read -r new_key

            if [[ "$new_key" =~ ^ssh- ]]; then
                echo "$new_key" >> "$authorized_keys"
                chmod 600 "$authorized_keys"
                chown "$username:$username" "$authorized_keys"
                echo -e "${GREEN}✓ SSH key added${RESET}"
                log_success "SSH key added for user: $username"
            else
                echo -e "${RED}Error: Invalid SSH key format${RESET}"
                return 1
            fi
            ;;
        2)
            if [[ ! -f "$authorized_keys" ]]; then
                echo "No keys to remove"
                return 0
            fi

            echo ""
            echo "Select key to remove:"
            cat "$authorized_keys" | grep "^ssh-" | nl
            echo ""

            read -p "Enter line number: " line_num

            if [[ "$line_num" =~ ^[0-9]+$ ]]; then
                sed -i "${line_num}d" "$authorized_keys"
                echo -e "${GREEN}✓ SSH key removed${RESET}"
                log_success "SSH key removed for user: $username"
            else
                echo -e "${RED}Error: Invalid line number${RESET}"
                return 1
            fi
            ;;
        3)
            if [[ -f "$authorized_keys" ]]; then
                cat "$authorized_keys"
            else
                echo "No authorized keys"
            fi
            ;;
        4)
            echo ""
            echo "Generating SSH key pair for $username..."
            local key_file="$ssh_dir/id_rsa"

            if [[ -f "$key_file" ]]; then
                echo -e "${YELLOW}Key already exists: $key_file${RESET}"
                read -p "Overwrite? (y/n): " confirm
                if [[ "$confirm" != "y" ]]; then
                    return 0
                fi
            fi

            sudo -u "$username" ssh-keygen -t rsa -b 4096 -f "$key_file" -N ""
            echo -e "${GREEN}✓ SSH key pair generated${RESET}"
            echo "  Private key: $key_file"
            echo "  Public key: ${key_file}.pub"
            log_success "SSH key pair generated for user: $username"
            ;;
        0)
            return 0
            ;;
        *)
            echo "Invalid option"
            return 1
            ;;
    esac
}

# ===========================
# User Information
# ===========================

display_user_info() {
    local username="$1"

    if ! user_exists "$username"; then
        echo -e "${RED}Error: User '$username' does not exist${RESET}"
        return 1
    fi

    local uid=$(id -u "$username")
    local gid=$(id -g "$username")
    local groups=$(id -Gn "$username" | tr ' ' ',')
    local home=$(eval echo ~"$username")
    local shell=$(getent passwd "$username" | cut -d: -f7)
    local locked=$(passwd -S "$username" 2>/dev/null | awk '{print $2}')

    echo -e "${BOLD}User Information: $username${RESET}"
    echo "  UID: $uid"
    echo "  GID: $gid"
    echo "  Groups: $groups"
    echo "  Home: $home"
    echo "  Shell: $shell"

    if [[ "$locked" == "L" ]]; then
        echo -e "  Status: ${RED}LOCKED${RESET}"
    else
        echo -e "  Status: ${GREEN}ACTIVE${RESET}"
    fi

    # Check for SSH keys
    local authorized_keys="$home/.ssh/authorized_keys"
    if [[ -f "$authorized_keys" ]]; then
        local key_count=$(grep -c "^ssh-" "$authorized_keys" 2>/dev/null || echo 0)
        echo "  SSH Keys: $key_count"
    fi

    # Last login
    local last_login=$(lastlog -u "$username" 2>/dev/null | tail -1 | awk '{$1=""; print $0}' | xargs)
    if [[ -n "$last_login" ]] && [[ "$last_login" != "Never logged in" ]]; then
        echo "  Last Login: $last_login"
    fi
}

list_all_users() {
    echo -e "${BOLD}${CYAN}System Users${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    printf "%-20s %-10s %-10s %-30s\n" "USERNAME" "UID" "GID" "HOME"
    echo "----------------------------------------"

    # List regular users (UID >= 1000)
    awk -F: -v min="$MIN_UID" -v max="$MAX_UID" \
        '$3 >= min && $3 <= max {printf "%-20s %-10s %-10s %-30s\n", $1, $3, $4, $6}' \
        /etc/passwd

    echo ""
}

list_all_groups() {
    echo -e "${BOLD}${CYAN}System Groups${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    printf "%-20s %-10s %-40s\n" "GROUP" "GID" "MEMBERS"
    echo "----------------------------------------"

    # List groups with GID >= 1000
    awk -F: -v min="$MIN_UID" \
        '$3 >= min {printf "%-20s %-10s %-40s\n", $1, $3, $4}' \
        /etc/group

    echo ""
}

# ===========================
# User Audit
# ===========================

generate_user_audit() {
    check_root

    mkdir -p "$AUDIT_DIR"
    local audit_file="$AUDIT_DIR/user-audit-$(date +%Y%m%d-%H%M%S).txt"

    echo -e "${BLUE}Generating user audit report...${RESET}"

    local report=$(cat <<EOF
========================================
User Audit Report
========================================
Generated: $(date '+%Y-%m-%d %H:%M:%S')
Hostname: $(hostname)

Total Users: $(awk -F: -v min="$MIN_UID" -v max="$MAX_UID" '$3 >= min && $3 <= max' /etc/passwd | wc -l)
Total Groups: $(awk -F: -v min="$MIN_UID" '$3 >= min' /etc/group | wc -l)

System Users:
-------------
$(awk -F: -v min="$MIN_UID" -v max="$MAX_UID" '$3 >= min && $3 <= max {printf "%-20s UID:%-6s GID:%-6s Home:%-30s Shell:%s\n", $1, $3, $4, $6, $7}' /etc/passwd)

Users with sudo access:
-----------------------
$(getent group sudo | cut -d: -f4 | tr ',' '\n' | sed 's/^/  /')

Users with docker access:
-------------------------
$(getent group docker 2>/dev/null | cut -d: -f4 | tr ',' '\n' | sed 's/^/  /' || echo "  Docker group not found")

Locked Accounts:
----------------
$(passwd -Sa 2>/dev/null | grep " L " | awk '{print "  " $1}' || echo "  None")

Accounts without password:
--------------------------
$(passwd -Sa 2>/dev/null | grep " NP " | awk '{print "  " $1}' || echo "  None")

Users logged in last 7 days:
----------------------------
$(lastlog -t 7 2>/dev/null | tail -n +2 | grep -v "Never logged in" | awk '{print "  " $1}' || echo "  None")

SSH Key Summary:
----------------
$(for user in $(awk -F: -v min="$MIN_UID" -v max="$MAX_UID" '$3 >= min && $3 <= max {print $1}' /etc/passwd); do
    home=$(eval echo ~"$user")
    if [[ -f "$home/.ssh/authorized_keys" ]]; then
        count=$(grep -c "^ssh-" "$home/.ssh/authorized_keys" 2>/dev/null || echo 0)
        if [[ $count -gt 0 ]]; then
            echo "  $user: $count key(s)"
        fi
    fi
done)

Security Concerns:
------------------
$(
# Check for users with UID 0
suspicious_uids=$(awk -F: '$3 == 0 && $1 != "root" {print "  WARNING: User " $1 " has UID 0"}' /etc/passwd)
if [[ -n "$suspicious_uids" ]]; then
    echo "$suspicious_uids"
fi

# Check for empty passwords
empty_passwords=$(passwd -Sa 2>/dev/null | grep " NP " | wc -l)
if [[ $empty_passwords -gt 0 ]]; then
    echo "  WARNING: $empty_passwords account(s) without password"
fi

# Check for users with no home directory
for user in $(awk -F: -v min="$MIN_UID" -v max="$MAX_UID" '$3 >= min && $3 <= max {print $1}' /etc/passwd); do
    home=$(eval echo ~"$user")
    if [[ ! -d "$home" ]]; then
        echo "  WARNING: User $user home directory does not exist: $home"
    fi
done

echo "  (No issues found)" 2>/dev/null
)

========================================
End of Audit Report
========================================
EOF
)

    echo "$report" | tee "$audit_file"

    echo ""
    echo -e "${GREEN}✓ Audit report saved: $audit_file${RESET}"
    log_success "User audit report generated: $audit_file"
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [COMMAND] [OPTIONS]

User and group management for ML infrastructure.

COMMANDS:
    add-user USER [OPTIONS]     Add a new user
    del-user USER [OPTIONS]     Delete a user
    mod-user USER [OPTIONS]     Modify user properties
    add-group GROUP [GID]       Add a new group
    del-group GROUP             Delete a group
    add-to-group USER GROUP     Add user to group
    list-users                  List all users
    list-groups                 List all groups
    audit                       Generate user audit report
    ssh-key USER                Manage SSH keys for user
    set-password USER           Set user password
    lock USER                   Lock user account
    unlock USER                 Unlock user account
    info USER                   Display user information

OPTIONS (for add-user, mod-user):
    --shell SHELL              Set user shell (default: $DEFAULT_SHELL)
    --home DIR                 Set home directory
    --groups GROUPS            Additional groups (comma-separated)
    --expire DATE              Account expiration date
    --uid UID                  Specific UID
    --gid GID                  Specific GID

GENERAL OPTIONS:
    -f, --force                Force operation
    -r, --remove-home          Remove home directory (with del-user)
    -v, --verbose              Verbose output
    -h, --help                 Display this help message

EXAMPLES:
    # Add ML engineer user
    $SCRIPT_NAME add-user mluser --groups docker,sudo

    # Add user with custom shell
    $SCRIPT_NAME add-user datauser --shell /bin/zsh

    # Delete user and remove home
    $SCRIPT_NAME del-user olduser --remove-home

    # Add user to docker group
    $SCRIPT_NAME add-to-group mluser docker

    # Manage SSH keys
    $SCRIPT_NAME ssh-key mluser

    # Lock account
    $SCRIPT_NAME lock mluser

    # Generate audit report
    $SCRIPT_NAME audit

    # List all users
    $SCRIPT_NAME list-users

COMMON ML INFRASTRUCTURE GROUPS:
    - docker: Access to Docker daemon
    - sudo: Administrative privileges
    - video: Access to GPUs

LOGS:
    Management log: $LOG_FILE
    Audit reports: $AUDIT_DIR

NOTE:
    Most operations require root privileges.
    Run with sudo for full functionality.

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

    local command="$1"
    shift

    case "$command" in
        add-user)
            if [[ $# -lt 1 ]]; then
                echo "Error: Username required"
                usage
                exit 1
            fi

            local username="$1"
            shift

            local shell="$DEFAULT_SHELL"
            local home=""
            local groups=""
            local uid=""

            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --shell)
                        shell="$2"
                        shift 2
                        ;;
                    --home)
                        home="$2"
                        shift 2
                        ;;
                    --groups)
                        groups="$2"
                        shift 2
                        ;;
                    --uid)
                        uid="$2"
                        shift 2
                        ;;
                    -v|--verbose)
                        VERBOSE=true
                        shift
                        ;;
                    *)
                        shift
                        ;;
                esac
            done

            add_user "$username" "$shell" "$home" "$groups" "$uid"
            ;;

        del-user)
            if [[ $# -lt 1 ]]; then
                echo "Error: Username required"
                usage
                exit 1
            fi

            local username="$1"
            shift

            local remove_home=false

            while [[ $# -gt 0 ]]; do
                case "$1" in
                    -r|--remove-home)
                        remove_home=true
                        shift
                        ;;
                    -f|--force)
                        FORCE=true
                        shift
                        ;;
                    -v|--verbose)
                        VERBOSE=true
                        shift
                        ;;
                    *)
                        shift
                        ;;
                esac
            done

            delete_user "$username" "$remove_home"
            ;;

        mod-user)
            if [[ $# -lt 1 ]]; then
                echo "Error: Username required"
                usage
                exit 1
            fi

            local username="$1"
            shift

            modify_user "$username" "$@"
            ;;

        add-group)
            if [[ $# -lt 1 ]]; then
                echo "Error: Group name required"
                usage
                exit 1
            fi

            add_group "$1" "${2:-}"
            ;;

        del-group)
            if [[ $# -lt 1 ]]; then
                echo "Error: Group name required"
                usage
                exit 1
            fi

            # Check for force flag
            if [[ "${2:-}" == "-f" ]] || [[ "${2:-}" == "--force" ]]; then
                FORCE=true
            fi

            delete_group "$1"
            ;;

        add-to-group)
            if [[ $# -lt 2 ]]; then
                echo "Error: Username and group name required"
                usage
                exit 1
            fi

            add_user_to_group "$1" "$2"
            ;;

        list-users)
            list_all_users
            ;;

        list-groups)
            list_all_groups
            ;;

        audit)
            generate_user_audit
            ;;

        ssh-key)
            if [[ $# -lt 1 ]]; then
                echo "Error: Username required"
                usage
                exit 1
            fi

            manage_ssh_key "$1"
            ;;

        set-password)
            if [[ $# -lt 1 ]]; then
                echo "Error: Username required"
                usage
                exit 1
            fi

            set_user_password "$1"
            ;;

        lock)
            if [[ $# -lt 1 ]]; then
                echo "Error: Username required"
                usage
                exit 1
            fi

            lock_user "$1"
            ;;

        unlock)
            if [[ $# -lt 1 ]]; then
                echo "Error: Username required"
                usage
                exit 1
            fi

            unlock_user "$1"
            ;;

        info)
            if [[ $# -lt 1 ]]; then
                echo "Error: Username required"
                usage
                exit 1
            fi

            display_user_info "$1"
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# ===========================
# Main Function
# ===========================

main() {
    # Ensure log file exists
    sudo mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    sudo touch "$LOG_FILE" 2>/dev/null || touch "$LOG_FILE" 2>/dev/null || true

    log_info "User management script started"

    parse_arguments "$@"
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
