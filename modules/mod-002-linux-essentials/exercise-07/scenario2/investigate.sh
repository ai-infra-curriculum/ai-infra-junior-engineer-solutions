#!/bin/bash
###############################################################################
# Scenario 2: Permission Denied - Investigation Script
###############################################################################
#
# Problem: PermissionError: [Errno 13] Permission denied: '/data/models/checkpoint.pth'
# Location: Trying to load model from /data/models/
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
echo -e "${BOLD}${RED}║  Scenario 2: Permission Denied Investigation              ║${NC}"
echo -e "${BOLD}${RED}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Problem:${NC} Cannot access model checkpoint file"
echo -e "${YELLOW}Location:${NC} /data/models/checkpoint.pth"
echo ""

# Target file/directory to investigate
TARGET="${1:-/data/models/checkpoint.pth}"

section "Step 1: Current User Information"
echo "Command: whoami, id, groups"
echo ""
log_info "Current user: $(whoami)"
log_info "User ID: $(id -u)"
log_info "Group ID: $(id -g)"
log_info "Groups: $(groups)"
echo ""
log_info "Full ID info:"
id

section "Step 2: Check if File/Directory Exists"
echo "Command: ls -la \"$TARGET\""
echo ""
if [ -e "$TARGET" ]; then
    log_success "File/directory exists"
    ls -la "$TARGET"
else
    log_error "File/directory does not exist: $TARGET"
    echo ""
    subsection "Checking parent directory:"
    parent_dir=$(dirname "$TARGET")
    if [ -d "$parent_dir" ]; then
        log_info "Parent directory exists: $parent_dir"
        ls -la "$parent_dir"
    else
        log_error "Parent directory does not exist: $parent_dir"
    fi
fi

section "Step 3: Check File Permissions"
if [ -e "$TARGET" ]; then
    echo "Detailed permission breakdown:"
    echo ""

    # Get file info
    perms=$(stat -c "%a" "$TARGET" 2>/dev/null || stat -f "%Lp" "$TARGET" 2>/dev/null)
    owner=$(stat -c "%U" "$TARGET" 2>/dev/null || stat -f "%Su" "$TARGET" 2>/dev/null)
    group=$(stat -c "%G" "$TARGET" 2>/dev/null || stat -f "%Sg" "$TARGET" 2>/dev/null)

    log_info "Numeric permissions: $perms"
    log_info "Owner: $owner"
    log_info "Group: $group"
    echo ""

    log_info "Full stat output:"
    stat "$TARGET" 2>/dev/null || ls -la "$TARGET"
    echo ""

    # Check if we're the owner
    current_user=$(whoami)
    if [ "$owner" = "$current_user" ]; then
        log_success "You are the owner"
    else
        log_warning "You are NOT the owner (owner is: $owner)"
    fi

    # Check if we're in the group
    if groups | grep -q "\b$group\b"; then
        log_success "You are in the file's group ($group)"
    else
        log_warning "You are NOT in the file's group (group is: $group)"
    fi
fi

section "Step 4: Check Parent Directory Permissions"
echo "Path traversal permission check:"
echo ""

if [ -e "$TARGET" ]; then
    path="$TARGET"
    # If it's a file, start from parent directory
    if [ -f "$TARGET" ]; then
        path=$(dirname "$TARGET")
    fi

    # Check each directory in the path
    current=""
    echo "$path" | tr '/' '\n' | while read dir; do
        if [ -z "$dir" ]; then
            current="/"
        else
            current="$current/$dir"
        fi

        if [ -d "$current" ]; then
            perms=$(stat -c "%a" "$current" 2>/dev/null || stat -f "%Lp" "$current" 2>/dev/null)
            owner=$(stat -c "%U" "$current" 2>/dev/null || stat -f "%Su" "$current" 2>/dev/null)

            # Check if we have execute permission
            if [ -x "$current" ]; then
                echo -e "  ${GREEN}✓${NC} $current ($perms, owner: $owner)"
            else
                echo -e "  ${RED}✗${NC} $current ($perms, owner: $owner) - No execute permission!"
            fi
        fi
    done

    echo ""
    log_info "Note: Execute (x) permission on directories is required to traverse them"
fi

section "Step 5: Test Read/Write Access"
if [ -e "$TARGET" ]; then
    echo "Testing actual access capabilities:"
    echo ""

    if [ -r "$TARGET" ]; then
        log_success "Read permission: YES"
    else
        log_error "Read permission: NO"
    fi

    if [ -w "$TARGET" ]; then
        log_success "Write permission: YES"
    else
        log_warning "Write permission: NO"
    fi

    if [ -x "$TARGET" ]; then
        log_success "Execute permission: YES"
    else
        log_info "Execute permission: NO (not needed for data files)"
    fi
fi

section "Step 6: Check ACLs (Access Control Lists)"
echo "Extended ACLs (if any):"
echo ""

if command -v getfacl &>/dev/null; then
    if [ -e "$TARGET" ]; then
        getfacl "$TARGET" 2>/dev/null || log_info "No ACLs set or getfacl failed"
    fi
else
    log_warning "getfacl command not found (install acl package)"
fi

section "Step 7: Check SELinux Context (if enabled)"
if command -v getenforce &>/dev/null && [ "$(getenforce 2>/dev/null)" != "Disabled" ]; then
    echo "SELinux is enabled"
    echo ""

    if [ -e "$TARGET" ]; then
        log_info "SELinux context:"
        ls -Z "$TARGET" 2>/dev/null || log_warning "Cannot get SELinux context"

        echo ""
        log_info "Current process context:"
        ps -eZ | grep $$ | head -1
    fi
else
    log_info "SELinux is not enabled or not installed"
fi

section "Step 8: Check for Immutable Attributes"
echo "File attributes (immutable, append-only, etc.):"
echo ""

if command -v lsattr &>/dev/null; then
    if [ -e "$TARGET" ]; then
        lsattr "$TARGET" 2>/dev/null || log_warning "Cannot check attributes"
        echo ""
        log_info "Key attributes:"
        log_info "  i = immutable (cannot be modified)"
        log_info "  a = append-only"
    fi
else
    log_warning "lsattr command not found"
fi

section "Step 9: Check Process Trying to Access File"
echo "If you know the process having issues, check its user:"
echo ""
log_info "Example: ps aux | grep python"
log_info "Example: ps -p <PID> -o user,group,cmd"
echo ""

# Look for Python processes that might be ML training
python_procs=$(ps aux | grep -E "python|train" | grep -v grep | wc -l)
if [ "$python_procs" -gt 0 ]; then
    log_info "Found $python_procs Python/training processes:"
    ps aux | grep -E "python|train" | grep -v grep | head -5
fi

section "Step 10: Check umask Setting"
echo "Current umask (affects new file creation):"
echo ""
log_info "umask: $(umask)"
log_info "Default file permissions will be: $(printf "%04o" $((0666 & ~$(umask))))"
log_info "Default directory permissions will be: $(printf "%04o" $((0777 & ~$(umask))))"

section "Analysis Summary"
echo -e "${BOLD}Diagnosis:${NC}"
echo ""

if [ ! -e "$TARGET" ]; then
    log_error "File does not exist"
    echo ""
    echo "Next steps:"
    echo "  1. Verify the correct path"
    echo "  2. Check if file was moved or deleted"
    echo "  3. Create the file/directory if needed"
else
    current_user=$(whoami)
    owner=$(stat -c "%U" "$TARGET" 2>/dev/null || stat -f "%Su" "$TARGET" 2>/dev/null)
    group=$(stat -c "%G" "$TARGET" 2>/dev/null || stat -f "%Sg" "$TARGET" 2>/dev/null)
    perms=$(stat -c "%a" "$TARGET" 2>/dev/null || stat -f "%Lp" "$TARGET" 2>/dev/null)

    if [ "$owner" != "$current_user" ] && ! groups | grep -q "\b$group\b"; then
        log_error "You are neither the owner nor in the group"
        echo ""
        echo "Possible solutions:"
        echo "  1. Change ownership: sudo chown $current_user:$current_user $TARGET"
        echo "  2. Add yourself to group: sudo usermod -aG $group $current_user"
        echo "  3. Adjust permissions: sudo chmod g+rw $TARGET"
    elif [ -r "$TARGET" ]; then
        log_success "You should have read access"
        log_info "If still getting errors, check parent directory permissions"
    else
        log_warning "Permission bits suggest you should not have access"
        echo ""
        echo "Current permissions: $perms"
        echo "Owner: $owner, Group: $group"
    fi
fi

echo ""
echo -e "${BOLD}${BLUE}Next Steps:${NC}"
echo "  1. Review the permission analysis above"
echo "  2. Identify the specific permission issue"
echo "  3. Run the fix script: ./fix.sh $TARGET"
echo "  4. Verify access after fixing"
echo ""
