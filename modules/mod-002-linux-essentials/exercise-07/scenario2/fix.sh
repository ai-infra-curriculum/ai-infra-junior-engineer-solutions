#!/bin/bash
###############################################################################
# Scenario 2: Permission Denied - Fix Script
###############################################################################
#
# Usage: ./fix.sh [TARGET] [--method METHOD] [--user USER] [--group GROUP]
#
# Methods:
#   owner    - Change ownership to current user
#   group    - Add current user to file's group
#   chmod    - Adjust file permissions
#   parent   - Fix parent directory permissions
#   auto     - Automatically detect and fix (default)
#

set -u

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ${NC} $*"; }
log_success() { echo -e "${GREEN}✓${NC} $*"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $*"; }
log_error() { echo -e "${RED}✗${NC} $*"; }

# Default values
TARGET=""
METHOD="auto"
TARGET_USER=""
TARGET_GROUP=""
DRY_RUN=false
RECURSIVE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --user)
            TARGET_USER="$2"
            shift 2
            ;;
        --group)
            TARGET_GROUP="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --recursive|-R)
            RECURSIVE=true
            shift
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [TARGET] [OPTIONS]

Fix permission issues for files and directories.

Arguments:
  TARGET              File or directory to fix (default: /data/models/checkpoint.pth)

Options:
  --method METHOD     Fix method: owner, group, chmod, parent, auto (default: auto)
  --user USER        Target user for ownership (default: current user)
  --group GROUP      Target group for ownership (default: current user's group)
  --recursive, -R    Apply changes recursively
  --dry-run          Show what would be done without doing it
  -h, --help         Show this help message

Methods:
  owner    Change file ownership to specified user
  group    Add user to file's group (requires logout)
  chmod    Adjust file permissions (add read/write for user)
  parent   Fix parent directory execute permissions
  auto     Automatically detect and apply best fix

Examples:
  $0 /data/models/checkpoint.pth
  $0 /data/models --recursive --method owner
  $0 /data/models/checkpoint.pth --method chmod --dry-run
  $0 /data/models --user mluser --group mlteam

EOF
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

# Set defaults
if [ -z "$TARGET" ]; then
    TARGET="/data/models/checkpoint.pth"
fi

if [ -z "$TARGET_USER" ]; then
    TARGET_USER=$(whoami)
fi

if [ -z "$TARGET_GROUP" ]; then
    TARGET_GROUP=$(id -gn)
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Permission Fix Utility                                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No changes will be made"
    echo ""
fi

log_info "Target: $TARGET"
log_info "Method: $METHOD"
log_info "User: $TARGET_USER"
log_info "Group: $TARGET_GROUP"
echo ""

# Check if target exists
if [ ! -e "$TARGET" ]; then
    log_error "Target does not exist: $TARGET"
    echo ""
    parent_dir=$(dirname "$TARGET")

    if [ ! -d "$parent_dir" ]; then
        log_info "Parent directory does not exist: $parent_dir"
        log_info "Creating parent directory..."

        if [ "$DRY_RUN" = false ]; then
            sudo mkdir -p "$parent_dir"
            sudo chown "$TARGET_USER:$TARGET_GROUP" "$parent_dir"
            sudo chmod 755 "$parent_dir"
            log_success "Parent directory created and configured"
        else
            log_info "Would run: sudo mkdir -p $parent_dir"
            log_info "Would run: sudo chown $TARGET_USER:$TARGET_GROUP $parent_dir"
            log_info "Would run: sudo chmod 755 $parent_dir"
        fi
    fi

    echo ""
    log_info "The target file will need to be created by your application"
    exit 0
fi

# Get current state
current_owner=$(stat -c "%U" "$TARGET" 2>/dev/null || stat -f "%Su" "$TARGET" 2>/dev/null)
current_group=$(stat -c "%G" "$TARGET" 2>/dev/null || stat -f "%Sg" "$TARGET" 2>/dev/null)
current_perms=$(stat -c "%a" "$TARGET" 2>/dev/null || stat -f "%Lp" "$TARGET" 2>/dev/null)

log_info "Current state:"
echo "  Owner: $current_owner"
echo "  Group: $current_group"
echo "  Permissions: $current_perms"
echo ""

# Determine fix method
if [ "$METHOD" = "auto" ]; then
    log_info "Auto-detecting best fix method..."
    echo ""

    # Check if we're trying to access as ourselves
    current_user=$(whoami)

    if [ "$current_owner" != "$current_user" ] && ! groups | grep -q "\b$current_group\b"; then
        # Not owner and not in group - change ownership is best
        METHOD="owner"
        log_info "Selected method: owner (you are not owner or in group)"
    elif [ ! -r "$TARGET" ]; then
        # Can't read - fix permissions
        METHOD="chmod"
        log_info "Selected method: chmod (insufficient read permissions)"
    else
        log_success "You already have read access!"
        if [ ! -w "$TARGET" ]; then
            log_info "But you don't have write access"
            METHOD="chmod"
        else
            log_success "No fixes needed - you have full access"
            exit 0
        fi
    fi
    echo ""
fi

# Apply fix based on method
case $METHOD in
    owner)
        log_info "Changing ownership to $TARGET_USER:$TARGET_GROUP..."

        if [ "$RECURSIVE" = true ] && [ -d "$TARGET" ]; then
            if [ "$DRY_RUN" = false ]; then
                sudo chown -R "$TARGET_USER:$TARGET_GROUP" "$TARGET"
                log_success "Ownership changed recursively"
            else
                log_info "Would run: sudo chown -R $TARGET_USER:$TARGET_GROUP $TARGET"
            fi
        else
            if [ "$DRY_RUN" = false ]; then
                sudo chown "$TARGET_USER:$TARGET_GROUP" "$TARGET"
                log_success "Ownership changed"
            else
                log_info "Would run: sudo chown $TARGET_USER:$TARGET_GROUP $TARGET"
            fi
        fi

        # Also set reasonable permissions
        if [ -d "$TARGET" ]; then
            new_perms="755"
        else
            new_perms="644"
        fi

        if [ "$DRY_RUN" = false ]; then
            if [ "$RECURSIVE" = true ] && [ -d "$TARGET" ]; then
                sudo chmod -R "$new_perms" "$TARGET"
            else
                sudo chmod "$new_perms" "$TARGET"
            fi
            log_success "Permissions set to $new_perms"
        else
            log_info "Would run: sudo chmod $new_perms $TARGET"
        fi
        ;;

    group)
        log_info "Adding user $TARGET_USER to group $current_group..."

        if [ "$DRY_RUN" = false ]; then
            sudo usermod -aG "$current_group" "$TARGET_USER"
            log_success "User added to group"
            log_warning "You must log out and back in for group changes to take effect!"
            echo ""
            log_info "Or run: newgrp $current_group"
        else
            log_info "Would run: sudo usermod -aG $current_group $TARGET_USER"
        fi

        # Also ensure group has read access
        if [ "$DRY_RUN" = false ]; then
            sudo chmod g+r "$TARGET"
            if [ -d "$TARGET" ]; then
                sudo chmod g+x "$TARGET"
            fi
            log_success "Group permissions updated"
        else
            log_info "Would run: sudo chmod g+r $TARGET"
        fi
        ;;

    chmod)
        log_info "Adjusting file permissions..."

        if [ "$current_owner" = "$TARGET_USER" ]; then
            # We're the owner - give ourselves read/write
            if [ -d "$TARGET" ]; then
                new_perms="u+rwx"
                log_info "Adding owner read/write/execute permissions"
            else
                new_perms="u+rw"
                log_info "Adding owner read/write permissions"
            fi
        elif groups | grep -q "\b$current_group\b"; then
            # We're in the group - give group read/write
            if [ -d "$TARGET" ]; then
                new_perms="g+rwx"
                log_info "Adding group read/write/execute permissions"
            else
                new_perms="g+rw"
                log_info "Adding group read/write permissions"
            fi
        else
            # Neither owner nor group - give others read
            log_warning "You are neither owner nor in group - giving 'others' read access"
            new_perms="o+r"
            if [ -d "$TARGET" ]; then
                new_perms="o+rx"
            fi
        fi

        if [ "$RECURSIVE" = true ] && [ -d "$TARGET" ]; then
            if [ "$DRY_RUN" = false ]; then
                sudo chmod -R "$new_perms" "$TARGET"
                log_success "Permissions changed recursively"
            else
                log_info "Would run: sudo chmod -R $new_perms $TARGET"
            fi
        else
            if [ "$DRY_RUN" = false ]; then
                sudo chmod "$new_perms" "$TARGET"
                log_success "Permissions changed"
            else
                log_info "Would run: sudo chmod $new_perms $TARGET"
            fi
        fi
        ;;

    parent)
        log_info "Fixing parent directory permissions..."

        # Get parent directory
        if [ -f "$TARGET" ]; then
            parent=$(dirname "$TARGET")
        else
            parent="$TARGET"
        fi

        # Check and fix each directory in path
        path_to_fix=""
        current=""
        echo "$parent" | tr '/' '\n' | while read dir; do
            if [ -z "$dir" ]; then
                current="/"
            else
                current="$current/$dir"
            fi

            if [ -d "$current" ] && [ ! -x "$current" ]; then
                log_warning "Directory $current lacks execute permission"

                if [ "$DRY_RUN" = false ]; then
                    sudo chmod +x "$current"
                    log_success "Added execute permission to $current"
                else
                    log_info "Would run: sudo chmod +x $current"
                fi
            fi
        done

        log_success "Parent directory permissions fixed"
        ;;

    *)
        log_error "Unknown method: $METHOD"
        echo "Valid methods: owner, group, chmod, parent, auto"
        exit 1
        ;;
esac

echo ""

# Verify fix
if [ "$DRY_RUN" = false ]; then
    log_info "Verifying fix..."
    echo ""

    new_owner=$(stat -c "%U" "$TARGET" 2>/dev/null || stat -f "%Su" "$TARGET" 2>/dev/null)
    new_group=$(stat -c "%G" "$TARGET" 2>/dev/null || stat -f "%Sg" "$TARGET" 2>/dev/null)
    new_perms=$(stat -c "%a" "$TARGET" 2>/dev/null || stat -f "%Lp" "$TARGET" 2>/dev/null)

    log_info "New state:"
    echo "  Owner: $new_owner"
    echo "  Group: $new_group"
    echo "  Permissions: $new_perms"
    echo ""

    if [ -r "$TARGET" ]; then
        log_success "Read access: YES"
    else
        log_error "Read access: NO - May need additional fixes"
    fi

    if [ -w "$TARGET" ]; then
        log_success "Write access: YES"
    else
        log_warning "Write access: NO (may be intentional)"
    fi

    echo ""
    log_success "Fix complete!"
else
    echo ""
    log_info "Dry run complete. Run without --dry-run to apply changes."
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Best Practices                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Use dedicated groups for team access:"
echo "   sudo groupadd mlteam"
echo "   sudo usermod -aG mlteam <username>"
echo ""
echo "2. Set appropriate umask in ~/.bashrc:"
echo "   umask 002  # Files created as 664, dirs as 775"
echo ""
echo "3. Use ACLs for fine-grained control:"
echo "   setfacl -m u:username:rw /data/models/"
echo ""
echo "4. For shared directories, use setgid bit:"
echo "   sudo chmod g+s /data/models/"
echo "   (New files inherit directory's group)"
echo ""
echo "5. Regular permission audits:"
echo "   find /data -type f ! -perm 644"
echo "   find /data -type d ! -perm 755"
echo ""
