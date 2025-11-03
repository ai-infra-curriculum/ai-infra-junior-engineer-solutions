#!/bin/bash
#
# manage_packages.sh - Unified Package Management Interface
#
# Usage: ./manage_packages.sh <command> [package_names...]
#

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' NC=''
fi

# Detect package manager
detect_package_manager() {
    if [ -f /etc/os-release ]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        case $ID in
            ubuntu|debian)
                echo "apt"
                ;;
            centos|rhel|fedora|rocky|alma)
                if command -v dnf &> /dev/null; then
                    echo "dnf"
                else
                    echo "yum"
                fi
                ;;
            arch|manjaro)
                echo "pacman"
                ;;
            *)
                echo "unknown"
                ;;
        esac
    else
        echo "unknown"
    fi
}

PKG_MANAGER=$(detect_package_manager)

# Usage
usage() {
    cat << EOF
Usage: $SCRIPT_NAME <command> [package_names...]

Unified interface for package management across different Linux distributions.

Commands:
  install [packages...]    Install packages
  remove [packages...]     Remove packages
  update                   Update package lists
  upgrade                  Upgrade installed packages
  search <keyword>         Search for packages
  info <package>           Show package information
  list                     List installed packages
  clean                    Clean package cache

Detected package manager: $PKG_MANAGER

Examples:
  $SCRIPT_NAME install python3-pip git
  $SCRIPT_NAME search tensorflow
  $SCRIPT_NAME update
  $SCRIPT_NAME upgrade
  $SCRIPT_NAME remove old-package

EOF
    exit 0
}

# Logging
log_info() { echo -e "${BLUE}ℹ${NC} $*"; }
log_success() { echo -e "${GREEN}✓${NC} $*"; }
log_error() { echo -e "${RED}✗${NC} $*" >&2; }

# Install packages
pkg_install() {
    local packages=("$@")

    if [ ${#packages[@]} -eq 0 ]; then
        log_error "No packages specified"
        exit 1
    fi

    log_info "Installing packages: ${packages[*]}"

    case $PKG_MANAGER in
        apt)
            sudo apt install -y "${packages[@]}"
            ;;
        yum)
            sudo yum install -y "${packages[@]}"
            ;;
        dnf)
            sudo dnf install -y "${packages[@]}"
            ;;
        pacman)
            sudo pacman -S --noconfirm "${packages[@]}"
            ;;
        *)
            log_error "Unknown package manager: $PKG_MANAGER"
            exit 1
            ;;
    esac

    log_success "Packages installed successfully"
}

# Remove packages
pkg_remove() {
    local packages=("$@")

    if [ ${#packages[@]} -eq 0 ]; then
        log_error "No packages specified"
        exit 1
    fi

    log_info "Removing packages: ${packages[*]}"

    case $PKG_MANAGER in
        apt)
            sudo apt remove -y "${packages[@]}"
            ;;
        yum)
            sudo yum remove -y "${packages[@]}"
            ;;
        dnf)
            sudo dnf remove -y "${packages[@]}"
            ;;
        pacman)
            sudo pacman -R --noconfirm "${packages[@]}"
            ;;
        *)
            log_error "Unknown package manager: $PKG_MANAGER"
            exit 1
            ;;
    esac

    log_success "Packages removed successfully"
}

# Update package lists
pkg_update() {
    log_info "Updating package lists..."

    case $PKG_MANAGER in
        apt)
            sudo apt update
            ;;
        yum)
            sudo yum check-update || true
            ;;
        dnf)
            sudo dnf check-update || true
            ;;
        pacman)
            sudo pacman -Sy
            ;;
        *)
            log_error "Unknown package manager: $PKG_MANAGER"
            exit 1
            ;;
    esac

    log_success "Package lists updated"
}

# Upgrade packages
pkg_upgrade() {
    log_info "Upgrading installed packages..."

    case $PKG_MANAGER in
        apt)
            sudo apt upgrade -y
            ;;
        yum)
            sudo yum update -y
            ;;
        dnf)
            sudo dnf upgrade -y
            ;;
        pacman)
            sudo pacman -Syu --noconfirm
            ;;
        *)
            log_error "Unknown package manager: $PKG_MANAGER"
            exit 1
            ;;
    esac

    log_success "Packages upgraded successfully"
}

# Search packages
pkg_search() {
    local keyword="$1"

    if [ -z "$keyword" ]; then
        log_error "No search keyword specified"
        exit 1
    fi

    log_info "Searching for: $keyword"

    case $PKG_MANAGER in
        apt)
            apt search "$keyword"
            ;;
        yum)
            yum search "$keyword"
            ;;
        dnf)
            dnf search "$keyword"
            ;;
        pacman)
            pacman -Ss "$keyword"
            ;;
        *)
            log_error "Unknown package manager: $PKG_MANAGER"
            exit 1
            ;;
    esac
}

# Show package info
pkg_info() {
    local package="$1"

    if [ -z "$package" ]; then
        log_error "No package specified"
        exit 1
    fi

    case $PKG_MANAGER in
        apt)
            apt show "$package"
            ;;
        yum)
            yum info "$package"
            ;;
        dnf)
            dnf info "$package"
            ;;
        pacman)
            pacman -Si "$package"
            ;;
        *)
            log_error "Unknown package manager: $PKG_MANAGER"
            exit 1
            ;;
    esac
}

# List installed packages
pkg_list() {
    log_info "Listing installed packages..."

    case $PKG_MANAGER in
        apt)
            apt list --installed
            ;;
        yum)
            yum list installed
            ;;
        dnf)
            dnf list --installed
            ;;
        pacman)
            pacman -Q
            ;;
        *)
            log_error "Unknown package manager: $PKG_MANAGER"
            exit 1
            ;;
    esac
}

# Clean package cache
pkg_clean() {
    log_info "Cleaning package cache..."

    case $PKG_MANAGER in
        apt)
            sudo apt clean
            sudo apt autoremove -y
            ;;
        yum)
            sudo yum clean all
            ;;
        dnf)
            sudo dnf clean all
            ;;
        pacman)
            sudo pacman -Sc --noconfirm
            ;;
        *)
            log_error "Unknown package manager: $PKG_MANAGER"
            exit 1
            ;;
    esac

    log_success "Package cache cleaned"
}

# Main function
main() {
    if [ $# -eq 0 ]; then
        usage
    fi

    local command="$1"
    shift

    case $command in
        install)
            pkg_install "$@"
            ;;
        remove)
            pkg_remove "$@"
            ;;
        update)
            pkg_update
            ;;
        upgrade)
            pkg_upgrade
            ;;
        search)
            pkg_search "$@"
            ;;
        info)
            pkg_info "$@"
            ;;
        list)
            pkg_list
            ;;
        clean)
            pkg_clean
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            ;;
    esac
}

main "$@"
