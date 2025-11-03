#!/bin/bash
#
# script_template.sh - Production-Ready Bash Script Template
#
# Usage: ./script_template.sh [options] <arguments>
#
# Description: This template includes best practices for production scripts
#

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Script metadata
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_VERSION="1.0.0"
readonly LOG_DIR="${SCRIPT_DIR}/../logs"
readonly LOG_FILE="${LOG_DIR}/${SCRIPT_NAME%.sh}.log"

# Default configuration
VERBOSE=false
DRY_RUN=false
DEBUG=false

# Colors for terminal output
if [[ -t 1 ]]; then
    readonly GREEN='\033[0;32m'
    readonly RED='\033[0;31m'
    readonly YELLOW='\033[1;33m'
    readonly BLUE='\033[0;34m'
    readonly NC='\033[0m'
else
    readonly GREEN=''
    readonly RED=''
    readonly YELLOW=''
    readonly BLUE=''
    readonly NC=''
fi

#
# Logging Functions
#

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "$@"
}

log_success() {
    log "SUCCESS" "$@"
    [ "$VERBOSE" = true ] && echo -e "${GREEN}✓${NC} $*"
}

log_warning() {
    log "WARNING" "$@"
    echo -e "${YELLOW}⚠${NC} $*" >&2
}

log_error() {
    log "ERROR" "$@"
    echo -e "${RED}✗${NC} $*" >&2
}

log_debug() {
    [ "$DEBUG" = true ] && log "DEBUG" "$@"
}

#
# Error Handling
#

error_exit() {
    log_error "$1"
    cleanup
    exit "${2:-1}"
}

error_handler() {
    local line_number="$1"
    log_error "Script failed at line $line_number"
    cleanup
    exit 1
}

# Trap errors
trap 'error_handler ${LINENO}' ERR

#
# Cleanup Functions
#

cleanup() {
    log_debug "Running cleanup"
    # Add cleanup tasks here (temp files, connections, etc.)
}

# Trap exit to ensure cleanup runs
trap cleanup EXIT

#
# Utility Functions
#

check_dependencies() {
    local missing_deps=()

    # List required commands
    local required_commands=(
        # "python3"
        # "git"
        # "docker"
    )

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [ ${#missing_deps[@]} -gt 0 ]; then
        error_exit "Missing dependencies: ${missing_deps[*]}"
    fi

    log_debug "All dependencies satisfied"
}

validate_input() {
    # Add input validation logic here
    log_debug "Validating input"
    return 0
}

#
# Main Business Logic
#

process_data() {
    local input="$1"

    log_info "Processing: $input"

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would process $input"
        return 0
    fi

    # Add your processing logic here

    log_success "Processed: $input"
}

#
# Usage and Help
#

usage() {
    cat << EOF
Usage: $SCRIPT_NAME [options] <arguments>

Description:
  Production-ready bash script template with best practices.

Options:
  -h, --help        Show this help message
  -v, --verbose     Enable verbose output
  -d, --debug       Enable debug output
  -n, --dry-run     Perform a dry run without making changes
  --version         Show script version

Arguments:
  <input>           Input data to process

Examples:
  $SCRIPT_NAME data.txt
  $SCRIPT_NAME --verbose data.txt
  $SCRIPT_NAME --dry-run --debug data.txt

Version: $SCRIPT_VERSION

EOF
    exit 0
}

#
# Argument Parsing
#

parse_arguments() {
    # Check if no arguments provided
    if [ $# -eq 0 ]; then
        usage
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                ;;
            -v|--verbose)
                VERBOSE=true
                log_debug "Verbose mode enabled"
                shift
                ;;
            -d|--debug)
                DEBUG=true
                log_debug "Debug mode enabled"
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                log_info "Dry run mode enabled"
                shift
                ;;
            --version)
                echo "$SCRIPT_NAME version $SCRIPT_VERSION"
                exit 0
                ;;
            -*)
                error_exit "Unknown option: $1"
                ;;
            *)
                # Positional arguments
                POSITIONAL_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

#
# Setup Functions
#

setup() {
    # Create required directories
    mkdir -p "$LOG_DIR"

    log_debug "Setup complete"
}

#
# Main Function
#

main() {
    # Store positional arguments
    local -a POSITIONAL_ARGS=()

    # Parse command line arguments
    parse_arguments "$@"

    # Setup environment
    setup

    # Check dependencies
    check_dependencies

    # Validate input
    validate_input

    log_info "=== Starting $SCRIPT_NAME ==="

    # Main logic
    if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
        for arg in "${POSITIONAL_ARGS[@]}"; do
            process_data "$arg"
        done
    else
        log_warning "No input provided"
    fi

    log_success "=== $SCRIPT_NAME Complete ==="
}

# Execute main function
main "$@"
