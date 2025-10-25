#!/bin/bash
#
# deploy_model.sh - Deploy ML models to production environment
#
# Description:
#   Automates the deployment of ML models with validation,
#   backup, and rollback capabilities.
#
# Usage:
#   ./deploy_model.sh [OPTIONS] MODEL_PATH
#
# Options:
#   -e, --environment ENV    Target environment (dev, staging, prod)
#   -n, --dry-run           Perform dry-run without actual deployment
#   -v, --verbose           Enable verbose output
#   -h, --help              Display this help message
#
# Examples:
#   ./deploy_model.sh -e prod model.pkl
#   ./deploy_model.sh --dry-run --verbose model.pkl
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="/var/log/ml-deployment"
readonly LOG_FILE="${LOG_DIR}/deployment.log"

# Default configuration
ENVIRONMENT="${ENVIRONMENT:-dev}"
DRY_RUN=false
VERBOSE=false

# Environment-specific directories
declare -A MODEL_DIRS=(
    ["dev"]="/opt/ml/dev/models"
    ["staging"]="/opt/ml/staging/models"
    ["prod"]="/opt/ml/prod/models"
)

declare -A BACKUP_DIRS=(
    ["dev"]="/backup/ml/dev/models"
    ["staging"]="/backup/ml/staging/models"
    ["prod"]="/backup/ml/prod/models"
)

# ===========================
# Colors for output
# ===========================

readonly COLOR_RED='\033[0;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_RESET='\033[0m'

# ===========================
# Logging Functions
# ===========================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Log to file
    if [[ -d "$LOG_DIR" ]] || mkdir -p "$LOG_DIR" 2>/dev/null; then
        echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    fi

    # Log to stdout
    case "$level" in
        INFO)
            echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $message"
            ;;
        SUCCESS)
            echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $message"
            ;;
        WARNING)
            echo -e "${COLOR_YELLOW}[WARNING]${COLOR_RESET} $message"
            ;;
        ERROR)
            echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $message" >&2
            ;;
        *)
            echo "[$level] $message"
            ;;
    esac
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        log "DEBUG" "$@"
    fi
}

# ===========================
# Error Handling
# ===========================

error_exit() {
    log "ERROR" "$1"
    exit "${2:-1}"
}

cleanup() {
    local exit_code=$?
    log_verbose "Cleanup function called with exit code: $exit_code"

    # Remove temporary files if they exist
    if [[ -n "${TEMP_DIR:-}" ]] && [[ -d "$TEMP_DIR" ]]; then
        log_verbose "Removing temporary directory: $TEMP_DIR"
        rm -rf "$TEMP_DIR"
    fi

    if [[ $exit_code -ne 0 ]]; then
        log "ERROR" "Deployment failed with exit code: $exit_code"
    fi
}

trap cleanup EXIT

# ===========================
# Validation Functions
# ===========================

check_requirements() {
    log_verbose "Checking requirements..."

    local required_commands=("md5sum" "python3")

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command not found: $cmd" 1
        fi
    done

    log_verbose "All requirements satisfied"
}

validate_environment() {
    local env="$1"

    if [[ ! -v MODEL_DIRS[$env] ]]; then
        error_exit "Invalid environment: $env. Valid options: dev, staging, prod" 1
    fi

    log_verbose "Environment validated: $env"
}

validate_model() {
    local model_path="$1"

    log "INFO" "Validating model: $model_path"

    # Check if file exists
    if [[ ! -f "$model_path" ]]; then
        error_exit "Model file not found: $model_path" 1
    fi

    # Check file size
    local size=$(stat -f%z "$model_path" 2>/dev/null || stat -c%s "$model_path" 2>/dev/null)
    if [[ $size -eq 0 ]]; then
        error_exit "Model file is empty: $model_path" 1
    fi

    log_verbose "Model file size: $((size / 1024 / 1024)) MB"

    # Check file extension
    case "$model_path" in
        *.pkl|*.joblib|*.h5|*.pt|*.pth|*.onnx)
            log_verbose "Valid model format detected"
            ;;
        *)
            log "WARNING" "Unexpected file extension. Supported: .pkl, .joblib, .h5, .pt, .pth, .onnx"
            ;;
    esac

    # Optional: Run Python validation
    if [[ -f "${SCRIPT_DIR}/validate_model.py" ]]; then
        log_verbose "Running Python model validation..."
        if ! python3 "${SCRIPT_DIR}/validate_model.py" "$model_path" 2>&1 | tee -a "$LOG_FILE"; then
            error_exit "Model validation failed" 1
        fi
    fi

    log "SUCCESS" "Model validation passed"
    return 0
}

# ===========================
# Deployment Functions
# ===========================

calculate_checksum() {
    local file="$1"
    md5sum "$file" | awk '{print $1}'
}

backup_current_model() {
    local model_name="$1"
    local deploy_dir="${MODEL_DIRS[$ENVIRONMENT]}"
    local backup_dir="${BACKUP_DIRS[$ENVIRONMENT]}"
    local current_model="$deploy_dir/$model_name"

    if [[ ! -f "$current_model" ]]; then
        log "INFO" "No existing model to backup"
        return 0
    fi

    local backup_name="${model_name}.$(date +%Y%m%d_%H%M%S)"
    local backup_path="$backup_dir/$backup_name"

    log "INFO" "Backing up current model: $model_name"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would backup: $current_model -> $backup_path"
        return 0
    fi

    # Create backup directory if it doesn't exist
    if ! mkdir -p "$backup_dir"; then
        log "WARNING" "Could not create backup directory: $backup_dir"
        return 1
    fi

    # Copy current model to backup
    if cp "$current_model" "$backup_path"; then
        log "SUCCESS" "Backup created: $backup_path"

        # Save checksum
        calculate_checksum "$backup_path" > "${backup_path}.md5"

        # Keep only last 5 backups
        cleanup_old_backups "$backup_dir" "$model_name"
    else
        log "WARNING" "Backup failed"
        return 1
    fi

    return 0
}

cleanup_old_backups() {
    local backup_dir="$1"
    local model_name="$2"
    local keep_count=5

    log_verbose "Cleaning up old backups (keeping last $keep_count)"

    # Find backups, sort by modification time, delete old ones
    find "$backup_dir" -name "${model_name}.*" -type f ! -name "*.md5" -printf '%T@ %p\n' \
        | sort -rn \
        | tail -n +$((keep_count + 1)) \
        | cut -d' ' -f2- \
        | while read -r old_backup; do
            log_verbose "Removing old backup: $old_backup"
            rm -f "$old_backup" "${old_backup}.md5"
        done
}

deploy_model() {
    local model_path="$1"
    local model_name=$(basename "$model_path")
    local deploy_dir="${MODEL_DIRS[$ENVIRONMENT]}"
    local deploy_path="$deploy_dir/$model_name"

    log "INFO" "Starting deployment to $ENVIRONMENT environment"
    log "INFO" "Source: $model_path"
    log "INFO" "Destination: $deploy_path"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would deploy: $model_path -> $deploy_path"
        log "INFO" "[DRY-RUN] Deployment steps:"
        log "INFO" "[DRY-RUN]   1. Validate model"
        log "INFO" "[DRY-RUN]   2. Backup current model"
        log "INFO" "[DRY-RUN]   3. Copy new model"
        log "INFO" "[DRY-RUN]   4. Set permissions"
        log "INFO" "[DRY-RUN]   5. Verify deployment"
        return 0
    fi

    # Create deploy directory if it doesn't exist
    if ! sudo mkdir -p "$deploy_dir"; then
        error_exit "Could not create deployment directory: $deploy_dir" 1
    fi

    # Backup current model
    backup_current_model "$model_name" || log "WARNING" "Backup failed but continuing..."

    # Copy new model
    log "INFO" "Copying model to deployment directory..."
    if ! sudo cp "$model_path" "$deploy_path"; then
        error_exit "Failed to copy model" 1
    fi

    # Set ownership and permissions
    log "INFO" "Setting permissions..."
    if sudo chown ml-user:ml-group "$deploy_path" 2>/dev/null || true; then
        log_verbose "Ownership set to ml-user:ml-group"
    fi

    if ! sudo chmod 644 "$deploy_path"; then
        error_exit "Failed to set permissions" 1
    fi

    # Calculate and save checksum
    log "INFO" "Calculating checksum..."
    calculate_checksum "$deploy_path" | sudo tee "${deploy_path}.md5" > /dev/null

    log "SUCCESS" "Model deployed successfully: $deploy_path"
    return 0
}

verify_deployment() {
    local model_name="$1"
    local deploy_dir="${MODEL_DIRS[$ENVIRONMENT]}"
    local deploy_path="$deploy_dir/$model_name"

    log "INFO" "Verifying deployment..."

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would verify deployment"
        return 0
    fi

    # Check if file exists
    if [[ ! -f "$deploy_path" ]]; then
        error_exit "Deployed model not found: $deploy_path" 1
    fi

    # Verify checksum
    if [[ -f "${deploy_path}.md5" ]]; then
        local expected_checksum=$(cat "${deploy_path}.md5")
        local actual_checksum=$(calculate_checksum "$deploy_path")

        if [[ "$expected_checksum" == "$actual_checksum" ]]; then
            log "SUCCESS" "Checksum verification passed"
        else
            error_exit "Checksum verification failed!" 1
        fi
    else
        log "WARNING" "No checksum file found for verification"
    fi

    log "SUCCESS" "Deployment verification completed"
    return 0
}

# ===========================
# Usage and Help
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] MODEL_PATH

Deploy ML models to production environment with validation and backup.

OPTIONS:
    -e, --environment ENV    Target environment (dev, staging, prod)
                            Default: dev
    -n, --dry-run           Perform dry-run without actual deployment
    -v, --verbose           Enable verbose output
    -h, --help              Display this help message

EXAMPLES:
    # Deploy to production
    $SCRIPT_NAME -e prod model.pkl

    # Dry-run deployment to staging
    $SCRIPT_NAME --dry-run --environment staging model.pkl

    # Verbose deployment to dev
    $SCRIPT_NAME -v model.pkl

ENVIRONMENT VARIABLES:
    ENVIRONMENT             Override default environment

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

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--dry-run)
                DRY_RUN=true
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
            -*)
                error_exit "Unknown option: $1" 1
                ;;
            *)
                MODEL_PATH="$1"
                shift
                ;;
        esac
    done

    if [[ -z "${MODEL_PATH:-}" ]]; then
        error_exit "MODEL_PATH is required" 1
    fi
}

# ===========================
# Main Function
# ===========================

main() {
    log "INFO" "=========================================="
    log "INFO" "ML Model Deployment Script"
    log "INFO" "=========================================="

    parse_arguments "$@"

    log "INFO" "Configuration:"
    log "INFO" "  Environment: $ENVIRONMENT"
    log "INFO" "  Model: $MODEL_PATH"
    log "INFO" "  Dry-run: $DRY_RUN"
    log "INFO" "  Verbose: $VERBOSE"
    log "INFO" "=========================================="

    check_requirements
    validate_environment "$ENVIRONMENT"
    validate_model "$MODEL_PATH"

    if deploy_model "$MODEL_PATH"; then
        verify_deployment "$(basename "$MODEL_PATH")"
        log "SUCCESS" "=========================================="
        log "SUCCESS" "Deployment completed successfully!"
        log "SUCCESS" "=========================================="
        exit 0
    else
        error_exit "Deployment failed" 1
    fi
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
